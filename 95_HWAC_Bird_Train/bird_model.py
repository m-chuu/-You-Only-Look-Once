"""
tiny-YOLOv2 for one class (bird), computed the way the board computes it.

The board runs layers 1-3 on the 416x256 picture alone and takes the grey
border above and below from padding_file.bin. forward() does exactly that:
layers 1-3 on the picture, layers 1-3 on a grey 416x416 frame for the border,
spliced, then layers 4-9 on the 416x416 frame. Layers 1-3 keep their
batch-norm statistics fixed so both paths use the same normalisation - and
so the exported padding file matches what the network was trained with.

Head: 25 channels = 5 anchors x [tx, ty, tw, th, objectness], the same
anchors HWAC's decode uses. hwac_format.head_to_hwac() spreads them over the
board's 500 channels at export.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import hwac_format as hw

CHANNELS = [3, 16, 32, 64, 128, 256, 512, 1024, 512]
ANCHORS = torch.tensor(hw.ANCHORS)
COCO_WEIGHTS = (Path(__file__).resolve().parent.parent / "97_YOLOv2" / "darknet_train"
                / "yolov2-tiny.weights")
FIXED_BN = 3


class TinyYoloV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(CHANNELS[i], CHANNELS[i + 1], 3, 1, 1, bias=False)
                                   for i in range(8))
        self.bns = nn.ModuleList(nn.BatchNorm2d(CHANNELS[i + 1]) for i in range(8))
        self.head = nn.Conv2d(512, 25, 1)

    def train(self, mode=True):
        super().train(mode)
        for bn in self.bns[:FIXED_BN]:
            bn.eval()
        return self

    def block(self, x, i):
        x = F.leaky_relu(self.bns[i](self.convs[i](x)), 0.1)
        return hw.pool(x, hw.POOLS[i])

    def forward(self, pic):
        x = pic
        for i in range(3):
            x = self.block(x, i)
        g = torch.full((1, 3, hw.FRAME, hw.FRAME), hw.GREY, device=pic.device, dtype=pic.dtype)
        for i in range(3):
            g = self.block(g, i)
        frame = g.to(x.dtype).expand(pic.shape[0], -1, -1, -1).clone()
        frame[:, :, hw.PAD_ROWS] = x
        x = frame
        for i in range(3, 8):
            x = self.block(x, i)
        return self.head(x)

    # ---------------------------------------------------------- export

    @torch.no_grad()
    def hwac_layers(self):
        """Fold batch-norm into the board's (w, a, b) per layer, 9 layers."""
        layers = []
        for conv, bn in zip(self.convs, self.bns):
            a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            b = bn.bias - bn.running_mean * a
            layers.append((conv.weight.float().cpu().numpy(), a.float().cpu().numpy(),
                           b.float().cpu().numpy()))
        layers.append(hw.head_to_hwac(self.head.weight.float().cpu().numpy(),
                                      self.head.bias.float().cpu().numpy()))
        return layers


def load_coco(model, path=COCO_WEIGHTS):
    """Start from Darknet's COCO tiny-YOLOv2: it already knows what a bird is."""
    with open(path, "rb") as fh:
        major, minor, _ = np.fromfile(fh, np.int32, 3)
        np.fromfile(fh, np.int64 if major * 10 + minor >= 2 else np.int32, 1)
        p = np.fromfile(fh, np.float32)
    cur = 0

    def take(n):
        nonlocal cur
        cur += n
        return torch.from_numpy(p[cur - n:cur].copy())

    with torch.no_grad():
        for i in range(8):
            f, c = CHANNELS[i + 1], CHANNELS[i]
            bn, conv = model.bns[i], model.convs[i]
            bn.bias.copy_(take(f))
            bn.weight.copy_(take(f))
            bn.running_mean.copy_(take(f))
            bn.running_var.copy_(take(f))
            conv.weight.copy_(take(f * c * 9).view(f, c, 3, 3))
        bias, weight = take(425), take(425 * 512).view(425, 512, 1, 1)
        if cur != p.size:
            raise ValueError(f"{p.size - cur} floats left over - not the stock COCO file")
        idx = [85 * n + k for n in range(5) for k in range(5)]   # tx ty tw th obj per anchor
        model.head.weight.copy_(weight[idx])
        model.head.bias.copy_(bias[idx])
    return model


# ------------------------------------------------------------------ loss

def box_iou(a, b):
    return torchvision.ops.box_iou(a, b)


def yolo_loss(out, targets, ignore_thresh=0.6, obj_scale=5.0, noobj_scale=1.0):
    """YOLOv2 region loss. out (B, 25, 13, 13); targets: list of (n, 4) cx,cy,w,h grid units."""
    B, G = out.shape[0], hw.GRID
    dev = out.device
    p = out.float().view(B, 5, 5, G, G).permute(0, 1, 3, 4, 2)
    anchors = ANCHORS.to(dev)
    gy, gx = torch.meshgrid(torch.arange(G, device=dev), torch.arange(G, device=dev), indexing="ij")
    sx, sy = torch.sigmoid(p[..., 0]), torch.sigmoid(p[..., 1])
    tw, th, to = p[..., 2], p[..., 3], p[..., 4]

    with torch.no_grad():
        bx, by = sx + gx, sy + gy
        bw = torch.exp(tw.clamp(-8, 8)) * anchors[:, 0].view(1, 5, 1, 1)
        bh = torch.exp(th.clamp(-8, 8)) * anchors[:, 1].view(1, 5, 1, 1)
        pred = torch.stack([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2], -1)

    obj = torch.zeros(B, 5, G, G, dtype=torch.bool, device=dev)
    noobj = torch.ones_like(obj)
    t = torch.zeros(5, B, 5, G, G, device=dev)        # tx, ty, tw, th, coord scale
    for i in range(B):
        gt = targets[i].to(dev)
        if gt.numel() == 0:
            continue
        gxyxy = torch.cat([gt[:, :2] - gt[:, 2:] / 2, gt[:, :2] + gt[:, 2:] / 2], 1)
        noobj[i].view(-1)[box_iou(pred[i].reshape(-1, 4), gxyxy).max(1).values > ignore_thresh] = False
        inter = torch.min(gt[:, 2:3], anchors[:, 0]) * torch.min(gt[:, 3:4], anchors[:, 1])
        best = (inter / (gt[:, 2:3] * gt[:, 3:4] + anchors[:, 0] * anchors[:, 1] - inter)).argmax(1)
        gi, gj = gt[:, 0].long().clamp(0, G - 1), gt[:, 1].long().clamp(0, G - 1)
        obj[i, best, gj, gi] = True
        noobj[i, best, gj, gi] = False
        t[0, i, best, gj, gi] = gt[:, 0] - gi
        t[1, i, best, gj, gi] = gt[:, 1] - gj
        t[2, i, best, gj, gi] = torch.log(gt[:, 2] / anchors[best, 0] + 1e-9)
        t[3, i, best, gj, gi] = torch.log(gt[:, 3] / anchors[best, 1] + 1e-9)
        t[4, i, best, gj, gi] = 2.0 - gt[:, 2] * gt[:, 3] / (G * G)

    m = obj
    s = t[4][m]
    l_xy = (s * ((sx[m] - t[0][m]) ** 2 + (sy[m] - t[1][m]) ** 2)).sum()
    l_wh = (s * ((tw[m] - t[2][m]) ** 2 + (th[m] - t[3][m]) ** 2)).sum()
    l_obj = obj_scale * F.binary_cross_entropy_with_logits(to[m], torch.ones_like(to[m]), reduction="sum")
    l_no = noobj_scale * F.binary_cross_entropy_with_logits(to[noobj], torch.zeros_like(to[noobj]),
                                                           reduction="sum")
    total = (l_xy + l_wh + l_obj + l_no) / B
    parts = {k: float(v.detach()) / B for k, v in
             (("xy", l_xy), ("wh", l_wh), ("obj", l_obj), ("noobj", l_no))}
    return total, parts


# ---------------------------------------------------------- evaluation

HWAC_TO_25 = [c for n in range(5) for c in hw.BOX_CH[n] + [hw.OBJ_CH[n]]]


def decode(out25):
    """(B, 25, 13, 13) -> boxes (B, 845, 4) picture px xyxy, scores (B, 845)."""
    B, G = out25.shape[0], hw.GRID
    dev = out25.device
    p = out25.float().view(B, 5, 5, G, G)
    gy, gx = torch.meshgrid(torch.arange(G, device=dev), torch.arange(G, device=dev), indexing="ij")
    anchors = ANCHORS.to(dev)
    cx = (torch.sigmoid(p[:, :, 0]) + gx) * 32
    cy = (torch.sigmoid(p[:, :, 1]) + gy) * 32 - hw.TOP
    w = torch.exp(p[:, :, 2].clamp(max=8)) * anchors[:, 0].view(1, 5, 1, 1) * 32
    h = torch.exp(p[:, :, 3].clamp(max=8)) * anchors[:, 1].view(1, 5, 1, 1) * 32
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).view(B, -1, 4)
    return boxes, torch.sigmoid(p[:, :, 4]).reshape(B, -1)


def average_precision(dets, gts, iou=0.5):
    """dets: list of (boxes, scores) per image after NMS; gts: list of boxes."""
    records, n_gt = [], sum(len(g) for g in gts)
    for img, ((boxes, scores), gt) in enumerate(zip(dets, gts)):
        for b, s in zip(boxes, scores):
            records.append((float(s), img, b))
    records.sort(key=lambda r: -r[0])
    used = [np.zeros(len(g), bool) for g in gts]
    tp = np.zeros(len(records))
    for k, (_, img, b) in enumerate(records):
        g = gts[img]
        if len(g):
            ious = box_iou(b[None], g)[0].numpy()
            j = int(ious.argmax())
            if ious[j] >= iou and not used[img][j]:
                used[img][j] = True
                tp[k] = 1
    if n_gt == 0 or not len(records):
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(records) + 1)
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[0], precision, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate(predict25, bird_loader, neg_loader, device="cuda"):
    """predict25: pics -> (B, 25, 13, 13). Board-relevant numbers:

    ap50       standard detection quality on the bird images
    top1_hit   share of bird images where the single top box (what the board
               draws) lands on a bird (IoU >= 0.3)
    fp30/fp50  share of people photos whose top box scores above 0.3 / 0.5
    """
    dets, gts, hits, top_scores = [], [], [], []
    for pics, boxes in bird_loader:
        b, s = decode(predict25(pics.to(device)))
        for i in range(len(pics)):
            gt = boxes[i]
            k = int(s[i].argmax())
            hits.append(bool(len(gt)) and float(box_iou(b[i, k:k + 1].cpu(), gt).max()) >= 0.3)
            top_scores.append(float(s[i, k]))
            keep = s[i] > 0.005
            bb, ss = b[i][keep], s[i][keep]
            keep = torchvision.ops.nms(bb, ss, 0.45)
            dets.append((bb[keep].cpu(), ss[keep].cpu()))
            gts.append(gt)
    neg = []
    for pics, _ in neg_loader:
        _, s = decode(predict25(pics.to(device)))
        neg += s.max(1).values.cpu().tolist()
    neg = np.array(neg)
    return {"ap50": average_precision(dets, gts), "top1_hit": float(np.mean(hits)),
            "bird_top_score": float(np.median(top_scores)),
            "fp30": float(np.mean(neg > 0.3)), "fp50": float(np.mean(neg > 0.5)),
            "neg_top_score": float(np.median(neg))}
