"""
Training and evaluation samples shaped like the picture the board's AI sees.

The board squeezes its 1920x1080 video into a 416x256 picture (16:9 pressed
into 13:8) and runs the network on a 416x416 frame with grey bars above and
below. So every sample here is a 16:9 scene squeezed to 416x256, and training
boxes come out in grid units of that 416x416 frame.

Sources
  birds_rf    100_BirdImages train split: labelled birds in the sky, closest
              to what the board sees
  birds_bf    BirdsFlying close-ups labelled by 01_Pseudo_Label.py, often
              shrunk here so they also look like distant birds
  negatives   photos of people (archive/Humans), never labelled: "not a bird"
Text and icon overlays are painted on some samples, unlabelled, so watermarks
and on-screen icons are learned as "not a bird" too. Red and blue are swapped
at random because the board's channel order is not confirmed.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]          # C:\LogikHaus\LogikHaus
HERE = Path(__file__).resolve().parent

SCENE_W, SCENE_H = 832, 468      # 16:9 working canvas
PIC_W, PIC_H = 416, 256          # what the board feeds the AI
TOP = 80                         # picture rows start here in the 416x416 frame
CELL = 32
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

WORDS = ["storyblocks", "gettyimages", "shutterstock", "SAMPLE", "LIVE", "REC", "CAM 1",
         "LogikHaus", "watermark", "12:45:07", "2026-10-09", "HD", "PREVIEW", "adobe stock",
         "iStock", "Dreamstime", "123RF", "pond5", "DEMO", "Copyright"]


# ------------------------------------------------------------- sources

def yolo_labels(path):
    if path is None or not Path(path).exists():
        return np.zeros((0, 4), np.float32)
    rows = [list(map(float, line.split()[1:5]))
            for line in Path(path).read_text().splitlines() if line.strip()]
    return np.array(rows, np.float32).reshape(-1, 4)


def rf_split(split):
    d = ROOT / "100_BirdImages" / split
    return [(str(p), str(d / "labels" / (p.stem + ".txt")))
            for p in sorted((d / "images").iterdir()) if p.suffix.lower() in IMG_EXT]


def birdsflying():
    d = HERE / "data" / "birdsflying"
    paths = [p for p in (d / "images.txt").read_text().splitlines() if p.strip()]
    return [(p, str(d / "labels" / (Path(p).stem + ".txt"))) for p in paths]


def humans(n_val=300, seed=0):
    paths = sorted(str(p) for p in (ROOT / "archive" / "Humans").iterdir()
                   if p.suffix.lower() in IMG_EXT)
    random.Random(seed).shuffle(paths)
    return [(p, None) for p in paths[n_val:]], [(p, None) for p in paths[:n_val]]


def load(item):
    img = cv2.imread(item[0], cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    lab = yolo_labels(item[1])
    xyxy = np.stack([(lab[:, 0] - lab[:, 2] / 2) * w, (lab[:, 1] - lab[:, 3] / 2) * h,
                     (lab[:, 0] + lab[:, 2] / 2) * w, (lab[:, 1] + lab[:, 3] / 2) * h], 1)
    return img, xyxy.reshape(-1, 4).astype(np.float32)


# ------------------------------------------------------------ geometry

def clip_boxes(b, region, min_keep=0.2, min_side=6.0):
    if len(b) == 0:
        return b.reshape(0, 4)
    x0, y0, x1, y1 = region
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(x0, x1)
    c[:, [1, 3]] = c[:, [1, 3]].clip(y0, y1)
    cw, ch = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    keep = (cw * ch >= min_keep * np.maximum(area, 1e-6)) & (cw >= min_side) & (ch >= min_side)
    return c[keep]


def border_colour(img):
    h, w = img.shape[:2]
    m = max(1, int(0.04 * min(h, w)))
    edge = np.concatenate([img[:m].reshape(-1, 3), img[-m:].reshape(-1, 3),
                           img[:, :m].reshape(-1, 3), img[:, -m:].reshape(-1, 3)])
    return np.median(edge, axis=0).astype(np.uint8)


def resize(img, w, h):
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(img, (int(w), int(h)), interpolation=interp)


def cover(scene, region, img, boxes, rng, zoom=(1.0, 1.5)):
    """Fill region with a random crop of img scaled to cover it."""
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    ih, iw = img.shape[:2]
    s = max(rw / iw, rh / ih) * rng.uniform(*zoom)
    sw, sh = max(rw, int(round(iw * s))), max(rh, int(round(ih * s)))
    im = resize(img, sw, sh)
    ox, oy = int(rng.integers(0, sw - rw + 1)), int(rng.integers(0, sh - rh + 1))
    scene[y0:y1, x0:x1] = im[oy:oy + rh, ox:ox + rw]
    b = boxes * np.array([sw / iw, sh / ih] * 2, np.float32) + np.array([x0 - ox, y0 - oy] * 2, np.float32)
    return clip_boxes(b, region)


def zoom_out(scene, img, boxes, rng):
    """Shrink img onto a flat background of its own border colour."""
    ih, iw = img.shape[:2]
    s = min(rng.uniform(0.2, 0.8) * SCENE_H / ih, 0.95 * SCENE_W / iw)
    sw, sh = max(8, int(iw * s)), max(8, int(ih * s))
    scene[:] = border_colour(img)
    noise = rng.normal(0, 2.0, scene.shape).astype(np.float32)
    scene[:] = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    px, py = int(rng.integers(0, SCENE_W - sw + 1)), int(rng.integers(0, SCENE_H - sh + 1))
    scene[py:py + sh, px:px + sw] = resize(img, sw, sh)
    b = boxes * np.array([sw / iw, sh / ih] * 2, np.float32) + np.array([px, py] * 2, np.float32)
    return clip_boxes(b, (0, 0, SCENE_W, SCENE_H))


def fit(img, boxes):
    """Deterministic: the whole image fitted into the 16:9 scene."""
    ih, iw = img.shape[:2]
    s = min(SCENE_W / iw, SCENE_H / ih)
    sw, sh = int(round(iw * s)), int(round(ih * s))
    scene = np.empty((SCENE_H, SCENE_W, 3), np.uint8)
    scene[:] = border_colour(img)
    px, py = (SCENE_W - sw) // 2, (SCENE_H - sh) // 2
    scene[py:py + sh, px:px + sw] = resize(img, sw, sh)
    b = boxes * np.array([sw / iw, sh / ih] * 2, np.float32) + np.array([px, py] * 2, np.float32)
    return scene, b


# ------------------------------------------------------- appearance

def hsv_jitter(scene, rng):
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-0.04, 0.04) * 180) % 180
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    hsv[..., 2] *= rng.uniform(0.6, 1.4)
    hsv[..., 1:] = hsv[..., 1:].clip(0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def blend(scene, layer, alpha):
    cv2.addWeighted(layer, alpha, scene, 1 - alpha, 0, dst=scene)


def text_overlay(scene, rng):
    if rng.random() < 0.7:
        txt = WORDS[int(rng.integers(len(WORDS)))]
    else:
        txt = "".join(chr(int(c)) for c in rng.integers(65, 91, int(rng.integers(3, 10))))
    font = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN][int(rng.integers(5))]
    scale, thick = rng.uniform(0.8, 3.0), int(rng.integers(1, 6))
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
    x = int(rng.integers(0, max(1, SCENE_W - tw)))
    y = int(rng.integers(th, max(th + 1, SCENE_H)))
    colour = [(255, 255, 255), (0, 0, 0), (128, 128, 128),
              tuple(int(v) for v in rng.integers(0, 256, 3))][int(rng.integers(4))]
    layer = scene.copy()
    cv2.putText(layer, txt, (x, y), font, scale, colour, thick, cv2.LINE_AA)
    blend(scene, layer, rng.uniform(0.35, 1.0))


def icon_overlay(scene, rng):
    w = int(rng.integers(16, 56))
    h = int(w * rng.uniform(0.6, 1.6))
    edge = int(rng.integers(4))
    m = int(rng.integers(2, 40))
    if edge == 0:
        x, y = SCENE_W - w - m, int(rng.integers(0, SCENE_H - h))
    elif edge == 1:
        x, y = int(rng.integers(0, SCENE_W - w)), m
    elif edge == 2:
        x, y = m, int(rng.integers(0, SCENE_H - h))
    else:
        x, y = int(rng.integers(0, SCENE_W - w)), SCENE_H - h - m
    layer = scene.copy()
    shade = int(rng.integers(20, 90))
    cv2.rectangle(layer, (x, y), (x + w, y + h), (shade, shade, shade), -1)
    cx, cy = x + w // 2, y + h // 2
    if rng.random() < 0.5:
        pts = np.array([[cx - w // 5, cy - h // 4], [cx - w // 5, cy + h // 4], [cx + w // 4, cy]])
        cv2.fillPoly(layer, [pts.astype(np.int32)], (255, 255, 255))
    else:
        cv2.rectangle(layer, (cx - w // 4, cy - 2), (cx + w // 4, cy + 2), (255, 255, 255), -1)
    blend(scene, layer, rng.uniform(0.6, 0.95))


# ------------------------------------------------------- final picture

def to_picture(scene, boxes, swap_rb=False):
    """Scene (BGR uint8) -> picture tensor (3, 256, 416) RGB 0..1, boxes in picture px."""
    pic = cv2.resize(scene, (PIC_W, PIC_H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    if swap_rb:
        pic = pic[:, :, ::-1]
    t = torch.from_numpy(np.ascontiguousarray(pic.transpose(2, 0, 1))).float().div_(255.0)
    sx, sy = PIC_W / SCENE_W, PIC_H / SCENE_H
    return t, (boxes * np.array([sx, sy, sx, sy], np.float32)).reshape(-1, 4)


def picture_to_grid(b):
    """Picture-px xyxy -> (cx, cy, w, h) in grid units of the 416x416 frame."""
    if len(b) == 0:
        return np.zeros((0, 4), np.float32)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2 + TOP
    return np.stack([cx, cy, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1).astype(np.float32) / CELL


# --------------------------------------------------------------- datasets

class TrainSet(torch.utils.data.Dataset):
    """Random board-shaped samples. sources: {name: (items, weight)}."""

    def __init__(self, sources, epoch_size=2400, overlays=True):
        self.names = list(sources)
        self.items = [sources[n][0] for n in self.names]
        w = np.array([sources[n][1] for n in self.names], np.float64)
        self.weights = w / w.sum()
        self.epoch_size = epoch_size
        self.overlays = overlays
        self._rng = None

    def __len__(self):
        return self.epoch_size

    def __getstate__(self):
        # each DataLoader worker must seed its own generator, not inherit one
        state = self.__dict__.copy()
        state["_rng"] = None
        return state

    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        return self._rng

    def draw(self, rng):
        for _ in range(20):
            src = self.items[int(rng.choice(len(self.items), p=self.weights))]
            img, boxes = load(src[int(rng.integers(len(src)))])
            if img is not None:
                return img, boxes
        raise RuntimeError("could not read any image")

    def scene(self, rng):
        scene = np.empty((SCENE_H, SCENE_W, 3), np.uint8)
        mode = rng.random()
        if mode < 0.35:                                   # 2x2 mosaic
            cx = int(rng.uniform(0.3, 0.7) * SCENE_W)
            cy = int(rng.uniform(0.3, 0.7) * SCENE_H)
            parts = []
            for region in ((0, 0, cx, cy), (cx, 0, SCENE_W, cy),
                           (0, cy, cx, SCENE_H), (cx, cy, SCENE_W, SCENE_H)):
                img, boxes = self.draw(rng)
                parts.append(cover(scene, region, img, boxes, rng, zoom=(1.0, 1.3)))
            boxes = np.concatenate(parts)
        elif mode < 0.70:                                 # one image, cropped to fill
            img, boxes = self.draw(rng)
            boxes = cover(scene, (0, 0, SCENE_W, SCENE_H), img, boxes, rng, zoom=(1.0, 1.6))
        else:                                             # one image, shrunk
            img, boxes = self.draw(rng)
            boxes = zoom_out(scene, img, boxes, rng)
        return scene, boxes

    def __getitem__(self, i):
        rng = self.rng()
        scene, boxes = self.scene(rng)
        if rng.random() < 0.5:
            scene = np.ascontiguousarray(scene[:, ::-1])
            boxes = boxes.copy()
            boxes[:, [0, 2]] = SCENE_W - boxes[:, [2, 0]]
        if rng.random() < 0.8:
            scene = hsv_jitter(scene, rng)
        if self.overlays and rng.random() < 0.3:
            text_overlay(scene, rng)
        if self.overlays and rng.random() < 0.2:
            icon_overlay(scene, rng)
        if rng.random() < 0.15:
            scene = cv2.GaussianBlur(scene, (3, 3) if rng.random() < 0.6 else (5, 5), 0)
        if rng.random() < 0.2:
            noise = rng.normal(0, rng.uniform(2, 6), scene.shape).astype(np.float32)
            scene = np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        pic, b = to_picture(scene, boxes, swap_rb=rng.random() < 0.5)
        return pic, torch.from_numpy(picture_to_grid(b))


class EvalSet(torch.utils.data.Dataset):
    """Whole images fitted into the board picture; boxes in picture px."""

    def __init__(self, items):
        self.items = [it for it in items if cv2.haveImageReader(it[0])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, boxes = load(self.items[i])
        scene, b = fit(img, boxes)
        pic, b = to_picture(scene, b)
        return pic, torch.from_numpy(b)


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def preview_sheet(dataset, path, n=12, cols=3):
    """Save training samples with their boxes drawn, to check the pipeline."""
    tiles = []
    for i in range(n):
        pic, grid = dataset[i]
        img = (pic.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8).copy()
        for cx, cy, w, h in grid.numpy() * CELL:
            cy -= TOP
            cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                          (0, 0, 255), 1)
        tiles.append(cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
    rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(n // cols)]
    cv2.imwrite(str(path), np.vstack(rows))
