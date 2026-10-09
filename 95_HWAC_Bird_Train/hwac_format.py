"""
HWAC weight format used by the LogikHaus SoC's AI core (tiny-YOLOv2, fp16).

Decoded from HWAC's own weights95tuned.npy, which is byte-identical to
sw1/weight_file.bin. Running those stock weights through BoardNet below finds
the eagle in board_test/T1.bin and nothing in the blank T4.bin, which is what
confirms the layout. 00_Selftest.py re-checks all of it.

weight_file.bin  (22,542,352 bytes, little-endian fp16, rows of 4, no header)
    Nine layers back to back. Within a layer, one block per group of 4
    output filters:
        for each group of 4 input channels (the last one padded with 0):
            [only before the LAST group]  row a  - scale of the 4 filters
                                          row b  - offset of the 4 filters
            4 filters x k*k taps (ky, kx row-major), one row per tap,
            each row holding the 4 input channels
    The core computes  y = a * conv(x, w) + b,  then leaky ReLU 0.1 on
    layers 1-8. Layer 9 is linear, with a = 1.

padding_file.bin (346,112 bytes)
    Layer-3 output (64 channels, 52x52) of a 416x416 frame filled with grey
    0.5, laid out (channel group, y, x, lane). Rows 10-41 are zero: that is
    where the features of the board's 416x256 picture go. The board runs
    layers 1-3 on the picture only and borrows the border from this file.

Layer 9 output order (500 channels)
    anchor n: objectness at 96n, 95 class scores at 96n+1 .. 96n+95,
              box tx, ty, tw, th at 480+4n .. 483+4n
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# (out filters, in channels, kernel) per layer, and the maxpool after it
LAYERS = [(16, 3, 3), (32, 16, 3), (64, 32, 3), (128, 64, 3), (256, 128, 3),
          (512, 256, 3), (1024, 512, 3), (512, 1024, 3), (500, 512, 1)]
POOLS = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 1), None, None, None]

WEIGHT_BYTES = 22_542_352
PADDING_BYTES = 346_112
STOCK_WEIGHT_SHA256 = "d8bd177ca902ec05de0f217c8bd6b12286bb8e9aca81ba1479699282eecc6df6"

IMG_W, IMG_H = 416, 256      # the picture the board feeds the AI
FRAME = 416                  # network frame; the picture sits in rows 80..335
TOP = (FRAME - IMG_H) // 2   # 80
GRID = FRAME // 32           # 13
PAD_ROWS = slice(TOP // 8, (TOP + IMG_H) // 8)   # rows 10..41 of the layer-3 map
GREY = 0.5

# Stock tiny-YOLOv2 anchors (grid units). HWAC's decode uses exactly these.
ANCHORS = np.array([[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],
                    [7.88282, 3.52778], [9.77052, 9.16828]], np.float32)
N_ANCHORS = 5
OBJ_CH = [96 * n for n in range(N_ANCHORS)]
BOX_CH = [[480 + 4 * n + k for k in range(4)] for n in range(N_ANCHORS)]

# the dashboard and the board report boxes on the 1920x1080 output
DISPLAY_W, DISPLAY_H = 1920, 1080


# --------------------------------------------------------------- layout

def _dims(C, k):
    cp = -(-C // 4) * 4
    return cp, cp // 4, k * k


def rows_per_layer(F_, C, k):
    _, ncg, kk = _dims(C, k)
    return (F_ // 4) * (2 + ncg * 4 * kk)


def pack_layer(w, a, b):
    """w (F, C, k, k), a (F,), b (F,)  ->  (rows, 4) float16 in HWAC order."""
    F_, C, k, _ = w.shape
    cp, ncg, kk = _dims(C, k)
    wp = np.zeros((F_, cp, kk), np.float32)
    wp[:, :C] = np.asarray(w, np.float32).reshape(F_, C, kk)
    # (g, j, cg, lane, t) -> (g, cg, j, t, lane): per channel group, 4 filters x taps
    blocks = (wp.reshape(F_ // 4, 4, ncg, 4, kk)
                .transpose(0, 2, 1, 4, 3)
                .reshape(F_ // 4, ncg, 4 * kk, 4))
    params = np.stack([np.asarray(a, np.float32).reshape(F_ // 4, 4),
                       np.asarray(b, np.float32).reshape(F_ // 4, 4)], axis=1)
    groups = np.concatenate([blocks[:, :-1].reshape(F_ // 4, -1, 4), params, blocks[:, -1]],
                            axis=1)
    return groups.reshape(-1, 4).astype(np.float16)


def unpack_layer(rows, F_, C, k):
    """Inverse of pack_layer: (rows, 4) -> w (F, C, k, k), a (F,), b (F,) as float32."""
    cp, ncg, kk = _dims(C, k)
    g = 2 + ncg * 4 * kk
    blk = np.asarray(rows, np.float32).reshape(F_ // 4, g, 4)
    p0 = (ncg - 1) * 4 * kk
    a = blk[:, p0].reshape(F_)
    b = blk[:, p0 + 1].reshape(F_)
    taps = np.concatenate([blk[:, :p0], blk[:, p0 + 2:]], axis=1)
    w = (taps.reshape(F_ // 4, ncg, 4, kk, 4)
             .transpose(0, 2, 1, 4, 3)
             .reshape(F_, cp, k, k)[:, :C])
    return np.ascontiguousarray(w), a, b


# ------------------------------------------------------------ file I/O

def read_weight_file(path):
    raw = np.fromfile(path, dtype=np.float16)
    if raw.nbytes != WEIGHT_BYTES:
        raise ValueError(f"{path}: {raw.nbytes:,} bytes, expected {WEIGHT_BYTES:,}")
    layers, off = [], 0
    for F_, C, k in LAYERS:
        n = rows_per_layer(F_, C, k) * 4
        layers.append(unpack_layer(raw[off:off + n].reshape(-1, 4), F_, C, k))
        off += n
    return layers


def weight_file_bytes(layers):
    out = []
    for (w, a, b), (F_, C, k) in zip(layers, LAYERS):
        if w.shape != (F_, C, k, k):
            raise ValueError(f"layer shape {w.shape}, expected {(F_, C, k, k)}")
        out.append(pack_layer(w, a, b))
    data = np.concatenate(out).tobytes()
    if len(data) != WEIGHT_BYTES:
        raise ValueError(f"packed {len(data):,} bytes, expected {WEIGHT_BYTES:,}")
    return data


def read_padding_file(path):
    raw = np.fromfile(path, dtype=np.float16)
    if raw.nbytes != PADDING_BYTES:
        raise ValueError(f"{path}: {raw.nbytes:,} bytes, expected {PADDING_BYTES:,}")
    return raw.astype(np.float32).reshape(16, 52, 52, 4).transpose(0, 3, 1, 2).reshape(64, 52, 52)


def padding_file_bytes(feat):
    """feat: layer-3 output of the grey frame, (64, 52, 52). Picture rows are zeroed."""
    feat = np.array(feat, np.float32, copy=True)
    feat[:, PAD_ROWS] = 0.0
    data = feat.reshape(16, 4, 52, 52).transpose(0, 2, 3, 1).astype(np.float16).tobytes()
    assert len(data) == PADDING_BYTES
    return data


def sha256(path_or_bytes):
    data = Path(path_or_bytes).read_bytes() if not isinstance(path_or_bytes, bytes) else path_or_bytes
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------- layer 9 <-> 5-anchor head

def head_to_hwac(w25, b25, class_logit=8.0):
    """Trained head (25 ch = 5 anchors x [tx, ty, tw, th, obj]) -> HWAC layer 9.

    The bird is class slot 0 of each anchor and the other 94 slots are pushed
    far down, so whether the core uses the class scores or not, the winning
    box is decided by objectness alone.
    """
    w = np.zeros((500, 512, 1, 1), np.float32)
    b = np.zeros(500, np.float32)
    w25 = np.asarray(w25, np.float32).reshape(25, 512, 1, 1)
    b25 = np.asarray(b25, np.float32).reshape(25)
    for n in range(N_ANCHORS):
        w[OBJ_CH[n]], b[OBJ_CH[n]] = w25[5 * n + 4], b25[5 * n + 4]
        b[OBJ_CH[n] + 1] = class_logit
        b[OBJ_CH[n] + 2:OBJ_CH[n] + 96] = -class_logit
        for k in range(4):
            w[BOX_CH[n][k]], b[BOX_CH[n][k]] = w25[5 * n + k], b25[5 * n + k]
    return w, np.ones(500, np.float32), b


# ------------------------------------------------- software board model

def pool(x, spec):
    if spec is None:
        return x
    size, stride = spec
    if stride == 1:   # Darknet: pad right/bottom with -inf, keep the size
        x = F.pad(x, (0, size - 1, 0, size - 1), value=float("-inf"))
    return F.max_pool2d(x, size, stride)


class BoardNet(torch.nn.Module):
    """What the board computes, given a weight file and a padding file.

    forward() takes the 416x256 picture (B, 3, 256, 416), values 0..1, and
    returns layer 9's raw output (B, 500, 13, 13).
    """

    def __init__(self, layers, padding):
        super().__init__()
        for i, (w, a, b) in enumerate(layers):
            self.register_buffer(f"w{i}", torch.from_numpy(np.ascontiguousarray(w, np.float32)))
            self.register_buffer(f"a{i}", torch.from_numpy(np.asarray(a, np.float32)).view(1, -1, 1, 1))
            self.register_buffer(f"b{i}", torch.from_numpy(np.asarray(b, np.float32)).view(1, -1, 1, 1))
        self.register_buffer("border", torch.from_numpy(np.asarray(padding, np.float32))[None])

    @classmethod
    def from_files(cls, weight_path, padding_path):
        return cls(read_weight_file(weight_path), read_padding_file(padding_path))

    def layer(self, x, i):
        w = getattr(self, f"w{i}")
        y = F.conv2d(x, w.to(x.dtype), padding=(w.shape[-1] - 1) // 2)
        y = y * getattr(self, f"a{i}").to(x.dtype) + getattr(self, f"b{i}").to(x.dtype)
        if i < 8:
            y = F.leaky_relu(y, 0.1)
        return pool(y, POOLS[i])

    def forward(self, img):
        x = img
        for i in range(3):
            x = self.layer(x, i)
        frame = self.border.to(x.dtype).expand(img.shape[0], -1, -1, -1).clone()
        frame[:, :, PAD_ROWS] = x
        x = frame
        for i in range(3, 9):
            x = self.layer(x, i)
        return x


def top_box(head):
    """The single box the board would draw: highest objectness over all cells.

    head: (500, 13, 13) tensor or array. Returns dict with objectness, the box
    in picture pixels (416x256) and in display pixels (1920x1080).
    """
    h = torch.as_tensor(head, dtype=torch.float32)
    obj = torch.sigmoid(h[OBJ_CH])                         # (5, 13, 13)
    n, gy, gx = np.unravel_index(int(torch.argmax(obj)), obj.shape)
    tx, ty, tw, th = (float(h[c, gy, gx]) for c in BOX_CH[n])
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    cx = (gx + sig(tx)) / GRID * FRAME
    cy = (gy + sig(ty)) / GRID * FRAME - TOP
    bw = np.exp(min(tw, 8.0)) * ANCHORS[n, 0] * 32
    bh = np.exp(min(th, 8.0)) * ANCHORS[n, 1] * 32
    l, r, t, b = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
    sx, sy = DISPLAY_W / IMG_W, DISPLAY_H / IMG_H
    return {
        "objectness": float(obj.max()), "anchor": int(n), "cell": (int(gy), int(gx)),
        "picture": (l, r, t, b),
        "display": tuple(int(round(v)) for v in (l * sx, r * sx, t * sy, b * sy)),
    }


def load_picture_bin(path):
    """A board image file (416x256x4 fp16) -> (1, 3, 256, 416) float tensor."""
    raw = np.fromfile(path, dtype=np.float16)
    pic = raw.reshape(IMG_H, IMG_W, 4)[:, :, :3].astype(np.float32).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(pic))[None]
