"""
Make a 12-second test video for lock-on tracking from the five-geese photo.

    python 05_Make_Test_Video.py

The geese are cut out of checks/five_geese_camera.png and flown over the same
sky (1280x720, 24 fps). What happens, with A being the bird auto should lock:

    0 s     only goose A, flying right            -> auto locks A
    0.9 s   goose B flies in from the right
    2.5 s   goose C (the biggest) flies in        -> lock must stay on A
    4.5 s   B crosses A, in front of it; for      -> lock must stay on A
            ~0.6 s the detector sees one bird
    6 s     goose D flies in from the left
    ~10 s   A leaves on the right                 -> after ~1 s, lock moves to the best bird left

Writes checks/geese_test.mp4 and checks/geese_test_truth.csv (where each
goose really is in every frame, and how much of it is visible).

No cloud on purpose: the board model scores a lone cloud ~0.44, but a goose
next to a cloud comes out as one wide box scoring up to 0.99 - a detector
problem that would hide what the tracker does.
"""

import argparse
import csv
import importlib
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
W, H, FPS, SECONDS = 1280, 720, 24, 12

# name, sprite (0 = biggest goose), flies left?, width px, appears at (s), path t -> (cx, cy)
BIRDS = [
    ("A", 1, False, 200, 0.0, lambda t: (100 + 125 * t, 430 + 20 * math.sin(1.3 * t))),
    ("B", 2, True, 190, 0.9, lambda t: (662 - 200 * (t - 4.5), 422 + 70 * (t - 4.5))),
    ("C", 0, True, 230, 2.5, lambda t: (1400 - 80 * (t - 2.5), 150 + 12 * math.sin(0.9 * t))),
    ("D", 3, False, 170, 6.0, lambda t: (-100 + 110 * (t - 6.0), 260 + 10 * math.sin(1.1 * t))),
]
ORDER = ["A", "B", "C", "D"]   # back to front: B passes in front of A


def sky_terms(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xx / w, yy / h
    return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], -1).reshape(-1, 6)


def fit_sky(img, rounds=3):
    """Smooth sky colour (2nd-order surface per channel), fitted around the birds."""
    terms = sky_terms(img.shape[1], img.shape[0])
    pix = img.reshape(-1, 3).astype(np.float32)
    keep = np.ones(len(pix), bool)
    for _ in range(rounds):
        coef = np.linalg.lstsq(terms[keep], pix[keep], rcond=None)[0]
        keep = np.abs(pix - terms @ coef).sum(1) < 30
    return coef


def cut_geese(img, coef):
    """Each goose as (colour, alpha), biggest first. Bits touching the edge are dropped."""
    sky = (sky_terms(img.shape[1], img.shape[0]) @ coef).reshape(img.shape)
    mask = (np.abs(img.astype(np.float32) - sky).sum(2) > 45).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    geese = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if x == 0 or y == 0 or x + w == img.shape[1] or y + h == img.shape[0] or area < 500:
            continue
        x0, y0, x1, y1 = max(x - 4, 0), max(y - 4, 0), x + w + 4, y + h + 4
        alpha = (labels[y0:y1, x0:x1] == k).astype(np.uint8) * 255
        alpha = cv2.GaussianBlur(cv2.dilate(alpha, np.ones((3, 3), np.uint8)), (3, 3), 0)
        geese.append((area, img[y0:y1, x0:x1].astype(np.float32), alpha.astype(np.float32) / 255))
    geese.sort(key=lambda g: -g[0])
    return [(c, a) for _, c, a in geese]


def place(colour, alpha, cx, cy):
    """Full-frame colour and alpha with the sprite centred at (cx, cy), cut at the frame edge."""
    h, w = alpha.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    fc, fa = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)
    sx0, sy0, sx1, sy1 = max(0, -x0), max(0, -y0), min(w, W - x0), min(h, H - y0)
    if sx1 > sx0 and sy1 > sy0:
        fc[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1] = colour[sy0:sy1, sx0:sx1]
        fa[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1] = alpha[sy0:sy1, sx0:sx1]
    return fc, fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, default=HERE / "checks" / "five_geese_camera.png")
    ap.add_argument("--out", type=Path, default=HERE / "checks" / "geese_test.mp4")
    args = ap.parse_args()

    try_picture = importlib.import_module("04_Try_Picture")
    img = cv2.imread(str(args.photo))
    if img is None:
        raise SystemExit(f"cannot read {args.photo}")
    img, _ = try_picture.remove_board_box(try_picture.cut_out_picture(img))
    coef = fit_sky(img)
    geese = cut_geese(img, coef)
    if len(geese) < 4:
        raise SystemExit(f"found {len(geese)} geese in {args.photo}, need 4")
    print(f"cut {len(geese)} geese out of {args.photo.name}")

    sky = (sky_terms(W, H) @ coef).reshape(H, W, 3)
    sprites = {}
    for name, idx, left, width, *_ in BIRDS:
        colour, alpha = geese[idx]
        size = (width, int(round(alpha.shape[0] * width / alpha.shape[1])))
        colour = cv2.resize(colour, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        if left:
            colour, alpha = colour[:, ::-1], alpha[:, ::-1]
        sprites[name] = (colour, alpha, float(alpha.sum()))

    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(0)
    rows = []
    for f in range(FPS * SECONDS):
        t = f / FPS
        layers = {}
        for name, _, _, _, start, path in BIRDS:
            if t >= start:
                layers[name] = place(*sprites[name][:2], *path(t))
        frame = sky.copy()
        for name in ORDER:
            if name in layers:
                c, a = layers[name]
                frame = frame * (1 - a[..., None]) + c * a[..., None]
        for i, name in enumerate(ORDER):
            if name not in layers:
                continue
            a = layers[name][1].copy()
            for later in ORDER[i + 1:]:
                if later in layers:
                    a *= 1 - layers[later][1]
            ys, xs = np.nonzero(layers[name][1] > 0.5)
            if len(xs):
                rows.append([f + 1, name, xs.min(), ys.min(), xs.max() + 1, ys.max() + 1,
                             round(float(a.sum()) / sprites[name][2], 3)])
        frame += rng.normal(0, 2, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()

    truth = args.out.with_name(args.out.stem + "_truth.csv")
    with open(truth, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["frame", "bird", "l", "t", "r", "b", "visible"])
        out.writerows(rows)
    print(f"saved {args.out}  ({SECONDS} s, {W}x{H}, {FPS} fps)\n      {truth}")


if __name__ == "__main__":
    main()
