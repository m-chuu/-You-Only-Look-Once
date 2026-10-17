"""
Predict the box the board will draw on a picture: stock files vs new files.

    python 04_Try_Picture.py photo.jpg
    python 04_Try_Picture.py camera_screenshot.png --camera-shot

The picture is squeezed to 416x256 the way the board does, then run through
BoardNet (the software model of the board) with the stock files in ../sw1 and
the new files in output/. The board draws only ONE box per frame, the
highest-scoring one (#1, thick red). Other candidates scoring above 0.3 are
numbered in yellow: when their scores are close to #1, the board's box can
jump between them from frame to frame.

--camera-shot is for screenshots of the camera view: it cuts out the picture
from the dark window around it and removes the red box the board drew, so
that box cannot influence the check.

Writes <name>_check.jpg (picture) and <name>_table.txt (spot/score table)
next to the picture.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

import bird_model as bm
import hwac_format as hw

HERE = Path(__file__).resolve().parent


def cut_out_picture(img):
    """The camera picture is the largest bright area in the screenshot."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright = (grey > 40).astype(np.uint8)
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    x, y, w, h, _ = stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    x1, y1 = x + w, y + h
    # drop edge rows/columns that are mostly dark (window frame, small icons at the edge)
    while x1 - x > 10 and bright[y:y1, x].mean() < 0.5:
        x += 1
    while x1 - x > 10 and bright[y:y1, x1 - 1].mean() < 0.5:
        x1 -= 1
    while y1 - y > 10 and bright[y, x:x1].mean() < 0.5:
        y += 1
    while y1 - y > 10 and bright[y1 - 1, x:x1].mean() < 0.5:
        y1 -= 1
    return img[y:y1, x:x1]


def remove_board_box(img, min_line=20):
    """Paint over the red rectangle the board's overlay drew.

    Only long straight red lines count, so red or orange parts of the scene
    (birds, flowers) are left alone.
    """
    b, g, r = cv2.split(img)
    red = ((r > 170) & (g < 90) & (b < 90)).astype(np.uint8)
    lines = (cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((1, min_line), np.uint8)) |
             cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((min_line, 1), np.uint8)))
    if lines.sum() == 0:
        return img, 0
    mask = cv2.dilate(lines, np.ones((5, 5), np.uint8)) * 255
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA), int(lines.sum())


def where(l, r, t, b):
    cx, cy = (l + r) / 2, (t + b) / 2
    col = "left" if cx < hw.IMG_W / 3 else "right" if cx > 2 * hw.IMG_W / 3 else "centre"
    row = "top" if cy < hw.IMG_H / 3 else "bottom" if cy > 2 * hw.IMG_H / 3 else "middle"
    return "centre" if (row, col) == ("middle", "centre") else f"{row}-{col}"


def candidates(net, x, dev, dtype, min_score=0.3):
    """All boxes above min_score after removing duplicates, best first (#1 = board's box)."""
    with torch.no_grad():
        head = net(x.to(dev, dtype)).float()
    boxes, scores = bm.decode(head[:, bm.HWAC_TO_25])
    b, s = boxes[0].cpu(), scores[0].cpu()
    keep = torchvision.ops.nms(b, s, 0.3)
    top = hw.top_box(head[0].cpu())
    found = [top]
    for k in keep:
        l, t, r, bt = b[k].tolist()
        if s[k] > min_score and not (abs(l - top["picture"][0]) < 1 and abs(t - top["picture"][2]) < 1):
            sx, sy = hw.DISPLAY_W / hw.IMG_W, hw.DISPLAY_H / hw.IMG_H
            found.append({"objectness": float(s[k]), "picture": (l, r, t, bt),
                          "display": tuple(int(round(v)) for v in (l * sx, r * sx, t * sy, bt * sy))})
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picture", type=Path)
    ap.add_argument("--camera-shot", action="store_true")
    ap.add_argument("--weights", type=Path, default=HERE / "output" / "weight_file.bin")
    ap.add_argument("--padding", type=Path, default=HERE / "output" / "padding_file.bin")
    args = ap.parse_args()

    img = cv2.imread(str(args.picture))
    if img is None:
        raise SystemExit(f"cannot read {args.picture}")
    if args.camera_shot:
        img = cut_out_picture(img)
        img, n_red = remove_board_box(img)
        print(f"camera picture {img.shape[1]}x{img.shape[0]}, board's red box removed ({n_red} pixels)")
    pic = cv2.resize(img, (hw.IMG_W, hw.IMG_H), interpolation=cv2.INTER_AREA)
    rgb = torch.from_numpy(np.ascontiguousarray(pic[:, :, ::-1].transpose(2, 0, 1))).float().div(255)[None]

    nets = {"stock HWAC": hw.BoardNet.from_files(HERE.parent / "sw1" / "weight_file.bin",
                                                 HERE.parent / "sw1" / "padding_file.bin"),
            "new bird model": hw.BoardNet.from_files(args.weights, args.padding)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    panels, table = [], []
    for name, net in nets.items():
        net = net.to(dev).eval()
        found = candidates(net, rgb, dev, dtype)
        bgr_top = candidates(net, rgb[:, [2, 1, 0]], dev, dtype)[0]
        canvas = cv2.resize(pic, (hw.IMG_W * 2, hw.IMG_H * 2), interpolation=cv2.INTER_LINEAR)
        table += [f"{name}  (board draws #1 only; if the board sent BGR colours, #1 would score "
                  f"{bgr_top['objectness']:.2f} at {where(*bgr_top['picture'])})",
                  "  #  | Score | Spot          | Box on screen (L R T B, 1920x1080)",
                  "  ---|-------|---------------|-----------------------------------"]
        for i, c in enumerate(found, 1):
            l, r, t, b = c["picture"]
            table.append(f"  {i:<3d}| {c['objectness']:.2f}  | {where(l, r, t, b):13s} | "
                         f"L{c['display'][0]} R{c['display'][1]} T{c['display'][2]} B{c['display'][3]}")
            colour, thick = ((0, 0, 255), 3) if i == 1 else ((0, 220, 255), 1)
            p = [int(v * 2) for v in (l, t, r, b)]
            cv2.rectangle(canvas, (p[0], p[1]), (p[2], p[3]), colour, thick)
            cv2.putText(canvas, f"#{i} {c['objectness']:.2f}", (p[0], max(14, p[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2 if i == 1 else 1, cv2.LINE_AA)
        table.append("")
        cv2.putText(canvas, f"{name}: board box #1 = {found[0]['objectness']:.2f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(canvas)

    out_img = args.picture.with_name(args.picture.stem + "_check.jpg")
    out_tab = args.picture.with_name(args.picture.stem + "_table.txt")
    cv2.imwrite(str(out_img), np.vstack(panels))
    out_tab.write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"saved {out_img}\n      {out_tab}")


if __name__ == "__main__":
    main()
