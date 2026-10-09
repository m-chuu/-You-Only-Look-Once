"""
Label the BirdsFlying photos with the existing YOLOv3-tiny bird model.

The 3,638 photos in 98_YOLOv3_ori/bird have no box labels. best.pt
(mAP50 ~0.85 on the bird dataset) draws them. To keep the labels clean:

  - boxes at confidence >= 0.35 are kept
  - a photo with any box between 0.15 and 0.35 is dropped: a half-seen bird
    would otherwise be taught as "not a bird"
  - a photo where the model finds no bird is dropped too - every photo here
    contains one, so that is a miss, not a negative
  - overlapping boxes are merged into one: on big close-ups the model, trained
    on distant birds, often splits a single bird into two boxes

    python 01_Pseudo_Label.py

Writes data/birdsflying/labels/*.txt (YOLO format), data/birdsflying/images.txt
and data/birdsflying/check.jpg - a sample sheet to eyeball.
"""

import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "98_YOLOv3_ori" / "bird"
TEACHER = HERE.parent / "98_YOLOv3_ori" / "best.pt"
OUT = HERE / "data" / "birdsflying"
KEEP, AMBIGUOUS = 0.35, 0.15


def main():
    images = sorted(SRC.glob("*.jpg"))
    print(f"{len(images)} photos, teacher {TEACHER.name}")
    model = YOLO(str(TEACHER))
    print(f"teacher classes: {model.names}")

    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    kept, ambiguous, missed, boxes_total = [], 0, 0, 0
    for i in range(0, len(images), 64):
        chunk = [str(p) for p in images[i:i + 64]]
        for path, r in zip(chunk, model.predict(chunk, imgsz=416, conf=AMBIGUOUS,
                                                verbose=False, device=0)):
            conf = r.boxes.conf.cpu().numpy()
            if ((conf >= AMBIGUOUS) & (conf < KEEP)).any():
                ambiguous += 1
                continue
            xywhn = merge_overlapping(r.boxes.xywhn.cpu().numpy()[conf >= KEEP])
            if len(xywhn) == 0:
                missed += 1
                continue
            lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in xywhn]
            (OUT / "labels" / (Path(path).stem + ".txt")).write_text("\n".join(lines) + "\n")
            kept.append(path)
            boxes_total += len(lines)

    (OUT / "images.txt").write_text("\n".join(kept) + "\n")
    print(f"kept {len(kept)} photos ({boxes_total} boxes), dropped {ambiguous} ambiguous, "
          f"{missed} with no bird found")
    sample_sheet(kept)


def merge_overlapping(xywhn, min_overlap=0.2):
    """Union boxes that overlap by more than min_overlap of the smaller box."""
    boxes = [np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) for cx, cy, w, h in xywhn]
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                iw = min(a[2], b[2]) - max(a[0], b[0])
                ih = min(a[3], b[3]) - max(a[1], b[1])
                smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if iw > 0 and ih > 0 and iw * ih > min_overlap * smaller:
                    boxes[i] = np.array([min(a[0], b[0]), min(a[1], b[1]),
                                         max(a[2], b[2]), max(a[3], b[3])])
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return np.array([[(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
                     for x0, y0, x1, y1 in boxes]).reshape(-1, 4)


def sample_sheet(kept, n=16, tile=224):
    random.seed(0)
    sheet = np.full((tile * 4, tile * 4, 3), 40, np.uint8)
    for i, path in enumerate(random.sample(kept, min(n, len(kept)))):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        for line in (OUT / "labels" / (Path(path).stem + ".txt")).read_text().split("\n"):
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                              (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 0, 255),
                              max(2, w // 150))
        s = tile / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        y, x = (i // 4) * tile, (i % 4) * tile
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.imwrite(str(OUT / "check.jpg"), sheet)
    print(f"sample sheet: {OUT / 'check.jpg'}")


if __name__ == "__main__":
    main()
