"""
Lay out the bird dataset for Darknet training and write a matching cfg.

The source dataset (98_YOLOv3_ori/01_Train_images) is a Roboflow export, and
its label format - "<class> <cx> <cy> <w> <h>" normalised - is already exactly
what Darknet wants. Only the directory layout and the list files are missing.

Why the files are copied rather than listed where they are
---------------------------------------------------------
Darknet finds a label by taking the image path and replacing the FIRST
occurrence of "images" with "labels". The source path contains it twice:

    .../01_Train_images/train/images/bird.jpg
         ^^^^^^^^^^^^^^ this one gets replaced

which yields .../01_Train_labels/train/images/bird.txt - a path that does not
exist, and Darknet would train against empty labels without complaining. So
the data is copied into data/<split>/images and data/<split>/labels, where the
first "images" is the right one.

The cfg is generated here too, because `classes` and the final conv `filters`
have to satisfy filters = anchors x (5 + classes). Editing one and forgetting
the other is the classic way to waste a training run.

Requires nothing outside the standard library.

Usage
-----
    python3 00_Prepare_Dataset.py --dataset ../98_YOLOv3_ori/01_Train_images
    python3 00_Prepare_Dataset.py --classes 95      # HWAC-shaped output
"""

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Stock tiny-YOLOv2 COCO anchors, unchanged - the accelerator's decode is
# hardcoded to these, and they are reasonable for birds anyway.
ANCHORS = ("0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, "
           "7.88282, 3.52778, 9.77052, 9.16828")
NUM_ANCHORS = 5

CFG_TEMPLATE = """[net]
# Training - CPU only, no CUDA in this darknet build.
batch={batch}
subdivisions={subdivisions}
width={size}
height={size}
channels=3
momentum=0.9
decay=0.0005

# Augmentation matters here: only {n_train} training images.
angle=0
saturation=1.5
exposure=1.5
hue=.1
jitter=.2

learning_rate=0.001
burn_in={burn_in}
max_batches={max_batches}
policy=steps
steps={step1},{step2}
scales=.1,.1

[convolutional]
batch_normalize=1
filters=16
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=1

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={filters}
activation=linear

[region]
anchors = {anchors}
bias_match=1
classes={classes}
coords=4
num={num}
softmax=1
jitter=.2
rescore=0

object_scale=5
noobject_scale=1
class_scale=1
coord_scale=1

absolute=1
thresh = .6
# random=1 would retrain at varying input sizes. Left off: on CPU it costs a
# lot of time for little gain on a dataset this small.
random=0
"""


def copy_split(src_root, dst_root, split, out_split):
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    if not src_images.is_dir():
        sys.exit(f"missing {src_images}")

    dst_images = dst_root / out_split / "images"
    dst_labels = dst_root / out_split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    listed = []
    missing = 0
    for image in sorted(src_images.iterdir()):
        if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label = src_labels / (image.stem + ".txt")
        if not label.exists():
            missing += 1
            continue
        shutil.copy2(image, dst_images / image.name)
        shutil.copy2(label, dst_labels / label.name)
        listed.append(dst_images / image.name)

    if missing:
        print(f"  {split}: skipped {missing} images with no label file")
    return listed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path,
                    default=SCRIPT_DIR.parent / "98_YOLOv3_ori" / "01_Train_images")
    ap.add_argument("--classes", type=int, default=1,
                    help="1 (default) trains a clean bird detector; 95 matches "
                         "HWAC's output width directly")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--subdivisions", type=int, default=16,
                    help="higher keeps memory down; irrelevant to CPU speed")
    ap.add_argument("--max-batches", type=int, default=2000,
                    help="2000 at batch 64 is ~450 epochs over 280 images - "
                         "ample for fine-tuning, and CPU training is slow")
    ap.add_argument("--size", type=int, default=416,
                    help="network input, multiple of 32. 320 costs ~40%% less "
                         "compute per iteration than 416")
    args = ap.parse_args()

    if args.size % 32:
        sys.exit(f"--size must be a multiple of 32, got {args.size}")

    if not args.dataset.is_dir():
        sys.exit(f"dataset not found: {args.dataset}\n"
                 f"Copy 98_YOLOv3_ori/01_Train_images across and pass --dataset.")

    data_root = SCRIPT_DIR / "data"
    print(f"source : {args.dataset}")
    print(f"target : {data_root}\n")

    train = copy_split(args.dataset, data_root, "train", "train")
    valid = copy_split(args.dataset, data_root, "valid", "valid")
    print(f"  train: {len(train)} images")
    print(f"  valid: {len(valid)} images")

    if not train:
        sys.exit("no training images found")

    (data_root / "train.txt").write_text(
        "\n".join(str(p.resolve()) for p in train) + "\n")
    (data_root / "valid.txt").write_text(
        "\n".join(str(p.resolve()) for p in valid) + "\n")

    # Darknet reads one name per class. Only class 0 ever appears in the
    # labels; the rest exist so the file length matches `classes`.
    names = ["bird"] + [f"unused_{i}" for i in range(1, args.classes)]
    (data_root / "obj.names").write_text("\n".join(names) + "\n")

    backup = SCRIPT_DIR / "backup"
    backup.mkdir(exist_ok=True)
    (data_root / "obj.data").write_text(
        f"classes = {args.classes}\n"
        f"train = {(data_root / 'train.txt').resolve()}\n"
        f"valid = {(data_root / 'valid.txt').resolve()}\n"
        f"names = {(data_root / 'obj.names').resolve()}\n"
        f"backup = {backup.resolve()}\n")

    filters = NUM_ANCHORS * (5 + args.classes)
    cfg_dir = SCRIPT_DIR / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    cfg_path = cfg_dir / f"yolov2-tiny-bird-{args.classes}c.cfg"
    cfg_path.write_text(CFG_TEMPLATE.format(
        batch=args.batch, subdivisions=args.subdivisions,
        burn_in=min(200, args.max_batches // 10),
        max_batches=args.max_batches,
        step1=int(args.max_batches * 0.8), step2=int(args.max_batches * 0.9),
        filters=filters, classes=args.classes, num=NUM_ANCHORS,
        anchors=ANCHORS, n_train=len(train), size=args.size))

    print(f"\nwrote {data_root/'train.txt'}")
    print(f"wrote {data_root/'valid.txt'}")
    print(f"wrote {data_root/'obj.names'}  ({args.classes} names)")
    print(f"wrote {data_root/'obj.data'}")
    print(f"wrote {cfg_path}")
    print(f"\nclasses={args.classes}  ->  final conv filters="
          f"{NUM_ANCHORS} x (5 + {args.classes}) = {filters}")
    print(f"max_batches={args.max_batches}, steps at "
          f"{int(args.max_batches*0.8)},{int(args.max_batches*0.9)}")
    print("\nNext: create the pretrained backbone, then train. See README.md")


if __name__ == "__main__":
    main()
