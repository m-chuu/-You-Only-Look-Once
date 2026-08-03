"""
Prove the exported .bin is correct by detecting with it and comparing against
darknet's own output.

Weights are read back out of yolov2-tiny_weights_fp16.bin using nothing but
the CSV manifest - no .weights file - so a wrong order, a wrong offset or a
botched batch-norm fold all show up as disagreeing detections rather than
passing silently.

Run 01_Detect_Images.py first; this compares against the darknet_detections.json
it leaves behind.

Requires numpy and Pillow (Pillow only to decode JPEG/PNG). Inference is pure
NumPy and takes a few seconds per image - it exists to check the export, not
to be fast.

Usage
-----
    python3 03_Verify_Weights.py [--source bin|weights] [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import yolov2_darknet as dn

SCRIPT_DIR = Path(__file__).resolve().parent


def load_plan_from_bin(blocks, bin_path, csv_path):
    """Rebuild the layer stack using only the .bin and its manifest."""
    import csv as csv_module

    plan = dn.layer_plan(blocks)
    with open(csv_path, newline="") as fh:
        rows = list(csv_module.DictReader(fh))

    blob = np.fromfile(bin_path, dtype=np.uint8)
    declared = sum(int(r["byte_length"]) for r in rows)
    if declared != blob.size:
        sys.exit(f"manifest declares {declared:,} bytes, {bin_path.name} holds {blob.size:,}")

    dtype = np.float16 if "fp16" in bin_path.name else np.float32
    convs = [layer for layer in plan if layer["type"] == "conv"]

    if len(rows) != 2 * len(convs):
        sys.exit(f"manifest has {len(rows)} rows, expected {2 * len(convs)}")

    for i, layer in enumerate(convs):
        layer["name"] = f"conv_{i + 1}"
        for role, row in (("weight", rows[2 * i]), ("bias", rows[2 * i + 1])):
            if row["role"] != role:
                sys.exit(f"manifest row {row['index']} is {row['role']}, expected {role}")
            start = int(row["byte_offset"])
            stop = start + int(row["byte_length"])
            shape = tuple(int(d) for d in row["shape"].split("x"))
            layer[role] = blob[start:stop].view(dtype).reshape(shape).astype(np.float32)

        expected = (layer["out_ch"], layer["in_ch"], layer["size"], layer["size"])
        if layer["weight"].shape != expected:
            sys.exit(f"conv_{i + 1} weight is {layer['weight'].shape}, cfg says {expected}")

    print(f"rebuilt {len(convs)} convolutions from {bin_path.name} "
          f"using only {csv_path.name}")
    return plan


def load_image(path):
    """Decode to float32 RGB in 0..1, matching Darknet's own loader."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow is needed to decode images: pip3 install --user Pillow")

    with Image.open(path) as handle:
        rgb = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    return rgb


def detect(plan, spec, names, image_path, thresh, nms_thresh):
    rgb = load_image(image_path)
    orig_h, orig_w = rgb.shape[:2]

    canvas = dn.letterbox(rgb, spec["width"], spec["height"])
    head = dn.forward(plan, np.ascontiguousarray(canvas.transpose(2, 0, 1)))

    boxes, objectness, class_probs = dn.decode_region(head, spec)
    boxes = dn.correct_boxes(boxes, orig_w, orig_h, spec["width"], spec["height"])

    found = dn.detections(boxes, objectness, class_probs, thresh, nms_thresh)
    for item in found:
        item["class"] = names[item["class_id"]]
        item["percent"] = int(round(item["prob"] * 100))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="bin", choices=["bin", "weights"],
                    help="rebuild from the exported .bin (default) or the original .weights")
    ap.add_argument("--bin", default=SCRIPT_DIR / "yolov2-tiny_weights_fp16.bin", type=Path)
    ap.add_argument("--csv", default=SCRIPT_DIR / "yolov2-tiny_weights_fp16.csv", type=Path)
    ap.add_argument("--cfg", default=SCRIPT_DIR / "cfg" / "yolov2-tiny.cfg", type=Path)
    ap.add_argument("--weights", default=SCRIPT_DIR / "yolov2-tiny.weights", type=Path)
    ap.add_argument("--names", default=SCRIPT_DIR / "data" / "coco.names", type=Path)
    ap.add_argument("--images", default=SCRIPT_DIR / "images", type=Path)
    ap.add_argument("--reference", default=SCRIPT_DIR / "darknet_detections.json", type=Path)
    ap.add_argument("--nms", default=0.45, type=float)
    ap.add_argument("--limit", type=int, help="only check the first N images")
    args = ap.parse_args()

    if not args.reference.exists():
        sys.exit(f"{args.reference.name} not found - run 01_Detect_Images.py first")

    reference = json.loads(args.reference.read_text())
    thresh = reference["thresh"]
    expected_by_image = reference["images"]

    blocks = dn.parse_cfg(args.cfg)
    spec = dn.describe_network(blocks)
    names = dn.load_names(args.names)

    if args.source == "bin":
        plan = load_plan_from_bin(blocks, args.bin, args.csv)
    else:
        plan = dn.load_folded_weights(dn.layer_plan(blocks), args.weights)

    image_names = sorted(expected_by_image)
    if args.limit:
        image_names = image_names[:args.limit]

    print(f"\nchecking {len(image_names)} images at thresh {thresh}, nms {args.nms}\n")

    agreed = 0
    disagreed = []

    for image_name in image_names:
        path = args.images / image_name
        if not path.exists():
            print(f"  {image_name}: SKIPPED (not found)")
            continue

        found = detect(plan, spec, names, path, thresh, args.nms)

        ours = sorted((d["class"], d["percent"]) for d in found)
        theirs = sorted((d["class"], d["percent"]) for d in expected_by_image[image_name])

        our_classes = sorted(c for c, _ in ours)
        their_classes = sorted(c for c, _ in theirs)

        if our_classes != their_classes:
            status = "CLASS MISMATCH"
            disagreed.append(image_name)
        else:
            drift = max((abs(a[1] - b[1]) for a, b in zip(ours, theirs)), default=0)
            if drift <= 2:
                status = f"ok (max {drift}% confidence drift)"
                agreed += 1
            else:
                status = f"CONFIDENCE DRIFT {drift}%"
                disagreed.append(image_name)

        print(f"  {image_name}: {status}")
        if our_classes != their_classes:
            print(f"      darknet: {theirs}")
            print(f"      ours   : {ours}")

    print(f"\n{agreed}/{len(image_names)} images agree with darknet")

    if disagreed:
        print("\nDisagreements:")
        for name in disagreed:
            print(f"  {name}")
        print("\nFAIL - the exported weights do not reproduce darknet's detections")
        sys.exit(1)

    print("PASS - the .bin reproduces darknet's detections")


if __name__ == "__main__":
    main()
