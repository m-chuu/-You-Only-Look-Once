"""
Dump tiny-YOLOv2's convolution weights to a raw binary file for hardware use.

Reads cfg/yolov2-tiny.cfg and yolov2-tiny.weights directly, folds each
batch-norm into the convolution that precedes it, converts every weight and
bias to half precision, and writes them nose-to-tail into a flat .bin with no
header - the same format 98_YOLOv3_ori/06_Export_Weights_Bin.py produces.

That sibling script gets its already-fused tensors from an ONNX export. There
is no ONNX file for this Darknet model and building one would mean installing
PyTorch on a CPU-only VM, so the fold is done here instead. The output format
is unchanged.

A manifest is written alongside, recording each tensor's byte offset, shape
and element count, because a headerless binary is unreadable without one.

Requires numpy only.

Usage
-----
    python3 02_Export_Weights_Bin.py [--dtype float16] [--no-verify]

Output
------
    yolov2-tiny_weights_fp16.bin     raw little-endian half-precision weights
    yolov2-tiny_weights_fp16.csv     offset / shape / count per tensor
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import yolov2_darknet as dn

# -------------------------------------------------
# Layout notes
# -------------------------------------------------
# Weights stay in Darknet's native [out_ch, in_ch, kh, kw] order, flattened
# row-major - the same order PyTorch uses, which is why the YOLOv3 .bin and
# this one agree without either being transposed. A real accelerator rarely
# wants this order: HWAC, for instance, interleaves 4 channels per row to feed
# 4 parallel multipliers, and expects a .npy of per-layer (N, 4) arrays rather
# than a flat file. That reordering is datapath-specific and belongs in a later
# pass once a board is chosen. See ../FPGA_SUITABILITY_YOLOV2.md.

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default=SCRIPT_DIR / "cfg" / "yolov2-tiny.cfg", type=Path)
    ap.add_argument("--weights", default=SCRIPT_DIR / "yolov2-tiny.weights", type=Path)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--no-verify", action="store_true",
                    help="skip reading the .bin back through the manifest")
    args = ap.parse_args()

    for required in (args.cfg, args.weights):
        if not required.exists():
            sys.exit(f"missing required file: {required}")

    dtype = np.dtype(args.dtype)
    suffix = "fp16" if dtype == np.float16 else "fp32"
    stem = f"yolov2-tiny_weights_{suffix}"
    bin_path = SCRIPT_DIR / (stem + ".bin")
    csv_path = SCRIPT_DIR / (stem + ".csv")

    blocks = dn.parse_cfg(args.cfg)
    spec = dn.describe_network(blocks)
    plan = dn.load_folded_weights(dn.layer_plan(blocks), args.weights)

    print(f"network {spec['width']}x{spec['height']}, {spec['num_classes']} classes, "
          f"{spec['num_anchors']} anchors")

    tensors = dn.conv_tensors(plan)

    offset = 0
    rows = []
    with open(bin_path, "wb") as fh:
        for i, (role, name, arr) in enumerate(tensors):
            flat = arr.astype(dtype).ravel()
            # tofile writes native byte order; x86 and ARM are both little-endian.
            flat.tofile(fh)
            rows.append({
                "index": i,
                "role": role,
                "name": name,
                "shape": "x".join(str(d) for d in arr.shape),
                "count": flat.size,
                "byte_offset": offset,
                "byte_length": flat.nbytes,
            })
            offset += flat.nbytes

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["count"] for r in rows)
    print(f"\n{len(rows)} tensors, {total:,} values, {offset:,} bytes ({offset / 1e6:.1f} MB)")
    print("wrote", bin_path)
    print("wrote", csv_path)

    for row in rows:
        print(f"  [{row['index']:2d}] {row['name']:<16} {row['role']:<6} "
              f"{row['shape']:<16} {row['count']:>9,} @ {row['byte_offset']:>10,}")

    if not args.no_verify:
        verify(bin_path, csv_path, tensors, dtype)


def verify(bin_path, csv_path, tensors, dtype):
    """Read the .bin back using only the manifest and compare to the source."""
    print("\nverifying")

    size = bin_path.stat().st_size
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    declared = sum(int(r["byte_length"]) for r in rows)
    if declared != size:
        sys.exit(f"FAIL manifest declares {declared:,} bytes, file holds {size:,}")
    print(f"  manifest byte_length sums to {declared:,}, matching the file exactly")

    if len(rows) != len(tensors):
        sys.exit(f"FAIL manifest has {len(rows)} rows for {len(tensors)} tensors")

    blob = np.fromfile(bin_path, dtype=np.uint8)

    # fp16 cannot represent everything float32 can. Values above 65504 become
    # infinity, which would corrupt the model; values below the smallest normal
    # round to zero, which is harmless. Report them separately - a single
    # relative-error figure conflates the two and reads as 1.0 whenever any
    # weight underflows, however tiny it was.
    smallest_normal = float(np.finfo(np.float16).tiny) if dtype == np.float16 else 0.0

    total = 0
    overflowed = 0
    flushed = 0
    max_abs = 0.0
    worst_rel = 0.0

    for row, (_role, name, arr) in zip(rows, tensors):
        if row["name"] != name:
            sys.exit(f"FAIL manifest row {row['index']} names {row['name']}, expected {name}")

        start = int(row["byte_offset"])
        stop = start + int(row["byte_length"])
        shape = tuple(int(d) for d in row["shape"].split("x"))
        restored = blob[start:stop].view(dtype).reshape(shape).astype(np.float64)

        expected = arr.astype(dtype).astype(np.float64)
        if not np.array_equal(restored, expected):
            sys.exit(f"FAIL {name} does not round-trip through the manifest")

        source = arr.astype(np.float64)
        finite = np.isfinite(restored)

        total += source.size
        overflowed += int(np.count_nonzero(~finite))
        flushed += int(np.count_nonzero((source != 0) & (restored == 0)))

        diff = np.abs(np.where(finite, restored, 0.0) - source)
        max_abs = max(max_abs, float(diff.max()))

        # Relative error is only meaningful where fp16 can hold the magnitude.
        representable = np.abs(source) >= smallest_normal
        if representable.any():
            worst_rel = max(worst_rel, float(
                (diff[representable] / np.abs(source[representable])).max()))

    print(f"  all {len(rows)} tensors round-trip byte-for-byte via the manifest")
    print(f"  {total:,} values: {overflowed} overflowed to inf, "
          f"{flushed:,} flushed to zero (under {smallest_normal:.2e})")
    print(f"  max absolute error {max_abs:.2e}, "
          f"max relative error {worst_rel:.2e} over representable values")

    if overflowed:
        sys.exit(f"FAIL {overflowed} values exceed the {dtype.name} range - "
                 f"the model would be corrupted on hardware")

    print("PASS")


if __name__ == "__main__":
    main()
