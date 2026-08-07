"""
Convert HWAC's published weights95tuned.npy into a flat binary weight file.

Source
------
    https://github.com/HWAC-DL/hwac_object_tracker
    py/overlay/hwac_object_tracker/weights95tuned.npy
    committed 2018-07-31 ("first commit", author Duvindu)

The .npy is an object array of nine per-layer float16 arrays, each (N, 4) -
four input channels per row, matching the four parallel multipliers in the
accelerator's conv_out_channel. Concatenating them row-major in layer order
gives the flat binary the board loads.

The result is byte-identical to the weight_file.bin already in use, which this
script verifies against a known SHA-256 rather than asking you to trust it.

Requires numpy. The download uses the standard library.

Usage
-----
    python3 01_Convert_NPY_To_Bin.py                  # download, convert, verify
    python3 01_Convert_NPY_To_Bin.py --npy local.npy  # use a local copy
    python3 01_Convert_NPY_To_Bin.py --compare a.bin  # also diff against a file

Output
------
    weights95tuned_fp16.bin   flat little-endian fp16, no header
    weights95tuned_fp16.csv   per-layer byte offsets
"""

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

URL = ("https://raw.githubusercontent.com/HWAC-DL/hwac_object_tracker/"
       "master/py/overlay/hwac_object_tracker/weights95tuned.npy")

# Verified 2026-08. The .npy container is 33,846,808 bytes; the flattened
# fp16 payload below is what the board actually consumes.
EXPECTED_SHA256 = "d8bd177ca902ec05de0f217c8bd6b12286bb8e9aca81ba1479699282eecc6df6"
EXPECTED_BYTES = 22542352

# tiny-YOLOv2 as HWAC synthesised it: 500 output filters = 5 anchors x (4+1+95).
LAYERS = [
    # (out_filters, in_channels, kernel)
    (16, 3, 3), (32, 16, 3), (64, 32, 3), (128, 64, 3), (256, 128, 3),
    (512, 256, 3), (1024, 512, 3), (512, 1024, 3), (500, 512, 1),
]


def expected_rows(out_ch, in_ch, kernel):
    """Rows HWAC's format uses for one layer.

    Two parameter rows per group of four filters, then one row per
    (channel-group, filter, tap) with four input channels packed per row.
    Input channels are padded up to a multiple of four - which is why layer 1,
    with three channels, has a zero in every fourth value.
    """
    padded_in = -(-in_ch // 4) * 4
    return (out_ch // 4) * (2 + (padded_in // 4) * 4 * kernel * kernel)


def fetch(npy_path):
    if npy_path.exists():
        print(f"using existing {npy_path.name} ({npy_path.stat().st_size:,} bytes)")
        return
    print(f"downloading {URL}")
    try:
        urllib.request.urlretrieve(URL, npy_path)
    except Exception as exc:
        sys.exit(f"download failed: {exc}\n"
                 f"Fetch it manually and pass --npy <path>.")
    print(f"saved {npy_path.name} ({npy_path.stat().st_size:,} bytes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy", type=Path, default=SCRIPT_DIR / "weights95tuned.npy")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "weights95tuned_fp16.bin")
    ap.add_argument("--compare", type=Path,
                    help="also compare the result against an existing .bin")
    args = ap.parse_args()

    fetch(args.npy)

    arrays = np.load(args.npy, allow_pickle=True)
    if len(arrays) != len(LAYERS):
        sys.exit(f"expected {len(LAYERS)} layers, found {len(arrays)}")

    print(f"\n{len(arrays)} per-layer arrays")
    print(f"{'layer':>5}  {'filters':>7} {'in_ch':>6} {'k':>2}  "
          f"{'shape':>14}  {'rows':>9}  {'expected':>9}")

    rows_meta = []
    offset = 0
    chunks = []

    for i, (arr, (out_ch, in_ch, kernel)) in enumerate(zip(arrays, LAYERS), start=1):
        if arr.ndim != 2 or arr.shape[1] != 4:
            sys.exit(f"layer {i} has shape {arr.shape}, expected (N, 4)")

        want = expected_rows(out_ch, in_ch, kernel)
        flag = "" if arr.shape[0] == want else "   <-- MISMATCH"
        print(f"{i:>5}  {out_ch:>7} {in_ch:>6} {kernel:>2}  {str(arr.shape):>14}  "
              f"{arr.shape[0]:>9,}  {want:>9,}{flag}")
        if flag:
            sys.exit(f"layer {i} row count does not match the documented format")

        flat = arr.astype(np.float16).ravel()
        chunks.append(flat)
        rows_meta.append({
            "layer": i,
            "filters": out_ch,
            "in_channels": in_ch,
            "kernel": kernel,
            "rows": arr.shape[0],
            "values": flat.size,
            "byte_offset": offset,
            "byte_length": flat.nbytes,
        })
        offset += flat.nbytes

    payload = np.concatenate(chunks)
    payload.tofile(args.out)

    with open(args.out.with_suffix(".csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_meta[0].keys()))
        writer.writeheader()
        writer.writerows(rows_meta)

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()

    print(f"\nwrote {args.out.name}  {offset:,} bytes ({payload.size:,} fp16 values)")
    print(f"wrote {args.out.with_suffix('.csv').name}")
    print(f"sha256 {digest}")

    print("\nverifying against the published reference")
    ok = True
    if offset != EXPECTED_BYTES:
        print(f"  FAIL size {offset:,}, expected {EXPECTED_BYTES:,}")
        ok = False
    else:
        print(f"  size   {offset:,} bytes as expected")
    if digest != EXPECTED_SHA256:
        print(f"  FAIL sha256 mismatch\n       expected {EXPECTED_SHA256}")
        ok = False
    else:
        print(f"  sha256 matches the published file")

    if args.compare:
        other = args.compare.read_bytes()
        same = other == args.out.read_bytes()
        print(f"  {args.compare.name}: {len(other):,} bytes, "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
        ok = ok and same

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
