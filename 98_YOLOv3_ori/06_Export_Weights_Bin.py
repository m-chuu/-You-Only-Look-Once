"""
Dump the model's convolution weights to a raw binary file for hardware use.

Reads best.onnx (already fused - batch-norm folded into the conv weights and
biases by the ONNX export), walks the graph in execution order, converts every
Conv weight and bias to half precision, and writes them nose-to-tail into a
flat .bin with no header, in the style of HWAC's weights95tuned.npy.

A manifest is written alongside it recording each tensor's byte offset, shape
and element count, because a headerless binary is unreadable without one.

Usage
-----
    python 06_Export_Weights_Bin.py [--onnx best.onnx] [--dtype float16]

Output
------
    best_weights_fp16.bin        raw little-endian half-precision weights
    best_weights_fp16.csv        offset / shape / count per tensor
"""

import argparse
import csv
import os

import numpy as np
import onnx
from onnx import numpy_helper

# -------------------------------------------------
# Layout notes
# -------------------------------------------------
# Weights stay in PyTorch's native [out_ch, in_ch, kh, kw] order, flattened
# row-major. A real accelerator almost never wants this order - HWAC, for
# instance, interleaves 4 channels per row to feed 4 parallel multipliers. Any
# reordering is datapath-specific, so it belongs in a later pass once the
# hardware exists, not here.


def collect_conv_tensors(graph):
    """Conv weights and biases, in graph execution order."""
    initializers = {i.name: i for i in graph.initializer}
    tensors = []

    for node in graph.node:
        if node.op_type != "Conv":
            continue
        # input[1] is always the kernel; input[2] is the bias when present.
        for role, idx in (("weight", 1), ("bias", 2)):
            if len(node.input) <= idx:
                continue
            init = initializers.get(node.input[idx])
            if init is None:
                continue
            tensors.append((role, node.input[idx], numpy_helper.to_array(init)))

    return tensors


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=os.path.join(script_dir, "best.onnx"))
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    dtype = np.dtype(args.dtype)
    stem = os.path.join(script_dir, f"best_weights_{'fp16' if dtype == np.float16 else 'fp32'}")
    bin_path, csv_path = stem + ".bin", stem + ".csv"

    tensors = collect_conv_tensors(onnx.load(args.onnx).graph)

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
    print(f"{len(rows)} tensors, {total:,} values, {offset:,} bytes ({offset/1e6:.1f} MB)")
    print("wrote", bin_path)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
