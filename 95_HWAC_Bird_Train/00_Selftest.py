"""
Proves the format code before anything is trained on it.

1. Unpacking and re-packing the stock weight_file.bin gives the same bytes.
2. Same for padding_file.bin.
3. The stock weights, run through BoardNet, find the eagle in T1.bin and
   nothing in the blank T4.bin - the same answers as the earlier analysis.

    python 00_Selftest.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

import hwac_format as hw

HERE = Path(__file__).resolve().parent
SW1 = HERE.parent / "sw1"
BOARD_TEST = HERE.parent / "99_HWAC_Weight_Convert" / "board_test"


def main():
    ok = True

    stock_w = SW1 / "weight_file.bin"
    layers = hw.read_weight_file(stock_w)
    again = hw.weight_file_bytes(layers)
    same = again == stock_w.read_bytes()
    print(f"1. weight file round trip: {'PASS' if same else 'FAIL'}  sha256 {hw.sha256(again)[:16]}...")
    ok &= same and hw.sha256(again) == hw.STOCK_WEIGHT_SHA256

    stock_p = SW1 / "padding_file.bin"
    border = hw.read_padding_file(stock_p)
    same = hw.padding_file_bytes(border) == stock_p.read_bytes()
    print(f"2. padding file round trip: {'PASS' if same else 'FAIL'}")
    ok &= same

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = hw.BoardNet(layers, border).to(dev).eval()
    with torch.no_grad():
        for name, want in (("T1", "eagle near L213 R249 T79 B155, objectness ~0.96"),
                           ("T4", "nothing: objectness ~0.01")):
            box = hw.top_box(net(hw.load_picture_bin(BOARD_TEST / f"{name}.bin").to(dev))[0].cpu())
            l, r, t, b = box["picture"]
            print(f"3. {name}: objectness {box['objectness']:.3f}  "
                  f"picture L{l:.0f} R{r:.0f} T{t:.0f} B{b:.0f}  display {box['display']}  (expected {want})")
            if name == "T1":
                ok &= box["objectness"] > 0.8 and abs(l - 213) < 6 and abs(t - 79) < 6
            else:
                ok &= box["objectness"] < 0.05

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
