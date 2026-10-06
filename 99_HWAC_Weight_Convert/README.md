# HWAC weight converter

Turns HWAC's published `weights95tuned.npy` into the flat binary the FPGA
loads, and proves the result is the file already in use.

One script, numpy the only dependency.

```bash
python3 01_Convert_NPY_To_Bin.py
```

Downloads the `.npy`, converts it, and checks the output against a known
SHA-256. Expect `PASS`.

To confirm it matches a weight file you already have:

```bash
python3 01_Convert_NPY_To_Bin.py --compare ../weight_file.bin
```

Offline, if you already have the `.npy`:

```bash
python3 01_Convert_NPY_To_Bin.py --npy /path/to/weights95tuned.npy
```

## Output

| File | |
|---|---|
| `weights95tuned_fp16.bin` | 22,542,352 bytes — flat little-endian fp16, no header |
| `weights95tuned_fp16.csv` | per-layer byte offsets |

`weights95tuned_fp16.bin` is **byte-identical to `weight_file.bin`**
(sha256 `d8bd177c…6df6`). Rename or copy it to drop into an existing setup.

## Where the weights come from

| | |
|---|---|
| Repository | https://github.com/HWAC-DL/hwac_object_tracker |
| Path | `py/overlay/hwac_object_tracker/weights95tuned.npy` |
| Committed | 2018-07-31, "first commit", author Duvindu |
| Git blob | `3df90106e5640f33f694fa1db0bde14575c04ad2` |
| Size | 33,846,808 bytes (`.npy` container) |

**These weights were downloaded, not trained here.** They are the HWAC team's
entry for the DAC 2018 System Design Contest — a 95-class model over drone
imagery, which they measured at average IoU 0.514 and 4.91 FPS on a PYNQ-Z1.
They have never seen bird training data.

That matters for interpreting results. The contest scored *localisation only*,
by IoU against a single ground-truth box per image — HWAC's own label files are
`<class> <cx> <cy> <w> <h>`, one line each, and their results legend reads only
"Green Line - FPGA infered Bounding Box / Blue Line - Actual Bounding Box". No
class is displayed because none is needed. So this model finds salient objects
and boxes them. A bird against open sky is an easy salient object; being boxed
is not evidence of bird recognition.

## The format

Nine per-layer `float16` arrays of shape `(N, 4)`, concatenated in layer order,
row-major. Four **input channels per row**, feeding the four parallel
multipliers in `conv_out_channel`. Row counts follow:

```
rows = (F/4) x (2 + (Cp/4) x 4 x k*k)        Cp = in_channels rounded up to 4
```

The script checks every layer against this and refuses to write if any
disagrees.

| layer | filters | in | k | rows |
|---|---|---|---|---|
| 1 | 16 | 3 | 3 | 152 |
| 2 | 32 | 16 | 3 | 1,168 |
| 3 | 64 | 32 | 3 | 4,640 |
| 4 | 128 | 64 | 3 | 18,496 |
| 5 | 256 | 128 | 3 | 73,856 |
| 6 | 512 | 256 | 3 | 295,168 |
| 7 | 1024 | 512 | 3 | 1,180,160 |
| 8 | 512 | 1024 | 3 | 1,179,904 |
| 9 | 500 | 512 | 1 | 64,250 |

Two corroborations that this is right. The cumulative row offsets — 0, 152,
1320, 5960, 24456, 98312, 393480, 1573640, 2753544 — are exactly the weight
addresses hardcoded in HWAC's `hwac.py`. And layer 9's 500 filters is
`FC_FLT_CNT = 500` in that same file: 5 anchors x (4 + 1 + 95 classes).

Layer 1 has three input channels padded to four, which is why every fourth
value in the first 608 is zero.

## What this does not do

It does not convert *your* model into this format. Going the other way needs
two things this repository does not yet have:

1. **The meaning of the two extra rows per group of four filters.** They are
   loaded into `norm_param_a` / `norm_param_b` by `conv_stream.v`, and
   `normalization.v` computes `a x conv + b`. But roughly half the row-0 values
   are negative in every layer, which a batch-norm scale `gamma/sqrt(var)`
   would not be, and reconstructing the network with that assumption makes
   activations collapse. Something about the semantics is still wrong.
2. **An answer on the output width.** The design expects 500 filters. A
   single-class bird detector needs 30. Whether that is a Verilog parameter
   that can be changed and resynthesised decides whether a bird model is
   retrained as 1 class or padded to 95.

Until both are settled, a converted custom model would be untested guesswork —
and an untested weight file costs a bring-up cycle to disprove.
