# 95_HWAC_Bird_Train — a bird weight file for the LogikHaus SoC

Trains tiny-YOLOv2 to find birds and packs it into the exact files the board's
AI core loads: `weight_file.bin` and `padding_file.bin`. Runs on the laptop's
GPU (RTX 5050), Windows, no Darknet needed.

The board's core is a port of HWAC (DAC 2018). Its weight layout was decoded
from HWAC's own weights; see `hwac_format.py` for the format and
`00_Selftest.py` for the proof.

## Steps

| Step | Command | What it does |
|---|---|---|
| 0 | `python 00_Selftest.py` | Re-packs the stock `sw1/weight_file.bin` byte for byte and finds the eagle in `board_test/T1.bin`. Must print PASS. |
| 1 | `python 01_Pseudo_Label.py` | Labels the 3,638 BirdsFlying photos with the YOLOv3 model (`98_YOLOv3_ori/best.pt`). Keeps 1,921. |
| 2 | `python 02_Train.py` | Trains on the GPU, starting from COCO tiny-YOLOv2. Writes `runs/bird_v1/`. |
| 3 | `python 03_Export.py` | Packs `runs/bird_v1/best.pt` into `output/weight_file.bin` + `output/padding_file.bin`, checks them against the trained network, and writes `output/report.txt`. |

## What the training data looks like

Every sample is shaped like the board's own picture: a 16:9 scene squeezed to
416×256, sitting in a 416×416 frame with grey bars (that is what the padding
file is). Birds come from `100_BirdImages` (labelled) and the BirdsFlying
close-ups (shrunk so they also look like distant birds). Photos of people
(`archive/Humans`), painted-on watermarks and screen icons are included with no
labels, so the model learns they are not birds. Red and blue are swapped at
random because the board's colour order is not confirmed.

## Loading the result onto the board

With `lh-dashboard-P4.5.pyw`, AI Control tab:

1. **Clear** (AI off).
2. File `weight_file.bin` → **Browse** to `output\weight_file.bin` → **UPLOAD TO SD** (~33 min at 115200 baud).
3. File `padding_file.bin` → **Browse** to `output\padding_file.bin` → **UPLOAD TO SD** (~30 s).
4. **LOAD FROM SD TO RAM**, wait for SUCCESS, then **RUN AI DETECTION**.

Both files must come from the same export — the padding file is computed from
the weights. To return to the stock model, upload `sw1\weight_file.bin` and
`sw1\padding_file.bin` the same way.

## Limits

- The core always draws its single best box, even when that box scores low.
  The new model scores birds high and people, logos and empty sky low, but
  without a confidence cut-off on the board a low-scoring box can still appear.
- tiny-YOLOv2 on a 13×13 grid struggles with very small, distant birds.
