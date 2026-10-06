"""
Build the test set that answers one question: does the stock weight file
already put a box on a bird?

Why this comes before any weight conversion
-------------------------------------------
The core on the board emits exactly one bounding box per image, with no class
label - it is a salient-object localiser, not a classifier. The stated goal is
"box a bird when one is in frame". That may already work, because a bird
against open sky is about the most salient object there is.

Testing it costs five uploads. Reverse-engineering HWAC's weight layout costs
days and is not guaranteed to succeed. So this runs first.

Reading the result
------------------
A box always comes back. "A box appeared" is therefore not a result. The test
is whether the box lands on the bird, which is why T4 is a flat grey image
with nothing in it - it shows what the hardware returns when there is nothing
to find, and every other result has to be read against that baseline.

The board reports boundaries in an unknown coordinate space. The dashboard's
own "Center" preset is (860, 1060, 440, 640), which straddles the middle of a
1920x1080 frame, so display coordinates are the best guess. The expected boxes
below are therefore given in three spaces at once; whichever one the numbers
resemble tells us the mapping, and T4 plus T1 together pin it down.

Usage
-----
    python3 03_Board_Test_Set.py

Output
------
    board_test/T*.bin        five 851,968-byte buffers to upload
    board_test/T*_check.png  what each one looks like decoded
    board_test/TEST_PLAN.md  the procedure and the expected boxes
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "board_test"
SRC_DIR = SCRIPT_DIR.parent / "98_YOLOv3_ori" / "DetectTheseImage"
CONVERTER = SCRIPT_DIR / "02_Make_Image_Bin.py"

WIDTH, HEIGHT = 416, 256

# Ground truth from 98_YOLOv3_ori/best.pt run against the 416x256 buffer
# contents, so these are already in buffer coordinates - no rescaling guesswork.
SINGLE_BIRD = "WhatsApp Image 2026-07-29 at 3.49.43 PM (1).jpeg"
TWO_BIRDS = "WhatsApp Image 2026-07-29 at 3.49.44 PM (2).jpeg"

TESTS = [
    {
        "id": "T1", "source": SINGLE_BIRD, "flags": [],
        "what": "One eagle, plain grey sky. The primary test.",
        "expect": (213.8, 247.5, 86.8, 149.6),
        "reading": "A box near this is the whole answer - the stock model "
                   "localises birds and no weight work is needed for the "
                   "stated goal.",
    },
    {
        "id": "T2", "source": SINGLE_BIRD, "flags": ["--bgr"],
        "what": "Same image, red and blue channels swapped.",
        "expect": (213.8, 247.5, 86.8, 149.6),
        "reading": "Settles channel order, which no document here resolves. "
                   "If this scores clearly better than T1, the hardware wants "
                   "BGR and every future image and weight file must match.",
    },
    {
        "id": "T3", "source": SINGLE_BIRD, "flags": ["--stretch"],
        "what": "Same image stretched to fill 416x256 instead of letterboxed.",
        "expect": None,
        "reading": "HWAC trained on 16:9 drone video squeezed into this 13:8 "
                   "buffer, so stretching may match what it expects. If T3 "
                   "beats T1, drop letterboxing.",
    },
    {
        "id": "T4", "source": None, "flags": [],
        "what": "Flat grey. Nothing in frame at all.",
        "expect": None,
        "reading": "The null control, and the most important test here. "
                   "Whatever box comes back is the hardware's output for "
                   "'nothing found'. If T1 returns roughly this same box, "
                   "then T1 found nothing either.",
    },
    {
        "id": "T5", "source": TWO_BIRDS, "flags": [],
        "what": "Two gulls, blue sky.",
        "expect": (106.8, 187.1, 0.0, 106.8),
        "reading": "One box, two birds. Shows which one wins. The expected "
                   "box is the higher-confidence gull; the other is at "
                   "L217 R274 T72 B145 and cannot also be returned.",
    },
]


def make_flat_grey(path, value=114):
    Image.new("RGB", (640, 400), (value, value, value)).save(path)


def spaces(box):
    """The same box in the three coordinate systems it might come back in."""
    if box is None:
        return None
    l, r, t, b = box
    return {
        "buffer": (l, r, t, b),
        "norm": (l / WIDTH, r / WIDTH, t / HEIGHT, b / HEIGHT),
        "display": (l / WIDTH * 1920, r / WIDTH * 1920,
                    t / HEIGHT * 1080, b / HEIGHT * 1080),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    tmp_grey = OUT_DIR / "_flat_grey.png"

    for test in TESTS:
        if test["source"] is None:
            make_flat_grey(tmp_grey)
            src = tmp_grey
        else:
            src = SRC_DIR / test["source"]
            if not src.exists():
                sys.exit(f"missing source image: {src}")

        out_bin = OUT_DIR / f"{test['id']}.bin"
        cmd = [sys.executable, str(CONVERTER), str(src),
               "--out", str(out_bin)] + test["flags"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"{test['id']} failed:\n{result.stdout}{result.stderr}")
        print(f"{test['id']}  {out_bin.name:<10} "
              f"{out_bin.stat().st_size:,} bytes  {' '.join(test['flags']) or '(default)'}")

    tmp_grey.unlink(missing_ok=True)
    write_plan()
    print(f"\nwrote {OUT_DIR / 'TEST_PLAN.md'}")


def write_plan():
    lines = [
        "# Board test: does the stock model already box a bird?",
        "",
        "Generated by `03_Board_Test_Set.py`. Five uploads, one sitting.",
        "",
        "## Procedure",
        "",
        "The bitstream and the weight file do not change during this test.",
        "Program the board as usual, upload `weight_file.bin` and",
        "`padding_file.bin` and press **LOAD FROM SD TO RAM** for each, exactly",
        "as you do today. Then for each test below:",
        "",
        "1. **AI Control** tab, section 2 *Image Input*. Leave the **Name**",
        "   field as `img0_file.bin` - the dashboard always writes to",
        "   `0x03000000` regardless of the name, so only the browsed file",
        "   matters.",
        "2. **Browse** to the `.bin` for the test.",
        "3. **UPLOAD IMAGE TO RAM**. Images go straight to DDR; there is no",
        "   LOAD FROM SD step for them.",
        "4. Press **RUN AI DETECTION**.",
        "5. Record the `Boundaries: L: R: T: B:` line from the Console tab.",
        "",
        "Optional but useful: press **Fetch DDR Snapshot** on the VPU tab after",
        "step 3. If the preview shows the photo, the upload path is confirmed",
        "good and any odd result afterwards is the model's doing, not the",
        "transfer's.",
        "",
        "## The tests",
        "",
    ]

    for test in TESTS:
        lines += [f"### {test['id']} - {test['what']}", ""]
        exp = spaces(test["expect"])
        if exp:
            b = exp["buffer"]
            n = exp["norm"]
            d = exp["display"]
            lines += [
                "Where the bird actually is, in the three coordinate spaces the",
                "board might be reporting in:",
                "",
                "| space | L | R | T | B |",
                "|---|---|---|---|---|",
                f"| 416x256 buffer | {b[0]:.0f} | {b[1]:.0f} | {b[2]:.0f} | {b[3]:.0f} |",
                f"| normalised | {n[0]:.3f} | {n[1]:.3f} | {n[2]:.3f} | {n[3]:.3f} |",
                f"| 1920x1080 display | {d[0]:.0f} | {d[1]:.0f} | {d[2]:.0f} | {d[3]:.0f} |",
                "",
            ]
        else:
            lines += ["No ground-truth box for this one.", ""]
        lines += [f"**How to read it.** {test['reading']}", "",
                  f"Upload: `{test['id']}.bin` - preview: `{test['id']}_check.png`", "", "---", ""]

    lines += [
        "## Recording the results",
        "",
        "| test | L | R | T | B | box on the bird? |",
        "|---|---|---|---|---|---|",
    ]
    for test in TESTS:
        lines.append(f"| {test['id']} | | | | | |")

    lines += [
        "",
        "## What each outcome means",
        "",
        "**T1 lands on the bird, T4 clearly differs.** The goal is met with the",
        "hardware exactly as it stands. No retraining, no weight conversion, no",
        "bitstream rebuild. The remaining work is only tuning - which of T1/T2/T3",
        "preprocesses best - and wiring the camera path to the same format.",
        "",
        "**T1 and T4 return much the same box.** The model is not finding the",
        "bird; it returns a default. Then a bird-trained weight file is genuinely",
        "required, and the blocker below applies.",
        "",
        "**Every test returns an identical box.** Suspect the image is not",
        "reaching the accelerator at all - check the DDR snapshot first, before",
        "concluding anything about the model.",
        "",
        "## If a new weight file turns out to be needed",
        "",
        "With bitstream-only access the shape is fixed: nine layers, 416x256",
        "input, layer 9 at 500 filters, 22,542,352 bytes. A bird model has to fit",
        "that mould exactly.",
        "",
        "The convenient part is that 500 filters is 5 anchors x (4 + 1 + 95",
        "classes). Training tiny-YOLOv2 with 95 class slots and using only slot 0",
        "for 'bird' keeps the file byte-for-byte the same size, so no rebuild is",
        "needed. Since the core emits one unlabelled box, only objectness has to",
        "fire on birds - the other 94 slots can stay unused.",
        "",
        "The blocker is the layout, not the training. HWAC packs weights as",
        "(N, 4) fp16 rows with two extra rows per group of four filters feeding",
        "`norm_param_a` / `norm_param_b`. Their meaning is unresolved: roughly",
        "half the row-0 values are negative, which a batch-norm scale would not",
        "be, and rebuilding the network on that assumption makes activations",
        "collapse. See `README.md`. That has to be solved numerically against",
        "`weights95tuned.npy` before any custom weight file is worth a bring-up",
        "cycle.",
        "",
    ]

    (OUT_DIR / "TEST_PLAN.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
