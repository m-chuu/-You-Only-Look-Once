"""
Run the bundled darknet binary over every image in images/ and save the
annotated results to outputs_images/.

Despite its old name (train.py) this script has only ever run inference.

Linux only - darknet here is a prebuilt x86-64 ELF (pjreddie's, CPU only, no
OpenCV and no CUDA). Two quirks of this build drive how the script works:

  * `detector test` ignores -out and always writes its result into the working
    directory, so each one is moved into place afterwards. It uses stb's JPEG
    encoder, giving predictions.jpg; older builds without it give
    predictions.png, so both are accepted.
  * load_alphabet() reads data/labels/*.png before the network is even loaded,
    and exits with status 0 if they are missing. That is what made this folder
    look broken for so long - it is checked for up front below.

Darknet's per-detection output is also parsed off stdout into
darknet_detections.json, which 03_Verify_Weights.py compares against.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ----------------------------
# USER SETTINGS
# ----------------------------

THRESH = 0.5

DARKNET_DIR = Path(__file__).resolve().parent
INPUT_DIR = DARKNET_DIR / "images"
OUTPUT_DIR = DARKNET_DIR / "outputs_images"

CFG = DARKNET_DIR / "cfg" / "yolov2-tiny.cfg"
WEIGHTS = DARKNET_DIR / "yolov2-tiny.weights"
DATA = DARKNET_DIR / "cfg" / "coco.data"
DARKNET = DARKNET_DIR / "darknet"

# ----------------------------
# Setup
# ----------------------------

for required in (CFG, WEIGHTS, DATA, DARKNET):
    if not required.exists():
        sys.exit(f"missing required file: {required}")

if not os.access(DARKNET, os.X_OK):
    sys.exit(f"{DARKNET} is not executable - run: chmod +x {DARKNET}")

# Darknet's load_alphabet() reads data/labels/<ascii>_<size>.png before it even
# loads the network, and exits with status 0 if any are missing - which looks
# like "no output was produced" rather than a failure. Check up front instead.
LABELS_DIR = DARKNET_DIR / "data" / "labels"
if not LABELS_DIR.is_dir() or not any(LABELS_DIR.glob("*.png")):
    sys.exit(f"missing {LABELS_DIR}\n"
             f"darknet needs 760 character bitmaps there to draw labels.\n"
             f"Generate them with: python3 make_darknet_labels.py")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# images/ holds the same pictures duplicated as .jpg/.jpeg/.png. Keep one copy
# of each, preferring the least lossy encoding available.
PREFERENCE = {".png": 0, ".jpg": 1, ".jpeg": 2}

by_stem = {}
for path in sorted(INPUT_DIR.iterdir()):
    rank = PREFERENCE.get(path.suffix.lower())
    if rank is None:
        continue
    current = by_stem.get(path.stem)
    if current is None or rank < PREFERENCE[current.suffix.lower()]:
        by_stem[path.stem] = path

images = [by_stem[stem] for stem in sorted(by_stem)]
print(f"Found {len(images)} distinct images in {INPUT_DIR}")

# ----------------------------
# Run Detection
# ----------------------------

failures = []
records = {}

# Darknet prints one "<class>: <percent>%" line per detection above -thresh.
DETECTION_RE = re.compile(r"^(.+?):\s+(\d+)%\s*$")

PREDICTION_CANDIDATES = (DARKNET_DIR / "predictions.png", DARKNET_DIR / "predictions.jpg")

for img in images:
    print(f"Processing: {img.name}")

    # Clear any stale prediction so a failed run cannot be mistaken for a
    # successful one by leaving the previous image's output in place.
    for stale in PREDICTION_CANDIDATES:
        if stale.exists():
            stale.unlink()

    cmd = [
        str(DARKNET),
        "detector", "test",
        str(DATA),
        str(CFG),
        str(WEIGHTS),
        str(img),
        "-thresh", str(THRESH),
    ]

    # IMPORTANT: run inside darknet folder - coco.data refers to data/coco.names
    # by a relative path, and predictions.png lands here too.
    result = subprocess.run(cmd, cwd=DARKNET_DIR, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

    written = next((p for p in PREDICTION_CANDIDATES if p.exists()), None)

    if result.returncode != 0:
        failures.append((img.name, f"darknet exited {result.returncode}"))
        print(f"  FAILED: darknet exited {result.returncode}")
        continue
    if written is None:
        failures.append((img.name, "darknet produced no predictions file"))
        print(f"  FAILED: no predictions.png in {DARKNET_DIR}")
        continue

    produced = OUTPUT_DIR / (img.stem + written.suffix)
    shutil.move(str(written), str(produced))

    detections = []
    for line in result.stdout.splitlines():
        match = DETECTION_RE.match(line.strip())
        if match:
            detections.append({"class": match.group(1), "percent": int(match.group(2))})

    records[img.name] = detections
    print(f"  saved {produced} ({len(detections)} detections)")

# ----------------------------
# Report
# ----------------------------

DETECTIONS_JSON = DARKNET_DIR / "darknet_detections.json"
DETECTIONS_JSON.write_text(json.dumps({
    "thresh": THRESH,
    "images": records,
}, indent=2, sort_keys=True))

total = sum(len(d) for d in records.values())
print(f"\n{len(images) - len(failures)}/{len(images)} images processed -> {OUTPUT_DIR}")
print(f"{total} detections recorded -> {DETECTIONS_JSON}")

if failures:
    print("\nFailures:")
    for name, why in failures:
        print(f"  {name}: {why}")
    sys.exit(1)
