"""
Generate the character bitmaps darknet needs in data/labels/.

Darknet's load_alphabet() reads data/labels/<ascii>_<size>.png for every
character from 32 to 126 at eight sizes - 760 files - and uses them to draw
text labels onto detection output. It runs before the network is even loaded,
and pjreddie's darknet calls exit(0) if any of them is missing, so a missing
labels/ directory looks like "no output was produced" rather than an error.

Those files ship with the darknet repository but were never included in this
folder, so this regenerates them.

Reproduces what darknet's own data/labels/make_labels.py does with ImageMagick:
black text on white, point size (index + 1) * 12, with a 4 pixel white border.
The glyphs only affect how labels look, not what gets detected.

Requires Pillow and a TrueType font.

Usage
-----
    python3 make_darknet_labels.py [--force]
"""

import argparse
import glob
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LABELS_DIR = SCRIPT_DIR / "data" / "labels"

FIRST_CHAR, LAST_CHAR = 32, 126
SIZE_COUNT = 8
BORDER = 4

FONT_CANDIDATES = [
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation-fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/gnu-free/FreeSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def find_font():
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    found = sorted(glob.glob("/usr/share/fonts/**/*.ttf", recursive=True))
    return found[0] if found else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if the files already exist")
    args = ap.parse_args()

    expected = SIZE_COUNT * (LAST_CHAR - FIRST_CHAR + 1)

    if LABELS_DIR.is_dir() and not args.force:
        existing = len(list(LABELS_DIR.glob("*.png")))
        if existing >= expected:
            print(f"{LABELS_DIR} already holds {existing} files - nothing to do")
            print("use --force to regenerate")
            return

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        sys.exit("Pillow is required:\n"
                 "  sudo dnf install -y python3-pillow\n"
                 "or  pip3 install --user Pillow")

    font_path = find_font()
    if font_path is None:
        sys.exit("no TrueType font found. Install one:\n"
                 "  sudo dnf install -y dejavu-sans-fonts")

    print(f"font: {font_path}")
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    ruler = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    written = 0

    for size_index in range(SIZE_COUNT):
        # Darknet's own generator uses pointsize (i + 1) * 12.
        point_size = (size_index + 1) * 12
        font = ImageFont.truetype(font_path, point_size)

        ascent, descent = font.getmetrics()
        height = ascent + descent

        for code in range(FIRST_CHAR, LAST_CHAR + 1):
            char = chr(code)

            advance = ruler.textlength(char, font=font)
            box = font.getbbox(char)
            width = max(int(round(advance)), box[2] if box else 0, 1)

            image = Image.new("RGB", (width + 2 * BORDER, height + 2 * BORDER), "white")
            ImageDraw.Draw(image).text((BORDER, BORDER), char, font=font, fill="black")
            image.save(LABELS_DIR / f"{code}_{size_index}.png")
            written += 1

        print(f"  size {size_index}: {point_size}pt, {height + 2 * BORDER}px tall")

    print(f"\nwrote {written} files to {LABELS_DIR}")
    if written != expected:
        sys.exit(f"expected {expected} files, wrote {written}")
    print("PASS - re-run: python3 01_Detect_Images.py")


if __name__ == "__main__":
    main()
