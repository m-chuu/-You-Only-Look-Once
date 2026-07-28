"""
Visually compare two YOLO inference output folders.

Works purely on the annotated JPGs that Ultralytics writes with save=True, so it
needs no re-run, no model, and no ground-truth labels. For every image present in
both folders it builds an A | B | diff triptych and ranks the images by how much
the annotations actually changed, so the biggest behaviour differences float to
the top instead of being buried in a 40-image folder.

Usage
-----
    python 03_Compare_Runs.py RUN_A RUN_B [options]

Example
-------
    python 03_Compare_Runs.py \
        runs/detect/Inference/detection_result-2 \
        runs/detect/Inference/detection_result-3 \
        --label-a "0724" --label-b "0727"
"""

import argparse
import csv
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# -------------------------------------------------
# Tunables
# -------------------------------------------------
PIXEL_DELTA = 30    # per-channel difference above which a pixel counts as "changed"
HIGHLIGHT = (255, 0, 60)   # colour used to paint changed pixels in the diff panel
BANNER_H = 34       # height of the caption strip above each triptych
GAP = 8             # gap in pixels between panels


def load_font(size=22):
    """Best-effort truetype lookup; PIL's default bitmap font is unreadably small."""
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def diff_mask(arr_a, arr_b):
    """Boolean mask of visibly-changed pixels, plus the raw per-pixel magnitude."""
    delta = np.abs(arr_a.astype(np.int16) - arr_b.astype(np.int16)).max(axis=2)
    return delta > PIXEL_DELTA, delta


def diff_panel(img_b, mask):
    """Desaturated copy of B with the changed pixels painted in HIGHLIGHT.

    The mask is dilated first: box outlines are 1-3px wide, and without dilation
    the changed pixels are too thin to see in a downscaled contact sheet.
    """
    base = img_b.convert("L").convert("RGB")
    base = Image.blend(base, Image.new("RGB", base.size, "white"), 0.45)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3))

    overlay = Image.new("RGB", base.size, HIGHLIGHT)
    return Image.composite(overlay, base, mask_img)


def build_triptych(img_a, img_b, panel_c, name, pct, label_a, label_b, font):
    """Stitch the three panels side by side under a single caption strip."""
    w, h = img_a.size
    total_w = w * 3 + GAP * 2
    canvas = Image.new("RGB", (total_w, h + BANNER_H), "white")
    canvas.paste(img_a, (0, BANNER_H))
    canvas.paste(img_b, (w + GAP, BANNER_H))
    canvas.paste(panel_c, (w * 2 + GAP * 2, BANNER_H))

    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), f"A: {label_a}", fill="black", font=font)
    draw.text((w + GAP + 6, 8), f"B: {label_b}", fill="black", font=font)
    draw.text(
        (w * 2 + GAP * 2 + 6, 8),
        f"diff {pct:.2f}% - {name[:40]}",
        fill=HIGHLIGHT,
        font=font,
    )
    return canvas


def compare(dir_a, dir_b, out_dir, label_a, label_b, top, scale):
    names_a = {f for f in os.listdir(dir_a) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    names_b = {f for f in os.listdir(dir_b) if f.lower().endswith((".jpg", ".jpeg", ".png"))}

    only_a, only_b = sorted(names_a - names_b), sorted(names_b - names_a)
    shared = sorted(names_a & names_b)

    if not shared:
        raise SystemExit("No filenames in common between the two folders.")

    os.makedirs(out_dir, exist_ok=True)
    font = load_font()
    rows = []

    for name in shared:
        img_a = Image.open(os.path.join(dir_a, name)).convert("RGB")
        img_b = Image.open(os.path.join(dir_b, name)).convert("RGB")

        if img_a.size != img_b.size:
            print(f"  ! size mismatch, skipping: {name} {img_a.size} vs {img_b.size}")
            continue

        mask, delta = diff_mask(np.asarray(img_a), np.asarray(img_b))
        rows.append(
            {
                "image": name,
                "pct_changed": round(float(mask.mean() * 100), 4),
                "mean_abs_delta": round(float(delta.mean()), 4),
                "identical": bool(not mask.any()),
                "_a": img_a,
                "_b": img_b,
                "_mask": mask,
            }
        )

    rows.sort(key=lambda r: -r["pct_changed"])

    # -------------------------------------------------
    # Write triptychs for the N most-changed images
    # -------------------------------------------------
    for rank, row in enumerate(rows[:top], start=1):
        img_a, img_b, mask = row["_a"], row["_b"], row["_mask"]
        panel_c = diff_panel(img_b, mask)

        if scale != 1.0:
            size = (int(img_a.width * scale), int(img_a.height * scale))
            img_a = img_a.resize(size, Image.LANCZOS)
            img_b = img_b.resize(size, Image.LANCZOS)
            panel_c = panel_c.resize(size, Image.LANCZOS)

        sheet = build_triptych(
            img_a, img_b, panel_c, row["image"], row["pct_changed"], label_a, label_b, font
        )
        stem = os.path.splitext(row["image"])[0][:50]
        sheet.save(os.path.join(out_dir, f"{rank:02d}_{stem}.jpg"), quality=92)

    # -------------------------------------------------
    # Report
    # -------------------------------------------------
    print(f"\nA = {dir_a}  ({label_a})")
    print(f"B = {dir_b}  ({label_b})")
    print(f"\n{len(shared)} images in common, {len(rows)} compared")
    if only_a:
        print(f"  only in A ({len(only_a)}): {', '.join(only_a[:5])}{' ...' if len(only_a) > 5 else ''}")
    if only_b:
        print(f"  only in B ({len(only_b)}): {', '.join(only_b[:5])}{' ...' if len(only_b) > 5 else ''}")

    identical = sum(r["identical"] for r in rows)
    print(f"  byte-visually identical: {identical}/{len(rows)}")
    print(f"  changed: {len(rows) - identical}/{len(rows)}\n")

    print(f"{'#':>3}  {'% pixels changed':>16}  {'mean |delta|':>12}  image")
    print("-" * 88)
    for rank, row in enumerate(rows, start=1):
        marker = " *" if rank <= top else "  "
        print(
            f"{rank:>3}{marker}{row['pct_changed']:>15.2f}%  "
            f"{row['mean_abs_delta']:>12.2f}  {row['image'][:44]}"
        )
    print(f"\n* = triptych written to {out_dir}")

    csv_path = os.path.join(out_dir, "compare_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "pct_changed", "mean_abs_delta", "identical"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(f"  summary CSV: {csv_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visually compare two YOLO inference output folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_a", help="first inference output folder")
    parser.add_argument("run_b", help="second inference output folder")
    parser.add_argument("-o", "--out", default=None, help="output folder (default: <run_b>_vs_<run_a>_compare)")
    parser.add_argument("--label-a", default=None, help="caption for panel A")
    parser.add_argument("--label-b", default=None, help="caption for panel B")
    parser.add_argument("--top", type=int, default=12, help="how many triptychs to write")
    parser.add_argument("--scale", type=float, default=1.0, help="resize factor for the panels")
    args = parser.parse_args()

    label_a = args.label_a or os.path.basename(os.path.normpath(args.run_a))
    label_b = args.label_b or os.path.basename(os.path.normpath(args.run_b))

    # Labels are free text and end up in a path, so strip anything shell-hostile.
    slug = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "run"
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.normpath(args.run_b)),
        f"compare_{slug(label_a)}_vs_{slug(label_b)}",
    )

    compare(args.run_a, args.run_b, out_dir, label_a, label_b, args.top, args.scale)


if __name__ == "__main__":
    main()
