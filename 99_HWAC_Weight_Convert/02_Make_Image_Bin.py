"""
Convert an ordinary photo into the img0_file.bin the LogikHaus SoC expects.

Format
------
The board's image buffer is 416 x 256 pixels, four fp16 values per pixel,
little-endian, row-major - 851,968 bytes exactly. Three of the four channels
are colour; the fourth is padding the datapath needs and is ignored.

This is not guesswork. The dashboard reads the same buffer back:

    expected_bytes = 416 * 256 * 4 * 2                 lh-dashboard-P4.pyw:570
    pixel_matrix = pixel_array.reshape((height, width, 4))
    rgb_matrix = pixel_matrix[:, :, :3]
    rgb_matrix = np.clip(rgb_matrix * 255.0, 0, 255)   lh-dashboard-P4.pyw:640

The multiply by 255 on readback is what fixes the scale: the buffer holds
0.0-1.0, not 0-255. The reshape order fixes everything else.

Channel order
-------------
The dashboard treats column 0 as R. HWAC's own driver instead does

    self.mem[4194304:4300800, 0] = np.reshape(img[:,:,2], (106496))

assigning index 2 of its array to column 0, which is the opposite order if its
array was RGB. The two disagree, and no document here settles it. So --bgr
writes the other order, and the first hardware test should try both: a swapped
red and blue channel is exactly the kind of error that still produces a
plausible-looking box and then quietly costs accuracy.

Usage
-----
    python3 02_Make_Image_Bin.py photo.jpg
    python3 02_Make_Image_Bin.py photo.jpg --bgr --out img0_bgr.bin
    python3 02_Make_Image_Bin.py --dir ../98_YOLOv3_ori/DetectTheseImage
    python3 02_Make_Image_Bin.py photo.jpg --stretch

Output
------
    <name>.bin        851,968 bytes, ready to upload to 0x03000000
    <name>_check.png  the buffer decoded back with the dashboard's own logic
    <name>.json       the letterbox transform, for mapping boxes back

Upload it with the dashboard's "UPLOAD IMAGE TO RAM", then press
"RUN AI DETECTION". The image goes straight to DDR - there is no
LOAD FROM SD step for images, unlike the weight and padding files.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT, CHANNELS = 416, 256, 4
EXPECTED_BYTES = WIDTH * HEIGHT * CHANNELS * 2

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(img, fill=(114, 114, 114)):
    """Resize preserving aspect ratio, centred on a flat background.

    Returns the image and the transform needed to map a box in the 416x256
    buffer back to a box in the original photo.
    """
    src_w, src_h = img.size
    scale = min(WIDTH / src_w, HEIGHT / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)

    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), fill)
    off_x, off_y = (WIDTH - new_w) // 2, (HEIGHT - new_h) // 2
    canvas.paste(resized, (off_x, off_y))

    return canvas, {
        "mode": "letterbox",
        "source_size": [src_w, src_h],
        "scale": scale,
        "offset_x": off_x,
        "offset_y": off_y,
        "content_size": [new_w, new_h],
    }


def stretch(img):
    """Fill the whole buffer, distorting aspect ratio.

    HWAC's contest footage was 16:9 drone video squeezed into 13:8, so this is
    plausibly what the model saw in training. It is the fallback if letterboxing
    detects nothing.
    """
    src_w, src_h = img.size
    return img.resize((WIDTH, HEIGHT), Image.BILINEAR), {
        "mode": "stretch",
        "source_size": [src_w, src_h],
        "scale_x": WIDTH / src_w,
        "scale_y": HEIGHT / src_h,
    }


def to_buffer(img, bgr=False, pad_value=0.0):
    """Pack an RGB PIL image into the four-lane fp16 buffer."""
    rgb = np.asarray(img, dtype=np.float32) / 255.0        # (H, W, 3), 0.0-1.0
    if bgr:
        rgb = rgb[:, :, ::-1]

    buf = np.full((HEIGHT, WIDTH, CHANNELS), pad_value, dtype=np.float16)
    buf[:, :, :3] = rgb.astype(np.float16)
    return buf


def decode_like_dashboard(raw):
    """Reproduce lh-dashboard-P4.pyw's update_live_preview, exactly.

    Reading the file back through the consumer's own code is the only check
    that matters - it catches a wrong reshape order or a wrong scale, which a
    self-consistent writer would never notice.
    """
    pixel_array = np.frombuffer(raw, dtype=np.float16)
    pixel_matrix = pixel_array.reshape((HEIGHT, WIDTH, CHANNELS))
    rgb_matrix = pixel_matrix[:, :, :3]
    return np.clip(rgb_matrix * 255.0, 0, 255).astype(np.uint8)


def convert(src, out_bin, args):
    img = Image.open(src).convert("RGB")
    prepared, transform = (stretch(img) if args.stretch else
                           letterbox(img, fill=tuple(args.fill)))

    buf = to_buffer(prepared, bgr=args.bgr, pad_value=args.pad_value)
    buf.tofile(out_bin)

    written = out_bin.stat().st_size
    if written != EXPECTED_BYTES:
        sys.exit(f"{out_bin.name}: wrote {written:,} bytes, expected {EXPECTED_BYTES:,}")

    # Round-trip through the dashboard's decoder and compare against what we
    # meant to send. fp16 across 0.0-1.0 is far finer than one 8-bit step, so
    # any real disagreement here is a layout bug, not precision loss.
    decoded = decode_like_dashboard(out_bin.read_bytes())
    intended = np.asarray(prepared, dtype=np.uint8)
    if args.bgr:
        intended = intended[:, :, ::-1]
    drift = int(np.abs(decoded.astype(int) - intended.astype(int)).max())

    Image.fromarray(decoded if not args.bgr else decoded[:, :, ::-1]).save(
        out_bin.with_name(out_bin.stem + "_check.png"))

    transform["channel_order"] = "BGR" if args.bgr else "RGB"
    transform["buffer_size"] = [WIDTH, HEIGHT]
    out_bin.with_suffix(".json").write_text(json.dumps(transform, indent=2))

    print(f"  {out_bin.name:<28} {written:,} bytes   round-trip drift {drift}/255"
          f"{'   <-- FAIL' if drift > 1 else ''}")
    return drift <= 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", type=Path, help="photo to convert")
    ap.add_argument("--dir", type=Path, help="convert every image in a directory")
    ap.add_argument("--out", type=Path, help="output .bin (single image only)")
    ap.add_argument("--outdir", type=Path, help="where to write (default: alongside)")
    ap.add_argument("--bgr", action="store_true",
                    help="write B,G,R instead of R,G,B in the first three lanes")
    ap.add_argument("--stretch", action="store_true",
                    help="fill the buffer instead of letterboxing")
    ap.add_argument("--pad-value", type=float, default=0.0,
                    help="value for the unused fourth lane (default 0.0)")
    ap.add_argument("--fill", type=int, nargs=3, default=[114, 114, 114],
                    metavar=("R", "G", "B"), help="letterbox bar colour")
    args = ap.parse_args()

    if not args.image and not args.dir:
        ap.error("give an image, or --dir")

    sources = []
    if args.image:
        sources.append(args.image)
    if args.dir:
        sources.extend(sorted(p for p in args.dir.iterdir()
                              if p.suffix.lower() in SUFFIXES))
    if not sources:
        sys.exit("no images found")

    outdir = args.outdir or (args.dir if args.dir else sources[0].parent)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"{WIDTH}x{HEIGHT}, {CHANNELS} lanes fp16, "
          f"{'BGR' if args.bgr else 'RGB'}, "
          f"{'stretched' if args.stretch else 'letterboxed'}\n")

    ok = True
    for i, src in enumerate(sources):
        if args.out and len(sources) == 1:
            out_bin = args.out
        else:
            # img0_file.bin is the name the dashboard defaults to; number the
            # rest so a batch does not overwrite itself.
            stem = "img0_file" if i == 0 else f"img{i}_file"
            out_bin = outdir / f"{stem}.bin"
        print(f"{src.name}")
        ok &= convert(src, out_bin, args)

    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
