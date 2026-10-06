# 97_YOLOv2 — tiny-YOLOv2 (Darknet), and its fp16 weight export

**Run this on Rocky Linux, not on macOS.** The `darknet` binary here is a
prebuilt Linux x86-64 ELF; there is no Darknet source in this folder to rebuild
from.

Everything below was written for:
Rocky Linux 10.1, QEMU guest, Xeon E5-2630L v4 ×12, 128 GiB.

---

## What is here

| File | Runs on | Needs |
|---|---|---|
| `darknet_train/00_Preflight_Check.sh` | Linux | bash |
| `darknet_train/make_darknet_labels.py` | anywhere | Pillow, a TrueType font |
| `darknet_train/01_Detect_Images.py` | Linux | Python 3 stdlib |
| `darknet_train/02_Export_Weights_Bin.py` | anywhere | numpy |
| `darknet_train/03_Verify_Weights.py` | anywhere | numpy, Pillow |
| `darknet_train/yolov2_darknet.py` | — | shared module, imported by 02 and 03 |
| `FPGA_SUITABILITY_YOLOV2.md` | — | the HWAC / Nexys Video assessment |

The model is stock **80-class COCO** tiny-YOLOv2, not bird-trained. "bird" is
class 15 of 80. Retraining on the bird dataset is a separate job.

---

## Step 0 — preflight

```bash
cd 97_YOLOv2/darknet_train
chmod +x 00_Preflight_Check.sh darknet
./00_Preflight_Check.sh
```

Must print `PASS` before you go further. It checks four things:

**CPU features.** The binary was compiled `-march=x86-64-v3`, so it needs AVX2,
FMA and BMI2. Your Xeon E5-2630L v4 (Broadwell) has all three — but a QEMU guest
running the default `qemu64` CPU model does **not** expose them, and darknet then
dies immediately with `Illegal instruction`. If the script reports these missing,
restart the VM with `-cpu host` (libvirt: `<cpu mode='host-passthrough'/>`).
That is a hypervisor-side change, not something a rebuild here can fix.

**Shared libraries.** It needs `libc`, `libm`, `libmvec` and `libgomp` only — no
OpenCV, no CUDA. If `libgomp.so.1` is missing:

```bash
sudo dnf install -y libgomp
```

**glibc.** The binary's highest symbol requirement is `GLIBC_2.34`; Rocky 10.1
ships 2.39, so this is fine. It was built with Red Hat GCC 14.3.1, which is
Rocky 10's own compiler — it very likely came off a machine like yours.

**Executable bit.** Git preserves it, but a zip or a copy through a Windows
share will not.

## Step 1 — supply data/labels (once)

**This folder shipped without `data/labels/`, and darknet cannot run without
it.** `load_alphabet()` reads 760 character bitmaps from there to draw text onto
the output, and it runs *before* the network is loaded. Worse, pjreddie's
darknet calls `exit(0)` — success — when an image fails to load, so a missing
`data/labels/` looks exactly like a clean run that happened to produce nothing.
This is the single reason this folder appeared broken.

Either fetch the official ones:

```bash
curl -L -o /tmp/darknet.tar.gz https://github.com/pjreddie/darknet/archive/refs/heads/master.tar.gz
tar -xzf /tmp/darknet.tar.gz -C /tmp
cp -r /tmp/darknet-master/data/labels data/
```

or generate them offline:

```bash
sudo dnf install -y python3-pillow dejavu-sans-fonts
python3 make_darknet_labels.py
```

Either way `ls data/labels/*.png | wc -l` must print **760**.

## Step 2 — run detection

```bash
python3 01_Detect_Images.py
```

Python stdlib only, no pip installs. Expect **19 annotated images** in
`outputs_images/`, plus `darknet_detections.json`.

`images/` holds 55 files that are only 19 distinct pictures duplicated as
`.jpg`/`.jpeg`/`.png`; the script keeps one copy of each.

> **On the old `train.py`.** It was named for training but only ever ran
> inference, and it hardcoded a colleague's `/home/patih-fauzi/...` paths. It
> also assumed `-out` works: this build's `detector test` ignores that flag and
> always writes into its working directory, so results are moved into place
> instead. It uses stb's JPEG encoder, so the file is `predictions.jpg`; builds
> lacking that write `predictions.png`, and both are accepted.

The model is stock COCO, so expect generic classes like `person` — not birds.

## Step 3 — export the weights to .bin

```bash
pip3 install --user numpy      # if not already present
python3 02_Export_Weights_Bin.py
```

Produces:

- `yolov2-tiny_weights_fp16.bin` — **22,459,026 bytes** expected
- `yolov2-tiny_weights_fp16.csv` — the manifest, **18 rows**

The script verifies itself: it reads the `.bin` back through the manifest alone
and confirms every tensor round-trips, then prints `PASS`.

Use `--dtype float32` for a full-precision export instead.

### The format

Identical to `98_YOLOv3_ori/06_Export_Weights_Bin.py`: convolution weights and
biases, fp16, little-endian, `[out_ch, in_ch, kh, kw]` row-major, concatenated
**with no header and no padding**. The CSV is the only index into it —
`index, role, name, shape, count, byte_offset, byte_length`.

That sibling script reads its already-fused tensors out of an ONNX file. There
is no ONNX export here, and building one would mean putting PyTorch on a
CPU-only VM for a single graph dump, so `02_...` folds batch-norm itself and
writes the same format.

The fold uses Darknet's own epsilon placement — `scale / (sqrt(var) + 1e-6)`,
not the `sqrt(var + eps)` most frameworks use — so it reproduces the C code
exactly rather than approximately.

**Why 22,459,026 bytes and not half of the 44.9 MB `.weights` file?** The raw
file holds 11,237,145 float32 values. Folding batch-norm collapses each layer's
scale, running mean and running variance into the convolution's weight and bias,
removing 3 × 2,544 = 7,632 values. 11,229,513 remain, at 2 bytes each.

## Step 4 — verify the export (recommended)

```bash
sudo dnf install -y python3-pillow
python3 03_Verify_Weights.py
```

This rebuilds the network **from the `.bin` and its CSV alone** — the `.weights`
file is not read — runs pure-NumPy inference, and compares against the
detections darknet produced in step 2.

Measured on the first 3 images: agreement with darknet at **0% confidence
drift**, so fp16 costs nothing in detection accuracy at this threshold.

That ordering is deliberate. A wrong tensor order, a wrong offset, or a
mis-folded batch-norm all produce disagreeing detections here rather than
passing silently and failing later on hardware.

Inference is a few seconds per image; it exists to check the export, not to be
fast. Use `--limit 5` for a quick pass, or `--source weights` to compare against
the original file as a control.

---

## FPGA / HWAC

See [FPGA_SUITABILITY_YOLOV2.md](FPGA_SUITABILITY_YOLOV2.md). Short version:
tiny-YOLOv2 **is** the architecture HWAC implements — identical layers, identical
anchors — but HWAC targets a PYNQ-Z1 (Zynq XC7Z020), not a Nexys Video
(Artix-7 XC7A200T), and its bitstream cannot load on that board. HWAC also wants
a lane-interleaved `.npy` rather than a flat `.bin`, a 416×256 input, and emits
one box per image.

---

## Not done

- **Bird retraining.** Would need the `98_YOLOv3_ori/01_Train_images/` dataset
  converted to Darknet format, `classes=1` and `filters=30` in the cfg, and a
  training run. The COCO weights were exported first to prove the path.
- **HWAC's `(N,4)` lane-interleaved layout.** Documented, not implemented —
  it only makes sense once a board is chosen.
