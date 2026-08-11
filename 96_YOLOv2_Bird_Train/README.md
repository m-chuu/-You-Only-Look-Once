# Fine-tune tiny-YOLOv2 on birds

Trains the bird detector with the same Darknet binary already verified in
`97_YOLOv2`. Runs on Rocky Linux, CPU only.

**Run every step on Rocky.** `data/obj.data` and the list files hold absolute
paths, so the prepare step must run on the machine that will train.

---

## Step 1 — copy to Rocky

Two things, alongside the existing `97_YOLOv2`:

| From the Mac | To Rocky |
|---|---|
| `96_YOLOv2_Bird_Train/` (scripts only — `data/` is regenerated) | next to `97_YOLOv2` |
| `98_YOLOv3_ori/01_Train_images/` | anywhere; pass its path in step 2 |

Set these once per shell, adjusting to your layout:

```bash
BASE=/shared/LH-AI-Modelling-WG/Matt-AI-Detection
DARKNET=$BASE/97_YOLOv2/97_YOLOv2/darknet_train
cd $BASE/96_YOLOv2_Bird_Train
```

## Step 2 — prepare the dataset

```bash
python3 00_Prepare_Dataset.py --dataset /path/to/01_Train_images
```

Standard library only. Expect `train: 280 images`, `valid: 82 images`, and five
files written.

It copies the data into `data/<split>/images` and `data/<split>/labels` rather
than pointing at it in place. That is not tidiness — Darknet locates a label by
replacing the **first** occurrence of `images` in the image path, and the source
path `01_Train_images/train/images/...` contains it twice. Left alone, Darknet
would silently train against labels that do not exist.

The cfg is generated here too, so `classes` and the final conv `filters` cannot
drift apart. With `--classes 1` that is `5 x (5 + 1) = 30`.

## Step 3 — build the pretrained backbone

Fine-tuning from the COCO backbone converges far faster than training from
scratch on 280 images. You already have the weights to make it:

```bash
$DARKNET/darknet partial $DARKNET/cfg/yolov2-tiny.cfg \
    $DARKNET/yolov2-tiny.weights yolov2-tiny.conv.13 13
```

This keeps layers 0–12 (through the 1024-filter conv) and discards the
COCO-specific head. Expect a `yolov2-tiny.conv.13` of roughly 44 MB.

## Step 4 — measure the speed before committing

CPU training is slow and worth timing before you leave it running:

```bash
$DARKNET/darknet detector train data/obj.data cfg/yolov2-tiny-bird-1c.cfg \
    yolov2-tiny.conv.13 2>&1 | tee smoke.log
```

Let it reach 5–10 iterations, note the seconds-per-iteration Darknet prints at
the end of each line, then stop it with Ctrl-C. Multiply by 2000.

If that projects past about a day, re-run step 2 smaller and start again:

```bash
python3 00_Prepare_Dataset.py --dataset /path/to/01_Train_images \
    --size 320 --max-batches 1000
```

320 costs roughly 40% less compute per iteration than 416.

## Step 5 — train

```bash
nohup $DARKNET/darknet detector train data/obj.data \
    cfg/yolov2-tiny-bird-1c.cfg yolov2-tiny.conv.13 > train.log 2>&1 &
echo $!    # note the PID
```

`nohup` survives a dropped SSH session. Watch it with:

```bash
tail -f train.log
grep -oE "avg loss: [0-9.]+" train.log | tail -20
```

**You can stop whenever the loss plateaus.** Darknet writes
`backup/yolov2-tiny-bird-1c.backup` every 100 iterations, and that file is a
complete, usable set of weights. There is no need to reach `max_batches`.

Read the per-iteration lines rather than only the loss. `Region Avg IOU` rising
towards 0.6+ and `Avg Recall` towards 1.0 mean it is learning; loss alone can
fall while the model predicts nothing.

## Step 6 — test

```bash
cp backup/yolov2-tiny-bird-1c.backup bird.weights
$DARKNET/darknet detector test data/obj.data cfg/yolov2-tiny-bird-1c.cfg \
    bird.weights /path/to/a/bird.jpg -thresh 0.4
```

Run this from a directory containing `data/labels/` (the 760 character
bitmaps), or Darknet exits with status 0 and produces nothing — the same trap
documented in `97_YOLOv2/README.md`. The simplest fix is to run it from
`$DARKNET`.

Set `batch=1` and `subdivisions=1` in the cfg for inference, or just use the
`97_YOLOv2` cfg with `classes=1` and `filters=30`.

---

## Why 1 class and not 95

You asked to assume 500 output filters for HWAC compatibility. This trains
1 class (30 filters) instead, and it costs nothing.

A 1-class model expands to HWAC's 500-filter shape **losslessly**. Each anchor
occupies `4 + 1 + classes` output channels: 6 for one class, 100 for 95. The
six real values map into the first six slots of each 100-slot block, and the
remaining 94 class logits are set strongly negative so softmax gives them zero.
The result is numerically identical to the 1-class model.

Training 95 classes directly would be worse — 94 outputs that never see a
positive example add noise to the softmax and waste capacity. So this way gives
a better detector and the same hardware compatibility.

If you want the 95-class version anyway:

```bash
python3 00_Prepare_Dataset.py --dataset /path/to/01_Train_images --classes 95
```

## Status of the FPGA path

The weight packer is **not solved**. Parameter rows and tap ordering are
proven, but the meaning of the two parameter rows is not, and HWAC published no
conversion script, testbench or reference model. See the analysis in
`97_YOLOv2/FPGA_SUITABILITY_YOLOV2.md`.

This training is worth doing regardless — it produces a real bird detector that
runs on CPU today, and it is the model that gets packed the moment the layout
question is answered.
