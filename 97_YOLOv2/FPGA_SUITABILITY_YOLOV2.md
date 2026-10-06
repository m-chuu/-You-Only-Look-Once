# Does 97_YOLOv2 suit the HWAC architecture on a Nexys Video Artix-7?

Assessment only. No RTL, bitstream or synthesis work was done.

Reference design: [HWAC-DL/hwac_object_tracker](https://github.com/HWAC-DL/hwac_object_tracker),
a DAC 2018 System Design Contest entry.

---

## Verdict

**The network matches. The board does not.**

Tiny-YOLOv2 is exactly the architecture HWAC's accelerator implements — a much
better fit than the YOLOv3-tiny model in `98_YOLOv3_ori`, which was rejected for
this purpose in [../98_YOLOv3_ori/HARDWARE_INTEGRATION.md](../98_YOLOv3_ori/HARDWARE_INTEGRATION.md).
Choosing YOLOv2 was the right call.

But HWAC targets a **Xilinx PYNQ-Z1 (Zynq XC7Z020)**, not a Nexys Video
(Artix-7 XC7A200T). Its bitstream physically cannot be loaded on the Nexys
Video, and its software stack assumes a hard ARM processor the Artix-7 does not
have. Nothing about that is fixable by converting weights.

So: keep the model, reconsider the board — or accept that "using HWAC" on a
Nexys Video means re-implementing it rather than reusing it.

---

## 1. Architecture — genuine match

`darknet_train/cfg/yolov2-tiny.cfg` describes nine convolutions:

| # | shape | kernel | after |
|---|---|---|---|
| 1 | 3 → 16 | 3×3 | maxpool 2/2 |
| 2 | 16 → 32 | 3×3 | maxpool 2/2 |
| 3 | 32 → 64 | 3×3 | maxpool 2/2 |
| 4 | 64 → 128 | 3×3 | maxpool 2/2 |
| 5 | 128 → 256 | 3×3 | maxpool 2/2 |
| 6 | 256 → 512 | 3×3 | maxpool 2/1 |
| 7 | 512 → 1024 | 3×3 | — |
| 8 | 1024 → 512 | 3×3 | — |
| 9 | 512 → 425 | 1×1 | region |

HWAC's `py/hwac_object_tracker/libraries/hwac.py` drives its core through
`setLayer1()` … `setLayer9()` — nine layers, same shapes, same 3×3-then-1×1
structure, same single detection scale. Its `inst_full()` calls carry the
channel counts 32→64, 64→128, 128→256, 256→512, 512→1024, 1024→512, and finally
512→`FC_FLT_CNT`, which is this exact network.

The anchors are identical, not merely similar. From the cfg:

```
anchors = 0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
```

From `hwac.py`:

```python
anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
```

Both are the stock tiny-YOLOv2 COCO anchors, and both decode with `/ 13`.
This is the same network, and it is the strongest argument for the YOLOv2
direction.

HWAC is also fp16 throughout (`HALF_WIDTH = 16` in `hw/YOLO/src/tiny_yolo_params.v`),
which is why `02_Export_Weights_Bin.py` writes half precision.

## 2. Board — hard mismatch

| Evidence | Finding |
|---|---|
| HWAC `README.md` | "implemented on the Xilinx **PYNQ-Z1** platform" |
| `py/overlay/.../hwac_object_tracker.bit` | **4,045,676 bytes** — an XC7Z020 configuration image (UG470 gives 4,045,564 bytes, plus a 112-byte `.bit` ASCII header). An XC7A200T image is 9,730,652 bytes. |
| `hwac.py` | uses PYNQ `Overlay`, `MMIO(0x40000000, 0x1000)` and `Xlnk.cma_array` — all require the Zynq PS |
| `hw/YOLO/src/common/axi3_params.v` | AXI3, the Zynq PS↔PL protocol. Nexys Video reaches DDR3 through a MIG controller over AXI4. |
| `design_1_processing_system7_0_0` in the block design | a hard `processing_system7` IP block, which only exists on Zynq |

The Nexys Video's XC7A200T is a different die in a different family with no hard
ARM core. The bitstream is not loadable, the Python driver has nothing to run
on, and the memory interface differs. This corroborates the independent analysis
already recorded in [../98_YOLOv3_ori/FPGA_PLATFORM_EVIDENCE.md](../98_YOLOv3_ori/FPGA_PLATFORM_EVIDENCE.md).

## 3. Four further mismatches, which apply even on a PYNQ-Z1

These matter because they are not solved by getting the right board.

**Weight format.** HWAC does not load a flat `.bin`. `configure()` does:

```python
W = np.load(weight_file)            # object array, shape (9,) - one entry per layer
woff = 0
for i in range(len(W)):
    self.mem[woff:(woff + W[i].shape[0]), :] = W[i]
    woff += W[i].shape[0]
```

`self.mem` is `xlnk.cma_array(shape=(0x800000, 4), dtype=np.float16)` — a 64 MiB
physically-contiguous DMA buffer that is **four fp16 lanes wide**, feeding four
parallel multipliers. Each layer's weights arrive as an `(N, 4)` array, and a
separate `l4padding95.npy` (346,192 bytes) supplies layer-4 padding at row
`7 << 20`. `weights95tuned.npy` is 33,846,808 bytes.

The `.bin` this project exports is a flat, headerless, `[out, in, kh, kw]`
row-major fp16 stream. It is the right artifact for interchange and matches the
YOLOv3 side, but it is **not drop-in for HWAC**. A lane-interleaving pass would
be needed, and HWAC ships no converter script — its layout has to be
reverse-engineered from `hwac.py` and the Verilog.

**Input size.** HWAC is hardcoded to **416×256**, not 416×416:

```python
img = np.empty((256, 416, 3), dtype=np.float16)
self.mem[4194304:4300800, 0] = np.reshape(img[:,:,2], (106496))   # 256*416
```

DAC-SDC 2018 used wide drone frames. A square-input model would need the
datapath's dimensions changed and the design resynthesised.

**Output.** HWAC emits a **single bounding box per image** — the contest task was
single-object tracking, and the results are "one line per image" of
`x_min, x_max, y_min, y_max`. The YOLOv2 region layer produces 13×13×5 = 845
candidate boxes followed by per-class NMS. Multi-bird detection is not something
HWAC's bounding-box module does.

**Class count.** `FC_FLT_CNT = 500` in `hwac.py` gives 5 × (4 + 1 + 95) = 95
classes. This model is 425 → 80 classes; a bird-only retrain would be 30 → 1
class. The value is parameterised in software, but the final convolution's
filter count is baked into the synthesised datapath, so changing it means
rebuilding the bitstream — which returns to needing Vivado and the right board.

## 4. Reported performance

On the contest's 1000-image set HWAC reported **average IoU 0.514 at 4.91 FPS**,
placing 5th in the FPGA category. Worth knowing before treating it as a
throughput solution: roughly 5 frames per second, at an accuracy tuned for
"roughly locate the object" rather than tight boxes.

## 5. Options

| Path | Effort | Notes |
|---|---|---|
| **Switch to a Zynq board** (PYNQ-Z1/Z2, Kria KV260) | Low | HWAC's bitstream and driver run as published. Still needs the weight relayout, 416×256 input, and a retrain for the class count. The only route that actually reuses HWAC. |
| **Keep Nexys Video, port HWAC's RTL** | High | The Verilog is there, but it needs an AXI3→AXI4/MIG rewrite, a soft MicroBlaze to replace the ARM PS, and full resynthesis for a different die. Effectively a new project that borrows compute kernels. |
| **Keep Nexys Video, use FINN/Brevitas** | High | Artix-7 is LUT-rich and DSP-poor, which suits FINN's 1–4 bit quantisation better than HWAC's fp16 datapath. Requires retraining a quantised model and self-building the MIG/DMA infrastructure, since Artix-7 is not on FINN's supported-board list. |
| **Drop the FPGA** | Lowest | A Raspberry Pi 5 or Jetson Orin Nano runs this model far above 4.91 FPS with no RTL work. |

**Recommendation:** if the goal is to reuse HWAC, get a Zynq board — that is the
single change that turns a blocked project into a working one, and tiny-YOLOv2
is already the right network for it. If the Nexys Video is fixed for other
reasons, treat HWAC as a design reference rather than something to be reused,
and budget for an RTL project rather than a weight conversion.

Either way, the immediate deliverables are unaffected: `97_YOLOv2` runs, and the
fp16 `.bin` plus manifest are the portable starting point for whichever path is
chosen.

## 6. What was checked

- HWAC `README.md`, `py/README.md`, `py/hwac_object_tracker/libraries/hwac.py`,
  `hw/YOLO/src/tiny_yolo_params.v`, `hw/YOLO/src/common/axi3_params.v`, and the
  repository file listing including bitstream and `.npy` sizes
- `darknet_train/cfg/yolov2-tiny.cfg`
- `darknet_train/yolov2-tiny.weights` — header parsed and parameter count
  confirmed at 11,237,145 float32, matching stock tiny-YOLOv2 exactly
- Xilinx UG470 bitstream length table, for the XC7Z020 / XC7A200T comparison
- The prior analysis in `../98_YOLOv3_ori/HARDWARE_INTEGRATION.md` and
  `../98_YOLOv3_ori/FPGA_PLATFORM_EVIDENCE.md`, which this agrees with
