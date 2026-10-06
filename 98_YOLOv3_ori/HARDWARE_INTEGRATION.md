# Hardware integration: getting `best.pt` onto an FPGA

Target board: **Digilent Nexys Video** (Artix-7 XC7A200T)
Reference project: [HWAC object tracker](https://github.com/HWAC-DL/hwac_object_tracker) (DAC 2018 SDC, 5th place FPGA category)
Model: `best.pt` — YOLOv3-tiny backbone + Ultralytics anchor-free head, 1 class (`birds`), 12,128,178 parameters

---

## 0. The blocker, stated plainly

Two things you asked for are mutually exclusive:

**HWAC's `hwac_object_tracker.bit` cannot be loaded onto a Nexys Video.** A bitstream is a
configuration of one specific silicon die. HWAC's targets the PYNQ-Z1's **Zynq XC7Z020**;
your board is an **Artix-7 XC7A200T**. Different family, different die, different resource
map, different pinout. The tools will refuse the file, and no conversion exists.

Compounding this: the whole HWAC flow runs on the Zynq's **hard ARM Cortex-A9** — Linux,
Python, Jupyter, the PYNQ `Overlay` class. The Nexys Video has no hard processor. You would
have to instantiate a **MicroBlaze** soft core and rebuild that entire software stack.

What *is* reusable is their **Verilog RTL** (`hw/YOLO/src/`), which can be re-synthesised for
Artix-7. That is the honest path, and section 3 covers it.

Second, independent problem: **your model is not the architecture their core implements.**
Section 2 quantifies this.

---

## 1. Done: portable export

```bash
cd 98_YOLOv3_ori
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', imgsz=416, opset=12)"
```

Produced **`best.onnx`** (46.3 MB, opset 12, input `1×3×416×416`, output `1×5×845`).

ONNX is the correct first step regardless of which hardware path you take — every toolchain
in section 4 ingests it. The `845` is `13×13 + 26×26`, the two detection grids at stride 32
and 16.

> Exported at `imgsz=416` to match training. You ran inference at 1280 in
> `05_Detect_New_Images.py`; if 1280 is the deployment resolution, re-export with
> `imgsz=1280` — the ONNX graph bakes in the input size.

Inspect it visually at [netron.app](https://netron.app) by dragging `best.onnx` in.

---

## 2. Why your model does not fit their core

### Their core: TinyYOLO v2

From `hwac.py`, the `FPGA` class has exactly **nine** layer methods, `setLayer1()` through
`setLayer9()`. Layer 1 takes a 416-wide input to 16 channels; layer 9 emits **500 filters at
13×13**. That 500 decodes cleanly:

```
5 anchors × (4 box + 1 objectness + 95 classes) = 5 × 100 = 500
```

A textbook anchor-based TinyYOLOv2 head. Nine conv layers, one detection scale, straight
conv → leaky-ReLU → maxpool chain with no branching.

### Your model: 24 convolutions across three stages

| Stage | Convs | Detail |
|---|---|---|
| Backbone | 1–7 | 16→32→64→128→256→512→1024, 3×3, each followed by maxpool |
| Neck | 8–11 | 1×1 reduce, upsample, **concat** (384ch), producing two scales |
| Head | 12–24 | Decoupled box + class branches per scale, plus the **DFL** conv |

Conv 24 is `[1, 16, 1, 1]` — the Distribution Focal Loss projection, `reg_max=16`. Convs 14
and 17 output 64 channels (`4 × reg_max`); convs 20 and 23 output 1 channel (`nc=1`).

### Operations their hardware does not have

The exported ONNX graph contains:

```
Conv 24 · Mul 22 · Sigmoid 20 · Concat 8 · Reshape 8 · MaxPool 6
Slice 3 · Transpose 3 · Add 3 · Div 2 · Sub 2 · Resize 1 · Softmax 1
```

`Resize` (upsample), `Concat` (route), `Softmax` (DFL), and the `Sub`/`Add`/`Div` box-decode
arithmetic have **no counterpart** in a nine-layer sequential TinyYOLOv2 pipeline. Their
instruction set is six commands wide — `CMD_LD_W`, `CMD_LD_D`, `CMD_SV_D`, `CMD_FULL`,
`CMD_LAYER_END`, `CMD_WAIT_END`. There is no route, no upsample, no softmax.

**Conclusion:** loading your weights into their core produces noise. Matching it would mean
retraining from a TinyYOLOv2 base and giving up the two-scale head — the very thing that
found 53 swifts in your flock image.

---

## 3. The HWAC weight binary, in depth

You asked how their `.npy` actually works. It is instructive, because it shows what a custom
FPGA weight binary really is: **not a model file, but a memory image shaped by the datapath.**

### The buffer

```python
self.mem = xlnk.cma_array(shape=(0x800000, 4), dtype=np.float16)
```

A **contiguous** DMA-able buffer — 8,388,608 rows × 4 columns of fp16 = **64 MiB**. `cma_array`
matters: the FPGA reads physical addresses directly, so the memory must not be paged or
scattered. The physical address is handed to the hardware in `configure()`:

```python
self.ctrl.write(PHY_ADDR_REG, self.mem.physical_address)
self.ctrl.write(RESET_REG, 1)
self.ctrl.write(RESET_REG, 0)
```

### The load

```python
W = np.load(weight_file)
for i in range(len(W)):
    self.mem[woff:(woff + W[i].shape[0]), :] = W[i]
    woff += W[i].shape[0]
```

`weights95tuned.npy` is an **object array of per-layer 2-D arrays**, each `(N, 4)` fp16,
concatenated nose-to-tail into the buffer. Padding data goes to a fixed offset of
`7 << 20` = row 7,340,032.

### Why four columns

From `tiny_yolo_params.v`:

```verilog
HALF_WIDTH        = 16      // fp16
WEIGHTS_LEN       = 36      // = 9 × 4
WEIGHTS_PARAM_LEN = 38      // 36 + 2 words of layer metadata
```

**`36 = 9 × 4`**: a 3×3 kernel is 9 taps, and the datapath has **4 parallel lanes**. So each
row of the array feeds four multiply-accumulate units in the same clock cycle, and nine
consecutive rows deliver one complete 3×3 kernel across four channels. The `(N, 4)` shape is
not a convention — it is the physical wiring of the multiplier array.

Two extra words per kernel (`38 - 36`) carry per-layer metadata: dimensions, channel counts,
stride, padding, encoded per the other params (`DIM_WIDTH=8`, `CH_COUNT_WIDTH=10`,
`FLT_COUNT_WIDTH=10`, `CONV_SIZE_WIDTH=2`, `PADDING_WIDTH=4`).

### What this means for you

To produce an equivalent binary for **any** custom core, including one you build, you must:

1. Know the datapath's parallelism (here: 4 lanes) and kernel unroll order.
2. Convert fp32 → fp16 (`np.float16` — trivial, `arr.astype(np.float16)`).
3. Re-order weights from PyTorch's `[out_ch, in_ch, kh, kw]` into the hardware's traversal
   order, which is generally *not* row-major.
4. Interleave the metadata words the controller expects.
5. Concatenate layers in execution order and record each layer's row offset.

**There is no converter in the HWAC repo.** No `convert.py`, no quantise script — `py/hwac_object_tracker/`
contains only the notebook, `preprocessing.py`, and `libraries/hwac.py`. Steps 1 and 3 would
have to be reverse-engineered from `tiny_yolo.v`. Budget weeks, not days, and note the repo
has been dormant since 2018.

---

## 4. Realistic paths, compared

Your workload is **periodic still images** — no real-time constraint. That materially weakens
the case for a custom accelerator, since throughput is not your bottleneck.

| Path | Board | Effort | Works with `best.pt`? | Notes |
|---|---|---|---|---|
| **Vitis-AI / DPU** | Zynq / Zynq-MPSoC (**not** Artix-7) | Medium | Yes, via ONNX | Xilinx's supported NN accelerator. Handles YOLO natively. **Requires a Zynq board** — DPU needs the hard PS. Rules out Nexys Video. |
| **FINN / Brevitas** | Artix-7 via **stitched IP only** — Nexys Video is *not* on FINN's supported board list ([`getting_started.rst`](https://github.com/Xilinx/finn/blob/main/docs/finn/getting_started.rst), "Supported FPGA Hardware": Pynq-Z1/Z2, Kria, Ultra96, ZCU102/104, Alveo) | High | No — needs retraining | AMD research flow for **quantised neural networks (QNNs)**. Generates an AXI-Stream IP block from a Brevitas-trained model, which the docs state "can be integrated onto any Xilinx-AMD FPGA as part of a larger system… **it's up to you** to… wire it up to your FPGA design". So you build MIG, DMA and MicroBlaze yourself: full bitfile + DMAs + Python driver are produced only for PYNQ/Alveo boards, and the bitstream step is `MakeZYNQProject` (Zynq) or `VitisLink` (Alveo) ([`hw_build.rst`](https://github.com/Xilinx/finn/blob/main/docs/finn/hw_build.rst)). Quantisation will cost accuracy; the bit-width is yours to choose and tune. |
| **Port HWAC RTL** | Artix-7, re-synthesised | Very high | No — architecture mismatch | Their Verilog can be re-targeted (Artix-7 XC7A200T has *more* fabric than XC7Z020: 740 DSPs vs 220, 13 Mb BRAM vs 4.9 Mb). But you must add a MicroBlaze, rebuild the software stack, and still retrain as TinyYOLOv2. |
| **hls4ml** | Artix-7 | High | No — needs heavy pruning | Aimed at small networks. 12M parameters is far beyond its comfortable range. |
| **Swap to a Zynq board** | PYNQ-Z2 / Kria KV260 | Low–medium | Yes | ~US$150–250. Kria KV260 with Vitis-AI runs YOLO models out of the box. **If FPGA is mandated, this is the pragmatic answer.** |
| **Non-FPGA edge** | Pi 5, Jetson Orin Nano | Very low | Yes | For periodic stills, a Pi 5 running `best.onnx` via ONNXRuntime is minutes of work. Worth stating as the baseline you are choosing to reject. |

### Resource reality check

Your model is 12.13 M parameters. At fp16 that is **24.3 MB of weights**.

- Nexys Video BRAM: 13 Mb ≈ **1.6 MB**
- PYNQ-Z1 BRAM: 4.9 Mb ≈ 0.6 MB

Weights cannot live on-chip on either board — they must stream from DDR3, exactly as HWAC
does through its 64 MiB CMA buffer. The Nexys Video's 512 MB DDR3 is ample. **Memory is not
your constraint; architecture and toolchain are.**

---

## 5. Recommended sequence

1. **`best.onnx` exists.** Verify it in Netron and confirm the graph matches expectations.
2. **Decide the board question.** If FPGA is genuinely mandated, moving to a **Kria KV260 or
   PYNQ-Z2** turns a multi-month RTL project into a supported Vitis-AI flow. If the Nexys
   Video is fixed, **FINN** is the only realistic route and requires retraining with Brevitas
   at low bit-width.
3. **If Nexys Video + FINN:** retrain a quantised model from scratch. Your 280-image dataset
   is small enough that this is a day of compute, but expect a real accuracy drop from
   aggressive quantisation.
4. **Do not attempt** to load HWAC's `.bit` or feed their core your weights. Neither can work.

---

## Appendix: files produced

| File | Purpose |
|---|---|
| `best.onnx` | Portable export, 46.3 MB, input 416×416 |
| `04_Read_Checkpoint.py` | Reads metrics embedded in any `.pt` |
| `05_Detect_New_Images.py` | Inference on `DetectTheseImage/` at imgsz 1280 |
