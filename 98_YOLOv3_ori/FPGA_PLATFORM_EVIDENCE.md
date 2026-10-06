# Platform evidence: why HWAC needs a Zynq board, and why FINN is the Artix-7 route

Prepared to support the architecture decision for bird detection on the **Digilent Nexys Video
(Artix-7 XC7A200T)**. Every claim below is backed by a primary source — vendor documentation
or the reference project's own code.

---

## Claim 1 — HWAC's bitstream is built for a Zynq XC7Z020 and cannot load on Artix-7

### 1a. The decisive arithmetic

Xilinx publishes the exact configuration bitstream length for every 7-series device.

| Device | Bitstream length (bits) | = bytes | Source |
|---|---:|---:|---|
| **XC7Z020** (PYNQ-Z1) | 32,364,512 | **4,045,564** | [U-Boot `XILINX_XC7Z020_SIZE`](https://lore.kernel.org/all/aa0505f33090ecda5c0af83eda30d69cc1aef9cb.1366641836.git.michal.simek@xilinx.com/) |
| **XC7A200T** (Nexys Video) | 77,845,216 | **9,730,652** | [UG470 Table 1-1](https://www.eng.auburn.edu/~nelsovp/courses/elec4200/FPGA/ug470_7Series_Config.pdf), p.14 |

Now measure HWAC's file:

```
py/overlay/hwac_object_tracker/hwac_object_tracker.bit    4,045,676 bytes
```

**4,045,676 − 4,045,564 = 112 bytes** — precisely the size of a `.bit` file's ASCII header
(design name, part, date, timestamp) prepended to the raw configuration frames.

The file is, to the byte, an XC7Z020 configuration image. An XC7A200T requires **2.4× more**
configuration data (9.73 MB). The Nexys Video's configuration logic would run out of bitstream
partway through and the DONE pin would never assert.

> UG470 Table 1-1 is titled "Bitstream Length" and lists a distinct, fixed value per device
> precisely because configuration data is a physical map of one specific die's frames. There is
> no conversion between them.

### 1b. HWAC's software stack requires the Zynq hard processor

From the project's own notebook, [`hwac_object_tracker.ipynb`](https://github.com/HWAC-DL/hwac_object_tracker/blob/master/py/hwac_object_tracker/hwac_object_tracker.ipynb):

```python
OVERLAY_PATH = '/home/xilinx/jupyter_notebooks/dac_2018/overlay/hwac_object_tracker/hwac_object_tracker.bit'
overlay = Overlay(OVERLAY_PATH)
xlnk = Xlnk()
fpga = hwac.FPGA(xlnk)
```

Three Zynq-only dependencies in four lines:

- **`Overlay`** — the PYNQ bitstream loader, which programs the PL *from* the PS.
- **`Xlnk`** — PYNQ's contiguous-memory allocator, backed by the Linux kernel running on the
  ARM cores.
- **`/home/xilinx/`** — the PYNQ Linux image's home directory.

The official PYNQ board list states plainly:

> "PYNQ supports Zynq based boards (Zynq, Zynq Ultrascale+, Zynq RFSoC), Kria SOMs."
> — [pynq.io/boards.html](http://www.pynq.io/boards.html)

Artix-7 is not a Zynq device and appears nowhere on that list. Running PYNQ on Artix-7 has been
asked about and answered on the PYNQ forum — it requires instantiating a MicroBlaze or Cortex-M
soft core, i.e. building the processor yourself
([discuss.pynq.io](https://discuss.pynq.io/t/artix-7-board/3914)).

### 1c. The RTL itself is wired to the Zynq PS

The repository's Verilog includes [`hw/YOLO/src/common/axi3_params.v`](https://github.com/HWAC-DL/hwac_object_tracker/blob/master/hw/YOLO/src/common/axi3_params.v).
**AXI3** is the protocol of the Zynq-7000's PS–PL interface ports; memory reaches DDR through
the PS memory controller. On a Nexys Video there is no PS — DDR3 is reached through a
[MIG](https://docs.amd.com/r/en-US/ug586_7Series_MIS) controller you instantiate in fabric, over
AXI4. That is a rework of the memory interface, not a re-synthesis.

### 1d. Board specifications, for the record

| | PYNQ-Z1 (HWAC's target) | Nexys Video (your board) |
|---|---|---|
| Device | Zynq **XC7Z020** | Artix-7 **XC7A200T** |
| Hard CPU | Dual Cortex-A9 | **None** |
| Logic cells | 85,000 | 215,360 |
| DSP slices | 220 | 740 |
| Block RAM | 4.9 Mb | 13 Mb |
| Runs PYNQ | Yes | No |

Sources: [PYNQ-Z1 board page](https://digilent.com/reference/programmable-logic/pynq-z1/reference-manual),
[Nexys Video product page](https://digilent.com/shop/nexys-video-artix-7-fpga-trainer-board-for-multimedia-applications/).

Note the Nexys Video has **more** programmable fabric — 3.4× the DSPs, 2.6× the BRAM. It is the
larger FPGA. What it lacks is the processor the entire PYNQ/HWAC software stack is built on.

---

## Claim 2 — FINN is the viable route for Artix-7, via stitched IP

### 2a. FINN's official board list also excludes Artix-7

From [FINN's getting-started documentation](https://github.com/Xilinx/finn/blob/main/docs/finn/getting_started.rst),
the supported platforms are:

> Pynq-Z1 · Pynq-Z2 · Kria SOM · Ultra96 · ZCU102 · ZCU104 · Alveo cards

Every one is Zynq, Zynq UltraScale+, or Alveo. **No Artix-7.** This must be stated honestly —
FINN does not "support" the Nexys Video in the packaged sense.

### 2b. What makes it work anyway: the stitched IP export

FINN's documentation is explicit that the generated accelerator is not platform-locked:

> "FINN generates a Vivado IP Integrator (IPI) design from the neural network with AXI stream
> (FIFO) in-out interfaces, which can be integrated onto **any Xilinx-AMD FPGA** as part of a
> larger system."
> — [FINN getting started](https://github.com/Xilinx/finn/blob/main/docs/finn/getting_started.rst)

And on the build flow:

> "…take the FINN-generated accelerator (what we call 'stitched IP' in the tutorials), wire it
> up to your FPGA design and send/receive neural network data to/from the accelerator."

So the `STITCHED_IP` build output yields an IP block with clean AXI-Stream interfaces. What FINN
will **not** do for a Nexys Video is generate the bitfile or the PYNQ driver — those steps
(`MakeZYNQProject`, `VitisLink`) are Zynq- and Alveo-specific
([hw_build.rst](https://github.com/Xilinx/finn/blob/main/docs/finn/hw_build.rst)).

**You would build the surrounding system yourself:** MIG for DDR3, AXI DMA, and a MicroBlaze or
host link to sequence inference.

### 2c. Why FINN's approach suits Artix-7 specifically

FINN targets **quantised neural networks at 1–4 bit**, which map to LUT-based XNOR/popcount
operations rather than DSP multipliers. The XC7A200T has 215,360 logic cells against only 740
DSPs — LUT-rich, DSP-poor. HWAC's fp16 datapath is the opposite shape: it consumes DSP
multipliers, which is the resource your board has least of relative to its logic.

### 2d. The alternative, for completeness

**Vitis-AI / DPU** is AMD's supported production flow and would ingest our existing `best.onnx`
without retraining — but the DPU IP requires a Zynq, Zynq UltraScale+, Versal, or Alveo target.
It cannot be deployed on Artix-7. Moving to a **Kria KV260** (~US$250) or **PYNQ-Z2** (~US$150)
would make this the shortest path by a wide margin.

---

## Summary of what the evidence supports

| Proposition | Verdict | Strength |
|---|---|---|
| HWAC's `.bit` can be loaded onto the Nexys Video | **False** | Conclusive — byte-exact size match to XC7Z020 |
| HWAC's Python flow can run on the Nexys Video | **False** | Conclusive — PYNQ is Zynq-only per vendor |
| HWAC's Verilog could be *re-synthesised* for Artix-7 | True, but costly | Requires MicroBlaze, MIG, AXI3→AXI4 rework |
| Our current model can run on HWAC's core | **False** | Architecture mismatch — 24 convs vs 9, plus unsupported ops |
| FINN is officially supported on Nexys Video | **False** | Not on the supported board list |
| FINN's stitched IP can target Nexys Video | **True** | Vendor-documented: "any Xilinx-AMD FPGA" |
| Vitis-AI can target Nexys Video | **False** | DPU requires Zynq/Versal/Alveo |

**The decision this supports:** if the Nexys Video is fixed, FINN with a self-built system is
the only credible route, and it requires retraining a quantised model from scratch — `best.pt`
cannot be converted. If the board is negotiable, a Zynq-family board reduces this from a
multi-month hardware project to a supported toolchain flow using the ONNX export we already have.

---

## Every source, in one place

**Vendor documentation**
- [UG470, 7 Series FPGAs Configuration User Guide](https://www.eng.auburn.edu/~nelsovp/courses/elec4200/FPGA/ug470_7Series_Config.pdf) — Table 1-1, p.14: bitstream lengths
- [PYNQ supported boards](http://www.pynq.io/boards.html) — "PYNQ supports Zynq based boards"
- [PYNQ forum: Artix-7 board](https://discuss.pynq.io/t/artix-7-board/3914) — soft-core requirement
- [Nexys Video product page](https://digilent.com/shop/nexys-video-artix-7-fpga-trainer-board-for-multimedia-applications/)
- [UG586, 7 Series Memory Interface Solutions (MIG)](https://docs.amd.com/r/en-US/ug586_7Series_MIS)

**FINN**
- [Getting started — supported boards, stitched IP](https://github.com/Xilinx/finn/blob/main/docs/finn/getting_started.rst)
- [Hardware build flow](https://github.com/Xilinx/finn/blob/main/docs/finn/hw_build.rst)

**HWAC reference project**
- [Repository](https://github.com/HWAC-DL/hwac_object_tracker)
- [Notebook — `Overlay`, `Xlnk`, `/home/xilinx`](https://github.com/HWAC-DL/hwac_object_tracker/blob/master/py/hwac_object_tracker/hwac_object_tracker.ipynb)
- [`hwac.py` — 9 layer methods, fp16 CMA buffer](https://github.com/HWAC-DL/hwac_object_tracker/blob/master/py/hwac_object_tracker/libraries/hwac.py)
- [`axi3_params.v` — Zynq PS interface protocol](https://github.com/HWAC-DL/hwac_object_tracker/blob/master/hw/YOLO/src/common/axi3_params.v)
- [`tiny_yolo_params.v` — `HALF_WIDTH=16`, `WEIGHTS_LEN=36`](https://github.com/HWAC-DL/hwac_object_tracker/blob/master/hw/YOLO/src/tiny_yolo_params.v)
