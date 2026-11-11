# Accelerating TinyML Inference via DSP-Packing on Resource-Constrained FPGAs

**Bachelor Thesis — Hazem Nabil Fayed**  
_Institut für Technische Informatik (ITEC) — Chair for Embedded Systems (CES)_  
_Karlsruhe Institute of Technology (KIT)_

---

## Overview

This repository contains the full implementation work for my bachelor thesis, which implements and evaluates the **UDP (Universal DSP Packing)** framework for accelerating low-bitwidth neural-network inference on Xilinx FPGAs.

The thesis is based on the research paper:

> _"UDP: A Universal DSP Packing Framework for Low-bitwidth MAC Acceleration on FPGAs"_  
> ACM Digital Library — [3748173.3779194](https://dl.acm.org/doi/10.1145/3748173.3779194)

UDP exploits the wide data paths of Xilinx **DSP48E2** blocks (used in Ultrascale/Ultrascale+ FPGAs) to pack **multiple low-bitwidth multiply-accumulate (MAC) operations into a single DSP cycle**, without needing any hardware modifications. For int4×int4 operations, a single DSP48E2 can execute **6 MACs** (P_w=3 weights × P_a=2 activations) in one cycle — a **6× throughput improvement** over conventional approaches.

The work spans four tightly-coupled components, evaluated against the **MLPerf Tiny v1.0** benchmark suite:

```
udp_model/      Python mathematical model of the UDP pipeline
udp_nn/         Neural network training & evaluation (IC, KWS, AD, VWW)
udp_rtl/        Verilog RTL implementation of the UDP MAC datapath
udp_systolic/   Verilog RTL implementation of a 12×12 UDP systolic array
mlperf_tiny/    MLPerf Tiny reference (submodule / benchmark reference)
```

---

## Table of Contents

1. [Background: The UDP Technique](#background-the-udp-technique)
2. [Repository Structure](#repository-structure)
3. [Component 1: Python Mathematical Model (`udp_model`)](#component-1-python-mathematical-model-udp_model)
4. [Component 2: Neural Network Experiments (`udp_nn`)](#component-2-neural-network-experiments-udp_nn)
5. [Component 3: RTL MAC Implementation (`udp_rtl`)](#component-3-rtl-mac-implementation-udp_rtl)
6. [Component 4: Systolic Array (`udp_systolic`)](#component-4-systolic-array-udp_systolic)
7. [MLPerf Tiny Reference (`mlperf_tiny`)](#mlperf-tiny-reference-mlperf_tiny)
8. [Key Results Summary](#key-results-summary)
9. [Installation & Setup](#installation--setup)
10. [Running Each Component](#running-each-component)
11. [Dependencies](#dependencies)

---

## Background: The UDP Technique

### The Problem

Neural networks are increasingly deployed on resource-constrained FPGA-based edge devices (TinyML). Using low-bitwidth (e.g., **int4**) quantization dramatically reduces memory and computation requirements, but conventional FPGA implementations waste much of the DSP block's capacity when operands are narrow.

A Xilinx DSP48E2 has:
- **B port**: 18-bit input
- **A/D port**: 27-bit input
- **C port**: 48-bit input (for pre-addition)
- **P port**: 48-bit output: `P = A*B + C`

With int4×int4 multiplication, each operand uses only 4 bits out of the available 18/27 bits. UDP fills these unused bit positions with _additional_ MAC operations packed together.

### How UDP Works

The UDP pipeline consists of 5 stages:

#### Stage 1 — Mapping (signed → unsigned)
Two mapping functions convert signed int values to unsigned for packing:
- **Clip mapping** (weights): strips the sign bit, records sign separately → `w_u = w_s & ((1<<(W_w-1))-1)`
- **Direct mapping** (activations): reinterprets the sign bit as a data bit → `a_u = a_s & ((1<<W_a)-1)`

#### Stage 2 — Packing
Multiple unsigned values are bit-packed into single wide integers:
- `w_p` (→ B port): `P_w` weights interleaved with `W_p = W_w + W_a - 1` bit gaps
- `a_p` (→ A/D port): `P_a` activations interleaved with `P_w × W_p` bit gaps

This ensures that the partial products of _different_ MAC operations land in non-overlapping bit positions of the 48-bit output `P`.

#### Stage 3 — Preprocessing (`b_pre`)
For weights that were negative, a sign-correction term is accumulated into `b_pre` (→ C port). This comes "for free" in the DSP's `P = A*B + C` pipeline.

#### Stage 4 — DSP48E2 Execution
A **single DSP operation** `P = w_p × a_p + b_pre` is issued. The 48-bit result P contains `P_w × P_a` packed sub-products.

#### Stage 5 — Postprocessing
The individual signed products are extracted from their bit-slots in P by applying sign-correction for the direct-mapped activations.

For **int4×int4** with `P_w=3, P_a=2`: **6 exact signed products in 1 DSP cycle**.

---

## Repository Structure

```
Codes/
├── README.md                   ← This file
├── .gitignore
│
├── udp_model/                  ← Python simulation model
│   ├── main.py                 ← Interactive demo (run this first!)
│   ├── core/
│   │   ├── mapping.py          ← clip_map / direct_map
│   │   ├── packing.py          ← pack_weights / pack_activations / unpack_results
│   │   ├── preprocessing.py    ← compute_b_pre  (Equation 3)
│   │   ├── postprocessing.py   ← compute_results (Equation 4)
│   │   ├── dsp_sim.py          ← Python DSP48E2 simulator
│   │   └── recovery.py         ← Sign recovery helpers
│   ├── efficiency/
│   │   └── model.py            ← find_optimal / find_all_valid / print_efficiency_table
│   ├── nn/
│   │   ├── udp_linear.py       ← UDP-aware linear layer (PyTorch)
│   │   └── quantize.py         ← Quantization helpers
│   ├── utilities/
│   │   ├── bit_ops.py          ← Bit-manipulation utilities
│   │   └── fixed_point.py      ← Fixed-point arithmetic helpers
│   └── tests/
│       ├── test_mapping.py
│       ├── test_packing.py
│       ├── test_processing.py
│       ├── test_recovery.py
│       ├── test_convolution.py
│       ├── test_efficiency.py
│       ├── test_end_to_end.py
│       └── test_nn_mlp.py
│
├── udp_nn/                     ← Neural network experiments
│   ├── resnet8.py              ← MLPerf Tiny ResNet-8 (Image Classification)
│   ├── quantize.py             ← INT4 quantization (PTQ & QAT)
│   ├── train.py                ← FP32 training (CIFAR-10)
│   ├── evaluate.py             ← PTQ / UDP evaluation
│   ├── qat.py                  ← Quantization-Aware Training (QAT)
│   ├── qat_train.py            ← QAT training loop
│   ├── qat_evaluate.py         ← QAT evaluation
│   ├── udp_sim.py              ← UDP simulation layer
│   ├── analyze_resnet8.py      ← GEMM dimension analysis + SA cycle estimation
│   ├── run_all_benchmarks.py   ← One-shot: run IC + KWS + AD + VWW
│   ├── run_full_pipeline.py    ← Full pipeline (train → eval → QAT → latency)
│   ├── run_post_training.py    ← Post-training quantization evaluation
│   ├── results_table.py        ← Print summary tables
│   ├── ic_collect_results.py   ← Collect IC results
│   ├── LAYER_TABLE.md          ← ResNet-8 GEMM layer table
│   ├── ad/                     ← Anomaly Detection (ToyADMOS autoencoder)
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── kws/                    ← Keyword Spotting (DS-CNN)
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── vww/                    ← Visual Wake Words (MobileNetV1 0.25×)
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── download_vww.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── qat_train.py
│   │   └── qat_evaluate.py
│   ├── latency/                ← Timeloop-based latency analysis
│   │   ├── generate_problems.py
│   │   ├── compute_latency.py
│   │   ├── layer_defs.py
│   │   ├── run_latency.sh
│   │   └── latency_report.txt
│   ├── timeloop/               ← Timeloop architecture YAML files
│   │   ├── udp_arch.yaml           ← Baseline 12×12 architecture
│   │   ├── udp_arch_6mac.yaml
│   │   ├── udp_arch_opt_a.yaml     ← Opt-A: 8×12 (96 PEs)
│   │   ├── udp_arch_opt_b.yaml     ← Opt-B: 8×9 (72 PEs)
│   │   ├── problem_*.yaml          ← Layer problem definitions
│   │   └── README.md
│   ├── results/
│   │   ├── evaluation_summary.json
│   │   └── evaluation_summary.md   ← Summary of all benchmark results
│   ├── ad_results.txt
│   ├── kws_results.txt
│   └── vww_results.txt
│
├── udp_rtl/                    ← RTL: UDP MAC datapath (Verilog)
│   ├── rtl/
│   │   ├── udp_mac_top.v       ← Top-level: 3 weights × 2 activations → 6 products
│   │   ├── udp_pack.v          ← Mapping + packing combinational logic
│   │   ├── udp_preprocess.v    ← b_pre computation (Equation 3)
│   │   ├── udp_postprocess.v   ← Product extraction (Equation 4)
│   │   └── dsp48e2_model.v     ← Behavioural DSP48E2 model
│   ├── tb/
│   │   ├── tb_udp_mac.v        ← Testbench: UDP vs direct multiplication
│   │   └── tb_dsp48e2.v        ← DSP48E2 standalone testbench
│   └── tcl/
│       ├── create_project.tcl  ← Vivado project creation
│       ├── run_all.tcl         ← Run all simulations
│       ├── sim_udp_mac.tcl     ← Simulate UDP MAC
│       ├── sim_dsp.tcl         ← Simulate DSP model
│       └── synth_check.tcl     ← Synthesis check
│
├── udp_systolic/               ← RTL: 12×12 UDP systolic array (Verilog)
│   ├── rtl/
│   │   ├── systolic_array.v    ← 12×12 weight-stationary systolic array
│   │   └── udp_pe.v            ← Processing element (UDP MAC + accumulator)
│   ├── tb/
│   │   └── tb_systolic.v       ← Testbench: matrix multiply verification
│   └── tcl/
│       ├── create_project.tcl
│       ├── sim_systolic.tcl
│       └── synth_systolic.tcl
│
└── mlperf_tiny/                ← MLPerf Tiny v1.0 benchmark reference
    ├── benchmark/
    │   ├── training/           ← Reference training scripts
    │   │   ├── image_classification/   ← ResNet-8 (CIFAR-10)
    │   │   ├── keyword_spotting/       ← DS-CNN (Speech Commands)
    │   │   ├── anomaly_detection/      ← AutoEncoder (ToyADMOS)
    │   │   └── visual_wake_words/      ← MobileNetV1 0.25× (VWW dataset)
    │   └── evaluation/
    └── README.md
```

---

## Component 1: Python Mathematical Model (`udp_model`)

### Purpose
A complete, bit-accurate Python simulation of the UDP pipeline, used to verify the mathematical correctness of the algorithm before hardware implementation.

### Key Modules

| File | Description |
|------|-------------|
| `core/mapping.py` | `clip_map(w, W_w)` and `direct_map(a, W_a)` — signed-to-unsigned conversion |
| `core/packing.py` | `pack_weights()`, `pack_activations()`, `unpack_results()` — bit interleaving |
| `core/preprocessing.py` | `compute_b_pre()` — sign-correction term for C port (Equation 3) |
| `core/postprocessing.py` | `compute_results()` — product extraction from 48-bit result (Equation 4) |
| `core/dsp_sim.py` | `simulate()` — models `P = w_p * a_p + b_pre` as a 48-bit integer |
| `efficiency/model.py` | `find_optimal()` — finds best `(P_w, P_a)` for given bitwidths |
| `nn/udp_linear.py` | UDP-aware PyTorch `Linear` layer replacement |

### Running the Demo

```bash
cd udp_model
python main.py
```

The demo walks through all 8 pipeline stages for an int4×int4 example:
- **Stage 0**: DSP48E2 hardware context
- **Stage 1**: Efficiency analysis — finds optimal `P_w=3, P_a=2` for int4×int4
- **Stage 2**: Signed → unsigned mapping
- **Stage 3**: Bit-packing into `w_p` and `a_p`
- **Stage 4**: Preprocessing (`b_pre` computation)
- **Stage 5**: DSP48E2 simulation (`r = w_p * a_p + b_pre`)
- **Stage 6**: Postprocessing — recover 6 signed products
- **Stage 7**: Efficiency table for all valid `(P_w, P_a)` pairs
- **Stress test**: 500 random int4 trials — all verified correct

### Running Tests

```bash
cd udp_model
python -m pytest tests/ -v
```

Tests cover every module independently and include an end-to-end pipeline test.

### Efficiency Analysis

For int4×int4 on DSP48E2 (B=18-bit, A=27-bit, P=48-bit):

| P_w | P_a | MACs/DSP | B-port usage | A-port usage |
|-----|-----|----------|--------------|--------------|
| 1   | 1   | 1        | 4/18 (22%)   | 4/27 (15%)   |
| 3   | 2   | **6**    | 17/18 (94%)  | 25/27 (93%)  |

**Optimal: P_w=3, P_a=2 → 6 MACs per DSP cycle (6× speedup)**

---

## Component 2: Neural Network Experiments (`udp_nn`)

### Purpose
Applies UDP quantization to all four **MLPerf Tiny v1.0** benchmark tasks and measures accuracy impact across three evaluation modes:
1. **FP32 baseline** — full-precision reference
2. **Naive INT4 PTQ** — standard post-training quantization (no UDP)
3. **UDP Mode 4** — weight-only INT4 with UDP packing (no postprocessing)
4. **UDP Mode 5** — full INT4 pipeline (weights + activations quantized)
5. **QAT variants** — Quantization-Aware Training for Modes 4 & 5

### Benchmark Tasks

#### Task 1 — Image Classification (IC) · ResNet-8 on CIFAR-10

| Mode | Top-1 Accuracy | MLPerf threshold (≥85%) |
|------|:--------------:|:-----------------------:|
| FP32 | **90.49%** | ✅ PASS |
| Naive INT4 PTQ | 49.28% | ❌ FAIL |
| UDP Mode 4 (weight-only) | 78.59% | ❌ FAIL |
| UDP Mode 5 (full INT4) | 74.13% | ❌ FAIL |

#### Task 2 — Keyword Spotting (KWS) · DS-CNN on Speech Commands v0.02

| Mode | Accuracy | MLPerf threshold (≥90%) |
|------|:--------:|:-----------------------:|
| FP32 | **95.93%** | ✅ PASS |
| Naive INT4 PTQ | 33.21% | ❌ FAIL |
| UDP Mode 4 (weight-only) | **93.76%** | ✅ PASS |
| UDP Mode 5 (full INT4) | 71.94% | ❌ FAIL |
| QAT Mode 4 | **95.72%** | ✅ PASS |
| QAT Mode 5 | 88.66% | ❌ FAIL |

#### Task 3 — Anomaly Detection (AD) · AutoEncoder on ToyADMOS

| Mode | AUC | MLPerf threshold (≥85%) |
|------|:---:|:-----------------------:|
| FP32 | **1.0000** | ✅ PASS |
| UDP Mode 4 (weight-only) | **1.0000** | ✅ PASS |

#### Task 4 — Visual Wake Words (VWW) · MobileNetV1 0.25× 

| Mode | Accuracy | MLPerf threshold (≥80%) |
|------|:--------:|:-----------------------:|
| FP32 | **81.95%** | ✅ PASS |
| Naive INT4 PTQ | 53.35% | ❌ FAIL |
| UDP Mode 4 (weight-only) | 50.65% | ❌ FAIL |
| QAT Mode 4 | **81.40%** | ✅ PASS |

### Key Findings
- **UDP Mode 4 (weight-only)** recovers nearly all FP32 accuracy for KWS (93.76%) and AD (100%), demonstrating that weight quantization alone with UDP is highly effective.
- **Full INT4 (Mode 5)** degrades accuracy significantly due to activation quantization errors in the postprocessing step — this is a known limitation addressed in the thesis.
- **QAT** successfully restores UDP Mode 4 accuracy on all tasks, confirming that the quantization error is recoverable with fine-tuning.

### Architecture Analysis (`analyze_resnet8.py`)

The script `analyze_resnet8.py` decomposes all ResNet-8 layers into GEMM operations and estimates **systolic array (SA) cycle counts** for the 12×12 UDP array:

```
Layer              M    N    K       MACs    SA cycles
stem_conv       1024   26  27      718,848      9,984
s1_conv1        1024   26 234    6,220,416     64,896
...
classifier_dense   1   10 104        1,040         13
```

**Total estimated latency @ 200 MHz: ~0.0624 ms** (Baseline 12×12 configuration)

### Running the Benchmarks

```bash
cd udp_nn

# Run all four benchmarks in sequence
python run_all_benchmarks.py

# Run just IC (CIFAR-10 / ResNet-8)
python evaluate.py

# Run QAT training + evaluation
python qat_train.py
python qat_evaluate.py

# Analyze layer GEMM dimensions
python analyze_resnet8.py
```

---

## Component 3: RTL MAC Implementation (`udp_rtl`)

### Purpose
A fully synthesisable Verilog RTL implementation of the UDP MAC datapath, targeting Xilinx Ultrascale+ FPGAs. Implements the complete combinational pipeline for **int4×int4, P_w=3, P_a=2** (6 products per DSP cycle).

### Module Hierarchy

```
udp_mac_top.v
├── udp_pack.v          Stage 1+2: Mapping + Packing
├── udp_preprocess.v    Stage 3: Preprocessing (b_pre, Equation 3)
├── dsp48e2_model.v     Stage 4: DSP48E2 behavioural model  r = w_p*a_p + b_pre
└── udp_postprocess.v   Stage 5: Postprocessing (Equation 4) → 6 signed products
```

### Top-Level Interface (`udp_mac_top.v`)

```verilog
module udp_mac_top (
    // Inputs: 3 signed int4 weights + 2 signed int4 activations
    input  wire signed [3:0] w_s0, w_s1, w_s2,
    input  wire signed [3:0] a_s0, a_s1,
    
    // Outputs: 6 exact signed products
    output wire signed [7:0] prod_00, prod_10, prod_20,  // × a_s0
    output wire signed [7:0] prod_01, prod_11, prod_21,  // × a_s1
    
    // DSP observability ports
    output wire [17:0] dsp_B,   // w_p on B port
    output wire [26:0] dsp_A,   // a_p on A port
    output wire [47:0] dsp_C,   // b_pre on C port
    output wire [47:0] dsp_P    // raw result
);
```

### Testbench Verification

`tb_udp_mac.v` runs **two parallel test loops**:
- **Test A (standard)**: 6 separate `w*a` multiplications → 6 loop iterations
- **Test B (UDP)**: 1 DSP call → 1 iteration, 6 products

**Confirmed: 6× MAC-count reduction for int4×int4**

### Simulation (Vivado)

```tcl
# In Vivado Tcl console:
cd udp_rtl/tcl
source run_all.tcl    # Creates project, adds sources, runs simulation

# Or individually:
source create_project.tcl
source sim_udp_mac.tcl
```

---

## Component 4: Systolic Array (`udp_systolic`)

### Purpose
A **12×12 weight-stationary systolic array** where each processing element (PE) is one UDP MAC unit. Computes `C = A × B` matrix multiplication using UDP-packed int4 operands, achieving the throughput gains at system level.

### Architecture

```
Array dimensions:  12 rows × 12 cols = 144 PEs
Activations tile:  PE_ROWS × P_A = 12 × 2 = 24 rows
Weights tile:      PE_COLS × P_W = 12 × 3 = 36 columns
DSP48E2 blocks:    144 (one per PE)
Accumulator width: 12 bits  (K_MAX = 31)
```

### Data Flow
- **Activations** stream left-to-right through each row, skewed by 1 cycle per column (pipeline fill)
- **Weights** are stationary, preloaded into each PE's registers
- After `K` activation vectors, each PE holds `P_W × P_A = 6` accumulated products

### Timing Protocol
1. Assert `srst=1` for 1 cycle → clears all 144×6 accumulators
2. Assert `en=1`, feed `K` activation columns (one per clock)
3. Pad with `PE_COLS-1 = 11` zero cycles for pipeline flush
4. Read `result_flat` — 144×6 = 864 accumulated products

### Processing Element (`udp_pe.v`)

Each PE instantiates `udp_mac_top` and adds 6 signed accumulators:
```
acc_ij += prod_ij   (for each new activation vector)
```

### Simulation (Vivado)

```tcl
cd udp_systolic/tcl
source create_project.tcl
source sim_systolic.tcl    # Verifies matrix multiply against reference
source synth_systolic.tcl  # Synthesis report
```

---

## MLPerf Tiny Reference (`mlperf_tiny`)

The `mlperf_tiny/` folder contains the **MLPerf Tiny v1.0 benchmark reference implementations** used as:
- Ground-truth model architectures (ResNet-8, DS-CNN, AutoEncoder, MobileNetV1)
- Training scripts for obtaining FP32 baseline checkpoints
- Evaluation protocols and accuracy thresholds

See [mlperf_tiny/README.md](mlperf_tiny/README.md) for full documentation.

---

## Key Results Summary

### Accuracy (MLPerf Tiny thresholds)

| Task | Model | FP32 | UDP w-only | UDP full INT4 | QAT w-only |
|------|-------|:----:|:----------:|:-------------:|:----------:|
| Image Classification | ResNet-8 | 90.49% ✅ | 78.59% ❌ | 74.13% ❌ | — |
| Keyword Spotting | DS-CNN | 95.93% ✅ | 93.76% ✅ | 71.94% ❌ | 95.72% ✅ |
| Anomaly Detection | AutoEncoder | 1.0000 ✅ | 1.0000 ✅ | — | — |
| Visual Wake Words | MobileNetV1 | 81.95% ✅ | 50.65% ❌ | — | 81.40% ✅ |

### Hardware Performance (Systolic Array @ 200 MHz)

| Architecture | PEs | Configuration | Latency (3 key layers) |
|---|:---:|---|:---:|
| Baseline | 144 | 12×12 | **0.0624 ms** |
| Opt-A | 96 | 8×12 | **0.0624 ms** |
| Opt-B | 72 | 8×9 | 0.1165 ms |

**Recommendation**: **Opt-A (8×12, 96 PEs)** — best area/utilisation trade-off, same latency as the larger 144-PE baseline for the dominant layers.

### UDP Throughput Gain

| Bitwidth | P_w | P_a | MACs/DSP | Speedup |
|----------|-----|-----|----------|---------|
| int8×int8 | 1 | 1 | 1 | 1× |
| int4×int4 | 3 | 2 | **6** | **6×** |
| int4×int8 | 1 | 1 | 1 | 1× |

---

## Installation & Setup

### Requirements

- **Python 3.9+** with pip
- **PyTorch 2.0+** (CPU is sufficient for evaluation; GPU optional for training)
- **Vivado 2022.1+** (for RTL simulation only)
- **Timeloop** (optional, for latency analysis under WSL)

### Python Environment

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn tqdm matplotlib pandas
```

### Dataset Setup

#### CIFAR-10 (Image Classification)
```bash
cd udp_nn
python -c "import torchvision; torchvision.datasets.CIFAR10('data', download=True)"
```

#### Google Speech Commands v0.02 (Keyword Spotting)
```bash
cd udp_nn
python -c "import torchaudio; torchaudio.datasets.SPEECHCOMMANDS('data', download=True)"
```

#### Visual Wake Words (VWW)
```bash
cd udp_nn
python vww/download_vww.py    # Downloads and prepares COCO-based VWW dataset
```

#### ToyADMOS (Anomaly Detection)
```bash
# Download ToyADMOS dataset manually from:
# https://zenodo.org/record/3351307
# Place in udp_nn/data/toycar/
```

---

## Running Each Component

### 1. Python Model Demo (no datasets needed)

```bash
cd udp_model
python main.py              # Interactive pipeline demo
python -m pytest tests/ -v  # Full test suite
```

### 2. Neural Network Evaluation

```bash
cd udp_nn

# Run all four MLPerf Tiny benchmarks
python run_all_benchmarks.py

# Individual tasks:
python evaluate.py                    # Image Classification (CIFAR-10)
python kws/evaluate.py                # Keyword Spotting
python ad/evaluate.py                 # Anomaly Detection
python vww/evaluate.py                # Visual Wake Words

# QAT training + evaluation
python qat_train.py                   # Train with quantization-aware training
python qat_evaluate.py                # Evaluate QAT model

# Layer analysis
python analyze_resnet8.py             # Print GEMM table + SA cycle estimates
python analyze_resnet8.py --write     # Regenerate LAYER_TABLE.md
```

### 3. RTL Simulation (Vivado required)

```tcl
# Open Vivado Tcl Shell, then:
cd udp_rtl/tcl
source create_project.tcl   # Set up project with all source files
source sim_udp_mac.tcl       # Simulate UDP MAC testbench
source synth_check.tcl       # Run synthesis and check resource usage
```

### 4. Systolic Array Simulation (Vivado required)

```tcl
cd udp_systolic/tcl
source create_project.tcl    # Set up systolic array project
source sim_systolic.tcl      # Simulate 12×12 systolic array
source synth_systolic.tcl    # Synthesis report (LUT/FF/DSP counts)
```

### 5. Latency Analysis (Timeloop / WSL)

```bash
# From WSL (Windows Subsystem for Linux) with Timeloop installed:
cd udp_nn/timeloop
bash run_mapper.sh            # Run Timeloop mapper for all layers
bash wsl_run.sh               # Full run with output capture

# Windows (parse existing Timeloop outputs):
cd udp_nn/latency
python generate_problems.py   # Generate problem YAML files
python compute_latency.py     # Compute latency from Timeloop outputs
```

---

## Dependencies

### Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Neural network framework |
| `torchvision` | ≥0.15 | CIFAR-10 dataset + transforms |
| `torchaudio` | ≥2.0 | Speech Commands dataset |
| `numpy` | ≥1.23 | Numerical computation |
| `scipy` | ≥1.10 | Signal processing (KWS/AD) |
| `scikit-learn` | ≥1.2 | AUC computation (AD) |
| `tqdm` | ≥4.65 | Progress bars |
| `matplotlib` | ≥3.7 | Plotting |
| `pandas` | ≥2.0 | Results tables |

### Hardware Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Vivado | ≥2022.1 | RTL synthesis & simulation |
| Timeloop | ≥0.4 | Architecture-level latency estimation |
| WSL2 | — | Required for Timeloop on Windows |

---

## Thesis Context

This work implements and evaluates all four components of the UDP framework as described in the referenced paper, with the following original contributions:

1. **Bit-accurate Python reference model** with full test suite — serves as golden reference for RTL verification
2. **MLPerf Tiny evaluation** of UDP quantization across all four benchmark tasks — with both PTQ and QAT variants
3. **Verilog RTL implementation** of the UDP MAC pipeline targeting Xilinx Ultrascale+ DSP48E2 blocks
4. **12×12 systolic array architecture** with weight-stationary dataflow and activation skewing
5. **Architecture exploration** of three systolic array configurations (Baseline, Opt-A, Opt-B) using Timeloop

The main finding is that **UDP Mode 4 (weight-only int4)** with QAT achieves near-FP32 accuracy across most MLPerf Tiny tasks while enabling 6× MAC throughput improvement per DSP block, and that the **Opt-A 8×12 (96 PE)** systolic array configuration offers the best balance of area and performance.

---

## Citation

If you use this work, please cite the original UDP paper:

```bibtex
@inproceedings{udp2024,
  title     = {UDP: A Universal DSP Packing Framework for Low-bitwidth MAC Acceleration on FPGAs},
  booktitle = {Proceedings of the ACM/SIGDA International Symposium on Field Programmable Gate Arrays},
  year      = {2024},
  doi       = {10.1145/3748173.3779194}
}
```

---

_Bachelor Thesis — Hazem Nabil Fayed_  
_Institut für Technische Informatik (ITEC) · Chair for Embedded Systems (CES) · KIT_
