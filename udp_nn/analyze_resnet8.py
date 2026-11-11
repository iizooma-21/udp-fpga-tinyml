"""
analyze_resnet8.py
==================
Extract GEMM dimensions (M, N, K) for every compute layer in the MLPerf Tiny
ResNet-8 model (resnet_v1_eembc, conv_filters=26).

Source architecture:
  mlperf_tiny/benchmark/training/image_classification/keras_model.py

Each conv layer is expressed as a matrix multiply via im2col:

    C[M, N] = A[M, K] @ B[K, N]

    M = output_height * output_width   (spatial positions)
    N = output_channels
    K = kernel_h * kernel_w * input_channels

The final classifier is a dense layer after global average pooling:

    M = batch_size (1 for single-image inference)
    N = num_classes (10)
    K = flattened feature dim (104)

SA cycle estimate (matches LAYER_TABLE.md placeholder scheduling):
  - Output tile: 24 rows (12 PE rows x P_a=2), 36 cols (12 PE cols x P_w=3)
  - Weight-stationary skew: +11 cycles per K-chunk (PE_COLS - 1)
  - K tiled when K > K_MAX=31 (ACC_WIDTH=12 on the systolic array)
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# ResNet-8 layer graph (mirrors keras_model.resnet_v1_eembc, conv_filters=26)
# ---------------------------------------------------------------------------

INPUT_H, INPUT_W, INPUT_C = 32, 32, 3
CONV_FILTERS = 26
NUM_CLASSES = 10
BATCH_SIZE = 1

# UDP systolic array parameters (12x12 PE grid, int4 P_w=3, P_a=2)
PE_ROWS = 12
PE_COLS = 12
P_W = 3
P_A = 2
TILE_M = PE_ROWS * P_A          # 24 output rows per tile
TILE_N = PE_COLS * P_W          # 36 output cols per tile
SKEW = PE_COLS - 1              # 11 activation pipeline fill cycles
K_MAX = 31                      # max inner dim per ACC_WIDTH=12 accumulation


@dataclass
class LayerSpec:
    name: str
    op: str
    in_shape: Tuple[int, int, int]   # H, W, C
    out_channels: int
    kernel: Tuple[int, int]
    stride: int


@dataclass
class LayerGEMM:
    name: str
    op: str
    M: int
    N: int
    K: int
    macs: int
    sa_cycles: int
    in_shape: Tuple[int, int, int]
    out_shape: Tuple[int, int, int]


def _out_spatial(h: int, w: int, stride: int) -> Tuple[int, int]:
    """Keras 'same' padding: spatial size = ceil(input / stride)."""
    return math.ceil(h / stride), math.ceil(w / stride)


def _build_layer_list(conv_filters: int = CONV_FILTERS) -> List[LayerSpec]:
    """Static layer list derived from keras_model.resnet_v1_eembc."""
    f1 = conv_filters
    f2 = conv_filters * 2
    f3 = conv_filters * 4

    return [
        LayerSpec("stem_conv", "conv", (32, 32, 3), f1, (3, 3), 1),
        LayerSpec("s1_conv1", "conv", (32, 32, f1), f1, (3, 3), 1),
        LayerSpec("s1_conv2", "conv", (32, 32, f1), f1, (3, 3), 1),
        LayerSpec("s2_conv1", "conv", (32, 32, f1), f2, (3, 3), 2),
        LayerSpec("s2_conv2", "conv", (16, 16, f2), f2, (3, 3), 1),
        LayerSpec("s2_shortcut", "conv", (32, 32, f1), f2, (1, 1), 2),
        LayerSpec("s3_conv1", "conv", (16, 16, f2), f3, (3, 3), 2),
        LayerSpec("s3_conv2", "conv", (8, 8, f3), f3, (3, 3), 1),
        LayerSpec("s3_shortcut", "conv", (16, 16, f2), f3, (1, 1), 2),
    ]


def conv_to_gemm(spec: LayerSpec) -> LayerGEMM:
    kh, kw = spec.kernel
    oh, ow = _out_spatial(spec.in_shape[0], spec.in_shape[1], spec.stride)
    _, _, ic = spec.in_shape

    M = oh * ow
    N = spec.out_channels
    K = kh * kw * ic
    macs = M * N * K

    return LayerGEMM(
        name=spec.name,
        op=spec.op,
        M=M,
        N=N,
        K=K,
        macs=macs,
        sa_cycles=estimate_sa_cycles(M, N, K),
        in_shape=spec.in_shape,
        out_shape=(oh, ow, spec.out_channels),
    )


def dense_to_gemm(
    name: str,
    in_features: int,
    out_features: int,
    batch: int = BATCH_SIZE,
) -> LayerGEMM:
    M = batch
    N = out_features
    K = in_features
    macs = M * N * K
    return LayerGEMM(
        name=name,
        op="dense",
        M=M,
        N=N,
        K=K,
        macs=macs,
        sa_cycles=estimate_sa_cycles(M, N, K),
        in_shape=(1, 1, in_features),
        out_shape=(1, 1, out_features),
    )


def k_chunk_cycles(k: int) -> int:
    """Clock cycles to accumulate inner dimension K on the SA (with K-tiling)."""
    k_full = k // K_MAX
    k_rem = k % K_MAX
    if k_rem == 0:
        return k_full * (K_MAX + SKEW)
    return k_full * (K_MAX + SKEW) + (k_rem + SKEW)


def estimate_sa_cycles(m: int, n: int, k: int) -> int:
    """Placeholder SA scheduling used in LAYER_TABLE.md."""
    tiles_m = math.ceil(m / TILE_M)
    tiles_n = math.ceil(n / TILE_N)
    return tiles_m * tiles_n * k_chunk_cycles(k)


def analyze_resnet8(conv_filters: int = CONV_FILTERS) -> List[LayerGEMM]:
    layers: List[LayerGEMM] = []
    for spec in _build_layer_list(conv_filters):
        layers.append(conv_to_gemm(spec))

    # Global average pool on 8x8x104 -> 104-d vector, then Dense(10)
    f3 = conv_filters * 4
    layers.append(dense_to_gemm("classifier_dense", f3, NUM_CLASSES))
    return layers


def verify_with_tensorflow(conv_filters: int = CONV_FILTERS) -> None:
    """Optional: build Keras model and print Conv2D/Dense output shapes."""
    mlperf_root = Path(__file__).resolve().parents[1] / "mlperf_tiny" / "benchmark" / "training" / "image_classification"
    sys.path.insert(0, str(mlperf_root))
    import keras_model  # noqa: WPS433

    model = keras_model.resnet_v1_eembc(conv_filters=conv_filters)
    print("\nTensorFlow layer shapes:")
    for layer in model.layers:
        if layer.__class__.__name__ in ("Conv2D", "Dense"):
            cfg = layer.get_config()
            print(f"  {layer.name:20s}  {layer.__class__.__name__:6s}  "
                  f"out={layer.output_shape}  k={cfg.get('kernel_size')}  s={cfg.get('strides')}")


def format_markdown(layers: List[LayerGEMM], conv_filters: int, clock_mhz: float = 200.0) -> str:
    total_macs = sum(l.macs for l in layers)
    total_cycles = sum(l.sa_cycles for l in layers)
    latency_ms = total_cycles / (clock_mhz * 1e6) * 1e3

    lines = [
        "# MLPerf Tiny ResNet-8 — Layer Table",
        "",
        f"Generated by `analyze_resnet8.py`. conv_filters={conv_filters}.",
        "",
        "| Layer | Op | M | N | K | MACs | SA cycles (est.) |",
        "|-------|----|---|---|---|------|------------------|",
    ]
    for layer in layers:
        lines.append(
            f"| {layer.name} | {layer.op} | {layer.M} | {layer.N} | {layer.K} | "
            f"{layer.macs:,} | {layer.sa_cycles:,} |"
        )
    lines += [
        "",
        f"**Total MACs:** {total_macs:,}  ",
        f"**Total SA cycles (est.):** {total_cycles:,} @ placeholder scheduling  ",
        f"**Placeholder latency:** {latency_ms:.4f} ms @ {clock_mhz:.0f} MHz",
        "",
    ]
    return "\n".join(lines)


def compare_with_reference(layers: List[LayerGEMM], ref_path: Path) -> bool:
    """Parse existing LAYER_TABLE.md and compare numeric columns."""
    if not ref_path.exists():
        print(f"Reference table not found: {ref_path}")
        return False

    ref_rows = {}
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("| Layer") or line.startswith("|-"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 7:
            continue
        name = parts[0]
        ref_rows[name] = {
            "op": parts[1],
            "M": int(parts[2]),
            "N": int(parts[3]),
            "K": int(parts[4]),
            "macs": int(parts[5].replace(",", "")),
            "cycles": int(parts[6].replace(",", "")),
        }

    all_ok = True
    print(f"\nComparing {len(layers)} layers against {ref_path.name}:\n")
    print(f"{'Layer':<18} {'Field':<8} {'Computed':>12} {'Reference':>12} {'Match':>6}")
    print("-" * 62)

    for layer in layers:
        ref = ref_rows.get(layer.name)
        if ref is None:
            print(f"{layer.name:<18} {'MISSING':<8}")
            all_ok = False
            continue

        checks = [
            ("M", layer.M, ref["M"]),
            ("N", layer.N, ref["N"]),
            ("K", layer.K, ref["K"]),
            ("MACs", layer.macs, ref["macs"]),
            ("cycles", layer.sa_cycles, ref["cycles"]),
            ("op", layer.op, ref["op"]),
        ]
        for field, got, expected in checks:
            ok = got == expected
            all_ok = all_ok and ok
            mark = "OK" if ok else "FAIL"
            print(f"{layer.name:<18} {field:<8} {str(got):>12} {str(expected):>12} {mark:>6}")

    print("-" * 62)
    print("RESULT:", "ALL MATCH" if all_ok else "MISMATCH — see FAIL rows above")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MLPerf Tiny ResNet-8 GEMM layers")
    parser.add_argument("--conv-filters", type=int, default=CONV_FILTERS)
    parser.add_argument("--write", action="store_true", help="Regenerate LAYER_TABLE.md")
    parser.add_argument("--verify-tf", action="store_true", help="Print Keras layer shapes")
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path(__file__).parent / "LAYER_TABLE.md",
        help="Reference table for comparison",
    )
    args = parser.parse_args()

    layers = analyze_resnet8(conv_filters=args.conv_filters)

    print("MLPerf Tiny ResNet-8 — GEMM analysis")
    print(f"conv_filters={args.conv_filters}, batch={BATCH_SIZE}")
    print(f"SA tile: {TILE_M}x{TILE_N}, K_MAX={K_MAX}, skew={SKEW}\n")
    print(f"{'Layer':<18} {'Op':<6} {'M':>6} {'N':>6} {'K':>6} {'MACs':>12} {'SA cyc':>10}")
    print("-" * 72)
    for layer in layers:
        print(
            f"{layer.name:<18} {layer.op:<6} {layer.M:>6} {layer.N:>6} {layer.K:>6} "
            f"{layer.macs:>12,} {layer.sa_cycles:>10,}"
        )

    total_macs = sum(l.macs for l in layers)
    total_cycles = sum(l.sa_cycles for l in layers)
    print("-" * 72)
    print(f"{'TOTAL':<18} {'':<6} {'':>6} {'':>6} {'':>6} {total_macs:>12,} {total_cycles:>10,}")
    print(f"\nPlaceholder latency @ 200 MHz: {total_cycles / 200e6 * 1e3:.4f} ms")

    compare_with_reference(layers, args.ref)

    if args.verify_tf:
        verify_with_tensorflow(args.conv_filters)

    if args.write:
        out = Path(__file__).parent / "LAYER_TABLE.md"
        out.write_text(format_markdown(layers, args.conv_filters), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
