"""
layer_defs.py — Network layer definitions for MLPerf Tiny latency estimation.

Each entry in a LAYERS list is a tuple:
    (name, C, K, R, S, P, Q, stride, count)

  name   : unique string identifier (used for problem YAML filenames)
  C      : input channels  (depthwise: C=1, K=num_channels)
  K      : output channels
  R, S   : kernel height, width
  P, Q   : OUTPUT spatial height, width (after stride/padding)
  stride : convolution stride (Hstride = Wstride = stride)
  count  : how many times this layer appears in the full network

EXISTING maps a layer name to the stem of an already-computed Timeloop stats
file in  ../timeloop/outputs/<stem>_<arch>_stats.txt  (or similar).
When a layer has an EXISTING entry, no new Timeloop run is needed.

Note: for KWS conv1, the stride is 2 in BOTH H and W dimensions.
      The problem YAML generator handles this via Hstride/Wstride fields.
"""

# =============================================================================
# IC  ─  ResNet-8  (MLPerf Tiny, conv_filters=26, CIFAR-10 32×32)
# =============================================================================
# Topology
#   Stem         : Conv(3→26, 3×3, s=1)  @ 32×32
#   Stack 1      : 2 × Conv(26→26, 3×3, s=1)  @ 32×32   (identity shortcut)
#   Stack 2 conv : Conv(26→52, 3×3, s=2) + Conv(52→52, 3×3, s=1)  @ 16×16
#   Stack 2 sc   : Conv(26→52, 1×1, s=2)  @ 16×16  (projection shortcut)
#   Stack 3 conv : Conv(52→104,3×3, s=2) + Conv(104→104,3×3,s=1)  @ 8×8
#   Stack 3 sc   : Conv(52→104, 1×1, s=2) @ 8×8   (projection shortcut)
#   Head         : AvgPool(8×8) → Linear(104→10)

IC_LAYERS = [
    # (name,       C,   K,   R, S,  P,  Q, stride, count)
    ("ic_stem",    3,  26,   3, 3, 32, 32,  1,  1),  # stem conv
    ("ic_s1c",    26,  26,   3, 3, 32, 32,  1,  2),  # stack1: 2 identical convs
    ("ic_s2c1",   26,  52,   3, 3, 16, 16,  2,  1),  # stack2 conv1 (stride-2)
    ("ic_s2c2",   52,  52,   3, 3, 16, 16,  1,  1),  # stack2 conv2
    ("ic_s2sc",   26,  52,   1, 1, 16, 16,  2,  1),  # stack2 shortcut (1×1 s=2)
    ("ic_s3c1",   52, 104,   3, 3,  8,  8,  2,  1),  # stack3 conv1
    ("ic_s3c2",  104, 104,   3, 3,  8,  8,  1,  1),  # stack3 conv2
    ("ic_s3sc",   52, 104,   1, 1,  8,  8,  2,  1),  # stack3 shortcut
    ("ic_cls",   104,  10,   1, 1,  1,  1,  1,  1),  # classifier FC
]

# Map layer_name → existing Timeloop stats file stem (in timeloop/outputs/)
# Format of existing stats: outputs/<stem>_<arch_tag>_stats.txt  OR
#                           outputs/<stem>_mapper.txt  (old format)
# We'll look for the cycles line in both formats.
IC_EXISTING = {
    "ic_stem": "stem_conv",         # timeloop/outputs/stem_conv_*
    "ic_s1c":  "s1_conv1",          # timeloop/outputs/s1_conv1_*  (×2!)
    "ic_cls":  "classifier_dense",  # timeloop/outputs/classifier_dense_*
}

# =============================================================================
# KWS  ─  DS-CNN  (MLPerf Tiny, num_filters=64, input 49×10)
# =============================================================================
# Topology
#   Stem   : Conv(1→64, 10×4, s=2×2)  → 25×5
#   × 4    : DW(64, 3×3, s=1) + PW(64→64, 1×1)  @ 25×5
#   Head   : AvgPool → Linear(64→12)

KWS_LAYERS = [
    # (name,        C,   K,   R,  S,  P,  Q, stride, count)
    ("kws_conv1",   1,  64,  10,  4, 25,  5,  2,  1),  # stem (stride H=W=2)
    ("kws_dw",      1,  64,   3,  3, 25,  5,  1,  4),  # × 4 DW blocks
    ("kws_pw",     64,  64,   1,  1, 25,  5,  1,  4),  # × 4 PW blocks
    ("kws_fc",     64,  12,   1,  1,  1,  1,  1,  1),  # FC head
]

KWS_EXISTING = {
    "kws_conv1": "kws_conv1",  # timeloop/outputs/kws_conv1_*
    "kws_dw":    "kws_dw1",    # timeloop/outputs/kws_dw1_*  (×4!)
    "kws_pw":    "kws_pw1",    # timeloop/outputs/kws_pw1_*  (×4!)
}

# KWS conv1 FAILS for opt_a / opt_b due to unusual kernel (10×4).
# When computing opt_a/opt_b latency for KWS, fall back to base result for this layer.
KWS_OPT_FALLBACK_TO_BASE = {"kws_conv1"}

# =============================================================================
# VWW  ─  MobileNetV1 α=0.25  (MLPerf Tiny, 96×96 RGB)
# =============================================================================
# Topology:  stem(stride=2) + 13 × DW+PW blocks + FC
#
# Block channel widths (after α=0.25 rounding, max(1,⌊c×0.25⌋)):
#   stem       : 3 → 8
#   block 0    : 8 → 16    (no stride)       @ 48×48
#   block 1    : 16 → 32   (stride=2 DW)     @ 24×24
#   block 2    : 32 → 32   (no stride)       @ 24×24
#   block 3    : 32 → 64   (stride=2 DW)     @ 12×12
#   block 4    : 64 → 64   (no stride)       @ 12×12
#   block 5    : 64 → 128  (stride=2 DW)     @  6×6
#   blocks 6-10: 128→ 128  (no stride, ×5)   @  6×6
#   block 11   : 128→ 256  (stride=2 DW)     @  3×3
#   block 12   : 256→ 256  (no stride)       @  3×3
#   head       : AvgPool → Linear(256→2)

VWW_LAYERS = [
    # (name,       C,   K,  R, S,  P,  Q, stride, count)
    ("vww_stem",   3,   8,  3, 3, 48, 48,  2,  1),   # stem conv
    ("vww_dw0",    1,   8,  3, 3, 48, 48,  1,  1),   # block0 DW
    ("vww_pw0",    8,  16,  1, 1, 48, 48,  1,  1),   # block0 PW
    ("vww_dw1",    1,  16,  3, 3, 24, 24,  2,  1),   # block1 DW (stride=2)
    ("vww_pw1",   16,  32,  1, 1, 24, 24,  1,  1),   # block1 PW
    ("vww_dw2",    1,  32,  3, 3, 24, 24,  1,  1),   # block2 DW
    ("vww_pw2",   32,  32,  1, 1, 24, 24,  1,  1),   # block2 PW
    ("vww_dw3",    1,  32,  3, 3, 12, 12,  2,  1),   # block3 DW (stride=2) ← vww_dw5 existing
    ("vww_pw3",   32,  64,  1, 1, 12, 12,  1,  1),   # block3 PW ← vww_pw5 existing
    ("vww_dw4",    1,  64,  3, 3, 12, 12,  1,  1),   # block4 DW
    ("vww_pw4",   64,  64,  1, 1, 12, 12,  1,  1),   # block4 PW
    ("vww_dw5b",   1,  64,  3, 3,  6,  6,  2,  1),   # block5 DW (stride=2)
    ("vww_pw5b",  64, 128,  1, 1,  6,  6,  1,  1),   # block5 PW
    ("vww_dw6",    1, 128,  3, 3,  6,  6,  1,  5),   # blocks 6-10 DW (×5 same shape)
    ("vww_pw6",  128, 128,  1, 1,  6,  6,  1,  5),   # blocks 6-10 PW (×5 same shape)
    ("vww_dw11",   1, 128,  3, 3,  3,  3,  2,  1),   # block11 DW (stride=2)
    ("vww_pw11", 128, 256,  1, 1,  3,  3,  1,  1),   # block11 PW
    ("vww_dw12",   1, 256,  3, 3,  3,  3,  1,  1),   # block12 DW
    ("vww_pw12", 256, 256,  1, 1,  3,  3,  1,  1),   # block12 PW
    ("vww_fc",   256,   2,  1, 1,  1,  1,  1,  1),   # FC head
]

VWW_EXISTING = {
    "vww_stem": "vww_conv1",  # timeloop/outputs/vww_conv1_*
    "vww_dw3":  "vww_dw5",   # timeloop/outputs/vww_dw5_*  (naming quirk)
    "vww_pw3":  "vww_pw5",   # timeloop/outputs/vww_pw5_*
}

# =============================================================================
# AD  ─  Deep AutoEncoder  (MLPerf Tiny, input_dim=640, bottleneck=8)
# =============================================================================
# Topology: Linear layers modelled as 1×1 convolutions (P=Q=1)
#   Encoder: 640→128→128→128→128→8
#   Decoder:           8→128→128→128→128→640
#
# Full dims list: [640, 128, 128, 128, 128, 8, 128, 128, 128, 128, 640]
# 10 linear layers:
#   640→128  ×1   (encoder input)
#   128→128  ×3   (encoder hidden)
#   128→8    ×1   (bottleneck)
#   8→128    ×1   (decoder input)
#   128→128  ×3   (decoder hidden)
#   128→640  ×1   (output reconstruction)

AD_LAYERS = [
    # (name,        C,   K,  R, S, P, Q, stride, count)
    ("ad_enc_in",  640, 128,  1, 1, 1, 1,  1,  1),  # 640→128
    ("ad_hidden",  128, 128,  1, 1, 1, 1,  1,  6),  # 128→128 × 6
    ("ad_bn_enc",  128,   8,  1, 1, 1, 1,  1,  1),  # 128→8
    ("ad_bn_dec",    8, 128,  1, 1, 1, 1,  1,  1),  # 8→128
    ("ad_out",     128, 640,  1, 1, 1, 1,  1,  1),  # 128→640
]

AD_EXISTING = {
    "ad_enc_in": "ad_dense1",  # timeloop/outputs/ad_dense1_*
}

# =============================================================================
# Combined registry
# =============================================================================
ALL_NETWORKS = {
    "ic":  (IC_LAYERS,  IC_EXISTING,  {}),
    "kws": (KWS_LAYERS, KWS_EXISTING, KWS_OPT_FALLBACK_TO_BASE),
    "vww": (VWW_LAYERS, VWW_EXISTING, {}),
    "ad":  (AD_LAYERS,  AD_EXISTING,  {}),
}
