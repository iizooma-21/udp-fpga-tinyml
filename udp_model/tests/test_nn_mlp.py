"""
tests/test_nn_mlp.py
====================
Tests and demo for UDP acceleration on a 2-layer MLP (fully-connected NN).

WHAT THE TEST SHOWS
-------------------
A linear layer computes:

    y[b][f] = sum_{n=0}^{N-1}  W[f][n] * x[b][n]

for each batch element b and output neuron f.

  Test A  --  standard loop: one scalar int4 MAC per iteration
              Total iterations = B * F * N

  Test B  --  UDP-accelerated loop: 6 MACs per DSP call
              Key idea: weight W[f][n] is SHARED across batch elements.
              Pack P_w=3 output neurons and P_a=2 batch elements together:

                w_s = [W[f0][n], W[f1][n], W[f2][n]]   <- 3 output neurons
                a_s = [x[b0][n], x[b1][n]]              <- 2 batch elements
                results = UDP_call(w_s, a_s)            <- 1 DSP cycle, 6 MACs

              All 6 products are USEFUL:
                results[0][0] = W[f0][n]*x[b0][n]  -> y[b0][f0]
                results[0][1] = W[f1][n]*x[b0][n]  -> y[b0][f1]
                results[0][2] = W[f2][n]*x[b0][n]  -> y[b0][f2]
                results[1][0] = W[f0][n]*x[b1][n]  -> y[b1][f0]
                results[1][1] = W[f1][n]*x[b1][n]  -> y[b1][f1]
                results[1][2] = W[f2][n]*x[b1][n]  -> y[b1][f2]

              Total calls = ceil(B/P_a) * N * ceil(F/P_w)
                          =      1      * 12 *     2      = 24   (for main config)

              Total scalar MACs = B * N * F = 2 * 12 * 6 = 144
              Loop reduction    = 144 / 24 = 6x  (= P_w * P_a)

KEY DIFFERENCE FROM CONVOLUTION TEST
--------------------------------------
In the convolution test, the SAME filter weights apply at different input
positions (weight sharing in space).  In a linear layer, W[f][n] is unique
to each (f, n) pair -- there is no spatial weight sharing.

The P_a=2 parallelism is therefore achieved across BATCH ELEMENTS: the
same weight column W[:,n] IS shared across all samples in a mini-batch,
which is exactly how neural networks operate in practice.

SETUP
-----
  W_w = W_a = 4   int4 weights and activations
  IN  = 12        input features (= N)
  HID = 6         hidden features (= F for Layer 1)
  OUT = 3         output features (= F for Layer 2)
  B   = 2         batch size (= P_a -> gives exact 6x loop reduction)
  P_w = 3         output neurons per UDP call
  P_a = 2         batch elements per UDP call

Run directly to see the comparison demo:
    python tests/test_nn_mlp.py
"""

import sys
import os
import math
import random
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.mapping        import clip_map, direct_map
from core.packing        import pack_weights, pack_activations
from core.preprocessing  import compute_b_pre
from core.postprocessing import compute_results
from core.dsp_sim        import simulate
from nn.quantize         import quantize_to_int4
from nn.udp_linear       import UDPLinear


# ============================================================================
# Configuration
# ============================================================================

W_w = W_a = 4    # int4
IN  = 12         # input features
HID = 6          # hidden features (Layer 1 output)
OUT = 3          # output features (Layer 2 output)
B   = 2          # batch size  (= P_a for exact 6x reduction)
P_w = 3          # output neurons per UDP call
P_a = 2          # batch elements per UDP call


# ============================================================================
# Shared UDP helper
# ============================================================================

def _udp_call(w_s_list: list, a_s_list: list) -> list:
    """Run one UDP pipeline call; return results[j][i] = w_s_list[i]*a_s_list[j]."""
    Pw    = len(w_s_list)
    w_u   = [clip_map(w, W_w)   for w in w_s_list]
    a_u   = [direct_map(a, W_a) for a in a_s_list]
    w_p   = pack_weights(w_u, W_w, W_a)
    a_p   = pack_activations(a_u, P_w=Pw, W_w=W_w, W_a=W_a)
    b_pre = compute_b_pre(w_s_list, a_s_list, W_w=W_w, W_a=W_a)
    r     = simulate(w_p, a_p, b_pre, W_w=W_w, W_a=W_a, P_w=Pw, P_a=len(a_s_list))
    return compute_results(r, w_s_list, a_s_list, W_w, W_a)


# ============================================================================
# TEST A  --  Standard int4 linear layer (scalar, one MAC per iteration)
# ============================================================================

def linear_standard(W_int, x_int):
    """
    Naive int4 linear layer.

    y[b][f] = sum_{n}  W[f][n] * x[b][n]

    Parameters
    ----------
    W_int : list[list[int]]  shape [F][N]  int4 weight matrix
    x_int : list[list[int]]  shape [B][N]  int4 input batch

    Returns
    -------
    y         : list[list[int]]  shape [B][F]
    mac_count : int   total scalar MAC iterations (= B * F * N)
    """
    F  = len(W_int)
    N  = len(W_int[0])
    Bb = len(x_int)
    y  = [[0] * F for _ in range(Bb)]
    mac_count = 0
    for b in range(Bb):
        for f in range(F):
            for n in range(N):
                y[b][f] += W_int[f][n] * x_int[b][n]
                mac_count += 1
    return y, mac_count


# ============================================================================
# TEST B  --  UDP-accelerated linear layer (loop divided by 6)
# ============================================================================

def linear_udp(W_int, x_int):
    """
    UDP-accelerated int4 linear layer.

    For each input column n, batch pair (b0, b1), and output block (f0,f1,f2):
      w_s = [W[f0][n], W[f1][n], W[f2][n]]   (P_w=3 output neurons)
      a_s = [x[b0][n], x[b1][n]]             (P_a=2 batch elements)
      results = UDP_call(w_s, a_s)            (1 DSP cycle = 6 MACs)

    If B is odd the last batch element is padded with a_s=0; those products
    are computed but not accumulated (zero contribution).

    Parameters
    ----------
    W_int : list[list[int]]  shape [F][N]
    x_int : list[list[int]]  shape [B][N]

    Returns
    -------
    y          : list[list[int]]  shape [B][F]  identical to linear_standard
    call_count : int   number of UDP (DSP) calls
    """
    F  = len(W_int)
    N  = len(W_int[0])
    Bb = len(x_int)
    y  = [[0] * F for _ in range(Bb)]
    call_count = 0

    for b_start in range(0, Bb, P_a):
        b_batch = [b_start + j for j in range(P_a) if b_start + j < Bb]

        for n in range(N):
            for f_start in range(0, F, P_w):
                f_block = [f_start + i for i in range(P_w) if f_start + i < F]

                w_s = [W_int[fi][n] for fi in f_block]
                a_s = [x_int[bj][n] for bj in b_batch]

                # Pad to full P_w / P_a for the DSP call
                w_s_pad = w_s + [0] * (P_w - len(w_s))
                a_s_pad = a_s + [0] * (P_a - len(a_s))

                results = _udp_call(w_s_pad, a_s_pad)
                call_count += 1

                # Accumulate only the valid (non-padded) products
                for j, bj in enumerate(b_batch):
                    for i, fi in enumerate(f_block):
                        y[bj][fi] += results[j][i]

    return y, call_count


# ============================================================================
# Helper for unit tests
# ============================================================================

def _random_int4(seed, F, N, Bb=B):
    """Return random int4 weight matrix [F][N] and input batch [Bb][N]."""
    random.seed(seed)
    lo, hi = -(2 ** (W_w - 1)) + 1, 2 ** (W_w - 1) - 1   # [-7, 7]
    W = [[random.randint(lo, hi) for _ in range(N)] for _ in range(F)]
    x = [[random.randint(lo, hi) for _ in range(N)] for _ in range(Bb)]
    return W, x


# ============================================================================
# Unit tests  --  TestLinearStandard
# ============================================================================

class TestLinearStandard(unittest.TestCase):

    def test_output_shape(self):
        W, x = _random_int4(0, HID, IN)
        y, _ = linear_standard(W, x)
        self.assertEqual(len(y), B)
        self.assertEqual(len(y[0]), HID)

    def test_mac_count(self):
        W, x = _random_int4(0, HID, IN)
        _, count = linear_standard(W, x)
        self.assertEqual(count, B * HID * IN)

    def test_zero_input_gives_zero(self):
        W, _ = _random_int4(0, HID, IN)
        x = [[0] * IN for _ in range(B)]
        y, _ = linear_standard(W, x)
        for row in y:
            self.assertEqual(row, [0] * HID)

    def test_known_single_element(self):
        """F=1, N=1, B=1: y[0][0] = w * x."""
        y, count = linear_standard([[3]], [[-2]])
        self.assertEqual(y, [[-6]])
        self.assertEqual(count, 1)


# ============================================================================
# Unit tests  --  TestLinearUDP
# ============================================================================

class TestLinearUDP(unittest.TestCase):

    def test_matches_standard_output(self):
        """UDP and standard loops produce bit-exact int4 results for 10 seeds."""
        for seed in range(10):
            W, x = _random_int4(seed, HID, IN)
            y_a, _ = linear_standard(W, x)
            y_b, _ = linear_udp(W, x)
            self.assertEqual(y_a, y_b, f"Mismatch at seed={seed}")

    def test_loop_divided_by_6_even_batch(self):
        """B=2, F divisible by P_w, N arbitrary: exact 6x reduction."""
        W, x = _random_int4(0, HID, IN, Bb=2)    # B=2, HID=6, IN=12
        _, mac_count  = linear_standard(W, x)
        _, call_count = linear_udp(W, x)
        expected_calls = math.ceil(B / P_a) * IN * math.ceil(HID / P_w)
        self.assertEqual(call_count, expected_calls)
        self.assertEqual(call_count * P_w * P_a, mac_count)

    def test_call_count_formula(self):
        """call_count = ceil(Bb/P_a) * N * ceil(F/P_w) for varied shapes."""
        for (Fb, Nb, Bb) in [(3, 12, 2), (5, 7, 3), (6, 8, 4), (4, 9, 1)]:
            W, x = _random_int4(0, Fb, Nb, Bb=Bb)
            _, call_count = linear_udp(W, x)
            expected = math.ceil(Bb / P_a) * Nb * math.ceil(Fb / P_w)
            self.assertEqual(call_count, expected,
                             f"F={Fb}, N={Nb}, B={Bb}: got {call_count}, expect {expected}")

    def test_single_sample_works(self):
        """B=1 (odd batch) produces correct output via zero-padding."""
        W, x = _random_int4(3, HID, IN, Bb=1)
        y_a, _ = linear_standard(W, x)
        y_b, _ = linear_udp(W, x)
        self.assertEqual(y_a, y_b)

    def test_zero_input(self):
        W, _ = _random_int4(0, HID, IN)
        x = [[0] * IN for _ in range(B)]
        y_b, _ = linear_udp(W, x)
        for row in y_b:
            self.assertEqual(row, [0] * HID)

    def test_extreme_values(self):
        """Boundary int4 values (±7) across the full linear layer."""
        W = [[-7 if (f + n) % 2 == 0 else 7 for n in range(IN)]
             for f in range(HID)]
        x = [[-7 if n % 2 == 0 else 7 for n in range(IN)]
             for _ in range(B)]
        y_a, _ = linear_standard(W, x)
        y_b, _ = linear_udp(W, x)
        self.assertEqual(y_a, y_b)

    def test_all_negative_weights(self):
        """All-negative weight matrix still gives exact match."""
        W = [[-1] * IN for _ in range(HID)]
        _, x = _random_int4(5, 1, IN)
        x = x * B   # duplicate to get B samples
        y_a, _ = linear_standard(W, x)
        y_b, _ = linear_udp(W, x)
        self.assertEqual(y_a, y_b)

    def test_odd_batch_still_correct(self):
        """B=3 (odd) produces correct output; last batch padded then discarded."""
        W, x = _random_int4(7, HID, IN, Bb=3)
        y_a, _ = linear_standard(W, x)
        y_b, _ = linear_udp(W, x)
        self.assertEqual(y_a, y_b)


# ============================================================================
# Unit tests  --  TestUDPLinearModule
# ============================================================================

class TestUDPLinearModule(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.layer = UDPLinear(IN, HID)

    def test_output_shape_batch(self):
        y = self.layer(torch.randn(B, IN))
        self.assertEqual(list(y.shape), [B, HID])

    def test_output_shape_single(self):
        y = self.layer(torch.randn(IN))
        self.assertEqual(list(y.shape), [HID])

    def test_output_is_float32(self):
        y = self.layer(torch.randn(B, IN))
        self.assertEqual(y.dtype, torch.float32)

    def test_dsp_call_count(self):
        """call_count == ceil(B/P_a) * IN * ceil(HID/P_w)."""
        self.layer(torch.randn(B, IN))
        expected = math.ceil(B / P_a) * IN * math.ceil(HID / P_w)
        self.assertEqual(self.layer.dsp_call_count, expected)

    def test_matches_reference_int4(self):
        """
        UDPLinear (minus bias) must exactly match the scalar int4 reference
        when both use the same quantized weights and activations.
        """
        x = torch.randn(B, IN)

        W_int, w_scale = quantize_to_int4(self.layer.weight.data)
        x_int, a_scale = quantize_to_int4(x)

        W_list = [[int(W_int[f, n].item()) for n in range(IN)] for f in range(HID)]
        x_list = [[int(x_int[b, n].item()) for n in range(IN)] for b in range(B)]

        y_ref, _ = linear_standard(W_list, x_list)

        y_udp = self.layer(x)
        if self.layer.bias is not None:
            y_udp = y_udp - self.layer.bias

        scale         = w_scale * a_scale
        y_ref_float   = torch.tensor(y_ref, dtype=torch.float32) * scale

        self.assertTrue(
            torch.allclose(y_udp, y_ref_float, atol=1e-4),
            "UDPLinear output (minus bias) does not match scalar int4 reference"
        )

    def test_no_bias_module(self):
        """UDPLinear with bias=False returns correct shape and has no bias."""
        layer_no_bias = UDPLinear(IN, HID, bias=False)
        y = layer_no_bias(torch.randn(B, IN))
        self.assertEqual(list(y.shape), [B, HID])
        self.assertIsNone(layer_no_bias.bias)

    def test_stacked_two_layers(self):
        """Two UDPLinear layers stack correctly without errors."""
        torch.manual_seed(0)
        l1 = UDPLinear(IN, HID)
        l2 = UDPLinear(HID, OUT)
        x  = torch.randn(B, IN)
        y  = l2(torch.relu(l1(x)))
        self.assertEqual(list(y.shape), [B, OUT])

    def test_sequential_compatible(self):
        """UDPLinear works inside nn.Sequential."""
        model = nn.Sequential(UDPLinear(IN, HID), nn.ReLU(), UDPLinear(HID, OUT))
        y = model(torch.randn(B, IN))
        self.assertEqual(list(y.shape), [B, OUT])


# ============================================================================
# Demo (run directly:  python tests/test_nn_mlp.py)
# ============================================================================

def _run_demo():
    SEP  = "=" * 72
    SEP2 = "-" * 72

    random.seed(42)

    lo, hi = -(2 ** (W_w - 1)) + 1, 2 ** (W_w - 1) - 1   # [-7, 7]

    W1 = [[random.randint(lo, hi) for _ in range(IN)]  for _ in range(HID)]
    W2 = [[random.randint(lo, hi) for _ in range(HID)] for _ in range(OUT)]
    x  = [[random.randint(lo, hi) for _ in range(IN)]  for _ in range(B)]

    # -----------------------------------------------------------------------
    print()
    print(SEP)
    print("  UDP  --  MLP Linear Layer (loop divided by 6)")
    print(SEP)
    print(f"""
  Config
  ------
  Bitwidths      :  int{W_w} weights  x  int{W_a} activations
  Network        :  IN={IN}  ->  HID={HID}  ->  OUT={OUT}
  Batch size (B) :  {B}  (P_a={P_a} batch elements per UDP call)
  Packing        :  P_w={P_w} output neurons  x  P_a={P_a} batch elements = {P_w*P_a} MACs/call

  Linear formula:
    y[b][f] = sum_{{n=0}}^{{{IN-1}}}  W[f][n] * x[b][n]
    for b = 0..{B-1},  f = 0..{{out_features-1}}
""")

    print("  Input batch  x[b][n]:")
    print("  b=0:  " + "  ".join(f"{v:3d}" for v in x[0]))
    print("  b=1:  " + "  ".join(f"{v:3d}" for v in x[1]))

    # -----------------------------------------------------------------------
    # LAYER 1: IN=12 -> HID=6
    # -----------------------------------------------------------------------
    print()
    print(SEP2)
    print(f"  LAYER 1   IN={IN}  ->  HID={HID}")
    print(SEP2)

    y1_A, mac1  = linear_standard(W1, x)
    y1_B, calls1 = linear_udp(W1, x)

    print(f"""
  TEST A  --  Standard loop
    for b in range({B}):
      for f in range({HID}):
        for n in range({IN}):
          y[b][f] += W[f][n] * x[b][n]   <- 1 scalar MAC

  Total iterations: B x F x N = {B} x {HID} x {IN} = {mac1}

  TEST B  --  UDP loop
    for b in range(0, {B}, {P_a}):           <- batch pairs, step={P_a}
      for n in range({IN}):                  <- input columns
        for f in range(0, {HID}, {P_w}):     <- output blocks, step={P_w}
          w_s = [W[f][n], W[f+1][n], W[f+2][n]]  (P_w={P_w})
          a_s = [x[b][n], x[b+1][n]]              (P_a={P_a})
          results = UDP_call(w_s, a_s)            <- 1 DSP cycle = {P_w*P_a} MACs
          accumulate all 6 products into y

  Total UDP calls : {calls1:3d}  =  ceil(B/P_a) x N x ceil(F/P_w)
                               =  {math.ceil(B/P_a)} x {IN} x {math.ceil(HID/P_w)}
  MACs per call   :   {P_w*P_a}  =  P_w x P_a = {P_w} x {P_a}
  Reduction       : {mac1//calls1:3d}x   (= P_w x P_a)
""")

    # Show one concrete call
    w_ex = [W1[fi][0] for fi in [0, 1, 2]]
    a_ex = [x[0][0], x[1][0]]
    res_ex = _udp_call(w_ex, a_ex)
    print(f"  Example UDP call  (n=0, output block f=0..2, batch b=0,1):")
    print(f"    w_s = {w_ex}   <- W[f=0..2][n=0]")
    print(f"    a_s = {a_ex}    <- x[b=0,1][n=0]")
    print(f"    6 products:")
    for j, bj in enumerate([0, 1]):
        for i, fi in enumerate([0, 1, 2]):
            print(f"      results[{j}][{i}] = W[{fi}][0]*x[{bj}][0] "
                  f"= {w_ex[i]:3d} * {a_ex[j]:3d} = {res_ex[j][i]:5d}  -> y[{bj}][{fi}]")

    print()
    print(f"  Test A output  y1_A[b][f]:")
    print("  b=0:  " + "  ".join(f"{y1_A[0][f]:6d}" for f in range(HID)))
    print("  b=1:  " + "  ".join(f"{y1_A[1][f]:6d}" for f in range(HID)))
    print()
    print(f"  Test B output  y1_B[b][f]:")
    print("  b=0:  " + "  ".join(f"{y1_B[0][f]:6d}" for f in range(HID)))
    print("  b=1:  " + "  ".join(f"{y1_B[1][f]:6d}" for f in range(HID)))

    # -----------------------------------------------------------------------
    # LAYER 2: HID=6 -> OUT=3
    # -----------------------------------------------------------------------
    print()
    print(SEP2)
    print(f"  LAYER 2   HID={HID}  ->  OUT={OUT}")
    print(SEP2)

    # Re-quantize intermediate activations back to int4 before Layer 2
    y1_tensor = torch.tensor(y1_A, dtype=torch.float32)
    y1_int, _ = quantize_to_int4(y1_tensor)
    y1_int_list = [[int(y1_int[b, f].item()) for f in range(HID)] for b in range(B)]

    y2_A, mac2   = linear_standard(W2, y1_int_list)
    y2_B, calls2 = linear_udp(W2, y1_int_list)

    print(f"""
  (Intermediate activations re-quantized to int4 before Layer 2.)

  TEST A: B x OUT x HID = {B} x {OUT} x {HID} = {mac2} iterations
  TEST B: ceil(B/P_a) x HID x ceil(OUT/P_w)
        = {math.ceil(B/P_a)} x {HID} x {math.ceil(OUT/P_w)} = {calls2} UDP calls  ({mac2//calls2}x reduction)

  Test A output  y2_A[b][f]:
  b=0:  {y2_A[0]}
  b=1:  {y2_A[1]}

  Test B output  y2_B[b][f]:
  b=0:  {y2_B[0]}
  b=1:  {y2_B[1]}
""")

    # -----------------------------------------------------------------------
    # VERIFICATION
    # -----------------------------------------------------------------------
    print(SEP2)
    print("  VERIFICATION  --  Test A vs Test B")
    print(SEP2)

    match1 = (y1_A == y1_B)
    match2 = (y2_A == y2_B)
    total_mac   = mac1 + mac2
    total_calls = calls1 + calls2

    print(f"""
  Layer 1:
    Test A iterations : {mac1:5d}
    Test B UDP calls  : {calls1:5d}   (each = {P_w*P_a} MACs)
    Reduction         : {mac1//calls1:5d}x
    Output match      : {"YES -- exact integer match" if match1 else "NO -- MISMATCH"}

  Layer 2:
    Test A iterations : {mac2:5d}
    Test B UDP calls  : {calls2:5d}   (each = {P_w*P_a} MACs)
    Reduction         : {mac2//calls2:5d}x
    Output match      : {"YES -- exact integer match" if match2 else "NO -- MISMATCH"}

  Total (both layers):
    Scalar MAC iterations : {total_mac}
    UDP DSP calls         : {total_calls}
    Overall reduction     : {total_mac // total_calls}x
""")

    if not match1:
        for b in range(B):
            for f in range(HID):
                if y1_A[b][f] != y1_B[b][f]:
                    print(f"  LAYER1 MISMATCH at y[{b}][{f}]: "
                          f"A={y1_A[b][f]}, B={y1_B[b][f]}")

    if not match2:
        for b in range(B):
            for f in range(OUT):
                if y2_A[b][f] != y2_B[b][f]:
                    print(f"  LAYER2 MISMATCH at y[{b}][{f}]: "
                          f"A={y2_A[b][f]}, B={y2_B[b][f]}")

    # -----------------------------------------------------------------------
    # PyTorch float32  vs  UDPLinear
    # -----------------------------------------------------------------------
    print(SEP2)
    print("  PYTORCH COMPARISON  --  float32 nn.Linear  vs  UDPLinear (int4)")
    print(SEP2)

    torch.manual_seed(7)
    x_float = torch.randn(B, IN)

    # float32 reference MLP
    float_l1 = nn.Linear(IN, HID)
    float_l2 = nn.Linear(HID, OUT)
    torch.manual_seed(7)   # same seed so weights start identical
    with torch.no_grad():
        y_float = float_l2(torch.relu(float_l1(x_float)))

    # UDP int4 MLP (same weight initialisation)
    torch.manual_seed(7)
    udp_l1 = UDPLinear(IN, HID)
    udp_l2 = UDPLinear(HID, OUT)
    with torch.no_grad():
        y_udp = udp_l2(torch.relu(udp_l1(x_float)))

    print(f"""
  float32 MLP output:
    b=0:  {["%.4f" % v for v in y_float[0].tolist()]}
    b=1:  {["%.4f" % v for v in y_float[1].tolist()]}

  UDP int4 MLP output:
    b=0:  {["%.4f" % v for v in y_udp[0].tolist()]}
    b=1:  {["%.4f" % v for v in y_udp[1].tolist()]}

  Max absolute difference : {(y_float - y_udp).abs().max().item():.4f}
  (Difference is due to int4 quantisation, NOT a UDP pipeline error.)

  DSP calls  Layer 1 : {udp_l1.dsp_call_count}
  DSP calls  Layer 2 : {udp_l2.dsp_call_count}
  Total DSP calls    : {udp_l1.dsp_call_count + udp_l2.dsp_call_count}
""")

    print(SEP)
    print("  Demo complete.")
    print(SEP)
    print()


if __name__ == "__main__":
    _run_demo()
