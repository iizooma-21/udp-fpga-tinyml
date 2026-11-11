"""
nn/udp_linear.py
================
UDPLinear: a PyTorch nn.Module that replaces nn.Linear with int4
UDP-simulated arithmetic.

HOW IT WORKS
------------
Standard nn.Linear computes  y = x @ W.T + b  in float32.

UDPLinear does the same computation in three stages:

  1. QUANTIZE  -- map float32 weights and activations to int4
                  (symmetric per-tensor, scale = max|x| / 7)

  2. UDP SIM   -- replace every float32 multiply with the full UDP
                  pipeline: clip_map + direct_map -> pack -> DSP sim
                  -> unpack -> postprocess (equation 4 of the paper)
                  Runs P_w=3 output neurons x P_a=2 batch elements
                  per DSP call -> 6 MACs per call -> 6x fewer calls
                  than a scalar loop over the same B x F x N products

  3. DEQUANTIZE -- multiply integer accumulator by (w_scale x a_scale)
                   and add the float32 bias

UDP PACKING SCHEME FOR LINEAR LAYERS
--------------------------------------
For y[b][f] = sum_n  W[f][n] * x[b][n]:

  Weight W[f][n] is shared across all batch elements.
  Packing TWO batch elements (b0, b1) as "activations" (P_a=2) and
  THREE output neurons (f0, f1, f2) as "weights" (P_w=3) gives:

    w_s = [W[f0][n], W[f1][n], W[f2][n]]     <- one column of W, 3 rows
    a_s = [x[b0][n], x[b1][n]]               <- same input position, 2 samples

  One DSP call returns all 6 useful cross-products:
    results[0][0] = W[f0][n]*x[b0][n]  -> y[b0][f0]
    results[0][1] = W[f1][n]*x[b0][n]  -> y[b0][f1]
    results[0][2] = W[f2][n]*x[b0][n]  -> y[b0][f2]
    results[1][0] = W[f0][n]*x[b1][n]  -> y[b1][f0]
    results[1][1] = W[f1][n]*x[b1][n]  -> y[b1][f1]
    results[1][2] = W[f2][n]*x[b1][n]  -> y[b1][f2]

  No cross-term waste -- all six products accumulate into different outputs.

  Total UDP calls = ceil(B / P_a) x N x ceil(F / P_w)

  For a batch of B=2 with N=12 inputs and F=6 outputs:
    calls = 1 x 12 x 2 = 24   vs   B x F x N = 144 scalar MACs -> 6x fewer
"""

import sys
import os
import math

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.mapping        import clip_map, direct_map
from core.packing        import pack_weights, pack_activations
from core.preprocessing  import compute_b_pre
from core.postprocessing import compute_results
from core.dsp_sim        import simulate
from nn.quantize         import quantize_to_int4


# Bitwidths and packing factors used by this module
W_w = 4   # weight bitwidth  (int4)
W_a = 4   # activation bitwidth (int4)
P_w = 3   # output neurons packed per UDP call
P_a = 2   # batch elements packed per UDP call


# ---------------------------------------------------------------------------
# Low-level UDP call (mirrors test_convolution._udp_call)
# ---------------------------------------------------------------------------

def _udp_call(w_s_list: list, a_s_list: list) -> list:
    """
    Execute one full UDP pipeline call.

    Runs: clip_map / direct_map -> pack -> b_pre -> DSP sim -> compute_results

    Parameters
    ----------
    w_s_list : list of int  P_w signed int4 weight values  (length must be P_w)
    a_s_list : list of int  P_a signed int4 activation values (length must be P_a)

    Returns
    -------
    list of list of int  results[j][i] = w_s_list[i] * a_s_list[j]
    """
    Pw    = len(w_s_list)
    w_u   = [clip_map(w, W_w)   for w in w_s_list]
    a_u   = [direct_map(a, W_a) for a in a_s_list]
    w_p   = pack_weights(w_u, W_w, W_a)
    a_p   = pack_activations(a_u, P_w=Pw, W_w=W_w, W_a=W_a)
    b_pre = compute_b_pre(w_s_list, a_s_list, W_w=W_w, W_a=W_a)
    r     = simulate(w_p, a_p, b_pre, W_w=W_w, W_a=W_a,
                     P_w=Pw, P_a=len(a_s_list))
    return compute_results(r, w_s_list, a_s_list, W_w, W_a)


# ---------------------------------------------------------------------------
# UDPLinear module
# ---------------------------------------------------------------------------

class UDPLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that simulates int4 UDP arithmetic.

    The forward pass:
      1. Quantizes self.weight and x to int4 (symmetric per-tensor)
      2. Runs every multiply through the full UDP hardware pipeline
         (P_w=3 output neurons x P_a=2 batch elements per DSP call)
      3. Dequantizes the integer accumulator back to float32
      4. Adds the float32 bias (if present)

    The output is float32 and fully compatible with PyTorch activations,
    loss functions, and nn.Sequential stacking.

    Parameters
    ----------
    in_features  : int
    out_features : int
    bias         : bool  default True -- bias stored and applied in float32

    Attributes
    ----------
    dsp_call_count : int
        Number of UDP (DSP) calls made during the most recent forward pass.
        Always equals  ceil(B/P_a) x in_features x ceil(out_features/P_w).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

        self.dsp_call_count: int = 0

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        UDP-simulated int4 forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (batch_size, in_features) or (in_features,)

        Returns
        -------
        torch.Tensor  float32, shape (batch_size, out_features) or (out_features,)
        """
        single_sample = x.dim() == 1
        if single_sample:
            x = x.unsqueeze(0)

        B, N = x.shape
        F    = self.out_features

        # ---- 1. Quantize ----
        W_int, w_scale = quantize_to_int4(self.weight.data)   # (F, N)  int32
        x_int, a_scale = quantize_to_int4(x.detach())         # (B, N)  int32

        # ---- 2. UDP simulation -- integer accumulator ----
        y_int  = [[0] * F for _ in range(B)]
        calls  = 0

        for b_start in range(0, B, P_a):
            b_batch = [b_start + j for j in range(P_a) if b_start + j < B]

            for n in range(N):
                for f_start in range(0, F, P_w):
                    f_block = [f_start + i for i in range(P_w) if f_start + i < F]

                    # Weights: one value per output neuron at this input column
                    w_s = [int(W_int[fi, n].item()) for fi in f_block]
                    # Activations: one per batch element in this pair
                    a_s = [int(x_int[bj, n].item()) for bj in b_batch]

                    # Pad to full P_w / P_a for the DSP call
                    w_s_pad = w_s + [0] * (P_w - len(w_s))
                    a_s_pad = a_s + [0] * (P_a - len(a_s))

                    results = _udp_call(w_s_pad, a_s_pad)
                    calls  += 1

                    # Accumulate only the valid (non-padded) products
                    for j, bj in enumerate(b_batch):
                        for i, fi in enumerate(f_block):
                            y_int[bj][fi] += results[j][i]

        self.dsp_call_count = calls

        # ---- 3. Dequantize and add bias ----
        scale = w_scale * a_scale
        y = torch.tensor(y_int, dtype=torch.float32) * scale

        if self.bias is not None:
            y = y + self.bias

        if single_sample:
            y = y.squeeze(0)

        return y

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"int{W_w}xint{W_a}, P_w={P_w}, P_a={P_a}")
