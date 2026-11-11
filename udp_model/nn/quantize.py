"""
nn/quantize.py
==============
Symmetric per-tensor int4 quantization helpers.

The UDP pipeline (clip_map + direct_map) works on signed integers in the
range [-2^(W_w-1), 2^(W_w-1)-1].  For W_w = W_a = 4 (int4) that is [-8, 7].

We use [-7, 7] as the quantized range (scale = max|x| / 7) rather than the
full [-8, 7], which keeps the values one step inside the hardware limits and
avoids the edge case where -8 * -8 = 64 can overflow a 7-bit slot.
"""

import torch


def quantize_to_int4(x: torch.Tensor):
    """
    Symmetric per-tensor quantization to int4 range [-7, 7].

    Parameters
    ----------
    x : torch.Tensor  float32 input (any shape)

    Returns
    -------
    x_int : torch.Tensor  int32 tensor with the same shape, values in [-7, 7]
    scale  : float         x_float ≈ x_int * scale  (reconstruction scale)

    Example
    -------
    >>> t = torch.tensor([0.5, -1.0, 0.25])
    >>> q, s = quantize_to_int4(t)
    >>> q   # tensor([ 4, -7,  2], dtype=torch.int32)
    >>> s   # 0.142857...
    """
    max_val = x.abs().max().item()
    if max_val == 0.0:
        return torch.zeros_like(x, dtype=torch.int32), 1.0
    scale = max_val / 7.0
    x_int = x.div(scale).round().clamp(-7, 7).to(torch.int32)
    return x_int, scale


def dequantize(x_int: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Reconstruct a float32 approximation from a quantized integer tensor.

    Parameters
    ----------
    x_int  : torch.Tensor  integer tensor (any dtype)
    scale  : float         quantization scale returned by quantize_to_int4

    Returns
    -------
    torch.Tensor  float32 approximation: x_int * scale
    """
    return x_int.float() * scale
