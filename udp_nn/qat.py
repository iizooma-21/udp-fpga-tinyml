"""
qat.py — Quantization-Aware Training (QAT) utilities for UDP ResNet-8

Provides two QAT targets matching the UDP hardware evaluation modes:

  w4   — per-channel INT4 weight fake-quant only (UDP Mode 4)
  full — per-channel INT4 weights + INT4 activation requantisation at the
         same 5 block boundaries as udp_sim.py (UDP Mode 5)

All quantisation uses the straight-through estimator (STE) so gradients
flow through the rounding operation during fine-tuning.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from resnet8 import ResNet8, build_resnet8

# ── INT4 ranges (match udp_sim.py) ───────────────────────────────────────────
W_QMIN = -8
W_QMAX =  7
A_QMIN =  0
A_QMAX = 15

ACT_KEYS = ("input", "stem", "s1_out", "s2_out", "s3_out")


# =============================================================================
# STE fake-quant helpers
# =============================================================================

def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


def fake_quant_weight_ste(w: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric INT4 fake-quant with STE."""
    oc = w.shape[0]
    scales = w.view(oc, -1).abs().max(dim=1).values.clamp(min=1e-8) / W_QMAX
    sv = scales.view(oc, *([1] * (w.dim() - 1)))
    w_int = _ste_round(w / sv).clamp(W_QMIN, W_QMAX)
    return w_int * sv


def asym_params(x_min: float, x_max: float) -> Tuple[float, int]:
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)
    scale = max((x_max - x_min) / (A_QMAX - A_QMIN), 1e-8)
    zp = int(round(A_QMIN - x_min / scale))
    zp = max(A_QMIN, min(A_QMAX, zp))
    return scale, zp


def fake_quant_act_ste(x: torch.Tensor, scale: torch.Tensor,
                       zero_point: torch.Tensor) -> torch.Tensor:
    """Asymmetric unsigned INT4 fake-quant with STE."""
    x_int = _ste_round(x / scale + zero_point).clamp(A_QMIN, A_QMAX)
    x_fq = (x_int - zero_point) * scale
    return x + (x_fq - x).detach()


# =============================================================================
# QAT modules
# =============================================================================

class QATConv2d(nn.Module):
    """Conv2d whose weights are fake-quantised per channel on every forward."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quant_weight_ste(self.conv.weight)
        return F.conv2d(
            x, w, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )


class FakeQuantActSTE(nn.Module):
    """
    Asymmetric INT4 activation fake-quant with STE.
    Scale/zero-point are set by calibration and frozen during QAT.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.calibrating = True
        self._obs_min = float("inf")
        self._obs_max = -float("inf")

    def observe(self, x: torch.Tensor):
        self._obs_min = min(self._obs_min, float(x.min()))
        self._obs_max = max(self._obs_max, float(x.max()))

    def freeze_from_stats(self):
        s, z = asym_params(self._obs_min, self._obs_max)
        self.scale.fill_(s)
        self.zero_point.fill_(float(z))
        self.calibrating = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self.observe(x)
            return x
        return fake_quant_act_ste(x, self.scale, self.zero_point)


# =============================================================================
# Model builders
# =============================================================================

def _get_parent(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def wrap_conv_qat(model: nn.Module) -> nn.Module:
    """Replace every Conv2d with QATConv2d (in-place on module tree)."""
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Conv2d):
            parent, attr = _get_parent(model, name)
            setattr(parent, attr, QATConv2d(mod))
    return model


def _inner_module(qat_model: nn.Module) -> nn.Module:
    if isinstance(qat_model, QATResNet8W4):
        return qat_model.model
    if isinstance(qat_model, QATResNet8Full):
        return qat_model.m
    raise TypeError(f"Not a QAT model: {type(qat_model)}")


def export_resnet8_state_dict(qat_model: nn.Module) -> dict:
    """
    Export a plain ResNet8-compatible state dict from a QAT model.
    Maps QATConv2d keys (stem.0.conv.weight) -> stem.0.weight.
    """
    raw = _inner_module(qat_model).state_dict()
    return {k.replace(".conv.", "."): v for k, v in raw.items()}


def load_resnet8_state_dict(qat_model: nn.Module, state: dict):
    """Load a plain ResNet8 state dict into a QAT-wrapped model."""
    inner = _inner_module(qat_model)
    mapped = {}
    for k, v in state.items():
        parts = k.split(".")
        # Insert .conv. before .weight/.bias for QATConv2d modules
        if parts[-1] in ("weight", "bias"):
            conv_key = ".".join(parts[:-1]) + ".conv." + parts[-1]
            if conv_key in inner.state_dict():
                mapped[conv_key] = v
                continue
        mapped[k] = v
    inner.load_state_dict(mapped, strict=True)


class QATResNet8W4(nn.Module):
    """ResNet-8 with per-channel INT4 weight QAT on all conv layers."""

    def __init__(self, base: ResNet8):
        super().__init__()
        self.model = wrap_conv_qat(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def base_state_dict(self):
        return export_resnet8_state_dict(self)


class QATResNet8Full(nn.Module):
    """
    Full INT4 QAT pipeline — matches udp_sim.py quantisation points:
      input -> stem -> s1_out -> s2_out -> s3_out
    Weights quantised per channel; activations asymmetric INT4 at boundaries.
    """

    def __init__(self, base: ResNet8):
        super().__init__()
        self.m = wrap_conv_qat(base)
        self.act_quants = nn.ModuleDict({
            k: FakeQuantActSTE() for k in ACT_KEYS
        })

    def _fqa(self, x: torch.Tensor, key: str) -> torch.Tensor:
        return self.act_quants[key](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.m

        x = self._fqa(x, "input")

        x = m.stem(x)
        x = self._fqa(x, "stem")

        sc = x
        x = m.stack1.conv1(x)
        x = m.stack1.bn1(x)
        x = m.stack1.relu(x)
        x = m.stack1.conv2(x)
        x = m.stack1.bn2(x)
        x = F.relu(x + sc)
        x = self._fqa(x, "s1_out")

        sc = x
        x = m.stack2.conv1(x)
        x = m.stack2.bn1(x)
        x = m.stack2.relu(x)
        x = m.stack2.conv2(x)
        x = m.stack2.bn2(x)
        sc = m.stack2.shortcut(sc)
        x = F.relu(x + sc)
        x = self._fqa(x, "s2_out")

        sc = x
        x = m.stack3.conv1(x)
        x = m.stack3.bn1(x)
        x = m.stack3.relu(x)
        x = m.stack3.conv2(x)
        x = m.stack3.bn2(x)
        sc = m.stack3.shortcut(sc)
        x = F.relu(x + sc)
        x = self._fqa(x, "s3_out")

        x = m.avgpool(x)
        x = m.flatten(x)
        return m.fc(x)

    def base_state_dict(self):
        return export_resnet8_state_dict(self)

    def activation_stats(self) -> Dict[str, Tuple[float, float]]:
        return {
            k: (float(q.scale.item()), int(q.zero_point.item()))
            for k, q in self.act_quants.items()
        }


def build_qat_model(mode: str, conv_filters: int = 26,
                    base: ResNet8 | None = None) -> nn.Module:
    base = base or build_resnet8(conv_filters=conv_filters)
    if mode == "w4":
        return QATResNet8W4(base)
    if mode == "full":
        return QATResNet8Full(base)
    raise ValueError(f"Unknown QAT mode: {mode}")


def load_fp32_weights(base: ResNet8, weights_path: str,
                      device: torch.device) -> ResNet8:
    """Load FP32 checkpoint into a plain ResNet8 (before QAT wrapping)."""
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    base.load_state_dict(state, strict=True)
    return base


def build_qat_from_checkpoint(mode: str, weights_path: str,
                              device: torch.device,
                              conv_filters: int = 26) -> nn.Module:
    """Build a QAT model with FP32 weights loaded before conv wrapping."""
    base = load_fp32_weights(build_resnet8(conv_filters), weights_path, device)
    return build_qat_model(mode, conv_filters, base).to(device)


# =============================================================================
# Activation calibration (full mode)
# =============================================================================

class _Stat:
    def __init__(self):
        self.mn = 1e9
        self._buf = []

    def __call__(self, _m, _inp, out):
        v = out.detach()
        self.mn = min(self.mn, float(v.min()))
        self._buf.append(v.cpu().float())

    def percentile_max(self, pct: float = 99.9, max_samples: int = 500_000) -> float:
        if not self._buf:
            return 0.0
        t = torch.cat([b.flatten() for b in self._buf])
        if t.numel() > max_samples:
            idx = torch.randperm(t.numel())[:max_samples]
            t = t[idx]
        return float(torch.quantile(t, pct / 100.0))


@torch.no_grad()
def calibrate_full_acts(model: QATResNet8Full, loader: DataLoader,
                        device: torch.device, num_batches: int = 16):
    """
    Calibrate activation scales on a plain FP32 ResNet8 (no weight quant noise),
    then copy scales into the QAT FakeQuantActSTE modules.
    """
    fp32 = build_resnet8(conv_filters=model.m.fc.in_features // 4).to(device)
    fp32.load_state_dict(export_resnet8_state_dict(model))
    fp32.eval()

    hooks, handles = {}, []
    hook_map = {
        "stem":   fp32.stem[2],
        "s1_out": fp32.stack1,
        "s2_out": fp32.stack2,
        "s3_out": fp32.stack3,
    }
    for name, mod in hook_map.items():
        h = _Stat()
        hooks[name] = h
        handles.append(mod.register_forward_hook(h))

    img_buf = []
    for i, (imgs, _) in enumerate(loader):
        if i >= num_batches:
            break
        imgs = imgs.to(device)
        img_buf.append(imgs.cpu().float())
        fp32(imgs)

    for h in handles:
        h.remove()

    stats: Dict[str, Tuple[float, int]] = {}
    for k, h in hooks.items():
        mx = h.percentile_max(99.9)
        stats[k] = asym_params(h.mn, mx)

    img_t = torch.cat([b.flatten() for b in img_buf])
    img_mn = float(img_t.min())
    if img_t.numel() > 500_000:
        idx = torch.randperm(img_t.numel())[:500_000]
        img_t = img_t[idx]
    img_mx = float(torch.quantile(img_t, 0.999))
    stats["input"] = asym_params(img_mn, img_mx)

    for k, (s, z) in stats.items():
        q = model.act_quants[k]
        q.scale.fill_(s)
        q.zero_point.fill_(float(z))
        q.calibrating = False

    return stats


@torch.no_grad()
def evaluate_qat(model: nn.Module, loader: DataLoader,
                 device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu()
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total
