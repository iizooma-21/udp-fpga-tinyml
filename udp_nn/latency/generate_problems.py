#!/usr/bin/env python3
"""
generate_problems.py
====================
Reads layer_defs.py and writes a Timeloop problem YAML for every layer
that does NOT already have an existing result.

Problem files are written to  latency/problems/<name>.yaml
"""

import os
import sys

# Make sure we can import layer_defs from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from layer_defs import ALL_NETWORKS

PROBLEMS_DIR = os.path.join(os.path.dirname(__file__), "problems")
os.makedirs(PROBLEMS_DIR, exist_ok=True)

TEMPLATE = """\
# Auto-generated problem YAML for {name}
# {desc}
problem:
  R: {R}
  S: {S}
  P: {P}
  Q: {Q}
  C: {C}
  K: {K}
  N: 1
  Wstride: {stride}
  Hstride: {stride}
  Wdilation: 1
  Hdilation: 1
"""


def write_problem(name, C, K, R, S, P, Q, stride):
    macs = C * K * R * S * P * Q
    desc = (f"C={C} K={K} R={R} S={S} P={P} Q={Q} stride={stride}  "
            f"MACs={macs:,}")
    text = TEMPLATE.format(
        name=name, desc=desc,
        R=R, S=S, P=P, Q=Q, C=C, K=K, stride=stride,
    )
    out = os.path.join(PROBLEMS_DIR, f"{name}.yaml")
    with open(out, "w") as f:
        f.write(text)
    return out


def main():
    generated = []
    for net, (layers, existing, _) in ALL_NETWORKS.items():
        for entry in layers:
            name, C, K, R, S, P, Q, stride, count = entry
            if name in existing:
                print(f"  skip {name:20s} (existing: {existing[name]})")
                continue
            out = write_problem(name, C, K, R, S, P, Q, stride)
            macs = C * K * R * S * P * Q * count
            print(f"  gen  {name:20s}  -> {os.path.basename(out)}  "
                  f"(count x{count}, MACs={macs:,})")
            generated.append(name)

    print(f"\nGenerated {len(generated)} problem files in {PROBLEMS_DIR}/")


if __name__ == "__main__":
    main()
