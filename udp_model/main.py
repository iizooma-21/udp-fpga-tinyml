"""
main.py  —  UDP Pipeline Demo
==============================
Interactive showcase of the Universal DSP Packing (UDP) framework from:

    "UDP: A Universal DSP Packing Framework for Low-bitwidth MAC
     Acceleration on FPGAs"  (ACM DL 3748173.3779194)

Run this file directly:
    python main.py

The demo walks through every stage of the pipeline in order:

    Stage 0  Hardware at a glance      (DSP48E2 port widths)
    Stage 1  Efficiency analysis       (optimal P_w, P_a for your bitwidths)
    Stage 2  Mapping                   (signed -> unsigned)
    Stage 3  Packing                   (many values -> one integer)
    Stage 4  Preprocessing  (b_pre)    (pack the sign-correction term)
    Stage 5  DSP simulation            (r = w_p * a_p + b_pre)
    Stage 6  Postprocessing            (recover signed products)
    Stage 7  Verification              (compare against direct Python multiply)
    Stage 8  Efficiency table          (all valid (P_w, P_a) pairs)
"""

import sys
import random

from core.mapping       import clip_map, direct_map
from core.packing       import pack_weights, pack_activations, unpack_results, compute_wp
from core.preprocessing import compute_b_pre
from core.postprocessing import compute_results
from core.dsp_sim       import simulate, DSP_SHORT, DSP_LONG, DSP_C, DSP_P
from efficiency.model   import find_optimal, find_all_valid, print_efficiency_table


# ============================================================================
# Helpers
# ============================================================================

def _sep(char="=", width=70):
    print(char * width)

def _header(title):
    _sep()
    print(f"  {title}")
    _sep()

def _sub(title):
    print(f"\n  -- {title} --")

def _bit_width(n):
    """Minimum bits needed to represent non-negative integer n."""
    return n.bit_length() if n > 0 else 1


def udp_pipeline(w_s_list, a_s_list, W_w, W_a, verbose=False):
    """Execute the complete UDP pipeline and return the recovered products."""
    P_w = len(w_s_list)

    w_u = [clip_map(w, W_w)   for w in w_s_list]
    a_u = [direct_map(a, W_a) for a in a_s_list]

    w_p   = pack_weights(w_u, W_w, W_a)
    a_p   = pack_activations(a_u, P_w=P_w, W_w=W_w, W_a=W_a)
    b_pre = compute_b_pre(w_s_list, a_s_list, W_w=W_w, W_a=W_a)

    r_packed = simulate(w_p, a_p, b_pre, W_w=W_w, W_a=W_a,
                        P_w=P_w, P_a=len(a_s_list), verbose=verbose)

    return compute_results(r_packed, w_s_list, a_s_list, W_w, W_a,
                           verbose=verbose)


# ============================================================================
# Stage 0  –  Hardware context
# ============================================================================

def stage0_hardware():
    _header("STAGE 0  |  DSP48E2 Hardware Ports")
    print(f"""
  The Xilinx DSP48E2 block has four relevant ports:

    Port   Width    UDP role
    -----  -------  ------------------------------------------
    B      {DSP_SHORT:2d}-bit   w_p   packed weights  (short port)
    A / D  {DSP_LONG:2d}-bit   a_p   packed activations (long port)
    C      {DSP_C:2d}-bit   b_pre preprocessing correction
    P      {DSP_P:2d}-bit   r = w_p * a_p + b_pre  (output)

  UDP fits MULTIPLE multiply-accumulate (MAC) operations inside a
  single DSP cycle by packing several weights and activations into
  the B and A/D operands.  The key constraint is that each packed
  operand must fit within its port width.
""")


# ============================================================================
# Stage 1  –  Efficiency analysis
# ============================================================================

def stage1_efficiency(W_w, W_a):
    _header(f"STAGE 1  |  Efficiency Analysis  (int{W_w} x int{W_a})")

    opt = find_optimal(W_w=W_w, W_a=W_a)
    if opt is None:
        print(f"\n  No valid packing exists for int{W_w} x int{W_a} "
              f"on a DSP48E2.")
        sys.exit(1)

    W_p = compute_wp(W_w, W_a)
    print(f"""
  Bitwidths      :  W_w = {W_w}  |  W_a = {W_a}  |  W_p = W_w+W_a-1 = {W_p}

  Port constraints (equations 6 & 7 from the paper):
    Eq 6:  W_pw = (P_w-1)*W_p + (W_w-1)  <=  {DSP_SHORT}   (B port)
    Eq 7:  W_pa = (P_a-1)*P_w*W_p + W_a  <=  {DSP_LONG}  (A/D port)

  Optimal config :  P_w = {opt.P_w}  |  P_a = {opt.P_a}
  MACs per DSP   :  P_w x P_a = {opt.parallelism}
  B  port usage  :  {opt.W_pw:2d} / {DSP_SHORT} bits  ({opt.short_util*100:.1f}%)
  A/D port usage :  {opt.W_pa:2d} / {DSP_LONG} bits  ({opt.long_util*100:.1f}%)
  Bottleneck     :  {opt.bottleneck} port
""")
    return opt.P_w, opt.P_a


# ============================================================================
# Stage 2  –  Mapping
# ============================================================================

def stage2_mapping(w_s_list, a_s_list, W_w, W_a):
    _header("STAGE 2  |  Mapping  (signed -> unsigned)")
    print("""
  UDP uses two mapping schemes (Section 3.1):
    Clip mapping  (weights)     : remove MSB, track sign separately
    Direct mapping (activations): reinterpret sign bit as data bit
""")

    w_u_list = [clip_map(w, W_w)   for w in w_s_list]
    a_u_list = [direct_map(a, W_a) for a in a_s_list]

    _sub("Weights  (clip_map)")
    print(f"  {'Signed':>10}  {'Unsigned':>10}  {'Negative?':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
    for ws, wu in zip(w_s_list, w_u_list):
        flag = "<-- sign folded" if ws < 0 else ""
        print(f"  {ws:>10}  {wu:>10}  {'yes' if ws < 0 else 'no':>10}  {flag}")

    _sub("Activations  (direct_map)")
    print(f"  {'Signed':>10}  {'Unsigned':>10}")
    print(f"  {'-'*10}  {'-'*10}")
    for as_, au in zip(a_s_list, a_u_list):
        print(f"  {as_:>10}  {au:>10}")

    return w_u_list, a_u_list


# ============================================================================
# Stage 3  –  Packing
# ============================================================================

def stage3_packing(w_u_list, a_u_list, P_w, W_w, W_a):
    _header("STAGE 3  |  Packing  (many values -> one integer)")

    W_p = compute_wp(W_w, W_a)

    print(f"""
  Each value occupies a W_p = {W_p}-bit slot inside the packed integer.
  Weights  : slots spaced {W_p} bits apart  (bits i*{W_p} .. (i+1)*{W_p}-1)
  Activations: slots spaced P_w*{W_p} = {P_w*W_p} bits apart
""")

    w_p = pack_weights(w_u_list, W_w, W_a)
    a_p = pack_activations(a_u_list, P_w=P_w, W_w=W_w, W_a=W_a)

    _sub(f"Packed weight  w_p  ({_bit_width(w_p)}-bit value)")
    print(f"  decimal : {w_p}")
    print(f"  binary  : {bin(w_p)}")
    for i, wu in enumerate(w_u_list):
        lo = i * W_p
        print(f"    slot[{i}] bits {lo:2d}-{lo+W_p-1:2d} : w_u[{i}] = {wu} "
              f"({bin(wu)})")

    _sub(f"Packed activation  a_p  ({_bit_width(a_p)}-bit value)")
    print(f"  decimal : {a_p}")
    print(f"  binary  : {bin(a_p)}")
    for j, au in enumerate(a_u_list):
        lo = j * P_w * W_p
        print(f"    slot[{j}] bits {lo:2d}-{lo+W_p-1:2d} : a_u[{j}] = {au} "
              f"({bin(au)})")

    return w_p, a_p


# ============================================================================
# Stage 4  –  Preprocessing (b_pre)
# ============================================================================

def stage4_preprocessing(w_s_list, a_s_list, W_w, W_a):
    _header("STAGE 4  |  Preprocessing  b_pre  (equation 3)")
    print("""
  For every negative weight w_s_i, the recovery formula (eq 1a) needs a
  correction term  -2^(W_w-1) * a_s_j  per activation group j.
  These are packed into a single integer b_pre and added via the DSP C port,
  so they come "for free" in the multiply-add pipeline.
""")

    has_neg = any(w < 0 for w in w_s_list)
    if not has_neg:
        print("  All weights are non-negative -> b_pre = 0  (no correction needed)\n")
    else:
        print("  Negative weights detected -> packing corrections:")
        for i, w in enumerate(w_s_list):
            if w < 0:
                print(f"    w_s[{i}] = {w}  (negative)")

    b_pre = compute_b_pre(w_s_list, a_s_list, W_w=W_w, W_a=W_a)
    print(f"\n  b_pre = {b_pre}  ({_bit_width(b_pre)}-bit)")
    return b_pre


# ============================================================================
# Stage 5  –  DSP simulation
# ============================================================================

def stage5_dsp(w_p, a_p, b_pre, w_s_list, a_s_list, W_w, W_a):
    _header("STAGE 5  |  DSP Simulation  r = w_p * a_p + b_pre")

    P_w = len(w_s_list)
    P_a = len(a_s_list)
    W_p = compute_wp(W_w, W_a)

    print(f"""
  w_p  = {w_p:>15d}  ({_bit_width(w_p)}-bit, B port   <= {DSP_SHORT}-bit)
  a_p  = {a_p:>15d}  ({_bit_width(a_p)}-bit, A/D port <= {DSP_LONG}-bit)
  b_pre= {b_pre:>15d}  ({_bit_width(b_pre)}-bit, C port   <= {DSP_C}-bit)
""")

    r_packed = simulate(w_p, a_p, b_pre, W_w=W_w, W_a=W_a, P_w=P_w, P_a=P_a)

    print(f"  r_packed = {r_packed}  ({_bit_width(r_packed)}-bit, P port <= {DSP_P}-bit)")

    _sub("Raw (unsigned) slot values inside r_packed")
    raw = unpack_results(r_packed, P_w=P_w, P_a=P_a, W_w=W_w, W_a=W_a)
    print(f"  {'slot (i,j)':>12}  {'raw value':>12}  {'note'}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*40}")
    for j, a_s in enumerate(a_s_list):
        for i, w_s in enumerate(w_s_list):
            w_u = clip_map(w_s, W_w)
            a_u = direct_map(a_s, W_a)
            note = f"w_u[{i}]*a_u[{j}]={w_u}*{a_u}={w_u*a_u}"
            print(f"  ({i},{j}){' ':>7}  {raw[j][i]:>12}  {note}")

    return r_packed


# ============================================================================
# Stage 6 & 7  –  Postprocessing and verification
# ============================================================================

def stage6_postprocessing(r_packed, w_s_list, a_s_list, W_w, W_a):
    _header("STAGE 6  |  Postprocessing  (equation 4)  +  Verification")
    print("""
  Apply the remaining correction terms (Terms A & B from eq 1a) and subtract
  the b_pre residual from each slot to recover the exact signed products.
""")

    results = compute_results(r_packed, w_s_list, a_s_list, W_w, W_a)

    P_w = len(w_s_list)
    P_a = len(a_s_list)

    print(f"  {'slot (i,j)':>12}  {'w_s[i]':>8}  {'a_s[j]':>8}  "
          f"{'recovered':>10}  {'direct':>10}  {'ok?':>6}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*6}")

    all_ok = True
    for j, a_s in enumerate(a_s_list):
        for i, w_s in enumerate(w_s_list):
            recovered = results[j][i]
            direct    = w_s * a_s
            ok        = recovered == direct
            all_ok    = all_ok and ok
            tick      = "PASS" if ok else "FAIL"
            print(f"  ({i},{j}){' ':>7}  {w_s:>8}  {a_s:>8}  "
                  f"{recovered:>10}  {direct:>10}  {tick:>6}")

    print()
    if all_ok:
        print(f"  All {P_w * P_a} products recovered exactly. "
              f"({P_w} weights x {P_a} activations = {P_w*P_a} MACs in 1 DSP cycle)")
    else:
        print("  ERROR: some products did not recover correctly!")

    return results


# ============================================================================
# Stage 8  –  Full efficiency table
# ============================================================================

def stage8_table(W_w, W_a):
    _header(f"STAGE 7  |  All Valid (P_w, P_a) Pairs  (int{W_w} x int{W_a})")
    print()
    print_efficiency_table(W_w=W_w, W_a=W_a)


# ============================================================================
# Stress demo: exhaustive int4 check
# ============================================================================

def stress_demo(W_w, W_a, P_w, P_a, n_trials=200):
    _header(f"STAGE 8  |  Stress Test  ({n_trials} random int{W_w}x{W_a} grids,"
            f" P_w={P_w}, P_a={P_a})")
    print(f"""
  Running {n_trials} random trials — each trial is a full UDP pipeline call
  (map -> pack -> b_pre -> DSP sim -> postprocessing) and compares the
  recovered products to direct Python multiplication.
""")
    lo = -(2 ** (W_w - 1))
    hi =  (2 ** (W_w - 1)) - 1
    lo_a = -(2 ** (W_a - 1))
    hi_a =  (2 ** (W_a - 1)) - 1

    random.seed(0)
    errors = 0
    for trial in range(n_trials):
        w_s = [random.randint(lo,   hi)   for _ in range(P_w)]
        a_s = [random.randint(lo_a, hi_a) for _ in range(P_a)]
        results = udp_pipeline(w_s, a_s, W_w, W_a)
        for j, a in enumerate(a_s):
            for i, w in enumerate(w_s):
                if results[j][i] != w * a:
                    errors += 1

    if errors == 0:
        print(f"  All {n_trials} trials passed. "
              f"({n_trials * P_w * P_a} products verified)")
    else:
        print(f"  {errors} products FAILED across {n_trials} trials!")


# ============================================================================
# Entry point
# ============================================================================

def main():
    # -----------------------------------------------------------------
    # Configuration — change these to explore other bitwidths
    # -----------------------------------------------------------------
    W_w = 4   # weight bitwidth  (e.g. 4, 8)
    W_a = 4   # activation bitwidth

    # Demo inputs for the step-by-step walkthrough
    w_s_list = [-3,  2, -1]   # 3 signed weights
    a_s_list = [ 5, -2]       # 2 signed activations
    # -----------------------------------------------------------------

    print()
    _sep("*")
    print("  UDP  —  Universal DSP Packing  |  Python Model Demo")
    _sep("*")
    print(f"""
  Paper : "UDP: A Universal DSP Packing Framework for
           Low-bitwidth MAC Acceleration on FPGAs"

  Config: int{W_w} weights  x  int{W_a} activations
  Inputs:
    weights     = {w_s_list}
    activations = {a_s_list}
""")

    # Stage 0: hardware context
    stage0_hardware()

    # Stage 1: find optimal packing factors
    P_w, P_a = stage1_efficiency(W_w, W_a)

    # Confirm demo inputs match the optimal P_w / P_a
    if len(w_s_list) != P_w or len(a_s_list) != P_a:
        print(f"  (Adjusting demo inputs to optimal P_w={P_w}, P_a={P_a})")
        random.seed(1)
        lo   = -(2 ** (W_w - 1))
        hi   =  (2 ** (W_w - 1)) - 1
        lo_a = -(2 ** (W_a - 1))
        hi_a =  (2 ** (W_a - 1)) - 1
        w_s_list = [random.randint(lo,   hi)   for _ in range(P_w)]
        a_s_list = [random.randint(lo_a, hi_a) for _ in range(P_a)]
        print(f"  weights     = {w_s_list}")
        print(f"  activations = {a_s_list}\n")

    # Stages 2–7: walk through the pipeline step by step
    w_u_list, a_u_list = stage2_mapping(w_s_list, a_s_list, W_w, W_a)
    w_p, a_p           = stage3_packing(w_u_list, a_u_list, P_w, W_w, W_a)
    b_pre              = stage4_preprocessing(w_s_list, a_s_list, W_w, W_a)
    r_packed           = stage5_dsp(w_p, a_p, b_pre, w_s_list, a_s_list, W_w, W_a)
    _                  = stage6_postprocessing(r_packed, w_s_list, a_s_list, W_w, W_a)

    # Stage 8: full (P_w, P_a) efficiency table
    stage8_table(W_w, W_a)

    # Stress test
    stress_demo(W_w, W_a, P_w, P_a, n_trials=500)

    _sep()
    print("  Demo complete.")
    _sep()
    print()


if __name__ == "__main__":
    main()
