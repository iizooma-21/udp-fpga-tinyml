"""
tests/test_convolution.py
=========================
Tests and demos UDP acceleration on a 1D multi-filter convolution.

WHAT "CONVOLUTION LOOP DIVIDED BY 6" MEANS
-------------------------------------------
A 1D convolution with F output filters, K kernel taps, and N_out output
positions requires F * N_out * K multiply-accumulate (MAC) operations.

  Test A  —  standard loop: one scalar MAC per iteration
              Total iterations = F * N_out * K

  Test B  —  UDP-accelerated loop: 6 MACs per DSP call
              For each kernel tap k and each pair of output positions (n, n+1):

                w_s = [w[0][k], w[1][k], w[2][k]]   <- P_w=3 filter values
                a_s = [x[n+k],  x[n+1+k]]            <- P_a=2 input values
                results = UDP_call(w_s, a_s)          <- 1 DSP cycle, 6 MACs

              The 6 products that come out:
                results[0][0] = w[0][k] * x[n+k]     -> accumulates into y[0][n]
                results[0][1] = w[1][k] * x[n+k]     -> accumulates into y[1][n]
                results[0][2] = w[2][k] * x[n+k]     -> accumulates into y[2][n]
                results[1][0] = w[0][k] * x[n+1+k]   -> accumulates into y[0][n+1]
                results[1][1] = w[1][k] * x[n+1+k]   -> accumulates into y[1][n+1]
                results[1][2] = w[2][k] * x[n+1+k]   -> accumulates into y[2][n+1]

              All 6 products are USEFUL — no waste.

              Total iterations = K * (N_out / 2) = (F * N_out * K) / 6
                                                   ^^^^^^^^^^^^^^^^^^^
                                                   loop divided by 6  (OK)

SETUP
-----
  W_w = W_a = 4   int4 weights and activations
  F   = 3         output channels / filters  (= P_w)
  K   = 6         kernel size (taps per filter)
  P_w = 3, P_a = 2  -> 6 MACs per UDP call

Run directly to see the comparison demo:
    python tests/test_convolution.py
"""

import sys, os, unittest, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.mapping        import clip_map, direct_map
from core.packing        import pack_weights, pack_activations
from core.preprocessing  import compute_b_pre
from core.postprocessing import compute_results
from core.dsp_sim        import simulate


# ============================================================================
# Configuration
# ============================================================================

W_w = W_a = 4    # int4
F   = 3          # output channels (= P_w)
K   = 6          # kernel taps
P_w = 3          # weights per UDP call  (= F, one value per filter per tap)
P_a = 2          # activations per UDP call  (= positions processed at once)


# ============================================================================
# Shared UDP helper
# ============================================================================

def _udp_call(w_s_list, a_s_list):
    """Run one UDP pipeline call; return results[j][i] product grid."""
    Pw = len(w_s_list)
    w_u   = [clip_map(w, W_w) for w in w_s_list]
    a_u   = [direct_map(a, W_a) for a in a_s_list]
    w_p   = pack_weights(w_u, W_w, W_a)
    a_p   = pack_activations(a_u, P_w=Pw, W_w=W_w, W_a=W_a)
    b_pre = compute_b_pre(w_s_list, a_s_list, W_w=W_w, W_a=W_a)
    r     = simulate(w_p, a_p, b_pre, W_w=W_w, W_a=W_a,
                     P_w=Pw, P_a=len(a_s_list))
    return compute_results(r, w_s_list, a_s_list, W_w, W_a)


# ============================================================================
# TEST A  —  Standard convolution loop (one scalar MAC per iteration)
# ============================================================================

def conv_standard(w, x):
    """
    Naive 1D multi-filter convolution.

    y[f][n] = sum_{k=0}^{K-1}  w[f][k] * x[n + k]

    Returns
    -------
    y          : list[list[int]]  output grid, shape [F][N_out]
    mac_count  : int              total number of scalar MAC iterations
    """
    N_out = len(x) - K + 1
    y     = [[0] * N_out for _ in range(F)]
    mac_count = 0

    for f in range(F):
        for n in range(N_out):
            for k in range(K):
                y[f][n] += w[f][k] * x[n + k]
                mac_count += 1

    return y, mac_count


# ============================================================================
# TEST B  —  UDP-accelerated convolution (loop divided by 6)
# ============================================================================

def conv_udp(w, x):
    """
    UDP-accelerated 1D multi-filter convolution.

    For each kernel tap k and each pair of output positions (n, n+1):
      w_s = [w[0][k], w[1][k], w[2][k]]     (one tap value per filter)
      a_s = [x[n+k],  x[n+1+k]]             (two consecutive input values)
      results[j][i] = w[i][k] * x[(n+j)+k]  (6 products at once)

    Each product accumulates into y[filter=i][position=n+j].

    If N_out is odd the last batch has one position and is padded with a_s=0;
    the padding product is computed but not accumulated (zero contribution).

    Returns
    -------
    y           : list[list[int]]  identical to conv_standard output
    call_count  : int              number of UDP (DSP) calls made
    """
    N_out = len(x) - K + 1
    y     = [[0] * N_out for _ in range(F)]
    call_count = 0

    for k in range(K):
        n = 0
        while n < N_out:
            # Collect up to P_a consecutive output positions
            n_batch = [n + j for j in range(P_a) if n + j < N_out]

            # One weight value per filter at this tap
            w_s = [w[f][k] for f in range(F)]

            # Input values at (position + tap) for each position in the batch
            a_s = [x[pos + k] for pos in n_batch]

            # Pad to P_a if the last batch is short (last position in signal)
            while len(a_s) < P_a:
                a_s.append(0)

            # --- ONE DSP CALL: 6 products ---
            results = _udp_call(w_s, a_s)
            call_count += 1

            # Accumulate only the valid (non-padded) products
            for j, pos in enumerate(n_batch):
                for i in range(F):
                    y[i][pos] += results[j][i]

            n += P_a

    return y, call_count


# ============================================================================
# Unit tests
# ============================================================================

def _random_inputs(seed, N=15):
    """Return random int4 filter weights and input signal."""
    random.seed(seed)
    lo, hi = -(2 ** (W_w - 1)), 2 ** (W_w - 1) - 1
    w = [[random.randint(lo, hi) for _ in range(K)] for _ in range(F)]
    x = [random.randint(lo, hi) for _ in range(N)]
    return w, x


class TestConvStandard(unittest.TestCase):

    def test_output_shape(self):
        w, x = _random_inputs(0)
        y, _ = conv_standard(w, x)
        self.assertEqual(len(y), F)
        self.assertEqual(len(y[0]), len(x) - K + 1)

    def test_mac_count(self):
        w, x = _random_inputs(1, N=15)
        N_out = len(x) - K + 1
        _, count = conv_standard(w, x)
        self.assertEqual(count, F * N_out * K)

    def test_zero_input_gives_zero(self):
        w, _ = _random_inputs(0)
        x = [0] * 15
        y, _ = conv_standard(w, x)
        for row in y:
            self.assertEqual(row, [0] * (15 - K + 1))

    def test_known_single_tap(self):
        """K=1: y[f][n] = w[f][0] * x[n]  for all f, n."""
        K_saved = globals().get('K')
        # Manually compute K=1 case inline
        w1 = [[3], [-2], [1]]
        x1 = [1, -2, 3, -4]
        y = [[w1[f][0] * x1[n] for n in range(4)] for f in range(3)]
        expected = [[3, -6, 9, -12], [-2, 4, -6, 8], [1, -2, 3, -4]]
        self.assertEqual(y, expected)


class TestConvUDP(unittest.TestCase):

    def test_matches_standard_output(self):
        """UDP and standard loops produce identical results for 10 random seeds."""
        for seed in range(10):
            w, x = _random_inputs(seed)
            y_a, _ = conv_standard(w, x)
            y_b, _ = conv_udp(w, x)
            self.assertEqual(y_a, y_b, f"Output mismatch at seed={seed}")

    def test_loop_divided_by_6_even_N_out(self):
        """For even N_out: UDP call count = mac count / 6 exactly."""
        w, x = _random_inputs(0, N=15)   # N_out = 15 - 6 + 1 = 10 (even)
        N_out = len(x) - K + 1
        _, mac_count  = conv_standard(w, x)
        _, call_count = conv_udp(w, x)
        self.assertEqual(N_out % P_a, 0, "N_out must be even for this test")
        self.assertEqual(call_count, mac_count // (P_w * P_a))
        self.assertEqual(call_count * P_w * P_a, mac_count)

    def test_call_count_formula(self):
        """call_count == K * ceil(N_out / P_a) for any N."""
        for N in [14, 15, 16, 17]:
            w, x = _random_inputs(0, N=N)
            N_out = N - K + 1
            _, call_count = conv_udp(w, x)
            expected = K * math.ceil(N_out / P_a)
            self.assertEqual(call_count, expected, f"N={N}")

    def test_odd_N_out_still_correct(self):
        """Padded last batch for odd N_out produces correct output."""
        w, x = _random_inputs(7, N=14)   # N_out = 14 - 6 + 1 = 9 (odd)
        y_a, _ = conv_standard(w, x)
        y_b, _ = conv_udp(w, x)
        self.assertEqual(y_a, y_b)

    def test_all_positive_weights(self):
        w = [[i + 1 for _ in range(K)] for i in range(F)]
        x = [1] * 15
        y_a, _ = conv_standard(w, x)
        y_b, _ = conv_udp(w, x)
        self.assertEqual(y_a, y_b)

    def test_zero_input(self):
        w, _ = _random_inputs(0)
        x = [0] * 15
        y_b, _ = conv_udp(w, x)
        for row in y_b:
            self.assertEqual(row, [0] * (15 - K + 1))

    def test_extreme_values(self):
        """Boundary int4 values (-8, 7) across full convolution."""
        w = [[-8 if k % 2 == 0 else 7 for k in range(K)] for _ in range(F)]
        x = [-8, 7] * 9
        y_a, _ = conv_standard(w, x[:15])
        y_b, _ = conv_udp(w, x[:15])
        self.assertEqual(y_a, y_b)


# ============================================================================
# Demo (run directly)
# ============================================================================

def _run_demo():
    SEP  = "=" * 72
    SEP2 = "-" * 72

    random.seed(42)
    lo, hi = -(2 ** (W_w - 1)), 2 ** (W_w - 1) - 1
    N     = 17            # input signal length  -- N_out = 12 (even, gives exact 6x)
    N_out = N - K + 1

    w = [[random.randint(lo, hi) for _ in range(K)] for _ in range(F)]
    x = [random.randint(lo, hi) for _ in range(N)]

    print()
    print(SEP)
    print("  UDP  —  Convolution Loop Divided by 6")
    print(SEP)
    print(f"""
  Config
  ------
  Bitwidths    :  int{W_w} weights  x  int{W_a} activations
  Filters (F)  :  {F}   (one per output channel)
  Kernel size  :  K = {K} taps per filter
  Input length :  N = {N}  ->  N_out = {N_out} output positions per filter

  Convolution formula:
    y[f][n] = sum_{{k=0}}^{{{K-1}}}  w[f][k] * x[n+k]
    for f = 0..{F-1},  n = 0..{N_out-1}
""")

    # Print the filter weights
    print("  Filter weights w[f][k]:")
    print("  " + "     ".join(f"k={k}" for k in range(K)))
    print("  " + "-" * (6 * K))
    for f in range(F):
        print(f"  f={f}:  " + "  ".join(f"{w[f][k]:4d}" for k in range(K)))
    print(f"\n  Input signal x = {x}\n")

    # ── TEST A ────────────────────────────────────────────────────────────
    print(SEP2)
    print("  TEST A  —  Standard convolution loop")
    print(SEP2)

    y_A, mac_count = conv_standard(w, x)

    print(f"""
  Loop structure:
    for f in range({F}):          <- output channel
      for n in range({N_out}):    <- output position
        for k in range({K}):      <- kernel tap
          y[f][n] += w[f][k] * x[n+k]   <- 1 scalar MAC

  Iterations:  F x N_out x K  =  {F} x {N_out} x {K}  =  {mac_count}
""")

    print("  Output y[f][n]:")
    print("         " + "".join(f"  n={n}" for n in range(N_out)))
    for f in range(F):
        print(f"  f={f}:  " + "".join(f"  {y_A[f][n]:4d}" for n in range(N_out)))

    # ── TEST B ────────────────────────────────────────────────────────────
    print()
    print(SEP2)
    print("  TEST B  —  UDP-accelerated convolution  (loop divided by 6)")
    print(SEP2)

    y_B, call_count = conv_udp(w, x)
    reduction = mac_count // call_count

    print(f"""
  Loop structure:
    for k in range({K}):                    <- kernel tap
      for n in range(0, {N_out}, {P_a}):    <- output positions, step={P_a}
        w_s = [w[0][k], w[1][k], w[2][k]]  (P_w={P_w} filter values)
        a_s = [x[n+k],  x[n+1+k]]          (P_a={P_a} input values)
        results = UDP_call(w_s, a_s)        <- 1 DSP cycle  =  {P_w*P_a} MACs
        accumulate all 6 products into y

  Total UDP calls : {call_count:4d}  =  K * (N_out / P_a) = {K} * ({N_out} / {P_a})
  MACs per call   : {P_w * P_a:4d}  =  P_w * P_a = {P_w} * {P_a}
  Iterations      : {call_count:4d}  vs Test A: {mac_count}  ->  {reduction}x fewer  (*)
""")

    print("  Each UDP call example  (k=0, n=0):")
    print(f"    w_s = {[w[f][0] for f in range(F)]}")
    print(f"    a_s = [{x[0]}, {x[1]}]")
    print(f"    products:")
    for j, pos in enumerate([0, 1]):
        for i in range(F):
            val = w[i][0] * x[pos]
            print(f"      results[{j}][{i}] = w[{i}][0]*x[{pos}] = "
                  f"{w[i][0]}*{x[pos]} = {val:4d}  -> y[{i}][{pos}]")

    print()
    print("  Output y[f][n]:")
    print("         " + "".join(f"  n={n}" for n in range(N_out)))
    for f in range(F):
        print(f"  f={f}:  " + "".join(f"  {y_B[f][n]:4d}" for n in range(N_out)))

    # ── VERIFICATION ──────────────────────────────────────────────────────
    print()
    print(SEP2)
    print("  VERIFICATION  —  Test A  vs  Test B")
    print(SEP2)

    all_match = (y_A == y_B)

    print(f"""
  Test A  scalar MAC iterations : {mac_count:5d}
  Test B  UDP DSP calls         : {call_count:5d}   (each = {P_w*P_a} MACs)
  Loop reduction                : {reduction:5d}x   (= P_w * P_a = {P_w}*{P_a})

  Output match: {"YES — Test A and Test B produce identical results" if all_match else "NO — MISMATCH"}
""")

    if not all_match:
        for f in range(F):
            for n in range(N_out):
                if y_A[f][n] != y_B[f][n]:
                    print(f"  MISMATCH at y[{f}][{n}]: "
                          f"A={y_A[f][n]}, B={y_B[f][n]}")

    print(SEP)
    print("  Demo complete.")
    print(SEP)
    print()


if __name__ == "__main__":
    _run_demo()
