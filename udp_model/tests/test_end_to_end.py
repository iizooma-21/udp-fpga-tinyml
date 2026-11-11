"""
tests/test_end_to_end.py
========================
Full UDP pipeline integration test.

Pipeline under test
-------------------
    raw signed ints (w_s_list, a_s_list)
        ↓  clip_map / direct_map
    unsigned mapped values (w_u_list, a_u_list)
        ↓  pack_weights / pack_activations
    packed operands (w_p, a_p)
        ↓  compute_b_pre
    preprocessing correction (b_pre)
        ↓  dsp_sim.simulate  →  r_packed = w_p * a_p + b_pre
    packed DSP result (r_packed)
        ↓  compute_results  (postprocessing, eq 4)
    recovered products  results[j][i]
        ↓  assert  results[j][i] == w_s_i * a_s_j  for all i, j

For int4×int4 with the optimal config (P_w=3, P_a=2) each DSP call
returns exactly 6 signed products — verified against direct Python
multiplication.
"""

import sys
import os
import unittest
import random

# Allow running this file directly from inside the tests/ folder
# (e.g. `python tests/test_end_to_end.py`).
# When discovered by unittest/pytest the path is already correct.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.mapping       import clip_map, direct_map
from core.packing       import pack_weights, pack_activations
from core.preprocessing import compute_b_pre
from core.postprocessing import compute_results
from core.dsp_sim       import simulate, DSPOverflowError
from efficiency.model   import find_optimal


# ---------------------------------------------------------------------------
# Full pipeline helper
# ---------------------------------------------------------------------------

def udp_pipeline(w_s_list, a_s_list, W_w, W_a, verbose=False):
    """Execute the complete UDP pipeline and return results[j][i].

    Parameters
    ----------
    w_s_list : list of signed int weights
    a_s_list : list of signed int activations
    W_w, W_a : bitwidths

    Returns
    -------
    results  : list[list[int]]  results[j][i] = w_s_i * a_s_j
    r_packed : int              raw DSP output (for inspection)
    """
    P_w = len(w_s_list)
    P_a = len(a_s_list)

    # Step 1 – mapping
    w_u = [clip_map(w, W_w)   for w in w_s_list]
    a_u = [direct_map(a, W_a) for a in a_s_list]

    # Step 2 – packing
    w_p = pack_weights(w_u, W_w, W_a)
    a_p = pack_activations(a_u, P_w=P_w, W_w=W_w, W_a=W_a)

    # Step 3 – preprocessing
    b_pre = compute_b_pre(w_s_list, a_s_list, W_w=W_w, W_a=W_a)

    # Step 4 – DSP simulate (enforces port-width constraints)
    r_packed = simulate(w_p, a_p, b_pre, W_w=W_w, W_a=W_a,
                        P_w=P_w, P_a=P_a, verbose=verbose)

    # Step 5 – postprocessing
    results = compute_results(r_packed, w_s_list, a_s_list, W_w, W_a,
                              verbose=verbose)
    return results, r_packed


# ===========================================================================
# End-to-end tests
# ===========================================================================

class TestEndToEndInt4Optimal(unittest.TestCase):
    """int4×int4, P_w=3, P_a=2  — the paper's canonical 6-MAC configuration."""

    W_w = W_a = 4

    def _verify(self, w_s_list, a_s_list):
        results, _ = udp_pipeline(w_s_list, a_s_list, self.W_w, self.W_a)
        self.assertEqual(len(results), len(a_s_list),
                         "Row count (P_a) mismatch")
        self.assertEqual(len(results[0]), len(w_s_list),
                         "Column count (P_w) mismatch")
        for j, a in enumerate(a_s_list):
            for i, w in enumerate(w_s_list):
                self.assertEqual(results[j][i], w * a,
                                 f"Mismatch at (i={i},j={j}): "
                                 f"w={w}, a={a}, got={results[j][i]}")

    def test_paper_example_produces_6_results(self):
        """Single DSP call yields exactly 6 signed products."""
        w_s = [-3,  2, -1]
        a_s = [ 5, -2]
        results, _ = udp_pipeline(w_s, a_s, self.W_w, self.W_a)
        flat = [results[j][i] for j in range(2) for i in range(3)]
        self.assertEqual(len(flat), 6)
        expected_flat = [w * a for a in a_s for w in w_s]
        self.assertEqual(flat, expected_flat)

    def test_all_negative_weights(self):
        self._verify([-3, -2, -1], [5, -2])

    def test_all_positive_weights(self):
        self._verify([3, 2, 1], [5, -2])

    def test_all_negative_activations(self):
        self._verify([-3, 2, -1], [-5, -2])

    def test_all_positive_activations(self):
        self._verify([-3, 2, -1], [5, 3])

    def test_zeros_in_weights(self):
        self._verify([0, 0, 0], [5, -2])

    def test_zeros_in_activations(self):
        self._verify([-3, 2, -1], [0, 0])

    def test_extreme_values(self):
        """Boundary: most-negative and most-positive int4 values."""
        self._verify([-8, 7, -8], [7, -8])

    def test_random_50_grids(self):
        """50 random int4 3×2 grids must all recover exactly."""
        random.seed(42)
        errors = 0
        for _ in range(50):
            w_s = [random.randint(-8, 7) for _ in range(3)]
            a_s = [random.randint(-8, 7) for _ in range(2)]
            try:
                self._verify(w_s, a_s)
            except AssertionError:
                errors += 1
        self.assertEqual(errors, 0, f"{errors} random grids failed")


class TestEndToEndInt4Pw1Pa1(unittest.TestCase):
    """int4×int4, degenerate P_w=1, P_a=1 — baseline single-MAC case."""

    def test_all_256_combinations(self):
        for w in range(-8, 8):
            for a in range(-8, 8):
                results, _ = udp_pipeline([w], [a], 4, 4)
                self.assertEqual(results[0][0], w * a,
                                 f"w={w}, a={a}")


class TestEndToEndInt4Pw2Pa1(unittest.TestCase):
    """int4×int4, P_w=2, P_a=1 — double-weight packing."""

    def test_all_sign_combinations(self):
        sign_cases = [
            ([-3, -2], [-2]),   # both weights negative, negative activation
            ([-3,  2], [ 5]),   # mixed weights, positive activation
            ([ 3,  2], [-4]),   # both positive weights, negative activation
            ([ 3,  2], [ 5]),   # all positive
        ]
        for w_s, a_s in sign_cases:
            results, _ = udp_pipeline(w_s, a_s, 4, 4)
            for i, w in enumerate(w_s):
                self.assertEqual(results[0][i], w * a_s[0],
                                 f"w_s={w_s}, a_s={a_s}")


class TestEndToEndInt8(unittest.TestCase):
    """int8×int8 — wider operands, optimal P_w=1, P_a=2."""

    def test_optimal_config_2_results_per_dsp(self):
        """int8 optimal: 1 weight × 2 activations = 2 MACs per DSP cycle."""
        opt = find_optimal(W_w=8, W_a=8)
        self.assertIsNotNone(opt)
        self.assertEqual(opt.parallelism, 2)

        random.seed(7)
        for _ in range(100):
            w_s = [random.randint(-128, 127)]
            a_s = [random.randint(-128, 127) for _ in range(opt.P_a)]
            results, _ = udp_pipeline(w_s, a_s, 8, 8)
            for j, a in enumerate(a_s):
                self.assertEqual(results[j][0], w_s[0] * a,
                                 f"w={w_s[0]}, a={a}")


class TestEndToEndDSPPortConstraints(unittest.TestCase):
    """Verify DSPOverflowError is raised for configs that exceed port widths."""

    def test_too_many_weights_raises_overflow(self):
        """P_w=4 for int4: W_pw = (4-1)*7 + 3 = 24 > 18 must raise DSPOverflowError."""
        from efficiency.model import is_feasible
        self.assertFalse(is_feasible(4, 1, W_w=4, W_a=4))   # confirm infeasible
        with self.assertRaises(DSPOverflowError):
            udp_pipeline([-3, 2, -1, 0], [5, -2], W_w=4, W_a=4)

    def test_valid_config_does_not_raise(self):
        """P_w=3, P_a=2 int4 fits perfectly — no exception."""
        try:
            udp_pipeline([-3, 2, -1], [5, -2], 4, 4)
        except DSPOverflowError as e:
            self.fail(f"Unexpected DSPOverflowError: {e}")


class TestEndToEndEdgeCases(unittest.TestCase):
    """Numerical edge cases across the full pipeline."""

    def test_most_negative_times_most_negative(self):
        """(-8)*(-8) = 64 overflows 7-bit signed range but integer is exact."""
        results, _ = udp_pipeline([-8], [-8], 4, 4)
        self.assertEqual(results[0][0], 64)

    def test_most_negative_times_most_positive(self):
        results, _ = udp_pipeline([-8], [7], 4, 4)
        self.assertEqual(results[0][0], -56)

    def test_all_zeros(self):
        results, r = udp_pipeline([0, 0, 0], [0, 0], 4, 4)
        self.assertEqual(r, 0)
        for row in results:
            for v in row:
                self.assertEqual(v, 0)

    def test_result_shape_matches_pw_pa(self):
        """results[j][i] shape must be [P_a][P_w]."""
        for P_w, P_a in [(1, 1), (2, 1), (1, 2), (3, 2)]:
            w_s = [1] * P_w
            a_s = [1] * P_a
            results, _ = udp_pipeline(w_s, a_s, 4, 4)
            self.assertEqual(len(results), P_a)
            for row in results:
                self.assertEqual(len(row), P_w)


def _run_comparison_demo():
    """Print a side-by-side comparison: UDP pipeline vs direct Python multiply."""

    SEP  = "=" * 72
    SEP2 = "-" * 72

    configs = [
        # (label,               W_w, W_a, w_s_list,          a_s_list)
        ("int4 x int4  (paper example,  P_w=3, P_a=2)",
         4, 4, [-3, 2, -1], [5, -2]),
        ("int4 x int4  (all negative weights,  P_w=3, P_a=2)",
         4, 4, [-3, -2, -1], [5, -7]),
        ("int4 x int4  (boundary values,  P_w=3, P_a=2)",
         4, 4, [-8,  7, -8], [7, -8]),
        ("int8 x int8  (single weight,   P_w=1, P_a=2)",
         8, 8, [-100], [64, -127]),
    ]

    print()
    print(SEP)
    print("  UDP Pipeline  vs  Direct Python Multiplication")
    print(SEP)
    print("""
  For each configuration the full pipeline runs:
    raw signed ints -> clip_map / direct_map
    -> pack_weights / pack_activations
    -> compute_b_pre  (preprocessing, eq 3)
    -> DSP simulate   (r = w_p * a_p + b_pre)
    -> compute_results (postprocessing, eq 4)
  and the recovered products are compared to w_s[i] * a_s[j] directly.
""")

    for label, W_w, W_a, w_s, a_s in configs:
        print(SEP2)
        print(f"  {label}")
        print(f"  W_w={W_w}, W_a={W_a}")
        print(f"  weights     : {w_s}")
        print(f"  activations : {a_s}")
        print()

        results, r_packed = udp_pipeline(w_s, a_s, W_w, W_a)
        P_w, P_a = len(w_s), len(a_s)

        print(f"  DSP output r_packed = {r_packed}  "
              f"({r_packed.bit_length()}-bit)")
        print()
        print(f"  {'i':>3}  {'j':>3}  {'w_s[i]':>8}  {'a_s[j]':>8}  "
              f"{'UDP result':>12}  {'direct':>12}  {'match?':>8}")
        print(f"  {'---':>3}  {'---':>3}  {'--------':>8}  {'--------':>8}  "
              f"{'----------':>12}  {'----------':>12}  {'--------':>8}")

        all_ok = True
        for j, a in enumerate(a_s):
            for i, w in enumerate(w_s):
                udp_val    = results[j][i]
                direct_val = w * a
                ok         = udp_val == direct_val
                all_ok     = all_ok and ok
                tick       = "OK" if ok else "MISMATCH"
                print(f"  {i:>3}  {j:>3}  {w:>8}  {a:>8}  "
                      f"{udp_val:>12}  {direct_val:>12}  {tick:>8}")

        print()
        status = f"All {P_w * P_a} products match." if all_ok else "ERRORS FOUND."
        print(f"  Result: {status}  "
              f"({P_w} weights x {P_a} activations = {P_w*P_a} MACs per DSP cycle)")
        print()

    print(SEP)
    print("  All comparisons complete.")
    print(SEP)
    print()


if __name__ == "__main__":
    _run_comparison_demo()
