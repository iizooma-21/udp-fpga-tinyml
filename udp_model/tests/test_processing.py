"""
tests/test_processing.py
========================
Unit tests for core/preprocessing.py and core/postprocessing.py.

Preprocessing  (eq 3):  b_pre = f(w_s_list, a_s_list)
Postprocessing (eq 4):  results[j][i] = w_s_i * a_s_j recovered from r_packed
"""

import unittest
from core.mapping import clip_map, direct_map
from core.packing import pack_weights, pack_activations, compute_wp
from core.preprocessing  import compute_b_pre,    needs_preprocessing
from core.postprocessing import compute_results,   needs_postprocessing


# ---------------------------------------------------------------------------
# Shared pipeline helper
# ---------------------------------------------------------------------------

def _full_pipeline(w_s_list, a_s_list, W_w, W_a):
    """map → pack → b_pre → r_packed → postprocessing → results grid."""
    P_w = len(w_s_list)
    w_u = [clip_map(w, W_w)   for w in w_s_list]
    a_u = [direct_map(a, W_a) for a in a_s_list]
    w_p   = pack_weights(w_u, W_w, W_a)
    a_p   = pack_activations(a_u, P_w=P_w, W_w=W_w, W_a=W_a)
    b_pre = compute_b_pre(w_s_list, a_s_list, W_w=W_w, W_a=W_a)
    r_packed = w_p * a_p + b_pre
    return compute_results(r_packed, w_s_list, a_s_list, W_w, W_a)


# ===========================================================================
# Preprocessing tests
# ===========================================================================

class TestComputeBPre(unittest.TestCase):

    def test_all_positive_weights_gives_zero(self):
        """No negative weights → b_pre = 0 (no correction needed)."""
        b = compute_b_pre([3, 2, 1], [5, -2], W_w=4, W_a=4)
        self.assertEqual(b, 0)

    def test_all_zero_activations_gives_zero(self):
        """Activations are all zero → correction terms vanish."""
        b = compute_b_pre([-3, -1], [0, 0], W_w=4, W_a=4)
        self.assertEqual(b, 0)

    def test_single_negative_weight_known_value(self):
        """P_w=1, P_a=1, w=-1, a=3: correction = neg_a << (W_w-1) = 13<<3 = 104."""
        b = compute_b_pre([-1], [3], W_w=4, W_a=4)
        neg_a = (-3) % (2 ** 4)       # 13
        expected = neg_a << (4 - 1)   # 104  (placed at bit W_w-1 within slot 0)
        self.assertEqual(b, expected)

    def test_b_pre_folds_preprocessing_term(self):
        """After adding b_pre the raw slot holds w_u*a_u + correction."""
        from core.packing import unpack_results
        W_w, W_a = 4, 4
        W_p = compute_wp(W_w, W_a)
        for w_s, a_s in [(-3, 5), (-1, -2), (3, -4), (-4, -4), (0, 7)]:
            w_u = clip_map(w_s, W_w)
            a_u = direct_map(a_s, W_a)
            w_p   = pack_weights([w_u], W_w, W_a)
            a_p   = pack_activations([a_u], P_w=1, W_w=W_w, W_a=W_a)
            b_pre = compute_b_pre([w_s], [a_s], W_w, W_a)
            r_p   = w_p * a_p + b_pre
            raw_slot = unpack_results(r_p, P_w=1, P_a=1, W_w=W_w, W_a=W_a)[0][0]
            neg_a = (-a_s) % (2 ** W_a)
            correction = (neg_a << (W_w - 1)) * (1 if w_s < 0 else 0)
            expected_slot = w_u * a_u + correction
            self.assertEqual(raw_slot, expected_slot,
                             f"w_s={w_s}, a_s={a_s}")

    def test_non_negative(self):
        """b_pre is always >= 0 (it's added to the DSP, not subtracted)."""
        cases = [
            ([-3, 2, -1], [5, -2]),
            ([-8, -8],    [-8, -8]),
            ([0, 0],       [0,  0]),
        ]
        for ws, acs in cases:
            b = compute_b_pre(ws, acs, W_w=4, W_a=4)
            self.assertGreaterEqual(b, 0)


class TestNeedsPreprocessing(unittest.TestCase):

    def test_negative_weight_triggers(self):
        self.assertTrue(needs_preprocessing([-1, 2], W_w=4))

    def test_all_positive_no_trigger(self):
        self.assertFalse(needs_preprocessing([1, 2, 3], W_w=4))

    def test_zero_only_no_trigger(self):
        self.assertFalse(needs_preprocessing([0, 0], W_w=4))


# ===========================================================================
# Postprocessing tests
# ===========================================================================

class TestComputeResults(unittest.TestCase):

    def test_pw1_pa1_all_int4(self):
        """All 256 int4 pairs must recover exactly."""
        W_w, W_a = 4, 4
        errors = 0
        for w in range(-8, 8):
            for a in range(-8, 8):
                res = _full_pipeline([w], [a], W_w, W_a)
                if res[0][0] != w * a:
                    errors += 1
        self.assertEqual(errors, 0)

    def test_pw2_pa1_mixed_signs(self):
        res = _full_pipeline([-3, -2], [-2], W_w=4, W_a=4)
        self.assertEqual(res[0][0], (-3) * (-2))   # 6
        self.assertEqual(res[0][1], (-2) * (-2))   # 4

    def test_pw1_pa2_most_negative_weight(self):
        """Edge case: w = -2^(W_w-1) with two activations."""
        res = _full_pipeline([-8], [7, -1], W_w=4, W_a=4)
        self.assertEqual(res[0][0], -8 *  7)
        self.assertEqual(res[1][0], -8 * -1)

    def test_pw3_pa2_paper_example(self):
        """3x2 product grid — the paper's running example."""
        w_s = [-3, 2, -1]
        a_s = [5, -2]
        res = _full_pipeline(w_s, a_s, W_w=4, W_a=4)
        for j, a in enumerate(a_s):
            for i, w in enumerate(w_s):
                self.assertEqual(res[j][i], w * a,
                                 f"res[{j}][{i}]: w={w}, a={a}")

    def test_all_positive_inputs(self):
        """No corrections needed — results equal direct products."""
        res = _full_pipeline([3, 2, 1], [5, 6], W_w=4, W_a=4)
        expected = [[3*5, 2*5, 1*5], [3*6, 2*6, 1*6]]
        self.assertEqual(res, expected)

    def test_inner_j_spill_pw3(self):
        """For P_w=3 the inner_j residual spills into slot i=1 — must be
        correctly extracted via get_bits, not just applied to slot 0."""
        w_s = [-3, 2, -1]
        a_s = [5]
        res = _full_pipeline(w_s, a_s, W_w=4, W_a=4)
        self.assertEqual(res[0][0], -3 * 5)
        self.assertEqual(res[0][1],  2 * 5)
        self.assertEqual(res[0][2], -1 * 5)

    def test_int8_random_sample(self):
        """500 random int8 pairs with P_w=1, P_a=1."""
        import random
        random.seed(0)
        for _ in range(500):
            w = random.randint(-128, 127)
            a = random.randint(-128, 127)
            res = _full_pipeline([w], [a], W_w=8, W_a=8)
            self.assertEqual(res[0][0], w * a, f"w={w}, a={a}")

    def test_most_negative_values_edge_case(self):
        """w = a = -2^(W_w-1): product may overflow W_p but integer is exact."""
        res = _full_pipeline([-8], [-8], W_w=4, W_a=4)
        self.assertEqual(res[0][0], 64)   # (-8)*(-8) = 64, overflows 7-bit signed


class TestNeedsPostprocessing(unittest.TestCase):

    def test_negative_activation_triggers(self):
        self.assertTrue(needs_postprocessing([-1, 2], W_a=4))

    def test_all_positive_no_trigger(self):
        self.assertFalse(needs_postprocessing([1, 2], W_a=4))

    def test_zero_no_trigger(self):
        self.assertFalse(needs_postprocessing([0, 7], W_a=4))


if __name__ == "__main__":
    unittest.main()
