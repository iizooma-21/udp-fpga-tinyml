"""
tests/test_packing.py
=====================
Unit tests for core/packing.py.

Covers compute_wp, pack_weights, pack_activations, unpack_results,
and verify_packing.  The key property tested throughout is:

    unpack_results(w_p * a_p)[j][i] == w_u_i * a_u_j   for all i, j
"""

import unittest
from core.packing import (
    compute_wp,
    pack_weights,
    pack_activations,
    unpack_results,
    verify_packing,
)
from core.mapping import clip_map, direct_map


class TestComputeWp(unittest.TestCase):

    def test_int4(self):
        self.assertEqual(compute_wp(4, 4), 7)

    def test_int8(self):
        self.assertEqual(compute_wp(8, 8), 15)

    def test_mixed(self):
        self.assertEqual(compute_wp(4, 8), 11)
        self.assertEqual(compute_wp(8, 4), 11)

    def test_minimal(self):
        self.assertEqual(compute_wp(2, 1), 2)


class TestPackWeights(unittest.TestCase):

    def _max_w_u(self, W_w):
        return 2 ** (W_w - 1) - 1

    def test_single_weight_at_slot_0(self):
        """Single weight occupies bits [0, W_w-2] of w_p."""
        W_w, W_a = 4, 4
        w_p = pack_weights([5], W_w, W_a)
        self.assertEqual(w_p, 5)

    def test_two_weights_non_overlapping(self):
        """Second slot starts at bit W_p = W_w + W_a - 1."""
        W_w, W_a = 4, 4
        W_p = compute_wp(W_w, W_a)          # 7
        w_p = pack_weights([3, 5], W_w, W_a)
        self.assertEqual(w_p, 3 | (5 << W_p))

    def test_three_weights_paper_example(self):
        """int4 paper example: [5,2,7] → W_p=7 → 5 + 2<<7 + 7<<14."""
        W_w, W_a = 4, 4
        W_p = 7
        w_p = pack_weights([5, 2, 7], W_w, W_a)
        self.assertEqual(w_p, 5 + (2 << 7) + (7 << 14))

    def test_all_zeros(self):
        w_p = pack_weights([0, 0, 0], 4, 4)
        self.assertEqual(w_p, 0)

    def test_max_values(self):
        """Values at boundary (2^(W_w-1)-1) must not overflow into next slot."""
        W_w, W_a, W_p = 4, 4, 7
        mv = self._max_w_u(W_w)              # 7
        w_p = pack_weights([mv, mv], W_w, W_a)
        # Slot 0: bits 0-2, slot 1: bits 7-9
        slot0 = w_p & 0x7F
        slot1 = (w_p >> W_p) & 0x7F
        self.assertEqual(slot0, mv)
        self.assertEqual(slot1, mv)


class TestPackActivations(unittest.TestCase):

    def test_single_activation(self):
        a_p = pack_activations([5], P_w=1, W_w=4, W_a=4)
        self.assertEqual(a_p, 5)

    def test_two_activations_stride(self):
        """Stride between activation slots is P_w * W_p."""
        W_w, W_a, P_w = 4, 4, 3
        W_p = compute_wp(W_w, W_a)   # 7
        stride = P_w * W_p           # 21
        a_p = pack_activations([5, 14], P_w=P_w, W_w=W_w, W_a=W_a)
        self.assertEqual(a_p, 5 + (14 << stride))

    def test_all_zeros(self):
        a_p = pack_activations([0, 0], P_w=2, W_w=4, W_a=4)
        self.assertEqual(a_p, 0)


class TestUnpackResults(unittest.TestCase):

    def test_single_slot_identity(self):
        """With P_w=1, P_a=1 the packed value IS the single product."""
        W_w, W_a = 4, 4
        w_u, a_u = 5, 13
        product = w_u * a_u
        w_p = pack_weights([w_u], W_w, W_a)
        a_p = pack_activations([a_u], P_w=1, W_w=W_w, W_a=W_a)
        r_p = w_p * a_p
        slots = unpack_results(r_p, P_w=1, P_a=1, W_w=W_w, W_a=W_a)
        self.assertEqual(slots[0][0], product)

    def test_multi_slot_layout(self):
        """Each slot[j][i] == w_u_i * a_u_j for the paper's 3x2 example."""
        W_w, W_a = 4, 4
        w_u_list = [5, 2, 7]
        a_u_list = [2, 9]
        w_p = pack_weights(w_u_list, W_w, W_a)
        a_p = pack_activations(a_u_list, P_w=3, W_w=W_w, W_a=W_a)
        r_p = w_p * a_p
        slots = unpack_results(r_p, P_w=3, P_a=2, W_w=W_w, W_a=W_a)
        for j, a_u in enumerate(a_u_list):
            for i, w_u in enumerate(w_u_list):
                self.assertEqual(slots[j][i], w_u * a_u,
                                 f"slot[{j}][{i}] mismatch")


class TestVerifyPacking(unittest.TestCase):

    def test_paper_example(self):
        self.assertTrue(
            verify_packing([5, 3, 6], [2, 9], W_w=4, W_a=4)
        )

    def test_single_pair(self):
        self.assertTrue(verify_packing([7], [15], W_w=4, W_a=4))

    def test_exhaustive_int4_pw2_pa2(self):
        """All valid unsigned int4 weight/activation pairs with P_w=2, P_a=2."""
        W_w, W_a = 4, 4
        max_w = 2 ** (W_w - 1) - 1     # 7
        max_a = 2 ** W_a - 1            # 15
        errors = 0
        for w0 in range(max_w + 1):
            for w1 in range(max_w + 1):
                for a0 in range(max_a + 1):
                    for a1 in range(max_a + 1):
                        ok = verify_packing([w0, w1], [a0, a1], W_w, W_a)
                        if not ok:
                            errors += 1
        self.assertEqual(errors, 0)

    def test_int8_single(self):
        self.assertTrue(verify_packing([100], [200], W_w=8, W_a=8))


if __name__ == "__main__":
    unittest.main()
