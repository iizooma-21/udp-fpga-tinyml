"""
tests/test_efficiency.py
========================
Unit tests for efficiency/model.py.

Covers is_feasible, find_all_valid, find_optimal, compute_port_widths,
and the PackingResult dataclass for correctness of constraint checking
(equations 6 & 7) and optimality of the parallelism solver.
"""

import unittest
from efficiency.model import (
    PackingResult,
    compute_port_widths,
    is_feasible,
    find_all_valid,
    find_optimal,
)
from core.dsp_sim import DSP_SHORT, DSP_LONG


class TestComputePortWidths(unittest.TestCase):

    def test_int4_pw3_pa2_paper_values(self):
        """int4 paper example: W_pw=17, W_pa=25, W_p=7."""
        W_p, W_pw, W_pa = compute_port_widths(3, 2, 4, 4)
        self.assertEqual(W_p,  7)
        self.assertEqual(W_pw, 17)
        self.assertEqual(W_pa, 25)

    def test_pw1_pa1_minimal(self):
        """Single weight and activation: no packing overhead."""
        W_p, W_pw, W_pa = compute_port_widths(1, 1, 4, 4)
        self.assertEqual(W_pw, 3)    # W_w - 1
        self.assertEqual(W_pa, 4)    # W_a

    def test_int8_pw1_pa1(self):
        W_p, W_pw, W_pa = compute_port_widths(1, 1, 8, 8)
        self.assertEqual(W_pw, 7)
        self.assertEqual(W_pa, 8)


class TestIsFeasible(unittest.TestCase):

    def test_paper_optimal_is_feasible(self):
        self.assertTrue(is_feasible(3, 2, 4, 4))

    def test_pw4_violates_eq6(self):
        """P_w=4, int4: W_pw = 3*7+3 = 24 > 18."""
        self.assertFalse(is_feasible(4, 1, 4, 4))

    def test_large_pa_violates_eq7(self):
        """P_w=3, P_a=3, int4: W_pa = 2*3*7+4 = 46 > 27."""
        self.assertFalse(is_feasible(3, 3, 4, 4))

    def test_pw1_pa1_always_feasible(self):
        for n in [2, 4, 8]:
            self.assertTrue(is_feasible(1, 1, n, n))

    def test_zero_and_negative_factors_infeasible(self):
        self.assertFalse(is_feasible(0, 1, 4, 4))
        self.assertFalse(is_feasible(1, 0, 4, 4))

    def test_custom_port_widths(self):
        """With tiny ports only (1,1) fits."""
        self.assertTrue( is_feasible(1, 1, 4, 4, W_short=3, W_long=4))
        self.assertFalse(is_feasible(2, 1, 4, 4, W_short=3, W_long=4))


class TestFindAllValid(unittest.TestCase):

    def test_int4_count_and_constraints(self):
        """All returned pairs must satisfy both equations."""
        results = find_all_valid(W_w=4, W_a=4)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertLessEqual(r.W_pw, DSP_SHORT, f"{r} violates eq6")
            self.assertLessEqual(r.W_pa, DSP_LONG,  f"{r} violates eq7")

    def test_sorted_descending_parallelism(self):
        results = find_all_valid(W_w=4, W_a=4)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].parallelism,
                                    results[i + 1].parallelism)

    def test_int4_includes_paper_optimum(self):
        results = find_all_valid(W_w=4, W_a=4)
        pairs = {(r.P_w, r.P_a) for r in results}
        self.assertIn((3, 2), pairs)

    def test_int32_no_valid_pairs(self):
        results = find_all_valid(W_w=32, W_a=32)
        self.assertEqual(results, [])

    def test_int2_has_more_options(self):
        """Smaller bitwidths allow more (P_w, P_a) combinations."""
        n2 = len(find_all_valid(W_w=2, W_a=2))
        n4 = len(find_all_valid(W_w=4, W_a=4))
        self.assertGreater(n2, n4)

    def test_no_duplicates(self):
        results = find_all_valid(W_w=4, W_a=4)
        pairs = [(r.P_w, r.P_a) for r in results]
        self.assertEqual(len(pairs), len(set(pairs)))


class TestFindOptimal(unittest.TestCase):

    def test_int4_optimal_is_3_2(self):
        """Paper result: optimal int4×int4 config is P_w=3, P_a=2 → 6 MACs."""
        opt = find_optimal(W_w=4, W_a=4)
        self.assertIsNotNone(opt)
        self.assertEqual(opt.P_w, 3)
        self.assertEqual(opt.P_a, 2)
        self.assertEqual(opt.parallelism, 6)

    def test_int8_optimal_is_feasible(self):
        opt = find_optimal(W_w=8, W_a=8)
        self.assertIsNotNone(opt)
        self.assertLessEqual(opt.W_pw, DSP_SHORT)
        self.assertLessEqual(opt.W_pa, DSP_LONG)
        self.assertGreaterEqual(opt.parallelism, 1)

    def test_int32_returns_none(self):
        self.assertIsNone(find_optimal(W_w=32, W_a=32))

    def test_custom_ports_respected(self):
        opt = find_optimal(W_w=4, W_a=4, W_short=10, W_long=10)
        if opt is not None:
            self.assertLessEqual(opt.W_pw, 10)
            self.assertLessEqual(opt.W_pa, 10)

    def test_optimal_maximises_over_all_valid(self):
        """find_optimal must return the max-parallelism entry."""
        all_valid = find_all_valid(W_w=4, W_a=4)
        opt = find_optimal(W_w=4, W_a=4)
        max_par = max(r.parallelism for r in all_valid)
        self.assertEqual(opt.parallelism, max_par)


class TestPackingResultFields(unittest.TestCase):

    def test_utilisation_bounds(self):
        """Port utilisation must be in (0, 1]."""
        for r in find_all_valid(W_w=4, W_a=4):
            self.assertGreater(r.short_util, 0)
            self.assertLessEqual(r.short_util, 1.0)
            self.assertGreater(r.long_util, 0)
            self.assertLessEqual(r.long_util, 1.0)

    def test_bottleneck_label(self):
        """Bottleneck must be whichever port has higher utilisation."""
        for r in find_all_valid(W_w=4, W_a=4):
            if r.short_util >= r.long_util:
                self.assertEqual(r.bottleneck, "short")
            else:
                self.assertEqual(r.bottleneck, "long")

    def test_str_representation(self):
        opt = find_optimal(W_w=4, W_a=4)
        s = str(opt)
        self.assertIn("P_w=3", s)
        self.assertIn("P_a=2", s)
        self.assertIn("parallelism=6", s)


if __name__ == "__main__":
    unittest.main()
