"""
tests/test_recovery.py
======================
Unit tests for core/recovery.py.

Covers recover_clip_direct (eq 1a), recover_clip_offset (eq 1b),
recover_clip_clip (eq 1c), and the generic recover() dispatcher,
for all four sign combinations and across multiple bitwidths.
"""

import unittest
from core.mapping import clip_map, direct_map, offset_map
from core.recovery import (
    recover_clip_direct,
    recover_clip_offset,
    recover_clip_clip,
    recover,
)


def _all_signed(n):
    """All signed integers representable in n bits."""
    return range(-(2 ** (n - 1)), 2 ** (n - 1))


class TestRecoverClipDirect(unittest.TestCase):
    """Equation 1a: clip weights + direct activations (UDP's choice)."""

    def _check(self, w_s, a_s, n):
        w_u = clip_map(w_s, n)
        a_u = direct_map(a_s, n)
        result = recover_clip_direct(w_u, a_u, w_s, a_s, n)
        self.assertEqual(result, w_s * a_s,
                         f"w_s={w_s}, a_s={a_s}, n={n}")

    def test_all_int4_combinations(self):
        """All 256 signed int4 pairs must recover exactly."""
        for w in _all_signed(4):
            for a in _all_signed(4):
                self._check(w, a, 4)

    def test_all_positive_weights(self):
        """When all weights >= 0, clip_map is identity and correction is zero."""
        for w in range(0, 8):
            for a in _all_signed(4):
                self._check(w, a, 4)

    def test_all_positive_activations(self):
        """When all activations >= 0, direct_map is identity."""
        for w in _all_signed(4):
            for a in range(0, 8):
                self._check(w, a, 4)

    def test_both_negative(self):
        """Both negative — cross-correction term (term 4) is active."""
        for w in range(-8, 0):
            for a in range(-8, 0):
                self._check(w, a, 4)

    def test_int8_sample(self):
        """Spot-check a selection of int8 pairs."""
        cases = [(-128, -128), (-128, 127), (127, -128), (0, -1),
                 (-1, 0), (-64, 64), (63, -63)]
        for w, a in cases:
            self._check(w, a, 8)


class TestRecoverClipOffset(unittest.TestCase):
    """Equation 1b: clip weights + offset activations."""

    def _check(self, w_s, a_s, n):
        w_u = clip_map(w_s, n)
        a_u = offset_map(a_s, n)
        result = recover_clip_offset(w_u, a_u, w_s, a_s, n)
        self.assertEqual(result, w_s * a_s,
                         f"w_s={w_s}, a_s={a_s}, n={n}")

    def test_all_int4_combinations(self):
        for w in _all_signed(4):
            for a in _all_signed(4):
                self._check(w, a, 4)


class TestRecoverClipClip(unittest.TestCase):
    """Equation 1c: clip weights + clip activations."""

    def _check(self, w_s, a_s, n):
        w_u = clip_map(w_s, n)
        a_u = clip_map(a_s, n)
        result = recover_clip_clip(w_u, a_u, w_s, a_s, n)
        self.assertEqual(result, w_s * a_s,
                         f"w_s={w_s}, a_s={a_s}, n={n}")

    def test_all_int4_combinations(self):
        for w in _all_signed(4):
            for a in _all_signed(4):
                self._check(w, a, 4)


class TestRecoverDispatcher(unittest.TestCase):
    """Generic recover() should route to the correct formula."""

    def test_clip_direct_routing(self):
        w_s, a_s, n = -3, 5, 4
        w_u = clip_map(w_s, n)
        a_u = direct_map(a_s, n)
        result = recover(w_u, a_u, w_s, a_s, n, scheme="clip_direct")
        self.assertEqual(result, w_s * a_s)

    def test_clip_offset_routing(self):
        w_s, a_s, n = -3, -2, 4
        w_u = clip_map(w_s, n)
        a_u = offset_map(a_s, n)
        result = recover(w_u, a_u, w_s, a_s, n, scheme="clip_offset")
        self.assertEqual(result, w_s * a_s)

    def test_clip_clip_routing(self):
        w_s, a_s, n = -3, -2, 4
        w_u = clip_map(w_s, n)
        a_u = clip_map(a_s, n)
        result = recover(w_u, a_u, w_s, a_s, n, scheme="clip_clip")
        self.assertEqual(result, w_s * a_s)

    def test_invalid_scheme_raises(self):
        with self.assertRaises((ValueError, KeyError, AssertionError)):
            recover(0, 0, 0, 0, 4, scheme="bogus")


if __name__ == "__main__":
    unittest.main()
