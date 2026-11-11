"""
tests/test_mapping.py
=====================
Unit tests for core/mapping.py.

Covers direct_map, direct_map_inverse, offset_map, offset_map_inverse,
clip_map, clip_map_inverse, and the generic apply_map / apply_map_inverse
wrappers for all three schemes.
"""

import unittest
from core.mapping import (
    direct_map, direct_map_inverse,
    offset_map, offset_map_inverse,
    clip_map,   clip_map_inverse,
    apply_map,  apply_map_inverse,
)


class TestDirectMap(unittest.TestCase):

    def test_positive_unchanged(self):
        """Positive signed values have identical bit patterns as unsigned."""
        for n in [4, 8]:
            max_pos = 2 ** (n - 1) - 1
            for v in range(0, max_pos + 1):
                self.assertEqual(direct_map(v, n), v)

    def test_negative_wraps(self):
        """Negative values are reinterpreted as their unsigned 2s-complement."""
        self.assertEqual(direct_map(-1, 4),  15)   # 0b1111
        self.assertEqual(direct_map(-8, 4),   8)   # 0b1000
        self.assertEqual(direct_map(-4, 4),  12)   # 0b1100
        self.assertEqual(direct_map(-128, 8), 128)

    def test_roundtrip(self):
        """direct_map_inverse undoes direct_map for all 4-bit values."""
        for v in range(-8, 8):
            self.assertEqual(direct_map_inverse(direct_map(v, 4), 4), v)

    def test_inverse_all_8bit(self):
        """direct_map_inverse is correct for all 8-bit signed values."""
        for v in range(-128, 128):
            u = direct_map(v, 8)
            self.assertEqual(direct_map_inverse(u, 8), v)

    def test_zero(self):
        self.assertEqual(direct_map(0, 4), 0)
        self.assertEqual(direct_map_inverse(0, 4), 0)


class TestOffsetMap(unittest.TestCase):

    def test_shifts_to_non_negative(self):
        """offset_map produces values in [0, 2^n - 1]."""
        for n in [4, 8]:
            half = 2 ** (n - 1)
            for v in range(-half, half):
                u = offset_map(v, n)
                self.assertGreaterEqual(u, 0)
                self.assertLessEqual(u, 2 ** n - 1)

    def test_known_values(self):
        self.assertEqual(offset_map(-8, 4),  0)
        self.assertEqual(offset_map(-1, 4),  7)
        self.assertEqual(offset_map( 0, 4),  8)
        self.assertEqual(offset_map( 7, 4), 15)

    def test_roundtrip(self):
        for v in range(-8, 8):
            self.assertEqual(offset_map_inverse(offset_map(v, 4), 4), v)


class TestClipMap(unittest.TestCase):

    def test_positive_unchanged(self):
        """clip_map leaves non-negative values unchanged."""
        for v in range(0, 8):        # [0, 2^(4-1) - 1]
            self.assertEqual(clip_map(v, 4), v)

    def test_negative_adds_half(self):
        """clip_map(-v, n) = -v + 2^(n-1), i.e. removes sign via offset."""
        self.assertEqual(clip_map(-1, 4),  7)   # -1 + 8
        self.assertEqual(clip_map(-8, 4),  0)   # -8 + 8
        self.assertEqual(clip_map(-4, 4),  4)

    def test_output_fits_in_n_minus_1_bits(self):
        """Result is always in [0, 2^(n-1) - 1]."""
        for n in [4, 8]:
            for v in range(-(2**(n-1)), 2**(n-1)):
                u = clip_map(v, n)
                self.assertGreaterEqual(u, 0)
                self.assertLessEqual(u, 2 ** (n - 1) - 1)

    def test_roundtrip(self):
        for v in range(-8, 8):
            self.assertEqual(clip_map_inverse(clip_map(v, 4), 4, was_negative=(v < 0)), v)

    def test_inverse_all_8bit(self):
        for v in range(-128, 128):
            self.assertEqual(clip_map_inverse(clip_map(v, 8), 8, was_negative=(v < 0)), v)


class TestApplyMap(unittest.TestCase):

    def test_direct_scheme(self):
        for v in range(-8, 8):
            u, meta = apply_map(v, 4, scheme="direct")
            self.assertEqual(u, direct_map(v, 4))
            self.assertEqual(apply_map_inverse(u, 4, scheme="direct", metadata=meta), v)

    def test_offset_scheme(self):
        for v in range(-8, 8):
            u, meta = apply_map(v, 4, scheme="offset")
            self.assertEqual(u, offset_map(v, 4))
            self.assertEqual(apply_map_inverse(u, 4, scheme="offset", metadata=meta), v)

    def test_clip_scheme(self):
        for v in range(-8, 8):
            u, meta = apply_map(v, 4, scheme="clip")
            self.assertEqual(u, clip_map(v, 4))
            self.assertEqual(apply_map_inverse(u, 4, scheme="clip", metadata=meta), v)

    def test_invalid_scheme_raises(self):
        with self.assertRaises((ValueError, KeyError, AssertionError)):
            apply_map(0, 4, scheme="unknown")


if __name__ == "__main__":
    unittest.main()
