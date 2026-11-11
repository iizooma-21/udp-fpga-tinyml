# =============================================================================
# mapping.py
# =============================================================================
# This file implements the three mapping schemes described in Section 3.1
# of the UDP paper.
#
# Recap of the problem:
#   DSP blocks work with unsigned numbers internally, but neural network
#   weights and activations are signed (they can be negative).
#   Mapping transforms signed inputs into unsigned representations that
#   the DSP can process, while keeping enough information to recover the
#   correct result afterwards.
#
# The three schemes:
#   1. Direct mapping  — reinterpret signed bits as unsigned, no transformation
#   2. Offset mapping  — add 2^(n-1) to shift everything into positive range
#   3. Clip mapping    — remove the MSB and handle it separately (UDP's choice)
#
# UDP uses clip mapping on weights and direct mapping on activations.
# This combination is proven in the paper to have the lowest total overhead.
# =============================================================================

from utilities.fixed_point import (
    to_unsigned,
    to_signed,
    assert_valid_signed,
    assert_valid_unsigned,
)
from utilities.bit_ops import get_msb, get_bits


# =============================================================================
# DIRECT MAPPING
# =============================================================================

def direct_map(value, bitwidth):
    """
    Direct mapping: reinterpret a signed integer as unsigned without
    any transformation. The bit pattern stays exactly the same —
    we just change how we read it.

    This is the simplest possible mapping. The signed value goes
    straight into the DSP as if it were unsigned. The sign bit becomes
    just another data bit, which causes errors in the multiplication
    result that must be corrected during recovery.

    Example (4-bit):
        direct_map(-4, 4)
        -4 in binary (two's complement) = 1100
        Read as unsigned                = 12
        So: direct_map(-4, 4) → 12

        direct_map(5, 4)
        5 in binary = 0101
        Read as unsigned = 5 (no change for positive numbers)
        So: direct_map(5, 4) → 5

    Args:
        value    : signed integer input (e.g. a neural network activation)
        bitwidth : how many bits wide this number is

    Returns:
        unsigned integer with the same bit pattern as the signed input
    """
    # Validate: input must be a valid signed integer of this bitwidth
    assert_valid_signed(value, bitwidth, name="direct_map input")

    # to_unsigned() reinterprets the bit pattern without changing any bits
    return to_unsigned(value, bitwidth)


def direct_map_inverse(value, bitwidth):
    """
    Inverse of direct mapping: reinterpret an unsigned integer back
    as signed. Again, no transformation — just a change in interpretation.

    Example (4-bit):
        direct_map_inverse(12, 4)
        12 in binary = 1100
        Read as signed two's complement = -4
        So: direct_map_inverse(12, 4) → -4

    Args:
        value    : unsigned integer (output of direct_map)
        bitwidth : how many bits wide this number is

    Returns:
        signed integer with the same bit pattern
    """
    assert_valid_unsigned(value, bitwidth, name="direct_map_inverse input")

    return to_signed(value, bitwidth)


# =============================================================================
# OFFSET MAPPING
# =============================================================================

def offset_map(value, bitwidth):
    """
    Offset mapping: shift the signed range up by 2^(n-1) so that
    all values become non-negative (unsigned).

    The signed range [-2^(n-1), 2^(n-1) - 1] gets shifted to [0, 2^n - 1].

    In hardware, adding 2^(n-1) is equivalent to flipping just the MSB
    (the sign bit), which is a very cheap XOR operation.

    Example (4-bit):
        Signed range is -8 to +7
        Offset = 2^(4-1) = 8

        offset_map(-8, 4) → -8 + 8 = 0    (minimum maps to 0)
        offset_map(-4, 4) → -4 + 8 = 4
        offset_map( 0, 4) →  0 + 8 = 8
        offset_map( 7, 4) →  7 + 8 = 15   (maximum maps to 15)

    Args:
        value    : signed integer input
        bitwidth : how many bits wide this number is

    Returns:
        unsigned integer in range [0, 2^bitwidth - 1]
    """
    assert_valid_signed(value, bitwidth, name="offset_map input")

    # The offset is 2^(n-1), which is the value of the sign bit position
    offset = 1 << (bitwidth - 1)

    # Add the offset to shift the entire range up
    result = value + offset

    # The result should now be a valid unsigned integer of this bitwidth
    assert_valid_unsigned(result, bitwidth, name="offset_map output")

    return result


def offset_map_inverse(value, bitwidth):
    """
    Inverse of offset mapping: subtract the offset to recover the
    original signed value.

    Example (4-bit):
        offset_map_inverse(4, 4) → 4 - 8 = -4
        offset_map_inverse(0, 4) → 0 - 8 = -8
        offset_map_inverse(15, 4) → 15 - 8 = 7

    Args:
        value    : unsigned integer (output of offset_map)
        bitwidth : how many bits wide this number is

    Returns:
        original signed integer
    """
    assert_valid_unsigned(value, bitwidth, name="offset_map_inverse input")

    offset = 1 << (bitwidth - 1)
    return value - offset


# =============================================================================
# CLIP MAPPING
# =============================================================================

def clip_map(value, bitwidth):
    """
    Clip mapping: UDP's core innovation.

    Instead of including the sign bit in the data sent to the DSP,
    we REMOVE it (clip it off) and handle it separately during recovery.

    This does two things:
        1. Converts the signed input to unsigned (so DSP can process it)
        2. REDUCES the bitwidth by 1 (from n bits to n-1 bits)

    The bitwidth reduction is the key benefit — it's what allows more
    values to be packed into a single DSP block.

    The mathematical formula from the paper (equation in Section 3.1):
        x_u = x_s + 2^(n-1) * [x_s < 0]

    Breaking this down:
        - If x_s >= 0: x_u = x_s              (positive: keep as is)
        - If x_s <  0: x_u = x_s + 2^(n-1)   (negative: add 2^(n-1))

    What this does in terms of bits:
        For positive numbers: the MSB is already 0, removing it changes nothing
        For negative numbers: adding 2^(n-1) is like flipping the sign bit,
        giving us the magnitude portion only

    Example (4-bit input → 3-bit output):
        clip_map(5, 4):
            5 = 0101 in binary, MSB = 0 (positive)
            x_u = 5 + 0 = 5
            Result as 3 bits: 101 = 5
            clip_map(5, 4) → 5

        clip_map(-4, 4):
            -4 = 1100 in binary, MSB = 1 (negative)
            x_u = -4 + 2^3 = -4 + 8 = 4
            Result as 3 bits: 100 = 4
            clip_map(-4, 4) → 4

        clip_map(-1, 4):
            -1 = 1111 in binary, MSB = 1 (negative)
            x_u = -1 + 8 = 7
            Result as 3 bits: 111 = 7
            clip_map(-1, 4) → 7

    Notice: positive 5 and negative -3 both map to different 3-bit values.
    The MSB information is LOST in the output — that's why we need to
    keep track of the original sign separately during recovery.

    Args:
        value    : signed integer input (e.g. a neural network weight)
        bitwidth : how many bits wide the INPUT is (output is bitwidth-1)

    Returns:
        unsigned integer of (bitwidth - 1) bits
    """
    assert_valid_signed(value, bitwidth, name="clip_map input")

    # Get the sign bit (MSB): 1 if negative, 0 if positive
    # We use the original signed value for this check
    is_negative = 1 if value < 0 else 0

    # Apply the formula: x_u = x_s + 2^(n-1) * [x_s < 0]
    x_u = value + (1 << (bitwidth - 1)) * is_negative

    # The result should fit in (bitwidth - 1) bits
    # Let's verify this is true
    clipped_bitwidth = bitwidth - 1
    assert_valid_unsigned(x_u, clipped_bitwidth, name="clip_map output")

    return x_u


def clip_map_inverse(value, original_bitwidth, was_negative):
    """
    Inverse of clip mapping: recover the original signed value.

    Since clip mapping loses the sign bit, we need to be told separately
    whether the original value was negative. In the full UDP pipeline,
    this sign information is tracked alongside the mapped value.

    Formula (reverse of clip_map):
        - If was_negative = 0: x_s = x_u              (just return as signed)
        - If was_negative = 1: x_s = x_u - 2^(n-1)   (subtract what we added)

    Example (4-bit original):
        clip_map_inverse(4, 4, was_negative=True)
        → 4 - 8 = -4   ✓ (we recover the original -4)

        clip_map_inverse(5, 4, was_negative=False)
        → 5             ✓ (positive 5, no change needed)

    Args:
        value            : unsigned integer output of clip_map (bitwidth-1 bits)
        original_bitwidth: bitwidth of the ORIGINAL signed input
        was_negative     : True/1 if the original value was negative, False/0 if not

    Returns:
        the original signed integer
    """
    clipped_bitwidth = original_bitwidth - 1
    assert_valid_unsigned(value, clipped_bitwidth, name="clip_map_inverse input")

    if was_negative:
        return value - (1 << (original_bitwidth - 1))
    else:
        return value


# =============================================================================
# CONVENIENCE FUNCTION: apply mapping by name
# =============================================================================

def apply_map(value, bitwidth, scheme):
    """
    Apply a mapping scheme by name. Useful for running comparisons
    between all three schemes without changing code structure.

    Args:
        value    : signed integer input
        bitwidth : how many bits wide this number is
        scheme   : one of "direct", "offset", "clip"

    Returns:
        tuple of (mapped_value, metadata)
        metadata is needed by the inverse function:
            - direct: None (no metadata needed)
            - offset: None (no metadata needed)
            - clip:   the original sign (True if negative)
    """
    if scheme == "direct":
        return direct_map(value, bitwidth), None

    elif scheme == "offset":
        return offset_map(value, bitwidth), None

    elif scheme == "clip":
        was_negative = value < 0
        return clip_map(value, bitwidth), was_negative

    else:
        raise ValueError(f"Unknown mapping scheme '{scheme}'. Choose: direct, offset, clip")


def apply_map_inverse(value, bitwidth, scheme, metadata=None):
    """
    Apply the inverse of a mapping scheme by name.

    Args:
        value    : mapped unsigned integer
        bitwidth : bitwidth of the ORIGINAL signed input
        scheme   : one of "direct", "offset", "clip"
        metadata : the metadata returned by apply_map (needed for clip)

    Returns:
        the original signed integer
    """
    if scheme == "direct":
        return direct_map_inverse(value, bitwidth)

    elif scheme == "offset":
        return offset_map_inverse(value, bitwidth)

    elif scheme == "clip":
        if metadata is None:
            raise ValueError("clip inverse requires metadata (was_negative flag)")
        return clip_map_inverse(value, bitwidth, was_negative=metadata)

    else:
        raise ValueError(f"Unknown mapping scheme '{scheme}'. Choose: direct, offset, clip")


# =============================================================================
# Quick self-test — run this file directly to verify everything works
# =============================================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Testing mapping.py")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # Direct mapping tests
    # -------------------------------------------------------------------------
    print("\n--- Direct Mapping ---")
    print("Positive numbers should be unchanged.")
    print("Negative numbers become their unsigned bit pattern.\n")

    test_cases = [
        (-8, 4, 8),   # most negative 4-bit value
        (-4, 4, 12),
        (-1, 4, 15),
        ( 0, 4, 0),
        ( 5, 4, 5),
        ( 7, 4, 7),   # most positive 4-bit value
    ]

    for signed_val, bw, expected in test_cases:
        mapped = direct_map(signed_val, bw)
        recovered = direct_map_inverse(mapped, bw)
        status = "OK" if mapped == expected and recovered == signed_val else "FAIL"
        print(f"  direct_map({signed_val:3d}, {bw}) = {mapped:2d}  "
              f"| inverse → {recovered:3d}  [{status}]")
        assert mapped == expected
        assert recovered == signed_val

    # -------------------------------------------------------------------------
    # Offset mapping tests
    # -------------------------------------------------------------------------
    print("\n--- Offset Mapping ---")
    print("All values shifted up by 2^(n-1) = 8 for 4-bit.")
    print("Range [-8, +7] becomes [0, 15].\n")

    test_cases = [
        (-8, 4, 0),   # -8 + 8 = 0
        (-4, 4, 4),   # -4 + 8 = 4
        (-1, 4, 7),   # -1 + 8 = 7
        ( 0, 4, 8),   #  0 + 8 = 8
        ( 5, 4, 13),  #  5 + 8 = 13
        ( 7, 4, 15),  #  7 + 8 = 15
    ]

    for signed_val, bw, expected in test_cases:
        mapped = offset_map(signed_val, bw)
        recovered = offset_map_inverse(mapped, bw)
        status = "OK" if mapped == expected and recovered == signed_val else "FAIL"
        print(f"  offset_map({signed_val:3d}, {bw}) = {mapped:2d}  "
              f"| inverse → {recovered:3d}  [{status}]")
        assert mapped == expected
        assert recovered == signed_val

    # -------------------------------------------------------------------------
    # Clip mapping tests
    # -------------------------------------------------------------------------
    print("\n--- Clip Mapping (UDP's choice) ---")
    print("MSB is removed. Output is (bitwidth - 1) = 3 bits for 4-bit input.")
    print("Sign information is tracked separately as metadata.\n")

    test_cases = [
        # (signed_val, bitwidth, expected_mapped)
        (-8, 4, 0),   # -8 + 8 = 0  (3-bit)
        (-4, 4, 4),   # -4 + 8 = 4  (3-bit)
        (-1, 4, 7),   # -1 + 8 = 7  (3-bit)
        ( 0, 4, 0),   #  0 + 0 = 0  (3-bit)
        ( 5, 4, 5),   #  5 + 0 = 5  (3-bit)
        ( 7, 4, 7),   #  7 + 0 = 7  (3-bit)
    ]

    for signed_val, bw, expected in test_cases:
        mapped, metadata = apply_map(signed_val, bw, scheme="clip")
        recovered = apply_map_inverse(mapped, bw, scheme="clip", metadata=metadata)
        status = "OK" if mapped == expected and recovered == signed_val else "FAIL"
        print(f"  clip_map({signed_val:3d}, {bw}) = {mapped}  "
              f"(was_negative={metadata})  "
              f"| inverse → {recovered:3d}  [{status}]")
        assert mapped == expected, f"Expected {expected}, got {mapped}"
        assert recovered == signed_val

    # -------------------------------------------------------------------------
    # Bitwidth reduction check
    # -------------------------------------------------------------------------
    print("\n--- Bitwidth Reduction (key UDP property) ---")
    print("Clip mapping outputs (bitwidth-1) bits.")
    print("This is what allows more values to fit in one DSP.\n")

    import math
    for val in [-4, -1, 0, 3, 7]:
        mapped, _ = apply_map(val, 4, scheme="clip")
        bits_needed = math.ceil(math.log2(mapped + 1)) if mapped > 0 else 1
        print(f"  clip_map({val:3d}, 4) = {mapped}  fits in 3 bits: "
              f"{'YES' if mapped <= 7 else 'NO'}")
        assert mapped <= 7, f"Clip map output {mapped} exceeds 3-bit range!"

    # -------------------------------------------------------------------------
    # Compare all three schemes side by side
    # -------------------------------------------------------------------------
    print("\n--- All Three Schemes Side by Side (4-bit) ---\n")
    print(f"  {'value':>6} | {'direct':>8} | {'offset':>8} | {'clip':>8} | {'sign':>6}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")

    for val in range(-8, 8):
        d, _  = apply_map(val, 4, "direct")
        o, _  = apply_map(val, 4, "offset")
        c, meta = apply_map(val, 4, "clip")
        sign = "neg" if meta else "pos"
        print(f"  {val:>6} | {d:>8} | {o:>8} | {c:>8} | {sign:>6}")

    print("\nAll tests passed!")
