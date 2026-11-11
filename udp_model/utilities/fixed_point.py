# =============================================================================
# fixed_point.py
# =============================================================================
# This file is responsible for one thing: making Python integers behave like
# real hardware integers of a specific bitwidth.
#
# The core problem:
#   In Python, integers can be any size. There is no overflow, no wrapping,
#   no concept of "this is a 4-bit number". Hardware doesn't work that way.
#
#   Example: in hardware, a 4-bit signed integer can only hold -8 to +7.
#            If you go above +7, it wraps around. Python won't do that on its own.
#
# This file gives us functions to enforce that hardware-like behavior.
# =============================================================================


def to_unsigned(value, bitwidth):
    """
    Reinterpret a signed integer as its unsigned bit pattern.

    In hardware, the same bits can mean different things depending on whether
    you treat them as signed or unsigned. This function takes a value and
    returns what those bits would look like if read as unsigned.

    Example (4-bit):
        -4 in signed two's complement is the bit pattern 1100
        1100 read as unsigned is 12
        So: to_unsigned(-4, 4) → 12

    Args:
        value    : the integer to convert (can be signed or unsigned)
        bitwidth : how many bits wide this number is (e.g. 4 for int4)

    Returns:
        An integer in the range [0, 2^bitwidth - 1]
    """

    # Step 1: create a mask that keeps only the lowest 'bitwidth' bits
    # For bitwidth=4: mask = 0b1111 = 15
    # For bitwidth=8: mask = 0b11111111 = 255
    mask = (1 << bitwidth) - 1

    # Step 2: AND the value with the mask to discard any higher bits
    # This is equivalent to "take only the last bitwidth bits"
    # Example: -4 in Python is ...11111100 (infinite 1s to the left)
    #          AND with 0b1111 gives 0b1100 = 12
    return value & mask


def to_signed(value, bitwidth):
    """
    Reinterpret an unsigned bit pattern as a signed two's complement integer.

    This is the reverse of to_unsigned(). Given a raw bit pattern (stored as
    an unsigned Python int), interpret it as a signed hardware integer.

    In two's complement, the most significant bit (MSB) is the sign bit.
    If the MSB is 1, the number is negative.

    Example (4-bit):
        12 in unsigned is the bit pattern 1100
        The MSB (bit 3) is 1, so this is negative in signed
        1100 in two's complement = -4
        So: to_signed(12, 4) → -4

    Args:
        value    : unsigned integer to reinterpret (range [0, 2^bitwidth - 1])
        bitwidth : how many bits wide this number is

    Returns:
        A signed integer in the range [-2^(bitwidth-1), 2^(bitwidth-1) - 1]
    """

    # First, make sure we're only looking at the relevant bits
    value = to_unsigned(value, bitwidth)

    # The MSB position is at index (bitwidth - 1)
    # If the MSB is 1, the number is negative in two's complement
    # We check this by seeing if value >= 2^(bitwidth-1)
    # Example: for bitwidth=4, 2^(bitwidth-1) = 8
    #          values 8-15 have MSB=1, so they are negative
    if value >= (1 << (bitwidth - 1)):
        # To convert: subtract 2^bitwidth
        # Example: 12 - 16 = -4
        value = value - (1 << bitwidth)

    return value


def clamp_unsigned(value, bitwidth):
    """
    Clamp a value to fit within an unsigned integer of the given bitwidth.

    Unlike to_unsigned() which wraps around (reinterprets bits), this function
    simply clips the value to the valid range. Useful for catching overflow
    bugs — if clamping changes the value, something went wrong upstream.

    Valid range for unsigned n-bit: [0, 2^n - 1]

    Example (4-bit unsigned range is 0 to 15):
        clamp_unsigned(20, 4) → 15   (clipped at max)
        clamp_unsigned(-1, 4) → 0    (clipped at min)
        clamp_unsigned(10, 4) → 10   (already in range, unchanged)

    Args:
        value    : the integer to clamp
        bitwidth : how many bits wide this number is

    Returns:
        value clamped to [0, 2^bitwidth - 1]
    """
    min_val = 0
    max_val = (1 << bitwidth) - 1
    return max(min_val, min(max_val, value))


def clamp_signed(value, bitwidth):
    """
    Clamp a value to fit within a signed integer of the given bitwidth.

    Valid range for signed n-bit two's complement: [-2^(n-1), 2^(n-1) - 1]

    Example (4-bit signed range is -8 to +7):
        clamp_signed(10, 4)  → 7    (clipped at max)
        clamp_signed(-10, 4) → -8   (clipped at min)
        clamp_signed(-3, 4)  → -3   (already in range, unchanged)

    Args:
        value    : the integer to clamp
        bitwidth : how many bits wide this number is

    Returns:
        value clamped to [-2^(bitwidth-1), 2^(bitwidth-1) - 1]
    """
    min_val = -(1 << (bitwidth - 1))
    max_val =  (1 << (bitwidth - 1)) - 1
    return max(min_val, min(max_val, value))


def is_valid_unsigned(value, bitwidth):
    """
    Check whether a value fits in an unsigned integer of the given bitwidth.

    Args:
        value    : the integer to check
        bitwidth : how many bits wide this number is

    Returns:
        True if value is in [0, 2^bitwidth - 1], False otherwise
    """
    return 0 <= value <= (1 << bitwidth) - 1


def is_valid_signed(value, bitwidth):
    """
    Check whether a value fits in a signed integer of the given bitwidth.

    Args:
        value    : the integer to check
        bitwidth : how many bits wide this number is

    Returns:
        True if value is in [-2^(bitwidth-1), 2^(bitwidth-1) - 1], False otherwise
    """
    return -(1 << (bitwidth - 1)) <= value <= (1 << (bitwidth - 1)) - 1


def assert_valid_unsigned(value, bitwidth, name="value"):
    """
    Assert that a value fits in an unsigned integer of the given bitwidth.
    Raises a clear error if it doesn't. Use this at the start of functions
    to catch bad inputs early.

    Args:
        value    : the integer to check
        bitwidth : how many bits wide this number is
        name     : a label for the value, used in the error message
    """
    if not is_valid_unsigned(value, bitwidth):
        raise ValueError(
            f"{name} = {value} does not fit in a {bitwidth}-bit unsigned integer "
            f"(valid range: 0 to {(1 << bitwidth) - 1})"
        )


def assert_valid_signed(value, bitwidth, name="value"):
    """
    Assert that a value fits in a signed integer of the given bitwidth.
    Raises a clear error if it doesn't.

    Args:
        value    : the integer to check
        bitwidth : how many bits wide this number is
        name     : a label for the value, used in the error message
    """
    if not is_valid_signed(value, bitwidth):
        raise ValueError(
            f"{name} = {value} does not fit in a {bitwidth}-bit signed integer "
            f"(valid range: {-(1 << (bitwidth - 1))} to {(1 << (bitwidth - 1)) - 1})"
        )


# =============================================================================
# Quick self-test — run this file directly to verify everything works
# =============================================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Testing fixed_point.py")
    print("=" * 50)

    # --- to_unsigned ---
    print("\n--- to_unsigned ---")
    # -4 as a 4-bit signed integer has bit pattern 1100 = 12 unsigned
    result = to_unsigned(-4, 4)
    print(f"to_unsigned(-4, 4)  = {result}  (expected 12)")
    assert result == 12

    # +5 stays as 5 (bit pattern 0101, no sign bit set)
    result = to_unsigned(5, 4)
    print(f"to_unsigned(5, 4)   = {result}   (expected 5)")
    assert result == 5

    # -1 as 4-bit is 1111 = 15 unsigned
    result = to_unsigned(-1, 4)
    print(f"to_unsigned(-1, 4)  = {result}  (expected 15)")
    assert result == 15

    # --- to_signed ---
    print("\n--- to_signed ---")
    # 12 as a 4-bit pattern (1100) is -4 in signed two's complement
    result = to_signed(12, 4)
    print(f"to_signed(12, 4)    = {result}   (expected -4)")
    assert result == -4

    # 5 as a 4-bit pattern (0101), MSB is 0, so still positive
    result = to_signed(5, 4)
    print(f"to_signed(5, 4)     = {result}    (expected 5)")
    assert result == 5

    # 15 as a 4-bit pattern (1111) is -1 in signed
    result = to_signed(15, 4)
    print(f"to_signed(15, 4)    = {result}   (expected -1)")
    assert result == -1

    # --- round trip ---
    print("\n--- round trip (signed → unsigned → signed) ---")
    for val in [-8, -4, -1, 0, 3, 7]:
        recovered = to_signed(to_unsigned(val, 4), 4)
        print(f"  {val:3d} → unsigned → signed → {recovered:3d}  {'OK' if recovered == val else 'FAIL'}")
        assert recovered == val

    # --- clamp ---
    print("\n--- clamp_unsigned ---")
    print(f"clamp_unsigned(20, 4)  = {clamp_unsigned(20, 4)}   (expected 15)")
    print(f"clamp_unsigned(-1, 4)  = {clamp_unsigned(-1, 4)}    (expected 0)")
    print(f"clamp_unsigned(10, 4)  = {clamp_unsigned(10, 4)}   (expected 10)")

    print("\n--- clamp_signed ---")
    print(f"clamp_signed(10, 4)    = {clamp_signed(10, 4)}    (expected 7)")
    print(f"clamp_signed(-10, 4)   = {clamp_signed(-10, 4)}   (expected -8)")
    print(f"clamp_signed(-3, 4)    = {clamp_signed(-3, 4)}   (expected -3)")

    # --- validation ---
    print("\n--- assert_valid_signed (should raise error) ---")
    try:
        assert_valid_signed(10, 4, name="test_value")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nAll tests passed!")
