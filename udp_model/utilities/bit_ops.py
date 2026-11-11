# =============================================================================
# bit_ops.py
# =============================================================================
# This file is responsible for one thing: manipulating bits structurally.
#
# The difference from fixed_point.py:
#   fixed_point.py asks "what does this number mean?" (signed vs unsigned)
#   bit_ops.py asks  "how do I move/extract/combine these bits?"
#
# These operations mirror exactly what hardware does at the gate level.
# When you read the UDP paper and see notation like X[a:b] (bit slice),
# or see values being shifted and concatenated during packing — this file
# is what implements all of that in Python.
# =============================================================================

from utilities.fixed_point import to_unsigned


def get_bit(value, position):
    """
    Extract a single bit at a given position.

    Position 0 is the least significant bit (rightmost).
    Position n-1 is the most significant bit for an n-bit number.

    Example:
        value = 0b1010 = 10
        get_bit(10, 0) → 0   (rightmost bit)
        get_bit(10, 1) → 1
        get_bit(10, 2) → 0
        get_bit(10, 3) → 1   (leftmost bit)

    Args:
        value    : the integer to extract from
        position : which bit to extract (0 = LSB)

    Returns:
        0 or 1
    """
    return (value >> position) & 1


def get_bits(value, high, low):
    """
    Extract a range of bits from a value. This is the X[high:low] notation
    used throughout the UDP paper.

    The range is INCLUSIVE on both ends: bits from position 'low' up to
    and including position 'high' are returned.

    Example:
        value = 0b11001010 = 202
        get_bits(202, 5, 2) extracts bits 5,4,3,2 → 0b0010 = 2

        Visually:
          bit:  7 6 5 4 3 2 1 0
          val:  1 1 0 0 1 0 1 0
                    ^-------^       ← extracting bits 5 down to 2
          result: 0 0 1 0 = 2

    Args:
        value : the integer to extract from
        high  : the highest bit position to include (inclusive)
        low   : the lowest bit position to include (inclusive)

    Returns:
        The extracted bits as an unsigned integer
    """
    # How many bits are we extracting?
    num_bits = high - low + 1

    # Create a mask of that many 1s
    # Example: num_bits=4 → mask = 0b1111 = 15
    mask = (1 << num_bits) - 1

    # Shift value right so 'low' becomes bit 0, then apply mask
    return (value >> low) & mask


def set_bit(value, position, bit):
    """
    Set a single bit at a given position to 0 or 1.

    Example:
        set_bit(0b1010, 2, 1) → 0b1110  (set bit 2 to 1)
        set_bit(0b1010, 1, 0) → 0b1000  (set bit 1 to 0)

    Args:
        value    : the original integer
        position : which bit position to modify
        bit      : the value to set (must be 0 or 1)

    Returns:
        The modified integer
    """
    if bit == 1:
        # OR with a 1 at the target position to force that bit to 1
        return value | (1 << position)
    else:
        # AND with NOT of a 1 at target position to force that bit to 0
        return value & ~(1 << position)


def shift_left(value, amount, bitwidth):
    """
    Shift a value left by 'amount' bits, keeping only 'bitwidth' bits.

    In Python, left shift never overflows because integers grow freely.
    This function enforces the hardware behavior by masking off any bits
    that would go beyond 'bitwidth'.

    Example (8-bit):
        shift_left(0b00001111, 3, 8)
        → 0b01111000 = 120   (shifted left 3, still fits in 8 bits)

        shift_left(0b10000000, 2, 8)
        → 0b00000000 = 0     (bits shifted out are lost)

    Args:
        value    : the integer to shift
        amount   : how many positions to shift left
        bitwidth : enforce this many bits (discard overflow)

    Returns:
        The shifted value masked to 'bitwidth' bits
    """
    mask = (1 << bitwidth) - 1
    return (value << amount) & mask


def shift_right(value, amount):
    """
    Shift a value right by 'amount' bits (logical shift — fills with 0s).

    This is a simple wrapper for clarity. Since we represent everything
    as unsigned integers internally, this is always a logical (zero-fill)
    right shift, which matches hardware behavior for unsigned values.

    Example:
        shift_right(0b11001100, 3) → 0b00011001 = 25

    Args:
        value  : the integer to shift (treat as unsigned)
        amount : how many positions to shift right

    Returns:
        The shifted value
    """
    return value >> amount


def concatenate_bits(high_value, low_value, low_bitwidth):
    """
    Concatenate two bit patterns: place high_value above low_value.

    This is like wiring two signals together in hardware — you take the
    bits of high_value and put them immediately to the left of low_value.

    You must specify how wide low_value is (low_bitwidth), so we know
    how far left to shift high_value before combining.

    Example:
        high_value  = 0b1010   (4 bits)
        low_value   = 0b0011   (4 bits, so low_bitwidth = 4)
        result      = 0b10100011   (8 bits)

        concatenate_bits(0b1010, 0b0011, 4) → 0b10100011 = 163

    Args:
        high_value    : the bits that go on the LEFT (more significant)
        low_value     : the bits that go on the RIGHT (less significant)
        low_bitwidth  : the bitwidth of low_value

    Returns:
        The concatenated integer
    """
    return (high_value << low_bitwidth) | to_unsigned(low_value, low_bitwidth)


def insert_zeros(value, value_bitwidth, gap):
    """
    Insert 'gap' zero bits below a value (i.e. shift it left by 'gap').

    This is used during packing: each data element gets zero bits inserted
    between it and the next element so that multiplication results don't
    overlap in the DSP output.

    Example:
        value = 0b1011  (4 bits)
        gap   = 3
        result = 0b1011_000  (value shifted left by 3, 3 zeros inserted below)

        insert_zeros(0b1011, 4, 3) → 0b1011000 = 88

    Args:
        value         : the integer to shift
        value_bitwidth: the bitwidth of value (for documentation clarity)
        gap           : how many zero bits to insert below the value

    Returns:
        value shifted left by gap positions
    """
    # This is just a left shift — inserting zeros below
    return value << gap


def get_msb(value, bitwidth):
    """
    Get the most significant bit (MSB) of a value of known bitwidth.

    The MSB is the sign bit in two's complement representation.
    In the UDP paper, many decisions depend on whether the MSB is 0 or 1
    (i.e. whether the number is positive or negative).

    Example (4-bit):
        get_msb(0b1010, 4) → 1   (negative number)
        get_msb(0b0110, 4) → 0   (positive number)

    Args:
        value    : the integer to inspect
        bitwidth : how wide the number is (determines MSB position)

    Returns:
        0 or 1
    """
    return get_bit(value, bitwidth - 1)


def count_bits(value):
    """
    Count how many bits are needed to represent a value (ignoring sign).

    Useful for sanity checking that packed values don't exceed DSP port widths.

    Example:
        count_bits(0)   → 1
        count_bits(7)   → 3   (0b111)
        count_bits(8)   → 4   (0b1000)
        count_bits(255) → 8

    Args:
        value : a non-negative integer

    Returns:
        Number of bits required
    """
    if value == 0:
        return 1
    return value.bit_length()


# =============================================================================
# Quick self-test — run this file directly to verify everything works
# =============================================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Testing bit_ops.py")
    print("=" * 50)

    # --- get_bit ---
    print("\n--- get_bit ---")
    val = 0b1010  # = 10
    print(f"value = {bin(val)}")
    for pos in range(4):
        b = get_bit(val, pos)
        print(f"  bit[{pos}] = {b}")
    assert get_bit(val, 0) == 0
    assert get_bit(val, 1) == 1
    assert get_bit(val, 2) == 0
    assert get_bit(val, 3) == 1

    # --- get_bits ---
    print("\n--- get_bits ---")
    val = 0b11001010  # = 202
    result = get_bits(val, 5, 2)
    print(f"get_bits({bin(val)}, high=5, low=2) = {bin(result)} = {result}  (expected 2)")
    assert result == 2

    result = get_bits(val, 7, 4)
    print(f"get_bits({bin(val)}, high=7, low=4) = {bin(result)} = {result}  (expected 12)")
    assert result == 12

    # --- concatenate_bits ---
    print("\n--- concatenate_bits ---")
    high = 0b1010   # 4 bits
    low  = 0b0011   # 4 bits
    result = concatenate_bits(high, low, 4)
    print(f"concatenate({bin(high)}, {bin(low)}, low_bitwidth=4) = {bin(result)} = {result}  (expected 163)")
    assert result == 0b10100011

    # --- insert_zeros ---
    print("\n--- insert_zeros ---")
    val = 0b1011
    result = insert_zeros(val, 4, 3)
    print(f"insert_zeros({bin(val)}, gap=3) = {bin(result)} = {result}  (expected 88)")
    assert result == 0b1011000

    # --- get_msb ---
    print("\n--- get_msb ---")
    print(f"get_msb(0b1010, 4) = {get_msb(0b1010, 4)}  (expected 1, negative)")
    print(f"get_msb(0b0110, 4) = {get_msb(0b0110, 4)}  (expected 0, positive)")
    assert get_msb(0b1010, 4) == 1
    assert get_msb(0b0110, 4) == 0

    # --- shift_left with overflow ---
    print("\n--- shift_left (with bitwidth enforcement) ---")
    result = shift_left(0b10000000, 2, 8)
    print(f"shift_left(0b10000000, 2, 8) = {bin(result)} = {result}  (expected 0, overflow discarded)")
    assert result == 0

    print("\nAll tests passed!")
