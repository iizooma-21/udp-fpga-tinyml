# =============================================================================
# recovery.py
# =============================================================================
# This file implements the three recovery formulas from Section 3.1 of
# the UDP paper (equations 1a, 1b, 1c).
#
# Recap of the problem:
#   After mapping, the DSP computes w_u * a_u instead of w_s * a_s.
#   The result is wrong because we changed the inputs before feeding them in.
#   Recovery corrects that wrong result to get back the true signed answer.
#
# UDP always applies clip mapping to weights.
# The three recovery formulas differ in how activations were mapped:
#   - Clip + Direct (equation 1a) → UDP's choice, lowest total overhead
#   - Clip + Offset (equation 1b) → more expensive than clip + direct
#   - Clip + Clip   (equation 1c) → most complex, only wins in rare cases
#
# The notation [condition] used in the paper means:
#   1 if the condition is true, 0 otherwise
#   We implement this as a simple Python conditional: 1 if condition else 0
# =============================================================================

from utilities.fixed_point import assert_valid_signed, assert_valid_unsigned


# =============================================================================
# HELPER
# =============================================================================

def _iverson(condition):
    """
    Iverson bracket notation from the paper: [condition]
    Returns 1 if condition is True, 0 if False.

    This is used throughout the recovery formulas to apply correction
    terms only when certain conditions hold (e.g. only when w_s < 0).

    Example:
        _iverson(True)      → 1
        _iverson(False)     → 0
        _iverson(-4 < 0)    → 1
        _iverson(5 < 0)     → 0
    """
    return 1 if condition else 0


# =============================================================================
# RECOVERY FORMULA 1A: Clip + Direct (UDP's choice)
# =============================================================================

def recover_clip_direct(w_u, a_u, w_s, a_s, bitwidth):
    """
    Recovery formula for clip mapping on weights + direct mapping on activations.
    This is equation (1a) from the paper and is the formula UDP uses.

    The paper proves this has the lowest total computational overhead
    across all data distributions.

    Full formula:
        w_s * a_s = w_u * a_u
                  - 2^(n-1)   * a_s * [w_s < 0]
                  - 2^n       * w_s * [a_s < 0]
                  - 2^(2n-1)         * [w_s < 0][a_s < 0]

    Breaking down each term:
        Term 1: w_u * a_u
            The raw DSP output. This is our starting point.

        Term 2: - 2^(n-1) * a_s * [w_s < 0]
            Corrects for clipping the weight's MSB.
            Only active when the weight was negative.
            Think of it as: "we removed the weight's sign bit,
            now we correct for the effect that had on the product."

        Term 3: - 2^n * w_s * [a_s < 0]
            Corrects for direct mapping the activation.
            Only active when the activation was negative.
            Direct mapping reinterprets the sign bit as a data bit,
            this term undoes that reinterpretation.

        Term 4: - 2^(2n-1) * [w_s < 0][a_s < 0]
            A cross-correction term.
            Only active when BOTH weight and activation are negative.
            This is a second-order correction needed because terms 2
            and 3 interact when both conditions are true simultaneously.

    Args:
        w_u      : unsigned mapped weight (output of clip_map)
        a_u      : unsigned mapped activation (output of direct_map)
        w_s      : original signed weight (needed to know if w was negative)
        a_s      : original signed activation (needed to know if a was negative)
        bitwidth : bitwidth of the original signed inputs (n in the formula)

    Returns:
        The correct signed product w_s * a_s
    """
    # Validate inputs
    assert_valid_unsigned(w_u, bitwidth - 1, name="w_u (clip mapped weight)")
    assert_valid_unsigned(a_u, bitwidth,     name="a_u (direct mapped activation)")
    assert_valid_signed(w_s, bitwidth,       name="w_s (original signed weight)")
    assert_valid_signed(a_s, bitwidth,       name="a_s (original signed activation)")

    n = bitwidth

    # Term 1: raw DSP multiplication result
    term1 = w_u * a_u

    # Term 2: weight clip correction
    # Only applied when weight was negative: [w_s < 0]
    term2 = (2 ** (n - 1)) * a_s * _iverson(w_s < 0)

    # Term 3: activation direct mapping correction
    # Only applied when activation was negative: [a_s < 0]
    term3 = (2 ** n) * w_s * _iverson(a_s < 0)

    # Term 4: cross-correction for when both are negative
    # Only applied when both weight AND activation were negative
    term4 = (2 ** (2 * n - 1)) * _iverson(w_s < 0) * _iverson(a_s < 0)

    # Apply all corrections
    result = term1 - term2 - term3 - term4

    # Sanity check: result must equal the direct multiplication
    expected = w_s * a_s
    assert result == expected, (
        f"Recovery error! Got {result}, expected {expected}. "
        f"Inputs: w_s={w_s}, a_s={a_s}, w_u={w_u}, a_u={a_u}"
    )

    return result


# =============================================================================
# RECOVERY FORMULA 1B: Clip + Offset
# =============================================================================

def recover_clip_offset(w_u, a_u, w_s, a_s, bitwidth):
    """
    Recovery formula for clip mapping on weights + offset mapping on activations.
    This is equation (1b) from the paper.

    The paper proves this is NEVER cheaper than clip + direct recovery.
    We implement it here for completeness and comparison purposes.

    Full formula:
        w_s * a_s = w_u * a_u
                  - 2^(n-1) * a_s * [w_s < 0]
                  - 2^(n-1) * w_s
                  - 2^(2n-2)       * [w_s < 0]

    Key difference from clip + direct:
        Term 3 here is - 2^(n-1) * w_s (always applied, not conditional)
        Term 4 here is - 2^(2n-2) * [w_s < 0] (simpler cross-term)

        This sounds simpler, but the paper shows the total cost is higher
        because term 3 must always be computed regardless of data.

    Args:
        w_u      : unsigned mapped weight (output of clip_map)
        a_u      : unsigned mapped activation (output of offset_map)
        w_s      : original signed weight
        a_s      : original signed activation
        bitwidth : bitwidth of the original signed inputs

    Returns:
        The correct signed product w_s * a_s
    """
    assert_valid_unsigned(w_u, bitwidth - 1, name="w_u (clip mapped weight)")
    assert_valid_unsigned(a_u, bitwidth,     name="a_u (offset mapped activation)")
    assert_valid_signed(w_s, bitwidth,       name="w_s (original signed weight)")
    assert_valid_signed(a_s, bitwidth,       name="a_s (original signed activation)")

    n = bitwidth

    # Term 1: raw DSP multiplication result
    term1 = w_u * a_u

    # Term 2: weight clip correction (same as in clip + direct)
    term2 = (2 ** (n - 1)) * a_s * _iverson(w_s < 0)

    # Term 3: activation offset correction
    # Note: this is NOT conditional — it always applies
    # This is one reason clip + offset costs more than clip + direct
    term3 = (2 ** (n - 1)) * w_s

    # Term 4: cross-correction
    term4 = (2 ** (2 * n - 2)) * _iverson(w_s < 0)

    result = term1 - term2 - term3 - term4

    # Sanity check
    expected = w_s * a_s
    assert result == expected, (
        f"Recovery error! Got {result}, expected {expected}. "
        f"Inputs: w_s={w_s}, a_s={a_s}, w_u={w_u}, a_u={a_u}"
    )

    return result


# =============================================================================
# RECOVERY FORMULA 1C: Clip + Clip
# =============================================================================

def recover_clip_clip(w_u, a_u, w_s, a_s, bitwidth):
    """
    Recovery formula for clip mapping on BOTH weights and activations.
    This is equation (1c) from the paper.

    Applying clip mapping to both inputs gives the most bitwidth compression
    (both inputs reduced by 1 bit), but the recovery formula becomes more
    complex.

    Full formula:
        w_s * a_s = w_u * a_u
                  - 2^(n-1)   * a_s * [w_s < 0]
                  - 2^(n-1)   * w_s * [a_s < 0]
                  - 2^(2n-2)         * [w_s < 0][a_s < 0]

    This looks very similar to clip + direct (1a), with one key difference:
        Term 3:  clip+direct uses 2^n    (one power higher)
                 clip+clip   uses 2^(n-1) (one power lower)

    The paper shows clip + clip only wins in the specific W4A8 case
    described in Section 6.3.2, due to how it handles a special overflow
    scenario in Section 3.3. For general use, clip + direct is preferred.

    Args:
        w_u      : unsigned mapped weight (output of clip_map, bitwidth-1 bits)
        a_u      : unsigned mapped activation (output of clip_map, bitwidth-1 bits)
        w_s      : original signed weight
        a_s      : original signed activation
        bitwidth : bitwidth of the original signed inputs

    Returns:
        The correct signed product w_s * a_s
    """
    assert_valid_unsigned(w_u, bitwidth - 1, name="w_u (clip mapped weight)")
    assert_valid_unsigned(a_u, bitwidth - 1, name="a_u (clip mapped activation)")
    assert_valid_signed(w_s, bitwidth,       name="w_s (original signed weight)")
    assert_valid_signed(a_s, bitwidth,       name="a_s (original signed activation)")

    n = bitwidth

    # Term 1: raw DSP multiplication result
    term1 = w_u * a_u

    # Term 2: weight clip correction (same as before)
    term2 = (2 ** (n - 1)) * a_s * _iverson(w_s < 0)

    # Term 3: activation clip correction
    # Note: 2^(n-1) here vs 2^n in clip+direct — this is the key difference
    term3 = (2 ** (n - 1)) * w_s * _iverson(a_s < 0)

    # Term 4: cross-correction (same exponent as clip+offset: 2n-2)
    term4 = (2 ** (2 * n - 2)) * _iverson(w_s < 0) * _iverson(a_s < 0)

    result = term1 - term2 - term3 - term4

    # Sanity check
    expected = w_s * a_s
    assert result == expected, (
        f"Recovery error! Got {result}, expected {expected}. "
        f"Inputs: w_s={w_s}, a_s={a_s}, w_u={w_u}, a_u={a_u}"
    )

    return result


# =============================================================================
# CONVENIENCE FUNCTION: recover by scheme name
# =============================================================================

def recover(w_u, a_u, w_s, a_s, bitwidth, scheme):
    """
    Apply a recovery formula by name.

    Args:
        w_u      : unsigned mapped weight
        a_u      : unsigned mapped activation
        w_s      : original signed weight
        a_s      : original signed activation
        bitwidth : bitwidth of original signed inputs
        scheme   : one of "clip_direct", "clip_offset", "clip_clip"

    Returns:
        The correct signed product w_s * a_s
    """
    if scheme == "clip_direct":
        return recover_clip_direct(w_u, a_u, w_s, a_s, bitwidth)
    elif scheme == "clip_offset":
        return recover_clip_offset(w_u, a_u, w_s, a_s, bitwidth)
    elif scheme == "clip_clip":
        return recover_clip_clip(w_u, a_u, w_s, a_s, bitwidth)
    else:
        raise ValueError(
            f"Unknown recovery scheme '{scheme}'. "
            f"Choose: clip_direct, clip_offset, clip_clip"
        )


# =============================================================================
# Quick self-test — run this file directly to verify everything works
# =============================================================================

if __name__ == "__main__":

    from core.mapping import clip_map, direct_map, offset_map

    print("=" * 60)
    print("Testing recovery.py")
    print("=" * 60)

    bitwidth = 4

    # Build the full list of valid 4-bit signed values: -8 to +7
    signed_values = list(range(-(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1)))

    # -------------------------------------------------------------------------
    # Test clip + direct recovery (equation 1a)
    # -------------------------------------------------------------------------
    print("\n--- Clip + Direct Recovery (equation 1a) ---")
    print("Testing all combinations of 4-bit signed weights and activations\n")

    errors = 0
    for w_s in signed_values:
        for a_s in signed_values:
            # Apply mappings
            w_u = clip_map(w_s, bitwidth)
            a_u = direct_map(a_s, bitwidth)

            # Recover
            result = recover_clip_direct(w_u, a_u, w_s, a_s, bitwidth)
            expected = w_s * a_s

            if result != expected:
                print(f"  FAIL: w_s={w_s}, a_s={a_s} → got {result}, expected {expected}")
                errors += 1

    if errors == 0:
        total = len(signed_values) ** 2
        print(f"  All {total} combinations passed!")
    else:
        print(f"  {errors} failures found!")

    # -------------------------------------------------------------------------
    # Test clip + offset recovery (equation 1b)
    # -------------------------------------------------------------------------
    print("\n--- Clip + Offset Recovery (equation 1b) ---")

    errors = 0
    for w_s in signed_values:
        for a_s in signed_values:
            w_u = clip_map(w_s, bitwidth)
            a_u = offset_map(a_s, bitwidth)

            result = recover_clip_offset(w_u, a_u, w_s, a_s, bitwidth)
            expected = w_s * a_s

            if result != expected:
                print(f"  FAIL: w_s={w_s}, a_s={a_s} → got {result}, expected {expected}")
                errors += 1

    if errors == 0:
        total = len(signed_values) ** 2
        print(f"  All {total} combinations passed!")
    else:
        print(f"  {errors} failures found!")

    # -------------------------------------------------------------------------
    # Test clip + clip recovery (equation 1c)
    # -------------------------------------------------------------------------
    print("\n--- Clip + Clip Recovery (equation 1c) ---")

    errors = 0
    for w_s in signed_values:
        for a_s in signed_values:
            w_u = clip_map(w_s, bitwidth)
            a_u = clip_map(a_s, bitwidth)

            result = recover_clip_clip(w_u, a_u, w_s, a_s, bitwidth)
            expected = w_s * a_s

            if result != expected:
                print(f"  FAIL: w_s={w_s}, a_s={a_s} → got {result}, expected {expected}")
                errors += 1

    if errors == 0:
        total = len(signed_values) ** 2
        print(f"  All {total} combinations passed!")
    else:
        print(f"  {errors} failures found!")

    # -------------------------------------------------------------------------
    # Detailed walkthrough for one example: -4 x -1
    # -------------------------------------------------------------------------
    print("\n--- Detailed Walkthrough: -4 x -1 (4-bit) ---")
    print("This matches Figure 2 in the paper\n")

    w_s, a_s = -4, -1
    n = 4

    w_u = clip_map(w_s, n)
    a_u = direct_map(a_s, n)

    print(f"  Inputs:        w_s = {w_s},  a_s = {a_s}")
    print(f"  After mapping: w_u = {w_u},  a_u = {a_u}")
    print(f"  DSP output:    w_u * a_u = {w_u} * {a_u} = {w_u * a_u}")
    print(f"  Expected:      w_s * a_s = {w_s} * {a_s} = {w_s * a_s}")
    print()

    term1 =  w_u * a_u
    term2 = (2 ** (n-1)) * a_s * _iverson(w_s < 0)
    term3 = (2 ** n)     * w_s * _iverson(a_s < 0)
    term4 = (2 ** (2*n-1))     * _iverson(w_s < 0) * _iverson(a_s < 0)

    print(f"  Term 1 (raw DSP):           {term1}")
    print(f"  Term 2 (weight correction): {term2}  [w_s<0]={_iverson(w_s<0)}")
    print(f"  Term 3 (activ correction):  {term3}  [a_s<0]={_iverson(a_s<0)}")
    print(f"  Term 4 (cross correction):  {term4}")
    print(f"  Result: {term1} - {term2} - {term3} - {term4} = {term1-term2-term3-term4}")

    # -------------------------------------------------------------------------
    # Compare correction terms across all three schemes
    # -------------------------------------------------------------------------
    print("\n--- Correction Terms Comparison (w_s=-4, a_s=-1, n=4) ---\n")
    print(f"  {'Scheme':<16} | {'Term1':>6} | {'Term2':>6} | {'Term3':>6} | {'Term4':>6} | {'Result':>6}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    for scheme in ["clip_direct", "clip_offset", "clip_clip"]:
        if scheme == "clip_direct":
            w_u_ = clip_map(w_s, n)
            a_u_ = direct_map(a_s, n)
        elif scheme == "clip_offset":
            w_u_ = clip_map(w_s, n)
            a_u_ = offset_map(a_s, n)
        else:
            w_u_ = clip_map(w_s, n)
            a_u_ = clip_map(a_s, n)

        result = recover(w_u_, a_u_, w_s, a_s, n, scheme)
        t1 = w_u_ * a_u_
        print(f"  {scheme:<16} | {t1:>6} | {'see fn':>6} | {'see fn':>6} | {'see fn':>6} | {result:>6}")

    print("\nAll tests passed!")
