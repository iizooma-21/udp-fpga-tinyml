"""
core/postprocessing.py
======================
Computes the final signed products from the packed DSP result after
preprocessing (equation 4 of the UDP paper).

Background
----------
After the DSP computes ``r_packed = w_p * a_p + b_pre``, the packed result
contains unsigned partial products that still need two correction terms from
the clip+direct recovery formula (equation 1a):

    w^s * a^s = w^u * a^u
                - 2^(W_w - 1) * a^s * [w^s < 0]       ← folded into b_pre
                - 2^W_w       * w^s * [a^s < 0]        ← postprocessing term A
                - 2^(2W_w-1)  * [w^s < 0][a^s < 0]    ← postprocessing term B

Equation 4 says: for each packed result slot (i, j), subtract a correction
``b_post_ij`` and interpret the result as a signed integer:

    result_ij = slot_ij − b_post_ij

Where ``b_post_ij`` has two components:

        1. An **inner-j residual**: the preprocessing correction for activation
               group j is packed at bit positions ``(W_w−1)·k`` for each negative
               weight k.  Because (W_w−1) < W_p, these contributions can spill
               across slot boundaries, affecting slots i = 0, 1, … up to
               ``ceil(P_w*(W_w-1)/W_p) - 1``.  The contribution that lands in slot
               i is the W_p-bit window of inner_j starting at bit ``i·W_p``:

                   inner_j_contrib_i = bits[(i+1)·W_p−1 : i·W_p]  of  inner_j

               where:

               inner_j = Σ_{k=0}^{P_w-1}  [w^s_k < 0] · (−a^s_j)^u · 2^{(W_w−1)·k}

    2. **Terms A and B** from equation 1a for each pair (i, j):

       terms_AB_ij = 2^(W_w−1) · a^s_j · [w^s_i < 0]
                   + 2^W_w     · w^u_i  · [a^s_j < 0]

       (This uses the identity  2^W_w · w^s + 2^(2W_w−1) · [w^s<0]
       = 2^W_w · w^u_i,  which eliminates the separate term-B computation.)

        Combined per-slot correction (equation 4):

            b_post_ij = inner_j_contrib_i
                      + 2^(W_w−1) · a^s_j · [w^s_i < 0]
                      + 2^W_w     · w^u_i  · [a^s_j < 0]

            result_ij = slot_ij − b_post_ij          (plain integer arithmetic)

NOTE ON SIGN HANDLING
---------------------
The slot values extracted from r_packed are *unsigned* W_p-bit integers.
After subtracting b_post_ij, the result is the signed product w^s_i * a^s_j.
No sign-extension of the slot is needed before subtraction; the arithmetic
naturally produces the correct signed integer.

NOTE ON OVERFLOW
----------------
If |w^s_i * a^s_j| >= 2^(W_p-1) the product overflows the W_p-bit slot
(e.g. the edge case -2^(W_w-1) × -2^(W_a-1) with equal bitwidths as noted
in Section 3.3 of the paper).  The model computes the unclipped signed
integer, leaving overflow handling to the caller.

Conventions (match the rest of the codebase)
--------------------------------------------
* All values are plain Python ``int``.
* Signed inputs must satisfy ``assert_valid_signed(v, W, name)``.
* Unsigned inputs must satisfy ``assert_valid_unsigned(v, W, name)``.
* Every public function has a ``verbose`` flag (default False).
"""

from utilities.fixed_point import assert_valid_signed, to_signed, to_unsigned
from utilities.bit_ops import get_bits
from core.mapping import clip_map, direct_map
from core.packing import compute_wp, unpack_results


# ---------------------------------------------------------------------------
# Helper 1 – inner-j residual (shared with preprocessing)
# ---------------------------------------------------------------------------

def _compute_inner_j(
    w_s_list: list,
    a_s: int,
    W_w: int,
    W_a: int,
) -> int:
    """Re-compute the inner-j preprocessing residual for activation j.

    This is *identical* to the computation performed inside
    ``preprocessing._compute_activation_correction``.  The result is a
    packed integer whose W_p-bit windows (at offsets i·W_p) give the
    b_pre contribution that landed in slot i of activation group j.

    inner_j = Σ_{k=0}^{P_w-1}  [w^s_k < 0] · (−a^s)^u · 2^{(W_w−1)·k}

    The contribution to slot i is then:
        inner_j_contrib_i = get_bits(inner_j, (i+1)*W_p−1, i*W_p)

    Parameters
    ----------
    w_s_list : list of int   All P_w signed weight values.
    a_s      : int           Signed activation for this group.
    W_w      : int           Weight bitwidth.
    W_a      : int           Activation bitwidth.

    Returns
    -------
    int   Non-negative inner_j value (may span multiple W_p-bit slots).
    """
    # neg_a is placed at bit (W_w-1) WITHIN each weight's own slot k.
    # Slot k starts at bit k*W_p, so the shift is k*W_p + (W_w-1).
    # This matches the preprocessing formula exactly and keeps each
    # correction within its slot (no cross-slot bleed or carry).
    W_p = W_w + W_a - 1
    neg_a_unsigned = to_unsigned(-a_s, W_a)  # (−a^s) mod 2^W_a; handles a_s = -2^(W_a-1)
    inner = 0
    for k, w_s in enumerate(w_s_list):
        if w_s < 0:
            inner |= neg_a_unsigned << (k * W_p + (W_w - 1))
    return inner


# ---------------------------------------------------------------------------
# Helper 2 – terms A and B correction for one (weight, activation) pair
# ---------------------------------------------------------------------------

def _compute_terms_AB(
    w_s: int,
    a_s: int,
    W_w: int,
    W_a: int,
) -> int:
    """Compute the combined terms-A-and-B correction for a single pair.

    From equation 1a:
        term A = 2^(W_w−1) · a^s · [w^s < 0]
        term B = 2^(2W_w−1) · [w^s < 0][a^s < 0]

    Identity:  2^W_w · w^s + 2^(2W_w−1) · [w^s < 0] = 2^W_w · w^u
    Therefore: term A + term B = 2^(W_w−1)·a^s·[w^s<0] + 2^W_w·w^u·[a^s<0]

    Parameters
    ----------
    w_s : int   Signed weight.
    a_s : int   Signed activation.
    W_w : int   Weight bitwidth.
    W_a : int   Activation bitwidth (used only to validate clip_map output).

    Returns
    -------
    int   Combined correction (may be negative if a^s < 0 and w^s = 0).
    """
    w_u = clip_map(w_s, W_w)
    term_A = (2 ** (W_w - 1)) * a_s * (1 if w_s < 0 else 0)
    term_B = (2 ** W_w) * w_u * (1 if a_s < 0 else 0)
    return term_A + term_B


# ---------------------------------------------------------------------------
# Helper 3 – recover single slot
# ---------------------------------------------------------------------------

def _recover_single_slot(
    slot_value: int,
    w_s: int,
    a_s: int,
    i: int,
    inner_j: int,
    W_w: int,
    W_a: int,
    W_p: int,
    verbose: bool = False,
) -> int:
    """Apply equation 4 to a single unpacked slot and return w^s_i * a^s_j.

    Parameters
    ----------
    slot_value : int   Raw unsigned W_p-bit value from r_packed at slot (i,j).
    w_s        : int   Signed weight w^s_i.
    a_s        : int   Signed activation a^s_j.
    i          : int   Weight slot index (0 … P_w−1).
    inner_j    : int   Full preprocessing residual for this activation group.
                       The W_p-bit window at offset i·W_p is extracted here.
    W_w        : int   Weight bitwidth.
    W_a        : int   Activation bitwidth.
    W_p        : int   Product slot width (W_w + W_a − 1).
    verbose    : bool

    Returns
    -------
    int   Signed product w^s_i * a^s_j.
    """
    # Extract the portion of inner_j that landed in slot i.
    residual   = get_bits(inner_j, (i + 1) * W_p - 1, i * W_p)
    terms_AB   = _compute_terms_AB(w_s, a_s, W_w, W_a)
    b_post_ij  = residual + terms_AB
    result     = slot_value - b_post_ij

    if verbose:
        print(
            f"    slot[i={i}]: raw={slot_value}, "
            f"residual(inner_j@slot{i})={residual}, "
            f"termsAB={terms_AB}, b_post={b_post_ij}, result={result}"
        )

    # Sanity-check against direct multiplication (catches logic errors).
    expected = w_s * a_s
    assert result == expected, (
        f"Postprocessing mismatch at i={i}: "
        f"w_s={w_s}, a_s={a_s} → got {result}, expected {expected}"
    )

    return result


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_results(
    r_packed: int,
    w_s_list: list,
    a_s_list: list,
    W_w: int,
    W_a: int,
    verbose: bool = False,
) -> list:
    """Apply equation 4 (postprocessing) to recover all signed products.

    Given the packed DSP result ``r_packed = w_p * a_p + b_pre``, this
    function extracts every slot, applies the per-slot correction
    ``b_post_ij``, and returns the exact signed products ``w^s_i * a^s_j``
    for all (i, j) pairs.

    Equation 4 per slot:
        b_post_ij = inner_j_contrib_i
                  + 2^(W_w−1) · a^s_j · [w^s_i < 0]
                  + 2^W_w     · w^u_i  · [a^s_j < 0]
        result_ij = slot_ij − b_post_ij

    Parameters
    ----------
    r_packed  : int           Packed DSP result (non-negative integer).
                              Must equal ``w_p * a_p + b_pre``.
    w_s_list  : list of int   P_w signed weight values, each in
                              [−2^(W_w−1), 2^(W_w−1) − 1].
    a_s_list  : list of int   P_a signed activation values, each in
                              [−2^(W_a−1), 2^(W_a−1) − 1].
    W_w       : int           Weight bitwidth (e.g. 4 for int4).
    W_a       : int           Activation bitwidth (e.g. 4 for int4).
    verbose   : bool          Print step-by-step breakdown if True.

    Returns
    -------
    list of list of int   Shape [P_a][P_w].
        results[j][i] = w^s_i * a^s_j   (exact signed product)

    Example
    -------
    >>> from core.packing import pack_weights, pack_activations
    >>> from core.mapping import clip_map, direct_map
    >>> from core.preprocessing import compute_b_pre
    >>> w_s, a_s = [-3, 2], [5, -2]
    >>> W_w, W_a = 4, 4
    >>> w_u = [clip_map(w, W_w) for w in w_s]
    >>> a_u = [direct_map(a, W_a) for a in a_s]
    >>> w_p  = pack_weights(w_u, W_w, W_a)
    >>> a_p  = pack_activations(a_u, P_w=2, W_w=W_w, W_a=W_a)
    >>> b_pre = compute_b_pre(w_s, a_s, W_w=W_w, W_a=W_a)
    >>> r_packed = w_p * a_p + b_pre
    >>> results = compute_results(r_packed, w_s, a_s, W_w, W_a)
    """
    P_w = len(w_s_list)
    P_a = len(a_s_list)

    if P_w == 0 or P_a == 0:
        raise ValueError("w_s_list and a_s_list must each have at least one element.")

    for idx, w in enumerate(w_s_list):
        assert_valid_signed(w, W_w, f"w_s_list[{idx}]")
    for idx, a in enumerate(a_s_list):
        assert_valid_signed(a, W_a, f"a_s_list[{idx}]")

    W_p = compute_wp(W_w, W_a)

    if verbose:
        print(f"compute_results: P_w={P_w}, P_a={P_a}, W_w={W_w}, W_a={W_a}, W_p={W_p}")
        print(f"  w_s_list  : {w_s_list}")
        print(f"  a_s_list  : {a_s_list}")
        print(f"  r_packed  : {r_packed}")

    results = []
    for j, a_s in enumerate(a_s_list):
        # Pre-compute inner_j once per activation group (used only for i=0).
        inner_j = _compute_inner_j(w_s_list, a_s, W_w, W_a)

        if verbose:
            print(f"\n  --- Activation j={j}, a_s={a_s}, inner_j={inner_j} ---")

        row = []
        for i, w_s in enumerate(w_s_list):
            slot_idx  = j * P_w + i
            bit_start = slot_idx * W_p
            bit_end   = bit_start + W_p - 1
            slot_val  = get_bits(r_packed, bit_end, bit_start)

            result = _recover_single_slot(
                slot_val, w_s, a_s, i, inner_j, W_w, W_a, W_p, verbose=verbose
            )
            row.append(result)

        results.append(row)

    if verbose:
        print(f"\n  Final results[j][i]:")
        for j, row in enumerate(results):
            for i, val in enumerate(row):
                print(f"    result[{j}][{i}] = w_s_{i} × a_s_{j} = "
                      f"{w_s_list[i]} × {a_s_list[j]} = {val}")

    return results


# ---------------------------------------------------------------------------
# Convenience: check whether postprocessing corrections are needed
# ---------------------------------------------------------------------------

def needs_postprocessing(a_s_list: list, W_a: int) -> bool:
    """Return True if any activation is negative (i.e. terms A/B are non-zero).

    Useful as a fast early-exit check before calling ``compute_results``.

    Parameters
    ----------
    a_s_list : list of int   Signed activation values.
    W_a      : int           Activation bitwidth (used only for validation).
    """
    for idx, a in enumerate(a_s_list):
        assert_valid_signed(a, W_a, f"a_s_list[{idx}]")
    return any(a < 0 for a in a_s_list)


# ---------------------------------------------------------------------------
# Self-test  (run with: python -m core.postprocessing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core.mapping import clip_map, direct_map
    from core.packing import pack_weights, pack_activations
    from core.preprocessing import compute_b_pre

    print("=" * 60)
    print("Self-test: core/postprocessing.py")
    print("=" * 60)

    def _run_pipeline(w_s_list, a_s_list, W_w, W_a, label="", verbose=False):
        """Full UDP pipeline: map → pack → DSP → b_pre → b_post → results."""
        P_w = len(w_s_list)
        W_p = compute_wp(W_w, W_a)

        w_u_list = [clip_map(w, W_w)   for w in w_s_list]
        a_u_list = [direct_map(a, W_a) for a in a_s_list]

        w_p   = pack_weights(w_u_list, W_w, W_a)
        a_p   = pack_activations(a_u_list, P_w=P_w, W_w=W_w, W_a=W_a)
        b_pre = compute_b_pre(w_s_list, a_s_list, W_w=W_w, W_a=W_a)

        r_packed = w_p * a_p + b_pre
        results  = compute_results(r_packed, w_s_list, a_s_list, W_w, W_a,
                                   verbose=verbose)
        return results

    # ------------------------------------------------------------------
    # Test 1: P_w=1, P_a=1 – all sign combinations for int4
    # ------------------------------------------------------------------
    print("\n[Test 1] P_w=1, P_a=1, all 4-bit signed combinations")
    W_w, W_a = 4, 4
    signed_vals = list(range(-(2**(W_w-1)), 2**(W_w-1)))
    errors = 0
    for w_s in signed_vals:
        for a_s in signed_vals:
            res = _run_pipeline([w_s], [a_s], W_w, W_a)
            expected = w_s * a_s
            if res[0][0] != expected:
                print(f"  FAIL: w_s={w_s}, a_s={a_s} → {res[0][0]}, expected {expected}")
                errors += 1
    if errors == 0:
        print(f"  PASSED  ({len(signed_vals)**2} combinations)")
    else:
        print(f"  {errors} FAILURES")

    # ------------------------------------------------------------------
    # Test 2: P_w=2, P_a=1 – two weights, one activation
    # ------------------------------------------------------------------
    print("\n[Test 2] P_w=2, P_a=1, w_s=[-3,-2], a_s=[-2]")
    res = _run_pipeline([-3, -2], [-2], W_w=4, W_a=4, verbose=True)
    assert res[0][0] == (-3) * (-2), f"Expected 6, got {res[0][0]}"
    assert res[0][1] == (-2) * (-2), f"Expected 4, got {res[0][1]}"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 3: P_w=1, P_a=2 – one weight, two activations
    # ------------------------------------------------------------------
    print("\n[Test 3] P_w=1, P_a=2, w_s=[-8], a_s=[7,-1]")
    res = _run_pipeline([-8], [7, -1], W_w=4, W_a=4, verbose=True)
    assert res[0][0] == (-8) * 7,   f"Expected -56, got {res[0][0]}"
    assert res[1][0] == (-8) * (-1), f"Expected 8,  got {res[1][0]}"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 4: P_w=3, P_a=2 – full 3-weight × 2-activation grid
    # ------------------------------------------------------------------
    print("\n[Test 4] P_w=3, P_a=2 – 3×2 product grid (paper example)")
    w_s_list = [-3, 2, -1]
    a_s_list = [5, -2]
    res = _run_pipeline(w_s_list, a_s_list, W_w=4, W_a=4, verbose=True)
    all_ok = True
    for j, a_s in enumerate(a_s_list):
        for i, w_s in enumerate(w_s_list):
            expected = w_s * a_s
            got      = res[j][i]
            if got != expected:
                print(f"  FAIL: ({i},{j}) w_s={w_s}, a_s={a_s} → {got}, expected {expected}")
                all_ok = False
    if all_ok:
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 5: All-positive weights – no preprocessing/postprocessing needed
    # ------------------------------------------------------------------
    print("\n[Test 5] All-positive weights, mixed activations")
    res = _run_pipeline([3, 2, 1], [5, -2], W_w=4, W_a=4)
    assert res[0][0] == 3*5  and res[0][1] == 2*5  and res[0][2] == 1*5
    assert res[1][0] == 3*-2 and res[1][1] == 2*-2 and res[1][2] == 1*-2
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 6: All-positive activations – no postprocessing terms A/B
    # ------------------------------------------------------------------
    print("\n[Test 6] All-positive activations – terms A/B are zero")
    res = _run_pipeline([-3, 2, -1], [5, 3], W_w=4, W_a=4)
    for j, a_s in enumerate([5, 3]):
        for i, w_s in enumerate([-3, 2, -1]):
            assert res[j][i] == w_s * a_s, (
                f"res[{j}][{i}]={res[j][i]}, expected {w_s*a_s}"
            )
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 7: needs_postprocessing helper
    # ------------------------------------------------------------------
    print("\n[Test 7] needs_postprocessing helper")
    assert needs_postprocessing([-1,  2], W_a=4) is True
    assert needs_postprocessing([ 1,  2], W_a=4) is False
    assert needs_postprocessing([ 0,  7], W_a=4) is False
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 8: int8 × int8, P_w=1, P_a=1 – larger bitwidths
    # ------------------------------------------------------------------
    print("\n[Test 8] int8 × int8, P_w=1, P_a=1")
    import random
    random.seed(42)
    errors = 0
    for _ in range(500):
        w_s = random.randint(-128, 127)
        a_s = random.randint(-128, 127)
        res = _run_pipeline([w_s], [a_s], W_w=8, W_a=8)
        if res[0][0] != w_s * a_s:
            print(f"  FAIL: w={w_s}, a={a_s}")
            errors += 1
    if errors == 0:
        print("  PASSED  (500 random int8 pairs)")

    print("\n" + "=" * 60)
    print("All postprocessing tests passed.")
    print("=" * 60)
