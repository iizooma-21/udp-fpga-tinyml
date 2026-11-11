"""
core/packing.py
===============
Implements the UDP packing and unpacking phases (Section 3.2 of the paper).

WHAT THIS FILE DOES
-------------------
After mapping (clip_map on weights, direct_map on activations), we have
unsigned values w_u_i and a_u_j.  The packing step interleaves these with
zero-padding so that a single wide DSP multiplication computes ALL P_w * P_a
cross-products simultaneously, with results sitting in non-overlapping bit
windows inside the product.

Key insight (Equation 2):
    r_p = w_p * a_p
         = SUM_{i,j} w_u_i * a_u_j << ((j*P_w + i) * W_p)

Each partial product occupies exactly W_p = W_w + W_a - 1 bits, and the
zero-padding guarantees no overlap between adjacent products.

Port assignment (Section 4):
    - w_p  → DSP short port  (narrower, W_pw bits)
    - a_p  → DSP long  port  (wider,   W_pa bits)

  W_pw = (P_w - 1)*W_p + (W_w - 1)      # highest weight slot ends here
  W_pa = (P_a - 1)*P_w*W_p + W_a        # highest activation slot ends here

  Because W_pa >= W_pw, this assignment maximises DSP utilisation.

NOTATION (matches Table 2 in paper)
------------------------------------
  W_w       : bitwidth of each weight  (before mapping, signed)
  W_a       : bitwidth of each activation (before mapping, signed)
  W_p       : W_w + W_a - 1  (max result bitwidth / slot width)
  P_w       : number of weights  packed together
  P_a       : number of activations packed together
  w_u_i     : unsigned-mapped weight   i   (W_w - 1 bits wide after clip)
  a_u_j     : unsigned-mapped activation j (W_a     bits wide after direct)
  w_p       : packed weight integer
  a_p       : packed activation integer
  r_p       : DSP product  w_p * a_p
  r_p[i,j]  : the (i,j) slot of r_p  = w_u_i * a_u_j

NOTE on clip-mapped weight bitwidth
-------------------------------------
clip_map removes the MSB, so w_u has W_w - 1 significant bits even though
we still call the nominal width W_w.  The product w_u * a_u therefore fits
in (W_w - 1) + W_a = W_p bits.  This is exactly why W_p = W_w + W_a - 1.
"""

from utilities.bit_ops import insert_zeros, get_bits, concatenate_bits


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def compute_wp(W_w: int, W_a: int) -> int:
    """
    Return W_p = W_w + W_a - 1, the slot width used throughout packing.

    W_p is simultaneously:
      - the maximum bitwidth of one w_u * a_u product
      - the spacing between adjacent result slots in r_p

    Parameters
    ----------
    W_w : int  bitwidth of weight input  (>= 1)
    W_a : int  bitwidth of activation input (>= 1)

    Returns
    -------
    int  W_p >= 1

    Example
    -------
    >>> compute_wp(4, 4)
    7
    >>> compute_wp(8, 8)
    15
    """
    assert W_w >= 1, f"W_w must be >= 1, got {W_w}"
    assert W_a >= 1, f"W_a must be >= 1, got {W_a}"
    return W_w + W_a - 1


def _packed_weight_width(P_w: int, W_p: int, W_w: int) -> int:
    """
    Bitwidth of the packed weight integer w_p.

        W_pw = (P_w - 1)*W_p + (W_w - 1)

    The highest weight slot starts at bit (P_w-1)*W_p and occupies W_w-1
    bits (clip-mapped, so MSB already removed).
    """
    return (P_w - 1) * W_p + (W_w - 1)


def _packed_activation_width(P_a: int, P_w: int, W_p: int, W_a: int) -> int:
    """
    Bitwidth of the packed activation integer a_p.

        W_pa = (P_a - 1)*P_w*W_p + W_a

    The highest activation slot starts at bit (P_a-1)*P_w*W_p and occupies
    W_a bits.
    """
    return (P_a - 1) * P_w * W_p + W_a


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def pack_weights(w_u_list: list, W_w: int, W_a: int, verbose: bool = False) -> int:
    """
    Pack a list of unsigned-mapped weights into a single integer w_p.

    Layout (slot i starts at bit i*W_p):
        w_p = SUM_{i=0}^{P_w-1}  w_u_i << (i * W_p)

    Each w_u_i occupies W_w-1 bits (clip mapping removed the MSB).
    Zero-padding of W_a bits follows each slot, preventing carry bleed.

    Parameters
    ----------
    w_u_list : list of int
        Unsigned-mapped weights [w_u_0, ..., w_u_{P_w-1}].
        Each value must be in [0, 2^(W_w-1) - 1].
    W_w : int   Original signed weight bitwidth.
    W_a : int   Original signed activation bitwidth.
    verbose : bool  Print step-by-step layout when True.

    Returns
    -------
    int  w_p >= 0

    Example (int4, P_w=3)
    ----------------------
    With W_w=4, W_a=4 → W_p=7.  Three weights occupy bits [0:3], [7:10],
    [14:17] of w_p.  The DSP short port needs 17 bits total.

    >>> pack_weights([5, 3, 6], W_w=4, W_a=4)
    # slot0=5 at bits[0:3], slot1=3 at bits[7:10], slot2=6 at bits[14:17]
    """
    P_w = len(w_u_list)
    assert P_w >= 1, "Need at least one weight."
    assert W_w >= 2, f"W_w must be >= 2 for clip mapping, got {W_w}"
    assert W_a >= 1, f"W_a must be >= 1, got {W_a}"

    W_p   = compute_wp(W_w, W_a)
    W_u_w = W_w - 1          # effective unsigned width after clip

    # Validate each weight is in the unsigned clip-mapped range [0, 2^(W_w-1)-1]
    max_w_u = (1 << W_u_w) - 1
    for idx, w in enumerate(w_u_list):
        assert isinstance(w, int), f"w_u_list[{idx}] must be int, got {type(w)}"
        assert 0 <= w <= max_w_u, (
            f"w_u_list[{idx}]={w} out of unsigned clip range [0, {max_w_u}]"
        )

    w_p = 0
    if verbose:
        W_pw = _packed_weight_width(P_w, W_p, W_w)
        print(f"\n=== pack_weights ===")
        print(f"  W_w={W_w}, W_a={W_a}, W_p={W_p}, P_w={P_w}")
        print(f"  Effective unsigned width per weight: W_u_w = W_w-1 = {W_u_w}")
        print(f"  Total packed weight width W_pw = {W_pw}")
        print(f"  Slot i starts at bit i*W_p = i*{W_p}")

    for i, w_u in enumerate(w_u_list):
        shift = i * W_p
        contribution = w_u << shift
        w_p |= contribution
        if verbose:
            print(f"  slot {i}: w_u={w_u} (0b{w_u:0{W_u_w}b}) "
                  f"<< {shift}  → bits [{shift}:{shift + W_u_w - 1}]  "
                  f"contribution=0b{contribution:b}")

    if verbose:
        W_pw = _packed_weight_width(P_w, W_p, W_w)
        print(f"  w_p = {w_p}  (0b{w_p:0{W_pw}b})")

    return w_p


def pack_activations(
    a_u_list: list,
    P_w: int,
    W_w: int,
    W_a: int,
    verbose: bool = False,
) -> int:
    """
    Pack a list of unsigned-mapped activations into a single integer a_p.

    Layout (activation slot j starts at bit j*P_w*W_p):
        a_p = SUM_{j=0}^{P_a-1}  a_u_j << (j * P_w * W_p)

    The stride between activation slots is P_w*W_p so that cross-products
    w_u_i * a_u_j land at bit position (j*P_w + i)*W_p — matching Eq. (2).

    Parameters
    ----------
    a_u_list : list of int
        Unsigned-mapped activations [a_u_0, ..., a_u_{P_a-1}].
        Each value must be in [0, 2^W_a - 1] (direct mapping keeps W_a bits).
    P_w  : int   Number of weights packed (needed to compute the stride).
    W_w  : int   Original signed weight bitwidth.
    W_a  : int   Original signed activation bitwidth.
    verbose : bool

    Returns
    -------
    int  a_p >= 0

    Example (int4, P_w=3, P_a=2)
    ------------------------------
    Stride = P_w*W_p = 3*7 = 21.
    a_u_0 at bits[0:3], a_u_1 at bits[21:24].
    """
    P_a = len(a_u_list)
    assert P_a >= 1, "Need at least one activation."
    assert P_w >= 1, f"P_w must be >= 1, got {P_w}"
    assert W_w >= 2, f"W_w must be >= 2, got {W_w}"
    assert W_a >= 1, f"W_a must be >= 1, got {W_a}"

    W_p    = compute_wp(W_w, W_a)
    stride = P_w * W_p      # gap between activation slots

    max_a_u = (1 << W_a) - 1
    for idx, a in enumerate(a_u_list):
        assert isinstance(a, int), f"a_u_list[{idx}] must be int"
        assert 0 <= a <= max_a_u, (
            f"a_u_list[{idx}]={a} out of unsigned direct range [0, {max_a_u}]"
        )

    a_p = 0
    if verbose:
        W_pa = _packed_activation_width(P_a, P_w, W_p, W_a)
        print(f"\n=== pack_activations ===")
        print(f"  W_w={W_w}, W_a={W_a}, W_p={W_p}, P_w={P_w}, P_a={P_a}")
        print(f"  Activation slot stride = P_w*W_p = {P_w}*{W_p} = {stride}")
        print(f"  Total packed activation width W_pa = {W_pa}")

    for j, a_u in enumerate(a_u_list):
        shift = j * stride
        contribution = a_u << shift
        a_p |= contribution
        if verbose:
            print(f"  slot {j}: a_u={a_u} (0b{a_u:0{W_a}b}) "
                  f"<< {shift}  → bits [{shift}:{shift + W_a - 1}]  "
                  f"contribution=0b{contribution:b}")

    if verbose:
        W_pa = _packed_activation_width(P_a, P_w, W_p, W_a)
        print(f"  a_p = {a_p}  (0b{a_p:0{W_pa}b})")

    return a_p


# ---------------------------------------------------------------------------
# Unpacking
# ---------------------------------------------------------------------------

def unpack_results(
    r_p: int,
    P_w: int,
    P_a: int,
    W_w: int,
    W_a: int,
    verbose: bool = False,
) -> list:
    """
    Extract all P_w * P_a unsigned partial products from the packed DSP result.

    Each slot (i, j) occupies W_p bits starting at bit (j*P_w + i)*W_p.
    Because the slot width W_p exactly equals the maximum product bitwidth,
    there is no overlap and extraction is a simple bit-field read.

    Returns
    -------
    list of list of int   shape [P_a][P_w]
        results[j][i] = w_u_i * a_u_j   (unsigned, W_p bits wide)

    This does NOT yet apply the clip+direct recovery formula — that is the
    job of postprocessing.py.  Here we only extract the raw unsigned products.

    Parameters
    ----------
    r_p  : int   Raw DSP multiplication result w_p * a_p  (or w_p*a_p + b_pre)
    P_w  : int   Number of weights packed.
    P_a  : int   Number of activations packed.
    W_w  : int   Original signed weight bitwidth.
    W_a  : int   Original signed activation bitwidth.
    verbose : bool

    Example (int4, P_w=3, P_a=2) → 6 products extracted
    """
    assert P_w >= 1 and P_a >= 1, "P_w and P_a must be >= 1"
    assert W_w >= 2 and W_a >= 1, "W_w >= 2, W_a >= 1"
    assert isinstance(r_p, int) and r_p >= 0, f"r_p must be non-negative int, got {r_p}"

    W_p = compute_wp(W_w, W_a)

    if verbose:
        print(f"\n=== unpack_results ===")
        print(f"  W_w={W_w}, W_a={W_a}, W_p={W_p}, P_w={P_w}, P_a={P_a}")
        print(f"  r_p = {r_p}")

    # Build 2D result grid:  results[j][i]
    results = []
    for j in range(P_a):
        row = []
        for i in range(P_w):
            slot_index  = j * P_w + i
            bit_start   = slot_index * W_p
            bit_end     = bit_start + W_p - 1        # inclusive
            slot_value  = get_bits(r_p, bit_end, bit_start)
            row.append(slot_value)
            if verbose:
                print(f"  r_p[{j},{i}]: bits [{bit_start}:{bit_end}] "
                      f"= {slot_value}  (w_u_{i} * a_u_{j})")
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def verify_packing(
    w_u_list: list,
    a_u_list: list,
    W_w: int,
    W_a: int,
    verbose: bool = False,
) -> bool:
    """
    End-to-end sanity check: pack both inputs, multiply, unpack, and compare
    every slot against the direct unsigned product.

    This verifies the core mathematical property:
        (w_p * a_p)[slot(i,j)] == w_u_i * a_u_j   for all i, j

    Parameters
    ----------
    w_u_list : list of int   Unsigned-mapped weights   (length = P_w)
    a_u_list : list of int   Unsigned-mapped activations (length = P_a)
    W_w : int
    W_a : int
    verbose : bool   Print a detailed trace when True.

    Returns
    -------
    bool   True if all slots match expected direct products.

    Raises
    ------
    AssertionError   if any mismatch is found (even with verbose=False).
    """
    P_w = len(w_u_list)
    P_a = len(a_u_list)
    W_p = compute_wp(W_w, W_a)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  verify_packing: P_w={P_w}, P_a={P_a}, W_w={W_w}, W_a={W_a}")
        print(f"  W_p = {W_p}")
        print(f"{'='*55}")

    # Step 1: pack
    w_p = pack_weights(w_u_list, W_w, W_a, verbose=verbose)
    a_p = pack_activations(a_u_list, P_w, W_w, W_a, verbose=verbose)

    # Step 2: simulate DSP multiply (pure software — no port-width constraints here)
    r_p = w_p * a_p

    if verbose:
        print(f"\n  DSP multiply: w_p={w_p}, a_p={a_p}")
        print(f"  r_p = w_p * a_p = {r_p}")

    # Step 3: unpack
    results = unpack_results(r_p, P_w, P_a, W_w, W_a, verbose=verbose)

    # Step 4: compare
    all_ok = True
    if verbose:
        print(f"\n  Slot verification:")
    for j, a_u in enumerate(a_u_list):
        for i, w_u in enumerate(w_u_list):
            expected = w_u * a_u
            got      = results[j][i]
            ok       = (got == expected)
            if not ok:
                all_ok = False
            if verbose:
                status = "✓" if ok else "✗ MISMATCH"
                print(f"    slot({i},{j}): w_u_{i}={w_u} * a_u_{j}={a_u} "
                      f"→ expected={expected}, got={got}  {status}")

    assert all_ok, "verify_packing: one or more slots produced wrong values!"

    if verbose:
        print(f"\n  All {P_w * P_a} slots verified correctly. ✓")

    return True


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("core/packing.py  —  self-test")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Test 1: int4 × int4, P_w=3, P_a=2  (the paper's running example)   #
    # ------------------------------------------------------------------ #
    print("\n--- Test 1: int4×int4, P_w=3, P_a=2 (paper Fig. 3 example) ---")
    W_w, W_a = 4, 4
    W_p = compute_wp(W_w, W_a)
    print(f"  W_p = W_w + W_a - 1 = {W_w} + {W_a} - 1 = {W_p}")

    # Three clip-mapped weights: values in [0, 7]  (3-bit unsigned)
    # Two direct-mapped activations: values in [0, 15] (4-bit unsigned)
    w_u_list = [5, 3, 6]      # three weights, clip-mapped
    a_u_list = [2, 9]         # two activations, direct-mapped

    verify_packing(w_u_list, a_u_list, W_w, W_a, verbose=True)

    # ------------------------------------------------------------------ #
    # Test 2: int8 × int8, P_w=1, P_a=1  (baseline: no packing)         #
    # ------------------------------------------------------------------ #
    print("\n--- Test 2: int8×int8, P_w=1, P_a=1 (degenerate / no packing) ---")
    W_w2, W_a2 = 8, 8
    verify_packing([100], [200], W_w2, W_a2, verbose=True)

    # ------------------------------------------------------------------ #
    # Test 3: int8 × int8, P_w=2, P_a=1  (double MAC, like paper [25])  #
    # ------------------------------------------------------------------ #
    print("\n--- Test 3: int8×int8, P_w=2, P_a=1 ---")
    verify_packing([50, 120], [30], 8, 8, verbose=True)

    # ------------------------------------------------------------------ #
    # Test 4: Exhaustive check — all valid 3-bit weights × 4-bit acts    #
    # for int4×int4, P_w=2, P_a=2                                        #
    # ------------------------------------------------------------------ #
    print("\n--- Test 4: Exhaustive int4×int4, P_w=2, P_a=2 ---")
    W_w4, W_a4 = 4, 4
    pass_count = 0
    fail_count = 0
    max_w_u = (1 << (W_w4 - 1)) - 1     # 7
    max_a_u = (1 << W_a4) - 1           # 15
    for w0 in range(max_w_u + 1):
        for w1 in range(max_w_u + 1):
            for a0 in range(max_a_u + 1):
                for a1 in range(max_a_u + 1):
                    ok = verify_packing([w0, w1], [a0, a1], W_w4, W_a4,
                                        verbose=False)
                    if ok:
                        pass_count += 1
                    else:
                        fail_count += 1
    total = pass_count + fail_count
    print(f"  Tested {total} combinations: {pass_count} passed, {fail_count} failed.")
    assert fail_count == 0, "Exhaustive test failed!"
    print("  All passed. ✓")

    # ------------------------------------------------------------------ #
    # Test 5: Helper width formulas                                        #
    # ------------------------------------------------------------------ #
    print("\n--- Test 5: Port width formulas ---")
    # int4×int4, P_w=3, P_a=2 → W_pw=17, W_pa=25 (matches paper Section 4)
    W_w5, W_a5, P_w5, P_a5 = 4, 4, 3, 2
    W_p5 = compute_wp(W_w5, W_a5)
    W_pw5 = _packed_weight_width(P_w5, W_p5, W_w5)
    W_pa5 = _packed_activation_width(P_a5, P_w5, W_p5, W_a5)
    print(f"  int4×int4, P_w=3, P_a=2:")
    print(f"    W_p  = {W_p5}  (expect 7)")
    print(f"    W_pw = {W_pw5}  (expect 17 → fits in 18-bit DSP short port)")
    print(f"    W_pa = {W_pa5}  (expect 25 → fits in 27-bit DSP long  port)")
    assert W_p5  == 7,  f"W_p  mismatch: {W_p5}"
    assert W_pw5 == 17, f"W_pw mismatch: {W_pw5}"
    assert W_pa5 == 25, f"W_pa mismatch: {W_pa5}"
    print("  Width formulas correct. ✓")

    print("\n" + "=" * 60)
    print("All packing tests passed. ✓")
    print("=" * 60)
