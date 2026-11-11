"""
core/preprocessing.py
=====================
Computes the preprocessing correction term ``b_pre`` that is fed into the
DSP's adder port *before* multiplication (equation 3 of the UDP paper).

Background
----------
After clip+direct mapping, the signed-signed product w^s * a^s is recovered
via (equation 1a):

    w^s * a^s = w^u * a^u
                - 2^(W_w - 1) * a^s * [w^s < 0]       ← preprocessing term
                - 2^W_w * w^s * [a^s < 0]              ← postprocessing term
                - 2^(2W_w - 1) * [w^s < 0][a^s < 0]   ← postprocessing term

The first correction term depends only on the *sign of w^s* and the *value
of a^s*, which are both known before the DSP multiply happens.  UDP folds
this term into the DSP's built-in add port so the multiply-add operation
computes:

    r_p = w_p * a_p + b_pre

in a *single* DSP cycle, achieving full hardware utilisation with no extra
adder logic outside the DSP.

NOTE ON EQUATION 3 TYPO
-----------------------
The paper (equation 3) writes the outer shift as ``W_p * P_w * (j - 1)``,
which would produce a *negative* shift for j = 0.  This is a typo.

Correct reasoning: from equation 2, activation-group j occupies bit positions
starting at ``j * P_w * W_p`` in the packed word.  The outer shift must
therefore be ``W_p * P_w * j`` (not j-1) to align each activation's
correction with its slot in the packed result.  We use the corrected formula
here.

Conventions (match the rest of the codebase)
--------------------------------------------
* All values are plain Python ``int``.
* Signed inputs must satisfy ``assert_valid_signed(v, W, name)``.
* Unsigned inputs must satisfy ``assert_valid_unsigned(v, W, name)``.
* Every public function has a ``verbose`` flag (default False).
"""

from utilities.fixed_point import assert_valid_signed, assert_valid_unsigned, to_unsigned
from utilities.bit_ops import get_msb
from core.mapping import clip_map, direct_map
from core.packing import compute_wp


# ---------------------------------------------------------------------------
# Helper 1 – single-weight correction slot
# ---------------------------------------------------------------------------

def _compute_single_weight_correction(
    w_s: int,
    a_s: int,
    i: int,
    W_w: int,
    W_a: int,
) -> int:
    """Compute the correction contribution for one (weight i, any activation).

    This implements the inner summand of equation 3 for a single weight slot:

        contrib_i = (-a^s)^u * [w^s_i < 0]  <<  (W_w - 1) * i

    where ``(-a^s)^u`` means: negate a^s, then treat the result as an
    unsigned W_a-bit number.

    Why ``(-a^s)^u``?
    -----------------
    From equation 1a the preprocessing term for one pair is:

        -2^(W_w-1) * a^s * [w^s < 0]
        = 2^(W_w-1) * (-a^s) * [w^s < 0]

    The factor ``2^(W_w-1)`` comes from the slot spacing (shift below).
    The factor ``(-a^s)`` must be stored as an unsigned W_a-bit number so
    it fits cleanly into the packed word without sign-extension artefacts.
    ``direct_map(-a^s, W_a)`` achieves exactly this: it reinterprets the
    W_a-bit pattern of (-a^s) as unsigned, which is what the hardware does
    when it reads the bits off the wire.

    Parameters
    ----------
    w_s : int   Signed weight value.
    a_s : int   Signed activation value.
    i   : int   Weight slot index (0 … P_w-1).
    W_w : int   Weight bitwidth (signed).
    W_a : int   Activation bitwidth (signed).

    Returns
    -------
    int   Unsigned correction contribution for slot i (not yet shifted to
          its activation group — that outer shift is applied in
          ``_compute_activation_correction``).
    """
    assert_valid_signed(w_s, W_w, f"w_s[{i}]")
    assert_valid_signed(a_s, W_a, f"a_s (for weight slot {i})")

    # Only non-zero when this weight is negative.
    if w_s >= 0:
        return 0

    # Negate a^s and reinterpret as unsigned W_a bits.
    # We use to_unsigned (= x mod 2^W_a) rather than direct_map so that the
    # edge case a_s = -2^(W_a-1) (where -a_s overflows the signed range) is
    # handled correctly: to_unsigned(-a_s, W_a) = (-a_s) & ((1<<W_a)-1).
    neg_a_unsigned = to_unsigned(-a_s, W_a)

    # Place neg_a at bit (W_w-1) WITHIN slot i.
    # Slot i starts at bit i*W_p, so the absolute bit position is
    # i*W_p + (W_w-1).  This ensures the correction lands entirely
    # inside slot i and does not bleed into adjacent slots.
    W_p = W_w + W_a - 1
    shift = i * W_p + (W_w - 1)
    return neg_a_unsigned << shift


# ---------------------------------------------------------------------------
# Helper 2 – all-weight correction for one activation
# ---------------------------------------------------------------------------

def _compute_activation_correction(
    w_s_list: list,
    a_s: int,
    j: int,
    W_w: int,
    W_a: int,
    verbose: bool = False,
) -> int:
    """Pack the per-weight corrections for activation j into one integer.

    This implements the inner sum of equation 3 for a single activation j:

        inner_j = Σ_{i=0}^{P_w-1}  (-a^s_j)^u * [w^s_i < 0]  <<  (W_w-1)*i

    The result is then shifted to activation j's position in the full packed
    word by the caller (``compute_b_pre``).

    Parameters
    ----------
    w_s_list : list of int   All P_w signed weight values.
    a_s      : int           Signed activation for this group (index j).
    j        : int           Activation slot index (0 … P_a-1), used only
                             for verbose labelling.
    W_w      : int           Weight bitwidth.
    W_a      : int           Activation bitwidth.
    verbose  : bool

    Returns
    -------
    int   Packed inner correction for activation j (unsigned, not yet
          shifted to the activation group position).
    """
    inner = 0
    P_w = len(w_s_list)

    for i, w_s in enumerate(w_s_list):
        contrib = _compute_single_weight_correction(w_s, a_s, i, W_w, W_a)
        inner |= contrib          # slots are non-overlapping by construction
        if verbose and contrib != 0:
            print(
                f"    weight slot i={i}: w_s={w_s} < 0, "
                f"contrib = {contrib} (0b{contrib:0b})"
            )

    if verbose:
        print(
            f"  activation j={j}: a_s={a_s}, "
            f"inner correction = {inner} (0b{inner:0b})"
        )

    return inner


# ---------------------------------------------------------------------------
# Helper 3 – shift inner correction to activation group position
# ---------------------------------------------------------------------------

def _shift_to_activation_slot(
    inner: int,
    j: int,
    P_w: int,
    W_p: int,
) -> int:
    """Shift the inner correction into activation j's position in b_pre.

    From equation 2, activation group j starts at bit ``j * P_w * W_p`` in
    the packed word.  We apply the same shift here so that b_pre lines up
    exactly with the packed product w_p * a_p.

    NOTE: The paper writes ``W_p * P_w * (j - 1)`` but this is a typo
    (negative for j=0).  The correct shift is ``W_p * P_w * j``.

    Parameters
    ----------
    inner : int   Inner correction (from ``_compute_activation_correction``).
    j     : int   Activation slot index.
    P_w   : int   Number of packed weights.
    W_p   : int   Product bitwidth (= W_w + W_a - 1).

    Returns
    -------
    int   Shifted contribution ready to be OR-ed into b_pre.
    """
    shift = W_p * P_w * j          # corrected from (j-1) in the paper
    return inner << shift


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_b_pre(
    w_s_list: list,
    a_s_list: list,
    W_w: int,
    W_a: int,
    verbose: bool = False,
) -> int:
    """Compute the full preprocessing correction term b_pre (equation 3).

    b_pre is the value added to the DSP's adder port so that the DSP
    computes:

        r_p = w_p * a_p + b_pre

    This folds the ``-2^(W_w-1) * a^s * [w^s < 0]`` terms from the
    clip+direct recovery formula (equation 1a) *into* the DSP multiply-add,
    saving external adder resources and eliminating guard bits.

    Parameters
    ----------
    w_s_list : list of int   P_w signed weight values, each in
                             [-2^(W_w-1), 2^(W_w-1) - 1].
    a_s_list : list of int   P_a signed activation values, each in
                             [-2^(W_a-1), 2^(W_a-1) - 1].
    W_w      : int           Weight bitwidth (e.g. 4 for int4).
    W_a      : int           Activation bitwidth (e.g. 4 for int4).
    verbose  : bool          Print step-by-step breakdown if True.

    Returns
    -------
    int   b_pre — a non-negative integer to be passed to the DSP adder port.
          Zero means no negative weights: no correction needed.

    Example
    -------
    >>> w_s_list = [-3, 2, -1]   # P_w = 3, int4
    >>> a_s_list = [5, -2]       # P_a = 2, int4
    >>> b = compute_b_pre(w_s_list, a_s_list, W_w=4, W_a=4, verbose=True)
    """
    P_w = len(w_s_list)
    P_a = len(a_s_list)

    if P_w == 0 or P_a == 0:
        raise ValueError("w_s_list and a_s_list must each have at least one element.")

    # Validate all inputs up front.
    for i, w in enumerate(w_s_list):
        assert_valid_signed(w, W_w, f"w_s_list[{i}]")
    for j, a in enumerate(a_s_list):
        assert_valid_signed(a, W_a, f"a_s_list[{j}]")

    W_p = compute_wp(W_w, W_a)   # W_w + W_a - 1

    if verbose:
        print(f"compute_b_pre: P_w={P_w}, P_a={P_a}, W_w={W_w}, W_a={W_a}, W_p={W_p}")
        print(f"  weights    : {w_s_list}")
        print(f"  activations: {a_s_list}")
        neg_w_indices = [i for i, w in enumerate(w_s_list) if w < 0]
        print(f"  negative weight indices: {neg_w_indices}")
        if not neg_w_indices:
            print("  No negative weights → b_pre = 0 (no correction needed)")

    b_pre = 0

    for j, a_s in enumerate(a_s_list):
        if verbose:
            print(f"\n  --- Activation j={j}, a_s={a_s} ---")

        inner = _compute_activation_correction(
            w_s_list, a_s, j, W_w, W_a, verbose=verbose
        )
        shifted = _shift_to_activation_slot(inner, j, P_w, W_p)
        b_pre += shifted           # slots are non-overlapping; += and |= equiv

        if verbose and inner != 0:
            print(
                f"  → shifted to activation slot j={j}: "
                f"{shifted} (0b{shifted:0b})"
            )

    if verbose:
        print(f"\n  b_pre = {b_pre} (0b{b_pre:0b})")

    return b_pre


# ---------------------------------------------------------------------------
# Convenience: check whether preprocessing is needed at all
# ---------------------------------------------------------------------------

def needs_preprocessing(w_s_list: list, W_w: int) -> bool:
    """Return True if any weight is negative (i.e. b_pre will be non-zero).

    Useful as a fast early-exit check before calling ``compute_b_pre``.

    Parameters
    ----------
    w_s_list : list of int   Signed weight values.
    W_w      : int           Weight bitwidth (used only for validation).
    """
    for i, w in enumerate(w_s_list):
        assert_valid_signed(w, W_w, f"w_s_list[{i}]")
    return any(w < 0 for w in w_s_list)


# ---------------------------------------------------------------------------
# Self-test  (run with: python -m core.preprocessing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Self-test: core/preprocessing.py")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: All-positive weights → b_pre must be 0
    # ------------------------------------------------------------------
    print("\n[Test 1] All-positive weights → b_pre == 0")
    b = compute_b_pre([3, 2, 1], [5, -2], W_w=4, W_a=4)
    assert b == 0, f"Expected 0, got {b}"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 2: All-zero activations → b_pre must be 0 regardless of signs
    # ------------------------------------------------------------------
    print("\n[Test 2] All-zero activations → b_pre == 0")
    b = compute_b_pre([-3, -1], [0, 0], W_w=4, W_a=4)
    assert b == 0, f"Expected 0, got {b}"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 3: Single weight, single activation, negative weight
    # ------------------------------------------------------------------
    print("\n[Test 3] P_w=1, P_a=1, w=-1, a=3, W_w=4, W_a=4")
    # W_p = 4 + 4 - 1 = 7
    # contrib_i=0: neg_a = (-3) mod 16 = 13
    #              shift = 0*W_p + (W_w-1) = 3
    #              contrib = 13 << 3 = 104
    # inner_j=0   = 104
    # outer shift for j=0: 104 << 0 = 104
    # b_pre = 104
    b = compute_b_pre([-1], [3], W_w=4, W_a=4, verbose=True)
    neg_a_unsigned = (-3) % (2 ** 4)              # 13
    expected = neg_a_unsigned << (4 - 1)          # 13 << 3 = 104
    assert b == expected, f"Expected {expected}, got {b}"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 4: Verify b_pre absorbs exactly the preprocessing correction.
    #
    # From equation 1a, the full recovery is:
    #   w^s * a^s = w^u * a^u
    #               - 2^(W_w-1) * a^s * [w^s < 0]    ← preprocessing
    #               - 2^W_w * w^s * [a^s < 0]         ← postprocessing
    #               - 2^(2W_w-1) * [w^s<0][a^s<0]     ← postprocessing
    #
    # b_pre is designed so that the *unpacked slot* of (w_p * a_p + b_pre)
    # equals (w^u * a^u - 2^(W_w-1) * a^s * [w^s < 0]).
    # i.e. the preprocessing correction has already been folded in.
    # We verify this directly without touching postprocessing logic.
    # ------------------------------------------------------------------
    print("\n[Test 4] b_pre folds the preprocessing term into the DSP result")
    from core.mapping import clip_map, direct_map
    from core.packing import compute_wp, pack_weights, pack_activations, unpack_results

    W_w, W_a = 4, 4
    W_p = compute_wp(W_w, W_a)

    for w_s, a_s in [(-3, 5), (-1, -2), (3, -4), (-4, -4), (0, 7)]:
        w_u = clip_map(w_s, W_w)
        a_u = direct_map(a_s, W_a)

        w_p = pack_weights([w_u], W_w, W_a)
        a_p = pack_activations([a_u], P_w=1, W_w=W_w, W_a=W_a)
        b_pre = compute_b_pre([w_s], [a_s], W_w, W_a)

        # DSP multiply-add
        r_p = w_p * a_p + b_pre

        # Unpack raw slot (unsigned W_p bits)
        raw_slot = unpack_results(r_p, P_w=1, P_a=1, W_w=W_w, W_a=W_a)[0][0]

        # What the slot SHOULD contain after preprocessing (before postprocessing):
        #   w^u * a^u + neg_a * 2^(W_w-1) * [w^s < 0]
        # where neg_a = (-a^s) mod 2^W_a.  No overflow guaranteed so no mod needed.
        neg_a = (-a_s) % (2 ** W_a)
        preprocessing_correction = (neg_a << (W_w - 1)) * (1 if w_s < 0 else 0)
        expected_slot = w_u * a_u + preprocessing_correction

        assert raw_slot == expected_slot, (
            f"w_s={w_s}, a_s={a_s}: slot={raw_slot}, expected={expected_slot}"
        )
        print(f"  w_s={w_s:3d}, a_s={a_s:3d} → slot=0x{raw_slot:04x}  PASSED")

    # ------------------------------------------------------------------
    # Test 5: needs_preprocessing helper
    # ------------------------------------------------------------------
    print("\n[Test 5] needs_preprocessing helper")
    assert needs_preprocessing([-1, 2], W_w=4) is True
    assert needs_preprocessing([1, 2, 3], W_w=4) is False
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All preprocessing tests passed.")
    print("=" * 60)
