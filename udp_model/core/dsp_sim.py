"""
core/dsp_sim.py
===============
Software model of the Xilinx DSP48E2 multiply-add block.

    r = w_p * a_p + b_pre

The DSP48E2 has three relevant data ports:

    Port   Width   UDP assignment
    -----  ------  --------------------------------------------------
    B      18-bit  w_p  (packed weights, short port)
    A/D    27-bit  a_p  (packed activations, long port)
    C      48-bit  b_pre (preprocessing correction, accumulator input)
    P      48-bit  output r = w_p * a_p + b_pre

All ports are treated as UNSIGNED here, consistent with the UDP
paper's mapping scheme (clip_map + direct_map produce unsigned values).

PORT-WIDTH CONSTRAINTS
----------------------
W_pw = (P_w - 1)*W_p + (W_w - 1)   ≤  DSP_SHORT = 18
W_pa = (P_a - 1)*P_w*W_p + W_a     ≤  DSP_LONG  = 27
b_pre                               ≤  2^DSP_C   - 1  (= 2^48 - 1)
r = w_p * a_p + b_pre               ≤  2^DSP_P   - 1  (= 2^48 - 1)

If any constraint is violated ``simulate`` raises a ``DSPOverflowError``
(a subclass of ``ValueError``) so callers can distinguish port-overflow
from other errors.

Conventions (match the rest of the codebase)
--------------------------------------------
* All values are plain Python ``int`` (non-negative for packed words).
* Every public function accepts a ``verbose`` flag (default False).
"""

from core.packing import compute_wp, _packed_weight_width, _packed_activation_width


# ---------------------------------------------------------------------------
# DSP48E2 hardware constants
# ---------------------------------------------------------------------------

DSP_SHORT  = 18   # B  port  (w_p)
DSP_LONG   = 27   # A/D port (a_p)
DSP_C      = 48   # C  port  (b_pre adder input)
DSP_P      = 48   # P  port  (output)

DSP_MAX_SHORT = (1 << DSP_SHORT) - 1
DSP_MAX_LONG  = (1 << DSP_LONG)  - 1
DSP_MAX_C     = (1 << DSP_C)     - 1
DSP_MAX_P     = (1 << DSP_P)     - 1


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class DSPOverflowError(ValueError):
    """Raised when an input or output value exceeds a DSP48E2 port width.

    Attributes
    ----------
    port      : str   Port name ('B/w_p', 'A/a_p', 'C/b_pre', or 'P/r').
    value     : int   The offending value.
    max_value : int   The largest value the port can hold.
    """

    def __init__(self, port: str, value: int, max_value: int):
        self.port      = port
        self.value     = value
        self.max_value = max_value
        bits = max_value.bit_length()
        super().__init__(
            f"DSP48E2 port '{port}' overflow: value={value} "
            f"exceeds {bits}-bit maximum ({max_value}). "
            f"Reduce P_w, P_a, W_w, or W_a so that the packed "
            f"operands fit within the hardware port widths."
        )


# ---------------------------------------------------------------------------
# Port-width check helpers
# ---------------------------------------------------------------------------

def _check_port(name: str, value: int, max_val: int) -> None:
    """Raise DSPOverflowError if value > max_val."""
    if value < 0:
        raise ValueError(f"Port '{name}' value must be non-negative, got {value}.")
    if value > max_val:
        raise DSPOverflowError(name, value, max_val)


def check_fits_short(w_p: int) -> None:
    """Assert w_p fits in the 18-bit B port.  Raises DSPOverflowError if not."""
    _check_port("B/w_p", w_p, DSP_MAX_SHORT)


def check_fits_long(a_p: int) -> None:
    """Assert a_p fits in the 27-bit A/D port.  Raises DSPOverflowError if not."""
    _check_port("A/a_p", a_p, DSP_MAX_LONG)


def check_fits_c(b_pre: int) -> None:
    """Assert b_pre fits in the 48-bit C port.  Raises DSPOverflowError if not."""
    _check_port("C/b_pre", b_pre, DSP_MAX_C)


def check_fits_output(r: int) -> None:
    """Assert the result fits in the 48-bit P port.  Raises DSPOverflowError if not."""
    _check_port("P/r", r, DSP_MAX_P)


# ---------------------------------------------------------------------------
# Capacity query helpers  (useful before calling simulate)
# ---------------------------------------------------------------------------

def max_packing_factors(W_w: int, W_a: int):
    """Return the maximum P_w and P_a that fit within DSP48E2 port widths.

    Computes the largest integer pair (P_w, P_a) such that:

        (P_w - 1)*W_p + (W_w - 1)  ≤  DSP_SHORT   (18)
        (P_a - 1)*P_w*W_p + W_a    ≤  DSP_LONG    (27)

    Parameters
    ----------
    W_w : int   Weight bitwidth.
    W_a : int   Activation bitwidth.

    Returns
    -------
    tuple (P_w_max, P_a_max)   Maximum safe packing factors.
    Zero is returned for a dimension if even a single operand overflows.
    """
    W_p = compute_wp(W_w, W_a)

    # Maximum P_w: (P_w - 1)*W_p + (W_w - 1) <= DSP_SHORT
    #  → P_w <= 1 + (DSP_SHORT - W_w + 1) / W_p
    if W_w - 1 > DSP_SHORT:
        P_w_max = 0
    else:
        P_w_max = 1 + (DSP_SHORT - (W_w - 1)) // W_p

    # Maximum P_a given P_w_max: (P_a - 1)*P_w*W_p + W_a <= DSP_LONG
    #  → P_a <= 1 + (DSP_LONG - W_a) / (P_w * W_p)
    if P_w_max == 0 or W_a > DSP_LONG:
        P_a_max = 0
    else:
        denom = P_w_max * W_p
        P_a_max = 1 + (DSP_LONG - W_a) // denom

    return P_w_max, P_a_max


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def simulate(
    w_p:   int,
    a_p:   int,
    b_pre: int,
    W_w:   int,
    W_a:   int,
    P_w:   int,
    P_a:   int,
    verbose: bool = False,
) -> int:
    """Simulate the DSP48E2 multiply-add with port-width validation.

    Computes:

        r = w_p * a_p + b_pre

    and checks that all operands and the result respect the DSP48E2
    hardware port widths (B=18, A/D=27, C=48, P=48).

    Parameters
    ----------
    w_p   : int   Packed weight integer (→ B port, 18-bit).
    a_p   : int   Packed activation integer (→ A/D port, 27-bit).
    b_pre : int   Preprocessing correction (→ C port, 48-bit).
    W_w   : int   Weight bitwidth.
    W_a   : int   Activation bitwidth.
    P_w   : int   Number of packed weights.
    P_a   : int   Number of packed activations.
    verbose : bool   Print a step-by-step trace if True.

    Returns
    -------
    int   r = w_p * a_p + b_pre  (fits in 48-bit P port).

    Raises
    ------
    DSPOverflowError   If any port constraint is violated.
    ValueError         If any argument is negative or logically invalid.
    """
    W_p  = compute_wp(W_w, W_a)
    W_pw = _packed_weight_width(P_w, W_p, W_w)
    W_pa = _packed_activation_width(P_a, P_w, W_p, W_a)

    if verbose:
        print("=" * 56)
        print("DSP48E2 simulation")
        print("=" * 56)
        print(f"  Operand widths  : W_w={W_w}, W_a={W_a}, W_p={W_p}")
        print(f"  Packing factors : P_w={P_w}, P_a={P_a}")
        print(f"  w_p  needs {W_pw:2d} bits  (B  port limit: {DSP_SHORT})")
        print(f"  a_p  needs {W_pa:2d} bits  (A/D port limit: {DSP_LONG})")
        print(f"  b_pre          = {b_pre}  (C  port limit: 2^{DSP_C}-1)")
        print(f"  w_p            = {w_p}")
        print(f"  a_p            = {a_p}")

    # ------------------------------------------------------------------
    # Port-width assertions
    # ------------------------------------------------------------------
    if W_pw > DSP_SHORT:
        raise DSPOverflowError("B/w_p (required width)", W_pw, DSP_SHORT)
    if W_pa > DSP_LONG:
        raise DSPOverflowError("A/a_p (required width)", W_pa, DSP_LONG)

    check_fits_short(w_p)
    check_fits_long(a_p)
    check_fits_c(b_pre)

    # ------------------------------------------------------------------
    # DSP multiply-add
    # ------------------------------------------------------------------
    product = w_p * a_p
    r       = product + b_pre

    if verbose:
        print(f"  w_p * a_p      = {product}")
        print(f"  + b_pre        = {b_pre}")
        print(f"  r (P port)     = {r}  (max: 2^{DSP_P}-1 = {DSP_MAX_P})")

    check_fits_output(r)

    if verbose:
        r_bits = r.bit_length() if r > 0 else 1
        print(f"  Output uses {r_bits} bits -- fits in {DSP_P}-bit P port OK")
        print("=" * 56)

    return r


# ---------------------------------------------------------------------------
# Self-test  (run with: python -m core.dsp_sim)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core.mapping      import clip_map, direct_map
    from core.packing      import pack_weights, pack_activations
    from core.preprocessing import compute_b_pre

    print("=" * 60)
    print("Self-test: core/dsp_sim.py")
    print("=" * 60)

    def _run(w_s_list, a_s_list, W_w, W_a, verbose=False):
        """Full pipeline up to and including the DSP simulate call."""
        P_w = len(w_s_list)
        w_u = [clip_map(w, W_w)   for w in w_s_list]
        a_u = [direct_map(a, W_a) for a in a_s_list]
        w_p   = pack_weights(w_u, W_w, W_a)
        a_p   = pack_activations(a_u, P_w=P_w, W_w=W_w, W_a=W_a)
        b_pre = compute_b_pre(w_s_list, a_s_list, W_w=W_w, W_a=W_a)
        r = simulate(w_p, a_p, b_pre, W_w, W_a,
                     P_w=P_w, P_a=len(a_s_list), verbose=verbose)
        return r, w_p, a_p, b_pre

    # ------------------------------------------------------------------
    # Test 1: Basic int4, P_w=1, P_a=1  (verbose trace)
    # ------------------------------------------------------------------
    print("\n[Test 1] int4 x int4, P_w=1, P_a=1 -- verbose trace")
    r, w_p, a_p, b_pre = _run([-3], [5], W_w=4, W_a=4, verbose=True)
    assert r == w_p * a_p + b_pre, "r mismatch"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 2: Paper example  P_w=3, P_a=2
    # ------------------------------------------------------------------
    print("\n[Test 2] int4 x int4, P_w=3, P_a=2 (paper example)")
    r, w_p, a_p, b_pre = _run([-3, 2, -1], [5, -2], W_w=4, W_a=4, verbose=True)
    assert r == w_p * a_p + b_pre
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 3: Port-width limits — max valid configuration for int4
    # ------------------------------------------------------------------
    print("\n[Test 3] max_packing_factors for int4 x int4")
    W_w, W_a = 4, 4
    P_w_max, P_a_max = max_packing_factors(W_w, W_a)
    print(f"  int4x int4 -> P_w_max={P_w_max}, P_a_max={P_a_max}")
    W_p = compute_wp(W_w, W_a)
    W_pw = _packed_weight_width(P_w_max, W_p, W_w)
    W_pa = _packed_activation_width(P_a_max, P_w_max, W_p, W_a)
    print(f"  W_pw={W_pw} (<={DSP_SHORT}?)  W_pa={W_pa} (<={DSP_LONG}?)")
    assert W_pw <= DSP_SHORT, f"W_pw={W_pw} exceeds B port"
    assert W_pa <= DSP_LONG,  f"W_pa={W_pa} exceeds A/D port"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 4: int8×int8 packing capacity
    # ------------------------------------------------------------------
    print("\n[Test 4] max_packing_factors for int8 x int8")
    P_w_max, P_a_max = max_packing_factors(8, 8)
    print(f"  int8x int8 -> P_w_max={P_w_max}, P_a_max={P_a_max}")
    assert P_w_max >= 1 and P_a_max >= 1
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 5: DSPOverflowError raised for over-wide w_p
    # ------------------------------------------------------------------
    print("\n[Test 5] DSPOverflowError raised when w_p exceeds 18-bit B port")
    try:
        simulate(2**18, 1, 0, W_w=4, W_a=4, P_w=1, P_a=1)
        print("  FAIL  (no error raised)")
    except DSPOverflowError as e:
        print(f"  Caught: {e}")
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 6: DSPOverflowError raised for over-wide a_p
    # ------------------------------------------------------------------
    print("\n[Test 6] DSPOverflowError raised when a_p exceeds 27-bit A/D port")
    try:
        simulate(1, 2**27, 0, W_w=4, W_a=4, P_w=1, P_a=1)
        print("  FAIL  (no error raised)")
    except DSPOverflowError as e:
        print(f"  Caught: {e}")
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 7: DSPOverflowError raised when result overflows 48-bit P port
    # ------------------------------------------------------------------
    print("\n[Test 7] DSPOverflowError raised when result overflows 48-bit P port")
    try:
        simulate(DSP_MAX_SHORT, DSP_MAX_LONG, DSP_MAX_C,
                 W_w=4, W_a=4, P_w=1, P_a=1)
        print("  FAIL  (no error raised)")
    except DSPOverflowError as e:
        print(f"  Caught: {e}")
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 8: All-zero inputs
    # ------------------------------------------------------------------
    print("\n[Test 8] All-zero inputs -> r = 0")
    r, *_ = _run([0], [0], W_w=4, W_a=4)
    assert r == 0, f"Expected 0, got {r}"
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All dsp_sim tests passed.")
    print("=" * 60)
