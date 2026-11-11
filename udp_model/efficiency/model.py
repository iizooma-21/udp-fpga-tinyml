"""
efficiency/model.py
===================
Parallelism constraint solver for the UDP packing framework.

Implements equations 6 and 7 from the paper to determine which
(P_w, P_a) packing factor pairs are feasible on a single DSP48E2,
and finds the pair that maximises the MAC parallelism P_w * P_a.

EQUATIONS 6 & 7 (port-width constraints)
-----------------------------------------
A (P_w, P_a) pair is valid when both packed operands fit within the
DSP48E2 hardware port widths:

    W_p  = W_w + W_a - 1                       (slot width)

    Eq 6:  W_pw = (P_w - 1) * W_p + (W_w - 1)  <=  W_short  (B port,   18 bits)
    Eq 7:  W_pa = (P_a - 1) * P_w * W_p + W_a  <=  W_long   (A/D port, 27 bits)

Parallelism and efficiency metrics
-----------------------------------
For a valid (P_w, P_a):

    parallelism  = P_w * P_a          MACs computed per DSP cycle
    short_util   = W_pw / W_short     fraction of B port used
    long_util    = W_pa / W_long      fraction of A/D port used
    bottleneck   = the port with higher utilisation

The solver finds the (P_w, P_a) that maximises parallelism.

Conventions
-----------
* DSP port widths default to Xilinx DSP48E2 values (W_short=18, W_long=27)
  but are parameterisable so the model generalises to other devices.
* All public functions accept an optional ``verbose`` flag (default False).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.packing import (
    compute_wp,
    _packed_weight_width,
    _packed_activation_width,
)
from core.dsp_sim import DSP_SHORT, DSP_LONG


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PackingResult:
    """Holds the efficiency metrics for one (P_w, P_a) configuration.

    Attributes
    ----------
    P_w          : int    Number of packed weights.
    P_a          : int    Number of packed activations.
    W_w          : int    Weight bitwidth.
    W_a          : int    Activation bitwidth.
    W_p          : int    Slot width  = W_w + W_a - 1.
    W_pw         : int    Required B  port bits  (eq 6 left-hand side).
    W_pa         : int    Required A/D port bits (eq 7 left-hand side).
    W_short      : int    B  port limit (default 18).
    W_long       : int    A/D port limit (default 27).
    parallelism  : int    MACs per DSP cycle  = P_w * P_a.
    short_util   : float  B  port utilisation  = W_pw / W_short.
    long_util    : float  A/D port utilisation  = W_pa / W_long.
    bottleneck   : str    Port with higher utilisation ('short' or 'long').
    """
    P_w:         int
    P_a:         int
    W_w:         int
    W_a:         int
    W_p:         int
    W_pw:        int
    W_pa:        int
    W_short:     int
    W_long:      int
    parallelism: int
    short_util:  float
    long_util:   float
    bottleneck:  str

    def __str__(self) -> str:
        return (
            f"PackingResult("
            f"P_w={self.P_w}, P_a={self.P_a}, "
            f"parallelism={self.parallelism}, "
            f"W_pw={self.W_pw}/{self.W_short} ({self.short_util:.1%}), "
            f"W_pa={self.W_pa}/{self.W_long} ({self.long_util:.1%}), "
            f"bottleneck={self.bottleneck})"
        )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_port_widths(
    P_w: int,
    P_a: int,
    W_w: int,
    W_a: int,
) -> Tuple[int, int, int]:
    """Return (W_p, W_pw, W_pa) for a given (P_w, P_a, W_w, W_a).

    Parameters
    ----------
    P_w : int   Number of packed weights.
    P_a : int   Number of packed activations.
    W_w : int   Weight bitwidth.
    W_a : int   Activation bitwidth.

    Returns
    -------
    (W_p, W_pw, W_pa)
        W_p  = W_w + W_a - 1
        W_pw = (P_w - 1)*W_p + (W_w - 1)   [eq 6 LHS]
        W_pa = (P_a - 1)*P_w*W_p + W_a      [eq 7 LHS]
    """
    W_p  = compute_wp(W_w, W_a)
    W_pw = _packed_weight_width(P_w, W_p, W_w)
    W_pa = _packed_activation_width(P_a, P_w, W_p, W_a)
    return W_p, W_pw, W_pa


def is_feasible(
    P_w: int,
    P_a: int,
    W_w: int,
    W_a: int,
    W_short: int = DSP_SHORT,
    W_long:  int = DSP_LONG,
) -> bool:
    """Return True when (P_w, P_a) satisfies equations 6 and 7.

    Parameters
    ----------
    P_w, P_a   : int   Candidate packing factors (>= 1).
    W_w, W_a   : int   Operand bitwidths.
    W_short    : int   B  port limit (default DSP48E2 = 18).
    W_long     : int   A/D port limit (default DSP48E2 = 27).
    """
    if P_w < 1 or P_a < 1:
        return False
    _, W_pw, W_pa = compute_port_widths(P_w, P_a, W_w, W_a)
    return W_pw <= W_short and W_pa <= W_long


def make_result(
    P_w: int,
    P_a: int,
    W_w: int,
    W_a: int,
    W_short: int = DSP_SHORT,
    W_long:  int = DSP_LONG,
) -> PackingResult:
    """Build a PackingResult for a feasible (P_w, P_a) pair."""
    W_p, W_pw, W_pa = compute_port_widths(P_w, P_a, W_w, W_a)
    short_util  = W_pw / W_short
    long_util   = W_pa / W_long
    bottleneck  = "short" if short_util >= long_util else "long"
    return PackingResult(
        P_w=P_w, P_a=P_a,
        W_w=W_w, W_a=W_a,
        W_p=W_p, W_pw=W_pw, W_pa=W_pa,
        W_short=W_short, W_long=W_long,
        parallelism=P_w * P_a,
        short_util=short_util,
        long_util=long_util,
        bottleneck=bottleneck,
    )


# ---------------------------------------------------------------------------
# Constraint solver
# ---------------------------------------------------------------------------

def find_all_valid(
    W_w: int,
    W_a: int,
    W_short: int = DSP_SHORT,
    W_long:  int = DSP_LONG,
    verbose: bool = False,
) -> List[PackingResult]:
    """Enumerate every feasible (P_w, P_a) pair (equations 6 and 7).

    Uses a nested loop: P_w from 1 upward until eq 6 fails; for each P_w,
    P_a from 1 upward until eq 7 fails.

    Parameters
    ----------
    W_w, W_a   : int   Operand bitwidths.
    W_short    : int   B  port width limit (default 18).
    W_long     : int   A/D port width limit (default 27).
    verbose    : bool  Print each accepted pair if True.

    Returns
    -------
    list of PackingResult  All valid configurations, sorted by
                           descending parallelism then descending P_w.
    """
    assert W_w >= 2, f"W_w must be >= 2 (clip mapping needs at least 2 bits), got {W_w}"
    assert W_a >= 1, f"W_a must be >= 1, got {W_a}"
    assert W_short >= 1 and W_long >= 1, "Port widths must be positive."

    W_p = compute_wp(W_w, W_a)
    results = []

    if verbose:
        print(f"find_all_valid: W_w={W_w}, W_a={W_a}, W_p={W_p}, "
              f"W_short={W_short}, W_long={W_long}")
        print(f"  Eq 6: W_pw = (P_w-1)*{W_p} + {W_w-1} <= {W_short}")
        print(f"  Eq 7: W_pa = (P_a-1)*P_w*{W_p} + {W_a}  <= {W_long}")
        print()

    P_w = 1
    while True:
        W_pw = _packed_weight_width(P_w, W_p, W_w)
        if W_pw > W_short:
            break               # eq 6 violated — no larger P_w can work

        P_a = 1
        while True:
            W_pa = _packed_activation_width(P_a, P_w, W_p, W_a)
            if W_pa > W_long:
                break           # eq 7 violated for this P_w — stop P_a loop

            result = make_result(P_w, P_a, W_w, W_a, W_short, W_long)
            results.append(result)

            if verbose:
                print(f"  VALID  P_w={P_w}, P_a={P_a}: "
                      f"W_pw={W_pw}/{W_short} ({result.short_util:.0%}), "
                      f"W_pa={W_pa}/{W_long} ({result.long_util:.0%}), "
                      f"parallelism={result.parallelism}")
            P_a += 1

        P_w += 1

    results.sort(key=lambda r: (-r.parallelism, -r.P_w))
    return results


def find_optimal(
    W_w: int,
    W_a: int,
    W_short: int = DSP_SHORT,
    W_long:  int = DSP_LONG,
    verbose: bool = False,
) -> Optional[PackingResult]:
    """Find the (P_w, P_a) pair that maximises P_w * P_a (equations 6 & 7).

    Ties are broken by preferring higher P_w (more weight reuse, which
    suits typical weight-stationary data-flows), then higher P_a.

    Parameters
    ----------
    W_w, W_a   : int   Operand bitwidths.
    W_short    : int   B  port width limit (default 18).
    W_long     : int   A/D port width limit (default 27).
    verbose    : bool  Print the full candidate table if True.

    Returns
    -------
    PackingResult   The optimal configuration.
    None            If no feasible pair exists (bitwidths too wide for the DSP).
    """
    candidates = find_all_valid(W_w, W_a, W_short, W_long, verbose=verbose)
    if not candidates:
        return None
    # Sorted list: first entry has maximum parallelism (ties broken by P_w).
    return candidates[0]


# ---------------------------------------------------------------------------
# Summary / report helper
# ---------------------------------------------------------------------------

def print_efficiency_table(
    W_w: int,
    W_a: int,
    W_short: int = DSP_SHORT,
    W_long:  int = DSP_LONG,
) -> None:
    """Print a human-readable table of all valid (P_w, P_a) configurations.

    Highlights the optimal row.

    Parameters
    ----------
    W_w, W_a   : int   Operand bitwidths.
    W_short    : int   B  port width limit (default 18).
    W_long     : int   A/D port width limit (default 27).
    """
    candidates = find_all_valid(W_w, W_a, W_short, W_long)
    optimal    = candidates[0] if candidates else None

    header = (f"Efficiency table  W_w={W_w}, W_a={W_a}, "
              f"W_short={W_short}, W_long={W_long}")
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    if not candidates:
        print("  No valid (P_w, P_a) pairs — operands too wide for this DSP.")
        return

    col = "{:>4} {:>4} {:>12} {:>10} {:>8} {:>10} {:>8} {:>12}"
    print(col.format("P_w", "P_a", "parallelism", "W_pw", "B util",
                     "W_pa", "A/D util", "bottleneck"))
    print("-" * 72)
    for r in candidates:
        marker = "  <-- optimal" if r is optimal else ""
        print(col.format(
            r.P_w, r.P_a,
            r.parallelism,
            f"{r.W_pw}/{r.W_short}",
            f"{r.short_util:.1%}",
            f"{r.W_pa}/{r.W_long}",
            f"{r.long_util:.1%}",
            r.bottleneck,
        ) + marker)

    print()
    if optimal:
        print(f"Optimal: P_w={optimal.P_w}, P_a={optimal.P_a} "
              f"-> {optimal.parallelism} MACs/DSP/cycle")


# ---------------------------------------------------------------------------
# Self-test  (run with: python -m efficiency.model)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Self-test: efficiency/model.py")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: int4 x int4 — paper's canonical example
    #   Expected optimal: P_w=3, P_a=2 (6 MACs/DSP/cycle)
    #   W_pw = 2*7 + 3 = 17 <= 18 (B  port)
    #   W_pa = 1*3*7 + 4 = 25 <= 27 (A/D port)
    # ------------------------------------------------------------------
    print("\n[Test 1] int4 x int4 -- paper canonical example")
    opt = find_optimal(W_w=4, W_a=4, verbose=True)
    print(f"\n  Optimal: {opt}")
    assert opt is not None,          "Expected a valid result"
    assert opt.P_w == 3,             f"Expected P_w=3, got {opt.P_w}"
    assert opt.P_a == 2,             f"Expected P_a=2, got {opt.P_a}"
    assert opt.parallelism == 6,     f"Expected parallelism=6, got {opt.parallelism}"
    assert opt.W_pw <= DSP_SHORT,    "B port violated"
    assert opt.W_pa <= DSP_LONG,     "A/D port violated"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 2: int8 x int8 — wider operands, lower parallelism
    # ------------------------------------------------------------------
    print("\n[Test 2] int8 x int8")
    opt8 = find_optimal(W_w=8, W_a=8, verbose=True)
    print(f"\n  Optimal: {opt8}")
    assert opt8 is not None
    # Sanity: W_pw and W_pa must not exceed port widths
    assert opt8.W_pw <= DSP_SHORT
    assert opt8.W_pa <= DSP_LONG
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 3: is_feasible spot-checks
    # ------------------------------------------------------------------
    print("\n[Test 3] is_feasible spot-checks")
    # P_w=3, P_a=2, int4: valid (from Test 1)
    assert is_feasible(3, 2, 4, 4) is True,  "3,2 int4 should be feasible"
    # P_w=4, P_a=1, int4: W_pw = 3*7+3 = 24 > 18, invalid
    assert is_feasible(4, 1, 4, 4) is False, "4,1 int4 should violate eq6"
    # P_w=1, P_a=1: always valid for any reasonable bitwidth
    assert is_feasible(1, 1, 4, 4) is True,  "1,1 always valid"
    assert is_feasible(1, 1, 8, 8) is True,  "1,1 int8 always valid"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 4: all valid pairs for int4 are correctly enumerated
    # ------------------------------------------------------------------
    print("\n[Test 4] All valid int4 x int4 pairs")
    all_valid = find_all_valid(W_w=4, W_a=4)
    # Every returned pair must satisfy both constraints
    errors = 0
    for r in all_valid:
        if not is_feasible(r.P_w, r.P_a, 4, 4):
            print(f"  FAIL: {r} marked valid but is_feasible says no")
            errors += 1
    # Sorted descending by parallelism
    for i in range(len(all_valid) - 1):
        assert all_valid[i].parallelism >= all_valid[i+1].parallelism
    print(f"  {len(all_valid)} valid pairs found, all constraints satisfied")
    if errors == 0:
        print("  PASSED")

    # ------------------------------------------------------------------
    # Test 5: full efficiency table for int4 x int4
    # ------------------------------------------------------------------
    print("\n[Test 5] Efficiency table for int4 x int4")
    print_efficiency_table(W_w=4, W_a=4)

    # ------------------------------------------------------------------
    # Test 6: efficiency table for int2 x int2 (sub-byte)
    # ------------------------------------------------------------------
    print("\n[Test 6] Efficiency table for int2 x int2 (sub-byte)")
    print_efficiency_table(W_w=2, W_a=2)
    opt2 = find_optimal(W_w=2, W_a=2)
    assert opt2 is not None
    assert opt2.W_pw <= DSP_SHORT
    assert opt2.W_pa <= DSP_LONG
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 7: no valid pair when operands are too wide (e.g. int32)
    # ------------------------------------------------------------------
    print("\n[Test 7] int32 x int32 -- too wide for one DSP48E2")
    result = find_optimal(W_w=32, W_a=32)
    assert result is None, f"Expected None for int32, got {result}"
    print("  PASSED (no valid config as expected)")

    # ------------------------------------------------------------------
    # Test 8: custom port widths
    # ------------------------------------------------------------------
    print("\n[Test 8] Custom DSP port widths (W_short=9, W_long=9)")
    opt_custom = find_optimal(W_w=4, W_a=4, W_short=9, W_long=9)
    print(f"  Optimal: {opt_custom}")
    if opt_custom is not None:
        assert opt_custom.W_pw <= 9
        assert opt_custom.W_pa <= 9
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All efficiency/model tests passed.")
    print("=" * 60)
