// =============================================================================
`timescale 1ns/1ps
// udp_preprocess.v
// Computes the preprocessing correction term b_pre (Equation 3 of the
// UDP paper) that is fed into the DSP's C port before multiplication.
//
// BACKGROUND
// -----------
// After clip_map + direct_map, a signed-signed product is recovered as:
//
//   w^s * a^s = w^u * a^u
//               - 2^(W_w-1) * a^s * [w^s < 0]    ← PREPROCESSING term
//               - 2^W_w     * w^s * [a^s < 0]     ← postprocessing term A
//               - 2^(2W_w-1)* [w^s<0][a^s<0]      ← postprocessing term B
//
// The preprocessing term depends only on the sign of each weight and the
// value of each activation — both known BEFORE the DSP multiply.
// UDP folds this into the DSP's adder port so the single operation:
//
//   r = w_p * a_p + b_pre
//
// absorbs the correction at zero extra hardware cost.
//
// FORMULA (corrected from paper's Equation 3; the paper has a typo — see below)
// ------------------------------------------------------------------------------
// For each activation j and each weight i where w_s_i < 0:
//
//   contrib(i,j) = neg_a_j  <<  (i*W_p + (W_w-1) + j*P_w*W_p)
//
// where neg_a_j = (-a_s_j) mod 2^W_a  (4-bit negate, computed as ~a_s_j + 1)
//
// PAPER TYPO NOTE: Equation 3 writes j-1 in the outer shift; the correct
// value is j (otherwise j=0 would produce a negative shift).  This
// implementation uses the corrected formula.
//
// PARAMETERS (fixed: int4 x int4, P_w=3, P_a=2)
//   W_w=4, W_a=4, W_p=7, P_w=3, P_a=2
//
// Shift table (shift = i*7 + 3 + j*21):
//   (i=0,j=0) = 3    (i=1,j=0) = 10   (i=2,j=0) = 17
//   (i=0,j=1) = 24   (i=1,j=1) = 31   (i=2,j=1) = 38
// =============================================================================

module udp_preprocess (
    input  wire signed [3:0] w_s0, w_s1, w_s2,   // 3 signed int4 weights
    input  wire signed [3:0] a_s0, a_s1,           // 2 signed int4 activations

    output wire [47:0] b_pre                        // preprocessing correction → C port
);
    // -------------------------------------------------------------------------
    // Compute neg_a_j = (-a_s_j) mod 16  for each activation.
    // In 4-bit two's-complement arithmetic: ~x + 1 gives -x mod 16.
    //   e.g.  a_s = -3 (1101) → ~1101 = 0010, +1 = 0011 = 3  → neg_a = 3
    //         a_s =  5 (0101) → ~0101 = 1010, +1 = 1011 = 11 → neg_a = 11
    //         a_s = -8 (1000) → ~1000 = 0111, +1 = 1000 = 8  → neg_a = 8
    //         a_s =  0 (0000) → ~0000 = 1111, +1 = 0000 = 0  → neg_a = 0  ✓
    // -------------------------------------------------------------------------
    wire [3:0] neg_a0 = ~a_s0 + 4'd1;
    wire [3:0] neg_a1 = ~a_s1 + 4'd1;

    // Extend to 48 bits for shifting without width-mismatch issues.
    wire [47:0] neg_a0_ext = {44'b0, neg_a0};
    wire [47:0] neg_a1_ext = {44'b0, neg_a1};

    // Sign bits of weights (1 = negative, 0 = positive/zero)
    wire s0 = w_s0[3];
    wire s1 = w_s1[3];
    wire s2 = w_s2[3];

    // -------------------------------------------------------------------------
    // b_pre is the OR-sum of all active contributions.
    // Slots are non-overlapping by construction (W_p-bit spacing), so OR = ADD.
    //
    // Activation j=0 (shifts 3, 10, 17):
    //   Each neg_a0 (4-bit) is placed at bit (i*W_p + W_w-1) = (i*7 + 3).
    //   Occupies bits [i*7+6 : i*7+3] — entirely within slot i's 7-bit window.
    //
    // Activation j=1 (shifts 24, 31, 38):
    //   Same pattern but shifted right by P_w*W_p = 21 to reach group j=1.
    // -------------------------------------------------------------------------
    assign b_pre =
        (s0 ? (neg_a0_ext << 3'd3)   : 48'b0) |   // (i=0, j=0)
        (s1 ? (neg_a0_ext << 4'd10)  : 48'b0) |   // (i=1, j=0)
        (s2 ? (neg_a0_ext << 5'd17)  : 48'b0) |   // (i=2, j=0)
        (s0 ? (neg_a1_ext << 5'd24)  : 48'b0) |   // (i=0, j=1)
        (s1 ? (neg_a1_ext << 5'd31)  : 48'b0) |   // (i=1, j=1)
        (s2 ? (neg_a1_ext << 6'd38)  : 48'b0);    // (i=2, j=1)

endmodule
