// =============================================================================
`timescale 1ns/1ps
// udp_postprocess.v
// Recovers 6 exact signed products from the packed DSP result (Equation 4).
//
// SLOT LAYOUT (W_p=7, P_w=3, P_a=2):
//   slot (i=0,j=0) : r[6:0]    →  w_s0 * a_s0
//   slot (i=1,j=0) : r[13:7]   →  w_s1 * a_s0
//   slot (i=2,j=0) : r[20:14]  →  w_s2 * a_s0
//   slot (i=0,j=1) : r[27:21]  →  w_s0 * a_s1
//   slot (i=1,j=1) : r[34:28]  →  w_s1 * a_s1
//   slot (i=2,j=1) : r[41:35]  →  w_s2 * a_s1
//
// SIMPLIFIED b_post FORMULA (derived from Equation 4)
// ----------------------------------------------------
//   b_post_ij = 128          if (w_si < 0) AND (a_sj > 0)
//             = 16 * w_u_i   if (a_sj < 0)
//             = 0            otherwise
//
// KEY LUT OPTIMISATION
// --------------------
// Both correction values (128 and 16*w_u_i) have ZERO in their lower 4 bits:
//   128         = 8'b1000_0000  → bits[3:0] = 4'b0000
//   16 * w_u_i  = {1'b0, w_u_i, 4'b0}  → bits[3:0] = 4'b0000
//
// Therefore: b_post[3:0] = 4'b0  ALWAYS.
//
// This means:
//   result[3:0] = slot_val[3:0]              ← pure wire, 0 LUTs
//   result[8:4] = {0,0,sv[6:4]} - {0,bp[7:4]} ← only 5-bit subtraction
//
// The 5-bit subtraction replaces the original 9-bit subtraction, saving
// 4 carry-chain LUTs per slot × 6 slots = 24 LUTs.
//
// b_post upper nibble:
//   bp[7]   = w_si[3] & a_pos_j        (bit 7 = 1 for the 128 correction)
//   bp[6:4] = a_neg_j ? w_u_i : 3'b0   (bits 6:4 = w_u_i for 16*w_u)
//   bp[3:0] = 4'b0                     (always)
//
// ACTIVATION SIGN FLAGS
// ----------------------
//   a_neg_j = a_sj[3]                        (1 if a_sj < 0)
//   a_pos_j = !a_sj[3] && |a_sj[2:0]        (1 if a_sj > 0, using 3-input OR)
//             Note: |a_sj[2:0] is cheaper than (a_sj != 4'b0) — same result
//             since a_sj[3]=0 already guarantees a_sj >= 0.
// =============================================================================

module udp_postprocess (
    input  wire [47:0]       r,

    input  wire signed [3:0] w_s0, w_s1, w_s2,
    input  wire        [2:0] w_u0, w_u1, w_u2,
    input  wire signed [3:0] a_s0, a_s1,

    output wire signed [7:0] prod_00,
    output wire signed [7:0] prod_10,
    output wire signed [7:0] prod_20,
    output wire signed [7:0] prod_01,
    output wire signed [7:0] prod_11,
    output wire signed [7:0] prod_21
);
    // -------------------------------------------------------------------------
    // Extract 7-bit slot values
    // -------------------------------------------------------------------------
    wire [6:0] slot_00 = r[6:0];
    wire [6:0] slot_10 = r[13:7];
    wire [6:0] slot_20 = r[20:14];
    wire [6:0] slot_01 = r[27:21];
    wire [6:0] slot_11 = r[34:28];
    wire [6:0] slot_21 = r[41:35];

    // -------------------------------------------------------------------------
    // Activation sign flags (shared across all weight slots for each activation)
    //   a_neg_j = 1 if a_sj < 0  (just the MSB)
    //   a_pos_j = 1 if a_sj > 0  (MSB=0 AND any of bits[2:0] set)
    //             Using |a_sj[2:0] instead of (a_sj != 0): saves 1 LUT input
    // -------------------------------------------------------------------------
    wire a_neg0 = a_s0[3];
    wire a_neg1 = a_s1[3];
    wire a_pos0 = ~a_s0[3] & |a_s0[2:0];
    wire a_pos1 = ~a_s1[3] & |a_s1[2:0];

    // -------------------------------------------------------------------------
    // Compute b_post upper nibble for each slot.
    // b_post[3:0] is always 0 — not computed (saves LUTs via wire-through below)
    //
    //   bp_hi[3]   = bp[7]   = w_si[3] & a_pos_j   (the "128" case)
    //   bp_hi[2:0] = bp[6:4] = a_neg_j ? w_u_i : 0 (the "16*w_u" case)
    //
    // These two cases are mutually exclusive (a_pos_j and a_neg_j cannot both
    // be 1 for the same activation), so OR and priority mux are equivalent.
    // -------------------------------------------------------------------------

    // --- Activation j=0 ---
    wire [3:0] bp_hi_00 = {w_s0[3] & a_pos0,  a_neg0 ? w_u0 : 3'b0};
    wire [3:0] bp_hi_10 = {w_s1[3] & a_pos0,  a_neg0 ? w_u1 : 3'b0};
    wire [3:0] bp_hi_20 = {w_s2[3] & a_pos0,  a_neg0 ? w_u2 : 3'b0};

    // --- Activation j=1 ---
    wire [3:0] bp_hi_01 = {w_s0[3] & a_pos1,  a_neg1 ? w_u0 : 3'b0};
    wire [3:0] bp_hi_11 = {w_s1[3] & a_pos1,  a_neg1 ? w_u1 : 3'b0};
    wire [3:0] bp_hi_21 = {w_s2[3] & a_pos1,  a_neg1 ? w_u2 : 3'b0};

    // -------------------------------------------------------------------------
    // Recover each product using the optimised subtraction.
    //
    // result[3:0] = slot_val[3:0]        (wire-through, 0 LUTs)
    // result[8:4] = {0,0,sv[6:4]} - {0,bp_hi}  (5-bit subtraction)
    // result[7:0] taken as the signed product (fits in 8 bits for int4 range)
    //
    // The 5-bit subtraction uses Vivado's carry chain: ~2-3 LUTs per slot
    // vs ~4-5 LUTs for the previous 9-bit subtraction.
    // -------------------------------------------------------------------------

    wire signed [4:0] hi_00 = {2'b00, slot_00[6:4]} - {1'b0, bp_hi_00};
    assign prod_00 = {hi_00[3:0], slot_00[3:0]};

    wire signed [4:0] hi_10 = {2'b00, slot_10[6:4]} - {1'b0, bp_hi_10};
    assign prod_10 = {hi_10[3:0], slot_10[3:0]};

    wire signed [4:0] hi_20 = {2'b00, slot_20[6:4]} - {1'b0, bp_hi_20};
    assign prod_20 = {hi_20[3:0], slot_20[3:0]};

    wire signed [4:0] hi_01 = {2'b00, slot_01[6:4]} - {1'b0, bp_hi_01};
    assign prod_01 = {hi_01[3:0], slot_01[3:0]};

    wire signed [4:0] hi_11 = {2'b00, slot_11[6:4]} - {1'b0, bp_hi_11};
    assign prod_11 = {hi_11[3:0], slot_11[3:0]};

    wire signed [4:0] hi_21 = {2'b00, slot_21[6:4]} - {1'b0, bp_hi_21};
    assign prod_21 = {hi_21[3:0], slot_21[3:0]};

endmodule
