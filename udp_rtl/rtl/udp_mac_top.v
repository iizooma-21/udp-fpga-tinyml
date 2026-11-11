// =============================================================================
`timescale 1ns/1ps
// udp_mac_top.v
// Top-level UDP MAC module for int4 x int4, P_w=3, P_a=2.
//
// Integrates the complete UDP pipeline in a single combinational datapath:
//
//   INPUTS: 3 signed int4 weights + 2 signed int4 activations
//   OUTPUTS: 6 exact signed products (one per weight-activation pair)
//
// Pipeline stages (all combinational — 1 DSP operation):
//
//   [1] Mapping
//       clip_map:   w_u_i = w_s_i[2:0]   (drop sign bit)
//       direct_map: a_u_j = a_s_j[3:0]   (reinterpret bits as unsigned)
//
//   [2] Packing
//       w_p = {w_u2, 4'b0, w_u1, 4'b0, w_u0}        → B port (17 bits)
//       a_p = {a_u1, 17'b0, a_u0}                     → A port (25 bits)
//
//   [3] Preprocessing  (b_pre = Equation 3)
//       For each (i,j) where w_s_i < 0:
//         b_pre |= neg_a_j << (i*7 + 3 + j*21)       → C port (48 bits)
//
//   [4] DSP48E2 multiply-add
//       r = w_p * a_p + b_pre                          → P port (48 bits)
//
//   [5] Postprocessing  (Equation 4)
//       For each slot: result = slot_val - b_post_ij   → 6 signed products
//
// LOOP-COUNT COMPARISON (see testbench):
//   Test A (standard):  6 separate MAC operations → 6 loop iterations
//   Test B (UDP):       1 DSP call         → 1 iteration, 6 products
//   Reduction:          6x
// =============================================================================

module udp_mac_top (
    // 3 signed int4 weights
    input  wire signed [3:0] w_s0, w_s1, w_s2,

    // 2 signed int4 activations
    input  wire signed [3:0] a_s0, a_s1,

    // 6 signed products: prod_ij = w_si * a_sj
    output wire signed [7:0] prod_00,   // w_s0 * a_s0
    output wire signed [7:0] prod_10,   // w_s1 * a_s0
    output wire signed [7:0] prod_20,   // w_s2 * a_s0
    output wire signed [7:0] prod_01,   // w_s0 * a_s1
    output wire signed [7:0] prod_11,   // w_s1 * a_s1
    output wire signed [7:0] prod_21,   // w_s2 * a_s1

    // DSP port observability (exposed for testbench inspection)
    output wire [17:0] dsp_B,           // w_p on B port
    output wire [26:0] dsp_A,           // a_p on A port
    output wire [47:0] dsp_C,           // b_pre on C port
    output wire [47:0] dsp_P            // raw r on P port
);
    // -------------------------------------------------------------------------
    // Internal wires
    // -------------------------------------------------------------------------
    wire [16:0] w_p_int;
    wire [24:0] a_p_int;
    wire [47:0] b_pre_int;
    wire [47:0] r_int;
    wire [2:0]  w_u0_int, w_u1_int, w_u2_int;

    // -------------------------------------------------------------------------
    // Stage 1+2: Mapping + Packing
    // -------------------------------------------------------------------------
    udp_pack u_pack (
        .w_s0 (w_s0),  .w_s1 (w_s1),  .w_s2 (w_s2),
        .a_s0 (a_s0),  .a_s1 (a_s1),
        .w_p  (w_p_int),
        .a_p  (a_p_int),
        .w_u0 (w_u0_int), .w_u1 (w_u1_int), .w_u2 (w_u2_int)
    );

    // -------------------------------------------------------------------------
    // Stage 3: Preprocessing — compute b_pre
    // -------------------------------------------------------------------------
    udp_preprocess u_pre (
        .w_s0  (w_s0),  .w_s1  (w_s1),  .w_s2  (w_s2),
        .a_s0  (a_s0),  .a_s1  (a_s1),
        .b_pre (b_pre_int)
    );

    // -------------------------------------------------------------------------
    // Stage 4: DSP48E2 multiply-add  r = w_p * a_p + b_pre
    // -------------------------------------------------------------------------
    dsp48e2_model u_dsp (
        .B (dsp_B),   // = {1'b0, w_p_int}
        .A (dsp_A),   // = {2'b0, a_p_int}
        .C (dsp_C),   // = b_pre_int
        .P (dsp_P)    // = r_int
    );

    assign dsp_B = {1'b0, w_p_int};
    assign dsp_A = {2'b0, a_p_int};
    assign dsp_C = b_pre_int;
    assign r_int = dsp_P;

    // -------------------------------------------------------------------------
    // Stage 5: Postprocessing — recover 6 signed products
    // -------------------------------------------------------------------------
    udp_postprocess u_post (
        .r      (r_int),
        .w_s0   (w_s0),   .w_s1   (w_s1),   .w_s2   (w_s2),
        .w_u0   (w_u0_int), .w_u1 (w_u1_int), .w_u2 (w_u2_int),
        .a_s0   (a_s0),   .a_s1   (a_s1),
        .prod_00(prod_00), .prod_10(prod_10), .prod_20(prod_20),
        .prod_01(prod_01), .prod_11(prod_11), .prod_21(prod_21)
    );

endmodule
