// =============================================================================
`timescale 1ns/1ps
// udp_pack.v
// Implements the mapping and packing phases of the UDP pipeline.
//
// MAPPING (Section 3.1 of the paper)
// ------------------------------------
//   Weights    → clip_map:   w_u_i = w_s_i[W_w-2:0]  (drop the sign bit)
//     The lower (W_w-1) bits of a two's-complement negative number equal
//     x + 2^(W_w-1), which is exactly what clip_map computes.  For positive
//     weights the MSB is already 0, so dropping it is lossless.
//
//   Activations → direct_map: a_u_j = a_s_j[W_a-1:0]  (same bit pattern)
//     Direct mapping reinterprets the signed bit pattern as unsigned with
//     no transformation.  In hardware this is just a wire.
//
// PACKING (Section 3.2 / Equation 2)
// ------------------------------------
// Weights are packed into the DSP's short (B) port:
//   w_p = w_u0 | (w_u1 << W_p) | (w_u2 << 2*W_p)
//   Bit layout [16:0]:  [2:0]=w_u0  [9:7]=w_u1  [16:14]=w_u2
//                       bits [6:3] and [13:10] are zero-padding guards
//
// Activations are packed into the DSP's long (A) port:
//   a_p = a_u0 | (a_u1 << P_w*W_p)
//   Bit layout [24:0]:  [3:0]=a_u0  [24:21]=a_u1
//                       bits [20:4] are zero-padding guards
//
// Parameters (fixed for int4 x int4, P_w=3, P_a=2):
//   W_w = 4  (weight bitwidth, signed)
//   W_a = 4  (activation bitwidth, signed)
//   W_p = 7  (W_w + W_a - 1  = slot width)
//   W_u_w = 3  (W_w - 1 = clip-mapped weight width)
//   W_pw = 17  ((P_w-1)*W_p + (W_w-1)  = packed weight width)
//   W_pa = 25  ((P_a-1)*P_w*W_p + W_a  = packed activation width)
// =============================================================================

module udp_pack (
    // Signed int4 inputs (weight slot indices 0..2, activation slot indices 0..1)
    input  wire signed [3:0] w_s0, w_s1, w_s2,   // 3 signed weights
    input  wire signed [3:0] a_s0, a_s1,           // 2 signed activations

    // Packed unsigned outputs for DSP ports
    output wire [16:0] w_p,   // packed weight  → B port [17:0] (upper bit unused)
    output wire [24:0] a_p,   // packed activation → A port [26:0] (upper bits unused)

    // Unsigned clip-mapped weights (needed by postprocessor for b_post)
    output wire [2:0] w_u0, w_u1, w_u2
);
    // -------------------------------------------------------------------------
    // CLIP MAPPING: drop the sign bit to get unsigned (W_w-1 = 3)-bit values
    //   clip_map(w_s) = w_s[2:0]
    //   For positives: MSB is 0, dropping it is identity.
    //   For negatives: w_s[2:0] = w_s + 8  (same as x + 2^(W_w-1)).
    // -------------------------------------------------------------------------
    assign w_u0 = w_s0[2:0];
    assign w_u1 = w_s1[2:0];
    assign w_u2 = w_s2[2:0];

    // DIRECT MAPPING: activations are already in the correct bit pattern.
    // a_u_j = a_s_j[3:0]  (just reinterpret; no transform needed in hardware)
    wire [3:0] a_u0 = a_s0[3:0];
    wire [3:0] a_u1 = a_s1[3:0];

    // -------------------------------------------------------------------------
    // PACK WEIGHTS into 17-bit w_p (→ B port)
    // Slot i starts at bit i*W_p = i*7.
    // w_u occupies (W_w-1)=3 bits; the remaining W_a=4 bits per slot are zeros.
    //
    //  w_p[16:0]:
    //    bits [2:0]   = w_u0  (slot 0 at offset 0)
    //    bits [6:3]   = 4'b0  (zero-padding)
    //    bits [9:7]   = w_u1  (slot 1 at offset 7)
    //    bits [13:10] = 4'b0  (zero-padding)
    //    bits [16:14] = w_u2  (slot 2 at offset 14)
    // -------------------------------------------------------------------------
    assign w_p = {w_u2, 4'b0, w_u1, 4'b0, w_u0};

    // -------------------------------------------------------------------------
    // PACK ACTIVATIONS into 25-bit a_p (→ A port)
    // Slot j starts at bit j*P_w*W_p = j*21.
    // a_u occupies W_a=4 bits; bits [20:4] are zeros.
    //
    //  a_p[24:0]:
    //    bits [3:0]   = a_u0  (slot 0 at offset 0)
    //    bits [20:4]  = 17'b0 (zero-padding)
    //    bits [24:21] = a_u1  (slot 1 at offset 21)
    // -------------------------------------------------------------------------
    assign a_p = {a_u1, 17'b0, a_u0};

endmodule
