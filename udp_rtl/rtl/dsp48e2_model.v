// =============================================================================
`timescale 1ns/1ps
// dsp48e2_model.v
// Dual-mode model of the DSP block for UDP.
//
// TARGET: Zynq UltraScale+ (ZCU106, xczu7ev-ffvc1156-2-e)
//   Physical DSP primitive on ZCU106 : DSP48E2
//   (Note: Vivado synthesis for xczu7ev uses DSP48E2, not DSP58E2)
//
// PORT MAP (int4 x int4, P_w=3, P_a=2):
//   w_p  → B port: 17-bit packed weights  (B port is 18-bit)
//   a_p  → A port: 25-bit packed activations (A port is 30-bit on DSP48E2)
//   b_pre → C port: 48-bit preprocessing correction (C port is 48-bit)
//   r    ← P port: 48-bit packed products  (P port is 48-bit)
//
// WHY BEHAVIORAL RTL ALWAYS USES 2 DSPs
// ----------------------------------------
// Verilog's expression-sizing rules apply the ASSIGNMENT CONTEXT WIDTH to all
// operands in an expression.  For "assign P = A * B + C" with P[47:0]:
//
//   Context width = 48 bits
//   → A[26:0] is zero-extended to 48 bits before multiply
//   → B[17:0] is zero-extended to 48 bits before multiply
//   → Vivado sees a 48×48 multiply → requires 2 DSPs (cascade mode)
//
// No amount of synthesis attributes (*use_dsp="yes"*) can overcome this
// because the attribute cannot change how Vivado interprets Verilog semantics.
//
// SOLUTION: ifdef SYNTHESIS split
// --------------------------------
//   SIMULATION (xsim — SYNTHESIS not defined):
//     Behavioral: "assign P = A * B + C" works correctly because Verilog's
//     zero-extension in the 48-bit context gives the exact same numerical
//     result as a 27×18 multiply.
//
    //   SYNTHESIS (Vivado — SYNTHESIS defined automatically by Vivado 2023.2):
    //     Direct DSP48E2 primitive instantiation.  All port widths are explicit
    //     and there is no ambiguity about operand widths.  Vivado maps to exactly
    //     1 DSP48E2 primitive with:
    //       A[29:0] ← {5'b0, a_p[24:0]}   (25-bit a_p, zero-padded to 30-bit)
    //       B[17:0] ← w_p (17-bit w_p, MSB=0 from udp_mac_top)
    //       C[47:0] ← b_pre (48-bit, exact fit)
    //       OPMODE  = 9'b001010101 → P = A×B + C
// =============================================================================

module dsp48e2_model (
    input  wire [17:0] B,    // w_p  — packed weights  (B port ≤18-bit)
    input  wire [26:0] A,    // a_p  — packed activations (A port ≤27-bit)
    input  wire [47:0] C,    // b_pre — preprocessing correction
    output wire [47:0] P     // r = A×B + C
);

`ifdef SYNTHESIS
    // =========================================================================
    // SYNTHESIS PATH — Direct DSP48E2 primitive instantiation
    // =========================================================================
    // DSP48E2 is the primitive used by Vivado for ZCU106 (xczu7ev-ffvc1156-2-e).
    // All pipeline registers disabled (XREG=0) → purely combinational output.
    //
    // OPMODE = 9'b001010101:
    //   W[8:7]=00 → 0 (no W term)
    //   Z[6:4]=101 → C (use C input as third addend)
    //   Y[3:2]=01  → M (use multiplier output as upper half)
    //   X[1:0]=01  → M (use multiplier output as lower half)
    //   ⇒  P = M + C  where M = A×B  (standard MAC mode)
    //
    // DSP48E2 port widths:
    //   A[29:0] : 30-bit  ← {5'b0, a_p[24:0]}  (a_p is 25 bits)
    //   B[17:0] : 18-bit  ← B (w_p is 17 bits, MSB=0 from udp_mac_top)
    //   C[47:0] : 48-bit  ← C (b_pre is exactly 48 bits)
    //   P[47:0] : 48-bit  → P
    // =========================================================================

    DSP48E2 #(
        .ACASCREG    (0),
        .ADREG       (0),
        .ALUMODEREG  (0),
        .AMULTSEL    ("A"),           // A port goes directly to multiplier
        .AREG        (0),
        .BCASCREG    (0),
        .BMULTSEL    ("B"),           // B port goes directly to multiplier
        .BREG        (0),
        .CARRYINREG  (0),
        .CARRYINSELREG(0),
        .CREG        (0),
        .DREG        (0),
        .INMODEREG   (0),
        .MREG        (0),
        .OPMODEREG   (0),
        .PREG        (0),
        .USE_MULT    ("MULTIPLY"),
        .USE_PATTERN_DETECT("NO_PATDET"),
        .USE_SIMD    ("ONE48"),       // Full 48-bit DSP (not SIMD split)
        .A_INPUT     ("DIRECT"),
        .B_INPUT     ("DIRECT"),
        .MASK        (48'h3FFFFFFFFFFF),
        .PATTERN     (48'h000000000000),
        .SEL_MASK    ("MASK"),
        .SEL_PATTERN ("PATTERN")
    ) dsp_inst (
        // Data inputs
        .A          ({3'b0, A}),           // 30-bit: a_p (27-bit, zero-padded to 30)
        .B          (B),                   // 18-bit: w_p (17-bit, MSB=0 from udp_mac_top)
        .C          (C),                   // 48-bit: b_pre (exact 48-bit fit)
        .D          (27'b0),               // Pre-adder D: not used
        // Control inputs
        .OPMODE     (9'b001010101),        // P = A×B + C (W=0,Z=C,Y=M,X=M)
        .ALUMODE    (4'b0000),             // Add
        .INMODE     (5'b00000),            // Standard: A to multiplier, B to multiplier
        .CARRYINSEL (3'b000),
        .CARRYIN    (1'b0),
        // Cascade inputs (not used)
        .ACIN       (30'b0),
        .BCIN       (18'b0),
        .PCIN       (48'b0),
        .CARRYCASCIN(1'b0),
        .MULTSIGNIN (1'b0),
        // Clock (irrelevant — PREG=0 → combinational P output)
        .CLK        (1'b0),
        // Clock enables (all 1 since no pipeline registers are used)
        .CEA1(1'b1), .CEA2(1'b1), .CEB1(1'b1), .CEB2(1'b1),
        .CEC (1'b1), .CED (1'b1), .CEAD(1'b1), .CEM(1'b1), .CEP(1'b1),
        .CEALUMODE(1'b1), .CECTRL(1'b1), .CEINMODE(1'b1), .CECARRYIN(1'b1),
        // Resets (all 0)
        .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0),
        .RSTM(1'b0), .RSTP(1'b0), .RSTALUMODE(1'b0),
        .RSTCTRL(1'b0), .RSTINMODE(1'b0), .RSTALLCARRYIN(1'b0),
        // Outputs
        .P          (P),                   // 48-bit result directly to module output
        // Unused outputs (safely left open — .port() means unconnected)
        .ACOUT(), .BCOUT(), .PCOUT(),
        .CARRYOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(),
        .PATTERNDETECT(), .PATTERNBDETECT(),
        .OVERFLOW(), .UNDERFLOW(), .XOROUT()
    );

`else
    // =========================================================================
    // SIMULATION PATH — Behavioral model (used by xsim)
    // =========================================================================
    // Functionally equivalent to the DSP primitive: P = A×B + C.
    // xsim does not have the DSP58E2 model without Vivado simulation libraries,
    // so we use this behavioral substitute for all simulation runs.
    //
    // Correctness: Verilog evaluates A*B+C in 48-bit context (from P's width).
    // Both A and B are zero-extended to 48 bits before multiply.  The actual
    // product bits beyond position 44 are zero (since A≤27 bits, B≤18 bits),
    // so the result is mathematically identical to a 27×18 MAC.
    // =========================================================================
    assign P = A * B + C;

`endif

endmodule
