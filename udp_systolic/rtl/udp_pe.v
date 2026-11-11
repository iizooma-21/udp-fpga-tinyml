// =============================================================================
`timescale 1ns/1ps
// udp_pe.v
// A single UDP Processing Element (PE) for the int4 systolic array.
//
// WHAT IT DOES
// ============
// Each PE is responsible for ONE output tile of the matrix multiply result.
// With P_w=3 weights and P_a=2 activations packed into 1 DSP call, each PE
// produces 6 partial products per clock cycle (one for every weight-activation
// pair: w0×a0, w1×a0, w2×a0, w0×a1, w1×a1, w2×a1).
//
// Over K input cycles, those 6 products are ACCUMULATED, so after K cycles
// each accumulator holds one element of the output matrix.
//
// INSIDE THE PE
// =============
//
//   a_in0, a_in1  ─►  ┌───────────────────────────────┐
//   w_s0,1,2       ──►│  udp_mac_top (1 DSP48E2)      │─► 6 products (combinational)
//                      └───────────────────────────────┘
//                                                          │
//   a_in0 ─► [pipeline FF] ─► a_out0  (1-cycle delay)    ▼
//   a_in1 ─► [pipeline FF] ─► a_out1                  [6 × ACC_WIDTH adder + reg]
//                                                          │
//                                                          ▼
//                                               acc_00 .. acc_21  (ACC_WIDTH-bit)
//
// The pipeline FF on the activation outputs is the KEY to the systolic array:
// it introduces a 1-cycle delay so that PE[r][c+1] sees the activation one
// cycle later than PE[r][c]. This "skewing" lets all PEs accumulate the same
// K values in parallel — they just start one cycle apart per column.
//
// PORT NAMING CONVENTION
// ======================
//   acc_WA  where W = weight index (0,1,2)  and  A = activation index (0,1)
//   e.g. acc_20 = accumulated result for w_s2 × a_s0
//
// RESOURCE TARGET (per PE, before array-level opt)
// =================================================
//   DSPs : 1  (via udp_mac_top → dsp48e2_model)
//   LUTs : ~34  (pre/post-processing from single-PE project)
//   FFs  : 8 (activation pipeline) + 6×ACC_WIDTH (accumulators) = 8+96 = 104
// =============================================================================

module udp_pe #(
    // Accumulator bit-width.
    // For int4 × int4 (max |product| = 64), K_MAX = (2^(ACC_WIDTH-1) - 1) / 64:
    //   ACC_WIDTH = 12  →  K_MAX = 31   (matches paper Table 4 / ~14,660 LUTs for 144 DSPs)
    //   ACC_WIDTH = 13  →  K_MAX = 63
    //   ACC_WIDTH = 14  →  K_MAX = 127
    //   ACC_WIDTH = 16  →  K_MAX = 511
    parameter ACC_WIDTH = 12
) (
    input  wire        clk,
    input  wire        srst,   // Synchronous reset — clears all 6 accumulators to zero

    // Accumulate enable: when high, each rising edge adds the current 6 products
    // to the running accumulators.  When low, accumulators hold their value.
    input  wire        en,

    // ── Stationary Weights ────────────────────────────────────────────────────
    // These are held constant throughout the K accumulation cycles.
    // In a weight-stationary systolic array they are loaded once, then frozen.
    input  wire signed [3:0] w_s0, w_s1, w_s2,

    // ── Streaming Activations ─────────────────────────────────────────────────
    // One new activation column arrives each cycle from the left neighbour.
    input  wire signed [3:0] a_in0, a_in1,

    // ── Activation Passthrough ────────────────────────────────────────────────
    // Registered copy of a_in (1-cycle delay).  Feeds the RIGHT neighbour PE.
    // This is what makes the array "systolic": data ripples right, one cycle
    // per column.
    output reg  signed [3:0] a_out0, a_out1,

    // ── Accumulated Results ───────────────────────────────────────────────────
    // Valid K cycles after the last activation is fed (plus pipeline skew offset).
    output reg  signed [ACC_WIDTH-1:0] acc_00, acc_10, acc_20,  // × a_s0
    output reg  signed [ACC_WIDTH-1:0] acc_01, acc_11, acc_21   // × a_s1
);

    // =========================================================================
    // Combinational UDP MAC — 6 products in ONE DSP call
    // =========================================================================
    wire signed [7:0] p00, p10, p20, p01, p11, p21;

    udp_mac_top u_mac (
        .w_s0    (w_s0),  .w_s1    (w_s1),  .w_s2    (w_s2),
        .a_s0    (a_in0), .a_s1    (a_in1),
        .prod_00 (p00),   .prod_10 (p10),   .prod_20 (p20),
        .prod_01 (p01),   .prod_11 (p11),   .prod_21 (p21),
        // Observability ports unused at array level — left unconnected
        .dsp_B   (),      .dsp_A   (),      .dsp_C   (),      .dsp_P   ()
    );

    // =========================================================================
    // Activation pipeline register (systolic skewing)
    // Reset to 0 so the pipeline is fully clean before each computation.
    // Without this, uninitialized X bits propagate into the accumulators.
    // =========================================================================
    always @(posedge clk) begin
        if (srst) begin
            a_out0 <= 4'sh0;
            a_out1 <= 4'sh0;
        end else begin
            a_out0 <= a_in0;
            a_out1 <= a_in1;
        end
    end

    // =========================================================================
    // 6 Accumulators
    // Sign-extend each 8-bit product to ACC_WIDTH, then add to running sum.
    // =========================================================================
    // Sign-extension wires: copy the MSB (sign bit) into upper bits
    wire signed [ACC_WIDTH-1:0] sx00 = {{(ACC_WIDTH-8){p00[7]}}, p00};
    wire signed [ACC_WIDTH-1:0] sx10 = {{(ACC_WIDTH-8){p10[7]}}, p10};
    wire signed [ACC_WIDTH-1:0] sx20 = {{(ACC_WIDTH-8){p20[7]}}, p20};
    wire signed [ACC_WIDTH-1:0] sx01 = {{(ACC_WIDTH-8){p01[7]}}, p01};
    wire signed [ACC_WIDTH-1:0] sx11 = {{(ACC_WIDTH-8){p11[7]}}, p11};
    wire signed [ACC_WIDTH-1:0] sx21 = {{(ACC_WIDTH-8){p21[7]}}, p21};

    always @(posedge clk) begin
        if (srst) begin
            // Clear all accumulators (synchronous reset)
            acc_00 <= {ACC_WIDTH{1'b0}};
            acc_10 <= {ACC_WIDTH{1'b0}};
            acc_20 <= {ACC_WIDTH{1'b0}};
            acc_01 <= {ACC_WIDTH{1'b0}};
            acc_11 <= {ACC_WIDTH{1'b0}};
            acc_21 <= {ACC_WIDTH{1'b0}};
        end else if (en) begin
            // Accumulate: add this cycle's products to running totals
            acc_00 <= acc_00 + sx00;
            acc_10 <= acc_10 + sx10;
            acc_20 <= acc_20 + sx20;
            acc_01 <= acc_01 + sx01;
            acc_11 <= acc_11 + sx11;
            acc_21 <= acc_21 + sx21;
        end
        // else: en=0 → hold (no write, FF retains value)
    end

endmodule
