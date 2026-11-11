// =============================================================================
`timescale 1ns/1ps
// systolic_array.v
// 12×12 weight-stationary int4 systolic array using UDP MAC.
//
// ─────────────────────────────────────────────────────────────────────────────
// HIGH-LEVEL PICTURE
// ─────────────────────────────────────────────────────────────────────────────
// The array computes a tiled matrix multiplication  C = A × B  where:
//
//   A  (activations) : (PE_ROWS×P_A)  ×  K  =  24 rows × K columns
//   B  (weights)     : K              ×  (PE_COLS×P_W) =  K rows × 36 columns
//   C  (result)      : 24 × 36   — each element is the dot product of one
//                                  row of A with one column of B
//
// With PE_ROWS=12, PE_COLS=12 and P_A=2, P_W=3:
//   • 144 PEs are arranged in a 12×12 grid
//   • Each PE at (r, c) is responsible for a 2×3 tile of C:
//       rows  {2r, 2r+1}  ×  columns  {3c, 3c+1, 3c+2}
//   • One DSP48E2 per PE  →  144 DSPs total
//
// ─────────────────────────────────────────────────────────────────────────────
// DATA FLOW (weight-stationary)
// ─────────────────────────────────────────────────────────────────────────────
//
//         act_flat ──► col 0  ──►  col 1  ──►  col 2  ──► ... ──►  col 11
//   row 0:             PE[0][0]──► PE[0][1]──► PE[0][2]  ...      PE[0][11]
//   row 1:             PE[1][0]    PE[1][1]    PE[1][2]            PE[1][11]
//   ...                  ...         ...         ...                  ...
//   row 11:            PE[11][0]   PE[11][1]   PE[11][2]           PE[11][11]
//
//   Activations (a_in) travel LEFT → RIGHT through each row.
//   Each PE passes its received activation through a 1-cycle pipeline FF to the
//   next column.  This means:
//     • PE[r][0]  sees the input activation at time t
//     • PE[r][1]  sees it at time t+1   (one cycle later)
//     • PE[r][c]  sees it at time t+c
//
//   Weights are STATIONARY: driven continuously from the weights_flat port.
//   In a real system these would come from weight-register banks, but for
//   synthesis the compiler just sees them as inputs (no statefulness assumed).
//
// ─────────────────────────────────────────────────────────────────────────────
// TIMING PROTOCOL
// ─────────────────────────────────────────────────────────────────────────────
//
//  1. Assert srst=1 for at least 1 cycle to zero all 144×6 accumulators.
//  2. Deassert srst, then assert en=1.
//  3. Feed K activation columns via act_flat, one per clock cycle.
//     After the last valid activation, pad with PE_COLS-1 = 11 zero cycles
//     so that the skewed PEs at higher column indices also see all K values.
//     Total en-high duration: K + PE_COLS - 1 = K + 11 cycles.
//  4. Deassert en.
//  5. result_flat now contains 144×6 = 864 valid accumulated products.
//
// ─────────────────────────────────────────────────────────────────────────────
// PORT LAYOUT (flattened vectors — Verilog-2001 has no 2D port arrays)
// ─────────────────────────────────────────────────────────────────────────────
//
//  weights_flat  [PE_ROWS × PE_COLS × 3 × 4 - 1 : 0]   = 1728 bits
//    Access:  weights_flat[(r*PE_COLS + c)*12 + w*4 +: 4]
//             gives the 4-bit signed weight w (0,1,2) of PE at row r, col c.
//
//  act_flat  [PE_ROWS × 2 × 4 - 1 : 0]   = 96 bits
//    Access:  act_flat[(r*2 + a)*4 +: 4]
//             gives the 4-bit signed activation a (0,1) for row r.
//
//  result_flat  [PE_ROWS × PE_COLS × 6 × ACC_WIDTH - 1 : 0]
//    Access:  result_flat[(r*PE_COLS + c)*6*ACC_WIDTH + p*ACC_WIDTH +: ACC_WIDTH]
//             gives accumulated product p (0..5) of PE(r,c).
//    Product index mapping:  0=acc_00, 1=acc_10, 2=acc_20,
//                            3=acc_01, 4=acc_11, 5=acc_21
// =============================================================================

module systolic_array #(
    parameter PE_ROWS   = 12,        // Number of PE rows
    parameter PE_COLS   = 12,        // Number of PE columns
    // Accumulator bit-width (see udp_pe.v for K_MAX vs ACC_WIDTH table).
    // 12 bits → K_MAX=31, matching the paper's ~14,660 LUT target.
    parameter ACC_WIDTH = 12
) (
    input  wire clk,
    input  wire srst,   // Synchronous reset (active high, 1 cycle clears all accumulators)
    input  wire en,     // Accumulation enable

    // Weight inputs for all 144 PEs: PE_ROWS × PE_COLS × 3 weights × 4 bits = 1728 bits
    input  wire [PE_ROWS*PE_COLS*12-1:0] weights_flat,

    // Activation inputs: PE_ROWS rows × 2 activations × 4 bits = 96 bits
    input  wire [PE_ROWS*8-1:0] act_flat,

    // Result outputs: PE_ROWS × PE_COLS × 6 products × ACC_WIDTH = 13824 bits (for defaults)
    output wire [PE_ROWS*PE_COLS*6*ACC_WIDTH-1:0] result_flat
);

    // =========================================================================
    // Activation skewing chain
    // =========================================================================
    // a0_chain[i] = activation-0 wire at position i in the flat chain.
    // Flat index:  row r, column c  →  i = r*(PE_COLS+1) + c
    //   • i = r*(PE_COLS+1) + 0        : external input for row r (col 0 entry)
    //   • i = r*(PE_COLS+1) + (c+1)    : output of PE[r][c] → input of PE[r][c+1]
    //
    // Total entries: PE_ROWS × (PE_COLS+1) = 12 × 13 = 156 per activation signal.
    // =========================================================================
    wire [3:0] a0_chain [0:PE_ROWS*(PE_COLS+1)-1];  // activation-0 pipeline chain
    wire [3:0] a1_chain [0:PE_ROWS*(PE_COLS+1)-1];  // activation-1 pipeline chain

    // =========================================================================
    // Connect external activation inputs to chain entry [r][0]
    // =========================================================================
    genvar ri;
    generate
        for (ri = 0; ri < PE_ROWS; ri = ri + 1) begin : ACT_IN
            assign a0_chain[ri*(PE_COLS+1) + 0] = act_flat[(ri*2 + 0)*4 +: 4];
            assign a1_chain[ri*(PE_COLS+1) + 0] = act_flat[(ri*2 + 1)*4 +: 4];
        end
    endgenerate

    // =========================================================================
    // 12×12 PE Array instantiation
    // =========================================================================
    genvar r, c;
    generate
        for (r = 0; r < PE_ROWS; r = r + 1) begin : GEN_ROW
            for (c = 0; c < PE_COLS; c = c + 1) begin : GEN_COL

                // -- Weight slices for PE(r,c) --
                // weights_flat layout: PE (r*PE_COLS+c) starts at bit offset
                //   (r*PE_COLS + c)*12,  width 12 (three 4-bit weights)
                wire signed [3:0] ws0, ws1, ws2;
                assign ws0 = weights_flat[(r*PE_COLS + c)*12 + 0 +: 4];
                assign ws1 = weights_flat[(r*PE_COLS + c)*12 + 4 +: 4];
                assign ws2 = weights_flat[(r*PE_COLS + c)*12 + 8 +: 4];

                // -- Result wires for PE(r,c) --
                wire signed [ACC_WIDTH-1:0] a00, a10, a20, a01, a11, a21;

                // Wire outputs into result_flat
                // PE(r,c) base offset: (r*PE_COLS + c)*6*ACC_WIDTH bits
                assign result_flat[(r*PE_COLS+c)*6*ACC_WIDTH + 0*ACC_WIDTH +: ACC_WIDTH] = a00;
                assign result_flat[(r*PE_COLS+c)*6*ACC_WIDTH + 1*ACC_WIDTH +: ACC_WIDTH] = a10;
                assign result_flat[(r*PE_COLS+c)*6*ACC_WIDTH + 2*ACC_WIDTH +: ACC_WIDTH] = a20;
                assign result_flat[(r*PE_COLS+c)*6*ACC_WIDTH + 3*ACC_WIDTH +: ACC_WIDTH] = a01;
                assign result_flat[(r*PE_COLS+c)*6*ACC_WIDTH + 4*ACC_WIDTH +: ACC_WIDTH] = a11;
                assign result_flat[(r*PE_COLS+c)*6*ACC_WIDTH + 5*ACC_WIDTH +: ACC_WIDTH] = a21;

                // -- PE instantiation --
                // Activation inputs:  chain[r][c]   (may be external or prior PE output)
                // Activation outputs: chain[r][c+1]  (feeds next PE in this row)
                udp_pe #(.ACC_WIDTH(ACC_WIDTH)) u_pe (
                    .clk    (clk),
                    .srst   (srst),
                    .en     (en),
                    .w_s0   (ws0),
                    .w_s1   (ws1),
                    .w_s2   (ws2),
                    .a_in0  (a0_chain[r*(PE_COLS+1) + c]),
                    .a_in1  (a1_chain[r*(PE_COLS+1) + c]),
                    .a_out0 (a0_chain[r*(PE_COLS+1) + c + 1]),
                    .a_out1 (a1_chain[r*(PE_COLS+1) + c + 1]),
                    .acc_00 (a00), .acc_10 (a10), .acc_20 (a20),
                    .acc_01 (a01), .acc_11 (a11), .acc_21 (a21)
                );

            end // GEN_COL
        end // GEN_ROW
    endgenerate

endmodule
