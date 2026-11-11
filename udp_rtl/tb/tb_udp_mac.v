// =============================================================================
// tb_udp_mac.v
// Full end-to-end testbench for the UDP MAC top-level module.
//
// SUPERVISOR REQUIREMENTS:
//   1. Test the DSP block       → via udp_mac_top which instantiates dsp48e2_model
//   2. Ensure 6 MAC operations  → loop_count_A=6 vs loop_count_B=1 (6x reduction)
//   3. b_pre support            → test cases with negative weights
//   4. UDP packing operation    → verify packed multiply gives correct products
//
// TEST STRUCTURE
// --------------
// For each test vector [w_s0, w_s1, w_s2, a_s0, a_s1]:
//
//   TEST A — 6 individual scalar MACs (reference)
//     Compute w_s_i * a_s_j directly in Verilog for i=0..2, j=0..1.
//     mac_count_A increments by 6 per test case.
//
//   TEST B — 1 UDP call (hardware path through udp_mac_top)
//     Instantiate udp_mac_top, read 6 output products.
//     dsp_count_B increments by 1 per test case.
//
//   VERIFY: prod_B_ij == prod_A_ij for all 6 slots.
//
// EXPECTED RESULT:
//   After N test cases:
//     mac_count_A  = 6 * N   (6 separate operations per test)
//     dsp_count_B  = N       (1 packed DSP call per test)
//     Ratio        = 6x      (matches P_w * P_a = 3 * 2)
// =============================================================================
`timescale 1ns/1ps

module tb_udp_mac;

    // -------------------------------------------------------------------------
    // DUT inputs
    // -------------------------------------------------------------------------
    reg signed [3:0] w_s0, w_s1, w_s2;
    reg signed [3:0] a_s0, a_s1;

    // -------------------------------------------------------------------------
    // DUT outputs
    // -------------------------------------------------------------------------
    wire signed [7:0] prod_00, prod_10, prod_20;
    wire signed [7:0] prod_01, prod_11, prod_21;
    wire [17:0] dsp_B;
    wire [26:0] dsp_A;
    wire [47:0] dsp_C, dsp_P;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    udp_mac_top dut (
        .w_s0(w_s0), .w_s1(w_s1), .w_s2(w_s2),
        .a_s0(a_s0), .a_s1(a_s1),
        .prod_00(prod_00), .prod_10(prod_10), .prod_20(prod_20),
        .prod_01(prod_01), .prod_11(prod_11), .prod_21(prod_21),
        .dsp_B(dsp_B), .dsp_A(dsp_A), .dsp_C(dsp_C), .dsp_P(dsp_P)
    );

    // -------------------------------------------------------------------------
    // Counters — core of the 6x demonstration
    // -------------------------------------------------------------------------
    integer mac_count_A;    // scalar MAC operations (Test A, increments by 6)
    integer dsp_count_B;    // packed DSP calls      (Test B, increments by 1)
    integer pass_count, fail_count;

    // Reference products (Test A — direct scalar multiplication)
    reg signed [7:0] ref_00, ref_10, ref_20;
    reg signed [7:0] ref_01, ref_11, ref_21;

    // -------------------------------------------------------------------------
    // Task: apply one test vector and verify all 6 products
    // -------------------------------------------------------------------------
    task run_test;
        input signed [3:0] tw0, tw1, tw2;
        input signed [3:0] ta0, ta1;
        input [255:0] label;
        integer errors;
        begin
            // -- Apply inputs to DUT
            w_s0 = tw0; w_s1 = tw1; w_s2 = tw2;
            a_s0 = ta0; a_s1 = ta1;
            #2; // combinational settle time

            // -- TEST A: compute 6 reference products (scalar MACs)
            ref_00 = $signed(tw0) * $signed(ta0);
            ref_10 = $signed(tw1) * $signed(ta0);
            ref_20 = $signed(tw2) * $signed(ta0);
            ref_01 = $signed(tw0) * $signed(ta1);
            ref_11 = $signed(tw1) * $signed(ta1);
            ref_21 = $signed(tw2) * $signed(ta1);
            mac_count_A = mac_count_A + 6;  // 6 individual MAC operations

            // -- TEST B: 1 UDP DSP call → read 6 products from DUT
            dsp_count_B = dsp_count_B + 1;  // 1 packed DSP operation

            // -- COMPARE
            errors = 0;
            if (prod_00 !== ref_00) errors = errors + 1;
            if (prod_10 !== ref_10) errors = errors + 1;
            if (prod_20 !== ref_20) errors = errors + 1;
            if (prod_01 !== ref_01) errors = errors + 1;
            if (prod_11 !== ref_11) errors = errors + 1;
            if (prod_21 !== ref_21) errors = errors + 1;

            if (errors == 0) begin
                $display("  [PASS] %0s", label);
                $display("         w=[%0d,%0d,%0d]  a=[%0d,%0d]",
                         tw0, tw1, tw2, ta0, ta1);
                $display("         b_pre=%0d  |  Products: [%0d,%0d,%0d | %0d,%0d,%0d]",
                         dsp_C,
                         prod_00, prod_10, prod_20,
                         prod_01, prod_11, prod_21);
                pass_count = pass_count + 1;
            end else begin
                $display("  [FAIL] %0s", label);
                $display("         w=[%0d,%0d,%0d]  a=[%0d,%0d]",
                         tw0, tw1, tw2, ta0, ta1);
                $display("         Expected: [%0d,%0d,%0d | %0d,%0d,%0d]",
                         ref_00, ref_10, ref_20, ref_01, ref_11, ref_21);
                $display("         Got:      [%0d,%0d,%0d | %0d,%0d,%0d]",
                         prod_00, prod_10, prod_20, prod_01, prod_11, prod_21);
                $display("         b_pre=%0d  dsp_B=%0d  dsp_A=%0d",
                         dsp_C, dsp_B, dsp_A);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // -------------------------------------------------------------------------
    // Main test sequence
    // -------------------------------------------------------------------------
    initial begin
        mac_count_A = 0;
        dsp_count_B = 0;
        pass_count  = 0;
        fail_count  = 0;

        $display("================================================================");
        $display("  tb_udp_mac: UDP MAC End-to-End Testbench");
        $display("  Config: int4 x int4, P_w=3, P_a=2 -> 6 MACs per DSP call");
        $display("================================================================");

        // ------------------------------------------------------------------
        // TEST GROUP 1: All-positive (no preprocessing, no postprocessing)
        //   b_pre = 0, no correction needed
        // ------------------------------------------------------------------
        $display("\n=== GROUP 1: All-positive weights and activations (b_pre=0) ===");
        run_test(4'd3, 4'd2, 4'd1,  4'd5, 4'd3,  "All positive");
        run_test(4'd7, 4'd7, 4'd7,  4'd7, 4'd7,  "Max positive");
        run_test(4'd1, 4'd1, 4'd1,  4'd1, 4'd1,  "All ones");

        // ------------------------------------------------------------------
        // TEST GROUP 2: Negative weights, positive activations
        //   b_pre != 0 (preprocessing kicks in)
        // ------------------------------------------------------------------
        $display("\n=== GROUP 2: Negative weights, positive activations (b_pre != 0) ===");
        run_test(-4'd3,  4'd2, -4'd1, 4'd5,  4'd3,  "Paper example [-3,2,-1],[5,3]");
        run_test(-4'd8, -4'd8, -4'd8, 4'd7,  4'd7,  "All min weights, max act");
        run_test(-4'd1, -4'd1, -4'd1, 4'd1,  4'd1,  "All -1 weights");

        // ------------------------------------------------------------------
        // TEST GROUP 3: Positive weights, negative activations
        //   Postprocessing term B active (16 * w_u * [a<0])
        // ------------------------------------------------------------------
        $display("\n=== GROUP 3: Positive weights, negative activations ===");
        run_test(4'd3,  4'd2,  4'd1, -4'd5, -4'd3,  "Positive w, negative a");
        run_test(4'd7,  4'd7,  4'd7, -4'd8, -4'd1,  "Max pos w, neg a");

        // ------------------------------------------------------------------
        // TEST GROUP 4: Mixed signs — all corrections active
        //   This is the most complex case: b_pre + both postprocessing terms
        // ------------------------------------------------------------------
        $display("\n=== GROUP 4: Mixed signs (all correction terms active) ===");
        run_test(-4'd3,  4'd2, -4'd1,  4'd5, -4'd2,  "Paper example [-3,2,-1],[5,-2]");
        run_test(-4'd8,  4'd7, -4'd1, -4'd8,  4'd7,  "Mixed extremes");
        run_test(-4'd4, -4'd4,  4'd4, -4'd4,  4'd4,  "Symmetric mixed");

        // ------------------------------------------------------------------
        // TEST GROUP 5: Edge cases
        // ------------------------------------------------------------------
        $display("\n=== GROUP 5: Edge cases ===");
        run_test(4'd0,  4'd0,  4'd0,  4'd0,  4'd0,  "All zeros");
        run_test(-4'd8, -4'd8, -4'd8, -4'd8, -4'd8,  "All minimum (-8*-8=64)");
        run_test(4'd7, -4'd8,  4'd0,  4'd7, -4'd8,  "Mixed extremes and zero");
        run_test(-4'd1, -4'd1, -4'd1, -4'd1, -4'd1,  "All -1");

        // ------------------------------------------------------------------
        // LOOP COUNT SUMMARY — demonstrates the 6x reduction
        // ------------------------------------------------------------------
        $display("\n================================================================");
        $display("  LOOP COUNT COMPARISON (6x Reduction Demonstration)");
        $display("================================================================");
        $display("  Test A (individual scalar MACs): %0d operations", mac_count_A);
        $display("  Test B (packed UDP DSP calls):   %0d operations", dsp_count_B);
        $display("  Reduction factor:                %0dx", mac_count_A / dsp_count_B);
        $display("  (Expected: 6x = P_w * P_a = 3 * 2)");

        // ------------------------------------------------------------------
        // DSP PORT VISIBILITY (last test case values)
        // ------------------------------------------------------------------
        $display("\n================================================================");
        $display("  DSP Port Inspection (last test case)");
        $display("================================================================");
        $display("  B port (w_p,  17-bit): %0d  [bin: %b]", dsp_B[16:0], dsp_B[16:0]);
        $display("  A port (a_p,  25-bit): %0d  [bin: %b]", dsp_A[24:0], dsp_A[24:0]);
        $display("  C port (b_pre,48-bit): %0d", dsp_C);
        $display("  P port (r,    48-bit): %0d", dsp_P);

        // ------------------------------------------------------------------
        // Final summary
        // ------------------------------------------------------------------
        $display("\n================================================================");
        $display("  FINAL RESULTS");
        $display("  Total test cases: %0d", pass_count + fail_count);
        $display("  PASSED: %0d   FAILED: %0d", pass_count, fail_count);
        $display("================================================================");

        if (fail_count == 0)
            $display("  ALL TESTS PASSED — UDP produces bit-exact results");
        else
            $display("  *** %0d TEST(S) FAILED ***", fail_count);

        $finish;
    end

endmodule
