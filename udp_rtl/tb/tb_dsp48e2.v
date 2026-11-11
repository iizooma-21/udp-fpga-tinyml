// =============================================================================
// tb_dsp48e2.v
// Testbench for the DSP48E2 behavioral model.
//
// Tests:
//   1. Basic: P = A * B + C with known values
//   2. b_pre = 0 (no preprocessing needed)
//   3. Non-zero b_pre (simulating the preprocessing correction)
//   4. Maximum port widths (port boundary check)
//   5. All-zero inputs
// =============================================================================
`timescale 1ns/1ps

module tb_dsp48e2;

    // DUT ports
    reg  [17:0] B;
    reg  [26:0] A;
    reg  [47:0] C;
    wire [47:0] P;

    integer pass_count, fail_count;
    integer test_num;

    // Expected result for checking
    reg [47:0] expected;

    // DUT instantiation
    dsp48e2_model dut (
        .B(B), .A(A), .C(C), .P(P)
    );

    // Task to check one test case
    task check;
        input [63:0] test_id;
        input [47:0] exp;
        input [127:0] desc; // unused but keeps call site readable
        begin
            #1; // let combinational logic settle
            if (P === exp) begin
                $display("  [PASS] Test %0d: P = %0d (expected %0d)", test_id, P, exp);
                pass_count = pass_count + 1;
            end else begin
                $display("  [FAIL] Test %0d: P = %0d (expected %0d) | A=%0d B=%0d C=%0d",
                         test_id, P, exp, A, B, C);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        pass_count = 0;
        fail_count = 0;

        $display("============================================================");
        $display("  tb_dsp48e2: DSP48E2 Behavioral Model Testbench");
        $display("  Tests: P = A * B + C  (all unsigned)");
        $display("============================================================");

        // ------------------------------------------------------------------
        // Test 1: Simple scalar — 3 * 5 + 0
        // ------------------------------------------------------------------
        $display("\n--- Test 1: 3 * 5 + 0 = 15 ---");
        A = 27'd3; B = 18'd5; C = 48'd0;
        expected = 48'd15;
        check(1, expected, "3*5+0");

        // ------------------------------------------------------------------
        // Test 2: With C term — 4 * 6 + 10 = 34
        // ------------------------------------------------------------------
        $display("\n--- Test 2: 4 * 6 + 10 = 34 ---");
        A = 27'd4; B = 18'd6; C = 48'd10;
        expected = 48'd34;
        check(2, expected, "4*6+10");

        // ------------------------------------------------------------------
        // Test 3: Simulate UDP for 1 weight*activation pair (no packing)
        //         w_s=-3 (w_u=5), a_s=5 (a_u=5), b_pre=88
        //         Expected DSP output: 5*5 + 88 = 113
        //         (Postprocessor will subtract 128 to get -15)
        // ------------------------------------------------------------------
        $display("\n--- Test 3: UDP single pair (w_u=5, a_u=5, b_pre=88) ---");
        A = 27'd5; B = 18'd5; C = 48'd88;
        expected = 48'd113;
        check(3, expected, "UDP_single_pair");

        // ------------------------------------------------------------------
        // Test 4: Maximum B-port value (18-bit: 2^18-1 = 262143)
        //         A=1, C=0: P = 262143
        // ------------------------------------------------------------------
        $display("\n--- Test 4: Maximum B port value ---");
        A = 27'd1; B = 18'h3FFFF; C = 48'd0;
        expected = 48'h3FFFF;
        check(4, expected, "max_B");

        // ------------------------------------------------------------------
        // Test 5: Maximum A-port value (27-bit: 2^27-1 = 134217727)
        //         B=1, C=0: P = 134217727
        // ------------------------------------------------------------------
        $display("\n--- Test 5: Maximum A port value ---");
        A = 27'h7FFFFFF; B = 18'd1; C = 48'd0;
        expected = 48'h7FFFFFF;
        check(5, expected, "max_A");

        // ------------------------------------------------------------------
        // Test 6: Non-trivial UDP packed multiply
        //         Paper example: w_s=[-3,2,-1], a_s=[5,-2]
        //         w_p = {7,4'b0,2,4'b0,5} = {111,0000,010,0000,101}
        //              = 17'b11100000100000101 = 114949 (decimal)
        //         a_p = {14,17'b0,5}
        //              = {1110,17'b0,0101} = 25'b1_11000000000000000000_0101 = 29360133
        //         We compute expected = w_p * a_p (ignoring b_pre for port check)
        // ------------------------------------------------------------------
        $display("\n--- Test 6: UDP packed operands (paper example, C=0) ---");
        A = 27'd29360133; B = 18'd114949; C = 48'd0;
        expected = 48'd114949 * 48'd29360133;  // just the product
        // Only check if it doesn't overflow 48 bits
        if (expected[47:44] == 4'b0) begin
            check(6, expected, "UDP_packed_no_C");
        end else begin
            $display("  [SKIP] Test 6: product too large for clean check (expected overflow check)");
        end

        // ------------------------------------------------------------------
        // Test 7: All-zero inputs
        // ------------------------------------------------------------------
        $display("\n--- Test 7: All-zero inputs ---");
        A = 27'd0; B = 18'd0; C = 48'd0;
        expected = 48'd0;
        check(7, expected, "all_zeros");

        // ------------------------------------------------------------------
        // Test 8: C-only (A=0, B=0, C=12345)
        // ------------------------------------------------------------------
        $display("\n--- Test 8: C passthrough (A=0, B=0, C=12345) ---");
        A = 27'd0; B = 18'd0; C = 48'd12345;
        expected = 48'd12345;
        check(8, expected, "C_passthrough");

        // ------------------------------------------------------------------
        // Summary
        // ------------------------------------------------------------------
        $display("\n============================================================");
        $display("  DSP48E2 Testbench Complete");
        $display("  PASSED: %0d   FAILED: %0d", pass_count, fail_count);
        $display("============================================================");

        if (fail_count == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  *** %0d TEST(S) FAILED ***", fail_count);

        $finish;
    end

endmodule
