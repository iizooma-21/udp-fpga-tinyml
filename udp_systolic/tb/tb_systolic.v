// =============================================================================
`timescale 1ns/1ps
// tb_systolic.v
// Testbench for systolic_array.v (12×12 UDP int4 systolic array).
//
// TEST METHODOLOGY
// ================
// Because there are 144 PEs × 6 accumulators = 864 outputs, we cannot
// exhaustively list every check.  Instead we use three focused test vectors
// and sample a REPRESENTATIVE SUBSET of outputs (first PE, last PE, and a
// mid-array PE) to confirm the array is working correctly.
//
// TIMING REMINDER (from systolic_array.v)
// ========================================
// Due to activation skewing, PE[r][c] starts accumulating c cycles AFTER the
// external activation is first presented.  So for K valid activations:
//   • Feed K valid activations, then pad with (PE_COLS-1)=11 zeros.
//   • Hold en=1 for K + 11 = K+11 cycles total.
//   • After those cycles, ALL PEs have completed exactly K accumulations.
//
// TEST CASES
// ==========
//   Test 1 — All-ones:  weights=+1, acts=+1, K=4 → expected acc = +4
//   Test 2 — Mixed neg: weights=-2, acts=+3, K=4 → expected acc = -24
//   Test 3 — Neg acts:  weights=+2, acts=-3, K=3 → expected acc = -18
//
// CHECKED OUTPUTS (bit offsets in result_flat)
// =============================================
//  PE(r,c), product p  →  base = (r*12 + c)*6*ACC_WIDTH + p*ACC_WIDTH
//  ACC_WIDTH = 16
//
//  PE(0, 0) acc_00 (p=0)  → bits [15: 0]
//  PE(0, 0) acc_21 (p=5)  → bits [95:80]
//  PE(6, 3) acc_11 (p=4)  → bits [7279:7264]   (mid-array PE)
//  PE(11,11) acc_00 (p=0) → bits [13728:13712]  (last PE in array)
//  PE(11,11) acc_21 (p=5) → bits [13823:13808]  (last PE, last product)
// =============================================================================

module tb_systolic;

    // ── Parameters matching DUT ─────────────────────────────────────────────
    localparam PE_ROWS   = 12;
    localparam PE_COLS   = 12;
    localparam ACC_WIDTH = 12;  // Must match systolic_array parameter

    // Derived constants
    localparam W_FLAT  = PE_ROWS * PE_COLS * 12;         // 1728 bits (weights)
    localparam A_FLAT  = PE_ROWS * 8;                    //   96 bits (acts)
    localparam R_FLAT  = PE_ROWS * PE_COLS * 6 * ACC_WIDTH; // 13824 bits (results)

    // ── DUT ports ────────────────────────────────────────────────────────────
    reg                   clk;
    reg                   srst;
    reg                   en;
    reg  [W_FLAT-1:0]     weights_flat;
    reg  [A_FLAT-1:0]     act_flat;
    wire [R_FLAT-1:0]     result_flat;

    // ── DUT instantiation ────────────────────────────────────────────────────
    systolic_array #(
        .PE_ROWS   (PE_ROWS),
        .PE_COLS   (PE_COLS),
        .ACC_WIDTH (ACC_WIDTH)
    ) dut (
        .clk          (clk),
        .srst         (srst),
        .en           (en),
        .weights_flat (weights_flat),
        .act_flat     (act_flat),
        .result_flat  (result_flat)
    );

    // ── Clock: 10 ns period ──────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Helper: read a signed ACC_WIDTH word from result_flat ─────────────────
    // PE(r,c), product index p  →  base bit = (r*12 + c)*6*ACC_WIDTH + p*ACC_WIDTH
    function signed [ACC_WIDTH-1:0] get_acc;
        input integer r;
        input integer c;
        input integer p;
        integer base;
        begin
            base = (r * PE_COLS + c) * 6 * ACC_WIDTH + p * ACC_WIDTH;
            get_acc = $signed(result_flat[base +: ACC_WIDTH]);
        end
    endfunction

    // ── Pass/Fail counter ─────────────────────────────────────────────────────
    integer pass_cnt, fail_cnt;

    task check;
        input signed [ACC_WIDTH-1:0] got;
        input signed [ACC_WIDTH-1:0] exp;
        input [127:0] label;
        begin
            if (got === exp) begin
                $display("[PASS] %s  got=%0d  exp=%0d", label, got, exp);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] %s  got=%0d  exp=%0d  *** MISMATCH ***", label, got, exp);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    // ── Task: reset and load weights everywhere ───────────────────────────────
    task reset_dut;
        begin
            en   = 0;
            srst = 1;
            @(posedge clk); #1;
            srst = 0;
            @(posedge clk); #1;
        end
    endtask

    // ── Task: fill all PE weights with a constant signed int4 value ───────────
    task fill_weights;
        input signed [3:0] wval;
        integer i;
        begin
            for (i = 0; i < PE_ROWS * PE_COLS * 3; i = i + 1)
                weights_flat[i*4 +: 4] = wval;
        end
    endtask

    // ── Task: fill all row activations with a constant signed int4 value ──────
    task fill_acts;
        input signed [3:0] aval;
        integer i;
        begin
            for (i = 0; i < PE_ROWS * 2; i = i + 1)
                act_flat[i*4 +: 4] = aval;
        end
    endtask

    // ── Task: run K valid + 11 padding cycles, then deassert en ──────────────
    // After this task returns, all 144 PEs have accumulated exactly K products.
    task run_accumulate;
        input integer K;
        integer cycle;
        begin
            en = 1;
            // K cycles of valid data
            repeat (K) @(posedge clk);
            // 11 padding cycles (acts=0 by caller, or previous value — caller
            // is responsible for setting act_flat to 0 before the last K cycle
            // if padding with zeros is needed)
            fill_acts(4'sh0);
            repeat (PE_COLS - 1) @(posedge clk);
            // One extra cycle for last FF update
            @(posedge clk);
            en = 0;
            #1;
        end
    endtask

    // =========================================================================
    // MAIN TEST SEQUENCE
    // =========================================================================
    initial begin
        pass_cnt    = 0;
        fail_cnt    = 0;
        weights_flat = {W_FLAT{1'b0}};
        act_flat     = {A_FLAT{1'b0}};
        en           = 0;
        srst         = 1;

        @(posedge clk); #1;

        // =====================================================================
        // TEST 1: All weights = +1, all activations = +1, K = 4 accumulations
        //         Expected: every acc_ij = +1 × +1 × 4 = +4
        // =====================================================================
        $display("\n======== TEST 1: weights=+1, acts=+1, K=4 ========");
        fill_weights(4'sh1);   // +1 in signed int4
        reset_dut;             // reset with act=0 → pipeline FFs cleared to 0

        // Set activations AFTER reset so pipeline starts clean
        fill_acts(4'sh1);
        en = 1;
        repeat (4) @(posedge clk);    // 4 valid cycles (act=+1)
        fill_acts(4'sh0);
        repeat (PE_COLS - 1) @(posedge clk);  // 11 padding cycles (act=0)
        @(posedge clk);               // extra cycle for last FF update
        en = 0; #1;

        // Check representative outputs — all should be +4
        check(get_acc(0,  0,  0), 12'sd4,  "PE(0,0)  acc_00");
        check(get_acc(0,  0,  5), 12'sd4,  "PE(0,0)  acc_21");
        check(get_acc(0, 11,  0), 12'sd4,  "PE(0,11) acc_00");
        check(get_acc(6,  3,  4), 12'sd4,  "PE(6,3)  acc_11");
        check(get_acc(11,11,  0), 12'sd4,  "PE(11,11) acc_00");
        check(get_acc(11,11,  5), 12'sd4,  "PE(11,11) acc_21");

        // =====================================================================
        // TEST 2: All weights = -2, all activations = +3, K = 4
        //         Expected: (-2) × (+3) = -6  ×  4 = -24
        // =====================================================================
        $display("\n======== TEST 2: weights=-2, acts=+3, K=4 ========");
        fill_weights(4'shE);   // -2 in signed int4 (1110b)
        reset_dut;

        fill_acts(4'sh3);      // +3 in signed int4
        en = 1;
        repeat (4) @(posedge clk);
        fill_acts(4'sh0);
        repeat (PE_COLS - 1) @(posedge clk);
        @(posedge clk);
        en = 0; #1;

        check(get_acc(0,  0, 0), -12'sd24, "PE(0,0)  acc_00");
        check(get_acc(0,  0, 3), -12'sd24, "PE(0,0)  acc_01");
        check(get_acc(6,  6, 1), -12'sd24, "PE(6,6)  acc_10");
        check(get_acc(11,11, 5), -12'sd24, "PE(11,11) acc_21");

        // =====================================================================
        // TEST 3: All weights = +2, all activations = -3, K = 3
        //         Expected: (+2) × (-3) = -6  ×  3 = -18
        // =====================================================================
        $display("\n======== TEST 3: weights=+2, acts=-3, K=3 ========");
        fill_weights(4'sh2);   // +2 in signed int4
        reset_dut;

        fill_acts(4'shD);      // -3 in signed int4 (1101b)
        en = 1;
        repeat (3) @(posedge clk);
        fill_acts(4'sh0);
        repeat (PE_COLS - 1) @(posedge clk);
        @(posedge clk);
        en = 0; #1;

        check(get_acc(0,  0, 0), -12'sd18, "PE(0,0)  acc_00");
        check(get_acc(5,  5, 2), -12'sd18, "PE(5,5)  acc_20");
        check(get_acc(11,11, 5), -12'sd18, "PE(11,11) acc_21");

        // =====================================================================
        // TEST 4: Accumulation does NOT reset between two en pulses
        //         Weights=+1, acts=+1.  First run K=3, second run K=2.
        //         Expected after second run: acc = 3 + 2 = 5
        // =====================================================================
        $display("\n======== TEST 4: two consecutive accumulations (K=3 then K=2) ========");
        fill_weights(4'sh1);
        reset_dut;

        // First K=3 run
        fill_acts(4'sh1);
        en = 1;
        repeat (3) @(posedge clk);
        fill_acts(4'sh0);
        repeat (PE_COLS - 1) @(posedge clk);
        @(posedge clk);
        en = 0; #1;

        // Second K=2 run (no reset between runs — accumulators continue from 3)
        fill_acts(4'sh1);
        en = 1;
        repeat (2) @(posedge clk);
        fill_acts(4'sh0);
        repeat (PE_COLS - 1) @(posedge clk);
        @(posedge clk);
        en = 0; #1;

        check(get_acc(0,  0, 0), 12'sd5, "PE(0,0)  acc_00  (3+2=5)");
        check(get_acc(11,11, 5), 12'sd5, "PE(11,11) acc_21 (3+2=5)");

        // =====================================================================
        // Summary
        // =====================================================================
        $display("\n================================================");
        $display("SIMULATION COMPLETE: %0d PASS  %0d FAIL", pass_cnt, fail_cnt);
        $display("================================================\n");

        if (fail_cnt == 0)
            $display(">>> ALL TESTS PASSED <<<");
        else
            $display(">>> %0d TEST(S) FAILED — review output above <<<", fail_cnt);

        $finish;
    end

    // Watchdog: abort if simulation runs too long (>50 us)
    initial begin
        #50000;
        $display("[WATCHDOG] Simulation exceeded 50 us — forcing stop.");
        $finish;
    end

endmodule
