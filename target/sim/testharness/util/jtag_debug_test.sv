// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// JTAG Debug Test for Multi-Chiplet HeMAiA
//
// This test verifies that JTAG debug works correctly on all four chiplets,
// including those with non-zero chip_id. For each chiplet it:
//   1. Reads the IDCODE register to verify JTAG TAP connectivity
//   2. Activates the debug module (dmactive=1)
//   3. Sends a halt request to the CVA6 core
//   4. Polls DMStatus until the core reports halted
//   5. Resumes the core and verifies it resumes
//
// Chiplets tested:
//   - chip00 (chip_id=0x00) - baseline
//   - chip01 (chip_id=0x01)
//   - chip10 (chip_id=0x10)
//   - chip11 (chip_id=0x11)
//
// This file is `included from testharness_rtl.sv and relies on:
//   - chip{00,01,10,11}_jtag_{trst_ni,tck_i,tms_i,tdi_i,tdo_o} wires
//   - The jtag_test package (from riscv-dbg) providing riscv_dbg class
//   - rst_ni signal to synchronize test start

localparam time JtagClkPeriod = 100ns; // 10 MHz JTAG clock

// JTAG clock generation
logic jtag_clk;
initial begin
    jtag_clk = 0;
    forever #(JtagClkPeriod / 2) jtag_clk = ~jtag_clk;
end

// JTAG_DV interfaces for all four chiplets
JTAG_DV jtag_if_chip00 (.clk_i(jtag_clk));
JTAG_DV jtag_if_chip01 (.clk_i(jtag_clk));
JTAG_DV jtag_if_chip10 (.clk_i(jtag_clk));
JTAG_DV jtag_if_chip11 (.clk_i(jtag_clk));

// Connect chip00 JTAG signals to interface
assign chip00_jtag_tck_i   = jtag_clk;
assign chip00_jtag_tdi_i   = jtag_if_chip00.tdi;
assign chip00_jtag_tms_i   = jtag_if_chip00.tms;
assign chip00_jtag_trst_ni = jtag_if_chip00.trst_n;
assign jtag_if_chip00.tdo  = chip00_jtag_tdo_o;

// Connect chip01 JTAG signals to interface
assign chip01_jtag_tck_i   = jtag_clk;
assign chip01_jtag_tdi_i   = jtag_if_chip01.tdi;
assign chip01_jtag_tms_i   = jtag_if_chip01.tms;
assign chip01_jtag_trst_ni = jtag_if_chip01.trst_n;
assign jtag_if_chip01.tdo  = chip01_jtag_tdo_o;

// Connect chip10 JTAG signals to interface
assign chip10_jtag_tck_i   = jtag_clk;
assign chip10_jtag_tdi_i   = jtag_if_chip10.tdi;
assign chip10_jtag_tms_i   = jtag_if_chip10.tms;
assign chip10_jtag_trst_ni = jtag_if_chip10.trst_n;
assign jtag_if_chip10.tdo  = chip10_jtag_tdo_o;

// Connect chip11 JTAG signals to interface
assign chip11_jtag_tck_i   = jtag_clk;
assign chip11_jtag_tdi_i   = jtag_if_chip11.tdi;
assign chip11_jtag_tms_i   = jtag_if_chip11.tms;
assign chip11_jtag_trst_ni = jtag_if_chip11.trst_n;
assign jtag_if_chip11.tdo  = chip11_jtag_tdo_o;

// Test task: run JTAG debug test on a given chip
task automatic run_jtag_debug_test(
    input string chip_name,
    input virtual JTAG_DV jtag_if,
    output bit pass
);
    // Create JTAG driver and debug module abstraction
    // TA=0ns (apply immediately), TT=JtagClkPeriod/2 (sample at mid-cycle)
    automatic jtag_test::jtag_driver #(
        .IrLength(5),
        .IDCODE('h1),
        .TA(JtagClkPeriod * 0.1),
        .TT(JtagClkPeriod * 0.9)
    ) jtag_drv = new(jtag_if);

    automatic jtag_test::riscv_dbg #(
        .IrLength(5),
        .IDCODE('h1),
        .DTMCSR('h10),
        .DMIACCESS('h11),
        .TA(JtagClkPeriod * 0.1),
        .TT(JtagClkPeriod * 0.9)
    ) dbg = new(jtag_drv);

    // Variables for test
    automatic logic [31:0] idcode;
    automatic logic [31:0] dmstatus;
    automatic logic [31:0] dmcontrol;
    automatic dm::dtm_op_status_e op;
    automatic int timeout_cnt;

    pass = 1;

    $display("[JTAG_TEST] ========================================");
    $display("[JTAG_TEST] Testing %s JTAG debug...", chip_name);
    $display("[JTAG_TEST] ========================================");

    // Step 1: Reset JTAG TAP
    $display("[JTAG_TEST] [%s] Resetting JTAG TAP...", chip_name);
    dbg.reset_master();
    dbg.wait_idle(10);

    // Step 2: Read IDCODE
    $display("[JTAG_TEST] [%s] Reading IDCODE...", chip_name);
    dbg.get_idcode(idcode);
    if (idcode == 32'h0 || idcode == 32'hFFFFFFFF) begin
        $error("[JTAG_TEST] [%s] FAIL: IDCODE read returned 0x%08x (JTAG TAP not responding)", chip_name, idcode);
        pass = 0;
        return;
    end
    $display("[JTAG_TEST] [%s] IDCODE = 0x%08x (OK)", chip_name, idcode);

    // Step 3: Activate debug module (set dmactive=1)
    $display("[JTAG_TEST] [%s] Activating debug module (dmactive=1)...", chip_name);
    dmcontrol = 32'h0000_0001; // dmactive = 1
    dbg.write_dmi(dm::DMControl, dmcontrol);
    dbg.wait_idle(10);

    // Read back DMStatus to check the debug module is active
    dbg.read_dmi_exp_backoff(dm::DMStatus, dmstatus);
    $display("[JTAG_TEST] [%s] DMStatus = 0x%08x (version=%0d)", chip_name, dmstatus, dmstatus[3:0]);

    // Check version field (bits [3:0]) should be 2 (v0.13)
    if (dmstatus[3:0] != 4'h2) begin
        $error("[JTAG_TEST] [%s] FAIL: DMStatus version=%0d, expected 2 (v0.13)", chip_name, dmstatus[3:0]);
        pass = 0;
        return;
    end

    // Check authenticated (bit 7) should be 1
    if (!dmstatus[7]) begin
        $error("[JTAG_TEST] [%s] FAIL: Debug module not authenticated", chip_name);
        pass = 0;
        return;
    end
    $display("[JTAG_TEST] [%s] Debug module authenticated (OK)", chip_name);

    // Step 4: Halt the core (haltreq=1, dmactive=1)
    $display("[JTAG_TEST] [%s] Sending halt request to CVA6 core...", chip_name);
    dmcontrol = 32'h8000_0001; // haltreq = 1, dmactive = 1
    dbg.write_dmi(dm::DMControl, dmcontrol);

    // Step 5: Poll DMStatus until anyhalted (bit 8) or timeout
    timeout_cnt = 0;
    while (timeout_cnt < 200) begin
        dbg.wait_idle(10);
        dbg.read_dmi_exp_backoff(dm::DMStatus, dmstatus);
        if (dmstatus[8]) begin // anyhalted bit
            break;
        end
        timeout_cnt++;
    end

    if (!dmstatus[8]) begin
        $error("[JTAG_TEST] [%s] FAIL: Core did not halt within timeout (DMStatus=0x%08x)", chip_name, dmstatus);
        $error("[JTAG_TEST] [%s] This indicates the debug module is not reachable at the chip_id-based address!", chip_name);
        pass = 0;
        return;
    end
    $display("[JTAG_TEST] [%s] Core halted successfully! (DMStatus=0x%08x)", chip_name, dmstatus);

    // Step 6: Clear haltreq and request resume
    $display("[JTAG_TEST] [%s] Resuming core...", chip_name);
    dmcontrol = 32'h4000_0001; // resumereq = 1, dmactive = 1
    dbg.write_dmi(dm::DMControl, dmcontrol);

    // Poll DMStatus until anyresumeack (bit 16) or timeout
    timeout_cnt = 0;
    while (timeout_cnt < 200) begin
        dbg.wait_idle(10);
        dbg.read_dmi_exp_backoff(dm::DMStatus, dmstatus);
        if (dmstatus[16]) begin // anyresumeack bit
            break;
        end
        timeout_cnt++;
    end

    if (!dmstatus[16]) begin
        $error("[JTAG_TEST] [%s] FAIL: Core did not resume within timeout (DMStatus=0x%08x)", chip_name, dmstatus);
        pass = 0;
        return;
    end
    $display("[JTAG_TEST] [%s] Core resumed successfully! (DMStatus=0x%08x)", chip_name, dmstatus);

    // Clear resumereq
    dmcontrol = 32'h0000_0001; // dmactive = 1 only
    dbg.write_dmi(dm::DMControl, dmcontrol);
    dbg.wait_idle(10);

    $display("[JTAG_TEST] [%s] PASS: JTAG debug test completed successfully!", chip_name);
endtask

// Main JTAG test process
initial begin
    automatic bit pass_chip00, pass_chip01, pass_chip10, pass_chip11;
    automatic int pass_count;

    // Wait for reset release and system stabilization
    @(posedge rst_ni);
    #(20us);

    $display("[JTAG_TEST] ========================================");
    $display("[JTAG_TEST] Starting JTAG Debug Multi-Chiplet Test");
    $display("[JTAG_TEST] Testing all 4 chiplets: 00, 01, 10, 11");
    $display("[JTAG_TEST] ========================================");

    // Test chip00 (chip_id = 0x00) - baseline
    run_jtag_debug_test("chip00 (id=0x00)", jtag_if_chip00, pass_chip00);
    #(5us);

    // Test chip01 (chip_id = 0x01)
    run_jtag_debug_test("chip01 (id=0x01)", jtag_if_chip01, pass_chip01);
    #(5us);

    // Test chip10 (chip_id = 0x10)
    run_jtag_debug_test("chip10 (id=0x10)", jtag_if_chip10, pass_chip10);
    #(5us);

    // Test chip11 (chip_id = 0x11)
    run_jtag_debug_test("chip11 (id=0x11)", jtag_if_chip11, pass_chip11);

    // Summary
    pass_count = pass_chip00 + pass_chip01 + pass_chip10 + pass_chip11;
    $display("[JTAG_TEST] ========================================");
    $display("[JTAG_TEST] JTAG Debug Test Summary: %0d/4 PASSED", pass_count);
    $display("[JTAG_TEST]   chip00 (id=0x00): %s", pass_chip00 ? "PASS" : "FAIL");
    $display("[JTAG_TEST]   chip01 (id=0x01): %s", pass_chip01 ? "PASS" : "FAIL");
    $display("[JTAG_TEST]   chip10 (id=0x10): %s", pass_chip10 ? "PASS" : "FAIL");
    $display("[JTAG_TEST]   chip11 (id=0x11): %s", pass_chip11 ? "PASS" : "FAIL");
    $display("[JTAG_TEST] ========================================");
    if (pass_count == 4) begin
        $display("[JTAG_TEST] ALL TESTS PASSED!");
    end else begin
        $error("[JTAG_TEST] %0d TEST(S) FAILED!", 4 - pass_count);
    end
end
