// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// This is the testharness of the Hemaia chip
// In the test harness there will be have three things:
// 1. off-chip driving signals
//    CLK and RST: clk_i, rst_i
//    PLL Driving Pins: pll_i, pll_o
//    UART Driving Pins: uart_i, uart_o
//    I2C Driving Pins:
//    JTAG Driving Pins:
//    SPI Driving Pins:
//    That mimic the real testing setup
// 2. The Design Under Test (DUT)
//    Which only exposed to the aformentioned CLK/PLL/Periph Pins
//    Inside the DUT, there are two setups
//    2.1 The Hemaia Chip Top
//        Can be configured to multiple chiplets
//    2.2 The Hemaia Chip Top with interposer (io pad + d2d link behavior model)
//        Now the 2.2 only supports 4 chiplets
//    The testharness will pass the SIM_WITH_INTERPOSER parameter to the DUT to select the two setups
// 3. The memchip around the DUT
//    This will be implemented with the FPGA in the real testing setup
// By doing so, we can maintain a practical and clear testing setup.
//
// The testharness will:
// 1. Drive CLK
// 2. Drive RST
// 3. Drive PLL pin if PLL presents
// 4. Load the Binary from the file using the load_data function in mem
// 5. Monitor simulation status
%if pll_present:
`timescale 1ps / 1fs
%else:
`timescale 1ns / 1ps
%endif
module testharness;
    // Drive the CLK signals
    %if pll_present:
    `define CLK_FREF_FREQ_MHZ  40.0
    // Matches with the real chip freq
    // In this freq, the uart simulation will be super slow
    `define PRI_FREQ_MHZ       32.0
    %if same_memchip_speed:
    `define MEMPOOL_FREQ_MHZ   4000.0 // 4 GHz, same as the PLL clock
    %else:
    // MEMPOOL_FREQ_MHZ must be 1/20 of the main clock to
    // make sure the d2d link alongside fpga works well
    `define MEMPOOL_FREQ_MHZ   200.0  // 200 MHz, 1/20 of the main clock
    %endif
    %else:
    // The master clock the compute chips divide down (cfg hemaia_multichip.sim_clock;
    // 500 MHz by default). At 4 GHz the chips run their logic at /8 = 500 MHz and their D2D
    // PHYs at the master itself, as the real chip's PLL does.
    `define CLK_FREF_FREQ_MHZ  ${"%.1f" % sim_clk_mhz}
    `define PRI_FREQ_MHZ       500.0 // 500 MHz
    %if same_memchip_speed:
    `define MEMPOOL_FREQ_MHZ   ${"%.1f" % sim_clk_mhz} // the memory chip's master: the same clock
    %else:
    `define MEMPOOL_FREQ_MHZ   25.0  //  25 MHz, 1/20 of the main clock
    %endif
    %endif
    // Use an explicit unit so simulator-wide -override_timescale flags cannot
    // reinterpret these numeric periods (notably 40 MHz as 40 kHz in PLL mode).
    `define CLK_FREF_PERIOD     (1us/`CLK_FREF_FREQ_MHZ)
    `define PRI_FREF_PERIOD     (1us/`PRI_FREQ_MHZ)
    `define MEMPOOL_FREF_PERIOD (1us/`MEMPOOL_FREQ_MHZ)
    logic mst_clk_drv, periph_clk_drv, mempool_clk_drv;
    wire  mst_clk_i, periph_clk_i, mempool_clk_i;
    assign mst_clk_i     = mst_clk_drv;
    assign periph_clk_i  = periph_clk_drv;
    assign mempool_clk_i = mempool_clk_drv;
    %for compute_chip in compute_chips:
    <%
        comp_chip_x = compute_chip.coordinate[0]
        comp_chip_y = compute_chip.coordinate[1]
    %>
    wire chip${comp_chip_x}${comp_chip_y}_clk_obs_o;
    %endfor

    initial begin
        mst_clk_drv = 0;
        forever #(`CLK_FREF_PERIOD/2.0) mst_clk_drv = ~mst_clk_drv;
    end

    initial begin
        periph_clk_drv = 0;
        forever #(`PRI_FREF_PERIOD/2.0) periph_clk_drv = ~periph_clk_drv;
    end

    initial begin
        mempool_clk_drv = 0;
        forever #(`MEMPOOL_FREF_PERIOD/2.0) mempool_clk_drv = ~mempool_clk_drv;
    end

    // CHIP ID
    %for compute_chip in compute_chips:
    <%
        comp_chip_x = compute_chip.coordinate[0]
        comp_chip_y = compute_chip.coordinate[1]
        comp_chip_x_str = "{:01x}".format(comp_chip_x)
        comp_chip_y_str = "{:01x}".format(comp_chip_y)
    %>
    wire [7:0] chip${comp_chip_x}${comp_chip_y}_id_i;
    assign chip${comp_chip_x}${comp_chip_y}_id_i = 8'h${comp_chip_x_str}${comp_chip_y_str};
    %endfor

    %for mem_chip in mem_chips:
    <%
        mem_chip_x = mem_chip.coordinate[0]
        mem_chip_y = mem_chip.coordinate[1]
        mem_chip_x_str = "{:01x}".format(mem_chip_x)
        mem_chip_y_str = "{:01x}".format(mem_chip_y)
    %>
    wire [7:0] mem_chip${mem_chip_x}${mem_chip_y}_id_i;
    assign mem_chip${mem_chip_x}${mem_chip_y}_id_i = 8'h${mem_chip_x_str}${mem_chip_y_str};
    %endfor 
    // PLL
    %for compute_chip in compute_chips:
    <%
        comp_chip_x = compute_chip.coordinate[0]
        comp_chip_y = compute_chip.coordinate[1]
    %>    
    logic       chip${comp_chip_x}${comp_chip_y}_pll_bypass_drv;
    logic       chip${comp_chip_x}${comp_chip_y}_pll_en_drv;
    logic [1:0] chip${comp_chip_x}${comp_chip_y}_pll_post_div_sel_drv;
    wire        chip${comp_chip_x}${comp_chip_y}_pll_bypass_i;
    wire        chip${comp_chip_x}${comp_chip_y}_pll_en_i;
    wire [1:0]  chip${comp_chip_x}${comp_chip_y}_pll_post_div_sel_i;
    wire        chip${comp_chip_x}${comp_chip_y}_pll_lock_o;
    assign chip${comp_chip_x}${comp_chip_y}_pll_bypass_i       = chip${comp_chip_x}${comp_chip_y}_pll_bypass_drv;
    assign chip${comp_chip_x}${comp_chip_y}_pll_en_i           = chip${comp_chip_x}${comp_chip_y}_pll_en_drv;
    assign chip${comp_chip_x}${comp_chip_y}_pll_post_div_sel_i = chip${comp_chip_x}${comp_chip_y}_pll_post_div_sel_drv;
    %endfor

    task init_pll_pins();
        begin
        %for compute_chip in compute_chips:
        <%
            comp_chip_x = compute_chip.coordinate[0]
            comp_chip_y = compute_chip.coordinate[1]
        %>
            chip${comp_chip_x}${comp_chip_y}_pll_bypass_drv = '0;
            chip${comp_chip_x}${comp_chip_y}_pll_en_drv = '0;
            chip${comp_chip_x}${comp_chip_y}_pll_post_div_sel_drv = '0;
        %endfor
        end
    endtask

    %if pll_present:
    task enable_pll_and_wait_lock();
        begin
        // Wait at least 1us
        #(1us);
        // PLL on
        // Drive the enable pin in sequence
        %for compute_chip in compute_chips:
        chip${compute_chip.coordinate[0]}${compute_chip.coordinate[1]}_pll_en_drv = 1'b1;
        // Wait for phase locked
        wait (chip${compute_chip.coordinate[0]}${compute_chip.coordinate[1]}_pll_lock_o === 1'b1);
        $display("Chip ${compute_chip.coordinate[0]}${compute_chip.coordinate[1]}'s PLL Lock asserted!");
        %endfor
        end
    endtask
    %endif

    // Drive rst
    logic rst_ni_drv, rst_periph_ni_drv;
    wire rst_ni, rst_periph_ni;
    assign rst_ni        = rst_ni_drv;
    assign rst_periph_ni = rst_periph_ni_drv;
    task set_rst();
        begin
            rst_ni_drv = 0;
            rst_periph_ni_drv = 0;
        end
    endtask

    task release_rst();
        begin
            rst_ni_drv = 1;
            rst_periph_ni_drv = 1;
        end
    endtask

    // Periph signals
    // Each chiplet should has its own peripherals
    localparam NUM_COMPUTE_CHPILET = ${num_compute_chiplet};
    localparam MAX_COMPUTE_CHIPLET_X = ${max_compute_chiplet_x};
    localparam MAX_COMPUTE_CHIPLET_Y = ${max_compute_chiplet_y};
    wire const_zero;
    wire const_one;
    assign const_zero = 1'b0;
    assign const_one  = 1'b1;
    %for compute_chip in compute_chips:
    <%
        comp_chip_x = compute_chip.coordinate[0]
        comp_chip_y = compute_chip.coordinate[1]
    %>
    // UART
    wire chip${comp_chip_x}${comp_chip_y}_uart_tx_o;
    wire chip${comp_chip_x}${comp_chip_y}_uart_rx_i;
    wire chip${comp_chip_x}${comp_chip_y}_uart_rts_no;
    wire chip${comp_chip_x}${comp_chip_y}_uart_cts_ni;
    assign chip${comp_chip_x}${comp_chip_y}_uart_cts_ni = const_zero;
    // The uart dpi interface
    uartdpi #(
        .BAUD(1),
        .FREQ(32),
        .NAME("uart_chip_${comp_chip_x}_${comp_chip_y}")
    ) i_uart_chip${comp_chip_x}${comp_chip_y} (
        .clk_i (periph_clk_i),
        .rst_ni(rst_ni),
        .tx_o  (chip${comp_chip_x}${comp_chip_y}_uart_rx_i),
        .rx_i  (chip${comp_chip_x}${comp_chip_y}_uart_tx_o)
    );
    // I2C
    wire chip${comp_chip_x}${comp_chip_y}_i2c_sda;
    wire chip${comp_chip_x}${comp_chip_y}_i2c_scl;
    // JTAG
    wire chip${comp_chip_x}${comp_chip_y}_jtag_trst_ni;
    wire chip${comp_chip_x}${comp_chip_y}_jtag_tck_i;
    wire chip${comp_chip_x}${comp_chip_y}_jtag_tms_i;
    wire chip${comp_chip_x}${comp_chip_y}_jtag_tdi_i;
    wire chip${comp_chip_x}${comp_chip_y}_jtag_tdo_o;
% if not sim_with_jtag_check:
    // Tie JTAG to zero when JTAG check is disabled
    assign chip${comp_chip_x}${comp_chip_y}_jtag_trst_ni = const_zero;
    assign chip${comp_chip_x}${comp_chip_y}_jtag_tck_i   = const_zero;
    assign chip${comp_chip_x}${comp_chip_y}_jtag_tms_i   = const_zero;
    assign chip${comp_chip_x}${comp_chip_y}_jtag_tdi_i   = const_zero;
% endif
    // When sim_with_jtag_check is enabled, JTAG signals are driven by jtag_debug_test.sv
    // SPI M
    wire chip${comp_chip_x}${comp_chip_y}_spim_sck_o;
    wire chip${comp_chip_x}${comp_chip_y}_spim_csb_o;
    wire [3:0] chip${comp_chip_x}${comp_chip_y}_spim_sd;
    // GPIO
    wire [3:0] chip${comp_chip_x}${comp_chip_y}_gpio;
    %endfor

    // Main Working Process
    // The load_binary function
% if sim_with_netlist:
    `include "util/load_binary_netlist.sv"
% elif sim_with_mem_macro:
    `include "util/load_binary_mem_macro.sv"
% else:
    `include "util/load_binary_rtl.sv"
% endif
    // The check_finish function
% if sim_with_netlist:
    `include "util/check_finish_netlist.sv"
% elif sim_with_mem_macro:
    `include "util/check_finish_mem_macro.sv"
% else:
    `include "util/check_finish_rtl.sv"
% endif
% if sim_with_jtag_check:
    ///////////////////////////////////////////
    // JTAG Debug Test for Multi-Chiplet HeMAiA
    // Verifies JTAG debug works on all chiplets
    ///////////////////////////////////////////

    localparam time JtagClkPeriod = 100ns; // 10 MHz JTAG clock

    // JTAG clock generation
    logic jtag_clk;
    initial begin
        jtag_clk = 0;
        forever #(JtagClkPeriod / 2) jtag_clk = ~jtag_clk;
    end

    // JTAG_DV interfaces and connections for each chiplet
    %for compute_chip in compute_chips:
    <%
        cx = compute_chip.coordinate[0]
        cy = compute_chip.coordinate[1]
    %>\
    JTAG_DV jtag_if_chip${cx}${cy} (.clk_i(jtag_clk));
    assign chip${cx}${cy}_jtag_tck_i   = jtag_clk;
    assign chip${cx}${cy}_jtag_tdi_i   = jtag_if_chip${cx}${cy}.tdi;
    assign chip${cx}${cy}_jtag_tms_i   = jtag_if_chip${cx}${cy}.tms;
    assign chip${cx}${cy}_jtag_trst_ni = jtag_if_chip${cx}${cy}.trst_n;
    assign jtag_if_chip${cx}${cy}.tdo  = chip${cx}${cy}_jtag_tdo_o;
    %endfor

    // JTAG debug test task: halt and resume a core via JTAG
    task automatic run_jtag_debug_test(
        input string chip_name,
        input virtual JTAG_DV jtag_if,
        output bit pass
    );
        automatic jtag_test::jtag_driver #(
            .IrLength(5), .IDCODE('h1),
            .TA(JtagClkPeriod * 0.1), .TT(JtagClkPeriod * 0.9)
        ) jtag_drv = new(jtag_if);
        automatic jtag_test::riscv_dbg #(
            .IrLength(5), .IDCODE('h1), .DTMCSR('h10), .DMIACCESS('h11),
            .TA(JtagClkPeriod * 0.1), .TT(JtagClkPeriod * 0.9)
        ) dbg = new(jtag_drv);
        automatic logic [31:0] idcode, dmstatus, dmcontrol;
        automatic dm::dtm_op_status_e op;
        automatic int timeout_cnt;
        pass = 1;
        $display("[JTAG_TEST] === Testing %s ===", chip_name);

        // Reset JTAG TAP
        dbg.reset_master();
        dbg.wait_idle(10);

        // Read IDCODE
        dbg.get_idcode(idcode);
        if (idcode == 32'h0 || idcode == 32'hFFFFFFFF) begin
            $error("[JTAG_TEST] [%s] FAIL: IDCODE=0x%08x (TAP not responding)", chip_name, idcode);
            pass = 0; return;
        end
        $display("[JTAG_TEST] [%s] IDCODE=0x%08x (OK)", chip_name, idcode);

        // Activate debug module
        dbg.write_dmi(dm::DMControl, 32'h0000_0001);
        dbg.wait_idle(10);
        dbg.read_dmi_exp_backoff(dm::DMStatus, dmstatus);
        $display("[JTAG_TEST] [%s] DMStatus=0x%08x (version=%0d)", chip_name, dmstatus, dmstatus[3:0]);
        if (dmstatus[3:0] != 4'h2) begin
            $error("[JTAG_TEST] [%s] FAIL: DMStatus version=%0d, expected 2 (v0.13)", chip_name, dmstatus[3:0]);
            pass = 0; return;
        end
        if (!dmstatus[7]) begin
            $error("[JTAG_TEST] [%s] FAIL: Not authenticated", chip_name);
            pass = 0; return;
        end
        $display("[JTAG_TEST] [%s] Debug module active, authenticated", chip_name);

        // Halt the core
        dbg.write_dmi(dm::DMControl, 32'h8000_0001);
        timeout_cnt = 0;
        while (timeout_cnt < 200) begin
            dbg.wait_idle(10);
            dbg.read_dmi_exp_backoff(dm::DMStatus, dmstatus);
            if (dmstatus[8]) break;
            timeout_cnt++;
        end
        if (!dmstatus[8]) begin
            $error("[JTAG_TEST] [%s] FAIL: Core did not halt (DMStatus=0x%08x)", chip_name, dmstatus);
            pass = 0; return;
        end
        $display("[JTAG_TEST] [%s] Core halted (OK)", chip_name);

        // Resume the core
        dbg.write_dmi(dm::DMControl, 32'h4000_0001);
        timeout_cnt = 0;
        while (timeout_cnt < 200) begin
            dbg.wait_idle(10);
            dbg.read_dmi_exp_backoff(dm::DMStatus, dmstatus);
            if (dmstatus[16]) break;
            timeout_cnt++;
        end
        if (!dmstatus[16]) begin
            $error("[JTAG_TEST] [%s] FAIL: Core did not resume (DMStatus=0x%08x)", chip_name, dmstatus);
            pass = 0; return;
        end
        dbg.write_dmi(dm::DMControl, 32'h0000_0001);
        dbg.wait_idle(10);
        $display("[JTAG_TEST] [%s] Core resumed (OK) - PASS", chip_name);
    endtask

    // Main JTAG test process
    initial begin
        automatic int pass_count = 0;
        automatic int total_chips = ${len(compute_chips)};
    %for compute_chip in compute_chips:
    <%
        cx = compute_chip.coordinate[0]
        cy = compute_chip.coordinate[1]
        cxs = "{:01x}".format(cx)
        cys = "{:01x}".format(cy)
    %>\
        automatic bit pass_chip${cx}${cy};
    %endfor

        @(posedge rst_ni);
        #(20us);
        $display("[JTAG_TEST] ========================================");
        $display("[JTAG_TEST] Starting JTAG Debug Multi-Chiplet Test");
        $display("[JTAG_TEST] Testing %0d chiplet(s)", total_chips);
        $display("[JTAG_TEST] ========================================");

    %for compute_chip in compute_chips:
    <%
        cx = compute_chip.coordinate[0]
        cy = compute_chip.coordinate[1]
        cxs = "{:01x}".format(cx)
        cys = "{:01x}".format(cy)
    %>\
        run_jtag_debug_test("chip${cx}${cy} (id=0x${cxs}${cys})", jtag_if_chip${cx}${cy}, pass_chip${cx}${cy});
        #(5us);
    %endfor

    %for compute_chip in compute_chips:
    <%
        cx = compute_chip.coordinate[0]
        cy = compute_chip.coordinate[1]
    %>\
        pass_count += pass_chip${cx}${cy};
    %endfor

        $display("[JTAG_TEST] ========================================");
        $display("[JTAG_TEST] Results: %0d/%0d PASSED", pass_count, total_chips);
    %for compute_chip in compute_chips:
    <%
        cx = compute_chip.coordinate[0]
        cy = compute_chip.coordinate[1]
        cxs = "{:01x}".format(cx)
        cys = "{:01x}".format(cy)
    %>\
        $display("[JTAG_TEST]   chip${cx}${cy} (id=0x${cxs}${cys}): %s", pass_chip${cx}${cy} ? "PASS" : "FAIL");
    %endfor
        $display("[JTAG_TEST] ========================================");
        if (pass_count == total_chips)
            $display("[JTAG_TEST] ALL TESTS PASSED!");
        else
            $error("[JTAG_TEST] %0d TEST(S) FAILED!", total_chips - pass_count);
    end
% endif
    initial begin
        set_rst();
        #(10ps);
        release_rst();
        #(10ps);
        set_rst();
        init_pll_pins();
        // Wait some random time
        #(11ns); 
        %if pll_present:
        // Enable the PLL and wait for locked signal
        enable_pll_and_wait_lock();
        %else:
        // PLL is not presents
        %endif
        // Wait some random time
        #(7ns);
        // Load binary
        load_binary();
        // Release the rst
        release_rst();
        // Monitor Simulation Status
        // This is done by inspecting a specific main mem location
        // In the real setup the chip will output the status by uart
        check_finish();
    end

    // Trigger the reusbale init task when reload_bin becomes high
    logic reload_bin = '0;
    always @(posedge reload_bin) begin
        set_rst();
        // Wait some random time
        #(11ns);
        // Load binary
        load_binary();
        // Release the rst
        release_rst();
        reload_bin = '0;
    end

    // The compute chiplets - the DUT (the ASICs) - and the memory chiplets - external memory
    // pools, an FPGA in the real setup - sit on one grid and talk over D2D links: a link
    // joins every two chips side by side (x to the east, y to the south, chip id
    // (x << 4) | y, coordinates 0..14). A memory chip may sit on the edge of the compute
    // array, between compute chips (a row C M C: it feeds both), or next to another memory
    // chip. For example:
    //+------+-------------+-------------+-------------+-------------+
    //|      | x=0         | x=1         | x=2         | x=3         |
    //+------+-------------+-------------+-------------+-------------+
    //| y=0  | comp_chip_00| mem_chip_10 | comp_chip_20| mem_chip_30 |
    //+------+-------------+-------------+-------------+-------------+
    //| y=1  | comp_chip_01| comp_chip_11| comp_chip_21|             |
    //+------+-------------+-------------+-------------+-------------+
    // The dut has a port on every compute-chip side with no compute chip beside it; the
    // links between compute chips are inside it (io_wrapper).

%if not sim_with_verilator:
    // The D2D links that leave the dut or join memory chips: per link a tri data bus and,
    // per end, the rts / cts / test-request that end drives. Dut ports with nothing beside
    // them get inputs tied to zero.
    %for (net, is_bus, tie_zero) in d2d_net_decls:
    %if is_bus:
    tri [2:0][19:0] ${net};
    %else:
    wire            ${net};
    %endif
    %if tie_zero:
    assign ${net} = const_zero;
    %endif
    %endfor
%endif

    dut i_dut (
%if not sim_with_verilator:
        /////////////////////////////////////
        // Off-array D2D ports
        /////////////////////////////////////
    %for (c, d) in dut_ports:
<%
    n = d2d_side_nets[(c[0], c[1], d)]
    P = "%d_%d" % (c[0], c[1])
%>\
        .${d}_d2d_link_${P}               (${n["bus"]}),
        .${d}_flow_control_rts_o_${P}     (${n["rts_o"]}),
        .${d}_flow_control_cts_i_${P}     (${n["cts_i"]}),
        .${d}_flow_control_rts_i_${P}     (${n["rts_i"]}),
        .${d}_flow_control_cts_o_${P}     (${n["cts_o"]}),
        .${d}_test_request_o_${P}         (${n["req_o"]}),
        .${d}_test_being_requested_i_${P} (${n["req_i"]}),
    %endfor
%endif
        /////////////////////////////////////
        // Each chiplet will have its own periphs
        /////////////////////////////////////
        %for compute_chip in compute_chips:
        <%
            comp_chip_x = compute_chip.coordinate[0]
            comp_chip_y = compute_chip.coordinate[1]
        %>
        // Chip ID
        .chip${comp_chip_x}${comp_chip_y}_id_i              (chip${comp_chip_x}${comp_chip_y}_id_i),
        // PLL
        .chip${comp_chip_x}${comp_chip_y}_pll_bypass_i      (chip${comp_chip_x}${comp_chip_y}_pll_bypass_i),
        .chip${comp_chip_x}${comp_chip_y}_pll_en_i          (chip${comp_chip_x}${comp_chip_y}_pll_en_i),
        .chip${comp_chip_x}${comp_chip_y}_pll_post_div_sel_i(chip${comp_chip_x}${comp_chip_y}_pll_post_div_sel_i),
        .chip${comp_chip_x}${comp_chip_y}_pll_lock_o        (chip${comp_chip_x}${comp_chip_y}_pll_lock_o),
        // UART
        .chip${comp_chip_x}${comp_chip_y}_uart_tx_o         (chip${comp_chip_x}${comp_chip_y}_uart_tx_o),
        .chip${comp_chip_x}${comp_chip_y}_uart_rx_i         (chip${comp_chip_x}${comp_chip_y}_uart_rx_i),
        .chip${comp_chip_x}${comp_chip_y}_uart_rts_no       (chip${comp_chip_x}${comp_chip_y}_uart_rts_no),
        .chip${comp_chip_x}${comp_chip_y}_uart_cts_ni       (chip${comp_chip_x}${comp_chip_y}_uart_cts_ni),
        // GPIO
        .chip${comp_chip_x}${comp_chip_y}_gpio              (chip${comp_chip_x}${comp_chip_y}_gpio),
        // SPI M
        .chip${comp_chip_x}${comp_chip_y}_spim_sck_o        (chip${comp_chip_x}${comp_chip_y}_spim_sck_o),
        .chip${comp_chip_x}${comp_chip_y}_spim_csb_o        (chip${comp_chip_x}${comp_chip_y}_spim_csb_o),
        .chip${comp_chip_x}${comp_chip_y}_spim_sd           (chip${comp_chip_x}${comp_chip_y}_spim_sd),
        // I2C
        .chip${comp_chip_x}${comp_chip_y}_i2c_sda           (chip${comp_chip_x}${comp_chip_y}_i2c_sda),
        .chip${comp_chip_x}${comp_chip_y}_i2c_scl           (chip${comp_chip_x}${comp_chip_y}_i2c_scl),
        // JTAG
        .chip${comp_chip_x}${comp_chip_y}_jtag_trst_ni      (chip${comp_chip_x}${comp_chip_y}_jtag_trst_ni),
        .chip${comp_chip_x}${comp_chip_y}_jtag_tck_i        (chip${comp_chip_x}${comp_chip_y}_jtag_tck_i),
        .chip${comp_chip_x}${comp_chip_y}_jtag_tms_i        (chip${comp_chip_x}${comp_chip_y}_jtag_tms_i),
        .chip${comp_chip_x}${comp_chip_y}_jtag_tdi_i        (chip${comp_chip_x}${comp_chip_y}_jtag_tdi_i),
        .chip${comp_chip_x}${comp_chip_y}_jtag_tdo_o        (chip${comp_chip_x}${comp_chip_y}_jtag_tdo_o),
        // CLK OBS
        .chip${comp_chip_x}${comp_chip_y}_clk_obs_o         (chip${comp_chip_x}${comp_chip_y}_clk_obs_o),
        %endfor
        // CLK and RST are shared by all the chiplets
        .mst_clk_i    (mst_clk_i    ),
        .rst_ni       (rst_ni       ),
        .periph_clk_i (periph_clk_i ),
        .rst_periph_ni(rst_periph_ni)
    );

%if not sim_with_verilator:
    // The memory chips: a PHY on every side that faces a chip, NumSysIdma push engines
    // (one per local port of its D2D link, so it can stream to that many neighbours at once).
    %for mem_chip in mem_chips:
<%
    mx, my = mem_chip.coordinate
    faces = {d: (mx, my, d) in d2d_side_nets for d in d2d_directions}
    ports = list(d2d_directions)
%>\

    hemaia_mem_chip #(
        .WideSRAMBankNum(16),
        .WideSRAMSize(${mem_chip.size}),
        // Simulated HBM, loaded by load_hbm (util/load_binary_*.sv)
        .EnableHbm(${1 if mem_chip.hbm else 0}),
        %if mem_chip.hbm:
        .HbmCfg(${mem_chip.hbm_sv}),
        %endif
        .EnableEastPhy(${int(faces["east"])}),
        .EnableWestPhy(${int(faces["west"])}),
        .EnableNorthPhy(${int(faces["north"])}),
        .EnableSouthPhy(${int(faces["south"])}),
        .NumSysIdma(${mem_chip.num_sys_idma}),
        .HostClkDiv(${memchip_clk_div})
    ) i_hemaia_mem_chip_${mx}_${my} (
        .clk_i    (mempool_clk_drv),
        .rst_ni   (rst_ni   ),
        .chip_id_i(mem_chip${mx}${my}_id_i),
        %for d in ports:
<%
    last = d == ports[-1]
    n = d2d_side_nets.get((mx, my, d))
    nb = chiplet_grid.neighbour((mx, my), d)
%>\
        %if n:
        // ${d}: to the ${"memory" if chiplet_grid.is_memory(nb) else "compute"} chip at (${nb[0]}, ${nb[1]})
        .${d}_d2d_io                (${n["bus"]}),
        .flow_control_${d}_rts_o    (${n["rts_o"]}),
        .flow_control_${d}_cts_i    (${n["cts_i"]}),
        .flow_control_${d}_rts_i    (${n["rts_i"]}),
        .flow_control_${d}_cts_o    (${n["cts_o"]}),
        .${d}_test_being_requested_i(${n["req_i"]}),
        .${d}_test_request_o        (${n["req_o"]})${"" if last else ","}
        %else:
        // ${d}: nothing there
        .${d}_d2d_io(),
        .flow_control_${d}_rts_o(),
        .flow_control_${d}_cts_i(const_zero),
        .flow_control_${d}_rts_i(const_zero),
        .flow_control_${d}_cts_o(),
        .${d}_test_being_requested_i(const_zero),
        .${d}_test_request_o()${"" if last else ","}
        %endif
        %endfor
    );
    %endfor
%endif

endmodule
