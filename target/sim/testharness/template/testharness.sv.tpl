// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// This is the testharness of the Hemaia chip
// In the test harness there will be have three things：
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
    localparam int SIM_WITH_PLL = ${sim_with_pll};
    %if pll_present:
    `define TIMESCALEVAL       1.0e-12    // 1ps
    `define CLK_FREF_FREQ_MHZ  40.0
    `define PRI_FREQ_MHZ       32.0
    `define MEMPOOL_FREQ_MHZ   200.0  // 200 MHz, 1/20 of the main clock
    %else:
    `define TIMESCALEVAL       1.0e-9 // 1ns
    `define CLK_FREF_FREQ_MHZ  500.0 // 500 MHz
    `define PRI_FREQ_MHZ       500.0 // 500 MHz
    `define MEMPOOL_FREQ_MHZ   25.0  //  25 MHz, 1/20 of the main clock
    %endif
    `define CLK_FREF_PERIOD     (1.0e-6/`CLK_FREF_FREQ_MHZ/`TIMESCALEVAL)
    `define PRI_FREF_PERIOD     (1.0e-6/`PRI_FREQ_MHZ/`TIMESCALEVAL)
    `define MEMPOOL_FREF_PERIOD (1.0e-6/`MEMPOOL_FREQ_MHZ/`TIMESCALEVAL)
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
    // Vendor PLL clk monitor
    %for compute_chip in compute_chips:
    <%
        comp_chip_x = compute_chip.coordinate[0]
        comp_chip_y = compute_chip.coordinate[1]
    %>
    clkmon #(.counts(2500), .pcterr(0.35)
    ) i_pll_mon_${comp_chip_x}_${comp_chip_y} (
        .CLK_REF(i_dut.i_hemaia_${comp_chip_x}_${comp_chip_y}.i_occamy_chip.i_hemaia_clk_rst_controller.i_pll.clk_o),
        .DONE(),
        .TIMEOUT()
    );
    %endfor
    task check_frequency();
        begin
        %for compute_chip in compute_chips:
        <%
            comp_chip_x = compute_chip.coordinate[0]
            comp_chip_y = compute_chip.coordinate[1]
        %>
            i_pll_mon_${comp_chip_x}_${comp_chip_y}.run();
        %endfor
        end
    endtask

    task enable_pll_and_wait_lock();
        begin
        // Wait at least 1us
        #(1us);
        // PLL on
        // Drive the enable pin in sequence
        %for compute_chip in compute_chips:
        chip${compute_chip.coordinate[0]}${compute_chip.coordinate[1]}_pll_en_drv = 1'b1;
        // Wait for phase locked
        @(posedge chip${compute_chip.coordinate[0]}${compute_chip.coordinate[1]}_pll_lock_o);
        $display("Chip ${compute_chip.coordinate[0]}${compute_chip.coordinate[1]}'s PLL Lock asserted!");
        %endfor
        check_frequency;        
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
    // Tie to zero
    assign chip${comp_chip_x}${comp_chip_y}_jtag_trst_ni = const_zero;
    assign chip${comp_chip_x}${comp_chip_y}_jtag_tck_i   = const_zero;
    assign chip${comp_chip_x}${comp_chip_y}_jtag_tms_i   = const_zero;
    assign chip${comp_chip_x}${comp_chip_y}_jtag_tdi_i   = const_zero;
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
    initial begin
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

    // The compute chiplet - DUT(The main ASIC)
    // The memory chiplet - External Memory Pool(Will be implemented on a FPGA)
    // And talk to the ASIC also via the D2D Link
    // The possible locations of the mem chip is around the ASIC
    // For example a 3x2 chiplet
    // The compute chip always starts from (x=0,y=0)
    //+------+------------------------+------------------------+------------------------+------------------------+------------------------+
    //|      | x=-1                   | x=0                    | x=1                    | x=2                    | x=3                    |
    //+------+------------------------+------------------------+------------------------+------------------------+------------------------+
    //| y=-1 | illegal                | possible mem_chip slot | possible mem_chip slot | possible mem_chip slot | illegal                |
    //+------+------------------------+------------------------+------------------------+------------------------+------------------------+
    //| y=0  | possible mem_chip slot | comp_chip_00           | comp_chip_10           | comp_chip_20           | possible mem_chip slot |
    //+------+------------------------+------------------------+------------------------+------------------------+------------------------+
    //| y=1  | possible mem_chip slot | comp_chip_01           | comp_chip_11           | comp_chip_21           | possible mem_chip slot |
    //+------+------------------------+------------------------+------------------------+------------------------+------------------------+
    //| y=2  | illegal                | possible mem_chip slot | possible mem_chip slot | possible mem_chip slot | illegal                |
    //+------+------------------------+------------------------+------------------------+------------------------+------------------------+

%if not sim_with_verilator:
    // Off-chip links to Mem Chip
    // North Links (#MAX_X links)
    %for x in range(max_compute_chiplet_x):
    tri [2:0][19:0] north_offchip_d2d_link_${x};
    wire            north_memchip_to_dut_link_rts_${x};
    wire            north_memchip_to_dut_link_cts_${x};
    wire            north_dut_to_memchip_link_rts_${x};
    wire            north_dut_to_memchip_link_cts_${x};
    wire            north_dut_to_memchip_link_test_request_${x};
    wire            north_memchip_to_dut_link_test_request_${x};
    %endfor

    // South Links (#MAX_X links)
    %for x in range(max_compute_chiplet_x):
    tri [2:0][19:0] south_offchip_d2d_link_${x};
    wire            south_memchip_to_dut_link_rts_${x};
    wire            south_memchip_to_dut_link_cts_${x};
    wire            south_dut_to_memchip_link_rts_${x};
    wire            south_dut_to_memchip_link_cts_${x};
    wire            south_dut_to_memchip_link_test_request_${x};
    wire            south_memchip_to_dut_link_test_request_${x};
    %endfor

    // West Links (#MAX_Y links)
    %for y in range(max_compute_chiplet_y):
    tri [2:0][19:0] west_offchip_d2d_link_${y};
    wire            west_memchip_to_dut_link_rts_${y};
    wire            west_memchip_to_dut_link_cts_${y};
    wire            west_dut_to_memchip_link_rts_${y};
    wire            west_dut_to_memchip_link_cts_${y};
    wire            west_dut_to_memchip_link_test_request_${y};
    wire            west_memchip_to_dut_link_test_request_${y};
    %endfor

    // East Links (#MAX_Y links)
    %for y in range(max_compute_chiplet_y):
    tri [2:0][19:0] east_offchip_d2d_link_${y};
    wire            east_memchip_to_dut_link_rts_${y};
    wire            east_memchip_to_dut_link_cts_${y};
    wire            east_dut_to_memchip_link_rts_${y};
    wire            east_dut_to_memchip_link_cts_${y};
    wire            east_dut_to_memchip_link_test_request_${y};
    wire            east_memchip_to_dut_link_test_request_${y};
    %endfor

    // Tie unconnected offchip D2D link input pins to const_zero
    // to avoid floating inputs on the DUT
    <%
        occupied_north = set()
        occupied_south = set()
        occupied_west  = set()
        occupied_east  = set()
        for mem_chip in mem_chips:
            mx = mem_chip.coordinate[0]
            my = mem_chip.coordinate[1]
            if my == -1:
                occupied_north.add(mx)
            if my == max_compute_chiplet_y:
                occupied_south.add(mx)
            if mx == -1:
                occupied_west.add(my)
            if mx == max_compute_chiplet_x:
                occupied_east.add(my)
    %>
    %for x in range(max_compute_chiplet_x):
    %if x not in occupied_north:
    // North link ${x} is not connected to any mem chip
    assign north_dut_to_memchip_link_cts_${x}      = const_zero;
    assign north_memchip_to_dut_link_rts_${x}       = const_zero;
    assign north_memchip_to_dut_link_test_request_${x} = const_zero;
    %endif
    %endfor
    %for x in range(max_compute_chiplet_x):
    %if x not in occupied_south:
    // South link ${x} is not connected to any mem chip
    assign south_dut_to_memchip_link_cts_${x}      = const_zero;
    assign south_memchip_to_dut_link_rts_${x}       = const_zero;
    assign south_memchip_to_dut_link_test_request_${x} = const_zero;
    %endif
    %endfor
    %for y in range(max_compute_chiplet_y):
    %if y not in occupied_west:
    // West link ${y} is not connected to any mem chip
    assign west_dut_to_memchip_link_cts_${y}      = const_zero;
    assign west_memchip_to_dut_link_rts_${y}       = const_zero;
    assign west_memchip_to_dut_link_test_request_${y} = const_zero;
    %endif
    %endfor
    %for y in range(max_compute_chiplet_y):
    %if y not in occupied_east:
    // East link ${y} is not connected to any mem chip
    assign east_dut_to_memchip_link_cts_${y}      = const_zero;
    assign east_memchip_to_dut_link_rts_${y}       = const_zero;
    assign east_memchip_to_dut_link_test_request_${y} = const_zero;
    %endif
    %endfor
%endif

    dut i_dut (
%if not sim_with_verilator:
        /////////////////////////////////////
        // Offchip D2D Links to Memchips
        /////////////////////////////////////
        // North Links (#MAX_X links)
    %for x in range(max_compute_chiplet_x):
        .north_d2d_link_${x}          (north_offchip_d2d_link_${x}),
        .north_flow_control_rts_o_${x}(north_dut_to_memchip_link_rts_${x}),
        .north_flow_control_cts_i_${x}(north_dut_to_memchip_link_cts_${x}),
        .north_flow_control_rts_i_${x}(north_memchip_to_dut_link_rts_${x}),
        .north_flow_control_cts_o_${x}(north_memchip_to_dut_link_cts_${x}),
        .north_test_request_o_${x}           (north_dut_to_memchip_link_test_request_${x}),
        .north_test_being_requested_i_${x}   (north_memchip_to_dut_link_test_request_${x}),
    %endfor
        // South Links (#MAX_X links)
    %for x in range(max_compute_chiplet_x):
        .south_d2d_link_${x}          (south_offchip_d2d_link_${x}),
        .south_flow_control_rts_o_${x}(south_dut_to_memchip_link_rts_${x}),
        .south_flow_control_cts_i_${x}(south_dut_to_memchip_link_cts_${x}),
        .south_flow_control_rts_i_${x}(south_memchip_to_dut_link_rts_${x}),
        .south_flow_control_cts_o_${x}(south_memchip_to_dut_link_cts_${x}),
        .south_test_request_o_${x}           (south_dut_to_memchip_link_test_request_${x}),
        .south_test_being_requested_i_${x}   (south_memchip_to_dut_link_test_request_${x}),
    %endfor
        // West Links (#MAX_Y links)
    %for y in range(max_compute_chiplet_y):
        .west_d2d_link_${y}          (west_offchip_d2d_link_${y}),
        .west_flow_control_rts_o_${y}(west_dut_to_memchip_link_rts_${y}),
        .west_flow_control_cts_i_${y}(west_dut_to_memchip_link_cts_${y}),
        .west_flow_control_rts_i_${y}(west_memchip_to_dut_link_rts_${y}),
        .west_flow_control_cts_o_${y}(west_memchip_to_dut_link_cts_${y}),
        .west_test_request_o_${y}           (west_dut_to_memchip_link_test_request_${y}),
        .west_test_being_requested_i_${y}   (west_memchip_to_dut_link_test_request_${y}),
    %endfor
        // East Links (#MAX_Y links)
    %for y in range(max_compute_chiplet_y):
        .east_d2d_link_${y}          (east_offchip_d2d_link_${y}),
        .east_flow_control_rts_o_${y}(east_dut_to_memchip_link_rts_${y}),
        .east_flow_control_cts_i_${y}(east_dut_to_memchip_link_cts_${y}),
        .east_flow_control_rts_i_${y}(east_memchip_to_dut_link_rts_${y}),
        .east_flow_control_cts_o_${y}(east_memchip_to_dut_link_cts_${y}),
        .east_test_request_o_${y}           (east_dut_to_memchip_link_test_request_${y}),
        .east_test_being_requested_i_${y}   (east_memchip_to_dut_link_test_request_${y}),
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
    // The mem chips
    %for mem_chip in mem_chips:
    <%
        mem_chip_x = mem_chip.coordinate[0]
        mem_chip_y = mem_chip.coordinate[1]
        enable_north = 0
        enable_south = 0
        enable_west  = 0
        enable_east  = 0
        #// Check the mem chip location
        #// if mem chip on the west side of the asic
        if mem_chip_x == -1:
            enable_east = 1
        #// if mem chip on the east side of the asic
        if mem_chip_x == max_compute_chiplet_x:
            enable_west = 1
        #// if mem chip on the north side of the asic
        if mem_chip_y == -1:
            enable_south = 1
        #// if mem chip on the south side of the asic
        if mem_chip_y == max_compute_chiplet_y:
            enable_north = 1
        #// Sanity Check
        #// Can only be 1
        enable_sum = enable_north+enable_south+enable_west+enable_east
        assert enable_sum == 1, "Illegal Memchip location!"
    %>

    hemaia_mem_chip #(
        .WideSRAMBankNum(16),
        .WideSRAMSize(${mem_chip.size}),
        .EnableEastPhy(${enable_east}),
        .EnableWestPhy(${enable_west}),
        .EnableNorthPhy(${enable_north}),
        .EnableSouthPhy(${enable_south})
    ) i_hemaia_mem_chip_${mem_chip_x}_${mem_chip_y} (
        .clk_i    (mempool_clk_drv),
        .rst_ni   (rst_ni   ),
        .chip_id_i(mem_chip${mem_chip_x}${mem_chip_y}_id_i),
        // Memchip East D2D Links
        // Should be connected to the Offchip West D2D Link of the asic
        %if enable_east:
        .east_d2d_io                (west_offchip_d2d_link_${mem_chip_y}),
        .flow_control_east_rts_o    (west_memchip_to_dut_link_rts_${mem_chip_y}),
        .flow_control_east_cts_i    (west_memchip_to_dut_link_cts_${mem_chip_y}),
        .flow_control_east_rts_i    (west_dut_to_memchip_link_rts_${mem_chip_y}),
        .flow_control_east_cts_o    (west_dut_to_memchip_link_cts_${mem_chip_y}),
        .east_test_being_requested_i(west_dut_to_memchip_link_test_request_${mem_chip_y}),
        .east_test_request_o        (west_memchip_to_dut_link_test_request_${mem_chip_y}),
        %else:
        .east_d2d_io(),
        .flow_control_east_rts_o(),
        .flow_control_east_cts_i(const_zero),
        .flow_control_east_rts_i(const_zero),
        .flow_control_east_cts_o(),
        .east_test_being_requested_i(const_zero),
        .east_test_request_o(),        
        %endif
        // Memchip West D2D Links
        // Should be connected to the Offchip East D2D Link of the asic
        %if enable_west:
        .west_d2d_io                (east_offchip_d2d_link_${mem_chip_y}),
        .flow_control_west_rts_o    (east_memchip_to_dut_link_rts_${mem_chip_y}),
        .flow_control_west_cts_i    (east_memchip_to_dut_link_cts_${mem_chip_y}),
        .flow_control_west_rts_i    (east_dut_to_memchip_link_rts_${mem_chip_y}),
        .flow_control_west_cts_o    (east_dut_to_memchip_link_cts_${mem_chip_y}),
        .west_test_being_requested_i(east_dut_to_memchip_link_test_request_${mem_chip_y}),
        .west_test_request_o        (east_memchip_to_dut_link_test_request_${mem_chip_y}),
        %else:
        .west_d2d_io(),
        .flow_control_west_rts_o(),
        .flow_control_west_cts_i(const_zero),
        .flow_control_west_rts_i(const_zero),
        .flow_control_west_cts_o(),
        .west_test_being_requested_i(const_zero),
        .west_test_request_o(),        
        %endif
        // Memchip North D2D Links
        // Should be connected to the Offchip South D2D Link of the asic
        %if enable_north:
        .north_d2d_io                (south_offchip_d2d_link_${mem_chip_x}),
        .flow_control_north_rts_o    (south_memchip_to_dut_link_rts_${mem_chip_x}),
        .flow_control_north_cts_i    (south_memchip_to_dut_link_cts_${mem_chip_x}),
        .flow_control_north_rts_i    (south_dut_to_memchip_link_rts_${mem_chip_x}),
        .flow_control_north_cts_o    (south_dut_to_memchip_link_cts_${mem_chip_x}),
        .north_test_being_requested_i(south_dut_to_memchip_link_test_request_${mem_chip_x}),
        .north_test_request_o        (south_memchip_to_dut_link_test_request_${mem_chip_x}),
        %else:
        .north_d2d_io(),
        .flow_control_north_rts_o(),
        .flow_control_north_cts_i(const_zero),
        .flow_control_north_rts_i(const_zero),
        .flow_control_north_cts_o(),
        .north_test_being_requested_i(const_zero),
        .north_test_request_o(),        
        %endif
        // Memchip South D2D Links
        // Should be connected to the Offchip North D2D Link of the asic
        %if enable_south:
        .south_d2d_io                (north_offchip_d2d_link_${mem_chip_x}),
        .flow_control_south_rts_o    (north_memchip_to_dut_link_rts_${mem_chip_x}),
        .flow_control_south_cts_i    (north_memchip_to_dut_link_cts_${mem_chip_x}),
        .flow_control_south_rts_i    (north_dut_to_memchip_link_rts_${mem_chip_x}),
        .flow_control_south_cts_o    (north_dut_to_memchip_link_cts_${mem_chip_x}),
        .south_test_being_requested_i(north_dut_to_memchip_link_test_request_${mem_chip_x}),
        .south_test_request_o        (north_memchip_to_dut_link_test_request_${mem_chip_x})
        %else:
        .south_d2d_io(),
        .flow_control_south_rts_o(),
        .flow_control_south_cts_i(const_zero),
        .flow_control_south_rts_i(const_zero),
        .flow_control_south_cts_o(),
        .south_test_being_requested_i(const_zero),
        .south_test_request_o()
        %endif
    );
    %endfor
%endif

endmodule