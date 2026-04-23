// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// DUT (top module): HeMAiA compute chiplet array + IO wrapper
//
// This module is the top-level Design Under Test. It:
//   1. Instantiates all hemaia compute chiplets
//   2. Exposes EVERY chiplet D2D port (all 4 directions) as internal wires
//   3. Delegates ALL D2D routing to the io_wrapper module
//   4. Connects per-chiplet peripherals directly to module ports
//   5. Passes peripheral/driving signals to io_wrapper (for interposer mode)
//
// This cleanly separates chip RTL (hemaia instances) from the interconnect
// fabric (io_wrapper), allowing different physical routing models to be
// selected via sim_with_interposer at template rendering time.
//
// Hierarchy:
//   dut
//   +-- i_hemaia_X_Y          (compute chiplets)
//   +-- i_io_wrapper           (interconnect routing)
//       +-- gen_direct / gen_interposer  (routing mode)
//
// Mesh layout (compute chiplets start at (0,0)):
//                    north boundary (offchip)
//              +--------+--------+--------+
//   west       | (0,0)  | (1,0)  | (2,0)  |  east
//   boundary   +--------+--------+--------+  boundary
//   (offchip)  | (0,1)  | (1,1)  | (2,1)  |  (offchip)
//              +--------+--------+--------+
//                    south boundary (offchip)
//
// Directions: east = +x, west = -x, south = +y, north = -y
// This module contains ONLY wires and connections (no behavioral code).

module dut (
%if not sim_with_verilator:
    /////////////////////////////////////
    // Off-chip D2D links to memory chips
    // (routed through io_wrapper to boundary chiplets)
    /////////////////////////////////////

    // North boundary (one per column, chiplets at y=0)
    %for x in range(max_compute_chiplet_x):
    inout tri [2:0][19:0] north_d2d_link_${x},
    inout wire             north_flow_control_rts_o_${x},
    inout wire             north_flow_control_cts_i_${x},
    inout wire             north_flow_control_rts_i_${x},
    inout wire             north_flow_control_cts_o_${x},
    inout wire             north_test_request_o_${x},
    inout wire             north_test_being_requested_i_${x},
    %endfor

    // South boundary (one per column, chiplets at y=${max_compute_chiplet_y - 1})
    %for x in range(max_compute_chiplet_x):
    inout tri [2:0][19:0] south_d2d_link_${x},
    inout wire             south_flow_control_rts_o_${x},
    inout wire             south_flow_control_cts_i_${x},
    inout wire             south_flow_control_rts_i_${x},
    inout wire             south_flow_control_cts_o_${x},
    inout wire             south_test_request_o_${x},
    inout wire             south_test_being_requested_i_${x},
    %endfor

    // West boundary (one per row, chiplets at x=0)
    %for y in range(max_compute_chiplet_y):
    inout tri [2:0][19:0] west_d2d_link_${y},
    inout wire             west_flow_control_rts_o_${y},
    inout wire             west_flow_control_cts_i_${y},
    inout wire             west_flow_control_rts_i_${y},
    inout wire             west_flow_control_cts_o_${y},
    inout wire             west_test_request_o_${y},
    inout wire             west_test_being_requested_i_${y},
    %endfor

    // East boundary (one per row, chiplets at x=${max_compute_chiplet_x - 1})
    %for y in range(max_compute_chiplet_y):
    inout tri [2:0][19:0] east_d2d_link_${y},
    inout wire             east_flow_control_rts_o_${y},
    inout wire             east_flow_control_cts_i_${y},
    inout wire             east_flow_control_rts_i_${y},
    inout wire             east_flow_control_cts_o_${y},
    inout wire             east_test_request_o_${y},
    inout wire             east_test_being_requested_i_${y},
    %endfor
%endif

    /////////////////////////////////////
    // Per-chiplet peripheral pins
    // (directly wired to hemaia instances AND passed to io_wrapper)
    /////////////////////////////////////
    %for compute_chip in compute_chips:
    <%
        cx = compute_chip.coordinate[0]
        cy = compute_chip.coordinate[1]
    %>
    // Chiplet (${cx}, ${cy})
    inout wire [7:0]      chip${cx}${cy}_id_i,
    inout wire            chip${cx}${cy}_pll_bypass_i,
    inout wire            chip${cx}${cy}_pll_en_i,
    inout wire [1:0]      chip${cx}${cy}_pll_post_div_sel_i,
    inout wire            chip${cx}${cy}_pll_lock_o,
    inout wire            chip${cx}${cy}_uart_tx_o,
    inout wire            chip${cx}${cy}_uart_rx_i,
    inout wire            chip${cx}${cy}_uart_rts_no,
    inout wire            chip${cx}${cy}_uart_cts_ni,
    inout wire [3:0]      chip${cx}${cy}_gpio,
    inout wire            chip${cx}${cy}_spim_sck_o,
    inout wire            chip${cx}${cy}_spim_csb_o,
    inout wire [3:0]      chip${cx}${cy}_spim_sd,
    inout wire            chip${cx}${cy}_i2c_sda,
    inout wire            chip${cx}${cy}_i2c_scl,
    inout wire            chip${cx}${cy}_jtag_trst_ni,
    inout wire            chip${cx}${cy}_jtag_tck_i,
    inout wire            chip${cx}${cy}_jtag_tms_i,
    inout wire            chip${cx}${cy}_jtag_tdi_i,
    inout wire            chip${cx}${cy}_jtag_tdo_o,
    inout wire            chip${cx}${cy}_clk_obs_o,
    %endfor

    // Shared clock and reset
    inout wire            mst_clk_i,
    inout wire            rst_ni,
    inout wire            periph_clk_i,
    inout wire            rst_periph_ni
);

    /////////////////////////////////////
    // Constants
    /////////////////////////////////////
    wire const_zero;
    wire const_one;
    assign const_zero = 1'b0;
    assign const_one  = 1'b1;

%if not sim_with_verilator:
    /////////////////////////////////////
    // Per-chiplet D2D wires
    // Each chiplet's 4 D2D interfaces are exposed as separate wires.
    // These wires connect to both the hemaia instance and the io_wrapper.
    // The io_wrapper decides how to route them (direct or interposer).
    /////////////////////////////////////
    %for compute_chip in compute_chips:
    <%
        cx = compute_chip.coordinate[0]
        cy = compute_chip.coordinate[1]
    %>
    // Chiplet (${cx}, ${cy}) D2D wires
    %for direction in ['east', 'west', 'north', 'south']:
    tri  [2:0][19:0] chip_${cx}_${cy}_${direction}_d2d;
    wire             chip_${cx}_${cy}_${direction}_rts_o;
    wire             chip_${cx}_${cy}_${direction}_cts_i;
    wire             chip_${cx}_${cy}_${direction}_rts_i;
    wire             chip_${cx}_${cy}_${direction}_cts_o;
    wire             chip_${cx}_${cy}_${direction}_test_request_o;
    wire             chip_${cx}_${cy}_${direction}_test_being_requested_i;
    %endfor

    %endfor
%endif

    /////////////////////////////////////
    // Compute chiplet instances
    /////////////////////////////////////
    %for compute_chip in compute_chips:
    <%
        cx = compute_chip.coordinate[0]
        cy = compute_chip.coordinate[1]
    %>
    // =============================================
    // Chiplet (${cx}, ${cy})
    // =============================================
    hemaia i_hemaia_${cx}_${cy} (
        // Clock, Reset, Aux
        .io_clk_i             (mst_clk_i),
        .io_rst_ni            (rst_ni),
        .io_clk_periph_i      (periph_clk_i),
        .io_rst_periph_ni     (rst_periph_ni),
        .io_test_mode_i       (const_zero),
        .io_boot_mode_i       (const_zero),
        .io_chip_id_i         (chip${cx}${cy}_id_i),
        .io_clk_obs_o         (chip${cx}${cy}_clk_obs_o),
        // PLL
        .io_pll_bypass_i      (chip${cx}${cy}_pll_bypass_i),
        .io_pll_en_i          (chip${cx}${cy}_pll_en_i),
        .io_pll_post_div_sel_i(chip${cx}${cy}_pll_post_div_sel_i),
        .io_pll_lock_o        (chip${cx}${cy}_pll_lock_o),
        // Peripherals
        .io_uart_tx_o         (chip${cx}${cy}_uart_tx_o),
        .io_uart_rx_i         (chip${cx}${cy}_uart_rx_i),
        .io_uart_rts_no       (chip${cx}${cy}_uart_rts_no),
        .io_uart_cts_ni       (chip${cx}${cy}_uart_cts_ni),
        .io_gpio              (chip${cx}${cy}_gpio),
        .io_spim_sck_o        (chip${cx}${cy}_spim_sck_o),
        .io_spim_csb_o        (chip${cx}${cy}_spim_csb_o),
        .io_spim_sd           (chip${cx}${cy}_spim_sd),
        .io_i2c_sda           (chip${cx}${cy}_i2c_sda),
        .io_i2c_scl           (chip${cx}${cy}_i2c_scl),
        .io_jtag_trst_ni      (chip${cx}${cy}_jtag_trst_ni),
        .io_jtag_tck_i        (chip${cx}${cy}_jtag_tck_i),
        .io_jtag_tms_i        (chip${cx}${cy}_jtag_tms_i),
        .io_jtag_tdi_i        (chip${cx}${cy}_jtag_tdi_i),
%if not sim_with_verilator and not multichip_cfg["single_chip"]:
        .io_jtag_tdo_o        (chip${cx}${cy}_jtag_tdo_o),
        // D2D — all 4 directions routed to io_wrapper
        .io_east_d2d                    (chip_${cx}_${cy}_east_d2d),
        .io_flow_control_east_rts_o     (chip_${cx}_${cy}_east_rts_o),
        .io_flow_control_east_cts_i     (chip_${cx}_${cy}_east_cts_i),
        .io_flow_control_east_rts_i     (chip_${cx}_${cy}_east_rts_i),
        .io_flow_control_east_cts_o     (chip_${cx}_${cy}_east_cts_o),
        .io_east_test_request_o         (chip_${cx}_${cy}_east_test_request_o),
        .io_east_test_being_requested_i (chip_${cx}_${cy}_east_test_being_requested_i),
        .io_west_d2d                    (chip_${cx}_${cy}_west_d2d),
        .io_flow_control_west_rts_o     (chip_${cx}_${cy}_west_rts_o),
        .io_flow_control_west_cts_i     (chip_${cx}_${cy}_west_cts_i),
        .io_flow_control_west_rts_i     (chip_${cx}_${cy}_west_rts_i),
        .io_flow_control_west_cts_o     (chip_${cx}_${cy}_west_cts_o),
        .io_west_test_request_o         (chip_${cx}_${cy}_west_test_request_o),
        .io_west_test_being_requested_i (chip_${cx}_${cy}_west_test_being_requested_i),
        .io_north_d2d                    (chip_${cx}_${cy}_north_d2d),
        .io_flow_control_north_rts_o     (chip_${cx}_${cy}_north_rts_o),
        .io_flow_control_north_cts_i     (chip_${cx}_${cy}_north_cts_i),
        .io_flow_control_north_rts_i     (chip_${cx}_${cy}_north_rts_i),
        .io_flow_control_north_cts_o     (chip_${cx}_${cy}_north_cts_o),
        .io_north_test_request_o         (chip_${cx}_${cy}_north_test_request_o),
        .io_north_test_being_requested_i (chip_${cx}_${cy}_north_test_being_requested_i),
        .io_south_d2d                    (chip_${cx}_${cy}_south_d2d),
        .io_flow_control_south_rts_o     (chip_${cx}_${cy}_south_rts_o),
        .io_flow_control_south_cts_i     (chip_${cx}_${cy}_south_cts_i),
        .io_flow_control_south_rts_i     (chip_${cx}_${cy}_south_rts_i),
        .io_flow_control_south_cts_o     (chip_${cx}_${cy}_south_cts_o),
        .io_south_test_request_o         (chip_${cx}_${cy}_south_test_request_o),
        .io_south_test_being_requested_i (chip_${cx}_${cy}_south_test_being_requested_i)
%else:
        .io_jtag_tdo_o        (chip${cx}${cy}_jtag_tdo_o)
%endif
    );
    %endfor

%if not sim_with_verilator:

    /////////////////////////////////////
    // IO Wrapper instance
    // Handles ALL D2D routing between chiplets and to off-chip boundaries.
    // Peripheral/driving signals are also passed for interposer mode
    // (where the chippad_powerpad contains its own hemaia instances).
    /////////////////////////////////////

    io_wrapper i_io_wrapper (
        // ---- Per-chiplet D2D ports ----
        %for compute_chip in compute_chips:
        <%
            cx = compute_chip.coordinate[0]
            cy = compute_chip.coordinate[1]
        %>
        %for direction in ['east', 'west', 'north', 'south']:
        .chip_${cx}_${cy}_${direction}_d2d                   (chip_${cx}_${cy}_${direction}_d2d),
        .chip_${cx}_${cy}_${direction}_rts_o                 (chip_${cx}_${cy}_${direction}_rts_o),
        .chip_${cx}_${cy}_${direction}_cts_i                 (chip_${cx}_${cy}_${direction}_cts_i),
        .chip_${cx}_${cy}_${direction}_rts_i                 (chip_${cx}_${cy}_${direction}_rts_i),
        .chip_${cx}_${cy}_${direction}_cts_o                 (chip_${cx}_${cy}_${direction}_cts_o),
        .chip_${cx}_${cy}_${direction}_test_request_o        (chip_${cx}_${cy}_${direction}_test_request_o),
        .chip_${cx}_${cy}_${direction}_test_being_requested_i(chip_${cx}_${cy}_${direction}_test_being_requested_i),
        %endfor
        %endfor

        // ---- Off-chip boundary D2D ports ----
        %for x in range(max_compute_chiplet_x):
        .north_d2d_link_${x}                  (north_d2d_link_${x}),
        .north_flow_control_rts_o_${x}        (north_flow_control_rts_o_${x}),
        .north_flow_control_cts_i_${x}        (north_flow_control_cts_i_${x}),
        .north_flow_control_rts_i_${x}        (north_flow_control_rts_i_${x}),
        .north_flow_control_cts_o_${x}        (north_flow_control_cts_o_${x}),
        .north_test_request_o_${x}            (north_test_request_o_${x}),
        .north_test_being_requested_i_${x}    (north_test_being_requested_i_${x}),
        .south_d2d_link_${x}                  (south_d2d_link_${x}),
        .south_flow_control_rts_o_${x}        (south_flow_control_rts_o_${x}),
        .south_flow_control_cts_i_${x}        (south_flow_control_cts_i_${x}),
        .south_flow_control_rts_i_${x}        (south_flow_control_rts_i_${x}),
        .south_flow_control_cts_o_${x}        (south_flow_control_cts_o_${x}),
        .south_test_request_o_${x}            (south_test_request_o_${x}),
        .south_test_being_requested_i_${x}    (south_test_being_requested_i_${x}),
        %endfor

        %for y in range(max_compute_chiplet_y):
        .west_d2d_link_${y}                   (west_d2d_link_${y}),
        .west_flow_control_rts_o_${y}         (west_flow_control_rts_o_${y}),
        .west_flow_control_cts_i_${y}         (west_flow_control_cts_i_${y}),
        .west_flow_control_rts_i_${y}         (west_flow_control_rts_i_${y}),
        .west_flow_control_cts_o_${y}         (west_flow_control_cts_o_${y}),
        .west_test_request_o_${y}             (west_test_request_o_${y}),
        .west_test_being_requested_i_${y}     (west_test_being_requested_i_${y}),
        %endfor
        %for y in range(max_compute_chiplet_y):
        .east_d2d_link_${y}                   (east_d2d_link_${y}),
        .east_flow_control_rts_o_${y}         (east_flow_control_rts_o_${y}),
        .east_flow_control_cts_i_${y}         (east_flow_control_cts_i_${y}),
        .east_flow_control_rts_i_${y}         (east_flow_control_rts_i_${y}),
        .east_flow_control_cts_o_${y}         (east_flow_control_cts_o_${y}),
        .east_test_request_o_${y}             (east_test_request_o_${y}),
        .east_test_being_requested_i_${y}     (east_test_being_requested_i_${y}),
        %endfor

        // ---- Driving signals (for interposer mode) ----
        .mst_clk_i    (mst_clk_i),
        .rst_ni       (rst_ni),
        .periph_clk_i (periph_clk_i),
        .rst_periph_ni(rst_periph_ni),
        %for compute_chip in compute_chips:
        <%
            cx = compute_chip.coordinate[0]
            cy = compute_chip.coordinate[1]
        %>
        .chip${cx}${cy}_id_i              (chip${cx}${cy}_id_i),
        .chip${cx}${cy}_pll_bypass_i      (chip${cx}${cy}_pll_bypass_i),
        .chip${cx}${cy}_pll_en_i          (chip${cx}${cy}_pll_en_i),
        .chip${cx}${cy}_pll_post_div_sel_i(chip${cx}${cy}_pll_post_div_sel_i),
        .chip${cx}${cy}_pll_lock_o        (chip${cx}${cy}_pll_lock_o),
        .chip${cx}${cy}_uart_tx_o         (chip${cx}${cy}_uart_tx_o),
        .chip${cx}${cy}_uart_rx_i         (chip${cx}${cy}_uart_rx_i),
        .chip${cx}${cy}_uart_rts_no       (chip${cx}${cy}_uart_rts_no),
        .chip${cx}${cy}_uart_cts_ni       (chip${cx}${cy}_uart_cts_ni),
        .chip${cx}${cy}_gpio              (chip${cx}${cy}_gpio),
        .chip${cx}${cy}_spim_sck_o        (chip${cx}${cy}_spim_sck_o),
        .chip${cx}${cy}_spim_csb_o        (chip${cx}${cy}_spim_csb_o),
        .chip${cx}${cy}_spim_sd           (chip${cx}${cy}_spim_sd),
        .chip${cx}${cy}_i2c_sda           (chip${cx}${cy}_i2c_sda),
        .chip${cx}${cy}_i2c_scl           (chip${cx}${cy}_i2c_scl),
        .chip${cx}${cy}_jtag_trst_ni      (chip${cx}${cy}_jtag_trst_ni),
        .chip${cx}${cy}_jtag_tck_i        (chip${cx}${cy}_jtag_tck_i),
        .chip${cx}${cy}_jtag_tms_i        (chip${cx}${cy}_jtag_tms_i),
        .chip${cx}${cy}_jtag_tdi_i        (chip${cx}${cy}_jtag_tdi_i),
        .chip${cx}${cy}_jtag_tdo_o        (chip${cx}${cy}_jtag_tdo_o),
        %if compute_chip == compute_chips[-1]:
        .chip${cx}${cy}_clk_obs_o         (chip${cx}${cy}_clk_obs_o)
        %else:
        .chip${cx}${cy}_clk_obs_o         (chip${cx}${cy}_clk_obs_o),
        %endif
        %endfor
    );
%endif

endmodule
