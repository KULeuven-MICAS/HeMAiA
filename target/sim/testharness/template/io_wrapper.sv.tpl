// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// IO Wrapper: D2D interconnect routing fabric + IO pad simulation
//
// This module receives ALL chiplet D2D ports from the DUT and handles routing:
//   - Internal mesh: connects adjacent chiplets' facing D2D ports
//   - Boundary: connects boundary chiplets' outward D2D ports to off-chip ports
//
// SIM_WITH_INTERPOSER = 0 (gen_direct):
//   Direct wire connections. Adjacent chiplets' D2D data buses are shorted
//   using tran gates. Flow control and test signals use assign.
//   This models an ideal interconnect with no physical effects.
//
// SIM_WITH_INTERPOSER = 1 (gen_interposer):
//   Wraps hemaia RTL into hemaia_io_pad (IO pad model), then wraps all 4
//   chiplets into four_chiplet_interposer_chippad_powerpad (interposer
//   physical routing model). Currently supports 2x2 array only.
//   Driving signals (clk, rst, peripherals) are routed from the DUT.
//
// This module contains ONLY wires and connections (no behavioral code).
//
// Signal direction convention (from chiplet's perspective):
//   _rts_o                : chiplet drives RTS outward
//   _cts_i                : chiplet receives CTS from outside
//   _rts_i                : chiplet receives RTS from neighbor
//   _cts_o                : chiplet drives CTS to neighbor
//   _test_request_o       : chiplet drives test request outward
//   _test_being_requested_i: chiplet receives test request from neighbor

module io_wrapper #(
    parameter int SIM_WITH_INTERPOSER = 0
) (
    /////////////////////////////////////
    // Per-chiplet D2D ports (all 4 directions)
    /////////////////////////////////////
    %for compute_chip in compute_chips:
    <%
        cx = compute_chip.coordinate[0]
        cy = compute_chip.coordinate[1]
    %>
    %for direction in ['east', 'west', 'north', 'south']:
    inout wire [2:0][19:0] chip_${cx}_${cy}_${direction}_d2d,
    inout wire             chip_${cx}_${cy}_${direction}_rts_o,
    inout wire             chip_${cx}_${cy}_${direction}_cts_i,
    inout wire             chip_${cx}_${cy}_${direction}_rts_i,
    inout wire             chip_${cx}_${cy}_${direction}_cts_o,
    inout wire             chip_${cx}_${cy}_${direction}_test_request_o,
    inout wire             chip_${cx}_${cy}_${direction}_test_being_requested_i,
    %endfor
    %endfor

    /////////////////////////////////////
    // Per-chiplet driving / peripheral signals
    /////////////////////////////////////
    %for compute_chip in compute_chips:
    <%
        cx = compute_chip.coordinate[0]
        cy = compute_chip.coordinate[1]
    %>
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
    inout wire            rst_periph_ni,

    /////////////////////////////////////
    // Off-chip boundary D2D ports
    /////////////////////////////////////
    %for x in range(max_compute_chiplet_x):
    inout wire [2:0][19:0] north_d2d_link_${x},
    inout wire             north_flow_control_rts_o_${x},
    inout wire             north_flow_control_cts_i_${x},
    inout wire             north_flow_control_rts_i_${x},
    inout wire             north_flow_control_cts_o_${x},
    inout wire             north_test_request_o_${x},
    inout wire             north_test_being_requested_i_${x},
    %endfor
    %for x in range(max_compute_chiplet_x):
    inout wire [2:0][19:0] south_d2d_link_${x},
    inout wire             south_flow_control_rts_o_${x},
    inout wire             south_flow_control_cts_i_${x},
    inout wire             south_flow_control_rts_i_${x},
    inout wire             south_flow_control_cts_o_${x},
    inout wire             south_test_request_o_${x},
    inout wire             south_test_being_requested_i_${x},
    %endfor
    %for y in range(max_compute_chiplet_y):
    inout wire [2:0][19:0] west_d2d_link_${y},
    inout wire             west_flow_control_rts_o_${y},
    inout wire             west_flow_control_cts_i_${y},
    inout wire             west_flow_control_rts_i_${y},
    inout wire             west_flow_control_cts_o_${y},
    inout wire             west_test_request_o_${y},
    inout wire             west_test_being_requested_i_${y},
    %endfor
    %for y in range(max_compute_chiplet_y):
    inout wire [2:0][19:0] east_d2d_link_${y},
    inout wire             east_flow_control_rts_o_${y},
    inout wire             east_flow_control_cts_i_${y},
    inout wire             east_flow_control_rts_i_${y},
    inout wire             east_flow_control_cts_o_${y},
    inout wire             east_test_request_o_${y},
    %if y < max_compute_chiplet_y - 1:
    inout wire             east_test_being_requested_i_${y},
    %else:
    inout wire             east_test_being_requested_i_${y}
    %endif
    %endfor
);

wire const_zero;
wire const_one;
assign const_zero = 1'b0;
assign const_one  = 1'b1;

generate
if (SIM_WITH_INTERPOSER == 0) begin : gen_direct
    // ==========================================================
    // Mode 0: Direct wire connections (ideal interconnect)
    // ==========================================================

    // ---- Internal mesh: East-West links ----
    %for x in range(max_compute_chiplet_x - 1):
    %for y in range(max_compute_chiplet_y):
    // (${x}, ${y}) east <--> (${x + 1}, ${y}) west
    for (genvar r = 0; r < 3; r++) begin : gen_ew_d2d_${x}_${y}_r
        for (genvar c = 0; c < 20; c++) begin : gen_ew_d2d_${x}_${y}_c
            tran(chip_${x}_${y}_east_d2d[r][c], chip_${x + 1}_${y}_west_d2d[r][c]);
        end
    end
    assign chip_${x + 1}_${y}_west_rts_i                  = chip_${x}_${y}_east_rts_o;
    assign chip_${x}_${y}_east_rts_i                      = chip_${x + 1}_${y}_west_rts_o;
    assign chip_${x}_${y}_east_cts_i                      = chip_${x + 1}_${y}_west_cts_o;
    assign chip_${x + 1}_${y}_west_cts_i                  = chip_${x}_${y}_east_cts_o;
    assign chip_${x + 1}_${y}_west_test_being_requested_i = chip_${x}_${y}_east_test_request_o;
    assign chip_${x}_${y}_east_test_being_requested_i     = chip_${x + 1}_${y}_west_test_request_o;
    %endfor
    %endfor

    // ---- Internal mesh: North-South links ----
    %for x in range(max_compute_chiplet_x):
    %for y in range(max_compute_chiplet_y - 1):
    // (${x}, ${y}) south <--> (${x}, ${y + 1}) north
    for (genvar r = 0; r < 3; r++) begin : gen_ns_d2d_${x}_${y}_r
        for (genvar c = 0; c < 20; c++) begin : gen_ns_d2d_${x}_${y}_c
            tran(chip_${x}_${y}_south_d2d[r][c], chip_${x}_${y + 1}_north_d2d[r][c]);
        end
    end
    assign chip_${x}_${y + 1}_north_rts_i                  = chip_${x}_${y}_south_rts_o;
    assign chip_${x}_${y}_south_rts_i                      = chip_${x}_${y + 1}_north_rts_o;
    assign chip_${x}_${y}_south_cts_i                      = chip_${x}_${y + 1}_north_cts_o;
    assign chip_${x}_${y + 1}_north_cts_i                  = chip_${x}_${y}_south_cts_o;
    assign chip_${x}_${y + 1}_north_test_being_requested_i = chip_${x}_${y}_south_test_request_o;
    assign chip_${x}_${y}_south_test_being_requested_i     = chip_${x}_${y + 1}_north_test_request_o;
    %endfor
    %endfor

    // ---- North boundary (y=0) ----
    %for x in range(max_compute_chiplet_x):
    for (genvar r = 0; r < 3; r++) begin : gen_nb_d2d_${x}_r
        for (genvar c = 0; c < 20; c++) begin : gen_nb_d2d_${x}_c
            tran(chip_${x}_0_north_d2d[r][c], north_d2d_link_${x}[r][c]);
        end
    end
    assign north_flow_control_rts_o_${x}            = chip_${x}_0_north_rts_o;
    assign chip_${x}_0_north_cts_i                  = north_flow_control_cts_i_${x};
    assign chip_${x}_0_north_rts_i                  = north_flow_control_rts_i_${x};
    assign north_flow_control_cts_o_${x}            = chip_${x}_0_north_cts_o;
    assign north_test_request_o_${x}                = chip_${x}_0_north_test_request_o;
    assign chip_${x}_0_north_test_being_requested_i = north_test_being_requested_i_${x};
    %endfor

    // ---- South boundary (y=${max_compute_chiplet_y - 1}) ----
    <% south_y = max_compute_chiplet_y - 1 %>
    %for x in range(max_compute_chiplet_x):
    for (genvar r = 0; r < 3; r++) begin : gen_sb_d2d_${x}_r
        for (genvar c = 0; c < 20; c++) begin : gen_sb_d2d_${x}_c
            tran(chip_${x}_${south_y}_south_d2d[r][c], south_d2d_link_${x}[r][c]);
        end
    end
    assign south_flow_control_rts_o_${x}                     = chip_${x}_${south_y}_south_rts_o;
    assign chip_${x}_${south_y}_south_cts_i                  = south_flow_control_cts_i_${x};
    assign chip_${x}_${south_y}_south_rts_i                  = south_flow_control_rts_i_${x};
    assign south_flow_control_cts_o_${x}                     = chip_${x}_${south_y}_south_cts_o;
    assign south_test_request_o_${x}                         = chip_${x}_${south_y}_south_test_request_o;
    assign chip_${x}_${south_y}_south_test_being_requested_i = south_test_being_requested_i_${x};
    %endfor

    // ---- West boundary (x=0) ----
    %for y in range(max_compute_chiplet_y):
    for (genvar r = 0; r < 3; r++) begin : gen_wb_d2d_${y}_r
        for (genvar c = 0; c < 20; c++) begin : gen_wb_d2d_${y}_c
            tran(chip_0_${y}_west_d2d[r][c], west_d2d_link_${y}[r][c]);
        end
    end
    assign west_flow_control_rts_o_${y}             = chip_0_${y}_west_rts_o;
    assign chip_0_${y}_west_cts_i                   = west_flow_control_cts_i_${y};
    assign chip_0_${y}_west_rts_i                   = west_flow_control_rts_i_${y};
    assign west_flow_control_cts_o_${y}             = chip_0_${y}_west_cts_o;
    assign west_test_request_o_${y}                 = chip_0_${y}_west_test_request_o;
    assign chip_0_${y}_west_test_being_requested_i  = west_test_being_requested_i_${y};
    %endfor

    // ---- East boundary (x=${max_compute_chiplet_x - 1}) ----
    <% east_x = max_compute_chiplet_x - 1 %>
    %for y in range(max_compute_chiplet_y):
    for (genvar r = 0; r < 3; r++) begin : gen_eb_d2d_${y}_r
        for (genvar c = 0; c < 20; c++) begin : gen_eb_d2d_${y}_c
            tran(chip_${east_x}_${y}_east_d2d[r][c], east_d2d_link_${y}[r][c]);
        end
    end
    assign east_flow_control_rts_o_${y}                    = chip_${east_x}_${y}_east_rts_o;
    assign chip_${east_x}_${y}_east_cts_i                  = east_flow_control_cts_i_${y};
    assign chip_${east_x}_${y}_east_rts_i                  = east_flow_control_rts_i_${y};
    assign east_flow_control_cts_o_${y}                    = chip_${east_x}_${y}_east_cts_o;
    assign east_test_request_o_${y}                        = chip_${east_x}_${y}_east_test_request_o;
    assign chip_${east_x}_${y}_east_test_being_requested_i = east_test_being_requested_i_${y};
    %endfor

end else begin : gen_interposer
    // ==========================================================
    // Mode 1: Interposer via hemaia_chiplet_interposer_2x2
    //
    // The interposer instantiates 4 hemaia_io_pad instances and
    // 4 hemaia_chiplet_interconnect_11x6 instances.
    // Each hemaia_io_pad wraps a hemaia chiplet with IO pad models,
    // so the interposer contains its own hemaia RTL.
    //
    // All pad-level ports are exposed from the interposer:
    //   - Internal D2D pads (between adjacent chiplets) routed through interconnect
    //   - Boundary D2D, corner (CRBL/CRBR/CRTL/CRTR), M pads -> module ports
    // IO signal ports of io_pad are driven via cds_thru from pad ports.
    //
    //
    // Hierarchy:
    //   i_io_wrapper.gen_interposer.i_interposer
    //     .chip_X_Y  (hemaia_io_pad containing hemaia instance)
    //     .conn_XX_YY (hemaia_chiplet_interconnect_11x6)
    // ==========================================================

    // ---- 2x2 array size check ----
    initial begin
        if (${max_compute_chiplet_x} != 2 || ${max_compute_chiplet_y} != 2) begin
            $fatal(1, "SIM_WITH_INTERPOSER=1 only supports 2x2 compute chiplet array. Got %0dx%0d.",
                   ${max_compute_chiplet_x}, ${max_compute_chiplet_y});
        end
    end
<%
## ============================================================
## Pad-to-signal mapping (from hemaia_io_pad.sv cds_thru)
## ============================================================
## Key: pad name -> (signal_name, bit_index_or_None)
pad_signal_map = {
    # Clocks and resets
    'CRTL_X1_Y3': ('clk_i', None),
    'CRTL_X1_Y1': ('rst_ni', None),
    'CRBL_X5_Y2': ('clk_obs_o', None),
    'CRTL_X2_Y4': ('clk_periph_i', None),
    'CRTL_X1_Y0': ('rst_periph_ni', None),
    # PLL
    'CRTL_X1_Y4': ('pll_bypass_i', None),
    'CRTL_X3_Y1': ('pll_en_i', None),
    'CRTL_X4_Y1': ('pll_post_div_sel_i_1', None),
    'CRTL_X4_Y2': ('pll_post_div_sel_i_0', None),
    'CRTL_X3_Y2': ('pll_lock_o', None),
    # Aux
    'CRTL_X1_Y2': ('test_mode_i', None),
    'CRBL_X3_Y4': ('boot_mode_i', None),
    # Chip ID [7:0]
    'CRBL_X2_Y1': ('chip_id_7', None),
    'CRBL_X3_Y1': ('chip_id_6', None),
    'CRBL_X4_Y1': ('chip_id_5', None),
    'CRBL_X5_Y1': ('chip_id_4', None),
    'CRBR_X0_Y1': ('chip_id_3', None),
    'CRBR_X1_Y1': ('chip_id_2', None),
    'CRBR_X2_Y1': ('chip_id_1', None),
    'CRBR_X3_Y1': ('chip_id_0', None),
    # East flow control
    'CRTR_X3_Y0': ('east_test_being_requested_i', None),
    'CRTR_X2_Y1': ('east_test_request_o', None),
    'CRBR_X2_Y4': ('flow_control_east_rts_o', None),
    'CRBR_X2_Y5': ('flow_control_east_cts_i', None),
    'CRBR_X2_Y3': ('flow_control_east_rts_i', None),
    'CRBR_X1_Y3': ('flow_control_east_cts_o', None),
    # West flow control
    'CRTL_X2_Y2': ('west_test_being_requested_i', None),
    'CRTL_X2_Y3': ('west_test_request_o', None),
    'CRBL_X2_Y2': ('flow_control_west_rts_o', None),
    'CRBL_X2_Y4': ('flow_control_west_cts_i', None),
    'CRBL_X3_Y3': ('flow_control_west_rts_i', None),
    'CRBL_X2_Y3': ('flow_control_west_cts_o', None),
    # North flow control
    'CRTL_X3_Y3': ('north_test_being_requested_i', None),
    'CRTL_X4_Y3': ('north_test_request_o', None),
    'CRTR_X3_Y2': ('flow_control_north_rts_o', None),
    'CRTR_X0_Y2': ('flow_control_north_cts_i', None),
    'CRTR_X2_Y2': ('flow_control_north_rts_i', None),
    'CRTR_X1_Y2': ('flow_control_north_cts_o', None),
    # South flow control
    'CRBL_X4_Y2': ('south_test_being_requested_i', None),
    'CRBL_X3_Y2': ('south_test_request_o', None),
    'CRBR_X0_Y2': ('flow_control_south_rts_o', None),
    'CRBR_X2_Y2': ('flow_control_south_cts_i', None),
    'CRBR_X1_Y2': ('flow_control_south_rts_i', None),
    'CRBR_X0_Y3': ('flow_control_south_cts_o', None),
    # UART
    'CRBL_X1_Y4': ('uart_tx_o', None),
    'CRBL_X1_Y3': ('uart_rx_i', None),
    'CRBL_X1_Y2': ('uart_rts_no', None),
    'CRBL_X1_Y1': ('uart_cts_ni', None),
    # GPIO
    'CRBR_X3_Y2': ('gpio_3', None),
    'CRBR_X3_Y3': ('gpio_2', None),
    'CRBR_X3_Y4': ('gpio_1', None),
    'CRBR_X3_Y5': ('gpio_0', None),
    # SPI Master
    'CRTL_X3_Y4': ('spim_sck_o', None),
    'CRTL_X4_Y4': ('spim_csb_o', None),
    'CRTR_X3_Y3': ('spim_sd_3', None),
    'CRTR_X2_Y3': ('spim_sd_2', None),
    'CRTR_X1_Y3': ('spim_sd_1', None),
    'CRTR_X0_Y3': ('spim_sd_0', None),
    # I2C
    'CRTL_X2_Y1': ('i2c_sda', None),
    'CRTL_X2_Y0': ('i2c_scl', None),
    # JTAG
    'CRTR_X4_Y3': ('jtag_trst_ni', None),
    'CRTR_X4_Y2': ('jtag_tck_i', None),
    'CRTR_X4_Y1': ('jtag_tms_i', None),
    'CRTR_X4_Y0': ('jtag_tdi_i', None),
    'CRTR_X3_Y1': ('jtag_tdo_o', None),
}

## Reverse map: signal name -> pad name
signal_to_pad = {v[0]: k for k, v in pad_signal_map.items()}

## 2x2 mesh connectivity
mesh_conn = {}
mesh_conn[(0,0,'east')]  = 'internal'
mesh_conn[(0,0,'west')]  = 'offchip'
mesh_conn[(0,0,'north')] = 'offchip'
mesh_conn[(0,0,'south')] = 'internal'
mesh_conn[(1,0,'east')]  = 'offchip'
mesh_conn[(1,0,'west')]  = 'internal'
mesh_conn[(1,0,'north')] = 'offchip'
mesh_conn[(1,0,'south')] = 'internal'
mesh_conn[(0,1,'east')]  = 'internal'
mesh_conn[(0,1,'west')]  = 'offchip'
mesh_conn[(0,1,'north')] = 'internal'
mesh_conn[(0,1,'south')] = 'offchip'
mesh_conn[(1,1,'east')]  = 'offchip'
mesh_conn[(1,1,'west')]  = 'internal'
mesh_conn[(1,1,'north')] = 'internal'
mesh_conn[(1,1,'south')] = 'offchip'

internal_links = {
    (0,0,'east'):  (1,0,'west'),
    (0,0,'south'): (0,1,'north'),
    (1,0,'west'):  (0,0,'east'),
    (1,0,'south'): (1,1,'north'),
    (0,1,'east'):  (1,1,'west'),
    (0,1,'north'): (0,0,'south'),
    (1,1,'west'):  (0,1,'east'),
    (1,1,'north'): (1,0,'south'),
}

## Flow control signal -> (direction_keyword, pad_signal_key)
## direction keyword is used to determine internal/offchip/tieoff routing
fc_signals = {
    'east': [
        ('flow_control_east_rts_o', 'rts_o'),
        ('flow_control_east_cts_i', 'cts_i'),
        ('flow_control_east_rts_i', 'rts_i'),
        ('flow_control_east_cts_o', 'cts_o'),
        ('east_test_request_o', 'req_o'),
        ('east_test_being_requested_i', 'req_i'),
    ],
    'west': [
        ('flow_control_west_rts_o', 'rts_o'),
        ('flow_control_west_cts_i', 'cts_i'),
        ('flow_control_west_rts_i', 'rts_i'),
        ('flow_control_west_cts_o', 'cts_o'),
        ('west_test_request_o', 'req_o'),
        ('west_test_being_requested_i', 'req_i'),
    ],
    'north': [
        ('flow_control_north_rts_o', 'rts_o'),
        ('flow_control_north_cts_i', 'cts_i'),
        ('flow_control_north_rts_i', 'rts_i'),
        ('flow_control_north_cts_o', 'cts_o'),
        ('north_test_request_o', 'req_o'),
        ('north_test_being_requested_i', 'req_i'),
    ],
    'south': [
        ('flow_control_south_rts_o', 'rts_o'),
        ('flow_control_south_cts_i', 'cts_i'),
        ('flow_control_south_rts_i', 'rts_i'),
        ('flow_control_south_cts_o', 'cts_o'),
        ('south_test_request_o', 'req_o'),
        ('south_test_being_requested_i', 'req_i'),
    ],
}

def conn_wire(cx, cy, d, signal):
    nx, ny, nd = internal_links[(cx,cy,d)]
    return "conn_%d_%d_%s_%s_to_%d_%d_%s" % (cx, cy, d, signal, nx, ny, nd)

def rev_conn_wire(cx, cy, d, signal):
    nx, ny, nd = internal_links[(cx,cy,d)]
    return "conn_%d_%d_%s_%s_to_%d_%d_%s" % (nx, ny, nd, signal, cx, cy, d)

def boundary_idx(cx, cy, d):
    """North/south boundaries are indexed by column (cx), east/west by row (cy)."""
    return cx if d in ('north', 'south') else cy

## Map flow control type to conn_ wire signal name
fc_wire_map = {
    'rts_o': ('rts', False),   # (signal, is_reverse)
    'cts_i': ('cts', True),
    'rts_i': ('rts', True),
    'cts_o': ('cts', False),
    'req_o': ('req', False),
    'req_i': ('req', True),
}

## Flow control pad signal name -> (direction, fc_type)
fc_pad_map = {}
for _d in ['east', 'west', 'north', 'south']:
    fc_pad_map['flow_control_%s_rts_o' % _d] = (_d, 'rts_o')
    fc_pad_map['flow_control_%s_cts_i' % _d] = (_d, 'cts_i')
    fc_pad_map['flow_control_%s_rts_i' % _d] = (_d, 'rts_i')
    fc_pad_map['flow_control_%s_cts_o' % _d] = (_d, 'cts_o')
    fc_pad_map['%s_test_request_o' % _d] = (_d, 'req_o')
    fc_pad_map['%s_test_being_requested_i' % _d] = (_d, 'req_i')

def get_pad_wire(sig_name, s, cx, cy):
    """Return io_wrapper wire expression for a pad signal, or None to leave unconnected."""
    # Shared clock/reset
    shared = {'clk_i': 'mst_clk_i', 'rst_ni': 'rst_ni',
              'clk_periph_i': 'periph_clk_i', 'rst_periph_ni': 'rst_periph_ni'}
    if sig_name in shared: return shared[sig_name]
    # Tied-off
    if sig_name in ('test_mode_i', 'boot_mode_i'): return 'const_zero'
    # Per-chip scalar
    scalar = {
        'clk_obs_o': '_clk_obs_o', 'pll_bypass_i': '_pll_bypass_i',
        'pll_en_i': '_pll_en_i', 'pll_lock_o': '_pll_lock_o',
        'uart_tx_o': '_uart_tx_o', 'uart_rx_i': '_uart_rx_i',
        'uart_rts_no': '_uart_rts_no', 'uart_cts_ni': '_uart_cts_ni',
        'i2c_sda': '_i2c_sda', 'i2c_scl': '_i2c_scl',
        'jtag_trst_ni': '_jtag_trst_ni', 'jtag_tck_i': '_jtag_tck_i',
        'jtag_tms_i': '_jtag_tms_i', 'jtag_tdi_i': '_jtag_tdi_i',
        'jtag_tdo_o': '_jtag_tdo_o', 'spim_sck_o': '_spim_sck_o',
        'spim_csb_o': '_spim_csb_o',
    }
    if sig_name in scalar: return s + scalar[sig_name]
    # Per-chip bit-indexed
    if sig_name == 'pll_post_div_sel_i_1': return s + '_pll_post_div_sel_i[1]'
    if sig_name == 'pll_post_div_sel_i_0': return s + '_pll_post_div_sel_i[0]'
    if sig_name.startswith('chip_id_'): return s + '_id_i[' + sig_name[-1] + ']'
    # GPIO individual bit pads (gpio_0 .. gpio_3)
    if sig_name.startswith('gpio_'):
        return s + '_gpio[' + sig_name.split('_')[-1] + ']'
    # SPI SD individual bit pads (spim_sd_0 .. spim_sd_3)
    if sig_name.startswith('spim_sd_'):
        return s + '_spim_sd[' + sig_name.split('_')[-1] + ']'
    # Flow control pads — direction-dependent routing
    if sig_name in fc_pad_map:
        _d, fc_type = fc_pad_map[sig_name]
        ct = mesh_conn[(cx, cy, _d)]
        if ct == 'internal':
            wire_sig, is_rev = fc_wire_map[fc_type]
            return rev_conn_wire(cx, cy, _d, wire_sig) if is_rev else conn_wire(cx, cy, _d, wire_sig)
        elif ct == 'offchip':
            bi = boundary_idx(cx, cy, _d)
            offchip = {
                'rts_o': '%s_flow_control_rts_o_%d' % (_d, bi),
                'cts_i': '%s_flow_control_cts_i_%d' % (_d, bi),
                'rts_i': '%s_flow_control_rts_i_%d' % (_d, bi),
                'cts_o': '%s_flow_control_cts_o_%d' % (_d, bi),
                'req_o': '%s_test_request_o_%d' % (_d, bi),
                'req_i': '%s_test_being_requested_i_%d' % (_d, bi),
            }
            return offchip.get(fc_type)
        else:  # tieoff
            return 'const_zero' if fc_type in ('cts_i', 'rts_i', 'req_i') else None
    return None
%>\

    // ---- Inter-chiplet flow control routing wires ----
    // These connect corner pads carrying flow control between adjacent chiplets
    // (0,0) east <-> (1,0) west
    wire conn_0_0_east_rts_to_1_0_west;
    wire conn_0_0_east_cts_to_1_0_west;
    wire conn_0_0_east_req_to_1_0_west;
    wire conn_1_0_west_rts_to_0_0_east;
    wire conn_1_0_west_cts_to_0_0_east;
    wire conn_1_0_west_req_to_0_0_east;
    // (0,0) south <-> (0,1) north
    wire conn_0_0_south_rts_to_0_1_north;
    wire conn_0_0_south_cts_to_0_1_north;
    wire conn_0_0_south_req_to_0_1_north;
    wire conn_0_1_north_rts_to_0_0_south;
    wire conn_0_1_north_cts_to_0_0_south;
    wire conn_0_1_north_req_to_0_0_south;
    // (1,0) south <-> (1,1) north
    wire conn_1_0_south_rts_to_1_1_north;
    wire conn_1_0_south_cts_to_1_1_north;
    wire conn_1_0_south_req_to_1_1_north;
    wire conn_1_1_north_rts_to_1_0_south;
    wire conn_1_1_north_cts_to_1_0_south;
    wire conn_1_1_north_req_to_1_0_south;
    // (0,1) east <-> (1,1) west
    wire conn_0_1_east_rts_to_1_1_west;
    wire conn_0_1_east_cts_to_1_1_west;
    wire conn_0_1_east_req_to_1_1_west;
    wire conn_1_1_west_rts_to_0_1_east;
    wire conn_1_1_west_cts_to_0_1_east;
    wire conn_1_1_west_req_to_0_1_east;

    // ---- Interposer instantiation ----
    // Drives both IO signal ports (hemaia RTL wires) and physical pad
    // ports (CRTL/CRBL/CRTR/CRBR corner pads) of the interposer.
    // D2D data pads and M pads are left unconnected (routed internally).
    hemaia_chiplet_interposer_2x2 i_interposer (
        %for chip_xy in [(0,0), (0,1), (1,0), (1,1)]:
        <%
            cx, cy = chip_xy
            c = "%d%d" % (cx, cy)
            s = "chip%d%d" % (cx, cy)
        %>
        // ================================================
        // Chip (${cx}, ${cy}) — IO signal ports
        // ================================================
        // Driving signals
        .chip${c}_io_clk_i          (mst_clk_i),
        .chip${c}_io_rst_ni         (rst_ni),
        .chip${c}_io_clk_periph_i   (periph_clk_i),
        .chip${c}_io_rst_periph_ni  (rst_periph_ni),
        .chip${c}_io_clk_obs_o      (${s}_clk_obs_o),
        .chip${c}_io_pll_bypass_i   (${s}_pll_bypass_i),
        .chip${c}_io_pll_en_i       (${s}_pll_en_i),
        .chip${c}_io_pll_post_div_sel_i(${s}_pll_post_div_sel_i),
        .chip${c}_io_pll_lock_o     (${s}_pll_lock_o),
        .chip${c}_io_test_mode_i    (const_zero),
        .chip${c}_io_chip_id_i      (${s}_id_i),
        .chip${c}_io_boot_mode_i    (const_zero),
        // Peripherals
        .chip${c}_io_uart_tx_o      (${s}_uart_tx_o),
        .chip${c}_io_uart_rx_i      (${s}_uart_rx_i),
        .chip${c}_io_uart_rts_no    (${s}_uart_rts_no),
        .chip${c}_io_uart_cts_ni    (${s}_uart_cts_ni),
        .chip${c}_io_gpio           (${s}_gpio),
        .chip${c}_io_spim_sck_o     (${s}_spim_sck_o),
        .chip${c}_io_spim_csb_o     (${s}_spim_csb_o),
        .chip${c}_io_spim_sd        (${s}_spim_sd),
        .chip${c}_io_i2c_sda        (${s}_i2c_sda),
        .chip${c}_io_i2c_scl        (${s}_i2c_scl),
        .chip${c}_io_jtag_trst_ni   (${s}_jtag_trst_ni),
        .chip${c}_io_jtag_tck_i     (${s}_jtag_tck_i),
        .chip${c}_io_jtag_tms_i     (${s}_jtag_tms_i),
        .chip${c}_io_jtag_tdi_i     (${s}_jtag_tdi_i),
        .chip${c}_io_jtag_tdo_o     (${s}_jtag_tdo_o),
        // D2D data buses
        .chip${c}_io_east_d2d       (chip_${cx}_${cy}_east_d2d),
        .chip${c}_io_west_d2d       (chip_${cx}_${cy}_west_d2d),
        .chip${c}_io_north_d2d      (chip_${cx}_${cy}_north_d2d),
        .chip${c}_io_south_d2d      (chip_${cx}_${cy}_south_d2d),
        // Flow control and test per direction
        %for d in ['east', 'west', 'north', 'south']:
        <%
            ct = mesh_conn[(cx, cy, d)]
        %>
        %if ct == 'internal':
        // ${d}: internal link to (${internal_links[(cx,cy,d)][0]}, ${internal_links[(cx,cy,d)][1]})
        .chip${c}_io_flow_control_${d}_rts_o     (${conn_wire(cx,cy,d,'rts')}),
        .chip${c}_io_flow_control_${d}_cts_i     (${rev_conn_wire(cx,cy,d,'cts')}),
        .chip${c}_io_flow_control_${d}_rts_i     (${rev_conn_wire(cx,cy,d,'rts')}),
        .chip${c}_io_flow_control_${d}_cts_o     (${conn_wire(cx,cy,d,'cts')}),
        .chip${c}_io_${d}_test_request_o         (${conn_wire(cx,cy,d,'req')}),
        .chip${c}_io_${d}_test_being_requested_i (${rev_conn_wire(cx,cy,d,'req')}),
        %elif ct == 'offchip':
        <%
            bi = boundary_idx(cx, cy, d)
        %>
        // ${d}: offchip ${d} boundary [${bi}]
        .chip${c}_io_flow_control_${d}_rts_o     (${d}_flow_control_rts_o_${bi}),
        .chip${c}_io_flow_control_${d}_cts_i     (${d}_flow_control_cts_i_${bi}),
        .chip${c}_io_flow_control_${d}_rts_i     (${d}_flow_control_rts_i_${bi}),
        .chip${c}_io_flow_control_${d}_cts_o     (${d}_flow_control_cts_o_${bi}),
        .chip${c}_io_${d}_test_request_o         (${d}_test_request_o_${bi}),
        .chip${c}_io_${d}_test_being_requested_i (${d}_test_being_requested_i_${bi}),
        %else:
        // ${d}: tied off (no neighbor)
        .chip${c}_io_flow_control_${d}_rts_o     (),
        .chip${c}_io_flow_control_${d}_cts_i     (const_zero),
        .chip${c}_io_flow_control_${d}_rts_i     (const_zero),
        .chip${c}_io_flow_control_${d}_cts_o     (),
        .chip${c}_io_${d}_test_request_o         (),
        .chip${c}_io_${d}_test_being_requested_i (const_zero),
        %endif
        %endfor
        // ================================================
        // Chip (${cx}, ${cy}) — Pad ports (directly drive physical pads)
        // ================================================
        // Connect corner pads (CRTL/CRBL/CRTR/CRBR) that have a known
        // signal mapping to the same io_wrapper wires used by the IO
        // signal ports above.  This allows selecting the simulation
        // path: drive IO signal ports OR drive pad ports.
        <%
            pad_ports = [(pn, pw) for pn, (sn, _) in pad_signal_map.items()
                         for pw in [get_pad_wire(sn, s, cx, cy)] if pw is not None]
            is_last_chip = (chip_xy == [(0,0), (0,1), (1,0), (1,1)][-1])
        %>\
        %for idx, (pad_name, pw) in enumerate(pad_ports):
        %if is_last_chip and idx == len(pad_ports) - 1:
        .chip${c}_${pad_name}    (${pw})
        %else:
        .chip${c}_${pad_name}    (${pw}),
        %endif
        %endfor
        %endfor
    );

end
endgenerate

endmodule
