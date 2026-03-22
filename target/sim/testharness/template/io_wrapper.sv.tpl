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
//   using shared tri wires. Flow control and test signals use assign.
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
    inout tri [2:0][19:0] chip_${cx}_${cy}_${direction}_d2d,
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
    inout tri [2:0][19:0] north_d2d_link_${x},
    inout wire             north_flow_control_rts_o_${x},
    inout wire             north_flow_control_cts_i_${x},
    inout wire             north_flow_control_rts_i_${x},
    inout wire             north_flow_control_cts_o_${x},
    inout wire             north_test_request_o_${x},
    inout wire             north_test_being_requested_i_${x},
    %endfor
    %for x in range(max_compute_chiplet_x):
    inout tri [2:0][19:0] south_d2d_link_${x},
    inout wire             south_flow_control_rts_o_${x},
    inout wire             south_flow_control_cts_i_${x},
    inout wire             south_flow_control_rts_i_${x},
    inout wire             south_flow_control_cts_o_${x},
    inout wire             south_test_request_o_${x},
    inout wire             south_test_being_requested_i_${x},
    %endfor
    %for y in range(max_compute_chiplet_y):
    inout tri [2:0][19:0] west_d2d_link_${y},
    inout wire             west_flow_control_rts_o_${y},
    inout wire             west_flow_control_cts_i_${y},
    inout wire             west_flow_control_rts_i_${y},
    inout wire             west_flow_control_cts_o_${y},
    inout wire             west_test_request_o_${y},
    inout wire             west_test_being_requested_i_${y},
    %endfor
    %for y in range(max_compute_chiplet_y):
    inout tri [2:0][19:0] east_d2d_link_${y},
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
    tri [2:0][19:0] ew_d2d_link_${x}_${y};
    alias ew_d2d_link_${x}_${y} = chip_${x}_${y}_east_d2d = chip_${x + 1}_${y}_west_d2d;
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
    tri [2:0][19:0] ns_d2d_link_${x}_${y};
    alias ns_d2d_link_${x}_${y} = chip_${x}_${y}_south_d2d = chip_${x}_${y + 1}_north_d2d;
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
    tri [2:0][19:0] nb_d2d_link_${x};
    alias nb_d2d_link_${x} = chip_${x}_0_north_d2d = north_d2d_link_${x};
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
    tri [2:0][19:0] sb_d2d_link_${x};
    alias sb_d2d_link_${x} = chip_${x}_${south_y}_south_d2d = south_d2d_link_${x};
    assign south_flow_control_rts_o_${x}                     = chip_${x}_${south_y}_south_rts_o;
    assign chip_${x}_${south_y}_south_cts_i                  = south_flow_control_cts_i_${x};
    assign chip_${x}_${south_y}_south_rts_i                  = south_flow_control_rts_i_${x};
    assign south_flow_control_cts_o_${x}                     = chip_${x}_${south_y}_south_cts_o;
    assign south_test_request_o_${x}                         = chip_${x}_${south_y}_south_test_request_o;
    assign chip_${x}_${south_y}_south_test_being_requested_i = south_test_being_requested_i_${x};
    %endfor

    // ---- West boundary (x=0) ----
    %for y in range(max_compute_chiplet_y):
    tri [2:0][19:0] wb_d2d_link_${y};
    alias wb_d2d_link_${y} = chip_0_${y}_west_d2d = west_d2d_link_${y};
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
    tri [2:0][19:0] eb_d2d_link_${y};
    alias eb_d2d_link_${y} = chip_${east_x}_${y}_east_d2d = east_d2d_link_${y};
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

## ============================================================
## Pad geometry definitions for hemaia_chiplet_interposer_2x2
## Used to enumerate ALL pad ports and connect remaining ones
## with () to avoid "missing connection" warnings.
## ============================================================
## Corner pads (same geometry for all chips)
corner_pad_geom = {
    'CRBL': [(x, y) for x in range(6) for y in range(5)],   # 30
    'CRBR': [(x, y) for x in range(5) for y in range(6)],   # 30
    'CRTL': [(x, y) for x in range(5) for y in range(6)],   # 30
    'CRTR': [(x, y) for x in range(6) for y in range(5)],   # 30
}
## M pads: border of 10x10 grid (same for all chips, 36 total)
m_pad_coords = (
    [(0, y) for y in range(10)] +
    [(9, y) for y in range(10)] +
    [(x, 0) for x in range(1, 9)] +
    [(x, 9) for x in range(1, 9)]
)
## D2D pads: boundary directions get full grid, internal get single corner pad
d2d_full_grid = {
    'N': [(x, y) for x in range(11) for y in range(6)],   # 66
    'S': [(x, y) for x in range(11) for y in range(6)],   # 66
    'E': [(x, y) for x in range(6) for y in range(11)],   # 66
    'W': [(x, y) for x in range(6) for y in range(11)],   # 66
}
d2d_corner_pad = {
    'E': [(5, 10)],
    'W': [(0, 0)],
    'N': [(0, 5)],
    'S': [(10, 0)],
}
dir_short = {'east': 'E', 'west': 'W', 'north': 'N', 'south': 'S'}

def get_all_pad_names(cx, cy):
    """Return set of all pad port names (without chip prefix) for chip at (cx, cy)."""
    pads = set()
    for corner, coords in corner_pad_geom.items():
        for x, y in coords:
            pads.add('%s_X%d_Y%d' % (corner, x, y))
    for x, y in m_pad_coords:
        pads.add('M_X%d_Y%d' % (x, y))
    for d in ['east', 'west', 'north', 'south']:
        ds = dir_short[d]
        ct = mesh_conn[(cx, cy, d)]
        if ct == 'offchip':
            coords = d2d_full_grid[ds]
        else:
            coords = d2d_corner_pad[ds]
        for x, y in coords:
            pads.add('D2D%s_X%d_Y%d' % (ds, x, y))
    return pads

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

## ============================================================
## Pin-map classification for remaining (non-signal-mapped) pads
## Derived from pin_map.csv — used to categorize pad connections
## in the generated SV with clear comments.
## ============================================================

## Power supply pads: pad_name -> supply type
power_pads = {
    # CRBL corner
    'CRBL_X0_Y0': 'VDD_IO', 'CRBL_X0_Y4': 'VDD', 'CRBL_X0_Y1': 'VSS',
    'CRBL_X0_Y3': 'VDD_D2D', 'CRBL_X3_Y0': 'VDD_IO', 'CRBL_X2_Y0': 'VSS',
    'CRBL_X1_Y0': 'VDD', 'CRBL_X5_Y0': 'VDD', 'CRBL_X4_Y0': 'VDD_D2D',
    'CRBL_X0_Y2': 'VDD_IO',
    # CRBR corner
    'CRBR_X4_Y0': 'VDD_IO', 'CRBR_X0_Y0': 'VDD', 'CRBR_X3_Y0': 'VSS',
    'CRBR_X1_Y0': 'VDD_D2D', 'CRBR_X4_Y3': 'VDD_IO', 'CRBR_X4_Y2': 'VSS',
    'CRBR_X4_Y1': 'VDD', 'CRBR_X4_Y5': 'VDD', 'CRBR_X4_Y4': 'VDD_D2D',
    'CRBR_X2_Y0': 'VDD_IO',
    # CRTL corner
    'CRTL_X4_Y5': 'VDD', 'CRTL_X3_Y5': 'VDD_D2D', 'CRTL_X2_Y5': 'VDD_IO',
    'CRTL_X1_Y5': 'VSS', 'CRTL_X0_Y5': 'VDD_IO', 'CRTL_X0_Y4': 'VDD',
    'CRTL_X0_Y3': 'VSS', 'CRTL_X0_Y2': 'VDD_IO', 'CRTL_X0_Y1': 'VDD_D2D',
    'CRTL_X0_Y0': 'VDD',
    # CRTR corner
    'CRTR_X5_Y4': 'VDD_IO', 'CRTR_X4_Y4': 'VDD', 'CRTR_X3_Y4': 'VSS',
    'CRTR_X2_Y4': 'VDD_IO', 'CRTR_X1_Y4': 'VDD_D2D', 'CRTR_X0_Y4': 'VDD',
    'CRTR_X5_Y3': 'VSS', 'CRTR_X5_Y2': 'VDD_IO', 'CRTR_X5_Y1': 'VDD_D2D',
    'CRTR_X5_Y0': 'VDD',
    # D2D power pads (East)
    'D2DE_X5_Y6': 'VSS', 'D2DE_X5_Y7': 'VDD_D2D', 'D2DE_X5_Y10': 'VSS',
    'D2DE_X5_Y2': 'VSS', 'D2DE_X5_Y3': 'VDD_D2D', 'D2DE_X2_Y5': 'VDD_D2D',
    # D2D power pads (North)
    'D2DN_X0_Y5': 'VSS', 'D2DN_X8_Y5': 'VDD_D2D', 'D2DN_X7_Y5': 'VSS',
    'D2DN_X4_Y5': 'VDD_D2D', 'D2DN_X3_Y5': 'VSS', 'D2DN_X5_Y2': 'VDD_D2D',
    # D2D power pads (South)
    'D2DS_X7_Y0': 'VDD_D2D', 'D2DS_X6_Y0': 'VSS', 'D2DS_X3_Y0': 'VDD_D2D',
    'D2DS_X2_Y0': 'VSS', 'D2DS_X10_Y0': 'VSS', 'D2DS_X5_Y3': 'VDD_D2D',
    # D2D power pads (West)
    'D2DW_X0_Y8': 'VDD_D2D', 'D2DW_X0_Y7': 'VSS', 'D2DW_X0_Y4': 'VDD_D2D',
    'D2DW_X0_Y3': 'VSS', 'D2DW_X0_Y0': 'VSS', 'D2DW_X3_Y5': 'VDD_D2D',
}

## NC (not connected) corner pads
nc_pads = {
    'CRBL_X4_Y4', 'CRBL_X3_Y4', 'CRBL_X4_Y3', 'CRBL_X5_Y4', 'CRBL_X5_Y3',
    'CRBR_X1_Y4', 'CRBR_X1_Y5', 'CRBR_X0_Y5', 'CRBR_X0_Y4',
    'CRTR_X0_Y1', 'CRTR_X1_Y1', 'CRTR_X0_Y0', 'CRTR_X1_Y0', 'CRTR_X2_Y0',
}

## PLL reserved pads
pll_rsv_pads = {'CRTL_X3_Y0': 'pll_rsv_2', 'CRTL_X4_Y0': 'pll_rsv_5'}

## D2D data pads: pad_name -> (direction_char, signal_index)
## Signal index maps to bus bits as: d2d_link[idx//20][idx%20]
d2d_data_pads = {
    # D2D East
    'D2DE_X5_Y0': ('E', 0), 'D2DE_X5_Y1': ('E', 1), 'D2DE_X5_Y8': ('E', 40),
    'D2DE_X5_Y9': ('E', 41), 'D2DE_X5_Y4': ('E', 20), 'D2DE_X5_Y5': ('E', 21),
    'D2DE_X4_Y8': ('E', 39), 'D2DE_X4_Y7': ('E', 38), 'D2DE_X4_Y6': ('E', 34),
    'D2DE_X3_Y6': ('E', 35), 'D2DE_X2_Y6': ('E', 31), 'D2DE_X1_Y6': ('E', 27),
    'D2DE_X0_Y6': ('E', 23), 'D2DE_X4_Y5': ('E', 30), 'D2DE_X3_Y5': ('E', 29),
    'D2DE_X1_Y5': ('E', 26), 'D2DE_X4_Y4': ('E', 33), 'D2DE_X3_Y4': ('E', 32),
    'D2DE_X2_Y4': ('E', 28), 'D2DE_X1_Y4': ('E', 25), 'D2DE_X4_Y3': ('E', 37),
    'D2DE_X4_Y2': ('E', 36), 'D2DE_X4_Y1': ('E', 3), 'D2DE_X0_Y4': ('E', 24),
    'D2DE_X0_Y5': ('E', 22), 'D2DE_X4_Y0': ('E', 2), 'D2DE_X0_Y0': ('E', 4),
    'D2DE_X0_Y1': ('E', 5), 'D2DE_X0_Y2': ('E', 6), 'D2DE_X0_Y3': ('E', 7),
    'D2DE_X1_Y0': ('E', 8), 'D2DE_X1_Y1': ('E', 9), 'D2DE_X1_Y2': ('E', 10),
    'D2DE_X1_Y3': ('E', 11), 'D2DE_X2_Y0': ('E', 12), 'D2DE_X2_Y1': ('E', 13),
    'D2DE_X2_Y2': ('E', 14), 'D2DE_X2_Y3': ('E', 15), 'D2DE_X3_Y0': ('E', 16),
    'D2DE_X3_Y1': ('E', 17), 'D2DE_X3_Y2': ('E', 18), 'D2DE_X3_Y3': ('E', 19),
    'D2DE_X0_Y7': ('E', 44), 'D2DE_X0_Y8': ('E', 45), 'D2DE_X0_Y9': ('E', 46),
    'D2DE_X0_Y10': ('E', 47), 'D2DE_X1_Y7': ('E', 48), 'D2DE_X1_Y8': ('E', 49),
    'D2DE_X1_Y9': ('E', 50), 'D2DE_X1_Y10': ('E', 51), 'D2DE_X2_Y7': ('E', 52),
    'D2DE_X2_Y8': ('E', 53), 'D2DE_X2_Y9': ('E', 54), 'D2DE_X2_Y10': ('E', 55),
    'D2DE_X3_Y7': ('E', 56), 'D2DE_X3_Y8': ('E', 57), 'D2DE_X3_Y9': ('E', 58),
    'D2DE_X3_Y10': ('E', 59), 'D2DE_X4_Y10': ('E', 43), 'D2DE_X4_Y9': ('E', 42),
    # D2D North
    'D2DN_X1_Y5': ('N', 0), 'D2DN_X2_Y5': ('N', 1), 'D2DN_X0_Y0': ('N', 2),
    'D2DN_X1_Y0': ('N', 3), 'D2DN_X0_Y1': ('N', 4), 'D2DN_X0_Y2': ('N', 5),
    'D2DN_X0_Y3': ('N', 6), 'D2DN_X0_Y4': ('N', 7), 'D2DN_X1_Y1': ('N', 8),
    'D2DN_X1_Y2': ('N', 9), 'D2DN_X1_Y3': ('N', 10), 'D2DN_X1_Y4': ('N', 11),
    'D2DN_X2_Y1': ('N', 12), 'D2DN_X2_Y2': ('N', 13), 'D2DN_X2_Y3': ('N', 14),
    'D2DN_X2_Y4': ('N', 15), 'D2DN_X3_Y1': ('N', 16), 'D2DN_X3_Y2': ('N', 17),
    'D2DN_X3_Y3': ('N', 18), 'D2DN_X3_Y4': ('N', 19), 'D2DN_X5_Y5': ('N', 20),
    'D2DN_X6_Y5': ('N', 21), 'D2DN_X4_Y3': ('N', 22), 'D2DN_X4_Y4': ('N', 23),
    'D2DN_X2_Y0': ('N', 24), 'D2DN_X4_Y0': ('N', 25), 'D2DN_X4_Y1': ('N', 26),
    'D2DN_X4_Y2': ('N', 27), 'D2DN_X3_Y0': ('N', 28), 'D2DN_X5_Y0': ('N', 29),
    'D2DN_X5_Y3': ('N', 30), 'D2DN_X5_Y4': ('N', 31), 'D2DN_X7_Y0': ('N', 32),
    'D2DN_X6_Y1': ('N', 33), 'D2DN_X5_Y1': ('N', 34), 'D2DN_X6_Y3': ('N', 35),
    'D2DN_X8_Y0': ('N', 36), 'D2DN_X6_Y0': ('N', 37), 'D2DN_X6_Y2': ('N', 38),
    'D2DN_X6_Y4': ('N', 39), 'D2DN_X9_Y5': ('N', 40), 'D2DN_X10_Y5': ('N', 41),
    'D2DN_X9_Y0': ('N', 42), 'D2DN_X10_Y0': ('N', 43), 'D2DN_X7_Y1': ('N', 44),
    'D2DN_X7_Y2': ('N', 45), 'D2DN_X7_Y3': ('N', 46), 'D2DN_X7_Y4': ('N', 47),
    'D2DN_X8_Y1': ('N', 48), 'D2DN_X8_Y2': ('N', 49), 'D2DN_X8_Y3': ('N', 50),
    'D2DN_X8_Y4': ('N', 51), 'D2DN_X9_Y1': ('N', 52), 'D2DN_X9_Y2': ('N', 53),
    'D2DN_X9_Y3': ('N', 54), 'D2DN_X9_Y4': ('N', 55), 'D2DN_X10_Y1': ('N', 56),
    'D2DN_X10_Y2': ('N', 57), 'D2DN_X10_Y3': ('N', 58), 'D2DN_X10_Y4': ('N', 59),
    # D2D South
    'D2DS_X0_Y0': ('S', 0), 'D2DS_X1_Y0': ('S', 1), 'D2DS_X0_Y1': ('S', 2),
    'D2DS_X1_Y1': ('S', 3), 'D2DS_X0_Y2': ('S', 4), 'D2DS_X0_Y3': ('S', 5),
    'D2DS_X0_Y4': ('S', 6), 'D2DS_X0_Y5': ('S', 7), 'D2DS_X1_Y2': ('S', 8),
    'D2DS_X1_Y3': ('S', 9), 'D2DS_X1_Y4': ('S', 10), 'D2DS_X1_Y5': ('S', 11),
    'D2DS_X2_Y2': ('S', 12), 'D2DS_X2_Y3': ('S', 13), 'D2DS_X2_Y4': ('S', 14),
    'D2DS_X2_Y5': ('S', 15), 'D2DS_X3_Y2': ('S', 16), 'D2DS_X3_Y3': ('S', 17),
    'D2DS_X3_Y4': ('S', 18), 'D2DS_X3_Y5': ('S', 19), 'D2DS_X4_Y0': ('S', 20),
    'D2DS_X5_Y0': ('S', 21), 'D2DS_X4_Y4': ('S', 22), 'D2DS_X4_Y5': ('S', 23),
    'D2DS_X2_Y1': ('S', 24), 'D2DS_X4_Y1': ('S', 25), 'D2DS_X4_Y2': ('S', 26),
    'D2DS_X4_Y3': ('S', 27), 'D2DS_X3_Y1': ('S', 28), 'D2DS_X5_Y1': ('S', 29),
    'D2DS_X5_Y4': ('S', 30), 'D2DS_X5_Y5': ('S', 31), 'D2DS_X7_Y1': ('S', 32),
    'D2DS_X6_Y2': ('S', 33), 'D2DS_X5_Y2': ('S', 34), 'D2DS_X6_Y4': ('S', 35),
    'D2DS_X8_Y1': ('S', 36), 'D2DS_X6_Y1': ('S', 37), 'D2DS_X6_Y3': ('S', 38),
    'D2DS_X6_Y5': ('S', 39), 'D2DS_X8_Y0': ('S', 40), 'D2DS_X9_Y0': ('S', 41),
    'D2DS_X9_Y1': ('S', 42), 'D2DS_X10_Y1': ('S', 43), 'D2DS_X7_Y2': ('S', 44),
    'D2DS_X7_Y3': ('S', 45), 'D2DS_X7_Y4': ('S', 46), 'D2DS_X7_Y5': ('S', 47),
    'D2DS_X8_Y2': ('S', 48), 'D2DS_X8_Y3': ('S', 49), 'D2DS_X8_Y4': ('S', 50),
    'D2DS_X8_Y5': ('S', 51), 'D2DS_X9_Y2': ('S', 52), 'D2DS_X9_Y3': ('S', 53),
    'D2DS_X9_Y4': ('S', 54), 'D2DS_X9_Y5': ('S', 55), 'D2DS_X10_Y2': ('S', 56),
    'D2DS_X10_Y3': ('S', 57), 'D2DS_X10_Y4': ('S', 58), 'D2DS_X10_Y5': ('S', 59),
    # D2D West
    'D2DW_X0_Y1': ('W', 0), 'D2DW_X0_Y2': ('W', 1), 'D2DW_X0_Y5': ('W', 20),
    'D2DW_X0_Y6': ('W', 21), 'D2DW_X0_Y9': ('W', 40), 'D2DW_X0_Y10': ('W', 41),
    'D2DW_X5_Y0': ('W', 2), 'D2DW_X5_Y1': ('W', 3), 'D2DW_X1_Y0': ('W', 4),
    'D2DW_X1_Y1': ('W', 5), 'D2DW_X1_Y2': ('W', 6), 'D2DW_X1_Y3': ('W', 7),
    'D2DW_X2_Y0': ('W', 8), 'D2DW_X2_Y1': ('W', 9), 'D2DW_X2_Y2': ('W', 10),
    'D2DW_X2_Y3': ('W', 11), 'D2DW_X3_Y0': ('W', 12), 'D2DW_X3_Y1': ('W', 13),
    'D2DW_X3_Y2': ('W', 14), 'D2DW_X3_Y3': ('W', 15), 'D2DW_X4_Y0': ('W', 16),
    'D2DW_X4_Y1': ('W', 17), 'D2DW_X4_Y2': ('W', 18), 'D2DW_X4_Y3': ('W', 19),
    'D2DW_X1_Y5': ('W', 22), 'D2DW_X1_Y6': ('W', 23), 'D2DW_X1_Y4': ('W', 24),
    'D2DW_X2_Y4': ('W', 25), 'D2DW_X2_Y5': ('W', 26), 'D2DW_X2_Y6': ('W', 27),
    'D2DW_X3_Y4': ('W', 28), 'D2DW_X4_Y5': ('W', 29), 'D2DW_X5_Y5': ('W', 30),
    'D2DW_X3_Y6': ('W', 31), 'D2DW_X4_Y4': ('W', 32), 'D2DW_X5_Y4': ('W', 33),
    'D2DW_X5_Y6': ('W', 34), 'D2DW_X4_Y6': ('W', 35), 'D2DW_X5_Y2': ('W', 36),
    'D2DW_X5_Y3': ('W', 37), 'D2DW_X5_Y7': ('W', 38), 'D2DW_X5_Y8': ('W', 39),
    'D2DW_X5_Y9': ('W', 42), 'D2DW_X5_Y10': ('W', 43), 'D2DW_X1_Y7': ('W', 44),
    'D2DW_X1_Y8': ('W', 45), 'D2DW_X1_Y9': ('W', 46), 'D2DW_X1_Y10': ('W', 47),
    'D2DW_X2_Y7': ('W', 48), 'D2DW_X2_Y8': ('W', 49), 'D2DW_X2_Y9': ('W', 50),
    'D2DW_X2_Y10': ('W', 51), 'D2DW_X3_Y7': ('W', 52), 'D2DW_X3_Y8': ('W', 53),
    'D2DW_X3_Y9': ('W', 54), 'D2DW_X3_Y10': ('W', 55), 'D2DW_X4_Y7': ('W', 56),
    'D2DW_X4_Y8': ('W', 57), 'D2DW_X4_Y9': ('W', 58), 'D2DW_X4_Y10': ('W', 59),
}

## Direction char to long name mapping
d2d_dir_to_long = {'E': 'east', 'W': 'west', 'N': 'north', 'S': 'south'}

def classify_remaining_pad(pad_name, cx, cy):
    """Classify a remaining (non-signal-mapped) pad.
    Returns (category, connection_wire_or_None, comment_string).
    Categories: 'nc', 'power', 'd2d_boundary', 'd2d_internal', 'd2d_power', 'pll_rsv'
    """
    # M pads — all NC
    if pad_name.startswith('M_'):
        return ('nc', None, 'NC (M pad)')
    # Power supply pads
    if pad_name in power_pads:
        return ('power', None, power_pads[pad_name])
    # NC corner pads
    if pad_name in nc_pads:
        return ('nc', None, 'NC')
    # PLL reserved pads
    if pad_name in pll_rsv_pads:
        return ('pll_rsv', None, pll_rsv_pads[pad_name])
    # D2D data pads
    if pad_name in d2d_data_pads:
        d_short, idx = d2d_data_pads[pad_name]
        d_long = d2d_dir_to_long[d_short]
        ct = mesh_conn[(cx, cy, d_long)]
        lane = idx // 20
        bit = idx % 20
        if ct == 'offchip':
            bi = boundary_idx(cx, cy, d_long)
            wire = '%s_d2d_link_%d[%d][%d]' % (d_long, bi, lane, bit)
            return ('d2d_boundary', wire,
                    'D2D%s_%d -> %s_d2d_link_%d[%d][%d]' % (d_short, idx, d_long, bi, lane, bit))
        else:
            return ('d2d_internal', None,
                    'D2D%s_%d (internal — routed by interposer)' % (d_short, idx))
    return ('nc', None, 'unclassified')
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
    // ports (CRTL/CRBL/CRTR/CRBR corner pads, D2D pads, M pads).
    // Signal-mapped corner pads connect to io_wrapper wires.
    // Remaining pads (power/ground, D2D physical, M) left open ().
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
        // Signal-mapped corner pads connect to io_wrapper wires.
        <%
            pad_ports = [(pn, pw) for pn, (sn, _) in pad_signal_map.items()
                         for pw in [get_pad_wire(sn, s, cx, cy)] if pw is not None]
            connected_pads = set(pn for pn, _ in pad_ports)
            all_pads = get_all_pad_names(cx, cy)
            remaining_pads = sorted(all_pads - connected_pads)
            is_last_chip = (chip_xy == [(0,0), (0,1), (1,0), (1,1)][-1])
        %>\
        %for pad_name, pw in pad_ports:
        .chip${c}_${pad_name}    (${pw}),
        %endfor
        // ------------------------------------------------
        // Remaining pads — classified by pin_map.csv
        // ------------------------------------------------
        <%
            ## Classify all remaining pads
            cats = {}
            for _pn in remaining_pads:
                _cat, _wire, _comment = classify_remaining_pad(_pn, cx, cy)
                cats.setdefault(_cat, []).append((_pn, _wire, _comment))

            ## Ordered categories with labels
            _cat_order = [
                ('d2d_boundary', 'Boundary D2D data pads — routed to top-level ports'),
                ('d2d_internal', 'Internal D2D data pads — routed via interposer (left open)'),
                ('nc',           'NC pads — left open'),
                ('power',        'Power/ground pads — left open'),
                ('d2d_power',    'D2D power pads — left open'),
                ('pll_rsv',      'PLL reserved pads — left open'),
            ]

            ## Build flat list: ('comment', label) or ('pad', pad_name, wire, comment)
            _flat = []
            for _ck, _cl in _cat_order:
                if _ck in cats and cats[_ck]:
                    _flat.append(('comment', _cl, '', ''))
                    for _pn, _w, _cm in cats[_ck]:
                        _flat.append(('pad', _pn, _w, _cm))

            _total_pads = sum(1 for _f in _flat if _f[0] == 'pad')
            _pad_counter = [0]
        %>\
        %for _item in _flat:
        %if _item[0] == 'comment':
        // ---- ${_item[1]} ----
        %else:
<%
    _pn = _item[1]
    _w  = _item[2]
    _cm = _item[3]
    _is_very_last = is_last_chip and _pad_counter[0] == _total_pads - 1
    _pad_counter[0] += 1
    _comma = '' if _is_very_last else ','
    _conn  = _w if _w else ''
%>\
        %if _conn:
        .chip${c}_${_pn}    (${_conn})${_comma}  // ${_cm}
        %else:
        .chip${c}_${_pn}    ()${_comma}  // ${_cm}
        %endif
        %endif
        %endfor
        %endfor
    );

end
endgenerate

endmodule
