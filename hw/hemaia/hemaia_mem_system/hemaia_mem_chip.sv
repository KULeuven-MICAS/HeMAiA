// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yunhao Deng <yunhao.deng@kuleuven.be>

`include "axi/assign.svh"
`include "axi/typedef.svh"
`include "register_interface/typedef.svh"
`include "common_cells/assertions.svh"

module hemaia_mem_chip #(
    // XDMA Defines
    parameter int unsigned ClusterAddressSpace = 48'h400000,
    parameter int unsigned MemBaseAddr = 32'h80000000,
    parameter int unsigned MemBankNum = 32,
    parameter int unsigned MemSize = 32'h100000,
    // Chip ID Type
    parameter type chip_id_t = logic [7:0],
    // D2D Link Phy Enables
    parameter bit EnableEastPhy = 0,
    parameter bit EnableWestPhy = 0,
    parameter bit EnableNorthPhy = 0,
    parameter bit EnableSouthPhy = 0
) (
    input  logic            clk_i,
    input  logic            rst_ni,
    input  chip_id_t        chip_id_i,
    // D2D Link - East side
    input  logic            east_test_being_requested_i,
    output logic            east_test_request_o,
    output logic            flow_control_east_rts_o,
    input  logic            flow_control_east_cts_i,
    input  logic            flow_control_east_rts_i,
    output logic            flow_control_east_cts_o,
    inout  wire      [19:0] east_d2d_io                 [3],
    // D2D Link - West side
    input  logic            west_test_being_requested_i,
    output logic            west_test_request_o,
    output logic            flow_control_west_rts_o,
    input  logic            flow_control_west_cts_i,
    input  logic            flow_control_west_rts_i,
    output logic            flow_control_west_cts_o,
    inout  wire      [19:0] west_d2d_io                 [3],
    // D2D Link - North side
    input  logic            north_test_being_requested_i,
    output logic            north_test_request_o,
    output logic            flow_control_north_rts_o,
    input  logic            flow_control_north_cts_i,
    input  logic            flow_control_north_rts_i,
    output logic            flow_control_north_cts_o,
    inout  wire      [19:0] north_d2d_io                [3],
    // D2D Link - South side
    input  logic            south_test_being_requested_i,
    output logic            south_test_request_o,
    output logic            flow_control_south_rts_o,
    input  logic            flow_control_south_cts_i,
    input  logic            flow_control_south_rts_i,
    output logic            flow_control_south_cts_o,
    inout  wire      [19:0] south_d2d_io                [3]
);

  ///////////////////
  // Chip ID Latch //
  ///////////////////

  // The latched chip_id
  (* false_path *) chip_id_t chip_id;

  always_latch begin
    if (~rst_ni) begin
      chip_id <= chip_id_i;
    end
  end

  ////////////////////////////////////////////
  //  HeMAiA Mem Chip Clock & Reset Manager //
  ////////////////////////////////////////////


  // Master Clock / clk_i: 3.6 GHz
  // Clock Channel 0 / clk_o[0]: SRAM Clock = 3.6 GHz / 4 = 0.9 GHz
  // Clock Channel 1 / clk_o[1]: West D2D TX Clock = 3.6 GHz

  `AXI_LITE_TYPEDEF_ALL_CT(axi_lite_a48_d32, axi_lite_a48_d32_req_t, axi_lite_a48_d32_rsp_t,
                           logic [47:0], logic [32:0], logic [3:0])

  localparam int HeMAiAMemChipDivision[5] = '{4, 1, 1, 1, 1};
  localparam int HeMAiAResetDelays[5] = '{default: 3};

  logic [4:0] clk_vec, rst_n_vec;
  (* syn_keep = 1, syn_preserve = 1 *)
  logic clk_host;
  (* syn_keep = 1, syn_preserve = 1 *)
  logic clk_d2d_phy_east, clk_d2d_phy_west, clk_d2d_phy_north, clk_d2d_phy_south;

  (* syn_keep = 1, syn_preserve = 1 *)
  logic rst_host_n;
  (* syn_keep = 1, syn_preserve = 1 *)
  logic rst_d2d_phy_east_n, rst_d2d_phy_west_n, rst_d2d_phy_north_n, rst_d2d_phy_south_n;


  assign clk_host = clk_vec[0];
  assign clk_d2d_phy_east = clk_vec[1];
  assign clk_d2d_phy_west = clk_vec[2];
  assign clk_d2d_phy_north = clk_vec[3];
  assign clk_d2d_phy_south = clk_vec[4];

  assign rst_host_n = rst_n_vec[0];
  assign rst_d2d_phy_east_n = rst_n_vec[1];
  assign rst_d2d_phy_west_n = rst_n_vec[2];
  assign rst_d2d_phy_north_n = rst_n_vec[3];
  assign rst_d2d_phy_south_n = rst_n_vec[4];

  axi_lite_a48_d32_req_t axi_lite_clk_rst_ctrl_req;
  axi_lite_a48_d32_rsp_t axi_lite_clk_rst_ctrl_rsp;

  hemaia_clk_rst_controller #(
      .NumClocks(5),
      .MaxDivisionWidth(8),
      .DefaultDivision(HeMAiAMemChipDivision),
      .ResetDelays(HeMAiAResetDelays),
      .axi_lite_req_t(axi_lite_a48_d32_req_t),
      .axi_lite_rsp_t(axi_lite_a48_d32_rsp_t)
  ) i_hemaia_clk_rst_controller (
      .test_mode_i('0),
      // for config the controller, same as uart clock
      .control_clk_i(clk_host),
      .control_rst_ni(rst_host_n),
      .axi_lite_req_i(axi_lite_clk_rst_ctrl_req),
      .axi_lite_rsp_o(axi_lite_clk_rst_ctrl_rsp),
      // source clock and reset
      .mst_clk_i(clk_i),
      .mst_rst_ni(rst_ni),
      // generated clocks and resets
      .clk_o(clk_vec),
      .rst_no(rst_n_vec)
  );

  ////////////////////////
  //  HeMAiA Mem System //
  ////////////////////////

  `AXI_TYPEDEF_ALL_CT(axi_a48_d512_i4_u1, axi_a48_d512_i4_u1_req_t, axi_a48_d512_i4_u1_resp_t,
                      logic [47:0], logic [3:0], logic [511:0], logic [63:0], logic [0:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d512_i6_u1, axi_a48_d512_i6_u1_req_t, axi_a48_d512_i6_u1_resp_t,
                      logic [47:0], logic [5:0], logic [511:0], logic [63:0], logic [0:0])

  `AXI_TYPEDEF_ALL_CT(axi_a48_d64_i3_u1, axi_a48_d64_i3_u1_req_t, axi_a48_d64_i3_u1_resp_t,
                      logic [47:0], logic [2:0], logic [63:0], logic [7:0], logic [0:0])
  `AXI_TYPEDEF_ALL_CT(axi_a48_d64_i4_u1, axi_a48_d64_i4_u1_req_t, axi_a48_d64_i4_u1_resp_t,
                      logic [47:0], logic [3:0], logic [63:0], logic [7:0], logic [0:0])

  axi_a48_d512_i4_u1_req_t  axi_wide_mem_sys_to_xbar_req;
  axi_a48_d512_i4_u1_resp_t axi_wide_mem_sys_to_xbar_rsp;
  axi_a48_d512_i6_u1_req_t  axi_wide_xbar_to_mem_sys_req;
  axi_a48_d512_i6_u1_resp_t axi_wide_xbar_to_mem_sys_rsp;

  axi_a48_d64_i3_u1_req_t   axi_narrow_mem_sys_to_xbar_req;
  axi_a48_d64_i3_u1_resp_t  axi_narrow_mem_sys_to_xbar_rsp;
  axi_a48_d64_i4_u1_req_t   axi_narrow_xbar_to_mem_sys_req;
  axi_a48_d64_i4_u1_resp_t  axi_narrow_xbar_to_mem_sys_rsp;

  hemaia_mem_system_for_mem_chip #(
      .chip_id_t(chip_id_t),
      .axi_wide_master_req_t(axi_a48_d512_i6_u1_req_t),
      .axi_wide_master_rsp_t(axi_a48_d512_i6_u1_resp_t),
      .axi_wide_slave_req_t(axi_a48_d512_i4_u1_req_t),
      .axi_wide_slave_rsp_t(axi_a48_d512_i4_u1_resp_t),
      .axi_narrow_master_req_t(axi_a48_d64_i4_u1_req_t),
      .axi_narrow_master_rsp_t(axi_a48_d64_i4_u1_resp_t),
      .axi_narrow_slave_req_t(axi_a48_d64_i3_u1_req_t),
      .axi_narrow_slave_rsp_t(axi_a48_d64_i3_u1_resp_t),
      .ClusterAddressSpace(ClusterAddressSpace),
      .MemBaseAddr(MemBaseAddr),
      .MemBankNum(MemBankNum),
      .MemSize(MemSize)
  ) i_hemaia_mem_system (
      .clk_i                  (clk_host),
      .rst_ni                 (rst_host_n),
      .chip_id_i              (chip_id),
      .axi_wide_master_req_i  (axi_wide_xbar_to_mem_sys_req),
      .axi_wide_master_rsp_o  (axi_wide_xbar_to_mem_sys_rsp),
      .axi_wide_slave_req_o   (axi_wide_mem_sys_to_xbar_req),
      .axi_wide_slave_rsp_i   (axi_wide_mem_sys_to_xbar_rsp),
      .axi_narrow_master_req_i(axi_narrow_xbar_to_mem_sys_req),
      .axi_narrow_master_rsp_o(axi_narrow_xbar_to_mem_sys_rsp),
      .axi_narrow_slave_req_o (axi_narrow_mem_sys_to_xbar_req),
      .axi_narrow_slave_rsp_i (axi_narrow_mem_sys_to_xbar_rsp)
  );

  //////////////////////////////////////////////////////////////////////////////
  //  HeMAiA Mem Chip DRAM Controller: Reserved for future connection to FPGA //
  //////////////////////////////////////////////////////////////////////////////
  // Assign 0 Now for the future usage
  axi_a48_d512_i6_u1_req_t  axi_wide_xbar_to_dram_req;
  axi_a48_d512_i6_u1_resp_t axi_wide_xbar_to_dram_rsp;
  assign axi_wide_xbar_to_dram_rsp = '0;

  ///////////////////////////////////////////////
  //  IDMA at Mem Chip for the Broadcast Usage //
  ///////////////////////////////////////////////

  // burst request
  typedef struct packed {
    logic [3:0]      id;
    logic [47:0]     src,         dst;
    logic [47:0]     num_bytes;
    axi_pkg::cache_t cache_src,   cache_dst;
    axi_pkg::burst_t burst_src,   burst_dst;
    logic            decouple_rw;
    logic            deburst;
    logic            serialize;
  } idma_burst_req_t;

  // local regbus definition
  `REG_BUS_TYPEDEF_ALL(idma_cfg_reg_a48_d64, logic [47:0], logic [63:0], logic [7:0])

  idma_burst_req_t idma_burst_req;
  logic idma_be_valid;
  logic idma_be_ready;
  logic idma_be_idle;
  logic idma_be_trans_complete;

  idma_cfg_reg_a48_d64_req_t idma_cfg_reg_req;
  idma_cfg_reg_a48_d64_rsp_t idma_cfg_reg_rsp;
  axi_a48_d64_i4_u1_req_t axi_narrow_xbar_to_idma_cfg_req;
  axi_a48_d64_i4_u1_resp_t axi_narrow_xbar_to_idma_cfg_rsp;

  axi_to_reg #(
      .ADDR_WIDTH(48),
      .DATA_WIDTH(64),
      .ID_WIDTH  (4),
      .USER_WIDTH(1),
      .axi_req_t (axi_a48_d64_i4_u1_req_t),
      .axi_rsp_t (axi_a48_d64_i4_u1_resp_t),
      .reg_req_t (idma_cfg_reg_a48_d64_req_t),
      .reg_rsp_t (idma_cfg_reg_a48_d64_rsp_t)
  ) i_axi_to_reg_sys_idma_cfg (
      .clk_i(clk_host),
      .rst_ni(rst_host_n),
      .testmode_i(1'b0),
      .axi_req_i(axi_narrow_xbar_to_idma_cfg_req),
      .axi_rsp_o(axi_narrow_xbar_to_idma_cfg_rsp),
      .reg_req_o(idma_cfg_reg_req),
      .reg_rsp_i(idma_cfg_reg_rsp)
  );

  idma_reg64_frontend #(
      .DmaAddrWidth  ('d64),
      .dma_regs_req_t(idma_cfg_reg_a48_d64_req_t),
      .dma_regs_rsp_t(idma_cfg_reg_a48_d64_rsp_t),
      .burst_req_t   (idma_burst_req_t)
  ) i_idma_reg64_frontend_sys_idma (
      .clk_i           (clk_host),
      .rst_ni          (rst_host_n),
      .dma_ctrl_req_i  (idma_cfg_reg_req),
      .dma_ctrl_rsp_o  (idma_cfg_reg_rsp),
      .burst_req_o     (idma_burst_req),
      .valid_o         (idma_be_valid),
      .ready_i         (idma_be_ready),
      .backend_idle_i  (idma_be_idle),
      .trans_complete_i(idma_be_trans_complete)
  );

  axi_a48_d512_i4_u1_req_t  axi_idma_to_soc_xbar_req;
  axi_a48_d512_i4_u1_resp_t axi_idma_to_soc_xbar_rsp;

  axi_dma_backend #(
      .DataWidth     (512),
      .AddrWidth     (48),
      .IdWidth       (4),
      .AxReqFifoDepth('d64),
      .TransFifoDepth('d16),
      .BufferDepth   ('d3),
      .axi_req_t     (axi_a48_d512_i4_u1_req_t),
      .axi_res_t     (axi_a48_d512_i4_u1_resp_t),
      .burst_req_t   (idma_burst_req_t),
      .DmaIdWidth    ('d32),
      .DmaTracing    (1'b1)
  ) i_axi_dma_backend_sys_idma (
      .clk_i           (clk_host),
      .rst_ni          (rst_host_n),
      .chip_id_i       (chip_id),
      .dma_id_i        ('d0),
      .axi_dma_req_o   (axi_idma_to_soc_xbar_req),
      .axi_dma_res_i   (axi_idma_to_soc_xbar_rsp),
      .burst_req_i     (idma_burst_req),
      .valid_i         (idma_be_valid),
      .ready_o         (idma_be_ready),
      .backend_idle_o  (idma_be_idle),
      .trans_complete_o(idma_be_trans_complete)
  );

  //////////////////////
  //  HeMAiA D2D Link //
  //////////////////////

  axi_lite_a48_d32_req_t axi_lite_d2d_link_ctrl_req;
  axi_lite_a48_d32_rsp_t axi_lite_d2d_link_ctrl_rsp;

  axi_a48_d512_i4_u1_req_t axi_soc_xbar_to_d2d_link_post_id_conv_req;
  axi_a48_d512_i4_u1_resp_t axi_soc_xbar_to_d2d_link_post_id_conv_rsp;
  axi_a48_d512_i6_u1_req_t axi_soc_xbar_to_d2d_link_pre_id_conv_req;
  axi_a48_d512_i6_u1_resp_t axi_soc_xbar_to_d2d_link_pre_id_conv_rsp;

  axi_a48_d512_i4_u1_req_t axi_d2d_link_to_soc_xbar_req;
  axi_a48_d512_i4_u1_resp_t axi_d2d_link_to_soc_xbar_rsp;

  hemaia_d2d_link #(
      .EnableEastPhy(EnableEastPhy),
      .EnableWestPhy(EnableWestPhy),
      .EnableNorthPhy(EnableNorthPhy),
      .EnableSouthPhy(EnableSouthPhy),
      .chip_id_t(chip_id_t),
      .axi_req_t(axi_a48_d512_i4_u1_req_t),
      .axi_rsp_t(axi_a48_d512_i4_u1_resp_t),
      .axi_lite_req_t(axi_lite_a48_d32_req_t),
      .axi_lite_rsp_t(axi_lite_a48_d32_rsp_t),
      .aw_chan_t(axi_a48_d512_i4_u1_aw_chan_t),
      .ar_chan_t(axi_a48_d512_i4_u1_ar_chan_t),
      .r_chan_t(axi_a48_d512_i4_u1_r_chan_t),
      .w_chan_t(axi_a48_d512_i4_u1_w_chan_t),
      .b_chan_t(axi_a48_d512_i4_u1_b_chan_t)
  ) i_d2d_link (
      .chip_id_i(chip_id),

      .control_clk_i(clk_host),
      .control_rst_ni(rst_host_n),
      .digital_clk_i(clk_host),
      .east_phy_tx_clk_i(clk_d2d_phy_east),
      .west_phy_tx_clk_i(clk_d2d_phy_west),
      .north_phy_tx_clk_i(clk_d2d_phy_north),
      .south_phy_tx_clk_i(clk_d2d_phy_south),
      .rst_ni(rst_host_n),

      .axi_lite_req_i(axi_lite_d2d_link_ctrl_req),
      .axi_lite_rsp_o(axi_lite_d2d_link_ctrl_rsp),

      .axi_in_req_i (axi_soc_xbar_to_d2d_link_post_id_conv_req),
      .axi_in_rsp_o (axi_soc_xbar_to_d2d_link_post_id_conv_rsp),
      .axi_out_req_o(axi_d2d_link_to_soc_xbar_req),
      .axi_out_rsp_i(axi_d2d_link_to_soc_xbar_rsp),

      .east_test_being_requested_i,
      .east_test_request_o,
      .flow_control_east_rts_o,
      .flow_control_east_cts_i,
      .flow_control_east_rts_i,
      .flow_control_east_cts_o,
      .east_phy_io(east_d2d_io),

      .west_test_being_requested_i,
      .west_test_request_o,
      .flow_control_west_rts_o,
      .flow_control_west_cts_i,
      .flow_control_west_rts_i,
      .flow_control_west_cts_o,
      .west_phy_io(west_d2d_io),

      .north_test_being_requested_i,
      .north_test_request_o,
      .flow_control_north_rts_o,
      .flow_control_north_cts_i,
      .flow_control_north_rts_i,
      .flow_control_north_cts_o,
      .north_phy_io(north_d2d_io),

      .south_test_being_requested_i,
      .south_test_request_o,
      .flow_control_south_rts_o,
      .flow_control_south_cts_i,
      .flow_control_south_rts_i,
      .flow_control_south_cts_o,
      .south_phy_io(south_d2d_io)
  );

  axi_id_remap #(
      .AxiSlvPortIdWidth(6),
      .AxiSlvPortMaxUniqIds(16),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(4),
      .slv_req_t(axi_a48_d512_i6_u1_req_t),
      .slv_resp_t(axi_a48_d512_i6_u1_resp_t),
      .mst_req_t(axi_a48_d512_i4_u1_req_t),
      .mst_resp_t(axi_a48_d512_i4_u1_resp_t)
  ) i_soc_xbar_to_d2d_link_iwc (
      .clk_i(clk_host),
      .rst_ni(rst_host_n),
      .slv_req_i(axi_soc_xbar_to_d2d_link_pre_id_conv_req),
      .slv_resp_o(axi_soc_xbar_to_d2d_link_pre_id_conv_rsp),
      .mst_req_o(axi_soc_xbar_to_d2d_link_post_id_conv_req),
      .mst_resp_i(axi_soc_xbar_to_d2d_link_post_id_conv_rsp)
  );

  //////////////////////////////////////////////////////////////////
  //  Connection 1: AXI to AXI Lite, the AXI Lite Peripheral XBAR //
  //////////////////////////////////////////////////////////////////

  localparam axi_pkg::xbar_cfg_t HeMAiAMemChipPeriphXbarCfg = '{
      NoSlvPorts: 1,
      NoMstPorts: 2,
      MaxSlvTrans: 4,
      MaxMstTrans: 4,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      PipelineStages: 0,
      AxiIdWidthSlvPorts: 0,
      AxiIdUsedSlvPorts: 0,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 32,
      NoAddrRules: 2
  };

  // M0 to clk_rst_controller, M1 to d2d_link_ctrl
  typedef struct packed {
    logic [31:0] idx;
    logic [47:0] start_addr;
    logic [47:0] end_addr;
  } xbar_rule_48_t;

  xbar_rule_48_t [1:0] HeMAiAMemChipPeriphXbarAddrmap;
  assign HeMAiAMemChipPeriphXbarAddrmap = '{
          '{idx: 0, start_addr: {chip_id, 40'h002005000}, end_addr: {chip_id, 40'h002006000}},
          '{idx: 1, start_addr: {chip_id, 40'h002007000}, end_addr: {chip_id, 40'h002008000}}
      };

  axi_lite_a48_d32_req_t [1:0] hemaia_mem_chip_periph_xbar_to_peripheral_req;
  axi_lite_a48_d32_rsp_t [1:0] hemaia_mem_chip_periph_xbar_to_peripheral_rsp;
  axi_lite_a48_d32_req_t [0:0] hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_lite_req;
  axi_lite_a48_d32_rsp_t [0:0] hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_lite_rsp;

  assign axi_lite_clk_rst_ctrl_req = hemaia_mem_chip_periph_xbar_to_peripheral_req[0];
  assign hemaia_mem_chip_periph_xbar_to_peripheral_rsp[0] = axi_lite_clk_rst_ctrl_rsp;
  assign axi_lite_d2d_link_ctrl_req = hemaia_mem_chip_periph_xbar_to_peripheral_req[1];
  assign hemaia_mem_chip_periph_xbar_to_peripheral_rsp[1] = axi_lite_d2d_link_ctrl_rsp;

  axi_lite_xbar #(
      .Cfg       (HeMAiAMemChipPeriphXbarCfg),
      .aw_chan_t (axi_lite_a48_d32_aw_chan_t),
      .w_chan_t  (axi_lite_a48_d32_w_chan_t),
      .b_chan_t  (axi_lite_a48_d32_b_chan_t),
      .ar_chan_t (axi_lite_a48_d32_ar_chan_t),
      .r_chan_t  (axi_lite_a48_d32_r_chan_t),
      .axi_req_t (axi_lite_a48_d32_req_t),
      .axi_resp_t(axi_lite_a48_d32_rsp_t),
      .rule_t    (xbar_rule_48_t)
  ) i_soc_axi_lite_periph_xbar (
      .clk_i                (clk_host),
      .rst_ni               (rst_host_n),
      .test_i               ('0),
      .slv_ports_req_i      (hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_lite_req),
      .slv_ports_resp_o     (hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_lite_rsp),
      .mst_ports_req_o      (hemaia_mem_chip_periph_xbar_to_peripheral_req),
      .mst_ports_resp_i     (hemaia_mem_chip_periph_xbar_to_peripheral_rsp),
      .addr_map_i           (HeMAiAMemChipPeriphXbarAddrmap),
      .en_default_mst_port_i('0),
      .default_mst_port_i   ('0)
  );

  `AXI_TYPEDEF_ALL_CT(axi_a48_d32_i4_u1, axi_a48_d32_i4_u1_req_t, axi_a48_d32_i4_u1_resp_t,
                      logic [47:0], logic [3:0], logic [31:0], logic [3:0], logic [0:0])

  axi_a48_d32_i4_u1_req_t  hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_req;
  axi_a48_d32_i4_u1_resp_t hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_rsp;

  axi_to_axi_lite #(
      .AxiAddrWidth(48),
      .AxiDataWidth(32),
      .AxiIdWidth(4),
      .AxiUserWidth(1),
      .AxiMaxWriteTxns(4),
      .AxiMaxReadTxns(4),
      .FallThrough(0),
      .full_req_t(axi_a48_d32_i4_u1_req_t),
      .full_resp_t(axi_a48_d32_i4_u1_resp_t),
      .lite_req_t(axi_lite_a48_d32_req_t),
      .lite_resp_t(axi_lite_a48_d32_rsp_t)
  ) i_axi_to_axi_lite_peripheral_xbar (
      .clk_i(clk_host),
      .rst_ni(rst_host_n),
      .test_i('0),
      .slv_req_i(hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_req),
      .slv_resp_o(hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_rsp),
      .mst_req_o(hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_lite_req[0]),
      .mst_resp_i(hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_lite_rsp[0])
  );

  axi_a48_d64_i4_u1_req_t  axi_narrow_xbar_to_periph_xbar_req;
  axi_a48_d64_i4_u1_resp_t axi_narrow_xbar_to_periph_xbar_rsp;

  axi_dw_converter #(
      .AxiMaxReads        (16),
      .AxiSlvPortDataWidth(64),
      .AxiMstPortDataWidth(32),
      .AxiAddrWidth       (48),
      .AxiIdWidth         (4),
      .aw_chan_t          (axi_a48_d64_i4_u1_aw_chan_t),
      .mst_w_chan_t       (axi_a48_d32_i4_u1_w_chan_t),
      .slv_w_chan_t       (axi_a48_d64_i4_u1_w_chan_t),
      .b_chan_t           (axi_a48_d32_i4_u1_b_chan_t),
      .ar_chan_t          (axi_a48_d32_i4_u1_ar_chan_t),
      .mst_r_chan_t       (axi_a48_d32_i4_u1_r_chan_t),
      .slv_r_chan_t       (axi_a48_d64_i4_u1_r_chan_t),
      .axi_mst_req_t      (axi_a48_d32_i4_u1_req_t),
      .axi_mst_resp_t     (axi_a48_d32_i4_u1_resp_t),
      .axi_slv_req_t      (axi_a48_d64_i4_u1_req_t),
      .axi_slv_resp_t     (axi_a48_d64_i4_u1_resp_t)
  ) i_axi_to_axi_lite_dw (
      .clk_i(clk_host),
      .rst_ni(rst_host_n),
      .slv_req_i(axi_narrow_xbar_to_periph_xbar_req),
      .slv_resp_o(axi_narrow_xbar_to_periph_xbar_rsp),
      .mst_req_o(hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_req),
      .mst_resp_i(hemaia_mem_chip_narrow_xbar_to_periph_xbar_dwc_rsp)
  );

  //////////////////////////////////
  //  Connection 2: AXI Wide XBAR //
  //////////////////////////////////
  localparam axi_pkg::xbar_cfg_t HeMAiAMemChipWideXbarCfg = '{
      NoSlvPorts: 4,
      NoMstPorts: 4,
      MaxSlvTrans: 16,
      MaxMstTrans: 16,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      PipelineStages: 0,
      AxiIdWidthSlvPorts: 4,
      AxiIdUsedSlvPorts: 4,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 512,
      NoAddrRules: 7
  };

  xbar_rule_48_t [6:0] HeMAiAMemChipWideXbarAddrmap;
  assign HeMAiAMemChipWideXbarAddrmap = '{
          '{
              idx: 1,
              start_addr: {chip_id, 40'h80000000},
              end_addr: {chip_id, 40'h80000000 + MemSize}
          },
          '{idx: 1, start_addr: {chip_id, 40'hFFFFC000}, end_addr: {chip_id, 40'hFFFFD000}},
          // XDMA Handshake goes to 2
          '{
              idx: 2,
              start_addr: {chip_id, 40'hFFFFB000},
              end_addr: {chip_id, 40'hFFFFC000}
          },
          '{idx: 2, start_addr: {chip_id, 40'hFFFFD000}, end_addr: {chip_id, 40'h100000000}},
          // Peripheral goes to 2
          '{
              idx: 2,
              start_addr: {chip_id, 40'h2000000},
              end_addr: {chip_id, 40'h2010000}
          },
          // IDMA Ctrl goes to 2
          '{
              idx: 2,
              start_addr: {chip_id, 40'h5000000},
              end_addr: {chip_id, 40'h5010000}
          },
          '{idx: 3, start_addr: {chip_id, 40'h100000000}, end_addr: {chip_id, 40'hFFFFFFFFFF}}
      };

  axi_a48_d512_i4_u1_req_t [3:0] master_to_axi_wide_xbar_req;
  axi_a48_d512_i4_u1_resp_t [3:0] master_to_axi_wide_xbar_rsp;
  axi_a48_d512_i6_u1_req_t [3:0] axi_wide_xbar_to_slave_req;
  axi_a48_d512_i6_u1_resp_t [3:0] axi_wide_xbar_to_slave_rsp;

  axi_a48_d512_i4_u1_req_t axi_narrow_xbar_to_wide_xbar_dwc_req;
  axi_a48_d512_i4_u1_resp_t axi_narrow_xbar_to_wide_xbar_dwc_rsp;

  axi_a48_d512_i6_u1_req_t axi_wide_xbar_to_narrow_xbar_req;
  axi_a48_d512_i6_u1_resp_t axi_wide_xbar_to_narrow_xbar_rsp;

  assign master_to_axi_wide_xbar_req[0] = axi_d2d_link_to_soc_xbar_req;
  assign axi_d2d_link_to_soc_xbar_rsp = master_to_axi_wide_xbar_rsp[0];
  assign master_to_axi_wide_xbar_req[1] = axi_wide_mem_sys_to_xbar_req;
  assign axi_wide_mem_sys_to_xbar_rsp = master_to_axi_wide_xbar_rsp[1];
  assign master_to_axi_wide_xbar_req[2] = axi_narrow_xbar_to_wide_xbar_dwc_req;
  assign axi_narrow_xbar_to_wide_xbar_dwc_rsp = master_to_axi_wide_xbar_rsp[2];
  assign master_to_axi_wide_xbar_req[3] = axi_idma_to_soc_xbar_req;
  assign axi_idma_to_soc_xbar_rsp = master_to_axi_wide_xbar_rsp[3];

  assign axi_soc_xbar_to_d2d_link_pre_id_conv_req = axi_wide_xbar_to_slave_req[0];
  assign axi_wide_xbar_to_slave_rsp[0] = axi_soc_xbar_to_d2d_link_pre_id_conv_rsp;
  assign axi_wide_xbar_to_mem_sys_req = axi_wide_xbar_to_slave_req[1];
  assign axi_wide_xbar_to_slave_rsp[1] = axi_wide_xbar_to_mem_sys_rsp;
  assign axi_wide_xbar_to_narrow_xbar_req = axi_wide_xbar_to_slave_req[2];
  assign axi_wide_xbar_to_slave_rsp[2] = axi_wide_xbar_to_narrow_xbar_rsp;
  assign axi_wide_xbar_to_dram_req = axi_wide_xbar_to_slave_req[3];
  assign axi_wide_xbar_to_slave_rsp[3] = axi_wide_xbar_to_dram_rsp;

  axi_xbar #(
      .Cfg          (HeMAiAMemChipWideXbarCfg),
      .Connectivity ('1),
      .ATOPs        (0),
      .slv_aw_chan_t(axi_a48_d512_i4_u1_aw_chan_t),
      .mst_aw_chan_t(axi_a48_d512_i6_u1_aw_chan_t),
      .w_chan_t     (axi_a48_d512_i4_u1_w_chan_t),
      .slv_b_chan_t (axi_a48_d512_i4_u1_b_chan_t),
      .mst_b_chan_t (axi_a48_d512_i6_u1_b_chan_t),
      .slv_ar_chan_t(axi_a48_d512_i4_u1_ar_chan_t),
      .mst_ar_chan_t(axi_a48_d512_i6_u1_ar_chan_t),
      .slv_r_chan_t (axi_a48_d512_i4_u1_r_chan_t),
      .mst_r_chan_t (axi_a48_d512_i6_u1_r_chan_t),
      .slv_req_t    (axi_a48_d512_i4_u1_req_t),
      .slv_resp_t   (axi_a48_d512_i4_u1_resp_t),
      .mst_req_t    (axi_a48_d512_i6_u1_req_t),
      .mst_resp_t   (axi_a48_d512_i6_u1_resp_t),
      .rule_t       (xbar_rule_48_t)
  ) i_axi_wide_xbar (
      .clk_i                (clk_host),
      .rst_ni               (rst_host_n),
      .test_i               ('0),
      .slv_ports_req_i      (master_to_axi_wide_xbar_req),
      .slv_ports_resp_o     (master_to_axi_wide_xbar_rsp),
      .mst_ports_req_o      (axi_wide_xbar_to_slave_req),
      .mst_ports_resp_i     (axi_wide_xbar_to_slave_rsp),
      .addr_map_i           (HeMAiAMemChipWideXbarAddrmap),
      .en_default_mst_port_i('1),
      .default_mst_port_i   ('0)
  );

  /////////////////////////////////////////////////
  //  Connection 3: AXI Wide to Narrow IWC + DWC //
  /////////////////////////////////////////////////
  axi_a48_d64_i4_u1_req_t  axi_narrow_xbar_to_wide_xbar_req;
  axi_a48_d64_i4_u1_resp_t axi_narrow_xbar_to_wide_xbar_rsp;

  axi_dw_converter #(
      .AxiMaxReads        (16),
      .AxiSlvPortDataWidth(64),
      .AxiMstPortDataWidth(512),
      .AxiAddrWidth       (48),
      .AxiIdWidth         (4),
      .aw_chan_t          (axi_a48_d64_i4_u1_aw_chan_t),
      .mst_w_chan_t       (axi_a48_d512_i4_u1_w_chan_t),
      .slv_w_chan_t       (axi_a48_d64_i4_u1_w_chan_t),
      .b_chan_t           (axi_a48_d64_i4_u1_b_chan_t),
      .ar_chan_t          (axi_a48_d64_i4_u1_ar_chan_t),
      .mst_r_chan_t       (axi_a48_d512_i4_u1_r_chan_t),
      .slv_r_chan_t       (axi_a48_d64_i4_u1_r_chan_t),
      .axi_mst_req_t      (axi_a48_d512_i4_u1_req_t),
      .axi_mst_resp_t     (axi_a48_d512_i4_u1_resp_t),
      .axi_slv_req_t      (axi_a48_d64_i4_u1_req_t),
      .axi_slv_resp_t     (axi_a48_d64_i4_u1_resp_t)
  ) i_axi_narrow_to_axi_wide_dw (
      .clk_i(clk_host),
      .rst_ni(rst_host_n),
      .slv_req_i(axi_narrow_xbar_to_wide_xbar_req),
      .slv_resp_o(axi_narrow_xbar_to_wide_xbar_rsp),
      .mst_req_o(axi_narrow_xbar_to_wide_xbar_dwc_req),
      .mst_resp_i(axi_narrow_xbar_to_wide_xbar_dwc_rsp)
  );

  `AXI_TYPEDEF_ALL_CT(axi_a48_d64_i6_u1, axi_a48_d64_i6_u1_req_t, axi_a48_d64_i6_u1_resp_t,
                      logic [47:0], logic [5:0], logic [63:0], logic [7:0], logic [0:0])

  axi_a48_d64_i6_u1_req_t  axi_wide_xbar_to_narrow_xbar_dwc_req;
  axi_a48_d64_i6_u1_resp_t axi_wide_xbar_to_narrow_xbar_dwc_rsp;

  axi_dw_converter #(
      .AxiMaxReads        (16),
      .AxiSlvPortDataWidth(512),
      .AxiMstPortDataWidth(64),
      .AxiAddrWidth       (48),
      .AxiIdWidth         (6),
      .aw_chan_t          (axi_a48_d512_i6_u1_aw_chan_t),
      .mst_w_chan_t       (axi_a48_d64_i6_u1_w_chan_t),
      .slv_w_chan_t       (axi_a48_d512_i6_u1_w_chan_t),
      .b_chan_t           (axi_a48_d512_i6_u1_b_chan_t),
      .ar_chan_t          (axi_a48_d512_i6_u1_ar_chan_t),
      .mst_r_chan_t       (axi_a48_d64_i6_u1_r_chan_t),
      .slv_r_chan_t       (axi_a48_d512_i6_u1_r_chan_t),
      .axi_mst_req_t      (axi_a48_d64_i6_u1_req_t),
      .axi_mst_resp_t     (axi_a48_d64_i6_u1_resp_t),
      .axi_slv_req_t      (axi_a48_d512_i6_u1_req_t),
      .axi_slv_resp_t     (axi_a48_d512_i6_u1_resp_t)
  ) i_wide_xbar_to_narrow_xbar_dwc (
      .clk_i(clk_host),
      .rst_ni(rst_host_n),
      .slv_req_i(axi_wide_xbar_to_narrow_xbar_req),
      .slv_resp_o(axi_wide_xbar_to_narrow_xbar_rsp),
      .mst_req_o(axi_wide_xbar_to_narrow_xbar_dwc_req),
      .mst_resp_i(axi_wide_xbar_to_narrow_xbar_dwc_rsp)
  );


  axi_a48_d64_i3_u1_req_t  axi_wide_xbar_to_narrow_xbar_dwc_iwc_req;
  axi_a48_d64_i3_u1_resp_t axi_wide_xbar_to_narrow_xbar_dwc_iwc_rsp;

  axi_id_remap #(
      .AxiSlvPortIdWidth(6),
      .AxiSlvPortMaxUniqIds(4),
      .AxiMaxTxnsPerId(4),
      .AxiMstPortIdWidth(3),
      .slv_req_t(axi_a48_d64_i6_u1_req_t),
      .slv_resp_t(axi_a48_d64_i6_u1_resp_t),
      .mst_req_t(axi_a48_d64_i3_u1_req_t),
      .mst_resp_t(axi_a48_d64_i3_u1_resp_t)
  ) i_wide_xbar_to_narrow_xbar_iwc (
      .clk_i(clk_host),
      .rst_ni(rst_host_n),
      .slv_req_i(axi_wide_xbar_to_narrow_xbar_dwc_req),
      .slv_resp_o(axi_wide_xbar_to_narrow_xbar_dwc_rsp),
      .mst_req_o(axi_wide_xbar_to_narrow_xbar_dwc_iwc_req),
      .mst_resp_i(axi_wide_xbar_to_narrow_xbar_dwc_iwc_rsp)
  );

  ////////////////////////////////////
  //  Connection 4: AXI Narrow XBar //
  ////////////////////////////////////
  localparam axi_pkg::xbar_cfg_t HeMAiAMemChipNarrowXbarCfg = '{
      NoSlvPorts: 2,
      NoMstPorts: 4,
      MaxSlvTrans: 16,
      MaxMstTrans: 16,
      FallThrough: 0,
      LatencyMode: axi_pkg::CUT_ALL_PORTS,
      PipelineStages: 0,
      AxiIdWidthSlvPorts: 3,
      AxiIdUsedSlvPorts: 3,
      UniqueIds: 0,
      AxiAddrWidth: 48,
      AxiDataWidth: 64,
      NoAddrRules: 4
  };

  xbar_rule_48_t [3:0] HeMAiAMemChipNarrowXbarAddrmap;
  assign HeMAiAMemChipNarrowXbarAddrmap = '{
          '{idx: 1, start_addr: {chip_id, 40'hFFFFB000}, end_addr: {chip_id, 40'hFFFFC000}},
          '{idx: 1, start_addr: {chip_id, 40'hFFFFD000}, end_addr: {chip_id, 40'h100000000}},
          '{idx: 2, start_addr: {chip_id, 40'h2000000}, end_addr: {chip_id, 40'h2010000}},
          '{idx: 3, start_addr: {chip_id, 40'h5000000}, end_addr: {chip_id, 40'h5010000}}
      };

  axi_a48_d64_i3_u1_req_t  [1:0] master_to_axi_narrow_xbar_req;
  axi_a48_d64_i3_u1_resp_t [1:0] master_to_axi_narrow_xbar_rsp;
  axi_a48_d64_i4_u1_req_t  [3:0] axi_narrow_xbar_to_slave_req;
  axi_a48_d64_i4_u1_resp_t [3:0] axi_narrow_xbar_to_slave_rsp;

  assign master_to_axi_narrow_xbar_req[0] = axi_wide_xbar_to_narrow_xbar_dwc_iwc_req;
  assign axi_wide_xbar_to_narrow_xbar_dwc_iwc_rsp = master_to_axi_narrow_xbar_rsp[0];
  assign master_to_axi_narrow_xbar_req[1] = axi_narrow_mem_sys_to_xbar_req;
  assign axi_narrow_mem_sys_to_xbar_rsp = master_to_axi_narrow_xbar_rsp[1];

  assign axi_narrow_xbar_to_wide_xbar_req = axi_narrow_xbar_to_slave_req[0];
  assign axi_narrow_xbar_to_slave_rsp[0] = axi_narrow_xbar_to_wide_xbar_rsp;
  assign axi_narrow_xbar_to_mem_sys_req = axi_narrow_xbar_to_slave_req[1];
  assign axi_narrow_xbar_to_slave_rsp[1] = axi_narrow_xbar_to_mem_sys_rsp;
  assign axi_narrow_xbar_to_periph_xbar_req = axi_narrow_xbar_to_slave_req[2];
  assign axi_narrow_xbar_to_slave_rsp[2] = axi_narrow_xbar_to_periph_xbar_rsp;
  assign axi_narrow_xbar_to_idma_cfg_req = axi_narrow_xbar_to_slave_req[3];
  assign axi_narrow_xbar_to_slave_rsp[3] = axi_narrow_xbar_to_idma_cfg_rsp;

  axi_xbar #(
      .Cfg          (HeMAiAMemChipNarrowXbarCfg),
      .Connectivity ('1),
      .ATOPs        (0),
      .slv_aw_chan_t(axi_a48_d64_i3_u1_aw_chan_t),
      .mst_aw_chan_t(axi_a48_d64_i4_u1_aw_chan_t),
      .w_chan_t     (axi_a48_d64_i3_u1_w_chan_t),
      .slv_b_chan_t (axi_a48_d64_i3_u1_b_chan_t),
      .mst_b_chan_t (axi_a48_d64_i4_u1_b_chan_t),
      .slv_ar_chan_t(axi_a48_d64_i3_u1_ar_chan_t),
      .mst_ar_chan_t(axi_a48_d64_i4_u1_ar_chan_t),
      .slv_r_chan_t (axi_a48_d64_i3_u1_r_chan_t),
      .mst_r_chan_t (axi_a48_d64_i4_u1_r_chan_t),
      .slv_req_t    (axi_a48_d64_i3_u1_req_t),
      .slv_resp_t   (axi_a48_d64_i3_u1_resp_t),
      .mst_req_t    (axi_a48_d64_i4_u1_req_t),
      .mst_resp_t   (axi_a48_d64_i4_u1_resp_t),
      .rule_t       (xbar_rule_48_t)
  ) i_axi_narrow_xbar (
      .clk_i                (clk_host),
      .rst_ni               (rst_host_n),
      .test_i               ('0),
      .slv_ports_req_i      (master_to_axi_narrow_xbar_req),
      .slv_ports_resp_o     (master_to_axi_narrow_xbar_rsp),
      .mst_ports_req_o      (axi_narrow_xbar_to_slave_req),
      .mst_ports_resp_i     (axi_narrow_xbar_to_slave_rsp),
      .addr_map_i           (HeMAiAMemChipNarrowXbarAddrmap),
      .en_default_mst_port_i('1),
      .default_mst_port_i   ('0)
  );

endmodule
