// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// This controller acts as a gatekeeper to a quadrant. It can isolate all its
// (downstream) AXI ports, gate its clock, and assert its reset through a
// register file mapped on the narrow AXI port.

// Author: Paul Scheffler <paulsc@iis.ee.ethz.ch>

// AUTOMATICALLY GENERATED by occamygen.py; edit the script instead.

module ${name}_quadrant_s1_ctrl
  import ${name}_pkg::*;
  import ${name}_quadrant_s1_reg_pkg::*;
#(
  parameter type tlb_entry_t = logic
)(
  input  logic clk_i,
  input  logic rst_ni,
  input  logic test_mode_i,
  input  chip_id_t chip_id_i,

  // Quadrant clock and reset
  output logic [${num_clusters-1}:0] clk_quadrant_cluster_o,
  output logic clk_quadrant_uncore_o,
  output logic rst_quadrant_no,

  // Quadrant control signals
  output ${name}_quadrant_s1_reg2hw_isolate_reg_t isolate_o,
  input  ${name}_quadrant_s1_hw2reg_isolated_reg_t isolated_i,
  output logic ro_enable_o,
  output logic ro_flush_valid_o,
  input  logic ro_flush_ready_i,
  
  output logic [${ro_cache_regions-1}:0][${soc_wide_xbar.in_quadrant_0.aw-1}:0] ro_start_addr_o,
  output logic [${ro_cache_regions-1}:0][${soc_wide_xbar.in_quadrant_0.aw-1}:0] ro_end_addr_o,

  // Upward (SoC) narrow ports
  output ${soc_narrow_xbar.in_s1_quadrant_0.req_type()} soc_out_req_o,
  input  ${soc_narrow_xbar.in_s1_quadrant_0.rsp_type()} soc_out_rsp_i,
  input  ${soc_narrow_xbar.out_s1_quadrant_0.req_type()} soc_in_req_i,
  output ${soc_narrow_xbar.out_s1_quadrant_0.rsp_type()} soc_in_rsp_o,

  // TLB narrow and wide configuration ports
  %if narrow_tlb_cfg:
  output tlb_entry_t [${narrow_tlb_entries-1}:0] narrow_tlb_entries_o,
  output logic narrow_tlb_enable_o,
  %endif
  %if wide_tlb_cfg:
  output tlb_entry_t [${wide_tlb_entries-1}:0] wide_tlb_entries_o,
  output logic wide_tlb_enable_o,
  %endif

  // Quadrant narrow ports
  output ${soc_narrow_xbar.out_s1_quadrant_0.req_type()} quadrant_out_req_o,
  input  ${soc_narrow_xbar.out_s1_quadrant_0.rsp_type()} quadrant_out_rsp_i,
  input  ${soc_narrow_xbar.in_s1_quadrant_0.req_type()} quadrant_in_req_i,
  output ${soc_narrow_xbar.in_s1_quadrant_0.rsp_type()} quadrant_in_rsp_o
);

  // Upper half of quadrant space reserved for internal use (same size as for all clusters)
  addr_t [0:0] internal_xbar_base_addr;
  assign internal_xbar_base_addr = {chip_id_i, S1QuadrantCfgBaseOffset[AddrWidth-ChipIdWidth-1:0]};

  // Controller crossbar: shims off for access to internal space
  ${module}

  // Connect upward (SoC) narrow ports
  assign soc_out_req_o = ${quadrant_s1_ctrl_xbars["quad_to_soc"].out_out.req_name()};
  assign ${quadrant_s1_ctrl_xbars["quad_to_soc"].out_out.rsp_name()} = soc_out_rsp_i;
  assign ${quadrant_s1_ctrl_xbars["soc_to_quad"].in_in.req_name()} = soc_in_req_i;
  assign soc_in_rsp_o = ${quadrant_s1_ctrl_xbars["soc_to_quad"].in_in.rsp_name()};

  // Connect quadrant narrow ports
  assign quadrant_out_req_o = ${quadrant_s1_ctrl_xbars["soc_to_quad"].out_out.req_name()};
  assign ${quadrant_s1_ctrl_xbars["soc_to_quad"].out_out.rsp_name()} = quadrant_out_rsp_i;
  assign ${quadrant_s1_ctrl_xbars["quad_to_soc"].in_in.req_name()} = quadrant_in_req_i;
  assign quadrant_in_rsp_o = ${quadrant_s1_ctrl_xbars["quad_to_soc"].in_in.rsp_name()};

  // Convert both internal ports to AXI lite, since only registers for now
  <%
    quad_internal_lite = quadrant_s1_ctrl_xbars["soc_to_quad"].out_internal \
      .serialize(context, iw=1, name="soc_to_quad_internal_ser") \
      .change_dw(context, 32, "axi_to_axi_lite_dw") \
      .to_axi_lite(context, name="quad_to_soc_internal_ser", to=quadrant_s1_ctrl_mux.in_soc)
    soc_internal_lite = quadrant_s1_ctrl_xbars["quad_to_soc"].out_internal \
      .serialize(context, iw=1, name="soc_internal_serialize") \
      .change_dw(context, 32, "soc_internal_change_dw") \
      .to_axi_lite(context, name="soc_internal_to_axi_lite", to=quadrant_s1_ctrl_mux.in_quad)
    quad_regs_regbus = quadrant_s1_ctrl_mux.out_out \
      .to_reg(context, "axi_lite_to_regbus_regs")
  %> \

  // Control registers
  ${name}_quadrant_s1_reg2hw_t reg2hw;
  ${name}_quadrant_s1_hw2reg_t hw2reg;

  ${name}_quadrant_s1_reg_top #(
    .reg_req_t (${quad_regs_regbus.req_type()}),
    .reg_rsp_t (${quad_regs_regbus.rsp_type()})
  ) i_${name}_quadrant_s1_reg_top (
    .clk_i,
    .rst_ni,
    .reg_req_i (${quad_regs_regbus.req_name()}),
    .reg_rsp_o (${quad_regs_regbus.rsp_name()}),
    .reg2hw,
    .hw2reg,
    .devmode_i (1'b1)
  );

  // Control quadrant control signals
  assign isolate_o = reg2hw.isolate;
  assign hw2reg.isolated = isolated_i;
  assign ro_enable_o = reg2hw.ro_cache_enable.q;

  // RO cache flush handshake
  assign ro_flush_valid_o = reg2hw.ro_cache_flush.q;
  assign hw2reg.ro_cache_flush.d = ro_flush_ready_i;
  assign hw2reg.ro_cache_flush.de = reg2hw.ro_cache_flush.q & hw2reg.ro_cache_flush.d;

  // Assemble RO cache start and end addresses from registers
  % for j in range(ro_cache_regions):
  assign ro_start_addr_o[${j}] = {reg2hw.ro_start_addr_high_${j}.q, reg2hw.ro_start_addr_low_${j}.q};
  assign ro_end_addr_o  [${j}] = {reg2hw.ro_end_addr_high_${j}.q,   reg2hw.ro_end_addr_low_${j}.q};
  % endfor

  %if narrow_tlb_cfg:
  // Narrow TLB enable
  assign narrow_tlb_enable_o = reg2hw.tlb_narrow_enable.q;

  // Assemble narrow TLB entries
  % for j in range(narrow_tlb_entries):
  assign narrow_tlb_entries_o[${j}] = '{
    first:  {reg2hw.tlb_narrow_entry_${j}_pagein_first_high.q,  reg2hw.tlb_narrow_entry_${j}_pagein_first_low.q},
    last:   {reg2hw.tlb_narrow_entry_${j}_pagein_last_high.q,   reg2hw.tlb_narrow_entry_${j}_pagein_last_low.q},
    base:   {reg2hw.tlb_narrow_entry_${j}_pageout_high.q,       reg2hw.tlb_narrow_entry_${j}_pageout_low.q},
    valid:  reg2hw.tlb_narrow_entry_${j}_flags.valid.q,
    read_only: reg2hw.tlb_narrow_entry_${j}_flags.read_only.q
  };
  % endfor
  % endif

  %if wide_tlb_cfg:
  // Wide TLB enable
  assign wide_tlb_enable_o = reg2hw.tlb_wide_enable.q;

  // Assemble wide TLB entries
  % for j in range(wide_tlb_entries):
  assign wide_tlb_entries_o[${j}] = '{
    first:  {reg2hw.tlb_wide_entry_${j}_pagein_first_high.q,  reg2hw.tlb_wide_entry_${j}_pagein_first_low.q},
    last:   {reg2hw.tlb_wide_entry_${j}_pagein_last_high.q,   reg2hw.tlb_wide_entry_${j}_pagein_last_low.q},
    base:   {reg2hw.tlb_wide_entry_${j}_pageout_high.q,       reg2hw.tlb_wide_entry_${j}_pageout_low.q},
    valid:  reg2hw.tlb_wide_entry_${j}_flags.valid.q,
    read_only: reg2hw.tlb_wide_entry_${j}_flags.read_only.q
  };
  % endfor
  % endif

  // Quadrant clock gate controlled by register
  % for cluster_idx in range(num_clusters):
  tc_clk_gating i_tc_clk_gating_quadrant_cluster_${cluster_idx} (
    .clk_i,
    .en_i (reg2hw.clk_ena.ena_cluster_${cluster_idx}.q),
    .test_en_i (test_mode_i),
    .clk_o (clk_quadrant_cluster_o[${cluster_idx}])
  );
  % endfor

  tc_clk_gating i_tc_clk_gating_quadrant_cluster_uncore (
    .clk_i,
    .en_i (reg2hw.clk_ena.en_quad_uncore.q),
    .test_en_i (test_mode_i),
    .clk_o (clk_quadrant_uncore_o)
  );
  

  // Reset directly from register (i.e. (de)assertion inherently synchronized)
  // Multiplex with glitchless multiplexor, top reset for testing purposes
`ifdef TARGET_XILINX
  // Using clk cells makes Vivado flag the reset as a clock tree
  assign rst_quadrant_no = (test_mode_i) ? rst_ni : reg2hw.reset_n.q;
`else
  tc_clk_mux2 i_tc_reset_mux (
    .clk0_i (reg2hw.reset_n.q),
    .clk1_i (rst_ni),
    .clk_sel_i (test_mode_i),
    .clk_o (rst_quadrant_no)
  );
`endif

endmodule
