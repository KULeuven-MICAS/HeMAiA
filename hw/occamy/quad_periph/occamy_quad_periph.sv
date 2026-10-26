// Author: Fanchen Kong <fanchen.kong@kuleuven.be>
`include "common_cells/registers.svh"

module occamy_quad_periph import occamy_quad_periph_reg_pkg::*; #(
  parameter BINGO_HW_MANAGER_NR_CORE_PER_CLUSTER = 4,
  parameter BINGO_HW_MANAGER_NR_CLUSTER = 1,
  // 32-bit words of the HW manager's CERF; must match the register file's NumCerfWords
  parameter int unsigned CERF_WORDS = 1,
  parameter REG_WIDTH = 32,
  parameter type reg_req_t = logic,
  parameter type reg_rsp_t = logic,
  parameter type reg_data_t = logic [REG_WIDTH-1:0]
) (
  input clk_i,
  input rst_ni,

  // Below Register interface can be changed
  input  reg_req_t reg_req_i,
  output reg_rsp_t reg_rsp_o,
  // To HW
  // Task Description Fetch
  output logic [47:0] bingo_hw_manager_task_list_base_addr_o,
  output reg_data_t bingo_hw_manager_num_task_o,
  output reg_data_t bingo_hw_manager_start_o,
  input  reg_data_t bingo_hw_manager_reset_start_i,
  input  logic        bingo_hw_manager_reset_start_en_i,
  // Power Management
  output logic [47:0] bingo_hw_manager_pm_base_addr_o,
  output reg_data_t bingo_hw_manager_enable_idle_pm_o,
  output reg_data_t bingo_hw_manager_idle_power_level_o,
  output reg_data_t bingo_hw_manager_norm_power_level_o,
  output reg_data_t [BINGO_HW_MANAGER_NR_CORE_PER_CLUSTER-1:0][BINGO_HW_MANAGER_NR_CLUSTER-1:0] bingo_hw_manager_core_power_domain_o,
  // DARTS CERF: a masked write, state = (state & ~mask) | (data & mask) when en is high
  output logic                       cerf_write_en_o,
  output logic [CERF_WORDS*32-1:0]   cerf_write_mask_o,
  output logic [CERF_WORDS*32-1:0]   cerf_write_data_o,
  input  logic [CERF_WORDS*32-1:0]   cerf_state_i,
  // DVFS: mode select, CLINT doorbell address, host ack, published request
  output reg_data_t   bingo_hw_manager_pm_mode_o,
  output logic [47:0] bingo_hw_manager_dvfs_clint_msip_addr_o,
  output reg_data_t   bingo_hw_manager_dvfs_ack_o,
  input  reg_data_t   bingo_hw_manager_dvfs_request_i
);

  occamy_quad_periph_hw2reg_t hw2reg;
  occamy_quad_periph_reg2hw_t reg2hw;
  // Task Description
  assign bingo_hw_manager_task_list_base_addr_o[31:0] = reg2hw.task_desc_list_base_addr_lo.q;
  assign bingo_hw_manager_task_list_base_addr_o[47:32] = reg2hw.task_desc_list_base_addr_hi.q[15:0];
  assign bingo_hw_manager_num_task_o = reg2hw.num_task.q;
  assign bingo_hw_manager_start_o = reg2hw.start_bingo_hw_manager.q;
  assign hw2reg.start_bingo_hw_manager.d  = bingo_hw_manager_reset_start_i;
  assign hw2reg.start_bingo_hw_manager.de = bingo_hw_manager_reset_start_en_i;
  // Power Management
  assign bingo_hw_manager_pm_base_addr_o[31:0] = reg2hw.pm_base_addr_lo.q;
  assign bingo_hw_manager_pm_base_addr_o[47:32] = reg2hw.pm_base_addr_hi.q[15:0];
  assign bingo_hw_manager_enable_idle_pm_o = reg2hw.en_idle_pm.q;
  assign bingo_hw_manager_idle_power_level_o = reg2hw.idle_power_level.q;
  assign bingo_hw_manager_norm_power_level_o = reg2hw.norm_power_level.q;
  always_comb begin
    for (int core = 0; core < BINGO_HW_MANAGER_NR_CORE_PER_CLUSTER; core++) begin
      for (int cluster = 0; cluster < BINGO_HW_MANAGER_NR_CLUSTER; cluster++) begin
        bingo_hw_manager_core_power_domain_o[core][cluster] = reg2hw.core_power_domain[core + cluster*BINGO_HW_MANAGER_NR_CORE_PER_CLUSTER].q;
      end
    end
  end
  // DARTS CERF. Three ways to write it, merged into one masked write:
  // - CERF_WRITE_EN latches every CERF_STATE word at once (mask all ones);
  // - a CERF_SET word sets the groups whose bits are 1 in it, a CERF_CLR word clears them,
  //   each in the cycle of its own store, leaving every other group as it is.
  // An EN latch and a SET/CLR store in the same cycle apply in that order.
  always_comb begin
    cerf_write_mask_o = '0;
    cerf_write_data_o = '0;
    if (reg2hw.cerf_write_en.q) begin
      cerf_write_mask_o = '1;
      for (int w = 0; w < CERF_WORDS; w++) cerf_write_data_o[w*32 +: 32] = reg2hw.cerf_state[w].q;
    end
    for (int w = 0; w < CERF_WORDS; w++) begin
      if (reg2hw.cerf_set[w].qe) begin
        cerf_write_mask_o[w*32 +: 32] |= reg2hw.cerf_set[w].q;
        cerf_write_data_o[w*32 +: 32] |= reg2hw.cerf_set[w].q;
      end
      if (reg2hw.cerf_clr[w].qe) begin
        cerf_write_mask_o[w*32 +: 32] |= reg2hw.cerf_clr[w].q;
        cerf_write_data_o[w*32 +: 32] &= ~reg2hw.cerf_clr[w].q;
      end
    end
  end
  assign cerf_write_en_o = |cerf_write_mask_o;
  // Auto-clear write_en: HW writes 0 back after latch takes effect
  assign hw2reg.cerf_write_en.d  = '0;
  assign hw2reg.cerf_write_en.de = reg2hw.cerf_write_en.q;
  // CERF_STATUS: HW writes actual controller state for SW read-back
  for (genvar w = 0; w < CERF_WORDS; w++) begin : gen_cerf_status
    assign hw2reg.cerf_status[w].d  = cerf_state_i[w*32 +: 32];
    assign hw2reg.cerf_status[w].de = 1'b1;  // always update
  end
  // DVFS: SW->HW config
  assign bingo_hw_manager_pm_mode_o = reg2hw.pm_mode.q;
  assign bingo_hw_manager_dvfs_clint_msip_addr_o[31:0]  = reg2hw.dvfs_clint_msip_addr_lo.q;
  assign bingo_hw_manager_dvfs_clint_msip_addr_o[47:32] = reg2hw.dvfs_clint_msip_addr_hi.q[15:0];
  assign bingo_hw_manager_dvfs_ack_o = reg2hw.dvfs_ack.q;
  // DVFS_REQUEST: HW publishes the request for SW read-back (always update)
  assign hw2reg.dvfs_request.d  = bingo_hw_manager_dvfs_request_i;
  assign hw2reg.dvfs_request.de = 1'b1;
  occamy_quad_periph_reg_top #(
    .reg_req_t ( reg_req_t ),
    .reg_rsp_t ( reg_rsp_t  )
  ) i_quad_periph (

    .clk_i           ( clk_i           ),
    .rst_ni          ( rst_ni          ),

    .reg_req_i       ( reg_req_i       ),
    .reg_rsp_o       ( reg_rsp_o       ),

    .reg2hw          ( reg2hw          ),
    .hw2reg          ( hw2reg          ),
    .devmode_i       ( 1'b0            )
  );
endmodule