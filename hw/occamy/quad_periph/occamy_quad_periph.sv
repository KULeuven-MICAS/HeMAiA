// Author: Fanchen Kong <fanchen.kong@kuleuven.be>
`include "common_cells/registers.svh"

module occamy_quad_periph import occamy_quad_periph_reg_pkg::*; #(
  parameter type reg_req_t = logic,
  parameter type reg_rsp_t = logic
) (
  input clk_i,
  input rst_ni,

  // Below Register interface can be changed
  input  reg_req_t reg_req_i,
  output reg_rsp_t reg_rsp_o,
  // To HW
  output logic [47:0] bingo_hw_manager_task_list_base_addr_o,
  output logic [31:0] bingo_hw_manager_num_task_o,
  output logic [31:0] bingo_hw_manager_start_o,
  input  logic [31:0] bingo_hw_manager_reset_start_i,
  input  logic        bingo_hw_manager_reset_start_en_i
);

  occamy_quad_periph_hw2reg_t hw2reg;
  occamy_quad_periph_reg2hw_t reg2hw;
  assign bingo_hw_manager_task_list_base_addr_o[31:0] = reg2hw.task_desc_list_base_addr_lo.q;
  assign bingo_hw_manager_task_list_base_addr_o[47:32] = reg2hw.task_desc_list_base_addr_hi.q[15:0];
  assign bingo_hw_manager_num_task_o = reg2hw.num_task.q;
  assign bingo_hw_manager_start_o = reg2hw.start_bingo_hw_manager.q;
  assign hw2reg.start_bingo_hw_manager.d  = bingo_hw_manager_reset_start_i;
  assign hw2reg.start_bingo_hw_manager.de = bingo_hw_manager_reset_start_en_i;
  // Assignment not ready yet
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