`include "common_cells/assertions.svh"
`include "register_interface/typedef.svh"

module hemaia_clk_rst_controller #(
    parameter int NumClocks = 4,
    parameter int MaxDivisionWidth = 8,  // Maximum width for clock division
    parameter int DefaultDivision[NumClocks] = '{default: 1},
    parameter int ResetDelays[NumClocks] = '{default: 1},
    parameter type axi_lite_req_t = logic,
    parameter type axi_lite_rsp_t = logic
) (
    // Test mode
    input logic test_mode_i,
    // The clock and rst for the controller itself
    input logic control_clk_i,
    (* false_path *) input logic control_rst_ni,
    // Data bus for controller
    input axi_lite_req_t axi_lite_req_i,
    output axi_lite_rsp_t axi_lite_rsp_o,

    // Input / Output clk, reset
    input logic mst_clk_i,
    input logic mst_rst_ni,
    input logic bypass_pll_division_i,
    output logic clk_obs_o,
    output logic [NumClocks-1:0] clk_o,
    output logic [NumClocks-1:0] rst_no
);

  //////////////////////////////////////
  //    PLL (Not implemented yet)     //
  //////////////////////////////////////
  logic mst_clk_after_pll;
  assign mst_clk_after_pll = mst_clk_i;


  ///////////////////////////////
  //    Reset Synchronizer     //
  ///////////////////////////////
  logic mst_rst_n_d1_mst_clk, mst_rst_n_d2_mst_clk;
  always_ff @(posedge mst_clk_after_pll or negedge mst_rst_ni) begin
    if (~mst_rst_ni) begin
      mst_rst_n_d1_mst_clk <= 1'b0;
      mst_rst_n_d2_mst_clk <= 1'b0;
    end else begin
      mst_rst_n_d1_mst_clk <= mst_rst_ni;
      mst_rst_n_d2_mst_clk <= mst_rst_n_d1_mst_clk;
    end
  end

  ///////////////////////
  //    CONTROLLER     //
  ///////////////////////

  import hemaia_clk_rst_controller_reg_pkg::*;
  `REG_BUS_TYPEDEF_ALL(reg_a48_d32, logic [47:0], logic [31:0], logic [3:0])

  reg_a48_d32_req_t controller_req;
  reg_a48_d32_rsp_t controller_rsp;

  axi_lite_to_reg #(
      .ADDR_WIDTH    (48),
      .DATA_WIDTH    (32),
      .axi_lite_req_t(axi_lite_req_t),
      .axi_lite_rsp_t(axi_lite_rsp_t),
      .reg_req_t     (reg_a48_d32_req_t),
      .reg_rsp_t     (reg_a48_d32_rsp_t)
  ) i_controller_axi_to_reg (
      .clk_i         (control_clk_i),
      .rst_ni        (control_rst_ni),
      .axi_lite_req_i(axi_lite_req_i),
      .axi_lite_rsp_o(axi_lite_rsp_o),
      .reg_req_o     (controller_req),
      .reg_rsp_i     (controller_rsp)
  );

  hemaia_clk_rst_controller_reg2hw_t reg2hw;
  hemaia_clk_rst_controller_hw2reg_t hw2reg;

  hemaia_clk_rst_controller_reg_top #(
      .reg_req_t(reg_a48_d32_req_t),
      .reg_rsp_t(reg_a48_d32_rsp_t)
  ) i_controller_reg_to_hw (
      .clk_i    (control_clk_i),
      .rst_ni   (control_rst_ni),
      .reg_req_i(controller_req),
      .reg_rsp_o(controller_rsp),
      .reg2hw   (reg2hw),
      .hw2reg   (hw2reg),
      .devmode_i('0)
  );

  ///////////////////////////////////////////////////
  //    Logic to reset clock division valid reg    //
  ///////////////////////////////////////////////////
  // Generate the spike for the sync bits
  assign hw2reg.clock_valid_register.valid_c0.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c1.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c2.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c3.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c4.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c5.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c6.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c7.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c8.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c9.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c10.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c11.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c12.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c13.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c14.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c15.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c16.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c17.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c18.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c19.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c20.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c21.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c22.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c23.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c24.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c25.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c26.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c27.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c28.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c29.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c30.d = 1'b0;
  assign hw2reg.clock_valid_register.valid_c31.d = 1'b0;

  assign hw2reg.clock_valid_register.valid_c0.de = reg2hw.clock_valid_register.valid_c0;
  assign hw2reg.clock_valid_register.valid_c1.de = reg2hw.clock_valid_register.valid_c1;
  assign hw2reg.clock_valid_register.valid_c2.de = reg2hw.clock_valid_register.valid_c2;
  assign hw2reg.clock_valid_register.valid_c3.de = reg2hw.clock_valid_register.valid_c3;
  assign hw2reg.clock_valid_register.valid_c4.de = reg2hw.clock_valid_register.valid_c4;
  assign hw2reg.clock_valid_register.valid_c5.de = reg2hw.clock_valid_register.valid_c5;
  assign hw2reg.clock_valid_register.valid_c6.de = reg2hw.clock_valid_register.valid_c6;
  assign hw2reg.clock_valid_register.valid_c7.de = reg2hw.clock_valid_register.valid_c7;
  assign hw2reg.clock_valid_register.valid_c8.de = reg2hw.clock_valid_register.valid_c8;
  assign hw2reg.clock_valid_register.valid_c9.de = reg2hw.clock_valid_register.valid_c9;
  assign hw2reg.clock_valid_register.valid_c10.de = reg2hw.clock_valid_register.valid_c10;
  assign hw2reg.clock_valid_register.valid_c11.de = reg2hw.clock_valid_register.valid_c11;
  assign hw2reg.clock_valid_register.valid_c12.de = reg2hw.clock_valid_register.valid_c12;
  assign hw2reg.clock_valid_register.valid_c13.de = reg2hw.clock_valid_register.valid_c13;
  assign hw2reg.clock_valid_register.valid_c14.de = reg2hw.clock_valid_register.valid_c14;
  assign hw2reg.clock_valid_register.valid_c15.de = reg2hw.clock_valid_register.valid_c15;
  assign hw2reg.clock_valid_register.valid_c16.de = reg2hw.clock_valid_register.valid_c16;
  assign hw2reg.clock_valid_register.valid_c17.de = reg2hw.clock_valid_register.valid_c17;
  assign hw2reg.clock_valid_register.valid_c18.de = reg2hw.clock_valid_register.valid_c18;
  assign hw2reg.clock_valid_register.valid_c19.de = reg2hw.clock_valid_register.valid_c19;
  assign hw2reg.clock_valid_register.valid_c20.de = reg2hw.clock_valid_register.valid_c20;
  assign hw2reg.clock_valid_register.valid_c21.de = reg2hw.clock_valid_register.valid_c21;
  assign hw2reg.clock_valid_register.valid_c22.de = reg2hw.clock_valid_register.valid_c22;
  assign hw2reg.clock_valid_register.valid_c23.de = reg2hw.clock_valid_register.valid_c23;
  assign hw2reg.clock_valid_register.valid_c24.de = reg2hw.clock_valid_register.valid_c24;
  assign hw2reg.clock_valid_register.valid_c25.de = reg2hw.clock_valid_register.valid_c25;
  assign hw2reg.clock_valid_register.valid_c26.de = reg2hw.clock_valid_register.valid_c26;
  assign hw2reg.clock_valid_register.valid_c27.de = reg2hw.clock_valid_register.valid_c27;
  assign hw2reg.clock_valid_register.valid_c28.de = reg2hw.clock_valid_register.valid_c28;
  assign hw2reg.clock_valid_register.valid_c29.de = reg2hw.clock_valid_register.valid_c29;
  assign hw2reg.clock_valid_register.valid_c30.de = reg2hw.clock_valid_register.valid_c30;
  assign hw2reg.clock_valid_register.valid_c31.de = reg2hw.clock_valid_register.valid_c31;

  ////////////////////////////////////////////////////////////////////////////////
  // Generate the “spike” and data‐enable bits for the reset_register registers //
  ////////////////////////////////////////////////////////////////////////////////

  assign hw2reg.reset_register.reset_c0.d = 1'b0;
  assign hw2reg.reset_register.reset_c1.d = 1'b0;
  assign hw2reg.reset_register.reset_c2.d = 1'b0;
  assign hw2reg.reset_register.reset_c3.d = 1'b0;
  assign hw2reg.reset_register.reset_c4.d = 1'b0;
  assign hw2reg.reset_register.reset_c5.d = 1'b0;
  assign hw2reg.reset_register.reset_c6.d = 1'b0;
  assign hw2reg.reset_register.reset_c7.d = 1'b0;
  assign hw2reg.reset_register.reset_c8.d = 1'b0;
  assign hw2reg.reset_register.reset_c9.d = 1'b0;
  assign hw2reg.reset_register.reset_c10.d = 1'b0;
  assign hw2reg.reset_register.reset_c11.d = 1'b0;
  assign hw2reg.reset_register.reset_c12.d = 1'b0;
  assign hw2reg.reset_register.reset_c13.d = 1'b0;
  assign hw2reg.reset_register.reset_c14.d = 1'b0;
  assign hw2reg.reset_register.reset_c15.d = 1'b0;
  assign hw2reg.reset_register.reset_c16.d = 1'b0;
  assign hw2reg.reset_register.reset_c17.d = 1'b0;
  assign hw2reg.reset_register.reset_c18.d = 1'b0;
  assign hw2reg.reset_register.reset_c19.d = 1'b0;
  assign hw2reg.reset_register.reset_c20.d = 1'b0;
  assign hw2reg.reset_register.reset_c21.d = 1'b0;
  assign hw2reg.reset_register.reset_c22.d = 1'b0;
  assign hw2reg.reset_register.reset_c23.d = 1'b0;
  assign hw2reg.reset_register.reset_c24.d = 1'b0;
  assign hw2reg.reset_register.reset_c25.d = 1'b0;
  assign hw2reg.reset_register.reset_c26.d = 1'b0;
  assign hw2reg.reset_register.reset_c27.d = 1'b0;
  assign hw2reg.reset_register.reset_c28.d = 1'b0;
  assign hw2reg.reset_register.reset_c29.d = 1'b0;
  assign hw2reg.reset_register.reset_c30.d = 1'b0;
  assign hw2reg.reset_register.reset_c31.d = 1'b0;

  assign hw2reg.reset_register.reset_c0.de = reg2hw.reset_register.reset_c0;
  assign hw2reg.reset_register.reset_c1.de = reg2hw.reset_register.reset_c1;
  assign hw2reg.reset_register.reset_c2.de = reg2hw.reset_register.reset_c2;
  assign hw2reg.reset_register.reset_c3.de = reg2hw.reset_register.reset_c3;
  assign hw2reg.reset_register.reset_c4.de = reg2hw.reset_register.reset_c4;
  assign hw2reg.reset_register.reset_c5.de = reg2hw.reset_register.reset_c5;
  assign hw2reg.reset_register.reset_c6.de = reg2hw.reset_register.reset_c6;
  assign hw2reg.reset_register.reset_c7.de = reg2hw.reset_register.reset_c7;
  assign hw2reg.reset_register.reset_c8.de = reg2hw.reset_register.reset_c8;
  assign hw2reg.reset_register.reset_c9.de = reg2hw.reset_register.reset_c9;
  assign hw2reg.reset_register.reset_c10.de = reg2hw.reset_register.reset_c10;
  assign hw2reg.reset_register.reset_c11.de = reg2hw.reset_register.reset_c11;
  assign hw2reg.reset_register.reset_c12.de = reg2hw.reset_register.reset_c12;
  assign hw2reg.reset_register.reset_c13.de = reg2hw.reset_register.reset_c13;
  assign hw2reg.reset_register.reset_c14.de = reg2hw.reset_register.reset_c14;
  assign hw2reg.reset_register.reset_c15.de = reg2hw.reset_register.reset_c15;
  assign hw2reg.reset_register.reset_c16.de = reg2hw.reset_register.reset_c16;
  assign hw2reg.reset_register.reset_c17.de = reg2hw.reset_register.reset_c17;
  assign hw2reg.reset_register.reset_c18.de = reg2hw.reset_register.reset_c18;
  assign hw2reg.reset_register.reset_c19.de = reg2hw.reset_register.reset_c19;
  assign hw2reg.reset_register.reset_c20.de = reg2hw.reset_register.reset_c20;
  assign hw2reg.reset_register.reset_c21.de = reg2hw.reset_register.reset_c21;
  assign hw2reg.reset_register.reset_c22.de = reg2hw.reset_register.reset_c22;
  assign hw2reg.reset_register.reset_c23.de = reg2hw.reset_register.reset_c23;
  assign hw2reg.reset_register.reset_c24.de = reg2hw.reset_register.reset_c24;
  assign hw2reg.reset_register.reset_c25.de = reg2hw.reset_register.reset_c25;
  assign hw2reg.reset_register.reset_c26.de = reg2hw.reset_register.reset_c26;
  assign hw2reg.reset_register.reset_c27.de = reg2hw.reset_register.reset_c27;
  assign hw2reg.reset_register.reset_c28.de = reg2hw.reset_register.reset_c28;
  assign hw2reg.reset_register.reset_c29.de = reg2hw.reset_register.reset_c29;
  assign hw2reg.reset_register.reset_c30.de = reg2hw.reset_register.reset_c30;
  assign hw2reg.reset_register.reset_c31.de = reg2hw.reset_register.reset_c31;

  // Synchronize valid bits into high frequencies
  (* async *)logic [31:0] clock_division_reg_valid_d1;
  logic [31:0] clock_division_reg_valid_d2;
  