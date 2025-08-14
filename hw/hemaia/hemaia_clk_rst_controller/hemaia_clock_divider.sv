// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yunhao Deng <yunhao.deng@kuleuven.be>

module hemaia_clock_counter #(
    parameter int Width = 8
) (
    input  logic             clk_i,
    input  logic             rst_ni,       // active-low async reset
    input  logic             tick_i,
    input  logic             clear_i,      // active-high sync clear
    input  logic [Width-1:0] ceiling_i,
    output logic [Width-1:0] count_o,
    output logic             last_value_o
);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      // Asynchronous reset to 0
      count_o <= '0;
    end else if (clear_i) begin
      // Synchronous "clear" input
      count_o <= '0;
    end else if (tick_i) begin
      // Only update on tick
      // If ceiling_i is 0, count_o should not increment
      // Compare against (ceiling_i - 1)
      if (count_o < (ceiling_i - 1'b1) && ceiling_i > 1'b0) count_o <= count_o + 1'b1;
      else count_o <= '0;
    end
  end

  assign last_value_o = (count_o == (ceiling_i - 1'b1)) && tick_i;

endmodule


(* no_ungroup *) (* no_boundary_optimization *) (* keep_hierarchy *)
module hemaia_clock_divider #(
    parameter int MaxDivisionWidth = 4,
    parameter int DefaultDivision  = 1
) (
    input  logic                        clk_i,
    input  logic                        rst_ni,
    input  logic [MaxDivisionWidth-1:0] divisor_i,
    input  logic                        divisor_valid_i,
    input  logic                        test_en_i,
    output logic                        clk_o
);

  logic [MaxDivisionWidth-1:0] divisor_q;
  logic new_divisor_ready;


  // Counters for edges and local flip-flops
  logic [MaxDivisionWidth-1:0] cnt;
  logic last_val;

  hemaia_clock_counter #(
      .Width(MaxDivisionWidth)
  ) pos_counter (
      .clk_i(clk_i),
      .rst_ni(rst_ni),
      .tick_i(1'b1),
      .clear_i(1'b0),
      .ceiling_i(divisor_q),
      .count_o(cnt),
      .last_value_o(last_val)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      new_divisor_ready <= '0;
      divisor_q <= DefaultDivision[MaxDivisionWidth-1:0];
    end else if (new_divisor_ready && (cnt == 0)) begin
      new_divisor_ready <= '0;
      divisor_q <= divisor_i;
    end else if (divisor_valid_i) begin
      new_divisor_ready <= 1'b1;
    end
  end

  (* syn_keep = 1, syn_preserve = 1 *) logic raw_div, raw_div_d1, raw_div_d2;
  always_comb begin
    if (cnt < (divisor_q >> 1)) begin
      raw_div = 1'b0;  // high for half the period
    end else begin
      raw_div = 1'b1;  // low for the other half
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      raw_div_d1 <= 1'b1;
    end else begin
      raw_div_d1 <= raw_div;
    end
  end

  always_ff @(negedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      raw_div_d2 <= 1'b1;
    end else begin
      raw_div_d2 <= raw_div_d1;
    end
  end

  logic clk_ungated, clk_divided, clk_odd, clk_even;
  assign clk_odd  /* synthesis keep */  = raw_div_d1 & raw_div_d2;
  assign clk_even  /* synthesis keep */ = raw_div_d1;

  tc_clk_mux2 i_clk_divided_mux (
      .clk0_i(clk_even),
      .clk1_i(clk_odd),
      .clk_sel_i(divisor_q[0]),
      .clk_o(clk_divided)
  );

  tc_clk_mux2 i_clk_o_mux (
      .clk0_i(clk_divided),
      .clk1_i(clk_i),
      .clk_sel_i(divisor_q == 'd1),
      .clk_o(clk_ungated)
  );

  tc_clk_gating i_clk_o_gate (
      .clk_i(clk_ungated),
      .en_i(divisor_q == '0),
      .test_en_i(test_en_i),
      .clk_o(clk_o)
  );
endmodule
