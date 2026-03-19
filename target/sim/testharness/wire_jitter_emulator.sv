// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>

module wire_jitter_emulator #(
    parameter int MAX_JITTER_CYCLES = 2
) (
    input  logic clk_i,
    input  logic data_i,
    output logic data_o
);

  initial begin
    data_o = data_i;
    forever begin
      // Wait for input flip
      @(data_i);
      // Wait for random number of clock cycles (0 to MAX_JITTER_CYCLES)
      repeat ($urandom_range(MAX_JITTER_CYCLES, 0)) @(posedge clk_i);
      // Update output
      data_o <= data_i;
    end
  end

endmodule
