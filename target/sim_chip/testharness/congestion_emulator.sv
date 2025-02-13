// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>

module congestion_emulator #(
    parameter int CONGESTION_LEVEL = 100  // 100 means full congestion, 0 means no congestion
) (
    input  logic clk_i,
    input  logic valid_i,
    output logic ready_o,
    output logic valid_o,
    input  logic ready_i
);

  logic stall;

  // Generate random stall based on congestion level
  initial begin
    forever begin
      @(posedge clk_i);
      if ($urandom_range(99, 0) >= CONGESTION_LEVEL) begin
        stall = 1'b0;
        wait(valid_i && ready_i);
        // After the data is consumed, we can generate a new stall
      end else begin
        stall = 1'b1;
      end
    end
  end

  // Control ready and valid signals
  assign ready_o = ~stall & ready_i;
  assign valid_o = valid_i & ~stall;

endmodule
