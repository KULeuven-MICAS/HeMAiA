// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yunhao Deng <yunhao.deng@kuleuven.be>

(* no_ungroup *)
(* no_boundary_optimization *)
(* keep_hierarchy *)

module hemaia_reset_controller #(
    // Number of reset channels, each synchronized with its own clock domain
    parameter int NumReset = 4,
    // Delay for the reset signal after being synchronized with its local clock.
    // Default: 1 cycle for each channel, automatically sized to NumReset.
    parameter int ResetDelays[NumReset] = '{default: 1}
) (
    input logic [NumReset-1:0] clk_i,
    input logic async_global_rst_ni,
    input logic [NumReset-1:0] async_local_rst_ni,
    output logic [NumReset-1:0] sync_rst_no
);

  // ---------------------------------------------------------------------------
  // Generate one independent synchroniser + pipeline per channel
  // ---------------------------------------------------------------------------
  genvar i;
  for (i = 0; i < NumReset; i++) begin : g_rst

    //--------------------------------------------------------------------------
    // Two-flop synchroniser (both flops have async set-to-0)
    //---------------------------------------------------------------------------
    (* async *) logic sync_ff0, sync_ff1;
    (* false_path *) logic sync_ff_rst;
    assign sync_ff_rst = async_local_rst_ni[i] & async_global_rst_ni;

    always_ff @(posedge clk_i[i] or negedge sync_ff_rst) begin
      if (~sync_ff_rst) begin
        sync_ff0 <= 1'b0;
        sync_ff1 <= 1'b0;
      end else begin
        sync_ff0 <= 1'b1;
        sync_ff1 <= sync_ff0;
      end
    end

    //---------------------------------------------------------------------------
    //  More pipeline to make the timing easier to meet
    //---------------------------------------------------------------------------
    if (ResetDelays[i] == 0) begin : g_no_delay
      // No extra flops requested.
      assign sync_rst_no[i] = sync_ff1;
    end else begin : g_delay
      localparam int PipeStages = ResetDelays[i];
      logic [PipeStages-1:0] pipe_ff;

      always_ff @(posedge clk_i[i] or negedge sync_ff_rst) begin
        if (~sync_ff_rst) begin
          pipe_ff <= '0;
        end else begin
          pipe_ff[0] <= sync_ff1;
          for (int k = 1; k < PipeStages; k++) begin
            pipe_ff[k] <= pipe_ff[k-1];
          end
        end
      end
      assign sync_rst_no[i] = pipe_ff[PipeStages-1];
    end
  end : g_rst

endmodule
