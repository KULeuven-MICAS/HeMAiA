// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Fanchen Kong <fanchen.kong@kuleuven.be>
// This mailbox module is mainly adopted from the axi_lite_mailbox.sv from 
// the PULP AXI IP repository.
// Different from the pulp version, this mailbox is not a dual-mailbox, but a single
// mailbox. 
// There are three types of mailbox in the Hemaia platform:
// 1. The host2host mailbox, which is resied in the 64bit soc axi lite periph xbar.
//    - There is only one host2host mailbox in one chiplet.
//    - This mailbox is used for communication between two host in different chiplets.
// 2. The cluster2host mailbox, which is resied in the 32bit soc axi lite narrow periph xbar.
//    - There is only one cluster2host mailbox in one chiplet.
//    - This mailbox is used for communication from clusters to the host in the same chiplet
//    - The field of this mailbox is:
//      * return value (4bit)
//      * chip id      (8bit)
//      * cluster id   (6bit)
//      * task id      (12bit)
//      * reserved     (2bit)
//    - The reason why we use a 32bit mailbox and pack all the information in one 32bit word is to
//      reduce the complexity of the software driver. The software driver only needs to read one
//      32bit word to get all the information. Beside, multiple clusters can send message to the
//      cluster2host mailbox, so it is better to ensure the message is atomic.
// 3. The host2cluster mailbox, which is reside in the 32bit quad ctrl xbar.
//    - Each cluster has its own host2cluster mailbox
`include "common_cells/registers.svh"
module hemaia_hw_mailbox #(
    parameter int unsigned MailboxDepth = 32'd0,
    parameter bit unsigned IrqEdgeTrig  = 1'b0,
    parameter bit unsigned IrqActHigh   = 1'b1,
    parameter int unsigned AxiAddrWidth = 32'd0,
    parameter int unsigned AxiDataWidth = 32'd0,
    parameter int unsigned ChipIdWidth  = 8,
    parameter type         req_lite_t   = logic,
    parameter type         resp_lite_t  = logic,
    // DEPENDENT PARAMETERS, DO NOT OVERRIDE!
    parameter type         addr_t       = logic [AxiAddrWidth-1:0],
    parameter type         data_t       = logic [AxiDataWidth-1:0],
    // usage type of the mailbox FIFO, also the type of the threshold comparison
    // is one bit wider, MSB is the fifo_full flag of the respective fifo
    parameter type         usage_t      = logic [$clog2(MailboxDepth):0]
) (
    input  logic                   clk_i,       // Clock
    input  logic                   rst_ni,      // Asynchronous reset active low
    input  logic                   test_i,      // Testmode enable
    input  logic [ChipIdWidth-1:0] chip_id_i,   // chip id input for multi-chip addressing
    input  req_lite_t              req_i,       // AXI-Lite request input
    output resp_lite_t             resp_o,      // AXI-Lite response output
    output logic                   irq_o,       // Interrupt output
    input  addr_t                  base_addr_i  // base address for this peripheral
);
    // FIFO signals
    logic w_mbox_flush, r_mbox_flush;
    logic mbox_push, mbox_pop;
    logic mbox_full, mbox_empty;
    logic [$clog2(MailboxDepth)-1:0] mbox_usage;
    data_t  mbox_w_data, mbox_r_data;
    usage_t mbox_usage_combined;
    // interrupt request from this slave port, level triggered, active high --> convert
    logic    slv_irq;
    logic    clear_irq;

    hemaia_axi_lite_mailbox_adapter #(
        .MailboxDepth ( MailboxDepth ),
        .AxiAddrWidth ( AxiAddrWidth ),
        .AxiDataWidth ( AxiDataWidth ),
        .req_lite_t   ( req_lite_t   ),
        .resp_lite_t  ( resp_lite_t  ),
        .addr_t       ( addr_t       ),
        .data_t       ( data_t       ),
        .usage_t      ( usage_t      )  // fill pointer from MBOX FIFO
    ) i_hemaia_axi_lite_mailbox_adapter (
        .clk_i,   // Clock
        .rst_ni,  // Asynchronous reset active low
        .chip_id_i      (chip_id_i     ), // chip id input for multi-chip addressing
        // slave port
        .slv_req_i      ( req_i        ),
        .slv_resp_o     ( resp_o       ),
        .base_addr_i    ( base_addr_i  ), // base address for the slave port
        // write FIFO port
        .mbox_w_data_o  ( mbox_w_data  ),
        .mbox_w_full_i  ( mbox_full    ),
        .mbox_w_push_o  ( mbox_push    ),
        .mbox_w_flush_o ( w_mbox_flush ),
        .mbox_w_usage_i ( mbox_usage_combined ),
        // read FIFO port
        .mbox_r_data_i  ( mbox_r_data  ),
        .mbox_r_empty_i ( mbox_empty   ),
        .mbox_r_pop_o   ( mbox_pop     ),
        .mbox_r_flush_o ( r_mbox_flush ),
        .mbox_r_usage_i ( mbox_usage_combined   ),
        // interrupt output, level triggered, active high, conversion in top
        .irq_o          ( slv_irq      ),
        .clear_irq_o    ( clear_irq    )
    );
    fifo_v3 #(
        .FALL_THROUGH ( 1'b0         ),
        .DEPTH        ( MailboxDepth ),
        .dtype        ( data_t       )
    ) i_mbox (
        .clk_i       ( clk_i       ),
        .rst_ni      ( rst_ni      ),
        .testmode_i  ( test_i      ),
        .flush_i     ( w_mbox_flush | r_mbox_flush  ),
        .full_o      ( mbox_full   ),
        .empty_o     ( mbox_empty  ),
        .usage_o     ( mbox_usage  ),
        .data_i      ( mbox_w_data ),
        .push_i      ( mbox_push   ),
        .data_o      ( mbox_r_data ),
        .pop_i       ( mbox_pop    )
    );
    // usage combined signal, MSB is the full flag
    assign mbox_usage_combined = {mbox_full, mbox_usage};
    // interrupt signal conversion
    if (IrqEdgeTrig) begin : gen_irq_edge
      logic irq_q, irq_d, update_irq;

      always_comb begin
        // default assignments
        irq_d      = irq_q;
        update_irq = 1'b0;
        // init the irq and pulse only on update
        irq_o   = ~IrqActHigh;
        if (clear_irq) begin
          irq_d      = 1'b0;
          update_irq = 1'b1;
        end else if (!irq_q && slv_irq) begin
          irq_d      = 1'b1;
          update_irq = 1'b1;
          irq_o   = IrqActHigh; // on update of the register pulse the irq signal
        end
      end

      `FFLARN(irq_q, irq_d, update_irq, '0, clk_i, rst_ni)
    end else begin : gen_irq_level
      assign irq_o = (IrqActHigh) ? slv_irq : ~slv_irq;
    end
endmodule