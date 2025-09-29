// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Fanchen Kong <fanchen.kong@kuleuven.be>
// This module is taken from the axi_lite_mailbox_slave.sv from the PULP AXI IP repository.
`include "common_cells/registers.svh"
`include "axi/typedef.svh"
// slave port module
module hemaia_axi_lite_mailbox_adapter #(
  parameter int unsigned MailboxDepth = 32'd16,
  parameter int unsigned AxiAddrWidth = 32'd32,
  parameter int unsigned AxiDataWidth = 32'd32,
  parameter int unsigned ChipIdWidth  = 8,
  parameter type         req_lite_t   = logic,
  parameter type         resp_lite_t  = logic,
  parameter type         addr_t       = logic [AxiAddrWidth-1:0],
  parameter type         data_t       = logic [AxiDataWidth-1:0],
  parameter type         usage_t      = logic                     // fill pointer from MBOX FIFO
) (
  input  logic       clk_i,   // Clock
  input  logic       rst_ni,  // Asynchronous reset active low
  input  logic [ChipIdWidth-1:0] chip_id_i, // chip id input for multi-chip addressing
  // slave port
  input  req_lite_t  slv_req_i,
  output resp_lite_t slv_resp_o,
  input  addr_t      base_addr_i, // base address for the slave port
  // write FIFO port
  output data_t      mbox_w_data_o,
  input  logic       mbox_w_full_i,
  output logic       mbox_w_push_o,
  output logic       mbox_w_flush_o,
  input  usage_t     mbox_w_usage_i,
  // read FIFO port
  input  data_t      mbox_r_data_i,
  input  logic       mbox_r_empty_i,
  output logic       mbox_r_pop_o,
  output logic       mbox_r_flush_o,
  input  usage_t     mbox_r_usage_i,
  // interrupt output, level triggered, active high, conversion in top
  output logic       irq_o,
  output logic       clear_irq_o // clear the edge trigger irq register in `axi_lite_mailbox`
);

  `AXI_LITE_TYPEDEF_B_CHAN_T(b_chan_lite_t)
  `AXI_LITE_TYPEDEF_R_CHAN_T(r_chan_lite_t, data_t)

  localparam int unsigned NoRegs = 32'd10;
  typedef enum logic [3:0] {
    MBOXW  = 4'd0, // Mailbox write register
    MBOXR  = 4'd1, // Mailbox read register
    STATUS = 4'd2, // Mailbox status register
    ERROR  = 4'd3, // Mailbox error register
    WIRQT  = 4'd4, // Write interrupt request threshold register
    RIRQT  = 4'd5, // Read interrupt request threshold register
    IRQS   = 4'd6, // Interrupt request status register
    IRQEN  = 4'd7, // Interrupt request enable register
    IRQP   = 4'd8, // Interrupt request pending register
    CTRL   = 4'd9  // Mailbox control register
  } reg_e;
  // address map rule struct, as required from `addr_decode` from `common_cells`
  typedef struct packed {
    int unsigned idx;
    addr_t       start_addr;
    addr_t       end_addr;
  } rule_t;
  // output type of the address decoders, to be casted onto the enum type `reg_e`
  typedef logic [$clog2(NoRegs)-1:0] idx_t;

  // LITE response signals, go into the output spill registers to prevent combinational response
  logic         b_valid, b_ready;
  b_chan_lite_t b_chan;
  logic         r_valid, r_ready;
  r_chan_lite_t r_chan; 
  // address map generation
  rule_t [NoRegs-1:0] addr_map;
  addr_t base_addr;
  assign base_addr = {chip_id_i, base_addr_i[AxiAddrWidth-ChipIdWidth-1:0]};
  for (genvar i = 0; i < NoRegs; i++) begin : gen_addr_map
    assign addr_map[i] = '{
        idx:        i,
        start_addr: base_addr +  i      * (AxiDataWidth / 8),
        end_addr:   base_addr + (i + 1) * (AxiDataWidth / 8),
        default:    '0
    };
  end
  // address decode flags
  idx_t w_reg_idx,   r_reg_idx;
  logic dec_w_valid, dec_r_valid;

  // mailbox register signal declarations, get extended when read, some of these regs
  // are build combinationally, indicated by the absence of the `*_d` signal

  logic [3:0] status_q;           // mailbox status register (read only)
  logic [1:0] error_q,   error_d; // mailbox error register
  data_t      wirqt_q,   wirqt_d; // write interrupt request threshold register
  data_t      rirqt_q,   rirqt_d; // read interrupt request threshold register
  logic [2:0] irqs_q,    irqs_d;  // interrupt request status register
  logic [2:0] irqen_q,   irqen_d; // interrupt request enable register
  logic [2:0] irqp_q;             // interrupt request pending register (read only)
  logic [1:0] ctrl_q;             // mailbox control register
  logic       update_regs;        // register enable signal

  // register instantiation
  `FFLARN(error_q, error_d, update_regs, '0, clk_i, rst_ni)
  `FFLARN(wirqt_q, wirqt_d, update_regs, '0, clk_i, rst_ni)
  `FFLARN(rirqt_q, rirqt_d, update_regs, '0, clk_i, rst_ni)
  `FFLARN(irqs_q, irqs_d, update_regs, '0, clk_i, rst_ni)
  `FFLARN(irqen_q, irqen_d, update_regs, '0, clk_i, rst_ni)

  // Mailbox FIFO data assignments
  for (genvar i = 0; i < (AxiDataWidth/8); i++) begin : gen_w_mbox_data
    assign mbox_w_data_o[i*8+:8] = slv_req_i.w.strb[i] ? slv_req_i.w.data[i*8+:8] : '0;
  end

  // combinational mailbox register assignments, for the read only registers
  assign status_q = { mbox_r_usage_i > usage_t'(rirqt_q),
                      mbox_w_usage_i > usage_t'(wirqt_q),
                      mbox_w_full_i,
                      mbox_r_empty_i };
  assign irqp_q   = irqs_q & irqen_q;                 // interrupt request pending is bit wise and
  assign ctrl_q   = {mbox_r_flush_o, mbox_w_flush_o}; // read ctrl_q is flush signals
  assign irq_o    = |irqp_q;                          // generate an active-high level irq

  always_comb begin
    // slave port channel outputs for the AW, W and R channel, other driven from spill register
    slv_resp_o.aw_ready = 1'b0;
    slv_resp_o.w_ready  = 1'b0;
    b_chan              = '{resp: axi_pkg::RESP_SLVERR};
    b_valid             = 1'b0;
    slv_resp_o.ar_ready = 1'b0;
    r_chan              = '{data: '0, resp: axi_pkg::RESP_SLVERR};
    r_valid             = 1'b0;
    // Default assignments for the internal registers
    error_d     = error_q;   // mailbox error register
    wirqt_d     = wirqt_q;   // write interrupt request threshold register
    rirqt_d     = rirqt_q;   // read interrupt request threshold register
    irqs_d      = irqs_q;    // interrupt request status register
    irqen_d     = irqen_q;   // interrupt request enable register
    update_regs = 1'b0;      // register update enable signal
    // MBOX FIFO control signals
    mbox_w_push_o  = 1'b0;
    mbox_w_flush_o = 1'b0;
    mbox_r_pop_o   = 1'b0;
    mbox_r_flush_o = 1'b0;
    // clear the edge triggered irq register if it is instantiated
    clear_irq_o    = 1'b0;

    // -------------------------------------------
    // Set the read and write interrupt FF (irqs), when the threshold triggers
    // -------------------------------------------
    // strict threshold interrupt these fields get cleared by acknowledge on write onto the register
    // read trigger, see status_q above
    if (!irqs_q[1] && status_q[3]) begin
      irqs_d[1]   = 1'b1;
      update_regs = 1'b1;
    end
    // write trigger, see status_q above
    if (!irqs_q[0] && status_q[2]) begin
      irqs_d[0]   = 1'b1;
      update_regs = 1'b1;
    end

    // -------------------------------------------
    // Read registers
    // -------------------------------------------
    // The logic of the read and write channels have to be in the same `always_comb` block.
    // The reason is that the error register could be cleared in the same cycle as a read from
    // the mailbox FIFO generates a new error. In this case the error is NOT cleared. Instead
    // it will generate a new irq when it is edge triggered, or the level will stay at its
    // active state.

    // Check if there is a pending read request on the slave port.
    if (slv_req_i.ar_valid) begin
      // set the right read channel output depending on the address decoding
      if (dec_r_valid) begin
        // when decode not valid, send the default slaveerror
        // read the right register when the transfer happens and decode is valid
        unique case (reg_e'(r_reg_idx))
          MBOXW: r_chan = '{data: data_t'( 32'hFEEDC0DE ), resp: axi_pkg::RESP_OKAY};
          MBOXR: begin
            if (!mbox_r_empty_i) begin
              r_chan       = '{data: data_t'( mbox_r_data_i ), resp: axi_pkg::RESP_OKAY};
              mbox_r_pop_o = 1'b1;
            end else begin
              // read mailbox is empty, set the read error Flip flop and respond with error
              r_chan      = '{data: data_t'( 32'hFEEDDEAD ), resp: axi_pkg::RESP_SLVERR};
              error_d[0]  = 1'b1;
              irqs_d[2]   = 1'b1;
              update_regs = 1'b1;
            end
          end
          STATUS: r_chan = '{data: data_t'( status_q ), resp: axi_pkg::RESP_OKAY};
          ERROR: begin // clear the error register
            r_chan      = '{data: data_t'( error_q  ), resp: axi_pkg::RESP_OKAY};
            error_d     = '0;
            update_regs = 1'b1;
          end
          WIRQT: r_chan = '{data: data_t'( wirqt_q  ), resp: axi_pkg::RESP_OKAY};
          RIRQT: r_chan = '{data: data_t'( rirqt_q  ), resp: axi_pkg::RESP_OKAY};
          IRQS:  r_chan = '{data: data_t'( irqs_q   ), resp: axi_pkg::RESP_OKAY};
          IRQEN: r_chan = '{data: data_t'( irqen_q  ), resp: axi_pkg::RESP_OKAY};
          IRQP:  r_chan = '{data: data_t'( irqp_q   ), resp: axi_pkg::RESP_OKAY};
          CTRL:  r_chan = '{data: data_t'( ctrl_q   ), resp: axi_pkg::RESP_OKAY};
          default: /*do nothing*/;
        endcase
      end
      r_valid = 1'b1;
      if (r_ready) begin
        slv_resp_o.ar_ready = 1'b1;
      end
    end // read register

    // -------------------------------------------
    // Write registers
    // -------------------------------------------
    // Wait for control and write data to be valid.
    if (slv_req_i.aw_valid && slv_req_i.w_valid) begin
      // Can do the handshake here as the b response goes into a spill register with latency one.
      // Without the B spill register, the B channel would violate the AXI stable requirement.
      b_valid = 1'b1;
      if (b_ready) begin
        // write to the register if required
        if (dec_w_valid) begin
          unique case (reg_e'(w_reg_idx))
            MBOXW:  begin
              if (!mbox_w_full_i) begin
                mbox_w_push_o = 1'b1;
                b_chan        = '{resp: axi_pkg::RESP_OKAY};
              end else begin
                // response with default error and set the error FF
                error_d[1]  = 1'b1;
                irqs_d[2]   = 1'b1;
                update_regs = 1'b1;
              end
            end
            // MBOXR:  read only
            // STATUS: read only
            // ERROR:  read only
            WIRQT:  begin
              for (int unsigned i = 0; i < AxiDataWidth/8; i++) begin
                wirqt_d[i*8+:8] = slv_req_i.w.strb[i] ? slv_req_i.w.data[i*8+:8] : 8'b0000_0000;
              end
              if (wirqt_d >= data_t'(MailboxDepth)) begin
                // the `-1` is to have the interrupt fireing when the FIFO is comletely full
                wirqt_d = data_t'(MailboxDepth) - data_t'(32'd1); // Threshold to maximal value
              end
              update_regs = 1'b1;
              b_chan      = '{resp: axi_pkg::RESP_OKAY};
            end
            RIRQT:  begin
              for (int unsigned i = 0; i < AxiDataWidth/8; i++) begin
                rirqt_d[i*8+:8] = slv_req_i.w.strb[i] ? slv_req_i.w.data[i*8+:8] : 8'b0000_0000;
              end
              if (rirqt_d >= data_t'(MailboxDepth)) begin
              // Threshold to maximal value, minus two to prevent overflow in usage
                rirqt_d = data_t'(MailboxDepth) - data_t'(32'd1);
              end
              update_regs = 1'b1;
              b_chan      = '{resp: axi_pkg::RESP_OKAY};
            end
            IRQS:   begin
              // Acknowledge and clear the register by asserting the respective one
              if (slv_req_i.w.strb[0]) begin
                // *_d signal is set in the beginning of this process, prevent accidental
                // overwrite of not acknowledged irq
                irqs_d[2]   = slv_req_i.w.data[2] ? 1'b0 : irqs_d[2]; // Error irq status
                irqs_d[1]   = slv_req_i.w.data[1] ? 1'b0 : irqs_d[1]; // Read  irq status
                irqs_d[0]   = slv_req_i.w.data[0] ? 1'b0 : irqs_d[0]; // Write irq status
                clear_irq_o = 1'b1;
                update_regs = 1'b1;
              end
              b_chan = '{resp: axi_pkg::RESP_OKAY};
            end
            IRQEN:  begin
              if (slv_req_i.w.strb[0]) begin
                irqen_d[2:0]  = slv_req_i.w.data[2:0]; // set the irq enable bits
                update_regs = 1'b1;
              end
              b_chan = '{resp: axi_pkg::RESP_OKAY};
            end
            // IRQP: read only
            CTRL:   begin
              if (slv_req_i.w.strb[0]) begin
                mbox_r_flush_o = slv_req_i.w.data[1]; // Flush read  FIFO
                mbox_w_flush_o = slv_req_i.w.data[0]; // Flush write FIFO
              end
              b_chan = '{resp: axi_pkg::RESP_OKAY};
            end
            default : /* use default b_chan */;
          endcase
        end
        slv_resp_o.aw_ready = 1'b1;
        slv_resp_o.w_ready  = 1'b1;
      end // if (b_ready): Does not violate AXI spec, because the ready comes from an internal
          // spill register and does not propagate the ready from the b channel.
    end // write register
  end

  // address decoder and response FIFOs for the LITE channel, the port can take a new transaction if
  // these FIFOs are not full, not fall through to prevent combinational paths to the return path
  addr_decode #(
    .NoIndices( NoRegs ),
    .NoRules  ( NoRegs ),
    .addr_t   ( addr_t ),
    .rule_t   ( rule_t )
  ) i_waddr_decode (
    .addr_i           ( slv_req_i.aw.addr ),
    .addr_map_i       ( addr_map          ),
    .idx_o            ( w_reg_idx         ),
    .dec_valid_o      ( dec_w_valid       ),
    .dec_error_o      ( /*not used*/      ),
    .en_default_idx_i ( 1'b0              ),
    .default_idx_i    ( '0                )
  );
  spill_register #(
    .T ( b_chan_lite_t )
  ) i_b_chan_outp (
    .clk_i,
    .rst_ni,
    .valid_i ( b_valid            ),
    .ready_o ( b_ready            ),
    .data_i  ( b_chan             ),
    .valid_o ( slv_resp_o.b_valid ),
    .ready_i ( slv_req_i.b_ready  ),
    .data_o  ( slv_resp_o.b       )
  );
  addr_decode #(
    .NoIndices( NoRegs ),
    .NoRules  ( NoRegs ),
    .addr_t   ( addr_t ),
    .rule_t   ( rule_t )
  ) i_raddr_decode (
    .addr_i           ( slv_req_i.ar.addr ),
    .addr_map_i       ( addr_map          ),
    .idx_o            ( r_reg_idx         ),
    .dec_valid_o      ( dec_r_valid       ),
    .dec_error_o      ( /*not used*/      ),
    .en_default_idx_i ( 1'b0              ),
    .default_idx_i    ( '0                )
  );
  spill_register #(
    .T ( r_chan_lite_t )
  ) i_r_chan_outp (
    .clk_i,
    .rst_ni,
    .valid_i ( r_valid            ),
    .ready_o ( r_ready            ),
    .data_i  ( r_chan             ),
    .valid_o ( slv_resp_o.r_valid ),
    .ready_i ( slv_req_i.r_ready  ),
    .data_o  ( slv_resp_o.r       )
  );
  // pragma translate_off
  `ifndef VERILATOR
  initial begin : proc_check_params
    assert (AxiAddrWidth == $bits(slv_req_i.aw.addr)) else $fatal(1, "AW AxiAddrWidth mismatch");
    assert (AxiDataWidth == $bits(slv_req_i.w.data))  else $fatal(1, " W AxiDataWidth mismatch");
    assert (AxiAddrWidth == $bits(slv_req_i.ar.addr)) else $fatal(1, "AR AxiAddrWidth mismatch");
    assert (AxiDataWidth == $bits(slv_resp_o.r.data)) else $fatal(1, " R AxiDataWidth mismatch");
  end
  `endif
  // pragma translate_on
endmodule
