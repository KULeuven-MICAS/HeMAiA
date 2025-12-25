// Authors:
// - Fanchen Kong <fanchen.kong@kuleuven.be>

/// A protocol converter from AXI Lite to CSR Interface
module axi_lite_to_csr #(
    /// The width of the address.
    parameter int AXI_LITE_ADDR_WIDTH = -1,
    /// The width of the data.
    parameter int AXI_LITE_DATA_WIDTH = -1,
    /// Buffer depth (how many outstanding transactions do we allow)
    parameter int BUFFER_DEPTH = 2,
    /// Whether the AXI-Lite W channel should be decoupled with a register. This
    /// can help break long paths at the expense of registers.
    parameter bit DECOUPLE_W = 1,
    /// AXI-Lite request struct type.
    parameter type axi_lite_req_t = logic,
    /// AXI-Lite response struct type.
    parameter type axi_lite_rsp_t = logic,
    /// CSR request struct type.
    parameter type csr_req_t = logic,
    /// CSR response struct type.
    parameter type csr_rsp_t = logic
    //   // CSR Req/Rsp interface
    //   typedef struct packed {
    //     addr_t   addr;
    //     logic [31:0]   data;
    //     logic    write;
    //   } csr_req_t;
    //   typedef struct packed {
    //     logic [31:0]   data;
    //   } csr_rsp_t;
) (
    input  logic           clk_i         ,
    input  logic           rst_ni        ,
    // Input AXI Lite
    input  axi_lite_req_t  axi_lite_req_i,
    output axi_lite_rsp_t  axi_lite_rsp_o,
    // Output CSR Interface
    output csr_req_t       csr_req_o     ,
    output logic           csr_req_valid_o,
    input  logic           csr_req_ready_i,
    input  csr_rsp_t       csr_rsp_i,
    input  logic           csr_rsp_valid_i,
    output logic           csr_rsp_ready_o
);
    typedef struct packed {
        logic [AXI_LITE_ADDR_WIDTH-1:0]   addr;
        logic [31:0]   data;
    } write_t;

    typedef struct packed {
        logic [AXI_LITE_ADDR_WIDTH-1:0] addr;
        logic write;
    } req_t;

    typedef struct packed {
        logic [31:0] data;
    } resp_t;

    ///////////////////////////////////
    // FIFO Write Req Signals for AW/W
    ///////////////////////////////////
    logic   write_fifo_full, write_fifo_empty;
    write_t write_fifo_in,   write_fifo_out;
    logic   write_fifo_push, write_fifo_pop;

    ///////////////////////////////////
    // FIFO Write Resp Signals for B
    ///////////////////////////////////
    logic   write_resp_fifo_full, write_resp_fifo_empty;
    logic   write_resp_fifo_in,   write_resp_fifo_out;
    logic   write_resp_fifo_push, write_resp_fifo_pop;

    ///////////////////////////////////
    // FIFO Read Req Signals for AR
    ///////////////////////////////////
    logic   read_fifo_full, read_fifo_empty;
    logic [AXI_LITE_ADDR_WIDTH-1:0]  read_fifo_in,   read_fifo_out;
    logic   read_fifo_push, read_fifo_pop;

    ///////////////////////////////////
    // FIFO Read Resp Signals for R
    ///////////////////////////////////
    logic   read_resp_fifo_full, read_resp_fifo_empty;
    resp_t  read_resp_fifo_in,   read_resp_fifo_out;
    logic   read_resp_fifo_push, read_resp_fifo_pop;

    //////////////////////////////////
    // Arbiter 
    //////////////////////////////////
    req_t read_req, write_req, arb_req;
    logic read_valid, write_valid, arb_req_valid;
    logic read_ready, write_ready, arb_req_ready;


    // Combine AW/W Channel
    fifo_v3 #(
        .FALL_THROUGH ( !DECOUPLE_W  ),
        .DEPTH        ( BUFFER_DEPTH ),
        .dtype        ( write_t      )
    ) i_fifo_write_req (
        .clk_i,
        .rst_ni,
        .flush_i    ( 1'b0             ),
        .testmode_i ( 1'b0             ),
        .full_o     ( write_fifo_full  ),
        .empty_o    ( write_fifo_empty ),
        .usage_o    ( /* open */       ),
        .data_i     ( write_fifo_in    ),
        .push_i     ( write_fifo_push  ),
        .data_o     ( write_fifo_out   ),
        .pop_i      ( write_fifo_pop   )
    );
    assign axi_lite_rsp_o.aw_ready = write_fifo_push; // not full and aw_valid presents
    assign axi_lite_rsp_o.w_ready = write_fifo_push;  // not full and w_valid presents
    assign write_fifo_push = axi_lite_req_i.aw_valid & axi_lite_req_i.w_valid & ~write_fifo_full;
    assign write_fifo_in.addr = axi_lite_req_i.aw.addr;
    assign write_fifo_in.data = axi_lite_req_i.w.data;
    assign write_fifo_pop = write_valid && write_ready;
    assign csr_req_o.data = arb_req.write ? write_fifo_out.data[31:0] : '0;
    // B Channel
    fifo_v3 #(
        .DEPTH        ( BUFFER_DEPTH ),
        .dtype        ( logic        )
    ) i_fifo_write_resp (
        .clk_i,
        .rst_ni,
        .flush_i    ( 1'b0                  ),
        .testmode_i ( 1'b0                  ),
        .full_o     ( write_resp_fifo_full  ),
        .empty_o    ( write_resp_fifo_empty ),
        .usage_o    ( /* open */            ),
        .data_i     ( write_resp_fifo_in    ),
        .push_i     ( write_resp_fifo_push  ),
        .data_o     ( write_resp_fifo_out   ),
        .pop_i      ( write_resp_fifo_pop   )
    );
    assign axi_lite_rsp_o.b_valid = ~write_resp_fifo_empty;
    assign axi_lite_rsp_o.b.resp = axi_pkg::RESP_OKAY;
    assign write_resp_fifo_in = '0; // always OKAY
    assign write_resp_fifo_pop = ~write_resp_fifo_empty & axi_lite_req_i.b_ready;
    assign write_resp_fifo_push = arb_req.write && arb_req_valid && arb_req_ready && ~write_resp_fifo_full ; // push when write req is accepted

    // AR Channel
    fifo_v3 #(
        .DEPTH        ( BUFFER_DEPTH ),
        .DATA_WIDTH   ( AXI_LITE_ADDR_WIDTH   )
    ) i_fifo_read (
        .clk_i,
        .rst_ni,
        .flush_i    ( 1'b0            ),
        .testmode_i ( 1'b0            ),
        .full_o     ( read_fifo_full  ),
        .empty_o    ( read_fifo_empty ),
        .usage_o    ( /* open */      ),
        .data_i     ( read_fifo_in    ),
        .push_i     ( read_fifo_push  ),
        .data_o     ( read_fifo_out   ),
        .pop_i      ( read_fifo_pop   )
    );
    assign read_fifo_in = axi_lite_req_i.ar.addr;
    assign axi_lite_rsp_o.ar_ready = ~read_fifo_full;
    assign read_fifo_push = ~read_fifo_full & axi_lite_req_i.ar_valid;
    assign read_fifo_pop = read_valid && read_ready;

    // R Channel
    fifo_v3 #(
        .DEPTH        ( BUFFER_DEPTH ),
        .dtype        ( resp_t       )
    ) i_fifo_read_resp (
        .clk_i,
        .rst_ni,
        .flush_i    ( 1'b0                 ),
        .testmode_i ( 1'b0                 ),
        .full_o     ( read_resp_fifo_full  ),
        .empty_o    ( read_resp_fifo_empty ),
        .usage_o    ( /* open */           ),
        .data_i     ( read_resp_fifo_in    ),
        .push_i     ( read_resp_fifo_push  ),
        .data_o     ( read_resp_fifo_out   ),
        .pop_i      ( read_resp_fifo_pop   )
    );
    assign axi_lite_rsp_o.r.data[AXI_LITE_DATA_WIDTH-1:32] = '0;
    assign axi_lite_rsp_o.r.data[31:0] = read_resp_fifo_out.data;
    assign axi_lite_rsp_o.r.resp = axi_pkg::RESP_OKAY;
    assign axi_lite_rsp_o.r_valid = ~read_resp_fifo_empty;
    assign read_resp_fifo_pop = ~read_resp_fifo_empty & axi_lite_req_i.r_ready;
    assign read_resp_fifo_in.data = csr_rsp_i.data;
    assign read_resp_fifo_push = ~arb_req.write && arb_req_valid && arb_req_ready && csr_rsp_valid_i && ~read_resp_fifo_full; // push when read req is accepted
    assign csr_rsp_ready_o = ~arb_req.write && arb_req_valid && ~read_resp_fifo_full;

    stream_arbiter #(
        .DATA_T  ( req_t ),
        .N_INP   ( 2     ),
        .ARBITER ( "rr"  )
    ) i_stream_arbiter (
        .clk_i,
        .rst_ni,
        .inp_data_i  ( {read_req,   write_req}   ),
        .inp_valid_i ( {read_valid, write_valid} ),
        .inp_ready_o ( {read_ready, write_ready} ),
        .oup_data_o  ( arb_req                   ),
        .oup_valid_o ( arb_req_valid             ),
        .oup_ready_i ( arb_req_ready             )
    );
    assign csr_req_o.addr = arb_req.addr;
    assign csr_req_o.write = arb_req.write;
    assign csr_req_valid_o = arb_req_valid;
    assign write_req.addr = write_fifo_out.addr;
    assign write_req.write = 1'b1;
    assign read_req.addr = read_fifo_out;
    assign read_req.write = 1'b0;
    // Make sure we can capture the responses (e.g. have enough fifo space)
    assign read_valid = ~read_fifo_empty & ~read_resp_fifo_full;
    assign write_valid = ~write_fifo_empty & ~write_resp_fifo_full;
    // If the request is write
    // The ready depends on: csr_req_ready_i
    // If the request is read
    // The ready depends on: csr_req_ready_i and csr_rsp_valid_i
    assign arb_req_ready = arb_req.write ? csr_req_ready_i : (csr_req_ready_i && csr_rsp_valid_i);
endmodule