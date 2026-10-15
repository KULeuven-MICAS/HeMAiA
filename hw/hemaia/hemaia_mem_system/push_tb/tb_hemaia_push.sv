// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// Standalone push testbench: a memory chip M streams weights to a compute chip C over one D2D
// link, launched by host models on C the way the weight prefetcher launches them.
//
//      M (0x20)                                              C (0x10)
//   source memory --> hemaia_sys_idma_engine k --> link port k ===== link ===> sink (checks data)
//   link port P --> axi_dw_converter --> axi_to_reg --> engine k's registers <=== host models
//
// Both ends are hemaia_d2d_link over the PHY model, in DDR (the link carries one 512-bit flit a
// digital cycle each way, never both ways at once). M has two local ports, one per engine;
// requests arriving from the west leave M's link on port 1 (the link's steering rule), which
// feeds the engines' registers. Each host model drives one engine through C's one local port:
//
//   mode 0, register launch: pick the run (PickCycles), then the 7 posted register writes of
//     sys_dma_engine_memcpy (destination, source, length high, conf, length low last), a fence
//     (every B back), reads of the length until it holds the new value, the launching read of
//     next_id (it returns once the backend takes the transfer), and the length cleared. Then the
//     next run.
//   mode 1, queue: pick the run, then the queue's writes (source, destination, length, doorbell,
//     8 bytes each, sys_dma_engine_queue) and on, without reading anything. At most
//     +inflight runs an engine are posted but not yet landed (the clusters' ring slots).
//
// Every run is RunChunks chunks of ChunkBytes, copied from the engine's source region to its own
// region of C. The sink checks every beat's data against the source pattern, and a run lands
// when its last byte has. The rate is the bytes landed over the time from the first launch to the
// last landing, per digital cycle, and in GB/s at the 500 MHz design point.
//
// Plusargs: +mode=0|1 +engines=1|2 +run_chunks=N +runs=N +pick=CYCLES +inflight=N
//           +host_path=CYCLES (each way between a host and C's link port) +rx_yield=W (C's RX
//           yield window, 0 off) +src_lat=CYCLES (source memory read latency) +tag=STRING
//           +run_bytes=N (runs of N bytes, a multiple of 64, instead of run_chunks chunks)

`include "axi/typedef.svh"
`include "register_interface/typedef.svh"
`include "idma/typedef.svh"

module tb_hemaia_push;

  timeunit 1ps;
  timeprecision 1ps;

  // 7:1, the design point: a 3.5 GHz PHY under a 500 MHz chip
  localparam realtime CyclTime = 2100ps;
  localparam realtime PhyCyclTime = 300ps;
  localparam realtime ApplTime = 400ps;
  localparam realtime TestTime = 1600ps;

  localparam int unsigned NumPorts = 2;  // M's local ports = its engines here
  localparam int unsigned RegPort = 1;  // where requests from the west leave M's link
  localparam logic [7:0] ChipM = 8'h20, ChipC = 8'h10;
  localparam logic [39:0] RegBase = 40'h0_0500_0000;  // SYS_IDMA_CFG_BASE_ADDR
  localparam logic [39:0] SrcBase = 40'h1_0000_0000;  // M's HBM
  localparam logic [39:0] DstBase = 40'h0_8000_0000;  // C's L3
  localparam logic [39:0] RegionBytes = 40'h0_1000_0000;  // per engine, source and destination
  localparam int unsigned ChunkBytes = 90112;  // 88 KiB, the weight chunk

  // ---------------------------------------------------------------------- types
  typedef logic [7:0] chip_id_t;
  typedef logic [47:0] addr_t;
  typedef logic [3:0] id_t;
  typedef logic [0:0] user_t;
  typedef logic [511:0] data_t;
  typedef logic [63:0] strb_t;
  typedef logic [31:0] data32_t;
  typedef logic [3:0] strb32_t;
  typedef logic [47:0] tf_len_t;

  `AXI_TYPEDEF_ALL(axi, addr_t, id_t, data_t, strb_t, user_t)
  `AXI_TYPEDEF_ALL(axi32, addr_t, id_t, data32_t, strb32_t, user_t)
  `REG_BUS_TYPEDEF_ALL(cfg, addr_t, data32_t, strb32_t)
  `IDMA_TYPEDEF_OPTIONS_T(idma_options_t, id_t)
  `IDMA_TYPEDEF_REQ_T(idma_req_t, tf_len_t, addr_t, idma_options_t, user_t)
  `IDMA_TYPEDEF_ERR_PAYLOAD_T(idma_err_payload_t, addr_t)
  `IDMA_TYPEDEF_RSP_T(idma_rsp_t, idma_err_payload_t)
  typedef struct packed {axi_ar_chan_t ar_chan;} idma_axi_read_meta_t;
  typedef struct packed {idma_axi_read_meta_t axi;} idma_read_meta_channel_t;
  typedef struct packed {axi_aw_chan_t aw_chan;} idma_axi_write_meta_t;
  typedef struct packed {idma_axi_write_meta_t axi;} idma_write_meta_channel_t;
  typedef hemaia_d2d_link_pkg::hemaia_d2d_link_default_axi_lite_req_t lite_req_t;
  typedef hemaia_d2d_link_pkg::hemaia_d2d_link_default_axi_lite_rsp_t lite_rsp_t;

  // ---------------------------------------------------------------------- settings
  int unsigned mode, engines, run_chunks, runs, pick, inflight_max, host_path, rx_yield, src_lat;
  bit debug;
  string tag;
  longint unsigned run_bytes;
  longint unsigned run_bytes_arg;

  // ---------------------------------------------------------------------- clocks, reset
  logic clk, phy_clk, rst_n;
  longint unsigned cycle;

  initial begin
    clk = 1'b0;
    forever #(CyclTime / 2) clk = ~clk;
  end
  initial begin
    phy_clk = 1'b0;
    forever #(PhyCyclTime / 2) phy_clk = ~phy_clk;
  end
  initial begin
    rst_n = 1'b0;
    repeat (20) @(posedge clk);
    rst_n = 1'b1;
  end
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) cycle <= 0;
    else cycle <= cycle + 1;
  end

  int unsigned errors;
  task automatic note_error(input string msg);
    errors++;
    if (errors <= 20) $error("%s", msg);
  endtask

  // ---------------------------------------------------------------------- the data pattern
  function automatic data_t src_data(input addr_t a);
    data_t d;
    for (int unsigned k = 0; k < 16; k++) d[32*k+:32] = a[37:6] ^ (32'h9E3779B9 * (k + 1)) ^ {16'h0, a[47:32]};
    return d;
  endfunction

  function automatic addr_t src_addr(input int unsigned e, input longint unsigned off);
    return {ChipM, SrcBase + RegionBytes * e + 40'(off)};
  endfunction
  function automatic addr_t dst_addr(input int unsigned e, input longint unsigned off);
    return {ChipC, DstBase + RegionBytes * e + 40'(off)};
  endfunction
  function automatic addr_t reg_addr(input int unsigned e, input logic [11:0] off);
    return {ChipM, RegBase + 40'h1000 * e + 40'(off)};
  endfunction

  // ======================================================================= M: the memory chip
  axi_req_t [NumPorts-1:0] m_to_remote_req, m_from_remote_req;
  axi_resp_t [NumPorts-1:0] m_to_remote_rsp, m_from_remote_rsp;
  axi_req_t [NumPorts-1:0] src_req;
  axi_resp_t [NumPorts-1:0] src_rsp;
  lite_req_t lite_req[2];
  lite_rsp_t lite_rsp[2];

  // the registers: M's link port RegPort -> 32-bit AXI -> register bus -> engine windows
  axi32_req_t reg32_req;
  axi32_resp_t reg32_rsp;
  cfg_req_t reg_req;
  cfg_rsp_t reg_rsp;
  cfg_req_t [NumPorts-1:0] eng_cfg_req;
  cfg_rsp_t [NumPorts-1:0] eng_cfg_rsp;

  axi_dw_converter #(
      .AxiMaxReads        (16),
      .AxiSlvPortDataWidth(512),
      .AxiMstPortDataWidth(32),
      .AxiAddrWidth       (48),
      .AxiIdWidth         (4),
      .aw_chan_t          (axi_aw_chan_t),
      .mst_w_chan_t       (axi32_w_chan_t),
      .slv_w_chan_t       (axi_w_chan_t),
      .b_chan_t           (axi_b_chan_t),
      .ar_chan_t          (axi_ar_chan_t),
      .mst_r_chan_t       (axi32_r_chan_t),
      .slv_r_chan_t       (axi_r_chan_t),
      .axi_mst_req_t      (axi32_req_t),
      .axi_mst_resp_t     (axi32_resp_t),
      .axi_slv_req_t      (axi_req_t),
      .axi_slv_resp_t     (axi_resp_t)
  ) i_reg_dw (
      .clk_i     (clk),
      .rst_ni    (rst_n),
      .slv_req_i (m_from_remote_req[RegPort]),
      .slv_resp_o(m_from_remote_rsp[RegPort]),
      .mst_req_o (reg32_req),
      .mst_resp_i(reg32_rsp)
  );

  axi_to_reg #(
      .ADDR_WIDTH(48),
      .DATA_WIDTH(32),
      .ID_WIDTH  (4),
      .USER_WIDTH(1),
      .axi_req_t (axi32_req_t),
      .axi_rsp_t (axi32_resp_t),
      .reg_req_t (cfg_req_t),
      .reg_rsp_t (cfg_rsp_t)
  ) i_axi_to_reg (
      .clk_i     (clk),
      .rst_ni    (rst_n),
      .testmode_i(1'b0),
      .axi_req_i (reg32_req),
      .axi_rsp_o (reg32_rsp),
      .reg_req_o (reg_req),
      .reg_rsp_i (reg_rsp)
  );

  logic [$clog2(NumPorts)-1:0] eng_sel;
  assign eng_sel = reg_req.addr[12+:$clog2(NumPorts)];

  reg_demux #(
      .NoPorts(NumPorts),
      .req_t  (cfg_req_t),
      .rsp_t  (cfg_rsp_t)
  ) i_eng_demux (
      .clk_i      (clk),
      .rst_ni     (rst_n),
      .in_select_i(eng_sel),
      .in_req_i   (reg_req),
      .in_rsp_o   (reg_rsp),
      .out_req_o  (eng_cfg_req),
      .out_rsp_i  (eng_cfg_rsp)
  );

  // +debug: every register access, as the engines see it
  always @(posedge clk) begin
    if (debug && rst_n && reg_req.valid && reg_rsp.ready)
      $display("PUSH_TB debug reg %s %h data %h strb %b", reg_req.write ? "W" : "R", reg_req.addr,
               reg_req.write ? reg_req.wdata : reg_rsp.rdata, reg_req.wstrb);
    if (debug && rst_n && reg32_req.aw_valid && reg32_rsp.aw_ready)
      $display("PUSH_TB debug axi32 AW %h len %0d size %0d", reg32_req.aw.addr, reg32_req.aw.len, reg32_req.aw.size);
    if (debug && rst_n && reg32_req.w_valid && reg32_rsp.w_ready)
      $display("PUSH_TB debug axi32 W %h strb %b last %b", reg32_req.w.data, reg32_req.w.strb, reg32_req.w.last);
  end

  // The link answers a write at the sender, so a register write M's register path rejects (the
  // burst splitter in axi_to_reg answers a non-modifiable burst with an error) never reaches a host.
  always @(posedge clk) begin
    if (rst_n && reg32_rsp.b_valid && reg32_req.b_ready && reg32_rsp.b.resp != axi_pkg::RESP_OKAY)
      note_error("M's register path answered a write with an error");
    if (rst_n && reg32_rsp.r_valid && reg32_req.r_ready && reg32_rsp.r.resp != axi_pkg::RESP_OKAY)
      note_error("M's register path answered a read with an error");
  end

  // the other port gets nothing from the west; anything that arrives is an error
  for (genvar p = 0; p < NumPorts; p++) begin : gen_other_ports
    if (p != RegPort) begin : gen_unused
      assign m_from_remote_rsp[p] = '0;
      always @(posedge clk) begin
        if (rst_n && (m_from_remote_req[p].aw_valid || m_from_remote_req[p].ar_valid))
          note_error($sformatf("a request left M's link on port %0d", p));
      end
    end
  end

  for (genvar e = 0; e < NumPorts; e++) begin : gen_engine
    axi_req_t write_req;
    axi_resp_t write_rsp;

    hemaia_sys_idma_engine #(
        .DataWidth           (512),
        .AddrWidth           (48),
        .AxiIdWidth          (4),
        .TFLenWidth          (48),
        .QueueDepth          (32),
        .reg_req_t           (cfg_req_t),
        .reg_rsp_t           (cfg_rsp_t),
        .idma_req_t          (idma_req_t),
        .idma_rsp_t          (idma_rsp_t),
        .axi_req_t           (axi_req_t),
        .axi_rsp_t           (axi_resp_t),
        .read_meta_channel_t (idma_read_meta_channel_t),
        .write_meta_channel_t(idma_write_meta_channel_t)
    ) i_engine (
        .clk_i          (clk),
        .rst_ni         (rst_n),
        .cfg_req_i      (eng_cfg_req[e]),
        .cfg_rsp_o      (eng_cfg_rsp[e]),
        .axi_read_req_o (src_req[e]),
        .axi_read_rsp_i (src_rsp[e]),
        .axi_write_req_o(write_req),
        .axi_write_rsp_i(write_rsp)
    );

    // writes to C go out on link port e; the backend's write port never reads
    assign m_to_remote_req[e] = write_req;
    assign write_rsp = m_to_remote_rsp[e];
  end

  // ---------------------------------------------------------------------- source memories
  // One per engine (M gives each engine its own HBM port): every read answered with the source
  // pattern, src_lat cycles after its AR, one beat a cycle.
  for (genvar e = 0; e < NumPorts; e++) begin : gen_src
    typedef struct {
      axi_ar_chan_t ar;
      longint unsigned due;
    } pending_t;
    pending_t pend[$];
    int unsigned beat;
    logic r_valid_q;
    axi_r_chan_t r_q;

    always_comb begin
      src_rsp[e] = '0;
      src_rsp[e].ar_ready = 1'b1;
      src_rsp[e].r_valid = r_valid_q;
      src_rsp[e].r = r_q;
    end

    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        pend.delete();
        beat = 0;
        r_valid_q <= 1'b0;
        r_q <= '0;
      end else begin
        if (src_req[e].ar_valid) begin
          if (src_req[e].ar.size != 3'd6 || src_req[e].ar.addr[5:0] != 0)
            note_error($sformatf("engine %0d reads %h size %0d", e, src_req[e].ar.addr, src_req[e].ar.size));
          pend.push_back('{src_req[e].ar, cycle + src_lat});
        end
        // the next beat once the current one is taken (or none is shown)
        if (!r_valid_q || src_req[e].r_ready) begin
          if (pend.size() > 0 && cycle >= pend[0].due) begin
            r_valid_q <= 1'b1;
            r_q <= '{id: pend[0].ar.id, data: src_data(pend[0].ar.addr + addr_t'(64 * beat)),
                     resp: axi_pkg::RESP_OKAY, last: (beat == pend[0].ar.len), user: '0};
            if (beat == pend[0].ar.len) begin
              void'(pend.pop_front());
              beat = 0;
            end else begin
              beat = beat + 1;
            end
          end else begin
            r_valid_q <= 1'b0;
          end
        end
        if (src_req[e].aw_valid) note_error($sformatf("engine %0d writes its source port", e));
      end
    end
  end

  // ======================================================================= C: the compute chip
  axi_req_t c_to_remote_req, c_from_remote_req;
  axi_resp_t c_to_remote_rsp, c_from_remote_rsp;

  // ---------------------------------------------------------------------- the link
  localparam int unsigned PhyIoWidth = 15 + 1 + 2 + 2;
  wire [2:0][PhyIoWidth-1:0] m_west_io, c_east_io;
  logic m_rts, m_cts, c_rts, c_cts, m_test, c_test;

  hemaia_d2d_link #(
      .SendFifoDepth (4),
      .RecvFifoDepth (8),
      .EnableEastPhy (0),
      .EnableWestPhy (1),
      .EnableNorthPhy(0),
      .EnableSouthPhy(0),
      .NumLocalPorts (NumPorts),
      .chip_id_t     (chip_id_t),
      .axi_req_t     (axi_req_t),
      .axi_rsp_t     (axi_resp_t),
      .axi_lite_req_t(lite_req_t),
      .axi_lite_rsp_t(lite_rsp_t),
      .aw_chan_t     (axi_aw_chan_t),
      .ar_chan_t     (axi_ar_chan_t),
      .r_chan_t      (axi_r_chan_t),
      .w_chan_t      (axi_w_chan_t),
      .b_chan_t      (axi_b_chan_t)
  ) i_link_m (
      .control_clk_i              (clk),
      .control_rst_ni             (rst_n),
      .digital_clk_i              (clk),
      .east_phy_tx_clk_i          (phy_clk),
      .west_phy_tx_clk_i          (phy_clk),
      .north_phy_tx_clk_i         (phy_clk),
      .south_phy_tx_clk_i         (phy_clk),
      .rst_ni                     (rst_n),
      .chip_id_i                  (ChipM),
      .to_remote_req_i            (m_to_remote_req),
      .to_remote_rsp_o            (m_to_remote_rsp),
      .from_remote_req_o          (m_from_remote_req),
      .from_remote_rsp_i          (m_from_remote_rsp),
      .axi_lite_req_i             (lite_req[0]),
      .axi_lite_rsp_o             (lite_rsp[0]),
      .east_test_being_requested_i(1'b0),
      .east_test_request_o        (),
      .flow_control_east_rts_o    (),
      .flow_control_east_cts_i    (1'b0),
      .flow_control_east_rts_i    (1'b0),
      .flow_control_east_cts_o    (),
      .east_phy_io                (),
      .west_test_being_requested_i(c_test),
      .west_test_request_o        (m_test),
      .flow_control_west_rts_o    (m_rts),
      .flow_control_west_cts_i    (c_cts),
      .flow_control_west_rts_i    (c_rts),
      .flow_control_west_cts_o    (m_cts),
      .west_phy_io                (m_west_io),
      .north_test_being_requested_i(1'b0),
      .north_test_request_o       (),
      .flow_control_north_rts_o   (),
      .flow_control_north_cts_i   (1'b0),
      .flow_control_north_rts_i   (1'b0),
      .flow_control_north_cts_o   (),
      .north_phy_io               (),
      .south_test_being_requested_i(1'b0),
      .south_test_request_o       (),
      .flow_control_south_rts_o   (),
      .flow_control_south_cts_i   (1'b0),
      .flow_control_south_rts_i   (1'b0),
      .flow_control_south_cts_o   (),
      .south_phy_io               ()
  );

  hemaia_d2d_link #(
      .SendFifoDepth (4),
      .RecvFifoDepth (8),
      .EnableEastPhy (1),
      .EnableWestPhy (0),
      .EnableNorthPhy(0),
      .EnableSouthPhy(0),
      .NumLocalPorts (1),
      .chip_id_t     (chip_id_t),
      .axi_req_t     (axi_req_t),
      .axi_rsp_t     (axi_resp_t),
      .axi_lite_req_t(lite_req_t),
      .axi_lite_rsp_t(lite_rsp_t),
      .aw_chan_t     (axi_aw_chan_t),
      .ar_chan_t     (axi_ar_chan_t),
      .r_chan_t      (axi_r_chan_t),
      .w_chan_t      (axi_w_chan_t),
      .b_chan_t      (axi_b_chan_t)
  ) i_link_c (
      .control_clk_i              (clk),
      .control_rst_ni             (rst_n),
      .digital_clk_i              (clk),
      .east_phy_tx_clk_i          (phy_clk),
      .west_phy_tx_clk_i          (phy_clk),
      .north_phy_tx_clk_i         (phy_clk),
      .south_phy_tx_clk_i         (phy_clk),
      .rst_ni                     (rst_n),
      .chip_id_i                  (ChipC),
      .to_remote_req_i            (c_to_remote_req),
      .to_remote_rsp_o            (c_to_remote_rsp),
      .from_remote_req_o          (c_from_remote_req),
      .from_remote_rsp_i          (c_from_remote_rsp),
      .axi_lite_req_i             (lite_req[1]),
      .axi_lite_rsp_o             (lite_rsp[1]),
      .east_test_being_requested_i(m_test),
      .east_test_request_o        (c_test),
      .flow_control_east_rts_o    (c_rts),
      .flow_control_east_cts_i    (m_cts),
      .flow_control_east_rts_i    (m_rts),
      .flow_control_east_cts_o    (c_cts),
      .east_phy_io                (c_east_io),
      .west_test_being_requested_i(1'b0),
      .west_test_request_o        (),
      .flow_control_west_rts_o    (),
      .flow_control_west_cts_i    (1'b0),
      .flow_control_west_rts_i    (1'b0),
      .flow_control_west_cts_o    (),
      .west_phy_io                (),
      .north_test_being_requested_i(1'b0),
      .north_test_request_o       (),
      .flow_control_north_rts_o   (),
      .flow_control_north_cts_i   (1'b0),
      .flow_control_north_rts_i   (1'b0),
      .flow_control_north_cts_o   (),
      .north_phy_io               (),
      .south_test_being_requested_i(1'b0),
      .south_test_request_o       (),
      .flow_control_south_rts_o   (),
      .flow_control_south_cts_i   (1'b0),
      .flow_control_south_rts_i   (1'b0),
      .flow_control_south_cts_o   (),
      .south_phy_io               ()
  );

  for (genvar ch = 0; ch < 3; ch++) begin : gen_phy_wire
    for (genvar b = 0; b < PhyIoWidth; b++) begin : gen_bit
      tran i_bit (m_west_io[ch][b], c_east_io[ch][b]);
    end
  end

  // who drives the link, per PHY cycle
  wire m_tx = i_link_m.gen_west_phy_enabled.gen_phy_channels[0].i_phy_interface_west.phy_tx_enable;
  wire c_tx = i_link_c.gen_east_phy_enabled.gen_phy_channels[0].i_phy_interface_east.phy_tx_enable;
  bit measuring;
  longint unsigned phy_cycles, m_tx_cycles, c_tx_cycles;
  always @(posedge phy_clk) begin
    if (rst_n && measuring) begin
      phy_cycles++;
      if (m_tx) m_tx_cycles++;
      if (c_tx) c_tx_cycles++;
    end
  end

  // ---------------------------------------------------------------------- AXI-Lite (link setup)
  task automatic lite_write(input int unsigned c, input logic [47:0] addr, input logic [31:0] data);
    bit aw_done, w_done;
    aw_done = 0;
    w_done = 0;
    @(posedge clk);
    #ApplTime;
    lite_req[c].aw.addr = addr;
    lite_req[c].aw.prot = '0;
    lite_req[c].aw_valid = 1'b1;
    lite_req[c].w.data = data;
    lite_req[c].w.strb = '1;
    lite_req[c].w_valid = 1'b1;
    lite_req[c].b_ready = 1'b1;
    while (!(aw_done && w_done)) begin
      #(TestTime - ApplTime);
      if (lite_rsp[c].aw_ready) aw_done = 1;
      if (lite_rsp[c].w_ready) w_done = 1;
      @(posedge clk);
      #ApplTime;
      if (aw_done) lite_req[c].aw_valid = 1'b0;
      if (w_done) lite_req[c].w_valid = 1'b0;
    end
    while (!lite_rsp[c].b_valid) begin
      @(posedge clk);
      #ApplTime;
    end
    @(posedge clk);
    #ApplTime;
    lite_req[c].b_ready = 1'b0;
  endtask

  // ======================================================================= C's sink
  // Every write into C: checked beat by beat against the source pattern; each run lands when its
  // last byte has.
  typedef struct {
    axi_aw_chan_t aw;
    int unsigned beat;
  } open_t;
  open_t open_w[$];
  axi_b_chan_t b_q[$];
  int unsigned n_open;
  logic b_valid_q;
  axi_b_chan_t b_head_q;
  longint unsigned landed_bytes[NumPorts][];
  longint unsigned landed_at[NumPorts][];
  int unsigned runs_landed[NumPorts];
  longint unsigned sink_bytes, last_land;

  always_comb begin
    c_from_remote_rsp = '0;
    c_from_remote_rsp.aw_ready = 1'b1;
    // a W may come with its own AW
    c_from_remote_rsp.w_ready = (n_open > 0) || c_from_remote_req.aw_valid;
    c_from_remote_rsp.ar_ready = 1'b0;
    c_from_remote_rsp.b_valid = b_valid_q;
    c_from_remote_rsp.b = b_head_q;
  end

  initial begin
    n_open = 0;
    b_valid_q = 1'b0;
    b_head_q = '0;
  end

  always @(posedge clk) begin
    if (rst_n) begin
      if (c_from_remote_req.aw_valid) begin
        open_w.push_back('{c_from_remote_req.aw, 0});
        n_open++;
      end
      if (c_from_remote_req.w_valid && c_from_remote_rsp.w_ready) begin
        automatic open_t o = open_w[0];
        automatic addr_t a = o.aw.addr + addr_t'(64 * o.beat);
        automatic longint unsigned rel = longint'(a[39:0] - DstBase);
        automatic int unsigned e = rel / RegionBytes;
        automatic longint unsigned off = rel % RegionBytes;
        automatic int unsigned r = off / run_bytes;
        if (a[47:40] != ChipC || e >= engines || r >= runs)
          note_error($sformatf("sink: write to %h is no run's", a));
        else begin
          if (c_from_remote_req.w.data !== src_data(src_addr(e, off)))
            note_error($sformatf("sink: engine %0d run %0d offset %0d: data mismatch", e, r, off % run_bytes));
          if (c_from_remote_req.w.strb !== '1) note_error($sformatf("sink: partial strobe at %h", a));
          landed_bytes[e][r] += 64;
          if (landed_bytes[e][r] == run_bytes) begin
            landed_at[e][r] = cycle;
            runs_landed[e]++;
            last_land = cycle;
          end
        end
        sink_bytes += 64;
        if (c_from_remote_req.w.last != (o.beat == o.aw.len))
          note_error($sformatf("sink: W last wrong at beat %0d of %0d", o.beat, o.aw.len + 1));
        if (o.beat == o.aw.len) begin
          void'(open_w.pop_front());
          n_open--;
          b_q.push_back('{id: o.aw.id, resp: axi_pkg::RESP_OKAY, user: '0});
        end else begin
          open_w[0].beat = o.beat + 1;
        end
      end
      // the B shown is taken: show the next
      if (!b_valid_q || c_from_remote_req.b_ready) begin
        if (b_q.size() > 0) begin
          b_valid_q <= 1'b1;
          b_head_q <= b_q.pop_front();
        end else begin
          b_valid_q <= 1'b0;
        end
      end
      if (c_from_remote_req.ar_valid) note_error("sink: C was read from the link");
    end
  end

  // ======================================================================= the hosts on C
  // One AXI master (C's local port) shared by the host models: a host holds a channel while it
  // drives it. Host h uses AXI ID h. host_path cycles pass before a request reaches the port and
  // after a response leaves it (the host core to C's link and back).
  semaphore aw_lock = new(1), ar_lock = new(1);
  int unsigned b_owed[NumPorts];
  data32_t r_box[NumPorts][$];

  int unsigned read_lane[NumPorts];

  initial begin
    c_to_remote_req = '0;
    c_to_remote_req.b_ready = 1'b1;
    c_to_remote_req.r_ready = 1'b1;
  end

  always @(posedge clk) begin
    if (rst_n) begin
      if (c_to_remote_rsp.b_valid) begin
        automatic int unsigned h = c_to_remote_rsp.b.id;
        if (c_to_remote_rsp.b.resp != axi_pkg::RESP_OKAY) note_error($sformatf("host %0d: B error", h));
        if (b_owed[h] == 0) note_error($sformatf("host %0d: B it did not wait for", h));
        else b_owed[h]--;
      end
      if (c_to_remote_rsp.r_valid) begin
        automatic int unsigned h = c_to_remote_rsp.r.id;
        if (c_to_remote_rsp.r.resp != axi_pkg::RESP_OKAY) note_error($sformatf("host %0d: R error", h));
        r_box[h].push_back(read_lane[h] < 16 ? c_to_remote_rsp.r.data[32*read_lane[h]+:32] : 32'hDEAD);
      end
    end
  end
  // a store of `bytes` (4 or 8) at addr, posted: the host goes on after its AW and W
  task automatic host_store(input int unsigned h, input addr_t addr, input logic [63:0] data,
                            input int unsigned bytes);
    automatic bit aw_done = 0, w_done = 0;
    automatic int unsigned lane = addr[5:0];
    repeat (host_path) @(posedge clk);
    aw_lock.get();
    @(posedge clk);
    #ApplTime;
    c_to_remote_req.aw = '0;
    c_to_remote_req.aw.id = id_t'(h);
    c_to_remote_req.aw.addr = addr;
    c_to_remote_req.aw.len = '0;
    c_to_remote_req.aw.size = (bytes == 8) ? 3'd3 : 3'd2;
    c_to_remote_req.aw.burst = axi_pkg::BURST_INCR;
    c_to_remote_req.aw.cache = axi_pkg::CACHE_MODIFIABLE;  // as CVA6 sets it
    c_to_remote_req.aw_valid = 1'b1;
    c_to_remote_req.w = '0;
    c_to_remote_req.w.data = data_t'(data) << (8 * lane);
    c_to_remote_req.w.strb = strb_t'((bytes == 8) ? 8'hFF : 8'h0F) << lane;
    c_to_remote_req.w.last = 1'b1;
    c_to_remote_req.w_valid = 1'b1;
    b_owed[h]++;
    while (!(aw_done && w_done)) begin
      #(TestTime - ApplTime);
      if (c_to_remote_rsp.aw_ready) aw_done = 1;
      if (c_to_remote_rsp.w_ready) w_done = 1;
      @(posedge clk);
      #ApplTime;
      if (aw_done) c_to_remote_req.aw_valid = 1'b0;
      if (w_done) c_to_remote_req.w_valid = 1'b0;
    end
    aw_lock.put();
  endtask

  // a 4-byte load: the host waits for its data
  task automatic host_load(input int unsigned h, input addr_t addr, output data32_t data);
    repeat (host_path) @(posedge clk);
    ar_lock.get();
    @(posedge clk);
    #ApplTime;
    c_to_remote_req.ar = '0;
    c_to_remote_req.ar.id = id_t'(h);
    c_to_remote_req.ar.addr = addr;
    c_to_remote_req.ar.len = '0;
    c_to_remote_req.ar.size = 3'd2;
    c_to_remote_req.ar.burst = axi_pkg::BURST_INCR;
    c_to_remote_req.ar.cache = axi_pkg::CACHE_MODIFIABLE;
    c_to_remote_req.ar_valid = 1'b1;
    read_lane[h] = addr[5:2];
    begin
      automatic bit done = 0;
      while (!done) begin
        #(TestTime - ApplTime);
        if (c_to_remote_rsp.ar_ready) done = 1;
        @(posedge clk);
        #ApplTime;
      end
    end
    c_to_remote_req.ar_valid = 1'b0;
    ar_lock.put();
    wait (r_box[h].size() > 0);
    data = r_box[h].pop_front();
    repeat (host_path) @(posedge clk);
  endtask

  // every store of host h acknowledged
  task automatic host_fence(input int unsigned h);
    wait (b_owed[h] == 0);
  endtask

  // ---------------------------------------------------------------------- the host loops
  longint unsigned t_first_launch, t_wr[NumPorts], t_go[NumPorts], n_readback[NumPorts];
  int unsigned posted[NumPorts];

  task automatic host_loop(input int unsigned h);
    data32_t v;
    for (int unsigned r = 0; r < runs; r++) begin
      automatic addr_t src = src_addr(h, longint'(r) * run_bytes);
      automatic addr_t dst = dst_addr(h, longint'(r) * run_bytes);
      automatic longint unsigned t0, t1, t2;
      repeat (pick) @(posedge clk);
      if (mode == 0) begin
        t0 = cycle;
        if (t0 < t_first_launch) t_first_launch = t0;
        host_store(h, reg_addr(h, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_DST_ADDR_LOW_OFFSET), dst[31:0], 4);
        host_store(h, reg_addr(h, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_DST_ADDR_HIGH_OFFSET), 32'(dst[47:32]), 4);
        host_store(h, reg_addr(h, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_SRC_ADDR_LOW_OFFSET), src[31:0], 4);
        host_store(h, reg_addr(h, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_SRC_ADDR_HIGH_OFFSET), 32'(src[47:32]), 4);
        host_store(h, reg_addr(h, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_LENGTH_HIGH_OFFSET), 32'h0, 4);
        host_store(h, reg_addr(h, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_CONF_OFFSET), 32'h0, 4);
        host_store(h, reg_addr(h, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_LENGTH_LOW_OFFSET), 32'(run_bytes), 4);
        host_fence(h);
        do begin
          host_load(h, reg_addr(h, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_LENGTH_LOW_OFFSET), v);
          n_readback[h]++;
        end while (v != 32'(run_bytes));
        t1 = cycle;
        host_load(h, reg_addr(h, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_NEXT_ID_0_OFFSET), v);
        t2 = cycle;
        host_store(h, reg_addr(h, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_LENGTH_LOW_OFFSET), 32'h0, 4);
        t_wr[h] += t1 - t0;
        t_go[h] += t2 - t1;
      end else begin
        wait (posted[h] - runs_landed[h] < inflight_max);
        t0 = cycle;
        if (t0 < t_first_launch) t_first_launch = t0;
        host_store(h, reg_addr(h, 12'h800), 64'(src), 8);
        host_store(h, reg_addr(h, 12'h808), 64'(dst), 8);
        host_store(h, reg_addr(h, 12'h810), 64'(run_bytes), 8);
        host_store(h, reg_addr(h, 12'h818), 64'h0, 8);
        posted[h]++;
        if (debug && r == 0) begin
          data32_t s_lo, s_hi, l_lo, cnt, iss;
          host_fence(h);
          host_load(h, reg_addr(h, 12'h800), s_lo);
          host_load(h, reg_addr(h, 12'h804), s_hi);
          host_load(h, reg_addr(h, 12'h810), l_lo);
          host_load(h, reg_addr(h, 12'h820), cnt);
          host_load(h, reg_addr(h, 12'h824), iss);
          $display("PUSH_TB debug host %0d after the first post: src %h_%h (want %h) len %0d (want %0d) queued %0d issued %0d",
                   h, s_hi, s_lo, src, l_lo, run_bytes, cnt, iss);
        end
        t_wr[h] += cycle - t0;
      end
    end
  endtask

  // ======================================================================= the test
  initial begin
    longint unsigned t_end, total, cyc;
    real bpc, gbs;
    data32_t v;

    if (!$value$plusargs("mode=%d", mode)) mode = 0;
    if (!$value$plusargs("engines=%d", engines)) engines = 1;
    if (!$value$plusargs("run_chunks=%d", run_chunks)) run_chunks = 4;
    if (!$value$plusargs("runs=%d", runs)) runs = 16;
    if (!$value$plusargs("pick=%d", pick)) pick = 2150;
    if (!$value$plusargs("inflight=%d", inflight_max)) inflight_max = 4;
    if (!$value$plusargs("host_path=%d", host_path)) host_path = 30;
    if (!$value$plusargs("rx_yield=%d", rx_yield)) rx_yield = 0;
    if (!$value$plusargs("src_lat=%d", src_lat)) src_lat = 53;
    if (!$value$plusargs("tag=%s", tag)) tag = "-";
    debug = $test$plusargs("debug");
    if (engines < 1 || engines > NumPorts) $fatal(1, "+engines=%0d: 1 or %0d", engines, NumPorts);
    run_bytes = longint'(run_chunks) * ChunkBytes;
    // +run_bytes=N runs N bytes (a multiple of 64) instead of run_chunks chunks: flag-sized runs
    if ($value$plusargs("run_bytes=%d", run_bytes_arg)) run_bytes = run_bytes_arg;
    if (run_bytes == 0 || run_bytes % 64 != 0) $fatal(1, "run bytes %0d: a non-zero multiple of 64", run_bytes);

    errors = 0;
    measuring = 0;
    sink_bytes = 0;
    last_land = 0;
    t_first_launch = '1;
    phy_cycles = 0;
    m_tx_cycles = 0;
    c_tx_cycles = 0;
    for (int unsigned e = 0; e < NumPorts; e++) begin
      landed_bytes[e] = new[runs];
      landed_at[e] = new[runs];
      for (int unsigned r = 0; r < runs; r++) landed_bytes[e][r] = 0;
      runs_landed[e] = 0;
      b_owed[e] = 0;
      t_wr[e] = 0;
      t_go[e] = 0;
      n_readback[e] = 0;
      posted[e] = 0;
    end
    lite_req[0] = '0;
    lite_req[1] = '0;

    @(posedge rst_n);
    // DDR on both ends (what the boot code does), and C's RX yield window
    fork
      lite_write(0, 48'(hemaia_d2d_link_reg_pkg::HEMAIA_D2D_LINK_PHY_DDR_MODE_REGISTER_OFFSET), 32'h0F);
      lite_write(1, 48'(hemaia_d2d_link_reg_pkg::HEMAIA_D2D_LINK_PHY_DDR_MODE_REGISTER_OFFSET), 32'h0F);
    join
    if (rx_yield != 0)
      lite_write(1, 48'(hemaia_d2d_link_reg_pkg::HEMAIA_D2D_LINK_RX_YIELD_WINDOW_REGISTER_OFFSET), 32'(rx_yield));
    repeat (200) @(posedge clk);

    // one uncontended register read: the bare round trip
    begin
      automatic longint unsigned t0 = cycle;
      host_load(0, reg_addr(0, idma_reg64_1d_reg_pkg::IDMA_REG64_1D_DONE_ID_0_OFFSET), v);
      $display("PUSH_TB idle register read round trip: %0d cycles (host_path %0d each way)", cycle - t0, host_path);
    end
    if (mode == 1) begin
      for (int unsigned h = 0; h < engines; h++) begin
        host_load(h, reg_addr(h, 12'h828), v);
        if (v != 32) note_error($sformatf("engine %0d queue depth reads %0d", h, v));
      end
    end

    $display("PUSH_TB start [%s]: %s, %0d engine(s), runs of %0d chunk(s) (%0d B), %0d runs each, pick %0d, host_path %0d, inflight %0d, rx_yield %0d",
             tag, mode ? "queue" : "register launch", engines, run_chunks, run_bytes, runs, pick, host_path,
             inflight_max, rx_yield);
    measuring = 1;
    for (int unsigned h = 0; h < engines; h++) begin
      automatic int unsigned hh = h;
      fork
        host_loop(hh);
      join_none
    end
    begin
      automatic longint unsigned limit = cycle + 200 + longint'(runs) * engines * (run_bytes / 8 + pick + 20000);
      bit all_done;
      do begin
        @(posedge clk);
        all_done = 1;
        for (int unsigned h = 0; h < engines; h++) if (runs_landed[h] < runs) all_done = 0;
      end while (!all_done && cycle < limit);
      if (!all_done) note_error("timeout: not every run landed");
    end
    measuring = 0;
    repeat (100) @(posedge clk);

    // results
    t_end = last_land;
    cyc = t_end - t_first_launch;
    total = sink_bytes;
    bpc = real'(total) / real'(cyc);
    gbs = bpc * 0.5;  // B a cycle at 500 MHz
    for (int unsigned h = 0; h < engines; h++) begin
      if (mode == 0)
        $display("PUSH_TB host %0d: %0d runs, per run %.0f cycles of writes + read-back (%.2f reads), %.0f cycles launching read",
                 h, runs, real'(t_wr[h]) / runs, real'(n_readback[h]) / runs, real'(t_go[h]) / runs);
      else
        $display("PUSH_TB host %0d: %0d runs, per run %.0f cycles of posting", h, runs, real'(t_wr[h]) / runs);
    end
    if (mode == 1) begin
      for (int unsigned h = 0; h < engines; h++) begin
        host_load(h, reg_addr(h, 12'h824), v);
        if (v != runs) note_error($sformatf("engine %0d queue issued %0d of %0d", h, v, runs));
      end
    end
    $display("PUSH_TB RESULT tag=%s mode=%s engines=%0d run_bytes=%0d runs=%0d pick=%0d host_path=%0d rx_yield=%0d bytes=%0d cycles=%0d B_per_cycle=%.2f GBps=%.2f link_util=%.3f m_tx=%.3f c_tx=%.3f errors=%0d",
             tag, mode ? "queue" : "register", engines, run_bytes, runs, pick, host_path, rx_yield,
             total, cyc, bpc, gbs, bpc / 64.0, real'(m_tx_cycles) / real'(phy_cycles),
             real'(c_tx_cycles) / real'(phy_cycles), errors);
    $display("PUSH_TB %s: %0d error(s)", errors ? "FAIL" : "PASS", errors);
    $finish;
  end

endmodule
