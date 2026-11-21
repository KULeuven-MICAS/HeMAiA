// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

module hemaia_vpk180_dram_bridge
  import occamy_pkg::*;
(
  input  logic host_clk_i,
  input  logic chip_rst_ni,
  input  chip_id_t chip_id_i,
  input  external_memory_req_t host_req_i,
  output external_memory_resp_t host_resp_o,

  output logic m_axi_aclk_o,
  output logic [47:0] m_axi_awaddr,
  output external_memory_id_t m_axi_awid,
  output logic [7:0] m_axi_awlen,
  output logic [2:0] m_axi_awsize,
  output logic [1:0] m_axi_awburst,
  output logic m_axi_awlock,
  output logic [3:0] m_axi_awcache,
  output logic [2:0] m_axi_awprot,
  output logic [3:0] m_axi_awqos,
  output logic m_axi_awvalid,
  input  logic m_axi_awready,
  output logic [511:0] m_axi_wdata,
  output logic [63:0] m_axi_wstrb,
  output logic m_axi_wlast,
  output logic m_axi_wvalid,
  input  logic m_axi_wready,
  input  external_memory_id_t m_axi_bid,
  input  logic [1:0] m_axi_bresp,
  input  logic m_axi_bvalid,
  output logic m_axi_bready,
  output logic [47:0] m_axi_araddr,
  output external_memory_id_t m_axi_arid,
  output logic [7:0] m_axi_arlen,
  output logic [2:0] m_axi_arsize,
  output logic [1:0] m_axi_arburst,
  output logic m_axi_arlock,
  output logic [3:0] m_axi_arcache,
  output logic [2:0] m_axi_arprot,
  output logic [3:0] m_axi_arqos,
  output logic m_axi_arvalid,
  input  logic m_axi_arready,
  input  external_memory_id_t m_axi_rid,
  input  logic [511:0] m_axi_rdata,
  input  logic [1:0] m_axi_rresp,
  input  logic m_axi_rlast,
  input  logic m_axi_rvalid,
  output logic m_axi_rready
);
  if (ExternalMemoryBase != 48'h0020_0000_0000 ||
      ExternalMemorySize != 48'h0003_0000_0000 || AddrWidth != 48 || ChipIdWidth != 8) begin : gen_bad_cfg
    initial $fatal(1, "VPK180 DRAM requires a 48-bit AXI address and 12 GiB at 0x??20_0000_0000");
  end

  // The SoC samples the same strap while chip_rst_ni is low.
  chip_id_t chip_id;
  always_latch begin
    if (!chip_rst_ni) chip_id <= chip_id_i;
  end

  addr_t aw_addr, ar_addr;
  hemaia_dram_address_mapper i_address_mapper (
    .chip_id_i (chip_id),
    .aw_addr_i (host_req_i.aw.addr),
    .ar_addr_i (host_req_i.ar.addr),
    .aw_addr_o (aw_addr),
    .ar_addr_o (ar_addr)
  );

  external_memory_req_t noc_req;
  external_memory_resp_t noc_resp;
  axi_modify_address #(
    .slv_req_t   (external_memory_req_t),
    .mst_addr_t  (addr_t),
    .mst_req_t   (external_memory_req_t),
    .axi_resp_t  (external_memory_resp_t)
  ) i_remap (
    .slv_req_i     (host_req_i),
    .slv_resp_o    (host_resp_o),
    .mst_aw_addr_i (aw_addr),
    .mst_ar_addr_i (ar_addr),
    .mst_req_o     (noc_req),
    .mst_resp_i    (noc_resp)
  );

  // The SoC wide crossbar excludes ATOP transactions. The full PL NMU is 512-bit AXI4.
  // Forward the source clock to its AXI interface. The NMU crosses into the hard NoC clock.
  assign m_axi_aclk_o = host_clk_i;
  assign m_axi_awaddr  = noc_req.aw.addr;
  assign m_axi_awid    = noc_req.aw.id;
  assign m_axi_awlen   = noc_req.aw.len;
  assign m_axi_awsize  = noc_req.aw.size;
  assign m_axi_awburst = noc_req.aw.burst;
  assign m_axi_awlock  = noc_req.aw.lock;
  assign m_axi_awcache = noc_req.aw.cache;
  assign m_axi_awprot  = noc_req.aw.prot;
  assign m_axi_awqos   = noc_req.aw.qos;
  assign m_axi_awvalid = noc_req.aw_valid;
  assign m_axi_wdata   = noc_req.w.data;
  assign m_axi_wstrb   = noc_req.w.strb;
  assign m_axi_wlast   = noc_req.w.last;
  assign m_axi_wvalid  = noc_req.w_valid;
  assign m_axi_bready  = noc_req.b_ready;
  assign m_axi_araddr  = noc_req.ar.addr;
  assign m_axi_arid    = noc_req.ar.id;
  assign m_axi_arlen   = noc_req.ar.len;
  assign m_axi_arsize  = noc_req.ar.size;
  assign m_axi_arburst = noc_req.ar.burst;
  assign m_axi_arlock  = noc_req.ar.lock;
  assign m_axi_arcache = noc_req.ar.cache;
  assign m_axi_arprot  = noc_req.ar.prot;
  assign m_axi_arqos   = noc_req.ar.qos;
  assign m_axi_arvalid = noc_req.ar_valid;
  assign m_axi_rready  = noc_req.r_ready;

  assign noc_resp = '{
    aw_ready: m_axi_awready,
    w_ready:  m_axi_wready,
    b_valid:  m_axi_bvalid,
    b: '{id: m_axi_bid, resp: m_axi_bresp, user: '0},
    ar_ready: m_axi_arready,
    r_valid:  m_axi_rvalid,
    r: '{id: m_axi_rid, data: m_axi_rdata, resp: m_axi_rresp, last: m_axi_rlast, user: '0}
  };
endmodule
