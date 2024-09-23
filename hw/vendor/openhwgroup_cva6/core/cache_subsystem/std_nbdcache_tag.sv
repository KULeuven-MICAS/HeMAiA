// Copyright 2024 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Fanchen Kong <fanchen.kong@kuleuven.be>

// The actual data memory. This memory is made into a module
// to support multiple power domain needed by the floor plan tool
(* no_ungroup *)
(* no_boundary_optimization *)

module std_nbdcache_tag #(
    parameter int unsigned NumWords   = 32'd1024, // Number of Words in
    parameter int unsigned DataWidth  = 32'd128,  // Data signal width
    parameter int unsigned ByteWidth  = 32'd8,    // Width of a data byte
    parameter int unsigned WAY_COUNT  = 32'd1, 
    parameter int unsigned Latency    = 32'd1,    // Latency when the read data is available
    parameter              SimInit      = "none",   // Simulation 
    parameter bit          PrintSimCfg  = 1'b0,     // Print configuration
    parameter type         impl_in_t  = logic,    // Type for implementation inputs
    parameter type         impl_out_t = logic,
    // DEPENDENT PARAMETERS, DO NOT OVERWRITE!
    parameter int unsigned AddrWidth  = (NumWords > 32'd1) ? $clog2(NumWords) : 32'd1,
    parameter int unsigned BeWidth    = (DataWidth + ByteWidth - 32'd1) / ByteWidth, // ceil_div
    parameter type         addr_t     = logic [AddrWidth-1:0],
    parameter type         data_t     = logic [DataWidth-1:0],
    parameter type         be_t       = logic [BeWidth-1:0]

) (
    input  logic                 clk_i,      // Clock
    input  logic                 rst_ni,     // Asynchronous reset active low
    // implementation-related IO
    input  impl_in_t             impl_i,
    output impl_out_t            impl_o,
    // input ports
    input  logic [WAY_COUNT-1:0] req_i,      // request
    input  logic                 we_i,       // write enable
    input  addr_t                addr_i,     // request address
    input  data_t                wdata_i,    // write data
    input  be_t                  be_i,       // write byte enable
    // output ports
    output data_t                rdata_o [WAY_COUNT-1:0]   // read data
);
  for (genvar i = 0; i < WAY_COUNT; i++) begin : gen_sram_std_nbdcache_tag
    // Tag RAM
    tc_sram_impl #(
      .impl_in_t ( impl_in_t ),
      .impl_out_t( impl_out_t),
      .DataWidth ( DataWidth ),
      .NumWords  ( NumWords  ),
      .NumPorts  ( 1         ),
      .Latency   ( 1         ),
      .SimInit   ( SimInit   ),
      .PrintSimCfg ( PrintSimCfg )
    ) tag_sram (
      .clk_i     ( clk_i                    ),
      .rst_ni    ( rst_ni                   ),
      .impl_i    ( impl_i                   ),
      .impl_o    ( /*Not Used*/             ),
      .req_i     ( req_i[i]                 ),
      .we_i      ( we_i                     ),
      .addr_i    ( addr_i                   ),
      .wdata_i   ( wdata_i                  ),
      .be_i      ( be_i                     ),
      .rdata_o   ( rdata_o[i]               )
    );
  end
    
endmodule