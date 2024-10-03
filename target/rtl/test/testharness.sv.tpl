// Copyright 2020 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

`include "axi/typedef.svh"

task automatic spi_read(input logic [63:0] addr, input integer length);
  // Inputs:
  //   addr   - 64-bit Address to read from
  //   length - Number of bytes to read
  // Output:
  //   data   - Array to store read data

  reg [7:0] cmd;  // SPI read command code
  integer i, j;
  reg [3:0] mosi_data;  // Data to send over SPI (master out)
  reg [3:0] miso_data;  // Data received from SPI (slave out)

  // Set the SPI Quad IO Fast Read command code (example: 0xEB)
  cmd        = 8'hEB;

  // Initialize signals
  spis_csb_i = 1;  // CSB is high (inactive)
  spis_sd_i  = '1;  // Data lines are high impedance

  // Wait for some time before starting
  #10;

  // Bring CSB low to start the transaction
  spis_csb_i = 0;

  // Wait for a clock edge to align
  @(posedge spis_sck_i);

  // Send the command code (8 bits) over 4 data lines (2 clock cycles)
  for (i = 7; i >= 0; i -= 4) begin
    @(negedge spis_sck_i);
    if (i >= 3) begin
      mosi_data = cmd[i-:4];
    end else begin
      // For i = 3 to 0
      mosi_data = cmd[i:0];
      mosi_data = mosi_data << (3 - i);  // Left-align to 4 bits
    end
    spis_sd_i = mosi_data;  // Drive data lines
  end

  // Send the 64-bit address over 4 data lines (16 clock cycles)
  for (i = 63; i >= 0; i -= 4) begin
    @(negedge spis_sck_i);
    if (i >= 3) begin
      mosi_data = addr[i-:4];
    end else begin
      // For i = 3 to 0
      mosi_data = addr[i:0];
      mosi_data = mosi_data << (3 - i);  // Left-align to 4 bits
    end
    spis_sd_i = mosi_data;  // Drive data lines
  end

  // Insert dummy cycles if required (e.g., 32 cycles)
  for (i = 0; i < 32; i = i + 1) begin
    @(posedge spis_sck_i);
    // Do nothing, just wait
  end

  // Now read the data from the slave
  for (i = 0; i < length; i = i + 1) begin
    reg [7:0] byte_data;
    for (j = 7; j >= 0; j -= 4) begin
      @(posedge spis_sck_i);
      miso_data = spis_sd_o;  // Read 4 bits from slave
      if (j >= 3) begin
        byte_data[j-:4] = miso_data;
      end else begin
        // For j = 3 to 0
        byte_data[j:0] = miso_data >> (3 - j);
      end
    end

    $display("Read byte %0d: %h", i, byte_data);  // Print the byte to the console
  end

  // Bring CSB high to end the transaction
  @(negedge spis_sck_i);
  spis_csb_i = 1;
endtask

module testharness import occamy_pkg::*; (
  input  logic        clk_i,
  input  logic        rst_ni
);

  // verilog_lint: waive explicit-parameter-storage-type
  localparam RTCTCK = 30.518us; // 32.768 kHz
  localparam SPITCK = 15.15ns; // SPI clock

  logic rtc_i;

  // Generate reset and clock.
  initial begin
    forever begin
      rtc_i = 1;
      #(RTCTCK/2);
      rtc_i = 0;
      #(RTCTCK/2);
    end
  end

  logic clk_periph_i, rst_periph_ni;
  assign clk_periph_i = clk_i;
  assign rst_periph_ni = rst_ni;

  // Generate SPI Clock.
  logic spis_sck_i;
  initial begin
    spis_sck_i = 0;
    forever begin
      #(SPITCK/2);
      spis_sck_i = ~spis_sck_i;
    end
  end

  // Generate logic to connect to the SPI device.
  logic spis_csb_i = '0;
  logic [3:0] spis_sd_o;
  logic [3:0] spis_sd_en_o;
  logic [3:0] spis_sd_i = '1;

<%def name="tb_memory(bus, name)">
  ${bus.req_type()} ${name}_req;
  ${bus.rsp_type()} ${name}_rsp;

  % if isinstance(bus, solder.AxiBus):
  tb_memory_axi #(
    .AxiAddrWidth (${bus.aw}),
    .AxiDataWidth (${bus.dw}),
    .AxiIdWidth (${bus.iw}),
    .AxiUserWidth (${bus.uw + 1}),
    .ATOPSupport (0),
  % else:
  tb_memory_regbus #(
    .AddrWidth (${bus.aw}),
    .DataWidth (${bus.dw}),
  % endif
    .req_t (${bus.req_type()}),
    .rsp_t (${bus.rsp_type()})
  ) i_${name}_channel (
    .clk_i,
    .rst_ni,
    .req_i (${name}_req),
    .rsp_o (${name}_rsp)
  );
</%def>

<%def name="tb_memory_no_def(bus, name)">
  % if isinstance(bus, solder.AxiBus):
  tb_memory_axi #(
    .AxiAddrWidth (${bus.aw}),
    .AxiDataWidth (${bus.dw}),
    .AxiIdWidth (${bus.iw}),
    .AxiUserWidth (${bus.uw + 1}),
    .ATOPSupport (0),
  % else:
  tb_memory_regbus #(
    .AddrWidth (${bus.aw}),
    .DataWidth (${bus.dw}),
  % endif
    .req_t (${bus.req_type()}),
    .rsp_t (${bus.rsp_type()})
  ) i_${name}_channel (
    .clk_i,
    .rst_ni,
    .req_i (${name}_req),
    .rsp_o (${name}_rsp)
  );
</%def>

  /// Uart signals
  logic tx, rx;

  /// SPM Wide Mem
  /// Simulated by fesvr
  ${tb_memory(soc_wide_xbar.out_spm_wide,"spm_wide")}

  /// Bootrom
  /// Simulated by fesvr
  axi_lite_a48_d32_req_t axi_lite_bootrom_req;
  axi_lite_a48_d32_rsp_t axi_lite_bootrom_rsp;

<% regbus_bootrom = soc_axi_lite_narrow_periph_xbar.out_bootrom.to_reg(context, "bootrom_regbus", fr="axi_lite_bootrom") %>

  ${tb_memory_no_def(regbus_bootrom, "bootrom_regbus")}

  occamy_top i_occamy (
    .clk_i,
    .rst_ni,
    .sram_cfgs_i ('0),
    .clk_periph_i,
    .rst_periph_ni,
    .rtc_i,
    .test_mode_i (1'b0),
    .chip_id_i ('0),
    .boot_mode_i ('0),
    // UART
    .uart_tx_o (tx),
    .uart_cts_ni ('0),
    .uart_rts_no (),
    .uart_rx_i (rx),
    // GPIO
    .gpio_d_i ('0),
    .gpio_d_o (),
    .gpio_oe_o (),
    // JTAG
    .jtag_trst_ni ('0),
    .jtag_tck_i ('0),
    .jtag_tms_i ('0),
    .jtag_tdi_i ('0),
    .jtag_tdo_o (),
    // I2C
    .i2c_sda_o (),
    .i2c_sda_i ('0),
    .i2c_sda_en_o (),
    .i2c_scl_o (),
    .i2c_scl_i ('0),
    .i2c_scl_en_o (),
    // SPI Master
    .spim_sck_o (),
    .spim_sck_en_o (),
    .spim_csb_o (),
    .spim_csb_en_o (),
    .spim_sd_o(),
    .spim_sd_en_o(),
    .spim_sd_i('0),
    // SPI Slave
    .spis_sck_i (spis_sck_i),
    .spis_csb_i (spis_csb_i),
    .spis_sd_o (spis_sd_o),
    .spis_sd_en_o (spis_sd_en_o),
    .spis_sd_i (spis_sd_i),
    // Bootrom
    .bootrom_req_o (axi_lite_bootrom_req),
    .bootrom_rsp_i (axi_lite_bootrom_rsp),
    // Main RAM
    .spm_axi_wide_req_o (spm_wide_req),
    .spm_axi_wide_rsp_i (spm_wide_rsp),
    .ext_irq_i ('0)
    );

  uartdpi #(
    .BAUD (1),
    // Frequency shouldn't matter since we are sending with the same clock.
    .FREQ (32),
    .NAME("uart0")
  ) i_uart0 (
    .clk_i (clk_i),
    .rst_ni (rst_ni),
    .tx_o (rx),
    .rx_i (tx)
  );

endmodule
