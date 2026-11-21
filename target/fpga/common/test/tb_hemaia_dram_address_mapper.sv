// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

module tb_hemaia_dram_address_mapper;
  timeunit 1ns;
  timeprecision 1ps;

  logic [ 7:0] chip_id;
  logic [47:0] aw_addr, ar_addr, mapped_aw, mapped_ar;

  hemaia_dram_address_mapper dut (
    .chip_id_i ( chip_id   ),
    .aw_addr_i ( aw_addr   ),
    .ar_addr_i ( ar_addr   ),
    .aw_addr_o ( mapped_aw ),
    .ar_addr_o ( mapped_ar )
  );

  task automatic check(
    input logic [ 7:0] local_chip,
    input logic [47:0] aw_in, ar_in,
    input logic [47:0] expected_aw, expected_ar
  );
    chip_id = local_chip;
    aw_addr = aw_in;
    ar_addr = ar_in;
    #1ns;
    if (mapped_aw !== expected_aw || mapped_ar !== expected_ar) begin
      $fatal(1, "Chip %02h: AW %012h -> %012h (expected %012h), AR %012h -> %012h (expected %012h)",
             chip_id, aw_addr, mapped_aw, expected_aw, ar_addr, mapped_ar, expected_ar);
    end
  endtask

  initial begin
    for (int unsigned id = 0; id < 256; id++) begin
      check(8'(id), {8'(id), 40'h20_0000_0000}, {8'(id), 40'h22_FFFF_FFFF},
            48'h0000_0000_0000, 48'h0600_FFFF_FFFF);
      check(8'(id), {8'(id), 40'h20_7FFF_FFFF}, {8'(id), 40'h20_8000_0000},
            48'h0000_7FFF_FFFF, 48'h0008_0000_0000);
      check(8'(id), {8'(id), 40'h20_FFFF_FFFF}, {8'(id), 40'h21_0000_0000},
            48'h0008_7FFF_FFFF, 48'h0500_0000_0000);
      check(8'(id), {8'(id), 40'h21_FFFF_FFFF}, {8'(id), 40'h22_0000_0000},
            48'h0500_FFFF_FFFF, 48'h0600_0000_0000);
      check(8'(id), {8'(id), 40'h22_4567_89AB}, {8'(id), 40'h20_1234_5678},
            48'h0600_4567_89AB, 48'h0000_1234_5678);
      check(8'(id), {8'(id), 40'h20_AAAA_BBCD}, {8'(id), 40'h21_ABCD_1234},
            48'h0008_2AAA_BBCD, 48'h0500_ABCD_1234);
      check(8'(id), {8'(id), 40'h1F_FFFF_FFFF}, {8'(id), 40'h23_0000_0000},
            48'h0010_0000_0FFF, 48'h0010_0000_0000);
      check(8'(id), {8'(id + 1), 40'h20_0000_0001}, {8'(id - 1), 40'h22_FFFF_FFFE},
            48'h0010_0000_0001, 48'h0010_0000_0FFE);
    end
    $display("HeMAiA DRAM address mapper passed for all 256 chip IDs");
    $finish;
  end
endmodule
