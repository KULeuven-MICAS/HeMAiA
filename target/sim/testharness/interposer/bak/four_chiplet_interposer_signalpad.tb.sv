// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>

module four_chiplet_interposer_signalpad (
    inout wire [199:0] P
);

  localparam CLKTCK = 1ns;  // 1 GHz
  localparam PRITCK = 1ns;  // 1 GHz
  localparam int SRAM_BANK = 16;  // 16 Banks architecture
  localparam int SRAM_DEPTH = 1024;
  localparam int SRAM_WIDTH = 8;  // 8 Bytes Wide

  // Driver regs and IO nets for DUT inout ports
  // Inout/output ports on the DUT must connect to nets (wire/tri). We drive them via separate regs.
  logic mst_clk_drv, periph_clk_drv, rst_ni_drv;

  // Shared Peripheral Clock and Reset
  assign P[141] = periph_clk_drv;
  assign P[138] = rst_ni_drv;
  // Pripheral Reset reuses the main reset
  assign P[137] = rst_ni_drv;
  // Chip_0_0 Master Clock
  assign P[154] = mst_clk_drv;
  // Chip_1_0 Master Clock
  assign P[120] = mst_clk_drv;
  // Chip_0_1 Master Clock
  assign P[178] = mst_clk_drv;
  // Chip_1_1 Master Clock
  assign P[27]  = mst_clk_drv;

  // Chip finish signal
  integer chip_finish[1:0][1:0];

  // The task to reset the chips , init clocks, and load the binaries.
  task automatic init_and_load();
    // Integer to save current time
    time current_time;
    // Init the chip_finish flags
    foreach (chip_finish[i, j]) begin
      chip_finish[i][j] = 0;
    end
    // Init the clk pins
    mst_clk_drv    = 0;
    periph_clk_drv = 0;
    // Init the reset pin
    rst_ni_drv = 1;
    #0;
    rst_ni_drv = 0;
    // Load the binaries
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[0].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_0.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[1].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_1.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[2].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_2.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[3].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_3.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[4].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_4.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[5].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_5.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[6].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_6.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[7].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_7.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[8].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_8.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[9].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_9.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[10].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_10.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[11].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_11.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[12].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_12.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[13].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_13.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[14].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_14.hex");
    i_interposer.i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_0/bank_15.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[0].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_0.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[1].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_1.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[2].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_2.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[3].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_3.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[4].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_4.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[5].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_5.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[6].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_6.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[7].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_7.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[8].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_8.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[9].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_9.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[10].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_10.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[11].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_11.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[12].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_12.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[13].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_13.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[14].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_14.hex");
    i_interposer.i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.load_data(
        "app_chip_0_1/bank_15.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[0].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_0.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[1].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_1.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[2].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_2.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[3].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_3.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[4].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_4.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[5].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_5.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[6].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_6.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[7].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_7.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[8].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_8.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[9].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_9.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[10].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_10.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[11].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_11.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[12].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_12.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[13].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_13.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[14].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_14.hex");
    i_interposer.i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_0/bank_15.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[0].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_0.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[1].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_1.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[2].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_2.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[3].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_3.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[4].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_4.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[5].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_5.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[6].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_6.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[7].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_7.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[8].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_8.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[9].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_9.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[10].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_10.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[11].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_11.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[12].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_12.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[13].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_13.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[14].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_14.hex");
    i_interposer.i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.load_data(
        "app_chip_1_1/bank_15.hex");

    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[0].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_0.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[1].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_1.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[2].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_2.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[3].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_3.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[4].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_4.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[5].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_5.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[6].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_6.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[7].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_7.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[8].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_8.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[9].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_9.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[10].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_10.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[11].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_11.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[12].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_12.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[13].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_13.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[14].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_14.hex");
    i_interposer.i_signalpad.i_hemaia_mem_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.load_data(
        "mempool/bank_15.hex");
    // Release the reset
    #(10 + $urandom % 10);
    current_time = $time / 1000;
    $display("Reset released at %tns", $time / 1000);
    rst_ni_drv = 1;
  endtask

  // Call the reusable init task from an initial block.
  initial begin
    init_and_load();
  end

  // Trigger the reusbale init task when reload_bin becomes high
  logic reload_bin = '0;
  always @(posedge reload_bin) begin
    init_and_load();
    reload_bin = '0;
  end

  always_comb begin
    automatic integer allFinished = 1;
    automatic integer allCorrect = 1;
    if (chip_finish[0][0] == 0) begin
      allFinished = 0;
    end
    if (chip_finish[0][0] == -1) begin
      allCorrect = 0;
    end
    if (chip_finish[0][1] == 0) begin
      allFinished = 0;
    end
    if (chip_finish[0][1] == -1) begin
      allCorrect = 0;
    end
    if (chip_finish[1][0] == 0) begin
      allFinished = 0;
    end
    if (chip_finish[1][0] == -1) begin
      allCorrect = 0;
    end
    if (chip_finish[1][1] == 0) begin
      allFinished = 0;
    end
    if (chip_finish[1][1] == -1) begin
      allCorrect = 0;
    end
    if (allFinished == 1) begin
      if (allCorrect == 1) begin
        $display("All chips finished successfully at %tns", $time / 1000);
        $finish;
      end else begin
        $error("All chips finished with errors at %tns", $time / 1000);
        $finish(-1);
      end
    end
  end

  always #(CLKTCK / 2) begin
    mst_clk_drv = ~mst_clk_drv;
  end

  always #(PRITCK / 2) begin
    periph_clk_drv = ~periph_clk_drv;
  end

  // Must be the frequency of i_uart0.clk_i in Hz
  localparam int unsigned UartDPIFreq = 1_000_000_000;

  // Const Ones and Zeros
  assign P[5]   = 1'b1;
  assign P[6]   = 1'b0;

  // Shared Chip ID 7,6,5,3,2,1
  assign P[12]  = 1'b0;
  assign P[11]  = 1'b0;
  assign P[10]  = 1'b0;
  assign P[9]   = 1'b0;
  assign P[8]   = 1'b0;
  assign P[7]   = 1'b0;

  // Shared Test mode
  assign P[139] = 1'b0;
  // Shared Bootmode
  assign P[142] = 1'b0;
  // Shared bypass PLL division
  assign P[140] = 1'b0;
  // Decide not to allocate a pin for this signal

  // Chiplet 0,0 Private signals
  uartdpi #(
      .BAUD('d31250000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_0_0")
  ) i_uart_0_0 (
      .clk_i (periph_clk_drv),
      .rst_ni(rst_ni_drv),
      .tx_o  (P[148]),
      .rx_i  (P[147])
  );
  // CTS
  assign P[151] = 1'b0;
  // JTAG
  assign P[166] = 1'b0;
  assign P[167] = 1'b0;
  assign P[169] = 1'b0;
  assign P[170] = 1'b0;

  // Chiplet 0,1 Private signals
  uartdpi #(
      .BAUD('d31250000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_0_1")
  ) i_uart_0_1 (
      .clk_i (periph_clk_drv),
      .rst_ni(rst_ni_drv),
      .tx_o  (P[3]),
      .rx_i  (P[4])
  );
  // CTS
  assign P[1]   = 1'b0;
  // JTAG
  assign P[190] = 1'b0;
  assign P[191] = 1'b0;
  assign P[193] = 1'b0;
  assign P[194] = 1'b0;

  // Chiplet 1,0 Private signals
  uartdpi #(
      .BAUD('d31250000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_1_0")
  ) i_uart_1_0 (
      .clk_i (periph_clk_drv),
      .rst_ni(rst_ni_drv),
      .tx_o  (P[135]),
      .rx_i  (P[136])
  );
  // CTS
  assign P[133] = 1'b0;
  // JTAG
  assign P[115] = 1'b0;
  assign P[123] = 1'b0;
  assign P[125] = 1'b0;
  assign P[126] = 1'b0;

  // Chiplet 1,1 Private signals
  uartdpi #(
      .BAUD('d31250000),
      // Frequency shouldn't matter since we are sending with the same clock.
      .FREQ(UartDPIFreq),
      .NAME("uart_1_1")
  ) i_uart_1_1 (
      .clk_i (periph_clk_drv),
      .rst_ni(rst_ni_drv),
      .tx_o  (P[23]),
      .rx_i  (P[24])
  );
  // CTS
  assign P[21] = 1'b0;
  // JTAG
  assign P[39] = 1'b0;
  assign P[40] = 1'b0;
  assign P[13] = 1'b0;
  assign P[14] = 1'b0;

  // Chip status monitor
  // Chip 0,0
  always @(i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32]) begin
    if (i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] != 0) begin
      if (i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] == 32'd1) begin
        $display("Simulation of chip_0_0 is finished at %tns", $time / 1000);
        chip_finish[0][0] = 1;
      end else begin
        $error(
            "Simulation of chip_0_0 is finished with errors %d at %tns",
            i_chippad_powerpad.chip_0_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32],
            $time / 1000);
        chip_finish[0][0] = -1;
      end
    end
  end

  // Chip 1,0
  always @(i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32]) begin
    if (i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] != 0) begin
      if (i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] == 32'd1) begin
        $display("Simulation of chip_1_0 is finished at %tns", $time / 1000);
        chip_finish[1][0] = 1;
      end else begin
        $error(
            "Simulation of chip_1_0 is finished with errors %d at %tns",
            i_chippad_powerpad.chip_1_0.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32],
            $time / 1000);
        chip_finish[1][0] = -1;
      end
    end
  end

  // Chip 0,1
  always @(i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32]) begin
    if (i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] != 0) begin
      if (i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] == 32'd1) begin
        $display("Simulation of chip_0_1 is finished at %tns", $time / 1000);
        chip_finish[0][1] = 1;
      end else begin
        $error(
            "Simulation of chip_0_1 is finished with errors %d at %tns",
            i_chippad_powerpad.chip_0_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32],
            $time / 1000);
        chip_finish[0][1] = -1;
      end
    end
  end

  // Chip 1,1
  always @(i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32]) begin
    if (i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] != 0) begin
      if (i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32] == 32'd1) begin
        $display("Simulation of chip_1_1 is finished at %tns", $time / 1000);
        chip_finish[1][1] = 1;
      end else begin
        $error(
            "Simulation of chip_1_1 is finished with errors %d at %tns",
            i_chippad_powerpad.chip_1_1.i_hemaia.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[15].i_data_mem.i_tc_sram.sram[SRAM_DEPTH-1][(SRAM_WIDTH*8-1)-:32],
            $time / 1000);
        chip_finish[1][1] = -1;
      end
    end
  end

  // Memory Chip
  wire const_one, const_zero;
  assign const_one  = 1'b1;
  assign const_zero = 1'b0;

  hemaia_mem_chip #(
      .WideSRAMBankNum(16),
      .WideSRAMSize(67108864),
      .EnableEastPhy(1'b0),
      .EnableWestPhy(1'b1),
      .EnableNorthPhy(1'b0),
      .EnableSouthPhy(1'b0)
  ) i_hemaia_mem_chip (
      .clk_i(mst_clk_drv),
      .rst_ni(rst_ni_drv),
      .chip_id_i(8'h20),
      .east_d2d_io(),
      .flow_control_east_rts_o(),
      .flow_control_east_cts_i(const_zero),
      .flow_control_east_rts_i(const_zero),
      .flow_control_east_cts_o(),
      .east_test_being_requested_i(const_zero),
      .east_test_request_o(),
      .west_d2d_io(P[107:48]),
      .flow_control_west_rts_o(P[43]),
      .flow_control_west_cts_i(P[44]),
      .flow_control_west_rts_i(P[42]),
      .flow_control_west_cts_o(P[45]),
      .west_test_being_requested_i(P[46]),
      .west_test_request_o(P[47]),
      .north_d2d_io(),
      .flow_control_north_rts_o(),
      .flow_control_north_cts_i(const_zero),
      .flow_control_north_rts_i(const_zero),
      .flow_control_north_cts_o(),
      .north_test_being_requested_i(const_zero),
      .north_test_request_o(),
      .south_d2d_io(),
      .flow_control_south_rts_o(),
      .flow_control_south_cts_i(const_zero),
      .flow_control_south_rts_i(const_zero),
      .flow_control_south_cts_o(),
      .south_test_being_requested_i(const_zero),
      .south_test_request_o()
  );

endmodule
