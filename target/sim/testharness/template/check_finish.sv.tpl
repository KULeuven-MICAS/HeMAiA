// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

// Simulation finish monitor
// Polls the last word of the last SRAM bank in each compute chip every clock cycle.
// When the software writes a non-zero value to that location:
//   1  => simulation passed
//   else => simulation failed with error code
//
// Hierarchy (both interposer and non-interposer modes):
//   i_dut.i_hemaia_X_Y.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem
//     .gen_banks[K].i_data_mem.i_tc_sram

<%
    sram_depth = int(mem_size / 8 / mem_bank)
    sram_width = 8  ## 8 Bytes Wide
%>

// check_finish task: polls SRAM and blocks until all compute chips have finished
task automatic check_finish();
    begin
% for compute_chip in compute_chips:
<%
    cx = compute_chip.coordinate[0]
    cy = compute_chip.coordinate[1]
%>
        automatic integer chip_finish_${cx}_${cy} = 0;
% endfor
        forever begin
            @(posedge mst_clk_i);
            begin
                automatic integer allFinished = 1;
                automatic integer allCorrect = 1;
% for compute_chip in compute_chips:
<%
    cx = compute_chip.coordinate[0]
    cy = compute_chip.coordinate[1]
    sram_path = "i_dut.i_hemaia_%d_%d.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[%d].i_data_mem.i_tc_sram" % (cx, cy, mem_bank - 1)
    sram_word = "%s.sram[%d][%d-:32]" % (sram_path, sram_depth - 1, sram_width * 8 - 1)
%>
                // Chip (${cx}, ${cy})
                if (chip_finish_${cx}_${cy} == 0 && ${sram_word} != 0) begin
                    if (${sram_word} == 32'd1) begin
                        $display("Simulation of chip_${cx}_${cy} is finished at %tns", $time / 1000);
                        chip_finish_${cx}_${cy} = 1;
                    end else begin
                        $error("Simulation of chip_${cx}_${cy} is finished with errors %d at %tns",
                               ${sram_word}, $time / 1000);
                        chip_finish_${cx}_${cy} = -1;
                    end
                end
                if (chip_finish_${cx}_${cy} == 0) allFinished = 0;
                if (chip_finish_${cx}_${cy} == -1) allCorrect = 0;
% endfor
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
        end
    end
endtask
