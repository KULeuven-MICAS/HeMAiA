// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

// Poll the final word of the final SPM bank in every compute chip.  Software
// writes 1 for success and another non-zero value for failure.  Unknown values
// are ignored until the memory has reached a stable, software-written state.

<%
    # The mapped tapeout netlist has a fixed 128 KiB compute-chip SPM:
    # 16 banks x 1024 64-bit words.  Do not derive its physical hierarchy
    # from a simulation configuration override.
    netlist_sram_bank_count = 16
    netlist_sram_bank_depth = 1024
    compute_bank_count = netlist_sram_bank_count if sim_with_netlist else mem_bank
    sram_depth = (
        netlist_sram_bank_depth
        if sim_with_netlist
        else int(mem_size / 8 / mem_bank)
    )
    sram_width_bits = 64
    # TSMC's 2048x64 macro uses M8; the other 64-bit macros used by tc_sram
    # use M4.  The mapped tapeout SRAM is 1024x64 M4.
    macro_column_count = 8 if sram_depth == 2048 else 4
    finish_macro_row = (sram_depth - 1) // macro_column_count
    finish_macro_column = (sram_depth - 1) % macro_column_count
%>

task automatic check_finish();
% for compute_chip in compute_chips:
<%
    cx = compute_chip.coordinate[0]
    cy = compute_chip.coordinate[1]
%>
    integer      chip_finish_${cx}_${cy};
    logic [31:0] chip_status_${cx}_${cy};
% endfor
    integer all_finished;
    integer all_correct;

    begin
% for compute_chip in compute_chips:
<%
    cx = compute_chip.coordinate[0]
    cy = compute_chip.coordinate[1]
%>
        chip_finish_${cx}_${cy} = 0;
% endfor

        forever begin
            @(posedge mst_clk_i);
            all_finished = 1;
            all_correct = 1;

% for compute_chip in compute_chips:
<%
    cx = compute_chip.coordinate[0]
    cy = compute_chip.coordinate[1]
    if sim_with_netlist:
        sram_word = (
            "i_dut.i_hemaia_%d_%d.\\i_occamy_chip/i_hemaia_mem_system/i_hemaia_mem "
            ".\\gen_banks_%d__i_data_mem/i_tc_sram/gen_mem_gen_%dx64_u_sram "
            ".MEMORY[%d][%d][%d-:32]"
            % (cx, cy, compute_bank_count - 1, sram_depth, finish_macro_row,
               finish_macro_column, sram_width_bits - 1)
        )
    elif sim_with_mem_macro:
        sram_word = (
            "i_dut.i_hemaia_%d_%d.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem"
            ".gen_banks[%d].i_data_mem.i_tc_sram.gen_mem.gen_%dx64.u_sram"
            ".MEMORY[%d][%d][%d-:32]"
            % (cx, cy, compute_bank_count - 1, sram_depth, finish_macro_row,
               finish_macro_column, sram_width_bits - 1)
        )
    else:
        sram_word = (
            "i_dut.i_hemaia_%d_%d.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem"
            ".gen_banks[%d].i_data_mem.i_tc_sram.sram[%d][%d-:32]"
            % (cx, cy, compute_bank_count - 1, sram_depth - 1, sram_width_bits - 1)
        )
%>
            chip_status_${cx}_${cy} = ${sram_word};
            if (chip_finish_${cx}_${cy} == 0 &&
                !$isunknown(chip_status_${cx}_${cy}) && chip_status_${cx}_${cy} != 0) begin
                if (chip_status_${cx}_${cy} == 32'd1) begin
                    $display("Simulation of chip_${cx}_${cy} finished successfully at %0t", $time);
                    chip_finish_${cx}_${cy} = 1;
                end else begin
                    $error("Simulation of chip_${cx}_${cy} finished with status %0d at %0t",
                           chip_status_${cx}_${cy}, $time);
                    chip_finish_${cx}_${cy} = -1;
                end
            end
            if (chip_finish_${cx}_${cy} == 0) all_finished = 0;
            if (chip_finish_${cx}_${cy} == -1) all_correct = 0;

% endfor
            if (all_finished == 1) begin
                if (all_correct == 1) begin
% for compute_chip in compute_chips:
<%
    cx, cy = compute_chip.coordinate
%>
                    if ($test$plusargs("check_pmic_dvfs")) begin
                        if (i_pmic_chip${cx}${cy}.read_count < 4 ||
                            i_pmic_chip${cx}${cy}.vset_write_count < 3 ||
                            chip${cx}${cy}_dvfs_phase != 3 ||
                            i_pmic_chip${cx}${cy}.output_voltage_mv() != 800.0 ||
                            !i_pmic_chip${cx}${cy}.voltage_settled() ||
                            i_pmic_chip${cx}${cy}.regs[1] != 8'h38 ||
                            i_pmic_chip${cx}${cy}.nack_count != 0)
                            $fatal(1, "chip${cx}${cy}: missing or failed PMIC/DVFS transactions");
%if not sim_with_netlist:
%for domain in range(1 + len(occamy_cfg["clusters"])):
                        if (chip${cx}${cy}_dvfs_div${domain} !== 8'd7)
                            $fatal(1, "chip${cx}${cy} domain ${domain}: final divider is not /7");
%endfor
%for domain in range(1 + len(occamy_cfg["clusters"]), 5 + len(occamy_cfg["clusters"])):
                        if (i_dut.i_hemaia_${cx}_${cy}.i_occamy_chip.i_hemaia_clk_rst_controller.gen_clock_divider[${domain}].i_clk_divider.divisor_q !== 8'd${20 if not same_memchip_speed and cx == 1 and cy == 0 and domain == 1 + len(occamy_cfg["clusters"]) else 1})
                            $fatal(1, "chip${cx}${cy} domain ${domain}: D2D TX divider changed");
%endfor
%endif
                        $display("[pmic-check] chip${cx}${cy} PASS: reads=%0d voltage_writes=%0d",
                                 i_pmic_chip${cx}${cy}.read_count, i_pmic_chip${cx}${cy}.vset_write_count);
                    end
% endfor
%if not sim_with_verilator:
%for mem_chip in mem_chips:
<%
    mx, my = mem_chip.coordinate
%>
                    if ($test$plusargs("check_pmic_dvfs")) begin
%for domain in range(5):
                        if (i_hemaia_mem_chip_${mx}_${my}.i_hemaia_clk_rst_controller.gen_clock_divider[${domain}].i_clk_divider.divisor_q !== 8'd${6 if domain == 0 else 1})
                            $fatal(1, "memchip${mx}${my} domain ${domain}: clock divider changed");
%endfor
                    end
%endfor
%endif
                    $display("All chips finished successfully at %0t", $time);
                    $finish;
                end else begin
                    $fatal(1, "All chips finished with errors at %0t", $time);
                end
            end
        end
    end
endtask
