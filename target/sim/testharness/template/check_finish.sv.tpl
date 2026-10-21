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
        else mem_size // 8 // mem_bank
    )
    sram_width_bits = 64
    # Match tc_sram_tsmc16: concatenate complete macros beyond the compiler limit.
    max_macro_depth = 8192
    depth_concatenated = sram_depth > max_macro_depth and sram_depth % max_macro_depth == 0
    macro_depth = max_macro_depth if depth_concatenated else sram_depth
    last_macro = (sram_depth - 1) // macro_depth
    last_macro_word = (sram_depth - 1) % macro_depth
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
        sram_path = (
            "i_dut.i_hemaia_%d_%d.\\i_occamy_chip/i_hemaia_mem_system/i_hemaia_mem "
            ".\\gen_banks_%d__i_data_mem/i_tc_sram/gen_mem_gen_%dx64_u_sram "
            % (cx, cy, compute_bank_count - 1, sram_depth)
        )
    else:
        sram_path = (
            "i_dut.i_hemaia_%d_%d.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem"
            ".gen_banks[%d].i_data_mem.i_tc_sram"
            % (cx, cy, compute_bank_count - 1)
        )
    if sim_with_netlist or sim_with_mem_macro:
        if sim_with_netlist:
            macro_path = sram_path
        elif depth_concatenated:
            macro_path = sram_path + f".gen_mem.gen_depth_concat.gen_depth[{last_macro}].u_sram"
        elif sram_depth in (128, 256, 512, 1024, 2048, 4096, max_macro_depth):
            macro_path = sram_path + f".gen_mem.gen_{sram_depth}x64.u_sram"
        else:
            macro_path = None
        if macro_path:
            # Use the last macro's own mux geometry to locate its final logical word.
            shift = macro_path + ".numCMAddr"
            sram_word = (f"{macro_path}.MEMORY[{last_macro_word} >> {shift}]"
                         f"[{last_macro_word} & ((1 << {shift}) - 1)]")
        else:
            sram_word = sram_path + f".gen_mem.gen_no_sram_matched.mem_array[{sram_depth - 1}]"
    else:
        sram_word = sram_path + f".sram[{sram_depth - 1}]"
    sram_word += f"[{sram_width_bits - 1}-:32]"
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
                    $display("All chips finished successfully at %0t", $time);
                    $finish;
                end else begin
                    $fatal(1, "All chips finished with errors at %0t", $time);
                end
            end
        end
    end
endtask
