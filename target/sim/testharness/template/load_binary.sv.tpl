// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

// Binary loading tasks for simulation.
//
// Compute chips use one of three memory implementations:
//   RTL:       tc_sram's behavioral memory and load_data task
//   MEM_MACRO: the unflattened TSMC SRAM macro
//   NETLIST:   the TSMC SRAM macro below the flattened mapped hierarchy
//
// The macro variants are loaded here through PRELOAD and MEMORY.  This keeps the
// generated vendor model pristine; no load_data task has to be appended to it.

<%
    # The mapped tapeout netlist has a fixed 128 KiB compute-chip SPM:
    # 16 banks x 1024 64-bit words.  Configuration-file memory overrides are
    # meaningful for RTL/macro simulation only and must not change this
    # physical hierarchy.
    netlist_sram_bank_count = 16
    netlist_sram_bank_depth = 1024
    compute_bank_count = netlist_sram_bank_count if sim_with_netlist else mem_bank
    sram_depth = (
        netlist_sram_bank_depth
        if sim_with_netlist
        else int(mem_size / 8 / mem_bank)
    )
    mem_chip_bank_count = 16
%>

// The memory chip remains a behavioral testbench component in every compute-chip
// simulation mode.  Always clear every bank first: workloads without a mempool
// image must not inherit data from a previous reload.  RTL uses tc_sram.sram;
// macro/netlist simulation selects TSMC tc_sram's gen_no_sram_matched.mem_array
// because the 64 MiB memory-chip configuration does not match a physical macro.
task automatic load_mem_chip;
    $display("-- Loading Bin to Mem Chip @%0g ns ---", $realtime);
    #10ns;
% for mem_chip_index, mem_chip in enumerate(mem_chips):
<%
    mem_chip_depth = int(mem_chip.size / 8 / mem_chip_bank_count)
    mem_chip_prefix = (
        "i_hemaia_mem_chip_%d_%d.i_hemaia_mem_system.i_hemaia_mem"
        % (mem_chip.coordinate[0], mem_chip.coordinate[1])
    )
    if sim_with_netlist or sim_with_mem_macro:
        mem_chip_array = (
            mem_chip_prefix +
            ".gen_banks[%d].i_data_mem.i_tc_sram.gen_mem.gen_no_sram_matched.mem_array"
        )
    else:
        mem_chip_array = (
            mem_chip_prefix + ".gen_banks[%d].i_data_mem.i_tc_sram.sram"
        )
%>
%   for k in range(mem_chip_bank_count):
    begin : preload_mem_chip_${mem_chip_index}_bank_${k}
        integer mempool_fd;
        integer mempool_word;
        string  mempool_filename;

        mempool_filename = "mempool/bank_${k}.hex";
        for (mempool_word = 0; mempool_word < ${mem_chip_depth}; mempool_word++) begin
            ${mem_chip_array % k}[mempool_word] = '0;
        end

        mempool_fd = $fopen(mempool_filename, "r");
        if (mempool_fd != 0) begin
            $fclose(mempool_fd);
            $readmemh(mempool_filename,
                ${mem_chip_array % k});
            $display("Loaded data from %s into memory-chip bank ${k}", mempool_filename);
        end else begin
            $display("No %s; memory-chip bank ${k} remains zero initialized", mempool_filename);
        end
    end
%   endfor
% endfor
    #10ns;
endtask

// Load compute-chip on-chip SPM banks.
task automatic load_main_mem;
    $display("-- Loading Bin to Main Mem @%0g ns ---", $realtime);
% if sim_with_netlist:
    $display("-- Loading MEM Type: NETLIST");
% elif sim_with_mem_macro:
    $display("-- Loading MEM Type: MEM_MACRO");
% else:
    $display("-- Loading MEM Type: SIM (RTL)");
% endif
    #10ns;
% if sim_with_netlist or sim_with_mem_macro:
%   for compute_chip in compute_chips:
%     for k in range(compute_bank_count):
<%
    cx = compute_chip.coordinate[0]
    cy = compute_chip.coordinate[1]
    if sim_with_netlist:
        macro_path = (
            "i_dut.i_hemaia_%d_%d.\\i_occamy_chip/i_hemaia_mem_system/i_hemaia_mem "
            ".\\gen_banks_%d__i_data_mem/i_tc_sram/gen_mem_gen_%dx64_u_sram "
            % (cx, cy, k, sram_depth)
        )
    else:
        macro_path = (
            "i_dut.i_hemaia_%d_%d.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem"
            ".gen_banks[%d].i_data_mem.i_tc_sram.gen_mem.gen_%dx64.u_sram"
            % (cx, cy, k, sram_depth)
        )
%>
    begin : preload_chip_${cx}_${cy}_bank_${k}
        integer preload_word;
        integer preload_row;
        integer preload_column;

        // Zero PRELOAD so a short hex file also deterministically clears the
        // unused tail, then reproduce the macro's logical-address mapping.
        for (preload_word = 0; preload_word < ${sram_depth}; preload_word++) begin
            ${macro_path}.PRELOAD[preload_word] = '0;
        end
        $readmemh("app_chip_${cx}_${cy}/bank_${k}.hex", ${macro_path}.PRELOAD);
        for (preload_word = 0; preload_word < ${sram_depth}; preload_word++) begin
            preload_row = preload_word >> ${macro_path}.numCMAddr;
            preload_column = preload_word & ((1 << ${macro_path}.numCMAddr) - 1);
            ${macro_path}.MEMORY[preload_row][preload_column] =
                ${macro_path}.PRELOAD[preload_word];
        end
        $display("Loaded app_chip_${cx}_${cy}/bank_${k}.hex into compute-chip bank ${k}");
    end
%     endfor
%   endfor
% else:
%   for compute_chip in compute_chips:
%     for k in range(compute_bank_count):
    i_dut.i_hemaia_${compute_chip.coordinate[0]}_${compute_chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${k}].i_data_mem.i_tc_sram.load_data("app_chip_${compute_chip.coordinate[0]}_${compute_chip.coordinate[1]}/bank_${k}.hex");
%     endfor
%   endfor
% endif
    #10ns;
endtask

// Top-level load task called from the testharness initial block.
task automatic load_binary;
% if len(mem_chips) > 0:
    load_mem_chip;
% endif
    load_main_mem;
endtask
