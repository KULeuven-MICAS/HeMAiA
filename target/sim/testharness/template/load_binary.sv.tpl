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
        else mem_size // 8 // mem_bank
    )
    mem_chip_bank_count = 16
    memory_mode = "netlist" if sim_with_netlist else "macro" if sim_with_mem_macro else "rtl"

    # Match tc_sram_tsmc16: larger 64-bit banks concatenate complete 8192-word macros.
    # Each entry is (logical offset, depth, physical path, is_macro).
    def memory_parts(depth, path, mode):
        if mode == "rtl":
            return [(0, depth, path + ".sram", False)]
        if mode == "netlist":
            return [(0, depth, path, True)]
        macro_depth = 8192
        if depth > macro_depth and depth % macro_depth == 0:
            return [(i * macro_depth, macro_depth,
                     path + f".gen_mem.gen_depth_concat.gen_depth[{i}].u_sram", True)
                    for i in range(depth // macro_depth)]
        if depth in (128, 256, 512, 1024, 2048, 4096, macro_depth):
            return [(0, depth, path + f".gen_mem.gen_{depth}x64.u_sram", True)]
        return [(0, depth, path + ".gen_mem.gen_no_sram_matched.mem_array", False)]
%>

<%def name="deposit_parts(parts, image)">\
% for offset, depth, path, is_macro in parts:
    for (int word_index = 0; word_index < ${depth}; word_index++) begin
%   if is_macro:
        ${path}.PRELOAD[word_index] = ${image}[${offset} + word_index];
        ${path}.MEMORY[word_index >> ${path}.numCMAddr]
            [word_index & ((1 << ${path}.numCMAddr) - 1)] = ${image}[${offset} + word_index];
%   else:
        ${path}[word_index] = ${image}[${offset} + word_index];
%   endif
    end
% endfor
</%def>\

<%def name="clear_parts(parts)">\
% for offset, depth, path, is_macro in parts:
    for (int word_index = 0; word_index < ${depth}; word_index++) begin
%   if is_macro:
        ${path}.PRELOAD[word_index] = '0;
        ${path}.MEMORY[word_index >> ${path}.numCMAddr]
            [word_index & ((1 << ${path}.numCMAddr) - 1)] = '0;
%   else:
        ${path}[word_index] = '0;
%   endif
    end
% endfor
</%def>\

// The memory chip remains a behavioral testbench component in every compute-chip
// simulation mode.  Always clear every bank first: workloads without a mempool
// image must not inherit data from a previous reload.  Each memory chip loads its own
// image, mempool_chip_<x>_<y>/, when the workload staged one for it, and the shared
// mempool/ otherwise -- so a workload that knows one memory chip loads the same data into
// all of them, as it always has. Use the same physical layout as compute-chip
// banks, including any depth concatenation. Memory-chip hierarchy is unflattened.
task automatic load_mem_chip;
    $display("-- Loading Bin to Mem Chip @%0g ns ---", $realtime);
    #10ns;
% for mem_chip_index, mem_chip in enumerate(mem_chips):
<%
    mem_chip_depth = mem_chip.size // 8 // mem_chip_bank_count
    mem_chip_prefix = (
        "i_hemaia_mem_chip_%d_%d.i_hemaia_mem_system.i_hemaia_mem"
        % (mem_chip.coordinate[0], mem_chip.coordinate[1])
    )
    mem_chip_mode = "macro" if sim_with_netlist or sim_with_mem_macro else "rtl"
%>
%   for k in range(mem_chip_bank_count):
<%
    mem_chip_path = mem_chip_prefix + f".gen_banks[{k}].i_data_mem.i_tc_sram"
    mem_chip_parts = memory_parts(mem_chip_depth, mem_chip_path, mem_chip_mode)
%>
    begin : preload_mem_chip_${mem_chip_index}_bank_${k}
        integer mempool_fd;
        integer mempool_word;
        string  mempool_filename;
%   if mem_chip_parts[0][3]:
        logic [63:0] mempool_data [0:${mem_chip_depth - 1}];
<% mempool_array = "mempool_data" %>
%   else:
<% mempool_array = mem_chip_parts[0][2] %>
%   endif

        mempool_filename = "mempool_chip_${mem_chip.coordinate[0]}_${mem_chip.coordinate[1]}/bank_${k}.hex";
        for (mempool_word = 0; mempool_word < ${mem_chip_depth}; mempool_word++) begin
            ${mempool_array}[mempool_word] = '0;
        end

        mempool_fd = $fopen(mempool_filename, "r");
        if (mempool_fd == 0) begin
            mempool_filename = "mempool/bank_${k}.hex";
            mempool_fd = $fopen(mempool_filename, "r");
        end
        if (mempool_fd != 0) begin
            $fclose(mempool_fd);
            $readmemh(mempool_filename,
                ${mempool_array});
            $display("Loaded data from %s into memory-chip (${mem_chip.coordinate[0]}, ${mem_chip.coordinate[1]}) bank ${k}", mempool_filename);
        end else begin
            $display("No %s; memory-chip (${mem_chip.coordinate[0]}, ${mem_chip.coordinate[1]}) bank ${k} remains zero initialized", mempool_filename);
        end
%   if mem_chip_parts[0][3]:
${deposit_parts(mem_chip_parts, mempool_array)}\
%   endif
    end
%   endfor
% endfor
    #10ns;
endtask

// The memory chips' simulated HBM
// (hw/hemaia/hemaia_mem_system/hbm). Cleared like the SRAM,
// then filled from the manifest the workload build staged in hbm/ -- or the one
// +hbm_manifest=<path> names. The files are mapped, not copied, so GiB images load
// instantly. No manifest: the HBM starts zeroed. One manifest serves every memory chip:
// an entry tagged chip=<id> loads into that chip only, an untagged one into all.
task automatic load_hbm;
    string manifest;
    if (!$value$plusargs("hbm_manifest=%s", manifest)) manifest = "hbm/manifest.txt";
    $display("-- Loading HBM from %s @%0g ns ---", manifest, $realtime);
% for mem_chip in mem_chips:
%   if mem_chip.hbm:
    i_hemaia_mem_chip_${mem_chip.coordinate[0]}_${mem_chip.coordinate[1]}.gen_hbm.i_hbm.clear();
    i_hemaia_mem_chip_${mem_chip.coordinate[0]}_${mem_chip.coordinate[1]}.gen_hbm.i_hbm.load_manifest(manifest, ${mem_chip.chip_id});
%   endif
% endfor
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
        sram_path = (
            "i_dut.i_hemaia_%d_%d.\\i_occamy_chip/i_hemaia_mem_system/i_hemaia_mem "
            ".\\gen_banks_%d__i_data_mem/i_tc_sram/gen_mem_gen_%dx64_u_sram "
            % (cx, cy, k, sram_depth)
        )
    else:
        sram_path = (
            "i_dut.i_hemaia_%d_%d.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem"
            ".gen_banks[%d].i_data_mem.i_tc_sram"
            % (cx, cy, k)
        )
    bank_parts = memory_parts(sram_depth, sram_path, memory_mode)
%>
    begin : preload_chip_${cx}_${cy}_bank_${k}
        integer preload_word;
        logic [63:0] preload_data [0:${sram_depth - 1}];

        // Load the complete logical bank, then distribute it over its macros.
        // Clearing first also initializes the unused tail of short hex files.
        for (preload_word = 0; preload_word < ${sram_depth}; preload_word++) begin
            preload_data[preload_word] = '0;
        end
        $readmemh("app_chip_${cx}_${cy}/bank_${k}.hex", preload_data);
${deposit_parts(bank_parts, "preload_data")}\
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

// Zero every compute chip's narrow SPM (L2) at time zero, before the boot ROM runs: the simulation
// boot ROM does not clear it (target/rtl/bootrom/bootrom_sim/bootrom.c). Clear every
// physical part of the configured bank. The netlist has no such path.
task automatic clear_narrow_spm;
% if sim_with_netlist:
    $warning("clear_narrow_spm: no path to the narrow SPM in the netlist, it starts uncleared");
% else:
%   for compute_chip in compute_chips:
<%
    narrow_path = (
        "i_dut.i_hemaia_%d_%d.i_occamy_chip.i_occamy.i_occamy_soc"
        ".i_spm_narrow_cut.i_mem.i_mem.i_tc_sram"
        % (compute_chip.coordinate[0], compute_chip.coordinate[1])
    )
    narrow_parts = memory_parts(occamy_cfg["spm_narrow"]["length"] // 8, narrow_path, memory_mode)
%>
${clear_parts(narrow_parts)}\
%   endfor
    $display("-- Cleared the narrow SPM of ${len(compute_chips)} compute chip(s) @%0g ns ---", $realtime);
% endif
endtask

// Top-level load task called from the testharness initial block.
task automatic load_binary;
    clear_narrow_spm;
% if len(mem_chips) > 0:
    load_mem_chip;
    load_hbm;
% endif
    load_main_mem;
endtask
