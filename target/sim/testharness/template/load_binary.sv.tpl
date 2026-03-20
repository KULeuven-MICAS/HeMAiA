// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

// Binary loading tasks for simulation
// Generated from Mako template — instance paths match the new testbench hierarchy:
//   Mem chips:     i_hemaia_mem_chip_X_Y.i_hemaia_mem_system.i_hemaia_mem.gen_banks[K]...
//   Compute chips: i_dut.i_hemaia_X_Y.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[K]...
// No interposer_prefix is needed — hemaia instances are always directly in i_dut.

<%
    sram_depth = int(mem_size / 8 / mem_bank)
%>

// Load external memory chips (mempool banks, fixed 16 banks)
task automatic load_mem_chip;
    $display("-- Loading Bin to Mem Chip@%0g ns ---",$realtime);
    #10ns;
% for mem_chip in mem_chips:
%   for k in range(0, 16):
    i_hemaia_mem_chip_${mem_chip.coordinate[0]}_${mem_chip.coordinate[1]}.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${k}].i_data_mem.i_tc_sram.load_data("mempool/bank_${k}.hex");
%   endfor
% endfor
    #10ns;
endtask

// Load compute chip main memory (on-chip SPM banks)
task automatic load_main_mem;
    input string MEM_TYPE;
    $display("-- Loading Bin to Main Mem @%0g ns ---",$realtime);
    $display("-- Loading MEM Type: %s", MEM_TYPE);
    #10ns;
    case (MEM_TYPE)
        "SIM": begin
        // Behavioral SRAM model
        // Path: i_dut.i_hemaia_X_Y.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[K].i_data_mem.i_tc_sram
% for compute_chip in compute_chips:
%   for k in range(0, mem_bank):
            i_dut.i_hemaia_${compute_chip.coordinate[0]}_${compute_chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${k}].i_data_mem.i_tc_sram.load_data("app_chip_${compute_chip.coordinate[0]}_${compute_chip.coordinate[1]}/bank_${k}.hex");
%   endfor
% endfor
        end
        "MEM_MARCO": begin
        // Vendor macro SRAM model
        // Depth = mem_size / (64/8) / num_bank = ${sram_depth}
        // Path: ...gen_banks[K].i_data_mem.i_tc_sram.gen_mem.gen_${sram_depth}x64.u_sram
% for compute_chip in compute_chips:
%   for k in range(0, mem_bank):
            i_dut.i_hemaia_${compute_chip.coordinate[0]}_${compute_chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks[${k}].i_data_mem.i_tc_sram.gen_mem.gen_${sram_depth}x${64}.u_sram.load_data("app_chip_${compute_chip.coordinate[0]}_${compute_chip.coordinate[1]}/bank_${k}.hex");
%   endfor
% endfor
        end
        "NETLIST": begin
        // Post-synthesis netlist model (gen_banks becomes gen_banks_K)
        // Path: ...gen_banks_K.i_data_mem.i_tc_sram.gen_mem.gen_${sram_depth}x64.u_sram
% for compute_chip in compute_chips:
%   for k in range(0, mem_bank):
            i_dut.i_hemaia_${compute_chip.coordinate[0]}_${compute_chip.coordinate[1]}.i_occamy_chip.i_hemaia_mem_system.i_hemaia_mem.gen_banks_${k}.i_data_mem.i_tc_sram.gen_mem.gen_${sram_depth}x${64}.u_sram.load_data("app_chip_${compute_chip.coordinate[0]}_${compute_chip.coordinate[1]}/bank_${k}.hex");
%   endfor
% endfor
        end
        default: begin
            $error("Input parameter MEM_TYPE is not supported!");
        end
    endcase
    #10ns;
endtask

// Top-level load task called from testharness initial block
task automatic load_binary;
    string mem_type;
    // Check for +MEM_TYPE=xxx plusarg, default to "SIM"
    if (!$value$plusargs("MEM_TYPE=%s", mem_type)) begin
% if sim_with_mem_macro:
        mem_type = "MEM_MARCO";
% elif sim_with_netlist:
        mem_type = "NETLIST";
% else:
        mem_type = "SIM";
% endif
    end
% if len(mem_chips) > 0:
    load_mem_chip;
% endif
    load_main_mem(mem_type);
endtask
