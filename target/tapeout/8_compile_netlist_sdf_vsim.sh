#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
sim_cfg_dir="$script_dir/../rtl/cfg"

# 
FILE=$script_dir/../../hw/hemaia/tech_cells_tsmc16/src/tsmc16/mem_macro/ts1n16ffcllsblvtd1024x64m4sws_150a/VERILOG/ts1n16ffcllsblvtd1024x64m4sws_150a.v

# Check if task is already present
if grep -q "task automatic load_data" "$FILE"; then
    echo "Task already exists. No changes made."
else
    echo "Task not found. Inserting before endmodule..."
    # Insert the block before endmodule
    sed -i '/endmodule/i \
    task automatic load_data(input string filename);\n\
      reg [numWordAddr:0] w;\n\
      reg [numWordAddr-numCMAddr-1:0] row;\n\
      reg [numCMAddr-1:0] col;\n\
      begin\n\
        // Initialization of the memory array\n\
        foreach (PRELOAD[i]) PRELOAD[i] = '\''0;\n\
        // Load data from file\n\
        $readmemh(filename, PRELOAD);\n\
        for (w = 0; w < numWord; w = w + 1) begin\n\
          {row, col} = w;\n\
          MEMORY[row][col] = PRELOAD[w];\n\
        end\n\
        $display(\"Loaded data from %s into memory\", filename);\n\
      end\n\
    endtask\n' "$FILE"
fi

# overwrite check_finish_netlist.sv, load_binary_netlist.sv, testharness_netlist.sv and compile.vsim.tcl for netlist sim

cp -f "/users/micas/shares/project_HeMAiAv2/hemaia_testing/netlist_sim_with_sdf/check_finish_netlist.sv" "$script_dir/../sim/testharness/check_finish_netlist.sv"
cp -f "/users/micas/shares/project_HeMAiAv2/hemaia_testing/netlist_sim_with_sdf/load_binary_netlist.sv" "$script_dir/../sim/testharness/load_binary_netlist.sv"
cp -f "/users/micas/shares/project_HeMAiAv2/hemaia_testing/netlist_sim_with_sdf/testharness_netlist.sv" "$script_dir/../sim/testharness/testharness.sv"

cp -f "/users/micas/shares/project_HeMAiAv2/hemaia_testing/netlist_sim_with_sdf/hemaia_mapped.sdf" "$script_dir/../sim/testharness/hemaia_mapped.sdf"

rm -rf "$script_dir/../sim/work-vsim/*"

cp -f "/users/micas/shares/project_HeMAiAv2/hemaia_testing/netlist_sim_with_sdf/compile_netlist.vsim.tcl" "$script_dir/../sim/work-vsim/compile.vsim.tcl"

# #!/bin/bash
# vsim +permissive -t 1ps -voptargs=+acc -do "log -r /*;  tcheck_set \"/testharness/i_hemaia/i_occamy_chip/i_occamy/i_periph_cdc/\\\i_axi_cdc_dst/i_cdc_fifo_gray_dst_ar/gen_sync_0__i_sync/reg_q_reg_0_  \" OFF; tcheck_set \"/testharness/i_hemaia/i_occamy_chip/i_occamy/i_periph_cdc/\\\i_axi_cdc_dst/i_cdc_fifo_gray_dst_ar/gen_sync_1__i_sync/reg_q_reg_0_  \" OFF; tcheck_set \"/testharness/i_hemaia/i_occamy_chip/i_occamy/i_periph_cdc/\\\i_axi_cdc_dst/i_cdc_fifo_gray_dst_ar/gen_sync_2__i_sync/reg_q_reg_0_  \" OFF; tcheck_set \"/testharness/i_hemaia/i_occamy_chip/i_occamy/i_periph_cdc/\\\i_axi_cdc_src/i_cdc_fifo_gray_dst_r/gen_sync_1__i_sync/reg_q_reg_0_ \" OFF; tcheck_set \"/testharness/i_hemaia/i_occamy_chip/i_occamy/i_periph_cdc/\\\i_axi_cdc_src/i_cdc_fifo_gray_dst_r/gen_sync_0__i_sync/reg_q_reg_0_ \" OFF; tcheck_set \"/testharness/i_hemaia/i_occamy_chip/i_occamy/i_periph_cdc/\\\i_axi_cdc_src/i_cdc_fifo_gray_dst_r/gen_sync_2__i_sync/reg_q_reg_0_ \" OFF; tcheck_set \"/testharness/i_hemaia/i_occamy_chip/i_occamy/\\\i_axi_lite_to_soc_cdc/i_axi_cdc_dst/i_cdc_fifo_gray_src_b/gen_sync_1__i_sync/reg_q_reg_0_ \" OFF; tcheck_set \"/testharness/i_hemaia/i_occamy_chip/i_occamy/\\\i_axi_lite_to_soc_cdc/i_axi_cdc_dst/i_cdc_fifo_gray_src_b/gen_sync_0__i_sync/reg_q_reg_0_ \" OFF; tcheck_set \"/testharness/i_hemaia/i_occamy_chip/i_occamy/\\\i_axi_lite_to_soc_cdc/i_axi_cdc_dst/i_cdc_fifo_gray_src_b/gen_sync_2__i_sync/reg_q_reg_0_ \" OFF; run -a" -sdfnoerror -work /users/micas/shares/project_snax/post_synthesis_testing/post_pnr_mapped_rtl/work-vsim \
#       -ldflags "-Wl,-rpath,/users/micas/shares/project_snax/projects/hemaia/HeMAiA/target/sim/work/lib -L/users/micas/shares/project_snax/projects/hemaia/HeMAiA/target/sim/work/lib -lfesvr -lutil" \
#       testharness_opt +permissive-off
 

# #!/bin/bash
# binary=$(realpath $1)
# mkdir -p logs
# echo $binary > logs/.rtlbinary
# vsim +permissive -64 -t 1ps -sdfmax /tb_bin/i_dut/i_snax_hypercorex_cluster=/users/micas/shares/project_snax/rambo-hulk-flow/rambo-flow/work-snax_hypercorex/outputs/snax_hypercorex_cluster_wrapper_mapped.sdf -sdfreport=/users/micas/shares/project_snax/snax_hypercorex/snax_cluster/target/snitch_cluster//work-vsim/sdf_report.txt -sdfnoerror +notimingchecks -do "log -r /*; run -a" -work /users/micas/shares/project_snax/snax_hypercorex/snax_cluster/target/snitch_cluster//work-vsim -ldflags "-Wl,-rpath,/users/micas/shares/project_snax/snax_hypercorex/snax_cluster/target/snitch_cluster/work/lib -L/users/micas/shares/project_snax/snax_hypercorex/snax_cluster/target/snitch_cluster/work/lib -lfesvr -lutil" tb_bin_opt +permissive-off ++$binary ++$2


# questa-vsim
make -C "$script_dir/../.." hemaia_system_vsim SIM_CFG="$sim_cfg_dir/sim_netlist.hjson"

cat > "$script_dir/../sim/bin/occamy_chip.vsim.gui" << 'EOF'
#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $script_dir
vsim +permissive -t 1fs -sdfmax /testharness/i_dut/i_hemaia_0_0=/users/micas/shares/project_HeMAiAv2/HeMAiA_2K_3D/target/sim/testharness/hemaia_mapped.sdf -sdfreport=$script_dir/../sdf_report.txt -sdfnoerror -voptargs=+acc -do "log -r /*" -warning 8386 -work ../work-vsim \
			-ldflags "-Wl,-rpath,-L -lutil" \
			testharness_opt +permissive-off ++$1 

EOF
