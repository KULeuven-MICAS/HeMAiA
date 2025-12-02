#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cfg_name="hemaia_tapeout.hjson"

# 
FILE=/users/micas/shares/project_HeMAiAv2/HeMAiA/hw/hemaia/tech_cells_tsmc16/src/tsmc16/mem_macro/ts1n16ffcllsblvtd1024x64m4sws_150a/VERILOG/ts1n16ffcllsblvtd1024x64m4sws_150a.v

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


cp -f "$script_dir/../sim/testharness/testharness_netlist.sv" "$script_dir/../sim/testharness/testharness.sv"
cp -f "$script_dir/../sim/work-vsim/compile_netlist.vsim.tcl" "$script_dir/../sim/work-vsim/compile.vsim.tcl"

make -C "$script_dir/../.." hemaia_system_vsim
