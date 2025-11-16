#!/bin/bash
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "$script_dir/../../hw/hemaia/tech_cells_tsmc16/src/tsmc16/mem_macro"
./gen_mem.sh
