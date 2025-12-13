# Mamba-Occamy Workflow commands


## Build RTL/simulator

`bash scripts/build_sim.sh`

## Test

(to do)

## Debug

[bash] Run test programs

`cd target/sim`

- `bin/occamy_top.vsim sw/host/apps/offload/build/offload-snax-simbacore-main.elf | tee vsim.log`

- `bin/occamy_top.vsim.gui sw/host/apps/offload/build/offload-snax-simbacore-main.elf`

[snax] Make traces (from .dasm to .txt)

`make traces`
