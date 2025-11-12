#!/bin/bash
# helper script to run internal CI tests

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# run inside of the docker container, @pj root
build_test() {
    local DEVICE_APP="$1"
    local HOST_APP="$2"

    # Run the sequence of make commands
    make clean-sw
    make sw CFG_OVERRIDE=target/rtl/cfg/hemaia_tapeout.hjson
    make apps DEVICE_APP="$DEVICE_APP" HOST_APP="$HOST_APP"
}

# run outside of the docker container, @target/sim/bin/
run_test() {
    ./occamy_chip.vsim.gui
}

# test that must be passed
build_test snax-bingo-offload offload_bingo_gemm_intra_chiplet
run_test

build_test snax-bingo-offload offload_bingo_gemm_multi_chiplet_vanilla
run_test

# test to be debugged
# build_test snax-bingo-offload offload_bingo_gemm_multi_chiplet_broadcast
# run_test

# build_test snax-bingo-offload offload_bingo_mempool_xdma
# run_test
