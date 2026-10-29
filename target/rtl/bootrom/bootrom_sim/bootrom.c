// Copyright 2024 KU Leuven
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Yunhao Deng <yunhao.deng@kuleuven.be>

// Minimal Bootrom for HeMAiA

// Please avoid using initialized global variable (.data
// region) in C. This will lead to the loss of initial value, as they are
// allocated in spm instead of bootrom. Thus, the initial data will lose.

// For global constant, use "const var"
// For values need to share between functions, use uninitialized global variable
// + initialization function

void bootrom() {
    // Nothing to initialise: the CVA6 fetches this ROM uncached over the peripheral bus, a few
    // hundred ns an instruction, so the work lives in the application, which runs from L3.
    // L2 (the narrow SPM) is zeroed by the simulation testharness at time zero
    // (target/sim/testharness/template/load_binary.sv.tpl, clear_narrow_spm); the clusters'
    // TCDM by the host runtime's start-up (target/sw/host/runtime/start.S, initialize_tcdm).
}
