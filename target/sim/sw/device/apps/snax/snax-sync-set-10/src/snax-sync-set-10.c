// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//-------------------------------
// Author: Ryan Antonio <ryan.antonio@esat.kuleuven.be>
//
// Program: Hypercorex Test CSRs
//
// This program is to test the capabilities
// of the HyperCoreX accelerator's CSRs so the test is
// to check if registers are working as intended.
//
// This includes checking for RW, RO, and WO registers.
//-------------------------------

#include "snrt.h"

#include "data.h"

#define PRINT_ADDR 0

uint32_t cycle_start[10];
uint32_t cycle_end[10];

int main() {

    // Set err value for checking
    int err = 0;

    //----------------------------
    // Start timestamps
    // Runs in parallel across all clusters
    //----------------------------
    if(snrt_is_compute_core()){
        cycle_start[snrt_cluster_idx()] = snrt_mcycle();
    };

    snrt_cluster_hw_barrier();

    // GLOBAL BARR
    snrt_global_barrier();

    //----------------------------
    // End timestamps
    // Runs in parallel across all clusters
    //----------------------------
    if(snrt_is_compute_core()){
        cycle_end[snrt_cluster_idx()] = snrt_mcycle();
    };

    snrt_cluster_hw_barrier();

    // GLOBAL BARR
    snrt_global_barrier();

    //----------------------------
    // Printing sessions
    //----------------------------

    for (uint32_t i = 0; i < 10; i++) {
        if (snrt_cluster_idx() == i) {
            if(snrt_is_compute_core()){
                printf_safe("C%d %d \n", i, cycle_end[i] - cycle_start[i]);
            };

            snrt_cluster_hw_barrier();
        }

        snrt_global_barrier();
    }

    return err;
}
