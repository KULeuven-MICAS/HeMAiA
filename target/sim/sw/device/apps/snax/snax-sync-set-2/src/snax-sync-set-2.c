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

// Addresses



int main() {

    // Set err value for checking
    int err = 0;

    uint32_t start_c0;
    uint32_t end_c0;

    uint32_t start_c1;
    uint32_t end_c1;

    for (uint32_t i = 0; i < 10; i++) {
        
   
        // Cycle 0
        if (snrt_cluster_idx() == 0) {
            if(snrt_is_compute_core()){
            
            start_c0 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 1) {
            if(snrt_is_compute_core()){
            start_c1 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        // GLOBAL BARR
        snrt_global_barrier();

        if (snrt_cluster_idx() == 0) {
            if(snrt_is_compute_core()){
            
            end_c0 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 1) {
            if(snrt_is_compute_core()){
            end_c1 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };
        

        // GLOBAL BARR
        snrt_global_barrier();

        if (snrt_cluster_idx() == 0) {
            if(snrt_is_compute_core()){
                printf("C0 %d \r\n", end_c0 - start_c0);
            };

            snrt_cluster_hw_barrier();
        };

        // GLOBAL BARR
        snrt_global_barrier();

        if (snrt_cluster_idx() == 1) {
            if(snrt_is_compute_core()){
                printf("C1 %d \r\n", end_c1 - start_c1);
            };

            snrt_cluster_hw_barrier();
        };

        // GLOBAL BARR
        snrt_global_barrier();
        
     };

    snrt_global_barrier();

    return err;
}
