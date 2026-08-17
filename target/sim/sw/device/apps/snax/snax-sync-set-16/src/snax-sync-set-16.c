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

    uint32_t start_c2;
    uint32_t end_c2;

    uint32_t start_c3;
    uint32_t end_c3;

    uint32_t start_c4;
    uint32_t end_c4;

    uint32_t start_c5;
    uint32_t end_c5;

    uint32_t start_c6;
    uint32_t end_c6;

    uint32_t start_c7;
    uint32_t end_c7;

    uint32_t start_c8;
    uint32_t end_c8;

    uint32_t start_c9;
    uint32_t end_c9;


    // for (uint32_t i = 0; i < 10; i++) {
        
   
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

        if (snrt_cluster_idx() == 2) {
            if(snrt_is_compute_core()){
            start_c2 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 3) {
            if(snrt_is_compute_core()){
            start_c3 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 4) {
            if(snrt_is_compute_core()){
            start_c4 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 5) {
            if(snrt_is_compute_core()){
            start_c5 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 6) {
            if(snrt_is_compute_core()){
            start_c6 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 7) {
            if(snrt_is_compute_core()){
            start_c7 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 8) {
            if(snrt_is_compute_core()){
            start_c8 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 9) {
            if(snrt_is_compute_core()){
            start_c9 = snrt_mcycle();
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

        if (snrt_cluster_idx() == 2) {
            if(snrt_is_compute_core()){
            end_c2 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 3) {
            if(snrt_is_compute_core()){
            end_c3 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 4) {
            if(snrt_is_compute_core()){
            end_c4 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 5) {
            if(snrt_is_compute_core()){
            end_c5 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

    
        if (snrt_cluster_idx() == 6) {
            if(snrt_is_compute_core()){
            end_c6 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 7) {
            if(snrt_is_compute_core()){
            end_c7 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 8) {
            if(snrt_is_compute_core()){
            end_c8 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };

        if (snrt_cluster_idx() == 9) {
            if(snrt_is_compute_core()){
            end_c9 = snrt_mcycle();
            };

            snrt_cluster_hw_barrier();
        };
        

        // GLOBAL BARR
        snrt_global_barrier();

        if (snrt_cluster_idx() == 0) {
            if(snrt_is_compute_core()){
                printf_safe("C0 %d \n", end_c0 - start_c0);
            };

            snrt_cluster_hw_barrier();
        };

        // GLOBAL BARR
        snrt_global_barrier();

        if (snrt_cluster_idx() == 1) {
            if(snrt_is_compute_core()){
                printf_safe("C1 %d \n", end_c1 - start_c1);
            };

            snrt_cluster_hw_barrier();
        };

        // GLOBAL BARR
        snrt_global_barrier();

        if (snrt_cluster_idx() == 2) {
            if(snrt_is_compute_core()){
                printf_safe("C2 %d \n", end_c2 - start_c2);
            };

            snrt_cluster_hw_barrier();
        };

        // GLOBAL BARR
        snrt_global_barrier();

        if (snrt_cluster_idx() == 3) {
            if(snrt_is_compute_core()){
                printf_safe("C3 %d \n", end_c3 - start_c3);
            };

            snrt_cluster_hw_barrier();
        };

        snrt_global_barrier();

        if (snrt_cluster_idx() == 4) {
            if(snrt_is_compute_core()){
                printf_safe("C4 %d \n", end_c4 - start_c4);
            };

            snrt_cluster_hw_barrier();
        };

        snrt_global_barrier();

        if (snrt_cluster_idx() == 5) {
            if(snrt_is_compute_core()){
                printf_safe("C5 %d \n", end_c5 - start_c5);
            };

            snrt_cluster_hw_barrier();
        };

        snrt_global_barrier();

        if (snrt_cluster_idx() == 6) {
            if(snrt_is_compute_core()){
                printf_safe("C6 %d \n", end_c6 - start_c6);
            };

            snrt_cluster_hw_barrier();
        };

        snrt_global_barrier();

        if (snrt_cluster_idx() == 7) {
            if(snrt_is_compute_core()){
                printf_safe("C7 %d \n", end_c7 - start_c7);
            };

            snrt_cluster_hw_barrier();
        }

        snrt_global_barrier();

        if (snrt_cluster_idx() == 8) {
            if(snrt_is_compute_core()){
                printf_safe("C8 %d \n", end_c8 - start_c8);
            };

            snrt_cluster_hw_barrier();
        }

        snrt_global_barrier();

        if (snrt_cluster_idx() == 9) {
            if(snrt_is_compute_core()){
                printf_safe("C9 %d \n", end_c9 - start_c9);
            };

            snrt_cluster_hw_barrier();
        }

        snrt_global_barrier();
        
    // };


    snrt_global_barrier();

    return err;
}
