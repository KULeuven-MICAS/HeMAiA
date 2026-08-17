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
uint32_t *tcdm_c0_in;
uint32_t *tcdm_c0_out;

uint32_t *tcdm_c1_in;
uint32_t *tcdm_c1_out;

uint32_t *tcdm_c2_in;
uint32_t *tcdm_c2_out;

uint32_t *tcdm_c3_in;
uint32_t *tcdm_c3_out;

uint32_t dma_start_c0;
uint32_t dma_end_c0;

uint32_t dma_start_c1;
uint32_t dma_end_c1;

uint32_t dma_start_c2;
uint32_t dma_end_c2;

uint32_t dma_start_c3;
uint32_t dma_end_c3;

int main() {

    // Set err value for checking
    int err = 0;

    //----------------------------
    // Pre-load stages
    // Needs to be done sequentially
    //----------------------------

    // Cluster 0
    if (snrt_cluster_idx() == 0) {
        if(snrt_is_dm_core()){
            tcdm_c0_in = (uint32_t*)snrt_cluster_base_addrl();
            tcdm_c0_out = tcdm_c0_in + half_super_bank;

            snrt_dma_start_1d(tcdm_c0_in, data_set, num_bytes);
            snrt_dma_wait_all();

            snrt_dma_start_1d(tcdm_c0_out, data_set, num_bytes);
            snrt_dma_wait_all();
        };

        snrt_cluster_hw_barrier();
    } 

    snrt_global_barrier();

    // Cluster 1
    if (snrt_cluster_idx() == 1){

       if(snrt_is_dm_core()){
            tcdm_c1_in = (uint32_t*)snrt_cluster_base_addrl();
            tcdm_c1_out = tcdm_c1_in + half_super_bank;

            snrt_dma_start_1d(tcdm_c1_in, data_set, num_bytes);
            snrt_dma_wait_all();

            snrt_dma_start_1d(tcdm_c1_out, data_set, num_bytes);
            snrt_dma_wait_all();
        };

        snrt_cluster_hw_barrier();
    };

    snrt_global_barrier();

    // Cluster 2
    if (snrt_cluster_idx() == 2){

       if(snrt_is_dm_core()){
            tcdm_c2_in = (uint32_t*)snrt_cluster_base_addrl();
            tcdm_c2_out = tcdm_c2_in + half_super_bank;

            snrt_dma_start_1d(tcdm_c2_in, data_set, num_bytes);
            snrt_dma_wait_all();

            snrt_dma_start_1d(tcdm_c2_out, data_set, num_bytes);
            snrt_dma_wait_all();
        };

        snrt_cluster_hw_barrier();
    };

    snrt_global_barrier();

    // Cluster 3
    if (snrt_cluster_idx() == 3){
         if(snrt_is_dm_core()){
            tcdm_c3_in = (uint32_t*)snrt_cluster_base_addrl();
            tcdm_c3_out = tcdm_c3_in + half_super_bank;

            snrt_dma_start_1d(tcdm_c3_in, data_set, num_bytes);
            snrt_dma_wait_all();

            snrt_dma_start_1d(tcdm_c3_out, data_set, num_bytes);
            snrt_dma_wait_all();
          };
    
          snrt_cluster_hw_barrier();
     };

    snrt_global_barrier();

    //----------------------------
    // Parallel transfer stages
    // Input-output parallel pipeline stages
    // First cluster pulls form L2 memory
    // Succeeding clusters pull continuously from the previous cluster's output
    //----------------------------

    // Cluster 0
    if (snrt_cluster_idx() == 0) {
        if(snrt_is_dm_core()){

            dma_start_c0 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c0_in, data_set, num_bytes);
            snrt_dma_wait_all();
            dma_end_c0 = snrt_mcycle();
        };

        snrt_cluster_hw_barrier();
    } 

    // Cluster 1
    if (snrt_cluster_idx() == 1) {
        if(snrt_is_dm_core()){

            dma_start_c1 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c1_in, tcdm_c0_out, num_bytes);
            snrt_dma_wait_all();
            dma_end_c1 = snrt_mcycle();
        };

        snrt_cluster_hw_barrier();
    } 

    // Cluster 2
    if (snrt_cluster_idx() == 2) {
        if(snrt_is_dm_core()){
            
            dma_start_c2 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c2_in, tcdm_c1_out, num_bytes);
            snrt_dma_wait_all();
            dma_end_c2 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    }

    // Cluster 3
    if (snrt_cluster_idx() == 3) {
        if(snrt_is_dm_core()){
            dma_start_c3 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c3_in, tcdm_c2_out, num_bytes);
            snrt_dma_wait_all();
            dma_end_c3 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    }

    //----------------------------
    // Printing sessions
    //----------------------------

    snrt_global_barrier();

    if (snrt_cluster_idx() == 0) {
        if(snrt_is_dm_core()){
            printf_safe("C0DM %d  \r\n", dma_end_c0 - dma_start_c0);
        };

        snrt_cluster_hw_barrier();
    } 

    snrt_global_barrier();

    if (snrt_cluster_idx() == 1) {
        if(snrt_is_dm_core()){
            printf_safe("C1DM %d  \r\n", dma_end_c1 - dma_start_c1);
        };

        snrt_cluster_hw_barrier();
    } 

    snrt_global_barrier();

    if (snrt_cluster_idx() == 2) {
        if(snrt_is_dm_core()){
            printf_safe("C2DM %d  \r\n", dma_end_c2 - dma_start_c2);
        };

        snrt_cluster_hw_barrier();
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 3) {
        if(snrt_is_dm_core()){
            printf_safe("C3DM %d  \r\n", dma_end_c3 - dma_start_c3);
        };

        snrt_cluster_hw_barrier();
    }

    snrt_global_barrier();

    return err;
}
