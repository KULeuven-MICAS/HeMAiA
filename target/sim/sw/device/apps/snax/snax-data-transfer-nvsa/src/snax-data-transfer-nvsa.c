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

#include "hdc_data.h"
#include "gemmx_data.h"

#include "snax-gemmx-params.h"
#include "snax-gemmx-lib.h"
#include "snax-xdma-lib.h"
#include "snax-hypercorex-lib.h"

// cgra
#define cgra_debug_mode 0

#include "snax-cgra-lib.h"
#include "cgra_data.h"

// Addresses
uint32_t *tcdm_c0_sb0;
uint32_t *tcdm_c0_sb1;
uint32_t *tcdm_c0_sb2;
uint32_t *tcdm_c0_sb3;

uint32_t *tcdm_c1_sb0;
uint32_t *tcdm_c1_sb1;
uint32_t *tcdm_c1_sb2;
uint32_t *tcdm_c1_sb3;

uint32_t *tcdm_c2_sb0;
uint32_t *tcdm_c2_sb1;
uint32_t *tcdm_c2_sb2;
uint32_t *tcdm_c2_sb3;

// GeMM specific addresses
// TCDM for DMA
int8_t *local_a_dma, *local_b_dma;
int32_t *local_c_dma, *local_d32_dma;
int8_t *local_d8_dma;

// TCDM for GeMM
int8_t *local_a, *local_b;
int32_t *local_c, *local_d32;
int8_t *local_d8;

// TCDM for FC
int8_t *local_a_fc, *local_b_fc;
int32_t *local_c_fc, *local_d32_fc;
int8_t *local_d8_fc;

// Measuring barrier run-times
uint32_t barr_start;
uint32_t barr_end;


uint32_t tcdm_baseaddress;
uint8_t *tcdm_maxpool_in;
// Put the output at the middle of tcdm
uint8_t *tcdm_maxpool_out;

int main() {

    // Set err value for checking
    int err = 0;

    //----------------------------
    // Initial pre-load stage!
    //----------------------------

    
    //----------------------------
    //----------------------------
    //----------------------------
    //----------------------------
    //----------------------------
    // CGRA for MatMuls
    //----------------------------
    //----------------------------
    //----------------------------
    //----------------------------

    if (snrt_cluster_idx() == 0) {
        // Set err value for checking
        // int err = 0;
        int err_counter;

        if(snrt_is_dm_core()){
            tcdm_c0_sb0 = (uint32_t*)snrt_cluster_base_addrl();
            tcdm_c0_sb1 = tcdm_c0_sb0 + 1280;

            printf("C0 TCDM ADDR %p \r\n", tcdm_c0_sb0);
            uint32_t dma_start = snrt_mcycle();
            // Just transfer 1 block 
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            snrt_dma_start_1d(tcdm_c0_sb0, am_list, 4096);
            snrt_dma_wait_all();
            uint32_t dma_end = snrt_mcycle();
            printf("C0DM %d  \r\n", dma_end - dma_start);
            snrt_dma_start_1d(tcdm_c0_sb1, am_list, 5120);
            snrt_dma_wait_all();
            // c0_barr_start = snrt_mcycle();
        };

        snrt_cluster_hw_barrier();

        if(snrt_cluster_core_idx() == 0){
            printf("CGRA done! \n");
        };

        snrt_cluster_hw_barrier();

    } 

    snrt_global_barrier();



    //----------------------------
    //----------------------------
    //----------------------------
    //----------------------------
    //----------------------------
    // HDC Block
    //----------------------------
    //----------------------------
    //----------------------------
    //----------------------------
    //----------------------------

    if (snrt_cluster_idx() == 2){

       if(snrt_is_dm_core()){
            tcdm_c2_sb0 = (uint32_t*)snrt_cluster_base_addrl();
            tcdm_c2_sb1 = tcdm_c2_sb0 + 1280;
            printf("C2 TCDM ADDR %p \r\n", tcdm_c2_sb0);
            uint32_t dma_start = snrt_mcycle();
            // Just transfer 1 block 
            snrt_dma_start_1d(tcdm_c2_sb0, tcdm_c0_sb1, 5120);
            snrt_dma_wait_all();
            uint32_t dma_end = snrt_mcycle();
            printf("C2DM %d  \r\n", dma_end - dma_start);
            snrt_dma_start_1d(tcdm_c2_sb1, am_list, 5120);
            snrt_dma_wait_all();
            // c0_barr_start = snrt_mcycle();
        };

        snrt_cluster_hw_barrier();

        if(snrt_cluster_core_idx() == 0){
            printf("HDC done! \n");
        };

        snrt_cluster_hw_barrier();

        // if(snrt_is_compute_core){
        //     return_to_cva6_single_cluster(err);
        // }
        snrt_cluster_hw_barrier();
    };
    snrt_global_barrier();


    //----------------------------
    //----------------------------
    //----------------------------
    //----------------------------
    // GeMM for Convolutions
    //----------------------------
    //----------------------------
    //----------------------------
    //----------------------------
    if (snrt_cluster_idx() == 1){

        if(snrt_is_dm_core()){
            tcdm_c1_sb0 = (uint32_t*)snrt_cluster_base_addrl();
            tcdm_c1_sb1 = tcdm_c1_sb0 + 1280;
            printf("C1 TCDM ADDR %p \r\n", tcdm_c1_sb0);
            uint32_t dma_start = snrt_mcycle();
            // Just transfer 1 block 
            snrt_dma_start_1d(tcdm_c1_sb0, tcdm_c2_sb1, 5120);
            snrt_dma_wait_all();
            uint32_t dma_end = snrt_mcycle();
            printf("C1DM %d  \r\n", dma_end - dma_start);
            snrt_dma_start_1d(tcdm_c1_sb1, am_list, 5120);
            snrt_dma_wait_all();
            // c0_barr_start = snrt_mcycle();
        };

        snrt_cluster_hw_barrier();

        if(snrt_cluster_core_idx() == 0){
            printf("HDC done! \n");
        };

        snrt_cluster_hw_barrier();

        // if(snrt_is_compute_core){
        //     return_to_cva6_single_cluster(err);
        // }
        snrt_cluster_hw_barrier();

    };

    snrt_global_barrier();

    return err;
}
