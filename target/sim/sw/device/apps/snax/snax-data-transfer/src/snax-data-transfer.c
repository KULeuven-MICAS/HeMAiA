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

uint32_t dma_start_c0;
uint32_t dma_end_c0;

uint32_t dma_start_c1;
uint32_t dma_end_c1;

uint32_t dma_start_c2;
uint32_t dma_end_c2;

int main() {

    // Set err value for checking
    int err = 0;

    //----------------------------
    // CGRA for MatMuls
    //----------------------------
    if (snrt_cluster_idx() == 0) {
        if(snrt_is_dm_core()){
            tcdm_c0_sb0 = (uint32_t*)snrt_cluster_base_addrl();
            tcdm_c0_sb1 = tcdm_c0_sb0 + num_32b;

            if (PRINT_ADDR){printf("C0ST %p \r\n", tcdm_c0_sb0);}
            dma_start_c0 = snrt_mcycle();

            snrt_dma_start_1d(tcdm_c0_sb0, data_set, num_bytes);
            snrt_dma_wait_all();
            dma_end_c0 = snrt_mcycle();
            printf("C0DM %d  \r\n", dma_end_c0 - dma_start_c0);

            snrt_dma_start_1d(tcdm_c0_sb1, data_set, num_bytes);
            snrt_dma_wait_all();
        };

        snrt_cluster_hw_barrier();
    } 

    snrt_global_barrier();


    //----------------------------
    // HDC Block
    //----------------------------
    if (snrt_cluster_idx() == 2){

       if(snrt_is_dm_core()){
            tcdm_c2_sb0 = (uint32_t*)snrt_cluster_base_addrl();
            tcdm_c2_sb1 = tcdm_c2_sb0 + num_32b;

            if (PRINT_ADDR){printf("C2ST %p \r\n", tcdm_c2_sb0);}
            dma_start_c2 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c2_sb0, tcdm_c0_sb1, num_bytes);
            snrt_dma_wait_all();
            dma_end_c2 = snrt_mcycle();
            printf("C2DM %d  \r\n", dma_end_c2 - dma_start_c2);
            snrt_dma_start_1d(tcdm_c2_sb1, data_set, num_bytes);
            snrt_dma_wait_all();
        };

        snrt_cluster_hw_barrier();
    };

    snrt_global_barrier();


    //----------------------------
    // GeMM for Convolutions
    //----------------------------
    if (snrt_cluster_idx() == 1){

        if(snrt_is_dm_core()){
            tcdm_c1_sb0 = (uint32_t*)snrt_cluster_base_addrl();
            tcdm_c1_sb1 = tcdm_c1_sb0 + num_32b;
            if (PRINT_ADDR){printf("C1ST %p \r\n", tcdm_c1_sb0);}
            dma_start_c1 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c1_sb0, tcdm_c2_sb1, num_bytes);
            snrt_dma_wait_all();
            dma_end_c1 = snrt_mcycle();
            printf("C1DM %d  \r\n", dma_end_c1 - dma_start_c1);
            snrt_dma_start_1d(tcdm_c1_sb1, data_set, num_bytes);
            snrt_dma_wait_all();
        };
        snrt_cluster_hw_barrier();

    };

    snrt_global_barrier();

    // Parallel transfers
    if (snrt_cluster_idx() == 0) {
        if(snrt_is_dm_core()){
            if (PRINT_ADDR){printf("C0ST %p \r\n", tcdm_c0_sb0);}
            dma_start_c0 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c0_sb0, data_set, num_bytes);
            snrt_dma_wait_all();
            dma_end_c0 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    } 

    if (snrt_cluster_idx() == 2) {
        if(snrt_is_dm_core()){
            if (PRINT_ADDR){printf("C2ST %p \r\n", tcdm_c2_sb0);}
            dma_start_c2 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c2_sb0, tcdm_c0_sb1, num_bytes);
            snrt_dma_wait_all();
            dma_end_c2 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    } 

    if (snrt_cluster_idx() == 1) {
        if(snrt_is_dm_core()){
            if (PRINT_ADDR){printf("C1ST %p \r\n", tcdm_c1_sb0);}
            dma_start_c1 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c1_sb0, tcdm_c2_sb1, num_bytes);
            snrt_dma_wait_all();
            dma_end_c1 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    } 

    snrt_global_barrier();

    // Parallel transfers
    if (snrt_cluster_idx() == 0) {
        if(snrt_is_dm_core()){
            if (PRINT_ADDR){printf("C0ST %p \r\n", tcdm_c0_sb0);}
            dma_start_c0 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c0_sb0, data_set, num_bytes);
            snrt_dma_wait_all();
            dma_end_c0 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    } 

    if (snrt_cluster_idx() == 2) {
        if(snrt_is_dm_core()){
            if (PRINT_ADDR){printf("C2ST %p \r\n", tcdm_c2_sb0);}
            dma_start_c2 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c2_sb0, tcdm_c0_sb1, num_bytes);
            snrt_dma_wait_all();
            dma_end_c2 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    } 

    if (snrt_cluster_idx() == 1) {
        if(snrt_is_dm_core()){
            if (PRINT_ADDR){printf("C1ST %p \r\n", tcdm_c1_sb0);}
            dma_start_c1 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c1_sb0, tcdm_c2_sb1, num_bytes);
            snrt_dma_wait_all();
            dma_end_c1 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    } 

    snrt_global_barrier();

    // Parallel transfers
    if (snrt_cluster_idx() == 0) {
        if(snrt_is_dm_core()){
            if (PRINT_ADDR){printf("C0ST %p \r\n", tcdm_c0_sb0);}
            dma_start_c0 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c0_sb0, data_set, num_bytes);
            snrt_dma_wait_all();
            dma_end_c0 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    } 

    if (snrt_cluster_idx() == 2) {
        if(snrt_is_dm_core()){
            if (PRINT_ADDR){printf("C2ST %p \r\n", tcdm_c2_sb0);}
            dma_start_c2 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c2_sb0, tcdm_c0_sb1, num_bytes);
            snrt_dma_wait_all();
            dma_end_c2 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    } 

    if (snrt_cluster_idx() == 1) {
        if(snrt_is_dm_core()){
            if (PRINT_ADDR){printf("C1ST %p \r\n", tcdm_c1_sb0);}
            dma_start_c1 = snrt_mcycle();
            snrt_dma_start_1d(tcdm_c1_sb0, tcdm_c2_sb1, num_bytes);
            snrt_dma_wait_all();
            dma_end_c1 = snrt_mcycle();
        };
        snrt_cluster_hw_barrier();
    } 

    snrt_global_barrier();

    if (snrt_cluster_idx() == 0) {
        if(snrt_is_dm_core()){
            printf("C0DM %d  \r\n", dma_end_c0 - dma_start_c0);
            printf("C2DM %d  \r\n", dma_end_c2 - dma_start_c2);
            printf("C1DM %d  \r\n", dma_end_c1 - dma_start_c1);
        };
        snrt_cluster_hw_barrier();
    } 

    snrt_global_barrier();

    return err;
}
