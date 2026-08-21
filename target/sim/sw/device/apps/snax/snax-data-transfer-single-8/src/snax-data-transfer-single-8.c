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

#define DMA_MCYCLE

#ifndef DMA_MCYCLE
uint32_t dma_start[8];
#endif

// Addresses
uint32_t *tcdm_in[8];

uint32_t dma_end[8];

int main() {

    // Set err value for checking
    int err = 0;

    //----------------------------
    // Parallel loading
    //----------------------------
    if(snrt_is_dm_core()){
        tcdm_in[snrt_cluster_idx()] = (uint32_t*)snrt_cluster_base_addrl();
#ifdef DMA_MCYCLE
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY, snrt_hartid());
#else
        dma_start[snrt_cluster_idx()] = snrt_mcycle();
#endif
        snrt_dma_start_1d(tcdm_in[snrt_cluster_idx()], data_set, num_bytes);
        snrt_dma_wait_all();
#ifdef DMA_MCYCLE
        dma_end[snrt_cluster_idx()] = snrt_get_perf_counter(SNRT_PERF_CNT0);
#else
        dma_end[snrt_cluster_idx()] = snrt_mcycle();
#endif
    };

    snrt_cluster_hw_barrier();
    snrt_global_barrier();

    if(snrt_is_dm_core()){
        tcdm_in[snrt_cluster_idx()] = (uint32_t*)snrt_cluster_base_addrl();
#ifdef DMA_MCYCLE
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY, snrt_hartid());
#else
        dma_start[snrt_cluster_idx()] = snrt_mcycle();
#endif
        snrt_dma_start_1d(tcdm_in[snrt_cluster_idx()], data_set, num_bytes);
        snrt_dma_wait_all();
#ifdef DMA_MCYCLE
        dma_end[snrt_cluster_idx()] = snrt_get_perf_counter(SNRT_PERF_CNT0);
#else
        dma_end[snrt_cluster_idx()] = snrt_mcycle();
#endif
    };

    snrt_cluster_hw_barrier();
    snrt_global_barrier();

    if(snrt_is_dm_core()){
        tcdm_in[snrt_cluster_idx()] = (uint32_t*)snrt_cluster_base_addrl();
#ifdef DMA_MCYCLE
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY, snrt_hartid());
#else
        dma_start[snrt_cluster_idx()] = snrt_mcycle();
#endif
        snrt_dma_start_1d(tcdm_in[snrt_cluster_idx()], data_set, num_bytes);
        snrt_dma_wait_all();
#ifdef DMA_MCYCLE
        dma_end[snrt_cluster_idx()] = snrt_get_perf_counter(SNRT_PERF_CNT0);
#else
        dma_end[snrt_cluster_idx()] = snrt_mcycle();
#endif
    };

    snrt_cluster_hw_barrier();
    snrt_global_barrier();
    

    //----------------------------
    // Printing sessions
    //----------------------------

    for (uint32_t i = 0; i < 8; i++) {
        snrt_global_barrier();

        if (snrt_cluster_idx() == i) {
            if(snrt_is_dm_core()){
#ifdef DMA_MCYCLE
                printf_safe("C%d %d  \r\n", i, dma_end[i]);
#else
                printf_safe("C%d %d  \r\n", i, dma_end[i] - dma_start[i]);
#endif
            };

            snrt_cluster_hw_barrier();
        }
    }

    snrt_global_barrier();

    return err;
}
