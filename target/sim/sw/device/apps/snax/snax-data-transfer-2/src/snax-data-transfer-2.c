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

// #define DMA_MCYCLE

// Addresses
uint32_t *tcdm_in[2];
uint32_t *tcdm_out[3];

#ifndef DMA_MCYCLE
uint32_t dma_start[2];
#endif

uint32_t dma_end[2];

int main() {

    // Set err value for checking
    int err = 0;

    //----------------------------
    // Pre-load stages
    // Runs in parallel across all clusters
    //----------------------------
    if(snrt_is_dm_core()){
        tcdm_in[snrt_cluster_idx()] = (uint32_t*)snrt_cluster_base_addrl();
        tcdm_out[snrt_cluster_idx() + 1] = tcdm_in[snrt_cluster_idx()] + half_super_bank;

        // tcdm_out[0] is a sentinel: it stands in for the "previous cluster"
        // of cluster 0, so the transfer stage can index tcdm_out uniformly.
        if (snrt_cluster_idx() == 0) {
            tcdm_out[0] = data_set;
        }

        snrt_dma_start_1d(tcdm_in[snrt_cluster_idx()], data_set, num_bytes);
        snrt_dma_wait_all();

        snrt_dma_start_1d(tcdm_out[snrt_cluster_idx() + 1], data_set, num_bytes);
        snrt_dma_wait_all();
    };

    snrt_cluster_hw_barrier();

    // All clusters must finish pre-loading before the pipeline stage
    // can safely read a previous cluster's output.
    snrt_global_barrier();

    //----------------------------
    // Parallel transfer stages
    // Input-output parallel pipeline stages
    // First cluster pulls form L2 memory
    // Succeeding clusters pull continuously from the previous cluster's output
    //----------------------------
    if(snrt_is_dm_core()){
#ifdef DMA_MCYCLE
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY, snrt_hartid());
#else
        dma_start[snrt_cluster_idx()] = snrt_mcycle();
#endif
        snrt_dma_start_1d(tcdm_in[snrt_cluster_idx()], tcdm_out[snrt_cluster_idx()], num_bytes);
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
#ifdef DMA_MCYCLE
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY, snrt_hartid());
#else
        dma_start[snrt_cluster_idx()] = snrt_mcycle();
#endif
        snrt_dma_start_1d(tcdm_in[snrt_cluster_idx()], tcdm_out[snrt_cluster_idx()], num_bytes);
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
#ifdef DMA_MCYCLE
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY, snrt_hartid());
#else
        dma_start[snrt_cluster_idx()] = snrt_mcycle();
#endif
        snrt_dma_start_1d(tcdm_in[snrt_cluster_idx()], tcdm_out[snrt_cluster_idx()], num_bytes);
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

    for (uint32_t i = 0; i < 2; i++) {
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
