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
#include "snax-hypercorex-lib.h"
#include "streamer_csr_addr_map.h"

int main() {

    // Set err value for checking
    int err = 0;

    if (snrt_cluster_idx() == 2){

        uint32_t *tcdm_c2_sb0;
        uint32_t *tcdm_c2_sb1;
        uint32_t *tcdm_c2_sb2;
        uint32_t *tcdm_c2_sb3;

        snrt_cluster_hw_barrier();

        // MOVE DATA FROM C0 TO C2
        if (snrt_is_dm_core()) {
            // Layout assignment
            tcdm_c2_sb0 = (uint32_t *)snrt_cluster_base_addrl();
            tcdm_c2_sb1 = tcdm_c2_sb0 + 16;
            tcdm_c2_sb2 = tcdm_c2_sb1 + 16;
            tcdm_c2_sb3 = tcdm_c2_sb2 + 16;


            size_t src_stride = 8 * sizeof(uint64_t);
            size_t dst_stride = 4 * src_stride;
            uint32_t num_of_rows_for_vectors = vec_num * 8;

            uint32_t dma_start = snrt_mcycle();

            // Initial AM loading
            snrt_dma_start_2d(
            // Destination address, source address
                tcdm_c2_sb0, am_list,
                // Size per chunk
                src_stride,
                // Destination stride, source stride
                dst_stride, src_stride,
                // Number of times to do
                num_am_search
            );

            // Data transferring
            snrt_dma_start_2d(
                // Destination address, source address
                tcdm_c2_sb1, vec_list,
                // Size per chunk
                src_stride,
                // Destination stride, source stride
                dst_stride, src_stride,
                // Number of times to do
                num_of_rows_for_vectors
            );

            snrt_dma_wait_all();
            uint32_t dma_end = snrt_mcycle();
            printf("[HPX] DMA %d \r\n", dma_end - dma_start);
        }

        // BARRIER
        snrt_cluster_hw_barrier();

        
        if(snrt_cluster_core_idx() == 0){
            // CONVERT AND COMPRESS
            uint32_t start_convert = snrt_mcycle();
            hypercorex_pack_512b(
                vec_num,
                tcdm_c2_sb1,
                tcdm_c2_sb3
            );
            uint32_t end_convert = snrt_mcycle();
            printf("[HPX] BIN %d \r\n", end_convert - start_convert);

            // ACTIVATE HDC TASK
            uint32_t hdc_cfg_start = snrt_mcycle();
            // Configure streamer for the input
            hypercorex_set_streamer_highdim_a(
                (uint32_t)tcdm_c2_sb2,  // Base pointer low
                0,                       // Base pointer high
                8,                       // Spatial stride
                vec_num,         // Inner loop bound
                1,                       // Outer loop bound
                256,                     // Inner loop stride
                0                        // Outer loop stride
            );

            // Configure streamer for the AM
            hypercorex_set_streamer_highdim_am(
                (uint32_t)tcdm_c2_sb0,  // Base pointer low
                0,                       // Base pointer high
                8,                       // Spatial stride
                num_am_search,          // Inner loop bound
                vec_num,                // Outer loop bound
                256,                     // Inner loop stride
                0                        // Outer loop stride
            );

            // Configure streamer for low dim predictions
            hypercorex_set_streamer_lowdim_predict(
                (uint32_t)tcdm_c2_sb3,  // Base pointer low
                0,                        // Base pointer high
                1,                        // Spatial stride
                vec_num,          // Inner loop bound
                1,                        // Outer loop bound
                256,                      // Inner loop stride
                0                         // Outer loop stride
            );

            // Start the streamers
            hypercorex_start_streamer();

            //-------------------------------
            // Configuring the Hypercorex
            //-------------------------------

            // Load instructions for hypercorex
            hypercorex_load_inst(5, 0, am_search_code);

            // Set number of classes to be predicted
            // During AM search mode
            csrw_ss(HYPERCOREX_AM_NUM_PREDICT_REG_ADDR, num_am_search);

            // Enable loop mode to 2D
            csrw_ss(HYPERCOREX_INST_LOOP_CTRL_REG_ADDR, 0x00000002);

            // Set loop jump addresses
            hypercorex_set_inst_loop_jump_addr(3, 0, 0);

            // Set loop end addresses
            hypercorex_set_inst_loop_end_addr(3, 4, 0);

            // Set loop counts
            hypercorex_set_inst_loop_count(num_am_search, vec_num, 0);

            // Write control registers
            csrw_ss(HYPERCOREX_CORE_SET_REG_ADDR, 0x00000010);

            uint32_t hdc_cfg_end = snrt_mcycle();
            printf("[HPX] CFG: %d \r\n", hdc_cfg_end - hdc_cfg_start);


            uint32_t hdc_core_start = snrt_mcycle();
            // Start hypercorex
            csrw_ss(HYPERCOREX_CORE_SET_REG_ADDR, 0x00000011);

            // Poll the busy-state of Hypercorex
            // Check both the Hypercorex and Streamer
            while ( hypercorex_is_streamer_busy()) {
            };

            // csrr_ss(STREAMER_BUSY_CSR)
            uint32_t hdc_core_end = snrt_mcycle();
            printf("[HPX] AM %d \r\n", hdc_core_end - hdc_core_start);

            // Reset core
            csrw_ss(HYPERCOREX_CORE_SET_REG_ADDR, 0x00000040);
        }

        snrt_cluster_hw_barrier();

        return_to_cva6_single_cluster(err);
    };
}
