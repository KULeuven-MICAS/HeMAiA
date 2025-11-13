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
#include "snax-cgra-workload.b32.hpp"

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
    // GeMM for Convolutions
    //----------------------------
    //----------------------------
    //----------------------------
    //----------------------------
    if (snrt_cluster_idx() == 1){

        int start_cycle = 0;
        int end_cycle = 0;
        int performance_counter = 0;

        uint32_t subtraction_setting = gen_subtraction_config(subtraction_a, subtraction_b);
        uint32_t csr0 = gen_csr0_config(input_zp_i, output_zp_i, max_int_i, min_int_i);
        uint32_t csr1 = gen_csr1_config(double_round_i);

        //----------------------------
        // GeMM DMA Task
        //----------------------------
        if (snrt_is_dm_core()) {
            // tcdm_c1_sb0 = (uint32_t*)snrt_cluster_base_addrl();
            // printf("C1 TCDM ADDR %p \r\n", tcdm_c1_sb0);

            // uint32_t dma_start = snrt_mcycle();
            // snrt_dma_start_1d(tcdm_c1_sb0, vec_list, 5120);
            // snrt_dma_wait_all();
            // uint32_t dma_end = snrt_mcycle();
            // printf("C1DM %d  \r\n", dma_end - dma_start);
            // c1_barr_start = snrt_mcycle();

            //--------------------------------------------
            // Some initial settings that we can exclude
            //--------------------------------------------

            // Allocate space in TCDM for DMA
            local_a_dma = (int8_t *)(snrt_l1_next() + delta_physical_a);
            local_b_dma = (int8_t *)(snrt_l1_next() + delta_physical_b);
            local_c_dma = (int32_t *)(snrt_l1_next() + delta_physical_c);
            local_d32_dma = (int32_t *)(snrt_l1_next() + delta_physical_d32);
            local_d8_dma = (int8_t *)(snrt_l1_next() + delta_physical_d8);

            // Allocate space in TCDM for streamer
            local_a = (int8_t *)(snrt_l1_next() + delta_local_a);
            local_b = (int8_t *)(snrt_l1_next() + delta_local_b);
            local_c = (int32_t *)(snrt_l1_next() + delta_local_c);
            local_d32 = (int32_t *)(snrt_l1_next() + delta_local_d32);
            local_d8 = (int8_t *)(snrt_l1_next() + delta_local_d8);

            snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
                                    snrt_hartid());

            //--------------------------------------------
            // Main DMA transfers
            //--------------------------------------------
            if (interleaved_address == 1) {
                asm volatile("" ::: "memory");
                start_cycle = snrt_mcycle();
                snrt_dma_start_1d(local_a, A,
                                Nbatch * (H + 2 * pad_h) * (W + 2 * pad_w) *
                                    Cin * sizeof(int8_t));
                snrt_dma_start_1d(local_b, B,
                                Cout * Kh * Kw * Cin * sizeof(int8_t));+
                snrt_dma_start_1d(local_c, C,
                                M * N * meshRow * meshCol * sizeof(int32_t));
            } else {
                asm volatile("" ::: "memory");
                start_cycle = snrt_mcycle();
                snrt_dma_start_2d(
                    local_a_dma, A, 64 * sizeof(int8_t), 256, 64,
                    Nbatch * (H + 2 * pad_h) * (W + 2 * pad_w) * Cin / 64);
                snrt_dma_start_2d(local_b_dma, B, 64 * sizeof(int8_t), 256, 64,
                                Cout * Kh * Kw * Cin / 64);
                snrt_dma_start_2d(local_c_dma, C, 16 * sizeof(int32_t), 256,
                                16 * sizeof(int32_t),
                                M * N * meshRow * meshCol / 16);
            }
            snrt_dma_wait_all();
            asm volatile("" ::: "memory");
            end_cycle = snrt_mcycle();
            // printf("C1DMHW %d \r\n", snrt_get_perf_counter(SNRT_PERF_CNT0));
            // snrt_reset_perf_counter(SNRT_PERF_CNT0);
            printf("C1DMMC %d \r\n", end_cycle - start_cycle);

        }

        snrt_cluster_hw_barrier();

        //----------------------------
        // GeMM Compute Task
        //----------------------------
        if (snrt_cluster_core_idx() == 0){
            // Set Streamer configuration CSR for conv2d
            start_cycle = snrt_mcycle();
            
            set_gemmx_streamer_csr(
                1, 8, 3, 8, 3,
                80, 4, 720, 8, 0,
                1, 64, 7, 80,
                0,

                1, 8, 36, 64, 8,
                2304, 7, 0, 0,

                1, 8, 8, 448, 1,
                64, 7, 64, 0,

                8, 64, 8, 1792, 1,
                256, 7, 256, 0,

                8, 64, 8, 1792,
                1, 256, 7, 256,
                0,

                0, 2880, 35648, 21312,
                35648, 0, 0, 0,
                0, 0);

            // Set GEMMX configuration CSR
            set_gemmx_csr(
                36, 8, 7, subtraction_setting, csr0, csr1,
                857224986, 638653443,
                -1115715826, -96688076, -943963355,
                -23202385, -1919332301, -43242793,
                1269514311, 166527003, 56, 0);

            
            end_cycle = snrt_mcycle();
            printf("C1CGT %d \r\n", end_cycle - start_cycle);

            // Set CSR to start Streamer for conv2d
            start_cycle = snrt_mcycle();
            set_gemmx_streamer_start();

            // Set CSR to start GEMM
            set_gemmx_start();

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();
            end_cycle = snrt_mcycle();

            printf("C1CRT %d \r\n", end_cycle - start_cycle);

            performance_counter = read_gemmx_streamer_perf_counter();

            printf("C1CRTP %d \r\n", performance_counter);

        }

        snrt_cluster_hw_barrier();


        //----------------------------
        // Maxpool Compute Task
        //----------------------------

        if (snrt_is_dm_core()) {
            tcdm_baseaddress = snrt_cluster_base_addrl();
            tcdm_maxpool_in = (uint8_t *)tcdm_baseaddress;
            // Put the output at the middle of tcdm
            tcdm_maxpool_out = (uint8_t *)(tcdm_baseaddress + delta_local_out);

            // Put the input at the starting of tcdm
            // The xdma core is the last compute core in the cluster
            uint32_t sstride_src[1] = {0};
            uint32_t sstride_dst[1] = {0};
            uint32_t tstride_src[5] = {0};
            uint32_t tbound_src[5] = {0};
            uint32_t tstride_dst[3] = {0};
            uint32_t tbound_dst[3] = {0};

            // Load the CFG from data.h
            sstride_src[0] = spatialStride1_in;
            sstride_dst[0] = spatialStride1_out;
            tstride_src[0] = tempStride0_in;
            tstride_src[1] = tempStride1_in;
            tstride_src[2] = tempStride2_in;
            tstride_src[3] = tempStride3_in;
            tstride_src[4] = tempStride4_in;
            tbound_src[0] = tempLoop0_in;
            tbound_src[1] = tempLoop1_in;
            tbound_src[2] = tempLoop2_in;
            tbound_src[3] = tempLoop3_in;
            tbound_src[4] = tempLoop4_in;
            tstride_dst[0] = tempStride0_out;
            tstride_dst[1] = tempStride1_out;
            tstride_dst[2] = tempStride2_out;
            tbound_dst[0] = tempLoop0_out;
            tbound_dst[1] = tempLoop1_out;
            tbound_dst[2] = tempLoop2_out;

            start_cycle = snrt_mcycle();
            uint32_t check;
            check = xdma_disable_dst_ext(0);
            uint32_t ext_param_maxpool_size[1] = {reduceLen};
            check = xdma_enable_dst_ext(1, ext_param_maxpool_size);
            check = xdma_disable_dst_ext(2);

            // if (xdma_disable_dst_ext(0) != 0) {
            //     printf("Error in disabling xdma extension 0\r\n");
            //     err++;
            // }

            // uint32_t ext_param_maxpool_size[1] = {reduceLen};
            // if (xdma_enable_dst_ext(1, ext_param_maxpool_size) != 0) {
            //     printf("Error in enabling xdma extension 1\r\n");
            //     err++;
            // }

            // if (xdma_disable_dst_ext(2) != 0) {
            //     printf("Error in disabling xdma extension 2\r\n");
            //     err++;
            // }

            xdma_memcpy_nd(tcdm_maxpool_in, tcdm_maxpool_out, sstride_src, sstride_dst, 5,
                           tstride_src, tbound_src, 3, tstride_dst, tbound_dst,
                           0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
            end_cycle = snrt_mcycle();
            printf("C1MPGT %d \r\n", end_cycle - start_cycle);

            start_cycle = snrt_mcycle();
            uint32_t task_id = xdma_start();
            xdma_wait(task_id);
            end_cycle = snrt_mcycle();

            printf("C1MPRT %d \r\n", end_cycle - start_cycle);
        }

        //----------------------------
        // Data Transfer for FC layer
        //----------------------------
        snrt_cluster_hw_barrier();

        if(snrt_is_dm_core()){
            local_a_fc = (int8_t *)(snrt_l1_next() + delta_local_a_fc);
            local_b_fc = (int8_t *)(snrt_l1_next() + delta_local_b_fc);
            local_c_fc = (int32_t *)(snrt_l1_next() + delta_local_c_fc);
            local_d32_fc = (int32_t *)(snrt_l1_next() + delta_local_d32_fc);
            local_d8_fc = (int8_t *)(snrt_l1_next() + delta_local_d8_fc);

            // snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
            //                         snrt_hartid());
            start_cycle = snrt_mcycle();
            snrt_dma_start_1d(
                local_b_fc, B_fc,
                N_fc * K_fc * tileSize * meshCol * sizeof(int8_t));

            snrt_dma_wait_all();
            end_cycle = snrt_mcycle();
            printf("DMAFC %d \r\n", end_cycle - start_cycle);
            // printf("DMAFC %d \r\n", snrt_get_perf_counter(SNRT_PERF_CNT0));
            // snrt_reset_perf_counter(SNRT_PERF_CNT0);
        }

        snrt_cluster_hw_barrier();

        //----------------------------
        // Compute FC layer
        //----------------------------
        if (snrt_cluster_core_idx() == 0) {
            // Set GEMMX configuration CSR
            uint32_t subtraction_setting_fc =
                gen_subtraction_config(subtraction_a_fc, subtraction_b_fc);

            uint32_t csr0_fc = gen_csr0_config(input_zp_i_fc, output_zp_i_fc,
                                               max_int_i_fc, min_int_i_fc);
            uint32_t csr1_fc = gen_csr1_config(double_round_i_fc);

            start_cycle = snrt_mcycle();

            set_gemmx_csr(8, 64, 1, subtraction_setting_fc, csr0_fc,
                          csr1_fc, 588778812,
                          67250211, -1252922048,
                          1028264373,
                          466306066,
                          -1819985696,
                          957777513,
                          1807171125,
                          -541852707,
                          2034129355,
                          64, 0);

            // Set Streamer configuration CSR for conv2d
            set_gemmx_streamer_csr(
                1, 8, 8, 64,
                64, 0, 1, 512,
                1, 0, 1, 0,
                1, 0, 0,

                1, 8, 8, 64,
                64, 512, 1, 0,
                0,

                1, 8, 64, 64,
                1, 4096, 1, 0,
                0,

                8, 64, 64, 256,
                1, 16384, 1, 0,
                0,

                8, 64, 64,
                256, 1, 16384,
                1, 0, 0,

                0, 512, 49664,
                33280, 49664, 0,
                0, 0, 0,
                0);

            end_cycle = snrt_mcycle();

            printf("FCGT %d \r\n", end_cycle - start_cycle);

            // Set CSR to start GEMM
            set_gemmx_start();

            // Set CSR to start Streamer for conv2d
            set_gemmx_streamer_start();
            // printf("Streamer and GeMM started \r\n");
            write_csr_obs(0x00d);

            // Poll until Streamer and GEMM accelerator finish
            wait_gemmx_and_streamer();

            performance_counter = read_gemmx_streamer_perf_counter();
            printf("FCRT %d \r\n", performance_counter);

            //local_d8_fc output of the fc layer
            barr_start = snrt_mcycle();
        }

        // Measurement purposes
        // asm volatile("" ::: "memory");
        // c1_barr_start = snrt_mcycle();
        // if(snrt_is_compute_core){
        //     return_to_cva6_single_cluster(err);
        // }
        // BARRIER
        snrt_cluster_hw_barrier();
    };

    snrt_global_barrier();

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
            printf("C0 TCDM ADDR %p \r\n", tcdm_c0_sb0);
            uint32_t dma_start = snrt_mcycle();
            // Just transfer 1 block 
            snrt_dma_start_1d(tcdm_c0_sb0, local_d8_fc, 512);
            snrt_dma_wait_all();
            uint32_t dma_end = snrt_mcycle();
            printf("C0DM %d  \r\n", dma_end - dma_start);
            // c0_barr_start = snrt_mcycle();
        };

        snrt_cluster_hw_barrier();

        if(snrt_cluster_core_idx() == 0){
            printf("CGRA done! \n");
        };

        snrt_cluster_hw_barrier();

        // int32_t *local_config_data;
        // int16_t *local_d16;

        // local_config_data = (int32_t *)(snrt_l1_next() + delta_config_data);
        // local_d16 = (int16_t *)(snrt_l1_next() + delta_store_data);

        // // Using DMA only
        // if (snrt_is_dm_core()) {
        //     snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_DMA_BUSY,
        //                             snrt_hartid());
        //     local_config_data = (int32_t *)(snrt_l1_next() + delta_config_lut);
        //     snrt_dma_start_1d(local_config_data, CONFIG_LUT,
        //                       CONFIG_SIZE_LUT * sizeof(uint32_t));
        //     snrt_dma_wait_all();

        //     local_config_data = (int32_t *)(snrt_l1_next() + delta_config_data);
        //     snrt_dma_start_1d(local_config_data, CONFIG_CONST,
        //                       CONFIG_SIZE_DATA * sizeof(uint32_t));
        //     snrt_dma_wait_all();

        //     if (CONFIG_SIZE_CMD_0 != 0) {
        //         local_config_data =
        //             (int32_t *)(snrt_l1_next() + delta_config_cmd_0);
        //         snrt_dma_start_1d(local_config_data, CONFIG_CMD,
        //                           CONFIG_SIZE_CMD_0 * sizeof(uint32_t));
        //         snrt_dma_wait_all();
        //     }

        //     local_config_data =
        //         (int32_t *)(snrt_l1_next() + delta_config_cmd_ss);
        //     snrt_dma_start_1d(local_config_data, CONFIG_CMD_SS,
        //                       CONFIG_SIZE_CMD_SS * sizeof(uint32_t));
        //     snrt_dma_wait_all();

        //     local_config_data = (int32_t *)(snrt_l1_next() + delta_comp_data);

        //     snrt_dma_start_1d(local_config_data, COMP_DATA,
        //                     COMP_DATA_SIZE * sizeof(uint32_t));
        //     snrt_dma_wait_all();
    
        //     printf("DMA %d \r\n", snrt_get_perf_counter(SNRT_PERF_CNT0));
        //     snrt_reset_perf_counter(SNRT_PERF_CNT0);
        // }
    
        // snrt_cluster_hw_barrier();

        // // testing csr -> cgra
        // if (snrt_is_compute_core()) {
        //     uint32_t mcycle_timestamps[7];

        //     launch_cgra_0_config(delta_config_data, delta_comp_data,
        //                          delta_store_data, mcycle_timestamps);
        //     launch_cgra_0_go(delta_config_data, delta_comp_data,
        //                      delta_store_data, mcycle_timestamps);

        //     cgra_hw_barrier_fast(10, 1e5);

        //     launch_cgra_0_relaunch(mcycle_timestamps);

        //     cgra_hw_barrier_fast(10, 1e5);
            
        //     // // cgra_hw_barrier(10, 1e5, 1, 1);
        //     // cgra_hw_barrier_fast(10, 1e5);

        //     cgra_hw_profiler();

		// 	printf("mcycle cgra_init = %d\r\n", mcycle_timestamps[1] - mcycle_timestamps[0]);
		// 	printf("mcycle cgra_config_prep = %d\r\n", mcycle_timestamps[2] - mcycle_timestamps[1]);
		// 	printf("mcycle cgra_data_prep = %d\r\n", mcycle_timestamps[3] - mcycle_timestamps[2]);
		// 	printf("mcycle cgra_launch_init = %d\r\n", mcycle_timestamps[4] - mcycle_timestamps[3]);
		// 	printf("mcycle cgra_relaunch_init = %d\r\n", mcycle_timestamps[5] - mcycle_timestamps[4]);
		// 	printf("mcycle cgra_relaunch_done = %d\r\n", mcycle_timestamps[6] - mcycle_timestamps[5]);



        // }
        // snrt_cluster_hw_barrier();
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

        // MEASUREMENT TIME
        // if(snrt_cluster_core_idx() == 0){
        //     barr_end = snrt_mcycle();
        //     printf("BS %d \r\n", barr_start);
        //     printf("BE %d \r\n", barr_end);
        // }
        // BARRIER
        snrt_cluster_hw_barrier();

        // MOVE DATA FROM C0 TO C2
        if (snrt_is_dm_core()) {
            // Layout assignment
            tcdm_c2_sb0 = (uint32_t *)snrt_cluster_base_addrl();
            tcdm_c2_sb1 = tcdm_c2_sb0 + 16;
            tcdm_c2_sb2 = tcdm_c2_sb1 + 16;
            tcdm_c2_sb3 = tcdm_c2_sb2 + 16;

            printf("C2 TCDM ADDR %p \r\n", tcdm_c2_sb0);

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
                tcdm_c2_sb1, tcdm_c0_sb0,
                // Size per chunk
                src_stride,
                // Destination stride, source stride
                dst_stride, src_stride,
                // Number of times to do
                num_of_rows_for_vectors
            );

            snrt_dma_wait_all();
            uint32_t dma_end = snrt_mcycle();
            printf("C2DM %d  \r\n", dma_end - dma_start);
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
            printf("C2CT %d\n", end_convert - start_convert);

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
            printf("C2CGT: %d\n", hdc_cfg_end - hdc_cfg_start);


            uint32_t hdc_core_start = snrt_mcycle();
            // Start hypercorex
            csrw_ss(HYPERCOREX_CORE_SET_REG_ADDR, 0x00000011);

            // Poll the busy-state of Hypercorex
            // Check both the Hypercorex and Streamer
            while ( hypercorex_is_streamer_busy()) {
            };

            // csrr_ss(STREAMER_BUSY_CSR)
            uint32_t hdc_core_end = snrt_mcycle();
            printf("C2CRT %d \r\n", hdc_core_end - hdc_core_start);

            csrw_ss(HYPERCOREX_CORE_SET_REG_ADDR, 0x00000040);

            printf("D %d \r\n",(uint32_t) * (tcdm_c2_sb3));

            printf("D finish!");
            printf("D finish!");
            printf("D finish!");
            printf("D finish!");
            printf("D finish!");
            

            // for (uint32_t i = 0; i < vec_num; i++) {
            //     printf("D %d \n",(uint32_t) * (tcdm_c2_sb3 + i * 64));
            // };
        }

        // if(snrt_is_compute_core){
        //     return_to_cva6_single_cluster(err);
        // }
        snrt_cluster_hw_barrier();
    };
    snrt_global_barrier();
    return err;
}
