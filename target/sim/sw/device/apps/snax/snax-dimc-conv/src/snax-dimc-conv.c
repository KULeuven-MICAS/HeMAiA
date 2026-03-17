// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"
#include "snax-dimc-csr.h"
#include "snax-dimc-lib.h"
#include "data_conv.h"

static int err = 0;

int main() {

    // printf("Starting the DIMC CONV test\r\n");

    // Set err value for checking
    err = 0;
    if(snrt_cluster_idx() == 3) {

        uint64_t kernel_length = 256;
        uint64_t qkv_activation_length = Q_LENGTH;
        uint64_t qkt_activation_length = Q1K1T_LENGTH;

        uint8_t  *local_kernel;
        uint64_t *local_activation_qkv, *local_activation_qkt, *local_final_res;

        local_kernel = (uint8_t *)snrt_l1_next();
        local_activation_qkv = (uint64_t *)(local_kernel + kernel_length);
        local_activation_qkt = (uint64_t *)(local_activation_qkv + qkv_activation_length);
        local_final_res = (uint64_t *)(local_activation_qkt + qkt_activation_length);

        if(snrt_is_dm_core()) {
            // printf("DMA core will be configured\r\n");

            // This measures the start of cycle count
            // for preloading the data to the L1 memory
            uint32_t start_dma_load = snrt_mcycle();
            /**********************************************************************/
            // initialize TCDM by DMA
            /**********************************************************************/
            // printf("INITIALIZING TCDM\r\n");

            // read matrix Q from data.h
            int kernel_size = kernel_length * sizeof(uint8_t);
            int activation_size_qkv = qkv_activation_length * sizeof(uint64_t);
            int activation_size_qkt = qkt_activation_length * sizeof(uint64_t);

            snrt_dma_start_1d(local_kernel,         kernel,  kernel_size);
            snrt_dma_start_1d(local_activation_qkv, Q,       activation_size_qkv);
            snrt_dma_start_1d(local_activation_qkt, Q1K1T_G, activation_size_qkt);

            // Measure the end of the transfer process
            snrt_dma_wait_all();

            uint32_t end_dma_load = snrt_mcycle();
            printf("DMTI %d\n", end_dma_load - start_dma_load);

            // printf("DMA core exits\r\n"); 
        }

        // Wait until DMA transfer is done
        snrt_cluster_hw_barrier();

        if (snrt_is_compute_core()){
            // printf("COMPUTE CORE\r\n");

            // printf("ENTERING CONV MODE\r\n");

            // uint32_t busy = dimc_query_busy();
            // printf("%d: busy\r\n", busy);
            // printf("QUERYING BUSY SUCCEEDED\r\n");
            uint32_t core_config_start = snrt_mcycle();
            configure_accelerator();

            // printf("CONFIGURING ACCELERATOR SUCCEEDED\r\n");

            // busy = dimc_query_busy();
            // printf("%d: busy\r\n", busy);
            // printf("QUERYING BUSY SUCCEEDED\r\n");

            // printf("CONFIGURING STREAMERS for KERNEL\r\n");
            dimc_set_streamer_dim_w((64*16), 1, 64, 0, 8, (uint32_t)(local_final_res));
            dimc_set_streamer_dim_r0(145, 1, 256, 0, 8, (uint32_t)(local_kernel));
            dimc_set_streamer_dim_r1(145, 1, 256, 0, 8, (uint32_t)(local_kernel + 64));
            dimc_set_streamer_dim_r2(145, 1, 256, 0, 8, (uint32_t)(local_kernel + 128));
            dimc_set_streamer_dim_r3(145, 1, 256, 0, 8, (uint32_t)(local_kernel + 192));

            uint32_t core_config_end = snrt_mcycle();
            printf("CGTP %d\n", core_config_end - core_config_start);

            uint32_t core_start = snrt_mcycle();
            dimc_start_streamer();

            dimc_start_conv();

            // LOAD KERNELS
            /*
            printf("CONFIGURING STREAMERS for KERNEL\r\n");
            dimc_set_streamer_dim_w(0, 0, 0, 0, 0, 0);
            dimc_set_streamer_dim_r0(1, 1, 256, 0, 8, (uint32_t)(local_kernel));
            dimc_set_streamer_dim_r1(1, 1, 256, 0, 8, (uint32_t)(local_kernel + 64));
            dimc_set_streamer_dim_r2(1, 1, 256, 0, 8, (uint32_t)(local_kernel + 128));
            dimc_set_streamer_dim_r3(1, 1, 256, 0, 8, (uint32_t)(local_kernel + 192));

            dimc_start_streamer();

            dimc_start_conv();

            while (dimc_is_streamer_busy()) {
                // printf("STREAMER BUSY\r\n");
            }

            // LOAD ACTIVATION
            printf("CONFIGURING STREAMERS for ACTIVATION\r\n");
            dimc_set_streamer_dim_w((64*16), 1, 64, 0, 8, (uint32_t)(local_final_res));
            dimc_set_streamer_dim_r0(144, 1, 256, 0, 8, (uint32_t)(local_activation_qkv));
            dimc_set_streamer_dim_r1(144, 1, 256, 0, 8, (uint32_t)(local_activation_qkv + 8));
            dimc_set_streamer_dim_r2(144, 1, 256, 0, 8, (uint32_t)(local_activation_qkv + 16));
            dimc_set_streamer_dim_r3(144, 1, 256, 0, 8, (uint32_t)(local_activation_qkv + 24));
            // printf("STREAMER CONFIGURED for ACTIVATION\r\n");

            dimc_start_streamer();
            */

            // busy = dimc_query_busy();
            // printf("%d: busy\r\n", busy);
            // printf("QUERYING BUSY SUCCEEDED\r\n");


            // Wait until the accelerator is done
            while (dimc_is_streamer_busy()) { }

            uint32_t core_end = snrt_mcycle();
            printf("CRTP %d\n", core_end - core_start);

            printf("GO CHECK FINAL RESULT\r\n");

            int mismatch = 0;

            // const int iterations = 8192 / 8;
            for (int i = 0; i < (8192 / 8); i++) {
                // Pointers to the current chunks
                uint64_t* res_chunk = &local_final_res[i * 8];
                uint8_t* gold_chunk = &gold_conv[i * 36];

                // Buffer to hold bits from local_final_res
                uint8_t res_bits[64]; // 512 bits / 8 bits per byte = 64 bytes

                // Convert local_final_res chunk to bytes
                for (int j = 0; j < 8; j++) {
                    uint64_t val = res_chunk[j];
                    for (int k = 0; k < 8; ++k) {
                        res_bits[j * 8 + k] = (val >> (k * 8)) & 0xFF; // Little-endian
                    }
                }

                // Buffer to hold bits from gold_conv with padding

                volatile uint8_t gold_bits[64];

                for (int j = 0; j < 36; j++) {
                    gold_bits[j] = gold_chunk[j];
                }

                for (int j = 36; j < 64; j++) {
                    gold_bits[j] = 0;
                }

                // Compare the two byte arrays
                bool match = true;
                for (int j = 0; j < 64; j++) {
                    if (res_bits[j] != gold_bits[j]) {
                        match = false;
                        err += 1;
                        printf("fuck\r\n");
                        break;
                    }
                }

            }
            printf("MISMATCHES: %d\r\n", err);
        }
    }

    return err;
}
