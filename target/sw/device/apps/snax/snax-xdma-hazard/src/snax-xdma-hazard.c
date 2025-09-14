// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yunhao Deng <yunhao.deng@kuleuven.be>
// Fanchen Kong <fanchen.kong@kuleuven.be>

// This SW is used to test the XDMA read and write between L3 and TCDM
// and between TCDMs of different clusters.
// The test includes 4 parts:
// 1. All clusters read data from L3 to their TCDM at the same time
// 2. All clusters (except cluster 0) read data from previous cluster's TCDM
// C1 read C0, C2 read C1, C3 read C2, ..., Cn read C(n-1)
// 3. All clusters (except the last cluster) write data to next cluster's TCDM
// C0 write C1, C1 write C2, C2 write C3, ..., C(n-1) write Cn
// 4. All clusters write data back to L3 from their TCDM at the same time
// The data size is defined in data.h file, which is 8192B by default.

#include "data.h"
#include "snrt.h"

uint32_t time[SNRT_CLUSTER_NUM];

int main() {
    // Set err value for checking
    int err = 0;
    // Obtain the start address of the TCDM memory
    uint32_t dma_load_input_start;
    uint32_t dma_load_input_end;
    uint32_t tcdm_baseaddress = snrt_cluster_base_addrl();
    if (snrt_global_core_idx() == 0) {
        printf("Now start to let clusters to read %dB data from L3\r\n", data_size * sizeof(data[0]));
    }
    if (snrt_is_dm_core()) {
        // All clusters try to read data from L3 at the same time
        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma writer extension 0\n");
            err++;
        }
        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma writer extension 1\n");
            err++;
        }
        if (xdma_disable_src_ext(0) != 0) {
            printf("Error in disabling xdma reader extension 0\n");
            err++;
        }
        xdma_memcpy_1d(data, (void *)tcdm_baseaddress,
                       data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        time[snrt_cluster_idx()] = xdma_last_task_cycle();
    }

    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        for (int i = 0; i < SNRT_CLUSTER_NUM; i++) {
            printf("XDMA remote read from L3 to TCDM C%d is done in %d cycles.\r\n", i,
                   time[i]);
        }
    }
    snrt_global_barrier();

    if (snrt_global_core_idx() == 0) {
        printf("Now start to read %dB data between Clusters\r\n", data_size * sizeof(data[0]));
    }
    if ((snrt_cluster_idx() != 0) && snrt_is_dm_core()) {
        // All remaining clusters try to read data from previous cluster's TCDM
        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma writer extension 0\n");
            err++;
        }
        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma writer extension 1\n");
            err++;
        }
        if (xdma_disable_src_ext(0) != 0) {
            printf("Error in disabling xdma reader extension 0\n");
            err++;
        }

        xdma_memcpy_1d((void *)tcdm_baseaddress - cluster_offset,
                       (void *)tcdm_baseaddress, data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        time[snrt_cluster_idx()] = xdma_last_task_cycle();
    }

    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        for (int i = 1; i < SNRT_CLUSTER_NUM; i++) {
            printf(
                "XDMA remote read from TCDM C%d to TCDM C%d is done in %d cycles.\r\n",
                i - 1, i, time[i]);
        }
    }
    snrt_global_barrier();

    if (snrt_global_core_idx() == 0) {
        printf("Now start to write %dB data between Clusters\r\n", data_size * sizeof(data[0]));
    }
    if ((snrt_cluster_idx() != snrt_cluster_num() - 1) && snrt_is_dm_core()) {
        // All remaining clusters try to write data to next cluster's TCDM
        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma writer extension 0\n");
            err++;
        }
        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma writer extension 1\n");
            err++;
        }
        if (xdma_disable_src_ext(0) != 0) {
            printf("Error in disabling xdma reader extension 0\n");
            err++;
        }
        xdma_memcpy_1d((void *)tcdm_baseaddress,
                       (void *)tcdm_baseaddress + cluster_offset,
                       data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        time[snrt_cluster_idx()] = xdma_last_task_cycle();
    }

    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        for (int i = 0; i < SNRT_CLUSTER_NUM - 1; i++) {
            printf(
                "XDMA remote write from TCDM C%d to TCDM C%d is done in %d cycles.\r\n",
                i, i + 1, time[i]);
        }
    }
    snrt_global_barrier();

#ifdef XDMA_CHECK_RESULT
    if (snrt_is_dm_core()) {
        // Check the result
        uint32_t *golden_result = (uint32_t *)tcdm_baseaddress;
        uint32_t *tcdm_result = (uint32_t *)data;

        for (int i = 0; i < data_size * sizeof(data[0]) / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("The data copy is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
    }
    snrt_global_barrier();
#endif
    if (snrt_global_core_idx() == 0) {
        printf("Now start to let cluster to write back the %dB data to L3\r\n", data_size * sizeof(data[0]));
    }    
    if (snrt_is_dm_core()) {
        // All clusters try to read data from L3 at the same time
        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma writer extension 0\n");
            err++;
        }
        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma writer extension 1\n");
            err++;
        }
        if (xdma_disable_src_ext(0) != 0) {
            printf("Error in disabling xdma reader extension 0\n");
            err++;
        }
        xdma_memcpy_1d((void *)tcdm_baseaddress, data,
                       data_size * sizeof(data[0]));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        time[snrt_cluster_idx()] = xdma_last_task_cycle();
    }
    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        for (int i = 0; i < SNRT_CLUSTER_NUM; i++) {
            printf("XDMA remote write from TCDM C%d to L3 is done in %d cycles.\r\n", i,
                   time[i]);
        }
    }
    snrt_global_barrier();

#ifdef XDMA_CHECK_RESULT
    if (snrt_is_dm_core()) {
        // Check the result
        uint32_t *golden_result = (uint32_t *)tcdm_baseaddress;
        uint32_t *tcdm_result = (uint32_t *)data;

        for (int i = 0; i < data_size * sizeof(data[0]) / 4; i++) {
            if (tcdm_result[i] != golden_result[i]) {
                printf("The data copy is incorrect at byte %d! \n", i << 2);
            }
        }
        printf("Checking is done. All values are right\n");
    }
    snrt_global_barrier();
#endif

    return 0;
}
