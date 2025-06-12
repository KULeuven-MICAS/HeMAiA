// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// smoke test for DIMC cluster & gemmx cluster

// Version 1 function: DIMC loads data to TCDM, GEMMX reads that chunk of data

// Version 2 function: DIMC computes one head, GEMMX reads the result

// Version 3 function: DIMC computes one head, GEMMX reads the result and compute MM

#include "data.h"
#include "snax-gemmx-params.h"
#include "snax-gemmx-lib.h"

#include "snax-dimc-csr.h"
#include "snax-dimc-lib.h"
#include "snrt.h"
#include "data_mha.h"

uint64_t * tcdm_1_start_addr; // starting address for GEMMX cluster
uint64_t * tcdm_3_start_addr; // starting address for DIMC cluster

int main() {

    // set error value for checking
    int err = 0;
    
    if (snrt_cluster_idx() == 3){

        // allocate 32+32+32KB in TCDM for activation and weight pair
        uint64_t *tcdm_ptr_0, *tcdm_ptr_1, *tcdm_ptr_2, *tcdm_ptr_danger;

        tcdm_ptr_0 = (uint64_t *)snrt_cluster_base_addrl();
        tcdm_ptr_1     = tcdm_ptr_0 + Q_LENGTH;
        tcdm_ptr_2     = tcdm_ptr_1 + Q_LENGTH;
        // must not fully use the 32KB space
        tcdm_ptr_danger     = tcdm_ptr_2 + Q_LENGTH;

        tcdm_3_start_addr = tcdm_ptr_0;

        if (snrt_is_dm_core()) {
            // read weight WK and ativation K from data.h
            size_t vector_size = Q_LENGTH * sizeof(uint64_t);
            snrt_dma_start_1d(tcdm_ptr_0, K,  vector_size);
            // snrt_dma_start_1d(tcdm_ptr_1, Q,  vector_size);
            // snrt_dma_start_1d(tcdm_ptr_2, WK, vector_size);

            snrt_dma_wait_all();
        }
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 1){
        
        tcdm_1_start_addr = (uint64_t *)snrt_cluster_base_addrl();

        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(tcdm_1_start_addr, tcdm_3_start_addr, Q_LENGTH * sizeof(uint64_t));
            snrt_dma_wait_all();
        }
    }

    snrt_global_barrier();

    if (snrt_cluster_idx() == 1) {
        if (snrt_cluster_core_idx() == 0) {
            printf("GEMMX cluster starts to check copied data]\n");
            for ( int i = 0; i < Q_LENGTH; i++) {
                if (tcdm_1_start_addr[i] != K[i]) {
                    printf("Error: tcdm_1_start_addr[%d] = %lu, expected %lu\n", i, tcdm_1_start_addr[i], K[i]);
                    err += 1;
                }
            }
        }
    }

    snrt_global_barrier();
    
    return err;
}   
