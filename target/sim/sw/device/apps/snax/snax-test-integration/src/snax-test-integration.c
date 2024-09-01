// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// The purpose of this program is to test the integration of the system
// We initialize two clusters
// cluster 0 fetch data from the l3 -> tcdm0
// cluster 1 fetch data from tcdm0 -> tcdm1
// Then each cluster check data is correct or not
#include "data.h"

#include "snrt.h"
int8_t* tcdm0_start_addr;
int8_t* tcdm1_start_addr;
int main(){
    int err = 0;
    // First set the addr of cluster 0
    // tcdm0_start_addr = (int8_t*)0x10000000;
    // tcdm1_start_addr = (int8_t*)0x10100000;
    if(snrt_cluster_idx()==0){
        if(snrt_is_dm_core()){
            tcdm0_start_addr = (int8_t*)snrt_cluster_base_addrl();
            printf("The C0 TCDM ADDR is %p \n", tcdm0_start_addr);
        }
    }
    snrt_global_barrier();

    if(snrt_cluster_idx()==1){
        if(snrt_is_dm_core()){
            tcdm1_start_addr = (int8_t*)snrt_cluster_base_addrl();
            printf("The C1 TCDM ADDR is %p \n", tcdm1_start_addr);
        }
    }
    snrt_global_barrier();
    // C0 Load the data from l3 -> l1
    if(snrt_cluster_idx()==0){
        if (snrt_is_dm_core()){
            printf("[C0] Start to load data from %p\n", test_data);
            snrt_dma_start_1d(tcdm0_start_addr,test_data,length_data);
            snrt_dma_wait_all();
        }
    }
    // wait the C0 is done
    snrt_global_barrier();

    // Thenc C1 fetches data from C0
    if(snrt_cluster_idx()==1){
        if (snrt_is_dm_core()){
            printf("[C1] Load data from C0 TCDM %p\n", tcdm0_start_addr);
            snrt_dma_start_1d(tcdm1_start_addr,tcdm0_start_addr,length_data);
            snrt_dma_wait_all();
        }
        
    }
    // wait the C1 is done
    snrt_global_barrier();

    // Start to check
    if(snrt_cluster_idx()==0){
        
        if (snrt_cluster_core_idx()==0){
            printf("C0 Checking the results\n");
            for (int i = 0; i < length_data; i++){
                if(tcdm0_start_addr[i] != test_data[i]){
                    err ++;
                    printf("C0 data is incorrect!\n");
                    printf("tcdm0[%d]=%d, test_data[%d]=%d\n", i,
                        tcdm0_start_addr[i], i, test_data[i]);             
                }
            }
        }
    }
    snrt_global_barrier();
    if(snrt_cluster_idx()==1){
        
        if (snrt_cluster_core_idx()==0){
            printf("C1 Checking the results\n");
            for (int i = 0; i < length_data; i++){
                if(tcdm1_start_addr[i] != test_data[i]){
                    err ++;
                    printf("C1 data is incorrect!\n");
                    printf("tcdm0[%d]=%d, test_data[%d]=%d\n", i,
                        tcdm1_start_addr[i], i, test_data[i]);            
                }
            }
        }
    }

    snrt_global_barrier();
    if(snrt_cluster_idx()==0){
        if(snrt_is_dm_core()){
           printf("Checking all done! No error!\n");
        }
    }

    return 0;
}