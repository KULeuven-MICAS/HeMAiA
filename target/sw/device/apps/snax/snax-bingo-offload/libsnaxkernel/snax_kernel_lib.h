// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once

#include <snax_versacore_lib.h>


#include <snax_versacore_lib.h>

#define SNAX_LIB_NAME_MAX_LEN 64
// symtab data structure
typedef struct __attribute__((packed)) {
    char name[SNAX_LIB_NAME_MAX_LEN];  // function name
    uint32_t addr;                     // function addr
} snax_symbol_t;

#define SNAX_SYMTAB_END_FN_NAME "SYMTAB_END"
#define SNAX_SYMTAB_END_FN_ADDR (uint32_t)(0xBAADF00D)

#define SNAX_LIB_DEFINE __attribute__((used))
#define SNAX_SYMTAB_SECTION __attribute__((used, section(".snax_symtab")))
#define SNAX_EXPORT_FUNC(name) {#name, (uint32_t)name}
#define SNAX_SYMTAB_END {SNAX_SYMTAB_END_FN_NAME, SNAX_SYMTAB_END_FN_ADDR}

inline uint64_t make_u64(uint32_t hi, uint32_t lo) {
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

//////////////////////// Kernel functions ////////////////////////
// Template for user defined kernels
// Each kernel function takes a single void* argument

// SNAX_LIB_DEFINE void __snax_kernel_template(void *arg){
//     // Some description of the kernel
//     // The arguments are all uin32_t type
//     // Arg0: ...
//     // Arg1: ...
// }

SNAX_LIB_DEFINE void __snax_kernel_dummy(void *arg) {
    uint32_t arg0 = ((uint32_t *)arg)[0];
    if (snrt_is_dm_core()) {
        printf(
            "Chip(%x, %x): [Cluster %d] Core(%d) Running Dummy Kernel with "
            "arg[0] = %d\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            snrt_cluster_idx(), snrt_cluster_core_idx(), arg0);
    }
}



SNAX_LIB_DEFINE void __snax_kernel_csr(void *arg) {
    // Arg0: csr_addr
    // Arg1: csr_value

    // shared_ptrs[0] arg_ptr
    // shared_ptrs[1] csr_read_value
    if (snrt_is_dm_core()) {
        get_cls_shared_ptrs()[0] = snrt_l1_malloc(256);
        get_cls_shared_ptrs()[1] = snrt_l1_malloc(4);
        get_cls_shared_ptrs()[0][0] = ((uint32_t *)arg)[0];
        get_cls_shared_ptrs()[0][1] = ((uint32_t *)arg)[1];
        uint32_t csr_addr = get_cls_shared_ptrs()[0][0];
        uint32_t csr_write_value = get_cls_shared_ptrs()[0][1];
        uint32_t csr_read_value = get_cls_shared_ptrs()[1][0];
        printf(
            "Chip(%x, %x): Running CSR Kernel with arg[0](csr_addr) = %d, "
            "arg[1](csr_value) = %x\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(), csr_addr,
            csr_write_value);
        csrw_ss(csr_addr, csr_write_value);
        csr_read_value = csrr_ss(csr_addr);
        if (csr_read_value != csr_write_value) {
            printf("Chip(%x, %x): Error! R/W CSR values are not the same!\n",
                   get_current_chip_loc_x(), get_current_chip_loc_y());
        } else {
            printf("Chip(%x, %x): Pass!\n", get_current_chip_loc_x(),
                   get_current_chip_loc_y());
        }
    }
}

SNAX_LIB_DEFINE void __snax_kernel_check_results(void *arg) {
    // This function is used to check the results of the computation
    // We assume the data type is uint32_t
    // Arg0: golden_data_addr
    // Arg1: output_data_addr
    // Arg2: data_size in Byte
    if (snrt_is_dm_core()) {
        uint32_t golden_data_addr = ((uint32_t *)arg)[0];
        uint32_t output_data_addr = ((uint32_t *)arg)[1];
        uint32_t data_size = ((uint32_t *)arg)[2];
        uint32_t num_errors = 0;
        for (uint32_t i = 0; i < data_size / sizeof(uint32_t); i++) {
            if (((uint32_t *)golden_data_addr)[i] !=
                ((uint32_t *)output_data_addr)[i]) {
                num_errors++;
                if (num_errors < 10) {
                    printf("Error at index %d: golden = %d, output = %d\n", i,
                           ((uint32_t *)golden_data_addr)[i],
                           ((uint32_t *)output_data_addr)[i]);
                }
            }
        }
        if (num_errors == 0) {
            printf("Check Results: PASS! No errors found.\n");
        } else {
            printf("Check Results: FAIL! %d errors found.\n", num_errors);
        }
    }
}

SNAX_LIB_DEFINE void __snax_kernel_load_compute_store(void *arg) {
    // Arg0: uint32_t input_data_addr
    // Arg1: uint32_t input_data_size in Byte
    // Arg2: uint32_t output_data_addr
    // Arg3: uint32_t output_data_size in Byte

    // shared_ptrs[0] arg_ptr
    // shared_ptrs[1] l1_input_data_ptr
    // shared_ptrs[2] l1_output_data_ptr
    if (snrt_is_dm_core()) {
        // We allocate 256 bytes of heap memory for the args
        // Each arg is 4 bytes, so we can store 64 args

        get_cls_shared_ptrs()[0] = snrt_l1_malloc(256);
        get_cls_shared_ptrs()[0][0] = ((uint32_t *)arg)[0];
        get_cls_shared_ptrs()[0][1] = ((uint32_t *)arg)[1];
        get_cls_shared_ptrs()[0][2] = ((uint32_t *)arg)[2];
        get_cls_shared_ptrs()[0][3] = ((uint32_t *)arg)[3];
        // We then allocate the space for input and output data
        get_cls_shared_ptrs()[1] = snrt_l1_malloc(get_cls_shared_ptrs()[0][1]);
        get_cls_shared_ptrs()[2] = snrt_l1_malloc(get_cls_shared_ptrs()[0][3]);

        // printf("[Cluster %d] Core(%d) Allocated heap memory for args, input
        // and output data\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        // printf("[Cluster %d] Core(%d) Shared pointer at 0x%x\n",
        // snrt_cluster_idx(), snrt_cluster_core_idx(),get_cls_shared_ptrs());
        // printf("[Cluster %d] Core(%d) Args at 0x%x\n", snrt_cluster_idx(),
        // snrt_cluster_core_idx(),get_cls_shared_ptrs()[0]); printf("[Cluster
        // %d] Core(%d) Input data at 0x%x\n", snrt_cluster_idx(),
        // snrt_cluster_core_idx(),get_cls_shared_ptrs()[1]); printf("[Cluster
        // %d] Core(%d) Output data at 0x%x\n", snrt_cluster_idx(),
        // snrt_cluster_core_idx(),get_cls_shared_ptrs()[2]);
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        // load the input data
        snrt_dma_start_1d((void *)get_cls_shared_ptrs()[1],
                          (void *)get_cls_shared_ptrs()[0][0],
                          get_cls_shared_ptrs()[0][1]);
        snrt_dma_wait_all();
        printf("[Cluster %d] Core(%d) Load Input Data\n", snrt_cluster_idx(),
               snrt_cluster_core_idx());
    }
    snrt_cluster_hw_barrier();
    if (snrt_cluster_core_idx() == 0) {
        // We choose the first core to do the dummy +1 computation
        // printf("[Cluster %d] Core(%d) Inputs...\n", snrt_cluster_idx(),
        // snrt_cluster_core_idx()); for(uint32_t
        // i=0;i<get_cls_shared_ptrs()[0][1]/sizeof(uint32_t);i++){
        //     printf("[Cluster %d] Core(%d) Input[%d] = %d\n",
        //     snrt_cluster_idx(), snrt_cluster_core_idx(), i,
        //     get_cls_shared_ptrs()[1][i]);
        // }

        printf("[Cluster %d] Core(%d) Compute...\n", snrt_cluster_idx(),
               snrt_cluster_core_idx());
        for (uint32_t i = 0; i < get_cls_shared_ptrs()[0][1] / sizeof(uint32_t);
             i++) {
            get_cls_shared_ptrs()[1][i] += 1;
        }
        // printf("[Cluster %d] Core(%d) Validate...\n", snrt_cluster_idx(),
        // snrt_cluster_core_idx()); for(uint32_t
        // i=0;i<get_cls_shared_ptrs()[0][1]/sizeof(uint32_t);i++){
        //     printf("[Cluster %d] Core(%d) Output[%d] = %d\n",
        //     snrt_cluster_idx(), snrt_cluster_core_idx(), i,
        //     get_cls_shared_ptrs()[1][i]);
        // }
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        // We store the output data
        snrt_dma_start_1d((void *)get_cls_shared_ptrs()[0][2],
                          (void *)get_cls_shared_ptrs()[1],
                          get_cls_shared_ptrs()[0][3]);
        snrt_dma_wait_all();
        printf("[Cluster %d] Core(%d) Store Data\n", snrt_cluster_idx(),
               snrt_cluster_core_idx());
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        // Free the allocated memory
        snrt_l1_free(get_cls_shared_ptrs()[0]);
        snrt_l1_free(get_cls_shared_ptrs()[1]);
        snrt_l1_free(get_cls_shared_ptrs()[2]);
        printf("[Cluster %d] Core(%d) Free allocated heap memory\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
        snrt_memset(get_cls_shared_ptrs(), 0,
                    sizeof(uint32_t *) * NUM_CLS_SHARED_PTRS);
        printf("[Cluster %d] Core(%d) Free shared pointers\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
    }
}


SNAX_LIB_DEFINE void __snax_kernel_double_buffer_example(void *arg) {
    // Example kernel for double buffering

    // Input data is assumed to be array of uint32_t
    // Arg0: uint32_t input_data_addr
    // Arg1: uint32_t output_data_addr
    // Arg2: uint32_t data_size in Byte
    // Arg3: uint32_t num_tiles (>3)
    // This kernel demonstrates how to do the double buffering
    // ----------------------
    // Double buffering idea
    // ----------------------
    // One standard way to to load_compute_store is as follows:
    // * - Bubbles indicate idle time
    // Core_DMA  Load0 |     *    | Store0
    // Core_Comp   *   | Compute0 |    *
    // The double buffering idea is to overlap the loading of the next tile
    // If we have 6 tiles, the computation flow will be as follows:
    //              ____________Sync points_______________________
    //              |    |       |       |       |        |       |
    //              v    v       v       v       v        v       v
    // Core_DMA  L0 | L1 | S0,L2 | S1,L3 | S2,L4 | S3,L5  | S4  | S5 
    // Core_Comp *  | C0 | C1    | C2    | C3    | C4     | C5  | *  
    // Time slot s0   s1   s2      s3      s4      s5       s6    s7  
    // Memory layout:
    // Buffer 0: t0                t3                       t6
    // Buffer 1:      t1                   t4                 
    // Buffer 2:           t2                     t5
    // So at the same time at most three tiles are in the system
    // Memory mappings:
    // T     load tile | compute tile | store tile
    // s=0   Buf0      | ---          | ---
    // s=1   Buf1      | Buf0         | ---
    // s=2   Buf2      | Buf1         | Buf0         
    // s=3   Buf0      | Buf2         | Buf1
    // s=4   Buf1      | Buf0         | Buf2
    // s=5   Buf2      | Buf1         | Buf0
    // s=x   x%3       | (x+2)%3      | (x+1)%3
    // ....
    // s=S-1 ----      | (S+1)%3      | (S)  %3
    // s=S   ----      | ----         | (S+1)%3
    // ----------- Main code -------------------

    // Notice the following vars are allocated in the Thread-Local Storage (TLS)
    // which means each core has its own copy
    // Since those are just arguments and will not be modified during the kernel execution, it is safe to put here
    uint32_t input_data_addr = ((uint32_t *)arg)[0];
    uint32_t output_data_addr = ((uint32_t *)arg)[1];
    uint32_t input_data_size = ((uint32_t *)arg)[2];
    uint32_t num_tiles = ((uint32_t *)arg)[3]; // assume num_tiles >= 3
    uint32_t tile_size = input_data_size / num_tiles;  // in Byte
    // if(snrt_is_dm_core()){
    //     printf("input_data_addr=0x%x, output_data_addr=0x%x, input_data_size=%d, num_tiles=%d, tile_size=%d bytes\n",
    //            input_data_addr, output_data_addr, input_data_size, num_tiles, tile_size);
    // }
    snrt_cluster_hw_barrier();

    // Use one core to allocate the shared storage
    if(snrt_is_dm_core()){
        if (num_tiles < 3){
            printf("Error: num_tiles should be >= 3 for double buffering example kernel\n");
            return;
        }
        get_cls_shared_ptrs()[0] = snrt_l1_malloc(tile_size); // ptr for temporal tile storage 1
        get_cls_shared_ptrs()[1] = snrt_l1_malloc(tile_size); // ptr for temporal tile storage 2
        get_cls_shared_ptrs()[2] = snrt_l1_malloc(tile_size); // ptr for temporal tile storage 3
        // printf("Allocated shared buffers, buf1=0x%x, buf2=0x%x, buf3=0x%x\n",
        //        get_cls_shared_ptrs()[0],
        //        get_cls_shared_ptrs()[1],
        //        get_cls_shared_ptrs()[2]);
        
    }
    snrt_cluster_hw_barrier();
    // Here all the cores can access the shared tile buffers
    uint32_t *tile_buf_0 = get_cls_shared_ptrs()[0];
    uint32_t *tile_buf_1 = get_cls_shared_ptrs()[1];
    uint32_t *tile_buf_2 = get_cls_shared_ptrs()[2];

    for (uint32_t s = 0; s < num_tiles + 2; s++){
        // Here all the varialbes are derived from the tile_bufs, hence they are shared among all cores
        uint32_t *load_buf = s % 3 == 0 ? tile_buf_0 : (s % 3 == 1 ? tile_buf_1 : tile_buf_2);
        uint32_t *compute_buf = (s + 2) % 3 == 0 ? tile_buf_0 : ((s + 2) % 3 == 1 ? tile_buf_1 : tile_buf_2);
        uint32_t *store_buf = (s + 1) % 3 == 0 ? tile_buf_0 : ((s + 1) % 3 == 1 ? tile_buf_1 : tile_buf_2);
        if (s==0){
            // T0: Load tile 0
            if (snrt_is_dm_core()){
                snrt_dma_start_1d((void *)load_buf, (void *)input_data_addr, tile_size);
                snrt_dma_wait_all();
            }
        }

        if (s==1){
            // T1: Load tile 1 + Compute tile 0
            if (snrt_is_dm_core()){
                snrt_dma_start_1d((void *)load_buf, (void *)(input_data_addr + tile_size), tile_size);
                snrt_dma_wait_all();
            }
            if(snrt_cluster_core_idx() == 0){
                for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++) {
                    compute_buf[i] += 1;
                }
            }
        }

        if (s >= 2 && s <= num_tiles - 1){
            // T2 to TS-2: Load tile s + Compute tile s-1 + Store tile s-2
            if (snrt_is_dm_core()) {
                // Load tile s
                snrt_dma_start_1d((void *)load_buf, (void *)(input_data_addr + s * tile_size), tile_size);
                // Store tile s-2
                snrt_dma_start_1d((void *)(output_data_addr + (s - 2) * tile_size), (void *)store_buf, tile_size);
                snrt_dma_wait_all();
            }
            // Compute tile s-1
            if (snrt_cluster_core_idx() == 0) {
                for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++) {
                    compute_buf[i] += 1;
                }
            }
        }

        if (s == num_tiles){
            // TS-1: Compute tile s-1 + Store tile s-2
            if (snrt_cluster_core_idx() == 0) {
                for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++) {
                    compute_buf[i] += 1;  // Example computation
                }
            }
            if (snrt_is_dm_core()){
                snrt_dma_start_1d((void *)(output_data_addr + (num_tiles - 2) * tile_size), (void *)store_buf, tile_size);
                snrt_dma_wait_all();
            }
        }

        if (s == num_tiles + 1){
            // TS: Store tile s-2
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d((void *)(output_data_addr + (num_tiles - 1) * tile_size), (void *)store_buf, tile_size);
                snrt_dma_wait_all();
            }
        }

        snrt_cluster_hw_barrier();
    }
}






SNAX_LIB_DEFINE void __snax_kernel_xdma_1d_copy(void *arg) {
    // Copy 1d data from src to dst using xdma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte

    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t data_size;

    if (snrt_is_dm_core()) {
        src_addr = ((uint64_t)((uint32_t *)arg)[0] << 32) |
                   ((uint64_t)((uint32_t *)arg)[1]);
        dst_addr = ((uint64_t)((uint32_t *)arg)[2] << 32) |
                   ((uint64_t)((uint32_t *)arg)[3]);
        data_size = ((uint32_t *)arg)[4];

        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma writer extension 0\n");
        }
        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma writer extension 1\n");
        }
        if (xdma_disable_src_ext(0) != 0) {
            printf("Error in disabling xdma reader extension 0\n");
        }

        xdma_memcpy_1d((void *)src_addr, (void *)dst_addr, data_size);

        int task_id = xdma_start();
        xdma_remote_wait(task_id);
    }
}

//////////////////////// VERSACORE ////////////////////////
SNAX_LIB_DEFINE void __snax_kernel_versacore_load_compute_store(void *arg){
    // Compute D = A*B + C using versacore
    // A: int8
    // B: int8
    // C: int32
    // D: int32
    // Inputs
    // Arg0: uint32_t input_A_addr_hi
    // Arg1: uint32_t input_A_addr_lo
    // Arg2: uint32_t input_A_size in Byte
    // Arg3: uint32_t input_B_addr_hi
    // Arg4: uint32_t input_B_addr_lo
    // Arg5: uint32_t input_B_size in Byte
    // Arg6: uint32_t input_C_addr_hi
    // Arg7: uint32_t input_C_addr_lo
    // Arg8: uint32_t input_C_size in Byte
    // Outputs
    // Arg9: uint32_t output_addr_hi
    // Arg10: uint32_t output_addr_lo

    // Streamer Arguments
    // Arg11: uint32_t streamer_cfg_addr_hi
    // Arg12: uint32_t streamer_cfg_addr_lo
    // Arg13: uint32_t streamer_cfg_size in Byte

    // Versacore Arguments
    // Arg14: uint32_t versacore_cfg_addr_hi
    // Arg15: uint32_t versacore_cfg_addr_lo
    // Arg16: uint32_t versacore_cfg_size in Byte

    // Arg length
    // Arg17: uint32_t argument length in Byte

    // shared_ptrs[0] arg_ptr
    // shared_ptrs[1] l1_A_data_ptr
    // shared_ptrs[2] l1_B_data_ptr
    // shared_ptrs[3] l1_C_data_ptr
    // shared_ptrs[4] l1_D_data_ptr
    // shared_ptrs[5] l1_streamer_cfg_ptr
    // shared_ptrs[6] l1_versacore_cfg_ptr
    uint64_t l3_input_A_addr, l3_input_B_addr, l3_input_C_addr, l3_streamer_cfg_addr, l3_versacore_cfg_addr;
    uint32_t input_A_size, input_B_size, input_C_size, streamer_cfg_size, versacore_cfg_size;

    if (snrt_is_dm_core()) {
        // We have two dma in the system
        // xdma is optmized for large data transfer
        // idma is optmized for 1d data transfer (preferred for loading args)
        ///////////////////////////////////////
        // Allocate the L1 memory for args   //
        ///////////////////////////////////////
        // We allocate 128 bytes of heap memory for the args
        // Each arg is 4 bytes, so we can store 32 args
        get_cls_shared_ptrs()[0] = snrt_l1_malloc(128);
        // l1_arg_ptr = get_cls_shared_ptrs()[0];

        ////////////////////////////////////
        // Call the idma to load the args //
        ////////////////////////////////////
        // First we need to use the DMA to load all the arguments from L3 into L1
        // Here we assume the arguments are packed in a contiguous way in L3

        snrt_dma_start_1d((void *)get_cls_shared_ptrs()[0], (void *)arg, ((uint32_t *)arg)[17]);
        snrt_dma_wait_all();

        // printf("[Cluster %d] Core(%d) Load Kernel Args: Dst = %x, Src = %x\n", snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[0], arg);

        // printf("[Cluster %d] Core(%d) Kernel Args:\n", snrt_cluster_idx(),
        //        snrt_cluster_core_idx());
        // for (uint32_t i = 0; i < ((uint32_t *)l1_arg_ptr)[17] / sizeof(uint32_t);
        //      i++) {
        //     printf("[Cluster %d] Core(%d) Arg[%d] = %x\n", snrt_cluster_idx(),
        //            snrt_cluster_core_idx(), i, ((uint32_t *)l1_arg_ptr)[i]);
        // }
        /////////////////////////////////////////
        // Allocate the L1 memory for Inputs   //
        /////////////////////////////////////////
        // Allocate the l1 space specifed by the arg
        // Input data A
        get_cls_shared_ptrs()[1] = snrt_l1_malloc(get_cls_shared_ptrs()[0][2]);
        // l1_A_data_ptr = get_cls_shared_ptrs()[1];
        // Input data B
        get_cls_shared_ptrs()[2] = snrt_l1_malloc(get_cls_shared_ptrs()[0][5]);
        // l1_B_data_ptr = get_cls_shared_ptrs()[2];
        // Input data C
        get_cls_shared_ptrs()[3] = snrt_l1_malloc(get_cls_shared_ptrs()[0][8]);
        // l1_C_data_ptr = get_cls_shared_ptrs()[3];
        // output data D
        // Size of D is the same as C
        get_cls_shared_ptrs()[4] = snrt_l1_malloc(get_cls_shared_ptrs()[0][8]);
        // l1_D_data_ptr = get_cls_shared_ptrs()[4];
        ///////////////////////////////
        // Call the dma to load A    //
        ///////////////////////////////
        l3_input_A_addr = make_u64(get_cls_shared_ptrs()[0][0], get_cls_shared_ptrs()[0][1]);
        input_A_size = (get_cls_shared_ptrs()[0][2]);
        xdma_memcpy_1d((void *)l3_input_A_addr, (void *)get_cls_shared_ptrs()[1], input_A_size);
        int task_id = xdma_start();
        xdma_remote_wait(task_id);  
        printf("[Cluster %d] Core(%d) Load Input A: Dst = %x, Src = %x\n", snrt_cluster_idx(),
               snrt_cluster_core_idx(), get_cls_shared_ptrs()[1], (uint32_t)l3_input_A_addr);
        printf("[Cluster %d] Core(%d) Input A[0] = %d\n", snrt_cluster_idx(),
               snrt_cluster_core_idx(), get_cls_shared_ptrs()[1][0]);
        ///////////////////////////////
        // Call the dma to load B    //
        ///////////////////////////////
        l3_input_B_addr = make_u64(get_cls_shared_ptrs()[0][3], get_cls_shared_ptrs()[0][4]);
        input_B_size = (get_cls_shared_ptrs()[0][5]);
        xdma_memcpy_1d((void *)l3_input_B_addr, (void *)get_cls_shared_ptrs()[2], input_B_size);
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        // printf("[Cluster %d] Core(%d) Load Input B: Dst = %x, Src = %x\n", snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[2], (uint32_t)l3_input_B_addr);
        printf("[Cluster %d] Core(%d) Input B[0] = %d\n", snrt_cluster_idx(),
               snrt_cluster_core_idx(), get_cls_shared_ptrs()[2][0]);
        ///////////////////////////////
        // Call the dma to load C    //
        ///////////////////////////////
        l3_input_C_addr = make_u64(get_cls_shared_ptrs()[0][6], get_cls_shared_ptrs()[0][7]);
        input_C_size = (get_cls_shared_ptrs()[0][8]);
        xdma_memcpy_1d((void *)l3_input_C_addr, (void *)get_cls_shared_ptrs()[3], input_C_size);
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("[Cluster %d] Core(%d) Input C[0] = %d\n", snrt_cluster_idx(),
               snrt_cluster_core_idx(), get_cls_shared_ptrs()[3][0]);
        // printf("[Cluster %d] Core(%d) Load Input C: Dst = %x, Src = %x\n", snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[3], (uint32_t)l3_input_C_addr);
        //////////////////////////////////////////////////////////////
        // Allocated the L1 memory for Stream and Versacore CSRs    //
        //////////////////////////////////////////////////////////////
        get_cls_shared_ptrs()[5] = snrt_l1_malloc(((uint32_t *)get_cls_shared_ptrs()[0])[13]);
        // l1_streamer_cfg_ptr = get_cls_shared_ptrs()[5];
        get_cls_shared_ptrs()[6] = snrt_l1_malloc(((uint32_t *)get_cls_shared_ptrs()[0])[16]);
        // l1_versacore_cfg_ptr = get_cls_shared_ptrs()[6];

        //////////////////////////////////////////
        // Call the dma to load streamer cfg    //
        //////////////////////////////////////////
        l3_streamer_cfg_addr = make_u64(get_cls_shared_ptrs()[0][11], get_cls_shared_ptrs()[0][12]);
        streamer_cfg_size = (get_cls_shared_ptrs()[0][13]);
        snrt_dma_start_1d((void *)get_cls_shared_ptrs()[5], (void *)l3_streamer_cfg_addr, streamer_cfg_size);
        snrt_dma_wait_all();
        // printf("[Cluster %d] Core(%d) Load Streamer Cfg: Dst = %x, Src = %x\n", snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[5], l3_streamer_cfg_addr);
        //////////////////////////////////////////
        // Call the dma to load versacore cfg   //
        //////////////////////////////////////////
        l3_versacore_cfg_addr = make_u64(get_cls_shared_ptrs()[0][14], get_cls_shared_ptrs()[0][15]);
        versacore_cfg_size = (get_cls_shared_ptrs()[0][16]);
        snrt_dma_start_1d((void *)get_cls_shared_ptrs()[6], (void *)l3_versacore_cfg_addr, versacore_cfg_size);
        snrt_dma_wait_all();
        // printf("[Cluster %d] Core(%d) Load Versacore Cfg: Dst = %x, Src = %x\n", snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[6], l3_versacore_cfg_addr);
    }
    snrt_cluster_hw_barrier();
    // The core 0 will be responsible for configuring and starting the versacore
    if (snrt_cluster_core_idx() == 0) {
        // Configuration the Steamer and Versacore
        // NOTE: The streamer configuration layout in l1_streamer_cfg_ptr is
        // currently a flat array of 32-bit words. We treat each index as the
        // start of a (possibly single-element) array for stride/bound/channel
        // parameters. Adjust the grouping and lengths once the exact packing
        // format is defined. For now this fixes the type mismatches so CSR
        // writes receive proper scalars vs pointers.
        set_versacore_streamer_csr(
            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[1],                  // A_addr
            (uint32_t*)get_cls_shared_ptrs()[5][0],              // Aslstride[] base
            (uint32_t*)get_cls_shared_ptrs()[5][1],              // Atlbound[] base
            (uint32_t*)get_cls_shared_ptrs()[5][2],              // Atlstride[] base
            get_cls_shared_ptrs()[5][3],               // set_addr_remap_index_A
            get_cls_shared_ptrs()[5][4],               // transpose_A
            (uint32_t*)get_cls_shared_ptrs()[5][5],              // channel_en_A []

            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[2],                  // B_addr
            (uint32_t*)get_cls_shared_ptrs()[5][6],              // Bslstride[] base
            (uint32_t*)get_cls_shared_ptrs()[5][7],              // Btlbound[] base
            (uint32_t*)get_cls_shared_ptrs()[5][8],              // Btlstride[] base
            get_cls_shared_ptrs()[5][9],               // set_addr_remap_index_B
            get_cls_shared_ptrs()[5][10],              // transpose_B
            (uint32_t*)get_cls_shared_ptrs()[5][11],             // channel_en_B []

            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[3],                  // C_addr
            (uint32_t*)get_cls_shared_ptrs()[5][12],             // Cslstride[] base
            (uint32_t*)get_cls_shared_ptrs()[5][13],             // Ctlbound[] base
            (uint32_t*)get_cls_shared_ptrs()[5][14],             // Ctlstride[] base
            get_cls_shared_ptrs()[5][15],              // set_addr_remap_index_C
            (uint32_t*)get_cls_shared_ptrs()[5][16],             // channel_en_C []

            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[4],                  // D_addr
            (uint32_t*)get_cls_shared_ptrs()[5][17],             // D32slstride[] base
            (uint32_t*)get_cls_shared_ptrs()[5][18],             // D32tlbound[] base
            (uint32_t*)get_cls_shared_ptrs()[5][19],             // D32tlstride[] base
            get_cls_shared_ptrs()[5][20],              // set_addr_remap_index_D32
            (uint32_t*)get_cls_shared_ptrs()[5][21]              // channel_en_D32 []
        );
        // printf("[Cluster %d] Core(%d) Config Streamer\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx());
        // printf("[Cluster %d] Core(%d) A_addr = %x\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[1]);
        // printf("[Cluster %d] Core(%d) B_addr = %x\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[2]);
        // printf("[Cluster %d] Core(%d) C_addr = %x\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[3]);
        // printf("[Cluster %d] Core(%d) D_addr = %x\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[4]);
        // for (int i=0; i<22; i++){
        //     printf("[Cluster %d] Core(%d) Streamer_cfg[%d] = %x\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), i, get_cls_shared_ptrs()[5][i]);
        // }

        set_versacore_csr(
            get_cls_shared_ptrs()[6][0],  //tempLoop0
            get_cls_shared_ptrs()[6][1],  //tempLoop1
            get_cls_shared_ptrs()[6][2],  //tempLoop2
            gen_subtraction_config((int8_t)get_cls_shared_ptrs()[6][3], (int8_t)get_cls_shared_ptrs()[6][4]), //subtraction
            get_cls_shared_ptrs()[6][5],  //array_shape
            get_cls_shared_ptrs()[6][6]   //data_type
        );
        // printf("[Cluster %d] Core(%d) Config Versacore\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx());
        // for (int i=0; i<7; i++){
        //     printf("[Cluster %d] Core(%d) Versacore_cfg[%d] = %x\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), i, get_cls_shared_ptrs()[6][i]);
        // }
        ////////////////////////////////////
        // Start the Streamer and Versacore
        ////////////////////////////////////
        start_versacore_and_streamer();
        printf("[Cluster %d] Core(%d) Start Versacore\n",snrt_cluster_idx(),
               snrt_cluster_core_idx());
        // Wait for the Versacore to finish
        wait_versacore_and_streamer();
    }
    
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        // Send the output data back to L3
        uint64_t l3_output_addr = make_u64(get_cls_shared_ptrs()[0][9], get_cls_shared_ptrs()[0][10]);
        uint32_t output_size = get_cls_shared_ptrs()[0][8];
        // D is the temporary output buffer in L1
        printf("[Cluster %d] Core(%d) Quick check on D[0] = %d\n", snrt_cluster_idx(),
               snrt_cluster_core_idx(), get_cls_shared_ptrs()[4][0]);
        xdma_memcpy_1d((void *)get_cls_shared_ptrs()[4], (void *)l3_output_addr, output_size);
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("[Cluster %d] Core(%d) Store Output: Dst = %x, Src = %x\n", snrt_cluster_idx(),
               snrt_cluster_core_idx(), get_cls_shared_ptrs()[4], (uint32_t)l3_output_addr);
        // Free the allocated L1 memory
        snrt_l1_free(get_cls_shared_ptrs()[0]);
        snrt_l1_free(get_cls_shared_ptrs()[1]);
        snrt_l1_free(get_cls_shared_ptrs()[2]);
        snrt_l1_free(get_cls_shared_ptrs()[3]);
        snrt_l1_free(get_cls_shared_ptrs()[4]);
        snrt_l1_free(get_cls_shared_ptrs()[5]);
        snrt_l1_free(get_cls_shared_ptrs()[6]);
        // Reset the shared pointers
        snrt_memset(get_cls_shared_ptrs(), 0,
                    sizeof(uint32_t *) * NUM_CLS_SHARED_PTRS);
    }
}

//////////////////////// SYMBOL TABLE ////////////////////////
// Here we create the symbol table
SNAX_SYMTAB_SECTION const snax_symbol_t __snax_symtab[] = {
    SNAX_EXPORT_FUNC(__snax_kernel_dummy),
    SNAX_EXPORT_FUNC(__snax_kernel_check_results),
    SNAX_EXPORT_FUNC(__snax_kernel_csr),
    SNAX_EXPORT_FUNC(__snax_kernel_load_compute_store),
    SNAX_EXPORT_FUNC(__snax_kernel_double_buffer_example),
    SNAX_EXPORT_FUNC(__snax_kernel_xdma_1d_copy),
    SNAX_EXPORT_FUNC(__snax_kernel_versacore_load_compute_store),
    SNAX_SYMTAB_END
};
