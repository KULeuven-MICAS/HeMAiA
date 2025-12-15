// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

#pragma once

#include <snax_versacore_lib.h>
#include "versacore_hw_param.h"
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

#define EXTRACT_BIT(x, pos) (((x) >> (pos)) & 1)

// Debug Prints
#ifdef IDMA_DEBUG
#define IDMA_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define IDMA_DEBUG_PRINT(...)
#endif

#ifdef XDMA_DEBUG
#define XDMA_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define XDMA_DEBUG_PRINT(...)
#endif

#ifdef CHECK_RESULT_DEBUG
#define CHECK_RESULT_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define CHECK_RESULT_DEBUG_PRINT(...)
#endif
//////////////////////// Kernel functions ////////////////////////
// Template for user defined kernels
// Each kernel function takes a single void* argument

// SNAX_LIB_DEFINE void __snax_kernel_template(void *arg){
//     // Some description of the kernel
//     // The arguments are all uin32_t type
//     // Arg0: ...
//     // Arg1: ...
// }

SNAX_LIB_DEFINE void __snax_kernel_dummy(void* arg) {
    uint32_t arg0 = ((uint32_t*)arg)[0];
    if (snrt_is_dm_core()) {
        printf(
            "Chip(%x, %x): [Cluster %d] Core(%d) Running Dummy Kernel with "
            "arg[0] = %d\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            snrt_cluster_idx(), snrt_cluster_core_idx(), arg0);
    }
}

SNAX_LIB_DEFINE void __snax_kernel_csr(void* arg) {
    // Arg0: csr_addr
    // Arg1: csr_value

    // shared_ptrs[0] arg_ptr
    // shared_ptrs[1] csr_read_value
    if (snrt_is_dm_core()) {
        get_cls_shared_ptrs()[0] = (uint32_t*)snrt_l1_malloc(256);
        get_cls_shared_ptrs()[1] = (uint32_t*)snrt_l1_malloc(4);
        get_cls_shared_ptrs()[0][0] = ((uint32_t*)arg)[0];
        get_cls_shared_ptrs()[0][1] = ((uint32_t*)arg)[1];
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

SNAX_LIB_DEFINE void __snax_kernel_check_results(void* arg) {
    // This function is used to check the results of the computation
    // We assume the data type is uint32_t
    // Arg0: golden_data_addr
    // Arg1: output_data_addr
    // Arg2: data_size in Byte
    if (snrt_is_dm_core()) {
        uint32_t golden_data_addr = ((uint32_t*)arg)[0];
        uint32_t output_data_addr = ((uint32_t*)arg)[1];
        uint32_t data_size = ((uint32_t*)arg)[2];
        uint32_t num_errors = 0;
        CHECK_RESULT_DEBUG_PRINT("Checking results...\n");
        CHECK_RESULT_DEBUG_PRINT("Golden data address: %x\n", golden_data_addr);
        CHECK_RESULT_DEBUG_PRINT("Output data address: %x\n", output_data_addr);
        for (uint32_t i = 0; i < data_size / sizeof(uint32_t); i++) {
            if (((uint32_t*)golden_data_addr)[i] !=
                ((uint32_t*)output_data_addr)[i]) {
                num_errors++;
                if (num_errors < 10) {
                    printf("Error at index %d: golden = %x, output = %x\n", i,
                           ((uint32_t*)golden_data_addr)[i],
                           ((uint32_t*)output_data_addr)[i]);
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

SNAX_LIB_DEFINE void __snax_kernel_check_results_full(void* arg) {
    // This function is used to check the results of the computation
    // Different from the previous check_results kernel, this function will use the DMA core to load data from L3 to L1 first
    // Arg0: golden_data_addr_hi
    // Arg1: golden_data_addr_lo
    // Arg2: output_data_addr_hi
    // Arg3: output_data_addr_lo
    // Arg4: data_size in Byte
    if (snrt_is_dm_core()) {
        uint32_t golden_data_addr_hi = ((uint32_t*)arg)[0];
        uint32_t golden_data_addr_lo = ((uint32_t*)arg)[1];
        uint32_t output_data_addr_hi = ((uint32_t*)arg)[2];
        uint32_t output_data_addr_lo = ((uint32_t*)arg)[3];
        uint32_t data_size = ((uint32_t*)arg)[4];

        uint32_t golden_data_l1 =
            snrt_l1_malloc(data_size);  // allocate L1 memory for golden data
        uint32_t output_data_l1 =
            snrt_l1_malloc(data_size);  // allocate L1 memory for output data

        snrt_dma_start_1d((void*)golden_data_l1,
                          (void*)make_u64(golden_data_addr_hi,golden_data_addr_lo),
                          data_size);
        snrt_dma_start_1d((void*)output_data_l1,
                          (void*)make_u64(output_data_addr_hi,output_data_addr_lo),
                          data_size);
        snrt_dma_wait_all();

        uint32_t num_errors = 0;
        for (uint32_t i = 0; i < data_size / sizeof(uint32_t); i++) {
            if (((uint32_t*)golden_data_l1)[i] != ((uint32_t*)output_data_l1)[i]) {
                num_errors++;
                if (num_errors < 10) {
                    printf("Error at index %d: golden = %d, output = %d\n", i,
                           ((uint32_t*)golden_data_l1)[i], ((uint32_t*)output_data_l1)[i]);
                }
            }
        }
        if (num_errors == 0) {
            printf("Check Results: PASS! No errors found.\n");
        } else {
            printf("Check Results: FAIL! %d errors found.\n", num_errors);
        }

        snrt_l1_free(golden_data_l1);
        snrt_l1_free(output_data_l1);
    }
}

SNAX_LIB_DEFINE void __snax_kernel_load_compute_store(void* arg) {
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

        get_cls_shared_ptrs()[0] = (uint32_t*)snrt_l1_malloc(256);
        get_cls_shared_ptrs()[0][0] = ((uint32_t*)arg)[0];
        get_cls_shared_ptrs()[0][1] = ((uint32_t*)arg)[1];
        get_cls_shared_ptrs()[0][2] = ((uint32_t*)arg)[2];
        get_cls_shared_ptrs()[0][3] = ((uint32_t*)arg)[3];
        // We then allocate the space for input and output data
        get_cls_shared_ptrs()[1] = (uint32_t*)snrt_l1_malloc(get_cls_shared_ptrs()[0][1]);
        get_cls_shared_ptrs()[2] = (uint32_t*)snrt_l1_malloc(get_cls_shared_ptrs()[0][3]);

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
        snrt_dma_start_1d((void*)get_cls_shared_ptrs()[1],
                          (void*)get_cls_shared_ptrs()[0][0],
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
        snrt_dma_start_1d((void*)get_cls_shared_ptrs()[0][2],
                          (void*)get_cls_shared_ptrs()[1],
                          get_cls_shared_ptrs()[0][3]);
        snrt_dma_wait_all();
        printf("[Cluster %d] Core(%d) Store Data\n", snrt_cluster_idx(),
               snrt_cluster_core_idx());
    }
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        // Free the allocated memory
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[0]);
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[1]);
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[2]);
        printf("[Cluster %d] Core(%d) Free allocated heap memory\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
        snrt_memset(get_cls_shared_ptrs(), 0,
                    sizeof(uint32_t*) * NUM_CLS_SHARED_PTRS);
        printf("[Cluster %d] Core(%d) Free shared pointers\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
    }
}

SNAX_LIB_DEFINE void __snax_kernel_double_buffer_example(void* arg) {
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
    // Since those are just arguments and will not be modified during the kernel
    // execution, it is safe to put here
    uint32_t input_data_addr = ((uint32_t*)arg)[0];
    uint32_t output_data_addr = ((uint32_t*)arg)[1];
    uint32_t input_data_size = ((uint32_t*)arg)[2];
    uint32_t num_tiles = ((uint32_t*)arg)[3];          // assume num_tiles >= 3
    uint32_t tile_size = input_data_size / num_tiles;  // in Byte
    // if(snrt_is_dm_core()){
    //     printf("input_data_addr=0x%x, output_data_addr=0x%x,
    //     input_data_size=%d, num_tiles=%d, tile_size=%d bytes\n",
    //            input_data_addr, output_data_addr, input_data_size, num_tiles,
    //            tile_size);
    // }
    snrt_cluster_hw_barrier();

    // Use one core to allocate the shared storage
    if (snrt_is_dm_core()) {
        if (num_tiles < 3) {
            printf(
                "Error: num_tiles should be >= 3 for double buffering example "
                "kernel\n");
            return;
        }
        get_cls_shared_ptrs()[0] =
            (uint32_t*)snrt_l1_malloc(tile_size);  // ptr for temporal tile storage 1
        get_cls_shared_ptrs()[1] =
            (uint32_t*)snrt_l1_malloc(tile_size);  // ptr for temporal tile storage 2
        get_cls_shared_ptrs()[2] =
            (uint32_t*)snrt_l1_malloc(tile_size);  // ptr for temporal tile storage 3
        // printf("Allocated shared buffers, buf1=0x%x, buf2=0x%x, buf3=0x%x\n",
        //        get_cls_shared_ptrs()[0],
        //        get_cls_shared_ptrs()[1],
        //        get_cls_shared_ptrs()[2]);
    }
    snrt_cluster_hw_barrier();
    // Here all the cores can access the shared tile buffers
    uint32_t* tile_buf_0 = get_cls_shared_ptrs()[0];
    uint32_t* tile_buf_1 = get_cls_shared_ptrs()[1];
    uint32_t* tile_buf_2 = get_cls_shared_ptrs()[2];

    for (uint32_t s = 0; s < num_tiles + 2; s++) {
        // Here all the varialbes are derived from the tile_bufs, hence they are
        // shared among all cores
        uint32_t* load_buf =
            s % 3 == 0 ? tile_buf_0 : (s % 3 == 1 ? tile_buf_1 : tile_buf_2);
        uint32_t* compute_buf =
            (s + 2) % 3 == 0 ? tile_buf_0
                             : ((s + 2) % 3 == 1 ? tile_buf_1 : tile_buf_2);
        uint32_t* store_buf =
            (s + 1) % 3 == 0 ? tile_buf_0
                             : ((s + 1) % 3 == 1 ? tile_buf_1 : tile_buf_2);
        if (s == 0) {
            // T0: Load tile 0
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d((void*)load_buf, (void*)input_data_addr,
                                  tile_size);
                snrt_dma_wait_all();
            }
        }

        if (s == 1) {
            // T1: Load tile 1 + Compute tile 0
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d((void*)load_buf,
                                  (void*)(input_data_addr + tile_size),
                                  tile_size);
                snrt_dma_wait_all();
            }
            if (snrt_cluster_core_idx() == 0) {
                for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++) {
                    compute_buf[i] += 1;
                }
            }
        }

        if (s >= 2 && s <= num_tiles - 1) {
            // T2 to TS-2: Load tile s + Compute tile s-1 + Store tile s-2
            if (snrt_is_dm_core()) {
                // Load tile s
                snrt_dma_start_1d((void*)load_buf,
                                  (void*)(input_data_addr + s * tile_size),
                                  tile_size);
                // Store tile s-2
                snrt_dma_start_1d(
                    (void*)(output_data_addr + (s - 2) * tile_size),
                    (void*)store_buf, tile_size);
                snrt_dma_wait_all();
            }
            // Compute tile s-1
            if (snrt_cluster_core_idx() == 0) {
                for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++) {
                    compute_buf[i] += 1;
                }
            }
        }

        if (s == num_tiles) {
            // TS-1: Compute tile s-1 + Store tile s-2
            if (snrt_cluster_core_idx() == 0) {
                for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++) {
                    compute_buf[i] += 1;  // Example computation
                }
            }
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(
                    (void*)(output_data_addr + (num_tiles - 2) * tile_size),
                    (void*)store_buf, tile_size);
                snrt_dma_wait_all();
            }
        }

        if (s == num_tiles + 1) {
            // TS: Store tile s-2
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(
                    (void*)(output_data_addr + (num_tiles - 1) * tile_size),
                    (void*)store_buf, tile_size);
                snrt_dma_wait_all();
            }
        }

        snrt_cluster_hw_barrier();
    }
}

SNAX_LIB_DEFINE void __snax_kernel_xdma_1d_copy(void* arg) {
    // Copy 1d data from src to dst using xdma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte

    if (snrt_is_dm_core()) {
        uint64_t src_addr = make_u64(((uint32_t*)arg)[0], ((uint32_t*)arg)[1]);
        uint64_t dst_addr = make_u64(((uint32_t*)arg)[2], ((uint32_t*)arg)[3]); 
        uint32_t data_size = ((uint32_t*)arg)[4];

        xdma_memcpy_1d_full_addr(src_addr, dst_addr, data_size);
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        XDMA_DEBUG_PRINT("XDMA copy completed\n");
        XDMA_DEBUG_PRINT("SRC ADDR = %lx\n", src_addr);
        XDMA_DEBUG_PRINT("DST ADDR = %lx\n", dst_addr);
    }

}

SNAX_LIB_DEFINE void __snax_kernel_idma_1d_copy(void* arg) {
    // Copy 1d data from src to dst using idma
    // Arg0: uint32_t src_addr_hi
    // Arg1: uint32_t src_addr_lo
    // Arg2: uint32_t dst_addr_hi
    // Arg3: uint32_t dst_addr_lo
    // Arg4: uint32_t size in Byte


    if (snrt_is_dm_core()) {
        uint64_t src_addr = make_u64(((uint32_t*)arg)[0], ((uint32_t*)arg)[1]);
        uint64_t dst_addr = make_u64(((uint32_t*)arg)[2], ((uint32_t*)arg)[3]); 
        uint32_t data_size = ((uint32_t*)arg)[4];

        snrt_dma_start_1d_wideptr(dst_addr, src_addr, data_size);
        snrt_dma_wait_all();
        IDMA_DEBUG_PRINT("IDMA copy completed\r\n");
        IDMA_DEBUG_PRINT("SRC ADDR = %lx\r\n", src_addr);
        IDMA_DEBUG_PRINT("DST ADDR = %lx\r\n", dst_addr);
    }
}

// //////////////////////// VERSACORE ////////////////////////
// SNAX_LIB_DEFINE void __snax_kernel_versacore_load_compute_store_w_streamer_args(
//     void* arg) {
//     // Compute D = A*B + C using versacore
//     // A: int8
//     // B: int8
//     // C: int32
//     // D: int32
//     // Inputs
//     // Arg0: uint32_t input_A_addr_hi
//     // Arg1: uint32_t input_A_addr_lo
//     // Arg2: uint32_t input_A_size in Byte
//     // Arg3: uint32_t input_B_addr_hi
//     // Arg4: uint32_t input_B_addr_lo
//     // Arg5: uint32_t input_B_size in Byte
//     // Arg6: uint32_t input_C_addr_hi
//     // Arg7: uint32_t input_C_addr_lo
//     // Arg8: uint32_t input_C_size in Byte
//     // Outputs
//     // Arg9: uint32_t output_addr_hi
//     // Arg10: uint32_t output_addr_lo

//     // Streamer Arguments
//     // Arg11: uint32_t streamer_cfg_addr_hi
//     // Arg12: uint32_t streamer_cfg_addr_lo
//     // Arg13: uint32_t streamer_cfg_size in Byte

//     // Versacore Arguments
//     // Arg14: uint32_t versacore_cfg_addr_hi
//     // Arg15: uint32_t versacore_cfg_addr_lo
//     // Arg16: uint32_t versacore_cfg_size in Byte

//     // Arg length
//     // Arg17: uint32_t argument length in Byte

//     // shared_ptrs[0] arg_ptr
//     // shared_ptrs[1] l1_A_data_ptr
//     // shared_ptrs[2] l1_B_data_ptr
//     // shared_ptrs[3] l1_C_data_ptr
//     // shared_ptrs[4] l1_D_data_ptr
//     // shared_ptrs[5] l1_streamer_cfg_ptr
//     // shared_ptrs[6] l1_versacore_cfg_ptr
//     uint64_t l3_input_A_addr, l3_input_B_addr, l3_input_C_addr,
//         l3_streamer_cfg_addr, l3_versacore_cfg_addr;
//     uint32_t input_A_size, input_B_size, input_C_size, streamer_cfg_size,
//         versacore_cfg_size;

//     if (snrt_is_dm_core()) {
//         // We have two dma in the system
//         // xdma is optmized for large data transfer
//         // idma is optmized for 1d data transfer (preferred for loading args)
//         ///////////////////////////////////////
//         // Allocate the L1 memory for args   //
//         ///////////////////////////////////////
//         // We allocate 128 bytes of heap memory for the args
//         // Each arg is 4 bytes, so we can store 32 args
//         get_cls_shared_ptrs()[0] = snrt_l1_malloc(128);
//         // l1_arg_ptr = get_cls_shared_ptrs()[0];

//         ////////////////////////////////////
//         // Call the idma to load the args //
//         ////////////////////////////////////
//         // First we need to use the DMA to load all the arguments from L3 into
//         // L1 Here we assume the arguments are packed in a contiguous way in L3

//         snrt_dma_start_1d((void*)get_cls_shared_ptrs()[0], (void*)arg,
//                           ((uint32_t*)arg)[17]);
//         snrt_dma_wait_all();

//         // printf("[Cluster %d] Core(%d) Load Kernel Args: Dst = %x, Src =
//         // %x\n", snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[0], arg);

//         // printf("[Cluster %d] Core(%d) Kernel Args:\n", snrt_cluster_idx(),
//         //        snrt_cluster_core_idx());
//         // for (uint32_t i = 0; i < ((uint32_t *)l1_arg_ptr)[17] /
//         // sizeof(uint32_t);
//         //      i++) {
//         //     printf("[Cluster %d] Core(%d) Arg[%d] = %x\n",
//         //     snrt_cluster_idx(),
//         //            snrt_cluster_core_idx(), i, ((uint32_t *)l1_arg_ptr)[i]);
//         // }
//         /////////////////////////////////////////
//         // Allocate the L1 memory for Inputs   //
//         /////////////////////////////////////////
//         // Allocate the l1 space specifed by the arg
//         // Input data A
//         get_cls_shared_ptrs()[1] = snrt_l1_malloc(get_cls_shared_ptrs()[0][2]);
//         // l1_A_data_ptr = get_cls_shared_ptrs()[1];
//         // Input data B
//         get_cls_shared_ptrs()[2] = snrt_l1_malloc(get_cls_shared_ptrs()[0][5]);
//         // l1_B_data_ptr = get_cls_shared_ptrs()[2];
//         // Input data C
//         get_cls_shared_ptrs()[3] = snrt_l1_malloc(get_cls_shared_ptrs()[0][8]);
//         // l1_C_data_ptr = get_cls_shared_ptrs()[3];
//         // output data D
//         // Size of D is the same as C
//         get_cls_shared_ptrs()[4] = snrt_l1_malloc(get_cls_shared_ptrs()[0][8]);
//         // l1_D_data_ptr = get_cls_shared_ptrs()[4];
//         ///////////////////////////////
//         // Call the dma to load A    //
//         ///////////////////////////////
//         l3_input_A_addr =
//             make_u64(get_cls_shared_ptrs()[0][0], get_cls_shared_ptrs()[0][1]);
//         input_A_size = (get_cls_shared_ptrs()[0][2]);
//         xdma_memcpy_1d((void*)l3_input_A_addr, (void*)get_cls_shared_ptrs()[1],
//                        input_A_size);
//         int task_id = xdma_start();
//         xdma_remote_wait(task_id);
//         printf("[Cluster %d] Core(%d) Load Input A: Dst = %x, Src = %x\n",
//                snrt_cluster_idx(), snrt_cluster_core_idx(),
//                get_cls_shared_ptrs()[1], (uint32_t)l3_input_A_addr);
//         printf("[Cluster %d] Core(%d) Input A[0] = %d\n", snrt_cluster_idx(),
//                snrt_cluster_core_idx(), get_cls_shared_ptrs()[1][0]);
//         ///////////////////////////////
//         // Call the dma to load B    //
//         ///////////////////////////////
//         l3_input_B_addr =
//             make_u64(get_cls_shared_ptrs()[0][3], get_cls_shared_ptrs()[0][4]);
//         input_B_size = (get_cls_shared_ptrs()[0][5]);
//         xdma_memcpy_1d((void*)l3_input_B_addr, (void*)get_cls_shared_ptrs()[2],
//                        input_B_size);
//         task_id = xdma_start();
//         xdma_remote_wait(task_id);
//         // printf("[Cluster %d] Core(%d) Load Input B: Dst = %x, Src = %x\n",
//         // snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[2],
//         //        (uint32_t)l3_input_B_addr);
//         printf("[Cluster %d] Core(%d) Input B[0] = %d\n", snrt_cluster_idx(),
//                snrt_cluster_core_idx(), get_cls_shared_ptrs()[2][0]);
//         ///////////////////////////////
//         // Call the dma to load C    //
//         ///////////////////////////////
//         l3_input_C_addr =
//             make_u64(get_cls_shared_ptrs()[0][6], get_cls_shared_ptrs()[0][7]);
//         input_C_size = (get_cls_shared_ptrs()[0][8]);
//         xdma_memcpy_1d((void*)l3_input_C_addr, (void*)get_cls_shared_ptrs()[3],
//                        input_C_size);
//         task_id = xdma_start();
//         xdma_remote_wait(task_id);
//         printf("[Cluster %d] Core(%d) Input C[0] = %d\n", snrt_cluster_idx(),
//                snrt_cluster_core_idx(), get_cls_shared_ptrs()[3][0]);
//         // printf("[Cluster %d] Core(%d) Load Input C: Dst = %x, Src = %x\n",
//         // snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[3],
//         //        (uint32_t)l3_input_C_addr);
//         //////////////////////////////////////////////////////////////
//         // Allocated the L1 memory for Stream and Versacore CSRs    //
//         //////////////////////////////////////////////////////////////
//         get_cls_shared_ptrs()[5] =
//             snrt_l1_malloc(((uint32_t*)get_cls_shared_ptrs()[0])[13]);
//         // l1_streamer_cfg_ptr = get_cls_shared_ptrs()[5];
//         get_cls_shared_ptrs()[6] =
//             snrt_l1_malloc(((uint32_t*)get_cls_shared_ptrs()[0])[16]);
//         // l1_versacore_cfg_ptr = get_cls_shared_ptrs()[6];

//         //////////////////////////////////////////
//         // Call the dma to load streamer cfg    //
//         //////////////////////////////////////////
//         l3_streamer_cfg_addr = make_u64(get_cls_shared_ptrs()[0][11],
//                                         get_cls_shared_ptrs()[0][12]);
//         streamer_cfg_size = (get_cls_shared_ptrs()[0][13]);
//         snrt_dma_start_1d((void*)get_cls_shared_ptrs()[5],
//                           (void*)l3_streamer_cfg_addr, streamer_cfg_size);
//         snrt_dma_wait_all();
//         // printf("[Cluster %d] Core(%d) Load Streamer Cfg: Dst = %x, Src =
//         // %x\n", snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[5],
//         //        l3_streamer_cfg_addr);
//         //////////////////////////////////////////
//         // Call the dma to load versacore cfg   //
//         //////////////////////////////////////////
//         l3_versacore_cfg_addr = make_u64(get_cls_shared_ptrs()[0][14],
//                                          get_cls_shared_ptrs()[0][15]);
//         versacore_cfg_size = (get_cls_shared_ptrs()[0][16]);
//         snrt_dma_start_1d((void*)get_cls_shared_ptrs()[6],
//                           (void*)l3_versacore_cfg_addr, versacore_cfg_size);
//         snrt_dma_wait_all();
//         // printf("[Cluster %d] Core(%d) Load Versacore Cfg: Dst = %x, Src =
//         // %x\n", snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[6],
//         //        l3_versacore_cfg_addr);
//     }
//     snrt_cluster_hw_barrier();
//     // The core 0 will be responsible for configuring and starting the versacore
//     if (snrt_cluster_core_idx() == 0) {
//         // Configuration the Steamer and Versacore
//         // NOTE: The streamer configuration layout in l1_streamer_cfg_ptr is
//         // currently a flat array of 32-bit words. We treat each index as the
//         // start of a (possibly single-element) array for stride/bound/channel
//         // parameters. Adjust the grouping and lengths once the exact packing
//         // format is defined. For now this fixes the type mismatches so CSR
//         // writes receive proper scalars vs pointers.
//         set_versacore_streamer_csr(
//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[1],  // A_addr
//             (uint32_t*)get_cls_shared_ptrs()[5][0],         // Aslstride
//             (uint32_t*)get_cls_shared_ptrs()[5][1],         // Atlbound[] base
//             (uint32_t*)get_cls_shared_ptrs()[5][2],         // Atlstride[] base
//             get_cls_shared_ptrs()[5][3],             // set_addr_remap_index_A
//             get_cls_shared_ptrs()[5][4],             // transpose_A
//             (uint32_t*)get_cls_shared_ptrs()[5][5],  // channel_en_A []

//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[2],  // B_addr
//             (uint32_t*)get_cls_shared_ptrs()[5][6],         // Bslstride[] base
//             (uint32_t*)get_cls_shared_ptrs()[5][7],         // Btlbound[] base
//             (uint32_t*)get_cls_shared_ptrs()[5][8],         // Btlstride[] base
//             get_cls_shared_ptrs()[5][9],              // set_addr_remap_index_B
//             get_cls_shared_ptrs()[5][10],             // transpose_B
//             (uint32_t*)get_cls_shared_ptrs()[5][11],  // channel_en_B []

//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[3],  // C_addr
//             (uint32_t*)get_cls_shared_ptrs()[5][12],        // Cslstride[] base
//             (uint32_t*)get_cls_shared_ptrs()[5][13],        // Ctlbound[] base
//             (uint32_t*)get_cls_shared_ptrs()[5][14],        // Ctlstride[] base
//             get_cls_shared_ptrs()[5][15],             // set_addr_remap_index_C
//             (uint32_t*)get_cls_shared_ptrs()[5][16],  // channel_en_C []

//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[4],  // D_addr
//             (uint32_t*)get_cls_shared_ptrs()[5][17],  // D32slstride[] base
//             (uint32_t*)get_cls_shared_ptrs()[5][18],  // D32tlbound[] base
//             (uint32_t*)get_cls_shared_ptrs()[5][19],  // D32tlstride[] base
//             get_cls_shared_ptrs()[5][20],            // set_addr_remap_index_D32
//             (uint32_t*)get_cls_shared_ptrs()[5][21],  // channel_en_D32 []

//             0, 0, 0, 0, 0, 0, 0
//         );
//         // printf("[Cluster %d] Core(%d) Config Streamer\n",snrt_cluster_idx(),
//         //        snrt_cluster_core_idx());
//         // printf("[Cluster %d] Core(%d) A_addr = %x\n",snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[1]);
//         // printf("[Cluster %d] Core(%d) B_addr = %x\n",snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[2]);
//         // printf("[Cluster %d] Core(%d) C_addr = %x\n",snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[3]);
//         // printf("[Cluster %d] Core(%d) D_addr = %x\n",snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[4]);
//         // for (int i=0; i<22; i++){
//         //     printf("[Cluster %d] Core(%d) Streamer_cfg[%d] =
//         //     %x\n",snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), i, get_cls_shared_ptrs()[5][i]);
//         // }

//         set_versacore_csr(
//             get_cls_shared_ptrs()[6][0],  // tempLoop0
//             get_cls_shared_ptrs()[6][1],  // tempLoop1
//             get_cls_shared_ptrs()[6][2],  // tempLoop2
//             gen_subtraction_config(
//                 (int8_t)get_cls_shared_ptrs()[6][3],
//                 (int8_t)get_cls_shared_ptrs()[6][4]),  // subtraction
//             get_cls_shared_ptrs()[6][5],               // array_shape
//             get_cls_shared_ptrs()[6][6]                // data_type
//         );
//         // printf("[Cluster %d] Core(%d) Config Versacore\n",snrt_cluster_idx(),
//         //        snrt_cluster_core_idx());
//         // for (int i=0; i<7; i++){
//         //     printf("[Cluster %d] Core(%d) Versacore_cfg[%d] =
//         //     %x\n",snrt_cluster_idx(),
//         //        snrt_cluster_core_idx(), i, get_cls_shared_ptrs()[6][i]);
//         // }
//         ////////////////////////////////////
//         // Start the Streamer and Versacore
//         ////////////////////////////////////
//         start_versacore_and_streamer();
//         printf("[Cluster %d] Core(%d) Start Versacore\n", snrt_cluster_idx(),
//                snrt_cluster_core_idx());
//         // Wait for the Versacore to finish
//         wait_versacore_and_streamer();
//     }

//     snrt_cluster_hw_barrier();
//     if (snrt_is_dm_core()) {
//         // Send the output data back to L3
//         uint64_t l3_output_addr =
//             make_u64(get_cls_shared_ptrs()[0][9], get_cls_shared_ptrs()[0][10]);
//         uint32_t output_size = get_cls_shared_ptrs()[0][8];
//         // D is the temporary output buffer in L1
//         printf("[Cluster %d] Core(%d) Quick check on D[0] = %d\n",
//                snrt_cluster_idx(), snrt_cluster_core_idx(),
//                get_cls_shared_ptrs()[4][0]);
//         xdma_memcpy_1d((void*)get_cls_shared_ptrs()[4], (void*)l3_output_addr,
//                        output_size);
//         int task_id = xdma_start();
//         xdma_remote_wait(task_id);
//         printf("[Cluster %d] Core(%d) Store Output: Dst = %x, Src = %x\n",
//                snrt_cluster_idx(), snrt_cluster_core_idx(),
//                get_cls_shared_ptrs()[4], (uint32_t)l3_output_addr);
//         // Free the allocated L1 memory
//         snrt_l1_free(get_cls_shared_ptrs()[0]);
//         snrt_l1_free(get_cls_shared_ptrs()[1]);
//         snrt_l1_free(get_cls_shared_ptrs()[2]);
//         snrt_l1_free(get_cls_shared_ptrs()[3]);
//         snrt_l1_free(get_cls_shared_ptrs()[4]);
//         snrt_l1_free(get_cls_shared_ptrs()[5]);
//         snrt_l1_free(get_cls_shared_ptrs()[6]);
//         // Reset the shared pointers
//         snrt_memset(get_cls_shared_ptrs(), 0,
//                     sizeof(uint32_t*) * NUM_CLS_SHARED_PTRS);
//     }
// }

// SNAX_LIB_DEFINE void __snax_kernel_gemm_intra_chiplet(void* arg) {
//     // move the args to local variables
//     if (snrt_is_dm_core()) {
//         // allocate L1 memory for global variables across the cluster
//         ///////////////////////////////////////
//         // Allocate the L1 memory //
//         ///////////////////////////////////////
//         get_cls_shared_ptrs()[0] = snrt_l1_malloc(128);
//         if (!get_cls_shared_ptrs()[0]) {
//             VERSACORE_DEBUG_PRINT("ERROR: arg alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated arg at %p\r\n",
//                                   (void*)get_cls_shared_ptrs()[0]);
//         }

//         ////////////////////////////////////
//         // Call the idma to load the args //
//         ////////////////////////////////////
//         // First we need to use the DMA to load all the arguments from L3 into
//         // L1 Here we assume the arguments are packed in a contiguous way in L3
//         VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Start!\r\n");

//         snrt_dma_start_1d(
//             (void*)get_cls_shared_ptrs()[0], (void*)arg,
//             sizeof(uint32_t) * 15);  // 15 uint32_t passed args in total
//         snrt_dma_wait_all();

//         // we use the following args, they are shared across the cluster
//         // arg0: uint32_t A_addr_hi get_cls_shared_ptrs()[0][0]
//         // arg1: uint32_t A_addr_lo get_cls_shared_ptrs()[0][1]
//         // arg2: uint32_t B_addr_hi get_cls_shared_ptrs()[0][2]
//         // arg3: uint32_t B_addr_lo get_cls_shared_ptrs()[0][3]
//         // arg4: uint32_t C_addr_hi get_cls_shared_ptrs()[0][4]
//         // arg5: uint32_t C_addr_lo get_cls_shared_ptrs()[0][5]
//         // arg6: uint32_t D_addr_hi get_cls_shared_ptrs()[0][6]
//         // arg7: uint32_t D_addr_lo get_cls_shared_ptrs()[0][7]
//         // arg8: uint32_t M get_cls_shared_ptrs()[0][8]
//         // arg9: uint32_t K get_cls_shared_ptrs()[0][9]
//         // arg10: uint32_t N get_cls_shared_ptrs()[0][10]
//         // arg11: uint32_t array_shape_idx get_cls_shared_ptrs()[0][11]
//         // arg12: transpose A get_cls_shared_ptrs()[0][12]
//         // arg13: transpose B get_cls_shared_ptrs()[0][13]
//         // arg14: get_cls_shared_ptrs()[0][14] is used for accumPrevC

//         // get_cls_shared_ptrs()[0][15] is used for addNonZeroC
//         // get_cls_shared_ptrs()[0][16] is used for meshRow
//         // get_cls_shared_ptrs()[0][17] is used for tileSize
//         // get_cls_shared_ptrs()[0][18] is used for meshCol

//         VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Load Args Done!\r\n");
//         /////////////////////////////////////////
//         // Allocate the L1 memory for Inputs   //
//         /////////////////////////////////////////
//         // Allocate the l1 space specifed by the arg
//         uint64_t l3_input_A_addr, l3_input_B_addr, l3_input_C_addr;

//         VERSACORE_DEBUG_PRINT(
//             "M = %d, K = %d, N = %d, array_shape_idx = %d\r\n",
//             get_cls_shared_ptrs()[0][8], get_cls_shared_ptrs()[0][9],
//             get_cls_shared_ptrs()[0][10], get_cls_shared_ptrs()[0][11]);
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             get_cls_shared_ptrs()[0][16] = meshRow_0;
//             get_cls_shared_ptrs()[0][17] = tileSize_0;
//             get_cls_shared_ptrs()[0][18] = meshCol_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 1) {
//             get_cls_shared_ptrs()[0][16] = meshRow_1;
//             get_cls_shared_ptrs()[0][17] = tileSize_1;
//             get_cls_shared_ptrs()[0][18] = meshCol_1;
//         } else if (get_cls_shared_ptrs()[0][11] == 2) {
//             get_cls_shared_ptrs()[0][16] = meshRow_2;
//             get_cls_shared_ptrs()[0][17] = tileSize_2;
//             get_cls_shared_ptrs()[0][18] = meshCol_2;
//         } else if (get_cls_shared_ptrs()[0][11] == 3) {
//             get_cls_shared_ptrs()[0][16] = meshRow_3;
//             get_cls_shared_ptrs()[0][17] = tileSize_3;
//             get_cls_shared_ptrs()[0][18] = meshCol_3;
//         } else if (get_cls_shared_ptrs()[0][11] == 4) {
//             get_cls_shared_ptrs()[0][16] = meshRow_4;
//             get_cls_shared_ptrs()[0][17] = tileSize_4;
//             get_cls_shared_ptrs()[0][18] = meshCol_4;
//         } else {
//             VERSACORE_DEBUG_PRINT("ERROR: array_shape_idx invalid!\r\n");
//             return;
//         }

//         // L1 data addresses for A
//         uint32_t A_size = (get_cls_shared_ptrs()[0][8]) *
//                           (get_cls_shared_ptrs()[0][9]) *
//                           get_cls_shared_ptrs()[0][16] *
//                           get_cls_shared_ptrs()[0][17] * sizeof(uint8_t);
//         A_size = (A_size + 63) & ~63;  // align to 64 bytes for xdma granularity
//         get_cls_shared_ptrs()[1] = snrt_l1_malloc(A_size);
//         if (!get_cls_shared_ptrs()[1]) {
//             VERSACORE_DEBUG_PRINT("ERROR: A alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated A at %p\r\n",
//                                   (void*)get_cls_shared_ptrs()[1]);
//         }

//         // Call the idma to load the inputs
//         // TODO: align with the sparse interconnect address granularity
//         l3_input_A_addr =
//             make_u64(get_cls_shared_ptrs()[0][0], get_cls_shared_ptrs()[0][1]);
//         xdma_memcpy_1d((void*)l3_input_A_addr, (void*)get_cls_shared_ptrs()[1],
//                        A_size);
//         int task_id = xdma_start();
//         xdma_remote_wait(task_id);

//         // L1 data addresses for B
//         get_cls_shared_ptrs()[2] = snrt_l1_malloc(
//             (get_cls_shared_ptrs()[0][9]) * (get_cls_shared_ptrs()[0][10]) *
//             get_cls_shared_ptrs()[0][17] * get_cls_shared_ptrs()[0][18] *
//             sizeof(uint8_t));  // B KxN in uint8_t
//         if (!get_cls_shared_ptrs()[2]) {
//             VERSACORE_DEBUG_PRINT("ERROR: B alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated B at %p\r\n",
//                                   (void*)get_cls_shared_ptrs()[2]);
//         }

//         // TODO: align with the sparse interconnect address granularity
//         l3_input_B_addr =
//             make_u64(get_cls_shared_ptrs()[0][2], get_cls_shared_ptrs()[0][3]);
//         xdma_memcpy_1d((void*)l3_input_B_addr, (void*)get_cls_shared_ptrs()[2],
//                        (get_cls_shared_ptrs()[0][9]) *
//                            (get_cls_shared_ptrs()[0][10]) *
//                            get_cls_shared_ptrs()[0][17] *
//                            get_cls_shared_ptrs()[0][18] * sizeof(uint8_t));
//         task_id = xdma_start();
//         xdma_remote_wait(task_id);

//         // L1 data addresses for C
//         // load C only if C_addr is not 0
//         // TODO: align with the sparse interconnect address granularity
//         l3_input_C_addr =
//             make_u64(get_cls_shared_ptrs()[0][4], get_cls_shared_ptrs()[0][5]);
//         if (get_cls_shared_ptrs()[0][14]) {
//             get_cls_shared_ptrs()[0][15] = 0;
//             get_cls_shared_ptrs()[3] = NULL;
//             VERSACORE_DEBUG_PRINT("skip loading C since b\r\n");
//         } else if (l3_input_C_addr != 0) {
//             get_cls_shared_ptrs()[0][15] = 1;
//             get_cls_shared_ptrs()[3] = snrt_l1_malloc(
//                 (get_cls_shared_ptrs()[0][8]) * (get_cls_shared_ptrs()[0][10]) *
//                 get_cls_shared_ptrs()[0][16] * get_cls_shared_ptrs()[0][18] *
//                 sizeof(uint32_t));  // C MxN in uint32
//             if (!get_cls_shared_ptrs()[3]) {
//                 VERSACORE_DEBUG_PRINT("ERROR: C alloc failed!\r\n");
//                 return;
//             } else {
//                 VERSACORE_DEBUG_PRINT("Allocated C at %p\r\n",
//                                       (void*)get_cls_shared_ptrs()[3]);
//             }
//             xdma_memcpy_1d(
//                 (void*)l3_input_C_addr, (void*)get_cls_shared_ptrs()[3],
//                 (get_cls_shared_ptrs()[0][8]) * (get_cls_shared_ptrs()[0][10]) *
//                     get_cls_shared_ptrs()[0][16] *
//                     get_cls_shared_ptrs()[0][18] * sizeof(uint32_t));
//             task_id = xdma_start();
//             xdma_remote_wait(task_id);
//         } else {
//             get_cls_shared_ptrs()[0][15] = 0;
//             get_cls_shared_ptrs()[3] = NULL;
//             VERSACORE_DEBUG_PRINT(
//                 "skip loading C since C_addr is 0, eg., C is always zero\r\n");
//         }

//         // L1 data addresses for D
//         get_cls_shared_ptrs()[4] = snrt_l1_malloc(
//             (get_cls_shared_ptrs()[0][8]) * (get_cls_shared_ptrs()[0][10]) *
//             get_cls_shared_ptrs()[0][16] * get_cls_shared_ptrs()[0][18] *
//             sizeof(uint32_t));  // D MxN in uint32
//         if (!get_cls_shared_ptrs()[4]) {
//             VERSACORE_DEBUG_PRINT("ERROR: D alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated D at %p\r\n",
//                                   (void*)get_cls_shared_ptrs()[4]);
//         }

//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Load Input Data Done!\r\n");
//     }

//     snrt_cluster_hw_barrier();

//     // The core 0 will be responsible for configuring and starting the versacore
//     if (snrt_cluster_core_idx() == 0) {
//         // compute the streamer cfg first
//         //////////////////////////////////////////////////////////////
//         // Allocated the L1 memory for Streamer CSRs    //
//         //////////////////////////////////////////////////////////////

//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Compute Streamer Cfg Start!\r\n");
//         // the snicth needs to generate these streamer cfgs
//         // 22 args in total
//         // A 6, B 6, C 5, D 5
//         get_cls_shared_ptrs()[5] = snrt_l1_malloc(sizeof(uint32_t) * 22);
//         if (!get_cls_shared_ptrs()[5]) {
//             VERSACORE_DEBUG_PRINT("ERROR: streamer cfg alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated streamer cfg at %p\r\n",
//                                   (void*)get_cls_shared_ptrs()[5]);
//         }
//         //////////////////////////////////////////////////////////////
//         // streamer cfg for A
//         //////////////////////////////////////////////////////////////

//         // Aslstride0
//         get_cls_shared_ptrs()[5][0] = bankWidth / 8;

//         // Atlbound0~5
//         uint32_t* Atlbound = snrt_l1_malloc(sizeof(uint32_t) * 6);
//         if (!Atlbound) {
//             VERSACORE_DEBUG_PRINT("ERROR: Atlbound alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Atlbound at %p\r\n",
//                                   (void*)Atlbound);
//         }
//         // Atlbound0
//         Atlbound[0] = (get_cls_shared_ptrs()[0][9]);
//         // Atlbound1
//         Atlbound[1] = (get_cls_shared_ptrs()[0][10]);
//         // Atlbound2
//         Atlbound[2] = (get_cls_shared_ptrs()[0][8]);
//         // Atlbound3
//         Atlbound[3] = 1;
//         // Atlbound4
//         Atlbound[4] = 1;
//         // Atlbound5
//         Atlbound[5] = 1;
//         get_cls_shared_ptrs()[5][1] = (uint32_t)(uintptr_t)Atlbound;

//         // Atlstride0~5
//         uint32_t* Atlstride = snrt_l1_malloc(sizeof(uint32_t) * 6);
//         if (!Atlstride) {
//             VERSACORE_DEBUG_PRINT("ERROR: Atlstride alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Atlstride at %p\r\n",
//                                   (void*)Atlstride);
//         }
//         // Atlstride0
//         Atlstride[0] =
//             get_cls_shared_ptrs()[0][16] * get_cls_shared_ptrs()[0][17];
//         // Atlstride1
//         Atlstride[1] = 0;
//         // Atlstride2
//         Atlstride[2] = get_cls_shared_ptrs()[0][16] *
//                        get_cls_shared_ptrs()[0][17] *
//                        get_cls_shared_ptrs()[0][9];
//         // Atlstride3
//         Atlstride[3] = 0;
//         // Atlstride4
//         Atlstride[4] = 0;
//         Atlstride[5] = 0;
//         get_cls_shared_ptrs()[5][2] = (uint32_t)(uintptr_t)Atlstride;

//         // set_addr_remap_index_A
//         get_cls_shared_ptrs()[5][3] = 0;
//         // transpose_A
//         get_cls_shared_ptrs()[5][4] = get_cls_shared_ptrs()[0][12];
//         // channel_en_A []
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             get_cls_shared_ptrs()[5][5] = channel_en_A_0_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 1) {
//             get_cls_shared_ptrs()[5][5] = channel_en_A_1_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 2) {
//             get_cls_shared_ptrs()[5][5] = channel_en_A_2_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 3) {
//             get_cls_shared_ptrs()[5][5] = channel_en_A_3_0;
//         }
//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Compute Streamer Cfg A Done!\r\n");

//         //////////////////////////////////////////////////////////////
//         // streamer cfg for B
//         //////////////////////////////////////////////////////////////

//         // Bslstride0
//         get_cls_shared_ptrs()[5][6] = bankWidth / 8;

//         // Btlbound0~2
//         uint32_t* Btlbound = snrt_l1_malloc(sizeof(uint32_t) * 3);
//         if (!Btlbound) {
//             VERSACORE_DEBUG_PRINT("ERROR: Btlbound alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Btlbound at %p\r\n",
//                                   (void*)Btlbound);
//         }
//         get_cls_shared_ptrs()[5][7] = (uint32_t)(uintptr_t)Btlbound;
//         // Btlbound0
//         Btlbound[0] = (get_cls_shared_ptrs()[0][9]);
//         // Btlbound1
//         Btlbound[1] = (get_cls_shared_ptrs()[0][10]);
//         // Btlbound2
//         Btlbound[2] = (get_cls_shared_ptrs()[0][8]);

//         // Btlstride0~2
//         uint32_t* Btlstride = snrt_l1_malloc(sizeof(uint32_t) * 3);
//         if (!Btlstride) {
//             VERSACORE_DEBUG_PRINT("ERROR: Btlstride alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Btlstride at %p\r\n",
//                                   (void*)Btlstride);
//         }
//         get_cls_shared_ptrs()[5][8] = (uint32_t)(uintptr_t)Btlstride;

//         // Btlstride0
//         Btlstride[0] =
//             get_cls_shared_ptrs()[0][17] * get_cls_shared_ptrs()[0][18];
//         // Btlstride1
//         Btlstride[1] = get_cls_shared_ptrs()[0][17] *
//                        get_cls_shared_ptrs()[0][18] *
//                        get_cls_shared_ptrs()[0][9];

//         // Btlstride2
//         Btlstride[2] = 0;

//         // set_addr_remap_index_B
//         get_cls_shared_ptrs()[5][9] = 0;
//         // transpose_B
//         get_cls_shared_ptrs()[5][10] = get_cls_shared_ptrs()[0][13];
//         // channel_en_B []
//         uint32_t* channel_en_B = snrt_l1_malloc(sizeof(uint32_t) * 2);
//         if (!channel_en_B) {
//             VERSACORE_DEBUG_PRINT("ERROR: channel_en_B alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated channel_en_B at %p\r\n",
//                                   (void*)channel_en_B);
//         }
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             channel_en_B[0] = channel_en_B_0_0;
//             channel_en_B[1] = channel_en_B_0_1;
//         } else if (get_cls_shared_ptrs()[0][11] == 1) {
//             channel_en_B[0] = channel_en_B_1_0;
//             channel_en_B[1] = channel_en_B_1_1;
//         } else if (get_cls_shared_ptrs()[0][11] == 2) {
//             channel_en_B[0] = channel_en_B_2_0;
//             channel_en_B[1] = channel_en_B_2_1;
//         } else if (get_cls_shared_ptrs()[0][11] == 3) {
//             channel_en_B[0] = channel_en_B_3_0;
//             channel_en_B[1] = channel_en_B_3_1;
//         }
//         get_cls_shared_ptrs()[5][11] = (uint32_t)(uintptr_t)channel_en_B;
//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Compute Streamer Cfg B Done!\r\n");

//         //////////////////////////////////////////////////////////////
//         // streamer cfg for C
//         //////////////////////////////////////////////////////////////
//         // Cslstride0
//         get_cls_shared_ptrs()[5][12] = bankWidth / 8;
//         // Ctlbound0~3
//         uint32_t* Ctlbound = snrt_l1_malloc(sizeof(uint32_t) * 4);
//         if (!Ctlbound) {
//             VERSACORE_DEBUG_PRINT("ERROR: Ctlbound alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Ctlbound at %p\r\n",
//                                   (void*)Ctlbound);
//         }
//         get_cls_shared_ptrs()[5][13] = (uint32_t)(uintptr_t)Ctlbound;
//         // Ctlbound0
//         if (get_cls_shared_ptrs()[0][14] == 1) {
//             // accumPrevC is true
//             Ctlbound[0] = 0;
//         } else if (get_cls_shared_ptrs()[0][11] == 0) { // else bound is normal
//             Ctlbound[0] = Ctlbound0_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 1) {
//             Ctlbound[0] = Ctlbound0_1;
//         } else if (get_cls_shared_ptrs()[0][11] == 2) {
//             Ctlbound[0] = Ctlbound0_2;
//         } else if (get_cls_shared_ptrs()[0][11] == 3) {
//             Ctlbound[0] = Ctlbound0_3;
//         }
//         Ctlbound[1] = (get_cls_shared_ptrs()[0][10]);
//         Ctlbound[2] = (get_cls_shared_ptrs()[0][8]);
//         Ctlbound[3] = 1;
//         // Ctlstride0~3
//         uint32_t* Ctlstride = snrt_l1_malloc(sizeof(uint32_t) * 4);
//         if (!Ctlstride) {
//             VERSACORE_DEBUG_PRINT("ERROR: Ctlstride alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Ctlstride at %p\r\n",
//                                   (void*)Ctlstride);
//         }
//         get_cls_shared_ptrs()[5][14] = (uint32_t)(uintptr_t)Ctlstride;

//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             // Ctlstride0
//             Ctlstride[0] = Ctlstride0_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 1) {
//             // Ctlstride0
//             Ctlstride[0] = Ctlstride0_1;
//         } else if (get_cls_shared_ptrs()[0][11] == 2) {
//             // Ctlstride0
//             Ctlstride[0] = Ctlstride0_2;
//         } else if (get_cls_shared_ptrs()[0][11] == 3) {
//             // Ctlstride0
//             Ctlstride[0] = Ctlstride0_3;
//         }

//         // Ctlstride1
//         Ctlstride[1] = C_elem_len * get_cls_shared_ptrs()[0][16] *
//                        get_cls_shared_ptrs()[0][18] / 8;
//         // Ctlstride2
//         Ctlstride[2] = get_cls_shared_ptrs()[0][10] * C_elem_len *
//                        get_cls_shared_ptrs()[0][16] *
//                        get_cls_shared_ptrs()[0][18] / 8;
//         // Ctlstride3
//         Ctlstride[3] = 0;
//         // set_addr_remap_index_C
//         get_cls_shared_ptrs()[5][15] = 0;
//         // channel_en_C []
//         // set channel_en_C to zero if accumPrevC or addNonZeroC
//         if (get_cls_shared_ptrs()[0][14] == 1 || get_cls_shared_ptrs()[0][15] == 0) {
//             get_cls_shared_ptrs()[5][16] = channel_en_C_null_0_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 0) {
//             get_cls_shared_ptrs()[5][16] = channel_en_C_0_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 1) {
//             get_cls_shared_ptrs()[5][16] = channel_en_C_1_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 2) {
//             get_cls_shared_ptrs()[5][16] = channel_en_C_2_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 3) {
//             get_cls_shared_ptrs()[5][16] = channel_en_C_3_0;
//         }
//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Compute Streamer Cfg C Done!\r\n");

//         //////////////////////////////////////////////////////////////
//         // streamer cfg for D
//         //////////////////////////////////////////////////////////////
//         // D32slstride0
//         get_cls_shared_ptrs()[5][17] = bankWidth / 8;
//         // D32tlbound0~3
//         uint32_t* D32tlbound = snrt_l1_malloc(sizeof(uint32_t) * 4);
//         if (!D32tlbound) {
//             VERSACORE_DEBUG_PRINT("ERROR: D32tlbound alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated D32tlbound at %p\r\n",
//                                   (void*)D32tlbound);
//         }
//         get_cls_shared_ptrs()[5][18] = (uint32_t)(uintptr_t)D32tlbound;
//         // D32tlbound0~3
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             D32tlbound[0] = D32tlbound0_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 1) {
//             D32tlbound[0] = D32tlbound0_1;
//         } else if (get_cls_shared_ptrs()[0][11] == 2) {
//             D32tlbound[0] = D32tlbound0_2;
//         } else if (get_cls_shared_ptrs()[0][11] == 3) {
//             D32tlbound[0] = D32tlbound0_3;
//         }
//         D32tlbound[1] = (get_cls_shared_ptrs()[0][10]);
//         D32tlbound[2] = (get_cls_shared_ptrs()[0][8]);
//         D32tlbound[3] = 1;
//         // D32tlstride0~3
//         uint32_t* D32tlstride = snrt_l1_malloc(sizeof(uint32_t) * 4);
//         if (!D32tlstride) {
//             VERSACORE_DEBUG_PRINT("ERROR: D32tlstride alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated D32tlstride at %p\r\n",
//                                   (void*)D32tlstride);
//         }
//         get_cls_shared_ptrs()[5][19] = (uint32_t)(uintptr_t)D32tlstride;
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             // D32tlstride0
//             D32tlstride[0] = D32tlstride0_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 1) {
//             // D32tlstride0
//             D32tlstride[0] = D32tlstride0_1;
//         } else if (get_cls_shared_ptrs()[0][11] == 2) {
//             // D32tlstride0
//             D32tlstride[0] = D32tlstride0_2;
//         } else if (get_cls_shared_ptrs()[0][11] == 3) {
//             // D32tlstride0
//             D32tlstride[0] = D32tlstride0_3;
//         }
//         // D32tlstride1
//         D32tlstride[1] = D32_elem_len * get_cls_shared_ptrs()[0][16] *
//                          get_cls_shared_ptrs()[0][18] / 8;
//         // D32tlstride2
//         D32tlstride[2] = (get_cls_shared_ptrs()[0][10]) * D32_elem_len *
//                          get_cls_shared_ptrs()[0][16] *
//                          get_cls_shared_ptrs()[0][18] / 8;
//         // D32tlstride3
//         D32tlstride[3] = 0;
//         // set_addr_remap_index_D32
//         get_cls_shared_ptrs()[5][20] = 0;
//         // channel_en_D32 []
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             get_cls_shared_ptrs()[5][21] = channel_en_D32_0_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 1) {
//             get_cls_shared_ptrs()[5][21] = channel_en_D32_1_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 2) {
//             get_cls_shared_ptrs()[5][21] = channel_en_D32_2_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 3) {
//             get_cls_shared_ptrs()[5][21] = channel_en_D32_3_0;
//         }
//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Compute Streamer Cfg D Done!\r\n");

//         //////////////////////////////////////////////////////////////
//         // Configuration the Steamer and Versacore
//         //////////////////////////////////////////////////////////////

//         set_versacore_streamer_csr(
//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[1],  // A_addr
//             (uint32_t*)(&get_cls_shared_ptrs()[5][0]),      // Aslstride
//             (uint32_t*)(get_cls_shared_ptrs()[5][1]),       // Atlbound[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][2]),       // Atlstride[] base
//             get_cls_shared_ptrs()[5][3],  // set_addr_remap_index_A
//             get_cls_shared_ptrs()[5][4],  // transpose_A
//             (uint32_t*)(&get_cls_shared_ptrs()[5][5]),  // channel_en_A []

//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[2],  // B_addr
//             (uint32_t*)(&get_cls_shared_ptrs()[5][6]),      // Bslstride[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][7]),       // Btlbound[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][8]),       // Btlstride[] base
//             get_cls_shared_ptrs()[5][9],   // set_addr_remap_index_B
//             get_cls_shared_ptrs()[5][10],  // transpose_B
//             (uint32_t*)(get_cls_shared_ptrs()[5][11]),  // channel_en_B []

//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[3],  // C_addr
//             (uint32_t*)(&get_cls_shared_ptrs()[5][12]),     // Cslstride[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][13]),      // Ctlbound[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][14]),      // Ctlstride[] base
//             get_cls_shared_ptrs()[5][15],  // set_addr_remap_index_C
//             (uint32_t*)(&get_cls_shared_ptrs()[5][16]),  // channel_en_C []

//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[4],  // D_addr
//             (uint32_t*)(&get_cls_shared_ptrs()[5][17]),  // D32slstride[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][18]),   // D32tlbound[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][19]),   // D32tlstride[] base
//             get_cls_shared_ptrs()[5][20],  // set_addr_remap_index_D32
//             (uint32_t*)(&get_cls_shared_ptrs()[5][21]),  // channel_en_D32 []

//             0, 0, 0, 0, 0, 0, 0

//         );

//         set_versacore_csr(
//             // accPrevC means takes new C
//             get_cls_shared_ptrs()[0][14] == 0, get_cls_shared_ptrs()[0][9],
//             get_cls_shared_ptrs()[0][10] * get_cls_shared_ptrs()[0][8], 0,
//             get_cls_shared_ptrs()[0][11], 0);

//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Streamer Configuration Done!\r\n");

//         // Set CSR to start Streamer
//         start_versacore_and_streamer();

//         // Poll until Streamer and GEMM accelerator finish
//         wait_versacore_and_streamer();

//         VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Compute Done!\r\n");

//         for (int i = 0;
//              i <
//              (get_cls_shared_ptrs()[0][8]) * (get_cls_shared_ptrs()[0][10]) *
//                  get_cls_shared_ptrs()[0][16] * get_cls_shared_ptrs()[0][18];
//              i++) {
//             VERSACORE_DEBUG_PRINT("D[%d] = %d\r\n", i,
//                                   get_cls_shared_ptrs()[4][i]);
//         }
//     }

//     snrt_cluster_hw_barrier();

//     // After computation, dma core will store the output back to L3
//     if (snrt_is_dm_core()) {
//         uint32_t output_size = (get_cls_shared_ptrs()[0][8]) *
//                                (get_cls_shared_ptrs()[0][10]) *
//                                get_cls_shared_ptrs()[0][16] *
//                                get_cls_shared_ptrs()[0][18] * sizeof(uint32_t);
//         uint64_t l3_output_D_addr;
//         // l3_output_D_addr output address in L3
//         l3_output_D_addr =
//             make_u64(get_cls_shared_ptrs()[0][6], get_cls_shared_ptrs()[0][7]);
//         VERSACORE_DEBUG_PRINT(
//             "[Cluster %d] Core(%d) Store Output: Dst = %x, Src = %x, size = "
//             "%d\r\n",
//             snrt_cluster_idx(), snrt_cluster_core_idx(),
//             get_cls_shared_ptrs()[4], (uint32_t)l3_output_D_addr, output_size);
//         xdma_memcpy_1d((void*)get_cls_shared_ptrs()[4], (void*)l3_output_D_addr, output_size);
//         int task_id = xdma_start();
//         xdma_remote_wait(task_id);
//         // Free the allocated L1 memory
//         snrt_l1_free(get_cls_shared_ptrs()[0]);
//         snrt_l1_free(get_cls_shared_ptrs()[1]);
//         snrt_l1_free(get_cls_shared_ptrs()[2]);
//         // Free C only if it is allocated when accNonZeroC is true
//         if (get_cls_shared_ptrs()[0][15]) {
//             snrt_l1_free(get_cls_shared_ptrs()[3]);
//         }
//         snrt_l1_free(get_cls_shared_ptrs()[4]);
//         snrt_l1_free(get_cls_shared_ptrs()[5]);
//         // Reset the shared pointers
//         snrt_memset(get_cls_shared_ptrs(), 0,
//                     sizeof(uint32_t*) * NUM_CLS_SHARED_PTRS);
//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Store Output Done!\r\n");
//     }
// }

// SNAX_LIB_DEFINE void __snax_kernel_gemm_compute_only_intra_chiplet(void* arg) {
//     // move the args to local variables
//     if (snrt_is_dm_core()) {
//         // allocate L1 memory for global variables across the cluster
//         ///////////////////////////////////////
//         // Allocate the L1 memory //
//         ///////////////////////////////////////
//         get_cls_shared_ptrs()[0] = (uint32_t*)snrt_l1_malloc(128);
//         if (!get_cls_shared_ptrs()[0]) {
//             VERSACORE_DEBUG_PRINT("ERROR: arg alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated arg at %p\r\n",
//                                   (void*)get_cls_shared_ptrs()[0]);
//         }

//         ////////////////////////////////////
//         // Call the idma to load the args //
//         ////////////////////////////////////
//         // First we need to use the DMA to load all the arguments from L3 into
//         // L1 Here we assume the arguments are packed in a contiguous way in L3
//         VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Start!\r\n");

//         snrt_dma_start_1d(
//             (void*)get_cls_shared_ptrs()[0], (void*)arg,
//             sizeof(uint32_t) * 15);  // 15 uint32_t passed args in total
//         snrt_dma_wait_all();

//         // we use the following args, they are shared across the cluster
//         // arg0: uint32_t A_addr_hi get_cls_shared_ptrs()[0][0]
//         // arg1: uint32_t A_addr_lo get_cls_shared_ptrs()[0][1]
//         // arg2: uint32_t B_addr_hi get_cls_shared_ptrs()[0][2]
//         // arg3: uint32_t B_addr_lo get_cls_shared_ptrs()[0][3]
//         // arg4: uint32_t C_addr_hi get_cls_shared_ptrs()[0][4]
//         // arg5: uint32_t C_addr_lo get_cls_shared_ptrs()[0][5]
//         // arg6: uint32_t D_addr_hi get_cls_shared_ptrs()[0][6]
//         // arg7: uint32_t D_addr_lo get_cls_shared_ptrs()[0][7]
//         // these address are all in L1! It's the TCDM local address.

//         // arg8: uint32_t M get_cls_shared_ptrs()[0][8]
//         // arg9: uint32_t K get_cls_shared_ptrs()[0][9]
//         // arg10: uint32_t N get_cls_shared_ptrs()[0][10]
//         // arg11: uint32_t array_shape_idx get_cls_shared_ptrs()[0][11]
//         // arg12: transpose A get_cls_shared_ptrs()[0][12]
//         // arg13: transpose B get_cls_shared_ptrs()[0][13]
//         // arg14: get_cls_shared_ptrs()[0][14] is used for accumPrevC

//         // get_cls_shared_ptrs()[0][15] is used for addNonZeroC
//         if (get_cls_shared_ptrs()[0][14]) {
//             get_cls_shared_ptrs()[0][15] = 0;
//         } else if (make_u64(get_cls_shared_ptrs()[0][4],
//                             get_cls_shared_ptrs()[0][5]) != 0) {
//             get_cls_shared_ptrs()[0][15] = 1;
//         } else {
//             get_cls_shared_ptrs()[0][15] = 0;
//         }

//         // get_cls_shared_ptrs()[0][16] is used for meshRow
//         // get_cls_shared_ptrs()[0][17] is used for tileSize
//         // get_cls_shared_ptrs()[0][18] is used for meshCol
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             get_cls_shared_ptrs()[0][16] = meshRow_0;
//             get_cls_shared_ptrs()[0][17] = tileSize_0;
//             get_cls_shared_ptrs()[0][18] = meshCol_0;
//         } else {
//             get_cls_shared_ptrs()[0][16] = meshRow_1;
//             get_cls_shared_ptrs()[0][17] = tileSize_1;
//             get_cls_shared_ptrs()[0][18] = meshCol_1;
//         }
//     }

//     snrt_cluster_hw_barrier();

//     // The core 0 will be responsible for configuring and starting the versacore
//     if (snrt_cluster_core_idx() == 0) {
//         // compute the streamer cfg first
//         //////////////////////////////////////////////////////////////
//         // Allocated the L1 memory for Streamer CSRs    //
//         //////////////////////////////////////////////////////////////

//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Compute Streamer Cfg Start!\r\n");
//         // the snicth needs to generate these streamer cfgs
//         // 22 args in total
//         // A 6, B 6, C 5, D 5
//         get_cls_shared_ptrs()[5] = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 22);
//         if (!get_cls_shared_ptrs()[5]) {
//             VERSACORE_DEBUG_PRINT("ERROR: streamer cfg alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated streamer cfg at %p\r\n",
//                                   (void*)get_cls_shared_ptrs()[5]);
//         }
//         //////////////////////////////////////////////////////////////
//         // streamer cfg for A
//         //////////////////////////////////////////////////////////////

//         // Aslstride0
//         get_cls_shared_ptrs()[5][0] = bankWidth / 8;

//         // Atlbound0~5
//         uint32_t* Atlbound = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 6);
//         if (!Atlbound) {
//             VERSACORE_DEBUG_PRINT("ERROR: Atlbound alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Atlbound at %p\r\n",
//                                   (void*)Atlbound);
//         }
//         // Atlbound0
//         Atlbound[0] = (get_cls_shared_ptrs()[0][9]);
//         // Atlbound1
//         Atlbound[1] = (get_cls_shared_ptrs()[0][10]);
//         // Atlbound2
//         Atlbound[2] = (get_cls_shared_ptrs()[0][8]);
//         // Atlbound3
//         Atlbound[3] = 1;
//         // Atlbound4
//         Atlbound[4] = 1;
//         // Atlbound5
//         Atlbound[5] = 1;
//         get_cls_shared_ptrs()[5][1] = (uint32_t)(uintptr_t)Atlbound;

//         // Atlstride0~5
//         uint32_t* Atlstride = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 6);
//         if (!Atlstride) {
//             VERSACORE_DEBUG_PRINT("ERROR: Atlstride alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Atlstride at %p\r\n",
//                                   (void*)Atlstride);
//         }
//         // Atlstride0
//         Atlstride[0] =
//             get_cls_shared_ptrs()[0][16] * get_cls_shared_ptrs()[0][17];
//         // Atlstride1
//         Atlstride[1] = 0;
//         // Atlstride2
//         Atlstride[2] = get_cls_shared_ptrs()[0][16] *
//                        get_cls_shared_ptrs()[0][17] *
//                        get_cls_shared_ptrs()[0][9];
//         // Atlstride3
//         Atlstride[3] = 0;
//         // Atlstride4
//         Atlstride[4] = 0;
//         Atlstride[5] = 0;
//         get_cls_shared_ptrs()[5][2] = (uint32_t)(uintptr_t)Atlstride;

//         // set_addr_remap_index_A
//         get_cls_shared_ptrs()[5][3] = 0;
//         // transpose_A
//         get_cls_shared_ptrs()[5][4] = get_cls_shared_ptrs()[0][12];
//         // channel_en_A []
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             get_cls_shared_ptrs()[5][5] = channel_en_A_0_0;
//         } else {
//             get_cls_shared_ptrs()[5][5] = channel_en_A_1_0;
//         }
//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Compute Streamer Cfg A Done!\r\n");

//         //////////////////////////////////////////////////////////////
//         // streamer cfg for B
//         //////////////////////////////////////////////////////////////

//         // Bslstride0
//         get_cls_shared_ptrs()[5][6] = bankWidth / 8;

//         // Btlbound0~2
//         uint32_t* Btlbound = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 3);
//         if (!Btlbound) {
//             VERSACORE_DEBUG_PRINT("ERROR: Btlbound alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Btlbound at %p\r\n",
//                                   (void*)Btlbound);
//         }
//         get_cls_shared_ptrs()[5][7] = (uint32_t)(uintptr_t)Btlbound;
//         // Btlbound0
//         Btlbound[0] = (get_cls_shared_ptrs()[0][9]);
//         // Btlbound1
//         Btlbound[1] = (get_cls_shared_ptrs()[0][10]);
//         // Btlbound2
//         Btlbound[2] = (get_cls_shared_ptrs()[0][8]);

//         // Btlstride0~2
//         uint32_t* Btlstride = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 3);
//         if (!Btlstride) {
//             VERSACORE_DEBUG_PRINT("ERROR: Btlstride alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Btlstride at %p\r\n",
//                                   (void*)Btlstride);
//         }
//         get_cls_shared_ptrs()[5][8] = (uint32_t)(uintptr_t)Btlstride;

//         // Btlstride0
//         Btlstride[0] =
//             get_cls_shared_ptrs()[0][17] * get_cls_shared_ptrs()[0][18];
//         // Btlstride1
//         Btlstride[1] = get_cls_shared_ptrs()[0][17] *
//                        get_cls_shared_ptrs()[0][18] *
//                        get_cls_shared_ptrs()[0][9];

//         // Btlstride2
//         Btlstride[2] = 0;

//         // set_addr_remap_index_B
//         get_cls_shared_ptrs()[5][9] = 0;
//         // transpose_B
//         get_cls_shared_ptrs()[5][10] = get_cls_shared_ptrs()[0][13];
//         // channel_en_B []
//         uint32_t* channel_en_B = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 2);
//         if (!channel_en_B) {
//             VERSACORE_DEBUG_PRINT("ERROR: channel_en_B alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated channel_en_B at %p\r\n",
//                                   (void*)channel_en_B);
//         }
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             channel_en_B[0] = channel_en_B_0_0;
//             channel_en_B[1] = channel_en_B_0_1;
//         } else {
//             channel_en_B[0] = channel_en_B_1_0;
//             channel_en_B[1] = channel_en_B_1_1;
//         }
//         get_cls_shared_ptrs()[5][11] = (uint32_t)(uintptr_t)channel_en_B;
//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Compute Streamer Cfg B Done!\r\n");

//         //////////////////////////////////////////////////////////////
//         // streamer cfg for C
//         //////////////////////////////////////////////////////////////
//         // Cslstride0
//         get_cls_shared_ptrs()[5][12] = bankWidth / 8;
//         // Ctlbound0~3
//         uint32_t* Ctlbound = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 4);
//         if (!Ctlbound) {
//             VERSACORE_DEBUG_PRINT("ERROR: Ctlbound alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Ctlbound at %p\r\n",
//                                   (void*)Ctlbound);
//         }
//         get_cls_shared_ptrs()[5][13] = (uint32_t)(uintptr_t)Ctlbound;
//         // Ctlbound0
//         if (get_cls_shared_ptrs()[0][14] == 1) {
//             // accumPrevC is true
//             Ctlbound[0] = 0;
//         } else if (get_cls_shared_ptrs()[0][11] == 0) {  // else bound is normal
//             Ctlbound[0] = Ctlbound0_0;
//         } else {
//             Ctlbound[0] = Ctlbound0_1;
//         }
//         Ctlbound[1] = (get_cls_shared_ptrs()[0][10]);
//         Ctlbound[2] = (get_cls_shared_ptrs()[0][8]);
//         Ctlbound[3] = 1;
//         // Ctlstride0~3
//         uint32_t* Ctlstride = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 4);
//         if (!Ctlstride) {
//             VERSACORE_DEBUG_PRINT("ERROR: Ctlstride alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated Ctlstride at %p\r\n",
//                                   (void*)Ctlstride);
//         }
//         get_cls_shared_ptrs()[5][14] = (uint32_t)(uintptr_t)Ctlstride;

//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             // Ctlstride0
//             Ctlstride[0] = Ctlstride0_0;
//         } else {
//             // Ctlstride0
//             Ctlstride[0] = Ctlstride0_1;
//         }
//         // Ctlstride1
//         Ctlstride[1] = C_elem_len * get_cls_shared_ptrs()[0][16] *
//                        get_cls_shared_ptrs()[0][18] / 8;
//         // Ctlstride2
//         Ctlstride[2] = get_cls_shared_ptrs()[0][10] * C_elem_len *
//                        get_cls_shared_ptrs()[0][16] *
//                        get_cls_shared_ptrs()[0][18] / 8;
//         // Ctlstride3
//         Ctlstride[3] = 0;
//         // set_addr_remap_index_C
//         get_cls_shared_ptrs()[5][15] = 0;
//         // channel_en_C []
//         // set channel_en_C to zero if accumPrevC or addNonZeroC
//         if (get_cls_shared_ptrs()[0][14] == 1 ||
//             get_cls_shared_ptrs()[0][15] == 0) {
//             get_cls_shared_ptrs()[5][16] = channel_en_C_null_0_0;
//         } else if (get_cls_shared_ptrs()[0][11] == 0) {
//             get_cls_shared_ptrs()[5][16] = channel_en_C_0_0;
//         } else {
//             get_cls_shared_ptrs()[5][16] = channel_en_C_1_0;
//         }
//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Compute Streamer Cfg C Done!\r\n");

//         //////////////////////////////////////////////////////////////
//         // streamer cfg for D
//         //////////////////////////////////////////////////////////////
//         // D32slstride0
//         get_cls_shared_ptrs()[5][17] = bankWidth / 8;
//         // D32tlbound0~3
//         uint32_t* D32tlbound = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 4);
//         if (!D32tlbound) {
//             VERSACORE_DEBUG_PRINT("ERROR: D32tlbound alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated D32tlbound at %p\r\n",
//                                   (void*)D32tlbound);
//         }
//         get_cls_shared_ptrs()[5][18] = (uint32_t)(uintptr_t)D32tlbound;
//         // D32tlbound0~3
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             D32tlbound[0] = D32tlbound0_0;
//         } else {
//             D32tlbound[0] = D32tlbound0_1;
//         }
//         D32tlbound[1] = (get_cls_shared_ptrs()[0][10]);
//         D32tlbound[2] = (get_cls_shared_ptrs()[0][8]);
//         D32tlbound[3] = 1;
//         // D32tlstride0~3
//         uint32_t* D32tlstride = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 4);
//         if (!D32tlstride) {
//             VERSACORE_DEBUG_PRINT("ERROR: D32tlstride alloc failed!\r\n");
//             return;
//         } else {
//             VERSACORE_DEBUG_PRINT("Allocated D32tlstride at %p\r\n",
//                                   (void*)D32tlstride);
//         }
//         get_cls_shared_ptrs()[5][19] = (uint32_t)(uintptr_t)D32tlstride;
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             // D32tlstride0
//             D32tlstride[0] = D32tlstride0_0;
//         } else {
//             // D32tlstride0
//             D32tlstride[0] = D32tlstride0_1;
//         }
//         // D32tlstride1
//         D32tlstride[1] = D32_elem_len * get_cls_shared_ptrs()[0][16] *
//                          get_cls_shared_ptrs()[0][18] / 8;
//         // D32tlstride2
//         D32tlstride[2] = (get_cls_shared_ptrs()[0][10]) * D32_elem_len *
//                          get_cls_shared_ptrs()[0][16] *
//                          get_cls_shared_ptrs()[0][18] / 8;
//         // D32tlstride3
//         D32tlstride[3] = 0;
//         // set_addr_remap_index_D32
//         get_cls_shared_ptrs()[5][20] = 0;
//         // channel_en_D32 []
//         if (get_cls_shared_ptrs()[0][11] == 0) {
//             get_cls_shared_ptrs()[5][21] = channel_en_D32_0_0;
//         } else {
//             get_cls_shared_ptrs()[5][21] = channel_en_D32_1_0;
//         }
//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Compute Streamer Cfg D Done!\r\n");

//         //////////////////////////////////////////////////////////////
//         // Configuration the Steamer and Versacore
//         //////////////////////////////////////////////////////////////

//         set_versacore_streamer_csr(
//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[0][1],  // A_addr
//             (uint32_t*)(&get_cls_shared_ptrs()[5][0]),         // Aslstride
//             (uint32_t*)(get_cls_shared_ptrs()[5][1]),  // Atlbound[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][2]),  // Atlstride[] base
//             get_cls_shared_ptrs()[5][3],               // set_addr_remap_index_A
//             get_cls_shared_ptrs()[5][4],               // transpose_A
//             (uint32_t*)(&get_cls_shared_ptrs()[5][5]),  // channel_en_A []

//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[0][3],  // B_addr
//             (uint32_t*)(&get_cls_shared_ptrs()[5][6]),  // Bslstride[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][7]),   // Btlbound[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][8]),   // Btlstride[] base
//             get_cls_shared_ptrs()[5][9],   // set_addr_remap_index_B
//             get_cls_shared_ptrs()[5][10],  // transpose_B
//             (uint32_t*)(get_cls_shared_ptrs()[5][11]),  // channel_en_B []

//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[0][5],  // C_addr
//             (uint32_t*)(&get_cls_shared_ptrs()[5][12]),  // Cslstride[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][13]),   // Ctlbound[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][14]),   // Ctlstride[] base
//             get_cls_shared_ptrs()[5][15],  // set_addr_remap_index_C
//             (uint32_t*)(&get_cls_shared_ptrs()[5][16]),  // channel_en_C []

//             (uint32_t)(uintptr_t)get_cls_shared_ptrs()[0][7],  // D_addr
//             (uint32_t*)(&get_cls_shared_ptrs()[5][17]),  // D32slstride[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][18]),   // D32tlbound[] base
//             (uint32_t*)(get_cls_shared_ptrs()[5][19]),   // D32tlstride[] base
//             get_cls_shared_ptrs()[5][20],  // set_addr_remap_index_D32
//             (uint32_t*)(&get_cls_shared_ptrs()[5][21])  // channel_en_D32 []
//             ,
//             0, 0, 0, 0, 0, 0, 0, 0, 0
//         );

//         set_versacore_csr(
//             // accPrevC means takes new C
//             get_cls_shared_ptrs()[0][14] == 0, get_cls_shared_ptrs()[0][9],
//             get_cls_shared_ptrs()[0][10] * get_cls_shared_ptrs()[0][8], 0,
//             get_cls_shared_ptrs()[0][11], 0);

//         VERSACORE_DEBUG_PRINT(
//             "GEMM Intra-Chiplet Kernel Streamer Configuration Done!\r\n");

//         // Set CSR to start Streamer
//         start_versacore_and_streamer();

//         // Poll until Streamer and GEMM accelerator finish
//         wait_versacore_and_streamer();

//         VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Compute Done!\r\n");

//         for (int i = 0; i < 10; i++) {
//             VERSACORE_DEBUG_PRINT("D[%d] = %d\r\n", i,
//                                   get_cls_shared_ptrs()[0][7][i]);
//         }
//     }
// }

SNAX_LIB_DEFINE void __snax_kernel_gemm(void* arg) {
    // Compute GeMM D = A*B + C
    // This kernel will check the A,B,C matrix address to decide whether to load inputs or
    // the inputs are already in the local TCDM
    // Usage 1: All the matrix are not in L1, so the kernel will load all inputs using DMA
    // Usage 2: Some matrix are already in L1, so the kernel will skip loading those matrix and assume they are in the local TCDM already 

    // move the args to local variables
    if (snrt_is_dm_core()) {
        // allocate L1 memory for global variables across the cluster
        ///////////////////////////////////////
        // Allocate the L1 memory //
        ///////////////////////////////////////
        get_cls_shared_ptrs()[0] = (uint32_t*)snrt_l1_malloc(128);
        if (!get_cls_shared_ptrs()[0]) {
            VERSACORE_DEBUG_PRINT("ERROR: arg alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated arg at %p\r\n",
                                  (void*)get_cls_shared_ptrs()[0]);
        }

        ////////////////////////////////////
        // Call the idma to load the args //
        ////////////////////////////////////
        // First we need to use the DMA to load all the arguments from L3 into
        // L1 Here we assume the arguments are packed in a contiguous way in L3
        VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Start!\r\n");

        snrt_dma_start_1d(
            (void*)get_cls_shared_ptrs()[0], (void*)arg,
            sizeof(uint32_t) * 15);  // 15 uint32_t passed args in total
        snrt_dma_wait_all();

        uint32_t* arg_ptr = get_cls_shared_ptrs()[0];   
        uint32_t A_addr_hi = arg_ptr[0];
        uint32_t A_addr_lo = arg_ptr[1];
        uint32_t B_addr_hi = arg_ptr[2];
        uint32_t B_addr_lo = arg_ptr[3];
        uint32_t C_addr_hi = arg_ptr[4];
        uint32_t C_addr_lo = arg_ptr[5];
        uint32_t D_addr_hi = arg_ptr[6];
        uint32_t D_addr_lo = arg_ptr[7];
        uint32_t M = arg_ptr[8];
        uint32_t K = arg_ptr[9];
        uint32_t N = arg_ptr[10];
        uint32_t array_shape_idx = arg_ptr[11];
        uint32_t transpose_A = arg_ptr[12];
        uint32_t transpose_B = arg_ptr[13];
        uint32_t accumPrevC = arg_ptr[14];
        // VERSACORE_DEBUG_PRINT("chip id: %d\r\n", get_current_chip_id());
        VERSACORE_DEBUG_PRINT("A addr: 0x%08x%08x\r\n", A_addr_hi, A_addr_lo);
        VERSACORE_DEBUG_PRINT("B addr: 0x%08x%08x\r\n", B_addr_hi, B_addr_lo);
        VERSACORE_DEBUG_PRINT("C addr: 0x%08x%08x\r\n", C_addr_hi, C_addr_lo);
        VERSACORE_DEBUG_PRINT("D addr: 0x%08x%08x\r\n", D_addr_hi, D_addr_lo);

        // we use the following args, they are shared across the cluster
        // arg0: uint32_t A_addr_hi get_cls_shared_ptrs()[0][0]
        // arg1: uint32_t A_addr_lo get_cls_shared_ptrs()[0][1]
        // arg2: uint32_t B_addr_hi get_cls_shared_ptrs()[0][2]
        // arg3: uint32_t B_addr_lo get_cls_shared_ptrs()[0][3]
        // arg4: uint32_t C_addr_hi get_cls_shared_ptrs()[0][4]
        // arg5: uint32_t C_addr_lo get_cls_shared_ptrs()[0][5]
        // arg6: uint32_t D_addr_hi get_cls_shared_ptrs()[0][6]
        // arg7: uint32_t D_addr_lo get_cls_shared_ptrs()[0][7]
        // these address are all in L1! It's the TCDM local address.

        // arg8: uint32_t M get_cls_shared_ptrs()[0][8]
        // arg9: uint32_t K get_cls_shared_ptrs()[0][9]
        // arg10: uint32_t N get_cls_shared_ptrs()[0][10]
        // arg11: uint32_t array_shape_idx get_cls_shared_ptrs()[0][11]
        // arg12: transpose A get_cls_shared_ptrs()[0][12]
        // arg13: transpose B get_cls_shared_ptrs()[0][13]
        // arg14: get_cls_shared_ptrs()[0][14] is used for accumPrevC

        // get_cls_shared_ptrs()[0][15] is used for addNonZeroC
        // get_cls_shared_ptrs()[0][16] is used for meshRow
        // get_cls_shared_ptrs()[0][17] is used for tileSize
        // get_cls_shared_ptrs()[0][18] is used for meshCol
        // get_cls_shared_ptrs()[0][19] is used for load_A
        // get_cls_shared_ptrs()[0][20] is used for load_B
        // get_cls_shared_ptrs()[0][21] is used for load_C
        // get_cls_shared_ptrs()[0][22] is used for store_D
        // get_cls_shared_ptrs()[0][23] is used for local_A_addr
        // get_cls_shared_ptrs()[0][24] is used for local_B_addr
        // get_cls_shared_ptrs()[0][25] is used for local_C_addr
        // get_cls_shared_ptrs()[0][26] is used for local_D_addr

        if (accumPrevC) {
            arg_ptr[15] = 0;
        } else if (make_u64(C_addr_hi, C_addr_lo) != 0) {
            arg_ptr[15] = 1;
        } else {
            arg_ptr[15] = 0;
        }

        // get_cls_shared_ptrs()[0][16] is used for meshRow
        // get_cls_shared_ptrs()[0][17] is used for tileSize
        // get_cls_shared_ptrs()[0][18] is used for meshCol
        if (array_shape_idx == 0) {
            arg_ptr[16] = meshRow_0;
            arg_ptr[17] = tileSize_0;
            arg_ptr[18] = meshCol_0;
        } else if (array_shape_idx == 1) {
            arg_ptr[16] = meshRow_1;
            arg_ptr[17] = tileSize_1;
            arg_ptr[18] = meshCol_1;
        } else if (array_shape_idx == 2) {
            arg_ptr[16] = meshRow_2;
            arg_ptr[17] = tileSize_2;
            arg_ptr[18] = meshCol_2;
        } else if (array_shape_idx == 3) {
            arg_ptr[16] = meshRow_3;
            arg_ptr[17] = tileSize_3;
            arg_ptr[18] = meshCol_3;
        } else if (array_shape_idx == 4) {
            arg_ptr[16] = meshRow_4;
            arg_ptr[17] = tileSize_4;
            arg_ptr[18] = meshCol_4;
        } else {
            VERSACORE_DEBUG_PRINT("ERROR: array_shape_idx invalid!\r\n");
            return;
        }
        uint32_t addNonZeroC = arg_ptr[15];
        uint32_t meshRow = arg_ptr[16];
        uint32_t tileSize = arg_ptr[17];
        uint32_t meshCol = arg_ptr[18];

        // We need to determine whether we need to use the dma to load the data from the other cores or 
        // the data has already been loaded into L1 by the other cores
        // If the addr is in L1 range, we assume the data is already in L1
        // Otherwise we need to use the DMA to load the data from the specified addr
        bool load_A = !((A_addr_lo > snrt_l1_start_addr()) && (A_addr_lo < snrt_l1_end_addr()));
        bool load_B = !((B_addr_lo > snrt_l1_start_addr()) && (B_addr_lo < snrt_l1_end_addr()));
        bool load_C = (!((C_addr_lo > snrt_l1_start_addr()) && (C_addr_lo < snrt_l1_end_addr()))) && (addNonZeroC == 1);
        // If D addr is in L3, we need to allocate the L1 memory for D and then store it back using DMA
        // If D addr is in L1, it means the host is allocated the L1 already, we do not need L1 alloc and the DMA
        bool store_D = !((D_addr_lo > snrt_l1_start_addr()) && (D_addr_lo < snrt_l1_end_addr()));
        arg_ptr[19] = load_A ? 1 : 0;
        arg_ptr[20] = load_B ? 1 : 0;
        arg_ptr[21] = load_C ? 1 : 0;
        arg_ptr[22] = store_D ? 1 : 0;
        VERSACORE_DEBUG_PRINT("load_A: %d, load_B: %d, load_C: %d, store_D: %d\r\n", load_A, load_B, load_C, store_D);
        if(load_A) {
            // Load A to the local L1
            uint64_t src_A_addr = make_u64(A_addr_hi, A_addr_lo);
            // Size A = M * K * meshRow * tileSize * sizeof(uint8_t)
            uint32_t A_size = M * K * meshRow * tileSize * sizeof(uint8_t);
            // Allocate L1 memory for A
            get_cls_shared_ptrs()[1] = (uint32_t*)snrt_l1_malloc(A_size);
            if (!get_cls_shared_ptrs()[1]) {
                VERSACORE_DEBUG_PRINT("ERROR: A alloc failed!\r\n");
                return;
            } else {
                VERSACORE_DEBUG_PRINT("Allocated A at %p\r\n",
                                      (void*)get_cls_shared_ptrs()[1]);
            }
            VERSACORE_DEBUG_PRINT("DMA Load A Src  0x%08x%08x\r\n", A_addr_hi, A_addr_lo);
            VERSACORE_DEBUG_PRINT("DMA Load A Dst  0x%lx\r\n", chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[1]));
            VERSACORE_DEBUG_PRINT("DMA Load A Size %d\r\n", A_size);
            // Dst is L1 addr, SRC is specified from the args
            snrt_dma_start_1d_wideptr(chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[1]),
                                     src_A_addr,
                                     A_size);
            snrt_dma_wait_all();
        }
        if(load_B){
            // Load B to the local L1
            uint64_t src_B_addr = make_u64(B_addr_hi, B_addr_lo);
            // Size B = K * N * tileSize * meshCol * sizeof(uint8_t)
            uint32_t B_size = K * N * tileSize * meshCol * sizeof(uint8_t);
            // Allocate L1 memory for B
            get_cls_shared_ptrs()[2] = (uint32_t*)snrt_l1_malloc(B_size);
            if (!get_cls_shared_ptrs()[2]) {
                VERSACORE_DEBUG_PRINT("ERROR: B alloc failed!\r\n");
                return;
            } else {
                VERSACORE_DEBUG_PRINT("Allocated B at %p\r\n",
                                      (void*)get_cls_shared_ptrs()[2]);
            }
            // Dst is L1 addr, SRC is specified from the args
            VERSACORE_DEBUG_PRINT("DMA Load B Src  0x%08x%08x\r\n", B_addr_hi, B_addr_lo);
            VERSACORE_DEBUG_PRINT("DMA Load B Dst  0x%lx\r\n", chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[2]));
            VERSACORE_DEBUG_PRINT("DMA Load B Size %d\r\n", B_size);
            snrt_dma_start_1d_wideptr(chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[2]),
                                     src_B_addr,
                                     B_size);
            snrt_dma_wait_all();
        }
        if(load_C){
            // Load C to the local L1
            uint64_t src_C_addr = make_u64(C_addr_hi, C_addr_lo);
            // Size C = M * N * meshRow * meshCol * sizeof(uint32_t)
            uint32_t C_size = M * N * meshRow * meshCol * sizeof(uint32_t);
            // Allocate L1 memory for C
            get_cls_shared_ptrs()[3] = (uint32_t*)snrt_l1_malloc(C_size);
            if (!get_cls_shared_ptrs()[3]) {
                VERSACORE_DEBUG_PRINT("ERROR: C alloc failed!\r\n");
                return;
            } else {
                VERSACORE_DEBUG_PRINT("Allocated C at %p\r\n",
                                      (void*)get_cls_shared_ptrs()[3]);
            }
            VERSACORE_DEBUG_PRINT("DMA Load C Src  0x%08x%08x\r\n", C_addr_hi, C_addr_lo);
            VERSACORE_DEBUG_PRINT("DMA Load C Dst  0x%lx\r\n", chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[3]));
            VERSACORE_DEBUG_PRINT("DMA Load C Size %d\r\n", C_size);
            // Dst is L1 addr, SRC is specified from the args
            snrt_dma_start_1d_wideptr(chiplet_addr_transform((uint64_t)get_cls_shared_ptrs()[3]),
                                     src_C_addr,
                                     C_size);
            snrt_dma_wait_all();
        }
        // Allocate the L1 memory for output D
        // If the D addr is not in L1, we need to allocate the L1 memory for D and then use the DMA to store according to the inputs
        // Otherwise the D addr is directly used
        if(store_D){
            // Allocate L1 memory for D
            uint32_t D_size = M * N * meshRow * meshCol * sizeof(uint32_t);
            get_cls_shared_ptrs()[4] = (uint32_t*)snrt_l1_malloc(D_size);
            if (!get_cls_shared_ptrs()[4]) {
                VERSACORE_DEBUG_PRINT("ERROR: D alloc failed!\r\n");
                return;
            } else {
                VERSACORE_DEBUG_PRINT("Allocated D at %p\r\n",
                                      (void*)get_cls_shared_ptrs()[4]);
            }
        }

        // Update the local A,B,C,D addr pointers in the shared ptrs
        get_cls_shared_ptrs()[0][23] = load_A? (uint32_t)get_cls_shared_ptrs()[1] : A_addr_lo;
        get_cls_shared_ptrs()[0][24] = load_B? (uint32_t)get_cls_shared_ptrs()[2] : B_addr_lo;
        get_cls_shared_ptrs()[0][25] = load_C? (uint32_t)get_cls_shared_ptrs()[3] : C_addr_lo;
        get_cls_shared_ptrs()[0][26] = store_D? (uint32_t)get_cls_shared_ptrs()[4] : D_addr_lo;

    }
    snrt_cluster_hw_barrier();

    // Here all the cores can access the args from L1
    // Specifically the compute core will configure the streamer and versacore accoring to the following shared ptrs
    uint32_t* arg_ptr = get_cls_shared_ptrs()[0]; 
    uint32_t M = arg_ptr[8];
    uint32_t K = arg_ptr[9];
    uint32_t N = arg_ptr[10];
    uint32_t array_shape_idx = arg_ptr[11];
    uint32_t transpose_A = arg_ptr[12];
    uint32_t transpose_B = arg_ptr[13];
    uint32_t accumPrevC = arg_ptr[14];

    // some inferenced args
    uint32_t addNonZeroC = arg_ptr[15];

    uint32_t meshRow = arg_ptr[16];
    uint32_t tileSize = arg_ptr[17];
    uint32_t meshCol = arg_ptr[18];

    uint32_t load_A = arg_ptr[19];
    uint32_t load_B = arg_ptr[20];
    uint32_t load_C = arg_ptr[21];
    uint32_t store_D = arg_ptr[22];
    uint32_t local_A_addr = arg_ptr[23];
    uint32_t local_B_addr = arg_ptr[24];
    uint32_t local_C_addr = arg_ptr[25];
    uint32_t local_D_addr = arg_ptr[26];
    // The core 0 will be responsible for configuring and starting the versacore
    if (snrt_cluster_core_idx() == 0) {
        // compute the streamer cfg first
        //////////////////////////////////////////////////////////////
        // Allocated the L1 memory for Streamer CSRs    //
        //////////////////////////////////////////////////////////////

        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg Start!\r\n");
        // the snicth needs to generate these streamer cfgs
        // 22 args in total
        // A 6, B 6, C 5, D 5
        get_cls_shared_ptrs()[5] = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 22);
        if (!get_cls_shared_ptrs()[5]) {
            VERSACORE_DEBUG_PRINT("ERROR: streamer cfg alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated streamer cfg at %p\r\n",
                                  (void*)get_cls_shared_ptrs()[5]);
        }
        //////////////////////////////////////////////////////////////
        // streamer cfg for A
        //////////////////////////////////////////////////////////////

        // Aslstride0
        get_cls_shared_ptrs()[5][0] = bankWidth / 8;

        // Atlbound0~5
        uint32_t* Atlbound = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 6);
        if (!Atlbound) {
            VERSACORE_DEBUG_PRINT("ERROR: Atlbound alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated Atlbound at %p\r\n",
                                  (void*)Atlbound);
        }
        // Atlbound0
        Atlbound[0] = K;
        // Atlbound1
        Atlbound[1] = N;
        // Atlbound2
        Atlbound[2] = M;
        // Atlbound3
        Atlbound[3] = 1;
        // Atlbound4
        Atlbound[4] = 1;
        // Atlbound5
        Atlbound[5] = 1;
        get_cls_shared_ptrs()[5][1] = (uint32_t)(uintptr_t)Atlbound;

        // Atlstride0~5
        uint32_t* Atlstride = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 6);
        if (!Atlstride) {
            VERSACORE_DEBUG_PRINT("ERROR: Atlstride alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated Atlstride at %p\r\n",
                                  (void*)Atlstride);
        }
        // Atlstride0
        Atlstride[0] = meshRow * tileSize;
        // Atlstride1
        Atlstride[1] = 0;
        // Atlstride2
        Atlstride[2] = meshRow * tileSize * K;
        // Atlstride3
        Atlstride[3] = 0;
        // Atlstride4
        Atlstride[4] = 0;
        Atlstride[5] = 0;
        get_cls_shared_ptrs()[5][2] = (uint32_t)(uintptr_t)Atlstride;

        // set_addr_remap_index_A
        get_cls_shared_ptrs()[5][3] = 0;
        // transpose_A
        get_cls_shared_ptrs()[5][4] = transpose_A;
        // channel_en_A []
        if (array_shape_idx == 0) {
            get_cls_shared_ptrs()[5][5] = channel_en_A_0_0;
        } else if (array_shape_idx == 1) {
            get_cls_shared_ptrs()[5][5] = channel_en_A_1_0;
        } else if (array_shape_idx == 2) {
            get_cls_shared_ptrs()[5][5] = channel_en_A_2_0;
        } else if (array_shape_idx == 3) {
            get_cls_shared_ptrs()[5][5] = channel_en_A_3_0;
        } else if (array_shape_idx == 4) {
            get_cls_shared_ptrs()[5][5] = channel_en_A_4_0;
        }
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg A Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for B
        //////////////////////////////////////////////////////////////

        // Bslstride0
        get_cls_shared_ptrs()[5][6] = bankWidth / 8;

        // Btlbound0~2
        uint32_t* Btlbound = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 3);
        if (!Btlbound) {
            VERSACORE_DEBUG_PRINT("ERROR: Btlbound alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated Btlbound at %p\r\n",
                                  (void*)Btlbound);
        }
        get_cls_shared_ptrs()[5][7] = (uint32_t)(uintptr_t)Btlbound;
        // Btlbound0
        Btlbound[0] = K;
        // Btlbound1
        Btlbound[1] = N;
        // Btlbound2
        Btlbound[2] = M;

        // Btlstride0~2
        uint32_t* Btlstride = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 3);
        if (!Btlstride) {
            VERSACORE_DEBUG_PRINT("ERROR: Btlstride alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated Btlstride at %p\r\n",
                                  (void*)Btlstride);
        }
        get_cls_shared_ptrs()[5][8] = (uint32_t)(uintptr_t)Btlstride;

        // Btlstride0
        Btlstride[0] = tileSize * meshCol;
        // Btlstride1
        Btlstride[1] = tileSize * meshCol * K;

        // Btlstride2
        Btlstride[2] = 0;

        // set_addr_remap_index_B
        get_cls_shared_ptrs()[5][9] = 0;
        // transpose_B
        get_cls_shared_ptrs()[5][10] = transpose_B;
        // channel_en_B []
        uint32_t* channel_en_B = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 2);
        if (!channel_en_B) {
            VERSACORE_DEBUG_PRINT("ERROR: channel_en_B alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated channel_en_B at %p\r\n",
                                  (void*)channel_en_B);
        }
        if (array_shape_idx == 0) {
            channel_en_B[0] = channel_en_B_0_0;
            channel_en_B[1] = channel_en_B_0_1;
        } else if (array_shape_idx == 1) {
            channel_en_B[0] = channel_en_B_1_0;
            channel_en_B[1] = channel_en_B_1_1;
        } else if (array_shape_idx == 2) {
            channel_en_B[0] = channel_en_B_2_0;
            channel_en_B[1] = channel_en_B_2_1;
        } else if (array_shape_idx == 3) {
            channel_en_B[0] = channel_en_B_3_0;
            channel_en_B[1] = channel_en_B_3_1;
        } else if (array_shape_idx == 4) {
            channel_en_B[0] = channel_en_B_4_0;
            channel_en_B[1] = channel_en_B_4_1;
        }
        get_cls_shared_ptrs()[5][11] = (uint32_t)(uintptr_t)channel_en_B;
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg B Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for C
        //////////////////////////////////////////////////////////////
        // Cslstride0
        get_cls_shared_ptrs()[5][12] = bankWidth / 8;
        // Ctlbound0~3
        uint32_t* Ctlbound = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!Ctlbound) {
            VERSACORE_DEBUG_PRINT("ERROR: Ctlbound alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated Ctlbound at %p\r\n",
                                  (void*)Ctlbound);
        }
        get_cls_shared_ptrs()[5][13] = (uint32_t)(uintptr_t)Ctlbound;
        if (accumPrevC == 1) {
            // accumPrevC is true
            Ctlbound[0] = 0;
        } else if (array_shape_idx == 0) { // else bound is normal
            Ctlbound[0] = Ctlbound0_0;
        } else if (array_shape_idx == 1) {
            Ctlbound[0] = Ctlbound0_1;
        } else if (array_shape_idx == 2) {
            Ctlbound[0] = Ctlbound0_2;
        } else if (array_shape_idx == 3) {
            Ctlbound[0] = Ctlbound0_3;
        } else if (array_shape_idx == 4) {
            Ctlbound[0] = Ctlbound0_4;
        }
        Ctlbound[1] = N;
        Ctlbound[2] = M;
        Ctlbound[3] = 1;
        // Ctlstride0~3
        uint32_t* Ctlstride = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!Ctlstride) {
            VERSACORE_DEBUG_PRINT("ERROR: Ctlstride alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated Ctlstride at %p\r\n",
                                  (void*)Ctlstride);
        }
        get_cls_shared_ptrs()[5][14] = (uint32_t)(uintptr_t)Ctlstride;

        if (array_shape_idx == 0) {
            // Ctlstride0
            Ctlstride[0] = Ctlstride0_0;
        } else if (array_shape_idx == 1) {
            // Ctlstride0
            Ctlstride[0] = Ctlstride0_1;
        } else if (array_shape_idx == 2) {
            // Ctlstride0
            Ctlstride[0] = Ctlstride0_2;
        } else if (array_shape_idx == 3) {
            // Ctlstride0
            Ctlstride[0] = Ctlstride0_3;
        } else if (array_shape_idx == 4) {
            // Ctlstride0
            Ctlstride[0] = Ctlstride0_4;
        }
        // Ctlstride1
        Ctlstride[1] = C_elem_len * meshRow *
                       meshCol / 8;
        // Ctlstride2
        Ctlstride[2] = N * C_elem_len *
                       meshRow *
                       meshCol / 8;
        // Ctlstride3
        Ctlstride[3] = 0;
        // set_addr_remap_index_C
        get_cls_shared_ptrs()[5][15] = 0;
        // channel_en_C []
        // set channel_en_C to zero if accumPrevC or addNonZeroC
        if (accumPrevC == 1 || addNonZeroC == 0) {
            get_cls_shared_ptrs()[5][16] = channel_en_C_null_0_0;
        } else if (array_shape_idx == 0) {
            get_cls_shared_ptrs()[5][16] = channel_en_C_0_0;
        } else if (array_shape_idx == 1) {
            get_cls_shared_ptrs()[5][16] = channel_en_C_1_0;
        } else if (array_shape_idx == 2) {
            get_cls_shared_ptrs()[5][16] = channel_en_C_2_0;
        } else if (array_shape_idx == 3) {
            get_cls_shared_ptrs()[5][16] = channel_en_C_3_0;
        } else if (array_shape_idx == 4) {
            get_cls_shared_ptrs()[5][16] = channel_en_C_4_0;
        }
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg C Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for D
        //////////////////////////////////////////////////////////////
        // D32slstride0
        get_cls_shared_ptrs()[5][17] = bankWidth / 8;
        // D32tlbound0~3
        uint32_t* D32tlbound = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!D32tlbound) {
            VERSACORE_DEBUG_PRINT("ERROR: D32tlbound alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated D32tlbound at %p\r\n",
                                  (void*)D32tlbound);
        }
        get_cls_shared_ptrs()[5][18] = (uint32_t)(uintptr_t)D32tlbound;
        // D32tlbound0~3
        if (array_shape_idx == 0) {
            D32tlbound[0] = D32tlbound0_0;
        } else if (array_shape_idx == 1) {
            D32tlbound[0] = D32tlbound0_1;
        } else if (array_shape_idx == 2) {
            D32tlbound[0] = D32tlbound0_2;
        } else if (array_shape_idx == 3) {
            D32tlbound[0] = D32tlbound0_3;
        } else if (array_shape_idx == 4) {
            D32tlbound[0] = D32tlbound0_4;
        }
        D32tlbound[1] = N;
        D32tlbound[2] = M;
        D32tlbound[3] = 1;
        // D32tlstride0~3
        uint32_t* D32tlstride = (uint32_t*)snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!D32tlstride) {
            VERSACORE_DEBUG_PRINT("ERROR: D32tlstride alloc failed!\r\n");
            return;
        } else {
            VERSACORE_DEBUG_PRINT("Allocated D32tlstride at %p\r\n",
                                  (void*)D32tlstride);
        }
        get_cls_shared_ptrs()[5][19] = (uint32_t)(uintptr_t)D32tlstride;
        if (array_shape_idx == 0) {
            // D32tlstride0
            D32tlstride[0] = D32tlstride0_0;
        } else if (array_shape_idx == 1) {
            // D32tlstride0
            D32tlstride[0] = D32tlstride0_1;
        } else if (array_shape_idx == 2) {
            // D32tlstride0
            D32tlstride[0] = D32tlstride0_2;
        } else if (array_shape_idx == 3) {
            // D32tlstride0
            D32tlstride[0] = D32tlstride0_3;
        } else if (array_shape_idx == 4) {
            // D32tlstride0
            D32tlstride[0] = D32tlstride0_4;
        }
        // D32tlstride1
        D32tlstride[1] = D32_elem_len * meshRow *
                         meshCol / 8;
        // D32tlstride2
        D32tlstride[2] = N * D32_elem_len *
                         meshRow *
                         meshCol / 8;
        // D32tlstride3
        D32tlstride[3] = 0;
        // set_addr_remap_index_D32
        get_cls_shared_ptrs()[5][20] = 0;
        // channel_en_D32 []
        if (array_shape_idx == 0) {
            get_cls_shared_ptrs()[5][21] = channel_en_D32_0_0;
        } else if (array_shape_idx == 1) {
            get_cls_shared_ptrs()[5][21] = channel_en_D32_1_0;
        } else if (array_shape_idx == 2) {
            get_cls_shared_ptrs()[5][21] = channel_en_D32_2_0;
        } else if (array_shape_idx == 3) {
            get_cls_shared_ptrs()[5][21] = channel_en_D32_3_0;
        } else if (array_shape_idx == 4) {
            get_cls_shared_ptrs()[5][21] = channel_en_D32_4_0;
        }
        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Compute Streamer Cfg D Done!\r\n");

        //////////////////////////////////////////////////////////////
        // Configuration the Steamer and Versacore
        //////////////////////////////////////////////////////////////

        set_versacore_streamer_csr(
            local_A_addr,  // A_addr
            (uint32_t*)(&get_cls_shared_ptrs()[5][0]),         // Aslstride
            (uint32_t*)(get_cls_shared_ptrs()[5][1]),  // Atlbound[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][2]),  // Atlstride[] base
            get_cls_shared_ptrs()[5][3],               // set_addr_remap_index_A
            get_cls_shared_ptrs()[5][4],               // transpose_A
            (uint32_t*)(&get_cls_shared_ptrs()[5][5]),  // channel_en_A []

            local_B_addr,  // B_addr
            (uint32_t*)(&get_cls_shared_ptrs()[5][6]),  // Bslstride[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][7]),   // Btlbound[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][8]),   // Btlstride[] base
            get_cls_shared_ptrs()[5][9],   // set_addr_remap_index_B
            get_cls_shared_ptrs()[5][10],  // transpose_B
            (uint32_t*)(get_cls_shared_ptrs()[5][11]),  // channel_en_B []

            local_C_addr,  // C_addr
            (uint32_t*)(&get_cls_shared_ptrs()[5][12]),  // Cslstride[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][13]),   // Ctlbound[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][14]),   // Ctlstride[] base
            get_cls_shared_ptrs()[5][15],  // set_addr_remap_index_C
            (uint32_t*)(&get_cls_shared_ptrs()[5][16]),  // channel_en_C []

            local_D_addr,  // D_addr
            (uint32_t*)(&get_cls_shared_ptrs()[5][17]),  // D32slstride[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][18]),   // D32tlbound[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][19]),   // D32tlstride[] base
            get_cls_shared_ptrs()[5][20],  // set_addr_remap_index_D32
            (uint32_t*)(&get_cls_shared_ptrs()[5][21]),  // channel_en_D32 []

            0, 0, 0, 0, 0, 0, 0, 0, 0
        );

        set_versacore_csr(
            // accPrevC means takes new C
            accumPrevC == 0,
            K,
            N*M,
            0,
            array_shape_idx,
            0);

        VERSACORE_DEBUG_PRINT(
            "GEMM Intra-Chiplet Kernel Streamer Configuration Done!\r\n");

        // Set CSR to start Streamer
        start_versacore_and_streamer();

        // Poll until Streamer and GEMM accelerator finish
        wait_versacore_and_streamer();

        VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel Compute Done!\r\n");

        for (int i = 0; i < 2; i++) {
            VERSACORE_DEBUG_PRINT("D[%d] = %d\r\n", i,
                                  *((uint32_t*)local_D_addr + i));
        }
    }

    snrt_cluster_hw_barrier();
    
    if (snrt_is_dm_core()) {
        // If the D addr is not in L1, we need to use the DMA to store the data back according to the input args
        if(store_D){
            uint32_t D_addr_hi = get_cls_shared_ptrs()[0][6];
            uint32_t D_addr_lo = get_cls_shared_ptrs()[0][7];
            uint64_t dst_D_addr = make_u64(D_addr_hi, D_addr_lo);
            uint32_t local_D_addr = get_cls_shared_ptrs()[0][26];

            uint32_t M = get_cls_shared_ptrs()[0][8];
            uint32_t N = get_cls_shared_ptrs()[0][10];
            uint32_t meshRow = get_cls_shared_ptrs()[0][16];
            uint32_t meshCol = get_cls_shared_ptrs()[0][18];
            // Size D = M * N * meshRow * meshCol * sizeof(uint32_t)
            uint32_t D_size = M *
                              N *
                              meshRow *
                              meshCol * sizeof(uint32_t);
            VERSACORE_DEBUG_PRINT("DMA Store D Dst addr: 0x%08x%08x\r\n", D_addr_hi, D_addr_lo);
            VERSACORE_DEBUG_PRINT("DMA Store D Src: 0x%lx\r\n", chiplet_addr_transform((uint64_t)local_D_addr));
            VERSACORE_DEBUG_PRINT("DMA Store D Size %d\r\n", D_size);
            // Dst is specified from the args, SRC is L1 addr
            snrt_dma_start_1d_wideptr(dst_D_addr,
                                     chiplet_addr_transform((uint64_t)local_D_addr),
                                     D_size);
            snrt_dma_wait_all();
        }
        ///////////////////////////////////////
        // Free the L1 memory //
        ///////////////////////////////////////
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[0]);
        if(load_A){
            snrt_l1_free((uint32_t)get_cls_shared_ptrs()[1]);
        }
        if(load_B){
            snrt_l1_free((uint32_t)get_cls_shared_ptrs()[2]);
        }
        if(load_C){
            snrt_l1_free((uint32_t)get_cls_shared_ptrs()[3]);
        }
        if(store_D){
            snrt_l1_free((uint32_t)get_cls_shared_ptrs()[4]);
        }
        // Free Atlbound
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][1]);
        // Free Atlstride
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][2]);
        // Free Btlbound
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][7]);
        // Free Btlstride
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][8]);
        // Free channel_en_B
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][11]);
        // Free Ctlbound
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][13]);
        // Free Ctlstride
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][14]);
        // Free D32tlbound
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][18]);
        // Free D32tlstride
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5][19]);
        // Free streamer cfg
        snrt_l1_free((uint32_t)get_cls_shared_ptrs()[5]);
        VERSACORE_DEBUG_PRINT("GEMM Intra-Chiplet Kernel L1 Free Done!\r\n");
    }
}

//////////////////////// SYMBOL TABLE ////////////////////////
// Here we create the symbol table
SNAX_SYMTAB_SECTION const snax_symbol_t __snax_symtab[] = {
    SNAX_EXPORT_FUNC(__snax_kernel_dummy),
    SNAX_EXPORT_FUNC(__snax_kernel_check_results),
    SNAX_EXPORT_FUNC(__snax_kernel_check_results_full),
    SNAX_EXPORT_FUNC(__snax_kernel_csr),
    SNAX_EXPORT_FUNC(__snax_kernel_load_compute_store),
    SNAX_EXPORT_FUNC(__snax_kernel_double_buffer_example),
    SNAX_EXPORT_FUNC(__snax_kernel_xdma_1d_copy),
    SNAX_EXPORT_FUNC(__snax_kernel_idma_1d_copy),
    // SNAX_EXPORT_FUNC(
    //     __snax_kernel_versacore_load_compute_store_w_streamer_args),
    // SNAX_EXPORT_FUNC(__snax_kernel_gemm_intra_chiplet),
    // SNAX_EXPORT_FUNC(__snax_kernel_gemm_compute_only_intra_chiplet),
    SNAX_EXPORT_FUNC(__snax_kernel_gemm),
    SNAX_SYMTAB_END
};
