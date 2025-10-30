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
        get_cls_shared_ptrs()[0] = snrt_l1_malloc(256);
        get_cls_shared_ptrs()[1] = snrt_l1_malloc(4);
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
        for (uint32_t i = 0; i < data_size / sizeof(uint32_t); i++) {
            if (((uint32_t*)golden_data_addr)[i] !=
                ((uint32_t*)output_data_addr)[i]) {
                num_errors++;
                if (num_errors < 10) {
                    printf("Error at index %d: golden = %d, output = %d\n", i,
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

        get_cls_shared_ptrs()[0] = snrt_l1_malloc(256);
        get_cls_shared_ptrs()[0][0] = ((uint32_t*)arg)[0];
        get_cls_shared_ptrs()[0][1] = ((uint32_t*)arg)[1];
        get_cls_shared_ptrs()[0][2] = ((uint32_t*)arg)[2];
        get_cls_shared_ptrs()[0][3] = ((uint32_t*)arg)[3];
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
        snrt_l1_free(get_cls_shared_ptrs()[0]);
        snrt_l1_free(get_cls_shared_ptrs()[1]);
        snrt_l1_free(get_cls_shared_ptrs()[2]);
        printf("[Cluster %d] Core(%d) Free allocated heap memory\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
        snrt_memset(get_cls_shared_ptrs(), 0,
                    sizeof(uint32_t*) * NUM_CLS_SHARED_PTRS);
        printf("[Cluster %d] Core(%d) Free shared pointers\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
    }
}

SNAX_LIB_DEFINE void __snax_kernel_xdma_1d_copy(void* arg) {
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
        src_addr = ((uint64_t)((uint32_t*)arg)[0] << 32) |
                   ((uint64_t)((uint32_t*)arg)[1]);
        dst_addr = ((uint64_t)((uint32_t*)arg)[2] << 32) |
                   ((uint64_t)((uint32_t*)arg)[3]);
        data_size = ((uint32_t*)arg)[4];

        if (xdma_disable_dst_ext(0) != 0) {
            printf("Error in disabling xdma writer extension 0\n");
        }
        if (xdma_disable_dst_ext(1) != 0) {
            printf("Error in disabling xdma writer extension 1\n");
        }
        if (xdma_disable_src_ext(0) != 0) {
            printf("Error in disabling xdma reader extension 0\n");
        }

        xdma_memcpy_1d((void*)src_addr, (void*)dst_addr, data_size);

        int task_id = xdma_start();
        xdma_remote_wait(task_id);
    }
}

//////////////////////// VERSACORE ////////////////////////
SNAX_LIB_DEFINE void __snax_kernel_versacore_load_compute_store_w_streamer_args(
    void* arg) {
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
    uint64_t l3_input_A_addr, l3_input_B_addr, l3_input_C_addr,
        l3_streamer_cfg_addr, l3_versacore_cfg_addr;
    uint32_t input_A_size, input_B_size, input_C_size, streamer_cfg_size,
        versacore_cfg_size;

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
        // First we need to use the DMA to load all the arguments from L3 into
        // L1 Here we assume the arguments are packed in a contiguous way in L3

        snrt_dma_start_1d((void*)get_cls_shared_ptrs()[0], (void*)arg,
                          ((uint32_t*)arg)[17]);
        snrt_dma_wait_all();

        // printf("[Cluster %d] Core(%d) Load Kernel Args: Dst = %x, Src =
        // %x\n", snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[0], arg);

        // printf("[Cluster %d] Core(%d) Kernel Args:\n", snrt_cluster_idx(),
        //        snrt_cluster_core_idx());
        // for (uint32_t i = 0; i < ((uint32_t *)l1_arg_ptr)[17] /
        // sizeof(uint32_t);
        //      i++) {
        //     printf("[Cluster %d] Core(%d) Arg[%d] = %x\n",
        //     snrt_cluster_idx(),
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
        l3_input_A_addr =
            make_u64(get_cls_shared_ptrs()[0][0], get_cls_shared_ptrs()[0][1]);
        input_A_size = (get_cls_shared_ptrs()[0][2]);
        xdma_memcpy_1d((void*)l3_input_A_addr, (void*)get_cls_shared_ptrs()[1],
                       input_A_size);
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("[Cluster %d] Core(%d) Load Input A: Dst = %x, Src = %x\n",
               snrt_cluster_idx(), snrt_cluster_core_idx(),
               get_cls_shared_ptrs()[1], (uint32_t)l3_input_A_addr);
        printf("[Cluster %d] Core(%d) Input A[0] = %d\n", snrt_cluster_idx(),
               snrt_cluster_core_idx(), get_cls_shared_ptrs()[1][0]);
        ///////////////////////////////
        // Call the dma to load B    //
        ///////////////////////////////
        l3_input_B_addr =
            make_u64(get_cls_shared_ptrs()[0][3], get_cls_shared_ptrs()[0][4]);
        input_B_size = (get_cls_shared_ptrs()[0][5]);
        xdma_memcpy_1d((void*)l3_input_B_addr, (void*)get_cls_shared_ptrs()[2],
                       input_B_size);
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        // printf("[Cluster %d] Core(%d) Load Input B: Dst = %x, Src = %x\n",
        // snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[2],
        //        (uint32_t)l3_input_B_addr);
        printf("[Cluster %d] Core(%d) Input B[0] = %d\n", snrt_cluster_idx(),
               snrt_cluster_core_idx(), get_cls_shared_ptrs()[2][0]);
        ///////////////////////////////
        // Call the dma to load C    //
        ///////////////////////////////
        l3_input_C_addr =
            make_u64(get_cls_shared_ptrs()[0][6], get_cls_shared_ptrs()[0][7]);
        input_C_size = (get_cls_shared_ptrs()[0][8]);
        xdma_memcpy_1d((void*)l3_input_C_addr, (void*)get_cls_shared_ptrs()[3],
                       input_C_size);
        task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("[Cluster %d] Core(%d) Input C[0] = %d\n", snrt_cluster_idx(),
               snrt_cluster_core_idx(), get_cls_shared_ptrs()[3][0]);
        // printf("[Cluster %d] Core(%d) Load Input C: Dst = %x, Src = %x\n",
        // snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[3],
        //        (uint32_t)l3_input_C_addr);
        //////////////////////////////////////////////////////////////
        // Allocated the L1 memory for Stream and Versacore CSRs    //
        //////////////////////////////////////////////////////////////
        get_cls_shared_ptrs()[5] =
            snrt_l1_malloc(((uint32_t*)get_cls_shared_ptrs()[0])[13]);
        // l1_streamer_cfg_ptr = get_cls_shared_ptrs()[5];
        get_cls_shared_ptrs()[6] =
            snrt_l1_malloc(((uint32_t*)get_cls_shared_ptrs()[0])[16]);
        // l1_versacore_cfg_ptr = get_cls_shared_ptrs()[6];

        //////////////////////////////////////////
        // Call the dma to load streamer cfg    //
        //////////////////////////////////////////
        l3_streamer_cfg_addr = make_u64(get_cls_shared_ptrs()[0][11],
                                        get_cls_shared_ptrs()[0][12]);
        streamer_cfg_size = (get_cls_shared_ptrs()[0][13]);
        snrt_dma_start_1d((void*)get_cls_shared_ptrs()[5],
                          (void*)l3_streamer_cfg_addr, streamer_cfg_size);
        snrt_dma_wait_all();
        // printf("[Cluster %d] Core(%d) Load Streamer Cfg: Dst = %x, Src =
        // %x\n", snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[5],
        //        l3_streamer_cfg_addr);
        //////////////////////////////////////////
        // Call the dma to load versacore cfg   //
        //////////////////////////////////////////
        l3_versacore_cfg_addr = make_u64(get_cls_shared_ptrs()[0][14],
                                         get_cls_shared_ptrs()[0][15]);
        versacore_cfg_size = (get_cls_shared_ptrs()[0][16]);
        snrt_dma_start_1d((void*)get_cls_shared_ptrs()[6],
                          (void*)l3_versacore_cfg_addr, versacore_cfg_size);
        snrt_dma_wait_all();
        // printf("[Cluster %d] Core(%d) Load Versacore Cfg: Dst = %x, Src =
        // %x\n", snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), get_cls_shared_ptrs()[6],
        //        l3_versacore_cfg_addr);
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
            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[1],  // A_addr
            (uint32_t*)get_cls_shared_ptrs()[5][0],         // Aslstride
            (uint32_t*)get_cls_shared_ptrs()[5][1],         // Atlbound[] base
            (uint32_t*)get_cls_shared_ptrs()[5][2],         // Atlstride[] base
            get_cls_shared_ptrs()[5][3],             // set_addr_remap_index_A
            get_cls_shared_ptrs()[5][4],             // transpose_A
            (uint32_t*)get_cls_shared_ptrs()[5][5],  // channel_en_A []

            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[2],  // B_addr
            (uint32_t*)get_cls_shared_ptrs()[5][6],         // Bslstride[] base
            (uint32_t*)get_cls_shared_ptrs()[5][7],         // Btlbound[] base
            (uint32_t*)get_cls_shared_ptrs()[5][8],         // Btlstride[] base
            get_cls_shared_ptrs()[5][9],              // set_addr_remap_index_B
            get_cls_shared_ptrs()[5][10],             // transpose_B
            (uint32_t*)get_cls_shared_ptrs()[5][11],  // channel_en_B []

            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[3],  // C_addr
            (uint32_t*)get_cls_shared_ptrs()[5][12],        // Cslstride[] base
            (uint32_t*)get_cls_shared_ptrs()[5][13],        // Ctlbound[] base
            (uint32_t*)get_cls_shared_ptrs()[5][14],        // Ctlstride[] base
            get_cls_shared_ptrs()[5][15],             // set_addr_remap_index_C
            (uint32_t*)get_cls_shared_ptrs()[5][16],  // channel_en_C []

            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[4],  // D_addr
            (uint32_t*)get_cls_shared_ptrs()[5][17],  // D32slstride[] base
            (uint32_t*)get_cls_shared_ptrs()[5][18],  // D32tlbound[] base
            (uint32_t*)get_cls_shared_ptrs()[5][19],  // D32tlstride[] base
            get_cls_shared_ptrs()[5][20],            // set_addr_remap_index_D32
            (uint32_t*)get_cls_shared_ptrs()[5][21]  // channel_en_D32 []
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
        //     printf("[Cluster %d] Core(%d) Streamer_cfg[%d] =
        //     %x\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), i, get_cls_shared_ptrs()[5][i]);
        // }

        set_versacore_csr(
            get_cls_shared_ptrs()[6][0],  // tempLoop0
            get_cls_shared_ptrs()[6][1],  // tempLoop1
            get_cls_shared_ptrs()[6][2],  // tempLoop2
            gen_subtraction_config(
                (int8_t)get_cls_shared_ptrs()[6][3],
                (int8_t)get_cls_shared_ptrs()[6][4]),  // subtraction
            get_cls_shared_ptrs()[6][5],               // array_shape
            get_cls_shared_ptrs()[6][6]                // data_type
        );
        // printf("[Cluster %d] Core(%d) Config Versacore\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx());
        // for (int i=0; i<7; i++){
        //     printf("[Cluster %d] Core(%d) Versacore_cfg[%d] =
        //     %x\n",snrt_cluster_idx(),
        //        snrt_cluster_core_idx(), i, get_cls_shared_ptrs()[6][i]);
        // }
        ////////////////////////////////////
        // Start the Streamer and Versacore
        ////////////////////////////////////
        start_versacore_and_streamer();
        printf("[Cluster %d] Core(%d) Start Versacore\n", snrt_cluster_idx(),
               snrt_cluster_core_idx());
        // Wait for the Versacore to finish
        wait_versacore_and_streamer();
    }

    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        // Send the output data back to L3
        uint64_t l3_output_addr =
            make_u64(get_cls_shared_ptrs()[0][9], get_cls_shared_ptrs()[0][10]);
        uint32_t output_size = get_cls_shared_ptrs()[0][8];
        // D is the temporary output buffer in L1
        printf("[Cluster %d] Core(%d) Quick check on D[0] = %d\n",
               snrt_cluster_idx(), snrt_cluster_core_idx(),
               get_cls_shared_ptrs()[4][0]);
        xdma_memcpy_1d((void*)get_cls_shared_ptrs()[4], (void*)l3_output_addr,
                       output_size);
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        printf("[Cluster %d] Core(%d) Store Output: Dst = %x, Src = %x\n",
               snrt_cluster_idx(), snrt_cluster_core_idx(),
               get_cls_shared_ptrs()[4], (uint32_t)l3_output_addr);
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
                    sizeof(uint32_t*) * NUM_CLS_SHARED_PTRS);
    }
}

SNAX_LIB_DEFINE void __snax_kernel_gemm_intra_chiplet(void* arg) {
    // move the args to local variables
    if (snrt_is_dm_core()) {
        // allocate L1 memory for global variables across the cluster
        ///////////////////////////////////////
        // Allocate the L1 memory //
        ///////////////////////////////////////
        get_cls_shared_ptrs()[0] = snrt_l1_malloc(128);
        if (!get_cls_shared_ptrs()[0]) {
            printf("ERROR: arg alloc failed!\r\n");
            return;
        } else {
            printf("Allocated arg at %p\r\n", (void*)get_cls_shared_ptrs()[0]);
        }

        ////////////////////////////////////
        // Call the idma to load the args //
        ////////////////////////////////////
        // First we need to use the DMA to load all the arguments from L3 into
        // L1 Here we assume the arguments are packed in a contiguous way in L3
        printf("GEMM Intra-Chiplet Kernel Start!\r\n");

        snrt_dma_start_1d(
            (void*)get_cls_shared_ptrs()[0], (void*)arg,
            sizeof(uint32_t) * 14);  // 14 uint32_t passed args in total
        snrt_dma_wait_all();

        // we use the following args, they are shared across the cluster
        // arg0: uint32_t A_addr_hi get_cls_shared_ptrs()[0][0]
        // arg1: uint32_t A_addr_lo get_cls_shared_ptrs()[0][1]
        // arg2: uint32_t B_addr_hi get_cls_shared_ptrs()[0][2]
        // arg3: uint32_t B_addr_lo get_cls_shared_ptrs()[0][3]
        // arg4: uint32_t C_addr_hi get_cls_shared_ptrs()[0][4]
        // arg5: uint32_t C_addr_lo get_cls_shared_ptrs()[0][5]
        // arg6: uint32_t D_addr_hi get_cls_shared_ptrs()[0][6]
        // arg7: uint32_t D_addr_lo get_cls_shared_ptrs()[0][7]
        // arg8: uint32_t M get_cls_shared_ptrs()[0][8]
        // arg9: uint32_t K get_cls_shared_ptrs()[0][9]
        // arg10: uint32_t N get_cls_shared_ptrs()[0][10]
        // arg11: uint32_t array_shape_idx get_cls_shared_ptrs()[0][11]
        // arg12: transpose A get_cls_shared_ptrs()[0][12]
        // arg13: transpose B get_cls_shared_ptrs()[0][13]
        // get_cls_shared_ptrs()[0][14] is used for C exist flag
        // get_cls_shared_ptrs()[0][15] meshRow
        // get_cls_shared_ptrs()[0][16] tileSize
        // get_cls_shared_ptrs()[0][17] meshCol

        printf("GEMM Intra-Chiplet Kernel Load Args Done!\r\n");
        /////////////////////////////////////////
        // Allocate the L1 memory for Inputs   //
        /////////////////////////////////////////
        // Allocate the l1 space specifed by the arg
        uint64_t l3_input_A_addr, l3_input_B_addr, l3_input_C_addr;

        printf("M = %d, K = %d, N = %d, array_shape_idx = %d\r\n",
               get_cls_shared_ptrs()[0][8], get_cls_shared_ptrs()[0][9],
               get_cls_shared_ptrs()[0][10], get_cls_shared_ptrs()[0][11]);
        if (get_cls_shared_ptrs()[0][11] == 0) {
            get_cls_shared_ptrs()[0][15] = meshRow_0;
            get_cls_shared_ptrs()[0][16] = tileSize_0;
            get_cls_shared_ptrs()[0][17] = meshCol_0;
        } else {
            get_cls_shared_ptrs()[0][15] = meshRow_1;
            get_cls_shared_ptrs()[0][16] = tileSize_1;
            get_cls_shared_ptrs()[0][17] = meshCol_1;
        }

        // L1 data addresses for A
        get_cls_shared_ptrs()[1] = snrt_l1_malloc(
            (get_cls_shared_ptrs()[0][8]) * (get_cls_shared_ptrs()[0][9]) *
            (get_cls_shared_ptrs()[0][15]) * (get_cls_shared_ptrs()[0][16]) *
            sizeof(uint8_t));  // A MxK in uint8_t
        if (!get_cls_shared_ptrs()[1]) {
            printf("ERROR: A alloc failed!\r\n");
            return;
        } else {
            printf("Allocated A at %p\r\n", (void*)get_cls_shared_ptrs()[1]);
        }
        // Call the idma to load the inputs
        l3_input_A_addr =
            make_u64(get_cls_shared_ptrs()[0][0], get_cls_shared_ptrs()[0][1]);
        xdma_memcpy_1d((void*)l3_input_A_addr, (void*)get_cls_shared_ptrs()[1],
                       (get_cls_shared_ptrs()[0][8]) *
                           (get_cls_shared_ptrs()[0][9]) *
                           get_cls_shared_ptrs()[0][15] *
                           get_cls_shared_ptrs()[0][16] * sizeof(uint8_t));
        int task_id = xdma_start();
        xdma_remote_wait(task_id);

        // L1 data addresses for B
        get_cls_shared_ptrs()[2] = snrt_l1_malloc(
            (get_cls_shared_ptrs()[0][9]) * (get_cls_shared_ptrs()[0][10]) *
            get_cls_shared_ptrs()[0][16] * get_cls_shared_ptrs()[0][17] *
            sizeof(uint8_t));  // B KxN in uint8_t
        if (!get_cls_shared_ptrs()[2]) {
            printf("ERROR: B alloc failed!\r\n");
            return;
        } else {
            printf("Allocated B at %p\r\n", (void*)get_cls_shared_ptrs()[2]);
        }

        l3_input_B_addr =
            make_u64(get_cls_shared_ptrs()[0][2], get_cls_shared_ptrs()[0][3]);
        xdma_memcpy_1d((void*)l3_input_B_addr, (void*)get_cls_shared_ptrs()[2],
                       (get_cls_shared_ptrs()[0][9]) *
                           (get_cls_shared_ptrs()[0][10]) *
                           get_cls_shared_ptrs()[0][16] *
                           get_cls_shared_ptrs()[0][17] * sizeof(uint8_t));
        task_id = xdma_start();
        xdma_remote_wait(task_id);

        // L1 data addresses for C
        // load C only if C_addr is not 0
        l3_input_C_addr =
            make_u64(get_cls_shared_ptrs()[0][4], get_cls_shared_ptrs()[0][5]);
        if (l3_input_C_addr != 0) {
            get_cls_shared_ptrs()[0][14] = 1;
            get_cls_shared_ptrs()[3] = snrt_l1_malloc(
                (get_cls_shared_ptrs()[0][8]) * (get_cls_shared_ptrs()[0][10]) *
                get_cls_shared_ptrs()[0][15] * get_cls_shared_ptrs()[0][17] *
                sizeof(uint32_t));  // C MxN in uint32
            if (!get_cls_shared_ptrs()[3]) {
                printf("ERROR: C alloc failed!\r\n");
                return;
            } else {
                printf("Allocated C at %p\r\n",
                       (void*)get_cls_shared_ptrs()[3]);
            }
            xdma_memcpy_1d(
                (void*)l3_input_C_addr, (void*)get_cls_shared_ptrs()[3],
                (get_cls_shared_ptrs()[0][8]) * (get_cls_shared_ptrs()[0][10]) *
                    get_cls_shared_ptrs()[0][15] *
                    get_cls_shared_ptrs()[0][17] * sizeof(uint32_t));
            task_id = xdma_start();
            xdma_remote_wait(task_id);
        } else {
            get_cls_shared_ptrs()[0][14] = 0;
            printf("skip loading C since C_addr is 0\r\n");
        }

        // L1 data addresses for D
        get_cls_shared_ptrs()[4] = snrt_l1_malloc(
            (get_cls_shared_ptrs()[0][8]) * (get_cls_shared_ptrs()[0][10]) *
            get_cls_shared_ptrs()[0][15] * get_cls_shared_ptrs()[0][17] *
            sizeof(uint32_t));  // D MxN in uint32
        if (!get_cls_shared_ptrs()[4]) {
            printf("ERROR: D alloc failed!\r\n");
            return;
        } else {
            printf("Allocated D at %p\r\n", (void*)get_cls_shared_ptrs()[4]);
        }

        printf("GEMM Intra-Chiplet Kernel Load Input Data Done!\r\n");
    }

    snrt_cluster_hw_barrier();

    // The core 0 will be responsible for configuring and starting the versacore
    if (snrt_cluster_core_idx() == 0) {
        // compute the streamer cfg first
        //////////////////////////////////////////////////////////////
        // Allocated the L1 memory for Streamer CSRs    //
        //////////////////////////////////////////////////////////////

        printf("GEMM Intra-Chiplet Kernel Compute Streamer Cfg Start!\r\n");
        // the snicth needs to generate these streamer cfgs
        // 22 args in total
        // A 6, B 6, C 5, D 5
        get_cls_shared_ptrs()[5] = snrt_l1_malloc(sizeof(uint32_t) * 22);
        if (!get_cls_shared_ptrs()[5]) {
            printf("ERROR: streamer cfg alloc failed!\r\n");
            return;
        } else {
            printf("Allocated streamer cfg at %p\r\n",
                   (void*)get_cls_shared_ptrs()[5]);
        }
        //////////////////////////////////////////////////////////////
        // streamer cfg for A
        //////////////////////////////////////////////////////////////

        // Aslstride0
        get_cls_shared_ptrs()[5][0] = bankWidth / 8;

        // Atlbound0~5
        uint32_t* Atlbound = snrt_l1_malloc(sizeof(uint32_t) * 6);
        if (!Atlbound) {
            printf("ERROR: Atlbound alloc failed!\r\n");
            return;
        } else {
            printf("Allocated Atlbound at %p\r\n", (void*)Atlbound);
        }
        // Atlbound0
        Atlbound[0] = (get_cls_shared_ptrs()[0][9]);
        // Atlbound1
        Atlbound[1] = (get_cls_shared_ptrs()[0][10]);
        // Atlbound2
        Atlbound[2] = (get_cls_shared_ptrs()[0][8]);
        // Atlbound3
        Atlbound[3] = 1;
        // Atlbound4
        Atlbound[4] = 1;
        // Atlbound5
        Atlbound[5] = 1;
        get_cls_shared_ptrs()[5][1] = (uint32_t)(uintptr_t)Atlbound;
        printf("Atlbound[0]: %d\r\n",
               ((uint32_t*)get_cls_shared_ptrs()[5][1])[0]);
        printf("Atlbound[1]: %d\r\n",
               ((uint32_t*)get_cls_shared_ptrs()[5][1])[1]);
        printf("Atlbound[2]: %d\r\n",
               ((uint32_t*)get_cls_shared_ptrs()[5][1])[2]);
        printf("Atlbound[3]: %d\r\n",
               ((uint32_t*)get_cls_shared_ptrs()[5][1])[3]);
        printf("Atlbound[4]: %d\r\n",
               ((uint32_t*)get_cls_shared_ptrs()[5][1])[4]);
        printf("Atlbound[5]: %d\r\n",
               ((uint32_t*)get_cls_shared_ptrs()[5][1])[5]);

        // Atlstride0~5
        uint32_t* Atlstride = snrt_l1_malloc(sizeof(uint32_t) * 6);
        if (!Atlstride) {
            printf("ERROR: Atlstride alloc failed!\r\n");
            return;
        } else {
            printf("Allocated Atlstride at %p\r\n", (void*)Atlstride);
        }
        // Atlstride0
        Atlstride[0] = get_cls_shared_ptrs()[0][15] * get_cls_shared_ptrs()[0][16];
        // Atlstride1
        Atlstride[1] = 0;
        // Atlstride2
        Atlstride[2] = get_cls_shared_ptrs()[0][15] * get_cls_shared_ptrs()[0][16] * (get_cls_shared_ptrs()[0][9]);
        // Atlstride3
        Atlstride[3] = 0;
        // Atlstride4
        Atlstride[4] = 0;
        get_cls_shared_ptrs()[5][2] = (uint32_t)(uintptr_t)Atlstride;

        // set_addr_remap_index_A
        get_cls_shared_ptrs()[5][3] = 0;
        // transpose_A
        get_cls_shared_ptrs()[5][4] = get_cls_shared_ptrs()[0][12];
        // channel_en_A []
        if (get_cls_shared_ptrs()[0][11] == 0) {
            get_cls_shared_ptrs()[5][5] = chanelEnA_0;
        } else {
            get_cls_shared_ptrs()[5][5] = chanelEnA_1;
        }
        printf("GEMM Intra-Chiplet Kernel Compute Streamer Cfg A Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for B
        //////////////////////////////////////////////////////////////

        // Bslstride0
        get_cls_shared_ptrs()[5][6] = bankWidth / 8;

        // Btlbound0~2
        uint32_t* Btlbound = snrt_l1_malloc(sizeof(uint32_t) * 3);
        if (!Btlbound) {
            printf("ERROR: Btlbound alloc failed!\r\n");
            return;
        } else {
            printf("Allocated Btlbound at %p\r\n", (void*)Btlbound);
        }
        get_cls_shared_ptrs()[5][7] = (uint32_t)(uintptr_t)Btlbound;
        // Btlbound0
        Btlbound[0] = (get_cls_shared_ptrs()[0][9]);
        // Btlbound1
        Btlbound[1] = (get_cls_shared_ptrs()[0][10]);
        // Btlbound2
        Btlbound[2] = (get_cls_shared_ptrs()[0][8]);

        // Btlstride0~2
        uint32_t* Btlstride = snrt_l1_malloc(sizeof(uint32_t) * 3);
        if (!Btlstride) {
            printf("ERROR: Btlstride alloc failed!\r\n");
            return;
        } else {
            printf("Allocated Btlstride at %p\r\n", (void*)Btlstride);
        }
        get_cls_shared_ptrs()[5][8] = (uint32_t)(uintptr_t)Btlstride;

        // Btlstride0
        Btlstride[0] = get_cls_shared_ptrs()[0][17] * get_cls_shared_ptrs()[0][16];
        // Btlstride1
        Btlstride[1] = get_cls_shared_ptrs()[0][17] * get_cls_shared_ptrs()[0][16] * (get_cls_shared_ptrs()[0][9]);

        // Btlstride2
        Btlstride[2] = 0;

        // set_addr_remap_index_B
        get_cls_shared_ptrs()[5][9] = 0;
        // transpose_B
        get_cls_shared_ptrs()[5][10] = get_cls_shared_ptrs()[0][13];
        // channel_en_B []
        uint32_t* channel_en_B = snrt_l1_malloc(sizeof(uint32_t) * 2);
        if (!channel_en_B) {
            printf("ERROR: channel_en_B alloc failed!\r\n");
            return;
        } else {
            printf("Allocated channel_en_B at %p\r\n", (void*)channel_en_B);
        }
        if (get_cls_shared_ptrs()[0][11] == 0) {
            channel_en_B[0] = chanelEnB_0_0;
            channel_en_B[1] = chanelEnB_0_1;
        } else {
            channel_en_B[0] = chanelEnB_1_0;
            channel_en_B[1] = chanelEnB_1_1;
        }
        get_cls_shared_ptrs()[5][11] = (uint32_t)(uintptr_t)channel_en_B;
        printf("GEMM Intra-Chiplet Kernel Compute Streamer Cfg B Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for C
        //////////////////////////////////////////////////////////////
        // Cslstride0
        get_cls_shared_ptrs()[5][12] = bankWidth / 8;
        // Ctlbound0~3
        uint32_t* Ctlbound = snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!Ctlbound) {
            printf("ERROR: Ctlbound alloc failed!\r\n");
            return;
        } else {
            printf("Allocated Ctlbound at %p\r\n", (void*)Ctlbound);
        }
        get_cls_shared_ptrs()[5][13] = (uint32_t)(uintptr_t)Ctlbound;
        // Ctlbound0
        if (get_cls_shared_ptrs()[0][11] == 0) {
            Ctlbound[0] = Ctlbound_0_0;
        } else {
            Ctlbound[0] = Ctlbound_1_0;
        }
        Ctlbound[1] = (get_cls_shared_ptrs()[0][10]);
        Ctlbound[2] = (get_cls_shared_ptrs()[0][8]);
        Ctlbound[3] = 1;
        // Ctlstride0~3
        uint32_t* Ctlstride = snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!Ctlstride) {
            printf("ERROR: Ctlstride alloc failed!\r\n");
            return;
        } else {
            printf("Allocated Ctlstride at %p\r\n", (void*)Ctlstride);
        }
        get_cls_shared_ptrs()[5][14] = (uint32_t)(uintptr_t)Ctlstride;

        if (get_cls_shared_ptrs()[0][11] == 0) {
            // Ctlstride0
            Ctlstride[0] = c_spatial_bound_0_0 * (bankWidth / 8);
        } else {
            // Ctlstride0
            Ctlstride[0] = c_spatial_bound_0_1 * (bankWidth / 8);
        }
        // Ctlstride1
        Ctlstride[1] = C_elem_len * get_cls_shared_ptrs()[0][15] * get_cls_shared_ptrs()[0][17] / 8;
        // Ctlstride2
        Ctlstride[2] =
            (get_cls_shared_ptrs()[0][10]) * C_elem_len * get_cls_shared_ptrs()[0][15] * get_cls_shared_ptrs()[0][17] / 8;
        // Ctlstride3
        Ctlstride[3] = 0;
        // set_addr_remap_index_C
        get_cls_shared_ptrs()[5][15] = 0;
        // channel_en_C []
        if (get_cls_shared_ptrs()[0][14] == 0) {
            get_cls_shared_ptrs()[5][16] = chanelEnC_C_null;
        } else if (get_cls_shared_ptrs()[0][11] == 0) {
            get_cls_shared_ptrs()[5][16] = chanelEnC_0;
        } else {
            get_cls_shared_ptrs()[5][16] = chanelEnC_1;
        }
        printf("GEMM Intra-Chiplet Kernel Compute Streamer Cfg C Done!\r\n");

        //////////////////////////////////////////////////////////////
        // streamer cfg for D
        //////////////////////////////////////////////////////////////
        // D32slstride0
        get_cls_shared_ptrs()[5][17] = bankWidth / 8;
        // D32tlbound0~3
        uint32_t* D32tlbound = snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!D32tlbound) {
            printf("ERROR: D32tlbound alloc failed!\r\n");
            return;
        } else {
            printf("Allocated D32tlbound at %p\r\n", (void*)D32tlbound);
        }
        get_cls_shared_ptrs()[5][18] = (uint32_t)(uintptr_t)D32tlbound;
        // D32tlbound0~3
        if (get_cls_shared_ptrs()[0][11] == 0) {
            D32tlbound[0] = D32tlbound_0_0;
        } else {
            D32tlbound[0] = D32tlbound_1_0;
        }
        D32tlbound[1] = (get_cls_shared_ptrs()[0][10]);
        D32tlbound[2] = (get_cls_shared_ptrs()[0][8]);
        D32tlbound[3] = 1;
        // D32tlstride0~3
        uint32_t* D32tlstride = snrt_l1_malloc(sizeof(uint32_t) * 4);
        if (!D32tlstride) {
            printf("ERROR: D32tlstride alloc failed!\r\n");
            return;
        } else {
            printf("Allocated D32tlstride at %p\r\n", (void*)D32tlstride);
        }
        get_cls_shared_ptrs()[5][19] = (uint32_t)(uintptr_t)D32tlstride;
        if (get_cls_shared_ptrs()[0][11] == 0) {
            // D32tlstride0
            D32tlstride[0] = d32_spatial_bound_0_0 * (bankWidth / 8);
        } else {
            // D32tlstride0
            D32tlstride[0] = d32_spatial_bound_0_1 * (bankWidth / 8);
        }
        // D32tlstride1
        D32tlstride[1] = D32_elem_len * get_cls_shared_ptrs()[0][15] * get_cls_shared_ptrs()[0][17] / 8;
        // D32tlstride2
        D32tlstride[2] = (get_cls_shared_ptrs()[0][10]) * D32_elem_len *
                         get_cls_shared_ptrs()[0][15] * get_cls_shared_ptrs()[0][17] / 8;
        // D32tlstride3
        D32tlstride[3] = 0;
        // set_addr_remap_index_D32
        get_cls_shared_ptrs()[5][20] = 0;
        // channel_en_D32 []
        if (get_cls_shared_ptrs()[0][11] == 0) {
            get_cls_shared_ptrs()[5][21] = chanelEnD32_0;
        } else {
            get_cls_shared_ptrs()[5][21] = chanelEnD32_1;
        }
        printf("GEMM Intra-Chiplet Kernel Compute Streamer Cfg D Done!\r\n");

        //////////////////////////////////////////////////////////////
        // Configuration the Steamer and Versacore
        //////////////////////////////////////////////////////////////

        set_versacore_streamer_csr(
            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[1],  // A_addr
            (uint32_t*)(&get_cls_shared_ptrs()[5][0]),      // Aslstride
            (uint32_t*)(get_cls_shared_ptrs()[5][1]),       // Atlbound[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][2]),       // Atlstride[] base
            get_cls_shared_ptrs()[5][3],  // set_addr_remap_index_A
            get_cls_shared_ptrs()[5][4],  // transpose_A
            (uint32_t*)(&get_cls_shared_ptrs()[5][5]),  // channel_en_A []

            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[2],  // B_addr
            (uint32_t*)(&get_cls_shared_ptrs()[5][6]),      // Bslstride[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][7]),       // Btlbound[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][8]),       // Btlstride[] base
            get_cls_shared_ptrs()[5][9],   // set_addr_remap_index_B
            get_cls_shared_ptrs()[5][10],  // transpose_B
            (uint32_t*)(get_cls_shared_ptrs()[5][11]),  // channel_en_B []

            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[3],  // C_addr
            (uint32_t*)(&get_cls_shared_ptrs()[5][12]),     // Cslstride[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][13]),      // Ctlbound[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][14]),      // Ctlstride[] base
            get_cls_shared_ptrs()[5][15],  // set_addr_remap_index_C
            (uint32_t*)(&get_cls_shared_ptrs()[5][16]),  // channel_en_C []

            (uint32_t)(uintptr_t)get_cls_shared_ptrs()[4],  // D_addr
            (uint32_t*)(&get_cls_shared_ptrs()[5][17]),  // D32slstride[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][18]),   // D32tlbound[] base
            (uint32_t*)(get_cls_shared_ptrs()[5][19]),   // D32tlstride[] base
            get_cls_shared_ptrs()[5][20],  // set_addr_remap_index_D32
            (uint32_t*)(&get_cls_shared_ptrs()[5][21])  // channel_en_D32 []

        );

        set_versacore_csr(
            get_cls_shared_ptrs()[0][14], (get_cls_shared_ptrs()[0][9]),
            (get_cls_shared_ptrs()[0][10]) * (get_cls_shared_ptrs()[0][8]), 0,
            get_cls_shared_ptrs()[0][11], 0);

        printf("GEMM Intra-Chiplet Kernel Compute Streamer Cfg-ed Done!\r\n");

        // Set CSR to start Streamer
        start_versacore_and_streamer();

        // Poll until Streamer and GEMM accelerator finish
        wait_versacore_and_streamer();

        printf("GEMM Intra-Chiplet Kernel Compute Done!\r\n");

        for (int i = 0;
             i < (get_cls_shared_ptrs()[0][8]) *
                     (get_cls_shared_ptrs()[0][10]) * get_cls_shared_ptrs()[0][15] * get_cls_shared_ptrs()[0][17];
             i++) {
            printf("D[%d] = %d\r\n", i, get_cls_shared_ptrs()[4][i]);
        }
    }

    snrt_cluster_hw_barrier();

    // After computation, dma core will store the output back to L3
    if (snrt_is_dm_core()) {
        uint32_t output_size = (get_cls_shared_ptrs()[0][8]) *
                               (get_cls_shared_ptrs()[0][10]) * get_cls_shared_ptrs()[0][15] *
                               get_cls_shared_ptrs()[0][17] * sizeof(uint32_t);
        uint64_t l3_output_D_addr;
        // l3_output_D_addr output address in L3
        l3_output_D_addr =
            make_u64(get_cls_shared_ptrs()[0][6], get_cls_shared_ptrs()[0][7]);
        xdma_memcpy_1d((void*)get_cls_shared_ptrs()[4], (void*)l3_output_D_addr,
                       output_size);
        int task_id = xdma_start();
        xdma_remote_wait(task_id);
        // Free the allocated L1 memory
        snrt_l1_free(get_cls_shared_ptrs()[0]);
        snrt_l1_free(get_cls_shared_ptrs()[1]);
        snrt_l1_free(get_cls_shared_ptrs()[2]);
        snrt_l1_free(get_cls_shared_ptrs()[3]);
        snrt_l1_free(get_cls_shared_ptrs()[4]);
        snrt_l1_free(get_cls_shared_ptrs()[5]);
        // Reset the shared pointers
        snrt_memset(get_cls_shared_ptrs(), 0,
                    sizeof(uint32_t*) * NUM_CLS_SHARED_PTRS);
        printf("GEMM Intra-Chiplet Kernel Store Output Done!\r\n");
    }
}

//////////////////////// SYMBOL TABLE ////////////////////////
// Here we create the symbol table
SNAX_SYMTAB_SECTION const snax_symbol_t __snax_symtab[] = {
    SNAX_EXPORT_FUNC(__snax_kernel_dummy),
    SNAX_EXPORT_FUNC(__snax_kernel_check_results),
    SNAX_EXPORT_FUNC(__snax_kernel_csr),
    SNAX_EXPORT_FUNC(__snax_kernel_load_compute_store),
    SNAX_EXPORT_FUNC(__snax_kernel_xdma_1d_copy),
    SNAX_EXPORT_FUNC(
        __snax_kernel_versacore_load_compute_store_w_streamer_args),
    SNAX_EXPORT_FUNC(__snax_kernel_gemm_intra_chiplet),
    SNAX_SYMTAB_END};
