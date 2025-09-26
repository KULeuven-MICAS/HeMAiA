// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once

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

// Getter function to retrive the cls shared pointers
inline uint32_t **get_cls_shared_ptrs() {
    return (uint32_t **)&(cls()->shared_ptrs);
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
SNAX_LIB_DEFINE void __snax_kernel_versacore_streamer_cfg(void *arg) {
    // config the versacore streamer
    // Arg0: a_streamer_cfg_addr
    // Arg1: b_streamer_cfg_addr
    // Arg2: c_streamer_cfg_addr
    // Arg3: d_streamer_cfg_addr

    if (snrt_cluster_core_idx() == 0) {
        // parse the args
        // We allocate 256 bytes of heap memory for the args
        // Each arg is 4 bytes, so we can store 64 args

        get_cls_shared_ptrs()[0] = snrt_l1_malloc(256);

        // a_streamer_cfg_addr in total 7 words
        // a_streamer_cfg_addr - int32_t delta_local_a
        get_cls_shared_ptrs()[0][0] = ((uint32_t *)(((uint32_t *)arg)[0]))[0];
        // a_streamer_cfg_addr - int32_t* Aslstride
        get_cls_shared_ptrs()[0][1] = ((uint32_t *)(((uint32_t *)arg)[0]))[1];
        // a_streamer_cfg_addr - int32_t* Atlbound
        get_cls_shared_ptrs()[0][2] = ((uint32_t *)(((uint32_t *)arg)[0]))[2];
        // a_streamer_cfg_addr - int32_t* Atlstride
        get_cls_shared_ptrs()[0][3] = ((uint32_t *)(((uint32_t *)arg)[0]))[3];
        // a_streamer_cfg_addr - int32_t set_addr_remap_index_A
        get_cls_shared_ptrs()[0][4] = ((uint32_t *)(((uint32_t *)arg)[0]))[4];
        // a_streamer_cfg_addr - int32_t transpose_A
        get_cls_shared_ptrs()[0][5] = ((uint32_t *)(((uint32_t *)arg)[0]))[5];
        // a_streamer_cfg_addr - int32_t channel_en_A
        get_cls_shared_ptrs()[0][6] = ((uint32_t *)(((uint32_t *)arg)[0]))[6];

        // b_streamer_cfg_addr in total 7 words
        // b_streamer_cfg_addr - int32_t delta_local_b
        get_cls_shared_ptrs()[0][7] = ((uint32_t *)(((uint32_t *)arg)[1]))[0];
        // b_streamer_cfg_addr - int32_t* Bslstride
        get_cls_shared_ptrs()[0][8] = ((uint32_t *)(((uint32_t *)arg)[1]))[1];
        // b_streamer_cfg_addr - int32_t* Btlbound
        get_cls_shared_ptrs()[0][9] = ((uint32_t *)(((uint32_t *)arg)[1]))[2];
        // b_streamer_cfg_addr - int32_t* Btlstride
        get_cls_shared_ptrs()[0][10] = ((uint32_t *)(((uint32_t *)arg)[1]))[3];
        // b_streamer_cfg_addr - int32_t set_addr_remap_index_B
        get_cls_shared_ptrs()[0][11] = ((uint32_t *)(((uint32_t *)arg)[1]))[4];
        // b_streamer_cfg_addr - int32_t transpose_B
        get_cls_shared_ptrs()[0][12] = ((uint32_t *)(((uint32_t *)arg)[1]))[5];
        // b_streamer_cfg_addr - int32_t channel_en_B
        get_cls_shared_ptrs()[0][13] = ((uint32_t *)(((uint32_t *)arg)[1]))[6];

        // c_streamer_cfg_addr in total 6 words
        // c_streamer_cfg_addr - delta_local_c
        get_cls_shared_ptrs()[0][14] = ((uint32_t *)(((uint32_t *)arg)[2]))[0];
        // c_streamer_cfg_addr - int32_t* Cslstride
        get_cls_shared_ptrs()[0][15] = ((uint32_t *)(((uint32_t *)arg)[2]))[1];
        // c_streamer_cfg_addr - int32_t* Ctlbound
        get_cls_shared_ptrs()[0][16] = ((uint32_t *)(((uint32_t *)arg)[2]))[2];
        // c_streamer_cfg_addr - int32_t* Ctlstride
        get_cls_shared_ptrs()[0][17] = ((uint32_t *)(((uint32_t *)arg)[2]))[3];
        // c_streamer_cfg_addr - int32_t set_addr_remap_index_C
        get_cls_shared_ptrs()[0][18] = ((uint32_t *)(((uint32_t *)arg)[2]))[4];
        // c_streamer_cfg_addr - int32_t* channel_en_C
        get_cls_shared_ptrs()[0][19] = ((uint32_t *)(((uint32_t *)arg)[2]))[5];

        // d_streamer_cfg_addr in total 6 words
        // d_streamer_cfg_addr - delta_local_d32
        get_cls_shared_ptrs()[0][20] = ((uint32_t *)(((uint32_t *)arg)[3]))[0];
        // d_streamer_cfg_addr - int32 * D32slstride
        get_cls_shared_ptrs()[0][21] = ((uint32_t *)(((uint32_t *)arg)[3]))[1];
        // d_streamer_cfg_addr - int32_t* D32tlbound
        get_cls_shared_ptrs()[0][22] = ((uint32_t *)(((uint32_t *)arg)[3]))[2];
        // d_streamer_cfg_addr - int32_t* D32tlstride
        get_cls_shared_ptrs()[0][23] = ((uint32_t *)(((uint32_t *)arg)[3]))[3];
        // d_streamer_cfg_addr - int32_t set_addr_remap_index_D32
        get_cls_shared_ptrs()[0][24] = ((uint32_t *)(((uint32_t *)arg)[3]))[4];
        // d_streamer_cfg_addr - int32_t* channel_en_D
        get_cls_shared_ptrs()[0][25] = ((uint32_t *)(((uint32_t *)arg)[3]))[5];

        set_versacore_streamer_csr(
            get_cls_shared_ptrs()[0][0], (int32_t *)get_cls_shared_ptrs()[0][1],
            (int32_t *)get_cls_shared_ptrs()[0][2],
            (int32_t *)get_cls_shared_ptrs()[0][3], get_cls_shared_ptrs()[0][4],
            get_cls_shared_ptrs()[0][5], (int32_t *)get_cls_shared_ptrs()[0][6],

            get_cls_shared_ptrs()[0][7], (int32_t *)get_cls_shared_ptrs()[0][8],
            (int32_t *)get_cls_shared_ptrs()[0][9],
            (int32_t *)get_cls_shared_ptrs()[0][10],
            get_cls_shared_ptrs()[0][11], get_cls_shared_ptrs()[0][12],
            (int32_t *)get_cls_shared_ptrs()[0][13],

            get_cls_shared_ptrs()[0][14],
            (int32_t *)get_cls_shared_ptrs()[0][15],
            (int32_t *)get_cls_shared_ptrs()[0][16],
            (int32_t *)get_cls_shared_ptrs()[0][17],
            get_cls_shared_ptrs()[0][18],
            (int32_t *)get_cls_shared_ptrs()[0][19],

            get_cls_shared_ptrs()[0][20],
            (int32_t *)get_cls_shared_ptrs()[0][21],
            (int32_t *)get_cls_shared_ptrs()[0][22],
            (int32_t *)get_cls_shared_ptrs()[0][23],
            get_cls_shared_ptrs()[0][24],
            (int32_t *)get_cls_shared_ptrs()[0][25]);

        // start the streamer
        csrw_ss(STREAMER_START_CSR, 1);

        printf(
            "Chip(%x, %x): [Cluster %d] Core(%d) Configured and Launched "
            "Versacore Streamer\n",
            "Streamer\n", get_current_chip_loc_x(), get_current_chip_loc_y(),
            snrt_cluster_idx(), snrt_cluster_core_idx());

        // Free the allocated memory
        snrt_l1_free(get_cls_shared_ptrs()[0]);

        printf("[Cluster %d] Core(%d) Free allocated heap memory\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());

        // clear the shared ptrs
        snrt_memset(get_cls_shared_ptrs(), 0,
                    sizeof(uint32_t *) * NUM_CLS_SHARED_PTRS);
        printf("[Cluster %d] Core(%d) Free shared pointers\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
    }
}

SNAX_LIB_DEFINE void __snax_kernel_versacore_cfg(void *arg) {
    // config the versacore
    // parse the args
    // We allocate 256 bytes of heap memory for the args
    // Each arg is 4 bytes, so we can store 64 args

    if (snrt_cluster_core_idx() == 0) {
        get_cls_shared_ptrs()[0] = snrt_l1_malloc(256);
        get_cls_shared_ptrs()[0][0] = ((uint32_t *)arg)[0];
        get_cls_shared_ptrs()[0][1] = ((uint32_t *)arg)[1];
        get_cls_shared_ptrs()[0][2] = ((uint32_t *)arg)[2];
        get_cls_shared_ptrs()[0][3] = ((uint32_t *)arg)[3];
        get_cls_shared_ptrs()[0][4] = ((uint32_t *)arg)[4];
        get_cls_shared_ptrs()[0][5] = ((uint32_t *)arg)[5];
        get_cls_shared_ptrs()[0][6] = ((uint32_t *)arg)[6];

        get_cls_shared_ptrs()[0][7] =
            gen_subtraction_config(((uint32_t *)arg)[3], ((uint32_t *)arg)[4]);

        set_versacore_csr(
            get_cls_shared_ptrs()[0][0], get_cls_shared_ptrs()[0][1],
            get_cls_shared_ptrs()[0][2], get_cls_shared_ptrs()[0][7],
            get_cls_shared_ptrs()[0][5], get_cls_shared_ptrs()[0][6]);

        // start the versacore
        csrw_ss(VERSACORE_START_CSR, 1);

        printf("Chip(%x, %x): [Cluster %d] Core(%d) Configured Versacore\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               snrt_cluster_idx(), snrt_cluster_core_idx());

        // Free the allocated memory
        snrt_l1_free(get_cls_shared_ptrs()[0]);
        printf("[Cluster %d] Core(%d) Free allocated heap memory\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
        // clear the shared ptrs
        snrt_memset(get_cls_shared_ptrs(), 0,
                    sizeof(uint32_t *) * NUM_CLS_SHARED_PTRS);
        printf("[Cluster %d] Core(%d) Free shared pointers\n",
               snrt_cluster_idx(), snrt_cluster_core_idx());
    }
}

SNAX_LIB_DEFINE void __snax_kernel_streamer_and_versacore_wait(void *arg) {
    if (snrt_cluster_core_idx() == 0) {
        // wait the streamer to finish
        while (csrr_ss(VERSACORE_BUSY)) {
        }
        while (csrr_ss(STREAMER_BUSY_CSR)) {
        }
        printf(
            "Chip(%x, %x): [Cluster %d] Core(%d) Versacore and Streamer "
            "Finished\n",
            get_current_chip_loc_x(), get_current_chip_loc_y(),
            snrt_cluster_idx(), snrt_cluster_core_idx());
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
    SNAX_EXPORT_FUNC(__snax_kernel_versacore_streamer_cfg),
    SNAX_EXPORT_FUNC(__snax_kernel_versacore_cfg),
    SNAX_EXPORT_FUNC(__snax_kernel_streamer_and_versacore_wait),
    SNAX_SYMTAB_END
};
