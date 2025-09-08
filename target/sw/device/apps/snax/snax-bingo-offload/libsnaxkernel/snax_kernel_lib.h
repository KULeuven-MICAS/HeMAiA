// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
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


// Here we create the symbol table
SNAX_SYMTAB_SECTION const snax_symbol_t __snax_symtab[] = {
    SNAX_EXPORT_FUNC(__snax_kernel_dummy),
    SNAX_EXPORT_FUNC(__snax_kernel_check_results),
    SNAX_EXPORT_FUNC(__snax_kernel_csr),
    SNAX_EXPORT_FUNC(__snax_kernel_load_compute_store),
    SNAX_EXPORT_FUNC(__snax_kernel_xdma_1d_copy),
    SNAX_SYMTAB_END
};
