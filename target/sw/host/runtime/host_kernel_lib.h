#pragma once

#include <stdint.h>
#include "uart.h"
#include "heterogeneous_runtime.h"
#include "perf_tracing.h"
#define EXIT_CODE_SUCC 1

// Host Bingo Kernel Implementations
// Normally the functions ret with 0
// Only the exit kernel returns the exit code defined by EXIT_CODE_SUCC, for now it is 1
static inline uint64_t __host_bingo_kernel_dummy(void *arg){
    // This is a dummy kernel to print a string from the host
    // Arg[0]: Dummy input
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t dummy_input = ((uint64_t *)arg)[0];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_safe("Chip(%x, %x): [Host] Kernel Dummy: %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), dummy_input);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    return 0;
}

static inline uint64_t __host_bingo_kernel_exit(void *arg){
    // This is a special kernel to exit the host kernel loop
    // We can add more clean up work here if needed
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t exit_code = ((uint64_t *)arg)[0];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_safe("Chip(%x, %x): [Host] Kernel Exit called with exit code %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), exit_code);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    return EXIT_CODE_SUCC;
}

static inline uint64_t __host_bingo_kernel_entry(void *arg){
    // This is a special kernel to book keeping the start CC of the Workload
    // In the future we can add more contents here when we need to do more initialization work
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    uint64_t start_cc = bingo_mcycle();
    printf_safe("Chip(%x, %x): [Host] Start at %d CC\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), start_cc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    return 0;

}
static inline uint64_t __host_bingo_kernel_check_result(void *arg){
    // Check the output data against the golden data
    // Arg0: uint64_t golden_data_addr
    // Arg1: uint64_t output_data_addr
    // Arg2: uint64_t data_size in Byte
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint8_t* golden_data_addr = (uint8_t*)(((uint64_t *)arg)[0]);
    uint8_t* output_data_addr = (uint8_t*)(((uint64_t *)arg)[1]);
    uint64_t data_size = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    uint32_t err = 0;
    for (uint64_t i = 0; i < data_size; i++) {
        if (output_data_addr[i] != golden_data_addr[i]) {
            err++;
            printf_safe("Unequals. output[%d] = %d, golden[%d] = %d\n", i,
                   output_data_addr[i], i, golden_data_addr[i]);
        }
    }
    if (err == 0) {
        printf_safe("Chip(%x, %x): [Host] Kernel Check Result: PASS! All %d bytes match.\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), data_size);
    } else {
        printf_safe("Chip(%x, %x): [Host] Kernel Check Result: FAIL! %d mismatches found out of %d bytes.\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), err, data_size);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    return err;
}

static inline uint64_t __host_bingo_kernel_idma(void *arg){
    // SoC DMA memcpy
    // Arg0: uint64_t src_addr
    // Arg1: uint64_t dst_addr
    // Arg2: uint64_t size in Byte
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t src_addr = ((uint64_t *)arg)[0];
    uint64_t dst_addr = ((uint64_t *)arg)[1];
    uint64_t size = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_CFG_START);
    uint64_t tf_id = sys_dma_memcpy(get_current_chip_id(), dst_addr, src_addr, size);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_RUN_START);
    while (*(sys_dma_done_ptr(get_current_chip_id())) != tf_id) {
        asm volatile("nop");
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_RUN_END);
    return 0;
}

