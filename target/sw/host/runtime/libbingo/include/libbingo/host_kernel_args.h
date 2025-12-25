// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include <stdint.h>
#define __HOST_BINGO_KERNEL_ARGS_DEFINE typedef struct __attribute__((packed, aligned(8)))

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_dummy_args {
    uint64_t dummy_arg_0;
} __host_bingo_kernel_dummy_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_exit_args {
    uint64_t exit_code;
} __host_bingo_kernel_exit_args_t;