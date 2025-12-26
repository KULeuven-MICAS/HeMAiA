#pragma once

#include <stdint.h>
#include "uart.h"
#include "heterogeneous_runtime.h"
static inline uint64_t __host_bingo_kernel_dummy(void *arg){
    // This is a special kernel to print a string from the host
    // Arg[0]: Dummy input
    // Arg[1]: global mutex ptr
    uint32_t* global_mutex_ptr = (uint32_t*)(((uint64_t *)arg)[1]);
    mutex_tas_acquire(global_mutex_ptr);
    printf("Host Kernel Dummy: %d\r\n", ((uint64_t *)arg)[0]);
    mutex_release(global_mutex_ptr);
    return 0;
}

static inline uint64_t __host_bingo_kernel_exit(void *arg){
    // This is a special kernel to exit the host kernel loop
    uint64_t exit_code = ((uint64_t *)arg)[0];
    printf("Host Kernel Exit called with exit code %d\r\n", exit_code);
    return 1;
}

