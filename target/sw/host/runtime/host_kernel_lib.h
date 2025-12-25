#pragma once

#include <stdint.h>
#include "uart.h"

static inline uint64_t __host_bingo_kernel_dummy(void *arg){
    // This is a special kernel to print a string from the host
    // The argument is a pointer to the string in the host memory
    printf("Host Kernel Dummy: %d\r\n", ((uint64_t *)arg)[0]);
    return 0;
}

static inline uint64_t __host_bingo_kernel_exit(void *arg){
    // This is a special kernel to exit the host kernel loop
    uint64_t exit_code = ((uint64_t *)arg)[0];
    printf("Host Kernel Exit called with exit code %d\r\n", exit_code);
    return 1;
}

