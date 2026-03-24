// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "host.c"
#include "data.h"

// static inline uint64_t cva6_mcycle() {
//     register uint64_t r;
//     asm volatile("csrr %0, mcycle" : "=r"(r));
//     return r;
// }

void matmul(
    const int8_t *vec_a, 
    const int8_t *vec_b, 
    int32_t *vec_out, 
    int M, int K, int N
){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)vec_a[i * K + k] * (int32_t)vec_b[k * N + j];
            }
            vec_out[i * N + j] = sum;
        }
    }
};

// Frequency at which the UART peripheral is clocked
int main() {
    init_uart(32, 1);
    asm volatile("fence" : : : "memory");

    uint16_t err = 0;
    int32_t result[M*N];

    uint64_t start_cycles = mcycle();
    matmul(vec_a, vec_b, result, M, K, N);
    uint64_t end_cycles = mcycle();

    // Compare against vec_o
    for (int i = 0; i < M * N; i++) {
        if (result[i] != vec_o[i]) {
            err++;
        }
    }

    // Printing results
    char uart_tx_buffer[512];
    if (err) {
        sprintf(uart_tx_buffer, "Fail: %d errors \r\n", err);
        print_uart(uart_tx_buffer);
    } else {
        print_uart("Pass! \r\n");
        sprintf(uart_tx_buffer, "EXET: %llu cycles \r\n", end_cycles - start_cycles);
        print_uart(uart_tx_buffer);

    }

    return 0;
}
