// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "host.c"
#include "data.h"

void compress_32b(
    const int8_t *data_in,
    int data_size,
    uint32_t *data_out
){
    int num_words = data_size / 32;
    for (int i = 0; i < num_words; i++) {
        uint32_t packed = 0;
        for (int j = 0; j < 32; j++) {
            if (data_in[i * 32 + j] >= 0) {
                packed |= ((uint32_t)1 << j);
            }
        }
        data_out[i] = packed;
    }
};

uint32_t ham_dist(
    const uint32_t *vec_a,
    const uint32_t *vec_b,
    uint32_t num_words
){
    uint32_t dist = 0;
    for (uint32_t i = 0; i < num_words; i++) {
        dist += __builtin_popcount(vec_a[i] ^ vec_b[i]);
    }
    return dist;
};

void class_compute(
    const uint32_t *query,
    const uint32_t *class_array,
    uint32_t num_classes,
    uint32_t num_words,
    uint32_t *distances_out
){
    for (uint32_t i = 0; i < num_classes; i++) {
        distances_out[i] = ham_dist(query, &class_array[i * num_words], num_words);
    }
};

int main() {
    init_uart(32, 1);
    asm volatile("fence" : : : "memory");

    // Initial components
    uint16_t err = 0;
    uint32_t data_size_32b = data_size / 32;
    uint32_t data_compressed[data_size_32b];

    // Compression
    uint64_t start_cycles = mcycle();
    compress_32b(data_vec, data_size, data_compressed);
    uint64_t end_cycles = mcycle();

    // Compare against bin_data_vec_32b
    for (uint32_t i = 0; i < data_size_32b; i++) {
        if (data_compressed[i] != bin_data_vec_32b[i]) {
            err++;
        }
    }

    // Profiling results
    char uart_tx_buffer[512];
    if (err) {
        sprintf(uart_tx_buffer, "Fail: %d errors \r\n", err);
        print_uart(uart_tx_buffer);
    } else {
        print_uart("Compress Pass! \r\n");
        sprintf(uart_tx_buffer, "EXE-Compress: %llu cycles \r\n", end_cycles - start_cycles);
        print_uart(uart_tx_buffer);

    }

    // Hamming distance check
    err = 0;
    uint32_t hd_result[class_size];

    start_cycles = mcycle();
    class_compute(data_compressed, bin_class_array_32b, class_size, data_size_32b, hd_result);
    end_cycles = mcycle();

    for (uint32_t i = 0; i < class_size; i++) {
        if (hd_result[i] != hamming_distances[i]) {
            err++;
        }
    }

    if (err) {
        sprintf(uart_tx_buffer, "HD Fail: %d errors \r\n", err);
        print_uart(uart_tx_buffer);
    } else {
        print_uart("HD Pass! \r\n");
        sprintf(uart_tx_buffer, "EXE-HD: %llu cycles \r\n", end_cycles - start_cycles);
        print_uart(uart_tx_buffer);
    }

    return 0;
}
