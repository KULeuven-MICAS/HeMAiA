// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "data.h"
#include "snrt.h"

#define MAX_PRINTED_MISMATCHES 16

#if SOFTMAX_ROWS < 1 || SOFTMAX_COLS < 1
#error "SOFTMAX_ROWS and SOFTMAX_COLS must be positive"
#endif

#if SOFTMAX_OUTPUT_SCALE < 1 || SOFTMAX_OUTPUT_SCALE > 255
#error "SOFTMAX_OUTPUT_SCALE must fit in uint8_t"
#endif

/**
 * Compute one row of quantized softmax using integer arithmetic only.
 *
 * The int8 input is Q(SOFTMAX_INPUT_FRAC_BITS). After subtracting the row
 * maximum, max - input is in [0, 255] and directly indexes exp_lut. LUT values
 * represent exp(input - max) in Q(SOFTMAX_EXP_FRAC_BITS). The common LUT scale
 * cancels during normalization.
 */
static void softmax_fixed_u8_row(const int8_t *input, uint8_t *output,
                                 const uint16_t *exp_lut) {
    int8_t max_value = input[0];

    for (uint32_t col = 1; col < SOFTMAX_COLS; ++col) {
        if (input[col] > max_value) {
            max_value = input[col];
        }
    }

    uint32_t exp_sum = 0;
    for (uint32_t col = 0; col < SOFTMAX_COLS; ++col) {
        uint32_t delta =
            (uint32_t)((int16_t)max_value - (int16_t)input[col]);
        uint16_t exp_value = exp_lut[delta];
        exp_sum += exp_value;
    }

    // exp_lut[0] is nonzero and each row contains at least one maximum.
    if (exp_sum == 0) {
        for (uint32_t col = 0; col < SOFTMAX_COLS; ++col) {
            output[col] = 0;
        }
        return;
    }

    for (uint32_t col = 0; col < SOFTMAX_COLS; ++col) {
        uint32_t delta =
            (uint32_t)((int16_t)max_value - (int16_t)input[col]);
        uint16_t exp_value = exp_lut[delta];
        uint32_t numerator =
            (uint32_t)exp_value * SOFTMAX_OUTPUT_SCALE + exp_sum / 2;
        output[col] = (uint8_t)(numerator / exp_sum);
    }
}

/** Split independent rows cyclically over all Snitch compute cores. */
static void softmax_fixed_u8_parallel(const int8_t *input, uint8_t *output,
                                      const uint16_t *exp_lut) {
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t compute_cores = snrt_cluster_compute_core_num();

    for (uint32_t row = core_idx; row < SOFTMAX_ROWS; row += compute_cores) {
        uint32_t row_offset = row * SOFTMAX_COLS;
        softmax_fixed_u8_row(&input[row_offset], &output[row_offset], exp_lut);
    }
}

int main() {
    if (snrt_cluster_idx() == SOFTMAX_TARGET_CLUSTER) {



        uint8_t *tcdm = (uint8_t *)snrt_l1_next();
        int8_t *local_input = (int8_t *)(tcdm + SOFTMAX_INPUT_OFFSET);
        uint8_t *local_output = tcdm + SOFTMAX_OUTPUT_OFFSET;
        uint8_t *local_golden = tcdm + SOFTMAX_GOLDEN_OFFSET;
        uint8_t *local_reference = tcdm + SOFTMAX_REFERENCE_OFFSET;
        uint16_t *local_exp_lut = (uint16_t *)(tcdm + SOFTMAX_LUT_OFFSET);
        volatile uint32_t *shared_error =
            (volatile uint32_t *)(tcdm + SOFTMAX_ERROR_OFFSET);

        uint32_t start_cycle = 0;
        uint32_t end_cycle = 0;

        if (snrt_is_dm_core()) {
            start_cycle = snrt_mcycle();
            snrt_dma_start_1d(local_input, softmax_input,
                            SOFTMAX_ELEMENTS * sizeof(int8_t));
            snrt_dma_start_1d(local_golden, softmax_golden,
                            SOFTMAX_ELEMENTS * sizeof(uint8_t));
            snrt_dma_start_1d(local_reference, softmax_reference,
                            SOFTMAX_ELEMENTS * sizeof(uint8_t));
            snrt_dma_start_1d(local_exp_lut, softmax_exp_lut,
                            SOFTMAX_LUT_SIZE * sizeof(uint16_t));
            snrt_dma_wait_all();
            end_cycle = snrt_mcycle();
            printf("DMA: %d\n", end_cycle - start_cycle);
            *shared_error = 0;
        }

        snrt_cluster_hw_barrier();

        // if (snrt_is_dm_core() == 0) {
        //     printf("Snitch fixed-point softmax: rows=%d cols=%d cores=%d "
        //         "input=Q%d output_scale=%d TCDM=%d bytes\n",
        //         SOFTMAX_ROWS, SOFTMAX_COLS, snrt_cluster_compute_core_num(),
        //         SOFTMAX_INPUT_FRAC_BITS, SOFTMAX_OUTPUT_SCALE,
        //         SOFTMAX_TCDM_BYTES);
        // }

        
        // if (snrt_cluster_core_idx() == 0) {
        //     start_cycle = snrt_mcycle();
        // }

        // This barrier is included in the reported parallel execution time.
        snrt_cluster_hw_barrier();

        if (snrt_is_compute_core()) {
            printf("Start! \n");
            start_cycle = snrt_mcycle();
            softmax_fixed_u8_parallel(local_input, local_output, local_exp_lut);
            end_cycle = snrt_mcycle();
            printf("End! \n");
        // }

        // snrt_cluster_hw_barrier();

        // if (snrt_cluster_core_idx() == 0) {
            

            // uint32_t exact_mismatches = 0;
            // uint32_t reference_mismatches = 0;
            // uint32_t max_reference_error = 0;
            // uint32_t min_row_sum = UINT32_MAX;
            // uint32_t max_row_sum = 0;

            // for (uint32_t index = 0; index < SOFTMAX_ELEMENTS; ++index) {
            //     if (local_output[index] != local_golden[index]) {
            //         if (exact_mismatches < MAX_PRINTED_MISMATCHES) {
            //             printf("Mismatch at %d: got %d, expected %d\n", index,
            //                 local_output[index], local_golden[index]);
            //         }
            //         ++exact_mismatches;
            //     }

            //     uint32_t reference_error =
            //         local_output[index] > local_reference[index]
            //             ? local_output[index] - local_reference[index]
            //             : local_reference[index] - local_output[index];
            //     if (reference_error > max_reference_error) {
            //         max_reference_error = reference_error;
            //     }
            //     if (reference_error > SOFTMAX_REFERENCE_TOLERANCE) {
            //         ++reference_mismatches;
            //     }
            // }

            // for (uint32_t row = 0; row < SOFTMAX_ROWS; ++row) {
            //     uint32_t row_sum = 0;
            //     for (uint32_t col = 0; col < SOFTMAX_COLS; ++col) {
            //         row_sum += local_output[row * SOFTMAX_COLS + col];
            //     }
            //     if (row_sum < min_row_sum) {
            //         min_row_sum = row_sum;
            //     }
            //     if (row_sum > max_row_sum) {
            //         max_row_sum = row_sum;
            //     }
            // }

            printf("Softmax cycles: %d\n", end_cycle - start_cycle);
            // printf("Bit-exact mismatches: %d\n", exact_mismatches);
            // printf("Float-reference max error: %d (tolerance %d), failures: %d\n",
            //     max_reference_error, SOFTMAX_REFERENCE_TOLERANCE,
            //     reference_mismatches);
            // printf("Quantized row-sum range: [%d, %d], nominal scale: %d\n",
            //     min_row_sum, max_row_sum, SOFTMAX_OUTPUT_SCALE);

            // *shared_error = exact_mismatches + reference_mismatches;
        }

        // Propagate core 0's result to the DM core, which reports it to CVA6.
        snrt_cluster_hw_barrier();
        // uint32_t exit_code = *shared_error;
        // return_to_cva6_single_cluster(exit_code);
        return 0;
    }
    return 0;
}
