// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matteo Perotti

// Include <riscv_vector.h> to use vector intrinsics
// Documentation: https://github.com/riscv/rvv-intrinsic-doc
// Compiler support:
// https://github.com/riscv/riscv-gnu-toolchain/tree/rvv-intrinsic

#include "host.h"
#include "vector_defines.h"
#include "kernels.h"
#include "data.h"
// Run also the scalar benchmark
#define SCALAR 1

// Check the vector results against golden vectors
#define CHECK 1

int main() {
  uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();
  // Initialize UART for output
  init_uart(address_prefix, 32, 1);
  // Turn on the RVV extension
  enable_vec();
  // Testing the dotp function
  printf("\n");
  printf("==========\n");
  printf("=  DOTP  =\n");
  printf("==========\n");
  printf("\n");
  printf("\n");
  // Timer variables
  uint64_t start_time_s = 0;
  uint64_t end_time_s = 0;
  uint64_t runtime_s = 0;
  uint64_t start_time_v = 0;
  uint64_t end_time_v = 0;
  uint64_t runtime_v = 0;
  // Output variables

  int32_t res32_v, res32_s;
  int16_t res16_v, res16_s;
  // Call the main kernel, and measure cycles
  // 32b dotp
  for (uint64_t avl = 8; avl <= (vsize >> 2); avl *= 8) {
    // Dotp
    printf("Calulating 32b dotp with vectors with length = %lu\n", avl);
    start_time_v = ara_get_cycle_count();
    res32_v = dotp_v32b(v32a, v32b, avl);
    end_time_v = ara_get_cycle_count();
    runtime_v = end_time_v - start_time_v;
    printf("Vector runtime: %ld\n", runtime_v);

    if (SCALAR) {
      start_time_s = ara_get_cycle_count();
      res32_s = dotp_s32b(v32a, v32b, avl);
      end_time_s = ara_get_cycle_count();
      runtime_s = end_time_s - start_time_s;
      printf("Scalar runtime: %ld\n", runtime_s);

      if (CHECK) {
        if (res32_v != res32_s) {
          printf("Error: v = %ld, g = %ld\n", res32_v, res32_s);
          return -1;
        }
      }
    }
  }

  for (uint64_t avl = 8; avl <= (vsize >> 1); avl *= 8) {
    // Dotp
    printf("Calulating 16b dotp with vectors with length = %lu\n", avl);
    start_time_v = ara_get_cycle_count();
    res16_v = dotp_v16b(v16a, v16b, avl);
    end_time_v = ara_get_cycle_count();
    runtime_v = end_time_v - start_time_v;
    printf("Vector runtime: %ld\n", runtime_v);

    if (SCALAR) {
      start_time_s = ara_get_cycle_count();
      res16_s = dotp_s16b(v16a, v16b, avl);
      end_time_s = ara_get_cycle_count();
      runtime_s = end_time_s - start_time_s;
      printf("Scalar runtime: %ld\n", runtime_s);

      if (CHECK) {
        if (res16_v != res16_s) {
          printf("Error: v = %ld, g = %ld\n", res16_v, res16_s);
          return -1;
        }
      }
    }
  }

  printf("\nAll tests passed!\n");
  return 0;
}
