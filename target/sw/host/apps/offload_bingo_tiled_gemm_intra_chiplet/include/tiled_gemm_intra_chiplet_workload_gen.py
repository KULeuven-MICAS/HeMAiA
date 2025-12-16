#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import re

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition, format_scalar_define, format_vector_define  # noqa E402

# # Add golden model path
from snax_utils import block_gemm_golden_model # noqa E402

np.random.seed(320)

def emit_workload_file(**kwargs):
    emit_str = ['''// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
''']
    emit_str += emit_workload_content(**kwargs)

    emit_str += ['''
    return num_tasks;
}''']

    return "\n".join(emit_str)

def emit_workload_content(**kwargs):
    # -------------------------------------------------------------
    # matmul workload settings
    # -------------------------------------------------------------

    # arguments
    M1 = kwargs["M1"]
    K1 = kwargs["K1"]
    N1 = kwargs["N1"]

    M2 = kwargs["M2"]
    K2 = kwargs["K2"]
    N2 = kwargs["N2"]
    assert K2 == N2 == 1, "K2 and N2 must be 1 in tiled_gemm_intra_chiplet kernel."

    array_shape = kwargs["array_shape"]

    data_type = 0  # int8 data type
    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        0
    ]
    tileSize = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        1
    ]
    meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        2
    ]

    # header part
    data_str = []

    # pipeline start
    data_str += [
        '''
#pragma once
#include "tiled_gemm_intra_chiplet_data.h"
#include "host.h"
#include "libbingo/bingo_api.h"

// This workload tests the single-chiplet
// 1. load A and B matrices from L3 to L1
// 2. perform GEMM on versacore, pass correct args to versacore
// 3. store the result D matrix back to L3
// 4. check the result D matrix with the golden data

#ifdef HOST_DEBUG
#define HOST_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define HOST_DEBUG_PRINT(...)
#endif

uint32_t __workload_versacore_intra_chiplet(bingo_task_t **task_list)
{
    ///////////////////
    // Main function //
    ///////////////////
    // 1. Get the kernel function address by the kernel name and prepare the
    // args
    // 2. Register the tasks
    // 3. Set the task dependency
    // 4. Set the assigned chiplet id and cluster id

    uint8_t current_chip_id = get_current_chip_id();
    // only test chip 0 for intra-chiplet gemm
    uint8_t task_chip_id = 0;
    uint8_t cluster_id = 0; // versacore is located at cluster

    // 1.1 Get the kernel function address by the kernel name
    check_kernel_tab_ready();
    uint32_t __snax_kernel_xdma_1d_copy_func_addr =
        get_device_function("__snax_kernel_xdma_1d_copy");
    uint32_t __snax_kernel_gemm_func_addr =
        get_device_function("__snax_kernel_gemm");
    uint32_t __snax_kernel_start_gemm_and_wait_func_addr =
        get_device_function("__snax_kernel_minimal_cfg_start_gemm_and_wait");
    uint32_t check_results_func_addr =
        get_device_function("__snax_kernel_check_results");

    if (__snax_kernel_xdma_1d_copy_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        __snax_kernel_gemm_func_addr == SNAX_SYMTAB_END_FN_ADDR || __snax_kernel_start_gemm_and_wait_func_addr == SNAX_SYMTAB_END_FN_ADDR ||
        check_results_func_addr == SNAX_SYMTAB_END_FN_ADDR)
    {
        printf("Error: Kernel symbol lookup failed!\\r\\n");
    }

    // common parameters
    uint32_t AdataTileSize = BINGO_CHIPLET_READW(M1) * BINGO_CHIPLET_READW(K1) *
                             BINGO_CHIPLET_READW(meshRow) *
                             BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t BdataTileSize = BINGO_CHIPLET_READW(K1) * BINGO_CHIPLET_READW(N1) *
                             BINGO_CHIPLET_READW(meshCol) *
                             BINGO_CHIPLET_READW(tileSize) * sizeof(uint8_t);
    uint32_t DdataTileSize = BINGO_CHIPLET_READW(M1) * BINGO_CHIPLET_READW(N1) *
                             BINGO_CHIPLET_READW(meshRow) *
                             BINGO_CHIPLET_READW(meshCol) * sizeof(uint32_t);

    // pipeline starting point: load A and B matrices from L3 to L1
    uint64_t A1_addr_l3 = chiplet_addr_transform((uint64_t)(uintptr_t)(A1));
    uint64_t B_addr_l3 = chiplet_addr_transform((uint64_t)(uintptr_t)(B));

    uint64_t cluster_l1_addr_A_ping = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(AdataTileSize));
    uint64_t cluster_l1_addr_B_ping = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(BdataTileSize));
    uint64_t cluster_l1_addr_D_ping = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(DdataTileSize));
    uint64_t cluster_l1_addr_A_pong = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(AdataTileSize));
    uint64_t cluster_l1_addr_D_pong = bingo_l1_alloc(current_chip_id, 0, BINGO_CHIPLET_READW(DdataTileSize));

    // load A1 matrix from L3 to L1
    __snax_kernel_xdma_1d_copy_args_t *task_l3_to_cluster_args_A1 =
        (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_l3_to_cluster_args_A1->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(A1_addr_l3));
    task_l3_to_cluster_args_A1->src_addr_lo = LOW32(BINGO_CHIPLET_READD(A1_addr_l3));
    task_l3_to_cluster_args_A1->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_A_ping));
    task_l3_to_cluster_args_A1->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_A_ping));
    task_l3_to_cluster_args_A1->size = ARRAY_SIZE_BYTES(A1);

    // load B matrix from L3 to L1
    __snax_kernel_xdma_1d_copy_args_t *task_l3_to_cluster_args_B =
        (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_l3_to_cluster_args_B->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(B_addr_l3));
    task_l3_to_cluster_args_B->src_addr_lo = LOW32(BINGO_CHIPLET_READD(B_addr_l3));
    task_l3_to_cluster_args_B->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_B_ping));
    task_l3_to_cluster_args_B->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_B_ping));
    task_l3_to_cluster_args_B->size = ARRAY_SIZE_BYTES(B);

    // load A2 matrix from L3 to L1
    uint64_t A2_addr_l3 = chiplet_addr_transform((uint64_t)(uintptr_t)(A2));
    __snax_kernel_xdma_1d_copy_args_t *task_l3_to_cluster_args_A2 =
        (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_l3_to_cluster_args_A2->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(A2_addr_l3));
    task_l3_to_cluster_args_A2->src_addr_lo = LOW32(BINGO_CHIPLET_READD(A2_addr_l3));
    task_l3_to_cluster_args_A2->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_A_pong));
    task_l3_to_cluster_args_A2->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_A_pong));
    task_l3_to_cluster_args_A2->size = ARRAY_SIZE_BYTES(A2);

    // args for gemm1
    __snax_kernel_gemm_intra_chiplet_args_t *gemm1_args = (__snax_kernel_gemm_intra_chiplet_args_t *)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_kernel_gemm_intra_chiplet_args_t));
    gemm1_args->input_A_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_A_ping));
    gemm1_args->input_A_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_A_ping));
    gemm1_args->input_B_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_B_ping));
    gemm1_args->input_B_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_B_ping));
    gemm1_args->input_C_addr_hi = 0;
    gemm1_args->input_C_addr_lo = 0; // no C matrix
    gemm1_args->output_D_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_D_ping));
    gemm1_args->output_D_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D_ping));
    gemm1_args->M = BINGO_CHIPLET_READW(M1);
    gemm1_args->K = BINGO_CHIPLET_READW(K1);
    gemm1_args->N = BINGO_CHIPLET_READW(N1);
    gemm1_args->array_shape = BINGO_CHIPLET_READW(array_shape);
    gemm1_args->transpose_A = BINGO_CHIPLET_READW(transposed_A);
    gemm1_args->transpose_B = BINGO_CHIPLET_READW(transposed_B);
    gemm1_args->accumPrevC = BINGO_CHIPLET_READW(accumPrevC); // false
'''
    ]

    # in the middle of the pipeline
    # 2 - (M2 - 1)
    for tile_idx in range(3, M2+1):
        # load A{tile_idx} from L3 to L1
        f_string_load_A = f"    // load A tile {tile_idx}\n"

        f_string_load_A += f'''    uint64_t A{tile_idx}_addr_l3 = chiplet_addr_transform((uint64_t)(uintptr_t)(A{tile_idx}));
    '''

        f_string_load_A += f'''__snax_kernel_xdma_1d_copy_args_t *task_l3_to_cluster_args_A{tile_idx} =
        (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(
            bingo_get_l3_heap_manager(current_chip_id),
            sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_l3_to_cluster_args_A{tile_idx}->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(A{tile_idx}_addr_l3));
    task_l3_to_cluster_args_A{tile_idx}->src_addr_lo = LOW32(BINGO_CHIPLET_READD(A{tile_idx}_addr_l3));
    task_l3_to_cluster_args_A{tile_idx}->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_A_{"pong" if tile_idx %2 ==0 else "ping"}));
    task_l3_to_cluster_args_A{tile_idx}->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_A_{"pong" if tile_idx %2 ==0 else "ping"}));
    task_l3_to_cluster_args_A{tile_idx}->size = ARRAY_SIZE_BYTES(A{tile_idx});
'''
        data_str += [f_string_load_A]

        # args for gemm{tile_idx-1}
        f_string_gemm_minimal_cfg = f'''    // gemm tile {tile_idx - 1}
    __snax_kernel_minimal_cfg_start_gemm_and_wait_args_t *gemm{tile_idx - 1}_mininal_cfg_args = (__snax_kernel_minimal_cfg_start_gemm_and_wait_args_t *)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_kernel_minimal_cfg_start_gemm_and_wait_args_t));
    gemm{tile_idx - 1}_mininal_cfg_args->input_A_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_A_{"ping" if tile_idx %2 ==0 else "pong"}));
    gemm{tile_idx - 1}_mininal_cfg_args->input_B_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_B_ping));
    gemm{tile_idx - 1}_mininal_cfg_args->input_C_addr_lo = 0; // no C matrix
    gemm{tile_idx - 1}_mininal_cfg_args->output_D_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D_{"ping" if tile_idx %2 ==0 else "pong"}));
    '''
        data_str += [f_string_gemm_minimal_cfg]

        # store D{tile_idx - 2} from L1 to L3
        f_string_stre_D = f'''    // store D tile {tile_idx - 2}
    uint64_t output_D{tile_idx - 2}_data_addr_L3 = o1heapAllocate(bingo_get_l3_heap_manager(current_chip_id), ARRAY_SIZE_BYTES(D{tile_idx - 2}));
    __snax_kernel_xdma_1d_copy_args_t *task_cluster_to_l3_args_D{tile_idx - 2} =
    (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(
        bingo_get_l3_heap_manager(current_chip_id),
        sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_cluster_to_l3_args_D{tile_idx - 2}->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_D_{"pong" if (tile_idx -2) %2 ==0 else "ping"}));
    task_cluster_to_l3_args_D{tile_idx - 2}->src_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D_{"pong" if (tile_idx -2) %2 ==0 else "ping"}));
    task_cluster_to_l3_args_D{tile_idx - 2}->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(output_D{tile_idx - 2}_data_addr_L3));
    task_cluster_to_l3_args_D{tile_idx - 2}->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(output_D{tile_idx - 2}_data_addr_L3));
    task_cluster_to_l3_args_D{tile_idx - 2}->size = ARRAY_SIZE_BYTES(D{tile_idx - 2});
'''
        data_str += [f_string_stre_D]

    # in the tile of the pipeline
    #M2 + 1
    tile_idx = M2 + 1
    # gemm{M2}
    f_string_gemm_minimal_cfg = f'''    // gemm tile {tile_idx - 1}
    __snax_kernel_minimal_cfg_start_gemm_and_wait_args_t *gemm{tile_idx - 1}_mininal_cfg_args = (__snax_kernel_minimal_cfg_start_gemm_and_wait_args_t *)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_kernel_minimal_cfg_start_gemm_and_wait_args_t));
    gemm{tile_idx - 1}_mininal_cfg_args->input_A_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_A_{"ping" if tile_idx %2 ==0 else "pong"}));
    gemm{tile_idx - 1}_mininal_cfg_args->input_B_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_B_ping));
    gemm{tile_idx - 1}_mininal_cfg_args->input_C_addr_lo = 0; // no C matrix
    gemm{tile_idx - 1}_mininal_cfg_args->output_D_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D_{"ping" if tile_idx %2 ==0 else "pong"}));
    '''
    data_str += [f_string_gemm_minimal_cfg]

    # store D{M2 - 1} from L1 to L3
    f_string_stre_D = f'''    // store D tile {tile_idx - 2}'''
    f_string_stre_D += f'''
    uint64_t output_D{tile_idx - 2}_data_addr_L3 = o1heapAllocate(bingo_get_l3_heap_manager(current_chip_id), ARRAY_SIZE_BYTES(D{tile_idx - 2}));
    __snax_kernel_xdma_1d_copy_args_t *task_cluster_to_l3_args_D{tile_idx - 2} =
    (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(
        bingo_get_l3_heap_manager(current_chip_id),
        sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_cluster_to_l3_args_D{tile_idx - 2}->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_D_{"pong" if (tile_idx -2) %2 ==0 else "ping"}));
    task_cluster_to_l3_args_D{tile_idx - 2}->src_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D_{"pong" if (tile_idx -2) %2 ==0 else "ping"}));
    task_cluster_to_l3_args_D{tile_idx - 2}->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(output_D{tile_idx - 2}_data_addr_L3));
    task_cluster_to_l3_args_D{tile_idx - 2}->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(output_D{tile_idx - 2}_data_addr_L3));
    task_cluster_to_l3_args_D{tile_idx - 2}->size = ARRAY_SIZE_BYTES(D{tile_idx - 2});
'''
    data_str += [f_string_stre_D]

    # pipeline end
    # store D{M2} from L1 to L3
    tile_idx = M2 + 2
    f_string_stre_D = f'''    // store D tile {tile_idx - 2}'''
    f_string_stre_D += f'''
    uint64_t output_D{tile_idx - 2}_data_addr_L3 = o1heapAllocate(bingo_get_l3_heap_manager(current_chip_id), ARRAY_SIZE_BYTES(D{tile_idx - 2}));
    __snax_kernel_xdma_1d_copy_args_t *task_cluster_to_l3_args_D{tile_idx - 2} =
    (__snax_kernel_xdma_1d_copy_args_t *)o1heapAllocate(
        bingo_get_l3_heap_manager(current_chip_id),
        sizeof(__snax_kernel_xdma_1d_copy_args_t));
    task_cluster_to_l3_args_D{tile_idx - 2}->src_addr_hi = HIGH32(BINGO_CHIPLET_READD(cluster_l1_addr_D_{"pong" if (tile_idx -2) %2 ==0 else "ping"}));
    task_cluster_to_l3_args_D{tile_idx - 2}->src_addr_lo = LOW32(BINGO_CHIPLET_READD(cluster_l1_addr_D_{"pong" if (tile_idx -2) %2 ==0 else "ping"}));
    task_cluster_to_l3_args_D{tile_idx - 2}->dst_addr_hi = HIGH32(BINGO_CHIPLET_READD(output_D{tile_idx - 2}_data_addr_L3));
    task_cluster_to_l3_args_D{tile_idx - 2}->dst_addr_lo = LOW32(BINGO_CHIPLET_READD(output_D{tile_idx - 2}_data_addr_L3));
    task_cluster_to_l3_args_D{tile_idx - 2}->size = ARRAY_SIZE_BYTES(D{tile_idx - 2});
'''
    data_str += [f_string_stre_D]

    # args for check results
    for tile_idx in range(1, M2 + 1):
        f_string_check_results = f'''    // check results for D tile {tile_idx}
        uint64_t D{tile_idx}_addr_golden = chiplet_addr_transform((uint64_t)(uintptr_t)(D{tile_idx}));
    __snax_kernel_check_results_args_t *check_results_args_D{tile_idx} = (__snax_kernel_check_results_args_t *)o1heapAllocate(bingo_get_l3_heap_manager(get_current_chip_id()), sizeof(__snax_kernel_check_results_args_t));
    check_results_args_D{tile_idx}->golden_data_addr = LOW32(BINGO_CHIPLET_READD(D{tile_idx}_addr_golden));
    check_results_args_D{tile_idx}->output_data_addr = LOW32(BINGO_CHIPLET_READD(output_D{tile_idx}_data_addr_L3));
    check_results_args_D{tile_idx}->data_size = ARRAY_SIZE_BYTES(D{tile_idx});
'''
        data_str += [f_string_check_results]

    # register tasks
    
    data_str += ['''// Register task: load B matrix
                     bingo_task_t *task_xdma_l3_to_cluster_B =
        bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                          (uint32_t)(uintptr_t)(task_l3_to_cluster_args_B),
                          task_chip_id, cluster_id);
                          '''
    ]
    for tile_idx in range(1, M2 + 1):
        # load A{tile_idx} from L3 to L1
        data_str += [
            f'''    // Register task: load A tile {tile_idx}
    bingo_task_t *task_xdma_l3_to_cluster_A{tile_idx} =
        bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                          (uint32_t)(uintptr_t)(task_l3_to_cluster_args_A{tile_idx}),
                          task_chip_id, cluster_id);
            '''
        ]
    data_str += ['''// Register task: gemm1
                     bingo_task_t *task_gemm1 =
        bingo_task_create(__snax_kernel_gemm_func_addr,
                          (uint32_t)(uintptr_t)(gemm1_args),
                          task_chip_id, cluster_id);
                          '''
    ]
    for tile_idx in range(2, M2 + 1):
        # gemm{tile_idx}
        data_str += [
            f'''    // Register task: gemm tile {tile_idx}
    bingo_task_t *task_gemm{tile_idx} =
        bingo_task_create(__snax_kernel_start_gemm_and_wait_func_addr,
                          (uint32_t)(uintptr_t)(gemm{tile_idx}_mininal_cfg_args),
                          task_chip_id, cluster_id);
            '''
        ]
    for tile_idx in range(1, M2 + 1):
        # store D{tile_idx} from L1 to L3
        data_str += [
            f'''    // Register task: store D tile {tile_idx}
    bingo_task_t *task_xdma_cluster_to_l3_D{tile_idx} =
        bingo_task_create(__snax_kernel_xdma_1d_copy_func_addr,
                          (uint32_t)(uintptr_t)(task_cluster_to_l3_args_D{tile_idx}),
                          task_chip_id, cluster_id);
            '''
        ]
    # check results
    for tile_idx in range(1, M2 + 1):
        data_str += [
            f'''    // Register task: check results for D tile {tile_idx}
    bingo_task_t *task_check_results_D{tile_idx} =
        bingo_task_create(check_results_func_addr,
                          (uint32_t)(uintptr_t)(check_results_args_D{tile_idx}),
                          task_chip_id, cluster_id);
            '''
        ]
    # dependency graph
    data_str += ['''    // Set task dependencies
    // load A and B matrices
    bingo_task_add_depend(task_xdma_l3_to_cluster_B, task_xdma_l3_to_cluster_A1);
    bingo_task_add_depend(task_gemm1, task_xdma_l3_to_cluster_B);
    '''
    ]

    for tile_idx in range(2, M2 + 1):
        data_str += [
            f'''    // dependencies for tile {tile_idx}
    bingo_task_add_depend(task_xdma_l3_to_cluster_A{tile_idx}, task_xdma_l3_to_cluster_A{tile_idx - 1});
    bingo_task_add_depend(task_gemm{tile_idx}, task_xdma_l3_to_cluster_A{tile_idx});
    bingo_task_add_depend(task_gemm{tile_idx}, task_gemm{tile_idx-1});
    bingo_task_add_depend(task_xdma_cluster_to_l3_D{tile_idx - 1}, task_gemm{tile_idx-1});
    bingo_task_add_depend(task_check_results_D{tile_idx - 1}, task_xdma_cluster_to_l3_D{tile_idx - 1});
            '''
        ]

    # final dependencies
    data_str += [f'''    // final dependencies
    bingo_task_add_depend(task_xdma_cluster_to_l3_D{M2}, task_gemm{M2});
    bingo_task_add_depend(task_check_results_D{M2}, task_xdma_cluster_to_l3_D{M2});
    ''']

    # add to task list
    data_str += ['''    asm volatile("fence" ::: "memory");''']

    for tile_idx in range(1, M2 + 1):
        data_str += [
            f'''    // Add tasks for tile {tile_idx}
    task_list[{(tile_idx -1) * 4 + 0}] = task_xdma_l3_to_cluster_A{tile_idx};
    task_list[{(tile_idx -1) * 4 + 1}] = task_gemm{tile_idx};
    task_list[{(tile_idx -1) * 4 + 2}] = task_xdma_cluster_to_l3_D{tile_idx};
    task_list[{(tile_idx -1) * 4 + 3}] = task_check_results_D{tile_idx};
            '''
        ]
    data_str += [f'''    // Add task for loading B matrix
    task_list[{M2 * 4}] = task_xdma_l3_to_cluster_B;
    ''']

    num_tasks = M2 * 4 + 1  # +1 for loading B matrix
    data_str += [f'''    uint32_t num_tasks = {num_tasks};''']

    return data_str

def main():
    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generating data for kernels")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        required=True,
        help="Select hardware config file kernel",
    )
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    # Load hardware config file
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    # Merge dictionaries (hw overrides param in case of conflicts)
    merged_config = {**param, **hw}

    # Emit header file
    print(emit_workload_file(**merged_config))

if __name__ == "__main__":
    main()
