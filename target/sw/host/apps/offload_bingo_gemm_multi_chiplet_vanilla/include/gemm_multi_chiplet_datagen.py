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

def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_matmul_data(**kwargs)
    return "\n\n".join(emit_str)

def emit_matmul_data(**kwargs):
    # -------------------------------------------------------------
    # matmul workload settings
    # -------------------------------------------------------------
    data_str = []

    # arguments
    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]

    data_str += [format_scalar_definition("uint32_t", "M", M)]
    data_str += [format_scalar_definition("uint32_t", "K", K)]
    data_str += [format_scalar_definition("uint32_t", "N", N)]

    array_shape = kwargs["array_shape"]
    data_str += [format_scalar_definition("uint32_t", "array_shape", array_shape)]

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
    data_str += [format_scalar_definition("uint32_t", "meshRow", meshRow)]
    data_str += [format_scalar_definition("uint32_t", "tileSize", tileSize)]
    data_str += [format_scalar_definition("uint32_t", "meshCol", meshCol)]
    
    transposed_A = kwargs["transposed_A"]
    data_str += [format_scalar_definition("uint32_t", "transposed_A", transposed_A)]

    transposed_B = kwargs["transposed_B"]
    data_str += [format_scalar_definition("uint32_t", "transposed_B", transposed_B)]

    data_str += [format_scalar_definition("uint32_t", "addNonZeroC", kwargs["addNonZeroC"])]
    data_str += [format_scalar_definition("uint32_t", "addZeroC", kwargs["addZeroC"])]
    data_str += [format_scalar_definition("uint32_t", "accumPrevC", kwargs["accumPrevC"])]
    assert sum([kwargs["addNonZeroC"], kwargs["addZeroC"], kwargs["accumPrevC"]]) == 1, "Only one of addNonZeroC, addZeroC, accumPrevC can be set to 1."

    if kwargs["accumPrevC"] == 1:
        assert M == 1 and N == 1, "When accumPrevC=1, M, N must be 1."

    # test data generation
    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127
    C_MIN, C_MAX = -2147483648, 2147483647

    A = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).reshape(-1)

    # Pad A to be multiple of 64 elements for xdma transfer
    length = A.size
    pad_len = (-length) % 64   # how many zeros to add to reach next multiple of 64
    if pad_len > 0:
        A_padded = np.pad(A, (0, pad_len), mode='constant', constant_values=0)
    else:
        A_padded = A
    data_str += [format_vector_definition("int8_t", "A1", A_padded)]

    B = np.random.randint(B_MIN, B_MAX, size=(K, N, tileSize, meshCol)).reshape(-1)
    data_str += [format_vector_definition("int8_t", "B", B)]

    if kwargs["addNonZeroC"] == 1:
        C = np.random.randint(C_MIN, C_MAX, size=(M, N, meshRow, meshCol)).reshape(-1)
        data_str += [format_vector_definition("int32_t", "C1", C)]
    elif kwargs["addZeroC"] == 1:
        C = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
        data_str += [format_scalar_definition("int32_t *", "C1", "(int32_t *)NULL")]
    else: # use accumPrevC
        C = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
        data_str += [format_scalar_definition("int32_t *", "C1", "(int32_t *)NULL")]

    if kwargs["transposed_A"] == 1:
        A = A.reshape(M, K, meshRow, tileSize)
        A = A.transpose(0, 1, 3, 2).reshape(-1)
    if kwargs["transposed_B"] == 1:
        B = B.reshape(K, N, tileSize, meshCol)
        B = B.transpose(0, 1, 3, 2).reshape(-1)

    subtraction_a = 0
    subtraction_b = 0
    
    D = block_gemm_golden_model(
        M,
        K,
        N,
        meshRow,
        tileSize,
        meshCol,
        A,
        B,
        subtraction_a,
        subtraction_b,
        C,
    )
    data_str += [format_vector_definition("int32_t", "D1", D)]

    for i in range(1, 4):
        A_i = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).reshape(-1)
        pad_len_i = (-A_i.size) % 64
        if pad_len_i > 0:
            A_i = np.pad(A_i, (0, pad_len_i), mode='constant', constant_values=0)

        # Apply transpose if needed
        if kwargs["transposed_A"] == 1:
            A_i = A_i.reshape(M, K, meshRow, tileSize)
            A_i = A_i.transpose(0, 1, 3, 2).reshape(-1)

        data_str += [format_vector_definition("int8_t", f"A{i+1}", A_i)]

        if kwargs["addNonZeroC"] == 1:
            C_i = np.random.randint(C_MIN, C_MAX, size=(M, N, meshRow, meshCol)).reshape(-1)
            data_str += [format_vector_definition("int32_t", f"C{i+1}", C_i)]
        elif kwargs["addZeroC"] == 1:
            C_i = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
            data_str += [format_scalar_definition("int32_t *", f"C{i+1}", "(int32_t *)NULL")]
        else: # use accumPrevC
            C_i = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
            data_str += [format_scalar_definition("int32_t *", f"C{i+1}", "(int32_t *)NULL")]

        # Compute corresponding D_i
        D_i = block_gemm_golden_model(M, K, N, meshRow, tileSize, meshCol, A_i, B, subtraction_a, subtraction_b, C_i)
        data_str += [format_vector_definition("int32_t", f"D{i+1}", D_i)]

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
    print(emit_header_file(**merged_config))

if __name__ == "__main__":
    main()
