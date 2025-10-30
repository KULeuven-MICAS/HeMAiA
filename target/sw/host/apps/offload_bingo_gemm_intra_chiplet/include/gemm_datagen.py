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

    if array_shape == 0:
        meshRow = 32
        tileSize = 4
        meshCol = 32
    elif array_shape == 1:
        meshRow = 1
        tileSize =32
        meshCol = 16
    else:
        raise ValueError("Unsupported array shape!")

    transposed_A = kwargs["transposed_A"]
    data_str += [format_scalar_definition("uint32_t", "transposed_A", transposed_A)]

    transposed_B = kwargs["transposed_B"]
    data_str += [format_scalar_definition("uint32_t", "transposed_B", transposed_B)]

    addNewC = kwargs["addNewC"]
    data_str += [format_scalar_definition("uint32_t", "addNewC", addNewC)]
    assert addNewC == 1, "Only support addNewC = 1 for now!"

    # test data generation
    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127
    C_MIN, C_MAX = -2147483648, 2147483647
    A = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).reshape(-1)
    data_str += [format_vector_definition("int8_t", "A", A)]
    B = np.random.randint(B_MIN, B_MAX, size=(K, N, tileSize, meshCol)).reshape(-1)
    data_str += [format_vector_definition("int8_t", "B", B)]

    if addNewC:
        C = np.random.randint(C_MIN, C_MAX, size=(M, N, meshRow, meshCol)).reshape(-1)
        data_str += [format_vector_definition("int32_t", "C", C)]
    else:
        C = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
        data_str += [format_vector_definition("int32_t", "C", C)]

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
    data_str += [format_vector_definition("int32_t", "D", D)]

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
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    # Emit header file
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
