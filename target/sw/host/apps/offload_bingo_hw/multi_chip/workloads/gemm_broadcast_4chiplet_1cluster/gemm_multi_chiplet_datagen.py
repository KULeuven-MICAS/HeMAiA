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
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition, format_scalar_define, format_vector_define  # noqa E402

# # Add golden model path
from snax_utils import block_gemm_golden_model # noqa E402

np.random.seed(320)

def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_matmul_data(**kwargs)
    return "\n\n".join(emit_str)

def emit_matmul_data(**kwargs):
    data_str = []

    # -------------------------------------------------------------
    # matmul workload settings
    # -------------------------------------------------------------
    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]

    data_str += [format_scalar_definition("uint32_t", "M", M)]
    data_str += [format_scalar_definition("uint32_t", "K", K)]
    data_str += [format_scalar_definition("uint32_t", "N", N)]

    array_shape = kwargs["array_shape"]
    data_str += [format_scalar_definition("uint32_t", "array_shape", array_shape)]
    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    data_type = 0  # int8 data type

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
    transposed_B = kwargs["transposed_B"]

    data_str += [
        format_scalar_definition("uint32_t", "transposed_A", transposed_A),
        format_scalar_definition("uint32_t", "transposed_B", transposed_B),
        format_scalar_definition("uint32_t", "addNonZeroC", kwargs["addNonZeroC"]),
        format_scalar_definition("uint32_t", "addZeroC", kwargs["addZeroC"]),
        format_scalar_definition("uint32_t", "accumPrevC", kwargs["accumPrevC"]),
    ]

    assert sum([kwargs["addNonZeroC"], kwargs["addZeroC"], kwargs["accumPrevC"]]) == 1, \
        "Only one of addNonZeroC, addZeroC, accumPrevC can be set to 1."

    if kwargs["accumPrevC"] == 1:
        assert M == 1 and N == 1, "When accumPrevC=1, M, N must be 1."

    # -------------------------------------------------------------
    # Data generation
    # -------------------------------------------------------------
    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127
    C_MIN, C_MAX = -2147483648, 2147483647

    # Common B
    B = np.random.randint(B_MIN, B_MAX, size=(K, N, tileSize, meshCol)).reshape(-1)

    A_all, D_all = [], []

    for i in range(1, 5):  # A1..A4
        A_i = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).reshape(-1)
        pad_len = (-A_i.size) % 64
        
        if kwargs["transposed_A"] == 1:
            A_i = A_i.reshape(M, K, meshRow, tileSize)
            A_i = A_i.transpose(0, 1, 3, 2).reshape(-1)

        if pad_len > 0:
            A_padded_i = np.pad(A_i, (0, pad_len), mode='constant', constant_values=0)
        else:
            A_padded_i = A_i

        if kwargs["transposed_B"] == 1:
            B_reshaped = B.reshape(K, N, tileSize, meshCol)
            B_transposed = B_reshaped.transpose(0, 1, 3, 2).reshape(-1)
            B_i = B_transposed
        else:
            B_i = B

        A_all.append(A_padded_i)

        if kwargs["addNonZeroC"] == 1:
            C_i = np.random.randint(C_MIN, C_MAX, size=(M, N, meshRow, meshCol)).reshape(-1)
        else:
            C_i = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)

        subtraction_a = 0
        subtraction_b = 0
        # D1..D4
        D_i = block_gemm_golden_model(M, K, N, meshRow, tileSize, meshCol,
                                      A_i, B_i, subtraction_a, subtraction_b, C_i)
        D_all.append(D_i)

    # Concatenate A1..A4 and D1..D4
    A_concat = np.concatenate(A_all)
    D_concat = np.concatenate(D_all)

    # -------------------------------------------------------------
    # Write A, B, D continuously to one file
    # -------------------------------------------------------------
    out_dir = kwargs.get("out_dir", "./build/")
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "mempool.bin")

    # Ensure correct data types for binary layout
    A_bytes = A_concat.astype(np.uint8)
    B_bytes = B.astype(np.uint8)
    D_bytes = D_concat.astype(np.uint32)

    with open(bin_path, "wb") as f:
        A_bytes.tofile(f)   # 1 byte per element
        B_bytes.tofile(f)   # 1 byte per element
        D_bytes.tofile(f)   # 4 bytes per element

    data_str += [format_vector_definition("int8_t", "A_data_L3", A_concat)]
    data_str += [format_vector_definition("int8_t", "B_data_L3", B)]
    data_str += [format_vector_definition("int32_t", "D_data_L3", D_concat.astype(np.int32))]

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
