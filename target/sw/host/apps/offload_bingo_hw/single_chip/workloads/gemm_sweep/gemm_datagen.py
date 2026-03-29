#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

# Add golden model path
from snax_utils import block_gemm_golden_model  # noqa E402

np.random.seed(320)

def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_matmul_data(**kwargs)
    return "\n\n".join(emit_str)

def emit_matmul_data(**kwargs):
    data_str = []

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
    meshRow = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][0]
    tileSize = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][1]
    meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][2]
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
    assert sum([kwargs["addNonZeroC"], kwargs["addZeroC"], kwargs["accumPrevC"]]) == 1, \
        "Only one of addNonZeroC, addZeroC, accumPrevC can be set to 1."

    # L1 storage size check
    L1_SIZE = kwargs.get("l1_size_kb", 512) * 1024
    A_size = M * K * meshRow * tileSize * 1  # int8
    B_size = K * N * tileSize * meshCol * 1  # int8
    D_size = M * N * meshRow * meshCol * 4   # int32
    L1_SIZE_KB = kwargs.get("l1_size_kb", 512)
    total_l1 = A_size + B_size + D_size
    assert total_l1 <= L1_SIZE, (
        f"GEMM data exceeds L1 storage! "
        f"A({A_size/1024:.1f}KB) + B({B_size/1024:.1f}KB) + D({D_size/1024:.1f}KB) = {total_l1/1024:.1f}KB > {L1_SIZE_KB}KB. "
        f"Reduce M={M}, K={K}, or N={N}."
    )

    # test data generation
    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127

    A = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).reshape(-1)

    # Pad A to be multiple of 64 elements for xdma transfer
    length = A.size
    pad_len = (-length) % 64
    if pad_len > 0:
        A_padded = np.pad(A, (0, pad_len), mode='constant', constant_values=0)
    else:
        A_padded = A
    data_str += [format_vector_definition("int8_t", "A", A_padded)]

    B = np.random.randint(B_MIN, B_MAX, size=(K, N, tileSize, meshCol)).reshape(-1)
    data_str += [format_vector_definition("int8_t", "B", B)]

    # addZeroC=1: C is zeros, pass NULL
    C = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
    data_str += [format_scalar_definition("int32_t *", "C", "NULL")]

    subtraction_a = 0
    subtraction_b = 0

    D = block_gemm_golden_model(
        M, K, N,
        meshRow, tileSize, meshCol,
        A, B,
        subtraction_a, subtraction_b,
        C,
    )
    data_str += [format_vector_definition("int32_t", "D", D)]

    return data_str

def main():
    parser = argparse.ArgumentParser(description="Generating data for kernels")
    parser.add_argument(
        "-c", "--cfg",
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

    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    merged_config = {**param, **hw}
    print(emit_header_file(**merged_config))

if __name__ == "__main__":
    main()
