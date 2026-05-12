#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import numpy as np


def get_gemm_mesh_dims(cfg):
    unrolling = cfg["snax_versacore_core_template"]["snax_acc_cfg"][0][
        "snax_versacore_spatial_unrolling"
    ][0][cfg["array_shape"]]
    return unrolling[0], unrolling[1], unrolling[2]


def define_gemm_workload_params(cfg_path, hwcfg_path):
    import hjson

    with open(cfg_path) as f:
        param = hjson.loads(f.read())
    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}
    M, K, N = merged["M"], merged["K"], merged["N"]
    meshRow, tileSize, meshCol = get_gemm_mesh_dims(merged)

    params = {
        "M": M,
        "K": K,
        "N": N,
        "addZeroC": merged.get("addZeroC", 1),
        "addNonZeroC": merged.get("addNonZeroC", 0),
        "accumPrevC": merged.get("accumPrevC", 0),
        "meshRow": meshRow,
        "tileSize": tileSize,
        "meshCol": meshCol,
        "arrayShapeIdx": merged["array_shape"],
        "transposeA": merged.get("transposed_A", 0),
        "transposeB": merged.get("transposed_B", 0),
    }
    params.update(
        A_size=M * K * meshRow * tileSize,
        B_size=K * N * tileSize * meshCol,
        C_size=M * N * meshRow * meshCol * 4,
        D_size=M * N * meshRow * meshCol * 4,
    )
    return params, merged


def emit_gemm_header_file(**kwargs):
    from data_utils import format_scalar_definition, format_vector_definition
    from sim_golden_models import block_gemm_golden_model

    np.random.seed(320)

    emit_str = ["#include <stdint.h>"]
    data_str = []

    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]

    data_str += [format_scalar_definition("uint32_t", "M", M)]
    data_str += [format_scalar_definition("uint32_t", "K", K)]
    data_str += [format_scalar_definition("uint32_t", "N", N)]

    array_shape = kwargs["array_shape"]
    data_str += [format_scalar_definition("uint32_t", "array_shape", array_shape)]

    meshRow, tileSize, meshCol = get_gemm_mesh_dims(kwargs)
    data_str += [format_scalar_definition("uint32_t", "meshRow", meshRow)]
    data_str += [format_scalar_definition("uint32_t", "tileSize", tileSize)]
    data_str += [format_scalar_definition("uint32_t", "meshCol", meshCol)]

    transposed_A = kwargs["transposed_A"]
    data_str += [format_scalar_definition("uint32_t", "transposed_A", transposed_A)]

    transposed_B = kwargs["transposed_B"]
    data_str += [format_scalar_definition("uint32_t", "transposed_B", transposed_B)]

    data_str += [
        format_scalar_definition("uint32_t", "addNonZeroC", kwargs["addNonZeroC"])
    ]
    data_str += [format_scalar_definition("uint32_t", "addZeroC", kwargs["addZeroC"])]
    data_str += [
        format_scalar_definition("uint32_t", "accumPrevC", kwargs["accumPrevC"])
    ]
    assert (
        sum([kwargs["addNonZeroC"], kwargs["addZeroC"], kwargs["accumPrevC"]]) == 1
    ), "Only one of addNonZeroC, addZeroC, accumPrevC can be set to 1."

    if kwargs["accumPrevC"] == 1:
        assert M == 1 and N == 1, "When accumPrevC=1, M, N must be 1."

    # test data generation
    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127
    C_MIN, C_MAX = -2147483648, 2147483647

    A = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).reshape(-1)

    # Pad A to be multiple of 64 elements for xdma transfer
    length = A.size
    pad_len = (-length) % 64
    if pad_len > 0:
        A_padded = np.pad(A, (0, pad_len), mode="constant", constant_values=0)
    else:
        A_padded = A
    data_str += [format_vector_definition("int8_t", "A", A_padded)]

    B = np.random.randint(B_MIN, B_MAX, size=(K, N, tileSize, meshCol)).reshape(-1)
    data_str += [format_vector_definition("int8_t", "B", B)]

    if kwargs["addNonZeroC"] == 1:
        C = np.random.randint(C_MIN, C_MAX, size=(M, N, meshRow, meshCol)).reshape(-1)
        data_str += [format_vector_definition("int32_t", "C", C)]
    elif kwargs["addZeroC"] == 1:
        C = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
        data_str += [format_scalar_definition("int32_t *", "C", "(int32_t *)NULL")]
    else:  # use accumPrevC
        C = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
        data_str += [format_scalar_definition("int32_t *", "C", "(int32_t *)NULL")]

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

    emit_str += data_str
    return "\n\n".join(emit_str)


def align_wide_addr(addr, alignment=64):
    if addr % alignment:
        addr = ((addr // alignment) + 1) * alignment
    return addr
