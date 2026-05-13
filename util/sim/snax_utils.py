#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import os

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


MULTICHIP_GEMM_CHIPLETS = (0x00, 0x01, 0x10, 0x11)
MULTICHIP_GEMM_CHECK_BYTES = 64
MULTICHIP_GEMM_RNG_SEED = 320


def load_merged_config(cfg_path, hwcfg_path):
    import hjson

    with open(cfg_path) as f:
        param = hjson.loads(f.read())
    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    return {**param, **hw}


def _pad_to_multichip_gemm_xdma_width(data):
    pad_len = (-data.size) % MULTICHIP_GEMM_CHECK_BYTES
    if pad_len == 0:
        return data
    return np.pad(data, (0, pad_len), mode="constant", constant_values=0)


def _emit_multichip_gemm_settings(kwargs, mesh_row, tile_size, mesh_col):
    from data_utils import format_scalar_definition

    return [
        format_scalar_definition("uint32_t", "M", kwargs["M"]),
        format_scalar_definition("uint32_t", "K", kwargs["K"]),
        format_scalar_definition("uint32_t", "N", kwargs["N"]),
        format_scalar_definition("uint32_t", "array_shape", kwargs["array_shape"]),
        format_scalar_definition("uint32_t", "meshRow", mesh_row),
        format_scalar_definition("uint32_t", "tileSize", tile_size),
        format_scalar_definition("uint32_t", "meshCol", mesh_col),
        format_scalar_definition("uint32_t", "transposed_A", kwargs["transposed_A"]),
        format_scalar_definition("uint32_t", "transposed_B", kwargs["transposed_B"]),
        format_scalar_definition("uint32_t", "addNonZeroC", kwargs["addNonZeroC"]),
        format_scalar_definition("uint32_t", "addZeroC", kwargs["addZeroC"]),
        format_scalar_definition("uint32_t", "accumPrevC", kwargs["accumPrevC"]),
    ]


def emit_multichip_gemm_matmul_data(**kwargs):
    from data_utils import format_vector_definition
    from sim_golden_models import block_gemm_golden_model

    np.random.seed(MULTICHIP_GEMM_RNG_SEED)
    data_str = []

    m = kwargs["M"]
    k = kwargs["K"]
    n = kwargs["N"]
    mesh_row, tile_size, mesh_col = get_gemm_mesh_dims(kwargs)
    data_str += _emit_multichip_gemm_settings(kwargs, mesh_row, tile_size, mesh_col)

    assert (
        sum([kwargs["addNonZeroC"], kwargs["addZeroC"], kwargs["accumPrevC"]]) == 1
    ), "Only one of addNonZeroC, addZeroC, accumPrevC can be set to 1."
    if kwargs["accumPrevC"] == 1:
        assert m == 1 and n == 1, "When accumPrevC=1, M, N must be 1."

    a_min, a_max = -128, 127
    b_min, b_max = -128, 127
    c_min, c_max = -2147483648, 2147483647

    b_data = np.random.randint(b_min, b_max, size=(k, n, tile_size, mesh_col)).reshape(
        -1
    )
    if kwargs["transposed_B"] == 1:
        b_for_model = b_data.reshape(k, n, tile_size, mesh_col)
        b_for_model = b_for_model.transpose(0, 1, 3, 2).reshape(-1)
    else:
        b_for_model = b_data

    a_tiles = []
    d_tiles = []
    for _ in MULTICHIP_GEMM_CHIPLETS:
        a_data = np.random.randint(
            a_min, a_max, size=(m, k, mesh_row, tile_size)
        ).reshape(-1)
        if kwargs["transposed_A"] == 1:
            a_for_model = a_data.reshape(m, k, mesh_row, tile_size)
            a_for_model = a_for_model.transpose(0, 1, 3, 2).reshape(-1)
        else:
            a_for_model = a_data
        a_tiles.append(_pad_to_multichip_gemm_xdma_width(a_for_model))

        if kwargs["addNonZeroC"] == 1:
            c_data = np.random.randint(
                c_min, c_max, size=(m, n, mesh_row, mesh_col)
            ).reshape(-1)
        else:
            c_data = np.zeros((m, n, mesh_row, mesh_col), dtype=np.int32).reshape(-1)

        d_tiles.append(
            block_gemm_golden_model(
                m,
                k,
                n,
                mesh_row,
                tile_size,
                mesh_col,
                a_for_model,
                b_for_model,
                0,
                0,
                c_data,
            )
        )

    a_concat = np.concatenate(a_tiles)
    d_concat = np.concatenate(d_tiles)

    out_dir = kwargs.get("out_dir", "./build/")
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "mempool.bin")

    with open(bin_path, "wb") as f:
        a_concat.astype(np.uint8).tofile(f)
        b_data.astype(np.uint8).tofile(f)
        d_concat.astype(np.uint32).tofile(f)

    data_str += [format_vector_definition("int8_t", "A_data_L3", a_concat)]
    data_str += [format_vector_definition("int8_t", "B_data_L3", b_data)]
    data_str += [
        format_vector_definition("int32_t", "D_data_L3", d_concat.astype(np.int32))
    ]
    return data_str


def emit_multichip_gemm_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_multichip_gemm_matmul_data(**kwargs)
    return "\n\n".join(emit_str)


def run_multichip_gemm_datagen():
    import argparse
    import pathlib

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
    merged_config = load_merged_config(args.cfg, args.hwcfg)
    print(emit_multichip_gemm_header_file(**merged_config))


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
