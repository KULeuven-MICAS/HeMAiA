#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import os
import random
import sys

import numpy as np

# Grouped util/sim layout: this module lives in util/sim/gemm/; its (lazy)
# `from data_utils import ...` / `from sim_golden_models import ...` need the
# sibling common/ dir on sys.path. Add common/ + the other engine subdirs.
_USIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # util/sim
for _sub in ("common", "gemm", "xdma", "ara"):
    _p = os.path.join(_USIM, _sub)
    if _p not in sys.path:
        sys.path.append(_p)


def _get_acc_cfg(cfg):
    return cfg["snax_versacore_core_template"]["snax_acc_cfg"][0]


def get_gemm_mesh_dims(cfg):
    data_type = cfg.get("data_type", 0)
    assert data_type in [0], "Only data_type - is supported in current hardware"
    unrolling = _get_acc_cfg(cfg)["snax_versacore_spatial_unrolling"][data_type][
        cfg["array_shape"]
    ]
    return unrolling[0], unrolling[1], unrolling[2]


def _bytes_for_elements(num_elements, elem_bits):
    total_bits = num_elements * elem_bits
    assert total_bits % 8 == 0, "Packed GEMM element count must be byte-aligned."
    return int(total_bits // 8)


def _signed_int_range(bits):
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def _signed_to_unsigned(val, bit_width):
    min_val, max_val = _signed_int_range(bit_width)
    if val < min_val or val > max_val:
        raise ValueError(f"Value {val} out of range for signed {bit_width}-bit integer")
    return val & ((1 << bit_width) - 1)


def _pack_signed_nbit(values, bit_width, pack_per_byte=2):

    if len(values) % pack_per_byte != 0:
        raise ValueError(
            f"Length {len(values)} must be a multiple of {pack_per_byte} for int{bit_width} packing"
        )
    packed = []
    for i in range(0, len(values), pack_per_byte):
        byte = 0
        for j in range(pack_per_byte):
            byte |= _signed_to_unsigned(int(values[i + j]), bit_width) << (
                j * bit_width
            )
        if byte >= 128:
            byte -= 256
        packed.append(byte)
    return np.array(packed, dtype=np.int8)


def _gemm_operand_widths(merged):
    acc = _get_acc_cfg(merged)
    data_type = merged.get("data_type", 0)
    a_len = (
        4
        if merged.get("int4_a_enable", 0)
        else acc["snax_versacore_input_a_element_width"][data_type]
    )
    b_len = (
        4
        if merged.get("int4_b_enable", 0)
        else acc["snax_versacore_input_b_element_width"][data_type]
    )
    c_len = 0
    # only have actual C when addNonZeroC is enabled
    if merged.get("addNonZeroC", 0) == 1:
        c_len = acc["snax_versacore_input_c_element_width"][data_type]
    if merged.get("quantization_enable", 0):
        d_len = 8
    elif merged.get("int32tofp16_enable", 0):
        d_len = 16
    else:
        d_len = acc["snax_versacore_output_d_element_width"][data_type]
    return a_len, b_len, c_len, d_len


def _generate_quantization_params():
    shift_i = random.randint(24, 48)
    multiplier_i = 1140768826 * 2 // (2 ** (48 - shift_i))
    input_zp_i = random.randint(-10000000, 10000000) // (2 ** (48 - shift_i))
    output_zp_i = random.randint(-3, 3)
    return {
        "shift_i": shift_i,
        "multiplier_i": multiplier_i,
        "input_zp_i": input_zp_i,
        "output_zp_i": output_zp_i,
    }


def define_gemm_workload_params(cfg_path, hwcfg_path):
    import hjson

    with open(cfg_path) as f:
        param = hjson.loads(f.read())
    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw, **_generate_quantization_params()}
    M, K, N = merged["M"], merged["K"], merged["N"]
    meshRow, tileSize, meshCol = get_gemm_mesh_dims(merged)
    a_len, b_len, c_len, d_len = _gemm_operand_widths(merged)

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
        "quantization_enable": merged.get("quantization_enable", 0),
        "shift_i": merged["shift_i"],
        "multiplier_i": merged["multiplier_i"],
        "input_zp_i": merged["input_zp_i"],
        "output_zp_i": merged["output_zp_i"],
        "int32tofp16_enable": merged.get("int32tofp16_enable", 0),
        "int4_a_enable": merged.get("int4_a_enable", 0),
        "int4_b_enable": merged.get("int4_b_enable", 0),
    }
    params.update(
        A_size=_bytes_for_elements(M * K * meshRow * tileSize, a_len),
        B_size=_bytes_for_elements(K * N * tileSize * meshCol, b_len),
        C_size=_bytes_for_elements(M * N * meshRow * meshCol, c_len),
        D_size=_bytes_for_elements(M * N * meshRow * meshCol, d_len),
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


def generate_gemm_test_data(**kwargs):

    from sim_golden_models import (
        block_gemm_golden_model,
        int32_to_fp16_golden,
        postprocessing_simd_golden_model_V3,
    )

    if "shift_i" not in kwargs:
        kwargs = {**kwargs, **_generate_quantization_params()}

    np.random.seed(320)

    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]
    meshRow, tileSize, meshCol = get_gemm_mesh_dims(kwargs)
    a_len, b_len, _, d_len = _gemm_operand_widths(kwargs)
    int4_a_enable = kwargs.get("int4_a_enable", 0)
    int4_b_enable = kwargs.get("int4_b_enable", 0)
    quantization_enable = kwargs.get("quantization_enable", 0)
    int32tofp16_enable = kwargs.get("int32tofp16_enable", 0)
    assert (
        quantization_enable + int32tofp16_enable <= 1
    ), "Only one of quantization and int32 to fp16 conversion can be enabled."
    assert (
        sum([kwargs["addNonZeroC"], kwargs["addZeroC"], kwargs["accumPrevC"]]) == 1
    ), "Only one of addNonZeroC, addZeroC, accumPrevC can be set to 1."

    if kwargs["accumPrevC"] == 1:
        assert M == 1 and N == 1, "When accumPrevC=1, M, N must be 1."

    A_MIN, A_MAX = _signed_int_range(a_len)
    B_MIN, B_MAX = _signed_int_range(b_len)
    C_MIN, C_MAX = -2147483648, 2147483647

    A = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).reshape(-1)
    if int4_a_enable:
        # The A reader's sparse interconnect wires read-port i only to banks of
        # parity (i % granularity_a); the device kernel rounds each int4 A K-tile
        # stride UP to granularity_a banks (offload_hw_kernels/gemm.h). Pad each
        # (meshRow*tileSize) K-tile here to that same width so the emitted byte
        # offsets match exactly what the streamer reads — the pad lands in the
        # skipped bank and is never consumed as A data. (For a tile that already
        # spans a granularity_a multiple of banks the pad is empty; only int4 A
        # at shape 1 actually grows.)
        GRANULARITY_A = 2            # banks; mirrors hwcfg granularity_a
        BANK_BITS = 64
        A_ELEM_BITS = 4
        elems_per_bank = BANK_BITS // A_ELEM_BITS       # 16 int4 per 64-bit bank
        tile_elems = meshRow * tileSize
        tile_banks = (tile_elems * A_ELEM_BITS + BANK_BITS - 1) // BANK_BITS
        tile_banks_aligned = (
            (tile_banks + GRANULARITY_A - 1) // GRANULARITY_A * GRANULARITY_A
        )
        padded_tile_elems = tile_banks_aligned * elems_per_bank
        A_tiles = A.reshape(M * K, tile_elems)
        A_tiles = np.pad(
            A_tiles, ((0, 0), (0, padded_tile_elems - tile_elems)),
            mode="constant", constant_values=0,
        )
        A_padded = A_tiles.reshape(-1)
        A_padded = np.pad(A_padded, (0, (-A_padded.size) % 64),
                          mode="constant", constant_values=0)
        A_to_emit = _pack_signed_nbit(A_padded, bit_width=4, pack_per_byte=2)
    else:
        A_padded = np.pad(A, (0, (-A.size) % 64), mode="constant",
                          constant_values=0)
        A_to_emit = A_padded

    B = np.random.randint(B_MIN, B_MAX, size=(K, N, tileSize, meshCol)).reshape(-1)
    B_to_emit = (
        _pack_signed_nbit(B, bit_width=4, pack_per_byte=2)
        if int4_b_enable
        else B
    )

    if kwargs["addNonZeroC"] == 1:
        C = np.random.randint(C_MIN, C_MAX, size=(M, N, meshRow, meshCol)).reshape(-1)
    else:
        C = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)

    A_for_golden = A
    B_for_golden = B
    if kwargs["transposed_A"] == 1:
        A_for_golden = A.reshape(M, K, meshRow, tileSize)
        A_for_golden = A_for_golden.transpose(0, 1, 3, 2).reshape(-1)
    if kwargs["transposed_B"] == 1:
        B_for_golden = B.reshape(K, N, tileSize, meshCol)
        B_for_golden = B_for_golden.transpose(0, 1, 3, 2).reshape(-1)

    D = block_gemm_golden_model(
        M,
        K,
        N,
        meshRow,
        tileSize,
        meshCol,
        A_for_golden,
        B_for_golden,
        0,
        0,
        C,
    )

    if quantization_enable:
        assert kwargs["multiplier_i"] > 0 and kwargs["shift_i"] > 0, (
            "quantization_enable requires positive multiplier_i and shift_i"
        )
        max_in_range = 8388608
        D = np.array(
            [
                postprocessing_simd_golden_model_V3(
                    elem,
                    kwargs["input_zp_i"],
                    kwargs["output_zp_i"],
                    kwargs["shift_i"],
                    max_in_range,
                    -max_in_range,
                    True,
                    kwargs["multiplier_i"],
                )
                for elem in D
            ],
            dtype=np.int8,
        )
    elif int32tofp16_enable:
        D = np.array([int32_to_fp16_golden(elem) for elem in D], dtype=np.uint16)
    else:
        assert d_len == 32
        D = D.astype(np.int32)

    return {
        "config": kwargs,
        "M": M,
        "K": K,
        "N": N,
        "meshRow": meshRow,
        "tileSize": tileSize,
        "meshCol": meshCol,
        "int4_a_enable": int4_a_enable,
        "int4_b_enable": int4_b_enable,
        "quantization_enable": quantization_enable,
        "int32tofp16_enable": int32tofp16_enable,
        "d_len": d_len,
        "A": A_to_emit,
        "B": B_to_emit,
        "C": C,
        "D": D,
    }


def emit_gemm_header_file(**kwargs):
    from data_utils import format_scalar_definition, format_vector_definition

    gemm_data = generate_gemm_test_data(**kwargs)
    cfg = gemm_data["config"]
    emit_str = ["#include <stdint.h>"]
    data_str = []

    data_str += [format_scalar_definition("uint32_t", "M", gemm_data["M"])]
    data_str += [format_scalar_definition("uint32_t", "K", gemm_data["K"])]
    data_str += [format_scalar_definition("uint32_t", "N", gemm_data["N"])]

    array_shape = cfg["array_shape"]
    data_str += [format_scalar_definition("uint32_t", "array_shape", array_shape)]
    data_type = cfg.get("data_type", 0)
    data_str += [format_scalar_definition("uint32_t", "data_type", data_type)]

    data_str += [format_scalar_definition("uint32_t", "meshRow", gemm_data["meshRow"])]
    data_str += [format_scalar_definition("uint32_t", "tileSize", gemm_data["tileSize"])]
    data_str += [format_scalar_definition("uint32_t", "meshCol", gemm_data["meshCol"])]

    data_str += [
        format_scalar_definition(
            "uint32_t", "int4_a_enable", gemm_data["int4_a_enable"]
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "int4_b_enable", gemm_data["int4_b_enable"]
        )
    ]
    data_str += [
        format_scalar_definition(
            "uint32_t", "quantization_enable", gemm_data["quantization_enable"]
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t", "int32tofp16_enable", gemm_data["int32tofp16_enable"]
        )
    ]

    transposed_A = cfg["transposed_A"]
    data_str += [format_scalar_definition("uint32_t", "transposed_A", transposed_A)]

    transposed_B = cfg["transposed_B"]
    data_str += [format_scalar_definition("uint32_t", "transposed_B", transposed_B)]

    data_str += [
        format_scalar_definition("uint32_t", "addNonZeroC", cfg["addNonZeroC"])
    ]
    data_str += [format_scalar_definition("uint32_t", "addZeroC", cfg["addZeroC"])]
    data_str += [
        format_scalar_definition("uint32_t", "accumPrevC", cfg["accumPrevC"])
    ]

    data_str += [format_vector_definition("int8_t", "A", gemm_data["A"])]
    data_str += [format_vector_definition("int8_t", "B", gemm_data["B"])]

    if cfg["addNonZeroC"] == 1:
        data_str += [format_vector_definition("int32_t", "C", gemm_data["C"])]
    else:
        data_str += [format_scalar_definition("int32_t *", "C", "(int32_t *)NULL")]

    shift_i = cfg["shift_i"]
    multiplier_i = cfg["multiplier_i"]
    input_zp_i = cfg["input_zp_i"]
    output_zp_i = cfg["output_zp_i"]
    data_str += [format_scalar_definition("uint32_t", "shift_i", shift_i)]
    data_str += [format_scalar_definition("uint32_t", "multiplier_i", multiplier_i)]
    data_str += [format_scalar_definition("int32_t", "input_zp_i", input_zp_i)]
    data_str += [format_scalar_definition("int32_t", "output_zp_i", output_zp_i)]

    if gemm_data["quantization_enable"]:
        data_str += [format_vector_definition("int8_t", "D", gemm_data["D"])]
    elif gemm_data["int32tofp16_enable"]:
        data_str += [format_vector_definition("uint16_t", "D", gemm_data["D"])]
    else:
        assert gemm_data["d_len"] == 32
        data_str += [format_vector_definition("int32_t", "D", gemm_data["D"])]

    emit_str += data_str
    return "\n\n".join(emit_str)


def emit_ksplit_gemm_header_file(workload_name, **kwargs):

    from data_utils import format_scalar_definition
    from sim_golden_models import block_gemm_golden_model

    lines = ["#include <stdint.h>"]

    array_shape = kwargs["array_shape"]
    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]
    k_split = kwargs["k_split"]
    if k_split <= 0:
        raise ValueError(f"k_split ({k_split}) must be positive")
    if k_split != 4:
        raise ValueError(f"{workload_name} expects k_split=4, got {k_split}")
    if K % k_split != 0:
        raise ValueError(f"K ({K}) must be divisible by k_split ({k_split})")

    data_type = kwargs.get("data_type", 0)
    meshRow, tileSize, meshCol = get_gemm_mesh_dims(kwargs)

    K_tile = K // k_split
    A_tile_bytes = M * K_tile * meshRow * tileSize
    B_tile_bytes = K_tile * N * tileSize * meshCol
    D_num_elements = M * N * meshRow * meshCol
    D_bytes = D_num_elements * 4

    lines.append(f"// K-split GEMM: M={M} K={K} N={N} k_split={k_split}")
    lines.append(f"// K_tile={K_tile} mesh={meshRow}x{tileSize}x{meshCol}")

    np.random.seed(42)
    A_full = np.random.randint(-128, 127, size=(M, K, meshRow, tileSize)).astype(
        np.int8
    )
    B_full = np.random.randint(-128, 127, size=(K, N, tileSize, meshCol)).astype(
        np.int8
    )

    A_tiles = []
    B_tiles = []
    partial_Ds = []
    C_zero = np.zeros(D_num_elements, dtype=np.int32)

    for k_idx in range(k_split):
        ks = k_idx * K_tile
        ke = ks + K_tile
        A_k = A_full[:, ks:ke, :, :].reshape(-1).copy()
        B_k = B_full[ks:ke, :, :, :].reshape(-1).copy()
        D_k = block_gemm_golden_model(
            M,
            K_tile,
            N,
            meshRow,
            tileSize,
            meshCol,
            A_k,
            B_k,
            0,
            0,
            C_zero.copy(),
        )
        A_tiles.append(A_k.astype(np.int8))
        B_tiles.append(B_k.astype(np.int8))
        partial_Ds.append(D_k.astype(np.int32))

    mempool_arrays = []
    mempool_arrays.extend(A_tiles)
    mempool_arrays.extend(B_tiles)
    mempool_arrays.extend(partial_Ds)

    running_sum = partial_Ds[0].copy()
    for k_idx in range(1, k_split):
        running_sum = (
            running_sum.astype(np.int64) + partial_Ds[k_idx].astype(np.int64)
        ).astype(np.int32)
        mempool_arrays.append(running_sum.astype(np.int32))

    combined_scale = np.float32(0.0001)
    golden_fp32_D = (running_sum.astype(np.float32) * combined_scale).astype(
        np.float32
    )
    mempool_arrays.append(golden_fp32_D.astype(np.float32))
    mempool_arrays.append(np.array([combined_scale], dtype=np.float32))

    out_dir = kwargs.get("out_dir", "./build/")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mempool.bin"), "wb") as f:
        for array in mempool_arrays:
            array.tofile(f)

    for name, val in [
        ("A_tile_bytes", A_tile_bytes),
        ("B_tile_bytes", B_tile_bytes),
        ("D_bytes", D_bytes),
        ("D_num_elements", D_num_elements),
        ("k_split", k_split),
        ("M", M),
        ("K", K),
        ("N", N),
        ("K_tile", K_tile),
        ("meshRow", meshRow),
        ("tileSize", tileSize),
        ("meshCol", meshCol),
        ("array_shape", array_shape),
        ("data_type", data_type),
    ]:
        lines.append(format_scalar_definition("uint32_t", name, val))

    return "\n\n".join(lines) + "\n"


def align_wide_addr(addr, alignment=64):
    if addr % alignment:
        addr = ((addr // alignment) + 1) * alignment
    return addr
