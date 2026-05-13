#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import numpy as np
import random


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
    return total_bits // 8


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


def emit_gemm_header_file(**kwargs):
    from data_utils import format_scalar_definition, format_vector_definition
    from sim_golden_models import (
        block_gemm_golden_model,
        int32_to_fp16_golden,
        postprocessing_simd_golden_model_V3,
    )

    if "shift_i" not in kwargs:
        kwargs = {**kwargs, **_generate_quantization_params()}

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
    data_type = kwargs.get("data_type", 0)
    data_str += [format_scalar_definition("uint32_t", "data_type", data_type)]

    meshRow, tileSize, meshCol = get_gemm_mesh_dims(kwargs)
    data_str += [format_scalar_definition("uint32_t", "meshRow", meshRow)]
    data_str += [format_scalar_definition("uint32_t", "tileSize", tileSize)]
    data_str += [format_scalar_definition("uint32_t", "meshCol", meshCol)]

    a_len, b_len, _, d_len = _gemm_operand_widths(kwargs)
    int4_a_enable = kwargs.get("int4_a_enable", 0)
    int4_b_enable = kwargs.get("int4_b_enable", 0)
    quantization_enable = kwargs.get("quantization_enable", 0)
    int32tofp16_enable = kwargs.get("int32tofp16_enable", 0)
    assert (
        quantization_enable + int32tofp16_enable <= 1
    ), "Only one of quantization and int32 to fp16 conversion can be enabled."

    data_str += [
        format_scalar_definition("uint32_t", "int4_a_enable", int4_a_enable)
    ]
    data_str += [
        format_scalar_definition("uint32_t", "int4_b_enable", int4_b_enable)
    ]
    data_str += [
        format_scalar_definition("uint32_t", "quantization_enable", quantization_enable)
    ]
    data_str += [
        format_scalar_definition("int32_t", "int32tofp16_enable", int32tofp16_enable)
    ]

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
    A_MIN, A_MAX = _signed_int_range(a_len)
    B_MIN, B_MAX = _signed_int_range(b_len)
    C_MIN, C_MAX = -2147483648, 2147483647

    A = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).reshape(-1)

    # Pad A to be multiple of 64 elements for xdma transfer
    length = A.size
    pad_len = (-length) % 64
    if pad_len > 0:
        A_padded = np.pad(A, (0, pad_len), mode="constant", constant_values=0)
    else:
        A_padded = A
    if int4_a_enable:
        A_to_emit = _pack_signed_nbit(A_padded, bit_width=4, pack_per_byte=2)
    else:
        A_to_emit = A_padded
    data_str += [format_vector_definition("int8_t", "A", A_to_emit)]

    B = np.random.randint(B_MIN, B_MAX, size=(K, N, tileSize, meshCol)).reshape(-1)
    if int4_b_enable:
        B_to_emit = _pack_signed_nbit(B, bit_width=4, pack_per_byte=2)
    else:
        B_to_emit = B
    data_str += [format_vector_definition("int8_t", "B", B_to_emit)]

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
    shift_i = kwargs["shift_i"]
    multiplier_i = kwargs["multiplier_i"]
    input_zp_i = kwargs["input_zp_i"]
    output_zp_i = kwargs["output_zp_i"]
    data_str += [format_scalar_definition("uint32_t", "shift_i", shift_i)]
    data_str += [format_scalar_definition("uint32_t", "multiplier_i", multiplier_i)]
    data_str += [format_scalar_definition("int32_t", "input_zp_i", input_zp_i)]
    data_str += [format_scalar_definition("int32_t", "output_zp_i", output_zp_i)]

    if quantization_enable:
        assert multiplier_i > 0 and shift_i > 0, (
            "quantization_enable requires positive multiplier_i and shift_i"
        )
        max_in_range = 8388608
        D = np.array(
            [
                postprocessing_simd_golden_model_V3(
                    elem,
                    input_zp_i,
                    output_zp_i,
                    shift_i,
                    max_in_range,
                    -max_in_range,
                    True,
                    multiplier_i,
                )
                for elem in D
            ],
            dtype=np.int8,
        )
        data_str += [format_vector_definition("int8_t", "D", D)]
    elif int32tofp16_enable:
        D = np.array([int32_to_fp16_golden(elem) for elem in D], dtype=np.uint16)
        data_str += [format_vector_definition("uint16_t", "D", D)]
    else:
        assert d_len == 32
        data_str += [format_vector_definition("int32_t", "D", D)]

    emit_str += data_str
    return "\n\n".join(emit_str)


def align_wide_addr(addr, alignment=64):
    if addr % alignment:
        addr = ((addr // alignment) + 1) * alignment
    return addr
