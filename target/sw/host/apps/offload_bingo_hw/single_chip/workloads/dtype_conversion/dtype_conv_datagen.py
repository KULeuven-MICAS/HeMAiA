#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Data generation for dtype conversion round-trip test:
#   FP32 input -> quantize INT8 -> GEMM (INT8xINT8->INT32) -> dequantize FP32

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import struct

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import block_gemm_golden_model  # noqa E402

np.random.seed(42)


def quantize_symmetric_int8(tensor):
    """Per-tensor symmetric INT8 quantization (matches quantize.py)."""
    abs_max = np.max(np.abs(tensor))
    if abs_max < 1e-10:
        abs_max = 1e-10
    scale = abs_max / 127.0
    quantized = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
    return quantized, float(scale)


def float_to_hex(f):
    """Convert float32 to its uint32 hex representation for C initializer."""
    return struct.unpack('<I', struct.pack('<f', f))[0]


def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_dtype_conv_data(**kwargs)
    return "\n\n".join(emit_str)


def emit_dtype_conv_data(**kwargs):
    data_str = []

    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]
    array_shape = kwargs["array_shape"]

    data_type = 0  # int8
    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][0]
    tileSize = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][1]
    meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][2]

    data_str += [format_scalar_definition("uint32_t", "M", M)]
    data_str += [format_scalar_definition("uint32_t", "K", K)]
    data_str += [format_scalar_definition("uint32_t", "N", N)]
    data_str += [format_scalar_definition("uint32_t", "array_shape", array_shape)]
    data_str += [format_scalar_definition("uint32_t", "meshRow", meshRow)]
    data_str += [format_scalar_definition("uint32_t", "tileSize", tileSize)]
    data_str += [format_scalar_definition("uint32_t", "meshCol", meshCol)]

    # --- FP32 input tensors (simulating RMSNorm / activation output) ---
    # Shape follows VersaCore blocked layout: A[M,K,meshRow,tileSize], B[K,N,tileSize,meshCol]
    num_A = M * K * meshRow * tileSize
    num_B = K * N * tileSize * meshCol

    fp32_A = np.random.uniform(-2.0, 2.0, size=num_A).astype(np.float32)
    fp32_B = np.random.uniform(-2.0, 2.0, size=num_B).astype(np.float32)

    data_str += [format_vector_definition("float", "fp32_input_A", fp32_A)]
    data_str += [format_vector_definition("float", "fp32_input_B", fp32_B)]

    # --- Quantize to INT8 ---
    int8_A, scale_A = quantize_symmetric_int8(fp32_A)
    int8_B, scale_B = quantize_symmetric_int8(fp32_B)

    # Pad A to multiple of 64 elements for xDMA transfer
    pad_len = (-int8_A.size) % 64
    if pad_len > 0:
        int8_A_padded = np.pad(int8_A, (0, pad_len), mode='constant', constant_values=0)
    else:
        int8_A_padded = int8_A

    data_str += [format_vector_definition("int8_t", "golden_int8_A", int8_A_padded)]
    data_str += [format_vector_definition("int8_t", "golden_int8_B", int8_B)]

    # Scale factors as float (stored in memory, read by kernels via address)
    data_str += [f"float golden_scale_A __attribute__((aligned(8))) = {scale_A:.10e}f;"]
    data_str += [f"float golden_scale_B __attribute__((aligned(8))) = {scale_B:.10e}f;"]

    # Combined scale for dequantize
    combined_scale = scale_A * scale_B
    data_str += [f"float combined_scale __attribute__((aligned(8))) = {combined_scale:.10e}f;"]

    # --- GEMM golden: INT8 x INT8 -> INT32 ---
    C_zero = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
    golden_D_int32 = block_gemm_golden_model(
        M, K, N, meshRow, tileSize, meshCol,
        int8_A, int8_B,
        subtraction_a=0, subtraction_b=0,
        C=C_zero,
    )
    data_str += [format_vector_definition("int32_t", "golden_int32_D", golden_D_int32)]

    # --- Dequantize golden: INT32 -> FP32 ---
    # y[i] = int32_D[i] * combined_scale
    # We compute this in float64 then round to float32 for exact golden match
    golden_fp32_D = (golden_D_int32.astype(np.float64) * combined_scale).astype(np.float32)
    data_str += [format_vector_definition("float", "golden_fp32_D", golden_fp32_D)]

    # Sizes for the DFG
    num_D = M * N * meshRow * meshCol
    data_str += [format_scalar_definition("uint32_t", "num_A_elements", num_A)]
    data_str += [format_scalar_definition("uint32_t", "num_B_elements", num_B)]
    data_str += [format_scalar_definition("uint32_t", "num_D_elements", num_D)]
    data_str += [format_scalar_definition("uint32_t", "A_int8_size", int8_A_padded.size)]
    data_str += [format_scalar_definition("uint32_t", "B_int8_size", int8_B.size)]
    data_str += [format_scalar_definition("uint32_t", "D_int32_size", num_D * 4)]
    data_str += [format_scalar_definition("uint32_t", "D_fp32_size", num_D * 4)]

    return data_str


def main():
    parser = argparse.ArgumentParser(description="Dtype conversion data generation")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

    header_content = emit_header_file(**merged)
    with open(args.output, "w") as f:
        f.write(header_content)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
