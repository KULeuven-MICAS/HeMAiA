#!/usr/bin/env python3
"""
Golden data generator for attn_test_ksplit_gemm.
K-split=4 GEMM across 4 clusters with int32_add reduction and dequantize.

Generates:
- Per K-chunk (0..3): A_k{k} (int8), B_k{k} (int8)
- Per cluster partial D: D_partial_c{k} (int32)
- Running reduction sums: golden_sum_c0_c1, golden_sum_c0_c1_c2, golden_sum_final
- Dequantized FP32 output: golden_fp32_D
- combined_scale scalar
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import block_gemm_golden_model  # noqa E402

np.random.seed(42)


def emit_header_file(**kwargs):
    lines = ["#include <stdint.h>"]

    array_shape = kwargs["array_shape"]
    seq_len = kwargs["seq_len"]
    d_model = kwargs["d_model"]
    d_head = kwargs["d_head"]
    k_split = kwargs["k_split"]

    data_type = 0  # int8
    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, tileSize, meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    # Full tile dimensions
    M_T = seq_len // meshRow
    K_T_full = d_model // tileSize
    N_T = d_head // meshCol
    K_T_tile = K_T_full // k_split

    lines.append(f"// K-split GEMM: M_T={M_T} K_T_full={K_T_full} N_T={N_T} k_split={k_split}")
    lines.append(f"// K_T_tile={K_T_tile} mesh={meshRow}x{tileSize}x{meshCol}")

    # Generate full A and B in VersaCore block layout
    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127

    A_full = np.random.randint(A_MIN, A_MAX, size=(M_T, K_T_full, meshRow, tileSize)).astype(np.int8)
    B_full = np.random.randint(B_MIN, B_MAX, size=(K_T_full, N_T, tileSize, meshCol)).astype(np.int8)

    # Per K-chunk tile inputs
    A_tile_size = M_T * K_T_tile * meshRow * tileSize   # int8 bytes
    B_tile_size = K_T_tile * N_T * tileSize * meshCol    # int8 bytes
    D_size = M_T * N_T * meshRow * meshCol               # int32 element count
    D_bytes = D_size * 4                                  # int32 bytes

    # Emit per K-chunk A and B tiles
    for k in range(k_split):
        ks = k * K_T_tile
        ke = ks + K_T_tile
        A_k = A_full[:, ks:ke, :, :].reshape(-1).copy()
        B_k = B_full[ks:ke, :, :, :].reshape(-1).copy()
        lines.append(format_vector_definition("int8_t", f"A_k{k}", A_k))
        lines.append(format_vector_definition("int8_t", f"B_k{k}", B_k))

    # Compute per-cluster partial D via block_gemm_golden_model
    # Each cluster handles one K-chunk (1:1 mapping: cluster i <-> K-chunk i)
    C_zero = np.zeros(D_size, dtype=np.int32)
    partial_Ds = []
    for k in range(k_split):
        ks = k * K_T_tile
        ke = ks + K_T_tile
        A_slice = A_full[:, ks:ke, :, :].reshape(-1).copy()
        B_slice = B_full[ks:ke, :, :, :].reshape(-1).copy()
        D_partial = block_gemm_golden_model(
            M_T, K_T_tile, N_T, meshRow, tileSize, meshCol,
            A_slice, B_slice, 0, 0, C_zero.copy(),
        )
        partial_Ds.append(D_partial)
        lines.append(format_vector_definition("int32_t", f"D_partial_c{k}", D_partial))

    # Running inter-cluster reduction sums
    running_sum = partial_Ds[0].copy()
    for i in range(1, k_split):
        running_sum = (running_sum.astype(np.int64) + partial_Ds[i].astype(np.int64)).astype(np.int32)
        label = "_".join(f"c{c}" for c in range(i + 1))
        lines.append(format_vector_definition("int32_t", f"golden_sum_{label}", running_sum))

    # The final sum after all 4 partials
    # golden_sum_c0_c1_c2_c3 is the same as running_sum at this point

    # Dequantize: combined_scale is arbitrary for this test.
    # Must be np.float32 to match HW arithmetic exactly (HW kernel does
    # int32 -> fp32, then fp32 * fp32). Using fp64 intermediate produces
    # LSB-different results.
    combined_scale = np.float32(0.0001)
    fp32_input = running_sum.astype(np.float32)
    golden_fp32_D = (fp32_input * combined_scale).astype(np.float32)
    lines.append(format_vector_definition("float", "golden_fp32_D", golden_fp32_D))
    # Declare combined_scale as a 1-element array so the kernel arg assignment
    # uses array-to-pointer decay (correct address). A scalar `float x = ...`
    # would be cast by value (truncated to int 0) when emitted as a BingoMemSymbol.
    lines.append(f"float combined_scale[1] __attribute__((aligned(8))) = {{ {float(combined_scale):.10e}f }};")

    # Emit byte size scalars
    for name, val in [
        ("A_tile_bytes", A_tile_size),
        ("B_tile_bytes", B_tile_size),
        ("D_bytes", D_bytes),
        ("D_num_elements", D_size),
        ("k_split", k_split),
        ("M_T", M_T),
        ("K_T_tile", K_T_tile),
        ("N_T", N_T),
        ("meshRow", meshRow),
        ("tileSize", tileSize),
        ("meshCol", meshCol),
        ("array_shape", array_shape),
    ]:
        lines.append(format_scalar_definition("uint32_t", name, val))

    return "\n\n".join(lines) + "\n"
