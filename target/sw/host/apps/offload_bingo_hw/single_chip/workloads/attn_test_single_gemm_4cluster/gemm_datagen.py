#!/usr/bin/env python3
"""
Golden data generator for attn_test_single_gemm.
Single GEMM: X[seq_len x d_model] @ Wq[d_model x d_head] -> Q[seq_len x d_head]

Follows the exact same pattern as gemm_sweep/gemm_datagen.py.
Uses block_gemm_golden_model for golden D computation.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from sim_golden_models import block_gemm_golden_model  # noqa E402

np.random.seed(42)


def emit_header_file(**kwargs):
    lines = ["#include <stdint.h>"]

    array_shape = kwargs["array_shape"]
    seq_len = kwargs["seq_len"]
    d_model = kwargs["d_model"]
    d_head = kwargs["d_head"]

    data_type = 0  # int8
    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, tileSize, meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    # Tile dimensions
    M = seq_len // meshRow
    K = d_model // tileSize
    N = d_head // meshCol

    lines.append(f"// GEMM: M={M} K={K} N={N} shape={array_shape} (mesh={meshRow}x{tileSize}x{meshCol})")

    # Generate random int8 A and B in VersaCore layout
    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127

    A = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).reshape(-1).astype(np.int8)
    # Pad A to multiple of 64 bytes for xDMA alignment
    pad_len = (-A.size) % 64
    if pad_len > 0:
        A = np.pad(A, (0, pad_len), mode="constant", constant_values=0)
    lines.append(format_vector_definition("int8_t", "A", A))

    B = np.random.randint(B_MIN, B_MAX, size=(K, N, tileSize, meshCol)).reshape(-1).astype(np.int8)
    lines.append(format_vector_definition("int8_t", "B", B))

    C = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
    D = block_gemm_golden_model(
        M, K, N, meshRow, tileSize, meshCol,
        A[: M * K * meshRow * tileSize],  # strip padding for golden
        B, 0, 0, C,
    )
    lines.append(format_vector_definition("int32_t", "D", D))

    # Sizes in bytes
    A_size = M * K * meshRow * tileSize      # int8
    B_size = K * N * tileSize * meshCol       # int8
    D_size = M * N * meshRow * meshCol * 4    # int32

    for name, val in [
        ("array_shape", array_shape),
        ("M", M), ("K", K), ("N", N),
        ("meshRow", meshRow), ("tileSize", tileSize), ("meshCol", meshCol),
        ("A_size", A_size), ("B_size", B_size), ("D_size", D_size),
    ]:
        lines.append(format_scalar_definition("uint32_t", name, val))

    return "\n\n".join(lines) + "\n"
