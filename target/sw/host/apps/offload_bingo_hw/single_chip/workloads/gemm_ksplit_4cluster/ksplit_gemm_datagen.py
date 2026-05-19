#!/usr/bin/env python3
"""
Golden data generator for gemm_ksplit_4cluster.
K-split=4 GEMM across 4 clusters with int32_add reduction and dequantize.

Generates `build/mempool.bin` with:
- A_k{0..3} (int8), then B_k{0..3} (int8), then golden_D_c{0..3} (int32)
- Running reduction sums: golden_sum_c0_c1, golden_sum_c0_c1_c2, golden_sum_final
- Dequantized FP32 output: golden_fp32_D
- combined_scale scalar
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition  # noqa E402
from sim_golden_models import block_gemm_golden_model  # noqa E402

np.random.seed(42)


def emit_header_file(**kwargs):
    lines = ["#include <stdint.h>"]

    array_shape = kwargs["array_shape"]
    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]
    k_split = kwargs["k_split"]
    if k_split <= 0:
        raise ValueError(f"k_split ({k_split}) must be positive")
    if k_split != 4:
        raise ValueError(f"gemm_ksplit_4cluster expects k_split=4, got {k_split}")
    if K % k_split != 0:
        raise ValueError(f"K ({K}) must be divisible by k_split ({k_split})")

    data_type = kwargs.get("data_type", 0)  # int8
    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, tileSize, meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    # Full tile dimensions. K_tile is the per-cluster K chunk.
    K_tile = K // k_split

    lines.append(f"// K-split GEMM: M={M} K={K} N={N} k_split={k_split}")
    lines.append(f"// K_tile={K_tile} mesh={meshRow}x{tileSize}x{meshCol}")

    # Generate full A and B in VersaCore block layout
    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127

    A_full = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).astype(np.int8)
    B_full = np.random.randint(B_MIN, B_MAX, size=(K, N, tileSize, meshCol)).astype(np.int8)

    # Per K-chunk tile inputs
    A_tile_size = M * K_tile * meshRow * tileSize        # int8 bytes
    B_tile_size = K_tile * N * tileSize * meshCol         # int8 bytes
    D_size = M * N * meshRow * meshCol                    # int32 element count
    D_bytes = D_size * 4                                  # int32 bytes

    A_tiles = []
    B_tiles = []
    golden_D_tiles = []
    mempool_arrays = []

    # Emit per K-chunk A and B tiles
    for k in range(k_split):
        ks = k * K_tile
        ke = ks + K_tile
        A_k = A_full[:, ks:ke, :, :].reshape(-1).copy()
        B_k = B_full[ks:ke, :, :, :].reshape(-1).copy()
        A_tiles.append(A_k.astype(np.int8))
        B_tiles.append(B_k.astype(np.int8))

    # Compute per-cluster partial D via block_gemm_golden_model
    # Each cluster handles one K-chunk (1:1 mapping: cluster i <-> K-chunk i)
    C_zero = np.zeros(D_size, dtype=np.int32)
    partial_Ds = []
    for k in range(k_split):
        ks = k * K_tile
        ke = ks + K_tile
        A_slice = A_full[:, ks:ke, :, :].reshape(-1).copy()
        B_slice = B_full[ks:ke, :, :, :].reshape(-1).copy()
        D_partial = block_gemm_golden_model(
            M, K_tile, N, meshRow, tileSize, meshCol,
            A_slice, B_slice, 0, 0, C_zero.copy(),
        )
        partial_Ds.append(D_partial)
        golden_D_tiles.append(D_partial.astype(np.int32))

    mempool_arrays.extend(A_tiles)
    mempool_arrays.extend(B_tiles)
    mempool_arrays.extend(golden_D_tiles)

    # Running inter-cluster reduction sums
    running_sum = partial_Ds[0].copy()
    for i in range(1, k_split):
        running_sum = (running_sum.astype(np.int64) + partial_Ds[i].astype(np.int64)).astype(np.int32)
        mempool_arrays.append(running_sum.astype(np.int32))

    # The final sum after all 4 partials
    # golden_sum_c0_c1_c2_c3 is the same as running_sum at this point

    # Dequantize: combined_scale is arbitrary for this test.
    # Must be np.float32 to match HW arithmetic exactly (HW kernel does
    # int32 -> fp32, then fp32 * fp32). Using fp64 intermediate produces
    # LSB-different results.
    combined_scale = np.float32(0.0001)
    fp32_input = running_sum.astype(np.float32)
    golden_fp32_D = (fp32_input * combined_scale).astype(np.float32)
    mempool_arrays.append(golden_fp32_D.astype(np.float32))
    mempool_arrays.append(np.array([combined_scale], dtype=np.float32))

    out_dir = kwargs.get("out_dir", "./build/")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mempool.bin"), "wb") as f:
        for array in mempool_arrays:
            array.tofile(f)

    # Emit byte size scalars
    for name, val in [
        ("A_tile_bytes", A_tile_size),
        ("B_tile_bytes", B_tile_size),
        ("D_bytes", D_bytes),
        ("D_num_elements", D_size),
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
