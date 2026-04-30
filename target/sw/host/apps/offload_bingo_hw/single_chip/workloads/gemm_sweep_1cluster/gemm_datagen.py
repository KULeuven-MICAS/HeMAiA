#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Data generator for the gemm_sweep RTL characterization workload.
# Emits per-config int8 A/B matrices + int32 golden D for every (M,K,N,shape)
# in CONFIGS.  The DFG in main_bingo.py runs each config twice (gemm_full
# once to configure the streamers, then gemm_minimal to measure the
# steady-state cost) and cross-checks the result against these goldens.

import numpy as np
import sys
import os

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_vector_definition  # noqa E402
from snax_utils import block_gemm_golden_model  # noqa E402

np.random.seed(320)


def mesh_dims(hwcfg: dict, array_shape: int):
    """Extract (meshRow, tileSize, meshCol) from an hjson hw config."""
    snax_acc_cfg = hwcfg["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = snax_acc_cfg["snax_versacore_spatial_unrolling"][0][array_shape]
    return unrolling[0], unrolling[1], unrolling[2]


def config_sizes(hwcfg: dict, M: int, K: int, N: int, array_shape: int):
    """Return (A_bytes, B_bytes, D_bytes) for a single config."""
    meshRow, tileSize, meshCol = mesh_dims(hwcfg, array_shape)
    return (
        M * K * meshRow * tileSize,           # int8
        K * N * tileSize * meshCol,           # int8
        M * N * meshRow * meshCol * 4,        # int32
    )


def emit_header_file(configs: list, hwcfg: dict) -> str:
    """Generate gemm_data.h with per-config A/B input + D golden arrays.

    Args:
        configs: list of (M, K, N, array_shape) tuples
        hwcfg:   the parsed snax_versacore_to_cluster.hjson config
    """
    lines = ["#include <stdint.h>"]

    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127

    for i, (M, K, N, shape) in enumerate(configs):
        meshRow, tileSize, meshCol = mesh_dims(hwcfg, shape)

        A = np.random.randint(
            A_MIN, A_MAX, size=(M, K, meshRow, tileSize)
        ).reshape(-1).astype(np.int8)
        # Pad A to multiple of 64 bytes for xDMA alignment
        pad_len = (-A.size) % 64
        if pad_len > 0:
            A = np.pad(A, (0, pad_len), mode="constant", constant_values=0)
        lines.append("")
        lines.append(
            f"// cfg{i}: M={M} K={K} N={N} shape={shape} "
            f"(mesh={meshRow}x{tileSize}x{meshCol})"
        )
        lines.append(format_vector_definition("int8_t", f"A_cfg{i}", A))

        B = np.random.randint(
            B_MIN, B_MAX, size=(K, N, tileSize, meshCol)
        ).reshape(-1).astype(np.int8)
        lines.append(format_vector_definition("int8_t", f"B_cfg{i}", B))

        C = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)
        D = block_gemm_golden_model(
            M, K, N, meshRow, tileSize, meshCol,
            A[: M * K * meshRow * tileSize],  # strip padding for golden
            B,
            0, 0,  # subtraction_a, subtraction_b
            C,
        )
        lines.append(format_vector_definition("int32_t", f"D_cfg{i}", D))

    return "\n".join(lines) + "\n"
