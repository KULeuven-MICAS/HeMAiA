#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os

# util paths
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import (
    format_scalar_definition,
    format_vector_definition,
)

from snax_utils import block_gemm_golden_model

np.random.seed(320)


def emit_header_file(**cfg):
    out = ["#pragma once", "#include <stdint.h>"]
    out += emit_stacked_gemm_data(**cfg)
    return "\n\n".join(out)


def emit_stacked_gemm_data(**cfg):
    data = []

    # --------------------------------------------------
    # GEMM dimensions
    # --------------------------------------------------
    M1 = cfg["M1"]
    K1 = cfg["K1"]
    N1 = cfg["N1"]

    M2 = cfg["M2"]
    K2 = cfg["K2"]
    N2 = cfg["N2"]

    data += [
        format_scalar_definition("uint32_t", "M1", M1),
        format_scalar_definition("uint32_t", "K1", K1),
        format_scalar_definition("uint32_t", "N1", N1),
        format_scalar_definition("uint32_t", "M2", M2),
        format_scalar_definition("uint32_t", "K2", K2),
        format_scalar_definition("uint32_t", "N2", N2),
    ]

    array_shape = cfg["array_shape"]
    data += [format_scalar_definition("uint32_t", "array_shape", array_shape)]

    acc_cfg = cfg["snax_versacore_core_template"]["snax_acc_cfg"][0]
    data_type = 0  # int8

    meshRow, tileSize, meshCol = acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    data += [
        format_scalar_definition("uint32_t", "meshRow", meshRow),
        format_scalar_definition("uint32_t", "tileSize", tileSize),
        format_scalar_definition("uint32_t", "meshCol", meshCol),
    ]

    assert M1 * meshRow == M2 * meshRow, "In stacked GEMM, the output rows of GEMM1 should match the input rows of GEMM2"
    assert N1 * meshCol == K2 * tileSize, "In stacked GEMM, the output cols of GEMM1 should match the input rows of GEMM2"

    data_granularity_a = int(cfg["snax_versacore_core_template"]["snax_acc_cfg"][0]["granularity_a"] * 64 / 8)
    data_granularity_b = int(cfg["snax_versacore_core_template"]["snax_acc_cfg"][0]["granularity_b"] * 64 / 8)
    data_granularity_c_d = int(cfg["snax_versacore_core_template"]["snax_acc_cfg"][0]["granularity_c_d"] * 64 / 8)
    data += [
        format_scalar_definition("uint32_t", "data_granularity_a", data_granularity_a),
        format_scalar_definition("uint32_t", "data_granularity_b", data_granularity_b),
        format_scalar_definition("uint32_t", "data_granularity_c_d", data_granularity_c_d),
    ]

    data += [
        format_scalar_definition("uint32_t", "transposed_A", cfg["transposed_A"]),
        format_scalar_definition("uint32_t", "transposed_B", cfg["transposed_B"]),
        format_scalar_definition("uint32_t", "accumPrevC", cfg["accumPrevC"]),
    ]

    # --------------------------------------------------
    # Random inputs
    # --------------------------------------------------
    A1 = np.random.randint(
        -128, 127, size=(M1, K1, meshRow, tileSize), dtype=np.int8
    )

    B1 = np.random.randint(
        -128, 127, size=(K1, N1, tileSize, meshCol), dtype=np.int8
    )

    B2 = np.random.randint(
        -128, 127, size=(K2, N2, tileSize, meshCol), dtype=np.int8
    )

    # --------------------------------------------------
    # Golden GEMM 1: D1 = A1 × B1
    # --------------------------------------------------
    C1 = np.zeros((M1, N1, meshRow, meshCol), dtype=np.int32)

    D1 = block_gemm_golden_model(
        M1, K1, N1,
        meshRow, tileSize, meshCol,
        A1.reshape(-1),
        B1.reshape(-1),
        0, 0,
        C1.reshape(-1),
    )

    D1_int32 = D1.reshape(-1)  # flatten if needed

    # Allocate int8 array (4x length of D1)
    D1_int8 = np.zeros(D1_int32.size * 4, dtype=np.int8)

    # Split each int32 into 4 int8 bytes
    for i, val in enumerate(D1_int32):
        D1_int8[i*4 + 0] = (val & 0xFF).astype(np.int8)
        D1_int8[i*4 + 1] = ((val >> 8) & 0xFF).astype(np.int8)
        D1_int8[i*4 + 2] = ((val >> 16) & 0xFF).astype(np.int8)
        D1_int8[i*4 + 3] = ((val >> 24) & 0xFF).astype(np.int8)

    # --------------------------------------------------
    # Golden GEMM 2: D2 = D1 × B2
    # --------------------------------------------------
    C2 = np.zeros((M2, N2, meshRow, meshCol), dtype=np.int32)

    D2 = block_gemm_golden_model(
        M2, K2, N2,
        meshRow, tileSize, meshCol,
        D1_int8.reshape(-1)[0:M2*K2*meshRow*tileSize],
        B2.reshape(-1),
        0, 0,
        C2.reshape(-1),
    )

    # --------------------------------------------------
    # Emit L3 symbols (flat)
    # --------------------------------------------------
    data += [
        format_vector_definition("int8_t",  "A1", A1.reshape(-1)),
        format_vector_definition("int8_t",  "B1", B1.reshape(-1)),
        format_vector_definition("int8_t",  "B2", B2.reshape(-1)),
        format_vector_definition("int32_t", "D1", D1.reshape(-1)),
        format_vector_definition("int8_t", "D1_int8", D1_int8.reshape(-1)),
        format_vector_definition("int32_t", "D2", D2.reshape(-1)),
    ]

    return data


def main():
    # Parsing cmd args
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

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    # Load hardware config file
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    # Merge dictionaries (hw overrides param in case of conflicts)
    merged_config = {**param, **hw}

    # Emit header file
    print(emit_header_file(**merged_config))


if __name__ == "__main__":
    main()
