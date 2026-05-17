#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/")
)
from gemm_sim_utils import get_gemm_mesh_dims  # noqa E402
from sim_golden_models import block_gemm_golden_model  # noqa E402

np.random.seed(320)


def emit_header_file(**kwargs):
    return "\n\n".join(["#include <stdint.h>", *emit_matmul_data(**kwargs)])


def emit_matmul_data(**kwargs):
    assert kwargs.get("data_type", 0) == 0
    assert kwargs.get("int4_a_enable", 0) == 0
    assert kwargs.get("int4_b_enable", 0) == 0
    assert kwargs.get("quantization_enable", 0) == 0
    assert kwargs.get("int32tofp16_enable", 0) == 0
    assert kwargs["addNonZeroC"] == 0
    assert kwargs["addZeroC"] == 1
    assert kwargs["accumPrevC"] == 0

    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]
    meshRow, tileSize, meshCol = get_gemm_mesh_dims(kwargs)

    A = np.random.randint(-128, 127, size=(M, K, meshRow, tileSize)).reshape(-1)
    B = np.random.randint(-128, 127, size=(K, N, tileSize, meshCol)).reshape(-1)

    A_golden = A
    B_golden = B
    if kwargs["transposed_A"] == 1:
        A_golden = (
            A.reshape(M, K, meshRow, tileSize).transpose(0, 1, 3, 2).reshape(-1)
        )
    if kwargs["transposed_B"] == 1:
        B_golden = (
            B.reshape(K, N, tileSize, meshCol).transpose(0, 1, 3, 2).reshape(-1)
        )

    C = np.zeros((M, N, meshRow, meshCol), dtype=np.int32).reshape(-1)

    D = block_gemm_golden_model(
        M, K, N, meshRow, tileSize, meshCol, A_golden, B_golden, 0, 0, C
    )

    out_dir = kwargs.get("out_dir", "./build/")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mempool.bin"), "wb") as f:
        A.astype(np.uint8).tofile(f)
        B.astype(np.uint8).tofile(f)
        D.astype(np.uint32).tofile(f)

    return []


def main():
    parser = argparse.ArgumentParser(description="Generating data for GEMM mem-chip test")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    print(emit_header_file(**{**param, **hw}))


if __name__ == "__main__":
    main()
