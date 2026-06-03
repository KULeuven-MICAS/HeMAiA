#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import os
import random
import sys

SIM_UTIL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim")
)
if SIM_UTIL_DIR not in sys.path:
    sys.path.append(SIM_UTIL_DIR)

from data_utils import (  # noqa E402
    format_vector_definition,
)


MULTICHIP_GEMM_CHIPLETS = (0x00, 0x01, 0x10, 0x11)
MULTICHIP_GEMM_CHECK_BYTES = 64
MULTICHIP_GEMM_RNG_SEED = 320


def get_gemm_mesh_dims(cfg):
    data_type = cfg.get("data_type", 0)
    array_shape = cfg.get("array_shape", 1)
    acc_cfg = cfg["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]
    return unrolling[0], unrolling[1], unrolling[2]


def pad_to_xdma_width(data):
    pad_len = (-len(data)) % MULTICHIP_GEMM_CHECK_BYTES
    return data + [0] * pad_len


def transpose_a_tile(data, m, k, mesh_row, tile_size):
    transposed = []
    for mi in range(m):
        for ki in range(k):
            for ti in range(tile_size):
                for ri in range(mesh_row):
                    idx = (((mi * k + ki) * mesh_row + ri) * tile_size + ti)
                    transposed.append(data[idx])
    return transposed


def write_int8_data(f, data):
    f.write(bytes((value & 0xFF for value in data)))


def emit_header_file(**kwargs):
    rng = random.Random(MULTICHIP_GEMM_RNG_SEED)

    m = kwargs["M"]
    k = kwargs["K"]
    n = kwargs["N"]
    mesh_row, tile_size, mesh_col = get_gemm_mesh_dims(kwargs)

    data_str = ["#include <stdint.h>"]

    b_size = k * n * tile_size * mesh_col
    b_data = [rng.randrange(-128, 127) for _ in range(b_size)]

    a_tiles = []
    for _ in MULTICHIP_GEMM_CHIPLETS:
        a_size = m * k * mesh_row * tile_size
        a_data = [rng.randrange(-128, 127) for _ in range(a_size)]
        if kwargs.get("transposed_A", 0) == 1:
            a_data = transpose_a_tile(a_data, m, k, mesh_row, tile_size)
        a_tiles.append(pad_to_xdma_width(a_data))

    a_concat = [value for tile in a_tiles for value in tile]

    out_dir = kwargs.get("out_dir", "./build/")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mempool.bin"), "wb") as f:
        write_int8_data(f, a_concat)
        write_int8_data(f, b_data)

    data_str += [format_vector_definition("int8_t", "A_data_L3", a_concat)]
    data_str += [format_vector_definition("int8_t", "B_data_L3", b_data)]
    return "\n\n".join(data_str)


def main():
    raise SystemExit("Use main_bingo.py to generate dma_for_gemm_data.h")


if __name__ == "__main__":
    main()
