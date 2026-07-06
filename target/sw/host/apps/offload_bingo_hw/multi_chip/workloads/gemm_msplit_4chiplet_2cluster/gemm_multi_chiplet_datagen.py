#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import os
import sys

import numpy as np

SIM_UTIL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim")
)
if SIM_UTIL_DIR not in sys.path:
    sys.path.append(SIM_UTIL_DIR)

import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from gemm_sim_utils import get_gemm_mesh_dims  # noqa E402
from sim_golden_models import block_gemm_golden_model  # noqa E402


XDMA_WIDTH_BYTES = 64
RNG_SEED = 320


def _pad_to_xdma_width(data):
    pad_len = (-data.size) % XDMA_WIDTH_BYTES
    if pad_len == 0:
        return data
    return np.pad(data, (0, pad_len), mode="constant", constant_values=0)


def _emit_settings(kwargs, mesh_row, tile_size, mesh_col):
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


def emit_header_file(**kwargs):
    # This workload assigns one independent M-split tile to each chiplet/cluster
    # slot. The shared multichip helper emits one tile per chiplet, which moves B
    # to the wrong mempool offset for the two-cluster schedule.
    np.random.seed(RNG_SEED)

    m = kwargs["M"]
    k = kwargs["K"]
    n = kwargs["N"]
    tile_count = int(kwargs["num_chiplets"]) * int(kwargs["num_clusters"])
    mesh_row, tile_size, mesh_col = get_gemm_mesh_dims(kwargs)

    assert (
        sum([kwargs["addNonZeroC"], kwargs["addZeroC"], kwargs["accumPrevC"]]) == 1
    ), "Only one of addNonZeroC, addZeroC, accumPrevC can be set to 1."
    if kwargs["accumPrevC"] == 1:
        assert m == 1 and n == 1, "When accumPrevC=1, M, N must be 1."

    b_data = np.random.randint(
        -128, 127, size=(k, n, tile_size, mesh_col)
    ).reshape(-1)
    if kwargs["transposed_B"] == 1:
        b_for_model = b_data.reshape(k, n, tile_size, mesh_col)
        b_for_model = b_for_model.transpose(0, 1, 3, 2).reshape(-1)
    else:
        b_for_model = b_data

    a_tiles = []
    d_tiles = []
    for _ in range(tile_count):
        a_data = np.random.randint(
            -128, 127, size=(m, k, mesh_row, tile_size)
        ).reshape(-1)
        if kwargs["transposed_A"] == 1:
            a_for_model = a_data.reshape(m, k, mesh_row, tile_size)
            a_for_model = a_for_model.transpose(0, 1, 3, 2).reshape(-1)
        else:
            a_for_model = a_data
        a_tiles.append(_pad_to_xdma_width(a_for_model))

        if kwargs["addNonZeroC"] == 1:
            c_data = np.random.randint(
                -2147483648, 2147483647, size=(m, n, mesh_row, mesh_col)
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
    with open(os.path.join(out_dir, "mempool.bin"), "wb") as f:
        a_concat.astype(np.uint8).tofile(f)
        b_data.astype(np.uint8).tofile(f)
        d_concat.astype(np.uint32).tofile(f)

    emit_str = ["#include <stdint.h>"]
    emit_str += _emit_settings(kwargs, mesh_row, tile_size, mesh_col)
    emit_str += [format_vector_definition("int8_t", "A_data_L3", a_concat)]
    emit_str += [format_vector_definition("int8_t", "B_data_L3", b_data)]
    emit_str += [
        format_vector_definition("int32_t", "D_data_L3", d_concat.astype(np.int32))
    ]
    return "\n\n".join(emit_str)


def main():
    import argparse
    import pathlib

    import hjson

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

    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())
    print(emit_header_file(**{**param, **hw}))


if __name__ == "__main__":
    main()
