#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Data + golden generator for the xdma_layout_transform workload.
# Emits one row-major source per layout direction and the matching blocked
# layout produced by the Python reference at HeMAiA/util/sim/layout_convert.py.
# The kernels are checked against the same goldens; row-major and blocked
# tensors play double duty as both the kernel src (per direction) and the
# expected output of the inverse direction.

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),
                             "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from layout_convert import (  # noqa E402
    row_major_to_a, row_major_to_b, row_major_to_d,
)

np.random.seed(42)


def emit_header_file(**kwargs):
    return "\n\n".join(["#include <stdint.h>"] + emit_data(**kwargs))


def emit_data(**kwargs):
    M_T = kwargs["M_T"]; K_T = kwargs["K_T"]; N_T = kwargs["N_T"]
    meshRow = kwargs["meshRow"]; tileSize = kwargs["tileSize"]; meshCol = kwargs["meshCol"]
    elem_bytes = kwargs["elem_bytes"]

    out = []
    for k, v in [("M_T", M_T), ("K_T", K_T), ("N_T", N_T),
                 ("meshRow", meshRow), ("tileSize", tileSize), ("meshCol", meshCol),
                 ("elem_bytes", elem_bytes)]:
        out.append(format_scalar_definition("uint32_t", k, v))

    A_rows = M_T * meshRow;  A_cols = K_T * tileSize
    B_rows = K_T * tileSize; B_cols = N_T * meshCol
    D_rows = M_T * meshRow;  D_cols = N_T * meshCol
    A_bytes = A_rows * A_cols * elem_bytes
    B_bytes = B_rows * B_cols * elem_bytes
    D_bytes = D_rows * D_cols * elem_bytes
    for k, v in [("A_bytes", A_bytes), ("B_bytes", B_bytes), ("D_bytes", D_bytes)]:
        out.append(format_scalar_definition("uint32_t", k, v))

    if elem_bytes == 1:
        dtype, lo, hi, ctype = np.int8, -128, 127, "int8_t"
    elif elem_bytes == 4:
        dtype, lo, hi, ctype = np.int32, -1_000_000, 1_000_000, "int32_t"
    else:
        raise ValueError(f"elem_bytes={elem_bytes} not supported")

    A_rm = np.random.randint(lo, hi, size=(A_rows, A_cols), dtype=dtype)
    B_rm = np.random.randint(lo, hi, size=(B_rows, B_cols), dtype=dtype)
    D_rm = np.random.randint(lo, hi, size=(D_rows, D_cols), dtype=dtype)

    A_layout = row_major_to_a(A_rm, M_T, K_T, meshRow, tileSize)
    B_layout = row_major_to_b(B_rm, K_T, N_T, tileSize, meshCol)
    D_layout = row_major_to_d(D_rm, M_T, N_T, meshRow, meshCol)

    out += [
        format_vector_definition(ctype, "src_A_rm",      A_rm.reshape(-1)),
        format_vector_definition(ctype, "src_A_layout",  A_layout),
        format_vector_definition(ctype, "src_B_rm",      B_rm.reshape(-1)),
        format_vector_definition(ctype, "src_B_layout",  B_layout),
        format_vector_definition(ctype, "src_D_rm",      D_rm.reshape(-1)),
        format_vector_definition(ctype, "src_D_layout",  D_layout),
        # Goldens: inverse-direction expected outputs (same content as src_*).
        format_vector_definition(ctype, "golden_A_layout", A_layout),
        format_vector_definition(ctype, "golden_A_rm",     A_rm.reshape(-1)),
        format_vector_definition(ctype, "golden_B_layout", B_layout),
        format_vector_definition(ctype, "golden_B_rm",     B_rm.reshape(-1)),
        format_vector_definition(ctype, "golden_D_layout", D_layout),
        format_vector_definition(ctype, "golden_D_rm",     D_rm.reshape(-1)),
    ]
    return out


def main():
    import argparse, pathlib, hjson
    p = argparse.ArgumentParser(description="xdma_layout_transform datagen")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = p.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
