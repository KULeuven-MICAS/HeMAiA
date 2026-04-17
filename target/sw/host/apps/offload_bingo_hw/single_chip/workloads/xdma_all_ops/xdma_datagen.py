#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Generate test data + golden references for ALL xDMA data layout ops.
# Produces xdma_data.h with input matrices and per-op golden outputs.

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

np.random.seed(42)


def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_xdma_data(**kwargs)
    return "\n\n".join(emit_str)


def emit_xdma_data(**kwargs):
    data_str = []
    rows = kwargs["rows"]
    cols = kwargs["cols"]
    elem = kwargs["elem_bytes"]
    total_elems = rows * cols

    data_str += [format_scalar_definition("uint32_t", "test_rows", rows)]
    data_str += [format_scalar_definition("uint32_t", "test_cols", cols)]
    data_str += [format_scalar_definition("uint32_t", "test_elem_bytes", elem)]
    data_str += [format_scalar_definition("uint32_t", "test_total_bytes", total_elems * elem)]

    # Input: [rows, cols] uint8 matrix (sequential values for easy visual checking)
    input_data = np.arange(total_elems, dtype=np.uint8)
    data_str += [format_vector_definition("uint8_t", "input_data", input_data)]

    # 1. COPY golden: same as input
    data_str += [format_vector_definition("uint8_t", "golden_copy", input_data)]

    # 2. TRANSPOSE golden: [rows, cols] -> [cols, rows]
    mat = input_data.reshape(rows, cols)
    golden_transpose = mat.T.flatten()
    data_str += [format_vector_definition("uint8_t", "golden_transpose", golden_transpose.astype(np.uint8))]

    # 3. SUBMATRIX/SLICE golden: extract [0:8, 8:cols]
    # All slice boundaries must be multiples of 8 for xDMA tile-level access.
    sub_row_start, sub_row_end = 0, 8
    sub_col_start, sub_col_end = 8, cols
    sub_rows = sub_row_end - sub_row_start
    sub_cols = sub_col_end - sub_col_start
    golden_submatrix = mat[sub_row_start:sub_row_end, sub_col_start:sub_col_end].flatten()
    data_str += [format_scalar_definition("uint32_t", "sub_row_start", sub_row_start)]
    data_str += [format_scalar_definition("uint32_t", "sub_row_end", sub_row_end)]
    data_str += [format_scalar_definition("uint32_t", "sub_col_start", sub_col_start)]
    data_str += [format_scalar_definition("uint32_t", "sub_col_end", sub_col_end)]
    data_str += [format_scalar_definition("uint32_t", "sub_rows", sub_rows)]
    data_str += [format_scalar_definition("uint32_t", "sub_cols", sub_cols)]
    data_str += [format_vector_definition("uint8_t", "golden_submatrix", golden_submatrix.astype(np.uint8))]

    # 4. EXPAND golden: broadcast first row [1, cols] -> [rows, cols]
    first_row = mat[0:1, :]  # [1, cols]
    golden_expand = np.tile(first_row, (rows, 1)).flatten()
    data_str += [format_vector_definition("uint8_t", "golden_expand", golden_expand.astype(np.uint8))]

    # 5. CONCAT golden: concat mat[0:rows//2] and mat[rows//2:rows] along axis=0
    # This should reconstruct the original matrix
    half = rows // 2
    data_str += [format_scalar_definition("uint32_t", "concat_half", half)]
    data_str += [format_vector_definition("uint8_t", "golden_concat", input_data)]  # reconstruct = original

    # 6. PAD golden: pad with 8 rows at top (all dims must be multiples of 8)
    pad_top, pad_bottom, pad_left, pad_right = 8, 0, 0, 0
    golden_pad = np.pad(mat, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='constant', constant_values=0).flatten()
    padded_rows = rows + pad_top + pad_bottom
    padded_cols = cols + pad_left + pad_right
    data_str += [format_scalar_definition("uint32_t", "pad_top", pad_top)]
    data_str += [format_scalar_definition("uint32_t", "pad_bottom", pad_bottom)]
    data_str += [format_scalar_definition("uint32_t", "pad_left", pad_left)]
    data_str += [format_scalar_definition("uint32_t", "pad_right", pad_right)]
    data_str += [format_scalar_definition("uint32_t", "padded_rows", padded_rows)]
    data_str += [format_scalar_definition("uint32_t", "padded_cols", padded_cols)]
    data_str += [format_vector_definition("uint8_t", "golden_pad", golden_pad.astype(np.uint8))]

    # 7. GATHER golden: gather every other row (stride=2)
    gather_start = 0
    gather_stride = 2
    gather_count = rows // gather_stride
    gathered = mat[gather_start::gather_stride, :].flatten()
    data_str += [format_scalar_definition("uint32_t", "gather_start", gather_start)]
    data_str += [format_scalar_definition("uint32_t", "gather_stride", gather_stride)]
    data_str += [format_scalar_definition("uint32_t", "gather_count", gather_count)]
    data_str += [format_vector_definition("uint8_t", "golden_gather", gathered.astype(np.uint8))]

    return data_str


def main():
    import argparse, pathlib, hjson
    parser = argparse.ArgumentParser(description="Generate xDMA all-ops test data")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
