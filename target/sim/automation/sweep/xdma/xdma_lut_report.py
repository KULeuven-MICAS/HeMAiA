#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Documents the gathered xDMA op cycle measurements (xdma_cycles.csv) as a
# markdown table -- thin wrapper over util/automation_scripts/lut_report.py (no
# bingo dependency; curve fitting lives in the bingo framework).
#
#   python3 xdma_lut_report.py [--csv FILE] [--out FILE]

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(
    os.path.join(_THIS, "../../../../../util/automation_scripts")))
from lut_report import run_cli  # noqa: E402

# Every op this sweep produces, so the report flags any that are missing.
#
# The 6 layout converters are NOT one op each: each is a family of 9 runnable kernels
# (3 array shapes x 3 element widths), one LUT apiece -- xdma_row_major_to_a_e2_M1K16
# and friends. All 54 variants have to be listed individually: a family name would
# match a family whose variant LUTs are still empty, and the gap would go unreported.
_SHAPE_TOKENS = {
    # family group -> the (shape 0, shape 1, shape 2) tokens, from the mesh triples
    # (32,2,32) / (1,16,32) / (16,8,16).  A: M<meshRow>K<tileSize>,
    # B: K<tileSize>N<meshCol>, D: M<meshRow>N<meshCol>.
    "A": ("M32K2", "M1K16", "M16K8"),
    "B": ("K2N32", "K16N32", "K8N16"),
    "D": ("M32N32", "M1N32", "M16N16"),
}
_LAYOUT_FAMILIES = {
    "xdma_row_major_to_a": "A", "xdma_a_to_row_major": "A",
    "xdma_row_major_to_b": "B", "xdma_b_to_row_major": "B",
    "xdma_row_major_to_d": "D", "xdma_d_to_row_major": "D",
}

OPS = [
    "xdma_6d", "xdma_transpose_2d", "xdma_submatrix_2d", "xdma_expand_2d",
    "xdma_concat_2d", "xdma_pad_2d", "xdma_gather_2d", "xdma_elementwise_add",
] + [
    f"{fam}_e{eb}_{tok}"
    for fam, grp in _LAYOUT_FAMILIES.items()
    for tok in _SHAPE_TOKENS[grp]
    for eb in (1, 2, 4)
]
_DEFAULT_CSV = os.path.join(_THIS, "xdma_cycles.csv")

if __name__ == "__main__":
    run_cli(_DEFAULT_CSV, OPS)
