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

# The 14 ops this sweep produces (so the report flags any that are missing).
OPS = [
    "xdma_6d", "xdma_transpose_2d", "xdma_submatrix_2d", "xdma_expand_2d",
    "xdma_concat_2d", "xdma_pad_2d", "xdma_gather_2d", "xdma_elementwise_add",
    "xdma_row_major_to_a", "xdma_a_to_row_major", "xdma_row_major_to_b",
    "xdma_b_to_row_major", "xdma_row_major_to_d", "xdma_d_to_row_major",
]
_DEFAULT_CSV = os.path.join(_THIS, "xdma_cycles.csv")

if __name__ == "__main__":
    run_cli(_DEFAULT_CSV, OPS)
