#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Documents the gathered Ara op cycle measurements (ara_cycles.csv) as a markdown
# table -- thin wrapper over util/automation_scripts/lut_report.py (no bingo
# dependency; curve fitting lives in the bingo framework).
#
#   python3 ara_lut_report.py [--csv FILE] [--out FILE]

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(
    os.path.join(_THIS, "../../../../../util/automation_scripts")))
from lut_report import run_cli  # noqa: E402

# The kernels this sweep produces (so the report flags any that are missing).
OPS = [
    "add", "sub", "mul", "div", "max", "min", "silu_mul",
    "exp", "sigmoid", "sqrt", "relu", "neg", "abs", "tanh", "reciprocal",
    "silu", "gelu", "reduce_sum", "reduce_max", "reduce_mean",
    "quantize", "dequantize", "softmax", "rmsnorm",
]
_DEFAULT_CSV = os.path.join(_THIS, "ara_cycles.csv")

if __name__ == "__main__":
    run_cli(_DEFAULT_CSV, OPS)
