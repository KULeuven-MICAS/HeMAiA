#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""parse_fa.py <run dir> [...]: the FA runs into results/fa.csv, with sweep/llm_blocks/ffn/parse_ffn.py
(the same windows, SRAM snapshots, checks and phases; trace_ffn.phases names the attention's
passes)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ffn"))
import parse_ffn  # noqa: E402

parse_ffn.OUT_CSV = "fa.csv"
parse_ffn.RESULTS = HERE / "results"

if __name__ == "__main__":
    parse_ffn.main()
