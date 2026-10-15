#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""parse_moe.py <run dir> [...]: the MoE runs into results/moe.csv, with sweep/llm_blocks/ffn/parse_ffn.py
(the same windows, SRAM snapshots, checks and phases; the route's phase is the last slot chip's
record)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ffn"))
import parse_ffn  # noqa: E402

parse_ffn.OUT_CSV = "moe.csv"
parse_ffn.RESULTS = HERE / "results"

if __name__ == "__main__":
    parse_ffn.main()
