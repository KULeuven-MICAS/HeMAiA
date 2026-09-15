#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Documents the gathered SIMD op cycle measurements (simd_cycles.csv) as a markdown
# table -- thin wrapper over util/automation_scripts/lut_report.py (no bingo
# dependency; curve fitting lives in the bingo framework).
#
#   python3 simd_lut_report.py [--csv FILE] [--out FILE]

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(
    os.path.join(_THIS, "../../../../../util/automation_scripts")))
from lut_report import run_cli  # noqa: E402

# Every op this sweep produces, so the report flags any that are missing.
#
# Five fused whole-op kernels (each a multi-pass pipeline the scheduler dispatches as
# ONE node), two of them in both an fp16 and an int8-output variant, plus the five
# primitive operators -- one armed chain each, which between them cover every stage of
# the chain: EW0, Map, Reduce, EW1 and the Fp16ToInt8 quantiser.
# The int8-output variants of softmax and rmsnorm are NOT listed: their workloads run the
# f16 chain only (48 host nodes for both chains do not fit the tapeout L3), so listing them
# would report a permanent gap rather than a missing measurement. The standalone snax apps
# cover them.
OPS = [
    "simd_softmax",
    "simd_rmsnorm",
    "simd_rope",
    "simd_silu",
    "simd_swiglu",
    "simd_stream_map",
    "simd_stream_reduce",
    "simd_stream_elementwise",
    "simd_stream_map_reduce",
    "simd_fp16_to_int8",
]
_DEFAULT_CSV = os.path.join(_THIS, "simd_cycles.csv")

if __name__ == "__main__":
    run_cli(_DEFAULT_CSV, OPS)
