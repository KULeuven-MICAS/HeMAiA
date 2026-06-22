# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-precision GEMM cycle sweep: i8i8_i32. All logic is shared in
# util/sim/gemm/gemm_psweep_lib.py; this just selects the precision. The size grid
# lives in gemm_psweep_lib.CONFIGS (shared across precisions for comparable LUTs).
import os
import sys

# repo root is 8 levels up from this workload dir; the shared lib is in util/sim.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 8)))
sys.path.insert(0, os.path.join(_ROOT, "util", "sim"))
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)
from gemm_psweep_lib import run  # noqa: E402

if __name__ == "__main__":
    run("i8i8_i32")
