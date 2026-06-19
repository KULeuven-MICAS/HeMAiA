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
# __usg__ grouped util/sim: make common/gemm/xdma/ara importable
import os as _usg_os, sys as _usg_sys
for _usg_p in [p for p in list(_usg_sys.path) if str(p).rstrip('/').endswith('util/sim')]:
    for _usg_s in ('common', 'gemm', 'xdma', 'ara'):
        _usg_sub = _usg_os.path.join(_usg_p, _usg_s)
        if _usg_sub not in _usg_sys.path:
            _usg_sys.path.append(_usg_sub)
from gemm_psweep_lib import run  # noqa: E402

if __name__ == "__main__":
    run("i8i8_i32")
