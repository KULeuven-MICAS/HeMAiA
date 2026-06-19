# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "concat" op.
# Each CONFIGS entry runs Load -> xDMA concat -> Store -> Check and emits one
# XDMA_RUN trace event (in CONFIGS order) for the cycle LUT. Shared machinery
# lives in util/sim/xdma/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")
# __usg__ grouped util/sim: make common/gemm/xdma/ara importable
import os as _usg_os, sys as _usg_sys
for _usg_p in [p for p in list(_usg_sys.path) if str(p).rstrip('/').endswith('util/sim')]:
    for _usg_s in ('common', 'gemm', 'xdma', 'ara'):
        _usg_sub = _usg_os.path.join(_usg_p, _usg_s)
        if _usg_sub not in _usg_sys.path:
            _usg_sys.path.append(_usg_sub)

from xdma_ops_lib import run_op_workload  # noqa E402

# Cycle-LUT sweep: rows x cols grid (elem=1) for the bilinear fit. rows is a
# multiple of 16 (concat assembles two rows/2 halves, each %8). The op emits
# TWO XDMA_RUN events per config (top+bottom) -> the extractor sums them.
_RC = [16, 64, 128]
CONFIGS = [{"rows": r, "cols": c, "elem_bytes": 1} for r in _RC for c in _RC]

if __name__ == "__main__":
    run_op_workload("concat", CONFIGS)
