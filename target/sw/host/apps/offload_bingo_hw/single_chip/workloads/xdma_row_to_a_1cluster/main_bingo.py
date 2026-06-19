# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "row_to_a" op.
# Each CONFIGS entry runs Load -> xDMA row_to_a -> Store -> Check and emits one
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

# Cycle-LUT sweep: rows x cols grid (elem=1) for the bilinear fit. Uses
# array_shape=2 (mesh [16,8,16]) so tileSize=8 (%8==0) takes the HW path; the
# A-operand shape is rows = M_T*meshRow = M_T*16, cols = K_T*tileSize = K_T*8.
CONFIGS = [{"M_T": m, "K_T": k, "N_T": 1, "elem_bytes": 1, "array_shape": 2}
           for m in (1, 2, 4) for k in (4, 8, 16)]   # rows 16..64, cols 32..128

if __name__ == "__main__":
    run_op_workload("row_to_a", CONFIGS)
