# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "submatrix" op.
# Each CONFIGS entry runs Load -> xDMA submatrix -> Store -> Check and emits one
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

# Cycle-LUT sweep: extract an R x C sub-block (output rows R, cols C) over a grid
# for the bilinear fit. Slice honors re-rs=R %8, ce-cs=C %8, cs=8 (cs*elem %8).
_RC = [16, 64, 128]
CONFIGS = [{"rows": R, "cols": C + 8, "elem_bytes": 1,
            "rs": 0, "re": R, "cs": 8, "ce": 8 + C}
           for R in _RC for C in _RC]

if __name__ == "__main__":
    run_op_workload("submatrix", CONFIGS)
