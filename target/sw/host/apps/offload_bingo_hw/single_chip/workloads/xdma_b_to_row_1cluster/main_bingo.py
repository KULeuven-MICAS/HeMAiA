# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "b_to_row" op.
# Each CONFIGS entry runs Load -> xDMA b_to_row -> Store -> Check and emits one
# XDMA_RUN trace event (in CONFIGS order) for the cycle LUT. Shared machinery
# lives in util/sim/xdma/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)

from xdma_ops_lib import run_op_workload  # noqa E402

# Cycle-LUT sweep: rows x cols grid (elem=1) for the bilinear fit. Uses
# array_shape=2 (mesh [16,8,16]) so tileSize=8 (%8==0) takes the HW path; the
# B-operand shape is rows = K_T*tileSize = K_T*8, cols = N_T*meshCol = N_T*16.
CONFIGS = [{"M_T": 1, "K_T": k, "N_T": n, "elem_bytes": 1, "array_shape": 2}
           for k in (4, 8, 16) for n in (1, 2, 4)]   # rows 32..128, cols 16..64

if __name__ == "__main__":
    run_op_workload("b_to_row", CONFIGS)
