# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "row_to_d" op.
# Each CONFIGS entry runs Load -> xDMA row_to_d -> Store -> Check and emits one
# XDMA_RUN trace event (in CONFIGS order) for the cycle LUT. Shared machinery
# lives in util/sim/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")

from xdma_ops_lib import run_op_workload  # noqa E402

# Cycle-LUT sweep: rows x cols grid (elem=1) for the bilinear fit. With
# array_shape=0 (meshRow=32, meshCol=32) the D-operand shape is
# rows = M_T*32, cols = N_T*32; K_T is unused here (set 1).
CONFIGS = [{"M_T": m, "K_T": 1, "N_T": n, "elem_bytes": 1}
           for m in (1, 2, 4) for n in (1, 2, 4)]   # rows 32..128, cols 32..128 (mesh 32x32)

if __name__ == "__main__":
    run_op_workload("row_to_d", CONFIGS)
