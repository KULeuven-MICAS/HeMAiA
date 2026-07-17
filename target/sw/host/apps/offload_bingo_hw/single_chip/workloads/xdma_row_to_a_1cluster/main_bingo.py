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

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Per-op row-major to VersaCore A-layout conversion sweep over nine
# configurations.
#
# Task dependency graph:
#
# For each config i = 0..8:
#   Load_i -> Op_i -> Store_row_to_a_cfg[i] -> Check_row_to_a_cfg[i]
#
# Config ordering:
#   Check_row_to_a_cfg[i] -> Load_[i+1]    for i = 0..7
# END WORKLOAD DESCRIPTION AND TASK GRAPH

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)

from xdma_ops_lib import run_op_workload  # noqa E402

# Cycle-LUT sweep: an (M_T, K_T) tile grid per variant, for the bilinear fit. The mesh
# comes from each config's array_shape, and the A-operand it converts has shape
# rows = M_T*meshRow, cols = K_T*tileSize.
# Sweep every RUNNABLE variant of this converter: 3 array shapes x 3 element widths
# x 9 (tile) points = 81 configs.  Each (shape, width) pair is a DISTINCT kernel and a
# DISTINCT LUT (xdma_<family>_e<N>_<SHAPE>), so one workload run fills 9 LUTs.
# The grid is capped so the largest single buffer stays <= 64 KiB: the L1 src/dst pair is
# shared across configs (xdma_ops_lib), but it is sized at the max over all of them.
CONFIGS = [{"M_T": m, "K_T": k, "N_T": 1, "elem_bytes": eb, "array_shape": s}
           for s in (0, 1, 2) for eb in (1, 2, 4)
           for m in (1, 2, 4) for k in (4, 8, 16)]

if __name__ == "__main__":
    run_op_workload("row_to_a", CONFIGS)
