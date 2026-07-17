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

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Per-op xDMA concat sweep over nine configurations. Each config concatenates a
# top half and bottom half before storing and checking the result.
#
# Task dependency graph:
#
# For each config i = 0..8:
#   LoadTop_i -> ConcatTop_i -> LoadBot_i -> ConcatBot_i
#   ConcatBot_i -> Store_concat_cfg[i] -> Check_concat_cfg[i]
#
# Config ordering:
#   Check_concat_cfg[i] -> LoadTop_[i+1]    for i = 0..7
# END WORKLOAD DESCRIPTION AND TASK GRAPH

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)

from xdma_ops_lib import run_op_workload  # noqa E402

# Cycle-LUT sweep: rows x cols grid (elem=1) for the bilinear fit. rows is a
# multiple of 16 (concat assembles two rows/2 halves, each %8). The op emits
# TWO XDMA_RUN events per config (top+bottom) -> the extractor sums them.
_RC = [16, 64, 128]
CONFIGS = [{"rows": r, "cols": c, "elem_bytes": 1} for r in _RC for c in _RC]

if __name__ == "__main__":
    run_op_workload("concat", CONFIGS)
