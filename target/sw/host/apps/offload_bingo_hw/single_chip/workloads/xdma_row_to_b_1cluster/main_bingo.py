# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "row_to_b" op.
# Each CONFIGS entry runs Load -> xDMA row_to_b -> Store -> Check and emits one
# XDMA_RUN trace event (in CONFIGS order) for the cycle LUT. Shared machinery
# lives in util/sim/xdma/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)

from xdma_ops_lib import run_op_workload  # noqa E402

# Cycle-LUT sweep: a (K_T, N_T) tile grid per variant, for the bilinear fit. The mesh
# comes from each config's array_shape, and the B-operand it converts has shape
# rows = K_T*tileSize, cols = N_T*meshCol.
# Sweep every RUNNABLE variant of this converter: 3 array shapes x 3 element widths
# x 9 (tile) points = 81 configs.  Each (shape, width) pair is a DISTINCT kernel and a
# DISTINCT LUT (xdma_<family>_e<N>_<SHAPE>), so one workload run fills 9 LUTs.
# The grid is capped so the largest single buffer stays <= 64 KiB: the L1 src/dst pair is
# shared across configs (xdma_ops_lib), but it is sized at the max over all of them.
CONFIGS = [{"M_T": 1, "K_T": k, "N_T": n, "elem_bytes": eb, "array_shape": s}
           for s in (0, 1, 2) for eb in (1, 2, 4)
           for k in (2, 4, 8) for n in (1, 2, 4)]

if __name__ == "__main__":
    run_op_workload("row_to_b", CONFIGS)
