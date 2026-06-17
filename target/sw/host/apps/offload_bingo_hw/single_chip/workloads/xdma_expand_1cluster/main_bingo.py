# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "expand" op.
# Each CONFIGS entry runs Load -> xDMA expand -> Store -> Check and emits one
# XDMA_RUN trace event (in CONFIGS order) for the cycle LUT. Shared machinery
# lives in util/sim/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")

from xdma_ops_lib import run_op_workload  # noqa E402

# Cycle-LUT sweep: rows x cols grid (elem=1) for the bilinear fit.
# rows = output rows (broadcast), cols = row width; both multiples of 8.
_RC = [16, 64, 128]
CONFIGS = [{"rows": r, "cols": c, "elem_bytes": 1} for r in _RC for c in _RC]

if __name__ == "__main__":
    run_op_workload("expand", CONFIGS)
