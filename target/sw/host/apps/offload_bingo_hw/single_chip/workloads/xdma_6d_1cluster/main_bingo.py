# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "xdma_6d" op.
# Each CONFIGS entry runs Load -> xDMA xdma_6d -> Store -> Check and emits one
# XDMA_RUN trace event (in CONFIGS order) for the cycle LUT. Shared machinery
# lives in util/sim/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")

from xdma_ops_lib import run_op_workload  # noqa E402

CONFIGS = [
    {"rows": 16, "cols": 16, "elem": 1},
    {"rows": 32, "cols": 16, "elem": 1},
    {"rows": 32, "cols": 32, "elem": 1},
]

if __name__ == "__main__":
    run_op_workload("xdma_6d", CONFIGS)
