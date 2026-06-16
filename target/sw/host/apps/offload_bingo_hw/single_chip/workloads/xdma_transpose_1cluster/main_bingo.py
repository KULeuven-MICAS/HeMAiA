# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "transpose" op.
# Each CONFIGS entry runs Load -> xDMA transpose -> Store -> Check and emits one
# XDMA_RUN trace event (in CONFIGS order) for the cycle LUT. Shared machinery
# lives in util/sim/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")

from xdma_ops_lib import run_op_workload  # noqa E402

# Element-width coverage (see __snax_bingo_kernel_xdma_transpose_2d):
#   - int8  -> HW transposer, 8-bit mode
#   - int16 -> HW transposer, native 16-bit mode (CSR0=1)
#   - int32 -> SW transpose (no native 32-bit HW mode)
# The int16 HW-16-bit path is newly wired; this sweep (real numpy goldens for
# every width) is the check that confirms it — run it in sim after a build.
CONFIGS = [
    {"rows": 8, "cols": 8, "elem_bytes": 1},
    {"rows": 16, "cols": 16, "elem_bytes": 1},
    {"rows": 32, "cols": 32, "elem_bytes": 1},
    {"rows": 16, "cols": 16, "elem_bytes": 2},   # int16 -> HW 16-bit mode
    {"rows": 16, "cols": 16, "elem_bytes": 4},   # int32 -> SW transpose
]

if __name__ == "__main__":
    run_op_workload("transpose", CONFIGS)
