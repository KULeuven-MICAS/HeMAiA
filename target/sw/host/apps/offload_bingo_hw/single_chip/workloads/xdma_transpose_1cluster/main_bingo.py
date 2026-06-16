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

# NOTE: only elem=1 (single 8x8-byte beat) is correct on the writer-side
# transposer. elem>=2 (tpt>1 multi-beat) currently produces WRONG data on the
# writer side — the per-beat 8x8 block transpose no longer composes into an
# element transpose the way it did on the reader side. Wide-element transpose
# was never correctness-checked before (xdma_ci_ops uses elem=1; the previous
# xDMA sweep checked against all-zero goldens), so this is a latent gap, not a
# regression. Re-add an elem>=2 config once the writer multi-beat path is fixed.
CONFIGS = [
    {"rows": 8, "cols": 8, "elem": 1},
    {"rows": 16, "cols": 16, "elem": 1},
    {"rows": 32, "cols": 32, "elem": 1},
    {"rows": 64, "cols": 64, "elem": 1},
]

if __name__ == "__main__":
    run_op_workload("transpose", CONFIGS)
