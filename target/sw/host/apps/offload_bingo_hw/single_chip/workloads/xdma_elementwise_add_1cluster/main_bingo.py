# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "elementwise_add" op.
# Each CONFIGS entry runs Load -> xDMA elementwise_add -> Store -> Check and emits one
# XDMA_RUN trace event (in CONFIGS order) for the cycle LUT. Shared machinery
# lives in util/sim/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")

from xdma_ops_lib import run_op_workload  # noqa E402

CONFIGS = [
    {"num_operands": 2, "per_op": 64},
    {"num_operands": 4, "per_op": 64},
    {"num_operands": 4, "per_op": 256},
    {"num_operands": 8, "per_op": 64},
]

if __name__ == "__main__":
    run_op_workload("elementwise_add", CONFIGS)
