# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "elementwise_add" op.
# Each CONFIGS entry runs Load -> xDMA elementwise_add -> Store -> Check and emits one
# XDMA_RUN trace event (in CONFIGS order) for the cycle LUT. Shared machinery
# lives in util/sim/xdma/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)

from xdma_ops_lib import run_op_workload  # noqa E402

# Cycle-LUT sweep: n = per_op (output int32 elems); num_operands fixed = 2 (the
# cost model's reduce_cc queries this op by n). per_op is a multiple of 16
# (one 512b bus word = 16 int32) for the linear n->cycles fit.
CONFIGS = [{"num_operands": 2, "per_op": p} for p in (16, 64, 256, 1024, 4096)]

if __name__ == "__main__":
    run_op_workload("elementwise_add", CONFIGS)
