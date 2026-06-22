# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op xDMA workload: functional check + cycle sweep for the "xdma_6d" op.
# Each CONFIGS entry runs Load -> xDMA xdma_6d -> Store -> Check and emits one
# XDMA_RUN trace event (in CONFIGS order) for the cycle LUT. Shared machinery
# lives in util/sim/xdma/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)

from xdma_ops_lib import run_op_workload  # noqa E402

# Cycle-LUT sweep: bytes = rows*cols (elem=1). Square shapes hit clean byte
# targets 64..65536 for the linear bytes->cycles fit (built with the single-chip
# cfg whose 16 MiB host WIDE_SPM holds the baked input+golden arrays).
CONFIGS = [{"rows": s, "cols": s, "elem_bytes": 1}
           for s in (8, 16, 32, 64, 128, 256)]

if __name__ == "__main__":
    run_op_workload("xdma_6d", CONFIGS)
