# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Per-op functional workload for the "swap" op (adjacent element-pair swap,
# dst[i] = src[i^1] — the data half of RoPE rotate_half, produced on-device via
# two strided iDMA copies so in-layer rope_q/rope_k can swap a runtime Q/K
# buffer). Each CONFIGS entry runs Load -> iDMA swap -> Store -> Check, comparing
# the device output byte-exact against a numpy golden; this sweep IS the per-op
# functional test. (The op runs on the iDMA, so it emits IDMA_RUN markers, not the
# XDMA_RUN the xDMA cycle-LUT keys on — cycle characterization is not its job
# here.) Shared machinery lives in util/sim/xdma/xdma_ops_lib.py.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/util/sim")
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)

from xdma_ops_lib import run_op_workload  # noqa E402

# Sweep total element count x element width. num_elems must be even (pairs never
# cross the buffer end); elem_bytes 2 is the fp16 RoPE case. The iDMA does two
# strided passes, so cost scales with num_elems (~2 cyc/elem).
_NE = [64, 256, 1024]
CONFIGS = [{"num_elems": ne, "elem_bytes": eb} for ne in _NE for eb in (1, 2, 4)]

if __name__ == "__main__":
    run_op_workload("swap", CONFIGS)
