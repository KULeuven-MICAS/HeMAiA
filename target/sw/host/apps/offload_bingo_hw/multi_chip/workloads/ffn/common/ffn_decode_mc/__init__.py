# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The FFN block of the decode layers: DBuild, ffn_prefill_mc's PBuild for ONE token, every weight multiply on
VersaCore's one-row GEMV. ../../../dsv2/common/dsv2_decode_mc.py and ../../../llama3/common/llama3_decode_layer_mc.py
extend it as their FfnPart. README.md explains the mapping and the parameters; the modules:

    params.py    the decode's switches and its refusals of the prefill's multi-row ones (DBuild.__init__)
    memory.py    X, Y and the weights in HBM, every L1 buffer a row
    rows.py      X's fetch, the gate|up GEMVs, the A16 row into the chip's A8 row
    passes.py    the pass: the down GEMVs, the ChainGathers, Y's row, Y into HBM
    checks.py    every array task a GEMV, the data checks
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # ffn/common: ffn_prefill_mc, ffn_data
sys.path.insert(0, HERE)
import ffn_prefill_mc as fp  # noqa: E402  (puts util/sim/multichip and the mini_compiler on sys.path)

from .params import DParams  # noqa: E402
from .memory import DMemory  # noqa: E402  (imports ffn_data, which puts util/sim/gemm on sys.path for checks.py)
from .rows import DRows  # noqa: E402
from .passes import DPasses  # noqa: E402
from .checks import DChecks  # noqa: E402

__all__ = ["DBuild"]


class DBuild(DParams, DMemory, DRows, DPasses, DChecks, fp.PBuild):
    """The FFN decode block (README.md), the prefill's PBuild run as one pass of one token on the GEMV. run_driver makes
    the program in four steps: __init__ (the parameters), mem_handles() (the data, the golden and the memory), build()
    (the task graph: every task and edge) and finish() (the buffer-hazard check, then the compile into the C header and
    the data image)."""
    TAG = "fd"
    MIN_POST_BLOCKS = 1
