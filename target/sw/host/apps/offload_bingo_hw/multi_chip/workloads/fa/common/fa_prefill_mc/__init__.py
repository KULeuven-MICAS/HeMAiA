# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaBuild, the FA prefill block (README.md): DeepSeek-V2-Lite's layer-1 MLA attention over a fresh prompt, head
parallel over the 16 + 4 chiplet platform. fa_stages_mc.FaStages builds the whole block from it.

    consts    the shapes and the mapping's arithmetic
    params    the parameters: the shapes, the head of each chip, every switch
    memory    the data, the staged inputs, the L1 buffers, where y lands
    ring      the weight ring and its loads, the sweep's key and value tiles
    ops       one task of one cluster: GEMM, SIMD, xDMA, copy
    inputs    a cluster's wait for the keys
    post      W_UV and W_O after the sweep
    reduce    the reduce chains and the stores of y
    checks    the checks
"""

from .consts import (BR, D_QK, D_V, EW_F16_ADD, HIDDEN, KB, KV_RANK, KVP, Q_HEAD, Q_NOPE, REPO, ROPE,  # noqa: F401
                     TILE16, VB, check)
import chiplet_mc as cm  # noqa: E402

from .checks import FaChecks  # noqa: E402
from .inputs import FaInputs  # noqa: E402
from .memory import FaMemory  # noqa: E402
from .ops import FaOps  # noqa: E402
from .params import FaParams  # noqa: E402
from .post import FaPost  # noqa: E402
from .reduce import FaReduce  # noqa: E402
from .ring import FaRing, FreshTileSource  # noqa: E402,F401


__all__ = ["BR", "D_QK", "D_V", "EW_F16_ADD", "HIDDEN", "KB", "KV_RANK", "KVP", "Q_HEAD", "Q_NOPE", "REPO", "ROPE",
           "TILE16", "VB", "check", "FaBuild", "FreshTileSource"]


class FaBuild(FaParams, FaMemory, FaRing, FaOps, FaInputs, FaPost, FaReduce, FaChecks, cm.ChipletBuild):
    """The MLA prefill block (README.md): its parameters (FaParams), data and memory (FaMemory), ring (FaRing), task
    helpers (FaOps), inputs (FaInputs), W_UV and W_O (FaPost), reduce (FaReduce) and checks (FaChecks). A builder
    (fa_stages_mc.FaStages) adds build()."""
    TAG = "fp"
