# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaStages, the builder of the whole FA prefill block (README.md): fa_prefill_mc.FaBuild's block, stages 0 to 8 in one
window (workloads/fa/dsv2_l1_prefill/full_n, t1024, stage01234 with stop q8, and the attention of the DSv2 layer
prefill). Each stage alone is its own app (workloads/fa/dsv2_l1_prefill/stage<N>/main_bingo.py).

    params    the block builder's switches
    inputs    the keys' flags, the copies before the window
    ring      the root memory chip's engine queue order
    x         stage 0: x from HBM into every chip's L3; a cluster's rows of a chunk
    keys      stage 1: the keys by columns (kv_balance: helpers for the two-pair chips)
    owner     stages 2 and 3: the owners finish the keys, the root multicasts them
    queries   stage 4: a chunk's query tiles; stage 5: its sweep
    build     build(): the whole block
    order     keys_by_chunk: the order of the collectives and the key tasks
    checks    the checks' waits; finish()
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # fa/common
import fa_prefill_mc as fa  # noqa: E402
from driver import run_driver  # noqa: E402

from .build import FaStagesBuild  # noqa: E402
from .checks import FaStagesChecks  # noqa: E402
from .inputs import FaStagesInputs  # noqa: E402
from .keys import FaStagesKeys  # noqa: E402
from .order import FaStagesOrder  # noqa: E402
from .owner import FaStagesOwner  # noqa: E402
from .params import FaStagesParams  # noqa: E402
from .queries import FaStagesQueries  # noqa: E402
from .ring import FaStagesRing  # noqa: E402
from .x import FaStagesX  # noqa: E402

__all__ = ["FaStages", "run"]


class FaStages(FaStagesParams, FaStagesInputs, FaStagesRing, FaStagesX, FaStagesKeys, FaStagesOwner, FaStagesQueries,
               FaStagesBuild, FaStagesOrder, FaStagesChecks, fa.FaBuild):
    """FaBuild's block whole (README.md), in FaBuild's four steps: __init__ (the parameters), mem_handles()
    (FaBuild's, whose _stage here adds the keys' flags), build() and finish()."""


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer 1 MLA attention prefill, stage full, {S.T} tokens, {len(S.chips)} chips x "
            f"{S.NC} clusters, x from HBM" + (", split loads" if S.x_xdma else ""))


def run():
    run_driver("fa prefill block workload generator", FaStages, _app_name)
