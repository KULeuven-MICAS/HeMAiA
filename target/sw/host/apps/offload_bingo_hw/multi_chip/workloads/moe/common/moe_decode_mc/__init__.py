# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MoE for ONE token (a decode step), every weight multiply on VersaCore's one-token GEMV:
DBuild, the MoE block of the whole layer-1 decode (../../dsv2/common/dsv2_decode_mc.py). README.md explains the
mapping, the steps and the parameters; one module a part of the build:

    params.py      the switches          placement.py   the pieces on the chips, their columns on the clusters
    memory.py      data, golden, L3, L1  build.py       the task graph, phase by phase
    checks.py      the checks            common.py      the a_row, the spread, the padded-row multiply, Piece
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # moe/common
# importing moe_prefill_mc first puts util/sim/multichip, util/sim/common and mini_compiler on sys.path
from moe_prefill_mc.common import ROUTE_B  # noqa: E402
from driver import run_driver  # noqa: E402

import chiplet_mc as cm  # noqa: E402

from .build import Build  # noqa: E402
from .checks import Checks  # noqa: E402
from .common import a_row_bytes  # noqa: E402
from .memory import Memory  # noqa: E402
from .params import Params  # noqa: E402
from .placement import Placement  # noqa: E402

__all__ = ["DBuild", "ROUTE_B", "a_row_bytes", "run"]


class DBuild(Params, Placement, Memory, Build, Checks, cm.ChipletBuild):
    """The MoE decode block (README.md). run_driver makes the program in four steps: __init__ (the parameters,
    params.py), mem_handles() (the data, the placement, the golden and the memory, memory.py), build() (the task graph:
    every task and edge, build.py) and finish() (the compile into the C header and the data image)."""
    TAG = "md"


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer-1 MoE decode, one token, {len(S.chips)} chips x {S.NC} clusters, routed experts "
            f"over {S.routed_split} chips each, X from HBM, y to HBM, every weight multiply on the GEMV")


def run():
    run_driver("moe decode workload generator", DBuild, _app_name)
