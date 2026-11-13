# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the ffn prefill workloads: PBuild, one LLM's dense SwiGLU FFN over a prompt of T tokens,
tensor-parallel over every cluster of the 16 + 4 chiplet platform. README.md explains the mapping and the parameters;
the modules:

    params.py    every switch, parsed and checked (PBuild.__init__)
    memory.py    the data, the plan (column split, gate|up chunks, down slices), the HBM images, the L1 buffers
    x.py         X: its landing area, its multicast from HBM, its fetches; Y_raw's places in L3
    gate_up.py   the weight loads, gate|up, the post-processing into the chip's A8
    passes.py    the window and the passes (sync edge): the down slices, the ChainGathers, the stores of Y_raw
    kernel.py    params sync barrier: the passes kernel by kernel and the reduce as one kernel
    checks.py    the data checks, the buffer-hazard proof, finish()
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # ffn/common: ffn_data, ffn_decode_mc
_REPO = os.path.abspath(__file__).split("/target/sw/host/")[0]
sys.path.insert(0, os.path.join(_REPO, "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(_REPO, "util", "sim", "common"))
sys.path.insert(0, f"{_REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
from driver import run_driver  # noqa: E402

import chiplet_mc as cm  # noqa: E402

from .checks import PChecks  # noqa: E402
from .consts import BLK, EW_F16_ADD, REPO, TILE  # noqa: E402,F401
from .gate_up import PGateUp  # noqa: E402
from .kernel import PKernel  # noqa: E402
from .memory import PMemory  # noqa: E402
from .params import PParams  # noqa: E402
from .passes import PPasses  # noqa: E402
from .x import PX  # noqa: E402

# what other modules take from here: the builder, the entry point, and the constants ffn_decode_mc shares
__all__ = ["PBuild", "run", "BLK", "EW_F16_ADD", "REPO", "TILE"]


class PBuild(PParams, PMemory, PX, PGateUp, PPasses, PKernel, PChecks, cm.ChipletBuild):
    """The FFN prefill block (README.md). run_driver makes the program in four steps: __init__ (the parameters,
    params.py), mem_handles() (the data, the golden and the memory, memory.py), build() (the task graph: every task and
    edge, passes.py) and finish() (the buffer-hazard check, then the compile into the C header and the data image, in
    which the compiler orders every core's tasks, checks.py)."""
    TAG = "pf"
    # the fewest post blocks a pass may have (params post_blocks); the decode (ffn_decode_mc), one row tile, has one
    MIN_POST_BLOCKS = 2


def _app_name(S):
    arms = "" if (S.x_cast, S.reduce_impl, S.sync) == ("multicast", "chain", "edge") else \
        f"; X {S.x_cast}, reduce {S.reduce_impl}, sync {S.sync}"
    return (f"{S.p.get('model', 'dsv2-lite-l0')} FFN prefill, {S.T} tokens in {S.NI} x {S.ROWS}, "
            f"{len(S.chips)} chips x {S.NC} clusters, X from HBM{arms}")


def run():
    run_driver("ffn prefill workload generator", PBuild, _app_name)
