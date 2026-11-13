# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The driver of the MoE prefill workloads (workloads/moe/dsv2_l1_prefill_ablation/<rung>) and the MoE block of the
whole DeepSeek-V2-Lite layer-1 prefill: MBuild. README.md explains the mapping, the phases and the parameters; one
module a part of the build:

    params.py      the switches          placement.py   experts on chips, columns on clusters
    memory.py      data, golden, L3, L1  tasks.py       a chunk's load and release, one GEMM, the HBM casts
    x_route.py     X, the route, the shared MLP's first pass
    gate_up.py     the experts' gate|up, the quantise, the A8 exchange
    down.py        the down projection, the reduce, the stores of Y
    checks.py      the checks and finish
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # moe/common
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "common"))
from driver import repo_root, run_driver  # noqa: E402
REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402

import chiplet_mc as cm  # noqa: E402

from .checks import Checks  # noqa: E402
from .common import DTMP_BUFS, MESH3, NOUT  # noqa: E402
from .down import Down  # noqa: E402
from .gate_up import GateUp  # noqa: E402
from .memory import Memory  # noqa: E402
from .params import Params  # noqa: E402
from .placement import Placement  # noqa: E402
from .tasks import Tasks  # noqa: E402
from .x_route import XRoute  # noqa: E402


__all__ = ["MBuild", "MESH3", "run"]


class MBuild(Params, Placement, Memory, Tasks, XRoute, GateUp, Down, Checks, cm.ChipletBuild):
    """The MoE prefill block (README.md). run_driver makes the program in four steps: __init__ (the parameters,
    params.py), mem_handles() (the data, the placement, the golden and the memory, memory.py), build() (the task graph,
    below: every task and edge) and finish() (the compile into the C header and the data image, in which the compiler
    orders every core's tasks; checks.py)."""
    TAG = "mp"

    def build(self):
        """The phases in the order their tasks are made: the window's start with X's and the router's casts; every
        cluster's route of its own tiles; X's row blocks with the shared MLP's first pass (and, tuned, the route's
        uploads, the root's table and every cluster's row lists behind the first blocks); the rings after X; the rest
        of the route; the experts' gate|up slot by slot; the down projection and the reduce slice by slice."""
        N_CL = self.N_CL
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        go = self._open_window()
        # the build's state across the phases
        self._xblk_reader = {cl: [[], []] for cl in range(N_CL)}    # each row buffer's readers
        self._sg_gms = {}                                   # (cl, chunk) -> its shared gate|up GEMMs, every X block
        self._gb_reader = {cl: [None, None] for cl in range(N_CL)}
        self.gcount = {cl: 0 for cl in range(N_CL)}
        self._sh_ld = {}
        self._last_sw = {cl: None for cl in range(N_CL)}
        self._rt_tile = {}                                  # tile -> (cluster, its top 6)
        self._rt_ups, self._rt_wait, self._disp = {}, {}, {}
        self._rputs, self._rput_to = [], {}                 # each block's table put; x_cast unicast: (block, chip)
        self._sh_qlast, self._sh_cp = {}, {}                # each cluster's last shared quantise and staged copy
        self._route_tiles_first(go)
        if self.per_kernel:
            self._route_whole(go)
        self._x_blocks(go)
        self._rings_after_x()
        self._route_rest(go)
        self._qa, self._ex_copy = {}, {}                    # (cl, slot or "sh") -> its quantise, its exchange copy
        self._a16_reader = {cl: None for cl in range(N_CL)}
        self._root_gat = {}                                 # each root cluster's last read of X
        # the row blocks' last readers (one GEMM core, in order) hold the expert row buffers first
        self._xe_reader = {cl: [list(self._xblk_reader[cl][i]) for i in range(2)] for cl in range(N_CL)}
        self._pend = {cl: [] for cl in range(N_CL)}         # exchange copies created one slot later
        self._held_ex = {cl: [] for cl in range(N_CL)}      # barrier_at kernel: every exchange, after the gate|up
        self._experts()
        self._xch_reader = {cl: [None] * NOUT for cl in range(N_CL)}
        self._dt_reader = {cl: [None] * DTMP_BUFS for cl in range(N_CL)}
        self._ndt_used = {cl: 0 for cl in range(N_CL)}
        self._p1_reader = {c: [None, None] for c in range(self.NC)}
        self._jbase, self._stores = {}, []
        self._last_red, self._last_any = {c: None for c in range(self.NC)}, None
        self._last_ld = {}                                  # every cluster's last weight load
        self._down()
        self.end_window(self._stores[-self.NC:])
        self._checks()


def _app_name(S):
    arms = "" if (S.x_cast, S.reduce_impl, S.sync) == ("multicast", "chain", "edge") else \
        f" x_cast={S.x_cast}" + (f" barrier_at={S.barrier_at}" if S.barrier_at else "")
    return (f"DeepSeek-V2-Lite layer-1 MoE prefill, {S.T} tokens, {len(S.chips)} chips x {S.NC} clusters, X from HBM, "
            f"bcast={S.bcast_impl} reduce={S.reduce_impl} sync={S.sync}" + arms)


def run():
    run_driver("moe prefill workload generator", MBuild, _app_name)
