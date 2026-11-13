# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MoE prefill built stage by stage (params stage).

Each stage is the one before it plus one step of the mapping (the MoE Prefill Tracker's "The stages, step by step"), so
every step is measured and optimised on its own before the next one is added. The data, the placement, the buffers and
the build are moe_prefill_mc.MBuild's; X and the router's weights come from HBM (params x_hbm) in every stage.

stage 0     X from HBM: the root's host queues its memory chip's push engine, which multicasts the router's weights
            behind one flag, then X a tile of 8 tokens (16 KiB) at a time, each behind a flag of its own, into every
            chip's L3 (MBuild.hbm_bcasts). Every cluster waits for the router's weights and every tile. No weight chunk
            is staged or pushed. Checks: windows of X on the root, a chip in the middle and the last chip.
stage "01"  + the route (MBuild's build, params stop route): every routing cluster loads the router's weights into L1
            once they land and routes its own tile as soon as it lands (the router GEMM, its factors, the softmax and
            the top 6 into the root's table); per row block the root multicasts the table, and every cluster fetches it
            and builds its chip's row lists. The window ends when every cluster has its row lists (stage 2's input).
            params route_after_x: every cluster's route waits for all of X instead (the reference without the overlap).
            Checks: X's windows, the route table on the root, the root cluster 1's row lists.
stage "012" + the shared MLP's gate|up (params stop shared): every shared chunk's pass over the 32-row blocks of X right
            after the route, no expert slot between them. The window ends at every cluster's last SwiGLU (and row lists).
stage "0123" + the experts' gate|up (stop gateup): the 4 expert slots in slot_order, the shared MLP's later passes after
            the slots shared_after names, as in the block; no quantise. The window ends at every cluster's last SwiGLU.
stage "01234" + the quantise and the A8 exchange (stop a8): each slot quantised while the next one runs, its columns
            copied into the other cluster's A8. Checks add the root's first expert's A8.
stage full  the whole block (MBuild's build), X from HBM: + the down projection (stage 5) and the reduce and the stores
            (stage 6), which run slice by slice together and are measured as full against 01234.

The stages are cumulative because the MoE's interleave: the shared MLP's second pass runs between the expert slots, a slot
is quantised while the next one runs, and each output slice's reduce follows its GEMMs.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_prefill_mc as mp  # noqa: E402
from bingo_kernel_args import SnaxBingoKernelWaitFlagsArgs  # noqa: E402
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from driver import run_driver  # noqa: E402

STAGES = (0, "01", "012", "0123", "01234", "full")
STOP = {0: "route", "01": "route", "012": "shared", "0123": "gateup", "01234": "a8", "full": "full"}


class MStages(mp.MBuild):
    """MBuild up to a stage (the module docstring). run_driver makes the program in MBuild's four steps: __init__ (the
    parameters), mem_handles() (MBuild's: the data, the placement, the golden and the memory), build() (stage 0's task
    graph here, every other stage's MBuild's) and finish()."""

    def __init__(self, args, p, plat, hw, shared=None):
        """The parameters: the stage (params stage) and the window's end it sets (params stop), then MBuild.__init__."""
        p = dict(p)
        self.stage = p.get("stage", "full")
        if self.stage not in STAGES:
            raise ValueError(f"params stage={self.stage!r}: one of {STAGES}")
        if p.get("x_device"):
            raise ValueError("the stage workloads take X from HBM (params x_hbm)")
        p["stop"] = STOP[self.stage]                        # 0, 01: no weight chunk staged, loaded or pushed
        super().__init__(args, p, plat, hw, shared=shared)

    def check_core_order(self, header, x_prefixes):
        """The base check wants an X task before every DM core's first ring load; a stage without ring loads (0, 01)
        has nothing to check."""
        if not any((n.kernel_name or "").endswith("idma_ring_load") for n in self.dfg.node_list):
            return
        super().check_core_order(header, x_prefixes)

    def build(self):
        if self.stage != 0:
            return super().build()
        R, Tg, dm = self.ROOT, self.TAG, self.roles["dm"]
        go = self.setup()
        self.rw_hbm_cast, self.x_hbm_cast = self.hbm_bcasts(go)
        self.rw_bcast, self.x_bcast = self.rw_hbm_cast.posts[-1], self.x_hbm_cast.posts[-1]
        if self.ring_mode == "prefetcher":                 # the root's prefetcher drives the same engine
            self.edge(self.x_hbm_cast.end, self.prefetch[R])
        last = {k: [] for k in self.chips}
        for k in self.chips:
            for c in range(self.NC):
                for name, flag, n, src in (("rw_wait", "rwflag", 1, self.rw_bcast),
                                           ("x_wait_all", "xtflag", self.NXT, self.x_bcast)):
                    w = self.node(k, c, dm, name, "__snax_bingo_kernel_wait_flags",
                                  SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_{flag}"), n, 1), [go[k]])
                    self.dfg.bingo_add_order_edge(src, w)
                    last[k].append(w)
        ends = [self.probe(k, f"fin{k:02x}", last[k]) for k in self.chips if k != R]
        ends.append(self.probe(R, f"fin{R:02x}", last[R] + [self.x_hbm_cast.end]))
        self.end_window(ends)
        n = 2048
        hq = np.asarray(self.d.hq, dtype=np.int8).reshape(-1)
        checks = []
        for k in dict.fromkeys((R, self.chips[len(self.chips) // 2], self.chips[-1])):
            for t in (0, self.NXT // 2, self.NXT - 1):
                off = t * self.XT + (self.XT - n if t == self.NXT - 1 else 0)
                checks.append((f"x_k{k:02x}_t{t}", self.xland(off, None if k == R else k), hq[off:off + n].view(np.uint8), n))
        self.run_checks(checks)


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer-1 MoE prefill, stage {S.stage}, {S.T} tokens, {len(S.chips)} chips x {S.NC} "
            f"clusters, X from HBM")


def run():
    run_driver("moe prefill stage workload generator", MStages, _app_name)
