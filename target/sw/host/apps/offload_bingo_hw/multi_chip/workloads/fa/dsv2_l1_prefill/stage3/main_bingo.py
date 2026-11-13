#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention, stage 3 of the 256-token prefill on sixteen chiplets alone: the keys'
pieces from their owners into every chip's L3.

Stage 2's result (each owner's pieces: key rows then value columns, INT8) is in the owner's memory chip's HBM and is
copied into the L1 of its tiles' cluster 0 before the window. Each owner puts its pieces into the root's L3 behind a
flag a piece (kvnflag); the root waits for a chunk's pieces and multicasts them from its L3 into every other chip's
L3 behind the chunk's flag (kvflag), one multicast at a time on its cluster 1's DM core; every chip then waits until
all its keys have landed.

          before the window     kvp_pre_t<lt>                 DM: each owner's pieces from HBM into L1
          S0_start
          every owner: kv_put_t<lt>                           DM: the piece into the root's L3, behind its flag
          the root, chunk h: kv_nwait_h<h>, kv_bput_h<h>      its pieces landed, then their multicast (a broadcast)
          every chip: kv_all                                  every piece (the root) or every chunk (the others)
          P_fin<chip> -> S1_end -> fp_report, the checks

Checks: three tiles' pieces in the L3 of the root, of the chip in the middle of the fold order and of the last (the
key rows' head, their k_pe, the value columns' tail).

params.hjson: the platform, the tokens, the chunk and key-tile sizes, the weight chunk and ring sizes, the weights'
width."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from stage_common import KV_RANK, KVP, BingoMemSymbol, StageBase, app_name  # noqa: E402
from bingo_kernel_args import SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelWaitFlagsArgs  # noqa: E402
from driver import run_driver  # noqa: E402

CHK = 2048                                  # bytes a checked window


class Stage3(StageBase):
    """Stage 3 alone (the module docstring)."""
    STAGE = 3

    def stage_inputs(self):
        """Each owner's pieces in its memory chip's HBM (kv_piece: key rows then value columns, INT8), copied into the
        owner's L1 before the window."""
        d, T = self.d, self.TAG
        self.h_kvp = {}
        for k in self.chips:
            img = np.concatenate([np.asarray(d.kv_piece(16 * q)).view(np.uint8).reshape(-1) for q in self.kv_tiles[k]])
            m = self.mem[k]
            self.h_kvp[k] = self.st.put_hbm(f"{T}_kvp_k{k:02x}", img.view(np.int8), mem_chip=(m >> 4, m & 15))

    def build(self):
        self.last_ld = {}
        for ls in self.ls:
            ls.strict = True
        go = self.setup(extra=self.stage3_preload())
        self.relays()
        self.x_puts = []
        last = {k: [] for k in self.chips}
        self.stage_keys_out(go, last)
        self.window_end(last)
        self.run_checks(self.piece_checks())

    def stage3_preload(self):
        """Before the window, per chip its pieces from HBM into the L1 of their tiles' cluster 0 (s3_piece)."""
        T = self.TAG
        pre = {k: [] for k in self.chips}
        self.s3_piece = {}
        for k in self.chips:
            for lt, q in enumerate(self.kv_tiles[k]):
                cl = self.chips.index(k) * self.NC
                self.s3_piece[q] = self.cx_chip(k).at(0).l1(f"{T}_s3_piece_t{lt}", KVP)
                pre[k] = [self.dm_copy(cl, f"kvp_pre_t{lt}", self.h_kvp[k].address + lt * KVP, self.s3_piece[q], KVP,
                                       pre[k])]
        return pre

    # ---- the pieces into every chip -------------------------------------------------------------------------------
    def stage_keys_out(self, go, last):
        """Each owner puts its pieces into the root's L3 behind a flag, the root multicasts a chunk's pieces at a time
        once they have landed (stage_kv_bcast), then a wait a chip until all have landed."""
        R, T, dm, n = self.ROOT, self.TAG, self.roles["dm"], self.T // 16

        def wait_on(k, c, name, flag, nf, puts):
            # the flags prove the landing; the puts are ordering edges where they are not this chip's
            w = self.node(k, c, dm, name, "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(flag[0], offset=flag[1]), nf, 1),
                          [t for t in puts if self.flag_dep(t, k, c)])
            for t in puts:
                if not self.flag_dep(t, k, c):
                    self.dfg.bingo_add_order_edge(t, w)
            return w

        self.kv_put = {}
        for q in sorted(self.kv_owner):
            k = self.kv_owner[q]
            on_root = lambda name, off: BingoMemSymbol(name, offset=off, chip_id=None if k == R else R)  # noqa: E731
            self.kv_put[q] = self.node(k, 0, dm, f"kv_put_t{self.kv_tiles[k].index(q)}",
                                       "__snax_bingo_kernel_idma_put_flagged",
                                       SnaxBingoKernelIdmaPutFlaggedArgs(self.s3_piece[q], on_root(f"{T}_kv", q * KVP),
                                                                         KVP, on_root(f"{T}_kvnflag", q * 64),
                                                                         value=1), [go[k]])
        self.stage_kv_bcast()
        for k in self.chips:
            if k == R:
                last[k].append(wait_on(k, 0, "kv_all", (f"{T}_kvnflag", 0), n, list(self.kv_put.values())))
            else:
                last[k].append(wait_on(k, 0, "kv_all", (f"{T}_kvflag", 0), self.NCH, self.kv_mc))

    def stage_kv_bcast(self):
        """The root multicasts the keys a chunk's pieces at a time, from its L3 into every other chip's, once their
        pieces have landed (on the DM core of x_put_cluster); the compiler orders the multicasts."""
        R, T, P = self.ROOT, self.TAG, self.C // 16
        c, dm = self.x_put_cluster, self.roles["dm"]
        self.kv_mc = []
        for h in range(self.NCH):
            puts = [self.kv_put[q] for q in range(h * P, (h + 1) * P)]
            w = self.node(R, c, dm, f"kv_nwait_h{h}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvnflag", offset=h * P * 64), P, 1),
                          [t for t in puts if self.flag_dep(t, R, c)] + self.kv_mc[-1:])
            for t in puts:
                if not self.flag_dep(t, R, c):
                    self.dfg.bingo_add_order_edge(t, w)
            src = BingoMemSymbol(f"{T}_kv", offset=h * P * KVP)
            self.kv_mc.append(self.broadcast_put(R, c, f"kv_bput_h{h}", SnaxBingoKernelIdmaPutFlaggedArgs(
                src, src, P * KVP, BingoMemSymbol(f"{T}_kvflag", offset=h * 64), value=1, bcast=True), [w]))

    # ---- checks -------------------------------------------------------------------------------------------------
    def piece_checks(self):
        """Three tiles' pieces in the L3 of the root, of the chip in the middle of the fold order and of the last."""
        d, T, R, n = self.d, self.TAG, self.ROOT, CHK
        far = self.order[len(self.order) // 2]
        checks = []
        for k in dict.fromkeys((R, far, self.order[-1])):
            for q in (0, self.NX // 2, self.NX - 1):
                img = np.asarray(d.kv_piece(16 * q)).view(np.uint8).reshape(-1)
                for nm, off in (("a", 0), ("p", 16 * KV_RANK), ("b", KVP - n)):
                    checks.append((f"kv_k{k:02x}_q{q}{nm}", BingoMemSymbol(f"{T}_kv", offset=q * KVP + off,
                                                                           chip_id=None if k == R else k),
                                   img[off:off + n], n))
        return checks


if __name__ == "__main__":
    run_driver("fa prefill stage 3 workload generator", Stage3, app_name)
