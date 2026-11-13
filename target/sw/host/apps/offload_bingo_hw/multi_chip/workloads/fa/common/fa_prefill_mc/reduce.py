# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaBuild's reduce: one ChainGather a W_O block and root cluster, folding the 16 heads' partials into the block's
collector, and the store of each block's sum into the collector's L3."""

import chiplet_mc as cm
from bingo_kernel_args import SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelXdmaChainGatherArgs

from .consts import EW_F16_ADD


class FaReduce:

    def stage_reduce(self, i, parts, war, gate, half, src_off, clusters):
        """Cluster index c's partials of W_O block `half` of chunk i into its collector's cluster c, one ChainGather
        along the fold cycle from the collector (y_root). parts {(k, c): [tasks after which the partial is there]},
        war {c: [tasks the destination waits for]}, gate {c: [tasks the reduce starts behind]}. Returns {c: the
        chain}. The last chunk's two clusters' chains run at the same time, after the chain before them and after every
        chip's ring has landed its last chunk; every other chain runs after the one before unless chain_concurrent."""
        R, xdma = self.y_root(i, half), self.roles["xdma"]
        others = self.y_order(R)[1:]
        out, before = {}, {}
        cs = list(clusters)
        for c in cs:
            before[c] = [t for k in self.chips for t in parts[(k, c)]] + list((gate or {}).get(c, []))
        last_pair = i == self.NCH - 1 and len(cs) == self.NC
        for c in cs:
            dstb = self.dst(i, c, half)
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, c, others, src=self.xch_addr(c, src_off),
                dst=f"((uint64_t)({dstb.get_c_var_name()}) + 0)", size=self.YB // 2,
                junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[dstb])
            serial = [] if self.chain_concurrent or self._last_chain is None else [self._last_chain]
            # the chains that wait for every chip's pushes: after the landing of its rings' last chunk
            # (after_pushes_landed)
            landed = False
            if last_pair:
                if self._pair_prev is None:
                    self._pair_prev = [self._last_chain] if self._last_chain is not None else []
                serial = list(self._pair_prev)
                landed = True
            out[c] = self.chain_gather(R, c, f"y_chain_c{c}_i{i}_h{half}", ga,
                               before[c] + list(war.get(c, [])) + serial)
            if landed:
                self.after_pushes_landed(self, out[c])
            self._last_chain = out[c]
            self._chains.append(out[c])
        if last_pair:
            self._pair_prev = [out[c] for c in cs]
        return out

    def stage_store(self, i, red, half, clusters):
        """The collector's cluster c stores W_O block `half` of chunk i's rows of y to its L3."""
        R, dm = self.y_root(i, half), self.roles["dm"]
        st = {}
        for c in clusters:
            nb, off = self.YB // 2, half * self.YB // 2
            st[c] = self.node(R, c, dm, f"y_store_c{c}_i{i}_h{half}", "__snax_bingo_kernel_idma_1d_copy",
                              SnaxBingoKernelIdma1dCopyArgs(self.dst(i, c, half), self.y_sym(i, c, off), nb),
                              list(red[c]) if isinstance(red[c], (list, tuple)) else [red[c]])
        return st

    def _block_chains(self, i, cs, war, gate, red, stores, xch_free):
        """Root clusters cs's chains of chunk i, one a W_O block, each block's sum in its own half-size buffer, stored
        right after its chain (the second block's store after the first's)."""
        cs = list(cs)
        half = self.YB // 2
        rh = []
        for r in range(2):
            parts = {(k, c): list(self._wo_blk[(k, c, i, r)]) for k in self.chips for c in cs}
            rh.append(self.stage_reduce(i, parts, war if r == 0 else {}, gate if r == 0 else None, half=r,
                                        src_off=r * half, clusters=cs))
            st = self.stage_store(i, rh[r], half=r, clusters=cs)
            if r == 0:
                h0 = st
            else:
                # the second half's store after the first's, so it stands for the chunk's stores
                for c in cs:
                    self.edge(h0[c], st[c])
        for c in cs:
            red[(i, c)] = rh[1][c]
            stores[(i, c)] = st[c]
            for k in self.chips:
                xch_free[(k, c)] = [rh[0][c], rh[1][c]]

    def chunk_start(self, i):
        """Where the tasks of chunk i (i >= 1) begin, in the order they are made: nothing here; a whole layer
        starts an args phase there."""

    def _reduce_chunk(self, i, red, stores, xch_free, gate=None):
        """Chunk i's chains and stores, the destinations behind the chunk before's stores."""
        war = {c: [stores[(i - 1, c)]] if (i - 1, c) in stores else [] for c in range(self.NC)}
        self._block_chains(i, range(self.NC), war, gate, red, stores, xch_free)
