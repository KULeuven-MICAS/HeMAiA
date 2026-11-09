# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The driver of workloads/fa/llama3_decode/<variant>/: Llama-3-8B's grouped-query attention for ONE decode token over a
KV cache of `keys` positions (synthetic INT4 weights, fa_llama_data.py), split by key over every cluster of the 16 + 4
chiplet platform -- fa_mc.py's mapping (DeepSeek-V2-Lite's MLA) for the 32 heads of a GQA layer. The platform, the
rings, the stamps, the sync arm, the broadcast and the reduce are util/sim/multichip/chiplet_mc.py's.

EVERY INPUT COMES FROM HBM. The query (both halves, 16 KiB) sits in the HBM of the root's memory chip: the root's host
has that memory chip's push engine copy it into the root's L3 behind a flag (__host_bingo_kernel_hbm_ucast, before the
root's cache stream on the same engine), and the root's cluster 0 fetches it into its symmetric buffer, the broadcast's
source (a symmetric-buffer address is one every chip's image can name, as the pulls need). Every
cluster's shard of the cache sits in its memory chip's HBM and streams through its weight ring.

THE HALVES (fa_llama_data.py). The 32 query heads run as two halves of 16 SIMD lanes, each over its four key / value
groups: a K tile is the four groups' keys side by side, the query block diagonal, so QK ([BC, 512] x [512, 16]) and PV
([512, BC] x [BC, 16]) are the shapes of fa_mc.py's; each half has its own softmax arenas. PV's [512, 16] FP16 output
holds every lane's O^T in its group's rows; one xDMA copy per cluster keeps those (8-byte atoms, four lanes of one
group) and leaves the partial [O_0 | O_1 | l_0 | l_1]: the halves' [128, 16] O^T, then their row-sum beats.

THE CACHE STREAMS. Every cluster's shard: its key tiles, half 0 then half 1 of each, then its value tiles the same way,
each one ring chunk of 32 KiB, pushed by the memory chip's engine under the chip's prefetcher, so the value tiles stream
in while the clusters exchange their maxima.

THE SPLIT, per half as fa_mc.py: the query broadcast into every cluster; pass 1, per K tile S^T = K Q^T (gemm_fa_qk) and
the FA softmax with its own seed, the shard's row maximum; each chip's two maxima met on its cluster 0 (the softmax
kernel over the two beats), the halves' maxima packed into one 64-B transfer; m* across the chips (reduce chain: one
ChainGather with the Elementwise junction's FP16 MAX; root: the root pulls every chip's pair, splits it into the halves'
rows, runs the max over each and packs m*), broadcast into every cluster and copied into its pass-2 arenas; pass 2, per
V tile the softmax SEEDED at m* and PV, the last tile leaving FP16; the compaction and the row sums; the partials
through the FFN's chip fold and reduce into the root, which normalises each half (o~ = O c / l) and transposes it: xt
[32, 128], one head a row.

THE ORDERS, as fa_mc.py: every DM core runs its tasks in order and a ring load blocks until its chunk lands, so the V
loads follow the cluster's share of its chip's maximum; the first `v_pre` tiles of them come before m*'s arrival, the
rest after it and the copies of m* into the arenas (on the root, the exchange's DM tasks also come after the first
ones). The first v_pre tiles of both halves need 2 v_pre ring slabs (w_buffers).

THE SOFTMAX STATISTICS, params stats (max | merge). max, as above: m* the lanewise maximum, the row sums reduced with O.
merge, with reduce chain: every cluster packs its (m, l) of both halves into one softmax-statistics group [m_0 | m_1 |
l_0 | l_1] (32 rows, FP16), each chip's two are folded by the junction on cluster 0's xDMA (cluster 1's the chain's
head), then the chips' by one ChainGather through every chip -- m* = max m_c, l* = sum exp(a (m_c - m*)) l_c, the
online softmax's merge, in the fabric -- into the root, which multicasts m* and normalises with l*; the partials are
O alone. With reduce root the root has no junction: stats merge falls back to max.

ONE PASS, params passes 1: each cluster streams its tiles once (K and V of both halves, tile by tile: the image's order)
and runs the online softmax and PV over them, every PV after the first scaling its INT32 accumulator by the softmax's
corr (C_COLSCALE): its own (m_c, l_c, O_c) in its own units. The statistics meet (stats), m* is multicast, and each
cluster re-bases O_c onto it with one more softmax step on its arena, a tile of FIX_ROWS scores that are all m*: the
step's corr is exp(a (m_c - m*)), and a PV with that column scale, zero values (OFF_ZV) and the D port leaves O_c corr
as FP16. Without the junction (stats max, or reduce root) each cluster's l_c is scaled by the same corr (a SIMD MUL)
and reduced with O.

PARAMETERS (params.hjson): keys (whole tiles, 4 a cluster at 8192), seed, wbits, n_calib, v_pre (two passes), passes,
stats, the ablation's switches bcast (bcast_l1 | pull), reduce (chain | root), sync (edge | barrier), barrier_mode, and
the platform's (w_chunk_bytes = one half's tile, 32 KiB; w_buffers; weight_ring; static_l1; l1_capacity; d2d_ddr).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# fa_mc puts util/sim/multichip and mini_compiler on sys.path
from fa_mc import (ARENA, BEAT, EW_F16_MAX, FA, LRUN, MRUN, PVK, REPO, XFER, FaBuild, _Part,  # noqa: E402
                   cm, run_driver)

from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelHbmUcastArgs, SnaxBingoKernelGemmFaQkArgs, SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelSimdMlaNormaliseArgs,
    SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelXdma6dArgs, SnaxBingoKernelXdmaChainGatherArgs,
    SnaxBingoKernelXdmaTranspose2dArgs, simd_fp16_lanes, xdma_softmax_stats_csrs)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs import at_offset  # noqa: E402

import fa_llama_data as fld  # noqa: E402

BC, BR, HD, D, NHF, GQ = fld.BC, fld.BR, fld.HD, fld.D, fld.HALVES, fld.GQ
KB, VB = BC * D, D * BC                 # a half's K tile, V tile (bytes)
SB = BEAT + BC * BR * 2                 # a score tile with its prefix beat
QH = BR * D                             # a half's block-diagonal query
QB = NHF * QH
FB = D * BR * 2                         # a half's PV output, FP16 [D, BR]
OH = HD * BR * 2                        # a half's compacted O^T, FP16 [HD, BR]
OB = NHF * OH
PB = OB + NHF * BEAT                    # a cluster's partial: [O_0 | O_1 | l_0 | l_1]
MB = NHF * BEAT                         # the halves' maxima, packed
ATOM = 8                                # the xDMA's lane: one 8-byte atom, four FP16 lanes
LANES = 8
GB = 2 * XFER                           # a softmax-statistics group: the m beat, then the l beat (both halves)
FIX_ROWS = 4                            # the re-base step's tile: m* in every row, one PV block of keys
ZV = D * FIX_ROWS                       # the re-base PV's values: zeros
P8F = FA.p8_bytes(FIX_ROWS, BR, FA.EXACT)
# the symmetric buffer's arenas: pass 1 (or the one pass), the chip's maximum, the root's maximum (reduce root), pass 2;
# one a half
AR1, AMX, ART, AR2 = range(4)


class LlamaFaBuild(FaBuild):
    TAG = "fa"

    def __init__(self, args, p, plat, hw):
        cm.ChipletBuild.__init__(self, args, p, plat, hw)
        if self.T != 1:
            raise ValueError("params tokens: the fa workloads run one decode token")
        if BR != simd_fp16_lanes():
            raise ValueError(f"fa_llama_data.BR={BR}: the softmax takes one query per FP16 lane of a SIMD beat, "
                             f"{simd_fp16_lanes()} on this cluster")
        if MB != XFER:
            raise ValueError(f"the halves' maxima are {MB} B, the m* exchange moves one {XFER}-B transfer")
        if self.chunk != KB:
            raise ValueError(f"params w_chunk_bytes={self.chunk}: one half's tile is {KB} B")
        for nb, what in ((QB, "the query"), (PB, "the partial"), (ARENA, "an FA arena")):
            if nb % XFER:
                raise ValueError(f"{what} is {nb} B, not whole {XFER}-B transfers")
        if (D // ATOM) * ATOM != D or HD % LANES:
            raise ValueError("the compaction moves whole atoms of a group's lanes")
        self.keys = int(p.get("keys", 8192))
        tiles = self.keys // BC
        if tiles % self.G:
            raise ValueError(f"{tiles} key tiles over {self.G} clusters")
        self.NT = NT = tiles // self.G
        self.passes = int(p.get("passes", 2))
        if self.passes not in (1, 2):
            raise ValueError(f"params passes={self.passes}: 1 or 2")
        stats = p.get("stats", "max")
        if stats not in ("max", "merge"):
            raise ValueError(f"params stats={stats}: max or merge")
        # the junction folds the statistics only on a chain; a root has none and takes the maximum
        self.merge = stats == "merge" and self.reduce_impl == "chain"
        self.v_pre = min(int(p.get("v_pre", 2)), NT)
        wbuf = int(p.get("w_buffers", 3))
        if self.passes == 2 and NHF * self.v_pre > wbuf:
            raise ValueError(f"params v_pre={self.v_pre}: its {NHF * self.v_pre} value chunks wait for m* in the "
                             f"ring slabs, w_buffers={wbuf}")
        cache = os.path.join(REPO, p.get("kv_cache_dir", "target/sim/automation/sweep/llm_blocks/fa/cache"))
        d = self.d = fld.LlamaFaData(cache, keys=self.keys, seed=int(p.get("seed", 1)),
                                     wbits=int(p.get("wbits", 4)), n_calib=int(p.get("n_calib", 64)))
        self.shards = {g: (g * NT, NT) for g in range(self.G)}
        chips = {k: self.gs(k) for k in self.chips}
        if self.passes == 2 and not self.merge:
            self.gold = d.golden(self.shards, chips, self.fold_order())
        else:
            self.gold = d.golden_v(self.shards, chips, self.fold_order(), passes=self.passes, merge=self.merge,
                                   fix_rows=FIX_ROWS)
        for line in d.report(self.gold["xt"]):
            print(f"[fa] {line}")
        print(f"[fa] {NT} tiles of {BC} keys per cluster, two halves each; fold order "
              f"{[f'{k:02x}' for k in self.order]}")
        self._stage()
        self.make_streams()
        self.make_rings()
        # ---- the symmetric buffer of every cluster, on whole transfer units (64 B) ----------------------------------
        self.OFF_X, self.OFF_FLAG, self.OFF_MFLAG = 0, QB, QB + XFER
        self.OFF_QIN = QB + 2 * XFER                 # the root's copy of the query from HBM: the broadcast's source
        # max: the chip's maxima, packed, the exchange's source; merge: the cluster's statistics group
        self.OFF_MLOC = self.OFF_QIN + QB
        if self.merge:
            self.OFF_GRPC = self.OFF_MLOC + GB       # the chip's group, its two clusters folded: the chain's source
            self.OFF_MS = self.OFF_GRPC + GB         # m*, as the broadcast leaves it in every cluster
            self.OFF_MR = self.OFF_MS + XFER         # the root's (m*, l*): the chain's destination, m* the broadcast's
            self.OFF_AR = self.OFF_MR + GB
        else:
            self.OFF_MS = self.OFF_MLOC + XFER       # m*, as the broadcast leaves it in every cluster
            self.OFF_MR = self.OFF_MS + XFER         # the root's m*: the chain's destination, the broadcast's source
            self.OFF_AR = self.OFF_MR + XFER         # the arenas (lowest addresses: below every score and P buffer)
        self.OFF_ZV = self.OFF_AR + 4 * NHF * ARENA  # one pass: the re-base PV's zero values
        self.OFF_LOC = self.OFF_ZV + (ZV if self.passes == 1 else 0)   # the chip's partial for the reduce
        self.PBE = OB if self.merge else PB          # a partial: O, and the row sums unless the junction folds them
        self.make_sym(self.OFF_LOC + self.PBE, self.PBE, dst_lead=BEAT)

    def ar(self, k, c, kind, h):
        """Chip k cluster c's arena `kind` (AR1, AMX, ART, AR2) of half h, in its symmetric buffer (zeroed at set-up:
        pass 2's lrun starts at 0)."""
        return self.sv(k, c, self.OFF_AR + (kind * NHF + h) * ARENA)

    # ---- data -------------------------------------------------------------------------------------------------------
    def _stage(self):
        st, d, T = self.st, self.d, self.TAG
        m0 = self.mem[self.ROOT]
        self.h_qhbm = st.put_hbm(f"{T}_qhbm", d.q8_image(), mem_chip=(m0 >> 4, m0 & 15))
        st.put_zeros(f"{T}_qland", "int8_t", QB, preload=True)
        st.put_zeros(f"{T}_qflag", "uint8_t", 64)
        st.put_zeros(f"{T}_qprog", "uint8_t", 64)
        self.img = {}
        for g, (t0, nt) in self.shards.items():
            k = self.chip(g)
            tl = d.tiles(t0, nt)                     # [half][tile] -> (k, v)
            if self.passes == 1:                     # tile by tile: its K of both halves, then its V
                img = np.concatenate([tl[h][i][kv] for i in range(nt) for kv in (0, 1)
                                      for h in range(NHF)]).astype(np.int8)
            else:
                img = np.concatenate([tl[h][i][0] for i in range(nt) for h in range(NHF)] +
                                     [tl[h][i][1] for i in range(nt) for h in range(NHF)]).astype(np.int8)
            self.img[g] = st.put_hbm(f"{T}_kv_g{g}", img, mem_chip=(self.mem[k] >> 4, self.mem[k] & 15))

    def _copy(self, k, c, name, src, dst, nbytes, deps):
        return self.node(k, c, self.roles["dm"], name, "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(src, dst, nbytes), deps)

    def _max(self, k, name, rows, n, p8, arena, deps):
        """The lanewise maximum of n beats at `rows` (a one-beat prefix below them) into `arena`'s mrun, on chip k's
        cluster 0: the FA softmax over an n-key tile, its own seed."""
        return self.node(k, 0, self.roles["simd"], name, "__snax_bingo_kernel_simd_fa_softmax",
                         FA(rows, p8, arena, n, 1, tile_idx=0, seed_state=1, geom_mode=FA.GEOM_SELF,
                            score_scale=1.0, p_mode=0, p8_pitch=0), deps)

    def _copy2d(self, k, c, name, src, dst, size, src_stride, dst_stride, reps, deps, **kw):
        return self.node(k, c, self.roles["dm"], name, "__snax_bingo_kernel_idma_2d_copy",
                         SnaxBingoKernelIdma2dCopyArgs(src, dst, size, src_stride, dst_stride, reps, **kw), deps)

    def _ssj_csrs(self):
        return xdma_softmax_stats_csrs(fld.SSJ_ROWS, self.d.a_exp)

    # ---- the statistics of every chip ---------------------------------------------------------------------------------
    def _chip_max(self, last):
        """Each chip's maxima on its cluster 0, packed [m_0 | m_1] into OFF_MLOC: per half the clusters' mrun (arena
        AR1, after last[g][h]) copied side by side and the softmax kernel's lanewise max over them. -> ({chip: the
        pack}, {g: the cluster's last DM task of it})"""
        mloc, mcp = {}, {}
        for k in self.chips:
            g0, g1 = self.gs(k)
            pk = []
            for h in range(NHF):
                mt = self._l1(g0, f"mt_h{h}", 3 * BEAT)          # the prefix beat, then the two clusters' rows
                p8m = self._l1(g0, f"p8m_h{h}", FA.p8_bytes(2, BR, 0))
                cp0 = self._copy(k, 0, f"m_cp0h{h}", at_offset(self.ar(k, 0, AR1, h), MRUN), at_offset(mt, BEAT),
                                 BEAT, [last[g0][h]])
                cp1 = self._copy(k, 1, f"m_cp1h{h}", at_offset(self.ar(k, 1, AR1, h), MRUN),
                                 at_offset(mt, 2 * BEAT), BEAT, [last[g1][h]])
                mcp[g0], mcp[g1] = cp0, cp1
                mx = self._max(k, f"m_maxh{h}", at_offset(mt, BEAT), 2, p8m, self.ar(k, 0, AMX, h), [cp0, cp1])
                pk.append(self._copy(k, 0, f"m_pack_h{h}", at_offset(self.ar(k, 0, AMX, h), MRUN),
                                     self.sv(k, 0, self.OFF_MLOC + h * BEAT), BEAT, [mx] + pk[-1:]))
            mloc[k] = pk[-1]
            mcp[g0] = pk[-1]
        return mloc, mcp

    def _chip_merge(self, last):
        """Every cluster's statistics group [m_0 | m_1 | l_0 | l_1] (arena AR1's mrun and lrun of both halves, after
        last[g]) into OFF_MLOC; each chip's two folded by the softmax-statistics junction on cluster 0's xDMA,
        cluster 1's the chain's head, into OFF_GRPC. -> ({chip: the fold}, {g: the cluster's last pack copy})"""
        csr0, csr1 = self._ssj_csrs()
        mloc, mcp = {}, {}
        for k in self.chips:
            pk = []
            for c, g in enumerate(self.gs(k)):
                a0 = self.ar(k, c, AR1, 0)
                pm = self._copy2d(k, c, "ml_pack_m", at_offset(a0, MRUN), self.sv(k, c, self.OFF_MLOC), BEAT,
                                  ARENA, BEAT, NHF, list(last[g]))
                mcp[g] = self._copy2d(k, c, "ml_pack_l", at_offset(a0, LRUN), self.sv(k, c, self.OFF_MLOC + XFER),
                                      BEAT, ARENA, BEAT, NHF, [pm])
                pk.append(mcp[g])
            mloc[k] = self.node(k, 0, self.roles["xdma"], "ml_chip", "__snax_bingo_kernel_xdma_chain_gather",
                                SnaxBingoKernelXdmaChainGatherArgs(
                                    local_src=self.sv(k, 0, self.OFF_MLOC),
                                    chain=[self.sv(k, 1, self.OFF_MLOC), self.sv(k, 0, self.OFF_GRPC)], size=GB,
                                    junction=str(self.jct["SoftmaxStats"]), jct_csr0=csr0, jct_csr1=csr1),
                                pk)
        return mloc, mcp

    def _exchange(self, before):
        """The chips' statistics met on the root into OFF_MR: m* (stats max: the Elementwise junction's FP16 MAX
        along the chain, or the root's pulls and max) or (m*, l*) (merge: the softmax-statistics junction along the
        chain). Sets self.m_made, self.msrc. -> the root DM core's tasks of it."""
        dm, xdma = self.roles["dm"], self.roles["xdma"]
        R, T = self.ROOT, self.TAG
        others = self.order[1:]
        duties = []
        self.msrc = self.sv(R, 0, self.OFF_MR)
        if self.merge:
            csr0, csr1 = self._ssj_csrs()
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, 0, others, src=self.sx(0, self.OFF_GRPC), dst=self.sx(0, self.OFF_MR), size=GB,
                junction=self.jct["SoftmaxStats"], jct_csr0=csr0, jct_csr1=csr1)
            made = self.node(R, 0, xdma, "ml_chain", "__snax_bingo_kernel_xdma_chain_gather", ga,
                             list(dict.fromkeys(before[k] for k in self.chips)))
        elif self.reduce_impl == "chain":
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, 0, others, src=self.sx(0, self.OFF_MLOC), dst=self.sx(0, self.OFF_MR), size=XFER,
                junction=self.jct["Elementwise"], jct_csr0=EW_F16_MAX)
            made = self.node(R, 0, xdma, "mg_chain", "__snax_bingo_kernel_xdma_chain_gather", ga,
                             list(dict.fromkeys(before[k] for k in self.chips)))
        else:
            n = len(self.chips)
            stg = self.cx_chip(R).at(0).l1(f"{T}_mstg", n * XFER)
            pulls = []
            for i, k in enumerate(self.order):
                src = self.sv(R, 0, self.OFF_MLOC) if k == R else cm.remote(k, self.sx(0, self.OFF_MLOC))
                pulls.append(self._copy(R, 0, f"mg_pull{i}", src, at_offset(stg, i * XFER), XFER,
                                        [before[k]] + pulls[-1:]))
            duties.append(pulls[0])
            pk = []
            for h in range(NHF):
                rows = self.cx_chip(R).at(0).l1(f"{T}_mrows_h{h}", BEAT + n * BEAT)
                p8r = self.cx_chip(R).at(0).l1(f"{T}_p8r_h{h}", FA.p8_bytes(n, BR, 0))
                sp = self.node(R, 0, dm, f"mg_split_h{h}", "__snax_bingo_kernel_idma_2d_copy",
                               SnaxBingoKernelIdma2dCopyArgs(at_offset(stg, h * BEAT), at_offset(rows, BEAT), BEAT,
                                                             XFER, BEAT, n),
                               [pulls[-1]] + pk[-1:])
                mx = self._max(R, f"mg_rowmax_h{h}", at_offset(rows, BEAT), n, p8r, self.ar(R, 0, ART, h), [sp])
                pk.append(self._copy(R, 0, f"mg_pack_h{h}", at_offset(self.ar(R, 0, ART, h), MRUN),
                                     self.sv(R, 0, self.OFF_MR + h * BEAT), BEAT, [mx] + pk[-1:]))
            made = pk[-1]
        self.m_made = made
        return duties

    # ---- the attention ----------------------------------------------------------------------------------------------
    def build(self):
        dm, xdma, simd, gemm = (self.roles[r] for r in ("dm", "xdma", "simd", "gemm"))
        d, R, T = self.d, self.ROOT, self.TAG
        go = self.setup()
        # ---- the query: HBM -> the root's L3 (its memory chip's engine, before the root's cache) -> its L1 ---------
        m = self.mem[R]
        self.q_push = self.cx_chip(R).host_node(
            f"q_hbm_ucast_k{R:02x}", "__host_bingo_kernel_hbm_ucast",
            HostBingoKernelHbmUcastArgs(self.h_qhbm.address, BingoMemSymbol(f"{T}_qland"),
                                        BingoMemSymbol(f"{T}_qflag"), BingoMemSymbol(f"{T}_qprog"),
                                        self.h_seq[m].address + 64, 1, QB, 1, m, self.engine[R], [R]),
            after=[go[R]])
        self.edge(self.q_push, self.prefetch[R])
        q_root = self.sv(R, 0, self.OFF_QIN)
        qin = self.node(R, 0, dm, "q_fetch", "__snax_bingo_kernel_idma_fetch_flagged",
                        SnaxBingoKernelIdmaFetchFlaggedArgs(BingoMemSymbol(f"{T}_qland"), q_root, QB,
                                                            BingoMemSymbol(f"{T}_qflag"), 1, 1), [go[R]])
        self.dfg.bingo_add_order_edge(self.q_push, qin)
        xin = self.broadcast("x", q_root, QB, self.OFF_X, self.OFF_FLAG,
                             {k: (qin if k == R else go[k]) for k in self.chips}, src_ready=qin)
        if self.passes == 1:
            self._one_pass(xin)
        else:
            self._two_pass(xin)
        # ---- the partials folded into the root, each half normalised and transposed --------------------------------
        self.reduce(self.part, self.PBE)
        self.ot = self.cx_chip(R).at(0).l1(f"{T}_ot", OB)
        self.xt = self.cx_chip(R).at(0).l1(f"{T}_xt", OB)
        nrm, tr = [], []
        for h in range(NHF):
            o_in = self.dst if h == 0 else self.dst.view(h * OH)
            latch = self.dst_buf if h == 0 else self.dst.view(h * OH - BEAT)
            # l: the junction's l* on the root, or the reduced row sums after O
            l_src = self.sv(R, 0, self.OFF_MR + XFER + h * BEAT) if self.merge else self.dst.view(OB + h * BEAT)
            # half 1's latch beat is half 0's last row, which half 0's normalise has read
            nrm.append(self.node(R, 0, simd, f"o_norm_h{h}", "__snax_bingo_kernel_simd_mla_normalise",
                                 SnaxBingoKernelSimdMlaNormaliseArgs(l_src, latch, o_in, self.ot.view(h * OH), HD,
                                                                     d.a_n_bits),
                                 list(self.reduce_end) + nrm[-1:]))
            tr.append(self.node(R, 0, xdma, f"o_transpose_h{h}", "__snax_bingo_kernel_xdma_transpose_2d",
                                SnaxBingoKernelXdmaTranspose2dArgs(self.ot.view(h * OH), self.xt.view(h * OH), HD,
                                                                   BR, 2),
                                [nrm[h]] + tr[-1:]))
        self.tr = tr[-1]
        self._finish_graph()

    def _compact(self, k, c, g, full, part, after):
        """Every lane's rows of its group: atom (four lanes) j of row 128 j + r, r over 8-row beats."""
        return self.node(k, c, self.roles["xdma"], f"o_compact_g{g}", "__snax_bingo_kernel_xdma_6d",
                         SnaxBingoKernelXdma6dArgs(full, part, BR * 2, BR * 2,
                                                   [LANES * BR * 2, HD * BR * 2 + ATOM, FB],
                                                   [HD // LANES, GQ, NHF],
                                                   [LANES * BR * 2, ATOM, OH], [HD // LANES, GQ, NHF]),
                         after)

    def _two_pass(self, xin):
        dm, xdma, simd, gemm = (self.roles[r] for r in ("dm", "xdma", "simd", "gemm"))
        d, NT, R = self.d, self.NT, self.ROOT
        # ---- pass 1: every cluster's row maximum per half over its K tiles ----------------------------------------
        self.s16, self.sm1 = {}, {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            self.ls[g].hold_until(xin[(k, c)])
            s16 = [self._l1(g, f"s16_h{h}", NT * SB) for h in range(NHF)]
            p8 = [self._l1(g, f"p8a_h{h}", FA.p8_bytes(BC, BR, FA.EXACT)) for h in range(NHF)]
            self.s16[g] = s16
            sm = [[] for _ in range(NHF)]
            for i in range(NT):
                for h in range(NHF):
                    ready, si, slab = self._load(g, f"K{i}h{h}_g{g}", (NHF * i + h) * KB, KB)
                    qk = self.node(k, c, gemm, f"qk{i}h{h}_g{g}", "__snax_bingo_kernel_gemm_fa_qk",
                                   SnaxBingoKernelGemmFaQkArgs(
                                       slab, self.sv(k, c, self.OFF_X + h * QH), 0, at_offset(s16[h], i * SB + BEAT),
                                       M=BC // 16, K=D // 4, N=BR // 16, d_shift=d.k_s),
                                   [ready, xin[(k, c)]])
                    self.ls[g].release(si, qk)
                    sm[h].append(self.node(k, c, simd, f"s1_{i}h{h}_g{g}", "__snax_bingo_kernel_simd_fa_softmax",
                                           FA(at_offset(s16[h], i * SB + BEAT), p8[h], self.ar(k, c, AR1, h), BC, 1,
                                              tile_idx=i, seed_state=1, geom_mode=FA.GEOM_SELF,
                                              score_scale=d.a_exp, p_mode=FA.EXACT, p8_pitch=0),
                                           [qk] + sm[h][-1:]))
            self.sm1[g] = [s[-1] for s in sm]
        # ---- each chip's statistics on its cluster 0: maxima packed [m_0 | m_1], or the clusters' groups folded ----
        self.mloc, mcp = (self._chip_merge if self.merge else self._chip_max)(self.sm1)
        before, bm = dict(self.mloc), {}
        if self.sync == "barrier":
            bm = self.barrier({k: [self.mloc[k]] for k in self.chips})
            before = {k: bm[R] for k in self.chips}
        # ---- the first v_pre value tiles, loaded while the maxima meet: made before the exchange, so with barriers
        # they belong between the chip's barrier after its maxima and the one before m*'s pulls -----------------------
        pre = self.v_pre
        vld = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            self.ls[g].wait_for(mcp[g])
            for i in range(pre):
                for h in range(NHF):
                    vld[g, i, h] = self._load(g, f"V{i}h{h}_g{g}", NT * NHF * KB + (NHF * i + h) * VB, VB,
                                              [bm[k]] if bm and c == 0 and i == 0 and h == 0 else [])
        # ---- m* (and l*): the chips' statistics met on the root, then m* into every cluster ------------------------
        duties = self._exchange(before)          # the root DM core's tasks of the exchange
        made = self.m_made
        self.bcast_puts = []
        marr = self.broadcast("m", self.msrc, XFER, self.OFF_MS, self.OFF_MFLAG,
                              {k: (made if k == R else self.mloc[k]) for k in self.chips}, src_ready=made)
        duties += self.bcast_puts if self.bcast_impl == "bcast_l1" else []
        mbar = dict(self.bcast_barrier)          # pull with barriers: the one before the pulls
        # ---- pass 2: per half the softmax seeded at m* and PV over the V tiles ------------------------------------
        self.part = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            if pre:
                # the exchange's arrival (and on the root its DM tasks) after the first value loads
                for t in [marr[(k, c)]] + (duties if (k, c) == (R, 0) else []) + ([mbar[k]] if mbar and c == 0 else []):
                    self.edge(vld[g, pre - 1, NHF - 1][0], t)
            p8 = [[self._l1(g, f"p8b{j}_h{h}", FA.p8_bytes(BC, BR, FA.EXACT)) for j in range(2)]
                  for h in range(NHF)]
            oacc = [self._l1(g, f"oacc_h{h}", D * BR * 4) for h in range(NHF)]
            full = self._l1(g, "ofull", NHF * FB)
            part = self._l1(g, "part", self.PBE)
            corr = FA.corr_offset(BC, BR)
            sm, pv, un = [[] for _ in range(NHF)], [[] for _ in range(NHF)], []

            def unpack():
                # m* into the halves' pass-2 arenas, behind its arrival: the DM core's next tasks after the first
                # value loads
                for h in range(NHF):
                    un.append(self._copy(k, c, f"m_unpack_h{h}_g{g}", self.sv(k, c, self.OFF_MS + h * BEAT),
                                         at_offset(self.ar(k, c, AR2, h), MRUN), BEAT, [marr[(k, c)]] + un[-1:]))
            for i in range(NT):
                if i == pre:
                    unpack()
                for h in range(NHF):
                    first, last = i == 0, i == NT - 1
                    if i < pre:
                        ready, si, slab = vld[g, i, h]
                    else:
                        ready, si, slab = self._load(g, f"V{i}h{h}_g{g}", NT * NHF * KB + (NHF * i + h) * VB, VB,
                                                        [un[-1]])
                    pb = p8[h][i % 2]
                    sm[h].append(self.node(k, c, simd, f"s2_{i}h{h}_g{g}", "__snax_bingo_kernel_simd_fa_softmax",
                                           FA(at_offset(self.s16[g][h], i * SB + BEAT), pb, self.ar(k, c, AR2, h),
                                              BC, 1, tile_idx=i, seed_state=0, geom_mode=FA.GEOM_SELF,
                                              score_scale=d.a_exp, p_mode=FA.EXACT, p8_pitch=0),
                                           [self.sm1[g][h]] + sm[h][-1:] + (pv[h][i - 2:i - 1] if i >= 2 else [])))
                    flags = PVK.B_KMAJOR | (0 if first else PVK.C_COLSCALE) | (PVK.D_FP16 if last else 0)
                    pv[h].append(self.node(k, c, gemm, f"pv{i}h{h}_g{g}", "__snax_bingo_kernel_gemm_fa_pv",
                                           PVK(slab, pb, 0 if first else oacc[h],
                                               at_offset(full, h * FB) if last else oacc[h], M=D // 16, K=BC // 4,
                                               N=BR // 16, flags=flags,
                                               corr_addr=0 if first else at_offset(pb, corr),
                                               d_shift=d.k_o if last else 0),
                                           [sm[h][i], ready] + pv[h][-1:]))
                    self.ls[g].release(si, pv[h][i])
            if pre == NT:
                unpack()
            for h in range(NHF):
                self.edge(un[h], sm[h][0])
            lcp = []
            if not self.merge:
                for h in range(NHF):
                    lcp.append(self._copy(k, c, f"l_cp_h{h}_g{g}", at_offset(self.ar(k, c, AR2, h), LRUN),
                                          at_offset(part, OB + h * BEAT), BEAT, [sm[h][-1]] + lcp[-1:]))
            cmp = self._compact(k, c, g, full, part, [pv[h][-1] for h in range(NHF)])
            self.part[g] = _Part(part, (cmp,) + tuple(lcp[-1:]))

    def _one_pass(self, xin):
        """Every cluster's online softmax and PV over its own tiles, its statistics met, O re-based onto m*."""
        dm, xdma, simd, gemm = (self.roles[r] for r in ("dm", "xdma", "simd", "gemm"))
        d, NT, R = self.d, self.NT, self.ROOT
        corr = FA.corr_offset(BC, BR)
        fcorr = FA.corr_offset(FIX_ROWS, BR)
        last, pvl, oacc, p8f = {}, {}, {}, {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            self.ls[g].hold_until(xin[(k, c)])
            s16 = [self._l1(g, f"s16_h{h}", NT * SB) for h in range(NHF)]
            p8 = [[self._l1(g, f"p8b{j}_h{h}", FA.p8_bytes(BC, BR, FA.EXACT)) for j in range(2)]
                  for h in range(NHF)]
            oacc[g] = [self._l1(g, f"oacc_h{h}", D * BR * 4) for h in range(NHF)]
            # the re-base step's P (then its corr), and one beat after it the cluster's l_c
            p8f[g] = [self._l1(g, f"p8f_h{h}", P8F + BEAT) for h in range(NHF)]
            sm, pv = [[] for _ in range(NHF)], [[] for _ in range(NHF)]
            for i in range(NT):
                qk = []
                for h in range(NHF):
                    ready, si, slab = self._load(g, f"K{i}h{h}_g{g}", (2 * NHF * i + h) * KB, KB)
                    qk.append(self.node(k, c, gemm, f"qk{i}h{h}_g{g}", "__snax_bingo_kernel_gemm_fa_qk",
                                        SnaxBingoKernelGemmFaQkArgs(
                                            slab, self.sv(k, c, self.OFF_X + h * QH), 0,
                                            at_offset(s16[h], i * SB + BEAT),
                                            M=BC // 16, K=D // 4, N=BR // 16, d_shift=d.k_s),
                                        [ready, xin[(k, c)]]))
                    self.ls[g].release(si, qk[h])
                for h in range(NHF):
                    first = i == 0
                    ready, si, slab = self._load(g, f"V{i}h{h}_g{g}", (2 * NHF * i + NHF + h) * KB, VB)
                    pb = p8[h][i % 2]
                    sm[h].append(self.node(k, c, simd, f"s1_{i}h{h}_g{g}", "__snax_bingo_kernel_simd_fa_softmax",
                                           FA(at_offset(s16[h], i * SB + BEAT), pb, self.ar(k, c, AR1, h), BC, 1,
                                              tile_idx=i, seed_state=1, geom_mode=FA.GEOM_SELF,
                                              score_scale=d.a_exp, p_mode=FA.EXACT, p8_pitch=0),
                                           [qk[h]] + sm[h][-1:] + (pv[h][i - 2:i - 1] if i >= 2 else [])))
                    pv[h].append(self.node(k, c, gemm, f"pv{i}h{h}_g{g}", "__snax_bingo_kernel_gemm_fa_pv",
                                           PVK(slab, pb, 0 if first else oacc[g][h], oacc[g][h], M=D // 16,
                                               K=BC // 4, N=BR // 16,
                                               flags=PVK.B_KMAJOR | (0 if first else PVK.C_COLSCALE),
                                               corr_addr=0 if first else at_offset(pb, corr), d_shift=0),
                                           [sm[h][i], ready] + pv[h][-1:]))
                    self.ls[g].release(si, pv[h][i])
            last[g] = [s[-1] for s in sm]
            pvl[g] = [p[-1] for p in pv]
        # ---- the statistics: the clusters' groups folded by the junction, or each chip's maxima ------------------
        self.mloc, _ = (self._chip_merge if self.merge else self._chip_max)(last)
        lkeep = {}
        if not self.merge:
            # l_c beside the re-base factor, before the re-base step moves the arena's lrun
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                for h in range(NHF):
                    lkeep[g, h] = self._copy(k, c, f"l_keep_h{h}_g{g}", at_offset(self.ar(k, c, AR1, h), LRUN),
                                             at_offset(p8f[g][h], P8F), BEAT, [last[g][h]])
        before = dict(self.mloc)
        if self.sync == "barrier":
            bm = self.barrier({k: [self.mloc[k]] for k in self.chips})
            before = {k: bm[R] for k in self.chips}
        self._exchange(before)
        made = self.m_made
        self.bcast_puts = []
        marr = self.broadcast("m", self.msrc, XFER, self.OFF_MS, self.OFF_MFLAG,
                              {k: (made if k == R else self.mloc[k]) for k in self.chips}, src_ready=made)
        # ---- each cluster's O re-based onto m*: a softmax step over FIX_ROWS rows of m*, its corr the PV's scale ----
        self.part = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            fx = self._l1(g, "fx", NHF * (1 + FIX_ROWS) * BEAT)     # per half: the latch beat, then the rows
            cpf = self._copy2d(k, c, f"m_fix_g{g}", self.sv(k, c, self.OFF_MS), at_offset(fx, BEAT), BEAT, 0, BEAT,
                               FIX_ROWS, [marr[(k, c)]], outer=NHF, src_outer=BEAT,
                               dst_outer=(1 + FIX_ROWS) * BEAT)
            full = self._l1(g, "ofull", NHF * FB)
            part = self._l1(g, "part", self.PBE)
            fs, fp_, lm = [], [], []
            for h in range(NHF):
                fs.append(self.node(k, c, simd, f"s_fix_h{h}_g{g}", "__snax_bingo_kernel_simd_fa_softmax",
                                    FA(at_offset(fx, h * (1 + FIX_ROWS) * BEAT + BEAT), p8f[g][h],
                                       self.ar(k, c, AR1, h), FIX_ROWS, 1, tile_idx=NT, seed_state=1,
                                       geom_mode=FA.GEOM_SELF, score_scale=d.a_exp, p_mode=FA.EXACT, p8_pitch=0),
                                    [cpf, last[g][h]] + ([lkeep[g, h]] if lkeep else []) + fs[-1:]))
                fp_.append(self.node(k, c, gemm, f"pv_fix_h{h}_g{g}", "__snax_bingo_kernel_gemm_fa_pv",
                                     PVK(self.sv(k, c, self.OFF_ZV), p8f[g][h], oacc[g][h], at_offset(full, h * FB),
                                         M=D // 16, K=FIX_ROWS // 4, N=BR // 16,
                                         flags=PVK.B_KMAJOR | PVK.C_COLSCALE | PVK.D_FP16,
                                         corr_addr=at_offset(p8f[g][h], fcorr), d_shift=d.k_o),
                                     [fs[h], pvl[g][h]] + fp_[-1:]))
                if not self.merge:
                    lm.append(self.node(k, c, simd, f"l_fix_h{h}_g{g}", "__snax_bingo_kernel_simd_stream_elementwise",
                                        SnaxBingoKernelSimdMulF16Args(at_offset(p8f[g][h], fcorr),
                                                                      at_offset(p8f[g][h], P8F),
                                                                      at_offset(part, OB + h * BEAT), 1, BR),
                                        [fs[h], lkeep[g, h]] + lm[-1:]))
            cmp = self._compact(k, c, g, full, part, fp_)
            self.part[g] = _Part(part, (cmp,) + tuple(lm[-1:]))

    # ---- the end ----------------------------------------------------------------------------------------------------
    def _finish_graph(self):
        R, gd = self.ROOT, self.gold
        self.end_window([self.tr])
        # the statistics on the root: the folded group (m*, l*) or m*
        sn, sb = ("ml", GB) if self.merge else ("mstar", MB)
        checks = [("xt", self.stash("xt", R, 0, self.xt, OB, []), np.asarray(gd["xt"], np.float16), OB),
                  (sn, self.stash(sn, R, 0, self.msrc, sb, [self.m_made]), np.asarray(gd[sn], np.float16), sb)]
        far, pb = self.order[len(self.order) // 2], self.PBE
        checks.append((f"loc_k{far:02x}", self.stash(f"loc_k{far:02x}", far, 0, self.sv(far, 0, self.OFF_LOC), pb, []),
                       np.asarray(gd["loc"][far], np.float16), pb))
        self.run_checks(checks)


def _app_name(S):
    return (f"Llama-3-8B GQA attention, one token over {S.keys} keys, {len(S.chips)} chips x {S.NC} clusters, "
            + ("" if S.passes == 2 and not S.merge else f"passes={S.passes} stats={'merge' if S.merge else 'max'} ")
            + f"bcast={S.bcast_impl} reduce={S.reduce_impl} sync={S.sync}")


def run():
    run_driver("fa llama3 decode workload generator", LlamaFaBuild, _app_name)
