# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the fa workloads (workloads/fa/<model>/<variant>/): DeepSeek-V2-Lite's
layer-1 MLA attention for one decode token over a REAL 8K-token latent cache, split by key over
every cluster of the 16 + 4 chiplet platform (the platform, the rings, the anchors, the sync arm,
the broadcast and the reduce are ffn/common/chiplet_mc.py's; the data and the golden are
fa_data.py's).

THE CACHE STREAMS. The cache lives in HBM: every cluster's shard of key tiles in its memory chip,
tile-major -- its K tiles (the key copy's rows, 64 keys x 576 B) then its V tiles (the value
copy's 512 x 64 B columns, as the attention reads them) -- and goes through the cluster's weight
ring like a weight (plain INT8; weight_crest, kept but not in the ablation, would push CREST
records expanded in place by the xDMA). Each tile
is one chunk: K0..K(n-1) for pass 1, then V0..V(n-1) for pass 2, so the V tiles stream in while
the clusters exchange their maxima.

THE SPLIT (dsv2_datagen.split_attention_golden, mode key). The query (Q8, BR lanes x 576 B) is
broadcast into every cluster. Pass 1: per K tile S^T = K Q^T (gemm_fa_qk) and the FA softmax
with its own seed -- the shard's row maximum. The maxima meet: each chip's two on its cluster 0
(the softmax kernel run over the two beats as a lanewise max), then across the chips -- m*, the
lanewise maximum. m* is broadcast into every cluster's pass-2 softmax arena, which sits in the
symmetric buffer. Pass 2: per V tile the softmax SEEDED at m* (no correction across clusters)
and PV (gemm_fa_pv); the last tile leaves O^T through the D port as FP16. Each cluster's partial
(O^T, then its row-sum beat) goes through the FFN's chip fold and reduce into the root, which
normalises (o~ = O c / l) and transposes.

THE m* EXCHANGE follows the reduce switch: chain, one ChainGather through every chip with the
Elementwise junction's FP16 MAX; root, the root pulls every chip's beat into one tile and runs
the max over it. The broadcasts (the query, m*) follow bcast, the synchronisation sync.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))     # workloads/common
from driver import repo_root, run_driver  # noqa: E402
REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, os.path.join(HERE, "..", "..", "ffn", "common"))
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    simd_fp16_lanes, SnaxBingoKernelGemmFaPvArgs, SnaxBingoKernelGemmFaQkArgs, SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelSimdFaSoftmaxArgs,
    SnaxBingoKernelSimdMlaNormaliseArgs, SnaxBingoKernelXdmaChainGatherArgs,
    SnaxBingoKernelXdmaTranspose2dArgs,
    XDMA_EWJCT_FMT_FP16, XDMA_EWJCT_OP_MAX, xdma_ewjct_csr0)
from libs import at_offset  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import fa_data  # noqa: E402

BC, BR, D_QK, D_V = fa_data.BC, fa_data.BR, fa_data.D_QK, fa_data.D_V
FA = SnaxBingoKernelSimdFaSoftmaxArgs
# BEAT: one SIMD beat, BR FP16 values -- one key of a score tile, and the m and l beats. XFER: the
# smallest unit the xDMA and the D2D paths move it in (their 64-B beat); a wider transfer of an
# m or l beat also carries the arena beat after it, which is scratch (corrL after mrun) or the
# same l again (lnew after lrun, equal after the commit).
BEAT = FA.beat()
XFER = max(64, BEAT)
KB, VB = BC * D_QK, D_V * BC              # a K tile, a V tile (bytes)
SB = BEAT + BC * BR * 2                   # a score tile with its prefix beat
QB = BR * D_QK                            # Q8
OB = D_V * BR * 2                         # O^T, FP16
PB = OB + XFER                            # a cluster's partial: O^T, then its row sums
EW_F16_MAX = xdma_ewjct_csr0(XDMA_EWJCT_OP_MAX, XDMA_EWJCT_FMT_FP16)
PVK = SnaxBingoKernelGemmFaPvArgs
LAY = FA.layout(BC, 1)
ARENA = FA.arena_bytes(BC, 1)
if ARENA % XFER:
    raise ValueError(f"the FA softmax arena is {ARENA} B, not whole {XFER}-B units")
if FA.layout(2, 1) != LAY or FA.arena_bytes(2, 1) != ARENA:
    raise ValueError("the FA softmax arena's layout depends on its tile height")
MRUN, LRUN = LAY["mrun"], LAY["lrun"]


class _Part:
    """A cluster's partial as ChipletBuild.reduce() reads it: a handle and its producers."""

    def __init__(self, handle, ends):
        self.handle, self.ends = handle, tuple(ends)


class FaBuild(cm.ChipletBuild):
    TAG = "fa"

    def __init__(self, args, p, plat, hw):
        super().__init__(args, p, plat, hw)
        if self.T != 1:
            raise ValueError("params tokens: the fa workloads run one decode token")
        if BR != simd_fp16_lanes():
            raise ValueError(f"fa_data.BR={BR}: the softmax takes one query per FP16 lane of a "
                             f"SIMD beat, {simd_fp16_lanes()} on this cluster")
        if self.chunk < KB:
            raise ValueError(f"params w_chunk_bytes={self.chunk}: a K tile is {KB} B")
        self.keys = int(p.get("keys", 8192))
        tiles = self.keys // BC
        if tiles % self.G:
            raise ValueError(f"{tiles} key tiles over {self.G} clusters")
        self.NT = NT = tiles // self.G
        kv = os.path.join(REPO, p.get("kv_cache", "target/sim/automation/sweep/llm_blocks/fa/cache/"
                                                  "dsv2_l1_kv_8193.npz"))
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        d = self.d = fa_data.FaData(kv, cache, args.dsv2_dir, keys=self.keys)
        # every cluster a contiguous shard of NT tiles, in cluster order
        self.shards = {g: (g * NT, NT) for g in range(self.G)}
        self.gold = d.golden(self.shards, {k: self.gs(k) for k in self.chips},
                             self.fold_order())
        for line in d.report(self.gold["xt"]):
            print(f"[fa] {line}")
        print(f"[fa] {NT} tiles of {BC} keys per cluster; fold order "
              f"{[f'{k:02x}' for k in self.order]}")
        self._stage()
        self.make_streams()
        self.make_rings(self.cimg)
        # ---- the symmetric buffer of every cluster -----------------------------------------
        # the flags and the arenas on whole transfer units (64 B)
        self.OFF_X, self.OFF_FLAG = 0, QB
        self.OFF_MFLAG = QB + XFER
        self.OFF_MX = QB + 2 * XFER                  # the chip's max: an FA softmax arena
        self.OFF_MLOC = self.OFF_MX + MRUN           # ... whose mrun is the chip's maximum
        self.OFF_A2 = self.OFF_MX + ARENA            # pass 2's arena: m* lands in its mrun
        self.OFF_M = self.OFF_A2 + MRUN
        # the root's m*: the mrun of one more arena (the root's max over the gathered beats, or
        # the chain's destination), in the symmetric buffer so every chip can name it
        self.OFF_MR = self.OFF_A2 + ARENA
        self.OFF_LOC = self.OFF_MR + ARENA           # the chip's (O^T, l) for the reduce
        # the root normalises dst in place: its latch beat sits below
        self.make_sym(self.OFF_LOC + PB, PB, dst_lead=BEAT)

    # ---- data -----------------------------------------------------------------------------
    def _stage(self):
        st, d, T = self.st, self.d, self.TAG
        self.h_q8 = st.put(f"{T}_q8", "int8_t", d.q8_a().reshape(-1))
        self.img, self.cimg = {}, {}
        for g, (t0, nt) in self.shards.items():
            k = self.chip(g)
            tl = d.tiles(t0, nt)
            img = np.concatenate([t[0] for t in tl] + [t[1] for t in tl]).astype(np.int8)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            hd = st.put_hbm(f"{T}_kv_g{g}", img, mem_chip=loc)
            self.img[g] = hd
            if self.crest:
                cap = img.nbytes + 64 * 1024
                self.cimg[g] = (hd.address, img.view(np.uint8).reshape(-1),
                                st.put_hbm(f"{T}_ckv_g{g}", np.zeros(cap, np.uint8), mem_chip=loc),
                                cap)

    # ---- the ring loads -------------------------------------------------------------------
    def _load(self, g, name, off, nbytes, deps=()):
        """One chunk of cluster g's cache image through its ring into the next slab of its
        stream: (the task after which the slab holds it, slab index, slab)."""
        k, c = self.chip(g), self.c(g)
        ls = self.ls[g]
        si, slab, war = ls.take(nbytes)
        ld, ex, _ = ls.rings.load_chunk(
            lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
            f"Ld_{name}", src=self.img[g].address + off, nbytes=nbytes, slab=slab, war=war,
            deps=deps, dm=self.roles["dm"], xdma=self.roles["xdma"], slab_bytes=ls.slab_bytes,
            expand_name=f"Ex_{name}")
        return ld, ld if ex is None else ex, si, slab

    def _l1(self, g, name, nbytes):
        return self.cx_chip(self.chip(g)).at(self.c(g)).l1(f"{self.TAG}_{name}", nbytes)

    # ---- the attention --------------------------------------------------------------------
    def build(self):
        dm, xdma, simd, gemm = (self.roles[r] for r in ("dm", "xdma", "simd", "gemm"))
        d, NT, R = self.d, self.NT, self.ROOT
        go = self.setup()
        xin = self.broadcast("x", self.h_q8, QB, self.OFF_X, self.OFF_FLAG, go)
        # ---- pass 1: every cluster's row maximum over its K tiles --------------------------
        self.s16, self.sm1 = {}, {}
        ar1 = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            self.ls[g].hold_until(xin[(k, c)])
            q8 = self.sv(k, c, self.OFF_X)
            ar1[g] = self._l1(g, "ar1", ARENA)
            s16 = self.s16[g] = self._l1(g, "s16", NT * SB)
            p8 = self._l1(g, "p8a", FA.p8_bytes(BC, BR, FA.EXACT))
            sm = []
            for i in range(NT):
                _, ready, si, slab = self._load(g, f"K{i}_g{g}", i * KB, KB)
                qk = self.node(k, c, gemm, f"qk{i}_g{g}", "__snax_bingo_kernel_gemm_fa_qk",
                               SnaxBingoKernelGemmFaQkArgs(
                                   slab, q8, 0, at_offset(s16, i * SB + BEAT),
                                   M=BC // 16, K=D_QK // 4, N=BR // 16, d_shift=d.k_s),
                               [ready, xin[(k, c)]])
                self.ls[g].release(si, qk)
                sm.append(self.node(k, c, simd, f"s1_{i}_g{g}", "__snax_bingo_kernel_simd_fa_softmax",
                                    FA(at_offset(s16, i * SB + BEAT), p8, ar1[g], BC, 1,
                                       tile_idx=i, seed_state=1, geom_mode=FA.GEOM_SELF,
                                       score_scale=d.a_exp, p_mode=FA.EXACT, p8_pitch=0),
                                    [qk] + sm[-1:]))
            self.sm1[g] = sm[-1]
        # ---- each chip's maximum, on its cluster 0 -----------------------------------------
        self.mloc, mcp = {}, {}
        for k in self.chips:
            g0, g1 = self.gs(k)
            mt = self._l1(g0, "mt", 3 * BEAT)             # the prefix beat, then the two rows
            p8m = self._l1(g0, "p8m", FA.p8_bytes(2, BR, 0))
            cp0 = self.node(k, 0, dm, "m_cp0", "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(at_offset(ar1[g0], MRUN),
                                                          at_offset(mt, BEAT), BEAT),
                            [self.sm1[g0]])
            cp1 = self.node(k, 1, dm, "m_cp1", "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(at_offset(ar1[g1], MRUN),
                                                          at_offset(mt, 2 * BEAT), BEAT),
                            [self.sm1[g1]])
            mcp[g0], mcp[g1] = cp0, cp1
            self.mloc[k] = self.node(k, 0, simd, "m_max", "__snax_bingo_kernel_simd_fa_softmax",
                                     FA(at_offset(mt, BEAT), p8m, self.sv(k, 0, self.OFF_MX), 2,
                                        1, tile_idx=0, seed_state=1, geom_mode=FA.GEOM_SELF,
                                        score_scale=1.0, p_mode=0, p8_pitch=0),
                                     [cp0, cp1])
        # ---- m*: the chips' maxima met on the root, then into every pass-2 arena -----------
        before, bm = dict(self.mloc), {}
        if self.sync == "barrier":
            bm = self.barrier({k: [self.mloc[k]] for k in self.chips})
            before = {k: bm[R] for k in self.chips}
        others = self.order[1:]
        duties = []                     # the root DM core's tasks of the exchange
        msrc = self.sv(R, 0, self.OFF_MR + MRUN)
        if self.reduce_impl == "chain":
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, 0, others, src=self.sx(0, self.OFF_MLOC),
                dst=self.sx(0, self.OFF_MR + MRUN), size=XFER,
                junction=self.jct["Elementwise"], jct_csr0=EW_F16_MAX)
            made = self.node(R, 0, xdma, "mg_chain", "__snax_bingo_kernel_xdma_chain_gather", ga,
                             list(dict.fromkeys(before[k] for k in self.chips)))
        else:
            rows = self.cx_chip(R).at(0).l1(f"{self.TAG}_mrows", BEAT + len(self.chips) * BEAT)
            p8r = self.cx_chip(R).at(0).l1(f"{self.TAG}_p8r", FA.p8_bytes(len(self.chips), BR, 0))
            pulls = []
            for i, k in enumerate(self.order):
                src = self.sv(R, 0, self.OFF_MLOC) if k == R else \
                    cm.remote(k, self.sx(0, self.OFF_MLOC))
                pulls.append(self.node(R, 0, dm, f"mg_pull{i}", "__snax_bingo_kernel_idma_1d_copy",
                                       SnaxBingoKernelIdma1dCopyArgs(
                                           src, at_offset(rows, BEAT + i * BEAT), BEAT),
                                       [before[k]] + pulls[-1:]))
            duties.append(pulls[0])
            made = self.node(R, 0, simd, "mg_rowmax", "__snax_bingo_kernel_simd_fa_softmax",
                             FA(at_offset(rows, BEAT), p8r, self.sv(R, 0, self.OFF_MR),
                                len(self.chips), 1, tile_idx=0, seed_state=1,
                                geom_mode=FA.GEOM_SELF, score_scale=1.0, p_mode=0, p8_pitch=0),
                             [pulls[-1]])
        self.m_made, self.msrc = made, msrc
        self.bcast_puts = []
        marr = self.broadcast("m", msrc, XFER, self.OFF_M, self.OFF_MFLAG,
                              {k: (made if k == R else self.mloc[k]) for k in self.chips},
                              src_ready=made)
        duties += self.bcast_puts if self.bcast_impl == "bcast_l1" else []
        mbar = dict(self.bcast_barrier)          # pull with barriers: the one before the pulls
        # ---- pass 2: the softmax seeded at m*, and PV over the V tiles ---------------------
        # Every DM core runs its tasks in order and a ring load blocks until its chunk lands.
        # The V loads follow the cluster's copy of its maximum (else the exchange starts only
        # once they land); the first PRE of them come before m*'s arrival, the rest after it,
        # and on the root before its tasks of the exchange: V0 and V1 land while m* is made.
        # With barriers, the chip's barrier after its maximum also comes before them (it holds
        # the DM core too, and the chip arrives there once its maximum is ready), and the one
        # before m*'s pulls after V0 and V1, as m*'s arrival.
        pre = min(2, NT)
        self.part = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            self.ls[g].wait_for(mcp[g])
            ar2 = self.sv(k, c, self.OFF_A2)
            p8 = [self._l1(g, f"p8b{j}", FA.p8_bytes(BC, BR, FA.EXACT)) for j in range(2)]
            oacc = self._l1(g, "oacc", D_V * BR * 4)
            part = self._l1(g, "part", PB)
            corr = FA.corr_offset(BC, BR)
            sm, pv = [], []
            for i in range(NT):
                first, last = i == 0, i == NT - 1
                deps = [marr[(k, c)]] if i >= pre else []
                deps += [bm[k]] if bm and c == 0 and i == 0 else []
                ld, ready, si, slab = self._load(g, f"V{i}_g{g}", NT * KB + i * VB, VB, deps)
                if i == pre - 1:
                    for t in [marr[(k, c)]] + (duties if (k, c) == (R, 0) else []) + \
                            ([mbar[k]] if mbar and c == 0 else []):
                        self.edge(ld, t)
                sm.append(self.node(k, c, simd, f"s2_{i}_g{g}", "__snax_bingo_kernel_simd_fa_softmax",
                                    FA(at_offset(self.s16[g], i * SB + BEAT), p8[i % 2], ar2, BC,
                                       1, tile_idx=i, seed_state=0, geom_mode=FA.GEOM_SELF,
                                       score_scale=d.a_exp, p_mode=FA.EXACT, p8_pitch=0),
                                    [marr[(k, c)], self.sm1[g]] + sm[-1:] +
                                    (pv[i - 2:i - 1] if i >= 2 else [])))
                flags = PVK.B_KMAJOR | (0 if first else PVK.C_COLSCALE) | \
                    (PVK.D_FP16 if last else 0)
                pv.append(self.node(k, c, gemm, f"pv{i}_g{g}", "__snax_bingo_kernel_gemm_fa_pv",
                                    PVK(slab, p8[i % 2], 0 if first else oacc,
                                        part if last else oacc, M=D_V // 16, K=BC // 4,
                                        N=BR // 16, flags=flags,
                                        corr_addr=0 if first else at_offset(p8[i % 2], corr),
                                        d_shift=d.k_o if last else 0),
                                    [sm[i], ready] + pv[-1:]))
                self.ls[g].release(si, pv[i])
            lcp = self.node(k, c, dm, f"l_cp_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(self.sv(k, c, self.OFF_A2 + LRUN),
                                                          at_offset(part, OB), XFER), [sm[-1]])
            self.part[g] = _Part(part, (pv[-1], lcp))
        # ---- the partials folded into the root, normalised and transposed ------------------
        self.reduce(self.part, PB)
        self.ot = self.cx_chip(R).at(0).l1(f"{self.TAG}_ot", OB)
        self.xt = self.cx_chip(R).at(0).l1(f"{self.TAG}_xt", OB)
        nrm = self.node(R, 0, simd, "o_norm", "__snax_bingo_kernel_simd_mla_normalise",
                        SnaxBingoKernelSimdMlaNormaliseArgs(self.dst.view(OB), self.dst_buf,
                                                            self.dst, self.ot, D_V, d.a_n_bits),
                        list(self.reduce_end))
        self.tr = self.node(R, 0, xdma, "o_transpose", "__snax_bingo_kernel_xdma_transpose_2d",
                            SnaxBingoKernelXdmaTranspose2dArgs(self.ot, self.xt, D_V, BR, 2), [nrm])
        self._finish_graph()

    # ---- the end --------------------------------------------------------------------------
    def _finish_graph(self):
        R, gd = self.ROOT, self.gold
        self.end_window([self.tr])
        checks = [("xt", self.stash("xt", R, 0, self.xt, OB, []),
                   np.asarray(gd["xt"], np.float16), OB),
                  ("mstar", self.stash("mstar", R, 0, self.msrc, BEAT, [self.m_made]),
                   np.asarray(gd["mstar"], np.float16), BEAT)]
        far = self.order[len(self.order) // 2]
        o, l = gd["loc"][far]
        checks.append((f"loc_k{far:02x}", self.stash(f"loc_k{far:02x}", far, 0,
                                                     self.sv(far, 0, self.OFF_LOC), PB, []),
                       self.d.part_bytes(o, l, XFER // BEAT), PB))
        self.run_checks(checks)


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer 1 MLA attention, one token over {S.keys} keys, "
            f"{len(S.chips)} chips x {S.NC} clusters, crest={int(S.crest)} "
            f"bcast={S.bcast_impl} reduce={S.reduce_impl} sync={S.sync}")


def run():
    run_driver("fa workload generator", FaBuild, _app_name)
