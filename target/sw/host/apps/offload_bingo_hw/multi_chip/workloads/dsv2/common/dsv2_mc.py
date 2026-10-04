# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the MULTI-CHIPLET staged DeepSeek-V2-Lite workloads
(workloads/dsv2/six_chiplet/stage*/), the counterpart of dsv2_staged.py (one chiplet).

THE PLATFORM (hemaia_sixchiplet_16MBL3_2cluster): compute chiplets with two clusters each and
memory chiplets between them, one C-M-C row per memory chiplet:

    C00 - M10 - C20        each memory chiplet feeds its west and its east neighbour, each
     |     |     |         with a push engine of its own (sys_dma.h engines), from its own
    C01 - M11 - C21        HBM; compute chips of one column share a direct link

THE MAPPING: TENSOR PARALLEL (docs/dsv2_multichiplet_plan.md, R4). The clusters of every
chip are one pool, global cluster g = (chip index) * clusters + cluster, chips in cfg order.
Every weight is split by OUTPUT columns -- each output column one whole dot product on one
cluster, so every value is the golden's to the bit -- and every link carries its share of
every weight, whatever the router picks:

    W_DKV        on the latent's cluster (LAT): the latent's norm, k_pe's RoPE and the cache
                 append are there
    W_Q          by head, heads / clusters a cluster (2 on 8 clusters)
    W_UK, W_UV   the same heads
    W_O          by output columns (256 a cluster), from the all-gathered o
    router       on EVERY chip: each chip routes for itself from the all-gathered h, with its
                 own expert table -- whose addresses are its own memory chiplet's slices -- so
                 no record crosses a chip
    shared       by intermediate columns (352 a cluster) for gate|up, all-gather of the SwiGLU
                 output, by output columns (256) for down
    routed       the same, 192 / 160 intermediate columns a cluster (an INT4 GEMV takes whole
                 32-column pairs), each chip's slices of each expert back to back in its memory
                 chiplet's HBM
    combine      each cluster its own 256 output columns: h + shared + the slots in top-k order

The attention (the latent cache and MlaAttention) stays on one cluster (ATT): its keys are
not split, so its arithmetic is the golden's own.

EVERY WEIGHT A CLUSTER STREAMS is ONE image in its memory chiplet's HBM, in stream order, so
the prefetcher (host_kernel_lib.h weight_prefetch, one per chip, on its own push engine)
merges chunk after chunk into long pushes. The memory chiplets hold different data: each
its own two chips' slices.

CROSSING A CHIP. A D2D write is acknowledged when it leaves the sender, so a hand-off is a
LOCAL write and a REMOTE read: the producer's DM core stashes its buffer into a staged array
in its own chip's L3, and the consumer's DM core reads that chip's copy (Collect). Every chip
runs one image, so a staged array sits at the same local address on each.

CHECKS are stashed into staged arrays on the producer's chip, and chip 0's host compares them
at the end, upstream first, byte-exact, reading the other chips' copies over the links.
"""

import argparse
import os
import sys
from dataclasses import replace

import hjson
import numpy as np

from dsv2_staged import ROOT_DIR, load_stage  # noqa: F401  (sets up the import paths)

from bingo_data_staging import DataStaging                             # noqa: E402
from bingo_dfg import BingoDFG                                         # noqa: E402
from bingo_kernel_args import (HostBingoKernelWeightPrefetchArgs,      # noqa: E402
                               SnaxBingoKernelGemmFaQkArgs, SnaxBingoKernelSimdAddF16Args,
                               SnaxBingoKernelSimdScaleF16Args,
                               SnaxBingoKernelSimdSwigluARowArgs, SnaxBingoKernelXdmaMemsetArgs)
from bingo_mem_handle import BingoMemSymbol                            # noqa: E402
from bingo_platform import (core_roles, guard_chiplet_count,           # noqa: E402
                            guard_cluster_count, load_cluster_cfg,
                            parse_platform_cfg)
from libs import (Ctx, DType, Layout, MemLevel, Pipeline, Port,        # noqa: E402
                  PortSpec, at_offset)
from libs.blocks import (After, Collect, Join, Linear, LoadStream, Pull, ScaleCols,  # noqa: E402
                        Stash, View, WeightRings, expert_table, record_bytes)
from libs.verify import checks as vchecks                              # noqa: E402
from libs.crest import CrestRings                                      # noqa: E402
import dsv2_datagen as dg                                              # noqa: E402

L1_CAPACITY = 514816
MESH = (16, 4, 16)
GEMV_SHAPE = (1, 4, 32)
D, KV, KVR, QH, RP = dg.D_MODEL, dg.KV, dg.KV_RANK, dg.Q_HEAD, dg.ROPE
HEADS, I_EXP, I_SH, N_EXP, TOP_K = dg.HEADS, dg.I_EXP, dg.I_SH, dg.N_EXP, dg.TOP_K
L1, L3, HBM = MemLevel.L1, MemLevel.L3, MemLevel.HBM
F16, I8 = DType.F16, DType.I8
RM, A, B, A_ROW = Layout.ROW_MAJOR, Layout.A, Layout.B, Layout.A_ROW
STAGES = {1: "norm", 2: "W_Q", 3: "W_DKV", 4: "latent + absorb", 5: "attention",
          6: "MLA output", 7: "routing", 8: "the layer"}


def f16(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float16))


def a_row_bytes(v):
    """int8 values as a_row: value c at (c // 4) * 8 + c % 4, the rest 0."""
    v = np.asarray(v, dtype=np.int8).reshape(-1)
    o = np.zeros(2 * v.size, dtype=np.int8)
    o.reshape(-1, 8)[:, :4] = v.reshape(-1, 4)
    return o


def split32(n, parts):
    """n columns over `parts` clusters in whole 32-column pairs (an INT4 GEMV's unit), the
    larger shares first -- alternating within a chip is the caller's choice."""
    if n % 32:
        raise ValueError(f"{n} columns are not whole pairs of 16-column blocks")
    q, r = divmod(n // 32, parts)
    return [32 * (q + (1 if i < r else 0)) for i in range(parts)]


class MBuild:
    """What one stage hands the next, for several chips: the params, the topology, the staged
    data, the graph under construction, every block a later stage reads, and the helpers."""

    def __init__(self, args, p, plat, hw, stop):
        self.args, self.p, self.plat, self.hw = args, p, plat, hw
        self.STOP = stop
        self.T = int(p.get("tokens", 1))
        if self.T != 1:
            raise ValueError("params tokens: the multi-chiplet stages run one token for now")
        self.wbits = int(p.get("wbits", 4))
        if self.wbits not in (4, 8):
            raise ValueError(f"params wbits={self.wbits}: 8 or 4")
        self.chunk = int(p.get("w_chunk_bytes", 128 * 1024))
        # params weight_crest: every weight chunk pushed CREST-compressed and expanded by the
        # cluster's xDMA (libs/crest.py); a slot then also holds the chunk's header, and a
        # chunk that does not compress is stored plain behind it
        self.crest = bool(p.get("weight_crest", False))
        self.slot_bytes = self.chunk + (128 if self.crest else 0)

        # ---- topology -------------------------------------------------------------------
        chips = self.chips = [int(k) for k in plat["chiplet_ids"]]
        NC = self.NC = int(plat["num_clusters_per_chiplet"])
        G = self.G = len(chips) * NC
        if HEADS % G:
            raise ValueError(f"{G} clusters do not split {HEADS} heads evenly")
        self.HPG = HEADS // G                        # heads per cluster
        mems = [int(m["id"]) for m in plat.get("mem_chips", [])]
        if not mems:
            raise ValueError(f"{args.platformcfg} lists no memory chips (an occamy.h from "
                             f"before MEM_CHIP_ID_<k>?)")
        # each compute chip's memory chiplet: the one beside it (C-M-C), and its push engine
        # there: 1 for the lower chip id beside that memory chiplet, 2 for the next
        self.mem, self.engine = {}, {}
        for k in chips:
            beside = [m for m in mems
                      if abs((m >> 4) - (k >> 4)) + abs((m & 15) - (k & 15)) == 1]
            if not beside:
                raise ValueError(f"compute chip {k:#04x} has no memory chip beside it")
            self.mem[k] = beside[0]
        for m in mems:
            fed = sorted(k for k in chips if self.mem[k] == m)
            for i, k in enumerate(fed):
                self.engine[k] = 1 + i
        self.LAT = int(p.get("latent_cluster", 0))   # W_DKV, the latent's norm, the append
        self.ATT = int(p.get("att_cluster", 1))      # the attention
        self.CHIP0 = chips[0]                         # its host runs the checks

        # the column splits, per global cluster
        self.o_cols = [D // G] * G                    # W_O, shared down, routed down, combine
        self.o0 = [sum(self.o_cols[:g]) for g in range(G)]
        self.sh_cols = split32(I_SH, G)               # shared intermediate columns
        self.sh0 = [sum(self.sh_cols[:g]) for g in range(G)]
        ex = split32(I_EXP, G)                        # routed: big and small shares
        big, small = ex[:G // 2], ex[G // 2:]         # alternate within a chip (balance)
        self.ex_cols = [big.pop(0) if g % 2 == 0 else small.pop(0) for g in range(G)]
        self.ex0 = [sum(self.ex_cols[:g]) for g in range(G)]
        # params expert_map: how the ROUTED experts are split.
        #   "tp8" (default): every slot over all G clusters -- each expert's SwiGLU output is
        #     all-gathered across every chip, through the memory chiplets' links.
        #   "colgroup": the compute chiplets' columns share the slots (slots 0-2 the first
        #     column, 3-5 the second); a slot is split over its column's clusters, so its
        #     all-gather crosses only the column's direct chip-to-chip link, which carries
        #     no weight pushes. Each cluster's down projection computes its own output
        #     columns AND its partner's in the other column (same row, same position); the
        #     partners swap those halves once, after every slot.
        self.expert_map = str(p.get("expert_map", "tp8"))
        if self.expert_map not in ("tp8", "colgroup"):
            raise ValueError(f"params expert_map={self.expert_map!r}: tp8 or colgroup")
        if self.expert_map == "colgroup":
            xs = sorted({k >> 4 for k in chips})
            self.groups = [[k for k in chips if k >> 4 == x] for x in xs]
            if len(self.groups) != 2 or TOP_K % 2:
                raise ValueError("expert_map colgroup: two columns of compute chiplets")
            self.gcl = [[g for k in grp for g in range(chips.index(k) * NC,
                                                         (chips.index(k) + 1) * NC)]
                        for grp in self.groups]
            self.slot_group = [s * 2 // TOP_K for s in range(TOP_K)]
            self.group_of = {g: gi for gi, gl in enumerate(self.gcl) for g in gl}
            exg = split32(I_EXP, len(self.gcl[0]))
            self.exg_cols, self.exg0, self.partner = {}, {}, {}
            for gl in self.gcl:
                for i, g in enumerate(gl):
                    self.exg_cols[g], self.exg0[g] = exg[i], sum(exg[:i])
            for a, b in zip(*self.gcl):
                self.partner[a], self.partner[b] = b, a

        # ---- data -------------------------------------------------------------------------
        data = self.data = dg.generate(p, args.dsv2_dir)
        for line in data["report"]:
            print(f"[dsv2] {line}")
        H = self.H = data["hw"]
        self.ks, self.inv = data["ks"], data["inv"]
        st = self.st = DataStaging(plat, on_host=True)
        if not st.has_hbm:
            raise ValueError(f"{args.platformcfg} has no HBM")
        self._stage_data()

        # ---- the graph --------------------------------------------------------------------
        dfg = self.dfg = BingoDFG(num_chiplets=len(chips),
                                  num_clusters_per_chiplet=NC,
                                  num_cores_per_cluster=plat["num_cores_per_cluster"],
                                  is_host_as_acc=True, chiplet_ids=chips,
                                  dep_tag_width=plat["dep_tag_width"])
        dfg.l1_capacity_bytes = int(p.get("l1_capacity", L1_CAPACITY))
        dfg.waiting_queue_depth = plat.get("waiting_queue_depth", 8)
        if "prune_fanout" in p:
            dfg.prune_fanout = bool(p["prune_fanout"])
        # params prune_implied: also drop edges another path already orders (e.g. a collect's
        # edge to a chip's first stash when gather_chain orders it before the second)
        dfg.prune_implied = bool(p.get("prune_implied", False))
        dfg.compact_task_tables = bool(p.get("compact_tables", True))
        # params remote_broadcast: a dependency set for every other chip is ONE broadcast
        # (a D2D multicast; the memory chips between the compute chips forward it, see
        # hemaia_d2d_link_initialize_grid); false: one targeted set per chip
        dfg.remote_broadcast = bool(p.get("remote_broadcast", True))
        ctx = self.ctx = Ctx(dfg=dfg, mesh=MESH, roles=core_roles(plat), hw=hw,
                             chiplet=self.CHIP0)
        self.pipe = Pipeline(ctx, verbose=True, gate_sources=True)
        wbuf = int(p.get("w_buffers", 2))
        # weight_crest: 1 KiB of slack per slab, the room an in-place expansion needs past a
        # full chunk (libs/crest.py checks every record against it)
        self.ls = [LoadStream(self.ctx_of(g), self.c(g), nbytes=self.chunk, nbuf=wbuf,
                              slack=1024 if self.crest else 0)
                   for g in range(G)]
        self._rings()
        if p.get("warm_kernels", False):
            self._warm_kernels()
        self.stash = []            # (name, stash stage, golden handle, got handle, nbytes)
        self.CHECKS_FROM = int(p.get("checks_from", 1))
        self.packs = []            # stage 1: each chip's x8 pack
        self.outs = []             # stage 8: every cluster's out slice
        self.xput = {}             # allgather: (name, g) -> its stash stage
        self.xget = {}             # allgather: (name, g) -> the destination's Collect / Pull

    def _warm_kernels(self):
        """params warm_kernels: every cluster runs its cold kernels once, on a zeroed 4 KiB
        scratch, before any weight lands. A kernel's first run pays its instruction-cache
        refills, and under iDMA traffic a refill waits behind the DMA (icache refills starved
        by iDMA), so a kernel's first run in the layer is much slower than the later ones.
        Here the refills happen on an idle fabric. These nodes are made before every block, so
        they head their cores' queues; nothing waits for them but each other, and they write
        only the scratch."""
        one = 0x3F800000                                # 1.0f
        for g in range(self.G):
            c, cg = self.c(g), self.ctx_of(g).at(self.c(g))
            buf = cg.l1(f"warm_g{g}", 4096)
            z = cg.node(f"warm_g{g}_Zero", cg.xdma, "__snax_bingo_kernel_xdma_memset",
                        SnaxBingoKernelXdmaMemsetArgs(buf, 4096,
                                                      SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO))
            if c == self.ATT:
                # one array block, C null (a non-null C is read even when not accumulated)
                cg.node(f"warm_g{g}_QK", cg.gemm, "__snax_bingo_kernel_gemm_fa_qk",
                        SnaxBingoKernelGemmFaQkArgs(at_offset(buf, 0), at_offset(buf, 64), 0,
                                                    at_offset(buf, 1024), M=1, K=1, N=1), [z])
            s = cg.node(f"warm_g{g}_Map", cg.simd, "__snax_bingo_kernel_simd_stream_map",
                        SnaxBingoKernelSimdScaleF16Args(at_offset(buf, 2048), at_offset(buf, 2112),
                                                        one, 1, 32), [z])
            s = cg.node(f"warm_g{g}_Ew", cg.simd, "__snax_bingo_kernel_simd_stream_elementwise",
                        SnaxBingoKernelSimdAddF16Args(at_offset(buf, 2176), at_offset(buf, 2240),
                                                      at_offset(buf, 2304), 1, 32), [s])
            cg.node(f"warm_g{g}_Swiglu", cg.simd, "__snax_bingo_kernel_simd_swiglu_a_row",
                    SnaxBingoKernelSimdSwigluARowArgs(at_offset(buf, 2368), at_offset(buf, 2496),
                                                      at_offset(buf, 2560), 32,
                                                      inv_scale_f32bits=one), [s])

    # ---- topology ---------------------------------------------------------------------------
    def chip(self, g):
        return self.chips[g // self.NC]

    def c(self, g):
        return g % self.NC

    def gs(self, k):
        """The global clusters of chip k."""
        i = self.chips.index(k)
        return list(range(i * self.NC, (i + 1) * self.NC))

    def ctx_of(self, g_or_chip, is_chip=False):
        k = g_or_chip if is_chip else self.chip(g_or_chip)
        return replace(self.ctx, chiplet=k)

    def mem_loc(self, k):
        m = self.mem[k]
        return (m >> 4, m & 15)

    # ---- data ------------------------------------------------------------------------------
    def _stage_data(self):
        """Everything but the weight images (_weight_images): the token, the factors, the
        RoPE tables, the latent cache, the goldens -- in the host image, on every chip."""
        st, data, H, G = self.st, self.data, self.H, self.G
        self.h = h = {}
        # the token, in every memory chiplet's HBM (the layer's input streams from memory)
        self.h_xh = {m: st.put_hbm(f"dsv2_xh_m{m:02x}", f16(H["x16"]).view(np.uint16),
                                   mem_chip=(m >> 4, m & 15))
                     for m in sorted(set(self.mem.values()))}
        self.h_x8 = st.put_zeros("dsv2_x8", "int8_t", 2 * D)          # each chip's own copy
        self.h_gx8 = st.put("dsv2_gold_x8", "int8_t", a_row_bytes(H["xq"]))
        for k, v in data["factors"].items():
            h[k] = st.put(f"dsv2_{k}", "uint16_t", v.view(np.uint16))
        # the shared expert's gate|up factors, cluster by cluster: its gate slice then its up
        s_gu = data["factors"]["s_sh_gu"]
        self.h_s_shgu = st.put("dsv2_s_shgu_split", "uint16_t", np.concatenate(
            [np.concatenate([s_gu[a: a + self.sh_cols[g]]
                             for a in (self.sh0[g], I_SH + self.sh0[g])]) for g in range(G)]
        ).view(np.uint16))
        # RoPE at the token's position: cos repeated per pair, sin signed; a table per row
        # count (the heads of a cluster, and one more for k_pe on the latent's cluster)
        c_rep = np.repeat(f16(H["cos16"]), 2)
        s_sgn = np.empty(RP, dtype=np.float16)
        s_sgn[0::2], s_sgn[1::2] = -f16(H["sin16"]), f16(H["sin16"])
        for n in (self.HPG, self.HPG + 1):
            h[f"cos{n}"] = st.put(f"dsv2_cos{n}", "uint16_t", np.tile(c_rep, n).view(np.uint16))
            h[f"sin{n}"] = st.put(f"dsv2_sin{n}", "uint16_t", np.tile(s_sgn, n).view(np.uint16))
        # the latent cache (ATT's chip reads its own copy), written by the append
        h["key"] = st.put("dsv2_key", "int8_t", data["key0"])
        h["val"] = st.put("dsv2_val", "int8_t", data["val0"])
        self._weight_images()

    def _weight_images(self):
        """Per cluster ONE image of the dense weights it streams, in stream order, in its
        memory chiplet's HBM; per chip its slices of every picked expert, and its expert
        table naming them. self.wh[(g, name)]: cluster g's slice of `name` (a handle)."""
        st, data, G, wb = self.st, self.data, self.G, self.wb
        bl = data["blobs"]
        self.wh = {}

        def cols(name, c0, n, d_in):
            b = bl[name]
            return b[wb(c0 * d_in): wb((c0 + n) * d_in)]

        img_parts = {g: [] for g in range(G)}
        self.cimg, self.cex = {}, {}
        for g in range(G):
            k = self.chip(g)
            hd0 = self.HPG * g
            parts = []
            if self.p.get("wdkv_split", False):
                # params wdkv_split: W_DKV's columns halved between LAT and ATT (stage 3)
                h_ = KV // 2
                if g == self.LAT:
                    parts.append(("wdkv", cols("wdkv", 0, h_, D)))
                elif g == self.ATT:
                    parts.append(("wdkv", cols("wdkv", h_, KV - h_, D)))
            elif g == self.LAT:
                parts.append(("wdkv", bl["wdkv"]))
            parts.append(("wq", cols("wq", QH * hd0, QH * self.HPG, D)))
            parts.append(("wuk", bl["wuk"][wb(hd0 * dg.Q_NOPE * KVR):
                                          wb((hd0 + self.HPG) * dg.Q_NOPE * KVR)]))
            parts.append(("wuv", bl["wuv"][wb(hd0 * KVR * dg.V_HEAD):
                                          wb((hd0 + self.HPG) * KVR * dg.V_HEAD)]))
            parts.append(("wo", cols("wo", self.o0[g], self.o_cols[g], D)))
            if self.c(g) == 0:                     # the router: every chip, its cluster 0
                parts.append(("wr", bl["wr"]))     # INT8 whatever wbits is
            parts.append(("sh_gu", np.concatenate(
                [cols("sh_gu", a, self.sh_cols[g], D) for a in (self.sh0[g],
                                                                I_SH + self.sh0[g])])))
            parts.append(("sh_dn", cols("sh_dn", self.o0[g], self.o_cols[g], I_SH)))
            img_parts[g] = parts
            img = np.concatenate([b for _, b in parts])
            hd = st.put_hbm(f"dsv2_wimg_g{g}", img, mem_chip=self.mem_loc(k))
            if self.crest:
                # the compressed copy the pushes read, written after the build (libs/crest.py)
                cap = img.nbytes + 64 * 1024
                self.cimg[g] = (hd.address, img.view(np.uint8).reshape(-1),
                                st.put_hbm(f"dsv2_cimg_g{g}", np.zeros(cap, np.uint8),
                                           mem_chip=self.mem_loc(k)), cap)
            off = 0
            for nm, b in parts:
                self.wh[(g, nm)] = at_offset(hd, off)
                off += len(b)
        self.img_bytes = {g: sum(len(b) for _, b in img_parts[g]) for g in range(G)}

        # the routed experts: each chip's slices of each picked expert, back to back per
        # field (its clusters in order), in its memory chiplet's HBM; its own table
        self.tables, self.table_bytes, self.slot_off = {}, {}, {}
        cg = self.expert_map == "colgroup"
        exc = self.exg_cols if cg else self.ex_cols
        ex0 = self.exg0 if cg else self.ex0
        # a cluster's down columns: its own -- and, colgroup, its partner's after them
        dcols = (lambda g: [(self.o0[g], self.o_cols[g]),
                            (self.o0[self.partner[g]], self.o_cols[self.partner[g]])]) if cg \
            else (lambda g: [(self.o0[g], self.o_cols[g])])
        for k in self.chips:
            entries = {}
            for e, (gus, dns, inv) in data["expert_f"].items():
                gu = np.concatenate([cols(f"e{e}_gu", a, exc[g], D)
                                     for g in self.gs(k) for a in (ex0[g], I_EXP + ex0[g])])
                gus_k = np.concatenate([gus[a: a + exc[g]] for g in self.gs(k)
                                        for a in (ex0[g], I_EXP + ex0[g])])
                dn = np.concatenate([cols(f"e{e}_dn", c0, n, I_EXP)
                                     for g in self.gs(k) for c0, n in dcols(g)])
                dns_k = np.concatenate([dns[c0: c0 + n] for g in self.gs(k)
                                        for c0, n in dcols(g)])
                loc = self.mem_loc(k)
                if self.crest:
                    # the region the compressed chunks are written into after the build; the
                    # table names it, so its address is fixed now (libs/crest.py)
                    for f, b in (("gu", gu), ("dn", dn)):
                        self.cex.setdefault((k, f), ({}, {}))[0][e] = \
                            np.ascontiguousarray(b).view(np.uint8).reshape(-1)
                    a_gu = st.put_hbm(f"dsv2_e{e}_gu_k{k:02x}", np.zeros(gu.nbytes, np.uint8),
                                      mem_chip=loc)
                    a_dn = st.put_hbm(f"dsv2_e{e}_dn_k{k:02x}", np.zeros(dn.nbytes, np.uint8),
                                      mem_chip=loc)
                    self.cex[(k, "gu")][1][e], self.cex[(k, "dn")][1][e] = a_gu, a_dn
                else:
                    a_gu = st.put_hbm(f"dsv2_e{e}_gu_k{k:02x}", gu, mem_chip=loc)
                    a_dn = st.put_hbm(f"dsv2_e{e}_dn_k{k:02x}", dn, mem_chip=loc)
                a_gus = st.put_hbm(f"dsv2_e{e}_gus_k{k:02x}", gus_k.view(np.int8), mem_chip=loc)
                a_dns = st.put_hbm(f"dsv2_e{e}_dns_k{k:02x}", dns_k.view(np.int8), mem_chip=loc)
                entries[e] = (a_gu.address, a_gus.address, a_dn.address, a_dns.address, inv)
            table = expert_table(N_EXP, entries)
            self.tables[k] = st.put(f"dsv2_expert_table_k{k:02x}", "int8_t", table)
            self.table_bytes[k] = table
        # where each cluster's slice starts in its chip's slice of a field, in bytes
        dn_n = lambda q: sum(n for _, n in dcols(q))
        self.slice_bytes = {g: {"gu": wb(2 * exc[g] * D), "dn": wb(dn_n(g) * I_EXP)}
                            for g in range(G)}
        for g in range(G):
            before = [q for q in self.gs(self.chip(g)) if q < g]
            self.slot_off[g] = {
                "gu": sum(wb(2 * exc[q] * D) for q in before),
                "gu_s": sum(2 * 2 * exc[q] for q in before),
                "dn": sum(wb(dn_n(q) * I_EXP) for q in before),
                "dn_s": sum(2 * dn_n(q) for q in before)}

    def _rings(self):
        """Per chip: its clusters' weight rings in its L3, filled by ITS memory chiplet's
        push engine under ITS host's prefetcher."""
        p, st = self.p, self.st
        wr = p.get("weight_ring", {"slots": 20, "batch": 4, "policy": 1})
        self.n_slots = n_slots = int(wr.get("slots", 20))
        self.ring_batch = int(wr.get("batch", 4))
        self.ring_batch_routed = int(wr.get("batch_routed", 0))     # 0: the same as batch
        # weight_ring.head / batch_head: a ring's runs that start within its first `head` chunks
        # batch by `batch_head` -- W_Q and W_UK, before any read crosses a link (0: off)
        self.ring_head = int(wr.get("head", 0))
        self.ring_batch_head = int(wr.get("batch_head", 0))
        # weight_ring.routed_rings: each cluster's routed chunks in a ring of their own (WeightRings
        # split), so a dense chunk taken after a routed one is still pushed before the route
        self.ring_split = bool(wr.get("routed_rings", False))
        self.NR = self.NC * (2 if self.ring_split else 1)       # rings per chip
        # weight_ring.queued: post the pushes to the engine's descriptor queue (no reads)
        self.ring_queued = int(bool(wr.get("queued", False)))
        # weight_ring.queued_inflight_routed: with queued, a routed run waits while this many
        # queued runs have not landed, so the link drains for the all-gathers' reads (0: no cap)
        self.ring_queued_inflight = int(wr.get("queued_inflight_routed", 0))
        # weight_ring.trailer (with weight_crest): every compressed record ends in a trailer
        # beat that says it has landed, so a run needs no flag transfer (libs/crest.py); the
        # slots are then zero at boot (.bss), so a poll never reads an unwritten word
        self.ring_trailer = bool(wr.get("trailer", False))
        if self.ring_trailer and not self.crest:
            raise ValueError("params weight_ring.trailer needs weight_crest")
        if self.ring_trailer and self.ring_queued:
            raise ValueError("params weight_ring.trailer with queued: the queue tracks landed "
                             "runs by their flags, which trailer runs do not push")
        seq = np.zeros((1024, 8), dtype=np.uint64)
        seq[:, 0] = np.arange(1024, dtype=np.uint64)
        self.h_seq = {m: st.put_hbm(f"dsv2_ring_seq_m{m:02x}", seq.view(np.int8).reshape(-1),
                                    mem_chip=(m >> 4, m & 15))
                      for m in sorted(set(self.mem.values()))}
        self.h_flags = st.put_zeros("dsv2_ring_flags", "uint8_t", self.NR * n_slots * 64)
        self.h_rel = st.put_zeros("dsv2_ring_release", "uint8_t", self.NR * n_slots * 64)
        self.h_rec_l3 = st.put_zeros("dsv2_rec_l3", "uint8_t", TOP_K * 128)
        self.rings = {}
        for k in self.chips:
            ck = self.ctx_of(k, is_chip=True)
            # (preloaded zeros, not .bss: the boot's DMA clear of these megabytes costs simulated
            # time before every run; params ring_preload: false restores it)
            slots = (st.put_zeros("dsv2_wring_slots", "uint8_t",
                                  self.NR * n_slots * self.slot_bytes,
                                  preload=bool(p.get("ring_preload", True)))
                     if self.ring_trailer and k == self.chips[0] else
                     self.h_wslots if self.ring_trailer else
                     ck.l3("wring_slots", self.NR * n_slots * self.slot_bytes))
            if self.ring_trailer:
                self.h_wslots = slots          # one symbol: every chip's own copy
            rings = self.rings[k] = WeightRings(
                slots=slots,
                flags=self.h_flags, releases=self.h_rel, n_rings=self.NR, n_slots=n_slots,
                slot_bytes=self.slot_bytes, batch=self.ring_batch,
                batch_routed=self.ring_batch_routed, head=self.ring_head,
                batch_head=self.ring_batch_head, split=self.ring_split)
            if self.crest:
                gs = self.gs(k)
                rings.crest = CrestRings(
                    _crest_codec(self.args.dsv2_dir),
                    dense={r: self.cimg[g] for r, g in enumerate(gs)},
                    routed={f: self.cex[(k, f)] for f in ("gu", "dn")},
                    slices={r: {f: (self.slot_off[g][f], self.slot_off[g][f] +
                                    self.slice_bytes[g][f]) for f in ("gu", "dn")}
                            for r, g in enumerate(gs)},
                    slab_bytes=self.ls[gs[0]].slab_bytes, trailer=self.ring_trailer,
                    size_only=bool(self.p.get("export_only", False)))
            for g in self.gs(k):
                self.ls[g].rings = rings
            ck.host_node(f"weight_prefetch_k{k:02x}", "__host_bingo_kernel_weight_prefetch",
                         HostBingoKernelWeightPrefetchArgs(
                             BingoMemSymbol(f"dsv2_ring_sched_k{k:02x}"), rings.slots,
                             self.h_flags, self.h_rel, self.h_seq[self.mem[k]], self.h_rec_l3,
                             TOP_K, self.NR, n_slots, self.slot_bytes, self.mem[k],
                             batch=self.ring_batch, policy=int(wr.get("policy", 1)),
                             engine=self.engine[k], batch_routed=self.ring_batch_routed,
                             queued=self.ring_queued,
                             queued_inflight_routed=self.ring_queued_inflight,
                             head=self.ring_head, batch_head=self.ring_batch_head))

    # ---- helpers the stages build with ------------------------------------------------------
    def wb(self, n, bits=None):
        """Bytes of n weights."""
        return n * (self.wbits if bits is None else bits) // 8

    def spec_row(self, n, g=None, level=L1, dtype=F16, rows=1, layout=RM):
        return PortSpec(layout, dtype, (rows, n), mem_level=level,
                        cluster=self.c(g) if level == L1 else None)

    @staticmethod
    def staged(spec, handle):
        """An array the datagen placed; nothing produced it."""
        return Port(spec, handle, ())

    @staticmethod
    def wspec(d_in, cols, bits, level=HBM):
        lay, dt = (Layout.B_W4, DType.I4) if bits == 4 else (B, I8)
        return PortSpec(lay, dt, (d_in, cols), mem_level=level)

    def add(self, block, name, at_g=None, at_chip=None, **binds):
        """`block` on the chip of global cluster `at_g` (or on chip `at_chip`). The binds are
        the block's ports, so these two names avoid every port name."""
        k = at_chip if at_chip is not None else self.chip(at_g)
        return self.pipe.add(block, name, chiplet=k, **binds)

    def gemv(self, name, g, d_in, d_out, k, x, w=None, groups=1, w_slot=None, rec=None,
             bits=None, x_layout=A, x_level=L1, first=False, wait_x=False):
        """A GEMV through cluster g's stream: the weight at handle `w`, or the record slot's
        slice of this cluster (w_slot = (slot, field))."""
        bits = self.wbits if bits is None else bits
        blk = Linear(tokens=1, d_in=d_in, d_out=d_out, mesh=MESH, cluster=self.c(g), gemv=True,
                     d_shift=k, x_level=x_level, x_layout=x_layout, w_level=HBM,
                     groups=groups, stream=self.ls[g], w_slot=w_slot, w_bits=bits,
                     rec_slots=TOP_K, stream_after_x=first, stream_wait_x=wait_x,
                     w_slot_off=self.slot_off[g][w_slot[1]] if w_slot else 0)
        if w_slot is None:
            return self.add(blk, name, g, x=x,
                            w=self.staged(self.wspec(d_in, groups * d_out, bits), w))
        return self.add(blk, name, g, x=x, rec=rec)

    def deq(self, name, g, n, x, s, s_off=0):
        """y (.) s: the factors from L3 (a handle) or from a stage in this L1."""
        c = self.c(g)
        if hasattr(s, "out"):
            return self.add(ScaleCols(cols=n, cluster=c, s_level=L1), name, g, x=x, s=s.out())
        return self.add(ScaleCols(cols=n, cluster=c, s_level=L3), name, g, x=x,
                        s=self.staged(self.spec_row(n, level=L3), at_offset(s, s_off)))

    def view(self, name, g, x, src, dst, offset=0):
        return self.add(View(src=src, dst=dst, offset=offset), name, g, x=x)

    def xsym(self, name, nbytes):
        """A staged hand-off array, one copy on every chip."""
        if name not in self.st._names:
            self.st.put_zeros(name, "uint8_t", nbytes)
        return name

    def allgather(self, name, parts, part_spec, part_bytes, dsts, dst_spec_of, chips=None):
        """Every cluster in `dsts` gets the parts of every cluster, back to back in global
        cluster order: each producer stashes its part into its OWN chip's copy of a staged
        array (at its place), then each destination's DM core reads every chip's run of it
        (Collect: its own chip's copy is a local read, the others' cross the links).

        `parts`: {g: port}; `part_bytes`: one count, or one per cluster; returns
        {g: the destination's stage}."""
        # `chips`: the chips taking part (default every one); `part_bytes` follows their
        # clusters in order, and so does the gathered layout
        chips = list(self.chips) if chips is None else list(chips)
        gl = [g for k in chips for g in self.gs(k)]
        pbl = [int(part_bytes)] * len(gl) if isinstance(part_bytes, (int, np.integer)) else \
            [int(v) for v in part_bytes]
        pb = dict(zip(gl, pbl))
        at = {g: sum(pbl[:i]) for i, g in enumerate(gl)}
        sym = self.xsym(f"dsv2_xchg_{name}", sum(pbl))
        runs, run_bytes = {}, {}
        # `gather_chain`: a chip's stashes in cluster order, each after the one before (an
        # ordering edge, no data), so a collect of the chip's run needs only the last one --
        # the others are implied (with dfg.prune_implied the compiler drops their edges):
        # one dependency tag per chip in the collector's cell instead of one per cluster.
        chain = bool(self.p.get("gather_chain", False))
        for k in chips:
            st_k = []
            for g in self.gs(k):
                h = BingoMemSymbol(sym, at[g], chip_id=k)
                x = parts[g]
                if chain and st_k:
                    gp = self.gs(k)[len(st_k) - 1]
                    x = self.add(After(x=part_spec(g), after=_l3(part_spec(gp))),
                                 f"xord_{name}_g{g}", g, x=x, after=st_k[-1].out()).out()
                st_k.append(self.add(Stash(src=part_spec(g), nbytes=pb[g], dst=h),
                                     f"xput_{name}_g{g}", g, x=x))
                # (name, g) -> the producer's stash, for a stream that must not load ahead
                # of it (stage 5's latent_after_q)
                self.xput[(name, g)] = st_k[-1]
            run_bytes[k] = sum(pb[g] for g in self.gs(k))
            runs[k] = self.add(
                Join(parts=[_l3(part_spec(g)) for g in self.gs(k)],
                     dst=PortSpec(RM, I8, (1, run_bytes[k]), mem_level=L3)),
                f"xrun_{name}_k{k:02x}", at_chip=k,
                **{f"x{i}": s.out() for i, s in enumerate(st_k)})
        # A chip's clusters are consecutive in g, so its run is one contiguous range of the
        # array: collecting the runs in chip order lays the parts out in g order.
        #
        # `gather_hier`: per chip only its first destination cluster collects across the
        # links, and the chip's other destinations copy that cluster's result over (a Pull,
        # L1 to L1 on the chip). A read across a link waits for whatever the memory chiplet
        # is pushing on it, so halving them halves those waits; and the second cluster then
        # waits on one local edge instead of one per producing cluster.
        hier = bool(self.p.get("gather_hier", False))
        out, first_on = {}, {}
        for g in dsts:
            k, spec = self.chip(g), dst_spec_of(g)
            if hier and k in first_on:
                # the destination as raw bytes (an A-layout row is a whole 16-row block, which
                # Pull will not cut), moved in one copy, then named again
                g0, tot = first_on[k], sum(pbl)
                raw = lambda c: PortSpec(RM, I8, (1, tot), mem_level=L1, cluster=c)
                src = self.view(f"xraw_{name}_g{g}", g0, out[g0].out(), dst_spec_of(g0),
                                raw(self.c(g0)))
                cp = self.add(Pull(rows=1, cols=tot, layout=RM, dtype=I8, src=self.c(g0),
                                   dst=self.c(g)), f"xget_{name}_g{g}", g, x0=src.out())
                self.xget[(name, g)] = cp
                out[g] = self.view(f"xnam_{name}_g{g}", g, cp.out(), raw(self.c(g)), spec)
                continue
            first_on.setdefault(k, g)
            out[g] = self.xget[(name, g)] = self.add(Collect(parts=[PortSpec(RM, I8, (1, run_bytes[q]), mem_level=L3)
                                             for q in chips],
                                      nbytes=[run_bytes[q] for q in chips],
                                      dst=spec, cluster=self.c(g)),
                              f"xget_{name}_g{g}", g,
                              **{f"x{i}": runs[q].out() for i, q in enumerate(chips)})
        return out

    def xfer(self, name, g_src, parts, part_spec, part_bytes, g_dst, dst_spec):
        """Cluster g_src's `parts` (ports of `part_spec`, `part_bytes` each) into cluster
        g_dst's L1, back to back: stashed into g_src's chip's copy of a staged array, read by
        g_dst's DM core in one copy (local write, remote read, like every hand-off here)."""
        k = self.chip(g_src)
        tot = part_bytes * len(parts)
        sym = self.xsym(f"dsv2_xchg_{name}", tot)
        st = [self.add(Stash(src=part_spec, nbytes=part_bytes,
                             dst=BingoMemSymbol(sym, i * part_bytes, chip_id=k)),
                       f"xput_{name}_{i}", g_src, x=pt)
              for i, pt in enumerate(parts)]
        run = self.add(Join(parts=[_l3(part_spec)] * len(parts),
                            dst=PortSpec(RM, I8, (1, tot), mem_level=L3)),
                       f"xrun_{name}", at_chip=k, **{f"x{i}": s.out() for i, s in enumerate(st)})
        return self.add(Collect(parts=[PortSpec(RM, I8, (1, tot), mem_level=L3)], nbytes=[tot],
                                dst=dst_spec, cluster=self.c(g_dst)),
                        f"xget_{name}", g_dst, x0=run.out())

    def scatter(self, name, g_src, port, spec, part_bytes, dsts, dst_spec_of):
        """Cluster g_src's buffer, one part per destination: g_src stashes the whole buffer
        into its chip's copy of a staged array, and each destination reads its part
        (`part_bytes[i]` bytes at the offset of the parts before it, in `dsts` order)."""
        k0 = self.chip(g_src)
        pb = [int(v) for v in part_bytes]
        sym = self.xsym(f"dsv2_xchg_{name}", sum(pb))
        s = self.add(Stash(src=spec, nbytes=sum(pb), dst=BingoMemSymbol(sym, chip_id=k0)),
                     f"xput_{name}", g_src, x=port)
        out, at = {}, 0
        for g, n in zip(dsts, pb):
            part = self.add(View(src=_l3(spec), dst=PortSpec(RM, I8, (1, n), mem_level=L3),
                                 offset=at),
                            f"xpart_{name}_g{g}", at_chip=k0, x=s.out())
            out[g] = self.add(Collect(parts=[PortSpec(RM, I8, (1, n), mem_level=L3)],
                                      nbytes=[n], dst=dst_spec_of(g), cluster=self.c(g)),
                              f"xget_{name}_g{g}", g, x0=part.out())
            at += n
        return out

    def check(self, name, stage, port, golden, nbytes, g, offset=0, st=99):
        """Stash `stage`'s output (on global cluster g) now, into a staged array on its chip;
        chip 0's host compares it at the end."""
        if st < self.CHECKS_FROM:
            return
        k = self.chip(g)
        sym = self.xsym(f"dsv2_chk_{name}", nbytes)
        sp = stage.out(port) if port else stage.out()
        s = self.add(Stash(src=_spec_of(stage, port), nbytes=nbytes, offset=offset,
                           dst=BingoMemSymbol(sym, chip_id=k)), f"stash_{name}", g, x=sp)
        self.stash.append((name, s, golden, BingoMemSymbol(sym, chip_id=k), nbytes))

    def gold(self, name, arr):
        """A golden, staged in L3 (chip 0 reads its own copy)."""
        a = np.ascontiguousarray(arr)
        if a.dtype == np.float16:
            return self.st.put(f"dsv2_g_{name}", "uint16_t", a.view(np.uint16).reshape(-1))
        return self.st.put(f"dsv2_g_{name}", "int8_t", a.view(np.int8).reshape(-1))

    # ---- after every stage: the checks, the graph, the emission -----------------------------
    def finish(self):
        dfg, ctx, p, st, args = self.dfg, self.ctx, self.p, self.st, self.args

        def all_checks():
            """After ALL compute, one at a time, upstream first, all byte-exact, on chip 0."""
            g0 = ctx.at(0)
            last = {}
            for _, s, _, _, _ in self.stash:
                nd = s.out().port.ends[-1]
                key = (nd.assigned_chiplet_id, nd.assigned_cluster_id)
                if key in last:
                    dfg.bingo_add_edge(last[key], nd)
                last[key] = nd
            prev = list(last.values()) + [p_.out().port.ends[-1] for p_ in self.packs]
            prev += [o.out().port.ends[-1] for o in self.outs]
            order = []
            for n, _s, gl, got, nb in self.stash:
                order.append((n, got, gl, nb))
            for name, got, golden, nbytes in order:
                prev = [vchecks.check_bytes(g0, f"Check_dsv2_{name}", golden=golden, got=got,
                                            nbytes=nbytes, after=prev, label=f"dsv2_{name}")]
        self.pipe.raw(all_checks, "checks")
        self.pipe.run()
        if self.crest:
            for k in self.chips:
                cr = self.rings[k].crest
                for kind, (u, n, r) in cr.finish(st.fill_hbm).items():
                    print(f"[dsv2] chip {k:#04x}: CREST {kind} {u / 2**20:.2f} MiB pushed as "
                          f"{n / 2**20:.2f} MiB ({r:.3f}x)")
                print(f"[dsv2] chip {k:#04x}: CREST {cr.plain} chunk(s) stored plain")

        for k in self.chips:
            r = self.rings[k]
            st.put(f"dsv2_ring_sched_k{k:02x}", "uint64_t",
                   np.array(r.schedule(), dtype=np.uint64))
            print(f"[dsv2] chip {k:#04x}: {r.chunks()} chunks pushed by memory chip "
                  f"{self.mem[k]:#04x} engine {self.engine[k]} "
                  f"({[len(e) for e in r.entries]} per cluster)")
        if args.data_h:
            st.emit(args.data_h, args.output_dir)
            if p.get("d2d_ddr", True):
                with open(args.data_h, "a") as f:
                    f.write("\n// params d2d_ddr: every D2D link switched to DDR at boot\n"
                            "#define HEMAIA_D2D_DDR 1\n")
            # params d2d_rx_yield: [compute chips, memory chips], the data links' RX yield
            # windows (hemaia_d2d_link_rx_yield_grid; 0 = off)
            if "d2d_rx_yield" in p:
                wc, wm = (int(v) for v in p["d2d_rx_yield"])
                if not (0 <= wc < 256 and 0 <= wm < 256):
                    raise ValueError(f"params d2d_rx_yield={p['d2d_rx_yield']}: 0..255 each")
                with open(args.data_h, "a") as f:
                    f.write(f"\n// params d2d_rx_yield: RX yield windows, compute / memory chips\n"
                            f"#define HEMAIA_D2D_RX_YIELD {wc}\n#define HEMAIA_D2D_RX_YIELD_MEM {wm}\n")
        extra = [os.path.basename(str(args.data_h))] if args.data_h else None
        # params fast_export: no graph pictures (a DSE export does not need them; minutes at 6k nodes)
        dfg.skip_viz = bool(p.get("fast_export", False) or p.get("export_only", False))
        dfg.bingo_compile_dfg(
            app_name=(f"DeepSeek-V2-Lite layer 1 to stage {self.STOP} ({STAGES[self.STOP]}), "
                      f"one token, INT{self.wbits}, {len(self.chips)} chips x {self.NC} "
                      f"clusters"),
            output_dir=args.output_dir,
            output_file_name=args.output_offload_file_name,
            extra_include_header_list=extra,
            static_l1=bool(p.get("static_l1", True)),
            desc_list_in_narrow_spm=not bool(p.get("desc_in_l3", False)))
        print(f"[dsv2] stage {self.STOP} ({STAGES[self.STOP]}); HBM {st.hbm_bytes / 2**20:.2f} "
              f"MiB over {len(set(self.mem.values()))} memory chips; "
              f"{sum(1 for _ in dfg.nodes)} nodes; per-cluster dense images "
              f"{[round(b / 2**20, 2) for b in self.img_bytes.values()]} MiB")
        print(f"Generated: {os.path.join(args.output_dir, args.output_offload_file_name)}")
        # params export_exec_ir: the compiled graph, rings and prefetch policy as the bingo
        # framework's Execution IR (its chiplet DSE simulates from it; passes/bingo_exec_export.py)
        if p.get("export_exec_ir", False):
            from bingo_exec_export import bingo_export_exec_ir, ring_table
            wr = p.get("weight_ring", {})
            out = p["export_exec_ir"] if isinstance(p["export_exec_ir"], str) else \
                os.path.join(args.output_dir, "exec_ir.json")
            bingo_export_exec_ir(
                dfg, out,
                ring_tables=[ring_table(self.rings[k], k, self.mem[k], self.engine[k])
                             for k in self.chips if k in getattr(self, "rings", {})],
                prefetch=dict(ring_slots=self.n_slots, batch=self.ring_batch,
                              batch_routed=self.ring_batch_routed, head=self.ring_head,
                              batch_head=self.ring_batch_head, policy=int(wr.get("policy", 1)),
                              queued=bool(self.ring_queued)),
                meta=dict(stage=self.STOP, chips=list(self.chips), nc=self.NC,
                          params={k: v for k, v in p.items()
                                  if isinstance(v, (int, float, str, bool, list, dict))},
                          trailer=bool(self.ring_trailer), crest=bool(self.crest),
                          slot_bytes=self.slot_bytes))
            print(f"[dsv2] Execution IR: {out}")


def _crest_codec(dsv2_dir):
    """snax_cluster's CREST reference codec (hw/chisel/doc/crest_decompressor/crest_codec.py),
    found from the golden's directory (target/snitch_cluster/sw/apps/dsv2 in snax_cluster)."""
    import importlib.util
    root = os.path.abspath(dsv2_dir)
    while root != os.path.dirname(root) and not os.path.isdir(os.path.join(root, "hw", "chisel")):
        root = os.path.dirname(root)
    path = os.path.join(root, "hw", "chisel", "doc", "crest_decompressor", "crest_codec.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"params weight_crest: no CREST codec at {path}")
    spec = importlib.util.spec_from_file_location("crest_codec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _l3(spec):
    """`spec`'s tensor in L3: what a Stash of it outputs."""
    return PortSpec(spec.layout, spec.dtype, spec.shape, mem_level=L3)


def _spec_of(stage, port):
    """The spec of a stage's output, before it is built."""
    blk = stage.template
    outs = blk.outputs
    return outs[port] if port else next(iter(outs.values()))


def run(stage):
    """Build stages 1..stage.STAGE, check them, and emit the workload's headers."""
    ap = argparse.ArgumentParser(description=(stage.__doc__ or "").strip().split("\n")[0])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_h", default=None)
    ap.add_argument("--output_offload_file_name", default="offload_bingo_hw.h")
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--hwcfg", required=True)
    ap.add_argument("--platformcfg", required=True)
    ap.add_argument("--dsv2_dir", required=True,
                    help="snax_cluster's target/snitch_cluster/sw/apps/dsv2 (the golden)")
    args = ap.parse_args()
    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    if int(p.get("stop", stage.STAGE)) != stage.STAGE:
        raise ValueError(f"{args.cfg}: stop={p['stop']}, but this is stage {stage.STAGE}")
    plat = parse_platform_cfg(args.platformcfg)
    if not (guard_chiplet_count(p, plat, args.output_dir, args.output_offload_file_name)
            and guard_cluster_count(p, plat, args.output_dir, args.output_offload_file_name)):
        return
    hw = load_cluster_cfg(args.hwcfg)
    shapes = [tuple(int(v) for v in s) for s in hw["snax_versacore_core_template"]
              ["snax_acc_cfg"][0]["snax_versacore_spatial_unrolling"][0]]
    if shapes[0] != MESH or GEMV_SHAPE not in shapes:
        raise ValueError(f"{args.hwcfg} declares array shapes {shapes}; this layer needs "
                         f"{MESH} and the one-token GEMV's {GEMV_SHAPE}")
    S = MBuild(args, p, plat, hw, stage.STAGE)
    stage.build(S)
    stage.checks(S)
    S.finish()
