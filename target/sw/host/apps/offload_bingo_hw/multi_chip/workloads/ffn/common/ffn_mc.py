# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the ffn workloads (workloads/ffn/<model>/<variant>/): one LLM's dense
SwiGLU FFN at its real size, tensor-parallel over every compute chiplet, its weights streamed
from the memory chiplets.

THE PLATFORM (hemaia_16c4m_8MBL3_2cluster): a 5 x 4 grid whose middle column is memory,
two clusters per compute chiplet:

    C00 C10 M20 C30 C40      each memory chiplet feeds the four compute chips of its row,
    C01 C11 M21 C31 C41      one push engine each, from its own HBM: C1y and C3y beside it,
    C02 C12 M22 C32 C42      C0y and C4y one chip further (their pushes cross C1y / C3y)
    C03 C13 M23 C33 C43

THE MAPPING (Megatron): every cluster g takes a contiguous range of the intermediate columns,
whole 32-column pairs (an INT4 GEMV's unit), the extra pairs spread over the links that carry
them:

    broadcast   the root's x (INT8, a_row) into every cluster's L1
    gate|up     its columns of gate and up, one GEMV, dequantised       -> g16 [2 n]
    SwiGLU      silu(gate) * up, quantised into the down GEMV's x         -> a8 [n]
    down        its ROWS of down: a partial y over every output column    -> part [hidden]
    local fold  cluster 1 hands its partial to cluster 0, which adds them -> loc [hidden]
    reduce      the chips' loc folded into the root's, in the fold order
    dequant     y16 = y_raw (.) s_dn, on the root

THE ABLATION. Each switch below replaces one mechanism by its conventional counterpart; the
compute is the same in every variant, so a difference in latency is the mechanism's. The paper
ablates three: bcast, reduce and sync (with barrier_mode bcast, the vanilla chip barrier, as
the conventional sync).

    weight_crest  true: every weight chunk pushed CREST-compressed and expanded in place in L1
                  (libs/crest.py); false: plain INT4. Not part of the ablation: the paper's
                  cluster has no decompressor, so every variant streams plain INT4
    bcast         bcast_l1: ONE D2D multicast per cluster index from the root's DM core into
                  that cluster's L1 on every chip, then a landed flag there; each cluster
                  waits on its own flag. pull: every cluster's DM core reads the root's copy.
    reduce        chain: ONE xDMA ChainGather by the root -- its partial travels the fold
                  order and every chip's writer junction adds that chip's loc in transit.
                  root: the root's DM core pulls every chip's loc in the fold order into two
                  L1 slots and its SIMD adds each, pipelined against the pulls.
    sync          edge: every cross-chip dependency is a BINGO hardware dependency, so a task
                  waits for exactly its producers. barrier: the window has no cross-chip
                  dependency; two chip barriers stand in for them (barrier_mode bcast: the
                  0xFF broadcast barrier; sw: the point-to-point one), each on cluster 0's DM
                  core of every chip: one after the start (where the edges fan the start out
                  to every chip), one once the chip's loc is ready (where they fan the locs
                  in to the reduce).

THE FOLD ORDER is a Hamiltonian cycle over the compute chips through the root, with as few
multi-hop steps as the grid allows (two here: one crossing of the memory column each way).
Both reduces follow it: the root's pulls and adds start from the root's own partial, the
ChainGather starts from the cycle's next chip and adds the root's own partial last. The
golden adds in the arm's order (ffn_data.py).

SYMMETRIC L1. The multicast writes x8 and its flag at the root's own addresses on every chip,
and the chain names the other chips' loc buffers the same way. So those buffers are one
allocation per cluster, `AAAA_ffn_sym` (x8, then its flag; loc is written over x8, which the
gate|up GEMV has read by then), which sorts before every other handle and therefore sits at
offset 0 of each cluster's static arena, the first allocation of each cluster's heap: the same
local address on every chip. finish() checks the placement and refuses a build where it
differs. The buffer is live from its setup (zeroed, so no flag reads X) to a hold task after
the checks, so the packer never lets anything else share its bytes.

TOKENS. params tokens: T tokens in one pass, every GEMV running one group per token over the
same weight chunk; x8, every partial and y are T rows.

TIMING. Two anchors on the root's DM core (__snax_bingo_kernel_sync_mark: mcycle stamps and an
SRAM-probe mark) bracket the FFN: start after every chip's setup, end after y16. The weight
prefetchers start after the start anchor, so the weight stream is inside the window.
params anchor_reduce: a third anchor once every chip's loc is ready, before the reduce, which
splits the window -- and makes the root's pulls wait for the last chip, so it is off by default.
"""

import argparse
import os
import sys

import hjson
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE.split("/target/sw/host/")[0]
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
from bingo_data_staging import DataStaging  # noqa: E402
from bingo_dfg import BingoDFG  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelChipBarrierArgs, HostBingoKernelSyncReportArgs,
    HostBingoKernelWeightPrefetchArgs, SnaxBingoKernelChipBarrierArgs,
    SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSyncMarkArgs, SnaxBingoKernelSyncProbeArgs,
    SnaxBingoKernelWaitFlagsArgs,
    SnaxBingoKernelXdmaChainGatherArgs, SnaxBingoKernelXdmaMemsetArgs)
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_platform import (core_roles, load_cluster_cfg, parse_platform_cfg,  # noqa: E402
                            writer_junction_index)
from libs import (Ctx, DType, Layout, MemLevel, Pipeline, Port, PortSpec,  # noqa: E402
                  at_offset)
from libs.blocks import Linear, LoadStream, ScaleCols, Stash, SwigluARow, WeightRings  # noqa: E402
from libs.crest import CrestRings  # noqa: E402
from libs.verify import checks as vchecks  # noqa: E402

import ffn_data  # noqa: E402

L1_CAPACITY = 990000
MESH = (16, 4, 16)
L1, L3, HBM = MemLevel.L1, MemLevel.L3, MemLevel.HBM
F16, I8 = DType.F16, DType.I8
RM, A_ROW = Layout.ROW_MAJOR, Layout.A_ROW
MASK40 = "0xFFFFFFFFFFULL"
EW_F16_ADD = (0 << 4) | 0          # elementwise junction CSR(0): fmt FP16, op ADD
SYM = "AAAA_ffn_sym"               # sorts before every other handle: offset 0 of the arena
PAIR = 32                          # an INT4 GEMV's column unit


def remote(chip, local_expr):
    """`local_expr` (a local address, the same on every chip) on chip `chip`."""
    return f"(chiplet_addr_transform_full(0x{chip:02x}, ((uint64_t)({local_expr}) & {MASK40})))"


def dist(a, b):
    return abs((a >> 4) - (b >> 4)) + abs((a & 15) - (b & 15))


def fold_cycle(chips, root):
    """A Hamiltonian cycle over `chips` through `root` (root first), each step to a chip at
    Manhattan distance 1, or 2 where the grid leaves no choice: the fewest distance-2 steps,
    found by backtracking (16 chips: instant). Falls back to a path if no cycle exists."""
    nb = {k: sorted((m for m in chips if m != k and dist(k, m) <= 2),
                    key=lambda m: (dist(k, m), m)) for k in chips}

    def search(path, seen, longs, budget, cycle):
        k = path[-1]
        if len(path) == len(chips):
            if not cycle:
                return path
            d = dist(k, root)
            if d == 1 or (d == 2 and longs < budget):
                return path
            return None
        free = [m for m in nb[k] if m not in seen]
        # fewest onward choices first (Warnsdorff), the near ones before the far ones
        free.sort(key=lambda m: (dist(k, m), sum(1 for q in nb[m] if q not in seen), m))
        for m in free:
            ln = longs + (dist(k, m) == 2)
            if ln > budget:
                continue
            seen.add(m)
            got = search(path + [m], seen, ln, budget, cycle)
            if got:
                return got
            seen.discard(m)
        return None

    for cycle in (True, False):
        for budget in range(0, 5):
            got = search([root], {root}, 0, budget, cycle)
            if got:
                return got
    raise ValueError(f"no fold order over {[hex(c) for c in chips]}")


def _crest_codec(dsv2_dir):
    """snax_cluster's CREST reference codec (hw/chisel/doc/crest_decompressor/crest_codec.py)."""
    import importlib.util
    root = os.path.abspath(dsv2_dir)
    while root != os.path.dirname(root) and not os.path.isdir(os.path.join(root, "hw", "chisel")):
        root = os.path.dirname(root)
    path = os.path.join(root, "hw", "chisel", "doc", "crest_decompressor", "crest_codec.py")
    spec = importlib.util.spec_from_file_location("crest_codec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FBuild:
    def __init__(self, args, p, plat, hw):
        self.args, self.p, self.plat, self.hw = args, p, plat, hw
        self.T = int(p.get("tokens", 1))
        if self.T < 1:
            raise ValueError(f"params tokens={self.T}: at least one")
        self.wbits = int(p.get("wbits", 4))
        self.chunk = int(p.get("w_chunk_bytes", 112 * 1024))
        self.crest = bool(p.get("weight_crest", False))
        self.slot_bytes = self.chunk + (128 if self.crest else 0)
        self.bcast_impl = str(p.get("bcast", "bcast_l1"))
        self.reduce_impl = str(p.get("reduce", "chain"))
        if self.bcast_impl not in ("bcast_l1", "pull"):
            raise ValueError(f"params bcast={self.bcast_impl!r}: bcast_l1 or pull")
        if self.reduce_impl not in ("chain", "root"):
            raise ValueError(f"params reduce={self.reduce_impl!r}: chain or root")
        self.sync = str(p.get("sync", "edge"))
        if self.sync not in ("edge", "barrier"):
            raise ValueError(f"params sync={self.sync!r}: edge or barrier")
        bm = str(p.get("barrier_mode", "bcast"))
        if bm not in ("bcast", "sw"):
            raise ValueError(f"params barrier_mode={bm!r}: bcast or sw")
        self.barrier_mode = (SnaxBingoKernelChipBarrierArgs.MODE_BCAST if bm == "bcast"
                             else SnaxBingoKernelChipBarrierArgs.MODE_SW)
        self.n_barrier = 0

        # ---- topology -------------------------------------------------------------------
        chips = self.chips = [int(k) for k in plat["chiplet_ids"]]
        NC = self.NC = int(plat["num_clusters_per_chiplet"])
        self.G = len(chips) * NC
        self.ROOT = chips[0]
        mems = {int(m["id"]): int(m.get("num_sys_idma", 4)) for m in plat.get("mem_chips", [])}
        if not mems:
            raise ValueError(f"{args.platformcfg} lists no memory chips")
        # each compute chip's memory chip: the nearest (same row first); its push engine there:
        # the chips a memory chip feeds, nearest first, engines 0, 1, ...
        self.mem, self.engine = {}, {}
        for k in chips:
            self.mem[k] = min(mems, key=lambda m: (dist(k, m), (m & 15) != (k & 15), m))
        for m, n_eng in mems.items():
            fed = sorted((k for k in chips if self.mem[k] == m), key=lambda k: (dist(k, m), k))
            if len(fed) > n_eng:
                raise ValueError(f"memory chip {m:#04x} feeds {len(fed)} chips with {n_eng} "
                                 f"push engines")
            for i, k in enumerate(fed):
                self.engine[k] = i
        # the link a chip's weights arrive over: its memory chip and the first step's side
        side = lambda k: ((k >> 4) > (self.mem[k] >> 4)) - ((k >> 4) < (self.mem[k] >> 4))
        self.link = {k: (self.mem[k], side(k)) for k in chips}
        self.cycle = fold_cycle(chips, self.ROOT)
        self.order = self.cycle                    # the fold order: the root, then the cycle

        # ---- data -----------------------------------------------------------------------
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/ffn/cache"))
        d = self.d = ffn_data.FfnData(cache, p.get("model", "dsv2-lite-l0"), args.dsv2_dir,
                                      seed=int(p.get("seed", 1)), wbits=self.wbits,
                                      tokens=self.T)
        self.D, self.I = d.H, d.I
        self.cols = self._split_columns()
        # the order the reduce adds the chips' partials: the root's pulls and adds start from
        # the root's own; a ChainGather starts from the chain's first chip, the root's own last
        fold = self.order[1:] + self.order[:1] if self.reduce_impl == "chain" else self.order
        self.gold = d.golden(self.cols, {k: self.gs(k) for k in chips}, fold)
        for line in d.report(self.gold["y16"]):
            print(f"[ffn] {line}")
        print(f"[ffn] fold order {[f'{k:02x}' for k in self.order]}; columns per cluster "
              f"{[n for _, n in self.cols]}; per link "
              f"{ {f'{m:02x}{s:+d}': sum(n for g, (_, n) in enumerate(self.cols) if self.link[self.chip(g)] == (m, s)) for m, s in sorted(set(self.link.values()))} }")
        if NC != 2:
            raise ValueError(f"{NC} clusters per chip: the local fold adds two")

        st = self.st = DataStaging(plat, on_host=True)
        if not st.has_hbm:
            raise ValueError(f"{args.platformcfg} has no HBM")
        dfg = self.dfg = BingoDFG(num_chiplets=len(chips), num_clusters_per_chiplet=NC,
                                  num_cores_per_cluster=plat["num_cores_per_cluster"],
                                  is_host_as_acc=True, chiplet_ids=chips,
                                  dep_tag_width=plat["dep_tag_width"])
        dfg.l1_capacity_bytes = int(p.get("l1_capacity", L1_CAPACITY))
        dfg.waiting_queue_depth = plat.get("waiting_queue_depth", 8)
        dfg.compact_task_tables = bool(p.get("compact_tables", True))
        dfg.remote_broadcast = bool(p.get("remote_broadcast", True))
        self.roles = core_roles(plat)
        self.ctx = Ctx(dfg=dfg, mesh=MESH, roles=self.roles, hw=hw, chiplet=self.ROOT)
        self.pipe = Pipeline(self.ctx, verbose=False, gate_sources=True)
        self.jct = writer_junction_index(hw, "HasElementwiseJunction")

        self._stage()
        wbuf = int(p.get("w_buffers", 3))
        self.ls = [LoadStream(self.cx(g), self.c(g), nbytes=self.chunk, nbuf=wbuf,
                              slack=1024 if self.crest else 0) for g in range(self.G)]
        self._rings()
        # ---- the symmetric collective buffer of every cluster ---------------------------
        # x8 lands at OFF_X (a_row) and its flag after it; the chip's loc is written over x8
        # on cluster 0, which the gate|up GEMV has read by then (loc follows the down GEMV)
        xb = 2 * self.D * self.T                  # x8 in a_row, and a partial in FP16
        self.OFF_X, self.OFF_FLAG, self.OFF_LOC = 0, xb, 0
        self.sym_bytes = xb + 64
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(SYM, self.sym_bytes)
                    for k in chips for c in range(NC)}
        # what needs no fixed address: cluster 1's partial handed to cluster 0, the root's
        # folded y_raw and its two pull slots
        self.p1 = {k: self.cx_chip(k).at(0).l1("ffn_p1", xb) for k in chips}
        self.dst = self.cx_chip(self.ROOT).at(0).l1("ffn_dst", xb)
        self.slots = self.cx_chip(self.ROOT).at(0).l1("ffn_slots", 2 * xb)
        self.n_anchor = 0
        self.stamps = BingoMemAlloc("A_ffn_stamps", size=4 * 16, mem_level="L2",
                                    chip_id=self.ROOT, cluster_id=0)
        self.mark = BingoMemAlloc("zz_ffn_sram_mark", size=64, mem_level="L1",
                                  chip_id=self.ROOT, cluster_id=0)

    # ---- topology helpers -----------------------------------------------------------------
    def chip(self, g):
        return self.chips[g // self.NC]

    def c(self, g):
        return g % self.NC

    def gs(self, k):
        i = self.chips.index(k)
        return list(range(i * self.NC, (i + 1) * self.NC))

    def cx(self, g):
        return Ctx(dfg=self.dfg, mesh=MESH, roles=self.roles, hw=self.hw,
                   chiplet=self.chip(g)).at(self.c(g))

    def cx_chip(self, k):
        return Ctx(dfg=self.dfg, mesh=MESH, roles=self.roles, hw=self.hw, chiplet=k)

    def _split_columns(self):
        """(c0, n) per global cluster: the intermediate columns in whole pairs, the pairs left
        over after an even split going one per link in turn (a link carries the weights of
        every chip behind it), each to the nearest chip's first cluster first."""
        units, G = self.I // PAIR, self.G
        if self.I % PAIR:
            raise ValueError(f"{self.I} intermediate columns are not whole {PAIR}-column pairs")
        base, extra = divmod(units, G)
        links = sorted(set(self.link.values()))
        per_link = {L: sorted((k for k in self.chips if self.link[k] == L),
                              key=lambda k: (dist(k, self.mem[k]), k)) for L in links}
        order = []
        for c in range(self.NC):
            for i in range(max(len(v) for v in per_link.values())):
                for L in links:
                    if i < len(per_link[L]):
                        order.append(self.chips.index(per_link[L][i]) * self.NC + c)
        n = {g: PAIR * (base + (1 if g in order[:extra] else 0)) for g in range(G)}
        cols, c0 = [], 0
        for g in range(G):
            cols.append((c0, n[g]))
            c0 += n[g]
        return cols

    # ---- data -----------------------------------------------------------------------------
    def _stage(self):
        st, d = self.st, self.d
        self.h_x8 = st.put("ffn_x8", "int8_t", d.x8())
        self.h_sdn = st.put("ffn_s_dn", "uint16_t", np.asarray(d.dn_f, np.float16).view(np.uint16))
        self.wh, self.h_sgu, self.cimg = {}, {}, {}
        for g, (c0, n) in enumerate(self.cols):
            k = self.chip(g)
            gu, dn = self.d.gu_blob(c0, n), self.d.dn_blob(c0, n)
            img = np.concatenate([gu, dn])
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            hd = st.put_hbm(f"ffn_wimg_g{g}", img, mem_chip=loc)
            self.wh[(g, "gu")], self.wh[(g, "dn")] = hd, at_offset(hd, gu.size)
            if self.crest:
                cap = img.nbytes + 64 * 1024
                self.cimg[g] = (hd.address, img.view(np.uint8).reshape(-1),
                                st.put_hbm(f"ffn_cimg_g{g}", np.zeros(cap, np.uint8), mem_chip=loc),
                                cap)
            self.h_sgu[g] = st.put(f"ffn_s_gu_g{g}", "uint16_t",
                                   np.asarray(d.gu_factors(c0, n), np.float16).view(np.uint16))
        self.img_bytes = sum(int(n) * (2 * self.D + self.D) * self.wbits // 8 for _, n in self.cols)

    def _rings(self):
        """Per chip: its clusters' weight rings in its L3, filled by its memory chip's push
        engine under its host's prefetcher (host_kernel_lib.h weight_prefetch)."""
        p, st = self.p, self.st
        wr = p.get("weight_ring", {})
        self.n_slots = n_slots = int(wr.get("slots", 16))
        self.ring_batch = int(wr.get("batch", 4))
        self.ring_trailer = bool(wr.get("trailer", self.crest))
        if self.ring_trailer and not self.crest:
            raise ValueError("params weight_ring.trailer needs weight_crest")
        NR = self.NC
        seq = np.zeros((1024, 8), dtype=np.uint64)
        seq[:, 0] = np.arange(1024, dtype=np.uint64)
        mems = sorted(set(self.mem.values()))
        self.h_seq = {m: st.put_hbm(f"ffn_ring_seq_m{m:02x}", seq.view(np.int8).reshape(-1),
                                    mem_chip=(m >> 4, m & 15)) for m in mems}
        self.h_flags = st.put_zeros("ffn_ring_flags", "uint8_t", NR * n_slots * 64)
        self.h_rel = st.put_zeros("ffn_ring_release", "uint8_t", NR * n_slots * 64)
        self.h_rec = st.put_zeros("ffn_rec_l3", "uint8_t", 128)
        # one symbol for every chip's slots (zero at boot, so a trailer poll never reads X)
        self.h_wslots = st.put_zeros("ffn_wring_slots", "uint8_t", NR * n_slots * self.slot_bytes,
                                     preload=True)
        self.rings, self.prefetch = {}, {}
        for k in self.chips:
            rings = self.rings[k] = WeightRings(
                slots=self.h_wslots, flags=self.h_flags, releases=self.h_rel, n_rings=NR,
                n_slots=n_slots, slot_bytes=self.slot_bytes, batch=self.ring_batch)
            gs = self.gs(k)
            if self.crest:
                rings.crest = CrestRings(_crest_codec(self.args.dsv2_dir),
                                         dense={r: self.cimg[g] for r, g in enumerate(gs)},
                                         routed={}, slices={},
                                         slab_bytes=self.ls[gs[0]].slab_bytes,
                                         trailer=self.ring_trailer)
            for g in gs:
                self.ls[g].rings = rings
            self.prefetch[k] = self.cx_chip(k).host_node(
                f"weight_prefetch_k{k:02x}", "__host_bingo_kernel_weight_prefetch",
                HostBingoKernelWeightPrefetchArgs(
                    BingoMemSymbol(f"ffn_ring_sched_k{k:02x}"), rings.slots, self.h_flags,
                    self.h_rel, self.h_seq[self.mem[k]], self.h_rec, 1, NR, n_slots,
                    self.slot_bytes, self.mem[k], batch=self.ring_batch,
                    policy=int(wr.get("policy", 1)), engine=self.engine[k]))

    # ---- nodes ----------------------------------------------------------------------------
    def node(self, k, c, core, name, kernel, kargs, after=()):
        return self.cx_chip(k).node(f"{name}_k{k:02x}", core, kernel, kargs, after, cluster=c)

    def edge(self, a, b):
        self.dfg.bingo_add_edge(a, b)

    def anchor(self, tag, after=()):
        nd = self.node(self.ROOT, 0, self.roles["dm"], f"A{self.n_anchor}_{tag}",
                       "__snax_bingo_kernel_sync_mark",
                       SnaxBingoKernelSyncMarkArgs(stamp_buf=self.stamps, slot=self.n_anchor,
                                                   mark_addr=self.mark), after)
        self.n_anchor += 1
        return nd

    def probe(self, k, tag, after=()):
        """A DM-core task on chip k's cluster 0 that does nothing but order."""
        return self.node(k, 0, self.roles["dm"], f"P_{tag}", "__snax_bingo_kernel_sync_probe",
                         SnaxBingoKernelSyncProbeArgs(stamp_buf=0, slot=0), after)

    def barrier(self, after):
        """One chip barrier over every compute chip: a task on each chip's cluster 0 DM core,
        after `after[k]`. Returns {chip: its barrier task}."""
        self.n_barrier += 1
        xs, ys = [k >> 4 for k in self.chips], [k & 15 for k in self.chips]
        tl, br = (min(xs) << 4) | min(ys), (max(xs) << 4) | max(ys)
        return {k: self.node(k, 0, self.roles["dm"], f"barrier{self.n_barrier}",
                             "__snax_bingo_kernel_chip_barrier",
                             SnaxBingoKernelChipBarrierArgs(self.barrier_mode, tl, br,
                                                            seq=self.n_barrier), after[k])
                for k in self.chips}

    def sv(self, k, c, off):
        """A view of chip k cluster c's symmetric buffer at `off`."""
        return self.sym[(k, c)].view(off)

    def sx(self, c, off):
        """The C expression of the root's own symmetric buffer at `off`: the same local
        address on every chip (see the module doc)."""
        return f"((uint64_t)({self.sym[(self.ROOT, c)].get_c_var_name()}) + {off})"

    # ---- the FFN --------------------------------------------------------------------------
    def build(self):
        p, D, T, NC = self.p, self.D, self.T, self.NC
        dm, xdma, simd = self.roles["dm"], self.roles["xdma"], self.roles["simd"]
        xb = 2 * D * T
        # ---- setup: zero every symmetric buffer (the flags must not read X) --------------
        setup_end = {}
        for k in self.chips:
            prev = None
            for c in range(NC):
                z = self.node(k, c, xdma, f"sym_zero_c{c}", "__snax_bingo_kernel_xdma_memset",
                              SnaxBingoKernelXdmaMemsetArgs(self.sym[(k, c)], self.sym_bytes,
                                                            SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO),
                              prev)
                prev = z
            if self.sync == "barrier":
                prev = self.cx_chip(k).host_node(
                    f"barrier_init_k{k:02x}", "__host_bingo_kernel_chip_barrier",
                    HostBingoKernelChipBarrierArgs(HostBingoKernelChipBarrierArgs.OP_INIT), [prev])
            setup_end[k] = prev
        if self.sync == "edge":
            # the start fans out to every chip as dependencies
            start = self.anchor("start", [setup_end[k] for k in self.chips])
            go = {k: start for k in self.chips}
        else:
            # every chip parks in the first barrier before the start: the barrier state is
            # zeroed everywhere (ready), every chip is armed, then the root stamps the start
            # and arrives last, so the window holds the barrier's release and nothing else
            ready = self.probe(self.ROOT, "ready", [setup_end[k] for k in self.chips])
            armed = {k: self.probe(k, "armed", [ready]) for k in self.chips}
            start = self.anchor("start", list(armed.values()))
            go = self.barrier({k: [start] if k == self.ROOT else [armed[k]] for k in self.chips})
        for k in self.chips:
            self.edge(go[k], self.prefetch[k])

        # ---- broadcast x ------------------------------------------------------------------
        xin = {}
        if self.bcast_impl == "bcast_l1":
            puts = []
            for c in range(NC):
                put = self.node(self.ROOT, 0, dm, f"x_bput_c{c}", "__snax_bingo_kernel_idma_put_flagged",
                                SnaxBingoKernelIdmaPutFlaggedArgs(
                                    self.h_x8, self.sv(self.ROOT, c, self.OFF_X), xb,
                                    self.sv(self.ROOT, c, self.OFF_FLAG), value=1, bcast=True,
                                    also_here=True), [go[self.ROOT]] + puts[-1:])
                puts.append(put)
                for k in self.chips:
                    w = self.node(k, c, dm, f"x_wait_c{c}", "__snax_bingo_kernel_wait_flags",
                                  SnaxBingoKernelWaitFlagsArgs(self.sv(k, c, self.OFF_FLAG), 1, 1),
                                  [go[k]] if k != self.ROOT else [put])
                    if k != self.ROOT:
                        self.dfg.bingo_add_order_edge(put, w)
                    xin[(k, c)] = w
        else:
            for k in self.chips:
                src = self.h_x8 if k == self.ROOT else BingoMemSymbol("ffn_x8", chip_id=self.ROOT)
                for c in range(NC):
                    xin[(k, c)] = self.node(k, c, dm, f"x_pull_c{c}", "__snax_bingo_kernel_idma_1d_copy",
                                            SnaxBingoKernelIdma1dCopyArgs(
                                                src, self.sv(k, c, self.OFF_X), xb), [go[k]])

        # ---- per cluster: gate|up, SwiGLU, down -----------------------------------------
        self.part = {}
        for g, (c0, n) in enumerate(self.cols):
            k, c = self.chip(g), self.c(g)
            xs = PortSpec(A_ROW, I8, (T, D), mem_level=L1, cluster=c)
            x = Port(xs, self.sv(k, c, self.OFF_X), (xin[(k, c)],))
            # the stream's first loads wait for x, which follows the start: a ring load has no
            # producer in the graph, so the in-order DM core would otherwise take it before
            # the start anchor -- and spin on a chunk the prefetcher pushes only after it
            self.ls[g].hold_until(xin[(k, c)])
            gu = self.pipe.add(
                Linear(tokens=T, d_in=D, d_out=2 * n, mesh=MESH, cluster=c, gemv=True,
                       d_shift=self.d.k_x, x_level=L1, x_layout=A_ROW, w_level=HBM,
                       stream=self.ls[g], w_bits=self.wbits),
                f"gu_g{g}", chiplet=k, x=x,
                w=Port(PortSpec(Layout.B_W4, DType.I4, (D, 2 * n), mem_level=HBM),
                       self.wh[(g, "gu")], ()))
            g16 = self.pipe.add(ScaleCols(cols=2 * n, rows=T, cluster=c, s_level=L3), f"g16_g{g}",
                                chiplet=k, x=gu.out(),
                                s=Port(PortSpec(RM, F16, (1, 2 * n), mem_level=L3),
                                       self.h_sgu[g], ()))
            a8 = self.pipe.add(SwigluARow(inter=n, cluster=c, inv_scale_f32bits=self.d.inv_a,
                                          rows=T, a_row=True), f"a8_g{g}", chiplet=k,
                               g=g16.out())
            self.part[g] = self.pipe.add(
                Linear(tokens=T, d_in=n, d_out=D, mesh=MESH, cluster=c, gemv=True,
                       d_shift=self.d.k_dn, x_level=L1, x_layout=A_ROW, w_level=HBM,
                       stream=self.ls[g], w_bits=self.wbits),
                f"dn_g{g}", chiplet=k, x=a8.out(),
                w=Port(PortSpec(Layout.B_W4, DType.I4, (n, D), mem_level=HBM),
                       self.wh[(g, "dn")], ()))
        self.pipe.raw(self._reduce, "reduce")
        # ---- the output: y16 = y_raw (.) s_dn on the root -------------------------------
        ys = PortSpec(RM, F16, (T, D), mem_level=L1, cluster=0)
        self.y16 = self.pipe.add(
            ScaleCols(cols=D, rows=T, cluster=0, s_level=L3), "y16", chiplet=self.ROOT,
            x=self.pipe.source("y_raw", ys, lambda: Port(ys, self.dst,
                                                         tuple(self.reduce_end))).out(),
            s=Port(PortSpec(RM, F16, (1, D), mem_level=L3), self.h_sdn, ()))
        self.pipe.raw(self._finish_graph, "finish")
        self.pipe.run()

    def _reduce(self):
        """The local folds, then the reduce over the chips into the root's dst."""
        D, T, NC = self.D, self.T, self.NC
        dm, xdma, simd = self.roles["dm"], self.roles["xdma"], self.roles["simd"]
        xb = 2 * D * T
        rows = xb // 64
        loc = {}
        for k in self.chips:
            gs = self.gs(k)
            ports = [self.part[g].out().port for g in gs]
            # every other cluster pushes its partial into cluster 0's symmetric buffer
            prev = None
            for i, (g, pt) in enumerate(zip(gs[1:], ports[1:]), start=1):
                prev = self.node(k, i, self.roles["dm"], f"part_push_c{i}",
                                 "__snax_bingo_kernel_idma_1d_copy",
                                 SnaxBingoKernelIdma1dCopyArgs(pt.handle, self.p1[k], xb),
                                 list(pt.ends))
            add = self.node(k, 0, simd, "loc_add", "__snax_bingo_kernel_simd_stream_elementwise",
                            SnaxBingoKernelSimdAddF16Args(ports[0].handle, self.p1[k],
                                                          self.sv(k, 0, self.OFF_LOC), rows, 32),
                            list(ports[0].ends) + ([prev] if prev is not None else []))
            loc[k] = add
        self.loc_nodes = loc
        R, others = self.ROOT, self.order[1:]
        before = []
        if self.sync == "barrier":
            # no dependency crosses a chip: every chip arrives once its loc is ready
            b = self.barrier({k: [loc[k]] for k in self.chips})
            loc = {k: b[R] for k in self.chips}
        if self.p.get("anchor_reduce", False):
            before = [self.anchor("reduce", list(dict.fromkeys(loc.values())))]
        if self.reduce_impl == "chain":
            chain = [remote(k, self.sx(0, self.OFF_LOC)) for k in others] + \
                    [remote(R, self.dst.get_c_var_name())]
            ga = SnaxBingoKernelXdmaChainGatherArgs(
                local_src=remote(R, self.sx(0, self.OFF_LOC)), chain=chain, size=xb,
                junction=str(self.jct), jct_csr0=EW_F16_ADD,
                hops=[(k, 0) for k in others] + [(R, 0)])
            # the chain names its buffers by address expressions, which the static L1 planner
            # does not see: name dst here, so its live range covers the gather
            ga.holds = [self.dst]
            gth = self.node(R, 0, xdma, "y_chain", "__snax_bingo_kernel_xdma_chain_gather", ga,
                            before or list(dict.fromkeys(loc[k] for k in self.chips)))
            self.reduce_end = [gth]
            return
        # root: pull i into slot i % 2, add it into dst (the first add starts from loc)
        pulls, adds = [], []
        for i, k in enumerate(others):
            slot = self.slots.view((i % 2) * xb)
            cp = self.node(R, 0, dm, f"y_pull{i}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(remote(k, self.sx(0, self.OFF_LOC)),
                                                         slot, xb),
                           (before or [loc[k]]) + pulls[-1:] + ([adds[i - 2]] if i >= 2 else []))
            pulls.append(cp)
            a_in = self.sv(R, 0, self.OFF_LOC) if i == 0 else self.dst
            add = self.node(R, 0, simd, f"y_add{i}", "__snax_bingo_kernel_simd_stream_elementwise",
                            SnaxBingoKernelSimdAddF16Args(a_in, slot, self.dst,
                                                          rows, 32),
                            [cp, loc[R]] + adds[-1:])
            adds.append(add)
        self.reduce_end = [adds[-1]]

    def _finish_graph(self):
        """The end anchor, the report, the checks, and the hold that keeps the symmetric
        buffers live to the end."""
        R, D, T = self.ROOT, self.D, self.T
        y_end = list(self.y16.out().port.ends)
        end = self.anchor("end", y_end)
        rep = self.cx_chip(R).host_node(
            "ffn_report", "__host_bingo_kernel_sync_report",
            HostBingoKernelSyncReportArgs(stamp_buf=self.stamps, num_phases=self.n_anchor - 1,
                                          local_phase=self.n_anchor - 1), [end])
        # ---- checks: copies into L3 symbols on the producer's chip, compared by the root --
        g0 = self.cx_chip(R)
        cps = []

        def stash(name, k, c, src, nbytes, after):
            sym = f"ffn_chk_{name}"
            self.st.put_zeros(sym, "uint8_t", nbytes)
            cp = self.node(k, c, self.roles["dm"], f"chk_{name}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(src, BingoMemSymbol(sym), nbytes),
                           [end] + list(after))
            cps.append(cp)
            return BingoMemSymbol(sym, chip_id=k)

        gold = self.gold
        f16b = lambda a: np.asarray(a, np.float16).view(np.uint16)
        checks = []
        y16p = self.y16.out().port
        xb = 2 * D * T
        checks.append(("y16", stash("y16", R, 0, y16p.handle, xb, y16p.ends), gold["y16"], xb))
        checks.append(("y_raw", stash("y_raw", R, 0, self.dst, xb, []), gold["y_raw"], xb))
        far = self.order[len(self.order) // 2]
        checks.append((f"loc_k{far:02x}", stash(f"loc_k{far:02x}", far, 0,
                                                self.sv(far, 0, self.OFF_LOC), xb, []),
                       gold["loc"][far], xb))
        prev = [rep] + cps
        for name, got, golden, nb in checks:
            gh = self.st.put(f"ffn_gold_{name}", "uint16_t", f16b(golden))
            prev = [vchecks.check_bytes(g0, f"Check_ffn_{name}", golden=gh, got=got, nbytes=nb,
                                        after=prev, label=f"ffn_{name}")]
        # the hold: every symmetric buffer named once more, after everything
        for (k, c), h in self.sym.items():
            self.node(k, c, self.roles["dm"], f"sym_hold_c{c}", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(h, 0, 0), prev if k == R else [end])

    # ---- emission -------------------------------------------------------------------------
    def finish(self):
        dfg, p, st, args = self.dfg, self.p, self.st, self.args
        if self.crest:
            for k in self.chips:
                cr = self.rings[k].crest
                for kind, (u, n, r) in cr.finish(st.fill_hbm).items():
                    if u:
                        print(f"[ffn] chip {k:#04x}: CREST {kind} {u / 2**20:.2f} MiB pushed as "
                              f"{n / 2**20:.2f} MiB ({r:.3f}x)")
                print(f"[ffn] chip {k:#04x}: CREST {cr.plain} chunk(s) stored plain")
        for k in self.chips:
            r = self.rings[k]
            st.put(f"ffn_ring_sched_k{k:02x}", "uint64_t", np.array(r.schedule(), dtype=np.uint64))
            print(f"[ffn] chip {k:#04x}: {r.chunks()} chunks pushed by memory chip "
                  f"{self.mem[k]:#04x} engine {self.engine[k]}")
        if args.data_h:
            st.emit(args.data_h, args.output_dir)
            if p.get("d2d_ddr", True):
                with open(args.data_h, "a") as f:
                    f.write("\n// params d2d_ddr: every D2D link switched to DDR at boot\n"
                            "#define HEMAIA_D2D_DDR 1\n")
        extra = [os.path.basename(str(args.data_h))] if args.data_h else None
        dfg.skip_viz = bool(p.get("fast_export", True))
        dfg.bingo_compile_dfg(
            app_name=(f"{self.d.W['model']} FFN {self.D}->{self.I}->{self.D}, {self.T} token(s), "
                      f"INT{self.wbits}, {len(self.chips)} chips x {self.NC} clusters, "
                      f"crest={int(self.crest)} bcast={self.bcast_impl} reduce={self.reduce_impl}"),
            output_dir=args.output_dir, output_file_name=args.output_offload_file_name,
            extra_include_header_list=extra, static_l1=bool(p.get("static_l1", True)),
            desc_list_in_narrow_spm=True)
        self._check_symmetric()
        self._check_core_order(os.path.join(args.output_dir, args.output_offload_file_name))
        print(f"[ffn] HBM {st.hbm_bytes / 2**20:.2f} MiB over {len(set(self.mem.values()))} "
              f"memory chips; {sum(1 for _ in dfg.nodes)} nodes; weights "
              f"{self.img_bytes / 2**20:.2f} MiB")

    def _check_core_order(self, header):
        """Every DM core reaches its cluster's x task before its first weight-ring load, in
        the order the hardware runs it: the descriptor list of the emitted header, which is
        not the node-id order. A ring load spins on a chunk the prefetcher pushes only after
        the start, and the start anchor shares the root's DM core: one ring load ahead of it
        is a deadlock no dependency check sees."""
        from chiplet_mc import desc_rows
        names = {str(n.node_id): n.node_name for n in self.dfg.node_list}
        txt = open(header).read()
        seen_x, bad = set(), []
        dm = str(self.roles["dm"])
        for chip, nid, ty, cl, core in desc_rows(txt):
            if ty != "0" or core != dm:
                continue
            nm = names.get(nid, "")
            key = (chip, cl)
            if nm.startswith(("x_wait_", "x_pull_")):
                seen_x.add(key)
            elif "_Ld_" in nm and key not in seen_x:
                bad.append(f"chip {chip} cluster {cl}: {nm}")
        if bad:
            raise ValueError(f"a weight load runs before its cluster's x: {bad[:4]}")
        if len(seen_x) != len(self.chips) * self.NC:
            raise ValueError(f"the core-order check found the x task on {len(seen_x)} of "
                             f"{len(self.chips) * self.NC} DM cores (header format changed?)")

    def _check_symmetric(self):
        """Every cluster's symmetric buffer at offset 0 of its static arena."""
        pl = getattr(self.dfg, "static_l1_placement", None)
        if not pl:
            raise ValueError("the ffn workloads need static_l1: the symmetric buffers' addresses")
        bad = [(k, c, pl[id(h)][1]) for (k, c), h in self.sym.items()
               if id(h) not in pl or pl[id(h)][1] != 0]
        if bad:
            raise ValueError(f"symmetric buffers not at offset 0: {bad[:4]}")


def run():
    ap = argparse.ArgumentParser(description="ffn workload generator")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_h", default=None)
    ap.add_argument("--output_offload_file_name", default="offload_bingo_hw.h")
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--hwcfg", required=True)
    ap.add_argument("--platformcfg", required=True)
    ap.add_argument("--dsv2_dir", required=True,
                    help="snax_cluster's target/snitch_cluster/sw/apps/dsv2 (the numerics models)")
    args = ap.parse_args()
    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    plat = parse_platform_cfg(args.platformcfg)
    if int(p.get("num_chiplets", plat["num_chiplets"])) != plat["num_chiplets"] or \
            int(p.get("num_clusters", plat["num_clusters_per_chiplet"])) != \
            plat["num_clusters_per_chiplet"]:
        raise ValueError(f"{args.cfg}: {p.get('num_chiplets')} chips x {p.get('num_clusters')} "
                         f"clusters, but {args.platformcfg} has {plat['num_chiplets']} x "
                         f"{plat['num_clusters_per_chiplet']}")
    hw = load_cluster_cfg(args.hwcfg)
    S = FBuild(args, p, plat, hw)
    S.build()
    S.finish()
