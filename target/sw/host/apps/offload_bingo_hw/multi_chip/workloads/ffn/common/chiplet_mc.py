# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""What the level-2 workloads share on the 16 + 4 chiplet platform: the topology, the weight rings,
the anchors, the synchronisation arm, the broadcast of the token, the chips' fold and the reduce,
the checks and the emission. ffn_mc.py documents each mechanism; the workloads (ffn, moe) build
their layer between `setup()` and `finish()`:

    S = Workload(args, p, plat, hw)     # its data, S.make_sym(...), S.make_rings(...)
    go = S.setup()                      # the setup, the start anchor, the first barrier
    xin = S.broadcast(...)              # the token into every cluster
    ...                                 # the layer: per cluster a partial over every output
    S.reduce(parts)                     # the chips' folds and the reduce into the root's dst
    S.finish(...)                       # end anchor, report, checks, compile

THE SWITCHES (params): bcast (bcast_l1 | pull), reduce (chain | root), sync (edge | barrier) and
barrier_mode (bcast | sw), as ffn_mc.py describes them; the ablation turns the first three on and
off, with barrier_mode bcast. weight_crest (CREST chunks expanded in place) is kept but is not part of
the ablation: the paper's cluster has no decompressor.
"""

import os
import re
import sys

import numpy as np

# workloads/common: fabric.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))

from bingo_data_staging import DataStaging  # noqa: E402
from bingo_dfg import BingoDFG  # noqa: E402


# The descriptor list of an emitted header, in the order the hardware runs it: one comment line
# a descriptor names its node and where it runs. (chip, node id, type, cluster, core) a row; the
# second pattern reads headers that wrote the list one store a word.
_DESC_IMAGE = re.compile(r"// Node ID (\d+)\s*\n\s*//\s+Type=(\d) TaskID=\d+ Assigned=(\w\w)/(\d+)/(\d+)")
_DESC_STORES = re.compile(r"bingo_hw_scheduler_task_desc_list_chip_(\w\w)\[\d+ \* BINGO_TASK_DESC_WORDS"
                          r" \+ 0\] = \w+; // Node ID (\d+)\s*\n\s*// Fields: Type=(\d), TaskID=\d+"
                          r"\s*\n\s*//\s*Assigned: Chiplet=\w+, Cluster=(\d+), Core=(\d+)")


def desc_rows(txt):
    """[(chip, node id, type, cluster, core)] of a header's descriptor lists, in list order."""
    rows = [(c, n, t, cl, co) for n, t, c, cl, co in _DESC_IMAGE.findall(txt)]
    return rows or _DESC_STORES.findall(txt)
from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelChipBarrierArgs, HostBingoKernelSyncReportArgs,
    HostBingoKernelWeightPrefetchArgs, SnaxBingoKernelChipBarrierArgs,
    SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdmaChainGatherArgs,
    SnaxBingoKernelXdmaMemsetArgs, XDMA_EWJCT_FMT_FP16, XDMA_EWJCT_OP_ADD, xdma_ewjct_csr0)
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_platform import core_roles, writer_junction_index  # noqa: E402
from libs import Ctx, MemLevel, Pipeline  # noqa: E402
from libs.blocks import LoadStream, WeightRings  # noqa: E402
from libs.crest import TRAILER_MAGIC, CrestRings  # noqa: E402
from libs.verify import checks as vchecks  # noqa: E402

# the shared mesh, address and fold helpers, re-exported for the workloads that import them from here
from fabric import (L1_CAPACITY, MASK40, MESH, FabricBase, crest_codec, dist,  # noqa: E402,F401
                    fold_cycle, remote)

EW_F16_ADD = xdma_ewjct_csr0(XDMA_EWJCT_OP_ADD, XDMA_EWJCT_FMT_FP16)
PAIR = 32                          # an INT4 GEMV's column unit


class ClusterNodes:
    """The node factory the mini_compiler blocks take (GemmQueue.dispatch, fa_sweep): chip k cluster c
    of a ChipletBuild, creating its nodes with the build's names."""

    def __init__(self, fb, k, c):
        self.fb, self.chiplet, self.cluster, self.dfg = fb, k, c, fb.dfg
        self.gemm, self.dm = fb.roles["gemm"], fb.roles["dm"]

    def node(self, name, core, kernel, args, after=()):
        return self.fb.node(self.chiplet, self.cluster, core, name, kernel, args, after)


class ChipletBuild(FabricBase):
    TAG = "l2"                          # the prefix of every staged symbol and printed line

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
        self.last_barrier = {}             # {chip: its newest barrier task}
        self.barrier_rounds = []           # every barrier(): {chip: its task}
        # the vanilla barrier fences every core of the chip (fence_barriers); off: only the
        # tasks a barrier names wait for it
        self.barrier_fence = self.sync == "barrier" and bool(p.get("barrier_fence", True))

        # ---- topology -------------------------------------------------------------------
        chips = self.chips = [int(k) for k in plat["chiplet_ids"]]
        NC = self.NC = int(plat["num_clusters_per_chiplet"])
        if NC != 2:
            raise ValueError(f"{NC} clusters per chip: the local fold adds two")
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
        self.order = fold_cycle(chips, self.ROOT)   # the fold order: the root, then the cycle

        # ---- the graph ------------------------------------------------------------------
        self.st = DataStaging(plat, on_host=True)
        if not self.st.has_hbm:
            raise ValueError(f"{args.platformcfg} has no HBM")
        dfg = self.dfg = BingoDFG(num_chiplets=len(chips), num_clusters_per_chiplet=NC,
                                  num_cores_per_cluster=plat["num_cores_per_cluster"],
                                  is_host_as_acc=True, chiplet_ids=chips,
                                  dep_tag_width=plat["dep_tag_width"])
        dfg.l1_capacity_bytes = int(p.get("l1_capacity", L1_CAPACITY))
        dfg.waiting_queue_depth = plat.get("waiting_queue_depth", 8)
        dfg.compact_task_tables = bool(p.get("compact_tables", True))
        # each chiplet's id maps hold its own entries only (bingo_g2l16_build, bingo_hw_scheduler_sparse)
        dfg.compact_id_maps = bool(p.get("compact_id_maps", False))
        # each cluster's L1 holds its own tasks' global ids instead of a table of every task's
        # (bingo_hw_scheduler_init_gid_lists): its L1 cost does not grow with the whole graph
        dfg.gid_lists = bool(p.get("gid_lists", False))
        # params per_chip_image: one host image per chiplet, holding the shared code and data and that chiplet's
        # own tables only (target/sw/host/apps/per_chip_image.py), so an image's L3 share does not grow with the
        # other chiplets' tasks; needs the one-function-per-chiplet host code (split_chip_functions)
        dfg.per_chip_image = bool(p.get("per_chip_image", False))
        dfg.static_args_image = bool(p.get("static_args_image", True))
        # the generator's faster host init (bingo_dfg_emit: L1 addresses at compile time, the kernel list
        # by one loop, the symbol lookup by signature, the g2l table runs by DMA)
        dfg.fast_host_init = bool(p.get("fast_host_init", False))
        dfg.remote_broadcast = bool(p.get("remote_broadcast", True))
        self.roles = core_roles(plat)
        self.ctx = Ctx(dfg=dfg, mesh=MESH, roles=self.roles, hw=hw, chiplet=self.ROOT)
        self.pipe = Pipeline(self.ctx, verbose=False, gate_sources=True)
        self.jct = {n: writer_junction_index(hw, f"Has{n}Junction")
                    for n in ("Elementwise", "TopK", "SoftmaxStats")}
        self.n_anchor = 0
        self.stamps = BingoMemAlloc(f"A_{self.TAG}_stamps", size=4 * 16, mem_level="L2",
                                    chip_id=self.ROOT, cluster_id=0)
        self.mark = BingoMemAlloc(f"zz_{self.TAG}_sram_mark", size=64, mem_level="L1",
                                  chip_id=self.ROOT, cluster_id=0)
        self.sym = {}
        self.checks = []                       # (name, got handle, golden array, bytes)

    # ---- topology helpers -----------------------------------------------------------------
    def fold_order(self):
        """The chips in the order the reduce adds their partials (FP16 addition does not
        associate, so the golden adds in it): the root's pulls and adds start from the root's
        own partial (reduce root); a ChainGather starts from the chain's first chip and the
        root's own partial, its local source, enters last (reduce chain)."""
        if self.reduce_impl == "chain":
            return self.order[1:] + self.order[:1]
        return list(self.order)

    def chain_order(self):
        """The order a ChainGather through every chip from the root combines their beats."""
        return self.order[1:] + self.order[:1]

    def split_columns(self, inter, clusters=None, last=(), unit=PAIR, fixed=None):
        """{g: (c0, n)}: `inter` columns over `clusters` (default every cluster) in whole units (default
        pairs), the units left over after an even split going one per link in turn (a link carries the
        weights of every chip behind it), each to the nearest chip's first cluster first; the
        clusters in `last` get a leftover unit only after every other cluster has one. `fixed`
        {g: columns}: those clusters get exactly that many, the rest is split over the others."""
        clusters = list(range(self.G)) if clusters is None else list(clusters)
        fixed = dict(fixed or {})
        if inter % unit or any(v % unit for v in fixed.values()):
            raise ValueError(f"{inter} intermediate columns are not whole {unit}-column units")
        if fixed:
            rest = self.split_columns(inter - sum(fixed.values()), [g for g in clusters if g not in fixed],
                                      last, unit)
            n = {g: (fixed[g] if g in fixed else rest[g][1]) for g in clusters}
            out, c0 = {}, 0
            for g in sorted(clusters):
                out[g] = (c0, n[g])
                c0 += n[g]
            return out
        units, G = inter // unit, len(clusters)
        base, extra = divmod(units, G)
        links = sorted(set(self.link[self.chip(g)] for g in clusters))
        per_link = {L: sorted({self.chip(g) for g in clusters if self.link[self.chip(g)] == L},
                              key=lambda k: (dist(k, self.mem[k]), k)) for L in links}
        order = []
        for c in range(self.NC):
            for i in range(max(len(v) for v in per_link.values())):
                for L in links:
                    if i < len(per_link[L]):
                        g = self.chips.index(per_link[L][i]) * self.NC + c
                        if g in clusters:
                            order.append(g)
        order = [g for g in order if g not in last] + [g for g in order if g in last]
        n = {g: unit * (base + (1 if g in order[:extra] else 0)) for g in clusters}
        out, c0 = {}, 0
        for g in sorted(clusters):
            out[g] = (c0, n[g])
            c0 += n[g]
        return out

    # ---- nodes ----------------------------------------------------------------------------
    def at(self, k, c):
        """Chip k's cluster c as the node factory a mini_compiler block takes."""
        return ClusterNodes(self, k, c)

    def barrier(self, after):
        """One chip barrier over every compute chip: a task on each chip's cluster 0 DM core,
        after `after[k]` and after the chip's previous barrier. A barrier matches the n-th
        arrival of every chip, so every chip must run its barriers in program order; two
        barriers with no path between them on a chip could otherwise be sorted either way.
        Returns {chip: its barrier task}."""
        self.n_barrier += 1
        xs, ys = [k >> 4 for k in self.chips], [k & 15 for k in self.chips]
        tl, br = (min(xs) << 4) | min(ys), (max(xs) << 4) | max(ys)
        out = {}
        for k in self.chips:
            deps = list(after[k]) + ([self.last_barrier[k]] if k in self.last_barrier else [])
            out[k] = self.node(k, 0, self.roles["dm"], f"barrier{self.n_barrier}",
                               "__snax_bingo_kernel_chip_barrier",
                               SnaxBingoKernelChipBarrierArgs(self.barrier_mode, tl, br,
                                                              seq=self.n_barrier),
                               list(dict.fromkeys(deps)))
        self.last_barrier = out
        self.barrier_rounds.append(out)
        return out

    def fence_barriers(self):
        """The vanilla chip barrier's fence (chip_sync.h snrt_hw_chip_global_barrier opens and
        closes with snrt_global_barrier): no core of a chip passes a barrier before every core
        of the chip has reached it. Each barrier task of chip k follows every device task of
        chip k created before it and precedes every one created after it, on both clusters.
        Creation order is the program order: a task belongs to the phase between the barriers
        it was created between. The host core (the prefetcher) stays outside, as the CVA6
        stays outside snrt_global_barrier. The compiler's prune passes keep one edge per core
        of each fan. Returns the number of edges added."""
        import networkx as nx
        dev = {self.roles[r] for r in ("gemm", "simd", "xdma", "dm")}
        added = 0
        for k in self.chips:
            bars = [rnd[k] for rnd in self.barrier_rounds]
            ids = [b.node_id for b in bars]
            if ids != sorted(ids):
                raise ValueError(f"chip {k:#04x}: barriers created out of order")
            bar_set = set(bars)
            for t in self.dfg.node_list:
                if (t.assigned_chiplet_id != k or t.assigned_core_id not in dev
                        or t in bar_set):
                    continue
                phase = sum(1 for i in ids if i < t.node_id)
                if phase > 0 and not self.dfg.has_edge(bars[phase - 1], t):
                    self.dfg.bingo_add_edge(bars[phase - 1], t)
                    added += 1
                if phase < len(bars) and not self.dfg.has_edge(t, bars[phase]):
                    self.dfg.bingo_add_edge(t, bars[phase])
                    added += 1
        if not nx.is_directed_acyclic_graph(self.dfg):
            cyc = nx.find_cycle(self.dfg)
            raise ValueError(f"the barrier fence closes a cycle: "
                             f"{[(a.node_name, b.node_name) for a, b in cyc[:6]]}")
        # each fan's members on one core are chained by the core order; keep the last of them
        self.dfg.prune_implied = True
        print(f"[{self.TAG}] barrier fence: {len(self.barrier_rounds)} barrier(s) per chip, "
              f"{added} edges before pruning")
        return added

    # ---- buffers --------------------------------------------------------------------------
    def make_sym(self, nbytes, xb, dst_lead=0):
        """Every cluster's symmetric buffer (`nbytes`), and the buffers the reduce needs where
        no fixed address is: cluster 1's partial handed to cluster 0, the root's dst (the
        folded output) and its two pull slots, `xb` bytes each. `dst_lead`: bytes kept below
        dst in its buffer (a SIMD kernel's latch beat, for a reader of dst)."""
        self.sym_bytes = int(nbytes)
        # "AAAA_" sorts before every other handle: offset 0 of each cluster's static arena
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_sym", self.sym_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.p1 = {k: self.cx_chip(k).at(0).l1(f"{self.TAG}_p1", xb) for k in self.chips}
        self.dst_lead = int(dst_lead)
        self.dst_buf = self.cx_chip(self.ROOT).at(0).l1(f"{self.TAG}_dst", xb + self.dst_lead)
        self.dst = self.dst_buf.view(self.dst_lead) if self.dst_lead else self.dst_buf
        self.slots = self.cx_chip(self.ROOT).at(0).l1(f"{self.TAG}_slots", 2 * xb)

    def dst_expr(self):
        """The C expression of the root's dst."""
        if not self.dst_lead:
            return self.dst_buf.get_c_var_name()
        return f"((uint64_t)({self.dst_buf.get_c_var_name()}) + {self.dst_lead})"

    def make_streams(self):
        wbuf = int(self.p.get("w_buffers", 3))
        self.ls = [LoadStream(self.cx(g), self.c(g), nbytes=self.chunk, nbuf=wbuf,
                              slack=1024 if self.crest else 0) for g in range(self.G)]

    @staticmethod
    def chunk_trailer():
        """The 64-B record a plain weight chunk ends in with weight_ring.trailer: the magic the
        ring load polls for (idma_ring_load), the rest zero."""
        t = np.zeros(64, dtype=np.int8)
        t[:8] = np.array([TRAILER_MAGIC], dtype="<u8").view(np.int8)
        return t

    def make_rings(self, dense, routed=None, slices=None, rec_l3=None, rec_slots=1):
        """Per chip: its clusters' weight rings in its L3, filled by its memory chip's push
        engine under its host's prefetcher (host_kernel_lib.h weight_prefetch).
        dense {g: (address, bytes, compressed handle, capacity)} with weight_crest (CrestRings
        dense); routed {chip: {field: ({expert: bytes}, {expert: handle})}} and slices
        {chip: {ring: {field: (first, end)}}} for routed weights; rec_l3 the L3 record each
        chip's prefetcher waits for (the routed runs)."""
        p, st, T = self.p, self.st, self.TAG
        wr = p.get("weight_ring", {})
        self.n_slots = n_slots = int(wr.get("slots", 16))
        self.ring_batch = int(wr.get("batch", 4))
        # weight_ring.trailer: a chunk's own last 64 B (a CREST record's, or without CREST a record
        # the workload appends to every plain chunk, chunk_trailer()) say it has landed, so the
        # prefetcher pushes no flag after it
        self.ring_trailer = bool(wr.get("trailer", self.crest))
        # weight_ring.head / batch_head: a ring's runs that start within its first `head` chunks batch
        # by `batch_head`, so a cluster's first chunks land one by one, not behind a whole run (0: off)
        self.ring_head = int(wr.get("head", 0))
        self.ring_batch_head = int(wr.get("batch_head", 0))
        # weight_ring.queued: the prefetcher posts each push to the engine's descriptor queue and goes on
        self.ring_queued = int(bool(wr.get("queued", False)))
        if self.ring_trailer and self.ring_queued:
            raise ValueError("params weight_ring.trailer with queued: the queue tracks landed runs by "
                             "their flags, which trailer runs do not push")
        NR = self.NC
        seq = np.zeros((1024, 8), dtype=np.uint64)
        seq[:, 0] = np.arange(1024, dtype=np.uint64)
        mems = sorted(set(self.mem.values()))
        self.h_seq = {m: st.put_hbm(f"{T}_ring_seq_m{m:02x}", seq.view(np.int8).reshape(-1),
                                    mem_chip=(m >> 4, m & 15)) for m in mems}
        self.h_flags = st.put_zeros(f"{T}_ring_flags", "uint8_t", NR * n_slots * 64)
        self.h_rel = st.put_zeros(f"{T}_ring_release", "uint8_t", NR * n_slots * 64)
        self.h_rec = rec_l3 if rec_l3 is not None else st.put_zeros(f"{T}_rec_l3", "uint8_t",
                                                                     128)
        # one symbol for every chip's slots (zero at boot, so a trailer poll never reads X)
        self.h_wslots = st.put_zeros(f"{T}_wring_slots", "uint8_t", NR * n_slots * self.slot_bytes,
                                     preload=True)
        self.rings, self.prefetch = {}, {}
        for k in self.chips:
            rings = self.rings[k] = WeightRings(
                slots=self.h_wslots, flags=self.h_flags, releases=self.h_rel, n_rings=NR,
                n_slots=n_slots, slot_bytes=self.slot_bytes, batch=self.ring_batch,
                batch_routed=int(wr.get("batch_routed", 0)), head=self.ring_head,
                batch_head=self.ring_batch_head)
            gs = self.gs(k)
            if self.crest:
                rings.crest = CrestRings(
                    crest_codec(self.args.dsv2_dir),
                    dense={r: dense[g] for r, g in enumerate(gs) if g in dense},
                    routed=(routed or {}).get(k, {}), slices=(slices or {}).get(k, {}),
                    slab_bytes=self.ls[gs[0]].slab_bytes, trailer=self.ring_trailer)
            if self.ring_trailer and not self.crest:
                rings.plain_trailer = True
            for g in gs:
                self.ls[g].rings = rings
            self.prefetch[k] = self.cx_chip(k).host_node(
                f"weight_prefetch_k{k:02x}", "__host_bingo_kernel_weight_prefetch",
                HostBingoKernelWeightPrefetchArgs(
                    BingoMemSymbol(f"{T}_ring_sched_k{k:02x}"), rings.slots, self.h_flags,
                    self.h_rel, self.h_seq[self.mem[k]], self.h_rec, rec_slots, NR, n_slots,
                    self.slot_bytes, self.mem[k], batch=self.ring_batch,
                    policy=int(wr.get("policy", 1)), engine=self.engine[k],
                    batch_routed=int(wr.get("batch_routed", 0)), queued=self.ring_queued,
                    head=self.ring_head, batch_head=self.ring_batch_head))

    # ---- the phases -----------------------------------------------------------------------
    def setup(self, extra=None):
        """Every symmetric buffer zeroed (no flag reads X), the barrier state too with barriers,
        the start anchor, and what each chip's window opens with: the start (edge) or the
        chip's first barrier, which every chip parks in before the root stamps the start and
        arrives last. `extra` {chip: [task]}: more of a chip's setup the start waits for.
        Returns {chip: go task}; the prefetchers follow it."""
        xdma = self.roles["xdma"]
        setup_end = {}
        # sym_zero False: a workload whose every read of a symmetric buffer follows a write of it in the
        # same run skips the zeroing, so the buffer is live only from its first writer on
        zero = getattr(self, "sym_zero", True)
        for k in self.chips:
            prev = None
            for c in range(self.NC if zero else 0):
                prev = self.node(k, c, xdma, f"sym_zero_c{c}", "__snax_bingo_kernel_xdma_memset",
                                 SnaxBingoKernelXdmaMemsetArgs(self.sym[(k, c)], self.sym_bytes,
                                                               SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO),
                                 prev)
            if self.sync == "barrier":
                prev = self.cx_chip(k).host_node(
                    f"barrier_init_k{k:02x}", "__host_bingo_kernel_chip_barrier",
                    HostBingoKernelChipBarrierArgs(HostBingoKernelChipBarrierArgs.OP_INIT), [prev])
            setup_end[k] = prev
        ends = [t for k in self.chips for t in [setup_end[k]] + list((extra or {}).get(k, [])) if t is not None]
        if self.sync == "edge":
            start = self.anchor("start", ends)
            go = {k: start for k in self.chips}
        else:
            ready = self.probe(self.ROOT, "ready", ends)
            armed = {k: self.probe(k, "armed", [ready]) for k in self.chips}
            start = self.anchor("start", list(armed.values()))
            go = self.barrier({k: [start] if k == self.ROOT else [armed[k]] for k in self.chips})
        for k in self.chips:
            self.edge(go[k], self.prefetch[k])
        self.go = go
        return go

    def broadcast(self, name, src, nbytes, off, off_flag, after, clusters=None, value=1,
                  src_ready=None):
        """The root's `src` (`nbytes`) at `off` of the symmetric buffer of every cluster in
        `clusters` (default all, by index): bcast_l1, one D2D multicast per cluster index
        (sequential: one multicast in flight) and a landed flag at `off_flag`, each cluster
        waiting on its own; pull, every cluster's DM core reading the root's copy. `after`
        {chip: task} orders it locally (the root's: the task that made src). `src_ready`: that
        task, when src is made inside the window -- a pull then waits for it (an edge, or with
        sync barrier one more barrier); a multicast's flag already says it. Returns
        {(chip, c): the task after which it is there}."""
        dm, R, NC = self.roles["dm"], self.ROOT, self.NC
        cls = range(NC) if clusters is None else clusters
        out = {}
        self.bcast_barrier = {}             # {chip: the barrier before the pulls}, if any
        if self.bcast_impl == "bcast_l1":
            puts = []
            for c in cls:
                put = self.node(R, 0, dm, f"{name}_bput_c{c}", "__snax_bingo_kernel_idma_put_flagged",
                                SnaxBingoKernelIdmaPutFlaggedArgs(
                                    src, self.sv(R, c, off), nbytes, self.sv(R, c, off_flag),
                                    value=value, bcast=True, also_here=True),
                                [after[R]] + puts[-1:])
                puts.append(put)
                self.bcast_puts = list(puts)
                for k in self.chips:
                    w = self.node(k, c, dm, f"{name}_wait_c{c}", "__snax_bingo_kernel_wait_flags",
                                  SnaxBingoKernelWaitFlagsArgs(self.sv(k, c, off_flag), 1, value),
                                  [after[k]] if k != R else [put])
                    if k != R:
                        self.dfg.bingo_add_order_edge(put, w)
                    out[(k, c)] = w
            return out
        deps = {k: [after[k]] for k in self.chips}
        if src_ready is not None:
            if self.sync == "barrier":
                bar = self.bcast_barrier = self.barrier(
                    {k: [after[k]] + ([src_ready] if k == R else []) for k in self.chips})
                deps = {k: [bar[k]] for k in self.chips}
            else:
                deps = {k: [after[k], src_ready] for k in self.chips}
        for k in self.chips:
            s = src if k == R else self.remote_of(src)
            for c in cls:
                out[(k, c)] = self.node(k, c, dm, f"{name}_pull_c{c}", "__snax_bingo_kernel_idma_1d_copy",
                                        SnaxBingoKernelIdma1dCopyArgs(s, self.sv(k, c, off), nbytes),
                                        list(dict.fromkeys(deps[k])))
        return out

    def remote_of(self, src):
        """The root's copy of `src` as another chip names it: a staged symbol by its chip, a
        root symmetric-buffer view by the root's address."""
        if isinstance(src, BingoMemSymbol):
            return BingoMemSymbol(src.symbol_name, offset=src.offset, chip_id=self.ROOT)
        h, off = (src.base, src.offset) if hasattr(src, "base") else (src, 0)
        return remote(self.ROOT, f"((uint64_t)({h.get_c_var_name()}) + {off})")

    def reduce(self, parts, xb):
        """`parts` {g: Port, FP16 [xb bytes] in cluster g's L1}: each chip's clusters added on its
        cluster 0 (into OFF_LOC of its symmetric buffer), then the chips' sums folded into the
        root's dst in the fold order: one ChainGather (reduce chain) or the root's pulls and
        SIMD adds (reduce root), in fold_order(). Sets self.reduce_end."""
        dm, xdma, simd = self.roles["dm"], self.roles["xdma"], self.roles["simd"]
        rows = xb // 64
        loc = {}
        for k in self.chips:
            gs = self.gs(k)
            ports = [parts[g] for g in gs]
            prev = None
            for i, pt in enumerate(ports[1:], start=1):
                prev = self.node(k, i, dm, f"part_push_c{i}", "__snax_bingo_kernel_idma_1d_copy",
                                 SnaxBingoKernelIdma1dCopyArgs(pt.handle, self.p1[k], xb),
                                 list(pt.ends))
            loc[k] = self.node(k, 0, simd, "loc_add", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdAddF16Args(ports[0].handle, self.p1[k],
                                                             self.sv(k, 0, self.OFF_LOC), rows, 32),
                               list(ports[0].ends) + ([prev] if prev is not None else []))
        self.loc_nodes = dict(loc)
        R, others = self.ROOT, self.order[1:]
        before = []
        if self.sync == "barrier":
            b = self.barrier({k: [loc[k]] for k in self.chips})
            loc = {k: b[R] for k in self.chips}
        if self.p.get("anchor_reduce", False):
            before = [self.anchor("reduce", list(dict.fromkeys(loc.values())))]
        if self.reduce_impl == "chain":
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                remote, R, 0, others, src=self.sx(0, self.OFF_LOC), dst=self.dst_expr(), size=xb,
                junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[self.dst_buf])
            gth = self.node(R, 0, xdma, "y_chain", "__snax_bingo_kernel_xdma_chain_gather", ga,
                            before or list(dict.fromkeys(loc[k] for k in self.chips)))
            self.reduce_end = [gth]
            return
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
                            SnaxBingoKernelSimdAddF16Args(a_in, slot, self.dst, rows, 32),
                            [cp, loc[R]] + adds[-1:])
            adds.append(add)
        self.reduce_end = [adds[-1]]

    def stash(self, name, k, c, src, nbytes, after):
        """A copy of chip k cluster c's `src` into an L3 symbol on chip k, after the end anchor
        and `after`: what the root's host compares. Returns the symbol as the root names it."""
        sym = f"{self.TAG}_chk_{name}"
        self.st.put_zeros(sym, "uint8_t", nbytes)
        cp = self.node(k, c, self.roles["dm"], f"chk_{name}", "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(src, BingoMemSymbol(sym), nbytes),
                       [self.end] + list(after))
        self.check_copies.append(cp)
        return BingoMemSymbol(sym, chip_id=k)

    def end_window(self, after):
        """The end anchor after `after`, and the root's report of the window."""
        self.end = self.anchor("end", after)
        self.report = self.cx_chip(self.ROOT).host_node(
            f"{self.TAG}_report", "__host_bingo_kernel_sync_report",
            HostBingoKernelSyncReportArgs(stamp_buf=self.stamps, num_phases=self.n_anchor - 1,
                                          local_phase=self.n_anchor - 1), [self.end])
        self.check_copies = []

    def run_checks(self, checks):
        """checks: (name, got symbol, golden array, bytes), compared byte for byte by the root's
        host, one at a time; then the hold that keeps every symmetric buffer live to the end."""
        g0 = self.cx_chip(self.ROOT)
        prev = [self.report] + self.check_copies
        for name, got, golden, nb in checks:
            a = np.ascontiguousarray(golden)
            gh = self.st.put(f"{self.TAG}_gold_{name}", "uint8_t", a.view(np.uint8).reshape(-1))
            prev = [vchecks.check_bytes(g0, f"Check_{self.TAG}_{name}", golden=gh, got=got,
                                        nbytes=nb, after=prev, label=f"{self.TAG}_{name}")]
        for (k, c), h in self.sym.items():
            self.node(k, c, self.roles["dm"], f"sym_hold_c{c}", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(h, 0, 0), prev if k == self.ROOT else [self.end])

    # ---- emission -------------------------------------------------------------------------
    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_")):
        dfg, p, st, args, T = self.dfg, self.p, self.st, self.args, self.TAG
        if self.crest:
            for k in self.chips:
                cr = self.rings[k].crest
                for kind, (u, n, r) in cr.finish(st.fill_hbm).items():
                    if u:
                        print(f"[{T}] chip {k:#04x}: CREST {kind} {u / 2**20:.2f} MiB pushed as "
                              f"{n / 2**20:.2f} MiB ({r:.3f}x)")
                print(f"[{T}] chip {k:#04x}: CREST {cr.plain} chunk(s) stored plain")
        for k in self.chips:
            r = self.rings[k]
            st.put(f"{T}_ring_sched_k{k:02x}", "uint64_t", np.array(r.schedule(), dtype=np.uint64))
            print(f"[{T}] chip {k:#04x}: {r.chunks()} chunks pushed by memory chip "
                  f"{self.mem[k]:#04x} engine {self.engine[k]}")
        if args.data_h:
            st.emit(args.data_h, args.output_dir)
            if p.get("d2d_ddr", True):
                with open(args.data_h, "a") as f:
                    f.write("\n// params d2d_ddr: every D2D link switched to DDR at boot\n"
                            "#define HEMAIA_D2D_DDR 1\n")
            # params d2d_rx_yield: [compute chips, memory chips], the data links' RX yield windows
            # (hemaia_d2d_link_rx_yield_grid; 0 = off): a side receiving a push stream hands the
            # half-duplex link over when it has its own requests waiting
            if "d2d_rx_yield" in p:
                wc, wm = (int(v) for v in p["d2d_rx_yield"])
                if not (0 <= wc < 256 and 0 <= wm < 256):
                    raise ValueError(f"params d2d_rx_yield={p['d2d_rx_yield']}: 0..255 each")
                with open(args.data_h, "a") as f:
                    f.write(f"\n// params d2d_rx_yield: RX yield windows, compute / memory chips\n"
                            f"#define HEMAIA_D2D_RX_YIELD {wc}\n#define HEMAIA_D2D_RX_YIELD_MEM {wm}\n")
        extra = [os.path.basename(str(args.data_h))] if args.data_h else None
        if self.barrier_fence:
            self.fence_barriers()
        dfg.skip_viz = bool(p.get("fast_export", True))
        dfg.bingo_compile_dfg(
            app_name=app_name, output_dir=args.output_dir,
            output_file_name=args.output_offload_file_name, extra_include_header_list=extra,
            static_l1=bool(p.get("static_l1", True)), desc_list_in_narrow_spm=True)
        self.check_symmetric()
        header = os.path.join(args.output_dir, args.output_offload_file_name)
        self.check_core_order(header, x_prefixes)
        self.check_barrier_order(header)
        print(f"[{T}] HBM {st.hbm_bytes / 2**20:.2f} MiB over {len(set(self.mem.values()))} "
              f"memory chips; {sum(1 for _ in dfg.nodes)} nodes")
        # params export_exec_ir: the compiled graph, rings and prefetch policy as the bingo
        # framework's Execution IR (its chiplet DSE prices the program from it;
        # passes/bingo_exec_export.py); a string names the file, else exec_ir.json in the output dir
        if p.get("export_exec_ir", False):
            from bingo_exec_export import bingo_export_exec_ir, ring_table
            wr = p.get("weight_ring", {})
            out = p["export_exec_ir"] if isinstance(p["export_exec_ir"], str) else \
                os.path.join(args.output_dir, "exec_ir.json")
            bingo_export_exec_ir(
                dfg, out,
                ring_tables=[ring_table(self.rings[k], k, self.mem[k], self.engine[k])
                             for k in self.chips],
                prefetch=dict(ring_slots=self.n_slots, batch=self.ring_batch,
                              batch_routed=int(wr.get("batch_routed", 0)), head=self.ring_head,
                              batch_head=self.ring_batch_head, policy=int(wr.get("policy", 1)),
                              queued=bool(self.ring_queued)),
                meta=dict(workload=T, app=app_name, chips=list(self.chips), nc=self.NC,
                          tokens=self.T,
                          params={k: v for k, v in p.items()
                                  if isinstance(v, (int, float, str, bool, list, dict))},
                          trailer=bool(self.ring_trailer), crest=bool(self.crest),
                          slot_bytes=self.slot_bytes))
            print(f"[{T}] Execution IR: {out}")

    def check_core_order(self, header, x_prefixes):
        """Every DM core reaches its cluster's x task before its first weight-ring load, in the
        order the hardware runs it: the descriptor list of the emitted header, which is not the
        node-id order. A ring load spins on a chunk the prefetcher pushes only after the start,
        and the start anchor shares the root's DM core: one ring load ahead of it is a deadlock
        no dependency check sees."""
        names = {str(n.node_id): n.node_name for n in self.dfg.node_list}
        ring = {str(n.node_id) for n in self.dfg.node_list
                if (n.kernel_name or "").endswith("idma_ring_load")}
        txt = open(header).read()
        seen_x, bad = set(), []
        dm = str(self.roles["dm"])
        for chip, nid, ty, cl, core in desc_rows(txt):
            if ty != "0" or core != dm:
                continue
            nm = names.get(nid, "")
            key = (chip, cl)
            if nm.startswith(tuple(x_prefixes)):
                seen_x.add(key)
            elif nid in ring and key not in seen_x:
                bad.append(f"chip {chip} cluster {cl}: {nm}")
        if bad:
            raise ValueError(f"a weight load runs before its cluster's x: {bad[:4]}")
        if len(seen_x) != len(self.chips) * self.NC:
            raise ValueError(f"the core-order check found the x task on {len(seen_x)} of "
                             f"{len(self.chips) * self.NC} DM cores (header format changed?)")

    def check_barrier_order(self, header):
        """Every chip's DM core runs barrier 1, 2, ... in that order (the descriptor order of
        the emitted header): a chip that took barrier n + 1 first would match the other
        chips' n-th arrival, and what barrier n + 1 orders would run early."""
        if not self.n_barrier:
            return
        names = {str(n.node_id): n.node_name for n in self.dfg.node_list}
        txt = open(header).read()
        seen = {}
        for chip, nid, ty, cl, core in desc_rows(txt):
            m = re.match(r"barrier(\d+)_k", names.get(nid, ""))
            if ty == "0" and m:
                seen.setdefault(chip, []).append(int(m.group(1)))
        want = list(range(1, self.n_barrier + 1))
        bad = {c: v for c, v in seen.items() if v != want}
        if bad or len(seen) != len(self.chips):
            raise ValueError(f"barriers out of program order on {len(bad)} chip(s): "
                             f"{dict(list(bad.items())[:3])}; {len(seen)} chips seen")

    def check_symmetric(self):
        """Every cluster's symmetric buffer at offset 0 of its static arena."""
        pl = getattr(self.dfg, "static_l1_placement", None)
        if not pl:
            raise ValueError("the level-2 workloads need static_l1: the symmetric buffers' "
                             "addresses")
        bad = [(k, c, pl[id(h)][1]) for (k, c), h in self.sym.items()
               if id(h) not in pl or pl[id(h)][1] != 0]
        if bad:
            raise ValueError(f"symmetric buffers not at offset 0: {bad[:4]}")


L1, L3, HBM = MemLevel.L1, MemLevel.L3, MemLevel.HBM
