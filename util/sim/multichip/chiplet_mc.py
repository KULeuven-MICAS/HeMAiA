# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""What the level-2 workloads share on the 16 + 4 chiplet platform: the topology, the weight rings,
the stamps, the synchronisation arm, the broadcast of the token, the chips' fold and the reduce,
the checks and the emission. The workloads (ffn, moe, fa, and the llama3 and dsv2 layers) build
their layer between `setup()` and `finish()`. It sits next to fabric.py (FabricBase, its base); a
caller puts util/sim/multichip and mini_compiler on sys.path before importing it:

    S = Workload(args, p, plat, hw)     # its data, S.make_xch(...), S.make_rings(...)
    go = S.setup()                      # the setup, the start stamp, the first barrier
    xin = S.broadcast(...)              # the token into every cluster
    ...                                 # the layer: per cluster a partial over every output
    S.reduce(parts)                     # the chips' folds and the reduce into the root's dst
    S.finish(...)                       # end stamp, report, checks, compile

THE SWITCHES (params): bcast (bcast_l1 | pull), reduce (chain | root), sync (edge | barrier) and
barrier_mode (bcast: one 0xFF broadcast store of the chip's checkpoint, then a local poll | sw: the
point-to-point software barrier through the top-left chip); the ablations turn the first three on and
off (ffn/common/ffn_prefill_mc.py, THE ABLATION, its own reduce root and barrier placement).
"""

import os
import re
import sys

import numpy as np

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
    HostBingoKernelChipBarrierArgs, HostBingoKernelStampReportArgs,
    HostBingoKernelWeightPrefetchArgs, SnaxBingoKernelChipBarrierArgs,
    SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelWaitFlagsArgs, SnaxBingoKernelXdmaChainGatherArgs,
    SnaxBingoKernelXdmaMemsetArgs, XDMA_EWJCT_FMT_FP16, XDMA_EWJCT_OP_ADD, xdma_ewjct_csr0)
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_platform import core_roles, writer_junction_index  # noqa: E402
from libs import Ctx, MemLevel, Pipeline  # noqa: E402
from libs.blocks import LoadStream, WeightRings  # noqa: E402
from libs.verify import checks as vchecks  # noqa: E402

# the shared mesh, address and fold helpers, re-exported for the workloads that import them from here
from fabric import (L1_CAPACITY, MASK40, MESH, FabricBase, dist, fold_cycle,  # noqa: E402,F401
                    remote)

# params weight_ring.trailer: the 64-bit magic a chunk's last beat holds once it has landed (the device's
# BINGO_RING_TRAILER_MAGIC, offload_hw_kernels/idma.h)
TRAILER_MAGIC = 0x5EED7A11C0DEF1A6
EW_F16_ADD = xdma_ewjct_csr0(XDMA_EWJCT_OP_ADD, XDMA_EWJCT_FMT_FP16)
PAIR = 32                          # an INT4 GEMV's column unit


class ClusterNodes:
    """The node factory the mini_compiler blocks take (fa_sweep): chip k cluster c
    of a ChipletBuild, creating its nodes with the build's names."""

    def __init__(self, fb, k, c):
        self.fb, self.chiplet, self.cluster, self.dfg = fb, k, c, fb.dfg
        self.gemm, self.dm = fb.roles["gemm"], fb.roles["dm"]

    def node(self, name, core, kernel, args, after=()):
        return self.fb.node(self.chiplet, self.cluster, core, name, kernel, args, after)


class AliasStaging:
    """The layer's DataStaging, with names that are one region: put_zeros of an aliased name stages nothing and
    returns its symbol; finish() stages the region at the largest extent asked and lines() are the defines that make
    each name the region. An alias is a region's name, or (region, offset): the name that many bytes into it."""

    def __init__(self, st, aliases):
        self._st, self.asked = st, {}
        self.aliases = {n: (a if isinstance(a, tuple) else (a, 0)) for n, a in aliases.items()}

    def __getattr__(self, name):
        return getattr(self._st, name)

    def put_zeros(self, name, ctype, count, mem_chip=None, preload=False):
        if name not in self.aliases:
            return self._st.put_zeros(name, ctype, count, mem_chip=mem_chip, preload=preload)
        width = {"uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2, "uint32_t": 4, "int32_t": 4}[ctype]
        self.asked[name] = count * width
        return BingoMemSymbol(name)

    def finish(self):
        for region in sorted({r for r, _ in self.aliases.values()}):
            ext = [o + self.asked[n] for n, (r, o) in self.aliases.items() if r == region and n in self.asked]
            if ext:
                self._st.put_zeros(region, "uint8_t", max(ext), preload=True)

    def lines(self):
        out = ["// one L3 region, several names (chiplet_mc.py AliasStaging)"]
        out += [f"#define {n} {r}" if not o else f"#define {n} ((uint8_t *){r} + {o})"
                for n, (r, o) in sorted(self.aliases.items()) if n in self.asked]
        return out


class ChipletBuild(FabricBase):
    TAG = "l2"                          # the prefix of every staged symbol and printed line

    def __init__(self, args, p, plat, hw, shared=None):
        """`shared`: another build whose graph, staging, stamps and node helpers this one adds to (a block
        of a whole layer, dsv2_layer_prefill_mc.py); None makes its own."""
        self.args, self.p, self.plat, self.hw = args, p, plat, hw
        self.T = int(p.get("tokens", 1))
        if self.T < 1:
            raise ValueError(f"params tokens={self.T}: at least one")
        self.wbits = int(p.get("wbits", 4))
        self.chunk = int(p.get("w_chunk_bytes", 112 * 1024))
        self.slot_bytes = self.chunk
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
        # params chip_rows: only the compute chips of the package's rows y = 0 .. chip_rows - 1 take part (each row with its
        # memory chip); the others are in no task, and their hosts return at once (bingo_dfg_emit idle_chiplet_ids)
        self.idle_chips = []
        if p.get("chip_rows") is not None:
            rows = int(p["chip_rows"])
            if not 1 <= rows <= 1 + max(k & 15 for k in chips):
                raise ValueError(f"params chip_rows={rows}: 1 to {1 + max(k & 15 for k in chips)}")
            self.idle_chips = [k for k in chips if (k & 15) >= rows]
            chips = self.chips = [k for k in chips if (k & 15) < rows]
        NC = self.NC = int(plat["num_clusters_per_chiplet"])
        if NC != 2:
            raise ValueError(f"{NC} clusters per chip: the local fold adds two")
        self.N_CL = len(chips) * NC
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
        self.xch = {}
        self.checks = []                       # (name, got handle, golden array, bytes)
        if shared is not None:
            for a in ("st", "dfg", "roles", "ctx", "pipe", "jct", "stamps", "mark", "ring_scheds"):
                setattr(self, a, getattr(shared, a))
            self.n_stamp = 0
            return
        self.st = DataStaging(plat, on_host=True)
        self.ring_scheds = {}                  # schedule symbol: (its WeightRings, TAG, chip), ring_schedule()
        if not self.st.has_hbm:
            raise ValueError(f"{args.platformcfg} has no HBM")
        dfg = self.dfg = BingoDFG(num_chiplets=len(chips), num_clusters_per_chiplet=NC,
                                  num_cores_per_cluster=plat["num_cores_per_cluster"],
                                  is_host_as_acc=True, chiplet_ids=chips,
                                  dep_tag_width=plat["dep_tag_width"])
        dfg.idle_chiplet_ids = list(self.idle_chips)
        # the static L1 plan's capacity a cluster: what the cluster's L1 heap leaves beside everything else it keeps
        # there once the schedule has fixed its tasks (passes/bingo_dfg_schedule.py _l1_cap); params l1_capacity: a
        # fixed one instead
        dfg.l1_capacity_auto = p.get("l1_capacity", "auto") == "auto"
        dfg.l1_capacity_bytes = L1_CAPACITY if dfg.l1_capacity_auto else int(p["l1_capacity"])
        dfg.l1_tcdm_bytes = int(hw["cluster"]["tcdm"]["size"]) * 1024     # the L1 heap's TCDM (bingo_l1_heap.py)
        dfg.waiting_queue_depth = plat.get("waiting_queue_depth", 8)
        dfg.compact_task_tables = bool(p.get("compact_tables", True))
        # each chiplet's id maps hold its own entries only (bingo_g2l16_build, bingo_hw_scheduler_sparse)
        dfg.compact_id_maps = bool(p.get("compact_id_maps", True))
        # each cluster's L1 holds its own tasks' global ids instead of a table of every task's
        # (bingo_hw_scheduler_init_gid_lists): its L1 cost does not grow with the whole graph
        dfg.gid_lists = bool(p.get("gid_lists", True))
        # each cluster's per-task list entries follow its args phases through two L1 windows (bingo_dfg_emit.py,
        # PAGED TASK LISTS), so they too stop growing with the whole graph; needs gid_lists and args phases (the device
        # runtime reads them: bingo.h, bingo_hw_offload_manager_paged)
        dfg.list_phases = bool(p.get("list_phases", False))
        # params per_chip_image: one host image per chiplet, holding the shared code and data and that chiplet's
        # own tables only (target/sw/host/apps/per_chip_image.py), so an image's L3 share does not grow with the
        # other chiplets' tasks; needs the one-function-per-chiplet host code (split_chip_functions)
        dfg.per_chip_image = bool(p.get("per_chip_image", False))
        dfg.static_args_image = bool(p.get("static_args_image", True))
        # the generator's faster host init (bingo_dfg_emit: L1 addresses at compile time, the kernel list
        # by one loop, the symbol lookup by signature, the g2l table runs by DMA)
        dfg.fast_host_init = bool(p.get("fast_host_init", False))
        dfg.remote_broadcast = bool(p.get("remote_broadcast", True))
        # the compiler orders every device core's program, and runs each accelerator's tasks as configure / start /
        # wait parts where its model says the program finishes sooner (passes/bingo_dfg_schedule.py); params
        # async_exec false keeps every task whole (the synchronous arm of an A/B)
        dfg.schedule_cores = True
        dfg.async_exec = bool(p.get("async_exec", True))
        self.roles = core_roles(plat)
        self.ctx = Ctx(dfg=dfg, mesh=MESH, roles=self.roles, hw=hw, chiplet=self.ROOT)
        self.pipe = Pipeline(self.ctx, verbose=False, gate_sources=True)
        self.jct = {n: writer_junction_index(hw, f"Has{n}Junction")
                    for n in ("Elementwise", "TopK", "SoftmaxStats")}
        self.n_stamp = 0
        self.stamps = BingoMemAlloc(f"A_{self.TAG}_stamps", size=4 * 16, mem_level="L2",
                                    chip_id=self.ROOT, cluster_id=0)
        self.mark = BingoMemAlloc(f"zz_{self.TAG}_sram_mark", size=64, mem_level="L1",
                                  chip_id=self.ROOT, cluster_id=0)

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
        """{cl: (c0, n)}: `inter` columns over `clusters` (default every cluster) in whole units (default
        pairs), the units left over after an even split going one per link in turn (a link carries the
        weights of every chip behind it), each to the nearest chip's first cluster first; the
        clusters in `last` get a leftover unit only after every other cluster has one. `fixed`
        {cl: columns}: those clusters get exactly that many, the rest is split over the others."""
        clusters = list(range(self.N_CL)) if clusters is None else list(clusters)
        fixed = dict(fixed or {})
        if inter % unit or any(v % unit for v in fixed.values()):
            raise ValueError(f"{inter} intermediate columns are not whole {unit}-column units")
        if fixed:
            rest = self.split_columns(inter - sum(fixed.values()), [cl for cl in clusters if cl not in fixed],
                                      last, unit)
            n = {cl: (fixed[cl] if cl in fixed else rest[cl][1]) for cl in clusters}
            out, c0 = {}, 0
            for cl in sorted(clusters):
                out[cl] = (c0, n[cl])
                c0 += n[cl]
            return out
        units, n_cl = inter // unit, len(clusters)
        base, extra = divmod(units, n_cl)
        links = sorted(set(self.link[self.chip(cl)] for cl in clusters))
        per_link = {L: sorted({self.chip(cl) for cl in clusters if self.link[self.chip(cl)] == L},
                              key=lambda k: (dist(k, self.mem[k]), k)) for L in links}
        order = []
        for c in range(self.NC):
            for i in range(max(len(v) for v in per_link.values())):
                for L in links:
                    if i < len(per_link[L]):
                        cl = self.chips.index(per_link[L][i]) * self.NC + c
                        if cl in clusters:
                            order.append(cl)
        order = [cl for cl in order if cl not in last] + [cl for cl in order if cl in last]
        n = {cl: unit * (base + (1 if cl in order[:extra] else 0)) for cl in clusters}
        out, c0 = {}, 0
        for cl in sorted(clusters):
            out[cl] = (c0, n[cl])
            c0 += n[cl]
        return out

    # ---- nodes ----------------------------------------------------------------------------
    def at(self, k, c):
        """Chip k's cluster c as the node factory a mini_compiler block takes."""
        return ClusterNodes(self, k, c)

    def barrier(self, after):
        """One chip barrier over every compute chip: a task on each chip's cluster 0 DM core (or the
        core of role self.barrier_role, if a build sets it: every barrier of a chip on one core),
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
            out[k] = self.node(k, 0, self.roles[getattr(self, "barrier_role", "dm")], f"barrier{self.n_barrier}",
                               "__snax_bingo_kernel_chip_barrier",
                               SnaxBingoKernelChipBarrierArgs(self.barrier_mode, tl, br,
                                                              seq=self.n_barrier),
                               list(dict.fromkeys(deps)))
        self.last_barrier = out
        self.barrier_rounds.append(out)
        # the graph's every round, whichever build of it made them: the hazard check's order across the chips
        if not hasattr(self.dfg, "barrier_rounds"):
            self.dfg.barrier_rounds = []
        self.dfg.barrier_rounds.append(out)
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
    def make_xch(self, nbytes, xb, dst_lead=0):
        """Every cluster's exchange buffer (`nbytes`), and the buffers the reduce needs where
        no fixed address is: cluster 1's partial handed to cluster 0, the root's dst (the
        folded output) and its two pull slots, `xb` bytes each. `dst_lead`: bytes kept below
        dst in its buffer (a SIMD kernel's latch beat, for a reader of dst)."""
        self.xch_bytes = int(nbytes)
        # "AAAA_" sorts before every other handle: offset 0 of each cluster's static arena
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.xch_bytes)
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
        # stream_name: the slabs' handle names (a layer's blocks each name their own)
        self.ls = [LoadStream(self.cx(cl), self.local(cl), nbytes=self.chunk, nbuf=wbuf,
                              name=getattr(self, "stream_name", "stream"))
                   for cl in range(self.N_CL)]

    @staticmethod
    def chunk_trailer():
        """The 64-B record a plain weight chunk ends in with weight_ring.trailer: the magic the
        ring load polls for (idma_ring_load), the rest zero."""
        t = np.zeros(64, dtype=np.int8)
        t[:8] = np.array([TRAILER_MAGIC], dtype="<u8").view(np.int8)
        return t

    def ring_schedule(self, k, rings):
        """The symbol of chip k's ring schedule, the table its prefetcher walks
        (WeightRings.schedule), for the prefetcher task made with `rings`. The rings are
        recorded in ring_scheds, which a layer's blocks share, and finish() stages every
        recorded table once the build has taken all its chunks."""
        name = f"{self.TAG}_ring_sched_k{k:02x}"
        self.ring_scheds[name] = (rings, self.TAG, k)
        return BingoMemSymbol(name)

    def make_rings(self, rec_l3=None, rec_slots=1):
        """Per chip: its clusters' weight rings in its L3, filled by its memory chip's push
        engine under its host's prefetcher (host_kernel_lib.h weight_prefetch); rec_l3 the L3
        record each chip's prefetcher waits for (the routed runs)."""
        p, st, T = self.p, self.st, self.TAG
        wr = p.get("weight_ring", {})
        self.n_slots = n_slots = int(wr.get("slots", 16))
        self.ring_batch = int(wr.get("batch", 4))
        # weight_ring.trailer: a chunk's own last 64 B (a record the workload appends to every chunk,
        # chunk_trailer()) say it has landed, so the prefetcher pushes no flag after it
        self.ring_trailer = bool(wr.get("trailer", False))
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
            cl_k = self.clusters(k)
            if self.ring_trailer:
                rings.plain_trailer = True
            for cl in cl_k:
                self.ls[cl].rings = rings
            self.prefetch[k] = self.cx_chip(k).host_node(
                f"weight_prefetch_k{k:02x}", "__host_bingo_kernel_weight_prefetch",
                HostBingoKernelWeightPrefetchArgs(
                    self.ring_schedule(k, rings), rings.slots, self.h_flags,
                    self.h_rel, self.h_seq[self.mem[k]], self.h_rec, rec_slots, NR, n_slots,
                    self.slot_bytes, self.mem[k], batch=self.ring_batch,
                    policy=int(wr.get("policy", 1)), engine=self.engine[k],
                    batch_routed=int(wr.get("batch_routed", 0)), queued=self.ring_queued,
                    head=self.ring_head, batch_head=self.ring_batch_head))

    # ---- the phases -----------------------------------------------------------------------
    def setup(self, extra=None):
        """Every exchange buffer zeroed (no flag reads X), the barrier state too with barriers,
        the start stamp, and what each chip's window opens with: the start (edge) or the
        chip's first barrier, which every chip parks in before the root stamps the start and
        arrives last. `extra` {chip: [task]}: more of a chip's setup the start waits for.
        Returns {chip: go task}; the prefetchers follow it."""
        xdma = self.roles["xdma"]
        setup_end = {}
        # xch_zero False: a workload whose every read of an exchange buffer follows a write of it in the
        # same run skips the zeroing, so the buffer is live only from its first writer on
        zero = getattr(self, "xch_zero", True)
        for k in self.chips:
            prev = None
            for c in range(self.NC if zero else 0):
                prev = self.node(k, c, xdma, f"xch_zero_c{c}", "__snax_bingo_kernel_xdma_memset",
                                 SnaxBingoKernelXdmaMemsetArgs(self.xch[(k, c)], self.xch_bytes,
                                                               SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO),
                                 prev)
            if self.sync == "barrier":
                prev = self.cx_chip(k).host_node(
                    f"barrier_init_k{k:02x}", "__host_bingo_kernel_chip_barrier",
                    HostBingoKernelChipBarrierArgs(HostBingoKernelChipBarrierArgs.OP_INIT), [prev])
            setup_end[k] = prev
        ends = [t for k in self.chips for t in [setup_end[k]] + list((extra or {}).get(k, [])) if t is not None]
        if self.sync == "edge":
            start = self.stamp("start", ends)
            go = {k: start for k in self.chips}
        else:
            ready = self.probe(self.ROOT, "ready", ends)
            armed = {k: self.probe(k, "armed", [ready]) for k in self.chips}
            start = self.stamp("start", list(armed.values()))
            go = self.barrier({k: [start] if k == self.ROOT else [armed[k]] for k in self.chips})
        for k in self.chips:
            self.edge(go[k], self.prefetch[k])
        self.go = go
        return go

    def broadcast(self, name, src, nbytes, off, off_flag, after, clusters=None, value=1,
                  src_ready=None):
        """The root's `src` (`nbytes`) at `off` of the exchange buffer of every cluster in
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
                                    src, self.xch_view(R, c, off), nbytes, self.xch_view(R, c, off_flag),
                                    value=value, bcast=True, also_here=True),
                                [after[R]] + puts[-1:])
                puts.append(put)
                self.bcast_puts = list(puts)
                for k in self.chips:
                    w = self.node(k, c, dm, f"{name}_wait_c{c}", "__snax_bingo_kernel_wait_flags",
                                  SnaxBingoKernelWaitFlagsArgs(self.xch_view(k, c, off_flag), 1, value),
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
                                        SnaxBingoKernelIdma1dCopyArgs(s, self.xch_view(k, c, off), nbytes),
                                        list(dict.fromkeys(deps[k])))
        return out

    def remote_of(self, src):
        """The root's copy of `src` as another chip names it: a staged symbol by its chip, a
        root exchange-buffer view by the root's address."""
        if isinstance(src, BingoMemSymbol):
            return BingoMemSymbol(src.symbol_name, offset=src.offset, chip_id=self.ROOT)
        h, off = (src.base, src.offset) if hasattr(src, "base") else (src, 0)
        return remote(self.ROOT, f"((uint64_t)({h.get_c_var_name()}) + {off})")

    def reduce(self, parts, xb):
        """`parts` {cl: Port, FP16 [xb bytes] in cluster cl's L1}: each chip's clusters added on its
        cluster 0 (into OFF_LOC of its exchange buffer), then the chips' sums folded into the
        root's dst in the fold order: one ChainGather (reduce chain) or the root's pulls and
        SIMD adds (reduce root), in fold_order(). Sets self.reduce_end."""
        dm, xdma, simd = self.roles["dm"], self.roles["xdma"], self.roles["simd"]
        rows = xb // 64
        loc = {}
        for k in self.chips:
            cl_k = self.clusters(k)
            ports = [parts[cl] for cl in cl_k]
            prev = None
            for i, pt in enumerate(ports[1:], start=1):
                prev = self.node(k, i, dm, f"part_push_c{i}", "__snax_bingo_kernel_idma_1d_copy",
                                 SnaxBingoKernelIdma1dCopyArgs(pt.handle, self.p1[k], xb),
                                 list(pt.ends))
            loc[k] = self.node(k, 0, simd, "loc_add", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdAddF16Args(ports[0].handle, self.p1[k],
                                                             self.xch_view(k, 0, self.OFF_LOC), rows, 32),
                               list(ports[0].ends) + ([prev] if prev is not None else []))
        self.loc_nodes = dict(loc)
        R, others = self.ROOT, self.order[1:]
        if self.sync == "barrier":
            b = self.barrier({k: [loc[k]] for k in self.chips})
            loc = {k: b[R] for k in self.chips}
        if self.reduce_impl == "chain":
            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                remote, R, 0, others, src=self.xch_addr(0, self.OFF_LOC), dst=self.dst_expr(), size=xb,
                junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[self.dst_buf])
            gth = self.node(R, 0, xdma, "y_chain", "__snax_bingo_kernel_xdma_chain_gather", ga,
                            list(dict.fromkeys(loc[k] for k in self.chips)))
            self.reduce_end = [gth]
            return
        pulls, adds = [], []
        for i, k in enumerate(others):
            slot = self.slots.view((i % 2) * xb)
            cp = self.node(R, 0, dm, f"y_pull{i}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(remote(k, self.xch_addr(0, self.OFF_LOC)),
                                                         slot, xb),
                           [loc[k]] + pulls[-1:] + ([adds[i - 2]] if i >= 2 else []))
            pulls.append(cp)
            a_in = self.xch_view(R, 0, self.OFF_LOC) if i == 0 else self.dst
            add = self.node(R, 0, simd, f"y_add{i}", "__snax_bingo_kernel_simd_stream_elementwise",
                            SnaxBingoKernelSimdAddF16Args(a_in, slot, self.dst, rows, 32),
                            [cp, loc[R]] + adds[-1:])
            adds.append(add)
        self.reduce_end = [adds[-1]]

    def stash(self, name, k, c, src, nbytes, after):
        """A copy of chip k cluster c's `src` into an L3 symbol on chip k, after the end stamp
        and `after`: what the root's host compares. Returns the symbol as the root names it."""
        sym = f"{self.TAG}_chk_{name}"
        self.st.put_zeros(sym, "uint8_t", nbytes)
        cp = self.node(k, c, self.roles["dm"], f"chk_{name}", "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(src, BingoMemSymbol(sym), nbytes),
                       [self.end] + list(after))
        self.check_copies.append(cp)
        return BingoMemSymbol(sym, chip_id=k)

    def end_window(self, after):
        """The end stamp after `after`, and the root's report of the window."""
        self.end = self.stamp("end", after)
        self.report = self.cx_chip(self.ROOT).host_node(
            f"{self.TAG}_report", "__host_bingo_kernel_stamp_report",
            HostBingoKernelStampReportArgs(stamp_buf=self.stamps, num_phases=self.n_stamp - 1,
                                          local_phase=self.n_stamp - 1), [self.end])
        self.check_copies = []

    def run_checks(self, checks):
        """checks: (name, got symbol, golden array, bytes), compared byte for byte by the root's
        host, one at a time; then the hold that keeps every exchange buffer live to the end."""
        rcx = self.cx_chip(self.ROOT)
        prev = [self.report] + self.check_copies
        for name, got, golden, nb in checks:
            a = np.ascontiguousarray(golden)
            gh = self.st.put(f"{self.TAG}_gold_{name}", "uint8_t", a.view(np.uint8).reshape(-1))
            prev = [vchecks.check_bytes(rcx, f"Check_{self.TAG}_{name}", golden=gh, got=got,
                                        nbytes=nb, after=prev, label=f"{self.TAG}_{name}")]
        for (k, c), h in self.xch.items():
            self.node(k, c, self.roles["dm"], f"xch_hold_c{c}", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(h, 0, 0), prev if k == self.ROOT else [self.end])

    # ---- emission -------------------------------------------------------------------------
    def _stage_ring_schedules(self):
        """Every ring schedule recorded by ring_schedule() into the data image."""
        for name, (r, tag, k) in self.ring_scheds.items():
            self.st.put(name, "uint64_t", np.array(r.schedule(), dtype=np.uint64))
            print(f"[{tag}] chip {k:#04x}: {r.chunks()} chunks pushed by memory chip "
                  f"{self.mem[k]:#04x} engine {self.engine[k]}")

    def report_ring_races(self):
        """The weight rings' overwrite races (WeightRings.races) under every prefetcher task of the
        graph, its parts' chunk ranges together: the runs the prefetcher's byte guard holds back."""
        pfs = {}
        for n in self.dfg.nodes:
            a = n.kernel_args
            if isinstance(a, HostBingoKernelWeightPrefetchArgs):
                pf = pfs.setdefault(a.sched_addr.symbol_name, (a, set()))
                pf[1].add((getattr(a, "chunk_lo", 0), getattr(a, "chunk_hi", 0)))
        if not pfs:
            return
        total = 0
        for name, (a, parts) in pfs.items():
            races = self.ring_scheds[name][0].races(a, sorted(parts))
            total += len(races)
            if races:
                print(f"[{self.TAG}] {name}: {len(races)} ring races (ring, unread chunk, pushed chunk, "
                      f"bytes of it covered) {races[:4]}")
        print(f"[{self.TAG}] ring races (runs the prefetcher's byte guard holds back): {total}")

    def data_h_extra(self):
        """More lines for the end of the data header."""
        return []

    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_")):
        dfg, p, st, args, T = self.dfg, self.p, self.st, self.args, self.TAG
        self._stage_ring_schedules()
        self.report_ring_races()
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
            more = self.data_h_extra()
            if more:
                with open(args.data_h, "a") as f:
                    f.write("\n" + "\n".join(more) + "\n")
        extra = [os.path.basename(str(args.data_h))] if args.data_h else None
        if self.barrier_fence:
            self.fence_barriers()
        dfg.skip_viz = bool(p.get("fast_export", True))
        # params desc_in_image: each chip's scheduler fetches its task descriptors from the packed list in the
        # program image (L3) instead of a copy in the narrow SPM, which holds about 7,500 descriptors
        # params l1_pin_last: name substrings of L1 buffers the static packer places last (a long-lived buffer that
        # would otherwise split the space the others pack into); l1_order: the packer's preference order (name,
        # size, search: the lowest peak of several largest-first orders)
        static_l1 = bool(p.get("static_l1", True))
        if static_l1 and (p.get("l1_pin_last") or p.get("l1_order")):
            from bingo_l1_packer import StaticL1Options
            static_l1 = StaticL1Options(enable=True, pin_last=tuple(p.get("l1_pin_last", ())),
                                        order=p.get("l1_order", "name"))
        # each cluster's device args in phases, two alternating L1 regions (passes/bingo_args_phases.py), where they do
        # not fit beside everything else the cluster keeps in L1: the compiler sizes each cluster's phases once it has
        # fixed each core's program (params args_phase_bytes: one size for every cluster instead)
        if static_l1 and not getattr(dfg, "args_pingpong", False):
            dm = self.roles["dm"]
            phase = p.get("args_phase_bytes", "auto")
            dfg.args_phases = dict(
                budget=None if phase == "auto" else int(phase), dm=dm,
                dev_cores=[self.roles[r] for r in ("gemm", "simd", "xdma", "dm")],
                node=lambda k, c, name, kernel, kargs, deps: self.node(k, c, dm, name, kernel, kargs, deps))
        dfg.bingo_compile_dfg(
            app_name=app_name, output_dir=args.output_dir,
            output_file_name=args.output_offload_file_name, extra_include_header_list=extra,
            static_l1=static_l1, desc_list_in_narrow_spm=True,
            desc_list_in_image=bool(p.get("desc_in_image", False)))
        self.check_xch_offsets()
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
                          trailer=bool(self.ring_trailer), slot_bytes=self.slot_bytes))
            print(f"[{T}] Execution IR: {out}")
        # params replay_cluster: [chip (hex string or int), cluster]: that cluster's tasks inside the window, with their
        # arguments, as a standalone program for the SNAX cluster simulator (passes/bingo_cluster_replay.py), into
        # replay_<chip>_<cluster>.h in the output dir
        if p.get("replay_cluster") is not None:
            from bingo_cluster_replay import emit_cluster_replay
            rc = p["replay_cluster"]
            chip = int(rc[0], 16) if isinstance(rc[0], str) else int(rc[0])
            # params replay_gates: [[task regex, anchor regex, cycles], ...], the external waits as delays after the
            # cluster's own anchor tasks; replay_durations: [[task regex, cycles], ...], tasks held to at least that long
            # (bingo_cluster_replay.py); replay_pre: the cluster's setup tasks before the window run too
            emit_cluster_replay(dfg, chip, int(rc[1]), os.path.join(args.output_dir, "static_l1_layout.csv"),
                                os.path.join(args.output_dir, f"replay_{chip:02x}_{int(rc[1])}.h"),
                                gates=p.get("replay_gates"), durations=p.get("replay_durations"),
                                include_pre=bool(p.get("replay_pre", False)))

    def check_core_order(self, header, x_prefixes=()):
        """No DM core runs a weight-ring load ahead of a task the prefetcher that pushes the load's chunk waits for, in
        the order the hardware runs them: the descriptor list of the emitted header, which is not the node-id order. A
        ring load spins on a chunk the prefetcher pushes only after it starts: one ring load ahead of a task that
        prefetcher (through the start) waits for, on the same core, is a deadlock no dependency check sees. A load's
        prefetcher: the one filling its ring's flags whose chunk range holds the load's chunk. (x_prefixes: unused, kept
        for the drivers that pass it.)"""
        import networkx as nx
        g = nx.DiGraph(self.dfg)
        g.add_edges_from(self.dfg.order_edges)

        def flags(a):
            f = getattr(a, "flag_addr", None) if a is not None else None
            return getattr(getattr(f, "base", f), "symbol_name", None) or getattr(f, "name", None)
        pfs, anc = {}, {}
        for n in self.dfg.node_list:
            if isinstance(n.kernel_args, HostBingoKernelWeightPrefetchArgs):
                pfs.setdefault(n.assigned_chiplet_id, []).append(n)
                anc[n] = nx.ancestors(g, n)

        def pushers(ld):
            out = []
            for p in pfs.get(ld.assigned_chiplet_id, []):
                a = p.kernel_args
                if flags(a) is not None and flags(a) != flags(ld.kernel_args):
                    continue
                chunk = getattr(ld, "exec_chunk", None)
                lo, hi = getattr(a, "chunk_lo", 0), getattr(a, "chunk_hi", 0)
                if chunk is None or (lo <= chunk[1] and (not hi or chunk[1] < hi)):
                    out.append(p)
            return out or pfs.get(ld.assigned_chiplet_id, [])[:1]
        by_id = {str(n.node_id): n for n in self.dfg.node_list}
        txt = open(header).read()
        loaded, bad = {}, []
        dm = str(self.roles["dm"])
        for chip, nid, ty, cl, core in desc_rows(txt):
            if ty != "0" or core != dm:
                continue
            n = by_id.get(nid)
            if n is None:
                continue
            seen = loaded.setdefault((chip, cl), set())
            if (n.kernel_name or "").endswith("idma_ring_load"):
                seen.update(pushers(n))
            elif any(n in anc[p] for p in seen):
                bad.append(f"chip {chip} cluster {cl}: {n.node_name}")
        if bad:
            raise ValueError(f"a weight-ring load runs ahead of a task its prefetcher waits for: {bad[:4]}")

    def check_barrier_order(self, header):
        """Every chip's barrier core runs barrier 1, 2, ... in that order (the descriptor order of
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

    def check_xch_offsets(self):
        """Every cluster's exchange buffer at offset 0 of its static arena."""
        pl = getattr(self.dfg, "static_l1_placement", None)
        if not pl:
            raise ValueError("the level-2 workloads need static_l1: the exchange buffers' "
                             "addresses")
        bad = [(k, c, pl[id(h)][1]) for (k, c), h in self.xch.items()
               if id(h) not in pl or pl[id(h)][1] != 0]
        if bad:
            raise ValueError(f"exchange buffers not at offset 0: {bad[:4]}")


L1, L3, HBM = MemLevel.L1, MemLevel.L3, MemLevel.HBM
