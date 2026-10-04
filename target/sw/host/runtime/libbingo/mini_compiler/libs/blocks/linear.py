# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""A projection: an INT8 GEMM (or a one-token GEMV) with its operand loads, as a block.

Every weight matrix in a transformer layer -- Wq, Wk, Wv, Wo, and each expert's up/gate/
down -- is this block with different shapes. It exists so a layer states `Linear(tokens,
d_in, d_out)` rather than restating the M/K/N tiling, the load pair and the C-address rule
four times over.

WHAT IT PRODUCES. FP16, because the narrowing rides the GEMM's own D port: an INT32 beat
carries half as many values as an FP16 one, so converting at the consumer would double both
the beats it reads and the L1 the tile occupies. `d_shift` is that port's power-of-two
scale, RNE(acc * 2^-k): a K-deep INT8 dot product overflows FP16 -- silently, to Inf --
unless 127^2 * K <= 65,504 * 2^k, which is 9 already at K = 2,048.

TWO ENGINES, ONE BLOCK.
  GEMM  (default) gemm_full on the array's (16, 4, 16) shape. A-layout x in, D-layout y out.
  GEMV  (gemv=True) one token on the (1, 4, 32) shape (offload_hw_kernels/gemv.h). The
        layouts stay the GEMM shape's -- x in ROW 0 of a 16-row A operand, W in the same
        B layout -- but a pass reads two 16-column weight blocks, 128 bytes, twice the GEMM
        shape's rate; and y comes out as a plain FP16 ROW, not the D layout. A one-token
        projection is nothing but weight bytes, so this is the shape that decides decode.

STREAMED WEIGHTS, AUTOMATICALLY. A weight that is not already in L1 is loaded by this block,
and one that is larger than a chunk (`w_chunk_bytes`, 128 KiB by default) is STREAMED: the
B layout is n-major, so columns [c0, c0 + n) are one contiguous run of K * n bytes and each
run is a complete B operand. The chunks rotate through `w_buffers` L1 buffers (two: double
buffering) on the DM core's iDMA while the array computes the chunk before:

      iDMA   Ld0  Ld1  .    Ld2  .    Ld3  ...          Ld(i) waits for task(i - buffers)
      GEMM   .    T0   .    T1   .    T2   ...          task(i) waits for Ld(i), task(i-1)

The order is carried by EDGES, not by creation order, which is dispatch order only where
the topological sort cannot reorder: the task chain is what makes the last task imply all
of them, and what keeps a later chunk from running before x exists. Within that, the loads
are created ahead of the tasks that free their buffers, so the iDMA never idles behind the
array. Each chunk yields FINISHED output columns -- nothing accumulates across tasks -- so
no partial sum ever leaves the array. A weight in main memory or in the memory chiplet's HBM streams the same
way; from the HBM every load crosses the D2D link, which is exactly why it is streamed once
and never hoisted.

ONE STREAM PER CLUSTER (LoadStream). By default each Linear allocates its own buffers, so a
layer of N projections on one cluster holds N sets of them. A `stream` instead hands every
streamed operand on the cluster -- every projection's chunks, and FlashAttention-style
tiles -- the same few L1 SLABS, in build order: a load goes to the next slab and waits for
whatever last read it, which may belong to the previous projection. So the next weight
starts streaming while the last chunks of this one compute, and the layer holds one set of
slabs however many projections it has. That is the snax reference's "one stream of loads,
one stream of GEMM tasks" (snax-dsv2-mla.h), expressed as edges.

GROUPS (per-head GEMVs). `groups = G` runs G GEMVs of d_in x d_out in one weight: group g
reads its own x -- row 0 of the g-th 16-row A operand, 16 d_in bytes on -- and its own
[d_in, d_out] B block, which are consecutive, i.e. the B layout of [W_0 | W_1 | ...]. A
chunk is then whole groups. The absorbed attention's W_UK and W_UV per head are this.

WEIGHTS CHOSEN AT RUN TIME (w_slot). A routed expert's weights are not an address the
compiler knows: the router picks the expert. With `w_slot = (slot, field)` the block reads
no `w` port; it takes `rec`, an expert-slot record in this cluster's L1
(offload_hw_kernels/moe_route.h), and every chunk load looks its source up there
(idma_copy_slot). The graph is the same whatever the router decides.

WHAT IT DOES NOT DO. It does not normalise, quantise or reshape its input. A projection
consumes A-layout int8 and that is what its port says; getting there from whatever the
previous stage emitted is the caller's composition, and the order is forced by hardware:
reshape at FP16, THEN quantise, because a row_major->A conversion needs an 8-byte run that
int8 does not have. See comm/nest.py. For the GEMV the quantiser writes the A operand's row
0 directly (block.simd.QuantizeARow). Dequantisation with a per-column weight scale is the
caller's as well (block.simd.ScaleRow): one SIMD pass over the finished row.
"""

import itertools
from dataclasses import dataclass, field
from typing import Optional

from bingo_kernel_args import (
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelGemvArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdmaCopySlotArgs,
    SnaxBingoKernelIdmaRingLoadArgs,
    SnaxBingoKernelXdmaCrestExpandArgs,
)
from bingo_mem_handle import BingoMemFixedAddr

from ..comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                    PortSpec)
from ..comm.ports import at_offset
from ..comm.variant import free_fields

# One streamed weight chunk, at most: 64 output columns at K = 2,048, which is also the
# chunk the snax reference streams (layout.GEMV_CHUNK). Two of them are half the L1.
DEFAULT_W_CHUNK_BYTES = 128 * 1024

# The layouts the GEMV reads are the (16, 4, 16) shape's, whatever shape it computes in.
GEMV_MESH = SnaxBingoKernelGemvArgs.MESH
GEMV_A_ROWS = GEMV_MESH[0]


# One slot of an expert-slot record (offload_hw_kernels/moe_route.h), and the record as a port:
# bytes, in the L1 of the cluster that reads it.
REC_SLOT_BYTES = 128


def record_spec(slots: int, cluster: int) -> PortSpec:
    return PortSpec(Layout.ROW_MAJOR, DType.I8, (slots, REC_SLOT_BYTES), mem_level=MemLevel.L1,
                    cluster=cluster, doc="expert-slot record (moe_route.h)")


class LoadStream:
    """One cluster's stream of loads: `nbuf` L1 slabs of `nbytes` that every streamed
    operand on the cluster goes through, in the order the blocks are built.

    take() hands out the next slab and the node that last READ it -- the WAR predecessor
    the load into it must wait for; release() records who reads what was just loaded. A
    block takes and releases in its own build order, so the rotation, and the edges, run
    straight across block boundaries: the first chunk of the next projection waits for the
    last-but-one chunk task of this one, not for the whole projection.

    The slabs are allocated once, here, and never belong to a block: they live for the
    whole graph, which is exactly the L1 a stream costs.
    """

    def __init__(self, ctx: Ctx, cluster: int, nbytes: int = DEFAULT_W_CHUNK_BYTES,
                 nbuf: int = 2, name: str = "stream", slack: int = 0):
        if nbuf < 1 or nbytes <= 0 or nbytes % 64 or slack < 0 or slack % 64:
            raise ValueError(f"LoadStream: {nbuf} slabs of {nbytes} B (+{slack}); at least one "
                             f"slab, a whole number of 64-B beats.")
        g = ctx.at(cluster)
        # nbytes is the most a load may bring; a slab is `slack` larger (weight_crest: a CREST
        # record lands at the slab's end and expands in place, libs/crest.py)
        self.cluster, self.nbytes = cluster, int(nbytes)
        self.slab_bytes = self.nbytes + int(slack)
        self.slabs = [g.l1(f"{name}_c{cluster}_s{i}", self.slab_bytes) for i in range(nbuf)]
        self._next = 0
        self._reader = [None] * nbuf
        self._extra = [[] for _ in range(nbuf)]      # wait_for(): one-shot extra waits

    def take(self, nbytes: int):
        """(slab index, slab handle, [WAR predecessor]) for the next load of nbytes."""
        if nbytes > self.nbytes:
            raise ValueError(f"LoadStream: a {nbytes} B load does not fit a {self.nbytes} B "
                             f"slab.")
        i = self._next
        self._next = (i + 1) % len(self.slabs)
        war = [self._reader[i]] if self._reader[i] is not None else []
        war += [n for n in self._extra[i] if n not in war]
        self._extra[i] = []
        return i, self.slabs[i], war

    def release(self, i: int, reader) -> None:
        """`reader` is the LAST node that reads slab i's current contents: the next load
        into slab i waits for it (so it must imply every other reader)."""
        self._reader[i] = reader

    def hold_until(self, node) -> None:
        """The stream's first loads wait for `node`: every slab nobody has read yet takes it
        as its reader. For a stream that must not run beside `node`'s stage -- a kernel
        starting cold fetches its code from L3, and those refills return behind the DMA
        reads in flight, any cluster's."""
        self._reader = [r if r is not None else node for r in self._reader]

    def wait_for(self, node) -> None:
        """The NEXT load into every slab also waits for `node`, besides whoever last read the
        slab (unlike hold_until, which only reaches slabs nobody has read yet). For a stream
        whose next block must not take the in-order DM core before `node` -- the shared
        expert's loads behind the router's tasks, say."""
        for e in self._extra:
            if node not in e:
                e.append(node)

    def fence(self, node) -> None:
        """Every later load waits for `node`, which must follow every read of the slabs so
        far -- the store of what those reads computed, say. The DM core runs its tasks in
        order and the sort puts a load as early as its WAR edge allows, so without the fence
        the next projection's first load -- waiting on a ring chunk the prefetcher has not
        pushed yet -- sits in front of this one's dequantisation load and store."""
        self._reader = [node] * len(self._reader)


class WeightRings:
    """Per-cluster rings of weight chunks in this chiplet's L3, PUSHED there by the memory
    chiplet's iDMA under the host's prefetcher (host_kernel_lib.h weight_prefetch),
    instead of every cluster PULLING its weights over the D2D link.

    WHY. The D2D link is half-duplex. A pull sends a read request toward the memory
    chiplet for every burst and the data comes back the other way, so the link turns
    around each time; a push only flows toward us. Measured on
    hemaia_twochiplet_16MBL3_4cluster, 4 GHz SDR (host_only d2d_push_pull_bw, 1 MiB):
    pull HBM -> L3 12.0 GB/s, pull L4 -> L3 15.7, push HBM or L4 -> L3 17.4-17.5 of the
    link's 18.3. The ring also decouples the link from the clusters: the prefetcher runs
    up to `n_slots` chunks ahead of each cluster, through the attention and the routing.

    The workload stages the ring memory (symbols in L3: the slots of every ring, a 64-B
    flag and a 64-B release word per slot) and passes the handles; a LoadStream with
    `rings` set sends each streamed weight chunk whose source the host can name -- a
    fixed HBM or memchip-SRAM address, or an expert-slot field of the router's record --
    through its cluster's ring (Linear._build_streamed), and schedule() is the table the
    prefetcher walks. Ring r is cluster r's.
    """

    FIELDS = {"gu": 0, "gu_s": 1, "dn": 2, "dn_s": 3}
    MAX_RINGS = 4

    def __init__(self, *, slots, flags, releases, n_rings: int, n_slots: int,
                 slot_bytes: int, batch: int = 1, batch_routed: int = 0, head: int = 0,
                 batch_head: int = 0, split: bool = False):
        if not 1 <= n_rings <= self.MAX_RINGS or n_slots < 2 or slot_bytes % 64:
            raise ValueError(f"WeightRings: {n_rings} rings of {n_slots} slots of "
                             f"{slot_bytes} B; 1..4 rings, 2+ slots, whole 64-B beats.")
        if not 1 <= batch <= n_slots:
            raise ValueError(f"WeightRings: batch {batch}; 1..n_slots ({n_slots}).")
        self.slots, self.flags, self.releases = slots, flags, releases
        self.n_rings, self.n_slots, self.slot_bytes = n_rings, n_slots, slot_bytes
        # `split`: every cluster has TWO rings, its dense chunks in ring c and its routed ones
        # in ring c + n_rings / 2. In one ring a dense chunk taken after a routed one (the
        # shared expert's down, built after the first routed gate|up for the all-gather
        # lag) waited behind it for the router's record and went over the link after the
        # route; in a ring of its own it is pushed before. take*() still name the CLUSTER.
        self.split = bool(split)
        if self.split and n_rings % 2:
            raise ValueError(f"WeightRings(split): {n_rings} rings; two per cluster.")
        self.n_clusters = n_rings // 2 if self.split else n_rings
        self.batch = batch
        # Routed chunks (kind 1) may batch differently: before the router a short run keeps a
        # read across the link from waiting long behind it, after it long runs keep the link
        # streaming. 0: the same as `batch`. The prefetcher applies the same rule.
        self.batch_routed = int(batch_routed) or batch
        if not 1 <= self.batch_routed <= n_slots:
            raise ValueError(f"WeightRings: batch_routed {batch_routed}; 1..n_slots ({n_slots}).")
        # A ring's run that STARTS within its first `head` chunks batches by `batch_head`
        # (0: off): before the attention the links push one-chunk runs and the launch -- a
        # register round trip across the link per run -- takes a large share of them, while no
        # read across the link is waiting yet. The prefetcher applies the same rule.
        self.head, self.batch_head = int(head), int(batch_head)
        if self.batch_head and not (self.head > 1 and 1 <= self.batch_head <= n_slots):
            raise ValueError(f"WeightRings: head {head} batch_head {batch_head}; head > 1 and "
                             f"1..n_slots ({n_slots}).")
        self.entries = [[] for _ in range(n_rings)]      # (src, kind, offset, size)
        # per entry, for an Execution-IR export (bingo_exec_export): the bytes the GEMV reads after
        # expansion, and whether the pushed record is compressed; `last_take` is the (ring, index)
        # of the newest chunk, which the streamed Linear tags its ring load with
        self.entry_meta = [[] for _ in range(n_rings)]
        self.last_take = None
        # The newest ring load per cluster: every load of a ring waits for the one before
        # (same DM core, no tag), so the loads run in the ring's order whatever block
        # took them -- a load dispatched ahead of an earlier chunk of its ring would wait
        # for a push that cannot come while that chunk holds its slot.
        self.last = {}
        # The run each ring's newest chunk belongs to: [first slot, chunks, bytes]
        self._run = [None] * n_rings
        # params weight_crest: the chip's CrestRings (libs/crest.py) -- every weight chunk is
        # pushed compressed and loaded by the xDMA through its CrestDecompressor
        self.crest = None

    def _continues(self, ring: int, k: int, src: int, kind: int, offset: int) -> bool:
        """Whether chunk k of `ring` joins its predecessor's run. The prefetcher applies the
        same rules at run time (host_kernel_lib.h weight_prefetch_run_of), so the chunk's
        place in the ring can be fixed now: at most `batch` chunks, no wrap, one source
        stream (the same kind; kind 0: the next byte of the same address range; kind 1: the
        same record field, the next offset -- two experts adjacent in HBM never merge)."""
        run = self._run[ring]
        # k == 1: a ring's first run is chunk 0 alone (host_kernel_lib.h, the same rule)
        lim = self.batch_routed if kind & 1 else self.batch
        if run is not None and self.batch_head and k - run[1] < self.head:
            lim = self.batch_head       # the run's first chunk is k - run[1]
        if run is None or run[1] >= lim or k % self.n_slots == 0 or k == 1:
            return False
        p_src, p_kind, p_off, p_n = self.entries[ring][k - 1]
        if kind != p_kind:
            return False
        if not kind & 1:
            return src + offset == p_src + p_off + p_n
        return src == p_src and offset == p_off + p_n

    def take(self, ring: int, src: int, nbytes: int, *, kind: int = 0, offset: int = 0):
        """The next chunk of `ring`: (its place in the ring, flag, release handles, sequence
        number). Chunks of one run sit back to back from the run's first slot, so a run of
        short chunks (a K = 1408 weight: 88 KiB per 64 columns) is still one push; the run
        never outgrows the slots it reserves, one per chunk."""
        return self.take_ex(ring, src, nbytes, kind=kind, offset=offset)[:4]

    def take_ex(self, ring: int, src: int, nbytes: int, *, kind: int = 0, offset: int = 0):
        """take(), then (bit 0 CREST record, bit 1 trailer) and the bytes pushed: with
        self.crest a dense weight chunk (kind 0) or a routed weight field (kind 1, gu / dn) is
        replaced by its CREST record -- source, offset and pushed size -- before it is placed;
        the routed scale vectors stay plain."""
        compressed = False
        raw_nbytes = int(nbytes)
        if self.crest is not None:
            fld = {v: k for k, v in self.FIELDS.items()}.get(int(src) & 0xFF) if kind else None
            if kind == 0:
                src, nbytes = self.crest.dense(ring, int(src) + int(offset), int(nbytes))
                offset, compressed = 0, True
            elif fld in self.crest.FIELDS:
                offset, nbytes = self.crest.routed(ring, fld, int(offset), int(nbytes))
                compressed = True
        # trailer mode: the chunk's own last beat says it has landed; its schedule entry is
        # marked (kind bit 1) so the prefetcher pushes no flag for it
        trailer = compressed and self.crest.trailer
        if trailer:
            kind |= 2
        if nbytes > self.slot_bytes:
            raise ValueError(f"WeightRings: a {nbytes} B chunk; slots are {self.slot_bytes} B")
        if self.split and kind & 1:
            ring += self.n_clusters          # the cluster's routed ring
        k = len(self.entries[ring])
        src, kind, offset, nbytes = int(src), int(kind), int(offset), int(nbytes)
        if self._continues(ring, k, src, kind, offset):
            run = self._run[ring]
        else:
            run = self._run[ring] = [k % self.n_slots, 0, 0]
        place = (ring * self.n_slots + run[0]) * self.slot_bytes + run[2]
        run[1] += 1
        run[2] += nbytes
        idx = ring * self.n_slots + k % self.n_slots
        self.entries[ring].append((src, kind, offset, nbytes))
        self.entry_meta[ring].append({"raw": raw_nbytes, "compressed": compressed})
        self.last_take = (ring, k)
        flag = at_offset(self.slots, place + nbytes - 64) if trailer else \
            at_offset(self.flags, idx * 64)
        return (at_offset(self.slots, place), flag, at_offset(self.releases, idx * 64), k + 1,
                int(compressed) | (2 if trailer else 0), nbytes)

    def schedule(self) -> list:
        """uint64 words: chunks per ring (4), first entry per ring (4), then 4 per entry."""
        rings = self.entries + [[] for _ in range(self.MAX_RINGS - self.n_rings)]
        words = [len(e) for e in rings]
        first = 0
        for e in rings:
            words.append(first)
            first += len(e)
        for e in rings:
            for src, kind, off, n in e:
                words += [src, kind, off, n]
        return words

    def chunks(self) -> int:
        return sum(len(e) for e in self.entries)


@dataclass(frozen=True)
class LinearCfg:
    """Shapes in ELEMENTS. The mesh-tile counts a descriptor wants are derived."""

    tokens: int
    d_in: int
    d_out: int
    mesh: tuple                       # (meshRow, tileSize, meshCol)
    cluster: int = 0
    # WHERE EACH OPERAND IS HANDED OVER. One in L1 is used where it lies; one further out
    # gets this block's own load. Left None, each is resolved from whatever produces it.
    x_level: Optional[MemLevel] = None
    w_level: Optional[MemLevel] = None
    array_shape_idx: int = 0
    transpose_a: int = 0
    transpose_b: int = 0
    # The D port's power-of-two output scale: FP16 out = RNE(acc * 2^-d_shift), 0..14.
    d_shift: int = 0
    # The (1, 4, 32) GEMV (see the module doc): one token, or a pass of several that share each
    # weight chunk, one GEMV group per token (x in a_row, y one row per token).
    gemv: bool = False
    # STREAMING. The most bytes one weight chunk may take in L1 (None: the default), and
    # how many L1 buffers the chunks rotate through. A weight that fits in one chunk is
    # loaded whole, as a single task; so is one already in L1.
    w_chunk_bytes: Optional[int] = None
    w_buffers: int = 2
    # G independent GEMVs of d_in x d_out, each with its own x (see the module doc).
    groups: int = 1
    # The cluster's LoadStream: every chunk goes through its slabs (and its size bounds a
    # chunk). None: this block's own w_buffers.
    stream: Optional[LoadStream] = field(default=None, compare=False, repr=False)
    # (slot, field) of an expert-slot record: the weight's address is chosen at run time,
    # and the block reads a `rec` port instead of `w`. field: gu or dn. rec_slots is the
    # record's length in slots, which its port states.
    w_slot: Optional[tuple] = None
    rec_slots: int = 6
    # w_slot: where this block's columns start in the record field's weight, in bytes -- a
    # cluster streaming its slice of an expert whose table entry names the slices of several
    # clusters back to back.
    w_slot_off: int = 0
    # The one-row GEMV's x: in the A layout (16 d_in bytes) or in a_row (2 d_in bytes, the
    # 8-byte word of each A block the GEMV reads; see comm/ports.py).
    x_layout: Layout = Layout.A
    # With a stream: its first loads wait for this block's own load of x. A DM core that
    # starts waiting on a weight chunk first -- a ring chunk not pushed yet -- would hold
    # x's load, and the whole block, behind it.
    stream_after_x: bool = False
    # ... and so is every slab's next load even when the stream already had readers
    # (LoadStream.wait_for; hold_until reaches only unread slabs): the shared expert's loads
    # behind the router's DM tasks (dsv2 params sh_wait_route)
    stream_wait_x: bool = False
    # The weights' width: 8, or 4 -- INT4 through B's converter, one-row GEMV only, the weight
    # in Layout.B_W4 (nibble-packed pairs of 16-column blocks, half the bytes; gemv.h INT4
    # WEIGHTS). Every weight byte count below goes through _wb.
    w_bits: int = 8

    def __post_init__(self):
        mr, ts, mc = self.mesh
        if self.groups < 1:
            raise ValueError(f"LinearCfg: groups={self.groups}; at least one.")
        if self.groups > 1 and not self.gemv:
            raise ValueError("LinearCfg: groups is a GEMV form (each group its own A row 0).")
        if self.w_slot is not None:
            if self.stream is None or self.w_level not in (None, MemLevel.HBM, MemLevel.L3):
                raise ValueError("LinearCfg(w_slot): a weight chosen at run time is streamed "
                                 "through a LoadStream, from L3 or the HBM.")
            if self.w_slot[1] not in ("gu", "dn"):
                raise ValueError(f"LinearCfg: w_slot field {self.w_slot[1]!r}: gu or dn.")
        if Layout(self.x_layout) not in (Layout.A, Layout.A_ROW) or (
                Layout(self.x_layout) == Layout.A_ROW and not self.gemv):
            raise ValueError(f"LinearCfg: x_layout={self.x_layout}; A, or a_row for a GEMV.")
        if self.gemv:
            if tuple(self.mesh) != GEMV_MESH:
                raise ValueError(
                    f"LinearCfg(gemv): mesh={self.mesh}, but the GEMV reads the "
                    f"{GEMV_MESH} shape's A and B layouts whatever shape it computes in.")
            # groups > 1 reads each token's groups' operands after the token before's, in
            # either layout: A (16 d_in bytes a group) or a_row (2 d_in, at 1/8 the L1)
            multi_ok = (self.groups == 1 and Layout(self.x_layout) == Layout.A_ROW) or \
                (self.groups > 1 and Layout(self.x_layout) in (Layout.A, Layout.A_ROW))
            if self.tokens < 1 or (self.tokens > 1 and not multi_ok):
                raise ValueError(
                    f"LinearCfg(gemv): tokens={self.tokens}. The GEMV's shape has one row: "
                    f"several tokens run one GEMV group each over the same chunk, each its own "
                    f"8-byte-aligned x in a_row; with groups (per-head GEMVs) each token is a "
                    f"task of its own over the chunk, its groups' A operands after the token "
                    f"before's (got x_layout={self.x_layout}, groups={self.groups}).")
            checks = (("d_in", self.d_in, ts, "tileSize"),
                      ("d_out", self.d_out, mc, "one 16-column B block"))
        else:
            checks = (("tokens", self.tokens, mr, "meshRow"),
                      ("d_in", self.d_in, ts, "tileSize"),
                      ("d_out", self.d_out, mc, "meshCol"))
        for nm, val, unit, why in checks:
            if val % unit:
                raise ValueError(
                    f"LinearCfg: {nm}={val} is not a multiple of {why}={unit}. A partial "
                    f"tile is not an error the array reports -- it computes over the tile "
                    f"it was given and the tail of the output is never written.")
        if not 0 <= self.d_shift <= 14:
            raise ValueError(f"LinearCfg: d_shift={self.d_shift}, the D port's k is 0..14.")
        if self.w_bits not in (8, 4):
            raise ValueError(f"LinearCfg: w_bits={self.w_bits}; weights are INT8 or INT4.")
        if self.w_bits == 4 and (not self.gemv or self.d_out % (2 * mc)):
            raise ValueError(
                f"LinearCfg: w_bits=4 needs the one-row GEMV and d_out a multiple of "
                f"{2 * mc} (INT4 weights come in pairs of {mc}-column blocks); got "
                f"gemv={self.gemv}, d_out={self.d_out}.")
        if self.w_buffers < 1:
            raise ValueError(f"LinearCfg: w_buffers={self.w_buffers}; at least one.")
        if self._chunk_budget < self._wb(self.d_in * self._col_unit):
            raise ValueError(
                f"LinearCfg: a {self._chunk_budget} B chunk holds less than one "
                f"{self._col_unit}-column block at d_in={self.d_in}.")

    # ---- what a descriptor counts ------------------------------------------------------
    @property
    def M_T(self):
        return 1 if self.gemv else self.tokens // self.mesh[0]

    @property
    def K_T(self):
        return self.d_in // self.mesh[1]

    @property
    def N_T(self):
        return self.d_out // self.mesh[2]

    @property
    def sizes(self) -> dict:
        """Every buffer this block needs, in BYTES (the weight whole; a streamed one takes
        `w_chunk_bytes_used` per buffer instead)."""
        mr, ts, mc = self.mesh
        if self.gemv:
            G, T = self.groups, self.tokens
            return {"x": T * G * self._x_rows * self.d_in,       # int8, A operand(s) or a_row
                    "w": self._wb(G * self.d_in * self.d_out),    # int8 B / int4 b_w4
                    "y": T * G * self.d_out * 2}                  # fp16, a row per token
        return {
            "x": self.M_T * self.K_T * mr * ts,            # int8  A-layout
            "w": self.K_T * self.N_T * ts * mc,            # int8  B-layout
            "y": self.M_T * self.N_T * mr * mc * 2,        # fp16  D-layout
        }

    def _wb(self, n: int) -> int:
        """Bytes of n weights at w_bits: n, or n / 2 nibble-packed."""
        return n * self.w_bits // 8

    @property
    def _x_rows(self) -> int:
        """Bytes of a GEMV's x per value: the A layout's 16 rows, or a_row's 2."""
        return 2 if Layout(self.x_layout) == Layout.A_ROW else GEMV_A_ROWS

    # ---- the weight stream ---------------------------------------------------------------
    @property
    def _col_unit(self) -> int:
        """The narrowest column run one chunk may hold: a whole B block (and, for the GEMV,
        a pair of them, so every pass runs full width) -- or, with groups, a whole group."""
        if self.groups > 1:
            return self.d_out
        return 2 * self.mesh[2] if self.gemv else self.mesh[2]

    @property
    def _chunk_budget(self) -> int:
        """The most bytes one chunk may take: the stream's slab, else w_chunk_bytes."""
        if self.stream is not None:
            return min(self.stream.nbytes, self.w_chunk_bytes or self.stream.nbytes)
        return self.w_chunk_bytes or DEFAULT_W_CHUNK_BYTES

    @property
    def cols_total(self) -> int:
        """Output columns across all groups: the width of y and of the B layout of w."""
        return self.groups * self.d_out

    @property
    def w_chunk_cols(self) -> int:
        """Output columns per streamed chunk: the most whole units the chunk budget holds."""
        unit = self._col_unit
        cols = max(unit, (self._chunk_budget // self._wb(self.d_in)) // unit * unit)
        return min(cols, self.cols_total)

    def w_chunks(self, streamed: bool) -> list:
        """(first column, columns) of each weight chunk. One chunk -- the whole matrix --
        when the weight is not streamed: already in L1, or small enough to load whole."""
        if not streamed:
            return [(0, self.cols_total)]
        step = self.w_chunk_cols
        return [(c0, min(step, self.cols_total - c0))
                for c0 in range(0, self.cols_total, step)]

    def streams(self, w_level) -> bool:
        """Is the weight streamed? Only one this block has to LOAD: past a chunk, or
        through the cluster's LoadStream whatever its size."""
        if w_level == MemLevel.L1:
            return False
        if self.stream is not None:
            return True
        return self.sizes["w"] > self._chunk_budget


class Linear(Block):
    """y = x @ W, INT8 in, FP16 out.

      GEMM   in   x  A-layout int8, [tokens, d_in]
                  w  B-layout int8, [d_in, d_out]
             out  y  D-layout fp16, [tokens, d_out], in this block's cluster L1
      GEMV   in   x  A-layout int8, [1, d_in]: row 0 of a 16-row operand (16 * d_in bytes)
                  w  B-layout int8, [d_in, d_out]; with w_bits=4, b_w4 int4 (half the bytes)
             out  y  row-major fp16, [1, d_out], in this block's cluster L1
    """

    name = "linear"

    def __init__(self, cfg: LinearCfg = None, **params):
        self.cfg = cfg if cfg is not None else LinearCfg(**params)
        # A weight the router picks has no level to resolve: its loads read the record.
        self.free = free_fields(self.cfg, ("x_level",) if self.cfg.w_slot is not None
                                else ("x_level", "w_level"))

    @property
    def inputs(self) -> dict:
        """Where each operand is handed over is a boundary this block resolves; what its
        ARRAY reads is always L1, and that is `needs`."""
        c = self.cfg
        self._check_realised()
        rows = c.tokens
        ins = {"x": self._port(Layout(c.x_layout), (rows, c.groups * c.d_in), c.x_level,
                               "activation")}
        if c.w_slot is None:
            w4 = c.w_bits == 4
            ins["w"] = self._port(Layout.B_W4 if w4 else Layout.B, (c.d_in, c.cols_total),
                                  c.w_level, "weight", DType.I4 if w4 else DType.I8)
        else:
            ins["rec"] = record_spec(c.rec_slots, c.cluster)
        return ins

    def _port(self, layout, shape, level, doc, dtype=DType.I8) -> PortSpec:
        return PortSpec(layout, dtype, shape, mem_level=level,
                        cluster=self.cfg.cluster if level == MemLevel.L1 else None,
                        doc=doc)

    def _check_realised(self) -> None:
        if self.free:
            raise ValueError(
                f"Linear is a template: {', '.join(self.free)} not decided. Either pin "
                f"them, or add it to a Pipeline -- run() resolves them from whatever "
                f"produces each operand.")

    def variants(self) -> list:
        """One realisation per place an operand can arrive from. A weight may also come
        from the memory chiplet's HBM, which only a weight does: it is streamed once."""
        if not self.free:
            return [{}]
        levels = {"x_level": (MemLevel.L1, MemLevel.L3),
                  "w_level": (MemLevel.L1, MemLevel.L3, MemLevel.HBM)}
        return [dict(zip(self.free, combo))
                for combo in itertools.product(*(levels[f] for f in self.free))]

    @property
    def needs(self) -> dict:
        """L1 -- what the ARRAY reads, which is not the same as where the caller keeps it.

        Saying L3 here would be a statement about this block's loads, and it would refuse
        an operand that is ALREADY in L1 because a previous stage produced it there. That
        is the common case in a composed layer and it is the cheap one: there is nothing
        to move. So the requirement is where the engine reads, and build() emits a load
        only for what is further out.
        """
        from dataclasses import replace
        return {n: replace(s, mem_level=MemLevel.L1, cluster=self.cfg.cluster)
                for n, s in self.inputs.items()}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        if c.gemv:
            return {"y": PortSpec(Layout.ROW_MAJOR, DType.F16, (c.tokens, c.cols_total),
                                  mem_level=MemLevel.L1, cluster=c.cluster,
                                  doc="projection output, one fp16 row per token, in L1")}
        return {"y": PortSpec(Layout.D, DType.F16, (c.tokens, c.d_out),
                              mem_level=MemLevel.L1, cluster=c.cluster,
                              doc="projection output, D-layout fp16, in L1")}

    # ---- building ---------------------------------------------------------------------------

    def _task(self, g, ctx, name, x, w, y, col0, ncols, after):
        """The compute of one weight chunk: columns [col0, col0 + ncols) of y, from `w`
        holding exactly those columns. Returns the task nodes, in dispatch order."""
        c = self.cfg
        mr, ts, mc = c.mesh
        if c.gemv:
            # y is one fp16 row, so the chunk's outputs are its slice of it. With groups a
            # chunk is whole groups: group g0 onwards, each reading its own A operand.
            if c.groups > 1:
                # One task per token over the chunk's groups (heads): token t's A operands
                # follow token t-1's, its outputs are row t of y. Chained, so the last task
                # implies every token's.
                g0, ng = col0 // c.d_out, ncols // c.d_out
                a_step = c._x_rows * c.d_in
                tasks = []
                for t in range(c.tokens):
                    tasks.append(g.node(
                        name if c.tokens == 1 else f"{name}_t{t}", ctx.gemm,
                        "__snax_bingo_kernel_gemv",
                        SnaxBingoKernelGemvArgs(
                            at_offset(x, (t * c.groups + g0) * a_step), w,
                            at_offset(y, 2 * (t * c.cols_total + col0)), K=c.d_in,
                            N=c.d_out, d_shift=c.d_shift, groups=ng, a_step=a_step,
                            a_blk=8 if c._x_rows == 2 else 64, w4=c.w_bits == 4),
                        after if not tasks else [tasks[-1]]))
                return tasks
            if c.tokens > 1:
                # One group per token over the same chunk (b_step 0): token t's x is its own
                # a_row, 2 d_in bytes on, and its outputs the chunk's slice of row t of y.
                return [g.node(name, ctx.gemm, "__snax_bingo_kernel_gemv",
                               SnaxBingoKernelGemvArgs(x, w, at_offset(y, 2 * col0), K=c.d_in,
                                                       N=ncols, d_shift=c.d_shift,
                                                       groups=c.tokens, a_step=2 * c.d_in,
                                                       a_blk=8, w4=c.w_bits == 4, b_step=0,
                                                       d_step=2 * c.cols_total),
                               after)]
            return [g.node(name, ctx.gemm, "__snax_bingo_kernel_gemv",
                           SnaxBingoKernelGemvArgs(x, w, at_offset(y, 2 * col0), K=c.d_in,
                                                   N=ncols, d_shift=c.d_shift,
                                                   a_blk=8 if c._x_rows == 2 else 64,
                                                   w4=c.w_bits == 4),
                           after)]
        # D-layout (m, n, r, c): at a fixed m-block the chunk's n-blocks are contiguous but
        # the m-blocks are N_T blocks apart. The whole matrix is one dense task; a CHUNK of a
        # multi-row GEMM is one task per m-block. A-layout m-blocks are contiguous K runs.
        n0, nb = col0 // mc, ncols // mc
        blk = mr * mc * 2
        whole = ncols == c.d_out
        rows = [None] if whole else list(range(c.M_T))
        tasks = []
        for m in rows:
            if whole:
                a_at, d_at, M = x, y, c.M_T
            else:
                a_at = at_offset(x, m * c.K_T * mr * ts)
                d_at = at_offset(y, (m * c.N_T + n0) * blk)
                M = 1
            # input_C_addr=0, NOT y. Every VersaCore GEMM computes D = A*B + C, and
            # accumPrevC=0 selects where C comes from: a NON-ZERO address READS C from
            # memory and adds it. Passing the output buffer as C reads it before anything
            # wrote it -- uninitialised TCDM, which simulates as X -- and D = A*B + X = X.
            # That X reaches the host, where check_result loads it and the CVA6 issues on an
            # unknown operand. It never fails a check; it kills the host.
            # Chained: a node with no edge to its sibling may be dispatched before it
            # (creation order is dispatch order only where the topological sort cannot
            # reorder), so each m-task follows the one before and the last implies all.
            tasks.append(g.node(
                name if whole or c.M_T == 1 else f"{name}_m{m}", ctx.gemm,
                "__snax_bingo_kernel_gemm_full",
                SnaxBingoKernelGemmFullArgs(
                    input_A_addr=a_at, input_B_addr=w, input_C_addr=0, output_D_addr=d_at,
                    M=M, K=c.K_T, N=nb, array_shape_idx=c.array_shape_idx,
                    transpose_A=c.transpose_a, transpose_B=c.transpose_b,
                    accumPrevC=0, int32tofp16_enable=1, d_shift=c.d_shift),
                after if not tasks else [tasks[-1]]))
        return tasks

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        self._check_realised()
        c, sz = self.cfg, self.cfg.sizes
        g = ctx.at(c.cluster)

        l1_y = g.l1(f"{self.name}_y", sz["y"])

        # ---- x: used where it lies if it is in L1, else loaded whole ---------------------
        # An operand ALREADY in L1 is used where it lies: copying it to a second L1 buffer
        # would cost a transfer, a dependency edge and a second allocation the static-L1
        # pass then has to fit, all to produce bytes that are already there. `ends` carries
        # whatever produced it, so the ordering is unchanged.
        xp = bound["x"]
        if xp.spec.mem_level == MemLevel.L1:
            l1_x, x_after, ld_x = xp.handle, list(xp.ends), None
        else:
            l1_x = g.l1(f"{self.name}_x", sz["x"])
            ld_x = g.node(f"Ld_{self.name}_x", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(xp.handle, l1_x, sz["x"]))
            x_after = [ld_x]

        # ---- W: in L1, loaded whole, or streamed in chunks ---------------------------------
        if c.w_slot is not None or c.stream is not None:
            return self._build_streamed(ctx, g, bound, l1_x, x_after, ld_x, l1_y)
        wp = bound["w"]
        w_in_l1 = wp.spec.mem_level == MemLevel.L1
        streamed = c.streams(wp.spec.mem_level)
        chunks = c.w_chunks(streamed)
        loads, tasks, bufs = [], [], []
        if w_in_l1:
            tasks += self._task(g, ctx, f"Gemm_{self.name}", l1_x, wp.handle, l1_y, 0,
                                c.d_out, x_after + list(wp.ends))
        else:
            nbuf = min(c.w_buffers, len(chunks))
            cbytes = c._wb(max(n for _, n in chunks) * c.d_in)
            bufs = [g.l1(f"{self.name}_w" if nbuf == 1 and not streamed
                         else f"{self.name}_w{b}", cbytes)
                    for b in range(nbuf)]

            def load(i):
                c0, n = chunks[i]
                # WAR: the buffer is free once every task that read its last chunk is done;
                # the tasks are chained, so the last of them is enough.
                war = [chunk_tasks[i - nbuf][-1]] if i >= nbuf else []
                return g.node(
                    f"Ld_{self.name}_w" if not streamed else f"Ld_{self.name}_w{i}",
                    ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(at_offset(wp.handle, c._wb(c0 * c.d_in)),
                                                  bufs[i % nbuf], c._wb(n * c.d_in)),
                    war)

            chunk_tasks = []
            loads = [load(i) for i in range(nbuf)]
            for i, (c0, n) in enumerate(chunks):
                name = f"Gemm_{self.name}" if not streamed else f"Gemm_{self.name}_c{i}"
                # RAW on the chunk's load, and a CHAIN through the tasks: the first waits
                # for x, every later one for the task before it. Creation order is dispatch
                # order only where the topological sort cannot reorder, and without the
                # chain a later chunk's task -- whose own load may have no predecessor --
                # could run before x exists, and the output's last task would not imply the
                # others (with two buffers it follows only every OTHER chunk).
                dep = [loads[i]] + (x_after if i == 0 else [chunk_tasks[-1][-1]])
                ts_i = self._task(g, ctx, name, l1_x, bufs[i % nbuf], l1_y, c0, n, dep)
                chunk_tasks.append(ts_i)
                tasks += ts_i
                if i + nbuf < len(chunks):
                    loads.append(load(i + nbuf))

        nodes = ([ld_x] if ld_x is not None else []) + loads + tasks
        # The first node that reads each input is what the linker joins its producer to; the
        # rest follow it on the same core. The output is finished when the LAST task is --
        # one end, not one per chunk, because every dependency edge costs a hardware tag.
        x_end = (ld_x,) if ld_x is not None else (tasks[0],)
        w_end = (loads[0],) if loads else (tasks[0],)
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], l1_y, (tasks[-1],), name="y")},
            inputs={"x": Port(self.inputs["x"], xp.handle, x_end, name="x"),
                    "w": Port(self.inputs["w"], wp.handle, w_end, name="w")},
            nodes=nodes,
            # The first loads have no predecessor, so no data edge can order them against an
            # earlier block's buffers. Pipeline(gate_sources=True) is what lets their L1 be
            # reused -- at the price of a prefetch that could have run under the producer;
            # naming them here is what makes either possible.
            sources=loads[:len(bufs)] if bufs else [],
            extra={"l1": {"x": l1_x, "w": bufs[0] if bufs else wp.handle, "y": l1_y},
                   "w_bufs": bufs, "chunks": chunks, "loads": loads, "tasks": tasks})

    def _build_streamed(self, ctx, g, bound, l1_x, x_after, ld_x, l1_y) -> BlockResult:
        """The weight through the cluster's LoadStream (or, with w_slot, from an address the
        router chose): one load and one task per chunk, the load into the next slab after
        whatever last read it, the task after its load and after the task before it."""
        c = self.cfg
        if c.stream is None:
            raise ValueError("Linear(w_slot) streams through a LoadStream; pass stream=.")
        if c.stream.cluster != c.cluster:
            raise ValueError(f"Linear on cluster {c.cluster} given cluster "
                             f"{c.stream.cluster}'s LoadStream.")
        chunks = c.w_chunks(True)
        if c.w_slot is None:
            wp = bound["w"]
            if wp.spec.mem_level == MemLevel.L1:
                raise ValueError("Linear(stream): the weight is already in L1; there is "
                                 "nothing to stream.")
            src_port, rec_after = wp, []
        else:
            src_port = bound["rec"]
            # EVERY load reads the record, not only the first: a later load with no path
            # from the record's producer may be sorted ahead of it. The first load waits for
            # the record and each later one for the load before it -- a CHAIN, not a fan-out
            # from the record's producer: fanned out, each of those edges would need a
            # dependency tag of its own in one (cluster, core, core) cell, and a slot has
            # dozens of chunks.
            rec_after = list(src_port.ends)
        if c.stream_wait_x and ld_x is not None:
            c.stream.wait_for(ld_x)
        if c.stream_after_x and ld_x is not None:
            c.stream.hold_until(ld_x)
        rings = getattr(c.stream, "rings", None)
        if rings is not None and c.w_slot is None and not isinstance(src_port.handle,
                                                                      BingoMemFixedAddr):
            rings = None        # not a numbered HBM / memchip address: pull it as before
        loads, tasks, expands = [], [], []
        for i, (c0, n) in enumerate(chunks):
            off, nbytes = c._wb(c0 * c.d_in), c._wb(n * c.d_in)
            si, slab, war = c.stream.take(nbytes)
            nm = f"Ld_{self.name}_w{i}" if len(chunks) > 1 else f"Ld_{self.name}_w"
            if rings is not None:
                # The prefetcher pushes the chunk into this cluster's ring; the load waits
                # for its flag and copies it from L3. A routed chunk's source is named by
                # its record slot and field; the prefetcher reads the address at run time.
                if c.w_slot is None:
                    ring_src = dict(src=src_port.handle.address + off)
                else:
                    slot, fld = c.w_slot
                    ring_src = dict(src=(slot << 8) | WeightRings.FIELDS[fld], kind=1,
                                    offset=c.w_slot_off + off)
                sl, fl, rl, seq, crest, pushed = rings.take_ex(c.cluster, nbytes=nbytes,
                                                               **ring_src)
                prev = [loads[-1]] if loads else \
                    ([rings.last[c.cluster]] if rings.last.get(c.cluster) is not None else [])
                deps = war + [n for n in prev if n not in war]
                if crest & 1:
                    # weight_crest: the record into the END of the slab, then the xDMA expands
                    # it in place into the slab's first beats (libs/crest.py)
                    ld = g.node(nm, ctx.dm, "__snax_bingo_kernel_idma_ring_load",
                                SnaxBingoKernelIdmaRingLoadArgs(
                                    fl, seq, sl, at_offset(slab, c.stream.slab_bytes - pushed),
                                    pushed, rl, trailer=bool(crest & 2)), deps)
                    expands.append(g.node(
                        f"Ex_{self.name}_w{i}" if len(chunks) > 1 else f"Ex_{self.name}_w",
                        ctx.xdma, "__snax_bingo_kernel_xdma_crest_expand",
                        SnaxBingoKernelXdmaCrestExpandArgs(slab, c.stream.slab_bytes), [ld]))
                else:
                    ld = g.node(nm, ctx.dm, "__snax_bingo_kernel_idma_ring_load",
                                SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, slab, nbytes, rl),
                                deps)
                rings.last[c.cluster] = ld
                ld.exec_chunk = rings.last_take       # (ring, index): the Execution-IR export
            elif c.w_slot is None:
                args = SnaxBingoKernelIdma1dCopyArgs(at_offset(src_port.handle, off), slab,
                                                     nbytes)
                ld = g.node(nm, ctx.dm, "__snax_bingo_kernel_idma_1d_copy", args, war)
            else:
                slot, fld = c.w_slot
                args = SnaxBingoKernelIdmaCopySlotArgs(src_port.handle, slot, fld,
                                                       c.w_slot_off + off, slab, nbytes)
                ld = g.node(nm, ctx.dm, "__snax_bingo_kernel_idma_copy_slot", args,
                            war + (rec_after if not loads else [loads[-1]]))
            loads.append(ld)
            name = f"Gemm_{self.name}_c{i}" if len(chunks) > 1 else f"Gemm_{self.name}"
            ready = expands[-1] if len(expands) == len(loads) else ld
            dep = [ready] + (x_after if i == 0 else [tasks[-1]])
            ts_i = self._task(g, ctx, name, l1_x, slab, l1_y, c0, n, dep)
            tasks += ts_i
            c.stream.release(si, ts_i[-1])
        nodes = ([ld_x] if ld_x is not None else []) + loads + expands + tasks
        x_end = (ld_x,) if ld_x is not None else (tasks[0],)
        ins = {"x": Port(self.inputs["x"], bound["x"].handle, x_end, name="x")}
        if c.w_slot is None:
            ins["w"] = Port(self.inputs["w"], src_port.handle, (loads[0],), name="w")
        else:
            ins["rec"] = Port(self.inputs["rec"], src_port.handle, (loads[0],), name="rec")
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], l1_y, (tasks[-1],), name="y")},
            inputs=ins, nodes=nodes,
            # NO sources: the loads are the stream's, ordered by the slabs' WAR edges and
            # nothing else. Gating them behind x's producer would stop the stream from
            # running ahead across block boundaries, which is the point of it.
            sources=[],
            extra={"l1": {"x": l1_x, "y": l1_y}, "chunks": chunks, "loads": loads,
                   "tasks": tasks})
