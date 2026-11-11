# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""How a cluster's streamed operands reach its L1: the slabs they rotate through, and the L3
rings the memory chiplet pushes weight chunks into.

  LoadStream   one cluster's stream of loads: a few L1 slabs that every streamed operand on
               the cluster goes through, in build order, each load after whatever last read
               its slab. Linear's weight chunks and MlaAttention's K/V tiles share it.
  WeightRings  per-cluster rings of weight chunks in a chiplet's L3, pushed there by the
               memory chiplet's iDMA under the host's prefetcher. A LoadStream with `rings`
               set loads its weight chunks from them (Linear._build_streamed), and
               schedule() is the table the prefetcher walks.
"""

from bingo_kernel_args import SnaxBingoKernelIdmaRingLoadArgs

from ..comm import Ctx
from ..comm.ports import at_offset

# One streamed weight chunk, at most: 64 output columns at K = 2,048, which is also the
# chunk the snax reference streams (layout.GEMV_CHUNK). Two of them are half the L1.
DEFAULT_W_CHUNK_BYTES = 128 * 1024


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

    # The cluster's WeightRings when its streamed weight chunks go through a ring; the workload
    # sets it on the instance.
    rings = None

    def __init__(self, ctx: Ctx, cluster: int, nbytes: int = DEFAULT_W_CHUNK_BYTES,
                 nbuf: int = 2, name: str = "stream"):
        if nbuf < 1 or nbytes <= 0 or nbytes % 64:
            raise ValueError(f"LoadStream: {nbuf} slabs of {nbytes} B; at least one "
                             f"slab, a whole number of 64-B beats.")
        g = ctx.at(cluster)
        # nbytes is the most a load may bring, and a slab's size
        self.cluster, self.nbytes = cluster, int(nbytes)
        self.slab_bytes = self.nbytes
        self.slabs = [g.l1(f"{name}_c{cluster}_s{i}", self.slab_bytes) for i in range(nbuf)]
        self._next = 0
        self._reader = [None] * nbuf
        self._extra = [[] for _ in range(nbuf)]      # wait_for(): one-shot extra waits
        # strict: refuse to hand out a slab whose last load has no reader recorded yet. The WAR edge
        # of such a take would be the reader before that load, so the new load could land before the
        # last one was read. Off by default; a caller whose takes may run ahead of its releases sets it.
        self.strict = False
        self._unread = [False] * nbuf

    def take(self, nbytes: int):
        """(slab index, slab handle, [WAR predecessor]) for the next load of nbytes."""
        if nbytes > self.nbytes:
            raise ValueError(f"LoadStream: a {nbytes} B load does not fit a {self.nbytes} B "
                             f"slab.")
        i = self._next
        self._next = (i + 1) % len(self.slabs)
        return self._take(i)

    def _take(self, i: int):
        if self.strict and self._unread[i]:
            raise ValueError(f"LoadStream: slab {i} of cluster {self.cluster} taken again before the "
                             f"reader of its last load was recorded: the new load would not wait "
                             f"for that read.")
        self._unread[i] = True
        r = self._reader[i]
        war = [] if r is None else list(r) if isinstance(r, (list, tuple)) else [r]
        war += [n for n in self._extra[i] if n not in war]
        self._extra[i] = []
        return i, self.slabs[i], war

    def take_at(self, i: int, nbytes: int):
        """take(), into slab i instead of the next in turn; the turn does not move. For a caller
        that keeps its own slab plan (a K/V tile sweep reusing each kind's slabs)."""
        if nbytes > self.nbytes:
            raise ValueError(f"LoadStream: a {nbytes} B load does not fit a {self.nbytes} B "
                             f"slab.")
        return self._take(i)

    def release(self, i: int, reader) -> None:
        """`reader`: the nodes that read slab i's current contents, a list, or one node that every other reader
        precedes by the graph's edges: the next load into slab i waits for them. A core's program order does not
        count, since the compiler chooses it."""
        self._reader[i] = reader
        self._unread[i] = False

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
    around each time; a push only flows toward us, so it keeps the link streaming. The
    ring also decouples the link from the clusters: the prefetcher runs up to `n_slots`
    chunks ahead of each cluster, through the attention and the routing.

    The workload stages the ring memory (symbols in L3: the slots of every ring, a 64-B
    flag and a 64-B release word per slot) and passes the handles; a LoadStream with
    `rings` set sends each streamed weight chunk whose source the host can name -- a
    fixed HBM or memchip-SRAM address, or an expert-slot field of the router's record --
    through its cluster's ring (Linear._build_streamed), and schedule() is the table the
    prefetcher walks. Ring c is cluster c's (with `split`, ring c + n_rings / 2 holds its
    routed chunks).
    """

    FIELDS = {"gu": 0, "gu_s": 1, "dn": 2, "dn_s": 3}
    MAX_RINGS = 4
    MAX_SLOTS = 32          # host runtime libbingo/weight_ring_guard.h WEIGHT_RING_GUARD_MAX_SLOTS
    # A dense chunk (kind 0) ends in a trailer its caller appended (params weight_ring.trailer);
    # the workload sets it on the instance.
    plain_trailer = False

    def __init__(self, *, slots, flags, releases, n_rings: int, n_slots: int,
                 slot_bytes: int, batch: int = 1, batch_routed: int = 0, head: int = 0,
                 batch_head: int = 0, split: bool = False):
        if not 1 <= n_rings <= self.MAX_RINGS or n_slots < 2 or slot_bytes % 64:
            raise ValueError(f"WeightRings: {n_rings} rings of {n_slots} slots of "
                             f"{slot_bytes} B; 1..4 rings, 2+ slots, whole 64-B beats.")
        # the prefetcher's byte guard tracks this many slots a ring (weight_ring_guard.h)
        if n_slots > self.MAX_SLOTS:
            raise ValueError(f"WeightRings: {n_slots} slots; the prefetcher guards at most {self.MAX_SLOTS}.")
        if not 1 <= batch <= n_slots:
            raise ValueError(f"WeightRings: batch {batch}; 1..n_slots ({n_slots}).")
        self.slots, self.flags, self.releases = slots, flags, releases
        self.n_rings, self.n_slots, self.slot_bytes = n_rings, n_slots, slot_bytes
        # `split`: every cluster has TWO rings, its dense chunks in ring c and its routed ones
        # in ring c + n_rings / 2. In one ring a dense chunk taken after a routed one (the
        # shared expert's down, built after the first routed gate|up for the all-gather
        # lag) waits behind it for the router's record; in a ring of its own it is pushed
        # first. take*() still name the CLUSTER.
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
        self.last_take = None
        # The newest ring load per cluster: every load of a ring waits for the one before
        # (same DM core, no tag), so the loads run in the ring's order whatever block
        # took them -- a load dispatched ahead of an earlier chunk of its ring would wait
        # for a push that cannot come while that chunk holds its slot.
        self.last = {}
        # The run each ring's newest chunk belongs to: [first slot, chunks, bytes]
        self._run = [None] * n_rings

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
        never outgrows the slots it reserves, one per chunk. A chunk can so start in a lower
        slot than its own, where the next wrap's run reserves nothing for it: the prefetcher's
        byte guard holds such a slot until the chunk is read (weight_ring_guard.h)."""
        return self.take_ex(ring, src, nbytes, kind=kind, offset=offset)[:4]

    def take_ex(self, ring: int, src: int, nbytes: int, *, kind: int = 0, offset: int = 0):
        """take(), then (bit 1: the chunk ends in a trailer) and the bytes pushed."""
        # trailer mode: the chunk's own last beat says it has landed; its schedule entry is
        # marked (kind bit 1) so the prefetcher pushes no flag for it. A dense chunk
        # (plain_trailer) ends in one its caller appended.
        trailer = self.plain_trailer and not kind & 1
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
        self.last_take = (ring, k)
        flag = at_offset(self.slots, place + nbytes - 64) if trailer else \
            at_offset(self.flags, idx * 64)
        return (at_offset(self.slots, place), flag, at_offset(self.releases, idx * 64), k + 1,
                2 if trailer else 0, nbytes)

    def load_chunk(self, node, cluster: int, name: str, *, src: int, nbytes: int, slab, war,
                   deps=(), kind: int = 0, offset: int = 0, dm: int, trailer: bool = False):
        """The ring load of one chunk into an L1 slab: take_ex(), then an idma_ring_load that
        waits for the prefetcher's push and copies the chunk from L3 into `slab` on core `dm`.
        It follows `war` (the slab's last readers, LoadStream.take), then the ring's newest
        load on `cluster` (so the loads run in the ring's order), then `deps`; the load becomes
        the ring's newest and is tagged for the Execution-IR export. `node(name, core, kernel,
        args, deps)` creates a node.

        The load honours the chunk's trailer only with `trailer`. Returns (the load, the chunk's
        place in the ring: the address the load reads)."""
        sl, fl, rl, seq, how, _ = self.take_ex(cluster, src=src, nbytes=nbytes, kind=kind,
                                               offset=offset)
        prev = [self.last[cluster]] if self.last.get(cluster) is not None else []
        after = list(dict.fromkeys(list(war) + prev + list(deps)))
        ld = node(name, dm, "__snax_bingo_kernel_idma_ring_load",
                  SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, slab, nbytes, rl,
                                                  trailer=trailer and bool(how & 2)), after)
        self.last[cluster] = ld
        ld.exec_chunk = self.last_take
        return ld, sl

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

    def races(self, pf, parts=((0, 0),)) -> list:
        """The overwrite races of this schedule under the prefetcher task `pf` (its
        HostBingoKernelWeightPrefetchArgs) walking the chunk ranges `parts` ([(chunk_lo, chunk_hi)],
        chunk_hi 0: to the end): [(ring, unread chunk, pushed chunk, first byte, end byte of the
        unread chunk the push covers)].

        Per ring the runs follow the prefetcher's rules (host_kernel_lib.h weight_prefetch_run_of:
        a ring's first run is one chunk, a run that starts within `head` chunks takes `batch_head`,
        a routed run `batch_routed`, else `batch`; one source stream; no wrap; a part's range ends
        its runs), the chunks of a run back to back from its first slot. A race is a run that the
        slot rule alone (at most n_slots chunks ahead of the reads) lets the prefetcher push while
        an earlier chunk, not read yet, has bytes where the run writes: a chunk shorter than a slot
        starts in a lower slot than its own. The prefetcher's byte guard (weight_ring_guard.h)
        holds such a run until those bytes are read."""
        ns, sb = pf.n_slots, pf.slot_bytes
        batch = pf.batch or 1
        batch_routed = pf.batch_routed or batch
        out = []
        for ring, ent in enumerate(self.entries[:pf.n_rings]):
            runs, place = [], [None] * len(ent)
            for lo, hi in parts:
                k0, n = lo, len(ent) if not hi or hi > len(ent) else hi
                while k0 < n:
                    lim = 1 if k0 == 0 else pf.batch_head if (pf.batch_head and k0 < pf.head) else \
                        (batch_routed if ent[k0][1] & 1 else batch)
                    s0, m, nbytes, kind0, src0, nxt = k0 % ns, 0, 0, None, None, None
                    while m < lim and k0 + m < n and s0 + m < ns:
                        src, kind, off, size = ent[k0 + m]
                        routed = kind & 1
                        if m > 0 and (kind != kind0 or ((src + off != nxt) if not routed else
                                                        (src != src0 or off != nxt))):
                            break
                        if m == 0:
                            kind0, src0 = kind, src
                        nxt = off + size if routed else src + off + size
                        place[k0 + m] = (s0 * sb + nbytes, size)
                        nbytes += size
                        m += 1
                    runs.append((k0, m, s0 * sb, s0 * sb + nbytes))
                    k0 += m
            for k0, m, lo, hi in runs:
                # the slot rule lets chunk k0 + m - 1 go once chunk k0 + m - 1 - n_slots is read
                for j in range(max(0, k0 + m - ns), k0):
                    if place[j] is not None and place[j][0] < hi and lo < place[j][0] + place[j][1]:
                        pj, nj = place[j]
                        out.append((ring, j, k0 + m - 1, max(lo, pj) - pj, min(hi, pj + nj) - pj))
        return out
