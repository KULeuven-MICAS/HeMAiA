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
               memory chiplet's iDMA, each run posted by its ring's cluster (push_from). A
               LoadStream with `rings` set loads its weight chunks from them
               (Linear._build_streamed).
"""

from bingo_kernel_args import SnaxBingoKernelEnginePostArgs, SnaxBingoKernelIdmaRingLoadArgs

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
        the next projection's first load -- waiting on a ring chunk not pushed yet -- sits in
        front of this one's dequantisation load and store."""
        self._reader = [node] * len(self._reader)


class WeightRings:
    """Per-cluster rings of weight chunks in this chiplet's L3, PUSHED there by the memory
    chiplet's iDMA, each run posted by its ring's cluster, instead of every cluster PULLING its
    weights over the D2D link.

    WHY. The D2D link is half-duplex. A pull sends a read request toward the memory
    chiplet for every burst and the data comes back the other way, so the link turns
    around each time; a push only flows toward us, so it keeps the link streaming. The
    ring also decouples the link from the clusters: the pushes run up to `n_slots` chunks
    ahead of each cluster, through the attention and the routing.

    The workload stages the ring memory (symbols in L3: the slots of every ring and a 64-B
    flag per slot) and passes the handles; a LoadStream with `rings` set sends each streamed
    weight chunk through its cluster's ring (Linear._build_streamed). Ring c is cluster c's.

    Each run is a task of its ring's cluster, a post on the DM core that queues the run into
    the push engine's queue, and its edges are the whole protocol (push_from).
    """

    def __init__(self, *, slots, flags, n_rings: int, n_slots: int, slot_bytes: int, batch: int = 1,
                 head: int = 0, batch_head: int = 0):
        if n_rings < 1 or n_slots < 2 or slot_bytes % 64:
            raise ValueError(f"WeightRings: {n_rings} rings of {n_slots} slots of "
                             f"{slot_bytes} B; 1+ rings, 2+ slots, whole 64-B beats.")
        if not 1 <= batch <= n_slots:
            raise ValueError(f"WeightRings: batch {batch}; 1..n_slots ({n_slots}).")
        self.slots, self.flags = slots, flags
        self.n_rings, self.n_slots, self.slot_bytes = n_rings, n_slots, slot_bytes
        self.batch = batch
        # A ring's run that STARTS within its first `head` chunks batches by `batch_head`
        # (0: off): a cluster's first chunks land one by one, not behind a whole run.
        self.head, self.batch_head = int(head), int(batch_head)
        if self.batch_head and not (self.head > 1 and 1 <= self.batch_head <= n_slots):
            raise ValueError(f"WeightRings: head {head} batch_head {batch_head}; head > 1 and "
                             f"1..n_slots ({n_slots}).")
        self.entries = [[] for _ in range(n_rings)]      # (src, 0, offset, size): the Execution IR's entry
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
        # push_from: the push engine, and per ring its runs [push, first load, first chunk] and its
        # chunks' bytes in the ring (lo, hi, chunk, load)
        self.pusher = None
        self._runs = [[] for _ in range(n_rings)]
        self._held = [[] for _ in range(n_rings)]
        # post_chunk / read_chunk (a ring several clusters read): per ring, chunk -> (its bytes, its run)
        self._chunk = [{} for _ in range(n_rings)]
        # boundary(): per ring, chunk -> (tasks after the run before it, tasks before its run)
        self._bounds = [{} for _ in range(n_rings)]
        # tail_before(): per ring, the tasks its last post comes before (close())
        self._tail = [[] for _ in range(n_rings)]
        # per ring, chunk -> the post of its run (post_of)
        self._posts = [{} for _ in range(n_rings)]

    def push_from(self, *, memchip: int, engine: int, seq, edge, depth: int = 32,
                  seq_entries: int = 1024) -> None:
        """Every run of a ring is posted by the ring's cluster, a post (offload_hw_kernels/idma.h
        engine_post) on its DM core, into the descriptor queue of push engine `engine` of memory
        chip `memchip` (`depth`: the transfers of the queue the rings may hold, all 32 or a share of
        them, the rest kept for other posters). Call it before any chunk is taken. `seq` is that
        memory chip's sequence table, `seq_entries` 64-B entries, entry i holding i; `edge(a, b)`
        adds a graph edge. A push's edges are the protocol:

          the newest load of the bytes the run overwrites -> push
          the ring's previous push -> push                                 (posted in ring order)
          the first load of the run `share` runs earlier -> push           (the queue never fills)
          push -> each load of its run                                     (which polls its flag)

        A push queues two transfers (the data, then the flags), the chip's rings share the
        queue, so a ring may have `share` = depth / (2 rings) runs posted and not landed: once a
        run's first flag has landed, the engine has taken that run's transfers off the queue.
        A ring's first push follows start_after()'s task."""
        share = depth // (2 * self.n_rings)
        if share < 1:
            raise ValueError(f"WeightRings.push_from: a queue of {depth} transfers for {self.n_rings} rings")
        if any(self.entries):
            raise ValueError("WeightRings.push_from: chunks were taken before the push engine was set")
        self.pusher = dict(memchip=int(memchip), engine=int(engine), seq=seq, edge=edge, share=share,
                           seq_entries=int(seq_entries), start=[])

    def boundary(self, ring: int, k: int, *, before=(), after=()) -> None:
        """Chunk k of `ring` starts a run, and the push engine's queue takes this
        ring's runs up to chunk k - 1, then another poster's entries, then the rest: the post of
        the run that ends at chunk k - 1 comes before every task of `before` (the other poster's
        first post), and the post of chunk k's run after every task of `after` (its last post).
        Call it before chunk k is taken; close() settles a boundary the ring never reaches."""
        if k < 1 or k < len(self.entries[ring]):
            raise ValueError(f"WeightRings.boundary: chunk {k} of ring {ring}; 1.. and not taken yet")
        b, a = self._bounds[ring].setdefault(k, ([], []))
        b += list(before)
        a += list(after)

    def runs_of(self, ring: int):
        """[(a run's post, the loads of its chunks)] of `ring`, in ring order (load_chunk and
        read_chunk's last readers; a post_chunk ring's other readers precede those)."""
        loads = {k: ld for _a, _b, k, ld in self._held[ring]}
        out, order = {}, []
        for k, post in sorted(self._posts[ring].items()):
            if post not in out:
                out[post] = []
                order.append(post)
            if k in loads:
                out[post].append(loads[k])
        return [(p, out[p]) for p in order]

    def queue_span(self):
        """(each ring's first post, each ring's last run's first load) -- once that load has
        run, every transfer the rings posted has left the engine's queue (push_from) -- or None if a ring's last
        run has no load yet."""
        runs = [r for r in self._runs if r]
        if any(r[-1][1] is None for r in runs):
            return None
        return [r[0][0] for r in runs], [r[-1][1] for r in runs]

    def post_of(self, ring: int, k: int):
        """The post of the run holding chunk k of `ring` (taken already)."""
        if k not in self._posts[ring]:
            raise ValueError(f"WeightRings.post_of: chunk {k} of ring {ring} has no post (not taken yet)")
        return self._posts[ring][k]

    def post_after(self, ring: int, k: int, tasks) -> None:
        """The post of the run holding chunk k of `ring` follows every task of `tasks`: an edge now if
        chunk k is taken, else a boundary() at k (chunk k starts a run) that they join as `after`."""
        if k < len(self.entries[ring]):
            for t in tasks:
                self.pusher["edge"](t, self.post_of(ring, k))
        else:
            self.boundary(ring, k, after=tasks)

    def tail_before(self, ring: int, tasks) -> None:
        """The ring's last post (made by the time close() runs) comes before every task of
        `tasks` (another poster's first post into the same queue: the engine runs the whole ring first)."""
        self._tail[ring] += list(tasks)

    def close(self) -> None:
        """A boundary() past a ring's last chunk puts the ring's last post before its
        `before` tasks (and its `after` tasks have no run to hold back); so does tail_before()."""
        if self.pusher is None:
            return
        for ring, bounds in enumerate(self._bounds):
            runs = self._runs[ring]
            for k, (before, _after) in bounds.items():
                if k >= len(self.entries[ring]) and runs:
                    for t in before:
                        self.pusher["edge"](runs[-1][0], t)
            bounds.clear()
            for t in self._tail[ring] if runs else ():
                self.pusher["edge"](runs[-1][0], t)
            self._tail[ring] = []

    def start_after(self, node) -> None:
        """Each ring's first push follows `node` (the window's start; another
        poster's last post into the same queue, so the engine runs that first). Calls add up."""
        p = self.pusher
        p["start"].append(node)
        for runs in self._runs:
            if runs:
                p["edge"](node, runs[0][0])

    def _continues(self, ring: int, k: int, src: int, offset: int) -> bool:
        """Whether chunk k of `ring` joins its predecessor's run: at most `batch` chunks (`batch_head` for a run that
        starts within the first `head`), no wrap, the next byte of the same address range; a ring's first run is chunk
        0 alone, and a boundary() starts a run."""
        run = self._run[ring]
        if k in self._bounds[ring]:
            return False                # boundary(): the chunk starts a run
        lim = self.batch
        if run is not None and self.batch_head and k - run[1] < self.head:
            lim = self.batch_head       # the run's first chunk is k - run[1]
        if run is None or run[1] >= lim or k % self.n_slots == 0 or k == 1:
            return False
        p_src, _kind, p_off, p_n = self.entries[ring][k - 1]
        return src + offset == p_src + p_off + p_n

    def _take(self, ring: int, src: int, nbytes: int, offset: int = 0):
        """The next chunk of `ring`: (its place in the ring, its flag, its sequence number, whether it starts a run, its
        first byte in the ring). Chunks of one run sit back to back from the run's first slot, so a run of short chunks
        (a K = 1408 weight: 88 KiB per 64 columns) is still one push; the run never outgrows the slots it reserves, one
        per chunk."""
        if nbytes > self.slot_bytes:
            raise ValueError(f"WeightRings: a {nbytes} B chunk; slots are {self.slot_bytes} B")
        k = len(self.entries[ring])
        src, offset, nbytes = int(src), int(offset), int(nbytes)
        new = not self._continues(ring, k, src, offset)
        if new:
            self._run[ring] = [k % self.n_slots, 0, 0]
        run = self._run[ring]
        rel = run[0] * self.slot_bytes + run[2]
        place = ring * self.n_slots * self.slot_bytes + rel
        run[1] += 1
        run[2] += nbytes
        idx = ring * self.n_slots + k % self.n_slots
        self.entries[ring].append((src, 0, offset, nbytes))
        self.last_take = (ring, k)
        return at_offset(self.slots, place), at_offset(self.flags, idx * 64), k + 1, new, rel

    def load_chunk(self, node, cluster: int, name: str, *, src: int, nbytes: int, slab, war,
                   deps=(), offset: int = 0, dm: int):
        """The ring load of one chunk of ring `cluster` into an L1 slab: an idma_ring_load that waits for the chunk's
        push and copies the chunk from L3 into `slab` on core `dm`. It follows `war` (the slab's last readers,
        LoadStream.take), then the ring's newest load on `cluster` (so the loads run in the ring's order), then `deps`;
        the load becomes the ring's newest and is tagged for the Execution-IR export. `node(name, core, kernel, args,
        deps)` creates a node. A chunk that starts a run makes the run's push, `<name>_push`, on core `dm` too
        (push_from), and the load follows it. Returns (the load, the chunk's place in the ring: the address the load
        reads)."""
        if self.pusher is None:
            raise ValueError("WeightRings.load_chunk: the push engine is not set (push_from)")
        ring = cluster
        sl, fl, seq, new, rel = self._take(ring, src, nbytes, offset)
        prev = [self.last[cluster]] if self.last.get(cluster) is not None else []
        after = list(dict.fromkeys(list(war) + prev + list(deps)))
        after.append(self._push(node, dm, name, ring, new, rel, int(src) + int(offset), nbytes, sl, fl, seq - 1))
        ld = node(name, dm, "__snax_bingo_kernel_idma_ring_load",
                  SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, slab, nbytes, 0, trailer=False), after)
        self.last[cluster] = ld
        ld.exec_chunk = self.last_take
        self._held[ring].append((rel, rel + nbytes, seq - 1, ld))
        runs = self._runs[ring]
        if new:
            runs[-1][1] = ld
        # the scheduler's arrival model: the ring's pushes stream from its first one
        ld.ring_start = runs[0][0]
        return ld, sl

    def _push(self, node, dm, name, ring, new, lo, src, nbytes, slot, flag, k, shared=False):
        """The push of chunk k's run, made when the chunk starts it, else grown by
        the chunk. Its edges: push_from(). `shared` (post_chunk): every last reader of the bytes
        it overwrites, not just the newest (the readers are on more than one cluster)."""
        p, runs = self.pusher, self._runs[ring]
        if new:
            if k + 1 >= p["seq_entries"]:
                raise ValueError(f"WeightRings: chunk {k} of ring {ring}; the sequence table has "
                                 f"{p['seq_entries']} entries")
            deps = [runs[-1][0]] if runs else list(p["start"])
            if k in self._bounds[ring]:
                before, after = self._bounds[ring][k]
                if runs:
                    for t in before:
                        p["edge"](runs[-1][0], t)
                deps += after
            if len(runs) >= p["share"]:
                deps.append(runs[-p["share"]][1])
            deps += self._readers(ring, len(runs), lo, lo + nbytes) if shared else \
                [self._reader(ring, len(runs), lo, lo + nbytes)]
            push = node(f"{name}_push", dm, SnaxBingoKernelEnginePostArgs.KERNEL_NAME,
                        SnaxBingoKernelEnginePostArgs(src, slot, nbytes, at_offset(p["seq"], 64 * (k + 1)), flag,
                                                    1, p["memchip"], p["engine"]),
                        list(dict.fromkeys(d for d in deps if d is not None)))
            runs.append([push, None, k, lo, lo + nbytes])
            self._posts[ring][k] = push
            return push
        push = runs[-1][0]
        push.kernel_args.size += nbytes
        push.kernel_args.chunks += 1
        runs[-1][4] = runs[-1][3] + push.kernel_args.size
        self._posts[ring][k] = push
        for r in (self._readers(ring, len(runs) - 1, lo, lo + nbytes) if shared else
                  [self._reader(ring, len(runs) - 1, lo, lo + nbytes)]):
            if r is not None:
                p["edge"](r, push)
        return push

    def post_chunk(self, node, ring: int, name: str, *, src: int, nbytes: int, dm: int, offset: int = 0):
        """A ring several clusters read (FlashAttention's one ring a chip, both clusters loading every chunk): the
        ring's next chunk and its run's post, `<name>_push` on core `dm` (`node(name, core, kernel, args, deps)` names
        the posting cluster), with no load. Returns ((the chunk's place in the ring, its flag, its sequence number,
        its bytes), the post). Every load of the chunk follows the post and declares itself with read_chunk(); the
        chunk's last reader is where the post that overwrites its bytes comes after."""
        if self.pusher is None:
            raise ValueError("WeightRings.post_chunk: the push engine is not set (push_from)")
        sl, fl, seq, new, rel = self._take(ring, src, nbytes, offset)
        post = self._push(node, dm, name, ring, new, rel, int(src) + int(offset), nbytes, sl, fl, seq - 1,
                          shared=True)
        self._chunk[ring][seq - 1] = (rel, rel + nbytes, len(self._runs[ring]) - 1)
        return (sl, fl, seq, int(nbytes)), post

    def read_chunk(self, ring: int, k: int, load, last: bool) -> None:
        """After post_chunk(): `load` reads chunk k (its sequence number - 1) of
        `ring`. The first load of a run's first chunk is the run's queue-share witness. `last`:
        no other load of the chunk comes after this one (they precede it by edges): the post that
        overwrites the chunk's bytes follows it. A post made already gets that edge here."""
        lo, hi, m = self._chunk[ring][k]
        runs = self._runs[ring]
        load.ring_start = runs[0][0]
        if runs[m][2] == k and runs[m][1] is None:
            runs[m][1] = load
        if not last:
            return
        held = self._held[ring]
        i = len(held)
        while i and held[i - 1][2] > k:
            i -= 1
        held.insert(i, (lo, hi, k, load))
        for later in runs[m + 1:]:
            if later[3] < hi and lo < later[4]:
                self.pusher["edge"](load, later[0])
                break

    def _reader(self, ring, m, lo, hi):
        """The newest load of a chunk with bytes in [lo, hi) of `ring`, which run m
        writes, or None. Its push follows it, and so every older load of the ring: the loads run
        in the ring's order, each after the one before. None too where that load comes before
        run m - share: the queue-share edge orders it already."""
        runs, share = self._runs[ring], self.pusher["share"]
        floor = runs[m - share][2] if m >= share else 0
        for a, b, k, ld in reversed(self._held[ring]):
            if k < floor:
                return None
            if a < hi and lo < b:
                return ld
        return None

    def _readers(self, ring, m, lo, hi):
        """post_chunk's _reader: the last reader of every chunk with bytes in [lo, hi) of `ring`
        down to run m - share (the queue-share edge orders the older ones)."""
        runs, share = self._runs[ring], self.pusher["share"]
        floor = runs[m - share][2] if m >= share else 0
        return [ld for a, b, k, ld in self._held[ring] if k >= floor and a < hi and lo < b]
