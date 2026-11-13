# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Broadcasts and gathers over the D2D fabric, and the rule that keeps them apart.

THE RULE (conflict()). Two broadcasts may not run at once: the later waits until the earlier is done. Two broadcasts
that one memory chip's push engine runs from its queue never overlap (the queue runs in order). Two gathers into
overlapping destination bytes may not run at once either. A broadcast beside a gather, and gathers into disjoint
destinations, may.

THE COLLECTIVES, each a span of tasks: from the task that starts it to the task after which it is done.

    broadcast   an iDMA put multicast from a cluster's L1 into every chip (idma_put_flagged with bcast: one task,
                broadcast_put), or a memory chip's HBM multicast a host queues (its engine posts up to the wait for its
                landing on the posting chip: mark_broadcast)
    gather      a ChainGather, one task on the collector's xDMA (chain_gather): its hops, the chips it passes in
                order, and its destination, the last chain entry on the collector

A broadcast is done after its landing on the posting chip, or after any receiver's wait on its data (a task its tasks
order by an ordering-only edge: a flag wait on what it delivers).

The primitives make the tasks; a broadcast of several tasks is recorded on the graph (mark_broadcast,
dfg.collective_spans), a one-task broadcast or gather is found by its kernel (spans()). The compiler orders them
(passes/bingo_dfg_collectives.py): every two spans that may not run at once get a path of edges, the missing ones added,
and the result is proved, whoever made the tasks."""

import re
from dataclasses import dataclass
from typing import Optional

from bingo_kernel_args import SnaxBingoKernelIdmaPutFlaggedArgs

BCAST_PUT = "__snax_bingo_kernel_idma_put_flagged"
CHAIN_GATHER = "__snax_bingo_kernel_xdma_chain_gather"

# fabric.remote(chip, local): the local address on another chip
_REMOTE = re.compile(r"^\(chiplet_addr_transform_full\(0x([0-9a-fA-F]+), \(\(uint64_t\)\((.*)\) & [0-9a-fA-FxX]+[uUlL]*\)\)\)$")
# a local address: a buffer's C name plus a byte offset
_LOCAL = re.compile(r"^\(\(uint64_t\)\(([A-Za-z_][A-Za-z_0-9\[\]\.]*)\) \+ (\d+)\)$")


@dataclass(eq=False)
class Span:
    """One collective: its kind (broadcast | gather), the task it starts with, what it is done after, and for a
    broadcast its push-engine queue (memory chip, engine; None for a put) and every task of it, for a gather its hops
    ((chip, cluster) in order; None when not known) and destination ((chip, buffer, first byte, end byte); None when
    not known). `end` is a task, or an object that makes it when asked (an HBM cast: its landing wait, `end`, made the
    first time it is read; `_end` until then)."""
    kind: str
    first: object
    end: object
    hops: Optional[list] = None
    queue: object = None
    tasks: Optional[list] = None
    dst: Optional[tuple] = None

    @property
    def name(self):
        return self.first.node_name

    def done(self):
        """The task after which the collective is done, or None while that task is not made."""
        e = self.end
        return e._end if hasattr(e, "_end") else e

    def resolve(self):
        """The task after which the collective is done, made now if it is not yet."""
        e = self.end
        return e.end if hasattr(e, "_end") else e


def _spans(dfg):
    if not hasattr(dfg, "collective_spans"):
        dfg.collective_spans = []
    return dfg.collective_spans


def mark_broadcast(dfg, first, end=None, queue=None, tasks=None):
    """Record a broadcast from `first` (the task that starts it) to `end` (the task after which it has landed on the
    posting chip, or an object that makes that task when its `end` is read; first when omitted). `queue`: the push
    engine's queue it runs from (memory chip, engine), `tasks` every task of it (its posts). Returns the span."""
    s = Span("broadcast", first, end if end is not None else first, queue=queue, tasks=list(tasks or [first]))
    _spans(dfg).append(s)
    return s


def broadcast_put(add, name, args, after=()):
    """An iDMA put multicast from L1 into every chip (`args` an SnaxBingoKernelIdmaPutFlaggedArgs with bcast), made by
    `add(name, kernel, args, deps)` on the source cluster's DM core. Returns the task."""
    if not isinstance(args, SnaxBingoKernelIdmaPutFlaggedArgs) or not getattr(args, "bcast", False):
        raise ValueError(f"broadcast_put {name}: an idma_put_flagged multicast (bcast) only")
    return add(name, BCAST_PUT, args, list(after))


def chain_gather(add, name, args, after=()):
    """One ChainGather, `add(name, kernel, args, deps)` making its task on the collector's xDMA core. Returns the task
    the gather's output is ready after and its operands are free after. With the DFG's async_exec the compiler splits
    it into a configure, a start and a wait (passes/bingo_dfg_schedule.py), the wait keeping the gather's name."""
    return add(name, CHAIN_GATHER, args, list(after))


def gather_hops(args):
    """The (chiplet, cluster) of every chain entry of a ChainGather's args: its `hops`, or each entry's own L1
    allocation. None when an entry names neither."""
    if getattr(args, "hops", None) is not None:
        return list(args.hops)
    hops = []
    for e in getattr(args, "chain", None) or ():
        base = getattr(e, "base", e)
        if getattr(base, "mem_level", None) != "L1" or not hasattr(base, "cluster_id"):
            return None
        hops.append((base.chip_id, base.cluster_id))
    return hops or None


def gather_dst(args):
    """A ChainGather's destination, its last chain entry: (collector chip, the buffer, first byte, end byte), or None
    when the entry is neither fabric.remote() of a buffer and offset nor an L1 allocation."""
    chain, size = getattr(args, "chain", None), int(getattr(args, "size", 0) or 0)
    if not chain or not size:
        return None
    e = chain[-1]
    if isinstance(e, str):
        m = _REMOTE.match(e)
        if not m:
            return None
        chip, local = int(m.group(1), 16), m.group(2).strip()
        lm = _LOCAL.match(local)
        name, off = (lm.group(1), int(lm.group(2))) if lm else (local, 0)
        return (chip, name, off, off + size)
    base = getattr(e, "base", e)
    if getattr(base, "mem_level", None) != "L1":
        return None
    off = int(getattr(e, "offset", 0)) if e is not base else 0
    return (getattr(base, "chip_id", None), ("l1", id(base)), off, off + size)


def conflict(a, b):
    """Whether spans a and b may not run at once (THE RULE)."""
    if a.kind == "broadcast" and b.kind == "broadcast":
        return a.queue is None or a.queue != b.queue
    if a.kind == "gather" and b.kind == "gather":
        if a.dst is None or b.dst is None:
            return True
        return a.dst[:2] == b.dst[:2] and a.dst[2] < b.dst[3] and b.dst[2] < a.dst[3]
    return False


def spans(dfg):
    """Every collective of the graph: the recorded spans, then each multicast put and ChainGather task no recorded span
    holds (found by its kernel), in node order."""
    out = list(_spans(dfg))
    held = {t for s in out for t in (s.tasks or []) + [s.first, s.done()] if t is not None}
    for t in sorted(dfg.nodes(), key=lambda n: n.node_id):
        if t in held:
            continue
        k = t.kernel_name or ""
        if k == CHAIN_GATHER:
            out.append(Span("gather", t, t, gather_hops(t.kernel_args), dst=gather_dst(t.kernel_args)))
        elif k == BCAST_PUT and getattr(t.kernel_args, "bcast", False):
            out.append(Span("broadcast", t, t, tasks=[t]))
    return out
