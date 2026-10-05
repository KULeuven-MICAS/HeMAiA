# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Collectives across chips: all-gather, send (one cluster to another) and scatter.

A COLLECTIVE IS A SMALL GRAPH OF BLOCKS, NOT ONE BLOCK. It runs on many clusters of many
chips, and its tasks interleave with everything else those cores run: a producer's stash goes
right after the producer, a destination's read goes wherever its edges allow. So it is built
the way a layer is -- blocks added to a Pipeline, each on its own chip -- by a function, as
fa_gather is. What it returns names every stage it added (`Collective`), because a layer
orders other work around them (a weight stream that must not load ahead of a stash, a slot
that waits for the previous expert's gather).

ONE INTERFACE, SEVERAL IMPLEMENTATIONS. Each collective takes `impl`, and every
implementation produces the same outputs -- the same layout in the same buffers' places -- so
the choice never changes what a consumer binds to:

  "idma"   the baseline, safe by construction. A producer's DM core copies its part into
           its OWN chip's copy of a staged hand-off array (a local write); a destination's
           DM core reads every chip's run of it (a remote read: only a read issued after the
           producer's own write proves the bytes are there -- a D2D write is acknowledged
           when it LEAVES the sender). Options: `chain` (a chip's stashes ordered, so a read
           waits for the chip's last one only: one dependency tag per chip), `tree` (a
           two-level gather: groups of `tree` chips gather onto their first chip, every
           destination reads one run per group), `hier` (per chip only the first destination
           reads across the links; the chip's others copy its result, L1 to L1).

  "bcast"  all-gather only. Each producer's DM core writes its part into its own chip's copy
           of the hand-off array and to the D2D broadcast address (chip 0xFF: every OTHER
           compute chip receives it at the same local address; the router never delivers a
           broadcast back to its source), then a landed flag the same two ways, on the same
           iDMA, so a landed flag says its part has landed. A destination's DM core spins
           until every producer's flag has landed in its OWN chip's memory and copies the
           whole array from there: no read crosses a link. The destinations wait for the
           producers by those flags, not by dependencies -- ordering-only edges
           (bingo_add_order_edge) that order the queues and the dispatch stream and that the
           hang check proves acyclic, and that spend no dependency tag. One use per run: a
           flag is never cleared. Every destination must also be a producer (its own part
           orders its read after its write).

Still to come behind the same call: "chain" (the xDMA chain gather, for reductions: a fold
along adjacent hops onto one collector), then "auto", a cost model that picks among the
legal ones. Until they exist, asking for one raises.

THE APPLICATION'S SIDE (`Fabric`): the chips taking part and their clusters, a way to add a
block on a given cluster's or chip's context, and a staged hand-off array present at the same
local address on every chip (named `xchg_prefix` + the collective's name).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Protocol

import numpy as np

from bingo_kernel_args import (SnaxBingoKernelIdmaBcastPutArgs,
                               SnaxBingoKernelIdmaFetchFlaggedArgs)
from bingo_mem_handle import BingoMemSymbol

from ..comm import Block, BlockResult, Ctx, DType, Layout, MemLevel, Port, PortSpec
from .move import After, Collect, Join, Pull, Stash, View

RM, I8 = Layout.ROW_MAJOR, DType.I8
L1, L3 = MemLevel.L1, MemLevel.L3


class Fabric(Protocol):
    """What a collective needs from the application.

    Clusters are numbered globally (g); `chips` lists the chips in order, and a chip's
    clusters are consecutive in g."""
    chips: list
    xchg_prefix: str                                        # hand-off arrays' name prefix

    def chip(self, g: int) -> int: ...                      # the chip of cluster g

    def c(self, g: int) -> int: ...                         # g's index on its chip

    def gs(self, k: int) -> list: ...                       # chip k's clusters, in order

    def add(self, block, name: str, at_g: int = None, at_chip: int = None, **binds): ...

    def xsym(self, name: str, nbytes: int) -> str: ...      # a staged array on every chip

    pipe: object                                            # the Pipeline (impl "bcast")
    dfg: object                                             # the BingoDFG (impl "bcast")


@dataclass
class Collective:
    """What a collective added: `out` {destination g: the stage holding its result}, `put`
    {producer g: its stash}, `get` {destination g: its read (a collect, or the copy from its
    chip's first destination)}."""
    impl: str
    out: Dict[int, object] = field(default_factory=dict)
    put: Dict[int, object] = field(default_factory=dict)
    get: Dict[int, object] = field(default_factory=dict)


def _l3(spec: PortSpec) -> PortSpec:
    """`spec`'s tensor in L3: what a Stash of it outputs."""
    return PortSpec(spec.layout, spec.dtype, spec.shape, mem_level=L3)


def _xsym(fab, name: str, nbytes: int) -> str:
    """The hand-off array `name`, staged on every chip."""
    return fab.xsym(f"{getattr(fab, 'xchg_prefix', 'xchg_')}{name}", nbytes)


def _impl(name: str, impl: str, have: tuple) -> str:
    if impl not in have:
        raise ValueError(f"{name}: impl={impl!r}; implemented: {', '.join(have)}.")
    return impl


# ======================================================================================
# all-gather
# ======================================================================================

def all_gather(fab: Fabric, name: str, parts: dict, part_spec: Callable, part_bytes,
               dsts: List[int], dst_spec_of: Callable, *, chips: list = None,
               chain: bool = False, tree: int = 0, hier: bool = False,
               impl: str = "idma") -> Collective:
    """Every cluster in `dsts` gets the parts of every cluster of `chips` (default: all),
    back to back in global cluster order.

    `parts` {g: port}; `part_spec(g)` the spec of g's part; `part_bytes` one count, or one
    per cluster in order; `dst_spec_of(g)` what destination g receives."""
    _impl("all_gather", impl, ("idma", "bcast"))
    if impl == "bcast":
        return _all_gather_bcast(fab, name, parts, part_spec, part_bytes, dsts, dst_spec_of,
                                 chips)
    return _all_gather_idma(fab, name, parts, part_spec, part_bytes, dsts, dst_spec_of,
                            chips, chain, tree, hier)


def _all_gather_idma(fab, name, parts, part_spec, part_bytes, dsts, dst_spec_of, chips,
                     chain, tree, hier) -> Collective:
    res = Collective("idma")
    chips = list(fab.chips) if chips is None else list(chips)
    gl = [g for k in chips for g in fab.gs(k)]
    pbl = [int(part_bytes)] * len(gl) if isinstance(part_bytes, (int, np.integer)) else \
        [int(v) for v in part_bytes]
    pb = dict(zip(gl, pbl))
    at = {g: sum(pbl[:i]) for i, g in enumerate(gl)}
    sym = _xsym(fab, name, sum(pbl))
    runs, run_bytes = {}, {}
    # `chain`: a chip's stashes in cluster order, each after the one before (an ordering
    # edge, no data), so a read of the chip's run needs only the last one -- the others are
    # implied (with dfg.prune_implied the compiler drops their edges): one dependency tag per
    # chip in the reader's cell instead of one per cluster.
    for k in chips:
        st_k = []
        for g in fab.gs(k):
            h = BingoMemSymbol(sym, at[g], chip_id=k)
            x = parts[g]
            if chain and st_k:
                gp = fab.gs(k)[len(st_k) - 1]
                x = fab.add(After(x=part_spec(g), after=_l3(part_spec(gp))),
                            f"xord_{name}_g{g}", g, x=x, after=st_k[-1].out()).out()
            st_k.append(fab.add(Stash(src=part_spec(g), nbytes=pb[g], dst=h),
                                f"xput_{name}_g{g}", g, x=x))
            res.put[g] = st_k[-1]
        run_bytes[k] = sum(pb[g] for g in fab.gs(k))
        runs[k] = fab.add(
            Join(parts=[_l3(part_spec(g)) for g in fab.gs(k)],
                 dst=PortSpec(RM, I8, (1, run_bytes[k]), mem_level=L3)),
            f"xrun_{name}_k{k:02x}", at_chip=k,
            **{f"x{i}": s.out() for i, s in enumerate(st_k)})
    # A chip's clusters are consecutive in g, so its run is one contiguous range of the
    # array: collecting the runs in chip order lays the parts out in g order.
    #
    # `tree` (a group size): with more chips than that, a two-level gather. The chips, in
    # order, form groups of `tree`; each group's first chip (its first cluster) collects the
    # group's runs and stashes them -- one contiguous range of a second staged array -- into
    # its own L3, and every destination reads one range per group. A reader then waits for
    # each group's leader instead of every chip: one dependency per group, where a flat
    # gather over twelve chips needs more dependency tags than the manager has once two
    # gathers overlap.
    src_chips, src_runs, src_bytes = chips, runs, run_bytes
    if tree and len(chips) > tree:
        tsym = _xsym(fab, f"{name}_t", sum(pbl))
        src_chips, src_runs, src_bytes = [], {}, {}
        for i in range(0, len(chips), tree):
            grp = chips[i: i + tree]
            lead, gl0 = grp[0], fab.gs(grp[0])[0]
            nb = sum(run_bytes[q] for q in grp)
            raw = PortSpec(RM, I8, (1, nb), mem_level=L1, cluster=fab.c(gl0))
            col = fab.add(Collect(parts=[PortSpec(RM, I8, (1, run_bytes[q]), mem_level=L3)
                                         for q in grp],
                                  nbytes=[run_bytes[q] for q in grp], dst=raw,
                                  cluster=fab.c(gl0)),
                          f"xgrp_{name}_k{lead:02x}", gl0,
                          **{f"x{j}": runs[q].out() for j, q in enumerate(grp)})
            st = fab.add(Stash(src=raw, nbytes=nb,
                               dst=BingoMemSymbol(tsym, at[gl0], chip_id=lead)),
                         f"xgput_{name}_k{lead:02x}", gl0, x=col.out())
            src_runs[lead] = fab.add(
                Join(parts=[_l3(raw)], dst=PortSpec(RM, I8, (1, nb), mem_level=L3)),
                f"xgrun_{name}_k{lead:02x}", at_chip=lead, x0=st.out())
            src_chips.append(lead)
            src_bytes[lead] = nb
    #
    # `hier`: per chip only its first destination cluster reads across the links, and the
    # chip's other destinations copy that cluster's result over (a Pull, L1 to L1 on the
    # chip). A read across a link waits for whatever the memory chiplet is pushing on it, so
    # halving them halves those waits; and the second cluster then waits on one local edge
    # instead of one per producing cluster.
    first_on = {}
    for g in dsts:
        k, spec = fab.chip(g), dst_spec_of(g)
        if hier and k in first_on:
            # the destination as raw bytes (an A-layout row is a whole 16-row block, which
            # Pull will not cut), moved in one copy, then named again
            g0, tot = first_on[k], sum(pbl)
            raw = lambda c: PortSpec(RM, I8, (1, tot), mem_level=L1, cluster=c)
            src = fab.add(View(src=dst_spec_of(g0), dst=raw(fab.c(g0))),
                          f"xraw_{name}_g{g}", g0, x=res.out[g0].out())
            cp = fab.add(Pull(rows=1, cols=tot, layout=RM, dtype=I8, src=fab.c(g0),
                              dst=fab.c(g)), f"xget_{name}_g{g}", g, x0=src.out())
            res.get[g] = cp
            res.out[g] = fab.add(View(src=raw(fab.c(g)), dst=spec), f"xnam_{name}_g{g}", g,
                                 x=cp.out())
            continue
        first_on.setdefault(k, g)
        res.out[g] = res.get[g] = fab.add(
            Collect(parts=[PortSpec(RM, I8, (1, src_bytes[q]), mem_level=L3)
                           for q in src_chips],
                    nbytes=[src_bytes[q] for q in src_chips], dst=spec, cluster=fab.c(g)),
            f"xget_{name}_g{g}", g,
            **{f"x{i}": src_runs[q].out() for i, q in enumerate(src_chips)})
    return res


FLAG_PITCH = 64       # one flag slot per producer, a beat apart


class BcastPut(Block):
    """A producer's part, out to every compute chip, with its landed flag (impl "bcast").

      in   x  the part, in cluster `cluster`'s L1
      out  y  the part in L3, at its place in the hand-off array (this chip's copy)

    One DM task: the part into this chip's copy of `dst` and to the same address on every
    other compute chip (the D2D broadcast), then `value` into `flag` the same two ways."""

    name = "bcast_put"

    def __init__(self, *, src: PortSpec, nbytes: int, dst, flag, cluster: int, value: int = 1):
        if src.mem_level != L1 or src.cluster != cluster:
            raise ValueError(f"BcastPut: {src.describe()} is not cluster {cluster}'s L1.")
        self.src, self.nbytes, self.dst, self.flag = src, int(nbytes), dst, flag
        self.cluster, self.value = int(cluster), int(value)

    @property
    def inputs(self) -> dict:
        return {"x": self.src}

    @property
    def outputs(self) -> dict:
        return {"y": _l3(self.src)}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        g = ctx.at(self.cluster)
        x = bound["x"]
        nd = g.node("BcastPut", ctx.dm, "__snax_bingo_kernel_idma_bcast_put",
                    SnaxBingoKernelIdmaBcastPutArgs(x.handle, self.dst, self.nbytes, self.flag,
                                                    self.value), list(x.ends))
        return BlockResult(outputs={"y": Port(self.outputs["y"], self.dst, (nd,), name="y")},
                           inputs={"x": Port(self.src, x.handle, (nd,), name="x")},
                           nodes=[nd], sources=[] if x.ends else [nd])


class FlagFetch(Block):
    """A destination's read of a broadcast collective (impl "bcast"): spin until `n_flags`
    flags in this chip's memory hold `value`, then copy `nbytes` from `src` (this chip's copy
    of the hand-off array) into this cluster's L1.

      in   after  this cluster's own BcastPut (orders the read after its own write)
      out  y      `dst`"""

    name = "flag_fetch"

    def __init__(self, *, src, nbytes: int, flags, n_flags: int, after: PortSpec,
                 dst: PortSpec, cluster: int, value: int = 1):
        if dst.mem_level != L1 or dst.cluster != cluster:
            raise ValueError(f"FlagFetch: {dst.describe()} is not cluster {cluster}'s L1.")
        self.src, self.nbytes, self.flags, self.n_flags = src, int(nbytes), flags, int(n_flags)
        self.after, self.dst, self.cluster, self.value = after, dst, int(cluster), int(value)

    @property
    def inputs(self) -> dict:
        return {"after": self.after}

    @property
    def outputs(self) -> dict:
        return {"y": self.dst}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        g = ctx.at(self.cluster)
        a = bound["after"]
        buf = g.l1(f"{self.name}_y", self.nbytes)
        nd = g.node("FlagFetch", ctx.dm, "__snax_bingo_kernel_idma_fetch_flagged",
                    SnaxBingoKernelIdmaFetchFlaggedArgs(self.src, buf, self.nbytes, self.flags,
                                                        self.n_flags, self.value),
                    list(a.ends))
        return BlockResult(outputs={"y": Port(self.dst, buf, (nd,), name="y")},
                           inputs={"after": Port(self.after, a.handle, (nd,), name="after")},
                           nodes=[nd], sources=[] if a.ends else [nd])


def _all_gather_bcast(fab, name, parts, part_spec, part_bytes, dsts, dst_spec_of,
                      chips) -> Collective:
    res = Collective("bcast")
    chips = list(fab.chips) if chips is None else list(chips)
    gl = [g for k in chips for g in fab.gs(k)]
    pbl = [int(part_bytes)] * len(gl) if isinstance(part_bytes, (int, np.integer)) else \
        [int(v) for v in part_bytes]
    pb = dict(zip(gl, pbl))
    at = {g: sum(pbl[:i]) for i, g in enumerate(gl)}
    missing = [g for g in dsts if g not in pb]
    if missing:
        raise ValueError(f"all_gather {name!r}, impl bcast: destinations {missing} hold no "
                         f"part; every destination must be a producer.")
    sym = _xsym(fab, name, sum(pbl))
    fsym = _xsym(fab, f"{name}_f", FLAG_PITCH * len(gl))
    for i, g in enumerate(gl):
        res.put[g] = fab.add(BcastPut(src=part_spec(g), nbytes=pb[g],
                                      dst=BingoMemSymbol(sym, at[g]),
                                      flag=BingoMemSymbol(fsym, FLAG_PITCH * i),
                                      cluster=fab.c(g)),
                             f"xbput_{name}_g{g}", g, x=parts[g])
    for g in dsts:
        res.out[g] = res.get[g] = fab.add(
            FlagFetch(src=BingoMemSymbol(sym, 0), nbytes=sum(pbl),
                      flags=BingoMemSymbol(fsym, 0), n_flags=len(gl),
                      after=_l3(part_spec(g)), dst=dst_spec_of(g), cluster=fab.c(g)),
            f"xbget_{name}_g{g}", g, after=res.put[g].out())

    def order():
        for g in dsts:
            rd = res.get[g].result.nodes[0]
            for p in gl:
                if p != g:
                    fab.dfg.bingo_add_order_edge(res.put[p].result.nodes[-1], rd)
    fab.pipe.raw(order, f"xborder_{name}")
    return res


# ======================================================================================
# send: one cluster's parts to another cluster
# ======================================================================================

def send(fab: Fabric, name: str, g_src: int, parts: list, part_spec: PortSpec,
         part_bytes: int, g_dst: int, dst_spec: PortSpec, *, impl: str = "idma"):
    """Cluster g_src's `parts` (ports of `part_spec`, `part_bytes` each) into cluster g_dst's
    L1, back to back. Returns the destination's read: its output is the result."""
    _impl("send", impl, ("idma",))
    # stashed into g_src's chip's copy of a staged array, read by g_dst's DM core in one copy
    # (local write, remote read, as for every hand-off)
    k = fab.chip(g_src)
    tot = part_bytes * len(parts)
    sym = _xsym(fab, name, tot)
    st = [fab.add(Stash(src=part_spec, nbytes=part_bytes,
                        dst=BingoMemSymbol(sym, i * part_bytes, chip_id=k)),
                  f"xput_{name}_{i}", g_src, x=pt)
          for i, pt in enumerate(parts)]
    run = fab.add(Join(parts=[_l3(part_spec)] * len(parts),
                       dst=PortSpec(RM, I8, (1, tot), mem_level=L3)),
                  f"xrun_{name}", at_chip=k, **{f"x{i}": s.out() for i, s in enumerate(st)})
    return fab.add(Collect(parts=[PortSpec(RM, I8, (1, tot), mem_level=L3)], nbytes=[tot],
                           dst=dst_spec, cluster=fab.c(g_dst)),
                   f"xget_{name}", g_dst, x0=run.out())


# ======================================================================================
# scatter: one cluster's buffer, one part per destination
# ======================================================================================

def scatter(fab: Fabric, name: str, g_src: int, port, spec: PortSpec, part_bytes: list,
            dsts: List[int], dst_spec_of: Callable, *, impl: str = "idma") -> Collective:
    """Cluster g_src's buffer, one part per destination: `part_bytes[i]` bytes at the offset
    of the parts before it, in `dsts` order, to destination dsts[i]."""
    _impl("scatter", impl, ("idma",))
    # g_src stashes the whole buffer into its chip's copy of a staged array, and each
    # destination reads its part
    res = Collective("idma")
    k0 = fab.chip(g_src)
    pb = [int(v) for v in part_bytes]
    sym = _xsym(fab, name, sum(pb))
    s = fab.add(Stash(src=spec, nbytes=sum(pb), dst=BingoMemSymbol(sym, chip_id=k0)),
                f"xput_{name}", g_src, x=port)
    res.put[g_src] = s
    at = 0
    for g, n in zip(dsts, pb):
        part = fab.add(View(src=_l3(spec), dst=PortSpec(RM, I8, (1, n), mem_level=L3),
                            offset=at),
                       f"xpart_{name}_g{g}", at_chip=k0, x=s.out())
        res.out[g] = res.get[g] = fab.add(
            Collect(parts=[PortSpec(RM, I8, (1, n), mem_level=L3)], nbytes=[n],
                    dst=dst_spec_of(g), cluster=fab.c(g)),
            f"xget_{name}_g{g}", g, x0=part.out())
        at += n
    return res
