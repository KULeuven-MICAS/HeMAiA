# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Choosing the orientation of every step in a chain, instead of pinning it by hand.

A per-row SIMD operator is three times cheaper in one orientation than the other -- the
reduction lands along the beats instead of across the lanes -- but reaching that
orientation costs xDMA passes, and whether those passes are real depends on what the
NEIGHBOURS do, not on the operator. A block cannot see its neighbours: `inputs` and
`outputs` are a SIGNATURE, read before anything is bound. So the choice does not belong
inside the block, and this is where it goes instead.

======================================================================================
WHY IT IS A PLANNER AND NOT A REWRITE
======================================================================================

Pipeline.add BUILDS EAGERLY, on purpose -- node creation order is dispatch order on this
machine, so a pass that reordered or re-emitted after the fact would silently move the
schedule. That rules out "build it, then optimise it". This pass therefore runs BEFORE
anything is added: it is handed the candidate configurations, it prices them, and it
returns the parameters to construct with. Nothing it does touches a graph.

That also makes LEGALITY free. Each candidate is constructed to price it, so a
configuration the block refuses -- RMSNorm at rows != 32, a Reshape whose conversion is
not a strided nest -- raises in its own constructor with its own message, and the pass
simply drops it from the search. There is no second copy of the rules here to drift.

======================================================================================
THE OBJECTIVE IS THE SIMD, AND THAT IS A CHOICE
======================================================================================

Cost is compared LEXICOGRAPHICALLY, in engine order: serialised SIMD work, then xDMA
passes, then iDMA passes. That ordering is the premise. The SIMD core is this machine's
constraint and the whole reason the pass exists; the xDMA is next because the relayouts
of the surrounding chain queue on it; the iDMA is last because the DM core is idle while
the other two work, so a plain copy put there is very nearly free -- which is exactly why
blocks are expected to put every non-transposing move on it.

Minimising total passes instead would be wrong, and the layer's first RMSNorm is the
counter-example: the row-major arm is ONE pass and the col-major arm is TWO, and the
col-major arm is the one worth having because it takes 2,062 cycles off the busy engine
to put one extra transfer on an idle one.

WHAT THE NUMBERS ARE, and what they are not. The SIMD quantity is SERIALISED CROSS-LANE
FOLDS, reported by the blocks themselves via an optional `simd_folds()`. It is a COUNT,
derived from the shape, of the one thing the orientation actually changes: StreamReduce
never moves sideways, so a reduction along beats is free and a reduction across the lanes
of a beat is a log-depth fold that holds the reader's port low until it drains.

THIS IS NOT A COST MODEL AND DELIBERATELY SO. Cycle numbers would be constants correct
for one shape on one RTL build, with nothing to stop them rotting into a model nobody
re-measures. A fold count cannot rot. It does not predict a runtime, it orders the arms,
and ordering is all a planner needs. A block that implements neither hook contributes
nothing and is ranked on passes alone.

======================================================================================
WHAT IT SEARCHES
======================================================================================

A chain of steps, each with one or more candidate parameter dicts. It is a shortest path
over (step, candidate), where the edge from one candidate to the next is whatever
comm.transfer.plan says it takes to get from the first one's output layout to the second
one's input layout -- including "that is impossible", which simply removes the edge. Two
candidates per step over a handful of steps, so an exact DP costs nothing and there is no
reason to be clever.

`start` and `end` pin the ends: the layout the chain is handed (an L3 staging choice, and
staging a tensor transposed is free because the host writes it either way) and the layout
the consumer demands (A, for a GEMM). Leave either None to let the chain choose it.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from .ports import DType, Layout, MemLevel, PortSpec
from .transfer import plan

_INF = float("inf")


@dataclass
class Step:
    """One stage of the chain, and the configurations that could fill it.

    `make` is called with each dict in `options` and must raise on an illegal one -- which
    is what every block here already does, so nothing needs a new method to take part.
    """

    name: str
    make: Callable
    options: list
    in_port: str = "x"
    out_port: str = "y"


@dataclass(frozen=True)
class Candidate:
    """A constructed, legal configuration, with the layouts it actually declares."""

    params: dict
    src: Layout
    dst: Layout
    simd_folds: int = 0
    xdma_passes: int = 0
    idma_passes: int = 0


@dataclass
class LayoutPlan:
    """What the pass decided, and enough of why to argue with it."""

    chosen: list = field(default_factory=list)          # one params dict per step
    candidates: list = field(default_factory=list)      # per step, every legal Candidate
    links: list = field(default_factory=list)           # conversion steps per boundary
    simd_folds: int = 0
    passes: int = 0
    idma_passes: int = 0

    def params(self, name_or_index):
        """The chosen parameters for a step, by name or position."""
        if isinstance(name_or_index, int):
            return self.chosen[name_or_index]
        for i, s in enumerate(self._names):
            if s == name_or_index:
                return self.chosen[i]
        raise KeyError(f"no step named {name_or_index!r}; have {self._names}.")


def assign_layouts(steps, *, mesh, elem_bytes, shape, start: Optional[Layout] = None,
                   end: Optional[Layout] = None, dtype: DType = DType.F16,
                   verbose: bool = True) -> LayoutPlan:
    """Pick one candidate per step so the SIMD core does the least serialised work.

    Returns a LayoutPlan whose `chosen[i]` is the parameter dict to construct step i with.
    Builds nothing -- see the module docstring for why it cannot.
    """
    # ---- enumerate what is legal, by construction ------------------------------------
    cands = []
    for st in steps:
        legal = []
        for params in st.options:
            try:
                blk = st.make(**params)
            except (ValueError, KeyError) as e:
                if verbose:
                    print(f"[layout] {st.name}: {_brief(params)} is not legal -- "
                          f"{_one_line(e)}")
                continue
            try:
                src = blk.inputs[st.in_port].layout
                dst = blk.outputs[st.out_port].layout
            except KeyError:
                raise ValueError(
                    f"[layout] {st.name}: no port {st.in_port!r}/{st.out_port!r} on "
                    f"{type(blk).__name__}. Set in_port/out_port on the Step -- the pass "
                    f"has to read the layouts off the block, not guess them.")
            # PASSES THE BLOCK ITSELF EMITS. Counting only the links would score a block
            # that transposes on BOTH its ports the same as one that transposes on
            # neither, which is the entire decision this pass exists to make.
            legal.append(Candidate(params, src, dst,
                                   int(getattr(blk, "simd_folds", int)() or 0),
                                   int(getattr(blk, "xdma_passes", int)() or 0),
                                   int(getattr(blk, "idma_passes", int)() or 0)))
        if not legal:
            raise ValueError(
                f"[layout] {st.name}: not one of its {len(st.options)} candidate "
                f"configurations is legal at this shape. The messages above are the "
                f"blocks' own; fix the shape or widen the options.")
        cands.append(legal)

    # ---- the cost of getting between two layouts -------------------------------------
    def gap(a: Optional[Layout], b: Optional[Layout]):
        """(number of xDMA passes, the step kinds) or (None, reason) when impossible."""
        if a is None or b is None or a == b:
            return 0, []
        pa = PortSpec(a, dtype, shape, mem_level=MemLevel.L1)
        pb = PortSpec(b, dtype, shape, mem_level=MemLevel.L1)
        try:
            got = plan(pa, pb, mesh=mesh, elem_bytes=elem_bytes)
        except ValueError as e:
            return None, _one_line(e)
        return len(got), [s.kind for s in got]

    # ---- shortest path over (step, candidate) ----------------------------------------
    # best[j] = (folds, passes, backpointer, the link taken INTO this candidate)
    best = []
    for j, c in enumerate(cands[0]):
        n, kinds = gap(start, c.src)
        if n is None:
            continue
        best.append(((c.simd_folds, n + c.xdma_passes, c.idma_passes), j, -1, kinds))
    if not best:
        raise ValueError(
            f"[layout] nothing can consume the chain's input layout {start}. Every "
            f"candidate for {steps[0].name} needs a conversion that is not expressible.")
    frontier = {b[1]: b for b in best}
    trail = [dict(frontier)]

    for i in range(1, len(cands)):
        nxt = {}
        for j, c in enumerate(cands[i]):
            for pj, (pk, _, _, _) in frontier.items():
                n, kinds = gap(cands[i - 1][pj].dst, c.src)
                if n is None:
                    continue
                key = (pk[0] + c.simd_folds, pk[1] + n + c.xdma_passes,
                       pk[2] + c.idma_passes)
                cur = nxt.get(j)
                if cur is None or key < cur[0]:
                    nxt[j] = (key, j, pj, kinds)
        if not nxt:
            raise ValueError(
                f"[layout] the chain cannot reach {steps[i].name}: no legal configuration "
                f"of it accepts anything {steps[i - 1].name} can produce.")
        frontier = nxt
        trail.append(dict(frontier))

    # ---- close the far end and pick the winner ---------------------------------------
    finals = []
    for j, (pk, _, pj, kinds) in frontier.items():
        m, mk = gap(cands[-1][j].dst, end)
        if m is None:
            continue
        finals.append(((pk[0], pk[1] + m, pk[2]), j, mk))
    if not finals:
        raise ValueError(
            f"[layout] no configuration of {steps[-1].name} can reach the chain's "
            f"required output layout {end}.")
    key, j, tail_kinds = min(finals, key=lambda t: t[0])

    # walk the backpointers
    order, links = [], [tail_kinds]
    for i in range(len(cands) - 1, -1, -1):
        order.append(j)
        _, _, pj, kinds = trail[i][j]
        links.append(kinds)
        j = pj
    order.reverse()
    links.reverse()

    out = LayoutPlan(chosen=[cands[i][k].params for i, k in enumerate(order)],
                     candidates=cands, links=links, simd_folds=key[0], passes=key[1],
                     idma_passes=key[2])
    out._names = [s.name for s in steps]
    if verbose:
        _report(steps, cands, order, links, out)
    return out


def _report(steps, cands, order, links, res):
    w = max(len(s.name) for s in steps)
    print("[layout] objective: fewest cross-lane folds, then xDMA passes, then iDMA"
          "  (shown as <x>x/<i>i)")
    for i, st in enumerate(steps):
        inc = links[i]
        if inc:
            print(f"[layout] {'':{w}}   ..link.. {' -> '.join(inc)}")
        for k, c in enumerate(cands[i]):
            mark = "->" if k == order[i] else "  "
            cost = (f"{c.simd_folds:>5,} folds" if c.simd_folds
                    else f"{'no folds':>11}")
            own = f"{c.xdma_passes}x/{c.idma_passes}i"
            print(f"[layout] {mark} {st.name:{w}}  {str(c.src):>9} -> {str(c.dst):<9} "
                  f"{cost}  {own:>8}   {_brief(c.params)}")
    if links[-1]:
        print(f"[layout] {'':{w}}   ..link.. {' -> '.join(links[-1])}")
    print(f"[layout] total: {res.simd_folds:,} cross-lane fold(s), {res.passes} xDMA "
          f"pass(es), {res.idma_passes} iDMA -- blocks' own plus the links between them")


def _brief(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "(defaults)"


def _one_line(e) -> str:
    return " ".join(str(e).split())[:96]
