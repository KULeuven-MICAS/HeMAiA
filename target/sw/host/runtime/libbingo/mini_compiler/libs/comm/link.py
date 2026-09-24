# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Connect blocks, resolve the boundaries between them, then let each block build itself.

THREE PHASES, AND THE MIDDLE ONE IS THE POINT.

    connect   `add` records a block and what feeds it. Nothing is constructed, nothing is
              built, and no layout has been decided.
    resolve   `run` asks every block what it COULD be (comm.variant), then picks one
              realisation per block so that each boundary agrees, at the least cost.
    build     each block emits its own sub-DFG for the realisation it was given, in the
              order it was added.

Splitting connect from build is what makes the middle phase possible at all: a block's
boundary is decided by its NEIGHBOURS, and until the chain is connected there are no
neighbours to ask.

COMPOSITION IS THE APPLICATION'S. The chain is exactly the blocks it added, in the order
it added them: a Reshape it wrote is built, a Quantize it wrote is built. Nothing is
inserted behind its back and nothing it asked for is optimised away, because which stages
a layer has is a statement about the layer, not something a cost model votes on.

    add(RMSNorm(rows, cols), "n1", x=x)
    add(Reshape(rows, cols, mesh), "to_a", x=n1)
    add(Quantize(rows, cols, scale), "q", x=rs)
    add(Linear(...), "qkv", x=q)
    run()

BOUNDARIES ARE NOT. `src`, `dst`, `layout` -- every layout field a block leaves unset
follows from what its neighbours turned out to be, so the application does not have to
work them out and keep them consistent, nor know which kernel each block will pick in
order to write the right thing around it. A field it DOES pin is a constraint every
realisation has to meet: that is how it keeps hold of a boundary it cares about, and
`demand()` is the same thing for the far end.

PLACEMENT IS THE APPLICATION'S TOO, AND IS NEVER RESOLVED. Which cluster a block runs on
is stated on its cfg and declared on its ports, and this file only CHECKS it. It is not a
knob the resolver turns, because `Cost` has one column per engine and no notion of which
cluster that engine is in: a fold on cluster 1 and a fold on cluster 0 are different
resources, so an objective built out of these counts cannot compare two placements. How
the work spreads over the machine also depends on what else is running there, which no
block can see. Spreading an operator is therefore something the layer writes -- `Scatter`,
one block per cluster, `Gather` -- and every node's placement stays visible in its source.

The blocks get a second freedom out of this, and it is the real prize: inside its own
boundary a block may choose between sub-DFGs that are not equivalent in cost. The RMSNorm
picks between two kernels whose per-row reduction lands along the beats or across the
lanes, and adds or drops its own transposes to reach the one it picked. That choice is
invisible from the outside, which is exactly why the block should be the one making it.

======================================================================================
WHY THE LINKER STILL INSERTS NOTHING
======================================================================================

Node CREATION order is dispatch order on this machine. A linker that injected a node
between two blocks would put it at a point in the order neither block chose, and the
schedule would move without anything in the source saying so. So the rule is unchanged:
this file adds EDGES, never nodes. A conversion is owned by the consuming or producing
block, inside its own build and in its own order, which is why a block has to be able to
say which conversions it is willing to own. That is `variants()`.

DEFERRING THE BUILD DOES NOT REORDER IT. Stages build in the order they were added, and
`raw()` puts non-block work -- an input load, a readback, a check -- into that same order.
Anything the application would otherwise have built between two `add` calls goes in a
`raw` thunk, and the dispatch order comes out exactly as it reads on the page.

======================================================================================
MATCHING IS EXACT, AND ASKS THE BLOCK
======================================================================================

A boundary is legal when the producer's output spec EQUALS the consumer's input spec --
same layout, same precision, same shape -- with memory level the one exception: a
consumer whose port says `mem_level=None` takes the operand wherever it lies and loads it
itself, which is what every block that fetches declares.

Equality, and not "could the hardware convert this", because those are different
questions. A block reading A-layout bytes that arrived row-major computes a scrambled
answer that no check catches, and whether some pair of strides exists says nothing about
whether this block will run it. So the only way a block accepts another layout is by
OFFERING a variant that does, which is a promise its build() has to keep.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import networkx as nx

from bingo_block_viz import BlockRecord, PortRecord

from .ctx import Ctx
from .ports import Block, BlockResult, Port, PortSpec
from .variant import Cost, Variant, variants_of

# ======================================================================================


def check_contract(src, need: PortSpec, *, where: str) -> Optional[str]:
    """Is `src` usable where `need` is required? Returns None, or the reason.

    `src` is a bound Port or a bare PortSpec. The rule is EQUALITY, on all five things a
    port states: shape, layout, precision, memory level and cluster. There is no
    satisfaction relation and no ordering over the memory hierarchy, because a port never
    says "anywhere" -- a block that would accept an operand from more than one level or in
    more than one layout offers a realisation for each, and the resolver picks the one
    that matches. So a mismatch here is always a real gap.

    Cluster equality falls out rather than being a case: only L1 ports carry a cluster, so
    two ports that agree on the level either both name one or neither does.

    NO GAP IS CLOSED HERE, and none is assumed to be closed elsewhere. A block that will
    convert says so by offering a variant that accepts the other layout; a gap nobody
    offered to close is a refusal, naming both ends.
    """
    spec = src.spec if hasattr(src, "spec") else src
    who = getattr(src, "handle_name", "the binding")
    if tuple(spec.shape) != tuple(need.shape):
        return f"{where}: shape {tuple(spec.shape)} cannot feed {tuple(need.shape)}."
    if spec.layout != need.layout:
        return (f"{where}: '{who}' is {spec.layout} but this port reads {need.layout}. No "
                f"realisation of the consuming block accepts {spec.layout}; give it a "
                f"variant that does, or convert in the producer.")
    if spec.dtype != need.dtype:
        return (f"{where}: '{who}' is {spec.dtype} but this port reads {need.dtype}. A "
                f"precision change needs a SCALE and there is no right default, so it is "
                f"a step of its own or a block that takes the scale as a parameter.")
    if spec.mem_level != need.mem_level:
        return (f"{where}: '{who}' is in {spec.mem_level} but this port reads "
                f"{need.mem_level}. No realisation of the consuming block is handed an "
                f"operand from {spec.mem_level}; give it one, or move the operand with a "
                f"block that does.")
    if spec.cluster != need.cluster:
        return (f"{where}: '{who}' is in cluster {spec.cluster}'s L1 but this port reads "
                f"cluster {need.cluster}'s. A GEMM or SIMD kernel addresses only its own "
                f"TCDM, and a remote handle does not fault -- the transfer completes "
                f"without writing and the destination keeps what it held. Move the "
                f"operand with an explicit block, or place the consumer on cluster "
                f"{spec.cluster}.")
    return None


def link(ctx: Ctx, src: Port, dst: Port, *, sources: Sequence = (),
         gate_sources: bool = True) -> int:
    """Join one block's output to the next block's input. Returns the edge count.

    Three things, and nothing else:

      1. the RAW edges, src.ends -> dst.ends. This is the real data dependency and it is
         what a hand-built graph would have.
      2. gating the destination's graph SOURCES behind the producer, so their buffers can
         reuse the producer block's L1. A data edge alone cannot do this: a node with no
         predecessor has no ancestors, so nothing upstream is ordered against it.
      3. nothing else. No transport (the block loads its own operands), no scheduling.

    Gating sources is a real trade and that is why it is a flag: a gated weight load cannot
    be prefetched during the previous block, and an ungated one cannot reuse its bytes. At
    toy sizes gate everything and measure; at widths where L1 binds, ungate the loads on
    the critical path.
    """
    n = 0
    for a in src.ends:
        for b in dst.ends:
            if a is not b:
                ctx.dfg.bingo_add_edge(a, b)
                n += 1
    if gate_sources and src.ends:
        anchor = src.ends[-1]
        for s in sources:
            if s is not anchor and s not in dst.ends:
                ctx.dfg.bingo_add_edge(anchor, s)
                n += 1
    return n


# ======================================================================================
# What `add` hands back before anything is built
# ======================================================================================

class Ref:
    """One output of a stage that has not been built yet.

    It is what `add` returns and what the next `add` binds, so the application writes the
    chain in the order it reads. After `run` it carries the real Port; before that,
    touching `.port` says so rather than handing back a half-built object.
    """

    __slots__ = ("stage", "port_name")

    def __init__(self, stage: "Stage", port_name: Optional[str]):
        self.stage, self.port_name = stage, port_name

    @property
    def port(self) -> Port:
        if self.stage.result is None:
            raise ValueError(
                f"'{self.stage.name}' has not been built yet, so it has no buffer and no "
                f"nodes. Pipeline.add only CONNECTS -- the boundary is resolved and the "
                f"sub-DFGs are emitted by run(). Work that needs a real Port goes in a "
                f"raw() thunk, which run() executes in the order you registered it.")
        outs = self.stage.result.outputs
        if self.port_name is None:
            if len(outs) != 1:
                raise ValueError(
                    f"'{self.stage.name}' has {len(outs)} outputs {sorted(outs)}; name one.")
            return next(iter(outs.values()))
        return outs[self.port_name]

    def __repr__(self):
        return f"<Ref {self.stage.name}.{self.port_name or '*'}>"


@dataclass
class Stage:
    """One block in the pipeline: what it is, what feeds it, and what it became."""

    name: str
    template: Block
    bind: dict                                   # {port: Ref | Port}
    variants: list = field(default_factory=list)
    chosen: Optional[Variant] = None
    result: Optional[BlockResult] = None

    @property
    def block(self) -> Block:
        """The realisation the resolver picked. Only meaningful after run()."""
        if self.chosen is None:
            raise ValueError(f"'{self.name}' has not been resolved yet; call run().")
        return self.chosen.block

    def out(self, port: str = None) -> Ref:
        return Ref(self, port)

    def __repr__(self):
        return f"<Stage {self.name}>"


@dataclass
class _Raw:
    """Non-block work, held in add-order so deferring the build does not reorder it."""
    name: str
    fn: Callable


class _SourceBlock(Block):
    """A block with no inputs whose build() is the application's own thunk.

    It exists so hand-built work takes part in the chain instead of sitting outside it:
    the resolver sees a real output spec to resolve against, and the nodes land at the
    point in the order where the application wrote them.
    """

    name = "source"

    def __init__(self, spec: PortSpec, fn: Callable, port: str = "y"):
        self.spec, self.fn, self.port = spec, fn, port

    @property
    def inputs(self) -> dict:
        return {}

    @property
    def outputs(self) -> dict:
        return {self.port: self.spec}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        got = self.fn()
        why = check_contract(got, self.spec, where=f"source {self.port!r}")
        if why:
            raise ValueError(
                f"{why}\n    A source's thunk has to return the Port its `spec` "
                f"promised: the resolver has already picked the rest of the chain "
                f"against that promise.")
        return BlockResult(outputs={self.port: got}, nodes=list(got.ends))


# ======================================================================================


class Pipeline:
    """Assemble blocks into one DFG: connect, resolve, build.

    `add` records; `raw` records a thunk for anything that is not a block; `run` resolves
    every boundary and then builds, in the order everything was recorded. Build order is
    add order, because node creation order is dispatch order on this machine.
    """

    def __init__(self, ctx: Ctx, *, gate_sources: bool = True, verbose: bool = True):
        self.ctx = ctx
        self.gate_sources = gate_sources
        self.verbose = verbose
        self.stages: list = []
        self.items: list = []           # stages and raws, in the order they were recorded
        self.demands: list = []         # (Ref, PortSpec) the application pinned
        self.edges = 0
        self.ran = False

    # ---- phase 1: connect --------------------------------------------------------------

    def add(self, block: Block, name: str, bind: dict = None, **binds) -> Stage:
        """Record a block and what feeds each of its inputs. Builds nothing.

        Inputs are bound by keyword -- `x=ref` -- or as a `bind` dict; both forms are the
        same thing, and the keyword form is there because a chain reads better without it.
        A bind value is a Ref from an earlier `add`, or a real Port for something the
        application built itself.
        """
        if self.ran:
            raise ValueError(
                f"'{name}' added after run(). A pipeline resolves its boundaries once, "
                f"over the whole chain; adding to it afterwards would build against "
                f"layouts that were chosen without it.")
        bind = dict(bind or {})
        bind.update(binds)
        st = Stage(name=name, template=block, bind=bind)
        self.stages.append(st)
        self.items.append(st)
        return st

    def source(self, name: str, spec: PortSpec, fn: Callable, port: str = "y") -> Stage:
        """Register a port the APPLICATION builds, as a stage of the chain.

        A layer's input load, a slice pulled from L3, anything hand-built that later
        stages read. `fn()` is called in add-order, like raw(), and must return the Port
        it built; `spec` is what that Port will be, which is what lets the resolver decide
        the boundary behind it before any of it exists.

        Building it outside the pipeline and binding the Port directly is equivalent only
        while nothing has been added ahead of it; after that its nodes land out of order.
        """
        return self.add(_SourceBlock(spec, fn, port), name)

    def raw(self, fn: Callable, name: str = "raw") -> None:
        """Register non-block work at this point in the order.

        An input load, a readback, a check, a hand-built node: anything the application
        would otherwise have written between two `add` calls. run() calls it in place, so
        the dispatch order comes out exactly as the source reads. Inside the thunk every
        earlier stage is built, so `ref.port` resolves.
        """
        if self.ran:
            raise ValueError(f"raw({name!r}) registered after run().")
        self.items.append(_Raw(name, fn))

    def demand(self, ref: Ref, spec: PortSpec = None, **kw) -> None:
        """Pin what a stage's output must be, because something outside the chain reads it.

        The resolver is free to pick any legal boundary, so an output whose layout the
        application depends on -- a golden it is checked against, a buffer the host reads
        back -- has to SAY so. Without this the chain's last layout is whatever was
        cheapest, which is correct for the graph and wrong for the reader.
        """
        if spec is None:
            spec = PortSpec(**kw)
        self.demands.append((ref, spec))

    # ---- phase 2: resolve --------------------------------------------------------------

    def _enumerate(self) -> None:
        """Ask every stage what it could be. Its own constructor decides what is legal."""
        for st in self.stages:
            refused = []
            st.variants = variants_of(
                st.template, on_refusal=lambda p, e: refused.append((p, e)))
            if not st.variants:
                lines = "\n".join(f"      {_brief(p)}: {_one_line(e)}" for p, e in refused)
                raise ValueError(
                    f"'{st.name}': no realisation of {type(st.template).__name__} is "
                    f"legal at this shape. The block's own reasons:\n{lines}")
            if self.verbose and len(st.variants) > 1:
                print(f"[pipeline] {st.name}: {len(st.variants)} realisation(s)"
                      + (f", {len(refused)} refused" if refused else ""))

    def _resolve(self) -> None:
        """Pick one variant per stage so every boundary agrees, at the least total cost.

        An exact search, not a heuristic. The state is the choice made for every stage
        whose output something later still reads -- the LIVE frontier, which on these
        graphs is one or two stages -- so the cost is a handful of combinations per stage
        and there is no reason to be clever.

        A greedy per-stage rule would get it wrong, and the RMSNorm is the counter-example
        in both directions: at rows=32 the right answer costs the chain two extra nodes to
        take the folds to zero, and at rows=64 -- where the fold-free kernel does not
        exist -- the right answer instead collapses the tail into one kernel. Neither is a
        local property of the norm.
        """
        idx = {id(st): i for i, st in enumerate(self.stages)}
        # deps[i][port] = ("stage", j, out_port) or ("fixed", Port)
        deps, consumers = [], {i: set() for i in range(len(self.stages))}
        for i, st in enumerate(self.stages):
            d = {}
            for port, val in st.bind.items():
                if isinstance(val, Ref):
                    j = idx.get(id(val.stage))
                    if j is None or j >= i:
                        raise ValueError(
                            f"'{st.name}.{port}' is bound to '{val.stage.name}', which is "
                            f"not an earlier stage of this pipeline. Blocks are resolved "
                            f"in the order they were added.")
                    d[port] = ("stage", j, val.port_name)
                    consumers[j].add(i)
                elif isinstance(val, Port):
                    d[port] = ("fixed", val)
                else:
                    raise ValueError(
                        f"'{st.name}.{port}' is bound to {type(val).__name__}; bind a Ref "
                        f"from an earlier add(), or a Port you built yourself.")
            deps.append(d)
        # BOUND TO SOMETHING THAT IS NOT AN INPUT. Caught over the union of every
        # realisation, because two realisations of one block declare the same port names;
        # a name in none of them is a typo, and saying so here names the stage.
        for i, st in enumerate(self.stages):
            known = set().union(*(set(v.inputs) for v in st.variants))
            unknown = sorted(set(deps[i]) - known)
            if unknown:
                raise ValueError(
                    f"'{st.name}': bound {unknown}, but this block's inputs are "
                    f"{sorted(known)}.")

        # A demand keeps its stage live to the very end, so the search cannot close the
        # frontier on a choice the reader then rejects.
        pinned = {}
        for ref, spec in self.demands:
            j = idx.get(id(ref.stage))
            if j is None:
                raise ValueError("demand() on a stage that is not in this pipeline.")
            pinned.setdefault(j, []).append((ref.port_name, spec))

        def live_after(i):
            return frozenset(j for j in range(i + 1)
                             if any(k > i for k in consumers[j]) or j in pinned)

        # states: {state -> (cost, choices)}; state = frozenset of (stage, variant index)
        states = {frozenset(): (Cost(), ())}
        why_none = []
        for i, st in enumerate(self.stages):
            nxt, live = {}, live_after(i)
            for state, (cost, choices) in states.items():
                chosen = dict(state)
                for vi, v in enumerate(st.variants):
                    bad = None
                    # AN INPUT WITH NOTHING ON IT. Per realisation, because a block may
                    # declare a port on one and not another; if that leaves no realisation
                    # standing, this is the reason the chain reports.
                    missing = sorted(set(v.inputs) - set(deps[i]))
                    if missing:
                        why_none.append(
                            f"{st.name}: input(s) {missing} are not bound. "
                            + "; ".join(f"{k} wants {v.inputs[k].describe()}"
                                        for k in missing))
                        continue
                    for port, dep in deps[i].items():
                        need = v.inputs.get(port)
                        if need is None:
                            bad = (f"{st.name}.{port}: this realisation has no such input "
                                   f"({sorted(v.inputs)})")
                            break
                        if dep[0] == "stage":
                            _, j, oport = dep
                            pv = self.stages[j].variants[chosen[j]]
                            outs = pv.outputs
                            got = (next(iter(outs.values())) if oport is None
                                   else outs.get(oport))
                            if got is None:
                                bad = (f"{st.name}.{port}: '{self.stages[j].name}' has no "
                                       f"output {oport!r}")
                                break
                            src = got
                        else:
                            src = dep[1]
                        bad = check_contract(src, need, where=f"{st.name}.{port}")
                        if bad:
                            break
                    if bad:
                        why_none.append(bad)
                        continue
                    full = dict(chosen)
                    full[i] = vi
                    key = frozenset((j, full[j]) for j in live if j in full)
                    cand = cost + v.cost
                    cur = nxt.get(key)
                    if cur is None or cand < cur[0]:
                        nxt[key] = (cand, choices + ((i, vi),))
            if not nxt:
                reasons = "\n".join(f"      {r}" for r in dict.fromkeys(why_none[-8:]))
                raise ValueError(
                    f"[pipeline] the chain cannot reach '{st.name}': no realisation of it "
                    f"accepts anything its producers can emit.\n{reasons}")
            states, why_none = nxt, []

        # Close the far end on whatever the application pinned.
        finals = []
        for state, (cost, choices) in states.items():
            chosen = dict(state)
            ok = True
            for j, wants in pinned.items():
                pv = self.stages[j].variants[chosen[j]]
                for oport, spec in wants:
                    outs = pv.outputs
                    got = (next(iter(outs.values())) if oport is None else outs.get(oport))
                    if got is None or check_contract(got, spec,
                                                     where=f"demand on {self.stages[j].name}"):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                finals.append((cost, choices))
        if not finals:
            want = "; ".join(f"{self.stages[j].name} -> {s.describe()}"
                             for j, w in pinned.items() for _, s in w)
            raise ValueError(
                f"[pipeline] no set of realisations satisfies what the application pinned "
                f"({want}). Either a block needs a variant that produces it, or the demand "
                f"is for something no block in the chain can emit.")
        cost, choices = min(finals, key=lambda t: t[0])
        for i, vi in choices:
            self.stages[i].chosen = self.stages[i].variants[vi]
        self.cost = cost

    # ---- phase 3: build ----------------------------------------------------------------

    def _build(self, st: Stage) -> None:
        blk, v = st.chosen.block, st.chosen
        bound = {}
        for port, val in st.bind.items():
            bound[port] = val.port if isinstance(val, Ref) else val
        for port, p in bound.items():
            why = check_contract(p, v.inputs[port], where=f"{st.name}.{port}")
            if why:
                raise ValueError(
                    f"{why}\n    The resolver chose this realisation, so a mismatch here "
                    f"means the block's declared ports and what it actually built "
                    f"disagree.")
        ctx = self.ctx.scope(st.name)
        res = blk.build(ctx, bound)
        for port, p in bound.items():
            dst = res.inputs.get(port)
            if dst is None:
                raise ValueError(f"'{st.name}': build() returned no input port {port!r}, "
                                 f"so the linker has nothing to join to.")
            self.edges += link(self.ctx, p, dst, sources=res.sources,
                               gate_sources=self.gate_sources)
        st.result = res
        if self.verbose:
            outs = ", ".join(f"{k}:{p.spec.describe()}" for k, p in res.outputs.items())
            extra = f"  {_brief(v.params)}" if v.params else ""
            print(f"[pipeline] {st.name}: {len(res.nodes)} nodes, "
                  f"{len(res.sources)} source(s) -> {outs}{extra}")

    def run(self) -> "Pipeline":
        """Resolve every boundary, then build everything in the order it was recorded."""
        if self.ran:
            raise ValueError("this pipeline has already been run().")
        self._enumerate()
        self._resolve()
        if self.verbose:
            self._plan_report()
        self.ran = True
        g = self.ctx.dfg
        for item in self.items:
            before = set(g.nodes)
            if isinstance(item, Stage):
                self._build(item)
            else:
                item.fn()
            self._record(item, [n for n in g.nodes if n not in before])
        return self

    def _record(self, item, nodes: list) -> None:
        """Remember what `item` built, for the per-block pictures (passes/bingo_block_viz).

        The nodes are taken as the DIFFERENCE of the graph's node set around the build, not
        from BlockResult.nodes: a block reports the nodes it chooses to, and a picture that
        is used to find a missing edge has to show every node the block created -- the
        loads bring_in emits on its behalf included. The records ride on the DFG, because
        the pictures are drawn by bingo_compile_dfg, which is the one place that knows the
        app's output directory.
        """
        g = self.ctx.dfg
        if getattr(g, "block_records", None) is None:
            g.block_records = []
            g.block_roles = dict(self.ctx.roles)
        rec = BlockRecord(index=len(g.block_records), name=item.name, kind="raw",
                          nodes=nodes, doc="application work registered with Pipeline.raw()")
        if isinstance(item, Stage):
            blk, v, res = item.chosen.block, item.chosen, item.result
            source = isinstance(blk, _SourceBlock)
            rec.kind = "source" if source else type(blk).__name__
            rec.doc = ("a port the application builds itself (Pipeline.source)" if source
                       else _first_sentence(type(blk).__doc__))
            rec.params = _brief(v.params) if v.params else ""
            c = v.cost
            rec.cost = f"folds {c.folds} · SIMD {c.simd} · xDMA {c.xdma} · iDMA {c.idma}"
            for port, val in item.bind.items():
                src = val.port if isinstance(val, Ref) else val
                if isinstance(val, Ref):
                    peer = f"{val.stage.name}.{val.port_name or next(iter(val.stage.result.outputs))}"
                elif src.ends:
                    peer = "built by the application"
                else:
                    peer = f"staged in {src.spec.mem_level}"
                dst = res.inputs.get(port)
                rec.inputs.append(PortRecord(port, src.spec.describe(), src.handle_name,
                                             list(dst.ends) if dst is not None else [],
                                             peer, list(src.ends), src.spec.cluster))
            for port, p in res.outputs.items():
                rec.outputs.append(PortRecord(port, p.spec.describe(), p.handle_name,
                                              list(p.ends), cluster=p.spec.cluster))
            # link() gates these behind each producer; the pictures tell those edges apart
            rec.sources = list(res.sources)
        g.block_records.append(rec)

    # ---- diagnostics -------------------------------------------------------------------

    def _plan_report(self) -> None:
        w = max(len(s.name) for s in self.stages)
        print(f"[pipeline] resolved {len(self.stages)} stage(s), total {self.cost} "
              f"(folds/simd/xdma/idma, compared in that order)")
        for st in self.stages:
            v = st.chosen
            ins = ", ".join(f"{k}:{s.layout}/{s.dtype}" for k, s in v.inputs.items())
            outs = ", ".join(f"{k}:{s.layout}/{s.dtype}" for k, s in v.outputs.items())
            alt = f"  ({len(st.variants)} realisations)" if len(st.variants) > 1 else ""
            print(f"[pipeline]   {st.name:{w}}  {ins:>34} -> {outs:<28} {v.cost}{alt}")

    def report(self) -> dict:
        """Say, per adjacent pair, whether the later block can reuse the earlier one's L1.

        This is the only build-time signal that static-L1 will actually overlap the stages;
        without it the symptom is a peak that quietly fails to drop, with nothing to point
        at. It answers the same question `can_share` will ask later, on the same graph.
        """
        if not self.ran:
            raise ValueError("report() before run(): there is no graph to walk yet.")
        g = self.ctx.dfg
        out = {}
        for a, b in zip(self.stages, self.stages[1:]):
            if not b.result.nodes:          # a source bound straight to a staged symbol
                out[(a.name, b.name)] = []
                continue
            anc = nx.ancestors(g, b.result.nodes[0])
            blocked = [n for n in a.result.nodes
                       if n not in anc and not nx.has_path(g, n, b.result.nodes[0])]
            out[(a.name, b.name)] = blocked
            if self.verbose:
                if blocked:
                    print(f"[pipeline] {a.name} -> {b.name}: {len(blocked)} node(s) of "
                          f"{a.name} are NOT ordered before {b.name}, so buffers they "
                          f"touch cannot be reused: "
                          + ", ".join(n.node_name for n in blocked[:4]))
                else:
                    print(f"[pipeline] {a.name} -> {b.name}: clean cut, "
                          f"{b.name} can reuse {a.name}'s L1")
        return out


def _brief(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in (params or {}).items()) or "(as constructed)"


def _one_line(e) -> str:
    return " ".join(str(e).split())[:120]


def _first_sentence(doc) -> str:
    """The first sentence of a docstring, on one line -- a block's own summary of itself."""
    text = " ".join((doc or "").strip().split("\n\n")[0].split())
    cut = text.find(". ")
    return (text if cut < 0 else text[:cut + 1])[:170]
