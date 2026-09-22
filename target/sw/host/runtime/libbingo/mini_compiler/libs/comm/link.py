# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The linker: bind one block's outputs to the next block's inputs, and add the edges.

WHAT IT IS NOT. It does not schedule, place or tile -- those are the caller's parameters.
It does not compile: the mini-compiler still sees ONE assembled DFG and runs static-L1, the
hang check and the CERF passes on it exactly as before. Assembly is the whole job.

WHY IT INSERTS NO NODES. A block loads its own operands, because only the block knows it
needs them in its own cluster's L1 -- the GEMM and SIMD have no AXI port. So the loads a
block already builds ARE the transport, and the linker only ever adds edges. That matters
more than it looks: node CREATION order is dispatch order on this machine, so a linker that
injected nodes would silently move the schedule. It is also why a block that closes its own
layout or location gaps has to say so (Block.closes_gaps) -- the conversion happens in the
BLOCK's build, in the block's own dispatch order, never here.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import networkx as nx

from .ctx import Ctx
from .ports import Block, BlockResult, Port, PortSpec

# ======================================================================================

# Bytes per element, for the contract check: whether a layout conversion is expressible
# depends on the precision, because the xDMA's run length is counted in BYTES.
_ELEM_BYTES = {"i8": 1, "f16": 2, "f32": 4, "i32": 4}


def _as_tuple(x):
    if x is None:
        return ()
    return tuple(x) if isinstance(x, (list, tuple, set)) else (x,)


def check_contract(src: Port, dst_spec: PortSpec, *, where: str,
                   mesh=None, elem_bytes=None, closes_gaps=False) -> Optional[str]:
    """Is `src` usable where `dst_spec` is required? Returns None, or the reason.

    LOSSLESS IS AUTOMATIC, LOSSY IS EXPLICIT. A layout difference is a permutation with
    exactly one right answer, and a buffer in the wrong memory only has to be moved, so a
    block can close either on its own. A precision difference needs a SCALE, and there is
    no correct default -- so it is refused by name and the caller inserts the quantiser.

    WHAT "CLOSABLE" MEANS IS STAGING'S ANSWER, NOT THIS FUNCTION'S. Asking `transfer.plan`
    is what keeps the two from drifting: this check used to refuse every layout mismatch
    outright, which made a block's own conversion unreachable -- the linker rejected the
    binding before the block could look at it. It also means a conversion the hardware
    cannot do (a transpose, or a run too narrow at this precision) is refused HERE, with
    transfer's reason, instead of being promised and then failing at build.
    """
    if tuple(src.shape) != tuple(dst_spec.shape):
        return f"{where}: shape {tuple(src.shape)} cannot feed {tuple(dst_spec.shape)}."
    from . import transfer
    # WHAT is needed first, and only then WHETHER it is possible. A consumer that closes
    # no gaps has to be handed the right thing whatever the hardware could have done, so
    # reporting the feasibility of a conversion it was never going to perform sends the
    # reader after the wrong problem.
    try:
        steps = transfer.plan(src.spec, dst_spec)
    except ValueError as e:
        return f"{where}: '{src.handle_name}' {e}"
    if not steps:
        return None
    if not closes_gaps:
        what = ", ".join(st.kind for st in steps)
        return (f"{where}: '{src.handle_name}' is {src.spec.describe()} but "
                f"{dst_spec.describe()} is required, and the consuming block does not "
                f"close gaps itself ({what} would be needed). Put an explicit conversion "
                f"block in the pipeline, or set closes_gaps on the consumer if its "
                f"build() calls transfer.bring_in.")
    try:
        transfer.plan(src.spec, dst_spec, mesh=mesh, elem_bytes=elem_bytes)
    except ValueError as e:
        return f"{where}: '{src.handle_name}' {e}"
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


@dataclass
class _Built:
    """One block instance in a pipeline."""
    name: str
    block: Block
    result: BlockResult

    def out(self, port: str = None) -> Port:
        o = self.result.outputs
        if port is None:
            if len(o) != 1:
                raise ValueError(f"{self.name} has {len(o)} outputs {sorted(o)}; name one.")
            return next(iter(o.values()))
        if port not in o:
            raise ValueError(f"{self.name} has no output {port!r}; it has {sorted(o)}.")
        return o[port]


class Pipeline:
    """Assemble blocks into one DFG.

    Binding is EAGER -- a block is built the moment it is added, in the order it is added --
    because node creation order is dispatch order on this machine. A pipeline that deferred
    building and reordered it would silently move the schedule.
    """

    def __init__(self, ctx: Ctx, *, gate_sources: bool = True, verbose: bool = True):
        self.ctx = ctx
        self.gate_sources = gate_sources
        self.verbose = verbose
        self.stages: list = []
        self.edges = 0

    def add(self, block: Block, name: str, bind: dict = None) -> _Built:
        bind = bind or {}
        want = block.inputs
        unknown = set(bind) - set(want)
        if unknown:
            raise ValueError(f"{name}: bound {sorted(unknown)}, but this block's inputs "
                             f"are {sorted(want)}.")
        missing = [k for k in want if k not in bind]
        if missing:
            raise ValueError(f"{name}: input(s) {missing} are not bound. "
                             + "; ".join(f"{k} wants {want[k].describe()}" for k in missing))

        # Contract first: a mismatch should be a message about layouts, not a crash deep
        # inside a kernel-args constructor.
        for k, port in bind.items():
            why = check_contract(port, want[k], where=f"{name}.{k}",
                                 mesh=self.ctx.mesh,
                                 elem_bytes=_ELEM_BYTES.get(want[k].dtype),
                                 closes_gaps=getattr(block, "closes_gaps", False))
            if why:
                raise ValueError(why)

        ctx = self.ctx.scope(name)
        res = block.build(ctx, bind)

        for k, port in bind.items():
            dst = res.inputs.get(k)
            if dst is None:
                raise ValueError(f"{name}: build() returned no input port {k!r}, so the "
                                 f"linker has nothing to join to.")
            self.edges += link(self.ctx, port, dst, sources=res.sources,
                               gate_sources=self.gate_sources)

        built = _Built(name=name, block=block, result=res)
        self.stages.append(built)
        if self.verbose:
            outs = ", ".join(f"{k}:{p.spec.describe()}" for k, p in res.outputs.items())
            print(f"[pipeline] {name}: {len(res.nodes)} nodes, "
                  f"{len(res.sources)} source(s) -> {outs}")
        return built

    # ---- diagnostics -------------------------------------------------------------------
    def report(self) -> dict:
        """Say, per adjacent pair, whether the later block can reuse the earlier one's L1.

        This is the only build-time signal that static-L1 will actually overlap the stages;
        without it the symptom is a peak that quietly fails to drop, with nothing to point
        at. It answers the same question `can_share` will ask later, on the same graph.
        """
        g = self.ctx.dfg
        out = {}
        for a, b in zip(self.stages, self.stages[1:]):
            anc = nx.ancestors(g, b.result.nodes[0]) if b.result.nodes else set()
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
