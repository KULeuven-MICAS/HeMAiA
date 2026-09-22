# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Getting an operand into the shape, precision and place a block needs.

WHY A BLOCK CANNOT JUST ASSUME. A block declares the layout, precision and memory level it
consumes. The thing bound to it was produced by something else -- a projection that emits
D-layout, a datagen that staged into the memory chiplet, a previous layer whose output is
fp16 where this one wants int8. Three kinds of mismatch, and NONE of them faults:

  LOCATION   a memory-chiplet address on a config without one is simply unmapped. The
             loads return whatever the fabric gives back and the checks compare one piece
             of garbage against another.
  LAYOUT     a permutation with the same flat length. Every byte is read, every byte is
             written, the arithmetic runs -- and the answer is a scrambled tensor. On
             random test data the golden is scrambled identically and it PASSES.
  PRECISION  fp16 bits read as int8 are two elements, not one. Nothing is out of range.

So this module's job is to see the mismatch and either close it or refuse. What it must
never do is proceed.

THE THREE CLOSURES, and why each is the engine it is:

  L4 -> L3   host iDMA. The memory-chiplet pool is reachable over the D2D link by the host
             iDMA; a cluster's GEMM and SIMD have no port at all and the cluster iDMA
             would be fetching across the die for every tile. So the pool is hoisted to
             main memory ONCE, and the per-tile traffic then runs entirely on-die.
  layout     one xDMA pass, FUSED INTO THE LOAD. The xDMA's AGU takes independent strides
             on each side, so reading L3 with the source layout's strides and writing L1
             with the destination's converts while it transports. That is strictly better
             than load-then-reshape: no intermediate buffer, no second traversal, and the
             conversion costs nothing beyond the load that had to happen anyway.
  precision  NOT DONE HERE. A precision change needs a scale, and there is no correct
             default -- the right one depends on the range of the data, which this module
             cannot see. Choosing one silently is how a tensor ends up 83% saturated and
             agreeing with its golden anyway. It is refused, with a pointer.

WHERE THE ARITHMETIC IS. This module decides WHAT a mismatch costs and emits the nodes;
deriving the actual strided transfer, and verifying it against ground-truth index maps,
is `nest`. The split matters because the two fail differently -- a wrong plan asks for a
conversion nobody wanted, a wrong nest silently scrambles the tensor.
"""

import numpy as np

from bingo_kernel_args import HostBingoKernelIdmaArgs

from .ports import Port, PortSpec
from .nest import convert_args

# How far a cluster can reach. A block's own loads close L3 -> L1; this module closes
# everything above that, and L2 is the descriptor list's home rather than an operand's.
SPACES = ("L1", "L2", "L3", "L4")


# ======================================================================================
# The plan, as data
# ======================================================================================

class Step:
    """One closure the prologue will make. Inspectable before anything is built."""

    def __init__(self, kind, why, engine=None):
        self.kind, self.why, self.engine = kind, why, engine

    def __repr__(self):
        return f"{self.kind}({self.engine})" if self.engine else self.kind


def plan(have: PortSpec, want: PortSpec, *, mesh=None, elem_bytes=None) -> list:
    """What it would take to satisfy `want` from `have`. Builds nothing.

    Separated from the building so a caller can print the prologue, assert that a
    composition needs none, or refuse one that would cost a pass it did not budget for.

    PASS `mesh` AND `elem_bytes` WHEN YOU HAVE THEM. Without them the plan can only say
    that a relayout is NEEDED, not that it is POSSIBLE -- and those differ: a transpose,
    or a run too narrow at this precision, is a conversion no pair of strides performs.
    With them the nest is derived here and the impossible case is refused at the contract
    check, by its real reason, instead of being promised and failing later at build.
    """
    steps = []
    if tuple(have.shape) != tuple(want.shape):
        raise ValueError(
            f"shape {tuple(have.shape)} cannot satisfy {tuple(want.shape)}: this module "
            f"moves and permutes, it does not tile or pad. Reshape upstream.")
    if have.space != want.space:
        if have.space == "L4":
            steps.append(Step("hoist", "the memory-chiplet pool is off-die; hoist it to "
                                       "main memory once instead of per tile",
                              engine="host iDMA"))
        elif (have.space, want.space) != ("L3", "L1"):
            raise ValueError(
                f"cannot bring an operand from {have.space} to {want.space}. The closures "
                f"are L4->L3 (host iDMA) and L3->L1 (the block's own loads).")
    if have.layout != want.layout:
        if mesh is not None and elem_bytes is not None:
            # Derive it now and throw it away: the derivation is the feasibility test, and
            # it is cheap next to being wrong about it.
            convert_args(have.layout, want.layout, have.shape[0], have.shape[1],
                         mesh, elem_bytes, 0, 0)
        steps.append(Step("relayout", f"{have.layout} -> {want.layout}, fused into the "
                                      f"load so it costs no extra traversal",
                          engine="xDMA 6d"))
    if have.dtype != want.dtype:
        raise ValueError(
            f"precision {have.dtype} -> {want.dtype} is not inserted automatically: it "
            f"needs a scale, and there is no correct default -- the right one depends on "
            f"the range of the data, which the linker cannot see. Put an explicit "
            f"quantise block in the pipeline and bind its output here.")
    return steps


# ======================================================================================
# Building it
# ======================================================================================

def hoist(ctx, name, src, *, nbytes, after=()):
    """L4 -> L3 on the host iDMA. Returns (handle, node)."""
    dst = ctx.l3(f"{name}_l3", int(nbytes))
    nd = ctx.host_node(f"Hoist_{name}", "__host_bingo_kernel_idma",
                       HostBingoKernelIdmaArgs(src, dst, int(nbytes)), after)
    return dst, nd


def bring_in(ctx, name, have: Port, want: PortSpec, *, mesh, elem_bytes, after=()):
    """Close whatever gap there is between `have` and `want`. Returns (Port, [nodes]).

    Only the gap: when the bound port already matches, NOTHING is emitted and the returned
    port is the one passed in. That is what lets a composed graph be identical to a
    hand-written one whenever the producer already emits what the consumer wants -- the
    library never charges for a conversion that was not needed.
    """
    steps = plan(have.spec, want, mesh=mesh, elem_bytes=elem_bytes)
    if not steps:
        return have, []

    nodes, handle, spec = [], have.handle, have.spec
    for st in steps:
        if st.kind == "hoist":
            nbytes = int(np.prod(spec.shape)) * elem_bytes
            handle, nd = hoist(ctx, name, handle, nbytes=nbytes, after=after)
            nodes.append(nd)
            spec = PortSpec(spec.layout, spec.dtype, spec.shape, space="L3", doc=spec.doc)
        elif st.kind == "relayout":
            rows, cols = spec.shape
            dst = ctx.l1(f"{name}_{want.layout.lower()}", rows * cols * elem_bytes)
            nd = ctx.node(f"Relayout_{name}", ctx.xdma, "__snax_bingo_kernel_xdma_6d",
                          convert_args(spec.layout, want.layout, rows, cols, mesh,
                                       elem_bytes, handle, dst),
                          nodes[-1] if nodes else after)
            nodes.append(nd)
            handle, spec = dst, PortSpec(want.layout, spec.dtype, spec.shape, space="L1",
                                         doc=spec.doc)
    return Port(spec, handle, (nodes[-1],), cluster=ctx.cluster, name=name), nodes
