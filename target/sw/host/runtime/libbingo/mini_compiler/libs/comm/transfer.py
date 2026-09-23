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
  AXES       a tensor stored [cols, rows] read as [rows, cols] is again a permutation with
             the same flat length, and again it computes cleanly and answers wrongly.

So this module's job is to see the mismatch and either close it or refuse. What it must
never do is proceed.

THE FOUR CLOSURES, and why each is the engine it is:

  L4 -> L3   host iDMA. The memory-chiplet pool is reachable over the D2D link by the host
             iDMA; a cluster's GEMM and SIMD have no port at all and the cluster iDMA
             would be fetching across the die for every tile. So the pool is hoisted to
             main memory ONCE, and the per-tile traffic then runs entirely on-die.
  layout     an xDMA pass with independent strides on each side. By default it is staged
             -- 1-D copy into L1, then permute in place -- because both halves are
             exercised on RTL today. The fused single-pass form reads main memory with the
             source layout's strides directly; the device kernel takes full addresses the
             same way the 1-D copy does, so it is supported by construction, but nothing
             in this tree does it. bring_in(fuse_relayout=True) opts in.
  transpose  the xDMA's 8x8 BLOCK TRANSPOSER, a real unit in the writer datapath and not a
             pair of strides -- no stride nest expresses an axis exchange, which is why
             nest.py refuses one by name and points here. Being a WRITER extension it
             scatters into the writer's OWN local TCDM, so the transposed result always
             lands in L1; a non-local source is staged by the kernel. Native at 1- and
             2-byte elements only, so this runs at fp16 and a quantiser comes after it.
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

from bingo_kernel_args import (HostBingoKernelIdmaArgs,
                               SnaxBingoKernelXdma1dCopyArgs,
                               SnaxBingoKernelXdmaTranspose2dArgs)

from .ports import Layout, Port, PortSpec
from .nest import convert_args

# The level names live in ports.py, next to the PortSpec field that holds one.


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
    if have.mem_level is None:
        # `have` is a BOUND port -- a real buffer, which is somewhere. None is only
        # meaningful on the declaration side, where it means "wherever you have it".
        raise ValueError(
            "the bound operand declares no mem_level. None is a DECLARATION saying the "
            "consuming block will fetch it from wherever it is; a buffer that actually "
            "exists has a level. Bind a Port whose spec says where the buffer lives.")
    if want.mem_level is None:
        # The consumer declared no level: it fetches the operand itself, from wherever the
        # caller keeps it. Nothing to plan here -- the block calls this again with the
        # level it actually needs, and THAT call plans the hoist.
        pass
    elif have.mem_level != want.mem_level:
        if have.mem_level == "L4":
            steps.append(Step("hoist", "the memory-chiplet pool is off-die; hoist it to "
                                       "main memory once instead of per tile",
                              engine="host iDMA"))
        elif (have.mem_level, want.mem_level) != ("L3", "L1"):
            raise ValueError(
                f"cannot bring an operand from {have.mem_level} to {want.mem_level}. The "
                f"closures are L4->L3 (host iDMA) and L3->L1 (the block's own loads).")
    # ONE RULE FOR BOTH HALVES. Layouts that disagree on their contiguous axis differ by
    # a transpose, which no stride nest expresses -- nest.py refuses exactly that pair. So
    # the plan is: put the operand in the orientation the destination blocks in, with the
    # 8x8 block transposer, and let convert_args do the rest.
    #
    #   row_major <-> col_major        the transposer alone
    #   col_major  -> A/B/D            transpose to row_major FIRST, then the nest
    #   A/B/D      -> col_major        the nest FIRST, then transpose
    #   row_major <-> A/B/D            the nest alone, as before
    #
    # ORDER IS FORCED, not chosen: the transposer permutes a PLAIN array, so it runs while
    # the operand is unblocked, and the nest is derived in the blocked side's own
    # dimensions. There is no flag to reconcile any more -- `xpose_first` and the
    # "needs a packed side" refusal both came from having orientation live outside the
    # layout, and both are gone with it. A transpose between two BLOCKED layouts is still
    # refused, by convert_args, where it belongs.
    UNBLOCKED = (Layout.ROW_MAJOR, Layout.COL_MAJOR)
    xpose = ((have.layout in UNBLOCKED) != (want.layout in UNBLOCKED)
             and Layout.COL_MAJOR in (have.layout, want.layout)) \
        or {have.layout, want.layout} == set(UNBLOCKED)
    xpose_first = xpose and have.layout == Layout.COL_MAJOR

    relayout_step = None
    if have.layout != want.layout and not {have.layout, want.layout} == set(UNBLOCKED):
        # The nest runs between row_major and the blocked layout, whichever side that is;
        # the transpose above or below it closes the orientation.
        src_lay = Layout.ROW_MAJOR if xpose_first else have.layout
        dst_lay = want.layout if want.layout not in UNBLOCKED else Layout.ROW_MAJOR
        if mesh is not None and elem_bytes is not None:
            # Derive it now and throw it away: the derivation is the feasibility test, and
            # it is cheap next to being wrong about it. In the tensor's own dimensions --
            # the blocked layouts are defined on (rows, cols) and the transpose, where
            # there is one, has already put the bytes that way round.
            r, c = want.shape
            convert_args(src_lay, dst_lay, r, c, mesh, elem_bytes, 0, 0)
        relayout_step = Step("relayout", f"{src_lay} -> {dst_lay}, fused into the load so "
                                         f"it costs no extra traversal", engine="xDMA 6d")

    xpose_step = None
    if xpose:
        if elem_bytes is not None and elem_bytes not in (1, 2):
            raise ValueError(
                f"a transpose at {elem_bytes}-byte elements has no hardware path: the "
                f"xDMA transposer's native modes are the cfg's elementWidth [8, 16], and "
                f"nothing composes a wider element. int32 falls back to a DM-core loop, "
                f"which is not a transfer this plans. Transpose at fp16 and convert after.")
        src_shape = have.stored_shape if xpose_first else want.stored_shape
        dst_shape = (src_shape[1], src_shape[0])
        xpose_step = Step("transpose",
                          f"{tuple(src_shape)} -> {tuple(dst_shape)}; an 8x8 block "
                          f"transposer, not a stride nest",
                          engine="xDMA transposer")

    ordered = (xpose_step, relayout_step) if xpose_first else (relayout_step, xpose_step)
    steps.extend(x for x in ordered if x is not None)
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


def bring_in(ctx, name, have: Port, want: PortSpec, *, mesh, elem_bytes, after=(),
             fuse_relayout: bool = False):
    """Close whatever gap there is between `have` and `want`. Returns (Port, [nodes]).

    Only the gap: when the bound port already matches, NOTHING is emitted and the returned
    port is the one passed in. That is what lets a composed graph be identical to a
    hand-written one whenever the producer already emits what the consumer wants -- the
    library never charges for a conversion that was not needed.

    `fuse_relayout` PICKS BETWEEN A PROVEN ROUTE AND A FASTER ONE, and the default is the
    proven one:

      False  copy main memory -> L1 with a 1-D transfer, then permute L1 -> L1. Two nodes
             and one extra buffer. Both halves are exercised on RTL today: the 1-D L3->L1
             copy is what FlashAttention's xload does on every K and V tile, and the
             L1->L1 permute is what the MoE reshape does.
      True   one 6-D transfer straight out of main memory, strided on both sides. The
             device kernel takes full 64-bit addresses through the same
             xdma_memcpy_nd_full_addr path the 1-D copy uses, so this is supported BY
             CONSTRUCTION -- but nothing in this tree does it, so it is unvalidated. It
             saves a buffer and a pass; turn it on once a sim has confirmed it.
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
            spec = PortSpec(spec.layout, spec.dtype, spec.shape, mem_level="L3",
                            doc=spec.doc)
        elif st.kind == "relayout":
            rows, cols = spec.stored_shape
            nbytes = rows * cols * elem_bytes
            prev = nodes[-1] if nodes else after
            if not fuse_relayout and spec.mem_level != "L1":
                # Stage it into L1 first, with the 1-D copy that every workload already
                # runs, and permute from there. The extra buffer is the price of using
                # only transfers this machine has been shown to perform.
                staged = ctx.l1(f"{name}_staged", nbytes)
                ld = ctx.node(f"Load_{name}", ctx.xdma,
                              "__snax_bingo_kernel_xdma_1d_copy",
                              SnaxBingoKernelXdma1dCopyArgs(handle, staged, nbytes), prev)
                nodes.append(ld)
                handle, prev = staged, ld
            # WHERE THIS NEST LANDS. When the destination is col_major the transpose
            # still has to run after it, so the nest targets row_major and the transpose
            # step below finishes the job; otherwise it goes straight to what was asked.
            to = Layout.ROW_MAJOR if want.layout == Layout.COL_MAJOR else want.layout
            dst = ctx.l1(f"{name}_{to.lower()}", nbytes)
            nd = ctx.node(f"Relayout_{name}", ctx.xdma, "__snax_bingo_kernel_xdma_6d",
                          convert_args(spec.layout, to, rows, cols, mesh,
                                       elem_bytes, handle, dst), prev)
            nodes.append(nd)
            handle, spec = dst, PortSpec(to, spec.dtype, spec.shape,
                                         mem_level="L1", doc=spec.doc)
        elif st.kind == "transpose":
            # The transposer is a datapath extension on whichever side the cfg declares
            # it (reader on snax_split_cluster, writer elsewhere -- BINGO_TRANSPOSER_ARM
            # hides which), and either way it is a LOCAL-LOOPBACK transform: the move ends
            # in the cluster's own TCDM. It does not have to START there -- the kernel
            # stages a non-local source itself -- but a block calling this already wants
            # its operand in L1, so the destination below always is.
            rows, cols = spec.stored_shape
            nbytes = rows * cols * elem_bytes
            prev = nodes[-1] if nodes else after
            dst = ctx.l1(f"{name}_t", nbytes)
            nd = ctx.node(f"Transpose_{name}", ctx.xdma,
                          "__snax_bingo_kernel_xdma_transpose_2d",
                          SnaxBingoKernelXdmaTranspose2dArgs(handle, dst, rows, cols,
                                                             elem_bytes), prev)
            nodes.append(nd)
            flipped = (Layout.COL_MAJOR if spec.layout == Layout.ROW_MAJOR
                       else Layout.ROW_MAJOR)
            handle, spec = dst, PortSpec(flipped, spec.dtype, spec.shape,
                                         mem_level="L1", doc=spec.doc)
    return Port(spec, handle, (nodes[-1],), cluster=ctx.cluster, name=name), nodes
