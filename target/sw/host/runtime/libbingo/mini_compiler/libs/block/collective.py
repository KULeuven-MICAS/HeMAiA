# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Moving one block's data between clusters, so the next block can be split across them.

WHY THIS FILE EXISTS. Every block in libs/block runs on ONE cluster, and the layer
measured on RTL spends 553 us on cluster 0 against 94 us on each of the other three --
FlashAttention is the only stage that uses the machine. Splitting anything else needs a
way to put an operand on several clusters and a way to bring the pieces back, and until
now the library had neither: the kernels exist (`xdma_multicast`, `xdma_1d_copy` with full
64-bit addresses) but nothing wrapped them.

THE ADDRESSING PROPERTY THAT MAKES THIS CHEAP. A handle allocated on cluster c already
resolves to a full (chip | cluster | offset) address, so a transfer names its destination
by passing that handle -- no cluster map, no assumption that the four heaps lay out the
same way. The chain gather has relied on this from the start; these do too.

WHAT IS HERE AND WHAT IS NOT

    Broadcast   one cluster's buffer -> N clusters, ONE xDMA task (hardware multicast,
                max_multicast=16 on this cluster, 8 destinations per kernel call).
    shard_rows  run a per-row operator on disjoint row slices across N clusters and push
                the pieces back to a root buffer.

An ALL-REDUCE is NOT here, and it is the piece a tensor-parallel GEMM needs: a K-split
matmul has to sum partial products across clusters. THE HARDWARE ALREADY DOES IT, and not
by a separate mechanism: `ElementwiseJunction` and `MonoidJunction` are the two operators
on ONE junction socket, with the same arity, the same chain position and the same beat
rate. FlashAttention's fold is `xdma_chain_gather` with the monoid index; a K-split sum is
the same call with the elementwise index and CSR(0) = {op: ADD, fmt: FP16}, which is why
`SnaxBingoKernelXdmaChainGatherArgs` takes `junction` and `jct_csr0` as arguments rather
than baking in the monoid.

NO WIDENING IS NEEDED. The junction's math is FP32 internally whatever the transport
format says, and at elemWidth=16 a beat is 32 FP32 lanes carrying FP16/BF16 at full rate.
Partials cross in the format the SIMD block already produces. `pack_fa_partial` exists
because (m, l) are two scalars per row that have to be placed in specific lanes, not
because fp16 needed promoting.

ONE PROPERTY TO DECIDE FIRST, THOUGH: FP16/FP32 ADD is not exactly associative, so a
chain's answer depends on the route. If a K-split reduction has to be bit-reproducible
across placements, use the junction's INTEGER grid instead -- ADD there wraps in Z_2^w,
which is exactly associative and commutative -- and keep the partials in the GEMM's int32
accumulator domain.

(An earlier version of this note said the all-reduce needed the partials widened first and
pointed at `__snax_bingo_kernel_xdma_elementwise_add`. Both were wrong. That kernel drives
the `HasElementwiseAdd` WRITER extension, which snax_split_cluster does not have -- its
writer_junctions are ElementwiseJunction and MonoidJunction -- so on this cluster it falls
back to a scalar loop on the xDMA hart and is not a collective at all.)
"""

from bingo_kernel_args import (
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdmaMulticastArgs,
)
from bingo_mem_handle import (BingoMemAllocView, BingoMemFixedAddr,
                              BingoMemSymbol)

from ..comm import Ctx

# One xDMA multicast kernel carries this many destinations (BINGO_XDMA_MCAST_MAX).
MCAST_MAX = SnaxBingoKernelXdmaMulticastArgs.DST_MAX


def broadcast(ctx: Ctx, src, nbytes: int, clusters, *, src_cluster: int = 0,
              name: str = "bcast", after=()):
    """Replicate `src` (on `src_cluster`) into a fresh L1 buffer on every cluster.

    Returns {cluster: handle}. The source cluster maps to `src` itself -- it already has
    the data, and copying a buffer onto itself would only cost a transfer.

    ONE TASK, NOT N. The xDMA reads the source once and writes every destination, so the
    cost is one read plus the fan-out rather than N separate copies. That is the whole
    reason to reach for this instead of a loop of 1-D copies.
    """
    dsts = [c for c in clusters if c != src_cluster]
    if not dsts:
        return {src_cluster: src}
    if len(dsts) > MCAST_MAX:
        raise ValueError(
            f"broadcast to {len(dsts)} destinations: one xdma_multicast carries at most "
            f"{MCAST_MAX}. Issue several, or reconsider the decomposition -- a fan-out "
            f"this wide usually means the operand should have been loaded per cluster "
            f"from L3 instead.")
    out = {src_cluster: src}
    handles = []
    for c in dsts:
        h = ctx.at(c).l1(f"{name}_dst", nbytes)
        out[c] = h
        handles.append(h)
    nd = ctx.at(src_cluster).node(
        f"Bcast_{name}", ctx.xdma, "__snax_bingo_kernel_xdma_multicast",
        SnaxBingoKernelXdmaMulticastArgs(src, handles, nbytes), after)
    return out, nd


def shard_rows(ctx: Ctx, *, src, rows: int, cols: int, clusters, make,
               root: int = 0, elem_bytes: int = 2, name: str = "shard", after=(),
               src_on_l1: bool = False):
    """Run a per-row operator over disjoint row slices, one slice per cluster.

    `src` is a [rows, cols] row-major tensor. Each cluster fetches ONLY its own slice --
    there is no broadcast, which is what makes a row-parallel operator nearly free to
    distribute. `make(g, in_h, nrows, after, out_h)` builds the operator on that cluster,
    writing its result to `out_h`, and returns its last node.

    WHY `make` IS HANDED ITS DESTINATION. Only the ROOT can write straight into the
    gathered buffer: a SIMD kernel checks that its destination is local TCDM
    (BINGO_SIMD_REQUIRE_LOCAL), so every other cluster has to produce its slice locally
    and push it. Passing the handle in lets the root skip a copy onto itself while the
    others still get a local buffer, without `make` needing to know which case it is.

    TWO SOURCES, AND THE DIRECTION DIFFERS. With `src_on_l1=False` the tensor is in L3 and
    each cluster pulls its own slice with its own iDMA -- no cross-cluster traffic at all,
    which is why that path is nearly free. With `src_on_l1=True` it sits in the ROOT's L1,
    where an intermediate lands, and the root SCATTERS the slices out by multicast. It is
    not symmetric with the gather on purpose: a cluster cannot reach another cluster's L1
    with a plain copy, so whichever side owns the data has to do the moving.

    The pieces are pushed back into one root buffer by each cluster's own xDMA, so the
    gather is N-1 concurrent writes rather than N-1 serialised reads by the root.

    WHEN THIS IS AND IS NOT LEGAL. Rows must be independent -- RMSNorm, quantise, the
    residual and the dequantise all are; attention across a row is not. And the CONSUMER
    sets the granularity: an operator whose output feeds a conversion into A-layout needs
    whole 16-row tiles (meshRow), so at T=32 only a 2-way split survives that step. Rows
    that stay row-major, as they do up to the reshape, split as finely as you like.
    """
    n = len(clusters)
    if rows % n:
        raise ValueError(
            f"shard_rows: {rows} rows over {n} clusters does not divide. A ragged split "
            f"is expressible but every consumer downstream would have to carry the "
            f"per-cluster row count, so it is refused here.")
    per = rows // n
    row_bytes = cols * elem_bytes
    slice_bytes = per * row_bytes

    g_root = ctx.at(root)
    dst = g_root.l1(f"{name}_out", rows * row_bytes)
    pushes = []
    for i, c in enumerate(clusters):
        g = ctx.at(c)
        off = i * slice_bytes
        if src_on_l1 and c == root:
            # already here: hand the operator a view of the root's own buffer
            src_h, ld = _at(src, off), after
        elif src_on_l1:
            # SCATTERED BY THE OWNER, NOT PULLED BY THE CONSUMER. xdma_1d_copy needs both
            # endpoints local to the cluster issuing it -- a remote endpoint fails
            # silently in EITHER direction, so a pull reads nothing and the slice stays X.
            # Measured: with pulls, llm_norm1 (L3 source) passed and llm_norm2 (L1 source)
            # never produced a result at all. So the ROOT issues one multicast per
            # destination, each with its own source offset. That serialises on the root's
            # xDMA, which for a few KB is cheaper than the alternative -- staging the
            # tensor out to L3 so every cluster can load it the way norm1 does.
            src_h = g.l1(f"{name}_in", slice_bytes)
            ld = ctx.at(root).node(
                f"Scatter_{name}_c{c}", ctx.xdma, "__snax_bingo_kernel_xdma_multicast",
                SnaxBingoKernelXdmaMulticastArgs(_at(src, off), [src_h], slice_bytes),
                after)
        else:
            src_h = g.l1(f"{name}_in", slice_bytes)
            ld = g.node(f"Ld_{name}_c{c}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(_at(src, off), src_h, slice_bytes),
                        after)
        if c == root:
            # writes its slice of the gathered buffer in place -- no push needed
            pushes.append(make(g, src_h, per, ld, _at(dst, off)))
        else:
            out_h = g.l1(f"{name}_part", slice_bytes)
            last = make(g, src_h, per, ld, out_h)
            # MULTICAST WITH ONE DESTINATION, NOT A 1-D COPY. Writing ANOTHER cluster's
            # TCDM needs the xDMA's multicast path, which arms a destination slot;
            # xdma_1d_copy goes through xdma_memcpy_1d_full_addr, whose writer targets
            # local TCDM. Pointed at a remote handle it completes without ever writing,
            # and the destination keeps whatever it held -- X, on a buffer nothing else
            # touched. That cost a full run: the layer output came back with rows 8..31
            # unwritten and the failure surfaced as a DMAWriteDataCorrect assertion in
            # the L3 mux when the host stored the X onward, not as a failed check.
            pushes.append(g.node(
                f"Push_{name}_c{c}", ctx.xdma, "__snax_bingo_kernel_xdma_multicast",
                SnaxBingoKernelXdmaMulticastArgs(out_h, [_at(dst, off)], slice_bytes),
                last))
    return dst, pushes


def _at(handle, offset: int):
    """`handle` advanced by `offset` bytes, without allocating anything new.

    All four handle kinds carry a byte offset, but each spells it differently: an
    allocation makes a view, a view deepens its own, a staged C array is a symbol plus a
    byte offset, and a memory-chiplet address is just a number. A row shard has to offset
    whichever one the caller staged its tensor as, so all four are handled here rather
    than at every call site.
    """
    if offset == 0:
        return handle
    if isinstance(handle, BingoMemAllocView):
        return BingoMemAllocView(handle.base, handle.offset + offset)
    if isinstance(handle, BingoMemSymbol):
        return BingoMemSymbol(handle.symbol_name, handle.offset + offset)
    if isinstance(handle, BingoMemFixedAddr):
        return BingoMemFixedAddr(handle.address + offset)
    if hasattr(handle, "view"):
        return handle.view(offset)
    raise TypeError(
        f"cannot offset a {type(handle).__name__} by {offset} B: shard_rows needs a "
        f"handle it can take a byte view of.")
