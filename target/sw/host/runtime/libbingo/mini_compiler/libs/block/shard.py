# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Splitting a tensor across clusters and putting the pieces back, as two blocks.

WHY THESE ARE BLOCKS. Every other block in libs/block runs on ONE cluster: its kernels
address its own TCDM and its ports say so. A layer that wants a row-independent operator
on four clusters therefore writes four of them, and these two blocks are the ends that
make that a graph -- one fans the rows out, one brings them back. They are ordinary
blocks with declared ports, so the pipeline checks their placement exactly as it checks
anyone's, and every node they emit is in the layer's own order.

    Scatter   [rows, cols] -> N row slices, one in each cluster's L1
    Gather    N row slices, one per cluster -> one [rows, cols] in the root's L1

ROW-MAJOR ONLY, AND THAT IS NOT A SIMPLIFICATION. A row slice is a contiguous byte range
only when rows are contiguous runs. In A, B or D layout a block of rows is interleaved
with every other block, so "the first eight rows" is not an address range and slicing one
off is a relayout, not a move. Convert first, or split along whatever axis that layout
does make contiguous.

THE DIRECTION IS NOT SYMMETRIC, because the hardware is not. Which side issues a transfer
is decided by which side can address both ends:

    scatter from L3   every cluster's own iDMA loads its own slice. No cross-cluster
                      traffic at all, which is what makes a distributed row operator
                      nearly free to start.
    scatter from L1   the ROOT issues one multicast per destination. A cluster cannot
                      reach another cluster's TCDM with a plain copy, so the side that
                      holds the data has to do the moving.
    gather            each cluster pushes its own piece with its own xDMA, so the N-1
                      writes are concurrent rather than N-1 serialised reads by the root.

MULTICAST, NOT xdma_1d_copy, FOR ANY REMOTE WRITE. `xdma_1d_copy` goes through
xdma_memcpy_1d_full_addr, whose writer targets local TCDM; pointed at a remote handle it
completes without writing and the destination keeps whatever it held. Writing another
cluster's TCDM needs the multicast path, which arms a destination slot.
"""

from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs,
                               SnaxBingoKernelXdmaMulticastArgs)

from ..comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                    PortSpec, at_offset)

_ELEM_BYTES = {DType.I8: 1, DType.F16: 2, DType.I32: 4, DType.F32: 4}


def _slice_shape(rows: int, cols: int, clusters, who: str):
    """The per-cluster row count, or the reason this split is refused."""
    n = len(clusters)
    if n < 1:
        raise ValueError(f"{who}: needs at least one cluster.")
    if len(set(clusters)) != n:
        raise ValueError(f"{who}: clusters={tuple(clusters)} repeats a cluster.")
    if rows % n:
        raise ValueError(
            f"{who}: {rows} rows over {n} clusters does not divide. A ragged split is "
            f"expressible, but every consumer would then have to carry the per-cluster "
            f"row count, so it is refused here rather than made everyone's problem.")
    return rows // n


def _check_row_major(layout: Layout, who: str) -> None:
    if Layout(layout) != Layout.ROW_MAJOR:
        raise ValueError(
            f"{who}: layout={layout}. A row slice is a contiguous byte range only in "
            f"row_major; in a blocked layout the rows of one slice are interleaved with "
            f"every other slice's, so taking one is a relayout rather than a move. "
            f"Reshape to row_major first.")


class Scatter(Block):
    """One tensor in, one row slice per cluster out.

    Output ports are named `y_c{cluster}` -- one per cluster in `clusters`, each declared
    in that cluster's L1, so binding one to a block on the wrong cluster is refused by the
    contract rather than discovered as an unwritten buffer.

    WHERE THE TENSOR STARTS IS A BOUNDARY FIELD, so `src_level` left unset is a knob the
    pipeline resolves from the producer. The two realisations are genuinely different
    graphs, not one graph with a flag: from main memory each cluster loads its own slice on
    its own iDMA, and from the ROOT's L1 the root multicasts the other slices out while its
    own is a view costing no node at all. An input already in some OTHER cluster's L1 is
    refused -- this block cannot read it, and pretending otherwise is the silent remote
    read above.
    """

    name = "scatter"

    def __init__(self, *, rows: int, cols: int, clusters, dtype: DType = DType.F16,
                 layout: Layout = Layout.ROW_MAJOR, root: int = None,
                 src_level: MemLevel = None):
        self.src_level = MemLevel(src_level) if src_level is not None else None
        if self.src_level not in (None, MemLevel.L1, MemLevel.L3):
            raise ValueError(
                f"Scatter: src_level={self.src_level}. The tensor is either in main "
                f"memory, where every cluster loads its own slice, or in the root's L1, "
                f"where the root multicasts them out. There is no third route.")
        self.clusters = tuple(int(c) for c in clusters)
        self.root = self.clusters[0] if root is None else int(root)
        if self.root not in self.clusters:
            raise ValueError(
                f"Scatter: root={self.root} is not among clusters={self.clusters}. The "
                f"root is the cluster that holds the tensor when it is already in L1.")
        _check_row_major(layout, "Scatter")
        self.rows, self.cols = int(rows), int(cols)
        self.layout, self.dtype = Layout(layout), DType(dtype)
        self.per = _slice_shape(self.rows, self.cols, self.clusters, "Scatter")
        self.elem = _ELEM_BYTES[self.dtype]

    @property
    def slice_bytes(self) -> int:
        return self.per * self.cols * self.elem

    def variants(self) -> list:
        """The two routes in, unless the caller pinned one."""
        if self.src_level is not None:
            return [{}]
        return [{"src_level": MemLevel.L3}, {"src_level": MemLevel.L1}]

    def respec(self, **params) -> "Scatter":
        return Scatter(rows=self.rows, cols=self.cols, clusters=self.clusters,
                       dtype=self.dtype, layout=self.layout, root=self.root,
                       src_level=params.get("src_level", self.src_level))

    def idma_passes(self) -> int:
        """One load per cluster out of main memory. Nothing on the L1 route: the root's
        slice is a view and the others go out on the xDMA."""
        return len(self.clusters) if self.src_level == MemLevel.L3 else 0

    def xdma_passes(self) -> int:
        """One multicast per destination on the L1 route, all issued by the root."""
        return 0 if self.src_level == MemLevel.L3 else len(self.clusters) - 1

    @property
    def inputs(self) -> dict:
        # The cluster is named on the L1 route and only there: from main memory this
        # block reads a buffer no cluster owns, and the root is where the SENDS are issued
        # rather than where the tensor is.
        return {"x": PortSpec(self.layout, self.dtype, (self.rows, self.cols),
                              mem_level=self.src_level,
                              cluster=(self.root if self.src_level == MemLevel.L1
                                       else None),
                              doc="the whole tensor; main memory, or the root's own L1")}

    @property
    def outputs(self) -> dict:
        return {f"y_c{c}": PortSpec(self.layout, self.dtype, (self.per, self.cols),
                                    mem_level=MemLevel.L1, cluster=c,
                                    doc=f"rows {i * self.per}..{(i + 1) * self.per - 1}")
                for i, c in enumerate(self.clusters)}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        if self.src_level is None:
            raise ValueError(
                "Scatter is a template: src_level not decided. Either pin it, or add the "
                "block to a Pipeline -- run() resolves it from the producer.")
        src = bound["x"].handle
        from_l1 = self.src_level == MemLevel.L1
        outs, nodes, readers = {}, [], []
        for i, c in enumerate(self.clusters):
            g, off = ctx.at(c), i * self.slice_bytes
            if from_l1 and c == self.root:
                # Already here: a view of the caller's own buffer, ordered straight behind
                # whoever wrote it.
                port = Port(self.outputs[f"y_c{c}"], at_offset(src, off),
                            tuple(bound["x"].ends), name=f"y_c{c}")
                outs[f"y_c{c}"] = port
                continue
            dst = g.l1(f"{self.name}_slice", self.slice_bytes)
            if from_l1:
                nd = ctx.at(self.root).node(
                    f"Scatter_c{c}", ctx.xdma, "__snax_bingo_kernel_xdma_multicast",
                    SnaxBingoKernelXdmaMulticastArgs(at_offset(src, off), [dst],
                                                     self.slice_bytes), ())
            else:
                nd = g.node(f"Ld_c{c}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(at_offset(src, off), dst,
                                                          self.slice_bytes), ())
            nodes.append(nd)
            readers.append(nd)
            outs[f"y_c{c}"] = Port(self.outputs[f"y_c{c}"], dst, (nd,), name=f"y_c{c}")
        return BlockResult(
            outputs=outs,
            inputs={"x": Port(self.inputs["x"], src, tuple(readers), name="x")},
            nodes=nodes)


class Gather(Block):
    """One row slice per cluster in, one whole tensor in the root's L1 out.

    Input ports are named `x_c{cluster}`, each declared in that cluster's L1. Every
    non-root cluster pushes its own piece with its own xDMA; the root's piece is a local
    copy on its idle DM core, because a block allocates its own output and cannot be told
    to write into someone else's buffer.
    """

    name = "gather"

    def __init__(self, *, rows: int, cols: int, clusters, dtype: DType = DType.F16,
                 layout: Layout = Layout.ROW_MAJOR, root: int = None):
        self.clusters = tuple(int(c) for c in clusters)
        self.root = self.clusters[0] if root is None else int(root)
        if self.root not in self.clusters:
            raise ValueError(
                f"Gather: root={self.root} is not among clusters={self.clusters}. The "
                f"root is where the gathered tensor lands.")
        _check_row_major(layout, "Gather")
        self.rows, self.cols = int(rows), int(cols)
        self.layout, self.dtype = Layout(layout), DType(dtype)
        self.per = _slice_shape(self.rows, self.cols, self.clusters, "Gather")
        self.elem = _ELEM_BYTES[self.dtype]

    @property
    def slice_bytes(self) -> int:
        return self.per * self.cols * self.elem

    def xdma_passes(self) -> int:
        """One push per non-root cluster, all concurrent."""
        return len(self.clusters) - 1

    def idma_passes(self) -> int:
        """The root's own slice, copied into the gathered buffer."""
        return 1

    @property
    def inputs(self) -> dict:
        return {f"x_c{c}": PortSpec(self.layout, self.dtype, (self.per, self.cols),
                                    mem_level=MemLevel.L1, cluster=c,
                                    doc=f"rows {i * self.per}..{(i + 1) * self.per - 1}")
                for i, c in enumerate(self.clusters)}

    @property
    def outputs(self) -> dict:
        return {"y": PortSpec(self.layout, self.dtype, (self.rows, self.cols),
                              mem_level=MemLevel.L1, cluster=self.root,
                              doc="the slices, back in row order")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        dst = ctx.at(self.root).l1(f"{self.name}_y", self.rows * self.cols * self.elem)
        nodes, ins = [], {}
        for i, c in enumerate(self.clusters):
            g, off = ctx.at(c), i * self.slice_bytes
            src = bound[f"x_c{c}"].handle
            if c == self.root:
                nd = g.node(f"Copy_c{c}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(src, at_offset(dst, off),
                                                          self.slice_bytes), ())
            else:
                nd = g.node(f"Push_c{c}", ctx.xdma, "__snax_bingo_kernel_xdma_multicast",
                            SnaxBingoKernelXdmaMulticastArgs(src, [at_offset(dst, off)],
                                                             self.slice_bytes), ())
            nodes.append(nd)
            ins[f"x_c{c}"] = Port(self.inputs[f"x_c{c}"], src, (nd,), name=f"x_c{c}")
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], dst, tuple(nodes), name="y")},
            inputs=ins, nodes=nodes)
