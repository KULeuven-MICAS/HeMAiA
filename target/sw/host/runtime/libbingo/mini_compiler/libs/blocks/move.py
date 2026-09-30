# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Moving a tensor between clusters, L1 to L1, as blocks -- and the two views that cost
nothing.

WHY MOVES ARE BLOCKS. check_contract refuses an L1 port on another cluster outright: a GEMM
or SIMD kernel addresses only its own TCDM, and a remote handle does not fault -- the
transfer completes without writing and the consumer reads whatever the buffer held. So a
tensor produced on one cluster and consumed on another needs a node that moves it, and the
linker may not insert one (node creation order is dispatch order). The layer names the move
the same way it names a Reshape, and it shows up in the graph and in the cost.

TWO ENGINES, CHOSEN BY WHO MAY BE BUSY.

  Broadcast   ONE xDMA multicast on the SOURCE cluster: one read, N writes. The cheapest
              fan-out there is -- but a remote write is absorbed by the RECEIVING cluster's
              xDMA, which must be idle while it lands (four busy receivers wedged the fabric,
              see libs/blocks/flash_attention.py on the headpar broadcast). Use it where the
              receivers have not started yet, e.g. the first thing a layer hands out.

  Pull        ONE iDMA read on the DESTINATION cluster's DM core. No xDMA on either side,
              so it is safe while both clusters are busy -- the pattern FlashAttention's own
              headpar V path runs on. Several sources may be stacked into one destination
              buffer, which is how per-tile producers become one contiguous operand.

Every move is a contiguous copy. A row range of a tensor is contiguous in row_major, A and D
(whole meshRow blocks), and never in B or col_major; the moves refuse a slice they cannot
express as one run rather than copy the wrong bytes.

THE VIEWS emit no node and allocate nothing: `Slice` names a row range of a buffer, and
`TransposeView` relabels A(X) as B(X^T) -- byte-identical when meshRow == meshCol, which is
what lets one A-layout activation feed a GEMM as either operand.
"""

from dataclasses import dataclass
from typing import Optional

from bingo_kernel_args import (
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaMulticastArgs,
)

from ..comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port, PortSpec,
                    at_offset)

_ELEM_BYTES = {DType.I8: 1, DType.F16: 2, DType.F32: 4, DType.I32: 4}
# Row ranges are one contiguous run only in these; B and col_major run down columns.
_ROW_CONTIGUOUS = (Layout.ROW_MAJOR, Layout.A, Layout.D)


def _row_span(layout, dtype, cols, row0, nrows, mesh):
    """(byte offset, byte count) of rows [row0, row0 + nrows), or a refusal.

    In A and D a meshRow block holds meshRow COMPLETE rows (A: [K/Ku][Mu][Ku], D:
    [N/Nu][Mu][Nu]), so a range of whole blocks is one run starting at row0 * cols elements
    -- the same arithmetic as row_major, provided the range is block-aligned.
    """
    if layout not in _ROW_CONTIGUOUS:
        raise ValueError(f"a row range of a {layout} tensor is not contiguous (it runs "
                         f"down columns); move the whole tensor or reshape first.")
    if layout in (Layout.A, Layout.D) and (row0 % mesh[0] or nrows % mesh[0]):
        raise ValueError(f"rows [{row0}, {row0 + nrows}) of a {layout} tensor split a "
                         f"meshRow={mesh[0]} block, which is not one contiguous run.")
    eb = _ELEM_BYTES[dtype]
    return row0 * cols * eb, nrows * cols * eb


@dataclass(frozen=True)
class MoveCfg:
    """What moves: a [rows, cols] tensor, or rows [row0, row0 + nrows) of it."""

    rows: int
    cols: int
    layout: Layout
    dtype: DType
    src: int                        # the cluster whose L1 holds the tensor
    dst: tuple                      # the clusters that receive it (Pull: exactly one)
    row0: int = 0
    nrows: Optional[int] = None     # None: to the end
    mesh: tuple = (16, 4, 16)

    @property
    def n(self) -> int:
        return self.rows - self.row0 if self.nrows is None else self.nrows


class Broadcast(Block):
    """One multicast on the source cluster's xDMA: `x` (or a row range of it) lands in the
    L1 of every cluster in `dst`, from ONE read.

      in   x       [rows, cols] in cluster `src`'s L1
      out  y_c{c}  [nrows, cols] in cluster c's L1, one port per destination

    THE RECEIVERS' xDMAs MUST BE IDLE while it lands -- see the module doc. The caller owns
    that ordering; this block cannot see what else the receivers run.
    """

    name = "broadcast"

    def __init__(self, cfg: MoveCfg = None, **params):
        self.cfg = cfg if cfg is not None else MoveCfg(**params)
        c = self.cfg
        if not c.dst:
            raise ValueError("Broadcast: no destination cluster.")
        if c.src in c.dst:
            raise ValueError(f"Broadcast: cluster {c.src} is the source; it already holds "
                             f"the tensor -- bind the producer's port there directly.")
        if not 1 <= len(c.dst) <= SnaxBingoKernelXdmaMulticastArgs.DST_MAX:
            raise ValueError(f"Broadcast: {len(c.dst)} destinations; the multicast takes "
                             f"1..{SnaxBingoKernelXdmaMulticastArgs.DST_MAX}.")
        self.off, self.nbytes = _row_span(c.layout, c.dtype, c.cols, c.row0, c.n, c.mesh)

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": PortSpec(c.layout, c.dtype, (c.rows, c.cols), mem_level=MemLevel.L1,
                              cluster=c.src, doc="the tensor, where it was produced")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {f"y_c{d}": PortSpec(c.layout, c.dtype, (c.n, c.cols), mem_level=MemLevel.L1,
                                    cluster=d, doc=f"cluster {d}'s copy")
                for d in c.dst}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        bufs = {d: ctx.at(d).l1(f"{self.name}_y", self.nbytes) for d in c.dst}
        nd = ctx.at(c.src).node(
            "Mcast", ctx.xdma, "__snax_bingo_kernel_xdma_multicast",
            SnaxBingoKernelXdmaMulticastArgs(at_offset(bound["x"].handle, self.off),
                                             [bufs[d] for d in c.dst], self.nbytes))
        outs = {f"y_c{d}": Port(self.outputs[f"y_c{d}"], bufs[d], (nd,), name=f"y_c{d}")
                for d in c.dst}
        return BlockResult(outputs=outs,
                           inputs={"x": Port(self.inputs["x"], bound["x"].handle, (nd,),
                                             name="x")},
                           nodes=[nd])


class Pull(Block):
    """The DESTINATION cluster's iDMA reads one or more tensors (or row ranges of them)
    from other clusters' L1 and stacks them, in order, into one buffer of its own.

      in   x{i}  [rows, cols] in cluster `src[i]`'s L1          (i = 0 .. len(src)-1)
      out  y     [sum of the moved rows, cols] in cluster `dst`'s L1

    Stacking is concatenation of the moved runs, so the output is itself a row-stacked
    tensor in the same layout -- which is what turns per-tile producers into the one
    contiguous operand a consumer wants. One copy node per source, all on the one DM core,
    in the order given. A source on `dst` itself is a local copy, which is still how two
    separate buffers become one.
    """

    name = "pull"

    def __init__(self, *, rows: int, cols: int, layout: Layout, dtype: DType, src,
                 dst: int, row0=0, nrows=None, mesh: tuple = (16, 4, 16)):
        self.src = tuple(src) if isinstance(src, (list, tuple)) else (src,)
        k = len(self.src)
        self.row0 = tuple(row0) if isinstance(row0, (list, tuple)) else (row0,) * k
        self.nrows = tuple(nrows) if isinstance(nrows, (list, tuple)) else (nrows,) * k
        if len(self.row0) != k or len(self.nrows) != k:
            raise ValueError("Pull: src, row0 and nrows must have one entry per source.")
        self.rows, self.cols, self.dst, self.mesh = rows, cols, dst, mesh
        self.layout, self.dtype = Layout(layout), DType(dtype)
        self.spans = []
        for r0, nr in zip(self.row0, self.nrows):
            n = rows - r0 if nr is None else nr
            self.spans.append((n,) + _row_span(self.layout, self.dtype, cols, r0, n, mesh))

    @property
    def inputs(self) -> dict:
        return {f"x{i}": PortSpec(self.layout, self.dtype, (self.rows, self.cols),
                                  mem_level=MemLevel.L1, cluster=s,
                                  doc=f"source {i}, on cluster {s}")
                for i, s in enumerate(self.src)}

    @property
    def outputs(self) -> dict:
        total = sum(n for n, _, _ in self.spans)
        return {"y": PortSpec(self.layout, self.dtype, (total, self.cols),
                              mem_level=MemLevel.L1, cluster=self.dst,
                              doc=f"the moved rows, stacked, on cluster {self.dst}")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        g = ctx.at(self.dst)
        total = sum(nb for _, _, nb in self.spans)
        buf = g.l1(f"{self.name}_y", total)
        nodes, ins, at = [], {}, 0
        for i, (_, off, nb) in enumerate(self.spans):
            nd = g.node(f"Pull{i}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(at_offset(bound[f"x{i}"].handle, off),
                                                      buf.view(at) if at else buf, nb))
            nodes.append(nd)
            ins[f"x{i}"] = Port(self.inputs[f"x{i}"], bound[f"x{i}"].handle, (nd,),
                                name=f"x{i}")
            at += nb
        return BlockResult(outputs={"y": Port(self.outputs["y"], buf, tuple(nodes), name="y")},
                           inputs=ins, nodes=nodes)


class Slice(Block):
    """Rows [row0, row0 + nrows) of a tensor, as a VIEW: no node, no copy, same buffer.

      in   x  [rows, cols]
      out  y  [nrows, cols], the same memory, level and cluster

    The output's producers are the input's: a consumer of the slice waits for whatever
    wrote the whole tensor.
    """

    name = "slice"

    def __init__(self, *, rows: int, cols: int, layout: Layout, dtype: DType,
                 row0: int, nrows: int, mem_level: MemLevel = MemLevel.L1,
                 cluster: Optional[int] = None, mesh: tuple = (16, 4, 16)):
        self.rows, self.cols, self.row0, self.nrows = rows, cols, row0, nrows
        self.layout, self.dtype = Layout(layout), DType(dtype)
        self.mem_level = MemLevel(mem_level)
        self.cluster = cluster if self.mem_level == MemLevel.L1 else None
        if row0 < 0 or row0 + nrows > rows:
            raise ValueError(f"Slice: rows [{row0}, {row0 + nrows}) outside [0, {rows}).")
        self.off, _ = _row_span(self.layout, self.dtype, cols, row0, nrows, mesh)

    def _spec(self, shape, doc):
        return PortSpec(self.layout, self.dtype, shape, mem_level=self.mem_level,
                        cluster=self.cluster, doc=doc)

    @property
    def inputs(self) -> dict:
        return {"x": self._spec((self.rows, self.cols), "the whole tensor")}

    @property
    def outputs(self) -> dict:
        return {"y": self._spec((self.nrows, self.cols),
                                f"rows [{self.row0}, {self.row0 + self.nrows})")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        x = bound["x"]
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], at_offset(x.handle, self.off),
                               tuple(x.ends), name="y")},
            inputs={"x": Port(self.inputs["x"], x.handle, (), name="x")},
            nodes=[])


class TransposeView(Block):
    """A(X) [rows, cols] relabelled as B(X^T) [cols, rows]: the same bytes, no node.

    A's block is [Mu][Ku] with a row of X along Ku; B's is [Nu][Ku] with a column of X^T --
    that is, the same row of X -- along Ku. With Mu == Nu the two orders coincide byte for
    byte, whatever the element width, which is what lets an A-layout activation feed a GEMM
    as its B operand (V^T = Wv^T . X^T takes X^T as B).
    """

    name = "transpose_view"

    def __init__(self, *, rows: int, cols: int, dtype: DType,
                 mem_level: MemLevel = MemLevel.L1, cluster: Optional[int] = None,
                 mesh: tuple = (16, 4, 16)):
        if mesh[0] != mesh[2]:
            raise ValueError(f"TransposeView: A(X) is B(X^T) only when meshRow == meshCol; "
                             f"mesh={tuple(mesh)}.")
        self.rows, self.cols, self.dtype = rows, cols, DType(dtype)
        self.mem_level = MemLevel(mem_level)
        self.cluster = cluster if self.mem_level == MemLevel.L1 else None

    @property
    def inputs(self) -> dict:
        return {"x": PortSpec(Layout.A, self.dtype, (self.rows, self.cols),
                              mem_level=self.mem_level, cluster=self.cluster, doc="A(X)")}

    @property
    def outputs(self) -> dict:
        return {"y": PortSpec(Layout.B, self.dtype, (self.cols, self.rows),
                              mem_level=self.mem_level, cluster=self.cluster,
                              doc="B(X^T): the same bytes")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        x = bound["x"]
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], x.handle, tuple(x.ends), name="y")},
            inputs={"x": Port(self.inputs["x"], x.handle, (), name="x")},
            nodes=[])


class View(Block):
    """The same bytes under another name: a relabel, no node and no copy.

      in   x  `src`
      out  y  `dst`, at x + offset

    For what the other views cannot say -- a column range of a row, a row read as several
    rows ([1, 2048] as [4, 512]), one row of a stack. Both specs must name the same memory
    and cluster; that the bytes agree is the caller's claim, as it is for any view.
    The output's producers are the input's.
    """

    name = "view"

    def __init__(self, *, src: PortSpec, dst: PortSpec, offset: int = 0):
        if src.mem_level != dst.mem_level or src.cluster != dst.cluster:
            raise ValueError(f"View: {src.describe()} -> {dst.describe()} changes memory; a "
                             f"view is the same bytes, a move is a block (Pull, Fetch).")
        if offset < 0:
            raise ValueError(f"View: offset {offset} < 0.")
        self.src, self.dst, self.offset = src, dst, int(offset)

    @property
    def inputs(self) -> dict:
        return {"x": self.src}

    @property
    def outputs(self) -> dict:
        return {"y": self.dst}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        x = bound["x"]
        return BlockResult(
            outputs={"y": Port(self.dst, at_offset(x.handle, self.offset), tuple(x.ends),
                               name="y")},
            inputs={"x": Port(self.src, x.handle, (), name="x")},
            nodes=[])


class Join(Block):
    """Consecutive pieces of one buffer under one name: a relabel, no node and no copy.

      in   x0 .. x{n-1}  `parts`, piece i of the buffer
      out  y             `dst`, at x0

    For a tensor several blocks write piecewise -- a pass's tokens, normalised on different
    clusters into one L3 buffer -- and later blocks read whole. The output waits for every
    piece's producers. That the pieces are where `dst` says, back to back from x0, is the
    caller's claim, as it is for a View.
    """

    name = "join"

    def __init__(self, *, parts: list, dst: PortSpec):
        if not parts:
            raise ValueError("Join: no parts.")
        if any(p.mem_level != dst.mem_level or p.cluster != dst.cluster for p in parts):
            raise ValueError(f"Join: every part must be in {dst.describe()}'s memory; a join "
                             f"is the same bytes, a move is a block (Pull, Fetch).")
        self.parts, self.dst = list(parts), dst

    @property
    def inputs(self) -> dict:
        return {f"x{i}": p for i, p in enumerate(self.parts)}

    @property
    def outputs(self) -> dict:
        return {"y": self.dst}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        ends = tuple(e for i in range(len(self.parts)) for e in bound[f"x{i}"].ends)
        return BlockResult(
            outputs={"y": Port(self.dst, bound["x0"].handle, ends, name="y")},
            inputs={f"x{i}": Port(p, bound[f"x{i}"].handle, (), name=f"x{i}")
                    for i, p in enumerate(self.parts)},
            nodes=[])


class Fetch(Block):
    """A tensor from L3 or the HBM into one cluster's L1: one iDMA copy on its DM core.

      in   x  `src` (L3 / HBM)
      out  y  the same spec in cluster `cluster`'s L1 (the first nbytes, from `offset`)
    """

    name = "fetch"

    def __init__(self, *, src: PortSpec, cluster: int, nbytes: int, offset: int = 0,
                 dst: PortSpec = None):
        if src.mem_level not in (MemLevel.L3, MemLevel.HBM):
            raise ValueError(f"Fetch: the source is in {src.mem_level}; Fetch reads L3 or "
                             f"the HBM (Pull moves between L1s).")
        self.src, self.cluster, self.nbytes, self.offset = src, cluster, int(nbytes), int(offset)
        self.dst = dst if dst is not None else PortSpec(src.layout, src.dtype, src.shape,
                                                        mem_level=MemLevel.L1, cluster=cluster,
                                                        doc=src.doc)

    @property
    def inputs(self) -> dict:
        return {"x": self.src}

    @property
    def outputs(self) -> dict:
        return {"y": self.dst}

    def idma_passes(self) -> int:
        return 1

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        g = ctx.at(self.cluster)
        buf = g.l1(f"{self.name}_y", self.nbytes)
        nd = g.node("Fetch", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(at_offset(bound["x"].handle, self.offset),
                                                  buf, self.nbytes),
                    list(bound["x"].ends))
        return BlockResult(outputs={"y": Port(self.dst, buf, (nd,), name="y")},
                           inputs={"x": Port(self.src, bound["x"].handle, (nd,), name="x")},
                           nodes=[nd], sources=[nd] if not bound["x"].ends else [])


class Stash(Block):
    """A copy of an L1 buffer in L3, made by its OWN cluster's DM core as soon as it exists.

      in   x  [..] in cluster c's L1
      out  y  the same bytes in L3

    For checking a layer whose L1 is reused: the host compares the L3 copy at the end, so
    the buffer itself may be recycled for the next stage the moment this copy is done, and
    no host read of an L1 races the clusters that are still computing.
    """

    name = "stash"

    def __init__(self, *, src: PortSpec, nbytes: int, offset: int = 0, dst=None):
        """`dst`: an L3 handle to copy into (a staged symbol, e.g. a hand-off buffer another
        chip reads); None allocates one on this chip."""
        if src.mem_level != MemLevel.L1:
            raise ValueError(f"Stash: {src.describe()} is not in L1; check it where it is.")
        self.src, self.nbytes, self.offset, self.dst = src, int(nbytes), int(offset), dst

    @property
    def inputs(self) -> dict:
        return {"x": self.src}

    @property
    def outputs(self) -> dict:
        s = self.src
        return {"y": PortSpec(s.layout, s.dtype, s.shape, mem_level=MemLevel.L3,
                              doc=f"L3 copy of {s.doc}")}

    def idma_passes(self) -> int:
        return 1

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        g = ctx.at(self.src.cluster)
        h = self.dst if self.dst is not None else ctx.l3(f"{self.name}_y", self.nbytes)
        nd = g.node("Stash", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(at_offset(bound["x"].handle, self.offset),
                                                  h, self.nbytes),
                    list(bound["x"].ends))
        return BlockResult(outputs={"y": Port(self.outputs["y"], h, (nd,), name="y")},
                           inputs={"x": Port(self.src, bound["x"].handle, (nd,), name="x")},
                           nodes=[nd])
