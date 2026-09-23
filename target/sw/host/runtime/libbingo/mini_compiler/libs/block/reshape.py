# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""An explicit layout change, as a block: one xDMA pass between two stages.

WHY THIS IS A BLOCK AND NOT AUTOMATIC. `comm.transfer` closes a layout gap for a block that
FETCHES its own operands -- it folds the conversion into the load, so it costs nothing
beyond the load that had to happen anyway. Between two blocks that both live in L1 there is
no load to fold into, and the linker may not insert a node of its own (node creation order
is dispatch order). So the conversion becomes a stage the layer names, which is also what
makes it visible in the graph and in the cost.

This is the block `check_contract` is pointing at when it says "put an explicit conversion
block in the pipeline".

THE CONVERSION IS DERIVED AND THEN CHECKED, not trusted: comm.nest walks the strides it
derived against ground-truth index maps of both layouts before the node is built. A
conversion the hardware cannot do -- a transpose, or a run too narrow at this precision --
is refused here, at graph build, with the reason.
"""

import itertools
from dataclasses import dataclass
from typing import Optional

from bingo_kernel_args import SnaxBingoKernelXdmaTranspose2dArgs

from ..comm import Block, BlockResult, Ctx, DType, Layout, MemLevel, Port, PortSpec
from ..comm.nest import convert_args
from ..comm.transfer import plan
from ..comm.variant import free_fields

_ELEM_BYTES = {DType.I8: 1, DType.F16: 2, DType.F32: 4, DType.I32: 4}

# What a reshape may be asked to convert between. d32 and monoid are left out on purpose:
# they are the D port's int32 scatter and the junction's lane geometry, neither of which is
# reached by a strided nest, and offering them would only fill the search with refusals.
_CONVERTIBLE = (Layout.ROW_MAJOR, Layout.COL_MAJOR, Layout.A, Layout.B, Layout.D)


@dataclass(frozen=True)
class ReshapeCfg:
    """`src`, `dst` and `dtype` may be left None: then they follow from the neighbours.

    A field the caller PINS is a constraint and survives into every realisation. That is
    how a layer keeps hold of a boundary it cares about while letting the rest be worked
    out -- the common case being a Reshape written between two stages whose layouts the
    layer has not decided yet and does not want to.
    """

    rows: int
    cols: int
    mesh: tuple
    src: Optional[Layout] = None
    dst: Optional[Layout] = None
    dtype: Optional[DType] = None
    cluster: int = 0


class Reshape(Block):
    """Permute a tensor from one blocked layout to another, in place of nothing else.

      in   x  `src` layout, in L1
      out  y  `dst` layout, in L1

    At FP16 this covers row_major<->A, row_major<->D and A<->D in ONE pass, and the
    col_major pairs in TWO -- comm.transfer.plan decomposes those through row_major,
    because the 8x8 transposer permutes a plain array and a stride nest has to be derived
    in the blocked side's own dimensions. Accepting them matters: a per-row SIMD operator
    upstream may well have chosen to emit col_major, and a Reshape that refused the result
    would make that choice unbuildable.

    It does NOT cover anything involving B-layout, which is a transpose between two
    BLOCKED layouts and needs the dedicated transposer kernels, and it does not cover int8
    for A-layout, whose tileSize run is 4 bytes against the xDMA's 8-byte lane. Both are
    refused by name rather than emitted and left to fail.
    """

    name = "reshape"

    def __init__(self, cfg: ReshapeCfg = None, **params):
        self.cfg = cfg if cfg is not None else ReshapeCfg(**params)
        c = self.cfg
        # A TEMPLATE, not a realisation: something is still to be decided, so there is no
        # conversion to plan and no port to declare. The resolver reaches the realisations
        # through variants(); anything else touching it gets told which fields are open.
        self.free = free_fields(c, ("src", "dst", "dtype"))
        if self.free:
            self.steps = None
            return
        object.__setattr__(c, "src", Layout(c.src))
        object.__setattr__(c, "dst", Layout(c.dst))
        object.__setattr__(c, "dtype", DType(c.dtype))
        if c.src == c.dst:
            raise ValueError(
                f"Reshape from {c.src} to itself does nothing. Drop the block; a stage "
                f"that moves no bytes still costs a dispatch and a dependency edge.")
        # PLAN IT NOW AND KEEP THE PLAN. The planning IS the feasibility test, and it is
        # cheap next to finding out at build that the conversion cannot be done -- by which
        # point a caller has already written the pipeline around it. plan() also derives
        # every nest it proposes, so an inexpressible conversion still raises here, with
        # its own reason.
        self.steps = plan(self.inputs["x"], self.outputs["y"],
                          mesh=c.mesh, elem_bytes=_ELEM_BYTES[c.dtype])

    def variants(self) -> list:
        """Every (src, dst, precision) this reshape could be. Illegal ones refuse in
        __init__ -- a transpose between two blocked layouts, an A-layout run too narrow at
        int8 -- and drop out of the search with their own message."""
        if not self.free:
            return [{}]
        choices = {"src": _CONVERTIBLE, "dst": _CONVERTIBLE,
                   "dtype": (DType.F16, DType.I8)}
        return [dict(zip(self.free, combo))
                for combo in itertools.product(*(choices[n] for n in self.free))]

    def xdma_passes(self) -> int:
        """One for a pure reshape, two when an orientation change rides along."""
        return len(self.steps)

    def _check_realised(self) -> None:
        if self.free:
            raise ValueError(
                f"Reshape is a template: {', '.join(self.free)} not decided. Either pin "
                f"them, or add it to a Pipeline -- run() resolves them from the "
                f"neighbours and builds the realisation it picked.")

    @property
    def inputs(self) -> dict:
        self._check_realised()
        c = self.cfg
        return {"x": PortSpec(c.src, c.dtype, (c.rows, c.cols), mem_level=MemLevel.L1,
                              doc=f"{c.src} layout")}

    @property
    def outputs(self) -> dict:
        self._check_realised()
        c = self.cfg
        return {"y": PortSpec(c.dst, c.dtype, (c.rows, c.cols), mem_level=MemLevel.L1,
                              doc=f"{c.dst} layout")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        self._check_realised()
        c = self.cfg
        g = ctx.at(c.cluster)
        eb = _ELEM_BYTES[c.dtype]
        nbytes = c.rows * c.cols * eb
        src, nodes = bound["x"].handle, []
        # WALK THE PLAN. One step for a pure reshape -- the same single node, with the same
        # name, this block has always emitted -- and two when an orientation change is in
        # play, since the transposer has to run on the unblocked side.
        lay = c.src
        for st in self.steps:
            if st.kind == "transpose":
                rows, cols = (c.cols, c.rows) if lay == Layout.COL_MAJOR else (c.rows, c.cols)
                dst = g.l1(f"{self.name}_t", nbytes)
                nodes.append(g.node(
                    f"Transpose_{self.name}", ctx.xdma,
                    "__snax_bingo_kernel_xdma_transpose_2d",
                    SnaxBingoKernelXdmaTranspose2dArgs(src, dst, rows, cols, eb),
                    nodes[-1] if nodes else ()))
                lay = Layout.ROW_MAJOR if lay == Layout.COL_MAJOR else Layout.COL_MAJOR
            else:
                to = Layout.ROW_MAJOR if c.dst == Layout.COL_MAJOR else c.dst
                dst = g.l1(f"{self.name}_{lay}2{to}", nbytes)
                nodes.append(g.node(
                    f"Reshape_{lay}2{to}", ctx.xdma, "__snax_bingo_kernel_xdma_6d",
                    convert_args(lay, to, c.rows, c.cols, c.mesh, eb, src, dst),
                    nodes[-1] if nodes else ()))
                lay = to
            src = dst
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], src, (nodes[-1],), cluster=c.cluster,
                               name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (nodes[0],), name="x")},
            nodes=nodes)
