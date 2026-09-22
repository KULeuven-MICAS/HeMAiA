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

from dataclasses import dataclass

from ..comm import Block, BlockResult, Ctx, DType, Layout, MemLevel, Port, PortSpec
from ..comm.nest import convert_args

_ELEM_BYTES = {DType.I8: 1, DType.F16: 2, DType.F32: 4, DType.I32: 4}


@dataclass(frozen=True)
class ReshapeCfg:
    rows: int
    cols: int
    src: Layout
    dst: Layout
    mesh: tuple
    dtype: DType = DType.F16
    cluster: int = 0


class Reshape(Block):
    """Permute a tensor from one blocked layout to another, in place of nothing else.

      in   x  `src` layout, in L1
      out  y  `dst` layout, in L1

    At FP16 this covers packed<->A, packed<->D and A<->D. It does NOT cover anything
    involving B-layout, which is a transpose and needs the xDMA transposer kernels, and it
    does not cover int8 for A-layout, whose tileSize run is 4 bytes against the xDMA's
    8-byte lane. Both are refused by name rather than emitted and left to fail.
    """

    name = "reshape"

    def __init__(self, cfg: ReshapeCfg = None, **params):
        self.cfg = cfg if cfg is not None else ReshapeCfg(**params)
        c = self.cfg
        object.__setattr__(c, "src", Layout(c.src))
        object.__setattr__(c, "dst", Layout(c.dst))
        object.__setattr__(c, "dtype", DType(c.dtype))
        if c.src == c.dst:
            raise ValueError(
                f"Reshape from {c.src} to itself does nothing. Drop the block; a stage "
                f"that moves no bytes still costs a dispatch and a dependency edge.")
        # DERIVE THE NEST NOW AND THROW IT AWAY. The derivation IS the feasibility test,
        # and it is cheap next to finding out at build that the conversion cannot be done
        # -- by which point a caller has already written the pipeline around it.
        convert_args(c.src, c.dst, c.rows, c.cols, c.mesh,
                     _ELEM_BYTES[c.dtype], 0, 0)

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": PortSpec(c.src, c.dtype, (c.rows, c.cols), mem_level=MemLevel.L1,
                              doc=f"{c.src} layout")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(c.dst, c.dtype, (c.rows, c.cols), mem_level=MemLevel.L1,
                              doc=f"{c.dst} layout")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        eb = _ELEM_BYTES[c.dtype]
        out = g.l1(f"{self.name}_{c.src}2{c.dst}", c.rows * c.cols * eb)
        nd = g.node(f"Reshape_{c.src}2{c.dst}", ctx.xdma,
                    "__snax_bingo_kernel_xdma_6d",
                    convert_args(c.src, c.dst, c.rows, c.cols, c.mesh, eb,
                                 bound["x"].handle, out))
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], out, (nd,), cluster=c.cluster, name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (nd,), name="x")},
            nodes=[nd])
