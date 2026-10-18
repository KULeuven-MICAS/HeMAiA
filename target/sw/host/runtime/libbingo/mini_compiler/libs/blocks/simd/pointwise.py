# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The operators that touch each value independently, and therefore do not care how the
values are ordered.

WHY THAT IS THE AXIS THIS FILE IS CUT ON. A permutation of the inputs of an elementwise op
is the same permutation of its output, so quantise is correct in ANY layout and in either
orientation. Its ports carry a `layout` so the contract can still state what it was handed
-- layout included, orientation with it -- but neither changes what it computes.

The per-row operators are the ones where layout is load-bearing (norm.py). The
split is not tidiness: it is the difference between an operand a block may accept as it
finds it and one it has to convert first.
"""

from bingo_kernel_args import SnaxBingoKernelSimdFp16ToInt8Args

from ...comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                     PortSpec)
from .common import RowCfg, check_row

# An elementwise operator is correct in ANY of these, because a permutation of its inputs
# is the same permutation of its output. So `layout` is a field these blocks leave open by
# default and the resolver fills from the neighbours; pinning it is how a caller states a
# boundary it wants to hold fixed.
_ELEMENTWISE = (Layout.ROW_MAJOR, Layout.COL_MAJOR, Layout.A, Layout.B, Layout.D,
                Layout.D32)


class Quantize(Block):
    """FP16 -> INT8 with an explicit scale. Elementwise, so the layout passes straight
    through -- whatever goes in comes out in the same order.

    THE SCALE IS AN ARGUMENT AND HAS NO DEFAULT. It has to come from the range of the data:
    a scale sized for activations near [-8, 8] saturates a tensor that reaches the
    thousands, and a chain whose every value is +-127 agrees with its golden no matter what
    produced it. Choose a power of two so the scaling is exact in fp16 and adds no error of
    its own -- see int8_scale_for() in the MoE datagen.
    """

    name = "quantize"

    def __init__(self, cfg: RowCfg = None, *, inv_scale_f32bits: int = None,
                 layout: Layout = None, **params):
        if inv_scale_f32bits is None:
            raise ValueError(
                "Quantize needs inv_scale_f32bits: the right scale depends on the range "
                "of the data, and a wrong one saturates the tensor rather than failing.")
        self.cfg = cfg if cfg is not None else RowCfg(**params)
        self.inv_scale_f32bits = int(inv_scale_f32bits)
        self.layout = Layout(layout) if layout is not None else None
        check_row(self.cfg.cols, "Quantize")

    def variants(self) -> list:
        """Any layout: quantising is elementwise, so it passes whatever it is handed."""
        return [{}] if self.layout is not None else [{"layout": l} for l in _ELEMENTWISE]

    def respec(self, **p) -> "Quantize":
        return Quantize(cfg=self.cfg, inv_scale_f32bits=self.inv_scale_f32bits,
                        layout=p.get("layout", self.layout))

    def simd_passes(self) -> int:
        """ONE SIMD task. What this step costs the engine the resolver ranks first."""
        return 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": PortSpec(self.layout, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1, cluster=c.cluster,
                              doc="fp16 in this block's layout")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(self.layout, DType.I8, (c.rows, c.cols),
                              mem_level=MemLevel.L1, cluster=c.cluster,
                              doc="int8, same layout -- quantising is elementwise")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        out = g.l1(f"{self.name}_y", c.rows * c.cols)
        nd = g.node("Quant", ctx.simd, "__snax_bingo_kernel_simd_fp16_to_int8",
                    SnaxBingoKernelSimdFp16ToInt8Args(
                        bound["x"].handle, out,
                        beats=(c.rows * c.cols * 2) // 64, rows=1,
                        inv_scale_f32bits=self.inv_scale_f32bits))
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], out, (nd,), name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (nd,), name="x")},
            nodes=[nd])
