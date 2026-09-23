# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The operators that touch each value independently, and therefore do not care how the
values are ordered.

WHY THAT IS THE AXIS THIS FILE IS CUT ON. A permutation of the inputs of an elementwise op
is the same permutation of its output, so quantise, dequantise and the residual add are
correct in ANY layout and in either orientation -- MoE runs its SwiGLU straight over a
D-layout block for exactly this reason. Their ports carry a `layout` so the contract can
still state what they were handed -- layout included, orientation with it -- but neither
changes what they compute.

The per-row operators are the ones where layout is load-bearing (norm.py, rope.py). The
split is not tidiness: it is the difference between an operand a block may accept as it
finds it and one it has to convert first.
"""

from dataclasses import dataclass, replace

from bingo_kernel_args import (SnaxBingoKernelSimdAddF16Args,
                               SnaxBingoKernelSimdFp16ToInt8Args,
                               SnaxBingoKernelSimdScaleF16Args)

from ...comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                     PortSpec)
from .common import RowCfg, check_row


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
                 layout: Layout = Layout.ROW_MAJOR, **params):
        if inv_scale_f32bits is None:
            raise ValueError(
                "Quantize needs inv_scale_f32bits: the right scale depends on the range "
                "of the data, and a wrong one saturates the tensor rather than failing.")
        self.cfg = cfg if cfg is not None else RowCfg(**params)
        self.inv_scale_f32bits = int(inv_scale_f32bits)
        self.layout = Layout(layout)
        check_row(self.cfg.cols, "Quantize")

    def simd_passes(self) -> int:
        """ONE SIMD task, for comm.layout_pass: what fusing this step into its producer saves."""
        return 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": PortSpec(self.layout, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1,
                              doc="fp16 in this block's layout")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(self.layout, DType.I8, (c.rows, c.cols),
                              mem_level=MemLevel.L1,
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
            outputs={"y": Port(self.outputs["y"], out, (nd,), cluster=c.cluster, name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (nd,), name="x")},
            nodes=[nd])


class Residual(Block):
    """a + b, elementwise, on the SIMD core. The skip connection.

    NOT THE xDMA ADD. `__snax_bingo_kernel_xdma_elementwise_add_ab` drives the
    HasElementwiseAdd WRITER extension, which snax_split_cluster does not have. That path
    still gives the right answer -- it falls back to scalar C -- but it sums every int32
    element one at a time on the xDMA hart. This drives HasStreamElementwise, a reader
    extension the cluster does have, so the residual stays a vector op.

    Elementwise, so the layout passes through; both operands must already agree on one,
    which the contract checks.
    """

    name = "residual"

    def __init__(self, cfg: RowCfg = None, *, layout: Layout = Layout.ROW_MAJOR,
                 dtype: DType = DType.F16, **params):
        self.cfg = cfg if cfg is not None else RowCfg(**params)
        self.layout, self.dtype = Layout(layout), DType(dtype)

    def simd_passes(self) -> int:
        """ONE SIMD task, for comm.layout_pass: what fusing this step into its producer saves."""
        return 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        spec = PortSpec(self.layout, self.dtype, (c.rows, c.cols), mem_level=MemLevel.L1)
        return {"a": replace(spec, doc="the branch output"),
                "b": replace(spec, doc="the skip")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(self.layout, self.dtype, (c.rows, c.cols),
                              mem_level=MemLevel.L1,
                              doc="a + b")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        if self.dtype != DType.F16:
            raise ValueError(
                f"Residual is fp16 only on this cluster: the add runs on "
                f"HasStreamElementwise, whose op is ADD_FP16. Got {self.dtype}.")
        g = ctx.at(c.cluster)
        out = g.l1(f"{self.name}_y", c.rows * c.cols * 2)
        nd = g.node("Add", ctx.simd, "__snax_bingo_kernel_simd_stream_elementwise",
                    SnaxBingoKernelSimdAddF16Args(
                        bound["a"].handle, bound["b"].handle, out,
                        rows=c.rows, cols=c.cols))
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], out, (nd,), cluster=c.cluster, name="y")},
            inputs={n: Port(self.inputs[n], bound[n].handle, (nd,), name=n)
                    for n in ("a", "b")},
            nodes=[nd])


class Dequantize(Block):
    """out = x * scale. The step back from GEMM output to activation range.

    WHY A LAYER NEEDS THIS, stated in the two hard limits that bracket the GEMM -- both
    fp16, both silent when exceeded:

      THE D PORT narrows the array's int32 accumulator to fp16, so `x_q @ w_q` must stay
      under 65504. That is what keeps the operands small: at d=128 with int8 activations
      an |w| of more than about 2 already saturates.

      RMSNorm reduces SUM(x^2) into an fp16 SCALAR before the rsqrt sees it, so its input
      needs sum_j x[j]^2 < 65504 -- about |x| < 22 at d=128. Past that the reduce saturates
      and 1/sqrt of a saturated sum is a power-of-two-wrong answer rather than an error:
      the output looks like a well-formed tensor scaled by 2^16. (That limit is the FP16
      narrow in the transport grid, not the rsqrt, so it did not move when the scalar
      epilogue did -- see norm.py.)

    Between the two sits a factor of roughly a thousand, which is exactly the product of
    the operands' quantisation scales, and nothing else in the chain is free to absorb it.
    So the scale is applied here, on the SIMD core, in one pass.

    Elementwise, so the layout passes straight through.
    """

    name = "dequantize"

    def __init__(self, cfg: RowCfg = None, *, scale_f32bits: int = None,
                 layout: Layout = Layout.ROW_MAJOR, **params):
        if scale_f32bits is None:
            raise ValueError(
                "Dequantize needs scale_f32bits: it is 1/(scale_x * scale_w) for the GEMM "
                "that produced this tensor, and no default can be right for every stage.")
        self.cfg = cfg if cfg is not None else RowCfg(**params)
        self.scale_f32bits = int(scale_f32bits)
        self.layout = Layout(layout)
        check_row(self.cfg.cols, "Dequantize")

    def simd_passes(self) -> int:
        """ONE SIMD task, for comm.layout_pass: what fusing this step into its producer saves."""
        return 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": PortSpec(self.layout, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1,
                              doc="fp16 straight off the GEMM's D port")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(self.layout, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1,
                              doc="fp16 back in activation range, same layout")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        out = g.l1(f"{self.name}_y", c.rows * c.cols * 2)
        nd = g.node("Dequant", ctx.simd, "__snax_bingo_kernel_simd_stream_map",
                    SnaxBingoKernelSimdScaleF16Args(
                        bound["x"].handle, out, scale_f32bits=self.scale_f32bits,
                        rows=c.rows, cols=c.cols))
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], out, (nd,), cluster=c.cluster, name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (nd,), name="x")},
            nodes=[nd])
