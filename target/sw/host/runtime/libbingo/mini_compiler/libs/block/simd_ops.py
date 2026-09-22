# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The per-element and per-row operators a layer is glued together with.

RMSNorm, RoPE, quantise and the residual add. Each is one fused kernel, so each is one
node, and the block around it exists to state the layouts -- which is the whole difficulty,
because these four do NOT all care about layout in the same way.

WHICH OPS CARE ABOUT LAYOUT, AND WHY IT IS NOT A DETAIL

  ELEMENTWISE ops -- quantise, residual add -- touch each value independently. Any layout
  works, because a permutation of the inputs is the same permutation of the output. They
  declare `packed` for readability but will consume whatever their operands agree on; MoE
  runs its SwiGLU straight over a D-layout block for exactly this reason.

  PER-ROW ops -- RMSNorm, RoPE -- reduce or rotate ALONG a row, so they need the row to be
  contiguous. In D-layout (m, n, r, c) a matrix row is NOT contiguous: consecutive columns
  of one row are `meshCol` apart across n-blocks. Handing D-layout to RMSNorm normalises
  groups that are not rows. It does not fault, it does not go out of range, and the answer
  is a well-formed tensor of wrong numbers. So their ports say `packed` and mean it.

THE ORDER A LAYER HAS TO USE, and it is forced by hardware rather than taste:

    GEMM (D/f16) -> reshape to packed (f16) -> RMSNorm (packed/f16) -> reshape to A (f16)
                 -> quantise (A/i8) -> GEMM

The two reshapes are FP16 because a conversion into or out of A-layout needs an 8-byte run
contiguous on both sides, and at int8 an A-layout tileSize run is 4 bytes and falls off the
hardware path. Quantising before the reshape would make the reshape impossible. See
comm/nest.py, which refuses it by name rather than emitting something that does not run.
"""

from dataclasses import dataclass, replace

from bingo_kernel_args import (
    SnaxBingoKernelSimdFp16ToInt8Args,
    SnaxBingoKernelSimdRmsnormF16F16Args,
    SnaxBingoKernelSimdRopeArgs,
    SnaxBingoKernelSimdAddF16Args,
)

from ..comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                    PortSpec)

# One 512-bit SIMD beat is 64 B = 32 FP16 lanes, and every fused kernel here works a whole
# beat at a time. A row that is not a multiple of this has a partial final beat, which the
# kernels do not handle.
_LANES_PER_BEAT = 32


def _check_row(cols: int, who: str) -> None:
    if cols % _LANES_PER_BEAT:
        raise ValueError(
            f"{who}: cols={cols} must be a multiple of {_LANES_PER_BEAT} -- one SIMD beat "
            f"is 64 B = {_LANES_PER_BEAT} fp16 lanes, and a partial final beat is not "
            f"handled.")


@dataclass(frozen=True)
class RowCfg:
    """A [rows, cols] FP16 tensor, one row per independent token position."""

    rows: int
    cols: int
    cluster: int = 0


class RMSNorm(Block):
    """RMSNorm over each row. FP16 in, FP16 out, in place-compatible shapes.

    THE INPUT MUST BE ROW-MAJOR. The kernel reduces sum-of-squares along `cols`, so it
    needs a row to be contiguous -- see the module docstring. The port says `packed` and
    the contract enforces it.

    There is no learnable gain here. The fused kernel is normalise-only; a layer that
    wants the usual per-channel weight applies it as a separate elementwise multiply.
    """

    name = "rmsnorm"

    def __init__(self, cfg: RowCfg = None, **params):
        self.cfg = cfg if cfg is not None else RowCfg(**params)
        _check_row(self.cfg.cols, "RMSNorm")

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": PortSpec(Layout.PACKED, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1,
                              doc="row-major fp16; rows are normalised independently")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(Layout.PACKED, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1, doc="row-major fp16, normalised")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        out = g.l1(f"{self.name}_y", c.rows * c.cols * 2)
        nd = g.node(f"{self.name.capitalize()}", ctx.simd,
                    "__snax_bingo_kernel_simd_rmsnorm_f16_f16",
                    SnaxBingoKernelSimdRmsnormF16F16Args(
                        input_addr=bound["x"].handle, output_addr=out,
                        rows=c.rows, cols=c.cols))
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], out, (nd,), cluster=c.cluster, name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (nd,), name="x")},
            nodes=[nd])


class RoPE(Block):
    """Rotary position embedding over each row. FP16, row-major, tables supplied.

    `cos` and `sin` are PRECOMPUTED TABLES, not parameters: the kernel has no trig and the
    device cores have no FPU, so the rotation arrives as data. They are ports rather than
    constructor arguments because they are tensors a caller stages, and because a decode
    step indexes a different slice of them than a prefill does.
    """

    name = "rope"

    def __init__(self, cfg: RowCfg = None, **params):
        self.cfg = cfg if cfg is not None else RowCfg(**params)
        _check_row(self.cfg.cols, "RoPE")

    @property
    def inputs(self) -> dict:
        c = self.cfg
        row = (c.rows, c.cols)
        return {
            "x": PortSpec(Layout.PACKED, DType.F16, row, mem_level=MemLevel.L1,
                          doc="row-major fp16, one row per token position"),
            "cos": PortSpec(Layout.PACKED, DType.F16, row, mem_level=MemLevel.L1,
                            doc="precomputed cos table, already sliced to these positions"),
            "sin": PortSpec(Layout.PACKED, DType.F16, row, mem_level=MemLevel.L1,
                            doc="precomputed sin table, sign already applied"),
        }

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(Layout.PACKED, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1, doc="row-major fp16, rotated")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        out = g.l1(f"{self.name}_y", c.rows * c.cols * 2)
        nd = g.node("Rope", ctx.simd, "__snax_bingo_kernel_simd_rope",
                    SnaxBingoKernelSimdRopeArgs(
                        x_addr=bound["x"].handle, cos_addr=bound["cos"].handle,
                        sin_addr=bound["sin"].handle, out_addr=out,
                        cols=c.cols, rows=c.rows))
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], out, (nd,), cluster=c.cluster, name="y")},
            inputs={n: Port(self.inputs[n], bound[n].handle, (nd,), name=n)
                    for n in ("x", "cos", "sin")},
            nodes=[nd])


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
                 layout: Layout = Layout.PACKED, **params):
        if inv_scale_f32bits is None:
            raise ValueError(
                "Quantize needs inv_scale_f32bits: the right scale depends on the range "
                "of the data, and a wrong one saturates the tensor rather than failing.")
        self.cfg = cfg if cfg is not None else RowCfg(**params)
        self.inv_scale_f32bits = int(inv_scale_f32bits)
        self.layout = Layout(layout)
        _check_row(self.cfg.cols, "Quantize")

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": PortSpec(self.layout, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1, doc="fp16 in this block's layout")}

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

    def __init__(self, cfg: RowCfg = None, *, layout: Layout = Layout.PACKED,
                 dtype: DType = DType.F16, **params):
        self.cfg = cfg if cfg is not None else RowCfg(**params)
        self.layout, self.dtype = Layout(layout), DType(dtype)

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
                              mem_level=MemLevel.L1, doc="a + b")}

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
