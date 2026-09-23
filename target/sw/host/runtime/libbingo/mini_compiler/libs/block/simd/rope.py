# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Rotary position embedding: the other per-row operator, and the other layout contract.

Like RMSNorm, RoPE works ALONG a row -- it pairs feature j with feature j + D/2 and rotates
them together -- so it needs the row contiguous and says `packed` and means it. Unlike
RMSNorm it does not REDUCE along the row, and that is why there is no transposed variant
here: transposing buys a reduction that falls out of the per-lane accumulators, and a
rotation has no reduction to buy. The pairing would simply become a stride the kernel does
not take.
"""

from bingo_kernel_args import SnaxBingoKernelSimdRopeArgs

from ...comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                     PortSpec)
from .common import RowCfg, check_row


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
        check_row(self.cfg.cols, "RoPE")

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
