# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Rotary position embedding: the other per-row operator, and the one that needs two cores.

Like RMSNorm, RoPE works ALONG a row -- it pairs feature j with feature j+1 and rotates
them together -- so it needs the row contiguous and says `packed` and means it. Unlike
RMSNorm it does not REDUCE along the row, and that is why there is no transposed variant:
transposing buys a reduction that falls out of the per-lane accumulators, and a rotation
has no reduction to buy.

======================================================================================
WHY THIS BLOCK EMITS TWO NODES
======================================================================================

    out = x (.) cos_full  +  xswap (.) sin_signed

Two products and their sum is a combine, then a combine, and the SIMD block has an
elementwise slot on EACH side of the Map -- so the arithmetic is ONE task over a four-deep
operand axis. What cannot join it is `xswap`, the adjacent fp16-pair permutation of x.

THE READER CANNOT DO THE SWAP, and the reason is granularity rather than awkwardness.
AddressGenUnit emits one address per spatial channel, and each channel fetches exactly one
64-bit TCDM word -- 8 bytes, delivered in memory order. The swap is a 2-byte reorder INSIDE
one of those words. `lane_stride` moves whole channels, not halfwords, and nothing between
the channel and the operator lanes reorders bytes. The same argument covers the xDMA, built
from the same ReaderWriter; its one byte-permuting unit is the 8x8 block Transposer, and a
block transpose is not an adjacent-pair swap.

TRANSPOSING DOES NOT HELP EITHER. For RMSNorm the win came from turning a cross-lane
REDUCTION into an along-beat one. RoPE has no reduction; its coupling is between the halves
of a pair, and transposing moves that to adjacent BEATS -- which the operand axis can
address -- but the two outputs of a pair then need cos and sin exchanged between them, and
that exchange rides the same axis as the operand selection. One materialisation traded for
another.

What CAN do it is the iDMA, a real byte-addressed DMA, as two strided 2-byte copies. So:

    DM core    Rope_swap   x -> slot 2          __snax_bingo_kernel_idma_pairwise_swap
    SIMD core  Rope        the whole rotation   __snax_bingo_kernel_simd_rope

That is not a worse decomposition than one node. The swap is the dominant cost of RoPE --
at [32, 128] it is rows*cols/2 element moves against a few hundred cycles for all the
arithmetic -- so putting it on the engine built for it, and off the busiest one, is the
whole point. A fused StreamRoPE rotating adjacent lanes inside the beat would remove it;
that is the only part of RoPE still asking for RTL.

======================================================================================
THE OPERAND BLOCK
======================================================================================

The reader adds ONE stride per axis, so the four operands must be equally spaced -- and
their ORDER is the pairing, because EW0 multiplies operand 0 by 1 and 2 by 3 and EW1 adds
the two:

    ops -> [ x | cos_full | xswap | sin_signed ]      each rows*cols*2 bytes

This block ALLOCATES that buffer and exposes slots 0, 1 and 3 as its input ports, so the
producer of Q/K and the two table loads write straight into it and no operand is ever
copied. Slot 2 is the block's own: nothing outside it may write there.

That is why the ports bind to views of one allocation rather than to three buffers the
caller chose. A three-buffer contract would look friendlier and cost three copies --
more than the swap it was trying to avoid.
"""

from bingo_kernel_args import (SnaxBingoKernelIdmaPairwiseSwapArgs,
                               SnaxBingoKernelSimdRopeArgs)

from ...comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                     PortSpec)
from ...comm.ports import at_offset
from .common import RowCfg, check_row

# Slot order inside the operand block. Not a convention -- it is the pairing the two
# elementwise slots perform, so renaming or reordering changes what the kernel computes.
_SLOT = {"x": 0, "cos": 1, "xswap": 2, "sin": 3}


class RoPE(Block):
    """Rotary position embedding over each row. FP16, row-major, tables supplied.

    `cos` and `sin` are PRECOMPUTED TABLES, not parameters: the kernel has no trig and the
    device cores have no FPU, so the rotation arrives as data. They are ports rather than
    constructor arguments because they are tensors a caller stages, and because a decode
    step indexes a different slice of them than a prefill does.

    THE THREE INPUT PORTS ARE VIEWS OF ONE BUFFER THIS BLOCK OWNS. Bind them by writing
    into them -- `block.slots` gives the handles before build, so a producer can target
    them directly. See the module docstring for why the operands cannot simply live
    wherever the caller had them.
    """

    name = "rope"

    def __init__(self, cfg: RowCfg = None, **params):
        self.cfg = cfg if cfg is not None else RowCfg(**params)
        check_row(self.cfg.cols, "RoPE")
        self._ops = None

    def alloc(self, ctx: Ctx):
        """Allocate the 4-row operand block and return {name: handle} for its slots.

        Called before build() by a caller that wants to aim its producers at the slots.
        build() calls it too if nobody did, so a caller that stages by other means still
        works.
        """
        if self._ops is None:
            c = self.cfg
            row_b = c.rows * c.cols * 2
            self._ops = ctx.at(c.cluster).l1(f"{self.name}_ops", 4 * row_b)
        return self.slots

    @property
    def slots(self) -> dict:
        if self._ops is None:
            raise ValueError(
                "RoPE.slots before the block has allocated: call alloc(ctx) first, or let "
                "build() do it and read the bound input ports afterwards.")
        row_b = self.cfg.rows * self.cfg.cols * 2
        return {k: at_offset(self._ops, i * row_b) for k, i in _SLOT.items()}

    @property
    def inputs(self) -> dict:
        c = self.cfg
        row = (c.rows, c.cols)
        return {
            "x": PortSpec(Layout.PACKED, DType.F16, row, mem_level=MemLevel.L1,
                          doc="row-major fp16, one row per token position (slot 0)"),
            "cos": PortSpec(Layout.PACKED, DType.F16, row, mem_level=MemLevel.L1,
                            doc="precomputed cos table, duplicated per pair (slot 1)"),
            "sin": PortSpec(Layout.PACKED, DType.F16, row, mem_level=MemLevel.L1,
                            doc="precomputed sin table, sign already applied (slot 3)"),
        }

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(Layout.PACKED, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1, doc="row-major fp16, rotated")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        self.alloc(g)
        slots = self.slots
        row_b = c.rows * c.cols * 2

        # REFUSE A BINDING THAT IS NOT THIS BLOCK'S OWN SLOT. The kernel derives its
        # operand stride from the shape, so an operand living anywhere else is read at the
        # wrong address -- and reads a well-formed tensor of the wrong numbers rather than
        # faulting. Checked by identity of the underlying allocation plus its offset.
        for name in ("x", "cos", "sin"):
            want, got = slots[name], bound[name].handle
            if _base_and_offset(got) != _base_and_offset(want):
                raise ValueError(
                    f"RoPE.{name} must be bound to this block's own operand slot "
                    f"(slot {_SLOT[name]} of {self.name}_ops). The four operands have to be "
                    f"equally spaced because the reader adds one stride per axis, so they "
                    f"cannot stay where the producer happened to put them. Call "
                    f"alloc(ctx) and aim the producer at the slot it returns.")

        out = g.l1(f"{self.name}_y", row_b)
        # The swap, on the engine that can address bytes. Reads slot 0, writes slot 2 --
        # they are 2*row_b apart and row_b long, so the two strided copies cannot overlap.
        sw = g.node("Rope_swap", ctx.dm, "__snax_bingo_kernel_idma_pairwise_swap",
                    SnaxBingoKernelIdmaPairwiseSwapArgs(
                        slots["x"], slots["xswap"], c.rows * c.cols, 2))
        nd = g.node("Rope", ctx.simd, "__snax_bingo_kernel_simd_rope",
                    SnaxBingoKernelSimdRopeArgs(self._ops, out, c.cols, c.rows), sw)
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], out, (nd,), cluster=c.cluster, name="y")},
            # x is read by the SWAP first, the tables by the rotation. Naming the real
            # first reader per operand is what lets the linker order a producer correctly.
            inputs={"x": Port(self.inputs["x"], slots["x"], (sw,), name="x"),
                    "cos": Port(self.inputs["cos"], slots["cos"], (nd,), name="cos"),
                    "sin": Port(self.inputs["sin"], slots["sin"], (nd,), name="sin")},
            nodes=[sw, nd])


def _base_and_offset(handle):
    """(underlying allocation, byte offset) for a handle or a view of one."""
    base = getattr(handle, "base", None)
    return (handle, 0) if base is None else (base, handle.offset)
