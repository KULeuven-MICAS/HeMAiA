# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""One token's row: the SIMD operators around a one-token GEMV.

A decode pass carries ONE token through the layer, so between two projections there is one
row of FP16, and the operators on it are shaped by that:

    RMSNormRow     the norm with no per-row plane: the row's 1/rms is a whole-TASK scalar,
                   so the SIMD latches it (STICKY_B) instead of writing it out
    QuantizeARow   FP16 -> INT8 into ROW 0 of the GEMV's 16-row A operand and nothing
                   else -- a quarter of the input beats of a fully written operand
    ARowLoad       an INT8 row into ROW 0 of that operand, one 2-D iDMA copy (after
                   RMSNormRow(quant_inv=...), which quantises in its own pass)
    ARowPack       an INT8 row into a_row, the GEMV's operand cut to the word it reads: one
                   2-D copy by the producer, then a plain 1-D load by every consumer
    ScaleCols      the per-column dequantisation y (.) s, s[n] = s_x * s_w[n] * 2^k: the
                   GEMV writes RNE(acc * 2^-k) and one scale per output column is not
                   something the D port can apply, so it is one SIMD pass over the row

    x --RMSNormRow--> xn --QuantizeARow--> A --Linear(gemv)--> y --ScaleCols--> y (.) s

The kernels are offload_hw_kernels/simd_row.h and the elementwise MUL; all three are ports
of what the snax reference (sw/apps/dsv2/include/snax-dsv2.h) runs bit-exactly against the
DeepSeek-V2-Lite golden model, so their outputs are checked with no tolerance.
"""

from dataclasses import dataclass
from typing import Optional

from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs,
                               SnaxBingoKernelSimdAddF16Args,
                               SnaxBingoKernelSimdMulF16Args,
                               SnaxBingoKernelSimdQuantARowArgs,
                               SnaxBingoKernelSimdRmsnormRowArgs,
                               SnaxBingoKernelSimdScaleF16Args,
                               SnaxBingoKernelSimdSoftmaxRowArgs,
                               SnaxBingoKernelSimdSwigluARowArgs,
                               SnaxBingoKernelXdmaMemsetArgs)

from ...comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                     PortSpec)
from ...comm.ports import at_offset
from ...comm.variant import free_fields
from .common import BEAT_BYTES, check_pow2, check_row

_LOADABLE = (MemLevel.L1, MemLevel.L3)
# Where RMSNormRow's x may come from when the level is PINNED: its iDMA copy reads the HBM
# as well (the memory chiplet's, over the D2D link). Not offered as a variant: a resolver
# choosing HBM for a row that already sits nearer would add a link crossing.
_NORM_PINNABLE = _LOADABLE + (MemLevel.HBM,)


def _row(cols, level, cluster, doc, rows=1):
    return _rows(rows, cols, level, cluster, doc)


def _rows(rows, cols, level, cluster, doc):
    return PortSpec(Layout.ROW_MAJOR, DType.F16, (rows, cols), mem_level=level,
                    cluster=cluster if level == MemLevel.L1 else None, doc=doc)


@dataclass(frozen=True)
class NormRowCfg:
    """One row of `cols` values on one cluster; `in_level` is where x is handed over (L1
    used in place where the block allows it, L3 loaded), None to resolve it."""

    cols: int
    cluster: int = 0
    in_level: Optional[MemLevel] = None
    # The activation quantiser's scale, as an FP32 bit pattern: set, the norm writes y as
    # INT8 -- a plain row of `cols` bytes -- in the same pass (the multiply's output goes on
    # through Fp16ToInt8); None, as FP16.
    quant_inv: Optional[int] = None


class RMSNormRow(Block):
    """y = x / sqrt(mean(x^2)) over one FP16 row, no gain (fold it into the weights after).

    Three SIMD tasks in one kernel: reduce SUMSQ -> RSQRT(ssq / cols) into a SEED beat ->
    MUL|STICKY_B over [seed | x]. The seed has to sit directly below x, so the block owns
    x's buffer: x is copied in on the iDMA from wherever it lies -- main memory, or another
    buffer in this L1 (4 KiB at d_model = 2,048, off the SIMD's critical path).

    cols is a power of two (the 1/cols is an exponent subtract) and the sum of squares
    must stay under FP16's 65,504: rms <= sqrt(65504 / cols), 5.66 at 2,048.

    quant_inv set: y is sat127(rne(norm(x) * inv)) as INT8, the quantiser fused into the
    multiply's pass -- one task instead of two, and half the bytes to move on. A consumer
    that needs the GEMV's A layout loads it with ARowLoad.
    """

    name = "rmsnorm_row"

    def __init__(self, cfg: NormRowCfg = None, **params):
        self.cfg = cfg if cfg is not None else NormRowCfg(**params)
        check_row(self.cfg.cols, "RMSNormRow")
        check_pow2(self.cfg.cols, "RMSNormRow")
        self.free = free_fields(self.cfg, ("in_level",))
        if self.cfg.in_level is not None and self.cfg.in_level not in _NORM_PINNABLE:
            raise ValueError(f"RMSNormRow: in_level={self.cfg.in_level}; L1, L3 or HBM.")

    def variants(self) -> list:
        return [{}] if not self.free else [{"in_level": lv} for lv in _LOADABLE]

    def idma_passes(self) -> int:
        return 1                       # x into the seed-headroom buffer, always

    def simd_passes(self) -> int:
        return 3

    @property
    def inputs(self) -> dict:
        c = self.cfg
        if self.free:
            raise ValueError("RMSNormRow: in_level not decided; pin it or use a Pipeline.")
        return {"x": _row(c.cols, c.in_level, c.cluster, "fp16 row")}

    @property
    def needs(self) -> dict:
        return {"x": _row(self.cfg.cols, MemLevel.L1, self.cfg.cluster, "fp16 row")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        if c.quant_inv is not None:
            return {"y": PortSpec(Layout.ROW_MAJOR, DType.I8, (1, c.cols), mem_level=MemLevel.L1,
                                  cluster=c.cluster, doc="normalised row, quantised to int8")}
        return {"y": _row(c.cols, MemLevel.L1, c.cluster, "normalised fp16 row")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        nbytes = 2 * c.cols
        buf = g.l1(f"{self.name}_x", BEAT_BYTES + nbytes)       # [seed | x]
        ssq = g.l1(f"{self.name}_ssq", BEAT_BYTES)
        y = g.l1(f"{self.name}_y", c.cols if c.quant_inv is not None else nbytes)
        ld = g.node("Ld_x", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(bound["x"].handle, buf.view(BEAT_BYTES),
                                                  nbytes))
        nd = g.node("RmsnormRow", ctx.simd, "__snax_bingo_kernel_simd_rmsnorm_row",
                    SnaxBingoKernelSimdRmsnormRowArgs(buf, buf.view(BEAT_BYTES), ssq, y,
                                                      c.cols, c.quant_inv or 0),
                    [ld])
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], y, (nd,), name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (ld,), name="x")},
            nodes=[ld, nd])


@dataclass(frozen=True)
class QuantARowCfg:
    cols: int
    inv_scale_f32bits: int
    cluster: int = 0
    # Which row of the 16-row operand: 0 for the token (1 token per pass), 4 for a second.
    row: int = 0
    # SEGMENTS: `segs` runs of `cols` values, seg_pitch BYTES apart in the input row, into
    # segs consecutive A operands -- every head's operand of a per-head GEMV in one task.
    # The input row is then segs * seg_pitch / 2 values long (the heads' whole slices).
    segs: int = 1
    seg_pitch: int = 0
    # a_row: the output in a_row (2 bytes a value, the segments back to back) instead of the
    # 16-row A layout -- a pass's tokens, or a per-head GEMV's heads, at 1/8 the L1
    a_row: bool = False

    @property
    def in_cols(self) -> int:
        return self.cols if self.segs == 1 else self.segs * self.seg_pitch // 2


class QuantizeARow(Block):
    """FP16 row -> INT8 into ROW `row` of a one-token GEMV's A operand.

    The output is the (16, 4, 16) A layout of a [16, cols] tensor -- 16 * cols bytes -- of
    which only the named row holds the token. Its port says (1, cols) in layout A: the
    operand of a Linear(gemv=True). The GEMV reads rows 0 and 1 of every block through one
    8-byte channel and the array takes row 0, so the rest has to be DEFINED, not
    meaningful: TCDM nobody wrote reads X on RTL. The block therefore zeroes the operand
    once, on the otherwise idle xDMA, before the quantiser writes its row.

    THE SCALE IS AN ARGUMENT AND HAS NO DEFAULT: one static inv_scale per tensor, calibrated
    offline, because the quantiser takes one per task and the SIMD has no abs-max.
    """

    name = "quant_a_row"

    def __init__(self, cfg: QuantARowCfg = None, **params):
        self.cfg = cfg if cfg is not None else QuantARowCfg(**params)
        check_row(self.cfg.cols, "QuantizeARow")
        # Validates the row and the scale the same way the kernel args will.
        SnaxBingoKernelSimdQuantARowArgs(0, 0, self.cfg.cols, self.cfg.row,
                                         self.cfg.inv_scale_f32bits, self.cfg.segs,
                                         self.cfg.seg_pitch, a_blk=8 if self.cfg.a_row else 64)

    def simd_passes(self) -> int:
        return 1

    def xdma_passes(self) -> int:
        return 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": _row(c.in_cols, MemLevel.L1, c.cluster, "fp16 row, in this L1")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        if c.a_row:
            return {"y": PortSpec(Layout.A_ROW, DType.I8, (1, c.segs * c.cols),
                                  mem_level=MemLevel.L1, cluster=c.cluster,
                                  doc="int8 a_row, 2 bytes a value")}
        return {"y": PortSpec(Layout.A, DType.I8, (1, c.segs * c.cols), mem_level=MemLevel.L1,
                              cluster=c.cluster,
                              doc="int8, row 0 of a 16-row A operand (16 * cols bytes)")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        nbytes = 2 * c.segs * c.cols if c.a_row else \
            c.segs * SnaxBingoKernelSimdQuantARowArgs.a_bytes(c.cols)
        a = g.l1(f"{self.name}_a", nbytes)
        zero = g.node("ZeroA", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                      SnaxBingoKernelXdmaMemsetArgs(a, nbytes,
                                                    SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO))
        nd = g.node("QuantARow", ctx.simd, "__snax_bingo_kernel_simd_quant_a_row",
                    SnaxBingoKernelSimdQuantARowArgs(bound["x"].handle, a, c.cols, c.row,
                                                     c.inv_scale_f32bits, c.segs, c.seg_pitch,
                                                     a_blk=8 if c.a_row else 64),
                    [zero])
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], a, (nd,), name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (nd,), name="x")},
            nodes=[zero, nd],
            # The memset has no predecessor: gated behind x's producer it cannot run early,
            # ungated it cannot reuse that producer's L1 -- the linker's call, as for loads.
            sources=[zero])


@dataclass(frozen=True)
class ARowLoadCfg:
    """An INT8 row of `cols` values into row 0 of a one-token GEMV's A operand on
    `cluster`. The row lies in L3 or the HBM (in_level), or in `src_cluster`'s L1."""

    cols: int
    cluster: int = 0
    in_level: MemLevel = MemLevel.L3
    src_cluster: Optional[int] = None


class ARowLoad(Block):
    """INT8 row -> ROW 0 of a one-token GEMV's A operand, in one iDMA copy.

    The operand is QuantizeARow's: the (16, 4, 16) A layout of a [16, cols] tensor, value i
    at (i // 4) * 64 + i % 4. So the load is a 2-D copy -- cols / 4 runs of 4 bytes, 4 bytes
    apart in the source and 64 in the operand -- after the operand is zeroed once on the
    otherwise idle xDMA (the GEMV reads rows 0 and 1 of every block, and TCDM nobody wrote
    reads X on RTL). With RMSNormRow(quant_inv=...) upstream this replaces a per-cluster
    FP16 fetch and QuantizeARow: the row moves at half the bytes and is quantised once.
    """

    name = "a_row_load"

    def __init__(self, cfg: ARowLoadCfg = None, **params):
        self.cfg = cfg if cfg is not None else ARowLoadCfg(**params)
        check_row(self.cfg.cols, "ARowLoad")
        if self.cfg.cols % 4:
            raise ValueError(f"ARowLoad: cols={self.cfg.cols}; whole A blocks of 4 values.")
        if self.cfg.in_level not in (MemLevel.L1, MemLevel.L3, MemLevel.HBM):
            raise ValueError(f"ARowLoad: in_level={self.cfg.in_level}; L1, L3 or HBM.")

    def idma_passes(self) -> int:
        return 1

    def xdma_passes(self) -> int:
        return 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        src = c.cluster if c.src_cluster is None else c.src_cluster
        return {"x": PortSpec(Layout.ROW_MAJOR, DType.I8, (1, c.cols), mem_level=c.in_level,
                              cluster=src if c.in_level == MemLevel.L1 else None,
                              doc="int8 row")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(Layout.A, DType.I8, (1, c.cols), mem_level=MemLevel.L1,
                              cluster=c.cluster,
                              doc="int8, row 0 of a 16-row A operand (16 * cols bytes)")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        nbytes = SnaxBingoKernelSimdQuantARowArgs.a_bytes(c.cols)
        a = g.l1(f"{self.name}_a", nbytes)
        zero = g.node("ZeroA", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                      SnaxBingoKernelXdmaMemsetArgs(a, nbytes,
                                                    SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO))
        ld = g.node("LdA", ctx.dm, "__snax_bingo_kernel_idma_2d_copy",
                    SnaxBingoKernelIdma2dCopyArgs(bound["x"].handle, a, 4, 4, 64, c.cols // 4),
                    [zero])
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], a, (ld,), name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (ld,), name="x")},
            nodes=[zero, ld], sources=[zero])


@dataclass(frozen=True)
class ARowPackCfg:
    """An INT8 row of `cols` values in `cluster`'s L1, packed into a_row at `dst`."""

    cols: int
    cluster: int = 0
    dst: object = None        # an L3 (or HBM) handle of 2 * cols bytes, staged as zeros
    dst_level: MemLevel = MemLevel.L3


class ARowPack(Block):
    """INT8 row -> a_row, the one-token GEMV's operand (comm/ports.py), in one iDMA copy.

    a_row keeps, of each 4-value A block, only the 8-byte word the GEMV reads: value c at
    (c // 4) * 8 + c % 4. So the pack is a 2-D copy -- cols / 4 runs of 4 bytes, 4 bytes
    apart in the row and 8 in a_row -- and the upper half of every word is whatever `dst`
    held, which must be zero: stage dst as zeros. Every consumer then loads the operand with
    one plain copy of 2 * cols bytes, straight into what its GEMV reads.
    """

    name = "a_row_pack"

    def __init__(self, cfg: ARowPackCfg = None, **params):
        self.cfg = cfg if cfg is not None else ARowPackCfg(**params)
        check_row(self.cfg.cols, "ARowPack")
        if self.cfg.cols % 4:
            raise ValueError(f"ARowPack: cols={self.cfg.cols}; whole blocks of 4 values.")
        if self.cfg.dst is None:
            raise ValueError("ARowPack: dst is required, 2 * cols bytes staged as zeros -- "
                             "the pack writes only the lower half of every word.")

    def idma_passes(self) -> int:
        return 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": PortSpec(Layout.ROW_MAJOR, DType.I8, (1, c.cols), mem_level=MemLevel.L1,
                              cluster=c.cluster, doc="int8 row")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(Layout.A_ROW, DType.I8, (1, c.cols), mem_level=c.dst_level,
                              doc="int8 row as a_row, 2 * cols bytes")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        nd = g.node("Pack", ctx.dm, "__snax_bingo_kernel_idma_2d_copy",
                    SnaxBingoKernelIdma2dCopyArgs(bound["x"].handle, c.dst, 4, 4, 8,
                                                  c.cols // 4),
                    list(bound["x"].ends))
        return BlockResult(outputs={"y": Port(self.outputs["y"], c.dst, (nd,), name="y")},
                           inputs={"x": Port(self.inputs["x"], bound["x"].handle, (nd,),
                                             name="x")},
                           nodes=[nd])


@dataclass(frozen=True)
class ScaleColsCfg:
    cols: int
    cluster: int = 0
    # Rows of x, one per token of a pass: each is its own SIMD task against the one factor
    # row, so every token's dequantisation is the single-row one, bit for bit.
    rows: int = 1
    # Where the factor row is handed over: L1 used in place, L3 loaded. None: resolved.
    s_level: Optional[MemLevel] = None


class ScaleCols(Block):
    """y = x (.) s over one FP16 row: one factor per column, one SIMD MUL pass.

    The GEMV's dequantisation: x is RNE(acc * 2^-k) off the D port and s[n] = s_x * s_w[n]
    * 2^k, rounded to FP16, so y = RNE(fp32(x) * fp32(s)) -- one rounding, bit-exact
    against the golden. A multiply of two FP16 is exact in FP32, so which elementwise
    instance runs it does not matter.
    """

    name = "scale_cols"

    def __init__(self, cfg: ScaleColsCfg = None, **params):
        self.cfg = cfg if cfg is not None else ScaleColsCfg(**params)
        check_row(self.cfg.cols, "ScaleCols")
        self.free = free_fields(self.cfg, ("s_level",))
        if self.cfg.s_level is not None and self.cfg.s_level not in _LOADABLE:
            raise ValueError(f"ScaleCols: s_level={self.cfg.s_level}; L1 or L3.")

    def variants(self) -> list:
        return [{}] if not self.free else [{"s_level": lv} for lv in _LOADABLE]

    def simd_passes(self) -> int:
        return 1

    def idma_passes(self) -> int:
        return 0 if self.cfg.s_level == MemLevel.L1 else 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        if self.free:
            raise ValueError("ScaleCols: s_level not decided; pin it or use a Pipeline.")
        return {"x": _rows(c.rows, c.cols, MemLevel.L1, c.cluster, "fp16 rows off the GEMV"),
                "s": _row(c.cols, c.s_level, c.cluster, "fp16 factor per column")}

    @property
    def needs(self) -> dict:
        c = self.cfg
        return {"x": _rows(c.rows, c.cols, MemLevel.L1, c.cluster, "fp16 rows off the GEMV"),
                "s": _row(c.cols, MemLevel.L1, c.cluster, "fp16 factor per column")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": _rows(c.rows, c.cols, MemLevel.L1, c.cluster, "x (.) s, fp16 rows")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        nbytes = 2 * c.cols
        sp = bound["s"]
        ld = None
        if sp.spec.mem_level == MemLevel.L1:
            s_l1, s_after = sp.handle, list(sp.ends)
        else:
            s_l1 = g.l1(f"{self.name}_s", nbytes)
            ld = g.node("Ld_s", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(sp.handle, s_l1, nbytes))
            s_after = [ld]
        y = g.l1(f"{self.name}_y", c.rows * nbytes)
        tasks = []
        for r in range(c.rows):
            # chained: the last row's task implies every row's
            tasks.append(g.node("ScaleCols" if c.rows == 1 else f"ScaleCols_r{r}", ctx.simd,
                                "__snax_bingo_kernel_simd_stream_elementwise",
                                SnaxBingoKernelSimdMulF16Args(
                                    at_offset(bound["x"].handle, r * nbytes), s_l1,
                                    at_offset(y, r * nbytes), rows=1, cols=c.cols),
                                s_after if r == 0 else [tasks[-1]]))
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], y, (tasks[-1],), name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (tasks[0],), name="x"),
                    "s": Port(self.inputs["s"], sp.handle, (ld,) if ld else (tasks[0],),
                              name="s")},
            nodes=([ld] if ld else []) + tasks,
            sources=[ld] if ld else [])


# ======================================================================================
# The rest of one token's row: the router's softmax, SwiGLU into the down GEMV's operand,
# a scale decided at run time, and the row add.
# ======================================================================================

@dataclass(frozen=True)
class SoftmaxRowCfg:
    cols: int
    cluster: int = 0
    rows: int = 1              # rows > 1: one task per row, the outputs back to back


class SoftmaxRow(Block):
    """softmax over ONE fp16 row (the router's): five SIMD tasks in one kernel, the max and
    the sum folded across lanes, 1/s as rsqrt(s * s) (offload_hw_kernels/simd_row.h).

    The kernel reads x behind a free latch beat, so the block owns x's buffer and copies
    the row in on the iDMA (cols * 2 bytes; 128 at 64 experts). Bit-exact hwmodel.softmax16.
    """

    name = "softmax_row"

    def __init__(self, cfg: SoftmaxRowCfg = None, **params):
        self.cfg = cfg if cfg is not None else SoftmaxRowCfg(**params)
        check_row(self.cfg.cols, "SoftmaxRow")
        if self.cfg.cols > 64 * 32:
            raise ValueError("SoftmaxRow: 1/s = rsqrt(s*s) is exact while s*s fits FP16.")

    def simd_passes(self) -> int:
        return 5

    def idma_passes(self) -> int:
        return 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": _row(c.cols, MemLevel.L1, c.cluster, "fp16 row", rows=c.rows)}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": _row(c.cols, MemLevel.L1, c.cluster, "softmax(x)", rows=c.rows)}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        nbytes, beats = 2 * c.cols, c.cols // 32
        tmp = g.l1(f"{self.name}_tmp", BEAT_BYTES)
        y = g.l1(f"{self.name}_y", nbytes * c.rows)
        nodes, cps = [], []
        for r in range(c.rows):
            sfx = "" if c.rows == 1 else f"_r{r}"
            xb = g.l1(f"{self.name}_x{sfx}", BEAT_BYTES + nbytes)              # [latch | x]
            eb = g.l1(f"{self.name}_e{sfx}", BEAT_BYTES + nbytes + BEAT_BYTES)  # [latch|e|s]
            cp = g.node(f"Ld_x{sfx}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(at_offset(bound["x"].handle, nbytes * r),
                                                      xb.view(BEAT_BYTES), nbytes))
            nd = g.node(f"SoftmaxRow{sfx}", ctx.simd, "__snax_bingo_kernel_simd_softmax_row",
                        SnaxBingoKernelSimdSoftmaxRowArgs(xb.view(BEAT_BYTES), tmp,
                                                          eb.view(BEAT_BYTES),
                                                          at_offset(y, nbytes * r), beats),
                        [cp] + nodes[-1:])          # chained: they share tmp
            cps.append(cp)
            nodes.append(nd)
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], y, (nodes[-1],), name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, tuple(cps), name="x")},
            nodes=cps + nodes)


@dataclass(frozen=True)
class SwigluARowCfg:
    inter: int
    cluster: int = 0
    # The output's quantiser: a static FP32 scale, or -- for a routed expert -- slot `slot`
    # of the expert-slot record bound to `rec` (moe_route.h: word 24 of the slot).
    inv_scale_f32bits: int = 0
    slot: Optional[int] = None
    rec_slots: int = 6
    # rows > 1: a pass's tokens, g one [gate | up] row each; the output is then a_row (the
    # layout a multi-token GEMV reads, 2 inter bytes a token), and so it is with a_row=True
    rows: int = 1
    a_row: bool = False


class SwigluARow(Block):
    """a8 = sat127(rne(silu(gate) * up * inv)) into ROW 0 of the down GEMV's A operand.

      in   g    [1, 2 * inter] fp16, L1: [gate | up], the dequantised gate|up GEMV
           rec  (with slot) the expert-slot record, whose slot holds inv
      out  y    [1, inter] int8, A layout: row 0 of a 16-row operand (16 * inter bytes)

    Two SIMD tasks (simd_row.h): SILU on gate into scratch, then the multiply straight into
    the quantiser's row write. Bit-exact hwmodel.moe_experts' mlp().
    """

    name = "swiglu_a_row"
    REC_INV_OFFSET = 24 * 4     # moe_route.h: word 24 of a 128-B slot

    def __init__(self, cfg: SwigluARowCfg = None, **params):
        self.cfg = cfg if cfg is not None else SwigluARowCfg(**params)
        c = self.cfg
        check_row(c.inter, "SwigluARow")
        if (c.slot is None) == (not c.inv_scale_f32bits):
            raise ValueError("SwigluARow: give exactly one of inv_scale_f32bits (static) or "
                             "slot (the routed expert's, from the record).")

    def simd_passes(self) -> int:
        return 2

    def xdma_passes(self) -> int:
        return 1

    @property
    def _a_row(self) -> bool:
        return self.cfg.a_row or self.cfg.rows > 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        ins = {"g": _row(2 * c.inter, MemLevel.L1, c.cluster, "[gate | up] fp16", rows=c.rows)}
        if c.slot is not None:
            from ..linear import record_spec
            ins["rec"] = record_spec(c.rec_slots, c.cluster)
        return ins

    @property
    def outputs(self) -> dict:
        c = self.cfg
        if self._a_row:
            return {"y": PortSpec(Layout.A_ROW, DType.I8, (c.rows, c.inter),
                                  mem_level=MemLevel.L1, cluster=c.cluster,
                                  doc="int8 a_row, one row per token")}
        return {"y": PortSpec(Layout.A, DType.I8, (1, c.inter), mem_level=MemLevel.L1,
                              cluster=c.cluster, doc="int8, row 0 of the down's A operand")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        nbytes = 2 * c.inter * c.rows if self._a_row else \
            SnaxBingoKernelSimdQuantARowArgs.a_bytes(c.inter)
        a = g.l1(f"{self.name}_a", nbytes)
        sg = g.l1(f"{self.name}_sg", 2 * c.inter)
        zero = g.node("ZeroA", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                      SnaxBingoKernelXdmaMemsetArgs(a, nbytes,
                                                    SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO))
        deps = [zero]
        inv_addr = 0
        if c.slot is not None:
            rec = bound["rec"]
            inv_addr = at_offset(rec.handle, c.slot * 128 + self.REC_INV_OFFSET)
            deps += list(rec.ends)
        nd = g.node("SwigluARow", ctx.simd, "__snax_bingo_kernel_simd_swiglu_a_row",
                    SnaxBingoKernelSimdSwigluARowArgs(bound["g"].handle, sg, a, c.inter,
                                                      c.inv_scale_f32bits, inv_addr,
                                                      rows=c.rows, g_pitch=4 * c.inter,
                                                      a_pitch=2 * c.inter,
                                                      a_blk=8 if self._a_row else 64),
                    deps)
        ins = {"g": Port(self.inputs["g"], bound["g"].handle, (nd,), name="g")}
        if c.slot is not None:
            ins["rec"] = Port(self.inputs["rec"], bound["rec"].handle, (nd,), name="rec")
        return BlockResult(outputs={"y": Port(self.outputs["y"], a, (nd,), name="y")},
                           inputs=ins, nodes=[zero, nd], sources=[zero])


@dataclass(frozen=True)
class ScaleRowCfg:
    cols: int
    cluster: int = 0
    slot: int = 0               # the record slot whose weight scales the row
    rec_slots: int = 6
    # rows > 1: a pass's tokens, row t scaled by slot `slot` of token t's record, the
    # records rec_slots slots apart (moe_route.h, tokens > 1); x and y [rows, cols]
    rows: int = 1


class ScaleRowBySlot(Block):
    """y = w * x over one fp16 row, w a routed expert's weight chosen at run time: the FP32
    word 1 of record slot `slot` (moe_route.h), read by the kernel as its Map scale. One
    SIMD pass; RNE(w * x), the golden's mul16 (a product of two FP16 is exact in FP32).
    """

    name = "scale_row_slot"
    REC_W_OFFSET = 1 * 4        # word 1: the weight as FP32 bits

    def __init__(self, cfg: ScaleRowCfg = None, **params):
        self.cfg = cfg if cfg is not None else ScaleRowCfg(**params)
        check_row(self.cfg.cols, "ScaleRowBySlot")

    def simd_passes(self) -> int:
        return 1

    @property
    def inputs(self) -> dict:
        from ..linear import record_spec
        c = self.cfg
        return {"x": _row(c.cols, MemLevel.L1, c.cluster, "fp16 row", rows=c.rows),
                "rec": record_spec(c.rows * c.rec_slots, c.cluster)}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": _row(c.cols, MemLevel.L1, c.cluster, "w * x", rows=c.rows)}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        y = g.l1(f"{self.name}_y", 2 * c.cols * c.rows)
        rec = bound["rec"]
        nodes = []
        for r in range(c.rows):
            nodes.append(g.node(
                "ScaleBySlot" if c.rows == 1 else f"ScaleBySlot_r{r}", ctx.simd,
                "__snax_bingo_kernel_simd_stream_map",
                SnaxBingoKernelSimdScaleF16Args(
                    at_offset(bound["x"].handle, 2 * c.cols * r), at_offset(y, 2 * c.cols * r),
                    0, rows=1, cols=c.cols,
                    scale_addr=at_offset(rec.handle, (r * c.rec_slots + c.slot) * 128 +
                                         self.REC_W_OFFSET)),
                list(rec.ends) + nodes[-1:]))
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], y, (nodes[-1],), name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, tuple(nodes), name="x"),
                    "rec": Port(self.inputs["rec"], rec.handle, (nodes[0],), name="rec")},
            nodes=nodes)


@dataclass(frozen=True)
class AddRowCfg:
    cols: int
    cluster: int = 0


class AddRow(Block):
    """y = a + b over one fp16 row, both in this L1: one SIMD pass, RNE(a + b).
    The residual of a decode pass and the combine's running sum."""

    name = "add_row"

    def __init__(self, cfg: AddRowCfg = None, **params):
        self.cfg = cfg if cfg is not None else AddRowCfg(**params)
        check_row(self.cfg.cols, "AddRow")

    def simd_passes(self) -> int:
        return 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"a": _row(c.cols, MemLevel.L1, c.cluster, "fp16 row"),
                "b": _row(c.cols, MemLevel.L1, c.cluster, "fp16 row")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": _row(c.cols, MemLevel.L1, c.cluster, "a + b")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        y = g.l1(f"{self.name}_y", 2 * c.cols)
        nd = g.node("AddRow", ctx.simd, "__snax_bingo_kernel_simd_stream_elementwise",
                    SnaxBingoKernelSimdAddF16Args(bound["a"].handle, bound["b"].handle, y,
                                                  rows=1, cols=c.cols))
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], y, (nd,), name="y")},
            inputs={n: Port(self.inputs[n], bound[n].handle, (nd,), name=n)
                    for n in ("a", "b")},
            nodes=[nd])
