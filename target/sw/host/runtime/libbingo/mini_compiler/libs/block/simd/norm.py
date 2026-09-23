# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""RMSNorm, and the layout pass that makes it cheap.

    out[t, f] = x[t, f] / sqrt( (1/D) * SUM_f x[t, f]^2 )

One multiply per element, the same arithmetic as a residual add -- which measures four
times cheaper over the same tile. All of the difference is the ONE SCALAR PER ROW: where
it is computed, and how it gets back to the data. This module is about that scalar.

======================================================================================
WHY ORIENTATION DECIDES THE COST
======================================================================================

StreamReduce carries one FP32 accumulator PER LANE, acc[0..31], persisting from beat to
beat; every beat, lane k folds into acc[k], and nothing ever moves sideways. So:

  A REDUCTION ALONG BEATS IS FREE. It is the accumulators doing what they already do, one
  beat per cycle, and the answer is sitting in them when the stream ends.

  A REDUCTION ACROSS THE LANES OF A BEAT IS A DIFFERENT MACHINE. A balanced binary fold
  through treeBuf, log2(32) = 5 rounds and 31 adds spread over `treeLanes` ALUs (2 today),
  holding the reader's input port low for ~35 cc -- once per row.

Which one RMSNorm gets is decided entirely by how x is stored:

  ROW-MAJOR x[T, D]   a beat is 32 consecutive FEATURES of one token, so lane k collects
                      every 32nd feature and a row's D terms scatter across all 32 lanes.
                      They have to be folded, once per row. Then the single scalar has to
                      be SPLATTED back across a beat and REPLICATED for every beat of the
                      row, because a 2-operand elementwise reads both operands from one
                      3-D affine stream {operand, beat, row} and the three loops share one
                      stride set -- `beat` cannot be zeroed for the scalar operand alone.
                      Measured: 1,387 cc of reduce for 128 beats, of which ~256 is the
                      arithmetic and ~1,130 is 32 rows x ~35 cc of fold.

  TRANSPOSED x^T[D,T] a beat is one FEATURE across all 32 tokens, and lane t is token t in
                      EVERY beat. acc[t] therefore collects token t's WHOLE row, and
                      SIMD_RED_LANEWISE just says "emit the accumulators": one beat holding
                      every token's sum of squares, no fold, no splat. The scale then rides
                      back in as a STICKY operand -- latched once for the whole task and
                      multiplied against every data beat, each lane by its own token's
                      scalar -- so nothing is replicated either. Measured: 270 cc for the
                      same 128 beats.

Measured end to end on snax_split_cluster at T=32, D=128 (see the snax reference app
target/snitch_cluster/sw/apps/snax-simd-rmsnorm):

    SIMD core (hart 1)          xDMA core (hart 2)
      row-major   3,135 cc        x   -> x^T   244 cc
      transposed  1,073 cc        y^T -> y     385 cc

Two engines, so two honest bounds: on the SIMD alone the transposed path is 65% cheaper;
fully serialised, with both conversions charged to it, still 42%. It always wins on hart 1,
which is the busier engine, and moving work off the SIMD and onto the xDMA is the direction
a layer wants.

======================================================================================
SO THE BLOCK CONVERTS, AND SAYS SO
======================================================================================

`in_transposed` / `out_transposed` are what the PRODUCER hands over and what the CONSUMER
wants. The block emits exactly the conversions the gap requires and no more, which is what
lets the orientation propagate along a chain instead of being paid twice:

    in    out    what gets built (at rows == 32)
    ----  -----  --------------------------------------------------------------
    F     F      Transpose(x -> x^T) . rmsnorm_t . Transpose(y^T -> y)
    T     F      Copy(x^T -> headroom) . rmsnorm_t . Transpose(y^T -> y)
    F     T      Transpose(x -> x^T) . rmsnorm_t
    T     T      Copy(x^T -> headroom) . rmsnorm_t

THE INPUT-SIDE TRANSPOSE NEED NOT BE PAID AT ALL. Transposing both sides of the PRODUCER's
matmul rewrites it with its axes exchanged -- (A.B)^T = B^T.A^T is the same GEMM with its
operands swapped, which is M and N exchanged in the config and nothing at run time. The
FlashAttention kernels already do this, which is why their softmax gets a LANEWISE rowmax
for free. Run the projection that way and its D-layout output relayouts straight to x^T,
so `in_transposed=True` costs the one-beat copy below and nothing else.

The OUTPUT side is not symmetric, and this is a hardware fact rather than an oversight. A
conversion into A-layout needs an 8-byte run contiguous on BOTH sides -- four consecutive
FEATURES of one token, exactly 8 B at fp16, which is also why the quantise has to come
after the reshape. In y those four are adjacent; in y^T what is contiguous is four
consecutive TOKENS, and no pair of strides makes a common run. comm/nest.py refuses
packed[D,T] -> B by name for the same reason. So y^T has to become y again before the
layer's reshape, whoever pays for it.

======================================================================================
THE ONE-BEAT HEADROOM, WHICH IS WHY THIS BLOCK OWNS THE BUFFER
======================================================================================

The sticky elementwise reads ONE FLAT SWEEP of 1 + D beats whose first beat is the scale.
There is no argument that separates the two: the seed must physically precede the tile. So
the block allocates a single (1 + D)-beat buffer, hands the kernel its base as `seed_addr`
and base + 64 as `input_addr`, and the kernel checks the adjacency rather than trusting it.

That is also why an already-transposed input is COPIED rather than used in place: the block
does not own the producer's buffer and cannot prepend to it. The copy is one xDMA pass over
D beats -- strictly cheaper than the transpose it replaces in the non-transposed case, so
`in_transposed=True` still comes out ahead.
"""

from dataclasses import dataclass

from bingo_kernel_args import (SnaxBingoKernelSimdRmsnormF16F16Args,
                               SnaxBingoKernelSimdRmsnormTF16F16Args,
                               SnaxBingoKernelXdma1dCopyArgs,
                               SnaxBingoKernelXdmaTranspose2dArgs)

from ...comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                     PortSpec)
from ...comm.ports import at_offset
from .common import BEAT_BYTES, LANES_PER_BEAT, check_pow2, check_row

_PATHS = ("auto", "transposed", "rowmajor")


def _check_transposer(rows: int, cols: int, who: str) -> None:
    """The xDMA transposer's shape rule, checked on BOTH directions of the round trip.

    Its fast path walks 8x8 element blocks and emits 8 spatial channels, so the source
    needs whole blocks down the rows and a whole number of 8-byte lanes across: M % 8 == 0
    and N * elem_bytes % 8 == 0. Off that path the kernel falls back to a DM-core loop,
    which is correct and roughly two orders of magnitude slower -- so it is a refusal here,
    not a silent downgrade, because the whole point of this route is the cycle count.
    """
    for m, n, tag in ((rows, cols, "x -> x^T"), (cols, rows, "y^T -> y")):
        if m % 8 or (n * 2) % 8:
            raise ValueError(
                f"{who}: the {tag} transpose is off the xDMA transposer's fast path "
                f"([{m}, {n}] at fp16 needs rows%8==0 and cols*2%8==0). It would fall back "
                f"to a DM-core element loop, which defeats the reason for transposing. "
                f"Use path='rowmajor'.")


@dataclass(frozen=True)
class NormCfg:
    """A [rows, cols] FP16 tile, plus the orientation of its two ends.

    `path` is the kernel choice and exists to be A/B'd, because that is the only honest way
    to bank the numbers in this module's docstring on a graph that is not the reference app:

      auto        transposed where it is legal (rows == LANES_PER_BEAT), row-major
                  otherwise. What a layer should use.
      transposed  refuse rather than fall back, so a shape that silently lost the fast path
                  is a build error instead of a quiet regression.
      rowmajor    the reference arm. Same goldens, same graph shape, one kernel different.
    """

    rows: int
    cols: int
    cluster: int = 0
    in_transposed: bool = False
    out_transposed: bool = False
    path: str = "auto"


class RMSNorm(Block):
    """RMSNorm over each row. FP16 in, FP16 out.

    There is no learnable gain here. The fused kernel is normalise-only; a layer that wants
    the usual per-channel weight applies it as a separate elementwise multiply.
    """

    name = "rmsnorm"

    def __init__(self, cfg: NormCfg = None, **params):
        self.cfg = cfg if cfg is not None else NormCfg(**params)
        c = self.cfg
        check_row(c.cols, "RMSNorm")
        # BOTH paths divide by the row length, so this is not a transposed-only rule.
        check_pow2(c.cols, "RMSNorm")
        if c.path not in _PATHS:
            raise ValueError(f"RMSNorm: path={c.path!r} is not one of {_PATHS}.")

        # WHICH KERNEL, decided here rather than in build(), so a shape that cannot take
        # the cheap path says so while the pipeline is still being written.
        legal = (c.rows == LANES_PER_BEAT)
        if c.path == "transposed" and not legal:
            raise ValueError(
                f"RMSNorm: path='transposed' needs rows == {LANES_PER_BEAT} -- one FP16 "
                f"lane per token, so a whole tile's per-token scalars are exactly one "
                f"beat. Got rows={c.rows}. At fewer rows a beat holds several features and "
                f"the per-lane accumulators mix tokens; at more, a feature spans several "
                f"beats and they mix the other way. Split the tile into "
                f"{LANES_PER_BEAT}-row slices, or use path='auto'.")
        self.transposed = legal and c.path != "rowmajor"
        if self.transposed:
            _check_transposer(c.rows, c.cols, "RMSNorm")

    # ---- the declared interface --------------------------------------------------------

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": PortSpec(Layout.PACKED, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1, transposed=c.in_transposed,
                              doc="row-major fp16; rows are normalised independently")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(Layout.PACKED, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1, transposed=c.out_transposed,
                              doc="fp16, normalised, in the orientation the cfg asked for")}

    # ---- building ----------------------------------------------------------------------

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        src = bound["x"].handle
        nodes = []
        out = (self._transposed if self.transposed else self._rowmajor)(g, ctx, src, nodes)
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], out, (nodes[-1],), cluster=c.cluster,
                               name="y")},
            inputs={"x": Port(self.inputs["x"], src, (nodes[0],), name="x")},
            nodes=nodes)

    def _xpose(self, g, ctx, nodes, name, src, dst, rows, cols):
        """One xDMA block transpose, [rows, cols] -> [cols, rows], fp16."""
        nodes.append(g.node(name, ctx.xdma, "__snax_bingo_kernel_xdma_transpose_2d",
                            SnaxBingoKernelXdmaTranspose2dArgs(src, dst, rows, cols, 2),
                            nodes[-1] if nodes else ()))
        return nodes[-1]

    def _transposed(self, g, ctx, src, nodes):
        """LANEWISE reduce + sticky scale, with the orientation closed at both ends."""
        c = self.cfg
        nbytes = c.rows * c.cols * 2
        # ONE allocation: the seed beat, then the tile. See the module docstring -- the
        # sticky elementwise sweeps 1 + D beats and the seed is simply its first.
        buf = g.l1(f"{self.name}_xt", BEAT_BYTES + nbytes)
        tile = at_offset(buf, BEAT_BYTES)

        if c.in_transposed:
            # Already the right orientation, wrong buffer: we cannot prepend a beat to one
            # we did not allocate, so it is copied down. One pass over D beats -- cheaper
            # than the transpose it replaces, so this case still comes out ahead.
            nodes.append(g.node("Stage_xt", ctx.xdma, "__snax_bingo_kernel_xdma_1d_copy",
                                SnaxBingoKernelXdma1dCopyArgs(src, tile, nbytes), ()))
        else:
            self._xpose(g, ctx, nodes, "Xpose_in", src, tile, c.rows, c.cols)

        yt = g.l1(f"{self.name}_yt", nbytes)
        nodes.append(g.node(
            "Rmsnorm_t", ctx.simd, "__snax_bingo_kernel_simd_rmsnorm_t_f16_f16",
            SnaxBingoKernelSimdRmsnormTF16F16Args(buf, tile, yt, c.rows, c.cols),
            nodes[-1]))
        if c.out_transposed:
            return yt
        y = g.l1(f"{self.name}_y", nbytes)
        # y^T is [cols, rows], so the transpose back runs the other way round.
        self._xpose(g, ctx, nodes, "Xpose_out", yt, y, c.cols, c.rows)
        return y

    def _rowmajor(self, g, ctx, src, nodes):
        """The reference arm: fold across lanes, broadcast the scale, multiply.

        Still one kernel and still three passes -- StreamMap's RSQRT func removed the core
        round trip from this path too -- it just pays the cross-lane fold and the [T, D]
        replication that the orientation above avoids entirely.
        """
        c = self.cfg
        nbytes = c.rows * c.cols * 2
        x = src
        if c.in_transposed:
            x = g.l1(f"{self.name}_x", nbytes)
            self._xpose(g, ctx, nodes, "Xpose_in", src, x, c.cols, c.rows)

        out = g.l1(f"{self.name}_y", nbytes)
        nodes.append(g.node(
            "Rmsnorm", ctx.simd, "__snax_bingo_kernel_simd_rmsnorm_f16_f16",
            SnaxBingoKernelSimdRmsnormF16F16Args(input_addr=x, output_addr=out,
                                                 rows=c.rows, cols=c.cols),
            nodes[-1] if nodes else ()))
        if not c.out_transposed:
            return out
        yt = g.l1(f"{self.name}_yt", nbytes)
        self._xpose(g, ctx, nodes, "Xpose_out", out, yt, c.rows, c.cols)
        return yt
