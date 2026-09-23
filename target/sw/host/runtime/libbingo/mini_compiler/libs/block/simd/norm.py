"""RMSNorm over each row, and the orientation that decides what it costs.

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

  row_major x[T, D]   a beat is 32 consecutive FEATURES of one token, so lane k collects
                      every 32nd feature and a row's D terms scatter across all 32 lanes.
                      They have to be folded, once per row. Then the single scalar has to
                      be SPLATTED back across a beat and REPLICATED for every beat of the
                      row, because a 2-operand elementwise reads both operands from one
                      3-D affine stream {operand, beat, row} and the three loops share one
                      stride set -- `beat` cannot be zeroed for the scalar operand alone.
                      Measured: 1,387 cc of reduce for 128 beats, of which ~256 is the
                      arithmetic and ~1,130 is 32 rows x ~35 cc of fold.

  col_major x^T[D,T]  a beat is one FEATURE across all 32 tokens, and lane t is token t in
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
      row_major   3,135 cc        x   -> x^T   244 cc
      col_major   1,073 cc        y^T -> y     385 cc

Two engines, so two honest bounds: on the SIMD alone col_major is 65% cheaper; fully
serialised, with both transposes charged to it, still 42%. It always wins on hart 1, the
busier engine, and moving work off the SIMD is the direction a layer wants.

======================================================================================
THE KERNEL IS INFERRED FROM THE TWO ENDS
======================================================================================

`in_layout` and `out_layout` are the only knobs: what the producer hands over and what
the consumer wants. The two implementations are not interchangeable options to pick
between on taste, they are what the hardware does to a row-major tile and to a
column-major one, so the input is WHICH TILE YOU HAVE and the kernel follows:

    col_major kernel  <=  an end asks for col_major, and rows == 32.
    row_major kernel  <=  otherwise -- both ends row_major, or a shape where a lane is not
                          one token and the cheap reduction does not exist at all.

THE ENDS DECIDE; THE BLOCK DOES NOT OVERRULE THEM. col_major is always cheaper on the
SIMD when it runs, but taking it unasked would be an optimality assumption rather than a
derivation: it would turn a caller who wants row_major on both ends into two transposes
and a different kernel. Whether col_major is worth PAYING for is a question about the
whole chain, not about this operator, so it belongs to comm/layout_pass.py, which asks by
varying these layouts. Saying row_major on both ends here means row_major, and costs
nothing.

TWO CASES ARE REFUSALS RATHER THAN FALLBACKS, because a fallback would be a lie:

  - the caller's two ends differ, so one transpose is unavoidable, and the shape is off
    the transposer's fast path. No kernel choice helps -- the transpose is between the
    caller's ends, not something this block picked.
  - an end asks for col_major, rows != 32 so the cheap kernel cannot run, and the
    row_major fallback would need a transpose this shape cannot do quickly either.

Keeping the block and the pass apart is why neither needs a cost model: the block's
answer is a hardware fact, and the pass is left counting passes per engine.

======================================================================================
WHICH ENGINE MOVES IT, AND WHERE IT IS MOVED FROM
======================================================================================

THE iDMA UNLESS THE MOVE TRANSPOSES. Only the xDMA can transpose -- the 8x8 block
transposer is a datapath extension it has and the iDMA does not -- but everything else
this block moves is a plain contiguous copy, and the iDMA does those on the DM core, idle
while the SIMD normalises and the xDMA transposes. A load put on the xDMA would queue
behind the very transpose its operand is waiting for, and the rest of the chain's
relayouts are queueing there too.

WHERE x LIVES IS READ OFF THE BINDING, not declared. `inputs` says mem_level=None --
wherever you have it -- and `needs` says L1, so the linker treats the gap as this block's
to close and build() reads the real level off the bound port. An operand left in main
memory therefore costs no ceremony at the call site, and when the tile is wanted
col_major the load IS the staging copy, so L3 costs no extra node at all.

    x in   in_layout   out_layout   what gets built (at rows == 32)
    -----  ----------  -----------  -------------------------------------------------
    L1     row_major   row_major    Rmsnorm
    L1     row_major   col_major    Xpose_in(x) . Rmsnorm_t
    L1     col_major   row_major    [Stage_xt(i)] . Rmsnorm_t . Xpose_out(x)
    L1     col_major   col_major    [Stage_xt(i)] . Rmsnorm_t
    L3     row_major   row_major    Load_x(i) . Rmsnorm
    L3     row_major   col_major    Load_x(i) . Xpose_in(x) . Rmsnorm_t
    L3     col_major   row_major    Stage_xt(i) . Rmsnorm_t . Xpose_out(x)
    L3     col_major   col_major    Stage_xt(i) . Rmsnorm_t

    (i) = iDMA, DM core.  (x) = xDMA.  The SIMD kernel is the only other node.

The bracketed copy is emitted only when x is in L1 and NOT already in this block's own
headroom buffer -- see THE ONE-BEAT HEADROOM below.

======================================================================================
THE ONE-BEAT HEADROOM, WHICH IS WHY THIS BLOCK OWNS THE BUFFER
======================================================================================

The sticky elementwise reads ONE FLAT SWEEP of 1 + D beats whose first beat is the scale.
No argument separates the two: the seed must physically precede the tile. So the block
allocates a single (1 + D)-beat buffer, hands the kernel its base as `seed_addr` and
base + 64 as `input_addr`, and the kernel checks the adjacency rather than trusting it.

That is also why an already-col_major x is COPIED when it arrives somewhere else: this
block cannot prepend a beat to an allocation it does not own. `alloc(ctx)` removes even
that. It hands out the tile handle before build(), so a producer can write there directly
and no staging node is emitted. The case that makes it worth having is the ordinary one:
a layer input arrives by a plain L3 -> L1 copy, so staging x TRANSPOSED in L3 costs the
host nothing, lands x^T in the headroom buffer, and leaves only the output transpose.

Orientation also propagates through elementwise operators for nothing -- a residual add
on two col_major operands emits a col_major result -- so a chain can stay col_major and
convert only where a GEMM demands a blocked layout.

======================================================================================
THE OUTPUT TRANSPOSE IS NOT REMOVABLE, AND IS THE EXPENSIVE HALF
======================================================================================

A conversion into A-layout needs an 8-byte run contiguous on BOTH sides -- four
consecutive FEATURES of one token, exactly 8 B at fp16, which is also why a quantise has
to come after a reshape rather than before. In y those four are adjacent; in y^T what is
contiguous is four consecutive TOKENS. So y^T has to become y again before the layer's
reshape, whoever pays for it. Two ways out of that look open and are not:

  FUSE IT INTO THE CONSUMER'S RELAYOUT. __snax_bingo_kernel_xdma_transpose_2d takes only
  (src, dst, M, N, elem_bytes) and derives its own 3-deep reader and writer nests from the
  8x8 tiling. Both nests are fully spent on the tiling, leaving no room to also apply an
  A-layout nest on the writer. y^T -> A is two passes in the hardware, not just in the
  planner.

  LET THE GEMM EAT y^T AS ITS B OPERAND. (A.B)^T = B^T.A^T would make the consumer read
  y^T directly, and the weight it pairs with is staged from L3 so transposing IT is free.
  But comm/nest.py refuses row_major -> B at EVERY shape, [D, T] included: B runs
  contiguously along columns and row_major along rows, so no pair of strides gives a
  common 8-byte run. The refusal is by name and shape-independent, not a near miss.

WHAT IS STILL OPEN is the bank conflict. The writer scatters 8 spatial channels spaced
`spatial_stride_dst = M * elem_bytes` apart, and the TCDM is 32 banks x 8 B = one 256 B
sweep. Xpose_in has M = rows = 32, so 64 B spacing across 4 distinct banks; Xpose_out has
M = cols = D = 128, so 256 B spacing and ALL EIGHT CHANNELS ON BANK 0. That is the 385 cc
against 244 cc for identical volume and an identical 128-transfer count. The condition is
`cols * elem_bytes % 256 == 0` -- every d_model that is a multiple of 128, so every real
one. Fixing it needs a destination ROW PITCH on the transpose kernel, which is a device
ABI change and is not done here.
"""

from dataclasses import dataclass

from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs,
                               SnaxBingoKernelSimdRmsnormArgs,
                               SnaxBingoKernelXdmaTranspose2dArgs)

from ...comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                     PortSpec)
from ...comm.ports import at_offset
from .common import BEAT_BYTES, LANES_PER_BEAT, check_pow2, check_row


def _transposer_ok(rows: int, cols: int) -> bool:
    """Can the xDMA transposer do BOTH directions of the round trip on its fast path?

    It walks 8x8 element blocks and emits 8 spatial channels, so the source needs whole
    blocks down the rows and a whole number of 8-byte lanes across: M % 8 == 0 and
    N * elem_bytes % 8 == 0. Off that path the kernel is still correct but falls back to a
    DM-core element loop, two orders of magnitude slower.

    A PREDICATE, NOT A REFUSAL, because the kernel is inferred: a shape the transposer
    cannot serve is a reason to keep the operand where it is, not a reason to fail. It
    becomes a refusal only when the caller's own layouts demand a transpose.
    """
    return all(m % 8 == 0 and (n * 2) % 8 == 0
               for m, n in ((rows, cols), (cols, rows)))


@dataclass(frozen=True)
class NormCfg:
    """A [rows, cols] FP16 tile, and the orientation of its two ends.

    THERE IS NO KERNEL KNOB. `in_layout` and `out_layout` are the CONSTRAINTS -- what the
    producer hands over and what the consumer wants -- and the implementation follows from
    them; see THE KERNEL IS INFERRED in the module docstring. That is deliberate: the two
    implementations are not interchangeable options, they are what the hardware does to a
    row-major tile and to a column-major one, and the only honest way to choose is to say
    which tile you have.
    """

    rows: int
    cols: int
    cluster: int = 0
    in_layout: Layout = Layout.ROW_MAJOR
    out_layout: Layout = Layout.ROW_MAJOR
    # THE PRODUCER WILL WRITE INTO alloc()'s SLOT. Declarative, because comm.layout_pass
    # has to price the block before anything is built and the staging copy is the whole
    # difference between the two col_major input arms. build() checks it against what was
    # actually bound and refuses a mismatch, so it cannot drift into a lie.
    x_in_place: bool = False


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
        # BOTH paths divide by the row length, so this is not a col_major-only rule.
        check_pow2(c.cols, "RMSNorm")
        for who, lay in (("in_layout", c.in_layout), ("out_layout", c.out_layout)):
            if lay not in (Layout.ROW_MAJOR, Layout.COL_MAJOR):
                raise ValueError(
                    f"RMSNorm: {who}={lay} is a BLOCKED layout. This operator reads a "
                    f"plain tile -- a row has to be contiguous in one orientation or the "
                    f"other -- so reshape out of {lay} before normalising.")

        # ---- INFERENCE. See THE KERNEL IS INFERRED FROM THE TWO ENDS, above. ---------
        # An end asking for col_major is what selects the col_major kernel; the shape is
        # what allows it. Neither overrules the caller, so row_major on both ends stays on
        # the row_major kernel even where the cheap one would be legal.
        wants_col = Layout.COL_MAJOR in (c.in_layout, c.out_layout)
        kernel_ok = (c.rows == LANES_PER_BEAT)
        self.col_major = kernel_ok and wants_col

        # THE TWO REFUSALS. A transpose the hardware would run as a DM-core element loop
        # is two orders of magnitude off the point of transposing, and neither kernel
        # choice can avoid it -- the transpose is between the caller's own ends.
        if c.in_layout != c.out_layout and not _transposer_ok(c.rows, c.cols):
            raise ValueError(
                f"RMSNorm: in_layout={c.in_layout} and out_layout={c.out_layout} differ, "
                f"so one xDMA transpose is unavoidable -- but [{c.rows}, {c.cols}] at fp16 "
                f"is off the transposer's fast path (it needs rows%8==0 and cols%4==0 in "
                f"both directions) and would fall back to a DM-core element loop, two "
                f"orders of magnitude slower. Ask for the same layout on both ends.")
        if not self.col_major and wants_col and not _transposer_ok(c.rows, c.cols):
            raise ValueError(
                f"RMSNorm: a col_major end was asked for at rows={c.rows}, but the "
                f"col_major kernel needs rows == {LANES_PER_BEAT} (one FP16 lane per "
                f"token) and the row_major fallback would need a transpose this shape "
                f"cannot do on the fast path. Split the tile into {LANES_PER_BEAT}-row "
                f"slices, or ask for row_major on both ends.")
        self._buf = None

    # ---- what it costs, for comm.layout_pass -------------------------------------------

    def simd_folds(self) -> int:
        """Serialised CROSS-LANE reductions this configuration performs. Exact, not a model.

        This is the whole cost difference between the two implementations and the only
        thing the layout choice changes on the SIMD core. StreamReduce carries one FP32
        accumulator per lane and never moves sideways, so a reduction ALONG beats is the
        accumulators doing what they already do, while a reduction ACROSS the lanes of a
        beat is a log-depth fold through treeBuf, serialised over treeLanes ALUs, holding
        the reader's input port low until it drains. row_major pays one per row; col_major
        pays none.

        COUNTED, NOT MEASURED, and that is on purpose. A cycle number would be one
        constant per arm, right for one shape on one RTL build, rotting silently into a
        cost model nobody re-measures. A fold count cannot rot: it is derived from the
        shape, it is what the hardware actually serialises, and it orders the arms without
        claiming to predict a cycle.
        """
        return self.cfg.rows if not self.col_major else 0

    def xdma_passes(self) -> int:
        """How many xDMA transfers build() will emit, for comm.layout_pass to count.

        ONLY THE TRANSPOSES; every other move is iDMA work and is counted by
        idma_passes(). Whether x arrives in L1 or L3 therefore does not change this number.

        Exact, not an estimate: it is the same decision build() makes, read off the cfg.
        """
        c = self.cfg
        n = 1 if c.in_layout != self._kernel_layout else 0         # Xpose_in
        return n + (1 if c.out_layout != self._kernel_layout else 0)   # Xpose_out

    def idma_passes(self, in_l3: bool = False) -> int:
        """Plain contiguous copies build() will emit, on the DM core.

        `in_l3` is not in the cfg because it is a property of the BINDING, not of the
        block -- build() reads it off the bound port. Pass it when costing a chain whose
        producer leaves x in main memory.
        """
        c = self.cfg
        if c.in_layout != self._kernel_layout:
            return 1 if in_l3 else 0                # Load_x, then the transpose
        if self.col_major:
            return 0 if (c.x_in_place and not in_l3) else 1     # Stage_xt
        return 1 if in_l3 else 0                    # Load_x

    @property
    def _kernel_layout(self) -> Layout:
        """The orientation the chosen kernel reads and writes."""
        return Layout.COL_MAJOR if self.col_major else Layout.ROW_MAJOR

    # ---- the headroom buffer, handed out before build() --------------------------------

    def alloc(self, ctx: Ctx):
        """Allocate the seed+tile buffer and return the TILE handle, for a producer to fill.

        Only the col_major path has one. Aiming an already-col_major producer at this
        handle is what removes the staging copy: build() compares what it was bound to
        against this exact slot and emits nothing when they match. Calling it is optional
        -- build() allocates the buffer itself otherwise.
        """
        if not self.col_major:
            raise ValueError(
                "RMSNorm.alloc() on the row-major path: there is no headroom buffer to "
                "write into. Only the col_major kernel takes a seed beat, and only it "
                "needs the tile placed 64 B after one. Check `.col_major` first -- it is "
                "inferred from the layouts and the shape, so at rows != 32 there is no "
                "slot to hand out.")
        if self._buf is None:
            c = self.cfg
            self._buf = ctx.at(c.cluster).l1(f"{self.name}_xt",
                                             BEAT_BYTES + c.rows * c.cols * 2)
        return self.tile

    @property
    def tile(self):
        """Where x^T goes: one beat into the buffer, because the seed beat precedes it."""
        if self._buf is None:
            raise ValueError("RMSNorm.tile before alloc(ctx); call alloc(ctx) first.")
        return at_offset(self._buf, BEAT_BYTES)

    # ---- the declared interface --------------------------------------------------------

    @property
    def inputs(self) -> dict:
        c = self.cfg
        # mem_level=None: WHEREVER YOU HAVE IT. This block loads its own operand, so a
        # producer that left x in main memory binds it here directly and build() emits the
        # load. `needs` below is what the block's own transfers read from once it has.
        return {"x": PortSpec(c.in_layout, DType.F16, (c.rows, c.cols),
                              mem_level=None,
                              doc="fp16, rows normalised independently. col_major means "
                                  "the same tensor with its bytes laid out [cols, rows] "
                                  "-- the shape is unchanged either way. L1 or L3; the "
                                  "block fetches")}

    @property
    def needs(self) -> dict:
        """What this block's OWN transfers read. The linker checks against this, not the
        declaration, so the L3 -> L1 gap is mine to close rather than a contract error."""
        return {"x": PortSpec(self.cfg.in_layout, DType.F16,
                              (self.cfg.rows, self.cfg.cols), mem_level=MemLevel.L1)}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(c.out_layout, DType.F16, (c.rows, c.cols),
                              mem_level=MemLevel.L1,
                              doc="fp16, normalised, in the orientation the cfg asked "
                                  "for")}

    # ---- building ----------------------------------------------------------------------

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        src = bound["x"].handle
        # WHERE x ACTUALLY IS, read off the bound port rather than assumed. Port fills this
        # in from the handle at bind time, so it is the real level and not the declaration.
        level = bound["x"].spec.mem_level
        if level not in (MemLevel.L1, MemLevel.L3):
            raise ValueError(
                f"RMSNorm: x is in {level}. This block loads from L3 or uses L1 in place; "
                f"the memory-chiplet pool is off-die and wants a host hoist first, which "
                f"is the caller's to emit because it is one move for the whole layer, not "
                f"one per operator.")
        nodes = []
        out = (self._col_major if self.col_major else self._row_major)(
            g, ctx, src, level, nodes)
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], out, (nodes[-1],), cluster=c.cluster,
                               name="y")},
            inputs={"x": Port(self.inputs["x"], src, (nodes[0],), name="x")},
            nodes=nodes)

    def _land(self, g, ctx, nodes, src, level, nbytes):
        """Get the operand into L1 if it is not already, on the iDMA. Returns the handle.

        On the iDMA because it transposes nothing -- see WHICH ENGINE MOVES IT, above.
        """
        if level == MemLevel.L1:
            return src
        dst = g.l1(f"{self.name}_x_l1", nbytes)
        nodes.append(g.node("Load_x", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(src, dst, nbytes),
                            nodes[-1] if nodes else ()))
        return dst

    def _xpose(self, g, ctx, nodes, name, src, dst, rows, cols):
        """One xDMA block transpose, [rows, cols] -> [cols, rows], fp16."""
        nodes.append(g.node(name, ctx.xdma, "__snax_bingo_kernel_xdma_transpose_2d",
                            SnaxBingoKernelXdmaTranspose2dArgs(src, dst, rows, cols, 2),
                            nodes[-1] if nodes else ()))
        return nodes[-1]

    def _col_major(self, g, ctx, src, level, nodes):
        """LANEWISE reduce + sticky scale, with the orientation closed at both ends."""
        c = self.cfg
        nbytes = c.rows * c.cols * 2
        # ONE allocation: the seed beat, then the tile. See the module docstring -- the
        # sticky elementwise sweeps 1 + D beats and the seed is simply its first.
        self.alloc(g)
        buf, tile = self._buf, self.tile

        if c.in_layout == Layout.COL_MAJOR:
            in_place = _same_slot(src, tile) and level == MemLevel.L1
            if c.x_in_place and not in_place:
                raise ValueError(
                    f"RMSNorm: x_in_place=True promises the producer wrote into this "
                    f"block's own headroom slot, but x is bound to a different buffer. "
                    f"Call alloc(ctx) and aim the producer at the handle it returns, or "
                    f"drop the flag and pay the staging copy. It is refused rather than "
                    f"quietly corrected because the flag is what the layout pass costed.")
            if in_place:
                # THE PRODUCER ALREADY WROTE HERE, so there is nothing to move. This is the
                # whole point of alloc(): the orientation and the headroom are both already
                # satisfied and the block emits no node at all.
                pass
            else:
                # Right orientation, wrong buffer -- this block cannot prepend a beat to
                # an allocation it does not own, so the tile is copied down. The iDMA
                # reaches main memory as readily as L1, so this one node also serves as
                # the load when x was never in L1 to begin with.
                nodes.append(g.node(
                    "Stage_xt", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(src, tile, nbytes), ()))
        else:
            src = self._land(g, ctx, nodes, src, level, nbytes)
            self._xpose(g, ctx, nodes, "Xpose_in", src, tile, c.rows, c.cols)

        yt = g.l1(f"{self.name}_yt", nbytes)
        nodes.append(g.node(
            "Rmsnorm_t", ctx.simd, "__snax_bingo_kernel_simd_rmsnorm",
            SnaxBingoKernelSimdRmsnormArgs(tile, yt, c.rows, c.cols,
                                           input_layout=Layout.COL_MAJOR,
                                           output_layout=Layout.COL_MAJOR,
                                           seed_addr=buf),
            nodes[-1] if nodes else ()))
        if c.out_layout == Layout.COL_MAJOR:
            return yt
        y = g.l1(f"{self.name}_y", nbytes)
        # y^T is [cols, rows], so the transpose back runs the other way round.
        self._xpose(g, ctx, nodes, "Xpose_out", yt, y, c.cols, c.rows)
        return y

    def _row_major(self, g, ctx, src, level, nodes):
        """Fold across lanes, broadcast the scale, multiply. Three passes, one kernel.

        The scalar stays in the datapath here too -- StreamMap's RSQRT computes it in the
        broadcast pass -- so what this pays over the col_major kernel is exactly the
        cross-lane fold and the [T, D] replication, once per row.
        """
        c = self.cfg
        nbytes = c.rows * c.cols * 2
        x = self._land(g, ctx, nodes, src, level, nbytes)
        if c.in_layout == Layout.COL_MAJOR:
            xt, x = x, g.l1(f"{self.name}_x", nbytes)
            self._xpose(g, ctx, nodes, "Xpose_in", xt, x, c.cols, c.rows)

        out = g.l1(f"{self.name}_y", nbytes)
        nodes.append(g.node(
            "Rmsnorm", ctx.simd, "__snax_bingo_kernel_simd_rmsnorm",
            SnaxBingoKernelSimdRmsnormArgs(x, out, c.rows, c.cols,
                                           input_layout=Layout.ROW_MAJOR,
                                           output_layout=Layout.ROW_MAJOR),
            nodes[-1] if nodes else ()))
        if c.out_layout != Layout.COL_MAJOR:
            return out
        yt = g.l1(f"{self.name}_yt", nbytes)
        self._xpose(g, ctx, nodes, "Xpose_out", out, yt, c.rows, c.cols)
        return yt


def _same_slot(a, b) -> bool:
    """Do two handles name the same bytes? Compared by allocation identity plus offset.

    A view and its base are different objects with the same address, so `is` would miss the
    match and charge a copy that is not needed; `==` on a dataclass view would match two
    different allocations that happen to share an offset. Both halves are needed.
    """
    def key(h):
        base = getattr(h, "base", None)
        return (id(h), 0) if base is None else (id(base), h.offset)
    return key(a) == key(b)
