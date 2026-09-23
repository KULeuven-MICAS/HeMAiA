"""RMSNorm over each row: which kernel runs, and what gets built around it.

    out[t, f] = x[t, f] / sqrt( (1/D) * SUM_f x[t, f]^2 )

One multiply per element -- the arithmetic of a residual add, which runs four times
cheaper over the same tile. All of the difference is the ONE SCALAR PER ROW: where it is
reduced, and how it gets back to the data. Everything below follows from that.

======================================================================================
1. THE TWO KERNELS, AND WHY THE TILE'S ORIENTATION PICKS ONE
======================================================================================

StreamReduce carries one FP32 accumulator PER LANE and never moves data sideways. A
reduction ALONG beats is therefore free; a reduction ACROSS the 32 lanes of a beat is a
serialised log-depth fold, ~35 cc a row, with the reader stalled.

  row_major kernel   x [T, D]. A beat is 32 features of ONE token, so a row's terms land
                     in all 32 lanes and must be FOLDED, once per row. The scalar is then
                     splatted and replicated into a [T, D] plane for the multiply.
                     reduce(SUMSQ) -> bcast_map(RSQRT) -> ew2(MUL)
  col_major kernel   x^T [D, T]. A beat is one feature of ALL 32 tokens -- lane t is token
                     t in every beat -- so the accumulators already hold every token's sum.
                     LANEWISE emits them as one beat; a STICKY multiply applies it. No fold,
                     no plane. Needs rows == 32 (one FP16 lane per token).
                     reduce(SUMSQ|LANEWISE) -> map(RSQRT), 1 beat -> ew(MUL|STICKY_B)

  Measured, snax_split_cluster, T = 32, D = 128:   row_major 3,135 cc   col_major 1,073 cc
                                                   x -> x^T 244 cc      y^T -> y  385 cc (xDMA)

Both are ONE device kernel, __snax_bingo_kernel_simd_rmsnorm, selected by the layout pair
in its args. It runs exactly four pairs:

    kernel route          reads        writes
    row_major             row_major    row_major               fp16 / int8
    col_major             col_major    col_major               fp16
    row_major -> A        row_major    the GEMM's A operand    fp16 / int8
    col_major -> B        col_major    the GEMM's B operand    fp16 / int8

======================================================================================
2. WRITING A GEMM OPERAND DIRECTLY: out_layout A or B
======================================================================================

When the consumer is a GEMM, the kernel can write its operand layout itself, and quantise
in the same pass (out_dtype I8). That removes the Reshape and the Quantize that would
otherwise sit between the norm and the GEMM. It works because the SIMD reader's lane
stride is programmable: each 8 B lane carries four consecutive elements of the tile, and
the operand's atom has to BE such a run --

    A (m, k, r, s)   atom = four FEATURES of one token  -> contiguous in a row_major tile
    B (n, k, c, s)   atom = four TOKENS of one feature  -> contiguous in a col_major tile

-- so A can only come from the row_major kernel and B only from the col_major one. The
other two pairings are a transpose, which only the xDMA does; this block puts it IN FRONT
(Xpose_in), never behind. A is what a projection x.W reads (contraction over d), B what a
GEMM contracting over the sequence reads (attention's P.V).

ANY MESH. The order the kernel reads its tile in to write a given mesh's blocks is derived
on the host (kernels/blocked_nest.py), verified against the operand's index map, and passed
down as a descriptor; `mesh` in the cfg is what it is derived for. A mesh no order fits is
refused at construction with the reason -- tileSize < 4 at fp16, or an int8 atom whose runs
land a block apart at uneven addresses (e.g. B on (1, 16, 32)).

======================================================================================
3. THE ROUTING TABLE: every combination, what it runs, what it builds
======================================================================================

The caller states `in_layout` (what the producer hands over), `out_layout` and `out_dtype`
(what the consumer wants). The route follows; there is no kernel knob.

  in_layout  out_layout  out_dtype  kernel route       nodes (x in L1)
  ---------  ----------  ---------  ----------------   ------------------------------------
  row_major  row_major   f16 / i8   row_major          Rmsnorm
  row_major  col_major   f16        col_major          Xpose_in . Rmsnorm_t
  row_major  A           f16 / i8   row_major -> A     Rmsnorm
  row_major  B           f16 / i8   col_major -> B     Xpose_in . Rmsnorm_t
  col_major  row_major   f16        col_major          Rmsnorm_t . Xpose_out
  col_major  row_major   i8         row_major          Xpose_in . Rmsnorm
  col_major  col_major   f16        col_major          Rmsnorm_t
  col_major  A           f16 / i8   row_major -> A     Xpose_in . Rmsnorm
  col_major  B           f16 / i8   col_major -> B     Rmsnorm_t
  any        col_major   i8         REFUSED -- no int8 col_major kernel; ask for B

  At rows != 32 the col_major kernel does not exist, so its rows change:
  row_major  col_major   f16        row_major          Rmsnorm . Xpose_out
  col_major  row_major   f16        row_major          Xpose_in . Rmsnorm
  col_major  col_major   f16        row_major          Xpose_in . Rmsnorm . Xpose_out
  any        B           any        REFUSED -- B needs the col_major kernel

  Around those, on the iDMA (DM core): x in L3 adds Load_x in front of a row_major-kernel
  route or of an Xpose_in; a col_major x feeding the col_major kernel is copied into this
  block's headroom buffer (Stage_xt) unless the producer already wrote there -- see 5.

THE BLOCK DOES NOT OVERRULE THE CALLER. `in_layout` and `out_layout` PINNED to row_major
stay on the row_major kernel although Xpose_in . col_major . Xpose_out would be cheaper on
the SIMD: a pinned field is a boundary the layer said it wanted, and a block that quietly
picked something else would make the layer's own source a lie. Leaving them unset is how a
layer says it does not mind -- see section 4.

TWO MORE REFUSALS, both about the transposer: a route that needs an Xpose on a shape off
its fast path (rows % 8 or cols % 4) is refused, because the fallback is a DM-core element
loop two orders of magnitude slower.

======================================================================================
4. WHICH ROUTE, AND WHO DECIDES
======================================================================================

`in_layout`, `out_layout` and `out_dtype` may be left UNSET. Then they are knobs this
block owns: `variants()` offers every combination, each one realised through the
constructor above -- so the refusals in section 3 are what prunes the list -- and the
pipeline picks the realisation whose boundary matches the neighbours at the least cost.

The block prices itself from the same decision build() makes, so nothing is measured and
nothing rots:

    simd_folds()    cross-lane folds: `rows` on the row_major kernel, 0 on col_major
    simd_passes()   SIMD tasks: 3 for the plain routes, 3 or 4 plus any repeat the
                    blocked nest needs for A / B
    xdma_passes()   Xpose_in + Xpose_out
    idma_passes()   Load_x / Stage_xt

Ranked folds first, then SIMD passes, then xDMA, then iDMA (comm/variant.py). That order
is why a pinned row_major/row_major pair is NOT what an open one resolves to: given the
choice, the block would rather pay two transposes on an idle engine than 32 serialised
folds on the busy one.

THE CHOICE IS INSIDE THE BOUNDARY, NOT ACROSS IT. Whatever the norm picks, it still
presents exactly the ports it declared, so a Reshape or a Quantize the layer wrote after
it is still built, still in the layer's own order. Writing A/int8 straight out of the
kernel is something a layer asks for by connecting the norm to the GEMM directly -- not
something that happens to stages it wrote down.

======================================================================================
5. BUFFERS AND ENGINES
======================================================================================

THE ONE-BEAT HEADROOM. The sticky multiply reads one flat sweep of 1 + D beats whose FIRST
beat is the scale, so the scale must physically precede x^T. The block allocates one
(1 + D)-beat buffer and hands the kernel its base as `seed_addr`, base + 64 as the tile;
the kernel checks the adjacency. That is why a col_major x arriving elsewhere is copied
(Stage_xt): a beat cannot be prepended to an allocation the block does not own. alloc()
hands the tile slot out before build(), so a producer can write there directly
(x_in_place) and no copy is emitted -- staging x transposed in L3 costs the host nothing.

PLAIN COPIES GO ON THE iDMA, TRANSPOSES ON THE xDMA. Only the xDMA has the 8x8 transposer;
every other move is a contiguous copy the idle DM core does, instead of queueing behind
the transposes on the xDMA.

WHERE x LIVES IS READ OFF THE BINDING. `inputs` says mem_level=None -- wherever you have it
-- and build() reads the real level off the bound port.

STILL OPEN: Xpose_out's bank conflict. The transposer's writer spaces its 8 channels
M * 2 bytes apart; at M = D = 128 that is 256 B, one TCDM sweep, so all eight hit bank 0 --
385 cc against Xpose_in's 244 for the same volume. It needs a destination row pitch on the
transpose kernel, a device ABI change. Routes with a blocked output never pay it: they
write the operand directly.
"""

import itertools
from dataclasses import dataclass
from typing import NamedTuple, Optional

from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs,
                               SnaxBingoKernelSimdRmsnormArgs,
                               SnaxBingoKernelXdmaTranspose2dArgs)

from ...comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                     PortSpec)
from ...comm.ports import at_offset
from ...comm.variant import free_fields
from .common import BEAT_BYTES, LANES_PER_BEAT, check_pow2, check_row

_ORIENTATIONS = (Layout.ROW_MAJOR, Layout.COL_MAJOR)
_BLOCKED = (Layout.A, Layout.B)


def _transposer_ok(rows: int, cols: int) -> bool:
    """Can the xDMA transposer do BOTH directions of the round trip on its fast path?

    It walks 8x8 element blocks and emits 8 spatial channels, so a source needs whole
    blocks down the rows and whole 8-byte lanes across: M % 8 == 0, N * 2 % 8 == 0.
    """
    return all(m % 8 == 0 and (n * 2) % 8 == 0
               for m, n in ((rows, cols), (cols, rows)))


class Route(NamedTuple):
    """What build() will emit, decided once in the constructor. See the table, section 3."""
    kernel: Layout          # ROW_MAJOR or COL_MAJOR: which kernel -- the orientation it reads
    kernel_out: Layout      # what the kernel writes: its own orientation, A, or B
    xpose_in: bool          # an xDMA transpose in front
    xpose_out: bool         # an xDMA transpose behind


def route(rows: int, in_layout: Layout, out_layout: Layout, out_dtype: DType) -> Route:
    """The routing table as code. Raises on the refused combinations."""
    col_ok = rows == LANES_PER_BEAT
    if out_dtype == DType.I8 and out_layout == Layout.COL_MAJOR:
        raise ValueError(
            "RMSNorm: there is no int8 col_major kernel. For an int8 GEMM operand ask for "
            "out_layout B, which quantises in the same kernel.")
    if out_layout == Layout.A:
        kernel, kernel_out = Layout.ROW_MAJOR, Layout.A
    elif out_layout == Layout.B:
        if not col_ok:
            raise ValueError(
                f"RMSNorm: out_layout B needs the col_major kernel -- B's atom is four "
                f"tokens of one feature, a col_major run -- and that kernel needs rows == "
                f"{LANES_PER_BEAT} (one FP16 lane per token). Got rows={rows}; split the "
                f"tile into {LANES_PER_BEAT}-row slices.")
        kernel, kernel_out = Layout.COL_MAJOR, Layout.B
    else:
        # A plain output: an end asking for col_major selects the col_major kernel where it
        # exists; an int8 output exists only on the row_major one.
        wants_col = Layout.COL_MAJOR in (in_layout, out_layout)
        kernel = (Layout.COL_MAJOR if col_ok and wants_col and out_dtype != DType.I8
                  else Layout.ROW_MAJOR)
        kernel_out = kernel
    return Route(kernel, kernel_out, xpose_in=in_layout != kernel,
                 xpose_out=out_layout in _ORIENTATIONS and out_layout != kernel_out)


@dataclass(frozen=True)
class NormCfg:
    """A [rows, cols] FP16 tile, what the producer hands over and what the consumer wants.

    There is no kernel knob: the route follows from these (section 3 of the module doc).
    """

    rows: int
    cols: int
    cluster: int = 0
    # THE THREE BOUNDARY FIELDS. Left None, each is a knob this block owns and the
    # pipeline resolves from the neighbours; pinned, it is a constraint every realisation
    # has to meet. A layer pins the ones it cares about and leaves the rest open.
    #   in_layout   what the producer hands over: ROW_MAJOR, or COL_MAJOR ([cols, rows])
    #   out_layout  an orientation, or a GEMM operand layout, A or B
    #   out_dtype   F16, or I8 quantised inside the kernel. Not on a col_major output.
    in_layout: Optional[Layout] = None
    out_layout: Optional[Layout] = None
    out_dtype: Optional[DType] = None
    # The Fp16ToInt8 scale, FP32 bits. 0 keeps the kernel's baked 64.0 (a normalised row
    # sits in ~[-2, 2]); a layer with a data-derived scale passes it here.
    inv_scale_f32bits: int = 0
    # (meshRow, tileSize, meshCol) of the GEMM that consumes an A or B output. Required
    # for those, ignored otherwise: the kernel's read order is derived for it.
    mesh: Optional[tuple] = None
    # The producer will write x^T into alloc()'s slot, so no staging copy is needed.
    # Declarative so the planner can price it; build() checks it against the binding.
    x_in_place: bool = False


class RMSNorm(Block):
    """RMSNorm over each row, FP16 in; FP16 or INT8 out, in an orientation or a GEMM
    operand layout. No learnable gain: a layer that wants one applies it separately."""

    name = "rmsnorm"

    def __init__(self, cfg: NormCfg = None, **params):
        self.cfg = cfg if cfg is not None else NormCfg(**params)
        c = self.cfg
        check_row(c.cols, "RMSNorm")
        check_pow2(c.cols, "RMSNorm")          # 1/D is an exponent subtract on every route
        # A TEMPLATE: a boundary field is still open, so there is no route to pick yet.
        # The shape checks above still run, because they hold for every realisation and
        # failing them early names the real problem instead of burying it in a list of
        # refused variants.
        self.free = free_fields(c, ("in_layout", "out_layout", "out_dtype"))
        if self.free:
            self.route, self._probe, self._buf = None, None, None
            return
        if c.in_layout not in _ORIENTATIONS:
            raise ValueError(
                f"RMSNorm: in_layout={c.in_layout} is a blocked layout. The kernel reads a "
                f"plain tile; reshape out of {c.in_layout} before normalising.")
        if c.out_layout not in _ORIENTATIONS + _BLOCKED:
            raise ValueError(
                f"RMSNorm: out_layout={c.out_layout}. The kernel writes row_major, "
                f"col_major, or a GEMM operand layout (A or B); D is the GEMM's output, "
                f"not an operand.")
        if c.out_dtype not in (DType.F16, DType.I8):
            raise ValueError(f"RMSNorm: out_dtype={c.out_dtype}; fp16 or int8.")

        self.route = route(c.rows, c.in_layout, c.out_layout, c.out_dtype)
        if (self.route.xpose_in or self.route.xpose_out) and not _transposer_ok(c.rows,
                                                                                c.cols):
            raise ValueError(
                f"RMSNorm: {c.in_layout} -> {c.out_layout} runs the {self.route.kernel} "
                f"kernel and so needs an xDMA transpose, but [{c.rows}, {c.cols}] is off "
                f"the transposer's fast path (rows % 8 == 0 and cols % 4 == 0 both ways) "
                f"and would fall back to a DM-core element loop. Hand over x "
                f"{self.route.kernel}, or ask for an output that needs no transpose.")

        # A blocked output: derive (and verify) the kernel's read order NOW, so a mesh no
        # order fits is refused at construction, which is also where the resolver drops it.
        self._probe = None
        if c.out_layout in _BLOCKED:
            if c.mesh is None:
                raise ValueError(
                    f"RMSNorm: out_layout {c.out_layout} needs mesh=(meshRow, tileSize, "
                    f"meshCol) of the consuming GEMM; the kernel's read order is derived "
                    f"for it.")
            self._probe = self._kernel_args(0, 0, 64)
        self._buf = None

    def variants(self) -> list:
        """Every boundary this norm could present, over the fields left open.

        `x_in_place` is deliberately NOT among them. It is a promise about what some other
        block will do -- that the producer wrote x^T into alloc()'s slot -- and a resolver
        choosing it would be choosing on another block's behalf. It stays the caller's,
        checked against the binding in build().
        """
        if not self.free:
            return [{}]
        choices = {"in_layout": _ORIENTATIONS,
                   "out_layout": _ORIENTATIONS + _BLOCKED,
                   "out_dtype": (DType.F16, DType.I8)}
        return [dict(zip(self.free, combo))
                for combo in itertools.product(*(choices[n] for n in self.free))]

    def _check_realised(self) -> None:
        if self.free:
            raise ValueError(
                f"RMSNorm is a template: {', '.join(self.free)} not decided. Either pin "
                f"them, or add it to a Pipeline -- run() resolves them from the "
                f"neighbours and builds the realisation it picked.")

    # ---- what it costs, for the pipeline resolver (section 4) --------------------------

    @property
    def col_major(self) -> bool:
        """Does this configuration run the col_major kernel?"""
        return self.route.kernel == Layout.COL_MAJOR

    def simd_folds(self) -> int:
        """Serialised cross-lane folds: one per row on the row_major kernel, none on the
        col_major one. Counted from the shape, not measured."""
        return 0 if self.col_major else self.cfg.rows

    def simd_passes(self) -> int:
        """SIMD tasks the kernel runs. Three on the plain routes; the blocked ones add their
        reordering read -- plus any repeat of it the mesh's nest needs -- and col_major -> B
        keeps its multiply as a separate pass before that read."""
        if self._probe is None:
            return 3
        reps = self._probe.nest.reps
        return (2 if self.route.kernel == Layout.ROW_MAJOR else 3) + reps

    def xdma_passes(self) -> int:
        """Transposes this block emits, Xpose_in plus Xpose_out."""
        return int(self.route.xpose_in) + int(self.route.xpose_out)

    def idma_passes(self, in_l3: bool = False) -> int:
        """Plain copies on the DM core. `in_l3` is a property of the BINDING, which build()
        reads off the bound port; pass it when pricing a chain whose x stays in main memory."""
        c = self.cfg
        if self.route.xpose_in or not self.col_major:
            return 1 if in_l3 else 0                              # Load_x
        return 0 if (c.x_in_place and not in_l3) else 1           # Stage_xt

    @property
    def _out_bytes(self) -> int:
        c = self.cfg
        return c.rows * c.cols * (1 if c.out_dtype == DType.I8 else 2)

    def _kernel_args(self, x, y, seed):
        """The kernel's args for this route: the layout pair, precision, and -- for A / B --
        the mesh the blocked nest is derived for."""
        c, r = self.cfg, self.route
        blocked = r.kernel_out in _BLOCKED
        return SnaxBingoKernelSimdRmsnormArgs(
            x, y, c.rows, c.cols, input_layout=r.kernel, output_layout=r.kernel_out,
            seed_addr=seed if self.col_major else 0,
            out_i8=c.out_dtype == DType.I8,
            inv_scale_f32bits=c.inv_scale_f32bits,
            mesh=c.mesh if blocked else None)

    # ---- the headroom buffer, handed out before build() --------------------------------

    def alloc(self, ctx: Ctx):
        """Allocate the seed + tile buffer and return the TILE handle, for a producer to
        fill. Only the col_major kernel has one; aiming a producer at it removes Stage_xt."""
        self._check_realised()
        if not self.col_major:
            raise ValueError(
                "RMSNorm.alloc() on a row_major-kernel route: there is no headroom buffer. "
                "Only the col_major kernel takes a seed beat. Check `.col_major` first -- "
                "it follows from the layouts and the shape.")
        if self._buf is None:
            c = self.cfg
            self._buf = ctx.at(c.cluster).l1(f"{self.name}_xt",
                                             BEAT_BYTES + c.rows * c.cols * 2)
        return self.tile

    @property
    def tile(self):
        """Where x^T goes: one beat into the buffer, after the seed beat."""
        if self._buf is None:
            raise ValueError("RMSNorm.tile before alloc(ctx); call alloc(ctx) first.")
        return at_offset(self._buf, BEAT_BYTES)

    # ---- the declared interface --------------------------------------------------------

    @property
    def inputs(self) -> dict:
        self._check_realised()
        c = self.cfg
        return {"x": PortSpec(c.in_layout, DType.F16, (c.rows, c.cols), mem_level=None,
                              doc="fp16. col_major means the same tensor stored [cols, "
                                  "rows]. L1 or L3; the block fetches")}

    @property
    def needs(self) -> dict:
        """What this block's OWN transfers read; the linker checks against this."""
        return {"x": PortSpec(self.cfg.in_layout, DType.F16,
                              (self.cfg.rows, self.cfg.cols), mem_level=MemLevel.L1)}

    @property
    def outputs(self) -> dict:
        self._check_realised()
        c = self.cfg
        return {"y": PortSpec(c.out_layout, c.out_dtype, (c.rows, c.cols),
                              mem_level=MemLevel.L1,
                              doc="normalised, in the layout and precision asked for")}

    # ---- building: the route, node by node ---------------------------------------------

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        self._check_realised()
        c, r = self.cfg, self.route
        g = ctx.at(c.cluster)
        src = bound["x"].handle
        level = bound["x"].spec.mem_level      # the real level, read off the binding
        if level not in (MemLevel.L1, MemLevel.L3):
            raise ValueError(
                f"RMSNorm: x is in {level}. This block loads from L3 or uses L1 in place; "
                f"the memory-chiplet pool wants a host hoist first, one move per layer.")
        nodes = []
        nbytes = c.rows * c.cols * 2

        # 1. x into the orientation the kernel reads, in L1 (and in the headroom slot for
        #    the col_major kernel).
        if self.col_major:
            self.alloc(g)
            seed, x = self._buf, self.tile
            if r.xpose_in:
                src = self._land(g, ctx, nodes, src, level, nbytes)
                self._xpose(g, ctx, nodes, "Xpose_in", src, x, c.rows, c.cols)
            else:
                in_place = _same_slot(src, x) and level == MemLevel.L1
                if c.x_in_place and not in_place:
                    raise ValueError(
                        "RMSNorm: x_in_place=True promises the producer wrote into alloc()'s "
                        "slot, but x is bound elsewhere. Aim the producer at the handle "
                        "alloc(ctx) returns, or drop the flag and pay the staging copy.")
                if not in_place:
                    nodes.append(g.node("Stage_xt", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                                        SnaxBingoKernelIdma1dCopyArgs(src, x, nbytes), ()))
        else:
            seed = 0
            x = self._land(g, ctx, nodes, src, level, nbytes)
            if r.xpose_in:
                xt, x = x, g.l1(f"{self.name}_x", nbytes)
                self._xpose(g, ctx, nodes, "Xpose_in", xt, x, c.cols, c.rows)

        # 2. The kernel, writing its route's output: its own orientation, A, or B.
        y = g.l1(f"{self.name}_y", nbytes if r.xpose_out else self._out_bytes)
        nodes.append(g.node("Rmsnorm_t" if self.col_major else "Rmsnorm", ctx.simd,
                            "__snax_bingo_kernel_simd_rmsnorm",
                            self._kernel_args(x, y, seed), nodes[-1] if nodes else ()))

        # 3. Back to the orientation the consumer asked for, if the kernel's differs.
        if r.xpose_out:
            out = g.l1(f"{self.name}_yo", nbytes)
            if self.col_major:     # y^T is [cols, rows]
                self._xpose(g, ctx, nodes, "Xpose_out", y, out, c.cols, c.rows)
            else:
                self._xpose(g, ctx, nodes, "Xpose_out", y, out, c.rows, c.cols)
            y = out

        return BlockResult(
            outputs={"y": Port(self.outputs["y"], y, (nodes[-1],), cluster=c.cluster,
                               name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (nodes[0],), name="x")},
            nodes=nodes)

    def _land(self, g, ctx, nodes, src, level, nbytes):
        """x into L1 on the iDMA if it is not there already. Returns the L1 handle."""
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


def _same_slot(a, b) -> bool:
    """Do two handles name the same bytes? Allocation identity plus offset: a view and its
    base are different objects at one address, so `is` would charge a needless copy."""
    def key(h):
        base = getattr(h, "base", None)
        return (id(h), 0) if base is None else (id(base), h.offset)
    return key(a) == key(b)
