# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""What a block DECLARES: its ports, and the vocabulary they are written in.

A port states five things about an operand -- its layout, its precision, its shape, the
memory it lives in and, when that memory is a cluster's TCDM, whose -- and every one of
them is part of the signature rather than an assumption, because none of them faults when
it is wrong. A mismatched layout is a permutation that computes a scrambled answer; a
mismatched precision reads two elements as one; an address in a memory pool the platform
does not have is simply unmapped; and a handle in another cluster's TCDM is written by a
transfer that completes without moving anything.

NONE OF THE FIVE MAY BE LEFT OPEN. A port that declined to say where its tensor is would
be declaring less than it knows, and the hole would be filled by whatever happened to be
bound to it. A block that can genuinely take an operand from more than one place is a
FAMILY -- one realisation per place, each pricing its own transfers -- and the resolver
picks the one its producer matches. That is comm.variant; this module only makes the
incomplete declaration impossible to write.

This module has no knowledge of HOW a gap gets closed -- that is comm.transfer and
comm.nest. What it does own is the VOCABULARY the ports are written in: Layout, DType and
MemLevel. The layouts are worked through with real index maps just below, because
"D-layout" is not self-explanatory and reading a buffer in the wrong one is the failure
this whole module exists to make impossible.
"""

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Optional

from bingo_mem_handle import (BingoMemAlloc, BingoMemAllocView,
                              BingoMemFixedAddr, BingoMemSymbol)

# WHAT A LAYOUT NAME MEANS: where element [row][col] physically sits in the buffer.
#
# Every one of these holds the same values in the same number of bytes. They differ only in
# the ORDER, which is exactly why getting one wrong does not fault -- every byte is read,
# every byte is written, the arithmetic runs, and the answer is a scrambled tensor. On
# random test data the golden is scrambled identically and it PASSES.
#
# The maps below are real, printed from index_map() for an 8x8 matrix on a small array,
# (meshRow, tileSize, meshCol) = (4, 2, 4). The production array is (16, 4, 16); the shape
# of the permutation is the same, the numbers are just bigger.
#
#   row_major                       A
#         c0 c1 c2 c3 c4 c5 c6 c7         c0 c1 c2 c3 c4 c5 c6 c7
#     r0   0  1  2  3  4  5  6  7     r0   0  1  8  9 16 17 24 25
#     r1   8  9 10 11 12 13 14 15     r1   2  3 10 11 18 19 26 27
#     r2  16 17 18 19 20 21 22 23     r2   4  5 12 13 20 21 28 29
#     r3  24 25 26 27 28 29 30 31     r3   6  7 14 15 22 23 30 31
#
#   col_major                       (row_major again, for the eye)
#         c0 c1 c2 c3 c4 c5 c6 c7         c0 c1 c2 c3 c4 c5 c6 c7
#     r0   0  8 16 24 32 40 48 56     r0   0  1  2  3  4  5  6  7
#     r1   1  9 17 25 33 41 49 57     r1   8  9 10 11 12 13 14 15
#     r2   2 10 18 26 34 42 50 58     r2  16 17 18 19 20 21 22 23
#     r3   3 11 19 27 35 43 51 59     r3  24 25 26 27 28 29 30 31
#
#           col_major IS a bijection on the SAME shape -- 8x8 in, 8x8 out, [r][c] simply
#           lands at c*rows + r. That is the whole content of the orientation axis, and
#           seeing it beside row_major is the quickest way to believe it belongs in this
#           enum rather than in a boolean next to it.
#
#   D                               B
#         c0 c1 c2 c3 c4 c5 c6 c7         c0 c1 c2 c3 c4 c5 c6 c7
#     r0   0  1  2  3 16 17 18 19     r0   0  2  4  6 32 34 36 38
#     r1   4  5  6  7 20 21 22 23     r1   1  3  5  7 33 35 37 39
#     r2   8  9 10 11 24 25 26 27     r2   8 10 12 14 40 42 44 46
#     r3  12 13 14 15 28 29 30 31     r3   9 11 13 15 41 43 45 47
#
# "row_major"
#           Plain row-major: [r][c] is at r*cols + c. The layout everything outside the
#           array uses -- a numpy array, a golden, a tensor from the host. It is the "no
#           blocking" layout, and it is also the ORIENTATION PIVOT: a conversion between
#           col_major and a blocked layout goes through it in two steps, because the
#           transposer permutes a plain array and a nest is derived in the blocked side's
#           dimensions. Conversions that need no transpose are still derived directly.
#
# "col_major"
#           The same values with the axes stored the other way: [r][c] is at c*rows + r.
#           The tensor's SHAPE is unchanged -- still (rows, cols), still 32 tokens of 128
#           features -- only the address map differs, which is why this is a layout and
#           not a flag. A per-row SIMD reduction wants it: see the essay below.
#
# "A" "B" "D"
#           The array's BLOCKED layouts, one per port: A is operand A (m, k, r, s), B is
#           operand B (n, k, c, s), D is the output (m, n, r, c). Each interleaves a block
#           of the matrix so the array can stream it a tile at a time.
#
#           NOTE WHAT B DOES. In the table above, +1 along a row moves +2, but +1 DOWN A
#           COLUMN moves +1: B runs contiguously down columns while row_major, A and D
#           run along rows. So every conversion into or out of B is a TRANSPOSE, not a
#           reshape,
#           and no pair of strides expresses it -- it needs the xDMA transposer kernels,
#           which are correct only at elem_bytes=1. comm/nest.py refuses it by name.
#
# "d32"     The D port's INT32 SCATTER, and a DIFFERENT bijection from "D" -- which is the
#           whole reason it has its own name rather than being "D with wider elements".
#
#           The port permutes differently depending on element width. Over a whole FA O
#           tile (Br x d), measured:
#
#               FP16   == row-major?  YES
#               INT32  == row-major?  NO -- 2048 of 4096 elements differ
#
#           At INT32, walking c along one row gives these byte offsets:
#
#               c     :  0  1  2  3  4  5  6  7   8   9  10  11  12  13  14  15
#               byte  :  0  4  8 12 16 20 24 28  64  68  72  76  80  84  88  92
#               delta :    4  4  4  4  4  4  4  36   4   4   4   4   4   4   4
#
#           Seven steps of 4 bytes, then a jump of 36. Anything that reads a d32 buffer as
#           a matrix without un-permuting gets scrambled values and no error anywhere,
#           which is why Attention types its output port d32: a downstream block physically
#           cannot bind it as if it were D.
#
#           The redeeming detail: split c into (c_lo = c % 8, c_hi = c // 8) and each has a
#           CONSTANT stride -- c_lo is 4 B x 8, c_hi is 64 B x 2. So the un-permute is
#           affine in five dimensions, which is one xdma_6d pass rather than a host repack.
#
#           index_map() in comm/nest.py deliberately has no entry for d32: it is an output
#           permutation, not a transport source.
#
# "monoid"  The monoid junction's lane geometry: lane = field*S + slot within a 16-lane
#           FP32 beat, field 0 = m and field 1 = l. It is what a partial (m, l) physically
#           is, and calling it row-major would invite a consumer to read it as rows.
#
# ======================================================================================
# ORIENTATION IS A LAYOUT, NOT A FLAG BESIDE ONE
# ======================================================================================
#
# `row_major` and `col_major` are two members of the SAME enum. Orientation is not a flag
# beside a layout, because it is a layout by the same definition every other name here
# satisfies: a BIJECTION ON A FIXED SHAPE. index_map("col_major", r, c) = c*rows + r --
# one line, same signature, same shape in and out. A [32, 128] tensor stored column-major
# still holds 32*128 elements indexed [r][c]; what changes is where element [r][c] sits,
# which is exactly what an index map says.
#
# IT COSTS ONE NAME, NOT SIX. Orientation is only free on the UNBLOCKED layout: A, B and D
# already fix theirs -- B is defined as the one that runs down columns -- so there is no
# A_T or D_T to name. Seven members, not twelve.
#
# KEEPING IT IN THE ENUM KEEPS BYTE ORDER IN ONE LANGUAGE: a verified permutation per
# name, which convert_args derives strides from and _verify_nest checks element by
# element. A boolean beside the layout would be a second, unverified description that
# every seam then has to reconcile -- which order a transpose and a relayout compose in, a
# `stored_shape` to translate between the two, a runtime refusal for `A^T` states that
# should not be spellable at all.
#
# THE SHAPE STAYS LOGICAL. A col_major port still declares shape = (rows, cols) -- the
# tensor's own dimensions, what the layer reasons about (32 tokens of 128 features) -- and
# the LAYOUT says the bytes are laid out [cols, rows]. That is what lets the linker compare
# a producer's output to a consumer's requirement at all: two specs whose shapes are
# (32, 128) and (128, 32) are, as far as check_contract can tell, different tensors.
#
# WHO CLOSES THE GAP: comm.transfer. A pair whose contiguous axes disagree is not a stride
# nest and nest.py refuses it; transfer.plan catches that and routes row_major <-> col_major
# to the xDMA's 8x8 block transposer, a REAL hardware unit, correct at 1- and 2-byte
# elements only (the cfg's elementWidth: [8, 16]). A blocked layout on one side and
# col_major on the other is planned as two steps THROUGH row_major, because the transposer
# permutes a plain array and the nest must be derived in the blocked side's own dimensions.
#
# WHY ANYTHING WANTS THIS. The SIMD block reduces ALONG BEATS for free -- one FP32
# accumulator per lane, lane k folding into acc[k] every beat -- and ACROSS the lanes of a
# beat only through a serialised log-depth fold that stalls the reader once per row. Which
# one a per-row operator gets is decided entirely by orientation: at [T, D] a row's terms
# scatter across all 32 lanes and must be folded, at [D, T] one lane IS one token and the
# answer is already in the accumulator. RMSNorm measures 3x cheaper in col_major for
# exactly that reason, and softmax's rowmax is the same argument.
class _Vocab(StrEnum):
    """A closed set of names, where a typo is a build error rather than a wrong answer.

    A member IS its string: it compares equal to it, hashes with it and formats as it, so
    "A" and Layout.A are interchangeable at any call site. What the enum adds is that the
    options are listable, each carries its own one-line `doc`, and an unknown value is
    refused with the valid ones named.
    """

    def __new__(cls, value, doc=""):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.doc = doc
        return obj

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value!r} is not a valid {cls.__name__}. Valid values are "
            f"{', '.join(repr(str(m)) for m in cls)}.")


class Layout(_Vocab):
    """How a matrix is ordered in memory. See the reference above for the index maps."""

    A = "A", "operand A of the GEMM: (m, k, r, s)"
    B = "B", "operand B of the GEMM: (n, k, c, s) -- runs down COLUMNS, so converting is a transpose"
    D = "D", "the GEMM output: (m, n, r, c)"
    ROW_MAJOR = "row_major", "plain row-major, what everything outside the array uses"
    COL_MAJOR = "col_major", "the same values stored [cols, rows]: what a per-row reduction wants"
    D32 = "d32", "the D port's INT32 scatter -- a DIFFERENT bijection from D"
    MONOID = "monoid", "the junction's lane geometry: lane = field*S + slot, field 0 = m, 1 = l"


class DType(_Vocab):
    """The precision of an element. Changing it needs a scale, so it is never automatic."""

    I8 = "i8", "signed 8-bit, the GEMM's operand precision"
    F16 = "f16", "half precision, what the SIMD computes in"
    I32 = "i32", "signed 32-bit, the GEMM's accumulator"
    F32 = "f32", "single precision, what the fabric junctions fold in"


class MemLevel(_Vocab):
    """Where a buffer lives. Spelled the way BingoMemAlloc spells it, because they are the
    same question and two vocabularies for one thing is how they drift.

    Part of the signature for the same reason the layout is: addressing the wrong one does
    not fault. A memory-chiplet address on a config with no memory chiplet is simply
    unmapped -- the loads return whatever the fabric gives back and the checks compare one
    piece of garbage against another.

    EVERY PORT NAMES ONE. A block whose engines read L1 -- which is all of them, since the
    GEMM and SIMD have no AXI port -- but which will fetch an operand from further out
    declares the level it is willing to be HANDED, and emits the load in its own build.
    Where it is willing to be handed more than one, it offers a realisation per level
    rather than a port that declines to say; `needs` is then the separate statement of
    what its engines read once the load has run.
    """

    L1 = "L1", "cluster TCDM. What the GEMM and SIMD can actually read"
    L2 = "L2", "the narrow SPM, where the descriptor list lives. Not an operand's home"
    L3 = "L3", "main memory (the host's wide SPM). Where a block's loads read from"
    L4 = "L4", "the memory-chiplet pool, off-die over the D2D link. Host iDMA reaches it"


# ======================================================================================
# The interface a block declares
# ======================================================================================

@dataclass(frozen=True)
class PortSpec:
    """One edge of a block's signature, before anything is bound to it.

    `layout` and `dtype` are part of the signature, not something fixed up afterwards: a
    block is INVOKED with the precision and layout it should consume and produce, the same
    way a function is invoked with argument types.

    `mem_level` and `cluster` say WHERE, and there is no "anywhere". Every tensor is in
    some memory at every point in the graph, so a port that declined to say which would be
    declaring less than it knows, and the gap would be filled by whatever happened to be
    bound. `mem_level` is therefore REQUIRED -- omitting it is a TypeError at graph-build
    time, before any node exists. A block that can genuinely read an operand from more
    than one level is a FAMILY, not a port with a hole: it offers one realisation per
    level (comm/variant.py), each stating a concrete level and pricing its own transfers,
    and the resolver picks the one the producer matches.

    `cluster` is the same statement one level down, and it has an answer for exactly one
    memory: L1 is per-cluster TCDM, everything else is reachable from every cluster. So an
    L1 port MUST name its cluster and a port anywhere else must NOT -- both halves are
    enforced below, because a missing cluster on an L1 port is a placement nobody decided
    and a cluster on an L3 port is a claim that means nothing and that the contract would
    then go on to compare.

    Naming it matters because a remote handle does not fault: a GEMM or SIMD kernel can
    address only its own cluster's TCDM, and pointed anywhere else the transfer completes
    without writing and the buffer keeps whatever it held.
    """
    layout: Layout
    dtype: DType
    shape: tuple
    mem_level: MemLevel                     # required: every tensor is somewhere
    cluster: Optional[int] = None           # required for L1, refused for anything else
    doc: str = ""

    def __post_init__(self):
        # COERCE, do not merely check: a plain "A" surviving as a str would make
        # spec.layout.doc an AttributeError on one path and not another, depending on how
        # the spec happened to be built. Every spec holds the enum member however it was
        # written, and an unknown value is refused with the valid ones named.
        object.__setattr__(self, "layout", Layout(self.layout))
        object.__setattr__(self, "dtype", DType(self.dtype))
        if self.mem_level is None:
            raise ValueError(
                f"PortSpec({self.layout}/{self.dtype} {tuple(self.shape)}): mem_level is "
                f"required. A tensor is in some memory at every point in the graph; a "
                f"block that reads an operand from more than one level offers one "
                f"realisation per level instead of leaving the port open.")
        object.__setattr__(self, "mem_level", MemLevel(self.mem_level))
        if self.mem_level == MemLevel.L1 and self.cluster is None:
            raise ValueError(
                f"PortSpec({self.layout}/{self.dtype} {tuple(self.shape)} in L1): L1 is "
                f"per-cluster TCDM, so this port has to name the cluster it is in. Pass "
                f"cluster=<the block's own>.")
        if self.mem_level != MemLevel.L1 and self.cluster is not None:
            raise ValueError(
                f"PortSpec(... in {self.mem_level}, cluster={self.cluster}): only L1 is "
                f"per-cluster. {self.mem_level} is reachable from every cluster, so "
                f"naming one here states something that is not true of the buffer and "
                f"that check_contract would then compare against.")

    @property
    def stored_shape(self) -> tuple:
        """The dimensions the BYTES have, for the kernels that take a shape as an argument.

        DERIVED FROM THE LAYOUT, which is the point: there is one source of truth for byte
        order and this reads it. `shape` is the tensor's own (32 tokens of 128 features);
        this is how those bytes sit (col_major lays them out [128, 32]).

        MEANINGFUL ONLY FOR THE UNBLOCKED PAIR. A, B and D flatten a four-deep tiling, so
        "the dimensions the bytes have" is not a question with an answer for them and the
        (rows, cols) returned here is just the tensor's. Only the transposer kernel, which
        takes a literal M and N, actually consumes this.
        """
        r, c = self.shape
        return (c, r) if self.layout == Layout.COL_MAJOR else (r, c)

    def describe(self) -> str:
        where = f"in {self.mem_level}"
        if self.cluster is not None:
            where += f" on cl{self.cluster}"
        return f"{self.layout}/{self.dtype} {tuple(self.shape)} {where}"


@dataclass(frozen=True)
class Port:
    """A bound PortSpec: the buffer, and the nodes at the edge of the block.

    `ends` is what makes the join mechanical, and it is the only thing the caller would
    otherwise have to know. For an OUTPUT it is the nodes that finish writing the buffer;
    for an INPUT it is the nodes that first read it. The linker joins one block's output
    ends to the next block's input ends -- a real RAW dependency, nothing more.
    """
    spec: PortSpec
    handle: object                     # BingoMemAlloc | BingoMemAllocView | BingoMemSymbol
    ends: tuple
    name: str = ""

    def __post_init__(self):
        # THE SPEC ALREADY STATES WHERE, so binding is a CHECK and never a fill: whatever
        # the handle knows about itself has to agree with what the port claims. A spec
        # that says L1 on cluster 0 while holding a cluster-1 allocation is the exact lie
        # the contract exists to catch, and letting the claim stand would carry it to
        # every consumer downstream as fact.
        #
        # A FIXED ADDRESS knows neither -- it is a number -- so there the spec is the only
        # statement there is and nothing is checked against it.
        lvl, cl = _derive(self.handle)
        if lvl is not None and MemLevel(lvl) != self.spec.mem_level:
            raise ValueError(
                f"port {self.name or '?'}: the spec says {self.spec.mem_level} but "
                f"'{self.handle_name}' is in {lvl}. The handle is what the kernels "
                f"address, so restating the level differently would be believed by every "
                f"consumer of this port.")
        if lvl == MemLevel.L1 and cl != self.spec.cluster:
            raise ValueError(
                f"port {self.name or '?'}: the spec says cluster {self.spec.cluster} but "
                f"'{self.handle_name}' is allocated on cluster {cl}. A kernel addresses "
                f"only its own TCDM, and a remote handle does not fault -- it is written "
                f"by a transfer that moves nothing.")

    @property
    def layout(self): return self.spec.layout
    @property
    def dtype(self): return self.spec.dtype
    @property
    def shape(self): return self.spec.shape
    @property
    def stored_shape(self): return self.spec.stored_shape
    @property
    def cluster(self): return self.spec.cluster

    @property
    def handle_name(self) -> str:
        h = self.handle
        if isinstance(h, BingoMemAllocView):
            return h.base.name
        return getattr(h, "name", getattr(h, "symbol_name", "?"))


def level_of(handle) -> str:
    """Which memory level a handle addresses. Derived, never guessed.

    An allocation carries its level; a view carries its base's. A SYMBOL is a C array in
    the workload image, which is main memory by construction. A FIXED ADDRESS cannot be
    derived -- it is a number, and nothing in the handle records which pool it points at --
    so it is refused rather than guessed: assuming would put a hoist in front of an operand
    that may already be in main memory.
    """
    if isinstance(handle, BingoMemAllocView):
        return handle.base.mem_level
    if isinstance(handle, BingoMemAlloc):
        return handle.mem_level
    if isinstance(handle, BingoMemSymbol):
        return "L3"
    raise ValueError(
        f"cannot tell which memory level a {type(handle).__name__} addresses, so the "
        f"PortSpec has to say. A fixed address is just a number: the staging helper emits "
        f"one for the memory-chiplet pool, but the handle does not record that, and "
        f"assuming it would put a hoist in front of an operand that may already be in "
        f"main memory.")


def _derive(handle):
    """(level, cluster) off a handle, or None for whichever cannot be told.

    A FIXED ADDRESS is just a number and records neither, which is why `level_of` refuses
    it; here that refusal means "nothing to check against", not an error, because a spec
    stating the level is the only way such a handle can be used at all.
    """
    try:
        lvl = level_of(handle)
    except ValueError:
        lvl = None
    return lvl, cluster_of(handle)


def cluster_of(handle) -> Optional[int]:
    """Which cluster's TCDM a handle addresses, or None if the question does not apply.

    ONLY L1 IS PER-CLUSTER. Main memory and the narrow SPM are reachable from every
    cluster, so "which cluster" has no answer there and None is the answer, not a guess.
    BingoMemAlloc carries a cluster_id whatever its level, so reading that field alone
    would report cluster 0 for every L3 buffer in the tree.

    A SYMBOL or a FIXED ADDRESS is not TCDM: a symbol is a C array in the workload image
    and a fixed address is a number. Both return None.
    """
    if isinstance(handle, BingoMemAllocView):
        return cluster_of(handle.base)
    if isinstance(handle, BingoMemAlloc):
        return handle.cluster_id if handle.mem_level == "L1" else None
    return None


def at_offset(handle, nbytes: int):
    """The same buffer, `nbytes` further in. No new allocation, whatever the handle is.

    A block that slices one operand per cluster needs this and cannot use `.view()`: only
    BingoMemAlloc has that method, and a staged array arrives as a SYMBOL on the host path
    or a FIXED ADDRESS on the memory-chiplet path. All three have to offset the same way,
    or the slicing works on one platform and silently reads cluster 0's tile on the other.
    """
    if not nbytes:
        return handle
    if isinstance(handle, BingoMemAlloc):
        return handle.view(nbytes)
    if isinstance(handle, BingoMemAllocView):
        return BingoMemAllocView(handle.base, handle.offset + nbytes)
    if isinstance(handle, BingoMemSymbol):
        return BingoMemSymbol(handle.symbol_name, handle.offset + nbytes)
    if isinstance(handle, BingoMemFixedAddr):
        return BingoMemFixedAddr(handle.address + nbytes)
    raise TypeError(f"cannot offset a {type(handle).__name__}; add it to at_offset().")


@dataclass
class BlockResult:
    """What a block hands back: its ports, its nodes, and its graph sources.

    `sources` are the nodes with no predecessor -- typically weight loads. They are called
    out because they are the one thing a data edge cannot order: nothing upstream is their
    ancestor, so their buffers can never reuse an earlier block's L1 until something gates
    them. See Pipeline(gate_sources=...).
    """
    outputs: dict
    inputs: dict = field(default_factory=dict)
    nodes: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class Block:
    """A sub-DFG with a declared interface.

    Subclasses state `inputs` / `outputs` as {name: PortSpec} and implement `build`, which
    appends nodes into the caller's DFG and returns a BlockResult. A block is built exactly
    once, in pipeline order, because node creation order is dispatch order.

    `inputs` is the CONTRACT -- what a caller must supply. `needs` is what this block's own
    transfers read from, and defaults to `inputs`. Overriding it is how a block says it
    FETCHES: the contract is then checked against `needs`, so any gap between what was
    bound and what the engines want is the block's to close in its own build().

    That the two are separate is what makes the linker's rule enforceable. The LINKER never
    inserts a node -- node creation order is dispatch order -- so a mismatch is only
    closable by the consuming block, where the nodes land in its own order. A block that
    does not override `needs` and is handed a D-layout operand would read it as A-layout
    and compute a scrambled answer that no check catches, so the contract refuses that
    binding outright.
    """
    name = "block"

    @property
    def inputs(self) -> dict:
        raise NotImplementedError

    @property
    def needs(self) -> dict:
        """The layout, precision and level this block's own transfers read from.

        Same as `inputs` unless the block fetches its operands, in which case it overrides
        this with the concrete requirement and hands it to transfer.bring_in.
        """
        return self.inputs

    @property
    def outputs(self) -> dict:
        raise NotImplementedError

    def build(self, ctx: "Ctx", bound: dict) -> BlockResult:
        raise NotImplementedError

    # ---- the family this block belongs to, for the resolver in link.py -----------------

    def variants(self) -> list:
        """The parameter dicts this block could equally well be realised with.

        DEFAULT: just as constructed. A block with a pinned signature has nothing to
        choose between, so it offers one variant and the resolver has no decision to make
        -- which is exactly right, and is why no existing block needs changing.

        A block that owns knobs -- a layout the caller left unset, a kernel it may pick
        between -- overrides this and returns one dict per realisation. It does NOT have
        to filter them: an illegal one raises in __init__ and the resolver drops it, so
        the rules live in the constructor and nowhere else.
        """
        return [{}]

    def respec(self, **params) -> "Block":
        """A sibling of this block with some configuration fields changed.

        Defaults to rebuilding from this block's own cfg, which every block in the library
        carries as a frozen dataclass. A block whose configuration is not one dataclass
        overrides this; nothing else about it has to change.
        """
        if not params:
            return self
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            raise NotImplementedError(
                f"{type(self).__name__}.variants() offers {sorted(params)} but the block "
                f"has no `cfg` to respec from. Give it one, or override respec().")
        return type(self)(cfg=replace(cfg, **params))


# ======================================================================================
# The factory a block builds through
