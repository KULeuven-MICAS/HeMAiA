# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""What a block DECLARES: its ports, and the vocabulary they are written in.

A port says four things about an operand -- its layout, its precision, its shape and the
memory it lives in -- and all four are part of the signature rather than an assumption,
because none of them faults when it is wrong. A mismatched layout is a permutation that
computes a scrambled answer; a mismatched precision reads two elements as one; an address
in a memory pool the platform does not have is simply unmapped.

This module has no knowledge of HOW a gap gets closed -- that is comm.transfer and
comm.nest. What it does own is the VOCABULARY: the layout, precision and space names a
port is written in. The layouts are worked through with real index maps below LAYOUTS,
because "D-layout" is not self-explanatory and reading a buffer in the wrong one is the
failure this whole module exists to make impossible.
"""

from dataclasses import dataclass, field
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
#   packed                          A
#         c0 c1 c2 c3 c4 c5 c6 c7         c0 c1 c2 c3 c4 c5 c6 c7
#     r0   0  1  2  3  4  5  6  7     r0   0  1  8  9 16 17 24 25
#     r1   8  9 10 11 12 13 14 15     r1   2  3 10 11 18 19 26 27
#     r2  16 17 18 19 20 21 22 23     r2   4  5 12 13 20 21 28 29
#     r3  24 25 26 27 28 29 30 31     r3   6  7 14 15 22 23 30 31
#
#   D                               B
#         c0 c1 c2 c3 c4 c5 c6 c7         c0 c1 c2 c3 c4 c5 c6 c7
#     r0   0  1  2  3 16 17 18 19     r0   0  2  4  6 32 34 36 38
#     r1   4  5  6  7 20 21 22 23     r1   1  3  5  7 33 35 37 39
#     r2   8  9 10 11 24 25 26 27     r2   8 10 12 14 40 42 44 46
#     r3  12 13 14 15 28 29 30 31     r3   9 11 13 15 41 43 45 47
#
# "packed"  Plain ROW-MAJOR: [r][c] is at r*cols + c. The layout everything outside the
#           array uses -- a numpy array, a golden, a tensor from the host. It is the "no
#           layout" layout. (comm/nest.py derives each conversion directly; it does not
#           route through packed as an intermediate.)
#
# "A" "B" "D"
#           The array's BLOCKED layouts, one per port: A is operand A (m, k, r, s), B is
#           operand B (n, k, c, s), D is the output (m, n, r, c). Each interleaves a block
#           of the matrix so the array can stream it a tile at a time.
#
#           NOTE WHAT B DOES. In the table above, +1 along a row moves +2, but +1 DOWN A
#           COLUMN moves +1: B runs contiguously down columns while packed, A and D run
#           along rows. So every conversion into or out of B is a TRANSPOSE, not a reshape,
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
#           is, and calling it "packed" would invite a consumer to read it as rows.
LAYOUTS = ("A", "B", "D", "packed", "d32", "monoid")
DTYPES = ("i8", "f16", "i32", "f32")
# Where a buffer lives, and it is part of the signature for the same reason the layout is:
# addressing the wrong one does not fault. A memory-chiplet address on a config with no
# memory chiplet is simply unmapped -- the loads return whatever the fabric gives back and
# the checks compare one piece of garbage against another.
#   L1  cluster TCDM. What the GEMM and SIMD can actually read.
#   L2  the narrow SPM, where the descriptor list lives. Not an operand's home.
#   L3  main memory (the host's wide SPM). Where a block's loads read from.
#   L4  the memory-chiplet pool, off-die over the D2D link. Reachable by the host iDMA.
SPACES = ("L1", "L2", "L3", "L4")


# ======================================================================================
# The interface a block declares
# ======================================================================================

@dataclass(frozen=True)
class PortSpec:
    """One edge of a block's signature, before anything is bound to it.

    `layout` and `dtype` are part of the signature, not something fixed up afterwards: a
    block is INVOKED with the precision and layout it should consume and produce, the same
    way a function is invoked with argument types.
    """
    layout: str
    dtype: str
    shape: tuple
    space: str = "L1"                 # one of SPACES
    doc: str = ""

    def __post_init__(self):
        if self.layout not in LAYOUTS:
            raise ValueError(f"PortSpec layout {self.layout!r} not in {LAYOUTS}.")
        if self.dtype not in DTYPES:
            raise ValueError(f"PortSpec dtype {self.dtype!r} not in {DTYPES}.")
        if self.space not in SPACES:
            raise ValueError(f"PortSpec space {self.space!r} not in {SPACES}.")

    def describe(self) -> str:
        return f"{self.layout}/{self.dtype} {tuple(self.shape)} in {self.space}"


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
    cluster: Optional[int] = None
    name: str = ""

    @property
    def layout(self): return self.spec.layout
    @property
    def dtype(self): return self.spec.dtype
    @property
    def shape(self): return self.spec.shape

    @property
    def handle_name(self) -> str:
        h = self.handle
        if isinstance(h, BingoMemAllocView):
            return h.base.name
        return getattr(h, "name", getattr(h, "symbol_name", "?"))


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

    `closes_gaps` says whether build() runs `transfer.bring_in` over its own inputs. It has
    to be declared rather than assumed: the LINKER never inserts a node (see the module
    docstring), so a mismatch is only closable if the CONSUMING BLOCK closes it, in its own
    build, where the nodes land in its own dispatch order. A block that leaves this False
    and is handed a D-layout operand would read it as A-layout and compute a scrambled
    answer that no check catches -- so the contract refuses that binding outright.
    """
    name = "block"
    closes_gaps = False

    @property
    def inputs(self) -> dict:
        raise NotImplementedError

    @property
    def outputs(self) -> dict:
        raise NotImplementedError

    def build(self, ctx: "Ctx", bound: dict) -> BlockResult:
        raise NotImplementedError


# ======================================================================================
# The factory a block builds through
