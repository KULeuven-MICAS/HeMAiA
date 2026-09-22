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

This module has no knowledge of HOW a gap gets closed. That is libs.comm.
"""

from dataclasses import dataclass, field
from typing import Optional

from bingo_mem_handle import (BingoMemAlloc, BingoMemAllocView,
                              BingoMemFixedAddr, BingoMemSymbol)

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
