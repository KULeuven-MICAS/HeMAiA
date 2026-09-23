# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The xDMA layout converter: one kernel, one pair of layouts per call.

    row_major <-> A      R[i, j] -> A[i/meshRow, j/tileSize, i%meshRow, j%tileSize]
    row_major <-> B      R[i, j] -> B[j/meshCol, i/tileSize, j%meshCol, i%tileSize]
    row_major <-> D      R[i, j] -> D[i/meshRow, j/meshCol, i%meshRow, j%meshCol]

THE PAIR AND THE ARRAY SHAPE ARE ARGUMENTS, not part of the kernel's identity. The device
dispatches on them, and the transfer is the same AGU pass however it was reached. Binding
either to the symbol meant a new direction or a new tiling needed a new device symbol, and
a tiling nobody had pre-declared -- a (16, 4, 16) array wants M16K4 -- simply had none.

ROWS AND COLS, NOT TILE COUNTS. The caller gives the ROW-MAJOR tensor's dimensions and the
device derives the tile counts from them and the mesh, so a tile count cannot be paired
with the wrong mesh. Which dimension divides by what depends on the blocked layout -- A
reads [M, K], B reads [K, N], D reads [M, N] -- and that derivation now lives in one place
instead of at every call site.

ANYTHING TOUCHING B IS A TRANSPOSE, and that is a hardware fact rather than a quirk of the
kernel: row_major, A and D all run contiguously along rows, B runs down columns. The
device takes the 8x8 block transposer where the shape allows it and a DM-core element loop
otherwise -- two orders of magnitude apart, so libs/comm/transfer.py prefers to decompose
such a pair rather than emit it.

FOR A PAIR WITH NO ROW-MAJOR SIDE (A <-> D, A <-> B, anything with col_major), this kernel
is not the answer: it converts to and from row_major only. libs/comm/transfer.py
decomposes the rest through row_major.

======================================================================================
THE NEST IS DERIVED HERE, NOT ON THE DEVICE
======================================================================================

An AGU pass has to put the xDMA's 8 lanes on an axis whose bound is a multiple of 8 and
whose stride is constant on both sides, and WHICH axis that is depends on the shape -- it
is often not the innermost one. libs/comm/nest.py searches for it, derives the whole nest,
and then WALKS THE RESULT against both layouts' ground-truth index maps: a nest that is
wrong by one dimension still moves the right number of bytes, so a byte count cannot catch
it and a simulation would only show a scrambled tensor several kernels downstream.

So the conversion is derived once, here, and handed to __snax_bingo_kernel_xdma_6d as
strides. The device is not asked to rediscover it.

WHEN NO NEST EXISTS -- an A-layout atom is 4 B at int8 against the xDMA's 8 B lane, which
is why a quantise belongs after a reshape rather than before -- this falls back to
__snax_bingo_kernel_xdma_layout_convert, whose element loop is correct at any shape and
about two orders of magnitude slower. `.KERNEL_NAME` on the INSTANCE says which of the two
it became; the class attribute is only the common case."""

from typing import Dict, Union

from bingo_mem_handle import BingoMemAlloc
from kernel_base import LAYOUT_CODE, BingoKernelArgs

# Which mesh dimensions the two axes of the row-major tensor divide by, per blocked
# layout. This IS the derivation the device performs; it is repeated here only to refuse a
# shape that does not tile, while the caller can still fix it.
_DIVISORS = {"A": ("meshRow", "tileSize"),
             "B": ("tileSize", "meshCol"),
             "D": ("meshRow", "meshCol")}


class SnaxBingoKernelXdmaLayoutConvertArgs(BingoKernelArgs):
    """One layout conversion, between row_major and one of the array's blocked layouts.

    `rows` and `cols` are the ROW-MAJOR tensor's dimensions, whichever side of the
    conversion it is on. `mesh` is (meshRow, tileSize, meshCol).
    """

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 rows: int, cols: int, src_layout: str, dst_layout: str,
                 mesh: tuple, elem_bytes: int):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.rows = int(rows)
        self.cols = int(cols)
        self.src_layout = str(src_layout)
        self.dst_layout = str(dst_layout)
        self.mesh = tuple(int(v) for v in mesh)
        self.elem_bytes = int(elem_bytes)

        if elem_bytes not in (1, 2, 4):
            raise ValueError(f"elem_bytes={elem_bytes} must be 1, 2 or 4.")
        if len(self.mesh) != 3 or any(v <= 0 for v in self.mesh):
            raise ValueError(
                f"mesh={mesh} must be a positive (meshRow, tileSize, meshCol).")
        for nm, v in (("rows", self.rows), ("cols", self.cols)):
            if v <= 0:
                raise ValueError(f"xdma_layout_convert: {nm}={v} must be positive.")
        for who, lay in (("src_layout", self.src_layout),
                         ("dst_layout", self.dst_layout)):
            if lay not in LAYOUT_CODE:
                raise ValueError(
                    f"xdma_layout_convert: {who}={lay!r} is not one of "
                    f"{sorted(LAYOUT_CODE)}.")
        # EXACTLY ONE ROW-MAJOR SIDE. This kernel converts to and from row_major; a pair
        # without it (A <-> D) or with col_major is a different transfer, and the device
        # refuses it rather than picking an interpretation. Refused here too, so the
        # message names the pair while the graph is still being written.
        pair = (self.src_layout, self.dst_layout)
        blocked = [l for l in pair if l in _DIVISORS]
        if pair.count("row_major") != 1 or len(blocked) != 1:
            raise ValueError(
                f"xdma_layout_convert: {pair[0]} -> {pair[1]} needs exactly one "
                f"row_major side and one of {sorted(_DIVISORS)}. A pair without one is "
                f"not a single pass of this kernel -- libs/comm/transfer.py decomposes "
                f"it through row_major and emits the halves.")
        # A SHAPE THAT DOES NOT TILE writes a short buffer and leaves the tail
        # uninitialised, which reads as X rather than faulting.
        names = dict(zip(("meshRow", "tileSize", "meshCol"), self.mesh))
        d0, d1 = (names[n] for n in _DIVISORS[blocked[0]])
        if self.rows % d0 or self.cols % d1:
            raise ValueError(
                f"xdma_layout_convert: a {self.rows}x{self.cols} tensor does not tile "
                f"{blocked[0]}-layout on a {self.mesh} array: it needs rows % {d0} == 0 "
                f"and cols % {d1} == 0. A partial tile leaves the tail of the "
                f"destination unwritten, and unwritten TCDM reads X.")

        # DERIVE THE NEST NOW. Success means this call IS an xdma_6d, with strides that
        # have been checked element by element against both index maps; failure means no
        # pair of strides expresses the conversion and the element loop is the only route.
        # Imported here, not at module scope: libs.comm.nest imports the kernel args, so
        # a top-level import would close the cycle.
        from libs.comm.nest import convert_args as _convert_args
        try:
            self._nest = _convert_args(self.src_layout, self.dst_layout,
                                       self.rows, self.cols, self.mesh,
                                       self.elem_bytes, src_addr, dst_addr)
        except ValueError:
            self._nest = None

    @property
    def KERNEL_NAME(self) -> str:                                  # noqa: N802
        """Which kernel this call became: the derived nest, or the element loop."""
        return ("__snax_bingo_kernel_xdma_6d" if self._nest is not None
                else "__snax_bingo_kernel_xdma_layout_convert")

    def get_struct_name(self) -> str:
        if self._nest is not None:
            return self._nest.get_struct_name()
        return "__snax_bingo_kernel_xdma_layout_convert_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        if self._nest is not None:
            return self._nest.get_c_field_assignments(handle_name_map)
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["rows"] = str(self.rows)
        a["cols"] = str(self.cols)
        a["src_layout"] = str(LAYOUT_CODE[self.src_layout])
        a["dst_layout"] = str(LAYOUT_CODE[self.dst_layout])
        a["meshRow"] = str(self.mesh[0])
        a["tileSize"] = str(self.mesh[1])
        a["meshCol"] = str(self.mesh[2])
        a["elem_bytes"] = str(self.elem_bytes)
        return a


# The six DIRECTIONS the one kernel serves, by the family names the op sweeps and the
# workloads use. They are no longer separate classes -- the pair is a constructor argument
# -- but the names stay because a caller thinks in directions, not in layout pairs.
# KEYED BY THE OLD SYMBOL NAMES, which is what every caller already passes. They no
# longer name a device symbol -- there is one, __snax_bingo_kernel_xdma_layout_convert --
# but they still name a DIRECTION, and a direction is what a caller picks.
_CONV_FAMILIES = {
    "xdma_row_major_to_a": ("row_major", "A"),
    "xdma_a_to_row_major": ("A", "row_major"),
    "xdma_row_major_to_b": ("row_major", "B"),
    "xdma_b_to_row_major": ("B", "row_major"),
    "xdma_row_major_to_d": ("row_major", "D"),
    "xdma_d_to_row_major": ("D", "row_major"),
}


def xdma_conv_args(family: str):
    """A constructor for one converter family: (src, dst, rows, cols, mesh, elem_bytes).

    The MESH AND THE ELEMENT WIDTH are constructor arguments, and so now is the direction:
    one device symbol serves every tiling and every pair, so there is nothing left for a
    family to select but which two layouts to fill in. `rows` and `cols` are the ROW-MAJOR
    tensor's dimensions on whichever side it sits; the tile counts follow from the mesh.
    """
    try:
        src, dst = _CONV_FAMILIES[family]
    except KeyError:
        raise LookupError(
            f"no converter family {family!r}; the six are "
            f"{sorted(_CONV_FAMILIES)}.") from None

    def make(src_addr, dst_addr, rows, cols, mesh, elem_bytes):
        return SnaxBingoKernelXdmaLayoutConvertArgs(
            src_addr, dst_addr, rows, cols, src, dst, mesh, elem_bytes)
    make.__name__ = f"{family}_args"
    make.layouts = (src, dst)
    # The symbol is the same for every family now, but callers still read it off whatever
    # xdma_conv_args hands back, so it is carried here rather than made their problem.
    make.KERNEL_NAME = "__snax_bingo_kernel_xdma_6d"
    return make
