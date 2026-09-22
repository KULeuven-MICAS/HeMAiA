# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The xDMA layout converters: dedicated kernels for the common permutations.

Six families (D<->row-major, row-major<->A, row-major<->B), each in three element widths
and three array shapes. They are NOT interchangeable with a generic strided transfer: the
row-major<->B pair drives the hardware TRANSPOSER, which is correct only at elem_bytes=1
and silently returns wrong data at 2 and 4.

For a permutation with no dedicated kernel, derive a strided nest instead -- see
libs/comm/nest.py, which checks the strides it derives against ground-truth index maps."""

from typing import Union, Dict
from bingo_mem_handle import BingoMemAlloc

from kernel_base import BingoKernelArgs


class SnaxBingoKernelXdmaDToRowMajorArgs(BingoKernelArgs):
    """D-layout -> row-major. D[m,n,r,c] -> R[m*meshRow+r, n*meshCol+c].

    ONE CLASS, ANY ARRAY. `meshRow`, `meshCol` and `elem_bytes` are ARGUMENTS, not part of the
    kernel's identity: the device impl already selects its AGU path from them at run time,
    and binding them per kernel meant a tiling nobody had pre-declared had no symbol to
    call -- a (16, 4, 16) array wants M16K4, and no wrapper ever defined one.
    """

    KERNEL_NAME = "__snax_bingo_kernel_xdma_d_to_row_major"

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, N_T: int, meshRow: int, meshCol: int, elem_bytes: int):
        if elem_bytes not in (1, 2, 4):
            raise ValueError(f"elem_bytes={elem_bytes} must be 1, 2 or 4.")
        for nm, v in (("meshRow", meshRow), ("meshCol", meshCol), ("M_T", M_T), ("N_T", N_T)):
            if v <= 0:
                raise ValueError(f"XdmaDToRowMajor: {nm}={v} must be positive.")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.M_T = M_T
        self.N_T = N_T
        self.meshRow = meshRow
        self.meshCol = meshCol
        self.elem_bytes = elem_bytes
        self.mesh = (meshRow, meshCol)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_d_to_row_major_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T)
        a["N_T"] = str(self.N_T)
        a["meshRow"] = str(self.meshRow)
        a["meshCol"] = str(self.meshCol)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


class SnaxBingoKernelXdmaRowMajorToAArgs(BingoKernelArgs):
    """row-major -> A-layout. R[i,j] -> A[i/meshRow, j/tileSize, i%meshRow, j%tileSize].

    ONE CLASS, ANY ARRAY. `meshRow`, `tileSize` and `elem_bytes` are ARGUMENTS, not part of the
    kernel's identity: the device impl already selects its AGU path from them at run time,
    and binding them per kernel meant a tiling nobody had pre-declared had no symbol to
    call -- a (16, 4, 16) array wants M16K4, and no wrapper ever defined one.
    """

    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_a"

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, K_T: int, meshRow: int, tileSize: int, elem_bytes: int):
        if elem_bytes not in (1, 2, 4):
            raise ValueError(f"elem_bytes={elem_bytes} must be 1, 2 or 4.")
        for nm, v in (("meshRow", meshRow), ("tileSize", tileSize), ("M_T", M_T), ("K_T", K_T)):
            if v <= 0:
                raise ValueError(f"XdmaRowMajorToA: {nm}={v} must be positive.")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.M_T = M_T
        self.K_T = K_T
        self.meshRow = meshRow
        self.tileSize = tileSize
        self.elem_bytes = elem_bytes
        self.mesh = (meshRow, tileSize)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_row_major_to_a_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T)
        a["K_T"] = str(self.K_T)
        a["meshRow"] = str(self.meshRow)
        a["tileSize"] = str(self.tileSize)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


class SnaxBingoKernelXdmaRowMajorToBArgs(BingoKernelArgs):
    """row-major -> B-layout. Drives the HW TRANSPOSER: correct only at elem_bytes=1.

    ONE CLASS, ANY ARRAY. `tileSize`, `meshCol` and `elem_bytes` are ARGUMENTS, not part of the
    kernel's identity: the device impl already selects its AGU path from them at run time,
    and binding them per kernel meant a tiling nobody had pre-declared had no symbol to
    call -- a (16, 4, 16) array wants M16K4, and no wrapper ever defined one.
    """

    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_b"

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 K_T: int, N_T: int, tileSize: int, meshCol: int, elem_bytes: int):
        if elem_bytes not in (1, 2, 4):
            raise ValueError(f"elem_bytes={elem_bytes} must be 1, 2 or 4.")
        for nm, v in (("tileSize", tileSize), ("meshCol", meshCol), ("K_T", K_T), ("N_T", N_T)):
            if v <= 0:
                raise ValueError(f"XdmaRowMajorToB: {nm}={v} must be positive.")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.K_T = K_T
        self.N_T = N_T
        self.tileSize = tileSize
        self.meshCol = meshCol
        self.elem_bytes = elem_bytes
        self.mesh = (tileSize, meshCol)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_row_major_to_b_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["K_T"] = str(self.K_T)
        a["N_T"] = str(self.N_T)
        a["tileSize"] = str(self.tileSize)
        a["meshCol"] = str(self.meshCol)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


class SnaxBingoKernelXdmaAToRowMajorArgs(BingoKernelArgs):
    """A-layout -> row-major. The inverse of row_major_to_a.

    ONE CLASS, ANY ARRAY. `meshRow`, `tileSize` and `elem_bytes` are ARGUMENTS, not part of the
    kernel's identity: the device impl already selects its AGU path from them at run time,
    and binding them per kernel meant a tiling nobody had pre-declared had no symbol to
    call -- a (16, 4, 16) array wants M16K4, and no wrapper ever defined one.
    """

    KERNEL_NAME = "__snax_bingo_kernel_xdma_a_to_row_major"

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, K_T: int, meshRow: int, tileSize: int, elem_bytes: int):
        if elem_bytes not in (1, 2, 4):
            raise ValueError(f"elem_bytes={elem_bytes} must be 1, 2 or 4.")
        for nm, v in (("meshRow", meshRow), ("tileSize", tileSize), ("M_T", M_T), ("K_T", K_T)):
            if v <= 0:
                raise ValueError(f"XdmaAToRowMajor: {nm}={v} must be positive.")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.M_T = M_T
        self.K_T = K_T
        self.meshRow = meshRow
        self.tileSize = tileSize
        self.elem_bytes = elem_bytes
        self.mesh = (meshRow, tileSize)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_a_to_row_major_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T)
        a["K_T"] = str(self.K_T)
        a["meshRow"] = str(self.meshRow)
        a["tileSize"] = str(self.tileSize)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


class SnaxBingoKernelXdmaBToRowMajorArgs(BingoKernelArgs):
    """B-layout -> row-major. Drives the HW TRANSPOSER: correct only at elem_bytes=1.

    ONE CLASS, ANY ARRAY. `tileSize`, `meshCol` and `elem_bytes` are ARGUMENTS, not part of the
    kernel's identity: the device impl already selects its AGU path from them at run time,
    and binding them per kernel meant a tiling nobody had pre-declared had no symbol to
    call -- a (16, 4, 16) array wants M16K4, and no wrapper ever defined one.
    """

    KERNEL_NAME = "__snax_bingo_kernel_xdma_b_to_row_major"

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 K_T: int, N_T: int, tileSize: int, meshCol: int, elem_bytes: int):
        if elem_bytes not in (1, 2, 4):
            raise ValueError(f"elem_bytes={elem_bytes} must be 1, 2 or 4.")
        for nm, v in (("tileSize", tileSize), ("meshCol", meshCol), ("K_T", K_T), ("N_T", N_T)):
            if v <= 0:
                raise ValueError(f"XdmaBToRowMajor: {nm}={v} must be positive.")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.K_T = K_T
        self.N_T = N_T
        self.tileSize = tileSize
        self.meshCol = meshCol
        self.elem_bytes = elem_bytes
        self.mesh = (tileSize, meshCol)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_b_to_row_major_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["K_T"] = str(self.K_T)
        a["N_T"] = str(self.N_T)
        a["tileSize"] = str(self.tileSize)
        a["meshCol"] = str(self.meshCol)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


class SnaxBingoKernelXdmaRowMajorToDArgs(BingoKernelArgs):
    """row-major -> D-layout. The inverse of d_to_row_major.

    ONE CLASS, ANY ARRAY. `meshRow`, `meshCol` and `elem_bytes` are ARGUMENTS, not part of the
    kernel's identity: the device impl already selects its AGU path from them at run time,
    and binding them per kernel meant a tiling nobody had pre-declared had no symbol to
    call -- a (16, 4, 16) array wants M16K4, and no wrapper ever defined one.
    """

    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_d"

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, N_T: int, meshRow: int, meshCol: int, elem_bytes: int):
        if elem_bytes not in (1, 2, 4):
            raise ValueError(f"elem_bytes={elem_bytes} must be 1, 2 or 4.")
        for nm, v in (("meshRow", meshRow), ("meshCol", meshCol), ("M_T", M_T), ("N_T", N_T)):
            if v <= 0:
                raise ValueError(f"XdmaRowMajorToD: {nm}={v} must be positive.")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.M_T = M_T
        self.N_T = N_T
        self.meshRow = meshRow
        self.meshCol = meshCol
        self.elem_bytes = elem_bytes
        self.mesh = (meshRow, meshCol)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_row_major_to_d_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T)
        a["N_T"] = str(self.N_T)
        a["meshRow"] = str(self.meshRow)
        a["meshCol"] = str(self.meshCol)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


_CONV_FAMILIES = {
    "xdma_d_to_row_major":  SnaxBingoKernelXdmaDToRowMajorArgs,
    "xdma_row_major_to_a":  SnaxBingoKernelXdmaRowMajorToAArgs,
    "xdma_row_major_to_b":  SnaxBingoKernelXdmaRowMajorToBArgs,
    "xdma_a_to_row_major":  SnaxBingoKernelXdmaAToRowMajorArgs,
    "xdma_b_to_row_major":  SnaxBingoKernelXdmaBToRowMajorArgs,
    "xdma_row_major_to_d":  SnaxBingoKernelXdmaRowMajorToDArgs,
}


def xdma_conv_args(family: str):
    """The args class for a converter family.

    It no longer takes a mesh or an element width. Those are CONSTRUCTOR ARGUMENTS of the
    class it returns, because the device kernel takes them too -- one symbol serves every
    tiling. The old form raised LookupError for any array shape nobody had generated a
    wrapper for, which on a (16, 4, 16) array was every one of them.
    """
    try:
        return _CONV_FAMILIES[family]
    except KeyError:
        raise LookupError(
            f"no converter family {family!r}; the six are "
            f"{sorted(_CONV_FAMILIES)}.") from None
