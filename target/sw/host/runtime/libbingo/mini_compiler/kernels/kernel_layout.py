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


class _SnaxBingoKernelXdmaDToRowMajorBase(BingoKernelArgs):
    """D-layout -> row-major. D[m,n,r,c] -> R[m*meshRow+r, n*meshCol+c].

    Subclasses bind MESH_1 / MESH_2 (the two mesh dims their block spans) and ELEM_BYTES."""
    KERNEL_NAME: str = None
    MESH_1: int = None
    MESH_2: int = None
    ELEM_BYTES: int = None

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, N_T: int):
        if type(self).KERNEL_NAME is None:
            raise TypeError(f"{type(self).__name__} binds no mesh; instantiate a per-(shape, "
                            f"elem_bytes) subclass, not the base")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.M_T = M_T
        self.N_T = N_T
        # exposed for cost/debug tooling; NOT emitted -- they are part of the kernel, not the args
        self.mesh = (type(self).MESH_1, type(self).MESH_2)
        self.elem_bytes = type(self).ELEM_BYTES

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_d_to_row_major_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T)
        a["N_T"] = str(self.N_T)
        return a


# meshRow=32, meshCol=32, elem_bytes=1
class SnaxBingoKernelXdmaDToRowMajorE1M32N32Args(_SnaxBingoKernelXdmaDToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_d_to_row_major_e1_M32N32"
    MESH_1 = 32
    MESH_2 = 32
    ELEM_BYTES = 1


# meshRow=32, meshCol=32, elem_bytes=2
class SnaxBingoKernelXdmaDToRowMajorE2M32N32Args(_SnaxBingoKernelXdmaDToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_d_to_row_major_e2_M32N32"
    MESH_1 = 32
    MESH_2 = 32
    ELEM_BYTES = 2


# meshRow=32, meshCol=32, elem_bytes=4
class SnaxBingoKernelXdmaDToRowMajorE4M32N32Args(_SnaxBingoKernelXdmaDToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_d_to_row_major_e4_M32N32"
    MESH_1 = 32
    MESH_2 = 32
    ELEM_BYTES = 4


# meshRow=1, meshCol=32, elem_bytes=1
class SnaxBingoKernelXdmaDToRowMajorE1M1N32Args(_SnaxBingoKernelXdmaDToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_d_to_row_major_e1_M1N32"
    MESH_1 = 1
    MESH_2 = 32
    ELEM_BYTES = 1


# meshRow=1, meshCol=32, elem_bytes=2
class SnaxBingoKernelXdmaDToRowMajorE2M1N32Args(_SnaxBingoKernelXdmaDToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_d_to_row_major_e2_M1N32"
    MESH_1 = 1
    MESH_2 = 32
    ELEM_BYTES = 2


# meshRow=1, meshCol=32, elem_bytes=4
class SnaxBingoKernelXdmaDToRowMajorE4M1N32Args(_SnaxBingoKernelXdmaDToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_d_to_row_major_e4_M1N32"
    MESH_1 = 1
    MESH_2 = 32
    ELEM_BYTES = 4


# meshRow=16, meshCol=16, elem_bytes=1
class SnaxBingoKernelXdmaDToRowMajorE1M16N16Args(_SnaxBingoKernelXdmaDToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_d_to_row_major_e1_M16N16"
    MESH_1 = 16
    MESH_2 = 16
    ELEM_BYTES = 1


# meshRow=16, meshCol=16, elem_bytes=2
class SnaxBingoKernelXdmaDToRowMajorE2M16N16Args(_SnaxBingoKernelXdmaDToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_d_to_row_major_e2_M16N16"
    MESH_1 = 16
    MESH_2 = 16
    ELEM_BYTES = 2


# meshRow=16, meshCol=16, elem_bytes=4
class SnaxBingoKernelXdmaDToRowMajorE4M16N16Args(_SnaxBingoKernelXdmaDToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_d_to_row_major_e4_M16N16"
    MESH_1 = 16
    MESH_2 = 16
    ELEM_BYTES = 4


class _SnaxBingoKernelXdmaRowMajorToABase(BingoKernelArgs):
    """row-major -> A-layout. R[i,j] -> A[i/meshRow, j/tileSize, i%meshRow, j%tileSize].

    Subclasses bind MESH_1 / MESH_2 (the two mesh dims their block spans) and ELEM_BYTES."""
    KERNEL_NAME: str = None
    MESH_1: int = None
    MESH_2: int = None
    ELEM_BYTES: int = None

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, K_T: int):
        if type(self).KERNEL_NAME is None:
            raise TypeError(f"{type(self).__name__} binds no mesh; instantiate a per-(shape, "
                            f"elem_bytes) subclass, not the base")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.M_T = M_T
        self.K_T = K_T
        # exposed for cost/debug tooling; NOT emitted -- they are part of the kernel, not the args
        self.mesh = (type(self).MESH_1, type(self).MESH_2)
        self.elem_bytes = type(self).ELEM_BYTES

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_row_major_to_a_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T)
        a["K_T"] = str(self.K_T)
        return a


# meshRow=32, tileSize=2, elem_bytes=1
class SnaxBingoKernelXdmaRowMajorToAE1M32K2Args(_SnaxBingoKernelXdmaRowMajorToABase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_a_e1_M32K2"
    MESH_1 = 32
    MESH_2 = 2
    ELEM_BYTES = 1


# meshRow=32, tileSize=2, elem_bytes=2
class SnaxBingoKernelXdmaRowMajorToAE2M32K2Args(_SnaxBingoKernelXdmaRowMajorToABase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_a_e2_M32K2"
    MESH_1 = 32
    MESH_2 = 2
    ELEM_BYTES = 2


# meshRow=32, tileSize=2, elem_bytes=4
class SnaxBingoKernelXdmaRowMajorToAE4M32K2Args(_SnaxBingoKernelXdmaRowMajorToABase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_a_e4_M32K2"
    MESH_1 = 32
    MESH_2 = 2
    ELEM_BYTES = 4


# meshRow=1, tileSize=16, elem_bytes=1
class SnaxBingoKernelXdmaRowMajorToAE1M1K16Args(_SnaxBingoKernelXdmaRowMajorToABase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_a_e1_M1K16"
    MESH_1 = 1
    MESH_2 = 16
    ELEM_BYTES = 1


# meshRow=1, tileSize=16, elem_bytes=2
class SnaxBingoKernelXdmaRowMajorToAE2M1K16Args(_SnaxBingoKernelXdmaRowMajorToABase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_a_e2_M1K16"
    MESH_1 = 1
    MESH_2 = 16
    ELEM_BYTES = 2


# meshRow=1, tileSize=16, elem_bytes=4
class SnaxBingoKernelXdmaRowMajorToAE4M1K16Args(_SnaxBingoKernelXdmaRowMajorToABase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_a_e4_M1K16"
    MESH_1 = 1
    MESH_2 = 16
    ELEM_BYTES = 4


# meshRow=16, tileSize=8, elem_bytes=1
class SnaxBingoKernelXdmaRowMajorToAE1M16K8Args(_SnaxBingoKernelXdmaRowMajorToABase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_a_e1_M16K8"
    MESH_1 = 16
    MESH_2 = 8
    ELEM_BYTES = 1


# meshRow=16, tileSize=8, elem_bytes=2
class SnaxBingoKernelXdmaRowMajorToAE2M16K8Args(_SnaxBingoKernelXdmaRowMajorToABase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_a_e2_M16K8"
    MESH_1 = 16
    MESH_2 = 8
    ELEM_BYTES = 2


# meshRow=16, tileSize=8, elem_bytes=4
class SnaxBingoKernelXdmaRowMajorToAE4M16K8Args(_SnaxBingoKernelXdmaRowMajorToABase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_a_e4_M16K8"
    MESH_1 = 16
    MESH_2 = 8
    ELEM_BYTES = 4


class _SnaxBingoKernelXdmaRowMajorToBBase(BingoKernelArgs):
    """row-major -> B-layout. R[i,j] -> B[j/meshCol, i/tileSize, j%meshCol, i%tileSize].

    Subclasses bind MESH_1 / MESH_2 (the two mesh dims their block spans) and ELEM_BYTES."""
    KERNEL_NAME: str = None
    MESH_1: int = None
    MESH_2: int = None
    ELEM_BYTES: int = None

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 K_T: int, N_T: int):
        if type(self).KERNEL_NAME is None:
            raise TypeError(f"{type(self).__name__} binds no mesh; instantiate a per-(shape, "
                            f"elem_bytes) subclass, not the base")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.K_T = K_T
        self.N_T = N_T
        # exposed for cost/debug tooling; NOT emitted -- they are part of the kernel, not the args
        self.mesh = (type(self).MESH_1, type(self).MESH_2)
        self.elem_bytes = type(self).ELEM_BYTES

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_row_major_to_b_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["K_T"] = str(self.K_T)
        a["N_T"] = str(self.N_T)
        return a


# tileSize=2, meshCol=32, elem_bytes=1
class SnaxBingoKernelXdmaRowMajorToBE1K2N32Args(_SnaxBingoKernelXdmaRowMajorToBBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_b_e1_K2N32"
    MESH_1 = 2
    MESH_2 = 32
    ELEM_BYTES = 1


# tileSize=2, meshCol=32, elem_bytes=2
class SnaxBingoKernelXdmaRowMajorToBE2K2N32Args(_SnaxBingoKernelXdmaRowMajorToBBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_b_e2_K2N32"
    MESH_1 = 2
    MESH_2 = 32
    ELEM_BYTES = 2


# tileSize=2, meshCol=32, elem_bytes=4
class SnaxBingoKernelXdmaRowMajorToBE4K2N32Args(_SnaxBingoKernelXdmaRowMajorToBBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_b_e4_K2N32"
    MESH_1 = 2
    MESH_2 = 32
    ELEM_BYTES = 4


# tileSize=16, meshCol=32, elem_bytes=1
class SnaxBingoKernelXdmaRowMajorToBE1K16N32Args(_SnaxBingoKernelXdmaRowMajorToBBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_b_e1_K16N32"
    MESH_1 = 16
    MESH_2 = 32
    ELEM_BYTES = 1


# tileSize=16, meshCol=32, elem_bytes=2
class SnaxBingoKernelXdmaRowMajorToBE2K16N32Args(_SnaxBingoKernelXdmaRowMajorToBBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_b_e2_K16N32"
    MESH_1 = 16
    MESH_2 = 32
    ELEM_BYTES = 2


# tileSize=16, meshCol=32, elem_bytes=4
class SnaxBingoKernelXdmaRowMajorToBE4K16N32Args(_SnaxBingoKernelXdmaRowMajorToBBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_b_e4_K16N32"
    MESH_1 = 16
    MESH_2 = 32
    ELEM_BYTES = 4


# tileSize=8, meshCol=16, elem_bytes=1
class SnaxBingoKernelXdmaRowMajorToBE1K8N16Args(_SnaxBingoKernelXdmaRowMajorToBBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_b_e1_K8N16"
    MESH_1 = 8
    MESH_2 = 16
    ELEM_BYTES = 1


# tileSize=8, meshCol=16, elem_bytes=2
class SnaxBingoKernelXdmaRowMajorToBE2K8N16Args(_SnaxBingoKernelXdmaRowMajorToBBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_b_e2_K8N16"
    MESH_1 = 8
    MESH_2 = 16
    ELEM_BYTES = 2


# tileSize=8, meshCol=16, elem_bytes=4
class SnaxBingoKernelXdmaRowMajorToBE4K8N16Args(_SnaxBingoKernelXdmaRowMajorToBBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_b_e4_K8N16"
    MESH_1 = 8
    MESH_2 = 16
    ELEM_BYTES = 4


class _SnaxBingoKernelXdmaAToRowMajorBase(BingoKernelArgs):
    """A-layout -> row-major (the inverse of row_major_to_a).

    Subclasses bind MESH_1 / MESH_2 (the two mesh dims their block spans) and ELEM_BYTES."""
    KERNEL_NAME: str = None
    MESH_1: int = None
    MESH_2: int = None
    ELEM_BYTES: int = None

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, K_T: int):
        if type(self).KERNEL_NAME is None:
            raise TypeError(f"{type(self).__name__} binds no mesh; instantiate a per-(shape, "
                            f"elem_bytes) subclass, not the base")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.M_T = M_T
        self.K_T = K_T
        # exposed for cost/debug tooling; NOT emitted -- they are part of the kernel, not the args
        self.mesh = (type(self).MESH_1, type(self).MESH_2)
        self.elem_bytes = type(self).ELEM_BYTES

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_a_to_row_major_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T)
        a["K_T"] = str(self.K_T)
        return a


# meshRow=32, tileSize=2, elem_bytes=1
class SnaxBingoKernelXdmaAToRowMajorE1M32K2Args(_SnaxBingoKernelXdmaAToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_a_to_row_major_e1_M32K2"
    MESH_1 = 32
    MESH_2 = 2
    ELEM_BYTES = 1


# meshRow=32, tileSize=2, elem_bytes=2
class SnaxBingoKernelXdmaAToRowMajorE2M32K2Args(_SnaxBingoKernelXdmaAToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_a_to_row_major_e2_M32K2"
    MESH_1 = 32
    MESH_2 = 2
    ELEM_BYTES = 2


# meshRow=32, tileSize=2, elem_bytes=4
class SnaxBingoKernelXdmaAToRowMajorE4M32K2Args(_SnaxBingoKernelXdmaAToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_a_to_row_major_e4_M32K2"
    MESH_1 = 32
    MESH_2 = 2
    ELEM_BYTES = 4


# meshRow=1, tileSize=16, elem_bytes=1
class SnaxBingoKernelXdmaAToRowMajorE1M1K16Args(_SnaxBingoKernelXdmaAToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_a_to_row_major_e1_M1K16"
    MESH_1 = 1
    MESH_2 = 16
    ELEM_BYTES = 1


# meshRow=1, tileSize=16, elem_bytes=2
class SnaxBingoKernelXdmaAToRowMajorE2M1K16Args(_SnaxBingoKernelXdmaAToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_a_to_row_major_e2_M1K16"
    MESH_1 = 1
    MESH_2 = 16
    ELEM_BYTES = 2


# meshRow=1, tileSize=16, elem_bytes=4
class SnaxBingoKernelXdmaAToRowMajorE4M1K16Args(_SnaxBingoKernelXdmaAToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_a_to_row_major_e4_M1K16"
    MESH_1 = 1
    MESH_2 = 16
    ELEM_BYTES = 4


# meshRow=16, tileSize=8, elem_bytes=1
class SnaxBingoKernelXdmaAToRowMajorE1M16K8Args(_SnaxBingoKernelXdmaAToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_a_to_row_major_e1_M16K8"
    MESH_1 = 16
    MESH_2 = 8
    ELEM_BYTES = 1


# meshRow=16, tileSize=8, elem_bytes=2
class SnaxBingoKernelXdmaAToRowMajorE2M16K8Args(_SnaxBingoKernelXdmaAToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_a_to_row_major_e2_M16K8"
    MESH_1 = 16
    MESH_2 = 8
    ELEM_BYTES = 2


# meshRow=16, tileSize=8, elem_bytes=4
class SnaxBingoKernelXdmaAToRowMajorE4M16K8Args(_SnaxBingoKernelXdmaAToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_a_to_row_major_e4_M16K8"
    MESH_1 = 16
    MESH_2 = 8
    ELEM_BYTES = 4


class _SnaxBingoKernelXdmaBToRowMajorBase(BingoKernelArgs):
    """B-layout -> row-major (the inverse of row_major_to_b).

    Subclasses bind MESH_1 / MESH_2 (the two mesh dims their block spans) and ELEM_BYTES."""
    KERNEL_NAME: str = None
    MESH_1: int = None
    MESH_2: int = None
    ELEM_BYTES: int = None

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 K_T: int, N_T: int):
        if type(self).KERNEL_NAME is None:
            raise TypeError(f"{type(self).__name__} binds no mesh; instantiate a per-(shape, "
                            f"elem_bytes) subclass, not the base")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.K_T = K_T
        self.N_T = N_T
        # exposed for cost/debug tooling; NOT emitted -- they are part of the kernel, not the args
        self.mesh = (type(self).MESH_1, type(self).MESH_2)
        self.elem_bytes = type(self).ELEM_BYTES

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_b_to_row_major_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["K_T"] = str(self.K_T)
        a["N_T"] = str(self.N_T)
        return a


# tileSize=2, meshCol=32, elem_bytes=1
class SnaxBingoKernelXdmaBToRowMajorE1K2N32Args(_SnaxBingoKernelXdmaBToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_b_to_row_major_e1_K2N32"
    MESH_1 = 2
    MESH_2 = 32
    ELEM_BYTES = 1


# tileSize=2, meshCol=32, elem_bytes=2
class SnaxBingoKernelXdmaBToRowMajorE2K2N32Args(_SnaxBingoKernelXdmaBToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_b_to_row_major_e2_K2N32"
    MESH_1 = 2
    MESH_2 = 32
    ELEM_BYTES = 2


# tileSize=2, meshCol=32, elem_bytes=4
class SnaxBingoKernelXdmaBToRowMajorE4K2N32Args(_SnaxBingoKernelXdmaBToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_b_to_row_major_e4_K2N32"
    MESH_1 = 2
    MESH_2 = 32
    ELEM_BYTES = 4


# tileSize=16, meshCol=32, elem_bytes=1
class SnaxBingoKernelXdmaBToRowMajorE1K16N32Args(_SnaxBingoKernelXdmaBToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_b_to_row_major_e1_K16N32"
    MESH_1 = 16
    MESH_2 = 32
    ELEM_BYTES = 1


# tileSize=16, meshCol=32, elem_bytes=2
class SnaxBingoKernelXdmaBToRowMajorE2K16N32Args(_SnaxBingoKernelXdmaBToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_b_to_row_major_e2_K16N32"
    MESH_1 = 16
    MESH_2 = 32
    ELEM_BYTES = 2


# tileSize=16, meshCol=32, elem_bytes=4
class SnaxBingoKernelXdmaBToRowMajorE4K16N32Args(_SnaxBingoKernelXdmaBToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_b_to_row_major_e4_K16N32"
    MESH_1 = 16
    MESH_2 = 32
    ELEM_BYTES = 4


# tileSize=8, meshCol=16, elem_bytes=1
class SnaxBingoKernelXdmaBToRowMajorE1K8N16Args(_SnaxBingoKernelXdmaBToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_b_to_row_major_e1_K8N16"
    MESH_1 = 8
    MESH_2 = 16
    ELEM_BYTES = 1


# tileSize=8, meshCol=16, elem_bytes=2
class SnaxBingoKernelXdmaBToRowMajorE2K8N16Args(_SnaxBingoKernelXdmaBToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_b_to_row_major_e2_K8N16"
    MESH_1 = 8
    MESH_2 = 16
    ELEM_BYTES = 2


# tileSize=8, meshCol=16, elem_bytes=4
class SnaxBingoKernelXdmaBToRowMajorE4K8N16Args(_SnaxBingoKernelXdmaBToRowMajorBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_b_to_row_major_e4_K8N16"
    MESH_1 = 8
    MESH_2 = 16
    ELEM_BYTES = 4


class _SnaxBingoKernelXdmaRowMajorToDBase(BingoKernelArgs):
    """row-major -> D-layout (the inverse of d_to_row_major).

    Subclasses bind MESH_1 / MESH_2 (the two mesh dims their block spans) and ELEM_BYTES."""
    KERNEL_NAME: str = None
    MESH_1: int = None
    MESH_2: int = None
    ELEM_BYTES: int = None

    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, N_T: int):
        if type(self).KERNEL_NAME is None:
            raise TypeError(f"{type(self).__name__} binds no mesh; instantiate a per-(shape, "
                            f"elem_bytes) subclass, not the base")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.M_T = M_T
        self.N_T = N_T
        # exposed for cost/debug tooling; NOT emitted -- they are part of the kernel, not the args
        self.mesh = (type(self).MESH_1, type(self).MESH_2)
        self.elem_bytes = type(self).ELEM_BYTES

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_row_major_to_d_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T)
        a["N_T"] = str(self.N_T)
        return a


# meshRow=32, meshCol=32, elem_bytes=1
class SnaxBingoKernelXdmaRowMajorToDE1M32N32Args(_SnaxBingoKernelXdmaRowMajorToDBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_d_e1_M32N32"
    MESH_1 = 32
    MESH_2 = 32
    ELEM_BYTES = 1


# meshRow=32, meshCol=32, elem_bytes=2
class SnaxBingoKernelXdmaRowMajorToDE2M32N32Args(_SnaxBingoKernelXdmaRowMajorToDBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_d_e2_M32N32"
    MESH_1 = 32
    MESH_2 = 32
    ELEM_BYTES = 2


# meshRow=32, meshCol=32, elem_bytes=4
class SnaxBingoKernelXdmaRowMajorToDE4M32N32Args(_SnaxBingoKernelXdmaRowMajorToDBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_d_e4_M32N32"
    MESH_1 = 32
    MESH_2 = 32
    ELEM_BYTES = 4


# meshRow=1, meshCol=32, elem_bytes=1
class SnaxBingoKernelXdmaRowMajorToDE1M1N32Args(_SnaxBingoKernelXdmaRowMajorToDBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_d_e1_M1N32"
    MESH_1 = 1
    MESH_2 = 32
    ELEM_BYTES = 1


# meshRow=1, meshCol=32, elem_bytes=2
class SnaxBingoKernelXdmaRowMajorToDE2M1N32Args(_SnaxBingoKernelXdmaRowMajorToDBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_d_e2_M1N32"
    MESH_1 = 1
    MESH_2 = 32
    ELEM_BYTES = 2


# meshRow=1, meshCol=32, elem_bytes=4
class SnaxBingoKernelXdmaRowMajorToDE4M1N32Args(_SnaxBingoKernelXdmaRowMajorToDBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_d_e4_M1N32"
    MESH_1 = 1
    MESH_2 = 32
    ELEM_BYTES = 4


# meshRow=16, meshCol=16, elem_bytes=1
class SnaxBingoKernelXdmaRowMajorToDE1M16N16Args(_SnaxBingoKernelXdmaRowMajorToDBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_d_e1_M16N16"
    MESH_1 = 16
    MESH_2 = 16
    ELEM_BYTES = 1


# meshRow=16, meshCol=16, elem_bytes=2
class SnaxBingoKernelXdmaRowMajorToDE2M16N16Args(_SnaxBingoKernelXdmaRowMajorToDBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_d_e2_M16N16"
    MESH_1 = 16
    MESH_2 = 16
    ELEM_BYTES = 2


# meshRow=16, meshCol=16, elem_bytes=4
class SnaxBingoKernelXdmaRowMajorToDE4M16N16Args(_SnaxBingoKernelXdmaRowMajorToDBase):
    KERNEL_NAME = "__snax_bingo_kernel_xdma_row_major_to_d_e4_M16N16"
    MESH_1 = 16
    MESH_2 = 16
    ELEM_BYTES = 4

# -------------------------------------------------------------------------
# Resolve a converter args class from (family, mesh, elem_bytes). Hand-written
# workloads parameterize their mesh at runtime; this is how they pick the kernel.
# -------------------------------------------------------------------------
def xdma_conv_args(family: str, mesh_1: int, mesh_2: int, elem_bytes: int):
    """The args class for `family` bound to that mesh and element width, e.g.
    xdma_conv_args("xdma_row_major_to_a", 32, 2, 1) -> SnaxBingoKernelXdmaRowMajorToAE1M32K2Args.

    Raises LookupError if the RTL build has no such kernel -- the mesh must be one of the array
    shapes and elem_bytes one of the widths the device wrappers were generated for."""
    prefix = f"__snax_bingo_kernel_{family}_e{elem_bytes}_"
    for name, obj in globals().items():
        if not (isinstance(obj, type) and name.startswith("SnaxBingoKernelXdma") and name.endswith("Args")):
            continue
        kn = getattr(obj, "KERNEL_NAME", None)
        if kn and kn.startswith(prefix) and (obj.MESH_1, obj.MESH_2) == (mesh_1, mesh_2):
            return obj
    raise LookupError(f"no device kernel for {family} at mesh ({mesh_1}, {mesh_2}) with "
                      f"{elem_bytes}-byte elements")


