# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""VersaCore GEMM: the general descriptor, the minimal one, and the typed variants.

The typed classes are a cross product -- (A precision, B precision, D precision) x (array
shape) -- and each fixes a kernel NAME. They are spelled out rather than generated because
the name is the contract with the device-side SNAX_EXPORT_FUNC registry: a generated name
that does not exist there links fine and dispatches to nothing."""

from typing import Union, Dict
from bingo_mem_handle import BingoMemAlloc

from kernel_base import BingoKernelArgs


class SnaxBingoKernelGemmFullArgs(BingoKernelArgs):
    def __init__(self, 
                 input_A_addr: Union[BingoMemAlloc, int],
                 input_B_addr: Union[BingoMemAlloc, int],
                 input_C_addr: Union[BingoMemAlloc, int],
                 output_D_addr: Union[BingoMemAlloc, int],
                 M: int,
                 K: int,
                 N: int,
                 array_shape_idx: int,
                 transpose_A: int,
                 transpose_B: int,
                 accumPrevC: int,
                 quantization_enable: int = 0,
                 shift_i: int = 0,
                 multiplier_i: int = 0,
                 input_zp_i: int = 0,
                 output_zp_i: int = 0,
                 int32tofp16_enable: int = 0,
                 int4_a_enable: int = 0,
                 int4_b_enable: int = 0):
        self.input_A_addr = input_A_addr
        self.input_B_addr = input_B_addr
        self.input_C_addr = input_C_addr
        self.output_D_addr = output_D_addr
        self.M = M
        self.K = K
        self.N = N
        self.array_shape_idx = array_shape_idx
        self.transpose_A = transpose_A
        self.transpose_B = transpose_B
        self.accumPrevC = accumPrevC
        self.quantization_enable = quantization_enable
        self.shift_i = shift_i
        self.multiplier_i = multiplier_i
        self.input_zp_i = input_zp_i
        self.output_zp_i = output_zp_i
        self.int32tofp16_enable = int32tofp16_enable
        self.int4_a_enable = int4_a_enable
        self.int4_b_enable = int4_b_enable

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_gemm_full_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.input_A_addr, "input_A_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.input_B_addr, "input_B_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.input_C_addr, "input_C_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.output_D_addr, "output_D_addr", assignments, handle_name_map, split_64bit=False)
        assignments["M"] = str(self.M)
        assignments["K"] = str(self.K)
        assignments["N"] = str(self.N)
        assignments["array_shape_idx"] = str(self.array_shape_idx)
        assignments["transpose_A"] = str(self.transpose_A)
        assignments["transpose_B"] = str(self.transpose_B)
        assignments["accumPrevC"] = str(self.accumPrevC)
        assignments["quantization_enable"] = str(self.quantization_enable)
        assignments["shift_i"] = str(self.shift_i)
        assignments["multiplier_i"] = str(self.multiplier_i)
        assignments["input_zp_i"] = str(self.input_zp_i)
        assignments["output_zp_i"] = str(self.output_zp_i)
        assignments["int32tofp16_enable"] = str(self.int32tofp16_enable)
        assignments["int4_a_enable"] = str(self.int4_a_enable)
        assignments["int4_b_enable"] = str(self.int4_b_enable)
        return assignments

# BINGO GEMM MINIMAL
class SnaxBingoKernelGemmMinimalArgs(BingoKernelArgs):
    def __init__(self, 
                 input_A_addr: Union[BingoMemAlloc, int],
                 input_B_addr: Union[BingoMemAlloc, int],
                 input_C_addr: Union[BingoMemAlloc, int],
                 output_D_addr: Union[BingoMemAlloc, int],
                 ):
        self.input_A_addr = input_A_addr
        self.input_B_addr = input_B_addr
        self.input_C_addr = input_C_addr
        self.output_D_addr = output_D_addr

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_gemm_minimal_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.input_A_addr, "input_A_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.input_B_addr, "input_B_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.input_C_addr, "input_C_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.output_D_addr, "output_D_addr", assignments, handle_name_map, split_64bit=False)
        return assignments

# -------------------------------------------------------------------------
# GEMM args: ONE class per runnable kernel = (precision mode, runtime array shape).
#
# The C side takes both as inputs to one `__bingo_gemm_run`: the mode picks the
# wrapper (which bakes int4_a / int4_b / int32tofp16), and `array_shape_idx` is a
# CSR the wrapper writes to reconfigure the spatial unrolling. But array_shape_idx
# is NOT free -- it must match the mesh the operands were blocked for and the mesh
# the cost model priced. Baking it into the class (ARRAY_SHAPE_IDX) instead of
# taking it as a constructor argument makes that agreement structural: you cannot
# instantiate a M32K2N32 kernel and hand it a shape-1 mesh.
#
# The five plain modes share one C struct (__snax_bingo_kernel_gemm_args_t) and the
# quantized one adds requant params (__snax_bingo_kernel_gemm_quant_args_t), so the
# struct name cannot name the dispatcher -- each class names it via KERNEL_NAME.
# Every reuse (Minimal) class dispatches to the single __snax_bingo_kernel_gemm_minimal,
# which reprograms only base addresses and inherits whatever CSRs (mode AND shape) the
# preceding configure programmed; its class name records which those were.
# -------------------------------------------------------------------------
class _SnaxBingoKernelGemmPlainArgs(BingoKernelArgs):
    """Shared base for the plain GEMM wrappers (int32 / int4-packed / fp16 out, no
    requantization). Subclasses bind KERNEL_NAME (the C dispatcher) and
    ARRAY_SHAPE_IDX (the ARRAY_SHAPE_CFG value) and add no fields."""
    ARRAY_SHAPE_IDX: int = None          # bound by each (mode, shape) subclass

    def __init__(self,
                 input_A_addr: Union[BingoMemAlloc, int],
                 input_B_addr: Union[BingoMemAlloc, int],
                 input_C_addr: Union[BingoMemAlloc, int],
                 output_D_addr: Union[BingoMemAlloc, int],
                 M: int, K: int, N: int,
                 transpose_A: int = 0,
                 transpose_B: int = 0,
                 accumPrevC: int = 0):
        if type(self).ARRAY_SHAPE_IDX is None:
            raise TypeError(f"{type(self).__name__} binds no ARRAY_SHAPE_IDX; instantiate a "
                            f"per-shape subclass, not the base")
        self.input_A_addr = input_A_addr
        self.input_B_addr = input_B_addr
        self.input_C_addr = input_C_addr
        self.output_D_addr = output_D_addr
        self.M = M
        self.K = K
        self.N = N
        self.array_shape_idx = type(self).ARRAY_SHAPE_IDX
        self.transpose_A = transpose_A
        self.transpose_B = transpose_B
        self.accumPrevC = accumPrevC

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_gemm_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.input_A_addr, "input_A_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.input_B_addr, "input_B_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.input_C_addr, "input_C_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.output_D_addr, "output_D_addr", assignments, handle_name_map, split_64bit=False)
        assignments["M"] = str(self.M)
        assignments["K"] = str(self.K)
        assignments["N"] = str(self.N)
        assignments["array_shape_idx"] = str(self.array_shape_idx)
        assignments["transpose_A"] = str(self.transpose_A)
        assignments["transpose_B"] = str(self.transpose_B)
        assignments["accumPrevC"] = str(self.accumPrevC)
        return assignments


class _SnaxBingoKernelGemmQuantArgs(BingoKernelArgs):
    """Shared base for the requantizing GEMM wrapper (int8 x int8 -> int8): the plain
    fields plus the requant params (shift / multiplier / zero-points)."""
    ARRAY_SHAPE_IDX: int = None          # bound by each shape subclass

    def __init__(self,
                 input_A_addr: Union[BingoMemAlloc, int],
                 input_B_addr: Union[BingoMemAlloc, int],
                 input_C_addr: Union[BingoMemAlloc, int],
                 output_D_addr: Union[BingoMemAlloc, int],
                 M: int, K: int, N: int,
                 shift_i: int,
                 multiplier_i: int,
                 input_zp_i: int,
                 output_zp_i: int,
                 transpose_A: int = 0,
                 transpose_B: int = 0,
                 accumPrevC: int = 0):
        if type(self).ARRAY_SHAPE_IDX is None:
            raise TypeError(f"{type(self).__name__} binds no ARRAY_SHAPE_IDX; instantiate a "
                            f"per-shape subclass, not the base")
        self.input_A_addr = input_A_addr
        self.input_B_addr = input_B_addr
        self.input_C_addr = input_C_addr
        self.output_D_addr = output_D_addr
        self.M = M
        self.K = K
        self.N = N
        self.array_shape_idx = type(self).ARRAY_SHAPE_IDX
        self.shift_i = shift_i
        self.multiplier_i = multiplier_i
        self.input_zp_i = input_zp_i
        self.output_zp_i = output_zp_i
        self.transpose_A = transpose_A
        self.transpose_B = transpose_B
        self.accumPrevC = accumPrevC

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_gemm_quant_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.input_A_addr, "input_A_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.input_B_addr, "input_B_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.input_C_addr, "input_C_addr", assignments, handle_name_map, split_64bit=False)
        self._process_addr(self.output_D_addr, "output_D_addr", assignments, handle_name_map, split_64bit=False)
        assignments["M"] = str(self.M)
        assignments["K"] = str(self.K)
        assignments["N"] = str(self.N)
        assignments["array_shape_idx"] = str(self.array_shape_idx)
        assignments["transpose_A"] = str(self.transpose_A)
        assignments["transpose_B"] = str(self.transpose_B)
        assignments["accumPrevC"] = str(self.accumPrevC)
        assignments["shift_i"] = str(self.shift_i)
        assignments["multiplier_i"] = str(self.multiplier_i)
        assignments["input_zp_i"] = str(self.input_zp_i)
        assignments["output_zp_i"] = str(self.output_zp_i)
        return assignments


class _SnaxBingoKernelGemmMinimalBase(SnaxBingoKernelGemmMinimalArgs):
    """Shared base for the reuse kernels. All 18 dispatch to the SAME C function and
    emit the same fields -- they differ only in name, which records the (mode, shape)
    configure whose CSRs they reuse. That is what makes the framework's kernel id and
    this class 1:1, and what lets the cost model price a reuse at its real shape."""
    KERNEL_NAME = "__snax_bingo_kernel_gemm_minimal"
    ARRAY_SHAPE_IDX: int = None


# i8i8_i32 @ mesh (mu,ku,nu)=(32, 2, 32) -> op id gemm_i8i8_i32_M32K2N32
class SnaxBingoKernelGemmI8I8I32M32K2N32Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i8_i32"
    ARRAY_SHAPE_IDX = 0


# i8i8_i32 @ mesh (mu,ku,nu)=(1, 16, 32) -> op id gemm_i8i8_i32_M1K16N32
class SnaxBingoKernelGemmI8I8I32M1K16N32Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i8_i32"
    ARRAY_SHAPE_IDX = 1


# i8i8_i32 @ mesh (mu,ku,nu)=(16, 8, 16) -> op id gemm_i8i8_i32_M16K8N16
class SnaxBingoKernelGemmI8I8I32M16K8N16Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i8_i32"
    ARRAY_SHAPE_IDX = 2


# i8i4_i32 @ mesh (mu,ku,nu)=(32, 2, 32) -> op id gemm_i8i4_i32_M32K2N32
class SnaxBingoKernelGemmI8I4I32M32K2N32Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i4_i32"
    ARRAY_SHAPE_IDX = 0


# i8i4_i32 @ mesh (mu,ku,nu)=(1, 16, 32) -> op id gemm_i8i4_i32_M1K16N32
class SnaxBingoKernelGemmI8I4I32M1K16N32Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i4_i32"
    ARRAY_SHAPE_IDX = 1


# i8i4_i32 @ mesh (mu,ku,nu)=(16, 8, 16) -> op id gemm_i8i4_i32_M16K8N16
class SnaxBingoKernelGemmI8I4I32M16K8N16Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i4_i32"
    ARRAY_SHAPE_IDX = 2


# i4i4_i32 @ mesh (mu,ku,nu)=(32, 2, 32) -> op id gemm_i4i4_i32_M32K2N32
class SnaxBingoKernelGemmI4I4I32M32K2N32Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i4i4_i32"
    ARRAY_SHAPE_IDX = 0


# i4i4_i32 @ mesh (mu,ku,nu)=(1, 16, 32) -> op id gemm_i4i4_i32_M1K16N32
class SnaxBingoKernelGemmI4I4I32M1K16N32Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i4i4_i32"
    ARRAY_SHAPE_IDX = 1


# i4i4_i32 @ mesh (mu,ku,nu)=(16, 8, 16) -> op id gemm_i4i4_i32_M16K8N16
class SnaxBingoKernelGemmI4I4I32M16K8N16Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i4i4_i32"
    ARRAY_SHAPE_IDX = 2


# i8i8_i8 @ mesh (mu,ku,nu)=(32, 2, 32) -> op id gemm_i8i8_i8_M32K2N32
class SnaxBingoKernelGemmI8I8I8M32K2N32Args(_SnaxBingoKernelGemmQuantArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i8_i8"
    ARRAY_SHAPE_IDX = 0


# i8i8_i8 @ mesh (mu,ku,nu)=(1, 16, 32) -> op id gemm_i8i8_i8_M1K16N32
class SnaxBingoKernelGemmI8I8I8M1K16N32Args(_SnaxBingoKernelGemmQuantArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i8_i8"
    ARRAY_SHAPE_IDX = 1


# i8i8_i8 @ mesh (mu,ku,nu)=(16, 8, 16) -> op id gemm_i8i8_i8_M16K8N16
class SnaxBingoKernelGemmI8I8I8M16K8N16Args(_SnaxBingoKernelGemmQuantArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i8_i8"
    ARRAY_SHAPE_IDX = 2


# i8i4_f16 @ mesh (mu,ku,nu)=(32, 2, 32) -> op id gemm_i8i4_f16_M32K2N32
class SnaxBingoKernelGemmI8I4F16M32K2N32Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i4_f16"
    ARRAY_SHAPE_IDX = 0


# i8i4_f16 @ mesh (mu,ku,nu)=(1, 16, 32) -> op id gemm_i8i4_f16_M1K16N32
class SnaxBingoKernelGemmI8I4F16M1K16N32Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i4_f16"
    ARRAY_SHAPE_IDX = 1


# i8i4_f16 @ mesh (mu,ku,nu)=(16, 8, 16) -> op id gemm_i8i4_f16_M16K8N16
class SnaxBingoKernelGemmI8I4F16M16K8N16Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i4_f16"
    ARRAY_SHAPE_IDX = 2


# i8i8_f16 @ mesh (mu,ku,nu)=(32, 2, 32) -> op id gemm_i8i8_f16_M32K2N32
class SnaxBingoKernelGemmI8I8F16M32K2N32Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i8_f16"
    ARRAY_SHAPE_IDX = 0


# i8i8_f16 @ mesh (mu,ku,nu)=(1, 16, 32) -> op id gemm_i8i8_f16_M1K16N32
class SnaxBingoKernelGemmI8I8F16M1K16N32Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i8_f16"
    ARRAY_SHAPE_IDX = 1


# i8i8_f16 @ mesh (mu,ku,nu)=(16, 8, 16) -> op id gemm_i8i8_f16_M16K8N16
class SnaxBingoKernelGemmI8I8F16M16K8N16Args(_SnaxBingoKernelGemmPlainArgs):
    KERNEL_NAME = "__snax_bingo_kernel_gemm_i8i8_f16"
    ARRAY_SHAPE_IDX = 2


# reuse of gemm_i8i8_i32_M32K2N32 -> op id gemm_i8i8_i32_M32K2N32_minimal
class SnaxBingoKernelGemmI8I8I32M32K2N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 0


# reuse of gemm_i8i8_i32_M1K16N32 -> op id gemm_i8i8_i32_M1K16N32_minimal
class SnaxBingoKernelGemmI8I8I32M1K16N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 1


# reuse of gemm_i8i8_i32_M16K8N16 -> op id gemm_i8i8_i32_M16K8N16_minimal
class SnaxBingoKernelGemmI8I8I32M16K8N16MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 2


# reuse of gemm_i8i4_i32_M32K2N32 -> op id gemm_i8i4_i32_M32K2N32_minimal
class SnaxBingoKernelGemmI8I4I32M32K2N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 0


# reuse of gemm_i8i4_i32_M1K16N32 -> op id gemm_i8i4_i32_M1K16N32_minimal
class SnaxBingoKernelGemmI8I4I32M1K16N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 1


# reuse of gemm_i8i4_i32_M16K8N16 -> op id gemm_i8i4_i32_M16K8N16_minimal
class SnaxBingoKernelGemmI8I4I32M16K8N16MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 2


# reuse of gemm_i4i4_i32_M32K2N32 -> op id gemm_i4i4_i32_M32K2N32_minimal
class SnaxBingoKernelGemmI4I4I32M32K2N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 0


# reuse of gemm_i4i4_i32_M1K16N32 -> op id gemm_i4i4_i32_M1K16N32_minimal
class SnaxBingoKernelGemmI4I4I32M1K16N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 1


# reuse of gemm_i4i4_i32_M16K8N16 -> op id gemm_i4i4_i32_M16K8N16_minimal
class SnaxBingoKernelGemmI4I4I32M16K8N16MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 2


# reuse of gemm_i8i8_i8_M32K2N32 -> op id gemm_i8i8_i8_M32K2N32_minimal
class SnaxBingoKernelGemmI8I8I8M32K2N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 0


# reuse of gemm_i8i8_i8_M1K16N32 -> op id gemm_i8i8_i8_M1K16N32_minimal
class SnaxBingoKernelGemmI8I8I8M1K16N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 1


# reuse of gemm_i8i8_i8_M16K8N16 -> op id gemm_i8i8_i8_M16K8N16_minimal
class SnaxBingoKernelGemmI8I8I8M16K8N16MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 2


# reuse of gemm_i8i4_f16_M32K2N32 -> op id gemm_i8i4_f16_M32K2N32_minimal
class SnaxBingoKernelGemmI8I4F16M32K2N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 0


# reuse of gemm_i8i4_f16_M1K16N32 -> op id gemm_i8i4_f16_M1K16N32_minimal
class SnaxBingoKernelGemmI8I4F16M1K16N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 1


# reuse of gemm_i8i4_f16_M16K8N16 -> op id gemm_i8i4_f16_M16K8N16_minimal
class SnaxBingoKernelGemmI8I4F16M16K8N16MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 2


# reuse of gemm_i8i8_f16_M32K2N32 -> op id gemm_i8i8_f16_M32K2N32_minimal
class SnaxBingoKernelGemmI8I8F16M32K2N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 0


# reuse of gemm_i8i8_f16_M1K16N32 -> op id gemm_i8i8_f16_M1K16N32_minimal
class SnaxBingoKernelGemmI8I8F16M1K16N32MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 1


# reuse of gemm_i8i8_f16_M16K8N16 -> op id gemm_i8i8_f16_M16K8N16_minimal
class SnaxBingoKernelGemmI8I8F16M16K8N16MinimalArgs(_SnaxBingoKernelGemmMinimalBase):
    ARRAY_SHAPE_IDX = 2

# BINGO XDMA 1D Copy
