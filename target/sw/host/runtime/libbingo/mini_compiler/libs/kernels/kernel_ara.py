# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Ara (RVV) host kernels, typed by precision.

One class per (operation, precision) pair, each fixing a kernel name and a PRECISION
constant the runtime dispatches on. Ara is a PARTIAL RVV implementation, so not every
precision exists for every operation -- the classes that are here are the ones that do."""

from typing import Union, Dict
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol

from kernel_base import BingoKernelArgs


# Multi-precision Ara kernel args (runtime-typed __host_bingo_kernel_<op>
# dispatchers in host_kernel_lib.h). ONE class per (op, precision): the precision
# is baked into the class name — callers pick e.g. HostBingoKernelAraAddI32Args(...)
# and never pass a `precision=` value. Only (op, precision) combos the C dispatchers
# implement get a class (see host_kernel_lib.h); an unsupported combo simply has no
# class, so the mistake surfaces at author time, not in sim.
#
# Naming: HostBingoKernelAra<Op><Suffix>Args, Suffix in {F32,F16,I8,I16,I32} = the
# operand element type (reductions: the INPUT type; int8/int16 reduce produce an
# int32 scalar output). Each class carries KERNEL_NAME + PRECISION, so a node may
# omit kernel_name (BingoNode infers it from the args object).
# Pair example: BingoNode(..., kernel_args=HostBingoKernelAraExpF16Args(in, out, n)).
# ══════════════════════════════════════════════════════════════════════
BINGO_PREC_FP32  = 0
BINGO_PREC_FP16  = 1
BINGO_PREC_INT8  = 2
BINGO_PREC_INT16 = 3
BINGO_PREC_INT32 = 4

_Addr = Union[BingoMemAlloc, BingoMemSymbol, int]


class _HostBingoKernelAraBinaryArgs(BingoKernelArgs):
    """Internal base, shape {input_a, input_b, output, num_elements, precision}.
    Concrete per-precision subclasses set KERNEL_NAME and PRECISION."""
    PRECISION = BINGO_PREC_FP32   # overridden per concrete subclass
    def __init__(self, input_a_addr: _Addr, input_b_addr: _Addr, output_addr: _Addr,
                 num_elements: int):
        self.input_a_addr = input_a_addr
        self.input_b_addr = input_b_addr
        self.output_addr = output_addr
        self.num_elements = num_elements
        self.precision = self.PRECISION

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_ara_binary_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_a_addr, "input_a_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.input_b_addr, "input_b_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr,  "output_addr",  a, handle_name_map, split_64bit=False, as_64bit=True)
        a["num_elements"] = str(self.num_elements)
        a["precision"] = str(self.precision)
        return a


# add: F32/F16/I8/I16 via the multi-precision dispatcher; I32 via the distinct
# __host_bingo_kernel_add_i32 kernel (K-split partial-D accumulation).
class HostBingoKernelAraAddF32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_add";     PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraAddF16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_add";     PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraAddI8Args(_HostBingoKernelAraBinaryArgs):  KERNEL_NAME = "__host_bingo_kernel_add";     PRECISION = BINGO_PREC_INT8
class HostBingoKernelAraAddI16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_add";     PRECISION = BINGO_PREC_INT16
class HostBingoKernelAraAddI32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_add_i32"; PRECISION = BINGO_PREC_INT32

class HostBingoKernelAraSubF32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_sub"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraSubF16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_sub"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraSubI8Args(_HostBingoKernelAraBinaryArgs):  KERNEL_NAME = "__host_bingo_kernel_sub"; PRECISION = BINGO_PREC_INT8
class HostBingoKernelAraSubI16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_sub"; PRECISION = BINGO_PREC_INT16
class HostBingoKernelAraSubI32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_sub"; PRECISION = BINGO_PREC_INT32

class HostBingoKernelAraMulF32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_mul"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraMulF16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_mul"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraMulI8Args(_HostBingoKernelAraBinaryArgs):  KERNEL_NAME = "__host_bingo_kernel_mul"; PRECISION = BINGO_PREC_INT8
class HostBingoKernelAraMulI16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_mul"; PRECISION = BINGO_PREC_INT16
class HostBingoKernelAraMulI32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_mul"; PRECISION = BINGO_PREC_INT32

class HostBingoKernelAraMaxF32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_max"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraMaxF16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_max"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraMaxI8Args(_HostBingoKernelAraBinaryArgs):  KERNEL_NAME = "__host_bingo_kernel_max"; PRECISION = BINGO_PREC_INT8
class HostBingoKernelAraMaxI16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_max"; PRECISION = BINGO_PREC_INT16
class HostBingoKernelAraMaxI32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_max"; PRECISION = BINGO_PREC_INT32

class HostBingoKernelAraMinF32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_min"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraMinF16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_min"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraMinI8Args(_HostBingoKernelAraBinaryArgs):  KERNEL_NAME = "__host_bingo_kernel_min"; PRECISION = BINGO_PREC_INT8
class HostBingoKernelAraMinI16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_min"; PRECISION = BINGO_PREC_INT16
class HostBingoKernelAraMinI32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_min"; PRECISION = BINGO_PREC_INT32

# div: float only
class HostBingoKernelAraDivF32Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_div"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraDivF16Args(_HostBingoKernelAraBinaryArgs): KERNEL_NAME = "__host_bingo_kernel_div"; PRECISION = BINGO_PREC_FP16


class _HostBingoKernelAraSiluMulArgs(_HostBingoKernelAraBinaryArgs):
    """silu_mul: out = silu(gate) * up (gate->input_a, up->input_b)."""
    KERNEL_NAME = "__host_bingo_kernel_silu_mul"
    def __init__(self, gate_addr: _Addr, up_addr: _Addr, output_addr: _Addr, num_elements: int):
        super().__init__(gate_addr, up_addr, output_addr, num_elements)

# silu_mul: float only
class HostBingoKernelAraSiluMulF32Args(_HostBingoKernelAraSiluMulArgs): PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraSiluMulF16Args(_HostBingoKernelAraSiluMulArgs): PRECISION = BINGO_PREC_FP16


class _HostBingoKernelAraUnaryArgs(BingoKernelArgs):
    """Internal base, shape {input, output, num_elements, precision} (elementwise + reduce).
    Concrete per-precision subclasses set KERNEL_NAME and PRECISION."""
    PRECISION = BINGO_PREC_FP32   # overridden per concrete subclass
    def __init__(self, input_addr: _Addr, output_addr: _Addr, num_elements: int):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.num_elements = num_elements
        self.precision = self.PRECISION

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_ara_unary_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_addr,  "input_addr",  a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr, "output_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        a["num_elements"] = str(self.num_elements)
        a["precision"] = str(self.precision)
        return a


# int-capable unary ops: F32/F16/I8/I16
class HostBingoKernelAraReluF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_relu"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraReluF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_relu"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraReluI8Args(_HostBingoKernelAraUnaryArgs):  KERNEL_NAME = "__host_bingo_kernel_relu"; PRECISION = BINGO_PREC_INT8
class HostBingoKernelAraReluI16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_relu"; PRECISION = BINGO_PREC_INT16
class HostBingoKernelAraReluI32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_relu"; PRECISION = BINGO_PREC_INT32

class HostBingoKernelAraNegF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_neg"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraNegF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_neg"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraNegI8Args(_HostBingoKernelAraUnaryArgs):  KERNEL_NAME = "__host_bingo_kernel_neg"; PRECISION = BINGO_PREC_INT8
class HostBingoKernelAraNegI16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_neg"; PRECISION = BINGO_PREC_INT16
class HostBingoKernelAraNegI32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_neg"; PRECISION = BINGO_PREC_INT32

class HostBingoKernelAraAbsF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_abs"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraAbsF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_abs"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraAbsI8Args(_HostBingoKernelAraUnaryArgs):  KERNEL_NAME = "__host_bingo_kernel_abs"; PRECISION = BINGO_PREC_INT8
class HostBingoKernelAraAbsI16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_abs"; PRECISION = BINGO_PREC_INT16
class HostBingoKernelAraAbsI32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_abs"; PRECISION = BINGO_PREC_INT32

# float-only unary ops: F32/F16
class HostBingoKernelAraExpF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_exp"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraExpF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_exp"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraSigmoidF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_sigmoid"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraSigmoidF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_sigmoid"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraSqrtF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_sqrt"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraSqrtF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_sqrt"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraTanhF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_tanh"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraTanhF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_tanh"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraReciprocalF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reciprocal"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraReciprocalF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reciprocal"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraSiluF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_silu"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraSiluF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_silu"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraGeluF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_gelu"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraGeluF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_gelu"; PRECISION = BINGO_PREC_FP16

# reduce ops share the unary shape (output is a scalar float/int32). Suffix = INPUT
# element type; int8/int16 inputs produce an int32 scalar.
class HostBingoKernelAraReduceSumF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reduce_sum"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraReduceSumF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reduce_sum"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraReduceSumI8Args(_HostBingoKernelAraUnaryArgs):  KERNEL_NAME = "__host_bingo_kernel_reduce_sum"; PRECISION = BINGO_PREC_INT8
class HostBingoKernelAraReduceSumI16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reduce_sum"; PRECISION = BINGO_PREC_INT16
class HostBingoKernelAraReduceSumI32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reduce_sum"; PRECISION = BINGO_PREC_INT32
class HostBingoKernelAraReduceMaxF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reduce_max"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraReduceMaxF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reduce_max"; PRECISION = BINGO_PREC_FP16
class HostBingoKernelAraReduceMaxI8Args(_HostBingoKernelAraUnaryArgs):  KERNEL_NAME = "__host_bingo_kernel_reduce_max"; PRECISION = BINGO_PREC_INT8
class HostBingoKernelAraReduceMaxI16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reduce_max"; PRECISION = BINGO_PREC_INT16
class HostBingoKernelAraReduceMaxI32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reduce_max"; PRECISION = BINGO_PREC_INT32
class HostBingoKernelAraReduceMeanF32Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reduce_mean"; PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraReduceMeanF16Args(_HostBingoKernelAraUnaryArgs): KERNEL_NAME = "__host_bingo_kernel_reduce_mean"; PRECISION = BINGO_PREC_FP16


class _HostBingoKernelAraSoftmaxArgs(BingoKernelArgs):
    """softmax: {input, output, num_rows, row_length, precision}."""
    KERNEL_NAME = "__host_bingo_kernel_softmax"
    PRECISION = BINGO_PREC_FP32   # overridden per concrete subclass
    def __init__(self, input_addr: _Addr, output_addr: _Addr,
                 num_rows: int, row_length: int):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.num_rows = num_rows
        self.row_length = row_length
        self.precision = self.PRECISION

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_ara_softmax_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_addr,  "input_addr",  a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr, "output_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        a["num_rows"] = str(self.num_rows)
        a["row_length"] = str(self.row_length)
        a["precision"] = str(self.precision)
        return a

# softmax: float only
class HostBingoKernelAraSoftmaxF32Args(_HostBingoKernelAraSoftmaxArgs): PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraSoftmaxF16Args(_HostBingoKernelAraSoftmaxArgs): PRECISION = BINGO_PREC_FP16


class _HostBingoKernelAraRmsnormArgs(BingoKernelArgs):
    """rmsnorm: {input, weight, output, hidden_dim, num_tokens, precision}."""
    KERNEL_NAME = "__host_bingo_kernel_rmsnorm"
    PRECISION = BINGO_PREC_FP32   # overridden per concrete subclass
    def __init__(self, input_addr: _Addr, weight_addr: _Addr, output_addr: _Addr,
                 hidden_dim: int, num_tokens: int):
        self.input_addr = input_addr
        self.weight_addr = weight_addr
        self.output_addr = output_addr
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.precision = self.PRECISION

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_ara_rmsnorm_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_addr,  "input_addr",  a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.weight_addr, "weight_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr, "output_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        a["hidden_dim"] = str(self.hidden_dim)
        a["num_tokens"] = str(self.num_tokens)
        a["precision"] = str(self.precision)
        return a

# rmsnorm: float only
class HostBingoKernelAraRmsnormF32Args(_HostBingoKernelAraRmsnormArgs): PRECISION = BINGO_PREC_FP32
class HostBingoKernelAraRmsnormF16Args(_HostBingoKernelAraRmsnormArgs): PRECISION = BINGO_PREC_FP16


# Conversions with a scale pointer (shared ara_convert shape). quantize WRITES
# the computed scale; dequantize READS it. `precision` is a no-op passthrough;
# the conversion types are fixed and encoded in the class name (f32->i8 / i32->f32).
class HostBingoKernelAraQuantizeF32I8Args(BingoKernelArgs):
    """FP32 -> INT8 per-tensor symmetric quantize. scale_out_addr receives the scale."""
    KERNEL_NAME = "__host_bingo_kernel_quantize_f32i8"
    def __init__(self, input_addr: _Addr, output_addr: _Addr,
                 scale_out_addr: _Addr, num_elements: int):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.scale_out_addr = scale_out_addr
        self.num_elements = num_elements

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_ara_convert_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_addr,     "input_addr",  a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr,    "output_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.scale_out_addr, "scale_addr",  a, handle_name_map, split_64bit=False, as_64bit=True)
        a["num_elements"] = str(self.num_elements)
        a["precision"] = "0"  # BINGO_PREC_FP32 (no-op for the conversion)
        return a


class HostBingoKernelAraQuantizeF16I8Args(BingoKernelArgs):
    """FP16 -> INT8 per-tensor symmetric quantize. scale_out_addr receives the scale."""
    KERNEL_NAME = "__host_bingo_kernel_quantize_f16i8"
    def __init__(self, input_addr: _Addr, output_addr: _Addr,
                 scale_out_addr: _Addr, num_elements: int):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.scale_out_addr = scale_out_addr
        self.num_elements = num_elements

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_ara_convert_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_addr,     "input_addr",  a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr,    "output_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.scale_out_addr, "scale_addr",  a, handle_name_map, split_64bit=False, as_64bit=True)
        a["num_elements"] = str(self.num_elements)
        a["precision"] = "0"  # no-op for the conversion (input type is fixed fp16)
        return a


class HostBingoKernelAraDequantizeI32F32Args(BingoKernelArgs):
    """INT32 -> FP32 dequantize. scale_addr is read (combined_scale = scale_a * scale_b)."""
    KERNEL_NAME = "__host_bingo_kernel_dequantize_i32f32"
    def __init__(self, input_addr: _Addr, output_addr: _Addr,
                 scale_addr: _Addr, num_elements: int):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.scale_addr = scale_addr
        self.num_elements = num_elements

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_ara_convert_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_addr,  "input_addr",  a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr, "output_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.scale_addr,  "scale_addr",  a, handle_name_map, split_64bit=False, as_64bit=True)
        a["num_elements"] = str(self.num_elements)
        a["precision"] = "0"  # BINGO_PREC_FP32 (no-op for the conversion)
        return a


# ================================================================
# DARTS Tier 1: MoE Gating Kernels
# ================================================================

# ================================================================
# DARTS Tier 1: Unified CERF Gating Args
# ================================================================
# Maps to __host_bingo_kernel_cerf_gating_args_t with a mode field.
# The compiler creates the appropriate instance based on cond_dic['mode'].

BINGO_GATING_MODE_TOP_K = 0
BINGO_GATING_MODE_THRESHOLD = 1
BINGO_GATING_MODE_STATIC = 2

class HostBingoKernelCerfGatingArgs(BingoKernelArgs):
    """Unified gating kernel args. Supports top_k, threshold, and static modes.

    For top_k with >32 experts (CERF group sharing), cond_activation_addr
    points to a uint8_t[num_experts] array that the gating kernel writes
    (1=selected, 0=skip). Expert kernels read their slot via SW guard.

    cond_weight_addr is the combine's half of the same decision: float[num_experts],
    renormalised over the winners. Both are views into one allocation, so a gate
    costs one L3 record rather than two.
    """
    def __init__(self,
                 mode: int = BINGO_GATING_MODE_STATIC,
                 pred_scratchpad_addr=None,   # wired at emit time by compiler
                 cerf_controlled_mask: int = 0,
                 top_k_or_threshold: Union[int, float] = 0,
                 cerf_group_ids_addr=None,
                 cond_activation_addr=None,
                 cond_weight_addr=None):
        self.mode = mode
        self.pred_scratchpad_addr = pred_scratchpad_addr
        self.cerf_controlled_mask = cerf_controlled_mask
        self.top_k_or_threshold = top_k_or_threshold
        self.cerf_group_ids_addr = cerf_group_ids_addr
        self.cond_activation_addr = cond_activation_addr
        self.cond_weight_addr = cond_weight_addr

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_cerf_gating_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        assignments["mode"] = str(self.mode)
        if self.pred_scratchpad_addr is not None:
            assignments["pred_scratchpad_addr"] = str(self.pred_scratchpad_addr)
        else:
            assignments["pred_scratchpad_addr"] = "0"
        assignments["cerf_controlled_mask"] = f"0x{self.cerf_controlled_mask:04x}"

        if self.mode == BINGO_GATING_MODE_TOP_K:
            assignments["top_k_or_threshold"] = str(int(self.top_k_or_threshold))
            if self.cerf_group_ids_addr is not None:
                self._process_addr(self.cerf_group_ids_addr, "cerf_group_ids_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
            else:
                assignments["cerf_group_ids_addr"] = "0"
        elif self.mode == BINGO_GATING_MODE_THRESHOLD:
            import struct
            thresh_bits = struct.unpack('<I', struct.pack('<f', float(self.top_k_or_threshold)))[0]
            assignments["top_k_or_threshold"] = f"0x{thresh_bits:08x}"
            assignments["cerf_group_ids_addr"] = "0"
        elif self.mode == BINGO_GATING_MODE_STATIC:
            assignments["top_k_or_threshold"] = f"0x{int(self.top_k_or_threshold):04x}"
            assignments["cerf_group_ids_addr"] = "0"
        else:
            assignments["top_k_or_threshold"] = str(self.top_k_or_threshold)
            assignments["cerf_group_ids_addr"] = str(self.cerf_group_ids_addr or 0)

        # Per-expert activation array (SW guard for CERF group sharing)
        if self.cond_activation_addr is not None:
            self._process_addr(self.cond_activation_addr, "cond_activation_addr",
                              assignments, handle_name_map, split_64bit=False, as_64bit=True)
        else:
            assignments["cond_activation_addr"] = "0"

        # Renormalised per-expert combine weight (float[num_experts]). The combine
        # reads these as raw FP32 bits; see host_kernel_args.h.
        if self.cond_weight_addr is not None:
            self._process_addr(self.cond_weight_addr, "cond_weight_addr",
                              assignments, handle_name_map, split_64bit=False, as_64bit=True)
        else:
            assignments["cond_weight_addr"] = "0"

        return assignments


# Backward-compat aliases
HostBingoKernelMoeGatingArgs = HostBingoKernelCerfGatingArgs
HostBingoKernelCerfThresholdArgs = HostBingoKernelCerfGatingArgs
HostBingoKernelCerfStaticArgs = HostBingoKernelCerfGatingArgs


# DEVICE: Dynamic MoE gating (32-bit address space)
class SnaxBingoKernelMoeGatingArgs(BingoKernelArgs):
    """Device-side MoE gating. Reads predecessor scratchpad for logits."""
    def __init__(self,
                 pred_scratchpad_addr=None,
                 top_k: int = 2,
                 cerf_controlled_mask: int = 0,
                 cerf_group_ids_addr: Union[BingoMemAlloc, BingoMemSymbol, int, None] = None):
        self.pred_scratchpad_addr = pred_scratchpad_addr
        self.top_k = top_k
        self.cerf_controlled_mask = cerf_controlled_mask
        self.cerf_group_ids_addr = cerf_group_ids_addr

    def get_struct_name(self) -> str:
        return "__snax_kernel_moe_gating_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        if self.pred_scratchpad_addr is not None:
            assignments["pred_scratchpad_addr"] = str(self.pred_scratchpad_addr)
        assignments["top_k"] = str(self.top_k)
        assignments["cerf_controlled_mask"] = f"0x{self.cerf_controlled_mask:04x}"
        if self.cerf_group_ids_addr is not None:
            self._process_addr(self.cerf_group_ids_addr, "cerf_group_ids_addr", assignments, handle_name_map, split_64bit=False)
        return assignments
