# Fanchen Kong <fanchen.kong@kuleuven.be>
from abc import ABC, abstractmethod
from typing import Union, Dict, Optional
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView, BingoMemSymbol, BingoMemFixedAddr
from bingo_helpers import _check_xdma_size_aligned

class BingoKernelArgs(ABC):
    """
    Abstract base class for Kernel Arguments.
    Subclasses define the specific arguments for each kernel type and how they map to C structs.
    """

    # Optional: the C dispatcher this args struct pairs with. When set on a
    # subclass, BingoNode infers `kernel_name` from the args if none is given.
    KERNEL_NAME: Optional[str] = None

    # Internal: set by the compiler during C emission, not by users.
    _scratchpad_c_expr: str = None    # C expr for this kernel's scratchpad pointer
    _gating_sp_c_expr: str = None     # C expr for gating kernel's scratchpad (SW guard)
    _cond_node_index: int = None         # This expert's index in the activation array

    def get_c_field_assignments_with_scratchpad(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        """Return field assignments including SW guard + scratchpad fields.

        For device kernels (uint32_t fields): gating_sp_addr, cond_node_index, scratchpad_ptr
        For host kernels (uint64_t fields): scratchpad_ptr only (host kernels don't need SW guard)
        The compiler sets _gating_sp_c_expr/_cond_node_index/_scratchpad_c_expr before calling this.
        """
        assignments = self.get_c_field_assignments(handle_name_map)
        # SW guard fields — only for device kernels (struct name starts with __snax)
        # Host kernel structs don't have gating_sp_addr/cond_node_index fields.
        is_device = self.get_struct_name().startswith("__snax")
        if is_device:
            if self._gating_sp_c_expr is not None:
                assignments["gating_sp_addr"] = self._gating_sp_c_expr
            else:
                assignments["gating_sp_addr"] = "0"
            if self._cond_node_index is not None:
                assignments["cond_node_index"] = str(self._cond_node_index)
            else:
                assignments["cond_node_index"] = "0"
        # Scratchpad pointer (always last field, both host and device)
        if self._scratchpad_c_expr is not None:
            assignments["scratchpad_ptr"] = self._scratchpad_c_expr
        return assignments

    @abstractmethod
    def get_struct_name(self) -> str:
        """Returns the C struct type definition name (e.g. __snax_kernel_dummy_args_t)"""
        pass
    
    @abstractmethod
    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        """
        Returns a dict of { c_field_name : c_value_string }.
        This allows the generator to emit:
        args_ptr->c_field_name = c_value_string;
        """
        pass
        
    def _process_addr(self, val: Union[int, BingoMemAlloc, BingoMemSymbol, BingoMemFixedAddr], base_name: str, assignments: Dict[str, str], handle_name_map: Dict[BingoMemAlloc, str], split_64bit: bool = True, as_64bit: bool = False):
        """
        Helper to generate address fields for a handle or integer address.
        
        Args:
            val: The value to process (int, handle, symbol, or absolute addr).
            base_name: The base name of the C struct field (e.g., "src_addr").
            assignments: The dictionary to populate with C field assignments.
            handle_name_map: Map from handle objects to their C variable names.
            split_64bit: If True, splits 64-bit address into Lo/Hi 32-bit fields.
                         e.g. src_addr_lo, src_addr_hi
            as_64bit: If True (and split_64bit=False), treats the field as a single uint64_t.
                      If False (and split_64bit=False), treats it as uint32_t.

        Examples:
            1. split_64bit=True (Default)
               => base_name_lo = (uint32_t)val
               => base_name_hi = (uint32_t)(val >> 32)
            
            2. split_64bit=False, as_64bit=False
               => base_name = (uint32_t)val

            3. split_64bit=False, as_64bit=True
               => base_name = (uint64_t)val
        """
        if isinstance(val, BingoMemAlloc):
            if val in handle_name_map:
                c_var = handle_name_map[val]
                offset_op = f" + {val.offset}" if val.offset != 0 else ""
                c_expr = f"{c_var}{offset_op}" if not offset_op else f"({c_var}{offset_op})"
                if not offset_op:
                    c_expr = c_var
                if split_64bit:
                    assignments[f"{base_name}_lo"] = f"(uint32_t){c_expr}"
                    assignments[f"{base_name}_hi"] = f"(uint32_t)({c_expr} >> 32)"
                elif as_64bit:
                    assignments[base_name] = f"(uint64_t){c_expr}"
                else:
                    assignments[base_name] = f"(uint32_t){c_expr}"
            else:
                 # Should not happen if handles are collected correctly before code gen
                if split_64bit:
                    assignments[f"{base_name}_lo"] = f"(uint32_t)0 /* UNREF_HANDLE: {val.name} */"
                    assignments[f"{base_name}_hi"] = f"(uint32_t)0"
                elif as_64bit:
                    assignments[base_name] = f"(uint64_t)0 /* UNREF_HANDLE: {val.name} */"
                else:
                    assignments[base_name] = f"(uint32_t)0 /* UNREF_HANDLE: {val.name} */"
        elif isinstance(val, BingoMemAllocView):
            # A byte offset into an allocation another node owns -> `ptr_<base> + offset`.
            # The base buffer is allocated once; this emits no allocation of its own.
            if val.base in handle_name_map:
                c_var = handle_name_map[val.base]
                c_expr = f"({c_var} + {val.offset})" if val.offset else c_var
                if split_64bit:
                    assignments[f"{base_name}_lo"] = f"(uint32_t){c_expr}"
                    assignments[f"{base_name}_hi"] = f"(uint32_t)({c_expr} >> 32)"
                elif as_64bit:
                    assignments[base_name] = f"(uint64_t){c_expr}"
                else:
                    assignments[base_name] = f"(uint32_t){c_expr}"
            else:
                if split_64bit:
                    assignments[f"{base_name}_lo"] = f"(uint32_t)0 /* UNREF_HANDLE: {val.base.name} */"
                    assignments[f"{base_name}_hi"] = f"(uint32_t)0"
                elif as_64bit:
                    assignments[base_name] = f"(uint64_t)0 /* UNREF_HANDLE: {val.base.name} */"
                else:
                    assignments[base_name] = f"(uint32_t)0 /* UNREF_HANDLE: {val.base.name} */"
        elif isinstance(val, BingoMemSymbol):
            c_var = val.symbol_name
            offset_op = f" + {val.offset}" if val.offset != 0 else ""
            
            # 1. Base expression: Cast symbol to uintptr_t and apply offset
            base_expr = f"(uintptr_t){c_var}{offset_op}"
            
            # 2. Apply transformation (chiplet_addr_transform)
            # Wrap with transformation function, casting input to uint64_t as commonly required
            final_expr = f"chiplet_addr_transform((uint64_t)({base_expr}))"

            # 3. Cast to final destination type/width
            if split_64bit:
                assignments[f"{base_name}_lo"] = f"(uint32_t)({final_expr})"
                assignments[f"{base_name}_hi"] = f"(uint32_t)(({final_expr}) >> 32)"
            elif as_64bit:
                assignments[base_name] = f"(uint64_t)({final_expr})"
            else:
                assignments[base_name] = f"(uint32_t)({final_expr})"
        elif isinstance(val, BingoMemFixedAddr):
            addr = val.address
            if split_64bit:
                assignments[f"{base_name}_lo"] = f"(uint32_t)0x{addr:x}"
                assignments[f"{base_name}_hi"] = f"(uint32_t)(0x{addr:x} >> 32)"
            elif as_64bit:
                assignments[base_name] = f"(uint64_t)0x{addr:x}"
            else:
                assignments[base_name] = f"(uint32_t)0x{addr:x}"
        else:
            if split_64bit:
                assignments[f"{base_name}_lo"] = f"(uint32_t){val}"
                assignments[f"{base_name}_hi"] = f"(uint32_t)({val} >> 32)"
            elif as_64bit:
                assignments[base_name] = f"(uint64_t){val}"
            else:
                assignments[base_name] = f"(uint32_t){val}"


# -------------------------------------------------------------
# Specific Kernel Argument Implementations
# -------------------------------------------------------------

# Dummy kernel args
class SnaxBingoKernelDummyArgs(BingoKernelArgs):
    def __init__(self, dummy_input: int):
        self.dummy_input = dummy_input

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_dummy_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        return {"dummy_input": str(self.dummy_input)}

# BINGO IDMA 1D Copy
class SnaxBingoKernelIdma1dCopyArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int], size: int):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.size = size

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_idma_1d_copy_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["size"] = str(self.size)
        return assignments

# BINGO IDMA BROADCAST
class SnaxBingoKernelIdmaBroadcastArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int], size: int):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.size = size

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_idma_broadcast_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["size"] = str(self.size)
        return assignments

# BINGO IDMA Pairwise Swap (flat adjacent-element-pair swap: dst[i] = src[i^1])
class SnaxBingoKernelIdmaPairwiseSwapArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 num_elems: int, elem_bytes: int = 2):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.num_elems = num_elems
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_idma_pairwise_swap_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["num_elems"] = str(self.num_elems)
        assignments["elem_bytes"] = str(self.elem_bytes)
        return assignments



# BINGO GEMM FULL
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
class SnaxBingoKernelXdma1dCopyArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int], size: int):
        _check_xdma_size_aligned(size, "SnaxBingoKernelXdma1dCopyArgs")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.size = size

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_1d_copy_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["size"] = str(self.size)
        return assignments

# BINGO XDMA 6D (fixed-size, exposes full AGU strides/bounds, max 6 dims)
class SnaxBingoKernelXdma6dArgs(BingoKernelArgs):
    """
    Args for __snax_bingo_kernel_xdma_6d.

    Fixed-size struct with 5 temporal dimension slots (1 spatial + 5 temporal = 6 total).
    Unused dimensions should have stride=0 and bound=1.
    """
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 spatial_stride_src: int, spatial_stride_dst: int,
                 temporal_strides_src: list, temporal_bounds_src: list,
                 temporal_strides_dst: list, temporal_bounds_dst: list):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.spatial_stride_src = spatial_stride_src
        self.spatial_stride_dst = spatial_stride_dst
        n = len(temporal_strides_src)
        assert n == len(temporal_bounds_src) == len(temporal_strides_dst) == len(temporal_bounds_dst)
        assert 1 <= n <= 5, f"num_temporal_dims must be 1..5, got {n}"
        # Pad to 5 slots: unused dims get stride=0, bound=1
        self.num_temporal_dims = n
        self.temporal_strides_src = list(temporal_strides_src) + [0] * (5 - n)
        self.temporal_bounds_src  = list(temporal_bounds_src)  + [1] * (5 - n)
        self.temporal_strides_dst = list(temporal_strides_dst) + [0] * (5 - n)
        self.temporal_bounds_dst  = list(temporal_bounds_dst)  + [1] * (5 - n)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_6d_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["spatial_stride_src"] = str(self.spatial_stride_src)
        assignments["spatial_stride_dst"] = str(self.spatial_stride_dst)
        assignments["num_temporal_dims"] = str(self.num_temporal_dims)
        for i in range(5):
            assignments[f"temporal_strides_src[{i}]"] = str(self.temporal_strides_src[i])
        for i in range(5):
            assignments[f"temporal_bounds_src[{i}]"] = str(self.temporal_bounds_src[i])
        for i in range(5):
            assignments[f"temporal_strides_dst[{i}]"] = str(self.temporal_strides_dst[i])
        for i in range(5):
            assignments[f"temporal_bounds_dst[{i}]"] = str(self.temporal_bounds_dst[i])
        return assignments

# BINGO XDMA Transpose 2D (high-level: user provides shape only)
class SnaxBingoKernelXdmaTranspose2dArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 M: int, N: int, elem_bytes: int = 1):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.M = M
        self.N = N
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_transpose_2d_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["M"] = str(self.M)
        assignments["N"] = str(self.N)
        assignments["elem_bytes"] = str(self.elem_bytes)
        return assignments

# BINGO XDMA Submatrix 2D (high-level: user provides shape + slice range)
class SnaxBingoKernelXdmaSubmatrix2dArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 src_rows: int, src_cols: int,
                 row_start: int, row_end: int, col_start: int, col_end: int,
                 elem_bytes: int = 1):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.src_rows = src_rows
        self.src_cols = src_cols
        self.row_start = row_start
        self.row_end = row_end
        self.col_start = col_start
        self.col_end = col_end
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_submatrix_2d_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["src_rows"] = str(self.src_rows)
        assignments["src_cols"] = str(self.src_cols)
        assignments["row_start"] = str(self.row_start)
        assignments["row_end"] = str(self.row_end)
        assignments["col_start"] = str(self.col_start)
        assignments["col_end"] = str(self.col_end)
        assignments["elem_bytes"] = str(self.elem_bytes)
        return assignments

# BINGO XDMA Expand 2D (high-level: broadcast [1, N] -> [M, N])
class SnaxBingoKernelXdmaExpand2dArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 M: int, N: int, elem_bytes: int = 1):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.M = M
        self.N = N
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_expand_2d_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["M"] = str(self.M)
        assignments["N"] = str(self.N)
        assignments["elem_bytes"] = str(self.elem_bytes)
        return assignments

class SnaxBingoKernelXdmaConcat2dArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 src_rows: int, src_cols: int, dst_rows: int, dst_cols: int,
                 axis: int, offset: int, elem_bytes: int = 1):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.src_rows = src_rows
        self.src_cols = src_cols
        self.dst_rows = dst_rows
        self.dst_cols = dst_cols
        self.axis = axis
        self.offset = offset
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_concat_2d_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["src_rows"] = str(self.src_rows)
        assignments["src_cols"] = str(self.src_cols)
        assignments["dst_rows"] = str(self.dst_rows)
        assignments["dst_cols"] = str(self.dst_cols)
        assignments["axis"] = str(self.axis)
        assignments["offset"] = str(self.offset)
        assignments["elem_bytes"] = str(self.elem_bytes)
        return assignments


class SnaxBingoKernelXdmaPad2dArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 src_rows: int, src_cols: int,
                 pad_top: int, pad_bottom: int, pad_left: int, pad_right: int,
                 elem_bytes: int = 1):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.src_rows = src_rows
        self.src_cols = src_cols
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_pad_2d_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["src_rows"] = str(self.src_rows)
        assignments["src_cols"] = str(self.src_cols)
        assignments["pad_top"] = str(self.pad_top)
        assignments["pad_bottom"] = str(self.pad_bottom)
        assignments["pad_left"] = str(self.pad_left)
        assignments["pad_right"] = str(self.pad_right)
        assignments["elem_bytes"] = str(self.elem_bytes)
        return assignments


class SnaxBingoKernelXdmaGather2dArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 src_rows: int, src_cols: int, num_indices: int,
                 index_start: int, index_stride: int, elem_bytes: int = 1):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.src_rows = src_rows
        self.src_cols = src_cols
        self.num_indices = num_indices
        self.index_start = index_start
        self.index_stride = index_stride
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_gather_2d_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["src_rows"] = str(self.src_rows)
        assignments["src_cols"] = str(self.src_cols)
        assignments["num_indices"] = str(self.num_indices)
        assignments["index_start"] = str(self.index_start)
        assignments["index_stride"] = str(self.index_stride)
        assignments["elem_bytes"] = str(self.elem_bytes)
        return assignments


# BINGO XDMA ElementwiseAdd (writer ext: dst = sum of `num_operands` int32
# operand buffers). Each operand holds `num_int32_elem_per_operand` int32 (must be
# a multiple of 16); consecutive operands are `operand_stride` bytes apart.
# Used to fuse the GEMM K-split partial-sum adds into one streaming pass.
class SnaxBingoKernelXdmaElementwiseAddArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 num_int32_elem_per_operand: int, num_operands: int, operand_stride: int):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.num_int32_elem_per_operand = num_int32_elem_per_operand
        self.num_operands = num_operands
        self.operand_stride = operand_stride

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_elementwise_add_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["num_int32_elem_per_operand"] = str(self.num_int32_elem_per_operand)
        a["num_operands"] = str(self.num_operands)
        a["operand_stride"] = str(self.operand_stride)
        return a


# BINGO XDMA ElementwiseAdd AB (two-operand) (convenience: dst = a + b, int32).
class SnaxBingoKernelXdmaElementwiseAddAbArgs(BingoKernelArgs):
    def __init__(self, src_a_addr: Union[BingoMemAlloc, int], src_b_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int], num_int32_elements: int):
        self.src_a_addr = src_a_addr
        self.src_b_addr = src_b_addr
        self.dst_addr = dst_addr
        self.num_int32_elements = num_int32_elements

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_elementwise_add_ab_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_a_addr, "src_a_addr", a, handle_name_map)
        self._process_addr(self.src_b_addr, "src_b_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["num_int32_elements"] = str(self.num_int32_elements)
        return a


# ══════════════════════════════════════════════════════════════════════
# xDMA FP16 streaming-SIMD primitives (reader extensions)
#
# The 3 generic ops the LLM layers (softmax/rmsnorm/silu/swiglu/rope) decompose
# into. The named sub-ops (reduce_max, map_exp, map_norm, ew_mul, ...) are these
# 3 classes constructed with preset op/func + FP32-bit operands. One row =
# `beats` x 64-byte beats (64 B = 32 FP16). csr_mode picks FULL (0, completely
# configure the AGU — the default) vs STICKY (1, retask-only, reuse the persisted
# same-shape config — the opt-in). dst_bound0 is the WRITER beat count.
#
# CSR encodings: StreamMap func 0=LINEAR(a*x+b) 1=EXP 2=SILU; StreamReduce op
# 0=MAX 1=ADD 2=SUMSQ, |0x100 = TAP, |0x200 = OUT_FP32; StreamElementwise op
# 0=MUL 1=ADD.
# ══════════════════════════════════════════════════════════════════════

# StreamReduce op-CSR flag bits (OR'd into `op`). REDUCE_OUT_FP32 keeps the per-row
# scalar in FP32 instead of narrowing it to the FP16 transport -- use it whenever the
# reduction can exceed fp16 range (e.g. SUMSQ of unscaled activations), since the FP16
# narrow wraps to garbage (NOT inf) on overflow. The host consumer must then read the
# scalar as fp32 (stride 16) instead of u16 (stride 32).
REDUCE_OP_TAP   = 1 << 8   # 0x100: pass the row through, then emit the scalar beat
REDUCE_OUT_FP32 = 1 << 9   # 0x200: emit the per-row scalar in FP32 (no FP16 narrow)


class SnaxBingoKernelXdmaStreamReduceArgs(BingoKernelArgs):
    """StreamReduce: per-row reduction (row -> scalar). op: 0=MAX 1=ADD 2=SUMSQ.
    Runs `rows` independent reductions in one dispatch (rows=1 = single row),
    emitting one splatted scalar beat per row (dst_bound0 defaults to rows).
    out_fp32=True ORs REDUCE_OUT_FP32 into op so the scalar reaches the host in FP32
    (the host reader then uses in_fp32=1); use it when the reduction can overflow fp16."""
    KERNEL_NAME = "__snax_bingo_kernel_xdma_stream_reduce"

    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 beats: int, op: int, rows: int = 1, csr_mode: int = 0,
                 dst_bound0: Optional[int] = None, out_fp32: bool = False):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.beats = beats
        self.op = (op | REDUCE_OUT_FP32) if out_fp32 else op
        self.rows = rows
        self.csr_mode = csr_mode
        self.dst_bound0 = rows if dst_bound0 is None else dst_bound0

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_stream_reduce_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["beats"] = str(self.beats)
        a["op"] = str(self.op)
        a["rows"] = str(self.rows)
        a["csr_mode"] = str(self.csr_mode)
        a["dst_bound0"] = str(self.dst_bound0)
        return a


class SnaxBingoKernelXdmaStreamMapArgs(BingoKernelArgs):
    """StreamMap: out = func(a*x + b) per element, over `rows*beats` flat beats.
    a_f32bits/b_f32bits are FP32 bit patterns (a defaults to 1.0f). out_dtype=1
    fuses FP16->INT8 quant with inv_scale_f32bits (pass dst_bound0 = rows*beats//2)."""
    KERNEL_NAME = "__snax_bingo_kernel_xdma_stream_map"

    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 beats: int, func: int, a_f32bits: int = 0x3F800000, b_f32bits: int = 0,
                 rows: int = 1, csr_mode: int = 0, dst_bound0: Optional[int] = None,
                 out_dtype: int = 0, inv_scale_f32bits: int = 0,
                 a_addr: Union[BingoMemAlloc, int, None] = None):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.beats = beats
        self.func = func
        self.a_f32bits = a_f32bits
        self.b_f32bits = b_f32bits
        self.rows = rows
        self.csr_mode = csr_mode
        self.dst_bound0 = rows * beats if dst_bound0 is None else dst_bound0
        self.out_dtype = out_dtype
        self.inv_scale_f32bits = inv_scale_f32bits
        # 0 = use a_f32bits; a handle = read the runtime 'a' (dequant qsc) from L1 at run time.
        self.a_addr = 0 if a_addr is None else a_addr

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_stream_map_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["beats"] = str(self.beats)
        a["a_f32bits"] = str(self.a_f32bits)
        a["b_f32bits"] = str(self.b_f32bits)
        a["func"] = str(self.func)
        a["rows"] = str(self.rows)
        a["csr_mode"] = str(self.csr_mode)
        a["dst_bound0"] = str(self.dst_bound0)
        a["out_dtype"] = str(self.out_dtype)
        a["inv_scale_f32bits"] = str(self.inv_scale_f32bits)
        if isinstance(self.a_addr, int):
            a["a_addr_lo"] = str(self.a_addr & 0xFFFFFFFF)
            a["a_addr_hi"] = str((self.a_addr >> 32) & 0xFFFFFFFF)
        else:
            self._process_addr(self.a_addr, "a_addr", a, handle_name_map)
        return a


class SnaxBingoKernelXdmaStreamMapReduceArgs(BingoKernelArgs):
    """MERGED StreamMap -||> StreamReduce: the map AND the reduce in ONE xDMA task,
    i.e. per row out = reduce(reduce_op, map(func, a*x + b)). Both reader extensions are
    enabled for a single task, so the map feeds the reduce inside the datapath and the
    mapped row is never written out and re-read -- this is the softmax exp+Sexp fusion,
    replacing a map task plus a reduce task that re-reads the whole mapped tensor.

    reduce_op: 0=MAX 1=ADD 2=SUMSQ. tap/out_fp32 OR the flag bits in for you.

    Output layout (and the dst_bound0 default) depends on `tap`:
      tap=True  -> the mapped row passes through 1:1 AND the row scalar is appended as a
                   trailing beat: a PADDED [rows, beats+1] beat tensor. Allocate
                   rows*(beats+1)*64 bytes, and mind the padded row stride downstream --
                   a consumer of the mapped data must either be single-row (rows=1: a
                   flat `beats`-beat read stops short of the scalar) or address rows at
                   the (beats+1)*64-byte stride.       dst_bound0 = rows*(beats+1)
      tap=False -> only the per-row scalar beats are written (a stream_reduce over the
                   MAPPED values, no passthrough).     dst_bound0 = rows
    """
    KERNEL_NAME = "__snax_bingo_kernel_xdma_stream_map_reduce"

    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 beats: int, func: int, reduce_op: int, a_f32bits: int = 0x3F800000,
                 b_f32bits: int = 0, tap: bool = True, out_fp32: bool = False,
                 rows: int = 1, csr_mode: int = 0, dst_bound0: Optional[int] = None,
                 a_addr: Union[BingoMemAlloc, int, None] = None):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.beats = beats
        self.func = func
        self.a_f32bits = a_f32bits
        self.b_f32bits = b_f32bits
        op = reduce_op
        if tap:
            op |= REDUCE_OP_TAP
        if out_fp32:
            op |= REDUCE_OUT_FP32
        self.reduce_op = op
        self.rows = rows
        self.csr_mode = csr_mode
        if dst_bound0 is None:
            # TAP writes beats mapped beats + 1 scalar beat per row; otherwise 1 beat/row.
            dst_bound0 = rows * (beats + 1) if tap else rows
        self.dst_bound0 = dst_bound0
        # 0 = use a_f32bits; a handle = read the runtime 'a' (dequant qsc) from L1 at run time.
        self.a_addr = 0 if a_addr is None else a_addr

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_stream_map_reduce_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["beats"] = str(self.beats)
        a["a_f32bits"] = str(self.a_f32bits)
        a["b_f32bits"] = str(self.b_f32bits)
        a["func"] = str(self.func)
        a["reduce_op"] = str(self.reduce_op)
        a["rows"] = str(self.rows)
        a["csr_mode"] = str(self.csr_mode)
        a["dst_bound0"] = str(self.dst_bound0)
        if isinstance(self.a_addr, int):
            a["a_addr_lo"] = str(self.a_addr & 0xFFFFFFFF)
            a["a_addr_hi"] = str((self.a_addr >> 32) & 0xFFFFFFFF)
        else:
            self._process_addr(self.a_addr, "a_addr", a, handle_name_map)
        return a


class SnaxBingoKernelXdmaFp16ToInt8Args(BingoKernelArgs):
    """Fp16ToInt8: out = clamp(round(x * inv_scale), -128, 127) over `rows*beats` flat beats,
    on the HasFp16ToInt8 xDMA datapath -- the dedicated activation fp16 -> int8 GEMM-operand
    requant that replaces the host quantize_f16i8. inv_scale_f32bits = FP32 bits of 127/max|x|
    (the producer computes max|x| via MAX(x)+MAX(-x) reduces). dst_bound0 = rows*beats//2
    (int8 packs two elements per fp16 lane)."""
    KERNEL_NAME = "__snax_bingo_kernel_xdma_fp16_to_int8"

    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 beats: int, rows: int, inv_scale_f32bits: int = 0,
                 csr_mode: int = 0, dst_bound0: Optional[int] = None,
                 inv_scale_addr: Union[BingoMemAlloc, int, None] = None):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.beats = beats
        self.rows = rows
        self.inv_scale_f32bits = inv_scale_f32bits
        self.csr_mode = csr_mode
        self.dst_bound0 = (rows * beats) // 2 if dst_bound0 is None else dst_bound0
        # 0 = use inv_scale_f32bits; a handle = read the runtime inv_scale (127/max|x| that
        # requant_scale wrote) from L1 at run time.
        self.inv_scale_addr = 0 if inv_scale_addr is None else inv_scale_addr

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_fp16_to_int8_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["beats"] = str(self.beats)
        a["rows"] = str(self.rows)
        a["inv_scale_f32bits"] = str(self.inv_scale_f32bits)
        a["csr_mode"] = str(self.csr_mode)
        a["dst_bound0"] = str(self.dst_bound0)
        if isinstance(self.inv_scale_addr, int):
            a["inv_scale_addr_lo"] = str(self.inv_scale_addr & 0xFFFFFFFF)
            a["inv_scale_addr_hi"] = str((self.inv_scale_addr >> 32) & 0xFFFFFFFF)
        else:
            self._process_addr(self.inv_scale_addr, "inv_scale_addr", a, handle_name_map)
        return a


class SnaxBingoKernelXdmaStreamElementwiseArgs(BingoKernelArgs):
    """StreamElementwise: out = op(operand_0, operand_1, ...) over `operand_count`
    interleaved streams operand_stride bytes apart, across `rows*beats` flat beats.
    op: 0=MUL 1=ADD. out_dtype=1 fuses FP16->INT8 quant with inv_scale_f32bits
    (pass dst_bound0 = rows*beats//2)."""
    KERNEL_NAME = "__snax_bingo_kernel_xdma_stream_elementwise"

    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 beats: int, op: int, operand_stride: int = 0, operand_count: int = 2,
                 rows: int = 1, csr_mode: int = 0, dst_bound0: Optional[int] = None,
                 out_dtype: int = 0, inv_scale_f32bits: int = 0,
                 src_b_addr: Union[BingoMemAlloc, int, None] = None,
                 src_row_stride: int = 0):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.beats = beats
        self.operand_stride = operand_stride
        self.operand_count = operand_count
        self.op = op
        self.rows = rows
        self.csr_mode = csr_mode
        self.dst_bound0 = rows * beats if dst_bound0 is None else dst_bound0
        self.out_dtype = out_dtype
        self.inv_scale_f32bits = inv_scale_f32bits
        # 0 = use operand_stride; a handle = derive stride from src_b - src_addr at run time.
        self.src_b_addr = 0 if src_b_addr is None else src_b_addr
        # 0 = flat/packed operands (2D reader). Nonzero = PADDED rows this many bytes apart,
        # of which only the first `beats` beats are data -> 3D {operand, beat, row} reader that
        # skips the slack. Use it to consume a TAP-padded map+reduce output (stride
        # (beats+1)*64); BOTH operands must share the stride, so the broadcast operand is built
        # at the same pitch by the xDMA stride-0 broadcast pass that produces it.
        self.src_row_stride = src_row_stride

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_stream_elementwise_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["beats"] = str(self.beats)
        a["operand_stride"] = str(self.operand_stride)
        a["operand_count"] = str(self.operand_count)
        a["op"] = str(self.op)
        a["rows"] = str(self.rows)
        a["csr_mode"] = str(self.csr_mode)
        a["dst_bound0"] = str(self.dst_bound0)
        a["out_dtype"] = str(self.out_dtype)
        a["inv_scale_f32bits"] = str(self.inv_scale_f32bits)
        if isinstance(self.src_b_addr, int):
            a["src_b_addr_lo"] = str(self.src_b_addr & 0xFFFFFFFF)
            a["src_b_addr_hi"] = str((self.src_b_addr >> 32) & 0xFFFFFFFF)
        else:
            self._process_addr(self.src_b_addr, "src_b_addr", a, handle_name_map)
        a["src_row_stride"] = str(self.src_row_stride)
        return a


class SnaxBingoKernelXdmaRopeArgs(BingoKernelArgs):
    """Fused FP16 RoPE: iDMA adjacent-pair swap of x + 3 StreamElementwise passes
    (x*cos_full, xswap*sin_signed, +) -> out. cos_full/sin_signed are precomputed
    tables; the kernel allocates xswap/tmp1/tmp2 scratch from L1. D = beats*32 fp16
    elements per row, rows independent token positions."""
    KERNEL_NAME = "__snax_bingo_kernel_xdma_rope"

    def __init__(self, x_addr: Union[BingoMemAlloc, int], cos_addr: Union[BingoMemAlloc, int],
                 sin_addr: Union[BingoMemAlloc, int], out_addr: Union[BingoMemAlloc, int],
                 cols: int, rows: int = 1):
        self.x_addr = x_addr
        self.cos_addr = cos_addr
        self.sin_addr = sin_addr
        self.out_addr = out_addr
        self.cols = cols        # per-row fp16 length D (a multiple of 32)
        self.rows = rows

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_rope_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.x_addr, "x_addr", a, handle_name_map)
        self._process_addr(self.cos_addr, "cos_addr", a, handle_name_map)
        self._process_addr(self.sin_addr, "sin_addr", a, handle_name_map)
        self._process_addr(self.out_addr, "out_addr", a, handle_name_map)
        a["cols"] = str(self.cols)
        a["rows"] = str(self.rows)
        return a


class _SnaxBingoKernelXdmaSimdArgs(BingoKernelArgs):
    """Shared base for the fused fp16 SIMD kernels (softmax, rmsnorm) that write a single
    output tensor. The user's args are HW-free: `input_addr` / `output_addr` are the
    [rows, cols] tensors (cols = per-row length D, a multiple of 32). The OUTPUT PRECISION is
    chosen by which subclass (kernel) you pick, not an arg:
      *F16F16Args -> fp16 output; output_addr is the fp16 buffer.
      *F16I8Args  -> int8 output (fused Fp16ToInt8, kernel-baked scale); output_addr is the
                     int8 buffer. No fp16 output is written (the fp16 result stays in scratch).
    Subclasses set KERNEL_NAME and STRUCT_NAME."""
    STRUCT_NAME = None  # subclass sets

    def __init__(self, input_addr: Union[BingoMemAlloc, int],
                 output_addr: Union[BingoMemAlloc, int], rows: int, cols: int):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.rows = rows
        self.cols = cols

    def get_struct_name(self) -> str:
        return self.STRUCT_NAME

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_addr, "input_addr", a, handle_name_map)
        self._process_addr(self.output_addr, "output_addr", a, handle_name_map)
        a["rows"] = str(self.rows)
        a["cols"] = str(self.cols)
        return a


class SnaxBingoKernelXdmaSoftmaxF16F16Args(_SnaxBingoKernelXdmaSimdArgs):
    """Whole FP16 softmax in ONE DM-core kernel -> fp16 output. reduce-MAX, device negate,
    sub-max, merged EXP+Sexp, integer reciprocal (rv32iM divu), normalize. Host does only
    Load / Store / Check."""
    KERNEL_NAME = "__snax_bingo_kernel_xdma_softmax_f16_f16"
    STRUCT_NAME = "__snax_bingo_kernel_xdma_softmax_args_t"


class SnaxBingoKernelXdmaSoftmaxF16I8Args(_SnaxBingoKernelXdmaSimdArgs):
    """Same fused softmax pipeline -> int8 output (fused Fp16ToInt8, baked 127.0 scale since
    softmax output is in [0,1]). output_addr is the int8 [rows, cols] buffer."""
    KERNEL_NAME = "__snax_bingo_kernel_xdma_softmax_f16_i8"
    STRUCT_NAME = "__snax_bingo_kernel_xdma_softmax_args_t"


class SnaxBingoKernelXdmaRmsnormF16F16Args(_SnaxBingoKernelXdmaSimdArgs):
    """Whole FP16 rmsnorm in ONE DM-core kernel -> fp16 output. reduce-SUMSQ, integer
    1/sqrt(Sxx/N) (device sqrt + reciprocal, no FPU), normalize. cols is a power-of-two
    multiple of 32. Host does only Load / Store / Check."""
    KERNEL_NAME = "__snax_bingo_kernel_xdma_rmsnorm_f16_f16"
    STRUCT_NAME = "__snax_bingo_kernel_xdma_rmsnorm_args_t"


class SnaxBingoKernelXdmaRmsnormF16I8Args(_SnaxBingoKernelXdmaSimdArgs):
    """Same fused rmsnorm pipeline -> int8 output (fused Fp16ToInt8, baked 64.0 scale).
    output_addr is the int8 [rows, cols] buffer."""
    KERNEL_NAME = "__snax_bingo_kernel_xdma_rmsnorm_f16_i8"
    STRUCT_NAME = "__snax_bingo_kernel_xdma_rmsnorm_args_t"


# ══════════════════════════════════════════════════════════════════════
# VersaCore blocked-layout conversion kernels (tile-shape-parameterized)
#
# Six primitive conversions between row-major and the three VersaCore
# blocked layouts {A, B, D}. All kernels take tile dimensions (M_T, K_T,
# N_T) and array-shape dims (meshRow, tileSize, meshCol) so they work
# for any DSE-chosen tiling. See HeMAiA/util/sim/xdma/layout_convert.py for
# the Python reference.
# ══════════════════════════════════════════════════════════════════════


# -------------------------------------------------------------------------
# xDMA blocked-layout converters: ONE class per runnable kernel =
# (converter, array shape, elem_bytes).
#
# meshRow / tileSize / meshCol and elem_bytes used to be struct fields the caller
# filled in. They are not arguments -- they decide which AGU path the kernel takes,
# and they must match the mesh the operands were actually blocked for. So each is a
# COMPILE-TIME constant of a device wrapper (`..._e1_M32K2`), and here a class
# constant. What is left in the struct is genuinely per-call: the addresses and the
# two tile counts.
#
# The 9 wrappers of a converter share one C struct, so the struct-name -> C-fn
# convention cannot name the dispatcher; each class names it via KERNEL_NAME.
#
# Naming: e<elem_bytes>_<mesh token>, where the mesh token names the two axes the
# block spans -- M<meshRow>K<tileSize> for A, K<tileSize>N<meshCol> for B,
# M<meshRow>N<meshCol> for D. Same (mu,ku,nu) order as the GEMM kernels.
# -------------------------------------------------------------------------
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


class HostBingoKernelDummyArgs(BingoKernelArgs):
    def __init__(self, dummy_input: int):
        self.dummy_input = dummy_input

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_dummy_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        return {"dummy_input": str(self.dummy_input)}

# HOST BINGO Check Result
# Check-mode constants (mirror #defines in host_kernel_args.h)
BINGO_CHECK_TYPE_BYTE_EXACT = 0
BINGO_CHECK_TYPE_FP32_TOL   = 1
BINGO_CHECK_TYPE_FP16_TOL   = 2
BINGO_CHECK_TYPE_FP16_RELTOL = 3  # fp16 relative tol: |out-g| <= rtol*|g| + 0.05 (magnitude-scaled)
BINGO_CHECK_TYPE_INT8_TOL   = 4   # signed-int8 abs tol: |out-g| <= tol LSBs (quantized activation)


# Bytes per element for each check mode — used for validation and
# conversion between num_elements and data_size (bytes).
_CHECK_TYPE_ELEM_BYTES = {
    BINGO_CHECK_TYPE_BYTE_EXACT: 1,  # data_size IS the byte count
    BINGO_CHECK_TYPE_FP32_TOL:   4,
    BINGO_CHECK_TYPE_FP16_TOL:   2,
    BINGO_CHECK_TYPE_FP16_RELTOL: 2,  # fp16 elements, relative-tolerance compare
    BINGO_CHECK_TYPE_INT8_TOL:   1,   # one signed int8 per element
}


class HostBingoKernelCheckResultArgs(BingoKernelArgs):
    def __init__(self,
                 golden_data_addr: Union[BingoMemAlloc, int],
                 output_data_addr: Union[BingoMemAlloc, int],
                 data_size: Optional[int] = None,
                 name: str = "",
                 check_type: int = BINGO_CHECK_TYPE_BYTE_EXACT,
                 tolerance: float = 0.0,
                 num_elements: Optional[int] = None):
        """Args for __host_bingo_kernel_check_result.

        The C kernel always reads `data_size` in BYTES, then for fp modes it
        iterates over `data_size / elem_bytes` floating-point elements
        (elem_bytes = 4 for fp32, 2 for fp16, 1 for byte-exact).

        This Python constructor accepts EITHER `data_size` (bytes, the raw
        kernel-level value) OR `num_elements` (logical element count), but
        not both. `num_elements` is the preferred, unambiguous form for
        tolerance modes; `data_size` remains for back-compat with byte-exact
        call-sites.

        check_type:
            0 (BYTE_EXACT) = byte-exact comparison. data_size = byte count
                             OR num_elements = byte count (they're identical).
            1 (FP32_TOL)   = fp32 absolute tolerance: |out[i]-golden[i]| <= tolerance.
                             num_elements = fp32 element count (→ data_size = num_elements*4)
            2 (FP16_TOL)   = fp16 absolute tolerance (elements promoted to fp32 for compare).
                             num_elements = fp16 element count (→ data_size = num_elements*2)
        tolerance: absolute fp32 tolerance (only meaningful when check_type != 0).
                   For fp16 mode this is still fp32 — the C kernel promotes
                   fp16 to fp32 before comparing.

        Validates that exactly one of data_size/num_elements is given and that
        data_size is a whole multiple of the element size.
        """
        check_type = int(check_type)
        if check_type not in _CHECK_TYPE_ELEM_BYTES:
            raise ValueError(f"Unknown check_type={check_type}. Must be one of "
                             f"{list(_CHECK_TYPE_ELEM_BYTES.keys())}.")
        elem_bytes = _CHECK_TYPE_ELEM_BYTES[check_type]

        if (data_size is None) == (num_elements is None):
            raise ValueError(
                "Exactly one of `data_size` (bytes) or `num_elements` must be "
                "given. For tolerance modes, prefer `num_elements` for clarity."
            )
        if num_elements is not None:
            if num_elements <= 0:
                raise ValueError(f"num_elements must be positive, got {num_elements}")
            data_size = int(num_elements) * elem_bytes
        else:
            if data_size <= 0:
                raise ValueError(f"data_size must be positive, got {data_size}")
            if data_size % elem_bytes != 0:
                raise ValueError(
                    f"data_size={data_size} is not a multiple of elem_bytes="
                    f"{elem_bytes} for check_type={check_type}. This would "
                    f"cause the kernel's `data_size / elem_bytes` to silently "
                    f"truncate. Pass num_elements instead or fix data_size."
                )

        self.golden_data_addr = golden_data_addr
        self.output_data_addr = output_data_addr
        self.data_size = int(data_size)
        self.name = name
        self.check_type = check_type
        self.tolerance = float(tolerance)

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_check_result_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        import struct
        assignments = {}
        self._process_addr(self.golden_data_addr, "golden_data_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_data_addr, "output_data_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        assignments["data_size"] = str(self.data_size)
        if self.name:
            assignments["name_addr"] = f'(uint64_t)"{self.name}"'
        # Always emit — L3 alloc is not zeroed, so garbage in these fields could
        # flip check_type to 1/2 with random tolerance_bits.
        assignments["check_type"] = str(self.check_type)
        tol_bits = struct.unpack('<I', struct.pack('<f', self.tolerance))[0]
        assignments["tolerance_bits"] = f"0x{tol_bits:08x}"
        return assignments
    
# HOST BINGO XDMA 1D Copy
class HostBingoKernelXdma1dCopyArgs(BingoKernelArgs):
    """Args for __host_bingo_kernel_xdma_1d_copy.

    Runtime note: the host implementation currently waits on the remote xDMA
    completion counter only. Use this kernel for transfers that complete as
    remote xDMA tasks; same-local-memory transfers may hang unless the host
    kernel is changed to wait on the local completion counter.
    """

    def __init__(self,
                 src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 size: int):
        _check_xdma_size_aligned(size, "HostBingoKernelXdma1dCopyArgs")
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.size = size

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_xdma_1d_copy_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        assignments["size"] = str(self.size)
        return assignments

HostBingoKernelXdmaArgs = HostBingoKernelXdma1dCopyArgs

# HOST BINGO IDMA
class HostBingoKernelIdmaArgs(BingoKernelArgs):
    def __init__(self,
                 src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 size: int):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.size = size

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_idma_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        assignments["size"] = str(self.size)
        return assignments


# HOST BINGO scalar broadcast bridge: read `rows` per-row reduce scalars from L1
# (row r one splatted 64-B beat apart), compute a per-row scalar on the CVA6, and
# splat each fp16 result across a [rows, D] fp16 broadcast buffer for a downstream
# device StreamElementwise. op: 0=NEG (-x), 1=RECIP (1/x), 2=RSQRT_MEAN (1/sqrt(x/N)).
class HostBingoKernelRequantScaleArgs(BingoKernelArgs):
    """Per-tensor fp16->int8 requant scale: reads xmax,nmax (max(x), max(-x) fp16 scalars the xDMA
    StreamReduce(MAX) passes wrote to cluster L1) and writes scale = max|x|/127 (fp32 dequant qsc) +
    inv_scale = 127/max|x| (fp32, the xDMA fp16_to_int8 runtime CSR). Replaces the host quantize_f16i8
    (which streamed the whole tensor from L3); this reads/writes 2+2 scalars only."""
    KERNEL_NAME = "__host_bingo_kernel_requant_scale"

    def __init__(self,
                 xmax_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 nmax_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 scale_out_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 inv_scale_out_addr: Union[BingoMemAlloc, BingoMemSymbol, int]):
        self.xmax_addr = xmax_addr
        self.nmax_addr = nmax_addr
        self.scale_out_addr = scale_out_addr
        self.inv_scale_out_addr = inv_scale_out_addr

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_requant_scale_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.xmax_addr, "xmax_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.nmax_addr, "nmax_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.scale_out_addr, "scale_out_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.inv_scale_out_addr, "inv_scale_out_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        return a


# ══════════════════════════════════════════════════════════════════════
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
    """
    def __init__(self,
                 mode: int = BINGO_GATING_MODE_STATIC,
                 pred_scratchpad_addr=None,   # wired at emit time by compiler
                 cerf_controlled_mask: int = 0,
                 top_k_or_threshold: Union[int, float] = 0,
                 cerf_group_ids_addr=None,
                 cond_activation_addr=None):
        self.mode = mode
        self.pred_scratchpad_addr = pred_scratchpad_addr
        self.cerf_controlled_mask = cerf_controlled_mask
        self.top_k_or_threshold = top_k_or_threshold
        self.cerf_group_ids_addr = cerf_group_ids_addr
        self.cond_activation_addr = cond_activation_addr

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
