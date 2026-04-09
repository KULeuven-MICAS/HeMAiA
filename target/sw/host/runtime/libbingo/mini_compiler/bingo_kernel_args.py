# Fanchen Kong <fanchen.kong@kuleuven.be>
from abc import ABC, abstractmethod
from typing import Union, Dict
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol, BingoMemFixedAddr

class BingoKernelArgs(ABC):
    """
    Abstract base class for Kernel Arguments.
    Subclasses define the specific arguments for each kernel type and how they map to C structs.
    """

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
                 accumPrevC: int):
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

# BINGO XDMA 1D Copy
class SnaxBingoKernelXdma1dCopyArgs(BingoKernelArgs):
    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int], size: int):
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


# ══════════════════════════════════════════════════════════════════════
# VersaCore blocked-layout conversion kernels (tile-shape-parameterized)
#
# Six primitive conversions between row-major and the three VersaCore
# blocked layouts {A, B, D}. All kernels take tile dimensions (M_T, K_T,
# N_T) and array-shape dims (meshRow, tileSize, meshCol) so they work
# for any DSE-chosen tiling. See HeMAiA/util/sim/layout_convert.py for
# the Python reference.
# ══════════════════════════════════════════════════════════════════════


class SnaxBingoKernelXdmaDToRowMajorArgs(BingoKernelArgs):
    """D-layout → row-major. D[m,n,r,c] -> R[m*meshRow+r, n*meshCol+c]."""
    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, N_T: int, meshRow: int, meshCol: int,
                 elem_bytes: int = 1):
        self.src_addr = src_addr; self.dst_addr = dst_addr
        self.M_T = M_T; self.N_T = N_T
        self.meshRow = meshRow; self.meshCol = meshCol
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_d_to_row_major_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T); a["N_T"] = str(self.N_T)
        a["meshRow"] = str(self.meshRow); a["meshCol"] = str(self.meshCol)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


class SnaxBingoKernelXdmaRowMajorToAArgs(BingoKernelArgs):
    """row-major → A-layout. R[i,j] -> A[i/meshRow, j/tileSize, i%meshRow, j%tileSize]."""
    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, K_T: int, meshRow: int, tileSize: int,
                 elem_bytes: int = 1):
        self.src_addr = src_addr; self.dst_addr = dst_addr
        self.M_T = M_T; self.K_T = K_T
        self.meshRow = meshRow; self.tileSize = tileSize
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_row_major_to_a_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T); a["K_T"] = str(self.K_T)
        a["meshRow"] = str(self.meshRow); a["tileSize"] = str(self.tileSize)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


class SnaxBingoKernelXdmaRowMajorToBArgs(BingoKernelArgs):
    """row-major → B-layout. R[i,j] -> B[j/meshCol, i/tileSize, j%meshCol, i%tileSize]."""
    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 K_T: int, N_T: int, tileSize: int, meshCol: int,
                 elem_bytes: int = 1):
        self.src_addr = src_addr; self.dst_addr = dst_addr
        self.K_T = K_T; self.N_T = N_T
        self.tileSize = tileSize; self.meshCol = meshCol
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_row_major_to_b_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["K_T"] = str(self.K_T); a["N_T"] = str(self.N_T)
        a["tileSize"] = str(self.tileSize); a["meshCol"] = str(self.meshCol)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


class SnaxBingoKernelXdmaAToRowMajorArgs(BingoKernelArgs):
    """A-layout → row-major (inverse of row_major_to_a)."""
    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, K_T: int, meshRow: int, tileSize: int,
                 elem_bytes: int = 1):
        self.src_addr = src_addr; self.dst_addr = dst_addr
        self.M_T = M_T; self.K_T = K_T
        self.meshRow = meshRow; self.tileSize = tileSize
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_a_to_row_major_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T); a["K_T"] = str(self.K_T)
        a["meshRow"] = str(self.meshRow); a["tileSize"] = str(self.tileSize)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


class SnaxBingoKernelXdmaBToRowMajorArgs(BingoKernelArgs):
    """B-layout → row-major (inverse of row_major_to_b)."""
    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 K_T: int, N_T: int, tileSize: int, meshCol: int,
                 elem_bytes: int = 1):
        self.src_addr = src_addr; self.dst_addr = dst_addr
        self.K_T = K_T; self.N_T = N_T
        self.tileSize = tileSize; self.meshCol = meshCol
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_b_to_row_major_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["K_T"] = str(self.K_T); a["N_T"] = str(self.N_T)
        a["tileSize"] = str(self.tileSize); a["meshCol"] = str(self.meshCol)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


class SnaxBingoKernelXdmaRowMajorToDArgs(BingoKernelArgs):
    """row-major → D-layout (inverse of d_to_row_major)."""
    def __init__(self, src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 M_T: int, N_T: int, meshRow: int, meshCol: int,
                 elem_bytes: int = 1):
        self.src_addr = src_addr; self.dst_addr = dst_addr
        self.M_T = M_T; self.N_T = N_T
        self.meshRow = meshRow; self.meshCol = meshCol
        self.elem_bytes = elem_bytes

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_row_major_to_d_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["M_T"] = str(self.M_T); a["N_T"] = str(self.N_T)
        a["meshRow"] = str(self.meshRow); a["meshCol"] = str(self.meshCol)
        a["elem_bytes"] = str(self.elem_bytes)
        return a


# HOST BINGO DUMMY
class HostBingoKernelDummyArgs(BingoKernelArgs):
    def __init__(self, dummy_input: int):
        self.dummy_input = dummy_input

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_dummy_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        return {"dummy_input": str(self.dummy_input)}

# HOST BINGO Check Result
class HostBingoKernelCheckResultArgs(BingoKernelArgs):
    def __init__(self,
                 golden_data_addr: Union[BingoMemAlloc, int],
                 output_data_addr: Union[BingoMemAlloc, int],
                 data_size: int,
                 name: str = ""):
        self.golden_data_addr = golden_data_addr
        self.output_data_addr = output_data_addr
        self.data_size = data_size
        self.name = name

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_check_result_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.golden_data_addr, "golden_data_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_data_addr, "output_data_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        assignments["data_size"] = str(self.data_size)
        if self.name:
            assignments["name_addr"] = f'(uint64_t)"{self.name}"'
        return assignments
    
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


# HOST BINGO FP32 Quantize (FP32 -> INT8)
class HostBingoKernelFp32QuantizeArgs(BingoKernelArgs):
    def __init__(self,
                 input_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 output_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 scale_out_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 num_elements: int):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.scale_out_addr = scale_out_addr
        self.num_elements = num_elements

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_fp32_quantize_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.input_addr, "input_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr, "output_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.scale_out_addr, "scale_out_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        assignments["num_elements"] = str(self.num_elements)
        return assignments


# HOST BINGO INT32 Dequantize (INT32 -> FP32)
class HostBingoKernelInt32DequantizeArgs(BingoKernelArgs):
    def __init__(self,
                 input_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 output_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 scale_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 num_elements: int):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.scale_addr = scale_addr
        self.num_elements = num_elements

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_int32_dequantize_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.input_addr, "input_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr, "output_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.scale_addr, "scale_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        assignments["num_elements"] = str(self.num_elements)
        return assignments


# HOST BINGO INT32 elementwise add
# For inter-cluster partial-D accumulation in K-split GEMM schemes.
class HostBingoKernelInt32AddArgs(BingoKernelArgs):
    def __init__(self,
                 input_a_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 input_b_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 output_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 num_elements: int):
        self.input_a_addr = input_a_addr
        self.input_b_addr = input_b_addr
        self.output_addr = output_addr
        self.num_elements = num_elements

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_int32_add_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_a_addr, "input_a_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.input_b_addr, "input_b_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr, "output_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        a["num_elements"] = str(self.num_elements)
        return a


# HOST BINGO FP32 Softmax (row-wise along last dim, Ara RVV)
class HostBingoKernelFp32SoftmaxArgs(BingoKernelArgs):
    def __init__(self,
                 input_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 output_addr: Union[BingoMemAlloc, BingoMemSymbol, int],
                 num_rows: int,
                 row_length: int):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.num_rows = num_rows
        self.row_length = row_length

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_fp32_softmax_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.input_addr, "input_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_addr, "output_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        assignments["num_rows"] = str(self.num_rows)
        assignments["row_length"] = str(self.row_length)
        return assignments


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
