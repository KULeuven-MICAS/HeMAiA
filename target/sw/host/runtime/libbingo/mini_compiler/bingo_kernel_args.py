# Fanchen Kong <fanchen.kong@kuleuven.be>
from abc import ABC, abstractmethod
from typing import Union, Dict
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol, BingoMemFixedAddr

class BingoKernelArgs(ABC):
    """
    Abstract base class for Kernel Arguments.
    Subclasses define the specific arguments for each kernel type and how they map to C structs.
    """
    
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
                 data_size: int):
        self.golden_data_addr = golden_data_addr
        self.output_data_addr = output_data_addr
        self.data_size = data_size

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_check_result_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.golden_data_addr, "golden_data_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_data_addr, "output_data_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        assignments["data_size"] = str(self.data_size)
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