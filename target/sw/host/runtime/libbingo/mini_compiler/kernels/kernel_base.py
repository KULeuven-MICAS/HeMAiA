# Fanchen Kong <fanchen.kong@kuleuven.be>
from abc import ABC, abstractmethod
from typing import Union, Dict, Optional, List, Tuple
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView, BingoMemSymbol, BingoMemFixedAddr

# THE LAYOUT CODES a kernel argument carries. The only other place they appear is
# device_kernel_args.h's BINGO_LAYOUT_*, and the two must agree: a mismatch would hand a
# kernel a layout it did not mean and the transfer would move the right byte count to the
# wrong offsets, which no byte count and no golden on random data would catch.
LAYOUT_CODE = {
    "row_major": 0,
    "col_major": 1,
    "A": 2,
    "B": 3,
    "D": 4,
}


class BingoKernelArgs(ABC):
    """
    Abstract base class for Kernel Arguments.
    Subclasses define the specific arguments for each kernel type and how they map to C structs.
    """

    # (earlier, later) attribute-name pairs constraining where this kernel's own buffers may
    # be placed relative to each other. Empty unless the kernel computes an address in one
    # buffer as an offset from another. See SnaxBingoKernelSimdFaSoftmaxArgs.
    PLACEMENT_ORDER: List[Tuple[str, str]] = []


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


