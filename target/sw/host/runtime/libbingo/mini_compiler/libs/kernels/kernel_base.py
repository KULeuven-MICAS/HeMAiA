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


# ---- ACCESSES: what a kernel does to the memory its address fields name ----------------------------------------------
READ, WRITE, READWRITE, SYNC, HOLD = "read", "write", "readwrite", "sync", "hold"


class Access:
    """One address field of a kernel's args and what the kernel does there:

        READ / WRITE / READWRITE   `nbytes` from `offset` bytes past the address (negative: before it; None: to the
                                   end of the buffer);
                                   with `rows`, that many runs of `nbytes`, `pitch` bytes apart (a strided access:
                                   the columns of a row-major block, which another task's columns sit between); with
                                   `dims`, a nest of runs: ((count, stride), ...) from the outermost dimension in
        SYNC                       memory a runtime protocol orders, not the graph: a flag polled or set, a stamp, a
                                   weight ring its posts fill behind its flags
        HOLD                       a buffer the task keeps live without touching it

    A field that holds a list (or tuple) applies to every address in it."""
    __slots__ = ("field", "mode", "nbytes", "offset", "dims")

    def __init__(self, field: str, mode: str, nbytes: Optional[int] = None, offset: int = 0, rows: int = 1,
                 pitch: int = 0, dims=()):
        if mode not in (READ, WRITE, READWRITE, SYNC, HOLD):
            raise ValueError(f"access mode {mode!r}")
        if nbytes is not None and int(nbytes) < 0:
            raise ValueError(f"{field}: {nbytes} bytes")
        dims = tuple((int(c), int(s)) for c, s in dims if int(c) > 1)
        if int(rows) > 1:
            dims = ((int(rows), int(pitch)),) + dims
        if any(c < 1 for c, _ in dims) or (dims and nbytes is None):
            raise ValueError(f"{field}: runs of {nbytes} bytes in the nest {dims}")
        self.field, self.mode = field, mode
        self.nbytes = None if nbytes is None else int(nbytes)
        self.offset = int(offset)
        self.dims = dims

    def __repr__(self):
        return (f"{self.mode}({self.field}, {self.nbytes}" + (f", +{self.offset}" if self.offset else "")
                + (f", {self.dims}" if self.dims else "") + ")")


def Read(field: str, nbytes: Optional[int] = None, offset: int = 0, rows: int = 1, pitch: int = 0, dims=()) -> Access:
    return Access(field, READ, nbytes, offset, rows, pitch, dims)


def Write(field: str, nbytes: Optional[int] = None, offset: int = 0, rows: int = 1, pitch: int = 0, dims=()) -> Access:
    return Access(field, WRITE, nbytes, offset, rows, pitch, dims)


def ReadWrite(field: str, nbytes: Optional[int] = None, offset: int = 0, rows: int = 1, pitch: int = 0,
              dims=()) -> Access:
    return Access(field, READWRITE, nbytes, offset, rows, pitch, dims)


def Sync(field: str) -> Access:
    return Access(field, SYNC)


def Hold(field: str) -> Access:
    return Access(field, HOLD)


# ---- SPLIT: a task as parts on its accelerator ----------------------------------------------------------------------
KEEPS = "keeps"                             # config_key(): the task leaves its accelerator's configuration as it was


class Split:
    """How a task of this kernel runs as parts on its cluster's accelerator, a CSR manager and a streamer (the graph's
    async_exec; passes/bingo_dfg_schedule.py ASYNCHRONOUS EXECUTION):

        accel       the core role whose accelerator the kernel programs: "gemm", "simd", "xdma"
        configure   (kernel, args) of a task that writes the configuration and starts nothing, or None
        start       start(configured) -> (kernel, args) of the task that starts the work: configured, after the
                    configure task; otherwise it configures too (None: there is no such kernel, the configure stays)
        wait        (kernel, args) of the task that returns once the work is done, or None: the start drains itself
        counter     the wait compares the accelerator's retired count with the dispatch's number (an args field
                    `seq` of every part): the compiler numbers each accelerator's dispatches in its core's program,
                    every one its counter counts (dispatch_accel()), a blocking one too
        depth       works started and not yet waited for, at most
        exclusive   no task of another kernel runs on the accelerator while one of these is in flight
        drain       drain(configured) -> (kernel, args) of a start that waits for its own work, used where the
                    schedule leaves the wait right behind the start, or None
        names       suffixes of the configure, start and wait tasks' names (the start keeps the task)
        prologue    (key, nbytes, init(dst) -> (kernel, args), field) or None: a task the accelerator's configuring
                    parts share, one a cluster and key, writing `nbytes` of its L1 at dst, which they read at `field`
    """
    __slots__ = ("accel", "configure", "start", "wait", "drain", "counter", "depth", "exclusive", "names",
                 "prologue")

    def __init__(self, accel: str, *, start, configure=None, wait=None, drain=None, counter: bool = False,
                 depth: int = 1, exclusive: bool = False, names=("_cfg", "", "_w"), prologue=None):
        self.accel, self.configure, self.start, self.wait, self.drain = accel, configure, start, wait, drain
        self.counter, self.depth, self.exclusive = bool(counter), int(depth), bool(exclusive)
        self.names, self.prologue = tuple(names), prologue


class BingoKernelArgs(ABC):
    """
    Abstract base class for Kernel Arguments.
    Subclasses define the specific arguments for each kernel type and how they map to C structs.

    Every kernel declares accesses(): each address field and what the kernel does there. The compiler checks with it
    that the graph orders every pair of tasks of which one writes bytes the other touches (passes/bingo_dfg_hazards.py).

    A kernel driven through a CSR manager and a streamer may also declare how it runs in parts (split()), whether its
    accelerator's retired counter counts it (dispatch_accel()), and the configuration it leaves in the CSRs
    (config_key()) with the cheaper task that reuses it (config_reuse()): the compiler's asynchronous execution and
    CSR reuse work from these alone (passes/bingo_dfg_schedule.py).
    """

    def split(self) -> Optional["Split"]:
        """This task as parts on its accelerator (Split), or None: it runs as one task."""
        return None

    def dispatch_accel(self) -> Optional[str]:
        """The accelerator whose retired counter counts this task's work ("gemm"), or None."""
        return None

    def config_key(self):
        """The configuration this task leaves in its accelerator's CSRs (hashable), KEEPS when it writes none, None
        when it changes them in a way not described (the next task cannot reuse them)."""
        return None

    def config_reuse(self):
        """(kernel, args) of the task that does this one's work when the CSRs already hold its config_key(), or None."""
        return None

    def accesses(self) -> List[Access]:
        """Every address field of these args and what the kernel does there (Read, Write, ReadWrite with the bytes from
        the address; Sync; Hold). Each kernel declares it; there is no default."""
        raise NotImplementedError(f"{type(self).__name__} declares no accesses(): every kernel states what it reads "
                                  f"and writes at each address field (kernel_base.py Access)")

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
            if getattr(val, "chip_id", None) is not None:
                # Another chip's copy: its chip id over the symbol's local 40-bit address.
                final_expr = (f"chiplet_addr_transform_full(0x{val.chip_id:02x}, "
                              f"((uint64_t)({base_expr})) & 0x000000ffffffffffULL)")

            # 3. Cast to final destination type/width
            if split_64bit:
                assignments[f"{base_name}_lo"] = f"(uint32_t)({final_expr})"
                assignments[f"{base_name}_hi"] = f"(uint32_t)(({final_expr}) >> 32)"
            elif as_64bit:
                assignments[base_name] = f"(uint64_t)({final_expr})"
            else:
                assignments[base_name] = f"(uint32_t)({final_expr})"
        # A LITERAL IS WIDENED BEFORE IT IS SHIFTED. `0` or `0x1000` is a 32-bit int in C,
        # so `(0 >> 32)` is undefined behaviour and -Werror=shift-count-overflow rejects the
        # host build; a literal is cast to uint64_t before it is shifted. The handle branches
        # above are uint64_t already.
        elif isinstance(val, BingoMemFixedAddr):
            addr = val.address
            if split_64bit:
                assignments[f"{base_name}_lo"] = f"(uint32_t)0x{addr:x}"
                assignments[f"{base_name}_hi"] = f"(uint32_t)((uint64_t)0x{addr:x} >> 32)"
            elif as_64bit:
                assignments[base_name] = f"(uint64_t)0x{addr:x}"
            else:
                assignments[base_name] = f"(uint32_t)0x{addr:x}"
        else:
            if split_64bit:
                assignments[f"{base_name}_lo"] = f"(uint32_t){val}"
                assignments[f"{base_name}_hi"] = f"(uint32_t)((uint64_t){val} >> 32)"
            elif as_64bit:
                assignments[base_name] = f"(uint64_t){val}"
            else:
                assignments[base_name] = f"(uint32_t){val}"


