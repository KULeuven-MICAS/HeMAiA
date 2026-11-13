# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Host-side kernels: transfers the CVA6 issues, and the result checks.

The check_result family is one class per PRECISION rather than one class with an integer
mode, because the integer silently couples three decisions -- which comparison runs, what
`num_elements` counts, and whether `tolerance` is a distance or a ratio -- and getting any
of them wrong passes rather than faults."""

from typing import Union, Dict, Optional
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol
from bingo_helpers import _check_xdma_size_aligned

from kernel_base import Read, Write, ReadWrite, Sync, Hold, BingoKernelArgs
from kernel_xdma import xdma_softmax_stats_csrs


class HostBingoKernelDummyArgs(BingoKernelArgs):
    def __init__(self, dummy_input: int):
        self.dummy_input = dummy_input

    def accesses(self):
        return []

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_dummy_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        return {"dummy_input": str(self.dummy_input)}


class HostBingoKernelChipBarrierArgs(BingoKernelArgs):
    """Args for __host_bingo_kernel_chip_barrier, on the host core of every chip.

    op 0 (OP_INIT) zeroes the chip's barrier state for both mechanisms (this kernel's and
    __snax_bingo_kernel_chip_barrier's); every chip runs it, ordered before any chip's first
    barrier. op 1 (OP_BCAST) is one 0xFF broadcast barrier over top_left..bottom_right; seq is
    its 1-based index in the program (0: unchecked), at most 255 per run (8-bit checkpoint)."""
    KERNEL_NAME = "__host_bingo_kernel_chip_barrier"
    OP_INIT = 0
    OP_BCAST = 1

    def __init__(self, op: int, top_left: int = 0, bottom_right: int = 0, seq: int = 0):
        self.op = int(op)
        self.top_left = int(top_left)
        self.bottom_right = int(bottom_right)
        self.seq = int(seq)
        if self.op == self.OP_BCAST and self.seq > 255:
            raise ValueError(f"broadcast barrier {self.seq}: the checkpoint is 8-bit, 255 per run")

    def accesses(self):
        return []

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_chip_barrier_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        return {"op": str(self.op), "top_left": str(self.top_left),
                "bottom_right": str(self.bottom_right), "seq": str(self.seq)}


class HostBingoKernelFoldArgs(BingoKernelArgs):
    """Args for __host_bingo_kernel_fold: n_parts partials of part_bytes each, back to back
    at src_addr, folded on the host core into dst_addr. op TOPK keeps the 8 largest
    (value, index) pairs per 64 B beat; op SOFTMAX_STATS folds the softmax-statistics
    junction's 128 B groups (a key beat m[0..31], a value beat l[0..31], FP16) with the
    junction's arithmetic, part by part: rows 0..n_valid-1 folded, `scale` the factor on the
    key difference (xdma_softmax_stats_csrs' CSR(0) and CSR(1)). Both buffers should be
    host-cacheable memory on the host's own chiplet."""
    KERNEL_NAME = "__host_bingo_kernel_fold"
    TOPK = 0
    SOFTMAX_STATS = 1

    def __init__(self, op: int, src_addr, n_parts: int, part_bytes: int, dst_addr,
                 n_valid: int = 0, scale: float = 1.0):
        unit = 128 if op == self.SOFTMAX_STATS else 64
        if part_bytes % unit:
            raise ValueError(f"fold: part_bytes {part_bytes} is not a multiple of {unit}")
        self.op = int(op)
        self.src_addr = src_addr
        self.n_parts = int(n_parts)
        self.part_bytes = int(part_bytes)
        self.dst_addr = dst_addr
        self.n_valid, self.scale_f32bits = 0, 0
        if self.op == self.SOFTMAX_STATS:
            self.n_valid, self.scale_f32bits = xdma_softmax_stats_csrs(n_valid, scale)

    def accesses(self):
        return [Read("src_addr", self.n_parts * self.part_bytes), Write("dst_addr", self.part_bytes)]

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_fold_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {"op": str(self.op), "n_parts": str(self.n_parts), "part_bytes": str(self.part_bytes),
             "n_valid": str(self.n_valid), "scale_f32bits": f"0x{self.scale_f32bits:08x}u"}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map,
                           split_64bit=False, as_64bit=True)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map,
                           split_64bit=False, as_64bit=True)
        return a


# HOST BINGO Check Result
# Check-mode constants (mirror #defines in host_kernel_args.h)
BINGO_CHECK_TYPE_BYTE_EXACT = 0
BINGO_CHECK_TYPE_FP32_TOL   = 1
BINGO_CHECK_TYPE_FP16_TOL   = 2
BINGO_CHECK_TYPE_FP16_RELTOL = 3  # fp16 relative tol: |out-g| <= rtol*|g| + 0.05 (magnitude-scaled)
BINGO_CHECK_TYPE_INT8_TOL   = 4   # signed-int8 abs tol: |out-g| <= tol LSBs (quantized activation)
BINGO_CHECK_TYPE_INT32_RELTOL = 5 # signed-int32 rel tol: |out-g| <= rtol*|g| + 0.001*max|g| (accumulator)


# Bytes per element for each check mode — used for validation and
# conversion between num_elements and data_size (bytes).
_CHECK_TYPE_ELEM_BYTES = {
    BINGO_CHECK_TYPE_BYTE_EXACT: 1,  # data_size IS the byte count
    BINGO_CHECK_TYPE_FP32_TOL:   4,
    BINGO_CHECK_TYPE_FP16_TOL:   2,
    BINGO_CHECK_TYPE_FP16_RELTOL: 2,  # fp16 elements, relative-tolerance compare
    BINGO_CHECK_TYPE_INT8_TOL:   1,   # one signed int8 per element
    BINGO_CHECK_TYPE_INT32_RELTOL: 4,  # one signed int32 per element
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
            0 (BYTE_EXACT)   = byte-exact comparison. data_size = byte count
                               OR num_elements = byte count (they're identical).
            1 (FP32_TOL)     = fp32 absolute tolerance: |out[i]-golden[i]| <= tolerance.
                               num_elements = fp32 element count (→ data_size = num_elements*4)
            2 (FP16_TOL)     = fp16 absolute tolerance (elements promoted to fp32 for compare).
                               num_elements = fp16 element count (→ data_size = num_elements*2)
            3 (FP16_RELTOL)  = fp16 relative tolerance: |out-g| <= tolerance*|g| + 0.05.
                               num_elements = fp16 element count (→ data_size = num_elements*2)
            4 (INT8_TOL)     = signed int8, |out-g| <= tolerance LSBs.
                               num_elements = int8 element count (→ data_size = num_elements)
            5 (INT32_RELTOL) = signed int32 accumulator: |out-g| <= tolerance*|g| + 0.001*max|g|.
                               num_elements = int32 element count (→ data_size = num_elements*4)
        tolerance: an absolute distance for 1, 2 and 4 (LSBs for 4), a ratio (rtol) for 3
                   and 5, unused for 0. It is always passed as fp32 — the C kernel
                   promotes fp16 to fp32 before comparing.

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

    def accesses(self):
        return [Read("golden_data_addr", self.data_size), Read("output_data_addr", self.data_size)]

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_check_result_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        import struct
        assignments = {}
        self._process_addr(self.golden_data_addr, "golden_data_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.output_data_addr, "output_data_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        assignments["data_size"] = str(self.data_size)
        # 0 means "no label": the host kernel prints no name for it
        assignments["name_addr"] = f'(uint64_t)"{self.name}"' if self.name else "0"
        # Always emit — L3 alloc is not zeroed, so garbage in these fields could
        # flip check_type to 1/2 with random tolerance_bits.
        assignments["check_type"] = str(self.check_type)
        tol_bits = struct.unpack('<I', struct.pack('<f', self.tolerance))[0]
        assignments["tolerance_bits"] = f"0x{tol_bits:08x}"
        return assignments
    
# ======================================================================================
# check_result, one class per precision
# ======================================================================================
# WHY THESE EXIST. HostBingoKernelCheckResultArgs selects its comparison with an integer,
# and the integer silently couples three decisions: which comparison runs, what
# `num_elements` COUNTS (int32s, fp16s or bytes), and whether `tolerance` is an absolute
# distance or a ratio. Get any of the three wrong and nothing faults -- the wrong element
# size compares a prefix of the buffer and passes, and an absolute tolerance passed where a
# ratio was wanted passes everything.
#
# So the precision picks the class, and the class fixes the integer and names the tolerance
# after what the kernel does with it: `tol` where the comparison is absolute, `rtol` where
# it is a ratio. The base class stays for the rare call site that genuinely computes its
# check_type; everything else should name one of these.

class _CheckResultTyped(HostBingoKernelCheckResultArgs):
    """Shared plumbing: a fixed check_type and a single tolerance argument."""
    CHECK_TYPE = BINGO_CHECK_TYPE_BYTE_EXACT

    def __init__(self, golden_data_addr, output_data_addr, num_elements,
                 tolerance=0.0, name=""):
        super().__init__(golden_data_addr, output_data_addr, name=name,
                         check_type=self.CHECK_TYPE, tolerance=tolerance,
                         num_elements=num_elements)


class HostBingoKernelCheckResultBytesArgs(_CheckResultTyped):
    """Byte-exact. `num_bytes` counts BYTES, and there is no tolerance.

    For anything whose bits must be reproduced exactly -- a copy, a layout conversion, a
    re-read of a buffer that should not have changed. A tolerance mode would accept a
    conversion that dropped the low bits of every element.
    """
    CHECK_TYPE = BINGO_CHECK_TYPE_BYTE_EXACT

    def __init__(self, golden_data_addr, output_data_addr, num_bytes, name=""):
        super().__init__(golden_data_addr, output_data_addr, num_bytes, name=name)


class HostBingoKernelCheckResultI8Args(_CheckResultTyped):
    """Signed int8, ABSOLUTE tolerance in LSBs. For a quantised activation.

    `tol` counts quantisation steps, not a fraction: tol=1 accepts a value that rounded
    the other way.
    """
    CHECK_TYPE = BINGO_CHECK_TYPE_INT8_TOL

    def __init__(self, golden_data_addr, output_data_addr, num_elements, tol=0.0, name=""):
        super().__init__(golden_data_addr, output_data_addr, num_elements, tol, name)


class HostBingoKernelCheckResultF16Args(_CheckResultTyped):
    """FP16, ABSOLUTE tolerance. Elements are promoted to fp32 to compare, so `tol` is an
    fp32 distance even though the buffer is fp16.

    Right where the quantity has a known scale -- a running max, a row sum. Wrong across a
    tensor spanning orders of magnitude, where one `tol` is strict at the bottom and
    vacuous at the top; use the Rel variant there.
    """
    CHECK_TYPE = BINGO_CHECK_TYPE_FP16_TOL

    def __init__(self, golden_data_addr, output_data_addr, num_elements, tol, name=""):
        super().__init__(golden_data_addr, output_data_addr, num_elements, tol, name)


class HostBingoKernelCheckResultF16RelArgs(_CheckResultTyped):
    """FP16, RELATIVE: |out - g| <= rtol*|g| + 0.05.

    The additive 0.05 is the kernel's own floor, not a parameter -- it keeps a golden of
    exactly zero from demanding a bit-exact zero back. That floor is also why this is not a
    drop-in for the absolute variant on small-magnitude data: below ~0.05 it accepts
    anything.
    """
    CHECK_TYPE = BINGO_CHECK_TYPE_FP16_RELTOL

    def __init__(self, golden_data_addr, output_data_addr, num_elements, rtol, name=""):
        super().__init__(golden_data_addr, output_data_addr, num_elements, rtol, name)


class HostBingoKernelCheckResultF32Args(_CheckResultTyped):
    """FP32, ABSOLUTE tolerance. `num_elements` counts fp32 words, so the byte count is 4x."""
    CHECK_TYPE = BINGO_CHECK_TYPE_FP32_TOL

    def __init__(self, golden_data_addr, output_data_addr, num_elements, tol, name=""):
        super().__init__(golden_data_addr, output_data_addr, num_elements, tol, name)


class HostBingoKernelCheckResultI32Args(_CheckResultTyped):
    """Signed int32, RELATIVE: |out - g| <= rtol*|g| + 0.001*max|g|.

    The accumulator compare. An int32 GEMM accumulator spans several orders of magnitude
    within one tile, so an absolute tolerance chosen for the large entries is meaningless
    for the small ones and vice versa. The 0.001*max|g| term is the kernel's floor for
    entries near zero, and is not a parameter.
    """
    CHECK_TYPE = BINGO_CHECK_TYPE_INT32_RELTOL

    def __init__(self, golden_data_addr, output_data_addr, num_elements, rtol, name=""):
        super().__init__(golden_data_addr, output_data_addr, num_elements, rtol, name)


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

    def accesses(self):
        return [Read("src_addr", self.size), Write("dst_addr", self.size)]

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
class HostBingoKernelIdmaMultiArgs(BingoKernelArgs):
    """Up to four host iDMA transfers in ONE BINGO task, all the same size.

    The system iDMA is a second 512-bit path into the quadrant, disjoint from the clusters'
    own pull. Batching up to four transfers in one task amortises the host's per-task
    dispatch cost, which is what makes the second pipe worth using.

    Every transfer in a batch shares the task's dependencies, so batching across clusters
    couples them: the batch cannot start until every source dependency is met. That is the
    price, and it is only worth paying while the clusters run in step."""
    KERNEL_NAME = "__host_bingo_kernel_idma_multi"
    MAX_N = 4

    def __init__(self, pairs, size: int):
        pairs = list(pairs)
        if not 1 <= len(pairs) <= self.MAX_N:
            raise ValueError(f"batch of {len(pairs)} transfers; the struct holds "
                             f"1..{self.MAX_N}")
        self.pairs = pairs
        self.size = size

    def accesses(self):
        return [ReadWrite("pairs", self.size)]

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_idma_multi_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {"n": str(len(self.pairs)), "size": str(self.size)}
        for i in range(self.MAX_N):
            # Unused slots are pinned to 0. The kernel reads all four into locals before
            # looking at n, so they must be well-defined even when never used.
            src, dst = self.pairs[i] if i < len(self.pairs) else (0, 0)
            self._process_addr(src, f"src_addr{i}", a, handle_name_map,
                               split_64bit=False, as_64bit=True)
            self._process_addr(dst, f"dst_addr{i}", a, handle_name_map,
                               split_64bit=False, as_64bit=True)
        return a


class HostBingoKernelHbmBcastArgs(BingoKernelArgs):
    """An HBM multicast as ChipletBuild.hbm_cast takes it, which posts it (engine_post) and waits for it (wait_flag):
    the memory chiplet's push engine multicasts n_tiles tiles from its HBM into every compute chiplet's L3, each
    followed by its flag (64 B apart). A description, not a task's arguments."""

    def __init__(self, src_addr, dst_addr, flag_addr, value_addr, value: int, tile_bytes: int,
                 n_tiles: int, memchip_id: int, engine: int):
        if not 0 <= int(engine) <= 3:
            raise ValueError(f"engine={engine}: 0..3, one of the memchip's push engines")
        if int(n_tiles) < 1 or int(tile_bytes) < 1:
            raise ValueError(f"n_tiles={n_tiles}, tile_bytes={tile_bytes}: both >= 1")
        self.src_addr, self.dst_addr, self.flag_addr = src_addr, dst_addr, flag_addr
        self.value_addr, self.value = value_addr, int(value)
        self.tile_bytes, self.n_tiles = int(tile_bytes), int(n_tiles)
        self.memchip_id, self.engine = int(memchip_id), int(engine)

    def accesses(self):
        n = self.tile_bytes * self.n_tiles
        return [Read("src_addr", n), Write("dst_addr", n), Sync("flag_addr"), Read("value_addr", 8)]

    def get_struct_name(self) -> str:
        raise TypeError(f"{type(self).__name__} describes an HBM cast for ChipletBuild.hbm_cast, not a task's arguments")

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        raise TypeError(f"{type(self).__name__} describes an HBM cast for ChipletBuild.hbm_cast, not a task's arguments")


class HostBingoKernelHbmUcastArgs(BingoKernelArgs):
    """An HBM unicast as ChipletBuild.hbm_cast takes it, the multicast's conventional counterpart: the memory chiplet's
    push engine writes the n_tiles tiles and their flags into each chiplet of `chips` in turn (one copy a chiplet), at
    the same addresses as the multicast, plus a progress flag a tile into `prog_addr` (64 B apart, n_tiles x len(chips)
    of them) on the issuing host's chiplet. A description, not a task's arguments."""

    def __init__(self, src_addr, dst_addr, flag_addr, prog_addr, value_addr, value: int,
                 tile_bytes: int, n_tiles: int, memchip_id: int, engine: int, chips):
        if not 0 <= int(engine) <= 3:
            raise ValueError(f"engine={engine}: 0..3, one of the memchip's push engines")
        if int(n_tiles) < 1 or int(tile_bytes) < 1:
            raise ValueError(f"n_tiles={n_tiles}, tile_bytes={tile_bytes}: both >= 1")
        chips = [int(c) for c in chips]
        if not 1 <= len(chips) <= 16 or any(not 0 <= c <= 0xFE for c in chips):
            raise ValueError(f"chips={chips}: 1..16 chiplet ids")
        self.src_addr, self.dst_addr, self.flag_addr, self.prog_addr = src_addr, dst_addr, flag_addr, prog_addr
        self.value_addr, self.value = value_addr, int(value)
        self.tile_bytes, self.n_tiles = int(tile_bytes), int(n_tiles)
        self.memchip_id, self.engine = int(memchip_id), int(engine)
        self.chips = chips

    def accesses(self):
        n = self.tile_bytes * self.n_tiles
        return [Read("src_addr", n), Write("dst_addr", n), Sync("flag_addr"), Read("prog_addr"),
                Read("value_addr", 8)]

    def get_struct_name(self) -> str:
        raise TypeError(f"{type(self).__name__} describes an HBM cast for ChipletBuild.hbm_cast, not a task's arguments")

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        raise TypeError(f"{type(self).__name__} describes an HBM cast for ChipletBuild.hbm_cast, not a task's arguments")



class HostBingoKernelEnginePostArgs(BingoKernelArgs):
    """One post into the descriptor queue of push engine `engine` of memory chip `memchip`, from a
    host as a task of its own (host_kernel_lib.h engine_post; ChipletBuild.hbm_cast): `entries`,
    [(src, dst, bytes)], queued in that order (an HBM multicast's tiles, each followed by its flag;
    an unicast's tiles into one chip, then a progress flag on the posting chip). The host's system
    iDMA writes each entry as one burst."""
    KERNEL_NAME = "__host_bingo_kernel_engine_post"
    MAX_ENTRIES = 32                    # host_kernel_args.h HOST_ENGINE_POST_MAX: the queue's depth

    def __init__(self, memchip: int, engine: int, entries):
        if not 0 <= int(engine) <= 3:
            raise ValueError(f"engine={engine}: 0..3, one of the memchip's push engines")
        entries = [(src, dst, int(nb)) for src, dst, nb in entries]
        if not 1 <= len(entries) <= self.MAX_ENTRIES or any(nb <= 0 for _, _, nb in entries):
            raise ValueError(f"engine post of {len(entries)} entries: 1..{self.MAX_ENTRIES}, each of >= 1 B")
        self.memchip, self.engine, self.entries = int(memchip), int(engine), entries

    def accesses(self):
        # the sources in HBM, the destinations and their flags: the graph's edges and the flags order them
        return [Sync("entries")]

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_engine_post_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        # the struct holds the entries as the queue window takes them (host_kernel_args.h)
        a = {"memchip": f"0x{self.memchip:02x}", "engine": str(self.engine), "n_entries": str(len(self.entries))}
        for i, (src, dst, nb) in enumerate(self.entries):
            self._process_addr(src, f"e[{i}].src", a, handle_name_map, split_64bit=False, as_64bit=True)
            self._process_addr(dst, f"e[{i}].dst", a, handle_name_map, split_64bit=False, as_64bit=True)
            a[f"e[{i}].len"], a[f"e[{i}].bell"] = str(nb), "0"
        return a


class HostBingoKernelWaitFlagArgs(BingoKernelArgs):
    """A host waits until the 4-byte flag at `flag_addr`, in its chiplet's L3, holds `value`
    (host_kernel_lib.h wait_flag): the landing of a tile an engine post sent."""
    KERNEL_NAME = "__host_bingo_kernel_wait_flag"

    def __init__(self, flag_addr, value: int = 1):
        self.flag_addr, self.value = flag_addr, int(value)

    def accesses(self):
        return [Sync("flag_addr")]

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_wait_flag_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.flag_addr, "flag_addr", a, handle_name_map, split_64bit=False, as_64bit=True)
        a["value"] = str(self.value)
        return a


class HostBingoKernelIdmaArgs(BingoKernelArgs):
    def __init__(self,
                 src_addr: Union[BingoMemAlloc, int],
                 dst_addr: Union[BingoMemAlloc, int],
                 size: int):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.size = size

    def accesses(self):
        return [Read("src_addr", self.size), Write("dst_addr", self.size)]

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_idma_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.src_addr, "src_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map, split_64bit=False, as_64bit=True)
        assignments["size"] = str(self.size)
        return assignments


class HostBingoKernelRequantScaleArgs(BingoKernelArgs):
    """Per-tensor fp16->int8 requant scale: reads xmax,nmax (max(x), max(-x) fp16 scalars that
    two MAX reduce passes wrote to cluster L1) and writes scale = max|x|/127 (fp32 dequant qsc) +
    inv_scale = 127/max|x| (fp32, the runtime inv_scale of simd_fp16_to_int8). It reads and
    writes 2+2 scalars only, not the tensor."""
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

    def accesses(self):
        return [Read("xmax_addr"), Read("nmax_addr"), Write("scale_out_addr"), Write("inv_scale_out_addr")]

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
