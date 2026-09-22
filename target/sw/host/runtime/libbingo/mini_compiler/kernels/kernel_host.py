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

from kernel_base import BingoKernelArgs


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
    own pull, and one transfer on it measures 56.4 B/cc. The catch is that a BINGO task on
    the host costs ~433 cc of dispatch, so one tile per task yields only 41.1 B/cc effective
    -- less than four concurrent cluster xDMAs already deliver. Batching four amortises that
    to 51.6 B/cc, which is what makes the second pipe worth using.

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
