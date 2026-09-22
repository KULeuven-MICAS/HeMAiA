# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""xDMA transfers and the in-fabric junctions.

Copies, multicast, memset, the general 6-dimensional AGU transfer, and the 2-D shape ops.
Then the junction CSR builders: an xDMA writer can fold data AS IT CROSSES THE FABRIC, and
what it computes is programmed through one packed CSR word -- which is why those are
functions that build a number rather than classes."""

from typing import Union, Dict
from bingo_mem_handle import BingoMemAlloc
from bingo_helpers import _check_xdma_size_aligned

from kernel_base import BingoKernelArgs


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


class SnaxBingoKernelXdmaMulticastArgs(BingoKernelArgs):
    """Args for __snax_bingo_kernel_xdma_multicast: one read, N destination clusters.

    The destinations are ordinary handles. A handle allocated on another cluster already
    resolves to a full (chip | cluster | offset) address, so nothing here has to know the
    cluster map or assume that the four heaps lay out the same -- the same property the
    chain gather relies on.
    """
    # Must match BINGO_XDMA_MCAST_MAX in device_kernel_args.h.
    DST_MAX = 8

    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_list: list, size: int):
        if not 1 <= len(dst_list) <= self.DST_MAX:
            raise ValueError(
                f"xdma multicast: {len(dst_list)} destinations; it must be 1.."
                f"{self.DST_MAX}. The hardware bound XDMA_MAX_DST_COUNT is generated "
                "per cfg and is checked on the device at call time.")
        _check_xdma_size_aligned(size, "SnaxBingoKernelXdmaMulticastArgs")
        self.src_addr = src_addr
        self.dst_list = list(dst_list)
        self.size = size

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_multicast_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        for i, dst in enumerate(self.dst_list):
            # _process_addr emits <base>_hi / <base>_lo and the base is a C lvalue, so an
            # array element indexes cleanly -- same trick the chain gather uses.
            tmp = {}
            self._process_addr(dst, "dst", tmp, handle_name_map)
            a[f"dst_hi[{i}]"] = tmp["dst_hi"]
            a[f"dst_lo[{i}]"] = tmp["dst_lo"]
        a["dst_num"] = str(len(self.dst_list))
        a["size"] = str(self.size)
        return a

# BINGO XDMA memset: fill local L1 with a repeating 32-bit pattern, on the writer path.
class SnaxBingoKernelXdmaMemsetArgs(BingoKernelArgs):
    """Args for __snax_bingo_kernel_xdma_memset.

    Generates a constant into L1 instead of loading it from main memory. FlashAttention
    seeds m to -inf and l to 0 and zeroes the O accumulator; those are CONSTANTS, and
    fetching them over the iDMA puts NQ separate transfers in front of K(0) on the load
    engine's critical head. The xDMA core is otherwise idle -- on HeMAiA it owns nothing
    but its exit node -- so it generates them in place.

    The pattern is 32 bits, not a byte, because FP16 -inf is 0xFBFF and no single byte
    repeats into it. One mechanism then covers INT8, FP16, BF16, FP32 and INT32.
    """

    # Handy patterns. FP16 is packed twice into the 32-bit word, so a beat of FP16 lanes
    # all take the value.
    PATTERN_ZERO = 0x00000000
    PATTERN_FP16_NEG_INF = 0xFBFFFBFF    # two FP16 -inf lanes per 32-bit word

    def __init__(self, dst_addr: Union[BingoMemAlloc, int], size: int, pattern: int):
        _check_xdma_size_aligned(size, "SnaxBingoKernelXdmaMemsetArgs")
        if not 0 <= int(pattern) <= 0xFFFFFFFF:
            raise ValueError(f"xdma memset: pattern {pattern:#x} is not a 32-bit value")
        self.dst_addr = dst_addr
        self.size = size
        self.pattern = int(pattern)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_memset_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.dst_addr, "dst_addr", assignments, handle_name_map)
        assignments["size"] = str(self.size)
        assignments["pattern"] = f"{self.pattern:#010x}u"
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


# BINGO XDMA ChainGather -- an in-fabric collective fold (see xdma.h).
#
# `chain` is the gather path in DATA order, ENDING at the collector's own destination
# buffer: [S1, S2, ..., dst_local]. Pass BingoMemAlloc handles; a handle allocated on
# another cluster already resolves to a full (chip|cluster|offset) address, which is
# what the hardware wants, so nothing here has to know the cluster map.
#
# The junction id is a GENERATED, cfg-dependent constant on the device side, so it is
# named here by its C identifier and emitted verbatim rather than hardcoded to a number
# that would silently shift if the cluster gained or lost a junction.
XDMA_JCT_ELEMENTWISE = "WRITER_JCT_ELEMENTWISEJUNCTION"
XDMA_JCT_MONOID = "WRITER_JCT_MONOIDJUNCTION"

# ElementwiseJunction CSR(0): [3:0] op, [6:4] fmt.
XDMA_EWJCT_OP_ADD = 0
XDMA_EWJCT_FMT_FP32 = 3


def xdma_ewjct_csr0(op=XDMA_EWJCT_OP_ADD, fmt=XDMA_EWJCT_FMT_FP32):
    """CSR(0) for the elementwise junction -- a plain (op, format) pair."""
    return (fmt << 4) | op


# MonoidJunction CSR(0)[14:12] transport format, from MonoidJunction.scala (FpHelpers.FMT_*).
XDMA_MONOID_FMT_FP16 = 0
XDMA_MONOID_FMT_BF16 = 1
XDMA_MONOID_FMT_FP8  = 2
XDMA_MONOID_FMT_FP32 = 3


def xdma_monoid_csr0(n_valid=8, n=1, n_exp=1, n_add=0, sigma=3, key_pol=0, key_mul=0,
                     fmt=XDMA_MONOID_FMT_FP32):
    """CSR(0) for the monoid junction -- a GEOMETRY word, not an operator id.

        [7:0] nValid | [11:8] n | [14:12] fmt | [21:18] nExp | [25:22] nAdd
        [27:26] sigma | [28] keyPol (0=max) | [29] keyMul (0 = the (R,max) key monoid)

    `fmt` IS NOT OPTIONAL and its zero value is a trap. It is the TRANSPORT format the
    junction slices a beat into -- 0 = FP16, 1 = BF16, 2 = FP8, 3 = FP32 -- and it was
    added to the junction after this encoder was written. A word built before the field
    existed has those bits zero, which names FP16: on the split cluster (elemWidth 16, so
    32 lanes per beat) the fold then reads the FP32 partial as 32 FP16 lanes, folds the
    wrong halves, and leaves the rest untouched.
    MEASURED when that happened: every per-shard check PASSED and fa_ml_merged FAILED with
    exactly 32 of 64 lanes reading 0x00000000, no watchdog. The default is FP32 because
    that is what pack_fa_partial produces and what the cluster cfg says the gather carries.

    Lanes are field-major, lane = field*S + slot with S = 1 << sigma, and nValid is how
    many of those S slots carry real data (the rest are fed their field's identity).

    The defaults are the online-softmax partial (m, l): key m plus ONE exp-twisted value
    coordinate, so n=1 (F = n+1 = 2 fields) and nExp=1, nAdd=0.

    sigma picks how many INDEPENDENT (m, l) pairs share a beat -- one per query row. A
    512b beat holds 16 FP32 lanes and the geometry uses F*S of them, so S=8 (sigma=3) is
    the largest legal choice and packs 8 query rows per beat: m at lanes 0..7, l at lanes
    8..15. Br=32 rows is then 4 beats. The sweep app uses nValid=1 because it folds a
    single scalar pair; FA folds a whole query tile, so it wants the full slot count --
    at nValid=1 only row 0 would be folded and the other seven silently keep the
    collector's own value.

    This is NOT the old StreamMomentMergeRt encoding ((1<<13)|1); under this layout that
    word decodes to n=0, sigma=0 -- a key-only geometry whose fold reads no value.
    """
    return ((n_valid & 0xFF) | ((n & 0xF) << 8) | ((fmt & 0x7) << 12) |
            ((n_exp & 0xF) << 18) | ((n_add & 0xF) << 22) | ((sigma & 0x3) << 26) |
            ((key_pol & 0x1) << 28) | ((key_mul & 0x1) << 29))


class SnaxBingoKernelPackFaPartialArgs(BingoKernelArgs):
    """Pack the softmax arena's (m, l) FP16 beats into the monoid's FP32 lane geometry.

    `slots` must equal the S the gather's monoid CSR(0) was built with, or the fold reads
    lanes the pack never wrote. Keep them derived from one place -- see
    xdma_monoid_csr0().
    """

    def __init__(self, src_m: Union[BingoMemAlloc, int], src_l: Union[BingoMemAlloc, int],
                 dst: Union[BingoMemAlloc, int], n_rows: int, slots: int = 8):
        if slots not in (1, 2, 4, 8):
            raise ValueError(f"pack_fa_partial: slots={slots} must be 1, 2, 4 or 8 "
                             "(two fields must fit the beat's 16 FP32 lanes).")
        if n_rows % slots:
            raise ValueError(f"pack_fa_partial: n_rows={n_rows} must be a multiple of "
                             f"slots={slots}.")
        self.src_m = src_m
        self.src_l = src_l
        self.dst = dst
        self.n_rows = n_rows
        self.slots = slots

    @staticmethod
    def packed_bytes(n_rows: int, slots: int = 8) -> int:
        """Size of the packed partial -- what the gather transfers per cluster."""
        return (n_rows // slots) * 64

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_pack_fa_partial_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_m, "src_m_addr", a, handle_name_map)
        self._process_addr(self.src_l, "src_l_addr", a, handle_name_map)
        self._process_addr(self.dst, "dst_addr", a, handle_name_map)
        a["n_rows"] = str(self.n_rows)
        a["slots"] = str(self.slots)
        return a


class SnaxBingoKernelXdmaChainGatherArgs(BingoKernelArgs):
    # Must match BINGO_XDMA_CHAIN_MAX in device_kernel_args.h.
    CHAIN_MAX = 8

    def __init__(self, local_src: Union[BingoMemAlloc, int], chain: list,
                 size: int, junction: str, jct_csr0: int):
        if not 2 <= len(chain) <= self.CHAIN_MAX:
            raise ValueError(
                f"chain gather: chain has {len(chain)} entries; it must have 2.."
                f"{self.CHAIN_MAX} -- the path in data order, ending at the "
                "collector's own destination buffer.")
        self.local_src = local_src
        self.chain = list(chain)
        self.size = size
        self.junction = junction
        self.jct_csr0 = jct_csr0

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_chain_gather_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.local_src, "local_src", a, handle_name_map)
        for i, hop in enumerate(self.chain):
            # _process_addr emits <base>_hi / <base>_lo; the base is a C lvalue, so an
            # array element indexes cleanly.
            tmp = {}
            self._process_addr(hop, "chain", tmp, handle_name_map)
            a[f"chain_hi[{i}]"] = tmp["chain_hi"]
            a[f"chain_lo[{i}]"] = tmp["chain_lo"]
        a["chain_num"] = str(len(self.chain))
        a["size"] = str(self.size)
        a["junction"] = self.junction
        a["jct_csr0"] = f"0x{self.jct_csr0:08x}u"
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


