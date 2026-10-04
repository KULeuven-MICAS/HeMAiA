# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Kernels that drive no accelerator, and the iDMA.

Dummy, sync probe, sync report -- placed wherever their consumer is, because they touch no
engine -- plus the cluster iDMA transfers. They share a file because none of them is big
enough to be worth its own, and because `_engine_of_kernel` treats them the same way."""

from typing import Union, Dict
from bingo_mem_handle import BingoMemAlloc

from kernel_base import BingoKernelArgs


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
class SnaxBingoKernelSyncProbeArgs(BingoKernelArgs):
    """Args for __snax_bingo_kernel_sync_probe (cross-chip sync latency, arm D).

    The probe does no work; it stamps mcycle into stamp_buf[slot]. Pass stamp_buf=0
    for a task whose timing is not needed (the remote "mid" tasks).

    WARNING: stamp_buf must be allocated on the SAME chiplet the task runs on. The
    kernel writes it with a plain local store, and mcycle is not comparable across
    chiplets anyway, so a cross-chiplet stamp would be meaningless even if it landed.
    """

    def __init__(self, stamp_buf: Union[BingoMemAlloc, int] = 0, slot: int = 0):
        self.stamp_buf = stamp_buf
        self.slot = int(slot)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_sync_probe_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.stamp_buf, "stamp_buf", assignments, handle_name_map,
                           split_64bit=False, as_64bit=False)
        assignments["slot"] = str(self.slot)
        return assignments


class HostBingoKernelSyncReportArgs(BingoKernelArgs):
    """Args for __host_bingo_kernel_sync_report (cross-chip sync latency, arm D).

    Runs as a HOST node at the end of the DFG and prints one line per phase, keyed by
    phase index. What each index means is written by the generator to sync_phases.csv:
    a DFG memory handle only reserves storage, so a metadata table cannot be preloaded
    into the device image.
    """

    def __init__(self, stamp_buf: Union[BingoMemAlloc, int],
                 num_phases: int, local_phase: int):
        self.stamp_buf = stamp_buf
        self.num_phases = int(num_phases)
        self.local_phase = int(local_phase)

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_sync_report_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        assignments = {}
        self._process_addr(self.stamp_buf, "stamp_buf", assignments, handle_name_map,
                           split_64bit=False, as_64bit=True)
        assignments["num_phases"] = str(self.num_phases)
        assignments["local_phase"] = str(self.local_phase)
        return assignments


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

# BINGO IDMA strided copy: `outer` repetitions of `reps` runs of `size` bytes
class SnaxBingoKernelIdma2dCopyArgs(BingoKernelArgs):
    """Up to three dimensions of runs in one node (offload_hw_kernels/idma.h): a gather of
    rows, a cache append into an A layout, a tile of a [512, cap] operand."""
    KERNEL_NAME = "__snax_bingo_kernel_idma_2d_copy"

    def __init__(self, src_addr, dst_addr, size: int, src_stride: int, dst_stride: int,
                 reps: int, outer: int = 1, src_outer: int = 0, dst_outer: int = 0):
        if size <= 0 or reps <= 0 or outer <= 0:
            raise ValueError(f"size={size}, reps={reps}, outer={outer}: all must be positive.")
        self.src_addr, self.dst_addr = src_addr, dst_addr
        self.size, self.src_stride, self.dst_stride = int(size), int(src_stride), int(dst_stride)
        self.reps, self.outer = int(reps), int(outer)
        self.src_outer, self.dst_outer = int(src_outer), int(dst_outer)

    @property
    def nbytes(self) -> int:
        return self.size * self.reps * self.outer

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_idma_2d_copy_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        for nm in ("size", "src_stride", "dst_stride", "reps", "outer", "src_outer",
                   "dst_outer"):
            a[nm] = str(getattr(self, nm))
        return a


# BINGO IDMA copy from an expert-slot record (moe_route.h): the source is chosen at run time
class SnaxBingoKernelIdmaCopySlotArgs(BingoKernelArgs):
    """Copy `size` bytes from [base + offset] to dst, base being word pair `field` of
    slot `slot` of an expert-slot record in this cluster's L1 (0 gate|up, 1 its factors,
    2 down, 3 its factors). A zero base fails the node."""
    KERNEL_NAME = "__snax_bingo_kernel_idma_copy_slot"
    FIELDS = {"gu": 0, "gu_s": 1, "dn": 2, "dn_s": 3}

    def __init__(self, record_addr, slot: int, field, offset: int, dst_addr, size: int):
        f = self.FIELDS[field] if isinstance(field, str) else int(field)
        if not 0 <= f <= 3:
            raise ValueError(f"field={field}: one of {sorted(self.FIELDS)}.")
        if size <= 0 or offset < 0 or slot < 0:
            raise ValueError(f"slot={slot}, offset={offset}, size={size}.")
        self.record_addr, self.slot, self.field = record_addr, int(slot), f
        self.offset, self.dst_addr, self.size = int(offset), dst_addr, int(size)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_idma_copy_slot_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.record_addr, "record_addr", a, handle_name_map,
                           split_64bit=False)
        a["slot"] = str(self.slot)
        a["field"] = str(self.field)
        a["offset"] = str(self.offset)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["size"] = str(self.size)
        return a


class SnaxBingoKernelIdmaRingLoadArgs(BingoKernelArgs):
    """One chunk of an L3 weight ring into an L1 slab (offload_hw_kernels/idma.h): wait
    until the slot's flag holds `seq`, copy `size` bytes from the slot, then write `seq`
    into the slot's release word. The memory chiplet's iDMA pushes the chunk and then its
    flag (libs/blocks/linear.py WeightRing); flag and release are local L3 words."""
    KERNEL_NAME = "__snax_bingo_kernel_idma_ring_load"

    def __init__(self, flag_addr, seq: int, src_addr, dst_addr, size: int, release_addr,
                 trailer: bool = False):
        if seq < 1 or size <= 0:
            raise ValueError(f"seq={seq} (from 1), size={size}.")
        self.flag_addr, self.seq, self.src_addr = flag_addr, int(seq), src_addr
        self.dst_addr, self.size, self.release_addr = dst_addr, int(size), release_addr
        # weight_ring.trailer: flag_addr is the record's last beat (a 64-bit magic, cleared
        # after the copy) instead of the ring's flag word
        self.trailer = int(bool(trailer))

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_idma_ring_load_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.flag_addr, "flag_addr", a, handle_name_map, split_64bit=False)
        a["seq"] = str(self.seq)
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["size"] = str(self.size)
        self._process_addr(self.release_addr, "release_addr", a, handle_name_map,
                           split_64bit=False)
        a["trailer"] = str(self.trailer)
        return a


class SnaxBingoKernelXdmaCrestExpandArgs(BingoKernelArgs):
    """CREST in place on the xDMA core (offload_hw_kernels/xdma.h): the slab's last 64-B beat
    is a record's tail (stream words W, 0 = plain; output beats N), the stream ends right
    before it, and it expands into the slab's first N beats, L1 to L1, through the writer's
    CrestDecompressor. params weight_crest (libs/crest.py)."""
    KERNEL_NAME = "__snax_bingo_kernel_xdma_crest_expand"

    def __init__(self, slab_addr, slab_bytes: int):
        if slab_bytes <= 64 or slab_bytes % 64:
            raise ValueError(f"slab_bytes={slab_bytes}: whole 64-B beats, more than one")
        self.slab_addr, self.slab_bytes = slab_addr, int(slab_bytes)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_xdma_crest_expand_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.slab_addr, "slab_addr", a, handle_name_map)
        a["slab_bytes"] = str(self.slab_bytes)
        return a


# BINGO MoE route: top k of n FP16 probabilities -> an expert-slot record
class SnaxBingoKernelMoeRouteArgs(BingoKernelArgs):
    """The router's decision as data (offload_hw_kernels/moe_route.h): top k of the n FP16
    probabilities at p_addr, and per chosen expert a 128-B record slot with its id, weight
    and its 64-B entry of the expert table at table_addr. On the DM core."""
    KERNEL_NAME = "__snax_bingo_kernel_moe_route"
    TABLE_ENTRY_BYTES = 64
    SLOT_BYTES = 128
    MAX_N, MAX_K = 64, 8

    MAX_TOKENS = 4

    def __init__(self, p_addr, n: int, k: int, table_addr, record_addr, tokens: int = 1,
                 p_pitch: int = 0, union_n: int = 0):
        if not 0 < n <= self.MAX_N or not 0 < k <= min(n, self.MAX_K):
            raise ValueError(f"n={n} (<= {self.MAX_N}), k={k} (<= {self.MAX_K}, <= n).")
        if not 1 <= tokens <= self.MAX_TOKENS or (tokens > 1 and (
                not k <= union_n <= min(n, tokens * k) or p_pitch < 2 * n or p_pitch % 2)):
            raise ValueError(f"tokens={tokens} (<= {self.MAX_TOKENS}): union_n={union_n} in "
                             f"[k, tokens k], p_pitch={p_pitch} past a row of {n} FP16.")
        self.p_addr, self.n, self.k = p_addr, int(n), int(k)
        self.table_addr, self.record_addr = table_addr, record_addr
        self.tokens, self.p_pitch = int(tokens), int(p_pitch)
        self.union_n = int(union_n) if tokens > 1 else 0

    @classmethod
    def record_bytes(cls, k: int) -> int:
        return cls.SLOT_BYTES * int(k)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_moe_route_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.p_addr, "p_addr", a, handle_name_map, split_64bit=False)
        a["n"] = str(self.n)
        a["k"] = str(self.k)
        self._process_addr(self.table_addr, "table_addr", a, handle_name_map)
        self._process_addr(self.record_addr, "record_addr", a, handle_name_map,
                           split_64bit=False)
        a["tokens"] = str(self.tokens)
        a["p_pitch"] = str(self.p_pitch)
        a["union_n"] = str(self.union_n)
        return a


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
