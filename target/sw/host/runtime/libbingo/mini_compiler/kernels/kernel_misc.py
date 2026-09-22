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
