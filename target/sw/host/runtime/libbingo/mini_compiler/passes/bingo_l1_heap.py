# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""A cluster's L1 heap, as the host's init and the device runtime fill it.

The device runtime puts the heap at the base of each cluster's TCDM and ends it below the cluster's private data, its
cores' stacks and a reserved block (device/runtime/src/snrt_l1_alloc.h). bingo_alloc keeps its instance at the base and
cuts each allocation as a fragment of BINGO_HEAP_FRAGMENT_SIZE(amount) bytes (shared/vendor/bingo_alloc). Everything a
cluster keeps in L1 is one of those allocations: the static plan's arena, the paged-list windows, the scratchpad and
args arena (bingo_dfg_emit.py), the scheduler's per-cluster lists (bingo_api.c, bingo_list_phases.h), and on the
device the SIMD scratch pool (simd.h).

The constants are read from those sources, so the model follows them; the device binary's cluster-private data
(.cdata + .cbss) is bounded by CLS_BOUND.
"""
import os
import re

CLS_BOUND = 512
# What a budget made before the graph is final leaves for what grows after: the task lists by the tasks the later
# passes add (LIST_SLACK), the scratchpad slots the packing makes (SLOTS_BOUND bytes); an args phase region is no
# smaller than MIN_PHASE bytes.
LIST_SLACK = 1.5
SLOTS_BOUND = 64 * 64
MIN_PHASE = 4096
_SW = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
_C = None


def _define(rel, name, default):
    try:
        text = open(os.path.join(_SW, rel)).read()
    except OSError:
        return default
    m = re.search(r"#define\s+%s\s+\(?\s*(0x[0-9a-fA-F]+|\d+)" % name, text)
    return int(m.group(1), 0) if m else default


def consts():
    """The heap's constants: {align, instance, reserved, row, log2_stack, simd_pool}."""
    global _C
    if _C is None:
        align = _define("shared/vendor/bingo_alloc/bingo_alloc/bingo_alloc.h", "BINGO_HEAP_ALIGNMENT", 128)
        bins = _define("shared/vendor/bingo_alloc/bingo_alloc/bingo_alloc.h", "BINGO_HEAP_NUM_BINS", 64)
        instance = bins * 8 + 8 + 5 * 8                 # bins, the non-empty mask, BingoHeapDiagnostics
        _C = dict(align=align, instance=(instance + align - 1) // align * align,
                  reserved=_define("device/runtime/src/snrt_l1_alloc.h", "RESERVED_SIZE", 4096),
                  row=_define("device/runtime/src/snrt_l1_alloc.h", "TCDM_ROW_SIZE", 512),
                  log2_stack=_define("device/runtime/generated/occamy_defs.h", "SNRT_LOG2_STACK_SIZE", 10),
                  # the cluster's Snitch cores, each with its stack (the bingo manager's core count adds the host)
                  cores=_define("shared/platform/generated/occamy.h", "N_CORES_PER_CLUSTER", 4),
                  simd_pool=_define(_SIMD, "BINGO_SIMD_SCRATCH_POOL", 16384),
                  norm_pool=_define(_SIMD, "BINGO_SIMD_NORM_POOL", 24576))
    return _C


_SIMD = "device/apps/snax/snax-bingo-offload/libsnaxkernel/offload_hw_kernels/simd.h"
# the SIMD kernels that keep a scratch pool of their own in each cluster's L1 heap, taken by their first task there
# (simd.h BINGO_SIMD_SCRATCH, _bingo_rmsnorm_scratch: rmsnorm only without scratch of its args)
_POOLS = {"__snax_bingo_kernel_simd_softmax_f16_f16": "simd_pool",
          "__snax_bingo_kernel_simd_softmax_t_f16_f16": "simd_pool",
          "__snax_bingo_kernel_simd_silu_f16_f16": "simd_pool",
          "__snax_bingo_kernel_simd_swiglu_f16_f16": "simd_pool",
          "__snax_bingo_kernel_simd_rmsnorm": "norm_pool"}


def device_pools(nodes):
    """{kernel name: bytes}: the scratch pools the device takes on the cluster of `nodes`."""
    out = {}
    for n in nodes:
        k = n.kernel_name or ""
        if k in _POOLS and k not in out:
            sa = getattr(n.kernel_args, "scratch_addr", 0)
            if k.endswith("_rmsnorm") and not (isinstance(sa, int) and sa == 0):
                continue                        # its args give it scratch
            out[k] = consts()[_POOLS[k]]
    return out


def frag(nbytes):
    """The bytes an allocation of `nbytes` takes from the heap (BINGO_HEAP_FRAGMENT_SIZE)."""
    a = consts()["align"]
    return (int(nbytes) + a + 2 * a - 1) // (2 * a) * (2 * a)


def heap_bytes(tcdm_bytes):
    """What the heap can hand out on one cluster: from the TCDM base to the top the device runtime leaves it, less
    bingo_alloc's instance."""
    c = consts()
    top = tcdm_bytes - CLS_BOUND - 8 - c["cores"] * ((1 << c["log2_stack"]) + 8) - c["reserved"]
    return top // c["row"] * c["row"] - c["instance"]


def list_allocs(mode, n, total, ncores, hdr_words=0):
    """The sizes of the scheduler's per-cluster L1 allocations at host init: n device tasks on the cluster, total
    tasks in the graph. mode: "paged" (bingo_hw_scheduler_init_list_phases: the header; the windows are the
    emitter's), "gid" (gid_lists), "compact" (compact id maps) or "full"."""
    up8 = lambda x: (x + 7) // 8 * 8
    if mode == "paged":
        return [hdr_words * 4]
    lists = up8(max(n, 1) * 4)
    if mode == "gid":
        return [lists, lists, up8((3 + ncores + n) * 4)]
    if mode == "compact":
        return [lists, lists, up8(4 + total * 2)]
    return [n * 4, n * 4, total * 4]
