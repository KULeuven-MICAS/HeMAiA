# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""GEMM configuration reuse: a rewrite a workload runs over its graph before compiling it.

A gemm_full task whose configuration equals the previous gemm_full's on the same core (every field
but the four operand addresses) becomes gemm_minimal, which rewrites the addresses and starts. The
previous GEMM on a core is the one created before it, so the core's tasks must run in creation order
(GemmQueue.program_order).
"""

from bingo_kernel_args import SnaxBingoKernelGemmMinimalArgs

GEMM_FULL = "__snax_bingo_kernel_gemm_full"
GEMM_MINIMAL = "__snax_bingo_kernel_gemm_minimal"
OPERAND_ADDRS = ("input_A_addr", "input_B_addr", "input_C_addr", "output_D_addr")


def reuse_gemm_config(dfg) -> int:
    """Turn every gemm_full of `dfg` that repeats its core's previous gemm_full configuration into a
    gemm_minimal. Returns the number converted."""
    prev, n = {}, 0
    for t in sorted(dfg.node_list, key=lambda t: t.node_id):
        if t.kernel_name != GEMM_FULL:
            continue
        key = (t.assigned_chiplet_id, t.assigned_cluster_id, t.assigned_core_id)
        sig = tuple(sorted((k, v) for k, v in vars(t.kernel_args).items()
                           if k not in OPERAND_ADDRS and not k.startswith("_")))
        if prev.get(key) == sig:
            a = t.kernel_args
            t.kernel_name = GEMM_MINIMAL
            t.kernel_args = SnaxBingoKernelGemmMinimalArgs(a.input_A_addr, a.input_B_addr, a.input_C_addr,
                                                           a.output_D_addr)
            n += 1
        prev[key] = sig
    return n
