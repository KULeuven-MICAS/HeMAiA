# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Accelerator configuration reuse along a core's program: rewrites a workload runs over its graph once
every device core's program is fixed (GemmQueue.program_order): its tasks in creation order, or by the
order_key a generator set (a task made early that runs later on its core), so the task before another on a
core is the one before it in that order, not the one created before it.

An accelerator's CSRs are sticky: a START snapshots the bank and leaves it as it was. A task that would write
the values its core's previous task left there can skip them, and only the host knows what that task was.

reuse_gemm_fa_config: a queued FlashAttention matmul (gemm_fa_qk_start / gemm_fa_pv_start) whose cluster's
previous VersaCore dispatch is the same kernel with the same shape, flags, d_shift, b_pitch and kind of C
takes GEMM_FA_REUSE: it writes its four operand addresses (and a PV its corr factors) and starts. Its seq must
follow the previous one's, so no dispatch the queue numbered sits between them; any other task on the GEMM
core but a gemm_wait ends the run. A dispatch configured ahead (CONFIGURE and START tasks, blocks/gemm_async.py)
counts as its configure task; the START task writes no configuration and is passed over.

reuse_gemm_start_config: the same for a queued generic GEMM (gemm_start): one that repeats its cluster's
previous dispatch, a gemm_start with the same arguments but the operand addresses, takes `reuse`. A dispatch
configured ahead (blocks/gemm_async.py cfg_ahead) is its configure task, which takes `reuse`, and a start task
that writes no CSR, which the run passes over.

prime_fa_softmax: a simd_fa_softmax whose arena an earlier softmax on its core has used takes GEOM_PRIMED:
the shapes and their memo are in the arena, so tile 0 of a later sweep does not rebuild them. It takes
GEOM_CSR_PRIMED when the task before it on the SIMD core is a softmax of the same geometry, whose shared CSRs
(snax_simd_program_fast) hold the values it would write. The earlier softmax on the same in-order core is
an ancestor of this one, which is what GEOM_PRIMED promises.
"""

from bingo_kernel_args import SnaxBingoKernelGemmFaPvStartArgs, SnaxBingoKernelSimdFaSoftmaxArgs
from bingo_mem_handle import BingoMemAllocView

FA_STARTS = ("__snax_bingo_kernel_gemm_fa_qk_start", "__snax_bingo_kernel_gemm_fa_pv_start")
GEMM_WAIT = "__snax_bingo_kernel_gemm_wait"
SOFTMAX = "__snax_bingo_kernel_simd_fa_softmax"
REUSE = SnaxBingoKernelGemmFaPvStartArgs.REUSE
FA_STAGE = REUSE | SnaxBingoKernelGemmFaPvStartArgs.CONFIGURE | SnaxBingoKernelGemmFaPvStartArgs.START
SM = SnaxBingoKernelSimdFaSoftmaxArgs


def _programs(dfg, cores):
    """{(chip, cluster, core): [tasks in program order]} for the device cores `cores`."""
    cores = set(cores)
    groups = {}
    for t in dfg.node_list:
        if t.assigned_core_id in cores:
            groups.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id, t.assigned_core_id), []).append(t)
    for ts in groups.values():
        ts.sort(key=program_key)
    return groups


def program_key(t):
    """A task's place in its core's program: GemmQueue.program_order's (its order_key, else its creation order)."""
    return (getattr(t, "order_key", t.node_id), t.node_id)


def reuse_gemm_fa_config(dfg, gemm_cores) -> int:
    """Mark GEMM_FA_REUSE on every FlashAttention start of `dfg` that repeats its cluster's previous
    dispatch. Returns the number marked."""
    n = 0
    for ts in _programs(dfg, gemm_cores).values():
        prev = None                         # (signature, seq) of the previous dispatch, if an FA start
        for t in ts:
            if t.kernel_name in FA_STARTS:
                a = t.kernel_args
                if a.flags & SnaxBingoKernelGemmFaPvStartArgs.START:
                    continue
                sig = (t.kernel_name, a.M, a.K, a.N, a.flags & ~FA_STAGE, a.d_shift, a.b_pitch,
                       bool(a.input_C_addr))
                if prev is not None and prev[0] == sig and a.seq == prev[1] + 1:
                    a.flags |= REUSE
                    n += 1
                prev = (sig, a.seq)
            elif t.kernel_name != GEMM_WAIT:
                prev = None
    return n


GEMM_START = "__snax_bingo_kernel_gemm_start"
_NOT_CONFIG = ("input_A_addr", "input_B_addr", "input_C_addr", "output_D_addr", "seq", "shape_addr", "reuse",
               "stage")
START_ONLY = 2


def reuse_gemm_start_config(dfg, gemm_cores) -> int:
    """Mark `reuse` on every queued gemm_start of `dfg` whose cluster's previous VersaCore dispatch is a
    gemm_start with the same arguments but its operand addresses (and a C of the same kind, NULL or not):
    the kernel writes the four addresses and starts. Its seq must follow the previous one's; any other task
    on the GEMM core but a gemm_wait or gemm_shape_init ends the run. Returns the number marked."""
    n = 0
    for ts in _programs(dfg, gemm_cores).values():
        prev = None                         # (signature, seq) of the previous dispatch, if a gemm_start
        for t in ts:
            if t.kernel_name == GEMM_START:
                a = t.kernel_args
                if a.stage == START_ONLY:
                    continue
                sig = (tuple(sorted((k, v) for k, v in vars(a).items()
                                    if k not in _NOT_CONFIG and not k.startswith("_"))),
                       bool(a.input_C_addr))
                if prev is not None and prev[0] == sig and a.seq == prev[1] + 1:
                    a.reuse = 1
                    n += 1
                prev = (sig, a.seq)
            elif t.kernel_name not in (GEMM_WAIT, "__snax_bingo_kernel_gemm_shape_init"):
                prev = None
    return n


def _buffer(h):
    """A key for the L1 buffer a handle names, whatever the view."""
    if isinstance(h, BingoMemAllocView):
        return (id(h.base), h.offset)
    return h if isinstance(h, int) else (id(h), 0)


def prime_fa_softmax(dfg, simd_cores):
    """Promote every softmax of `dfg` whose arena an earlier softmax on its core has used to GEOM_PRIMED,
    or GEOM_CSR_PRIMED after a same-geometry softmax; a softmax its builder made GEOM_PRIMED (its arena's
    geometry built by an ancestor: a prologue, or fa_sweep geom_core's build on another core) likewise
    becomes GEOM_CSR_PRIMED after a same-geometry softmax. Returns (primed, csr_primed)."""
    primed = csr = 0
    for ts in _programs(dfg, simd_cores).values():
        used = set()
        prev = None
        for t in ts:
            if t.kernel_name == SOFTMAX:
                a = t.kernel_args
                key = _buffer(a.arena)
                if (key in used and a.geom_mode == SM.GEOM_SELF) or a.geom_mode == SM.GEOM_PRIMED:
                    p = prev.kernel_args if prev is not None and prev.kernel_name == SOFTMAX else None
                    if p is not None and (p.bc, p.dhead, p.p_mode, p.p8_pitch, p.score_scale) == \
                            (a.bc, a.dhead, a.p_mode, a.p8_pitch, a.score_scale):
                        a.geom_mode = SM.GEOM_CSR_PRIMED
                        csr += 1
                    else:
                        a.geom_mode = SM.GEOM_PRIMED
                        primed += 1
                used.add(key)
            prev = t
    return primed, csr
