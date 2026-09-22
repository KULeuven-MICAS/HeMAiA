# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Folding the per-cluster attention partials, in the fabric.

WHAT IT FOLDS, AND WHAT IT DOES NOT. A kvsplit shard's partial state is the TRIPLE
(m, l, O): the running max, the running sum, and the accumulator. The complete combine is

    m* = max_c m_c
    l* = sum_c exp(m_c - m*) * l_c
    O* = sum_c exp(m_c - m*) * O_c

and this folds the FIRST TWO. O is left where each cluster computed it, as `o_c{c}`, still
scaled by that cluster's own m_c. A caller that wants one attention output has to apply the
third line itself, using the m* this returns.

That is a real limit, not an oversight of documentation: the monoid junction folds the two
scalars per query row as the partials cross the fabric, and O is Br x d of INT32 in the D
port's scatter layout, which is a different transfer entirely. Treating `ml` as "the merged
result" and reading O_c as final gives an answer that is wrong by a per-cluster exponential
-- and on data where the four maxima are close, wrong by little enough to look plausible.

A SEPARATE STAGE, AND NOT A BLOCK. Two reasons, and the second is why it is not just a
method on FlashAttention.

First, the fold is a CHOICE, not a consequence. Under kvsplit each cluster holds a partial
over its own slice of the KV axis and the answer is the online-softmax combine of the four;
but where that combine happens is open -- in the fabric as the partials cross the monoid
junction, on the host, or nowhere at all, because a next stage sharded the same way would
pay for a gather and then re-split what it gathered. Building it into FlashAttention would make
that choice for every caller and hide the one case where it is wrong.

Second, it cannot be expressed as a Block. A Block owns its operands through ports, and a
port names a buffer with a layout and a producer. What this fold reads is each cluster's
softmax ARENA -- a buffer the previous block is still updating in place, a running state
rather than a produced tensor. A port would promise something the model cannot honour, so
this takes the shards directly.

Under headpar, or with a single cluster, there is nothing to fold and this is a no-op.
"""

import numpy as np

from bingo_kernel_args import (
    HostBingoKernelIdmaArgs,
    SnaxBingoKernelPackFaPartialArgs,
    SnaxBingoKernelSimdFaSoftmaxArgs,
    SnaxBingoKernelXdmaChainGatherArgs,
    xdma_monoid_csr0,
)
from bingo_mem_handle import BingoMemAlloc
from bingo_platform import writer_junction_index

from ..verify import checks


def merged_golden(cfg, m_c, l_c):
    """The (m*, l*) the junction should produce, in the lane geometry it writes.

        m* = max_c m_c            l* = sum_c exp(m_c - m*) * l_c

    Computed in FP32 on FP16 inputs, mirroring the device: the arena holds m and l in
    FP16, pack_fa_partial widens the bit pattern to FP32, and the junction folds in FP32.

    The packing matters as much as the arithmetic. The collector's buffer is laid out
    lane = field*S + slot -- field 0 is m, field 1 is l, S = monoid_slots rows per beat,
    16 FP32 lanes per 512-bit beat -- so a golden in row order would disagree with a
    correct fold everywhere.
    """
    m_c = np.asarray(m_c, dtype=np.float16)
    l_c = np.asarray(l_c, dtype=np.float16)
    m_star = m_c.astype(np.float32).max(axis=0)
    l_star = (np.exp(m_c.astype(np.float32) - m_star) *
              l_c.astype(np.float32)).sum(axis=0)

    beats = cfg.br // cfg.monoid_slots
    merged = np.zeros(beats * 16, dtype=np.float32)
    for beat in range(beats):
        for slot in range(cfg.monoid_slots):
            row = beat * cfg.monoid_slots + slot
            merged[beat * 16 + 0 * cfg.monoid_slots + slot] = m_star[row]
            merged[beat * 16 + 1 * cfg.monoid_slots + slot] = l_star[row]
    return merged


def fa_gather(ctx, cfg, shards, merged_h=None, jct_monoid=None, verify=True):
    """Fold the per-cluster partials into one (m*, l*) in the fabric. SEPARATE ON PURPOSE.

    Under kvsplit each cluster holds a partial over its own slice of the KV axis, and the
    answer is not any one of them -- it is the online-softmax combine

        m* = max_c m_c            l* = sum_c exp(m_c - m*) * l_c

    computed by the monoid junction as the partials cross the fabric. Under headpar there
    is nothing to fold: each cluster owns a different query head and its O is already final.

    WHY IT IS NOT PART OF FlashAttention. The fold is a CHOICE about how the shards are
    recombined, and there is more than one: fold in the fabric here, fold on the host, or
    do not fold at all and let the next block consume the four partials where they lie.
    Building it into the block would make that choice for every caller and hide the one
    case that matters -- a decode step whose next stage is already sharded the same way
    pays for a gather and then re-splits what it gathered.

    So FlashAttention exposes the partials and the caller composes this when it wants them
    merged. Calling it with clusters < 2, or under headpar, is a no-op that returns the
    shards untouched.
    """
    if cfg.clusters < 2 or cfg.decomp == "headpar":
        return shards
    if jct_monoid is None:
        # DERIVED, not defaulted. The junction id is the extension's POSITION in the
        # cluster cfg's writer_junctions list, so there is no literal that is right on
        # every cluster -- and a caller that forgets to pass one must not silently emit
        # whatever it had. Reading it from the cfg the RTL was elaborated from is the only
        # source that stays in step with the hardware.
        if getattr(ctx, "hw", None) is None:
            raise ValueError(
                "fa_gather needs the monoid junction's index: pass jct_monoid=, or build "
                "the Ctx with hw=<parsed cluster cfg> so it can be derived. It is a "
                "position in that cfg's writer_junctions list, not a constant.")
        # The cfg key carries the `Has` prefix; the device macro does not
        # (WRITER_JCT_MONOIDJUNCTION). The cfg spelling is the one that indexes.
        jct_monoid = writer_junction_index(ctx.hw, "HasMonoidJunction")
    g = ctx.at(0)
    # ---- pack each shard's (m, l) into the junction's lanes -----------------------------
    # On each cluster's own xDMA core, so the gather that consumes it is the very next
    # thing that core does. The pack is scalar FP16->FP32 bit work; that core has no FPU,
    # which is the whole reason it is a kernel and not two lines in the caller.
    part_bytes = SnaxBingoKernelPackFaPartialArgs.packed_bytes(cfg.br, cfg.monoid_slots)
    parts, packs = [], []
    for c, sh in enumerate(shards):
        gc = g.at(c)
        part = gc.l1("fa_ml_partial", part_bytes)
        parts.append(part)
        packs.append(gc.node(
            f"PackPartial_c{c}", ctx.xdma, "__snax_bingo_kernel_pack_fa_partial",
            SnaxBingoKernelPackFaPartialArgs(
                src_m=sh["arena"].view(SnaxBingoKernelSimdFaSoftmaxArgs.layout(cfg.bc, cfg.dhead)["mrun"]),
                src_l=sh["p8_last"].view((cfg.bc // 2) * 64),
                dst=part, n_rows=cfg.br, slots=cfg.monoid_slots),
            sh["last_sm"]))

    # ---- the fold itself ----------------------------------------------------------------
    # chain is the path in DATA order, ENDING at the collector's own destination, and
    # local_src is the collector's own operand. Cluster 0 collects.
    #
    # The gather may not start until EVERY shard has packed: it reads the other three
    # clusters' buffers directly, and nothing in the fabric would tell it that a partial is
    # still being written. Those three edges are the entire synchronisation.
    merged = g.l1("fa_ml_merged", part_bytes)
    gather = g.node(
        "GatherML", ctx.xdma, "__snax_bingo_kernel_xdma_chain_gather",
        SnaxBingoKernelXdmaChainGatherArgs(
            local_src=parts[0], chain=parts[1:] + [merged],
            size=part_bytes,
            # The identifier WRITER_JCT_MONOIDJUNCTION is device-side only; see
            # writer_junction_index() for why the host emits the derived number instead.
            junction=f"{jct_monoid} /* WRITER_JCT_MONOIDJUNCTION */",
            # nValid is the number of ROWS folded per beat, not a count of operands: the
            # geometry packs MONOID_SLOTS query rows into one beat, and at nValid=1 seven
            # of every eight rows would silently keep the collector's own value.
            jct_csr0=xdma_monoid_csr0(n_valid=cfg.monoid_slots, n=1, n_exp=1, n_add=0,
                                      sigma=3)),
        packs)

    # ---- check the merged result --------------------------------------------------------
    # Compared in the junction's own lane order as FP32, which is what the collector's
    # buffer holds -- m in lanes 0..S-1 of each beat, l in lanes S..2S-1.
    if not verify or merged_h is None:
        return shards
    l3_ml = BingoMemAlloc("out_fa_ml_merged", size=part_bytes, mem_level="L3")
    st_ml = g.node("Store_ml", ctx.host, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(merged, l3_ml, part_bytes),
                   [gather] + [sh["checks"] for sh in shards if sh["checks"]],
                   cluster=0)
    checks.check_fp32(ctx, "Check_ml", golden=merged_h, got=l3_ml,
                      elems=part_bytes // 4, tol=0.02, after=st_ml, label="fa_ml_merged")
    return shards
