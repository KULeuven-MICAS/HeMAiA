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

That is a real limit, not an oversight of documentation: the softmax-statistics junction
folds the two scalars per query row as the partials cross the fabric, and O is Br x d of INT32 in the D
port's scatter layout, which is a different transfer entirely. Treating `ml` as "the merged
result" and reading O_c as final gives an answer that is wrong by a per-cluster exponential
-- and on data where the four maxima are close, wrong by little enough to look plausible.

A SEPARATE STAGE, AND NOT A BLOCK. Two reasons, and the second is why it is not just a
method on FlashAttention.

First, the fold is a CHOICE, not a consequence. Under kvsplit each cluster holds a partial
over its own slice of the KV axis and the answer is the online-softmax combine of the four;
but where that combine happens is open -- in the fabric as the partials cross the
softmax-statistics junction, on the host, or nowhere at all, because a next stage sharded the same way would
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
    XDMA_SSJ_IDENTITY,
    XDMA_SSJ_ROWS,
    xdma_softmax_stats_csrs,
    xdma_softmax_stats_groups,
)
from bingo_mem_handle import BingoMemAlloc
from bingo_platform import writer_junction_index

from ..verify import checks


def merged_golden(cfg, m_c, l_c):
    """The (m*, l*) the junction should produce, in the groups it writes.

        m* = max_c m_c            l* = sum_c exp(a * (m_c - m*)) * l_c,   a = cfg.exp_scale

    `m_c` is each shard's running max as the softmax leaves it -- the max of the CONVERTED
    scores, S * 2^-k when the D port shifts -- and `l_c` its running sum, FP16, one value per
    query row. The softmax computed P = exp(a * (S - m)) with a = exp_scale, so the fold
    applies the same a to the key difference: the junction's CSR(1). m* comes back in the
    same units as m_c.

    The groups: per 32 query rows, a beat of 32 FP16 keys then a beat of 32 FP16 values;
    rows past Br hold the identity row, key 0xFBFF and value 0. Returned as FP16 bit patterns. Computed in FP32
    on FP16 inputs; the junction folds each hop in FP32 and rounds l to FP16 on the way out,
    which the merged check's relative tolerance covers.
    """
    m_c = np.asarray(m_c, dtype=np.float16).astype(np.float32)
    l_c = np.asarray(l_c, dtype=np.float16).astype(np.float32)
    m_star = m_c.max(axis=0)
    l_star = (np.exp(np.float32(cfg.exp_scale) * (m_c - m_star)) * l_c).sum(axis=0)
    groups, _ = xdma_softmax_stats_groups(cfg.br)
    out = np.zeros((groups, 2, XDMA_SSJ_ROWS), dtype=np.uint16)
    out[:, 0, :] = XDMA_SSJ_IDENTITY
    for g in range(groups):
        r0, r1 = g * XDMA_SSJ_ROWS, min(cfg.br, (g + 1) * XDMA_SSJ_ROWS)
        out[g, 0, :r1 - r0] = m_star[r0:r1].astype(np.float16).view(np.uint16)
        out[g, 1, :r1 - r0] = l_star[r0:r1].astype(np.float16).view(np.uint16)
    return out.reshape(-1)


def fa_gather(ctx, cfg, shards, merged_h=None, jct=None, verify=True, tol=0.02,
              out=None):
    """Fold the per-cluster partials into one (m*, l*) in the fabric. SEPARATE ON PURPOSE.

    Under kvsplit each cluster holds a partial over its own slice of the KV axis, and the
    answer is not any one of them -- it is the online-softmax combine

        m* = max_c m_c            l* = sum_c exp(a * (m_c - m*)) * l_c

    computed by the softmax-statistics junction as the partials cross the fabric. Under headpar there
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

    `out`, a dict, receives the nodes a caller orders against: "gather" (the fold),
    "merged" (the collector's buffer) and "check" (the merged check, or the gather when
    there is none).
    """
    if cfg.ncl < 2 or cfg.decomp == "headpar":
        return shards
    if jct is None:
        # DERIVED, not defaulted. The junction id is the extension's POSITION in the
        # cluster cfg's writer_junctions list, so there is no literal that is right on
        # every cluster -- and a caller that forgets to pass one must not silently emit
        # whatever it had. Reading it from the cfg the RTL was elaborated from is the only
        # source that stays in step with the hardware.
        if getattr(ctx, "hw", None) is None:
            raise ValueError(
                "fa_gather needs the softmax-statistics junction's index: pass jct=, or "
                "build the Ctx with hw=<parsed cluster cfg> so it can be derived. It is a "
                "position in that cfg's writer_junctions list, not a constant.")
        # The cfg key carries the `Has` prefix; the device macro does not
        # (WRITER_JCT_SOFTMAXSTATSJUNCTION). The cfg spelling is the one that indexes.
        jct = writer_junction_index(ctx.hw, "HasSoftmaxStatsJunction")
    g = ctx.at(0)
    # ---- copy each shard's (m, l) into the junction's group ------------------------------
    # The junction reads a key beat followed by a value beat; the arena keeps m and l apart,
    # so each shard copies them into one buffer, on its own xDMA core, so that the gather
    # that consumes it is the very next thing that core does. m is the running max as the
    # softmax leaves it and l the running sum (lrun), committed by the last softmax; the row
    # sum in the last P buffer is only the last tile's contribution.
    part_bytes = SnaxBingoKernelPackFaPartialArgs.packed_bytes(cfg.br)
    _, n_valid = xdma_softmax_stats_groups(cfg.br)
    csr0, csr1 = xdma_softmax_stats_csrs(n_valid, cfg.exp_scale)
    lay = SnaxBingoKernelSimdFaSoftmaxArgs.layout(cfg.bc, cfg.dhead)
    parts, packs = [], []
    for c, sh in enumerate(shards):
        gc = g.at(c)
        part = gc.l1("fa_ml_partial", part_bytes)
        parts.append(part)
        packs.append(gc.node(
            f"PackPartial_c{c}", ctx.xdma, "__snax_bingo_kernel_pack_fa_partial",
            SnaxBingoKernelPackFaPartialArgs(
                src_m=sh["arena"].view(lay["mrun"]), src_l=sh["arena"].view(lay["lrun"]),
                dst=part, n_rows=cfg.br),
            [sh["last_sm"]]))

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
            # The identifier WRITER_JCT_SOFTMAXSTATSJUNCTION is device-side only; see
            # writer_junction_index() for why the host emits the derived number instead.
            junction=f"{jct} /* WRITER_JCT_SOFTMAXSTATSJUNCTION */",
            # nValid is the number of query rows folded in every group; CSR(1) is the
            # softmax's own temperature, applied to the key difference before the exp.
            jct_csr0=csr0, jct_csr1=csr1),
        packs)

    # ---- check the merged result --------------------------------------------------------
    # Compared in the junction's own group order as FP16, which is what the collector's
    # buffer holds -- per 32 rows a key beat then a value beat.
    if out is not None:
        out.update(gather=gather, merged=merged, check=gather)
    if not verify or merged_h is None:
        return shards
    l3_ml = BingoMemAlloc("out_fa_ml_merged", size=part_bytes, mem_level="L3")
    st_ml = g.node("Store_ml", ctx.host, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(merged, l3_ml, part_bytes),
                   [gather] + [sh["checks"] for sh in shards if sh["checks"]],
                   cluster=0)
    ck = checks.check_fp16_rel(ctx, "Check_ml", golden=merged_h, got=l3_ml,
                               elems=part_bytes // 2, rtol=tol, after=st_ml,
                               label="fa_ml_merged")
    if out is not None:
        out["check"] = ck
    return shards
