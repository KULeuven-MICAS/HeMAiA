# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The key-split decode's sizes: the FA softmax's arena and beats, the tiles, the exchanged rows, the weight chunks, and
the constants' layouts in L1. Also the a_row packing (a_row) and FP32 bits (f32bits) the steps share."""

import numpy as np

from bingo_kernel_args import (SnaxBingoKernelGemmFaPvArgs, SnaxBingoKernelSimdFaSoftmaxArgs, XDMA_EWJCT_FMT_FP16,
                               XDMA_EWJCT_OP_MAX, xdma_ewjct_csr0)

BC, BR, D_QK, D_V = 64, 16, 576, 512   # keys a tile, query lanes (one SIMD beat of FP16), a key's / a value's bytes
HIDDEN, KV_RANK, ROPE, Q_NOPE, V_HEAD = 2048, 512, 64, 128, 128
Q_HEAD = Q_NOPE + ROPE

FA = SnaxBingoKernelSimdFaSoftmaxArgs
PVK = SnaxBingoKernelGemmFaPvArgs
# BEAT: one SIMD beat, BR FP16 values -- one key of a score tile, and the m and l beats. XFER: the smallest unit the xDMA
# and the D2D paths move it in (their 64-B beat); a wider transfer of an m or l beat also carries the arena beat after
# it, which is scratch (corrL after mrun) or the same l again (lnew after lrun, equal after the commit).
BEAT = FA.beat()
XFER = max(64, BEAT)
EW_F16_MAX = xdma_ewjct_csr0(XDMA_EWJCT_OP_MAX, XDMA_EWJCT_FMT_FP16)
LAY = FA.layout(BC, 1)
ARENA = FA.arena_bytes(BC, 1)
if ARENA % XFER:
    raise ValueError(f"the FA softmax arena is {ARENA} B, not whole {XFER}-B units")
if FA.layout(2, 1) != LAY or FA.arena_bytes(2, 1) != ARENA:
    raise ValueError("the FA softmax arena's layout depends on its tile height")
MRUN, LRUN = LAY["mrun"], LAY["lrun"]

QC = 32                              # a GEMV chunk: 32 columns (the paired INT4 layout's unit)
WCH = HIDDEN * QC // 2               # a chunk of 32 columns over the hidden depth, INT4
ATOM, LANES = 8, 8                   # the xDMA's lane (an 8-byte atom) and its lanes a beat
KB, VB = BC * D_QK, D_V * BC         # a K tile, a V tile
SB = BEAT + BC * BR * 2              # a score tile with its prefix beat
QB = BR * D_QK                       # the query tile Q8, A layout
OB = D_V * BR * 2                    # O^T, FP16
PB = OB + XFER                       # a partial: O^T, then its row sums
PK, PVB = 16 * D_QK, D_V * 16        # the new piece: its key rows, its value columns
QROW = 2 * D_QK                      # a head's query row, FP16 [q~ | q_pe]
XB = 2 * HIDDEN                      # x in a_row
LAT = 2 * KV_RANK                    # a head's o~ in a_row
OAR = 2 * V_HEAD                     # a head's o8 in a_row
WQ_ORDER = (4, 5, 0, 1, 2, 3)        # W_q's chunks of a head as the ring streams them: q_pe's two, then q_nope's four
N_KVA = D_QK // QC                   # W_KVA's chunks
LINK_BW = 231.2e9 / 8                # bytes a second of one memory-chip link (the eight links' aggregate over eight)

# the constants of a cluster 0 (FP16, in L1 before the window): q_nope's factors spread into a_row, q_pe's, the RoPE
# tables at P, q~'s factors, W_UV's factors spread into a_row
C0_NSP, C0_FPE, C0_COS, C0_SIN, C0_FQT, C0_UVSP = 0, 512, 640, 768, 896, 1920
C0_BYTES = 2432
# the constants of the new key's cluster: W_KVA's factors, the RoPE tables at P, the new tile's mask (INT32 [64, 16])
C1_FKV, C1_COS, C1_SIN, C1_MASK = 0, 1152, 1280, 1408
C1_BYTES = C1_MASK + BC * BR * 4


def a_row(v):
    """A row of values in a_row: value c at (c // 4) * 8 + c % 4, the other places zero (the GEMV's a_blk 8)."""
    v = np.asarray(v).reshape(-1)
    c = np.arange(v.size)
    out = np.zeros(2 * v.size, dtype=v.dtype)
    out[(c // 4) * 8 + c % 4] = v
    return out


def f32bits(v):
    return int(np.asarray(v, dtype=np.float32).reshape(()).view(np.uint32))
