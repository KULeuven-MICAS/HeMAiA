# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The MoE drivers' constants and small helpers (moe_prefill_mc and moe_decode_mc share them)."""

import numpy as np

import chiplet_mc as cm
import moe_prefill_data as md

BLK = 16                      # a GEMM column block (Nu)
D_BLOCK = 256                 # one D-layout block: 8 rows x 16 FP16
SIMD_BEAT = 32                # one SIMD beat: a SIMD task writes whole ones
SHAPE3 = 3                    # VersaCore array shape (8, 8, 16)
MESH3 = md.MESH3
ROUTE_B = 32                  # one token's entry of the route table
SOFTMAX_ROWS = 32             # the softmax's rows a task (its scratch fits the SIMD pool)
B_PARITY_OFF = 8              # a parity-layout weight chunk starts one word into its slab: A's rows
                              # start 64-B aligned, so B is an odd number of words after A
EW_F16_ADD = cm.EW_F16_ADD
NOUT = 2                      # the exchange buffer's output slots the down phase's slices rotate through
DTMP_BUFS = 2                 # the expert down GEMMs' output buffers a cluster rotates through
SCALE_BUFS = 4                # the scale-tile buffers a cluster's loads rotate through


def chain_path_param(p, chips, root, whole=True):
    """params chain_path: the chips a ChainGather passes in data order, as hex chip ids, the root last (None:
    the default fold cycle). whole: the path must pass every chip."""
    cp = p.get("chain_path")
    if cp is None:
        return None
    path = [int(str(x), 16) for x in cp]
    if path[-1] != root or len(set(path)) != len(path) or not set(path) <= set(chips):
        raise ValueError(f"params chain_path={cp}: distinct chips of the platform, the root {root:02x} last")
    if whole and set(path) != set(chips):
        raise ValueError(f"params chain_path={cp}: must pass every chip")
    return path


def row_pitch(nbytes):
    """A row-major operand's row pitch: at least a row and 16 bytes (an odd quantise writes 16
    values past it), beat-aligned, and 32 mod 64 so the 8 rows of a GEMM pass sit in 8 banks."""
    p = int(nbytes) + 16
    p += (32 - p % 64) % 64
    return p


def beat_aligned(off, what):
    """A SIMD task's output offset inside its buffer: the kernels write whole beats."""
    if off % SIMD_BEAT:
        raise ValueError(f"{what}: output at byte {off} of its buffer, not on a {SIMD_BEAT}-byte SIMD beat")
    return off


def route_bytes(d, T):
    """The route table's bytes for T tokens (moe_topk_rows): per token its top 6 experts (0xFF-padded to 8),
    then their FP16 weights."""
    out = np.zeros((T, ROUTE_B), dtype=np.uint8)
    out[:, :8] = 0xFF
    out[:, :md.TOP_K] = d.ids.astype(np.uint8)
    w = np.zeros((T, 8), dtype=np.float16)
    w[:, :md.TOP_K] = d.w16
    out[:, 8:24] = w.view(np.uint8).reshape(T, 16)
    return out.reshape(-1)


def lists_bytes(d, held):
    """The row lists (moe_dispatch) of a chip holding experts `held`: per expert its rows, then their weights."""
    parts = []
    for e in held:
        rows, w = d.rows_of(e)
        parts += [np.array(rows, dtype=np.uint16).view(np.uint8), np.asarray(w, dtype=np.float16).view(np.uint8)]
    return np.concatenate(parts)


class Scale:
    """A weight chunk's scale tile as its dequantisation reads it: the address, and (the prefill) the copy task that
    put it there and the index of its buffer."""

    def __init__(self, addr, copy=None, buf=None):
        self.addr, self.copy, self.buf = addr, copy, buf

    @property
    def deps(self):
        return [self.copy] if self.copy is not None else []
