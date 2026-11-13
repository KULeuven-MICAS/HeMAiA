# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The MoE decode's constants and helpers: the token's a_row, the spread of an A16 row into the quantiser's input, the
padded-row stream multiply, and a piece of an MLP on a chip."""

import numpy as np
from bingo_kernel_args import SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelXdma6dArgs

import moe_prefill_data as md
from moe_prefill_mc.common import BLK, D_BLOCK, SIMD_BEAT

ROWS = md.TILE_ROWS           # the token's tile of rows (the token, then zero rows): the softmax's rows, a slot's
                              # row-list capacity
GAP = 64                      # the HBM image's gap before a ring's first chunk that waits for the route
GV_MESH = (16, 4, 16)         # the layouts of VersaCore's shapes 0..2, which the GEMV reads
ATOM, LANES = 8, 8            # the xDMA's 8-byte atom (4 FP16 values) and its lanes a beat
SPREAD_ATOMS = 2 * LANES      # the atoms of the A16 row one step of the spread's outer loop moves


def a_row_bytes(row):
    """An INT8 row as the GEMV's a_row: value c at byte (c // 4) * 8 + c % 4, the other 4 bytes of each word zero."""
    row = np.asarray(row, dtype=np.int8).reshape(-1)
    c = np.arange(row.size)
    out = np.zeros(2 * row.size, dtype=np.int8)
    out[(c // 4) * 8 + c % 4] = row
    return out


def spread_steps(cols):
    """The outer steps of the spread of an A16 row of `cols` values (SPREAD_ATOMS atoms a step)."""
    return -(-(cols // 4) // SPREAD_ATOMS)


def spread_args(src, dst, cols):
    """The xDMA pass that spreads an FP16 row of `cols` values (src) into simd_quant_rows' input (dst): atom t
    (values 4t .. 4t + 3) to byte 256 (t // 2) + 16 (t % 2), so the 32-byte beat m, 256 bytes on from beat m - 1, holds
    atoms 2m and 2m + 1, each followed by 8 bytes the pass never writes. The lanes take atoms 2 apart (16 bytes in the
    row, 256 in dst), the inner loop the atom of the pair, the outer loop SPREAD_ATOMS atoms at a time; the last step
    moves whole lanes, so src is read and dst written up to spread_steps(cols) steps (128 and 2048 bytes each). The
    nest is walked here and compared with that mapping."""
    steps = spread_steps(cols)
    args = SnaxBingoKernelXdma6dArgs(src, dst, 2 * ATOM, D_BLOCK, [ATOM, SPREAD_ATOMS * ATOM], [2, steps],
                                     [2 * ATOM, LANES * D_BLOCK], [2, steps])
    j, b, lane = np.meshgrid(np.arange(2), np.arange(steps), np.arange(LANES), indexing="ij")
    so = (j * ATOM + b * SPREAD_ATOMS * ATOM + lane * 2 * ATOM).reshape(-1)
    do = (j * 2 * ATOM + b * LANES * D_BLOCK + lane * D_BLOCK).reshape(-1)
    t = so // ATOM
    if so.min() < 0 or np.any(so % ATOM) or not np.array_equal(np.sort(t), np.arange(steps * SPREAD_ATOMS)) \
            or not np.array_equal(do, D_BLOCK * (t // 2) + 2 * ATOM * (t % 2)):
        raise ValueError("the spread's xDMA nest does not make the quantiser's input")
    return args


class SimdMulRowsF16Args(SnaxBingoKernelSimdMulF16Args):
    """out = a (.) b over `rows` rows of `cols` FP16 values, each operand's rows `row_stride` bytes apart and out packed:
    the stream kernel's padded-row read (src_row_stride, device_kernel_args.h), both operands at the same stride."""

    def __init__(self, a_addr, b_addr, out_addr, rows: int, cols: int, row_stride: int):
        super().__init__(a_addr, b_addr, out_addr, rows, cols)
        if row_stride % SIMD_BEAT or row_stride < 2 * cols:
            raise ValueError(f"row_stride={row_stride}: whole SIMD beats, at least a row of {cols} values")
        self.row_stride = int(row_stride)

    def get_c_field_assignments(self, handle_name_map):
        a = super().get_c_field_assignments(handle_name_map)
        a["src_row_stride"] = str(self.row_stride)
        return a


class Piece:
    """A run of one MLP's intermediate columns on a chip: e (None: the shared MLP), its first column k0 within the MLP,
    n columns, b0 of its column blocks on cluster 0 (the rest on cluster 1)."""

    def __init__(self, e, k0, n):
        self.e, self.k0, self.n, self.b0 = e, int(k0), int(n), 0

    @property
    def nb(self):
        return self.n // BLK

    def cols(self, c):
        """Cluster c's columns of the piece: (first column within the piece, columns)."""
        return (0, BLK * self.b0) if c == 0 else (BLK * self.b0, self.n - BLK * self.b0)

    def __repr__(self):
        return f"{'sh' if self.e is None else f'e{self.e}'}[{self.k0}:{self.k0 + self.n}]"
