# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The data of DeepSeek-V2-Lite's layer-1 MLA attention for ONE DECODE STEP over a cache (FaDecodeData), and the D2D
links a unicast or a broadcast crosses (_route, _bcast_tree): what the key-split decode (fa_decode_ks_mc.py) and the
whole-layer decode (../../dsv2/common/dsv2_decode_mc.py) build on.

The token at position P (params cache, default 1,024) attends to keys 0 .. P: the P cached keys and its own. The
golden is fa_prefill_data.py's for a fresh causal prompt of P + 1 tokens, the token being its last: the same
calibration, the same keys, the same device arithmetic. The cached keys are P / 16 pieces [key rows 9 KiB | value
columns 8 KiB] as the prefill leaves them (fa_prefill_data kv_piece).
"""

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa_prefill_data  # noqa: E402
from fa_prefill_mc import BR, D_QK, D_V  # noqa: E402

PIECE = 16                     # tokens a piece of the keys


class FaDecodeData(fa_prefill_data.FaPrefillData):
    """fa_prefill_data's data of a fresh prompt of cache + 1 tokens, the decode token at position P = cache its last:
    the calibration (the last n_calib positions), the latent's c_fold, the scales and the device's keys of all P + 1
    positions, as that prefill's golden makes them. Then the decode's views:

        rows, c8          the sweep's keys: positions 0 .. P - 1 (the cache), then the new tile: the token's key in all
                          16 rows of its piece (the device makes it from the token's 16 rows), the rest zero
        hq, cos_rep, ..   the query tile: the token in all 16 rows (head(), x_chunk_a(0, 16), rope_tables(0, 16))
        mask_patterns     the last tile's mask: every key after position P, in every lane

    head(h) is then the parent's chain for the 16 rows, attention() the sweep over the decode's tiles."""

    def __init__(self, kv_path, weights_cache, cache=1024, bc=64, n_calib=32, model="dsv2-lite-bf16",
                 rope_perm=False, mask_c=False, wbits=8):
        if cache % bc or cache // bc < 4:
            raise ValueError(f"cache {cache}: whole key tiles of {bc}, at least four (the new key starts a tile)")
        # the parent: the checkpoint's weights packed, and the prefill that left the cache (its checks take whole key
        # tiles, keys = tokens, one chunk); the T-dependent part is made again below for the P + 1 positions
        super().__init__(kv_path, weights_cache, tokens=cache, keys=cache, bc=bc, n_calib=n_calib,
                         model=model, fresh=True, chunk=cache, rope_perm=rope_perm, mask_c=mask_c, wbits=wbits)
        P, n = int(cache), int(cache) + 1
        z = np.load(kv_path)
        if z["c_unit"].shape[0] < n:
            raise ValueError(f"{kv_path}: {z['c_unit'].shape[0]} tokens; the decode needs {n}")
        self.P, self.T, self.keys = P, n, n
        self.c_unit = z["c_unit"][:n].astype(np.float32)
        self.kpe = z["kpe"][:n].astype(np.float32)
        self.pos = np.arange(n)
        x1 = z["x1"][:n].astype(np.float32)
        self.k_o = self.fp.d_shift_for_depth(n)          # the last PV's depth: the P + 1 keys it adds
        self._calibrate(x1, n_calib)
        self.hq = np.stack([self._input(x) for x in x1])
        self._cos_sin()
        self._device_kv()
        self.x1 = x1
        # the device's keys of every position (the parent's rows / c8), and the sweep's tiles
        rep = lambda a: np.repeat(np.asarray(a)[P:P + 1], BR, axis=0)    # noqa: E731
        self.rows_all, self.c8_all = self.rows, self.c8
        self.NT = -(-n // bc)
        K = np.zeros((self.NT * bc, D_QK), dtype=np.int8)
        V = np.zeros((self.NT * bc, D_V), dtype=np.int8)
        K[:P], V[:P] = self.rows_all[:P], self.c8_all[:P]
        K[P:P + PIECE], V[P:P + PIECE] = rep(self.rows_all), rep(self.c8_all)
        self.rows, self.c8 = K, V
        self.hq_all = self.hq
        self.hq, self.cos_rep, self.sin_sgn = rep(self.hq_all), rep(self.cos_rep), rep(self.sin_sgn)
        # the last tile's mask, S^T [bc keys, 16 lanes]: a key after the token's position
        keys = (self.NT - 1) * bc + np.arange(bc)[:, None]
        self.mask_patterns = [np.where(np.broadcast_to(keys > P, (bc, BR)), np.float16(self.hm.MASKED_SCORE),
                                       np.float16(0)).astype(np.float16)]
        self._plans = {0: (self.NT, {(self.NT - 1, 0): 0})}
        self._heads = {}
        with open(os.path.abspath(__file__), "rb") as f:
            src = f.read()
        self._cache_key = hashlib.sha1(repr((self._cache_key, "decode", P, bc, n_calib,
                                             hashlib.sha1(src).hexdigest())).encode()).hexdigest()[:16]

    def attention(self, q8):
        """The sweep of the query tile's lanes over the decode's key tiles, the last one masked after position P."""
        nt, plan = self._plans[0]
        return self._sweep(q8, nt, plan)

    def piece(self, q):
        """Piece q (tokens 16 q .. 16 q + 15) as the sweep reads it: [key rows | value columns] INT8."""
        return self.kv_piece(PIECE * q)

    def report(self, heads_in_order):
        """How far the device's y of the token is from the float attention of the checkpoint."""
        y = self.y16(self.reduce(heads_in_order, 0, 1))[0].astype(np.float64)
        ref = np.asarray(self.y_ref, dtype=np.float64)
        cos = float(y @ ref / max(np.linalg.norm(y) * np.linalg.norm(ref), 1e-30))
        rel = float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30))
        a, b = np.asarray(self.c_unit16, dtype=np.float64).ravel(), self.c_unit.astype(np.float64).ravel()
        ccos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        sc = lambda q: f"{q.s:.4g}"     # noqa: E731
        return [f"DeepSeek-V2-Lite layer 1 MLA attention, one decode token at position {self.P} over keys 0..{self.P} "
                f"(a fresh causal prompt of {self.T} tokens, the token its last; the keys made as the device makes "
                f"them: latent vs the model's cosine {ccos:.5f}, c_fold {self.c_fold:g}), {self.NT} key tiles of "
                f"{self.bc}, {self.H} heads, INT{self.wbits} per column",
                f"D-port shifts k_x={self.k_x} k_uk={self.k_uk} k_uv={self.k_uv} k_s={self.k_s} k_o={self.k_o} "
                f"k_wo={self.k_wo}; scales s_x={sc(self.s_x)} s_qn={sc(self.s_qn)} s_qt={sc(self.s_qt)} "
                f"s_qpe={sc(self.s_qpe)} s_c={sc(self.s_c)} s_ot={sc(self.s_ot)} s_o={sc(self.s_o)}",
                f"device y of the token vs the float attention: cosine {cos:.5f}, relative L2 error {rel:.4f}"]


def _route(a, b):
    """The D2D links a unicast from chip a to chip b crosses: X first, then Y (hemaia_d2d_link_router_rc)."""
    x, y, bx, by, out = a >> 4, a & 15, b >> 4, b & 15, []
    while x != bx:
        nx_ = x + (1 if bx > x else -1)
        out.append(tuple(sorted(((x << 4) | y, (nx_ << 4) | y))))
        x = nx_
    while y != by:
        ny = y + (1 if by > y else -1)
        out.append(tuple(sorted(((x << 4) | y, (x << 4) | ny))))
        y = ny
    return out


def _bcast_tree(src, grid):
    """The D2D links a broadcast (chip id 0xFF) from `src` crosses, by hemaia_d2d_link_router_rc's rules: the source
    sends every way; a chip above it in its column north, east and west; a chip in its row away from it and south; a
    chip above its row away from its column; a chip below its row south."""
    sx, sy, out = src >> 4, src & 15, []
    N, S, E, W = (0, -1), (0, 1), (1, 0), (-1, 0)

    def dirs(x, y):
        if (x, y) == (sx, sy):
            return (N, S, E, W)
        if x == sx and y < sy:
            return (N, E, W)
        if y == sy:
            return ((W,) if x < sx else (E,)) + (S,)
        if y < sy:
            return (W,) if x < sx else (E,)
        return (S,)

    stack = [(sx, sy)]
    while stack:
        x, y = stack.pop()
        for dx, dy in dirs(x, y):
            nx_, ny = x + dx, y + dy
            if nx_ < 0 or ny < 0 or ((nx_ << 4) | ny) not in grid:
                continue
            out.append(tuple(sorted(((x << 4) | y, (nx_ << 4) | ny))))
            stack.append((nx_, ny))
    return out

