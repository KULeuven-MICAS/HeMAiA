# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaBuild's data and memory: the data and its golden, everything staged in L3 and HBM, the L1 buffers, and where y
lands."""

import numpy as np

import fa_prefill_data
from bingo_mem_handle import BingoMemSymbol
from libs.blocks import FaSweepCfg, fa_sweep_alloc

from .consts import BR, D_QK, HIDDEN, KV_RANK, KVP, Q_NOPE, ROPE


class FaMemory:

    def mem_handles(self):
        """The data and the memory; no task yet: the data and its golden (fa_prefill_data.FaPrefillData) and their
        report; _stage (every input staged in L3 and HBM: x, the key pieces and their flags, W_KVA, the weights and the
        dequantisation constants of each head); the load streams' L1 slabs (make_streams); one weight ring a chip
        (_make_rings); the L1 buffers (_buffers: the exchange buffer, the constants, the keys' constants)."""
        d = self.d = fa_prefill_data.FaPrefillData(self.kv_path, self.w_cache, tokens=self.T, keys=self.keys,
                                                   bc=self.bc, n_calib=self.n_calib, fresh=True, chunk=self.C,
                                                   rope_perm=True, mask_c=True, wbits=self.wbits,
                                                   skip_masked=self.skip_masked)
        self.lay = d.layout
        for line in d.report(self.fold_heads):
            print(f"[fp] {line}")
        print(f"[fp] {self.T} tokens in {self.NCH} chunks of {self.C}: {self.NQ} query tiles of {BR} rows a "
              f"cluster, {self.NT} K/V tiles of {self.bc} keys; fold order "
              f"{[f'{k:02x}' for k in self.order]}")
        self._stage()
        self.make_streams()
        self._make_rings()
        self._buffers()

    def flag_dep(self, t, k, c):
        """Whether task t enters a flag wait on chip k cluster c as a dependency (a task of that chip), else as an
        ordering edge (the flag proves the landing)."""
        return t.assigned_chiplet_id == k

    def xh(self, i, c):
        """The half of chunk i's rows that cluster c takes."""
        return c ^ (i & 1) if self.row_swap else c

    def r0(self, i, c):
        """The first row of chunk i that cluster c takes."""
        return i * self.C + self.xh(i, c) * self.ROWS

    # ---- data -----------------------------------------------------------------------------------
    def _stage(self):
        st, d, T = self.st, self.d, self.TAG
        # the prompt's rows, A layout per chunk: the root's source and every chip's landing (every chip's copy in L3
        # starts as zeros for a multicast inside the window to fill, a tile of 16 tokens at a time behind its own flag,
        # from the prompt's rows in the HBM of the root's memory chip)
        self.xsrc = np.concatenate([d.x_chunk_a(i * self.C, self.C) for i in range(self.NCH)]).astype(np.int8)
        self.h_xsrc = st.put_zeros(f"{T}_xsrc", "int8_t", self.xsrc.size, preload=True)
        m = self.mem[self.ROOT]
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", self.xsrc, mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_xtflag", "uint8_t", (self.T // 16) * 64)
        st.put_zeros(f"{T}_xland", "int8_t", self.NCH * self.XB, preload=True)
        st.put_zeros(f"{T}_xflag", "uint8_t", self.NCH * 64)
        # the root's y lives in its x buffers (y of chunk i, cluster 0 in the landing buffer of chunk i, cluster 1 in
        # the source of chunk i): every chip's image is the same, and an 8 MiB L3 cannot hold y beside both x buffers,
        # the ring and the task tables. A chunk's y is stored after its reduce, which follows every use of its x.
        if self.NC != 2 or self.XB != self.YB:
            raise ValueError("y over x needs two clusters and a chunk's x the size of a cluster's y")
        # RoPE tables, per chunk and cluster: cos then sin_signed, D layout [rows, 64]
        rope = []
        for i in range(self.NCH):
            for c in range(self.NC):
                cs, sn = d.rope_tables(self.r0(i, c), self.ROWS)
                rope += [cs, sn]
        self.h_rope = st.put(f"{T}_rope", "uint16_t", np.concatenate(rope).astype(np.float16).view(np.uint16))
        self.ROPE_B = self.ROWS * ROPE * 2
        self.h_w, self.h_c, self.h_wkv, self.h_kc = {}, {}, {}, {}
        # W_KVA whole in each memory chip's HBM (no task reads it: each chip streams its own slice, wkvc); the 16-token
        # pieces and their landed flags in every chip's L3
        wkv = np.concatenate(d.kva_chunks()).astype(np.int8)
        for m in sorted(set(self.mem.values())):
            self.h_wkv[m] = st.put_hbm(f"{T}_wkv_m{m:02x}", wkv, mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_kv", "int8_t", (self.T // 16) * KVP, preload=True)
        st.put_zeros(f"{T}_kvflag", "uint8_t", self.NCH * 64)          # a chunk's pieces landed (multicast)
        st.put_zeros(f"{T}_kvnflag", "uint8_t", (self.T // 16) * 64)   # a piece landed on the root
        # each chip's W_KVA slice in its memory chip's HBM; per owned tile the gathered kv and a flag a chip
        self.h_wkvc = {k: st.put_hbm(f"{T}_wkvc_k{k:02x}", d.kva_cols(16 * self.kvg[k][0], 16 * self.kvg[k][1]),
                                     mem_chip=(self.mem[k] >> 4, self.mem[k] & 15)) for k in self.chips}
        st.put_zeros(f"{T}_kvraw", "uint8_t", self.KVR_TILES * self.KVD)
        st.put_zeros(f"{T}_kvrflag", "uint8_t", self.KVR_TILES * len(self.chips) * 64)
        # per chip: W_KVA's dequantisation (D layout [16, 576]), then each of its tiles' RoPE tables (cos, sin_signed),
        # KC_TILE bytes a tile
        kva = d.scale_tile(d.kva_scale()).view(np.uint8)
        for k in self.chips:
            tabs = []
            for q in self.kv_tiles[k]:
                cs, sn = d.rope_tables(16 * q, 16)
                tabs += [np.asarray(t, dtype=np.float16).view(np.uint8) for t in (cs, sn)]
            img = np.concatenate([kva] + tabs)
            self.h_kc[k] = st.put_hbm(f"{T}_kc_k{k:02x}", img.astype(np.uint8),
                                      mem_chip=(self.mem[k] >> 4, self.mem[k] & 15))
        self.KC_ROPE, self.KC_TILE = kva.size, 2 * 16 * ROPE * 2
        self.KC_BYTES = kva.size + len(self.kv_tiles[self.ROOT]) * self.KC_TILE
        # per head: the weights as the ring streams them, and the dequantisation constants: q_nope's 128 scales and
        # o's in the A layout of the cluster's rows, q~'s of one query tile, where the fused tasks multiply the
        # reshaped values; q_pe's stay D (RoPE reads them dequantised); then the causal masks as the score matmul's
        # INT32 [bc, 16] biases
        self.w_off = {}
        for k in self.chips:
            h = self.head[k]
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            chunks = d.weight_chunks(h, wo_block=None)
            self.w_off[k], off = [], 0
            for nm, b in chunks:
                self.w_off[k].append((nm, off, b.size))
                off += b.size
            self.h_w[k] = st.put_hbm(f"{T}_w_k{k:02x}", np.concatenate([b for _, b in chunks]), mem_chip=loc)
            qs = d.q_scale(h)
            sq = np.concatenate([d.scale_tile_a(qs[:Q_NOPE], self.ROWS).view(np.uint8),
                                 np.tile(d.scale_tile(qs[Q_NOPE:]), self.ROWS // 16).view(np.uint8)])
            sqt, suv = d.scale_tile_a(d.qt_scale(h), 16), d.scale_tile_a(d.uv_scale(h), self.ROWS)
            masks = d.mask_bias_patterns()
            const = np.concatenate([sq.view(np.uint8), sqt.view(np.uint8), suv.view(np.uint8),
                                    np.concatenate([m.reshape(-1) for m in masks]).view(np.uint8)])
            self.h_c[k] = st.put_hbm(f"{T}_c_k{k:02x}", const.astype(np.uint8), mem_chip=loc)
        self.SQ_B = (self.ROWS // 16) * 2048           # one W_q block's scales: 16 rows x 64 FP16 a row tile
        self.C_SQ, self.C_QT = 0, 3 * self.SQ_B
        self.C_UV = self.C_QT + 16384
        self.C_MASK = self.C_UV + self.ROWS * Q_NOPE * 2
        self.MASK_B = self.bc * BR * 4
        self.C_BYTES = self.C_MASK + len(d.mask_patterns) * self.MASK_B

    # ---- buffers --------------------------------------------------------------------------------
    def _buffers(self):
        d = self.d
        self.xch_bytes = self.YB
        self.xch_zero = False
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.xch_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.sweep_cfg = FaSweepCfg(bc=self.bc, nt=self.NT, nq=self.NQ, br=BR, k_s=d.k_s, k_o=d.k_o,
                                    a_exp=d.a_exp, a_n_f32bits=d.a_n_bits)
        self.consts = {cl: self.cx(cl).l1(f"{self.TAG}_consts", self.C_BYTES)
                       for cl in range(self.N_CL)}
        # W_KVA's dequantisation and the chip's tiles' RoPE tables, on cluster 0 (the tiles' keys)
        self.kvc = {(k, 0): self.cx_chip(k).at(0).l1(f"{self.TAG}_kvc", self.KC_BYTES) for k in self.chips}
        self._bufs, self._dst = {}, {}

    def kbuf(self, cl):
        """kv_slabs split_kb: cluster cl's buffer for its odd key tiles (one key tile)."""
        if cl not in self._kbuf:
            self._kbuf[cl] = self.cx(cl).l1(f"{self.TAG}_kbuf", self.bc * D_QK)
        return self._kbuf[cl]

    def bufs(self, cl, i):
        """The buffers of chunk i on cluster cl. A chunk has its own, so the static L1 planner overlays
        the ones whose lifetimes cannot meet (the projection's with the previous sweep's, say)."""
        if (cl, i) in self._bufs:
            return self._bufs[cl, i]
        k, c = self.chip(cl), self.local(cl)
        rows, nq = self.ROWS, self.NQ
        l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}_i{i}", nb)    # noqa: E731
        b = {}
        b["xrows"] = l1("xrows", rows * HIDDEN)
        # q: three D regions of 64 columns (the last is q_pe), rows x 64 FP16 each; simd_rope's operand block
        # [x | cos | xswap | sin], ROPE_B each: q_pe's GEMM and dequantisation write slot 0, the tables slots 1 and 3,
        # the pair swap slot 2
        b["rope4"] = l1("rope4", 4 * self.ROPE_B)
        b["q"] = [l1(f"q{r}", rows * 64 * 2) for r in range(2)] + [b["rope4"]]
        b["a16n"] = l1("a16n", rows * Q_NOPE * 2)             # q_nope, A layout FP16
        b["qn8"] = l1("qn8", rows * Q_NOPE)
        b["qt"] = l1("qt", rows * KV_RANK * 2)                # q~, D layout FP16
        b["a16q"] = l1("a16q", rows * KV_RANK * 2)
        b["pe"] = l1("pe", rows * ROPE * 2)
        b["a16p"] = l1("a16p", rows * ROPE * 2)
        b["q8"] = [l1(f"q8_{q}", D_QK * BR) for q in range(nq)]
        b["uv"] = l1("uv", rows * Q_NOPE * 2)                 # o_h, D layout FP16
        b["a16u"] = l1("a16u", rows * Q_NOPE * 2)
        b["o8"] = l1("o8", rows * Q_NOPE)
        # the sweep's names keep their sort order inside a chunk (the softmax's strides need it)
        b["swp"] = fa_sweep_alloc(lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}_i{i}", nb),
                                  self.sweep_cfg)
        self._bufs[cl, i] = b
        return b

    # ---- where y lands --------------------------------------------------------------------------
    def y_root(self, i, half=None):
        """The chip that collects and stores chunk i's y (half `half`: its W_O block): params y_roots, else the root."""
        if not self.y_roots:
            return self.ROOT
        r = self.y_roots[(2 * i + (half or 0)) % len(self.y_roots)]
        if r not in self.order:
            raise ValueError(f"y_roots: chip {r:02x} is not a compute chip")
        return r

    def y_order(self, r):
        """The fold cycle from collector r: r, then the cycle on from it (self.order is one cycle, so every hop of a
        chain stays between neighbours)."""
        j = self.order.index(r)
        return self.order[j:] + self.order[:j]

    def y_fold_heads(self, r):
        """The heads in the order collector r's chain adds their partials: the chain's chips, r's own partial last."""
        o = self.y_order(r)
        return [self.head[k] for k in o[1:] + o[:1]]

    def dst(self, i, c, half=None):
        """The collector's cluster c landing buffer of chunk i's reduce: the reduced rows, then stored (a W_O block's
        half in a buffer of its own)."""
        key = (i, c) if half is None else (i, c, half)
        if key not in self._dst:
            name, nb = (f"{self.TAG}_dst_i{i}", self.YB) if len(key) == 2 else (f"{self.TAG}_dst_i{i}h{half}",
                                                                                 self.YB // 2)
            self._dst[key] = self.cx_chip(self.y_root(i, half)).at(c).l1(name, nb)
        return self._dst[key]

    def y_sym(self, i, c, off=0, chip_id=None):
        """Where the collector stores cluster c's rows of y of chunk i: over x (cluster 0 in chunk i's landing area,
        cluster 1 in its source)."""
        name, base = (f"{self.TAG}_xland" if c == 0 else f"{self.TAG}_xsrc"), i * self.XB
        if chip_id is None:
            return BingoMemSymbol(name, offset=base + off)
        return BingoMemSymbol(name, offset=base + off, chip_id=chip_id)
