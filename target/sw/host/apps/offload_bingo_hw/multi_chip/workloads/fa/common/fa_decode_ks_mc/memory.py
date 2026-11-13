# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The data and the memory (README.md, THE CACHE and THE ORDERS): the data and its golden, the staging in L3 and in
the memory chips' HBM, one weight ring a chip and its loads, every cluster's exchange buffer."""

import numpy as np

from bingo_kernel_args import SnaxBingoKernelIdmaRingLoadArgs
from libs.blocks import WeightRings

import chiplet_mc as cm
import fa_decode_data as fdd

from .consts import (ARENA, BC, BEAT, C0_BYTES, C1_BYTES, HIDDEN, KB, LAT, MRUN, N_KVA, OAR, PB, Q_HEAD, Q_NOPE, QB, QC,
                     QROW, VB, WCH, WQ_ORDER, XB, XFER, a_row)
from .golden import KsGolden


class KsMemory:
    """FaDecodeKs.mem_handles and the ring: no task is made here but the ring's loads (load), which build() asks
    for."""

    def mem_handles(self):
        """The data (fa_decode_data.FaDecodeData) and its golden (KsGolden) with their report and the cross-check of
        head 0's Q8; _stage (the gathers' and multicasts' areas in L3, every chip's weights, cache shard and constants
        in its memory chip's HBM); one weight ring a chip (_make_ring); every cluster's exchange buffer (its offsets:
        the query, the flags, the arenas, the partial)."""
        R, nchip, extra = self.ROOT, len(self.chips), self.kva_extra
        d = self.d = fdd.FaDecodeData(self.kv_path, self.w_cache, cache=self.P, bc=BC, n_calib=self.n_calib,
                                      rope_perm=True, mask_c=True, wbits=4)
        if d.H != nchip:
            raise ValueError(f"{nchip} chips: the attention has {d.H} heads, one a chip")
        self.gold = KsGolden(d, self.units, self.chips, self.fold_order(), self.NTC)
        sc = lambda q: f"{q.s:.4g}"     # noqa: E731
        print(f"[{self.TAG}] DeepSeek-V2-Lite layer 1 MLA, one decode token at position {self.P} over keys "
              f"0..{self.P}, {d.H} heads, INT{d.wbits} per column; D-port shifts k_x={d.k_x} k_uk={d.k_uk} "
              f"k_uv={d.k_uv} k_s={d.k_s} k_o={d.k_o} k_wo={d.k_wo}; scales s_x={sc(d.s_x)} s_qn={sc(d.s_qn)} "
              f"s_qt={sc(d.s_qt)} s_qpe={sc(d.s_qpe)} s_c={sc(d.s_c)} s_ot={sc(d.s_ot)} s_o={sc(d.s_o)}")
        for line in self.gold.report(d):
            print(f"[{self.TAG}] {line}")
        self._cross_check()
        print(f"[{self.TAG}] key split: {self.NTC} cached tiles of {BC} keys, {self.NTC // nchip} a chip on its "
              f"cluster 0, the new tile (key {self.P}) on {R:02x}'s cluster 1; W_KVA's chunks 16.. on "
              f"{[f'{k:02x}' for k in extra]}; m* by chain; fold order {[f'{k:02x}' for k in self.order]}")
        self._stage()
        self._make_ring()
        # ---- every cluster's exchange buffer, on whole transfer units -----------------------------------------------
        self.OFF_X, self.OFF_FLAG, self.OFF_MFLAG = 0, QB, QB + XFER
        self.OFF_MX = QB + 2 * XFER                  # an FA arena: the pass-1 arena of a chip's cluster 0 (the root's:
        self.OFF_MLOC = self.OFF_MX + MRUN           # its two clusters' maximum), whose mrun is the chip's maximum
        self.OFF_A2 = self.OFF_MX + ARENA            # the pass-2 arena: m* lands in its mrun
        self.OFF_M = self.OFF_A2 + MRUN
        self.OFF_MR = self.OFF_A2 + ARENA            # the root's arena of m*
        self.OFF_LOC = self.OFF_MR + ARENA           # the chip's partial for the chain
        self.make_xch(self.OFF_LOC + PB, PB, dst_lead=BEAT)

    def _cross_check(self):
        """Head 0's Q8 against FaDecodeData's own chain of the head (the 16-row prefill arithmetic, its row 0)."""
        ref = np.asarray(self.d.head(0)["q8"][0], np.int8)
        if not np.array_equal(ref, self.gold.Q8[0]):
            raise AssertionError("head 0's Q8 on the GEMV path differs from FaDecodeData.head(0)")

    # ---- data --------------------------------------------------------------------------------------------------------
    def _stage(self):
        st, d, gold, T = self.st, self.d, self.gold, self.TAG
        R, nchip = self.ROOT, len(self.chips)
        loc = lambda k: (self.mem[k] >> 4, self.mem[k] & 15)       # noqa: E731
        f16 = lambda a: np.asarray(a, dtype=np.float16).reshape(-1).view(np.uint8)    # noqa: E731
        # x's landing area and flag in every chip's L3 (the layer's input norm multicasts x there)
        st.put_zeros(f"{T}_xland", "int8_t", XB)
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64)
        # y: the row the clusters fill in the root's L3, a flag each
        st.put_zeros(f"{T}_yrow", "uint8_t", 2 * HIDDEN)
        st.put_zeros(f"{T}_yflag", "uint8_t", self.N_CL * 64)
        # the gathers into the root's L3: the W_KVA rows, the query rows, the o8 rows; the multicasts' landing areas in
        # every chip's L3: o~ and the o8 row
        for name, nb in (("kvraw", N_KVA * 2 * QC), ("kvrflag", N_KVA * 64), ("qrows", nchip * QROW),
                         ("qflag", nchip * 64), ("olat", nchip * LAT), ("olflag", 64), ("oland", nchip * OAR),
                         ("oflag", nchip * 64), ("omflag", 64)):
            st.put_zeros(f"{T}_{name}", "uint8_t", nb)
        # per chip, in its memory chip's HBM: its weights in ring order, its cache shard, its constants
        self.w_off, self.h_w, self.h_kv, self.h_c0, self.kv_new = {}, {}, {}, {}, None
        for i, k in enumerate(self.chips):
            h = self.head[k]
            chunks = [(f"kva{n}", d.wblob(d.wkva.q[:, QC * n:QC * (n + 1)])) for n in self.kva[k]]
            chunks += [(f"wq{c}", d.wblob(d.wq.q[:, h * Q_HEAD + QC * c:h * Q_HEAD + QC * (c + 1)])) for c in WQ_ORDER]
            chunks += [("wuk", d.wblob(d.wuk.q[h])), ("wuv", d.wblob(d.wuv.q[h]))]
            for c in (self.KC, self.QC):
                for s in range(2):
                    c0 = 64 * (i * self.NC + c) + QC * s
                    chunks.append((f"wo{c}{s}", d.wblob(d.wo.q[:, c0:c0 + QC])))
            self.w_off[k], off = {}, 0
            for nm, b in chunks:
                if b.size != WCH:
                    raise ValueError(f"chunk {nm}: {b.size} B, not {WCH}")
                self.w_off[k][nm] = (off, b.size)
                off += b.size
            self.h_w[k] = st.put_hbm(f"{T}_w_k{k:02x}", np.concatenate([b for _, b in chunks]).astype(np.int8),
                                     mem_chip=loc(k))
            tiles = []
            for t in self.tiles[k]:
                tiles += [d.layout.to_a(d.rows_all[t * BC:(t + 1) * BC], d.mesh),
                          d.layout.to_a(np.ascontiguousarray(d.c8_all[t * BC:(t + 1) * BC].T), d.mesh)]
            if k == R:
                self.kv_new = len(self.tiles[k]) * (KB + VB)          # the next tile's slot: the appended piece
                tiles.append(np.zeros(KB + VB, dtype=np.int8))
            self.h_kv[k] = st.put_hbm(f"{T}_kv_k{k:02x}", np.concatenate(tiles).astype(np.int8), mem_chip=loc(k))
            qs = np.asarray(d.q_scale(h), np.float16)
            c0b = np.concatenate([f16(a_row(qs[:Q_NOPE])), f16(qs[Q_NOPE:]), f16(gold.cos), f16(gold.sin),
                                  f16(d.qt_scale(h)), f16(a_row(np.asarray(d.uv_scale(h), np.float16)))])
            if c0b.size != C0_BYTES:
                raise ValueError(f"cluster 0's constants are {c0b.size} B, not {C0_BYTES}")
            self.h_c0[k] = st.put_hbm(f"{T}_c0_k{k:02x}", c0b, mem_chip=loc(k))
        c1b = np.concatenate([f16(d.kva_scale()), f16(gold.cos), f16(gold.sin), gold.mask.reshape(-1).view(np.uint8)])
        if c1b.size != C1_BYTES:
            raise ValueError(f"the new key's constants are {c1b.size} B, not {C1_BYTES}")
        self.h_c1 = st.put_hbm(f"{T}_c1_k{R:02x}", c1b, mem_chip=loc(R))

    # ---- the ring ----------------------------------------------------------------------------------------------------
    def ring_order(self, k):
        """Chip k's ring: (name, HBM address, bytes, the cluster that reads it), in push order."""
        w, kv = self.h_w[k].address, self.h_kv[k].address
        wo = lambda nm: (nm, w + self.w_off[k][nm][0], self.w_off[k][nm][1])       # noqa: E731
        out = [wo(f"kva{n}") + (self.KC,) for n in self.kva[k]]
        out += [wo(f"wq{c}") + (self.QC,) for c in WQ_ORDER] + [wo("wuk") + (self.QC,)]
        out += [(f"K{t}", kv + i * (KB + VB), KB, self.QC) for i, t in enumerate(self.tiles[k])]
        out += [(f"V{t}", kv + i * (KB + VB) + KB, VB, self.QC) for i, t in enumerate(self.tiles[k])]
        out += [wo("wuv") + (self.QC,)]
        out += [wo(f"wo{c}{s}") + (c,) for c in (self.KC, self.QC) for s in range(2)]
        return out

    def _make_ring(self):
        """One ring a chip (every chunk of both clusters, in ring order), pushed by its memory chip's engine; each run
        posted by the DM core of the cluster that reads its first chunk (WeightRings post_chunk, made by load() in ring
        order), the queue's weight_ring.queue entries the ring's share; each chunk read by one cluster."""
        p, st, T = self.p, self.st, self.TAG
        wr = cm.ring_params(p)
        self.ring_queue = int(wr.get("queue", 16))
        if not 2 <= self.ring_queue <= 32:
            raise ValueError(f"params weight_ring.queue={self.ring_queue}: 2..32 entries (two a run, one ring)")
        self.n_slots, self.ring_batch = int(wr.get("slots", 16)), int(wr.get("batch", 4))
        self.ring_head, self.ring_batch_head = int(wr.get("head", 0)), int(wr.get("batch_head", 0))
        seq = np.zeros((1024, 8), dtype=np.uint64)
        seq[:, 0] = np.arange(1024, dtype=np.uint64)
        self.h_seq = {m: st.put_hbm(f"{T}_ring_seq_m{m:02x}", seq.view(np.int8).reshape(-1), mem_chip=(m >> 4, m & 15))
                      for m in sorted(set(self.mem.values()))}
        self.h_flags = st.put_zeros(f"{T}_ring_flags", "uint8_t", self.n_slots * 64)
        self.h_wslots = st.put_zeros(f"{T}_wring_slots", "uint8_t", self.n_slots * self.slot_bytes, preload=True)
        self.rings, self.tk, self.ring_bytes = {}, {}, {}
        self.ring_posted = {k: 0 for k in self.chips}       # the chunks posted, in ring order
        self.last_chunk = {k: None for k in self.chips}
        for k in self.chips:
            rings = self.rings[k] = WeightRings(
                slots=self.h_wslots, flags=self.h_flags, n_rings=1, n_slots=self.n_slots, slot_bytes=self.slot_bytes,
                batch=self.ring_batch, head=self.ring_head, batch_head=self.ring_batch_head)
            order = self.ring_order(k)
            if len(order) > self.n_slots:
                raise ValueError(f"params weight_ring.slots={self.n_slots}: chip {k:02x} streams {len(order)} chunks "
                                 f"(every chunk keeps its slot)")
            self.tk[k] = {}
            for nm, src, nb, c in order:
                if nb > self.slot_bytes:
                    raise ValueError(f"chunk {nm}: {nb} B over a {self.slot_bytes}-B slot")
            self.ring_bytes[k] = sum(nb for _, _, nb, _ in order)
            rings.push_from(memchip=self.mem[k], engine=self.engine[k], seq=self.h_seq[self.mem[k]], edge=self.edge,
                            depth=self.ring_queue)
            self.queue_poster((self.mem[k], self.engine[k]), f"{T} ring, chip {k:#04x}", self.ring_queue,
                              rings.queue_span)
        self.last_name = {k: self.ring_order(k)[-1][0] for k in self.chips}

    def load(self, k, name, dst, deps=()):
        """Chip k's ring chunk `name` into the L1 buffer dst, on its reader's DM core: waits for the chunk's flag. Every
        chunk of the ring up to this one is posted first, in ring order (WeightRings post_chunk on the DM core of the
        cluster that reads the run's first chunk), and the load follows its post."""
        order = self.ring_order(k)
        i = next(j for j, o in enumerate(order) if o[0] == name)
        while self.ring_posted[k] <= i:
            nm, src, nb_, c_ = order[self.ring_posted[k]]
            tk, post = self.rings[k].post_chunk(
                lambda n, core, kern, args, after, c_=c_: self.node(k, c_, core, n, kern, args, after), 0,
                f"Ld_{nm}", src=src, nbytes=nb_, dm=self.roles["dm"])
            self.tk[k][nm] = tk + (c_, self.rings[k].last_take, post)
            self.ring_posted[k] += 1
        sl, fl, seq, nb, c, ex, post = self.tk[k][name]
        ld = self.nd(k, c, "dm", f"Ld_{name}", "__snax_bingo_kernel_idma_ring_load",
                     SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, dst, nb), list(deps) + [post])
        self.rings[k].read_chunk(0, seq - 1, ld, last=True)
        ld.exec_chunk = ex
        if name == self.last_name[k]:
            self.last_chunk[k] = ld
        return ld
