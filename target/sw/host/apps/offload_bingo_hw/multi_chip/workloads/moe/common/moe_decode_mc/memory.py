# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DBuild's data and memory: the token's data, the golden, the HBM images of the weights (their split at the route),
the L3 landing areas and every cluster's L1 buffers (README.md, THE X LAYOUT)."""

import os

import numpy as np
from bingo_kernel_args import SnaxBingoKernelMoeDispatchArgs
from bingo_mem_handle import BingoMemSymbol
from driver import repo_root

import moe_prefill_data as md
from moe_prefill_mc.common import BLK, D_BLOCK, DTMP_BUFS

from .common import ATOM, GAP, GV_MESH, LANES, ROWS, SPREAD_ATOMS, a_row_bytes, spread_steps

REPO = repo_root(__file__)


class Memory:

    def mem_handles(self):
        """The data, the placement, the golden and the memory; no task is made here: the token's data (_data) and the
        sizes it gives (the output slices, params slice, its default a cluster's outputs; X's multicast, the token's
        a_row); the pieces of the token's experts and of the shared MLP on the chips (_place, params routed_split); the
        golden (_golden); then _stage(): X and Y's row in the HBM of the root's memory chip, every chip's landing areas
        and flags, every cluster's weight chunks and factor rows in its memory chip's HBM; make_streams(): each
        cluster's L1 weight slabs; make_rings(): each chip's weight rings in its L3; _buffers(): every cluster's L1
        buffers."""
        d = self.d = self._data(self.args, self.p, self.hw)
        self.lay = d.layout
        self.H, self.IE, self.ISH = d.H, d.I_e, d.I_sh
        self.EB, self.SHB = self.IE // BLK, self.ISH // BLK
        self.SW = int(self.p.get("slice", self.H // self.NC))
        if (self.H // self.NC) % self.SW or self.SW % BLK:
            raise ValueError(f"params slice={self.SW}: whole column blocks dividing a cluster's {self.H // self.NC} "
                             f"outputs")
        self.NSLC = self.H // self.NC // self.SW
        self.SL = self.SW * 2                             # one output slice: the token's row of it, FP16
        self.XT, self.NXT = 2 * self.H, 1                 # X's multicast: one tile, the token's a_row
        self._place(self.p)
        self._golden()
        self._stage()
        self.make_streams()
        self.make_rings()
        self._buffers()

    def _data(self, args, p, hw):
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        return md.MoePrefillData(cache, hw=hw, seed=int(p.get("seed", 1)),
                                 n_calib=int(p.get("n_calib", 8)), n_cal_act=int(p.get("n_cal_act", 32)),
                                 tokens=self.T, inter_e=p.get("inter_e"), inter_sh=p.get("inter_sh"))

    # ---- the golden -------------------------------------------------------------------------------------------------
    def _golden(self):
        d, add16 = self.d, self.d.fp.add16
        a8sh = d.a8_shared()                              # [1, I_sh]
        y = np.zeros(self.H, dtype=np.float16)
        for c in range(self.NC):
            for c0, w in self.out_slices(c):
                parts = {}
                for k in self.chips:
                    sh = self.pieces[k][0]
                    yk = d.d16(d.sh, a8sh, sh.k0, sh.k0 + sh.n, c0, w)[0].copy()
                    for pc in self.pieces[k][1:]:
                        yk = add16(yk, d.d16(d.expert(pc.e), d.a8_expert(pc.e), pc.k0, pc.k0 + pc.n, c0, w)[0])
                    parts[k] = yk
                order = self.chain_order(c)
                acc = parts[order[0]]
                for k in order[1:]:
                    acc = add16(acc, parts[k])
                y[c0:c0 + w] = acc
        self.y_row = y
        print(f"[md] {d.repo}@{d.revision[:12]} layer {d.layer} MoE, one token: experts {self.route_ids} (slot order), "
              f"route weights {[round(float(x), 4) for x in d.w16[0]]}; experts of {self.IE}, shared {self.ISH}")
        for line in d.report(y[None, :])[1:]:              # the D-port shifts, the float layer's error
            print(f"[md] {line}")

    # ---- the weights as streamed -------------------------------------------------------------------------------------
    def _pack(self, q):
        """INT4 weights [K, N] as the GEMV reads them: the paired layout of the (16, 4, 16) B blocks, nibble-packed."""
        return self.lay.pack_int4(self.lay.to_b_pairs(np.asarray(q), GV_MESH))

    @staticmethod
    def _chunk(w_bytes, f):
        """A GEMV weight chunk as it is streamed: the weights, then (64-B aligned) its output columns' factors as one
        FP16 row. Returns (bytes, offset of the factor row)."""
        wb = np.asarray(w_bytes, dtype=np.int8).reshape(-1)
        off = -(-wb.size // 64) * 64
        row = np.ascontiguousarray(np.asarray(f, dtype=np.float16)).view(np.int8)
        return np.concatenate([wb, np.zeros(off - wb.size, dtype=np.int8), row]), off

    @staticmethod
    def _gu_cols(inter, a, n):
        """The gate|up columns of the n intermediate columns from `a` of an MLP of `inter`, as a GEMV chunk takes them:
        each 16-column block's gate columns, then its up columns."""
        return np.concatenate([np.r_[a + b + np.arange(BLK), inter + a + b + np.arange(BLK)] for b in range(0, n, BLK)])

    def _stage(self):
        """X as the token's a_row; the router's INT8 chunk in the B layout with its factor row; every gate|up and
        down chunk in the GEMV's paired INT4 layout with its factor row, in the order the clusters load them."""
        st, d, T, R = self.st, self.d, self.TAG, self.ROOT
        m = self.mem[R]
        loc_r = (m >> 4, m & 15)
        # X: the token's a_row in the HBM of the root's memory chip; every chip's landing of it
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", a_row_bytes(d.hq[0]), mem_chip=loc_r)
        st.put_zeros(f"{T}_xland", "int8_t", self.XT, preload=True)
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64)
        st.put_zeros(f"{T}_rland", "uint8_t", 64, preload=True)
        st.put_zeros(f"{T}_rflag", "uint8_t", 64)
        st.put_zeros(f"{T}_zero1k", "uint8_t", 1024)
        self.h_yhbm = st.put_hbm(f"{T}_yhbm", np.zeros(2 * self.H, dtype=np.uint8), mem_chip=loc_r)
        rw = self.lay.to_b(np.asarray(d.wr_q, dtype=np.int8), GV_MESH).astype(np.int8).reshape(-1)
        self.rw_bytes = rw.size
        free, blobs = {}, {}
        sh_mp = d.sh
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            sh = self.pieces[k][0]
            items = []                                    # (key, blob, factor row offset)
            if cl == 0:                                   # the router's INT8 weights, then its factors
                items.append((("rw",),) + self._chunk(rw, d.wr_f))
            e0, en = sh.cols(c)
            for j, (b0, n) in enumerate(self.chunks(en)):
                cols = self._gu_cols(self.ISH, sh.k0 + e0 + b0, n)
                items.append((("sg", j),) + self._chunk(self._pack(sh_mp["gu_q"][:, cols]), sh_mp["gu_f"][cols]))
            for jl, (o0, w) in enumerate(self.out_slices(c)):
                items.append((("sd", jl),) + self._chunk(self._pack(sh_mp["dn_q"][sh.k0:sh.k0 + sh.n, o0:o0 + w]),
                                                            sh_mp["dn_f"][o0:o0 + w]))
            free[cl] = len(items)
            for pi, pc in enumerate(self.pieces[k][1:], start=1):
                emp = d.expert(pc.e)
                e0, en = pc.cols(c)
                for j, (b0, n) in enumerate(self.chunks(en)):
                    cols = self._gu_cols(self.IE, pc.k0 + e0 + b0, n)
                    items.append((("gu", pi, j),) + self._chunk(self._pack(emp["gu_q"][:, cols]),
                                                                   emp["gu_f"][cols]))
            for jl, (o0, w) in enumerate(self.out_slices(c)):
                for pi, pc in enumerate(self.pieces[k][1:], start=1):
                    emp = d.expert(pc.e)
                    items.append((("dn", jl, pi),) + self._chunk(
                        self._pack(emp["dn_q"][pc.k0:pc.k0 + pc.n, o0:o0 + w]), emp["dn_f"][o0:o0 + w]))
            blobs[cl] = items
        self._images(blobs, free)

    def _images(self, blobs, free):
        """Each cluster's weight image in its memory chip's HBM: blobs {cl: [(key, chunk bytes, factors' offset)]} in
        the order the cluster loads them, free {cl: its first chunks no route decides}; the gap before each ring's
        first chunk that waits for the route; then the stream report."""
        st, T = self.st, self.TAG
        for cl, items in blobs.items():
            for key, b, _ in items:
                if b.size > self.chunk:
                    raise ValueError(f"cluster {cl}: weight chunk {key} of {b.size} B; params w_chunk_bytes is "
                                     f"{self.chunk}")
        self.img, self.chunk_off, self.cidx = {}, {}, {}
        # each chip's split at the route: the fewer route-free chunks of its two rings
        self.n_free = {k: min(free[cl] for cl in self.clusters(k)) for k in self.chips}
        for cl in range(self.N_CL):
            k = self.chip(cl)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts, offs, keys = [], [], {}
            off = 0
            for i, (key, b, soff) in enumerate(blobs[cl]):
                if i == self.n_free[k] and i:
                    parts.append(np.zeros(GAP, dtype=np.int8))
                    off += GAP
                keys[key] = len(offs)
                parts.append(np.asarray(b, dtype=np.int8))
                offs.append((off, b.size, soff))
                off += b.size
            self.img[cl] = st.put_hbm(f"{T}_w_cl{cl}", np.concatenate(parts), mem_chip=loc)
            self.chunk_off[cl], self.cidx[cl] = offs, keys
        self._stream_report()

    def _stream_report(self):
        """The bytes a token streams: per memory chip, per link (a memory chip's link carries the chips behind it), and
        per chip the route-free chunks, pushed before the route is known (each ring's runs before its boundary), which
        cover the route's latency."""
        R = self.ROOT
        chip_b = {k: sum(sum(nb for _, nb, _ in self.chunk_off[cl]) for cl in self.clusters(k)) for k in self.chips}
        free_b = {k: sum(sum(nb for _, nb, _ in self.chunk_off[cl][:self.n_free[k]]) for cl in self.clusters(k))
                  for k in self.chips}
        mem_b = {m: sum(b for k, b in chip_b.items() if self.mem[k] == m) for m in sorted(set(self.mem.values()))}
        mem_b[self.mem[R]] += self.XT                     # X's multicast
        link_b = {}
        for k, b in chip_b.items():
            link_b[self.link[k]] = link_b.get(self.link[k], 0) + b
        total = sum(mem_b.values())
        w_only = sum(3 * self.H * pc.n for k in self.chips for pc in self.pieces[k]) // 2
        print(f"[md] stream: {total:,} B a token ({w_only:,} B of INT4 weights, the rest their factor rows, the "
              f"router's INT8 chunk and X); per memory chip " + ", ".join(f"{m:02x} {b:,}" for m, b in mem_b.items()))
        print("[md] stream per link: " + ", ".join(f"{m:02x}{'+' if s > 0 else '-'} {b:,}"
                                                  for (m, s), b in sorted(link_b.items())))
        print("[md] pushed before the route (route-free chunks a ring, bytes a chip): "
              + ", ".join(f"{k:02x} {self.n_free[k]} {free_b[k]:,}" for k in self.chips))

    def xland(self, off=0, chip_id=None):
        """Every chip's landing of X (the token's a_row) in its L3."""
        return BingoMemSymbol(f"{self.TAG}_xland", offset=off, chip_id=chip_id)

    # ---- L1 ---------------------------------------------------------------------------------------------------------
    def _buffers(self):
        """Every activation a row. The exchange buffer and the root's fold buffer one row a slice; the fold rows
        are y's halves (yrow). Per cluster: X (a_row); per piece its G row (64 B a block of the cluster's columns), its
        SiLU, its A16 row and the quantiser's input, and its A8 row (a_row, both clusters' columns); the routed downs'
        row buffers, the route table, the row lists; on the root's cluster 0 the logits' row and the softmax's 8 rows.
        The quantisers' inputs, the A16 rows (padded to whole spread steps) and the softmax's rows share one buffer the
        xDMA zeroes before the window ("rowz"): the spread writes only the atoms, the SiLU'd product only the A16 row's
        own beats and the router's factors only row 0, so the rest stays zero."""
        B = {}
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.NSLC * self.SL)
                    for k in self.chips for c in range(self.NC)}
        self.xch_zero = False                             # the shared down GEMV writes each slot whole first
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", self.NSLC * self.SL) for c in range(self.NC)}
        self.yrow = self.p1c                              # slice after slice: the cluster's half of y in column order
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nb)
            pcs = self.pieces[k]
            ens = [pc.cols(c)[1] for pc in pcs]
            steps = [spread_steps(en) for en in ens]
            zb = sum(LANES * D_BLOCK * s + SPREAD_ATOMS * ATOM * s for s in steps) + (ROWS * 128 if cl == 0 else 0)
            z = l1("rowz", zb)
            off, sq, a16 = 0, [], []
            for s in steps:                               # the quantisers' inputs
                sq.append(z.view(off))
                off += LANES * D_BLOCK * s
            for s in steps:                               # the A16 rows
                a16.append(z.view(off))
                off += SPREAD_ATOMS * ATOM * s
            b = B[cl] = dict(
                x=l1("x", self.XT),
                rowz=(z, zb),
                gr=[l1(f"gr{i}", 4 * en) for i, en in enumerate(ens)],
                sgr=[l1(f"sgr{i}", 4 * en) for i, en in enumerate(ens)],
                a16=a16, sq=sq,
                a8=[l1(f"a8p{i}", 2 * pc.n) for i, pc in enumerate(pcs)],
                dtmp=[l1(f"dtmp{i}", self.SL) for i in range(DTMP_BUFS)],
                rt=l1("rt", 64),
                lists=l1("lists", SnaxBingoKernelMoeDispatchArgs.list_bytes([ROWS] * len(self.held[k]))))
            if cl == 0:                                   # the root's cluster 0: the router
                b["lg"] = l1("lg", 2 * md.N_EXP)
                b["pin"] = z.view(off)                    # the softmax's 8 rows of 64, 128 B apart
                b["pp"] = l1("pp", ROWS * 128)
            b["wz"] = l1("wz", 1024)
        self.B = B
