# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""MBuild's data and memory: the MoE's data, the golden, the HBM images of the weights, every L3 area and every
cluster's L1 buffers."""

import os

import numpy as np
from bingo_kernel_args import SnaxBingoKernelMoeDispatchArgs
from bingo_mem_handle import BingoMemSymbol
from driver import repo_root

import moe_prefill_data as md

from .common import (B_PARITY_OFF, BLK, D_BLOCK, DTMP_BUFS, MESH3, NOUT, ROUTE_B, SCALE_BUFS, SOFTMAX_ROWS,
                     row_pitch)

REPO = repo_root(__file__)


class Memory:

    def mem_handles(self):
        """The data, the placement, the golden and the memory; no task is made. The MoE's data (_data: the experts,
        the shared MLP, the router, X and the route) and the sizes it gives: the shared MLP's column blocks a chip
        (shk, an even split) and each chip's first (sho), the slices a cluster, the row pitches; the placement of the
        experts on the chips (_place, params slot_order) and their split over each chip's clusters (_split_all); the
        golden (_golden); then _stage(): X, the router's weights and every cluster's weight chunks and factors into
        HBM, every L3 landing area, table and flag; make_streams(): each cluster's L1 weight slabs; make_rings(): each
        chip's weight rings in its L3; _buffers(): every cluster's L1 buffers."""
        d = self.d = self._data(self.args, self.p, self.hw)
        self.lay = d.layout
        self.H, self.IE, self.ISH = d.H, d.I_e, d.I_sh
        self.EB, self.SHB = self.IE // BLK, self.ISH // BLK
        nch = len(self.chips)
        if self.SHB % nch:
            raise ValueError(f"{self.SHB} shared blocks over {nch} chips")
        self.shk = {k: self.SHB // nch for k in self.chips}
        self.sho, acc = {}, 0
        for k in self.chips:
            self.sho[k], acc = acc, acc + self.shk[k]
        if (self.H // self.NC) % self.SW or self.SW % BLK:
            raise ValueError(f"params slice={self.SW}: whole column blocks dividing a cluster's {self.H // self.NC} outputs")
        self.NSLC = self.H // self.NC // self.SW           # slices a cluster
        self.XT = 8 * self.H                               # a multicast tile of X, 8 tokens
        self.PX = row_pitch(self.H)
        self.PE = row_pitch(self.IE)
        self.PS = row_pitch(BLK * max(self.shk.values()))
        self._place()
        if self.slot_order is not None:
            for k in self.chips:
                es = sorted(self.held[k], key=lambda e: (md.pad8(d.cnt[e]), e))
                self.held[k] = [es[r] for r in self.slot_order]
        self._split_all()
        print("[mp] placement (chip: experts, padded rows): " + "; ".join(
            f"{k:02x}: {self.held[k]} {sum(md.pad8(d.cnt[e]) for e in self.held[k])}" for k in self.chips))
        print(f"[mp] slot rows (params slot_order {self.slot_order}): " + "; ".join(
            f"{k:02x}: {[md.pad8(d.cnt[e]) for e in self.held[k]]}" for k in self.chips))
        self._golden()
        self._stage()
        self.make_streams()
        # The shared gate|up keeps each of its weight chunks in an L1 slab for every row block of X
        # (loaded at the first block, released after the last), so a cluster needs a slab per shared
        # chunk: with fewer, a later chunk's load overwrites one still to be read.
        for cl in range(self.N_CL):
            n_sh = len(self.chunks(self.ch(self.chip(cl), self.local(cl))[0][1]))
            if n_sh > len(self.ls[cl].slabs):
                raise ValueError(f"cluster {cl} holds {n_sh} shared gate|up chunks in "
                                 f"{len(self.ls[cl].slabs)} slabs: params w_buffers >= {n_sh}")
        self.make_rings()
        self._buffers()

    def _data(self, args, p, hw):
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        return md.MoePrefillData(cache, hw=hw, seed=int(p.get("seed", 1)),
                                 n_calib=int(p.get("n_calib", 8)), n_cal_act=int(p.get("n_cal_act", 32)),
                                 tokens=self.T, inter_e=p.get("inter_e"), inter_sh=p.get("inter_sh"))

    # ---- the golden ---------------------------------------------------------------------------
    def _golden(self):
        d, add16 = self.d, self.d.fp.add16
        y = np.zeros((self.T, self.H), dtype=np.float16)
        for c in range(self.NC):
            for c0, w in self.out_slices(c):
                parts = {}
                for k in self.chips:
                    ks0 = BLK * self.sho[k]
                    init = (self.resid["h16"][:, c0:c0 + w] if self.resid is not None and k == self.resid["chip"]
                            else None)
                    parts[k] = d.chip_partial((ks0, ks0 + BLK * self.shk[k]), self.held[k], c0, w, init=init)
                order = self.fold_order_c(c)
                acc = parts[order[0]]
                for k in order[1:]:
                    acc = add16(acc, parts[k])
                y[:, c0:c0 + w] = acc
        self.y_raw = y
        for line in d.report(y, resid=None if self.resid is None else self.resid["h16"]):
            print(f"[mp] {line}")

    # ---- the weights as streamed ----------------------------------------------------------------
    def _pack_b(self, q):
        """INT4 weights [K, N] in the parity layout (gemm.h): the B layout with each pair of passes (64 B each, eight
        words) interleaved word by word, so the B reader and the row-major A reader keep to banks of opposite parity."""
        b = self.lay.pack_int4(self.lay.to_b(q, MESH3))
        K, N = q.shape
        if (K // MESH3[1]) % 2:
            raise ValueError(f"the parity layout: {K // MESH3[1]} passes of K is odd")
        return np.ascontiguousarray(np.asarray(b).reshape(N // MESH3[2], K // MESH3[1] // 2, 2, 8, 8)
                                    .transpose(0, 1, 3, 2, 4)).reshape(-1)

    def _scale_tile(self, f):
        return self.lay.to_d(np.tile(np.asarray(f, dtype=np.float16), (8, 1)), MESH3).astype(np.float16)

    def _chunk(self, w_bytes, f):
        """A weight chunk as it is streamed: B_PARITY_OFF zero bytes, the INT4 weights, then (64-B aligned) the scale
        tile of its output columns. Returns (bytes, offset of the scale tile)."""
        wb = np.asarray(w_bytes, dtype=np.int8).reshape(-1)
        wb = np.concatenate([np.zeros(B_PARITY_OFF, dtype=np.int8), wb])
        off = -(-wb.size // 64) * 64
        st = self._scale_tile(f).view(np.int8)
        parts = [wb, np.zeros(off - wb.size, dtype=np.int8), st]
        return np.concatenate(parts), off

    # ---- L3 and HBM -----------------------------------------------------------------------------
    def _stage(self):
        st, d, T, R = self.st, self.d, self.TAG, self.ROOT
        hq = np.asarray(d.hq, dtype=np.int8)
        # X's source, then every chip's X landing (the prompt's rows and the zero row the experts' padding rows read);
        # Y lies over the first T * H * 2 bytes
        assert self.T * self.H * 2 <= (2 * self.T + 1) * self.H
        st.put_zeros(f"{T}_xy", "int8_t", (2 * self.T + 1) * self.H, preload=True)
        m = self.mem[R]
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", hq.reshape(-1), mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64 * self.NXT)
        if self.x_cast == "unicast":                        # the root's progress flags, one a (chip, tile)
            st.put_zeros(f"{T}_xuprog", "uint8_t", 64 * self.NXT * len(self.chips))
        st.put_zeros(f"{T}_rland", "uint8_t", self.T * ROUTE_B, preload=True)
        st.put_zeros(f"{T}_rflag", "uint8_t", 64 * self.XB)
        st.put_zeros(f"{T}_rup", "uint8_t", 64 * (self.T // 8))   # the root's: one flag a tile
        st.put_zeros(f"{T}_zero1k", "uint8_t", 1024)
        rw = self.lay.to_b(np.asarray(d.wr_q, dtype=np.int8), MESH3).astype(np.int8)
        rws = self._scale_tile(d.wr_f).view(np.int8)
        # the router's weights and factors in the HBM of the root's memory chip; the multicast lands them in every
        # chip's L3 behind one flag
        blob = np.concatenate([rw.reshape(-1), rws.reshape(-1)])
        self.rw_bytes = blob.size
        self.h_rwhbm = st.put_hbm(f"{T}_rwhbm", blob, mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_rwland", "int8_t", blob.size, preload=True)
        st.put_zeros(f"{T}_rwflag", "uint8_t", 64)
        if self.x_cast == "unicast":
            st.put_zeros(f"{T}_rwuprog", "uint8_t", 64 * len(self.chips))
        self.h_rw, self.h_rws = BingoMemSymbol(f"{T}_rwland"), BingoMemSymbol(f"{T}_rwland", offset=rw.size)
        # each cluster's weights in the order it loads them: its shared gate|up chunks, its experts'
        # gate|up chunks slot by slot, then per output slice the shared down chunk and each slot's
        self.img, self.chunk_off = {}, {}
        sh_mp = d.sh
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            (sc0, sn), ex = self.ch(k, c)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts, offs = [], []

            def add(blob, soff):
                off0 = sum(b.size for b in parts)
                parts.append(blob)
                offs.append((off0, blob.size, soff))

            for b0, n in self.chunks(sn):
                cols = np.concatenate([np.arange(sc0 + b0, sc0 + b0 + n), self.ISH + np.arange(sc0 + b0, sc0 + b0 + n)])
                add(*self._chunk(self._pack_b(sh_mp["gu_q"][:, cols]), sh_mp["gu_f"][cols]))
            for e in self.held[k]:
                mp = d.expert(e)
                e0, en = ex[e]
                for b0, n in self.chunks(en):
                    cols = np.concatenate([np.arange(e0 + b0, e0 + b0 + n), self.IE + np.arange(e0 + b0, e0 + b0 + n)])
                    add(*self._chunk(self._pack_b(mp["gu_q"][:, cols]), mp["gu_f"][cols]))
            ks0 = BLK * self.sho[k]
            for o0, w in self.out_slices(c):
                add(*self._chunk(self._pack_b(sh_mp["dn_q"][ks0:ks0 + BLK * self.shk[k], o0:o0 + w]),
                                 sh_mp["dn_f"][o0:o0 + w]))
                for e in self.held[k]:
                    mp = d.expert(e)
                    add(*self._chunk(self._pack_b(mp["dn_q"][:, o0:o0 + w]), mp["dn_f"][o0:o0 + w]))
            for b in parts:
                if b.size > self.chunk:
                    raise ValueError(f"a weight chunk of {b.size} B; params w_chunk_bytes is {self.chunk}")
            self.img[cl] = st.put_hbm(f"{T}_w_cl{cl}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[cl] = offs

    def xland(self, off=0, chip_id=None):
        """Every chip's X landing in its L3 (behind X's source on the root's)."""
        return BingoMemSymbol(f"{self.TAG}_xy", offset=self.T * self.H + off, chip_id=chip_id)

    def ysym(self, off=0, chip_id=None):
        """The output Y in the root's L3, over X's source and landing."""
        return BingoMemSymbol(f"{self.TAG}_xy", offset=off, chip_id=chip_id)

    def y_off(self, c, jl):
        return (c * self.NSLC + jl) * self.SL

    # ---- L1 -----------------------------------------------------------------------------------
    def _buffers(self):
        d, T = self.d, self.T
        # the 8-token tiles each cluster routes, spread evenly over the clusters (each cluster's a run of consecutive
        # tiles in one row block of X, routed as one, at most SOFTMAX_ROWS rows)
        ntile = T // 8
        self.rt_tiles = {cl: [i for i in range(ntile) if i * self.N_CL // ntile == cl] for cl in range(self.N_CL)}
        if (max(len(v) for v in self.rt_tiles.values()) * 8 > SOFTMAX_ROWS or any(
                v and (v != list(range(v[0], v[0] + len(v))) or 8 * v[0] // self.XR != 8 * v[-1] // self.XR)
                for v in self.rt_tiles.values())):
            raise ValueError(f"the route: {ntile} tiles over {self.N_CL} clusters, each cluster's consecutive, in one "
                             f"row block of X, {SOFTMAX_ROWS // 8} at most")
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", NOUT * self.SL)
                    for k in self.chips for c in range(self.NC)}
        self.xch_zero = False                    # the shared down GEMM writes each slot whole first
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", 2 * self.SL) for c in range(self.NC)}
        B = {}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nb)    # noqa: E731
            (_, sn), ex = self.ch(k, c)
            rows = [md.pad8(d.cnt[e]) for e in self.held[k]]
            mrow = max(rows)
            # the gate|up passes' rows (params expert_rows: an expert's row blocks), pass p into xe[p % 2]
            prow = [n for r in rows for _, n in self.row_blocks(r)]
            brow = max(prow)
            gt = max(self.XR, brow) // 8 * 2 * self.BPC * D_BLOCK
            nmax = max(n for _, n in ex.values()) // BLK
            b = B[cl] = dict(
                # slot s gathers its rows into xe[s % 2], so each holds the most rows of its slots; X's row blocks land
                # in them too
                xe=[l1(f"xe{i}", max([r for s, r in enumerate(prow) if s % 2 == i] + [self.XR]) * self.PX)
                    for i in range(2)],
                gbuf=[l1(f"g{i}", gt) for i in range(2)],
                sg=l1("sg", gt),
                # one A16 for every expert: the SIMD core quantises an expert before its next SwiGLU
                a16e=l1("a16e", brow // 8 * nmax * D_BLOCK + D_BLOCK),
                a8sh=l1("a8sh", T * self.PS),
                a8e=[l1(f"a8e{s}", r * self.PE) for s, r in enumerate(rows)],
                dtmp=[l1(f"dtmp{i}", mrow * self.SW * 2) for i in range(DTMP_BUFS)],
                rt=l1("rt", T * ROUTE_B),
                lists=l1("lists", SnaxBingoKernelMoeDispatchArgs.list_bytes(rows)))
            b["xblk"] = b["xe"]
            b["dst"] = l1("dst", 64)                       # the row lists' counts between blocks
            if sn:
                b["a16sh"] = l1("a16sh", self.XR // 8 * (sn // BLK) * D_BLOCK + D_BLOCK)
                if self.sh_stage_pitch(k, c):
                    b["shq_st"] = l1("shq_st", T * self.sh_stage_pitch(k, c))
            if self.rt_tiles[cl]:                          # the cluster's tiles to route
                nr = 8 * len(self.rt_tiles[cl])
                b["rx"] = l1("rx", nr * self.PX)
                b["rw"] = l1("rw", self.H * 64)
                b["rws"] = l1("rws", 4 * D_BLOCK)
                b["lg"] = l1("lg", nr * 64 * 2)
                b["pin"] = l1("pin", nr * 128)
                b["pp"] = l1("pp", nr * 128)
            b["wz"] = l1("wz", 1024)
            if self.resid is not None and k == self.resid["chip"]:
                b["hres"] = l1("hres", self.resid["piece"])
            # an 8-row D-layout tile of a chunk's widest columns; named to sort after the weight slabs: the static
            # packer places buffers in name order, and these live for the whole run, so placed before the long-lived
            # row buffers they would split their space
            b["sc"] = [self.cx_chip(k).at(c).l1(f"zy_{self.TAG}_sc{i}", 16 * max(self.SW, 2 * BLK * self.BPC))
                       for i in range(SCALE_BUFS)]
        self.B = B
        self._hres_reader = {}
        self.sc_next = {cl: 0 for cl in range(self.N_CL)}
        self.sc_reader = {cl: [None] * SCALE_BUFS for cl in range(self.N_CL)}
