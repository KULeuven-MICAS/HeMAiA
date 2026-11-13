# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The block's data, plan and memory: the column split, the gate|up chunks and down slices, the golden, the HBM images
and every L1 buffer."""

import os

import numpy as np

import ffn_data

from .consts import BLK, REPO, TILE


class PMemory:
    """PBuild.mem_handles and what it plans with."""

    def mem_handles(self):
        """The data, the golden and the memory; no task is made. The FFN's data (_data) and the mapping its sizes give:
        the intermediate columns each cluster computes (split_columns) and their gate|up chunks, each chip's columns (K)
        for the down projection, each cluster's output slices (_dn_out_split); the golden Y, added in the fold order the
        ChainGathers add in; the sizes of X and Y_raw a pass and the output slots; then _stage(): every pass's X and every
        cluster's weight chunks and gate|up factors into HBM, X's landing area and Y_raw's place in L3; make_streams():
        each cluster's L1 weight slabs; make_rings(): each chip's weight rings in its L3; _buffers(): every cluster's L1
        buffers."""
        d = self.d = self._data(self.args, self.p)
        R = self.ROWS
        self.lay = d.layout
        self.D, self.I = d.H, d.I
        if self.D % self.DNC:
            raise ValueError(f"{self.D} outputs are not whole {self.DNC}-column down chunks")
        self.NDN = self.D // self.DNC
        rc = self.root_cols
        self.cols = self.split_columns(self.I, last=self.clusters(self.ROOT), unit=BLK,
                                       fixed=dict(zip(self.clusters(self.ROOT), rc)) if any(rc) else None)
        self.chunks = {cl: self._gu_chunks(n) for cl, (_, n) in self.cols.items()}
        # each cluster takes the chip's columns (K) for its half of the outputs, so it holds the chip's sum there and
        # nothing is folded inside a chip (the two clusters exchange their A8 once, before the first gather)
        self.kcols = {}                             # {chip: (first column, columns)}
        if self.NC != 2 or self.NDN % 2:
            raise ValueError("the down projection split by output columns needs 2 clusters and an even number of slices")
        for k in self.chips:
            (a0, an), (b0, bn) = (self.cols[cl] for cl in self.clusters(k))
            if b0 != a0 + an:
                raise ValueError(f"chip {k:02x}'s columns are not contiguous")
            self.kcols[k] = (a0, an + bn)
        self.gold = d.golden([self.kcols[k] for k in self.chips],
                             {k: [i] for i, k in enumerate(self.chips)}, self.fold_order())
        self.dn_sl = self._dn_out_split(self.dn_balance)
        if self.chain_alt:
            loc, add16 = self.gold["loc"], d.fp.add16
            rev = self.order[1:][::-1] + self.order[:1]
            y = loc[rev[0]]
            for k in rev[1:]:
                y = add16(y, loc[k])
            yr = self.gold["y_raw"].copy()
            for c0, w in self.dn_sl[1]:
                yr[:, c0:c0 + w] = y[:, c0:c0 + w]
            self.gold["y_raw"], self.gold["y16"] = yr, d.fp.mul16(yr, d.dn_f[None, :])
        for line in d.report(self.gold["y16"]):
            print(f"[pf] {line}")
        print(f"[pf] {self.NI * R} tokens in {self.NI} passes of {R}; fold order "
              f"{[f'{k:02x}' for k in self.order]}; columns per cluster "
              f"{sorted({n for _, n in self.cols.values()})}; gate|up chunks per cluster "
              f"{sorted({len(c) for c in self.chunks.values()})}")
        self.XB = R * self.D                    # X of a pass, INT8
        self.YB = R * self.D * 2                # Y_raw of a pass, FP16
        self.XT = self.XB // self.x_blocks      # the multicast's tile, each behind a flag: a first-pass block of X
        self.TPP = self.XB // self.XT           # tiles a pass
        self.YT_B = 16 * self.D * 2             # y_scatter: a 16-row tile of Y_raw, D layout, FP16
        h = self.NDN // self.NC
        if h % self.chain_group or (self.out_slots % self.chain_group and not self.per_kernel):
            raise ValueError(f"params chain_group={self.chain_group}: must divide the {h} slices a cluster and "
                             f"out_slots={self.out_slots}")
        if self.per_kernel:
            # kernel by kernel, no task of one kernel runs inside another: the down slices all wait in their output slots
            # until the reduce kernel, slot j for slice j, the exchange buffer's two slots then the X buffer's (dead
            # until the next pass's X), so the X buffer sits right behind the exchange buffer on every cluster
            self.out_slots = h                  # slot j for slice j of the pass
        self._stage()
        self.make_streams()
        self.make_rings()
        self._buffers()

    def _data(self, args, p):
        """The FFN's data and golden (a layer's block gives it the layer's rows)."""
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/llm_blocks/ffn/cache"))
        return ffn_data.FFNData(cache, p.get("model", "dsv2-lite-l0"), seed=int(p.get("seed", 1)),
                                wbits=self.wbits, tokens=self.T, inter=p.get("inter"))

    # ---- the plan -------------------------------------------------------------------------
    def _gu_chunks(self, n):
        """[(first gate block, blocks)] of a cluster's n columns: params first_blocks, then chunks of blocks."""
        nb = n // BLK
        f = self.first_blocks
        if not f:
            return [(b0, min(self.BPC, nb - b0)) for b0 in range(0, nb, self.BPC)]
        if f > self.BPC or f > nb:
            raise ValueError(f"params first_blocks={f}: at most blocks={self.BPC} and the {nb} blocks of a cluster")
        return [(0, f)] + [(b0, min(self.BPC, nb - b0)) for b0 in range(f, nb, self.BPC)]

    def _gu_cols(self, cl, b0, nb_):
        """The gate|up chunk's columns of [gate | up]: its gate blocks, then the same up blocks."""
        c0, _ = self.cols[cl]
        gate = np.arange(c0 + BLK * b0, c0 + BLK * (b0 + nb_))
        return np.concatenate([gate, self.I + gate])

    def dn_slices(self, c):
        """The down slices (DNC output columns each) cluster c computes: its half of the outputs."""
        h = self.NDN // self.NC
        return range(c * h, (c + 1) * h)

    def _dn_out_split(self, balance):
        """{cluster: [(first output column, columns)]}, NDN / 2 slices each. Default: the two halves of the outputs in
        DNC-column slices. params dn_balance: cluster 0's share of the outputs moves (a multiple of 16, the same on every
        chip, since a gather takes the same slice from every chip) to where the busiest cluster's array steps (gate|up
        of its own columns + down of its chip's) are fewest, and each share is cut into slices of whole 16-column blocks
        as even as possible."""
        h = self.NDN // self.NC
        if not balance:
            return {c: [(j * self.DNC, self.DNC) for j in self.dn_slices(c)] for c in range(self.NC)}
        mt, kd = self.MT, self.D // 4

        def worst(S):
            w = 0
            for k in self.chips:
                kn = self.kcols[k][1]
                for c, cl in enumerate(self.clusters(k)):
                    out = S if c == 0 else self.D - S
                    w = max(w, mt * kd * 2 * self.cols[cl][1] // BLK + mt * (kn // 4) * out // BLK)
            return w
        u = self.dn_unit                            # whole pairs of column tiles
        S = min(range(u, self.D, u), key=lambda v: (worst(v), abs(v - self.D // 2)))

        def cut(o0, width):
            nb = width // u
            sizes = [nb // h + (1 if i < nb % h else 0) for i in range(h)]
            out, c0 = [], o0
            for b in sizes:
                out.append((c0, b * u))
                c0 += b * u
            return out
        print(f"[pf] dn_balance: cluster 0 takes outputs 0..{S - 1}, cluster 1 {S}..{self.D - 1}; busiest cluster "
              f"{worst(S):,} array steps (even halves: {worst(self.D // 2):,})")
        return {0: cut(0, S), 1: cut(S, self.D - S)}

    def y_slices(self):
        """[(first column, columns)] of Y in the order its D-layout slices sit in L3."""
        return [sl for c in range(self.NC) for sl in self.dn_sl[c]]

    # ---- data -----------------------------------------------------------------------------
    def _pack_b(self, q):
        """An INT4 weight chunk [K, N] as the GEMMs read it: the paired layout."""
        lay, mesh = self.lay, self.d.mesh
        if q.shape[1] % (2 * mesh[2]):
            raise ValueError(f"a weight chunk of {q.shape[1]} columns is not whole pairs of {mesh[2]}-column tiles")
        return lay.pack_int4(lay.to_b_pairs(q, mesh))

    def x_a(self):
        """Every pass's X, A layout, pass after pass (INT8)."""
        R = self.ROWS
        return np.concatenate([self.lay.to_a(self.d.hq[i * R:(i + 1) * R]) for i in range(self.NI)]).astype(np.int8)

    def _stage(self):
        st, d, T, lay = self.st, self.d, self.TAG, self.lay
        # every chip's X landing area; params l3_overlay: one region a pass as long as Y_raw of the pass, X landing in its
        # second half
        if self.l3_overlay:
            st.put_zeros(f"{T}_xy", "int8_t", 2 * self.NI * self.XB, preload=True)
        else:
            st.put_zeros(f"{T}_xland", "int8_t", self.NI * self.XB, preload=True)
        # every pass's X in the HBM of the root's memory chip, a flag a tile
        m = self.mem[self.ROOT]
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", self.x_a().reshape(-1), mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64 * self.NI * self.TPP)
        if self.x_cast == "unicast":            # the root's progress flags, one a (chip, tile) of every pass
            st.put_zeros(f"{T}_xuprog", "uint8_t", 64 * self.NI * len(self.chips) * self.TPP)
        if not self.l3_overlay:
            st.put_zeros(f"{T}_y", "uint8_t", self.NI * self.YB, preload=True)
        if self.y_scatter:
            st.put_zeros(f"{T}_yt", "uint8_t", self.NTL * self.YT_B, preload=True)    # the chip's tiles of Y_raw
            st.put_zeros(f"{T}_ytflag", "uint8_t", self.NTL * self.NDN * 64)          # a slice of one landed
        self.img, self.chunk_off, self.sdup = {}, {}, {}
        for cl, (c0, n) in self.cols.items():
            k = self.chip(cl)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts, offs, sc = [], [], []
            for b0, nb_ in self.chunks[cl]:
                cols = self._gu_cols(cl, b0, nb_)
                parts.append(self._pack_b(d.gu_q[:, cols]))
                s = np.tile(d.gu_f[cols].astype(np.float16), (16, 1))
                tile = lay.to_d(s, d.mesh)
                # params deq_per_row: the scales of one row tile, applied to every row tile in turn
                sc.append(np.concatenate([tile] * (1 if self.deq_per_row else self.MT)))
            kc0, kn = self.kcols[k]             # the chip's columns, this cluster's share of the outputs
            for o0, w in self.dn_sl[self.local(cl)]:
                parts.append(self._pack_b(d.dn_q[kc0:kc0 + kn, o0:o0 + w]))
            off = 0
            for b in parts:
                offs.append((off, b.size))
                off += b.size
            self.img[cl] = st.put_hbm(f"{T}_w_cl{cl}", np.concatenate(parts).astype(np.int8),
                                     mem_chip=loc)
            self.chunk_off[cl] = offs
            s = np.concatenate(sc).astype(np.float16)
            self.sdup[cl] = (st.put_hbm(f"{T}_sgu_cl{cl}", s.view(np.int8), mem_chip=loc), s.nbytes)

    def _buffers(self):
        R, MT = self.ROWS, self.MT
        # the down output is reduced slice by slice (a slice's columns of all rows), through out_slots slots of each
        # cluster's exchange buffer and two fold slots (of chain_group slices) of the root clusters' fold buffers
        self.SLICE = R * max(w for v in self.dn_sl.values() for _, w in v) * 2
        # sync barrier: two output slots in the exchange buffer, the rest in the X buffer behind it (named to sort right
        # after it, so the static packer puts it there on every cluster: check_xch_offsets)
        self.xch_slots = 2 if self.per_kernel else self.out_slots
        if self.per_kernel and (self.out_slots - self.xch_slots) * self.SLICE > self.XB:
            raise ValueError(f"sync barrier: {self.out_slots - self.xch_slots} output slots of {self.SLICE} B do "
                             f"not fit the {self.XB}-byte X buffer")
        self.xch_bytes = self.xch_slots * self.SLICE
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.xch_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", 2 * self.chain_group * self.SLICE)
                    for c in range(self.NC)}
        self.xbuf, self.gbuf, self.sbuf, self.a16d, self.a16a, self.a8 = {}, {}, {}, {}, {}, {}
        for cl, (_, n) in self.cols.items():
            l1 = lambda name, nb: self.cx(cl).l1(f"{self.TAG}_{name}", nb)     # noqa: E731
            self.xbuf[cl] = [l1("AAAB_xo" if self.per_kernel else "x0", self.XB)]
            gb = MT * 2 * self.BPC * TILE
            self.gbuf[cl] = [l1(f"g{b}", gb) for b in range(2)]
            self.sbuf[cl] = l1("sgu", self.sdup[cl][1])
            self.a16d[cl] = l1("a16d", R * n * 2)
            self.a16a[cl] = l1("a16a", R * n * 2)
            a8 = l1("a8_0", R * self.kcols[self.chip(cl)][1])   # the chip's A8: both clusters' columns, one buffer
            self.a8[cl] = [a8, a8]
