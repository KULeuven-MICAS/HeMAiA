# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The decode's data and memory: X, Y and the weights in HBM, every L1 buffer a row."""

import os

import numpy as np

import ffn_data
import gemv_utils  # util/sim/gemm: ffn_data puts it on sys.path
from bingo_mem_handle import BingoMemSymbol
from ffn_prefill_mc import REPO

from .params import ATOM, LANES


class DMemory:
    """DBuild.mem_handles: PBuild's plan of the one-token tile, its own staging and its row buffers."""

    def mem_handles(self):
        """PBuild's data, golden and memory of the one-token tile (this class's _data, _stage and _buffers: X and
        Y_raw's row in the root memory chip's HBM, the row buffers in L1); then X's multicast as what the HBM holds, one
        tile behind one flag."""
        super().mem_handles()
        # x_hbm_bcasts multicasts what the HBM holds as one tile behind one flag
        self.XT, self.TPP = self.XR, 1
        print(f"[{self.TAG}] decode: one token as rows, gate|up and down on the GEMV; X from HBM as the row in "
              f"a_row ({self.XR} B), {'Y16' if self.y_dequant else 'Y_raw'} ({2 * self.D} B) into HBM; "
              f"{self.NDN // self.NC} slices a cluster, "
              f"{self.NDN // self.NC // self.chain_group} gathers a cluster")

    def _data(self, args, p):
        """The FFN's data and golden of the one token."""
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/llm_blocks/ffn/cache"))
        return ffn_data.FFNData(cache, p.get("model", "dsv2-lite-l0"), seed=int(p.get("seed", 1)),
                                wbits=self.wbits, tokens=1, inter=p.get("inter"))

    def x_a(self):
        """X as the HBM holds it: the token's INT8 row in a_row (README.md, THE X DELIVERY)."""
        return gemv_utils.a_row_bytes(self.d.hq[0])

    def _stage(self):
        """X in the root memory chip's HBM, its landing area in every chip's L3 and its flag; Y_raw's row (and with
        y_dequant s_dn) in the same HBM; every cluster's weight chunks (as PBuild._stage packs them) and the gate|up
        chunks' column factors, one FP16 row a chunk ([gate | up]), in its memory chip's HBM."""
        st, d, T = self.st, self.d, self.TAG
        m = self.mem[self.ROOT]
        x = self.x_a().reshape(-1)
        self.XR = x.size                        # bytes of X in the HBM and in every chip's L3
        st.put_zeros(f"{T}_xland", "int8_t", self.XR, preload=True)
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", x, mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64)
        self.h_yhbm = st.put_hbm(f"{T}_yhbm", np.zeros(2 * self.D, dtype=np.uint8), mem_chip=(m >> 4, m & 15))
        if self.y_dequant:                      # s_dn, FP16 a column
            self.h_sdn = st.put_hbm(f"{T}_sdn", d.dn_f.astype(np.float16).view(np.int8), mem_chip=(m >> 4, m & 15))
        self.img, self.chunk_off, self.sdup = {}, {}, {}
        for cl, (c0, n) in self.cols.items():
            k = self.chip(cl)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts, offs, sc = [], [], []
            for b0, nb_ in self.chunks[cl]:
                cols = self._gu_cols(cl, b0, nb_)
                parts.append(self._pack_b(d.gu_q[:, cols]))
                sc.append(d.gu_f[cols].astype(np.float16))
            kc0, kn = self.kcols[k]             # the chip's columns, this cluster's share of the outputs
            for o0, w in self.dn_sl[self.local(cl)]:
                parts.append(self._pack_b(d.dn_q[kc0:kc0 + kn, o0:o0 + w]))
            off = 0
            for b in parts:
                offs.append((off, b.size))
                off += b.size
            self.img[cl] = st.put_hbm(f"{T}_w_cl{cl}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[cl] = offs
            s = np.concatenate(sc).astype(np.float16)
            self.sdup[cl] = (st.put_hbm(f"{T}_sgu_cl{cl}", s.view(np.int8), mem_chip=loc), s.nbytes)

    def _buffers(self):
        """Every buffer a row. The exchange buffer's output slots and the root's fold slots a row of a slice each
        (SLICE); per cluster X (a_row), its gate|up rows (FP16, chunk j's at its first gate block x 64 B, so no chunk's
        row overlaps another's), their factors, its A16 row and the row's a_row form in one padded buffer (the spread's
        whole beats), and the chip's A8 row (a_row); Y_raw's row (with y_dequant also s_dn) on the root's cluster 0."""
        self.SLICE = 2 * max(w for v in self.dn_sl.values() for _, w in v)
        self.xch_slots = self.out_slots
        self.xch_bytes = self.xch_slots * self.SLICE
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.xch_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", 2 * self.chain_group * self.SLICE)
                    for c in range(self.NC)}
        self.xbuf, self.sbuf, self.grow, self.rowp, self.a16r, self.ar16, self.a8 = {}, {}, {}, {}, {}, {}, {}
        for cl, (_, n) in self.cols.items():
            l1 = lambda name, nb: self.cx(cl).l1(f"{self.TAG}_{name}", nb)     # noqa: E731
            self.xbuf[cl] = [l1("xr", self.XR)]
            self.sbuf[cl] = l1("sgu", self.sdup[cl][1])
            self.grow[cl] = l1("gr", 2 * 2 * n)
            a16 = -(-2 * n // (LANES * ATOM)) * LANES * ATOM    # the A16 row, whole xDMA beats
            self.rowp[cl] = l1("rowp", 3 * a16)                  # [A16 row | its a_row form, twice as long]
            self.a16r[cl], self.ar16[cl] = self.rowp[cl].view(0), self.rowp[cl].view(a16)
            a8 = l1("a8r", 2 * self.kcols[self.chip(cl)][1])
            self.a8[cl] = [a8, a8]
        self.yrow = self.cx_chip(self.ROOT).at(0).l1(f"{self.TAG}_yrow", 2 * self.D)
        if self.y_dequant:
            self.sdn = self.cx_chip(self.ROOT).at(0).l1(f"{self.TAG}_sdn", 2 * self.D)

    def _xland(self, off, chip_id=None):
        """Where X lands in every chip's L3, at byte `off`."""
        return BingoMemSymbol(f"{self.TAG}_xland", offset=off, chip_id=chip_id)
