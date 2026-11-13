# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The task graph of the step (README.md, THE STEPS): build() runs the steps in their order; the preloads before the
window; the hooks a layer provides (x_multicast, stage_store)."""

from .consts import BR, C0_BYTES, C1_BYTES, KB, LAT, PK, PVB, Q_NOPE, V_HEAD, VB


class KsBuild:
    """FaDecodeKs.build: the steps of query.py, new_key.py, sweep.py and output.py in the order they run."""

    def build(self):
        self._y_stores, self._chains = [], []
        self.vbuf, self.wuv, self.wuv_ld, self.wo, self.wo_ld = {}, {}, {}, {}, {}
        pre = self._preload()
        go = self.setup(extra=pre)
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        xm = self.x_multicast(go)
        xa, xw = self._x_fetch(go, xm)
        kv_puts = self._kva_columns(xa, xw)
        q_puts = self._q_rows(xa, xw)
        new_ready = self._new_key(kv_puts)
        qin = self._query_tile(q_puts)
        s16, sm1, ar1 = self._pass1(qin, new_ready)
        marr, vld = self._m_star(sm1, ar1, qin)
        part, lcp = self._pass2(s16, sm1, marr, vld, new_ready)
        ow = self._w_uv(self._o_tilde(self._reduce(part, lcp)))
        yw = self._w_o(ow)
        ys = self.stage_store(yw)
        self.append = self.kv_append([ys] if ys is not None else [yw])
        self.link_report()
        self.end_window([t for t in (ys, self.append, yw) if t is not None])

    def _preload(self):
        """Before the window: every cluster 0's constants, the new key's, and the buffers whose unwritten places must
        be zero (the a_row spreads, the new tile, the zero rows of the new piece's tiles). One chain a cluster."""
        R = self.ROOT
        pre = {k: [] for k in self.chips}
        self.b0 = {}
        for k in self.chips:
            b = self.b0[k] = dict(c=self.l1(k, 0, "c0", C0_BYTES), nsp=self.l1(k, 0, "nsp", 4 * Q_NOPE),
                                  osp=self.l1(k, 0, "ohsp", 4 * V_HEAD))
            prev = [self.copy(k, 0, "c0_ld", self.h_c0[k].address, b["c"], C0_BYTES, [])]
            prev = [self.memset(k, 0, "z_nsp", b["nsp"], 4 * Q_NOPE, prev)]
            prev = [self.memset(k, 0, "z_ohsp", b["osp"], 4 * V_HEAD, prev)]
            if k == R:
                self.br = dict(osp=self.l1(R, 0, "osp", 2 * BR * LAT))
                prev = [self.memset(R, 0, "z_osp", self.br["osp"], 2 * BR * LAT, prev)]
            pre[k] += prev
        sizes = dict(c=C1_BYTES, kt=KB, vt=VB, ka16=2 * PK, vr16=2 * PVB)
        b = self.b1 = {nm: self.l1(R, 1, {"kt": "kt_new", "vt": "vt_new"}.get(nm, nm), nb) for nm, nb in sizes.items()}
        prev = [self.copy(R, 1, "c1_ld", self.h_c1.address, b["c"], C1_BYTES, [])]
        for nm in ("kt", "vt", "ka16", "vr16"):
            prev = [self.memset(R, 1, f"z_{nm}", b[nm], sizes[nm], prev)]
        pre[R] += prev
        return pre

    # ---- the hooks a layer provides ----------------------------------------------------------------------------------
    def x_multicast(self, go):
        """The task the x fetches are ordered after: the layer's input norm puts x into fk_xland on every chip behind
        fk_xtflag (the a_row, INT8 at the block's s_x, XB bytes)."""
        raise NotImplementedError("the key-split decode runs in a layer, which multicasts x (dsv2_decode_mc)")

    def stage_store(self, yw):
        """Where y goes after yw (the root's wait for every cluster's columns of its L3 row fk_yrow): the layer copies
        the row into its own buffer. Returns the task."""
        raise NotImplementedError("the key-split decode runs in a layer, which takes y (dsv2_decode_mc)")
