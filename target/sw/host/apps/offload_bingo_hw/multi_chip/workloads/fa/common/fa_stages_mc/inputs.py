# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaStages' inputs: the keys' flags beside FaBuild's image, and the copies before the window that put every cluster's
constants in place (and, kx_per_chunk, each chip's W_KVA slice and key constants in its L3)."""

from bingo_mem_handle import BingoMemSymbol
from fa_prefill_mc import HIDDEN


class FaStagesInputs:

    def _stage(self):
        super()._stage()
        self.kv_helpers()
        # the k_pe parts' flags on the root, one a tile
        self.st.put_zeros(f"{self.TAG}_kvpflag", "uint8_t", (self.T // 16) * 64)
        if self.kvx:
            # the helpers' slices' flags: per owned tile, one a two-pair chip
            self.st.put_zeros(f"{self.TAG}_kvxflag", "uint8_t", self.KVR_TILES * len(self.kv_two) * 64)

    # ---- before the window ---------------------------------------------------------------------------------------
    def const_preload(self, pre):
        """Every cluster's dequantisation constants into its L1 (const_ld), one chain a chip ending on cluster 0 behind
        `pre` {chip: [tasks]} (the start stamp sits on the root's cluster 0 DM core behind that core's previous task,
        so a chain ending there adds no second edge into the stamp)."""
        for k in self.chips:
            for c in reversed(range(self.NC)):
                cl = self.chips.index(k) * self.NC + c
                pre[k] = [self.dm_copy(cl, "const_ld", self.h_c[k].address, self.consts[cl], self.C_BYTES, pre[k])]
        return pre

    def stage1_preload(self):
        """Before the window, per chip one chain ending on cluster 0: every cluster's key constants (W_KVA's
        dequantisation scales and RoPE tables: kvc on the tiles' cluster, kvc_pe on the other, for k_pe)."""
        T = self.TAG
        pre = {k: [] for k in self.chips}
        self.kvc_pe = {}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                cl = self.chips.index(k) * self.NC + c
                if (k, c) in self.kvc:
                    pre[k] = [self.dm_copy(cl, "kvc_ld", self.h_kc[k].address, self.kvc[(k, c)], self.KC_BYTES,
                                           pre[k])]
            for c in range(self.NC):
                if (k, c) not in self.kvc:
                    cl = self.chips.index(k) * self.NC + c
                    self.kvc_pe[(k, c)] = self.cx_chip(k).at(c).l1(f"{T}_kvcpe", self.KC_BYTES)
                    pre[k] = [self.dm_copy(cl, "kvc_ld", self.h_kc[k].address, self.kvc_pe[(k, c)],
                                           self.KC_BYTES, pre[k])]
        return pre

    def _kx_l3_preload(self, pre):
        """kx_per_chunk: before the window, each chip copies its W_KVA slice, its key constants and (a helper) the
        helped chip's second pair from HBM into its own L3, so the per-chunk copies in the window read the chip's L3,
        never HBM: reads across the D2D links beside the memory chips' ring pushes stall the fabric."""
        T = self.TAG
        pair = HIDDEN * 16 * 2 // 2
        st = self.st
        st.put_zeros(f"{T}_wkvc3", "uint8_t", max(self._src(k, 0)[1] for k in self.chips))
        st.put_zeros(f"{T}_kc3", "uint8_t", self.KC_BYTES)
        if self.kvx:
            st.put_zeros(f"{T}_wkvx3", "uint8_t", pair)
        for k in self.chips:
            cl = self.chips.index(k) * self.NC
            pre[k] = [self.dm_copy(cl, "wkvc3_ld", self.h_wkvc[k].address, BingoMemSymbol(f"{T}_wkvc3"),
                                   self._src(k, 0)[1], pre[k])]
            pre[k] = [self.dm_copy(cl, "kc3_ld", self.h_kc[k].address, BingoMemSymbol(f"{T}_kc3"), self.KC_BYTES,
                                   pre[k])]
            if k in self.kvx:
                pre[k] = [self.dm_copy(cl, "wkvx3_ld", self.h_wkvc[self.kvx[k][0]].address + pair,
                                       BingoMemSymbol(f"{T}_wkvx3"), pair, pre[k])]
