# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Stages 2 and 3: each owner finishes its tiles' keys on its two clusters (c on the tile's cluster 0, k_pe and the
value columns on cluster 1) behind stage 1's waits, puts the pieces into the root's L3, and the root multicasts a
chunk's pieces at a time into every other chip's L3."""

import numpy as np

from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
                               SnaxBingoKernelSimdRmsnormArgs, SnaxBingoKernelWaitFlagsArgs,
                               SnaxBingoKernelXdmaTranspose2dArgs, simd_fp16_lanes)
from bingo_mem_handle import BingoMemSymbol
from fa_prefill_mc import KV_RANK, KVP, ROPE, TILE16
from libs.comm.nest import convert_args, d_to_a_args


class FaStagesOwner:

    def stage_owner(self, go, last, end_wait=True):
        """Stage 2: stage_kv_split on every chip, each tile behind stage 1's owner waits (own_wait), then the root's
        waits for every piece, a chunk at a time, on the cluster that multicasts them; stage 3: the root multicasts each
        chunk once its waits are done; every other chip then waits for both chunks (end_wait)."""
        R, T, dm, P = self.ROOT, self.TAG, self.roles["dm"], self.C // 16
        self.kvp_put = {}
        for k in self.chips:
            self.stage_kv_split(k)
            self._key_runs.append((self._next_id(), None))
            last[k] += [self.kv_put[q] for q in self.kv_tiles[k]] + [self.kvp_put[q] for q in self.kv_tiles[k]
                                                                    if q in self.kvp_put]
        c, prev = self.x_put_cluster, []
        self.kv_mc = []
        waits = [("kv_nwait", "kvnflag", self.kv_put), ("kv_pwait", "kvpflag", self.kvp_put)]
        for h in range(self.NCH):
            self._key_runs.append((self._next_id(), h))
            for name, flag, src in waits:
                puts = [src[q] for q in range(h * P, (h + 1) * P)]
                w = self.node(R, c, dm, f"{name}_h{h}", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_{flag}", offset=h * P * 64), P, 1),
                              [t for t in puts if self.flag_dep(t, R, c)] + prev)
                for t in puts:
                    if not self.flag_dep(t, R, c):
                        self.dfg.bingo_add_order_edge(t, w)
                prev = [w]
                last[R].append(w)
            self.root_ready = getattr(self, "root_ready", {})
            self.root_ready[h] = prev[0]
            if self.keys_by_chunk:
                # one multicast source at a time: chunk 0's keys after chunk 1's x has landed, the keys of the chunks
                # after chunk 1 after the rest of x (x_wait: the flag proves the multicast's data)
                t = 2 * P - 1 if h == 0 else self.NX - 1 if h >= (1 if self.x_rest_early else 2) else None
                if t is not None and t < self.NX:
                    prev = prev + [self.x_wait(R, c, t, go, name=f"x_wait_h{h}")]
            # stage 3: the chunk's pieces from the root's L3 into every other chip's, one multicast at a time
            src = BingoMemSymbol(f"{T}_kv", offset=h * P * KVP)
            self.kv_mc.append(self.broadcast_put(R, c, f"kv_bput_h{h}",
                                                 SnaxBingoKernelIdmaPutFlaggedArgs(
                                                     src, src, P * KVP, BingoMemSymbol(f"{T}_kvflag", offset=h * 64),
                                                     value=1, bcast=True), prev))
            prev = self.kv_mc[-1:]
            last[R].append(self.kv_mc[-1])
        self._key_runs.append((self._next_id(), None))
        if end_wait:
            for k in self.chips:
                if k != R:
                    last[k].append(self.kv_all(k))

    def kv_all(self, k):
        """Chip k's wait (cluster 0's DM core) for both chunks' keys from the stage-3 multicasts."""
        w = self.node(k, 0, self.roles["dm"], "kv_all", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_kvflag"), self.NCH, 1), [])
        for t in self.kv_mc:
            self.dfg.bingo_add_order_edge(t, w)
        return w

    def kv_wait(self, cl, i):
        # the root's pieces carry two flags; its sweep follows the root's own waits for chunks 0 .. i
        if self.chip(cl) == self.ROOT:
            return self.root_ready[i]
        return super().kv_wait(cl, i)

    def stage_kv_split(self, k):
        """Chip k's tiles, each as two parts (kv_tile). keys_by_chunk: the tiles after the window read their own copies
        of the clusters' constants, made after the window (the first DM task of the first such tile), so the window's
        copies are free once chunk 1's keys are made; kx_per_chunk: every chunk after the first reads copies of its own
        too (made with its key tasks, behind the chunk before's q projections, from the chip's L3 copy)."""
        d, mesh = self.d, self.d.mesh
        c8 = np.arange(16 * KV_RANK).reshape(16, KV_RANK)
        assert np.array_equal(d.layout.to_b(c8, mesh), d.layout.to_a(np.ascontiguousarray(c8.T), mesh))
        self._kvc_late = getattr(self, "_kvc_late", {})
        self._kvc_late_ld = getattr(self, "_kvc_late_ld", {})      # (k, cluster, chunk) -> the copy that fills it
        late_from = self.NCH * (self.C // 16) if self.keys_by_chunk else self.NX
        for lt, q in enumerate(self.kv_tiles[k]):
            self.kv_tile(k, lt, q, late_from)

    def kv_late_consts(self, k, q, late_from):
        """The constants tile q's clusters read when it is a late tile (after the window, or kx_per_chunk a chunk after
        the first): their copies, made once a chip, cluster and chunk. Returns (late, hk)."""
        T = self.TAG
        hk = q // (self.C // 16) if self.kx_per_chunk and q // (self.C // 16) >= 1 else None
        late = q >= late_from or hk is not None
        if late:
            for cc, src in ((0, self.kvc), (1, self.kvc_pe)):
                if (k, cc, hk) not in self._kvc_late and (k, cc) in src:
                    cl = self.chips.index(k) * self.NC + cc
                    buf = self.cx_chip(k).at(cc).l1(f"{T}_lkvc{'pe' if src is self.kvc_pe else ''}"
                                                    + ("" if hk is None else f"_h{hk}"), self.KC_BYTES)
                    self._kvc_late_ld[(k, cc, hk)] = self.dm_copy(
                        cl, "kvc_lld" + ("" if hk is None else f"_q{q}"),
                        self.h_kc[k].address if hk is None else BingoMemSymbol(f"{T}_kc3"),
                        buf, self.KC_BYTES, [])
                    self._kvc_late[(k, cc, hk)] = buf
        return late, hk

    def kv_tile(self, k, lt, q, late_from):
        """Chip k's tile q (its lt-th) as two parts. Cluster 0 takes c: its 512 columns (the first 16 KB of the tile's
        D layout) into L1, their dequantisation, and RMSNorm row_major -> A (INT8) writes the key rows' c. Cluster 1
        builds c transposed itself (its own load and dequantisation of c, D -> row major and the transpose on its own
        xDMA) and RMSNorm col_major -> B (INT8) writes the value columns from it (B of c [16, 512] is A of c^T); k_pe
        (the last 2 KB) is dequantised and rotated (RoPE) into the key rows' k_pe there too, first on its in-order SIMD
        core. Cluster 0 puts the key rows' c behind the tile's flag (kvnflag); cluster 1 puts k_pe and the value
        columns (adjacent in the slot) behind a flag of its own (kvpflag)."""
        d, T, mesh = self.d, self.TAG, self.d.mesh
        simd, xdma, dm = self.roles["simd"], self.roles["xdma"], self.roles["dm"]
        R, KR, RP = self.ROOT, KV_RANK, ROPE
        on_root = lambda name, off: BingoMemSymbol(name, offset=off, chip_id=None if k == R else R)    # noqa: E731
        cb, n_pe, half = (KR // 16) * TILE16, 16 * RP, (RP // 32) * TILE16
        sc = self.fbits(d.s_c.inv)
        beat = simd_fp16_lanes() * 2
        self._key_runs.append((self._next_id(), q // (self.C // 16)))
        s = "" if lt == 0 else f"_t{lt}"
        c0, c1 = 0, 1
        late, hk = self.kv_late_consts(k, q, late_from)
        kvcp = (lambda cc: self._kvc_late[(k, cc, hk)]) if late else (lambda cc: self.kvc_pe[(k, cc)])
        # the tile's loads (and the RoPE table copy) behind the copy of the constants their readers read
        kdep = (lambda cc: [self._kvc_late_ld[(k, cc, hk)]] if late and (k, cc, hk) in self._kvc_late_ld else [])
        cl0, cl1 = self.chips.index(k) * self.NC + c0, self.chips.index(k) * self.NC + c1
        src = BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD)
        # a chip's later tiles (more tiles than chips) take buffers of their own names: one handle a buffer
        l0 = lambda name, nb, s=s: self.cx_chip(k).at(c0).l1(f"{T}_kvs_{name}{s}", nb)    # noqa: E731
        l1 = lambda name, nb, s=s: self.cx_chip(k).at(c1).l1(f"{T}_kvp_{name}{s}", nb)    # noqa: E731
        kvd, crow, piece = l0("d", cb), l0("crow", 16 * KR * 2), l0("piece", KVP)
        pv = l1("pv", KVP - 16 * KR)              # [k_pe | value columns], the slot's bytes 16 KR .. KVP
        kvc = self._kvc_late[(k, c0, hk)] if late else self.kvc[(k, c0)]
        # stage 1's slices of the tile have landed (its owner waits): both clusters read them
        w0 = w1 = list(self.own_wait[q])
        ld = self.dm_copy(cl0, f"kvr_ld{s}", src, kvd, cb, list(w0) + kdep(c0))
        dq = self.simd_mul(cl0, f"kv_dq{s}", kvd, kvc, kvd, 16 * KR, [ld])
        cv = self.xd(cl0, f"kv_crm{s}", convert_args("D", "row_major", 16, KR, mesh, 2, kvd, crow), [dq])
        # c^T on cluster 1, first (its SIMD's long dequantisation ahead of k_pe): [seed beat | c^T [512, 16] FP16]
        kvd1, crow1, xt1 = l1("cd", cb), l1("crow", 16 * KR * 2), l1("xt", beat + 16 * KR * 2)
        ldv = self.dm_copy(cl1, f"kvv_ld{s}", src, kvd1, cb, list(w1) + kdep(c1))
        dqv = self.simd_mul(cl1, f"kvv_dq{s}", kvd1, kvcp(c1), kvd1, 16 * KR, [ldv])
        cvv = self.xd(cl1, f"kvv_crm{s}", convert_args("D", "row_major", 16, KR, mesh, 2, kvd1, crow1), [dqv])
        trv = self.node(k, c1, xdma, f"kvv_vt{s}", "__snax_bingo_kernel_xdma_transpose_2d",
                        SnaxBingoKernelXdmaTranspose2dArgs(crow1, xt1.view(beat), 16, KR, 2), [cvv])
        # k_pe on cluster 1: its own load and dequantisation, RoPE, D -> A, quantised into the slot
        kvc1 = kvcp(c1)
        pd, pe, pa = l1("d", n_pe * 2), l1("pe", n_pe * 2), l1("pa", n_pe * 2)
        rc = self.KC_ROPE + lt * self.KC_TILE
        ld1 = self.dm_copy(cl1, f"kvp_ld{s}", BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD + cb), pd,
                           n_pe * 2, list(w1) + kdep(c1))
        xs, t1, t2 = l1("xs", n_pe * 2), l1("t1", n_pe * 2), l1("t2", n_pe * 2)
        dq1 = self.simd_mul(cl1, f"kvp_dq{s}", pd, kvc1.view(cb), pd, n_pe, [ld1])
        sw = [self.node(k, c1, dm, f"kvp_swap{h}{s}", "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(pd.view((1 - h) * half), xs.view(h * half), half),
                        [dq1]) for h in range(2)]
        m1 = self.simd_mul(cl1, f"kvp_cos{s}", pd, kvc1.view(rc), t1, n_pe, [dq1])
        m2 = self.simd_mul(cl1, f"kvp_sin{s}", xs, kvc1.view(rc + n_pe * 2), t2, n_pe, sw)
        ad = self.simd_add(cl1, f"kvp_rope{s}", t1, t2, pe, n_pe, [m1, m2])
        d2p = self.xd(cl1, f"kvp_pa{s}", d_to_a_args(mesh, 1, 4, 16, pe, pa), [ad])
        pq = self.quant(cl1, f"kvp_p8{s}", pa, pv, n_pe * 2, self.fbits(d.s_kpe.inv), [d2p])
        # c: RMSNorm row_major -> A (INT8) on cluster 0 into the key rows, col_major -> B (INT8) on cluster 1 into the
        # value columns, each with the INT8 scale applied
        ra = lambda i, o, la, sd=0: SnaxBingoKernelSimdRmsnormArgs(   # noqa: E731
            i, o, 16, KR, "row_major" if not sd else "col_major", "A" if not sd else "B", seed_addr=sd,
            out_i8=True, inv_scale_f32bits=sc, mesh=mesh, scratch_addr=la(
                "scrb" if sd else "scra", SnaxBingoKernelSimdRmsnormArgs(
                    0, 0, 16, KR, "col_major" if sd else "row_major", "B" if sd else "A", seed_addr=1,
                    out_i8=True, mesh=mesh).scratch_bytes()))
        kq = self.node(k, c0, simd, f"kv_rmsa{s}", "__snax_bingo_kernel_simd_rmsnorm", ra(crow, piece, l0), [cv])
        vq = self.node(k, c1, simd, f"kv_rmsb{s}", "__snax_bingo_kernel_simd_rmsnorm",
                       ra(xt1.view(beat), pv.view(n_pe), l1, xt1), [trv])
        self.kv_put[q] = self.node(k, c0, dm, f"kv_put{s}", "__snax_bingo_kernel_idma_put_flagged",
                                   SnaxBingoKernelIdmaPutFlaggedArgs(piece, on_root(f"{T}_kv", q * KVP), 16 * KR,
                                                                     on_root(f"{T}_kvnflag", q * 64), value=1),
                                   [kq])
        self.kvp_put[q] = self.node(k, c1, dm, f"kvp_put{s}", "__snax_bingo_kernel_idma_put_flagged",
                                    SnaxBingoKernelIdmaPutFlaggedArgs(
                                        pv, on_root(f"{T}_kv", q * KVP + 16 * KR), KVP - 16 * KR,
                                        on_root(f"{T}_kvpflag", q * 64), value=1), [pq, vq])
