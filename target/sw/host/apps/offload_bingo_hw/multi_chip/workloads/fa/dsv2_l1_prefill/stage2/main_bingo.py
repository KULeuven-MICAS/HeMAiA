#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention, stage 2 of the 256-token prefill on sixteen chiplets alone: every owner
finishes its tiles' keys and puts the pieces into the root's L3.

Stage 1's result is in each owner's memory chip's HBM (its tiles' latent rows as the D port leaves them, kvraw, with
every slice's flag set) and is copied into the owner's L3 before the window, with every cluster's key constants
(W_KVA's dequantisation scales and the tiles' RoPE tables). Per tile, in two parts: cluster 0 takes c (its 512 columns)
into L1, dequantises it, lays it out row major and RMSNorms it into the key rows' c (INT8, A layout); cluster 1 builds
c transposed itself (its own load, dequantisation, relayout and transpose on its xDMA) and RMSNorms it into the value
columns (INT8, B of c = A of c^T), and dequantises and rotates k_pe (RoPE) into the key rows' k_pe. Cluster 0 puts the
key rows' c into the root's L3 behind the tile's flag (kvnflag), cluster 1 puts [k_pe | value columns] behind a flag of
its own (kvpflag); the root waits for every piece, a chunk at a time, on the cluster that multicasts them in stage 3.

          before the window     kvc_ld, kvr_pre, kvr_pref     DM: the key constants into L1, stage 1's result into L3
          S0_start
          every owner, every tile (its lt-th: suffix _t<lt> from the second):
            cluster 0: kvr_wait, kvr_ld, kv_dq, kv_crm, kv_rmsa, kv_put
            cluster 1: kvp_wait, kvv_ld, kvv_dq, kvv_crm, kvv_vt, kvp_ld, kvp_dq, kvp_swap<h>, kvp_cos, kvp_sin,
                       kvp_rope, kvp_pa, kvp_p8, kv_rmsb, kvp_put
          the root: kv_nwait_h<h>, kv_pwait_h<h>: every piece of chunk h has landed
          P_fin<chip> -> S1_end -> fp_report, the checks

Checks: the root's copy of three tiles' pieces (key rows [16, 576] then value columns [512, 16], INT8, A layout),
the key rows' head and tail, their k_pe and the value columns' tail.

params.hjson: the platform, the tokens, the chunk and key-tile sizes, the weight chunk and ring sizes, the weights'
width."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from stage_common import KV_RANK, KVP, ROPE, TILE16, BingoMemSymbol, StageBase, app_name  # noqa: E402
from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,  # noqa: E402
                               SnaxBingoKernelSimdRmsnormArgs, SnaxBingoKernelWaitFlagsArgs,
                               SnaxBingoKernelXdmaTranspose2dArgs, simd_fp16_lanes)
from driver import run_driver  # noqa: E402
from libs.comm.nest import convert_args, d_to_a_args  # noqa: E402

CHK = 2048                                  # bytes a checked window


class Stage2(StageBase):
    """Stage 2 alone (the module docstring)."""
    STAGE = 2

    def stage_inputs(self):
        """Stage 1's result in each owner's memory chip's HBM (stage1_result), then the k_pe parts' flags on the
        root, one a tile."""
        self.stage1_result()
        self.st.put_zeros(f"{self.TAG}_kvpflag", "uint8_t", (self.T // 16) * 64)

    def stage1_result(self):
        """Each owner's stage 1 result, copied into its L3 before the window: its tiles' latent rows as the D port
        leaves them (kvraw, FP16, D layout) and every slice's flag (kvrflag)."""
        d, lay, T = self.d, self.lay, self.TAG
        raw = np.asarray(d.fp.d_port(d._mm(d.hq, d.wkva.q), d.k_x), dtype=np.float16)
        nr, nf = self.KVR_TILES * self.KVD, self.KVR_TILES * len(self.chips) * 64
        self.h_kvpre = {}
        for k in self.chips:
            img = np.zeros(nr + nf, dtype=np.uint8)
            for lt, q in enumerate(self.kv_tiles[k]):
                img[lt * self.KVD:(lt + 1) * self.KVD] = lay.to_d(raw[16 * q:16 * q + 16], d.mesh).astype(
                    np.float16).view(np.uint8).reshape(-1)
            img[nr:].view(np.uint32).reshape(-1, 16)[:, 0] = 1
            m = self.mem[k]
            self.h_kvpre[k] = self.st.put_hbm(f"{T}_kvpre_k{k:02x}", img.view(np.int8), mem_chip=(m >> 4, m & 15))

    def build(self):
        self.last_ld = {}
        for ls in self.ls:
            ls.strict = True
        go = self.setup(extra=self.stage1_preload())
        self.relays()
        self.x_puts = []
        last = {k: [] for k in self.chips}
        self.stage_owner(go, last)
        self.window_end(last)
        self.run_checks(self.piece_checks())

    def stage1_preload(self):
        """Before the window, per chip one chain ending on cluster 0: every cluster's key constants (W_KVA's
        dequantisation scales and RoPE tables: kvc on the tiles' cluster, kvc_pe on the other, for k_pe), then stage
        1's result from HBM into the chip's L3."""
        T, nr, nf = self.TAG, self.KVR_TILES * self.KVD, self.KVR_TILES * len(self.chips) * 64
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
            cl = self.chips.index(k) * self.NC
            pre[k] = [self.dm_copy(cl, "kvr_pre", self.h_kvpre[k].address, BingoMemSymbol(f"{T}_kvraw"), nr, pre[k])]
            pre[k] = [self.dm_copy(cl, "kvr_pref", self.h_kvpre[k].address + nr, BingoMemSymbol(f"{T}_kvrflag"), nf,
                                   pre[k])]
        return pre

    # ---- the owners -----------------------------------------------------------------------------------------------
    def stage_owner(self, go, last):
        """kv_tile on every chip's tiles, its waits behind the chip's start (the flags are already set), then the
        root's waits for every piece, a chunk at a time, on the cluster that multicasts them in stage 3."""
        R, T, dm, P = self.ROOT, self.TAG, self.roles["dm"], self.C // 16
        self.kvc_put = {(q, k2): go[k2] for q in range(self.NX) for k2 in self.chips}
        self.kvp_put = {}
        d, mesh = self.d, self.d.mesh
        c8 = np.arange(16 * KV_RANK).reshape(16, KV_RANK)
        assert np.array_equal(d.layout.to_b(c8, mesh), d.layout.to_a(np.ascontiguousarray(c8.T), mesh))
        for k in self.chips:
            for lt, q in enumerate(self.kv_tiles[k]):
                self.kv_tile(k, lt, q)
            last[k] += [self.kv_put[q] for q in self.kv_tiles[k]] + [self.kvp_put[q] for q in self.kv_tiles[k]
                                                                    if q in self.kvp_put]
        c, prev = self.x_put_cluster, []
        waits = [("kv_nwait", "kvnflag", self.kv_put), ("kv_pwait", "kvpflag", self.kvp_put)]
        for h in range(self.NCH):
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

    def kv_tile(self, k, lt, q):
        """Chip k's tile q (its lt-th) as two parts. Cluster 0 takes c: its 512 columns (the first 16 KB of the tile's
        D layout) into L1, their dequantisation, and RMSNorm row_major -> A (INT8) writes the key rows' c. Cluster 1
        builds c transposed itself (its own load and dequantisation of c, D -> row major and the transpose on its own
        xDMA) and RMSNorm col_major -> B (INT8) writes the value columns from it (B of c [16, 512] is A of c^T); k_pe
        (the last 2 KB) is dequantised and rotated (RoPE) into the key rows' k_pe there too, first on its in-order SIMD
        core. Cluster 0 puts the key rows' c behind the tile's flag (kvnflag); cluster 1 puts k_pe and the value
        columns (adjacent in the slot) behind a flag of its own (kvpflag)."""
        d, T, mesh = self.d, self.TAG, self.d.mesh
        simd, xdma, dm = self.roles["simd"], self.roles["xdma"], self.roles["dm"]
        R, nch, KR, RP = self.ROOT, len(self.chips), KV_RANK, ROPE
        on_root = lambda name, off: BingoMemSymbol(name, offset=off, chip_id=None if k == R else R)    # noqa: E731
        cb, n_pe, half = (KR // 16) * TILE16, 16 * RP, (RP // 32) * TILE16
        sc = self.fbits(d.s_c.inv)
        beat = simd_fp16_lanes() * 2
        s = "" if lt == 0 else f"_t{lt}"
        c0, c1 = 0, 1
        cl0, cl1 = self.chips.index(k) * self.NC + c0, self.chips.index(k) * self.NC + c1
        puts = [self.kvc_put[(q, k2)] for k2 in self.chips]

        def wait(c, name):
            # stage 1's slices of the tile have landed: both clusters read them
            w = self.node(k, c, dm, f"{name}{s}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_kvrflag", offset=lt * nch * 64), nch, 1),
                          [t for t in puts if self.flag_dep(t, k, c)])
            for t in puts:
                if not self.flag_dep(t, k, c):
                    self.dfg.bingo_add_order_edge(t, w)
            return [w]
        src = BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD)
        # a chip's later tiles (more tiles than chips) take buffers of their own names: one handle a buffer
        l0 = lambda name, nb, s=s: self.cx_chip(k).at(c0).l1(f"{T}_kvs_{name}{s}", nb)    # noqa: E731
        l1 = lambda name, nb, s=s: self.cx_chip(k).at(c1).l1(f"{T}_kvp_{name}{s}", nb)    # noqa: E731
        kvd, crow, piece = l0("d", cb), l0("crow", 16 * KR * 2), l0("piece", KVP)
        pv = l1("pv", KVP - 16 * KR)              # [k_pe | value columns], the slot's bytes 16 KR .. KVP
        kvc = self.kvc[(k, c0)]
        w0 = wait(c0, "kvr_wait")
        w1 = wait(c1, "kvp_wait")
        ld = self.dm_copy(cl0, f"kvr_ld{s}", src, kvd, cb, list(w0))
        dq = self.simd_mul(cl0, f"kv_dq{s}", kvd, kvc, kvd, 16 * KR, [ld])
        cv = self.xd(cl0, f"kv_crm{s}", convert_args("D", "row_major", 16, KR, mesh, 2, kvd, crow), [dq])
        # c^T on cluster 1, first (its SIMD's long dequantisation ahead of k_pe): [seed beat | c^T [512, 16] FP16]
        kvd1, crow1, xt1 = l1("cd", cb), l1("crow", 16 * KR * 2), l1("xt", beat + 16 * KR * 2)
        ldv = self.dm_copy(cl1, f"kvv_ld{s}", src, kvd1, cb, list(w1))
        dqv = self.simd_mul(cl1, f"kvv_dq{s}", kvd1, self.kvc_pe[(k, c1)], kvd1, 16 * KR, [ldv])
        cvv = self.xd(cl1, f"kvv_crm{s}", convert_args("D", "row_major", 16, KR, mesh, 2, kvd1, crow1), [dqv])
        trv = self.node(k, c1, xdma, f"kvv_vt{s}", "__snax_bingo_kernel_xdma_transpose_2d",
                        SnaxBingoKernelXdmaTranspose2dArgs(crow1, xt1.view(beat), 16, KR, 2), [cvv])
        # k_pe on cluster 1: its own load and dequantisation, RoPE, D -> A, quantised into the slot
        kvc1 = self.kvc_pe[(k, c1)]
        pd, pe, pa = l1("d", n_pe * 2), l1("pe", n_pe * 2), l1("pa", n_pe * 2)
        rc = self.KC_ROPE + lt * self.KC_TILE
        ld1 = self.dm_copy(cl1, f"kvp_ld{s}", BingoMemSymbol(f"{T}_kvraw", offset=lt * self.KVD + cb), pd,
                           n_pe * 2, list(w1))
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

    # ---- checks -------------------------------------------------------------------------------------------------
    def piece_checks(self):
        """The root's copy of three tiles' pieces: key rows [16, 576] then value columns [512, 16], INT8, A layout,
        and the key rows' k_pe from the other cluster."""
        d, T, n = self.d, self.TAG, CHK
        checks = []
        for q in (0, self.NX // 2, self.NX - 1):
            img = np.asarray(d.kv_piece(16 * q)).view(np.uint8).reshape(-1)
            for nm, off in (("a", 0), ("k", KVP // 2 - n), ("b", KVP - n), ("p", 16 * KV_RANK)):
                checks.append((f"kv_q{q}{nm}", BingoMemSymbol(f"{T}_kv", offset=q * KVP + off), img[off:off + n], n))
        return checks


if __name__ == "__main__":
    run_driver("fa prefill stage 2 workload generator", Stage2, app_name)
