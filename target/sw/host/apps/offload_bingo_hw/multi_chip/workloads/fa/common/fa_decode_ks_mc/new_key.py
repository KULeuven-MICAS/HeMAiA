# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The new key (README.md, THE STEPS: new key, append): on the root's cluster 1, W_KVA's 18 rows dequantised, RMSNorm
of c, RoPE of k_pe at P, the piece of 16 keys and the new tile's K and V; then its append into the cache in HBM."""

from bingo_kernel_args import SnaxBingoKernelSimdRmsnormArgs, SnaxBingoKernelXdmaTranspose2dArgs
from libs.comm.nest import convert_args

import chiplet_mc as cm

from .consts import BC, C1_COS, C1_FKV, C1_SIN, D_QK, D_V, KB, KV_RANK, N_KVA, PK, PVB, ROPE


class KsNewKey:
    """FaDecodeKs's new key: made after the query rows (build), appended by the layer at its end (kv_append)."""

    def _new_key(self, kv_puts):
        """The root's cluster 1 waits for the 18 W_KVA rows (kv_puts) and makes the new piece: its key rows (row 0 of a
        zeroed FP16 A tile [16, 576], quantised in two parts, c at s_c and k_pe at s_kpe) into the new tile's K, its
        value columns (c~ in row 0 of a zeroed [16, 512] tile, transposed, the A layout, quantised) into the first four
        columns of every 16-row block of the new tile's V. Returns {"K": tasks, "V": tasks} after which they are
        there."""
        d, R, c, b = self.d, self.ROOT, self.KC, self.b1
        kvl, kv16, krow = self.l1(R, c, "kvraw", 2 * D_QK), self.l1(R, c, "kv16", 2 * D_QK), self.l1(R, c, "krow",
                                                                                                    2 * D_QK)
        kw = self.fetch(R, c, "kv_wait", self.l3("kvraw"), kvl, 2 * D_QK, self.l3("kvrflag"), N_KVA, kv_puts)
        dkv = self.mul(R, c, "dq_kv", kvl, b["c"].view(C1_FKV), kv16, D_QK, [kw])
        rms_a = SnaxBingoKernelSimdRmsnormArgs(kv16, krow, 1, KV_RANK)
        rscr = self.l1(R, c, "kvn_scr", rms_a.scratch_bytes())
        rms = self.nd(R, c, "simd", "kv_rms", "__snax_bingo_kernel_simd_rmsnorm",
                      SnaxBingoKernelSimdRmsnormArgs(kv16, krow, 1, KV_RANK, scratch_addr=rscr), [dkv])
        kpe, rt, rs = kv16.view(2 * KV_RANK), self.l1(R, c, "krt", 2 * ROPE), self.l1(R, c, "krs", 2 * ROPE)
        hf = ROPE // 2
        tc = self.mul(R, c, "kp_cos", kpe, b["c"].view(C1_COS), rt, ROPE, [dkv])
        sa = self.mul(R, c, "kp_sa", kpe.view(2 * hf), b["c"].view(C1_SIN), rs, hf, [dkv])
        sb = self.mul(R, c, "kp_sb", kpe, b["c"].view(C1_SIN + 2 * hf), rs.view(2 * hf), hf, [dkv])
        kr = self.add(R, c, "kp_add", rt, rs, krow.view(2 * KV_RANK), ROPE, [tc, sa, sb, rms])
        # the key rows: row 0 of the FP16 A tile [16, 576], quantised (c at s_c, k_pe at s_kpe) into the new tile's K
        r2a = self.row_to_a(R, c, "k_toa", krow, b["ka16"], D_QK, [rms, kr])
        qk1 = self.quant(R, c, "k8_c", b["ka16"], b["kt"], 2 * 16 * KV_RANK, d.s_c.inv, [r2a])
        qk2 = self.quant(R, c, "k8_pe", b["ka16"].view(2 * 16 * KV_RANK), b["kt"].view(16 * KV_RANK), 2 * 16 * ROPE,
                         d.s_kpe.inv, [r2a])
        # the value columns: c~ in row 0 of the FP16 [16, 512] tile, transposed, the A layout, quantised at s_c, then
        # the first four columns of every row block of the new tile's V
        cpv = self.copy(R, c, "v_row", krow, b["vr16"], 2 * KV_RANK, [rms])
        vt16, va16 = self.l1(R, c, "vt16", 2 * PVB), self.l1(R, c, "va16", 2 * PVB)
        b["vp8"] = self.l1(R, c, "vp8", PVB)
        trv = self.nd(R, c, "xdma", "v_tr", "__snax_bingo_kernel_xdma_transpose_2d",
                      SnaxBingoKernelXdmaTranspose2dArgs(b["vr16"], vt16, 16, KV_RANK, 2), [cpv])
        cva = self.xd(R, c, "v_toa", convert_args("row_major", "A", KV_RANK, 16, cm.MESH, 2, vt16, va16), [trv])
        qv = self.quant(R, c, "v8", va16, b["vp8"], 2 * PVB, d.s_c.inv, [cva])
        vsc = self.copy2d(R, c, "v_tile", b["vp8"], b["vt"], 256, 256, 16 * BC, D_V // 16, [qv])
        return {"K": [qk1, qk2], "V": [vsc]}

    def kv_append(self, deps):
        """The new piece from the new key's cluster into its slot of the root's cache shard in HBM: the key rows, then
        the value columns (32 runs of 256 B one 16-row block of the V tile apart). Returns the last task."""
        R, c = self.ROOT, self.KC
        base = self.h_kv[R].address + self.kv_new
        ak = self.copy(R, c, "kv_app_k", self.b1["kt"], base, PK, deps)
        return self.copy2d(R, c, "kv_app_v", self.b1["vp8"], base + KB, 256, 256, 16 * BC, D_V // 16, [ak])
