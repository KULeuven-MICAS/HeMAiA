# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""GEMM shapes in isolation on one cluster of the FFN prefill platform (hemaia_16c4m_8MBL3_2cluster).

The prefill's GEMMs always share their cluster with SIMD and DMA work, so their time does not say how
fast the array runs a shape. This workload runs a list of shapes one after another on the root chip's
cluster 0 with nothing else on that cluster, twice each (the second pass is the measurement: the
first also pays the instruction fetch), then the gate|up shape again next to a DMA copy and next to a
SIMD add on the same cluster. Each GEMM's start and end are in the scheduler monitor's log
(bingo_progress.log); the operand values are the prompt bytes, so nothing is checked.

Shapes are in array tiles: M row tiles of 16, K steps of 4, N column tiles of 16 (mesh (16, 4, 16)).
The rest of the platform is set up as for the prefill (it is PBuild with other buffers and program).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ffn_prefill_mc as M  # noqa: E402
from ffn_prefill_mc import (SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs,  # noqa: E402
                            SnaxBingoKernelSimdAddF16Args, BingoMemSymbol)

KIB = 1024
# name, M, K, N: the prefill's gate|up (32 rows, K 2048, 6 column tiles) and down (32 rows, K 352, 32 tiles),
# the same with 64 and 128 rows and with twice the columns, and three diagnostic shapes
SHAPES = [("gu32", 2, 512, 6), ("dn32", 2, 88, 32),
          ("gu64", 4, 512, 6), ("dn64", 4, 88, 32),
          ("gu128", 8, 512, 6), ("dn128", 8, 88, 32),
          ("gu32w", 2, 512, 12), ("gu128w", 8, 512, 12),
          # diagnosis of the gate|up rate: shallower K, and the down shape with gate|up's 6 column tiles
          ("gu32_k128", 2, 128, 6), ("gu32_k32", 2, 32, 6), ("dn32_n6", 2, 88, 6)]
# params bench_set offsets: the gate|up shape with the B (then A) base moved by a few bytes: the TCDM bank of each
# operand's first access, and so how often the A and B streams ask for the same bank
OFFSETS = [0, 8, 16, 24, 32, 48, 64, 96, 128]


class GBench(M.PBuild):
    TAG = "gb"

    def _buffers(self):
        self.OFF_LOC = 0
        self.sym_bytes = 64
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_sym", self.sym_bytes)
                    for k in self.chips for c in range(self.NC)}
        cl = self.cx_chip(self.ROOT).at(0)
        self.ba = cl.l1(f"{self.TAG}_a", 256 * KIB + 256)  # A: up to 128 rows x K 2048, int8 (+ offset room)
        self.bb = cl.l1(f"{self.TAG}_b", 192 * KIB + 256)  # B: up to K 2048 x 192 columns, int4 (+ offset room)
        self.bd = cl.l1(f"{self.TAG}_d", 128 * KIB)       # D: up to 8 x 32 FP16 tiles

    def build(self):
        dm, gemm, simd = self.roles["dm"], self.roles["gemm"], self.roles["simd"]
        R, T = self.ROOT, self.TAG
        src = lambda off=0: BingoMemSymbol(f"{T}_xsrc", offset=off)
        # every DM core opens with a bench_ task (the emitter checks each DM core has one first)
        pre = {k: [] for k in self.chips}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            pre[k].append(self.node(k, c, dm, f"bench_init_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(src(), self.sym[(k, c)], self.sym_bytes)))
        go = self.setup(extra=pre)
        a = self.node(R, 0, dm, "bench_lda", "__snax_bingo_kernel_idma_1d_copy",
                      SnaxBingoKernelIdma1dCopyArgs(src(), self.ba, 256 * KIB), [go[R]])
        prev = self.node(R, 0, dm, "bench_ldb", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(src(), self.bb, 192 * KIB), [a])

        def gemm_task(name, m, k, n, after, a_off=0, b_off=0, int4_b=1):
            return self.node(R, 0, gemm, f"bench_{name}", "__snax_bingo_kernel_gemm_full",
                             SnaxBingoKernelGemmFullArgs(
                                 input_A_addr=self.ba.view(a_off) if a_off else self.ba,
                                 input_B_addr=self.bb.view(b_off) if b_off else self.bb, input_C_addr=0,
                                 output_D_addr=self.bd, M=m, K=k, N=n, array_shape_idx=0,
                                 transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1,
                                 int4_b_enable=int4_b, d_shift=9), after)
        # params bench_set int8b: the gate|up shapes with INT8 B (the B reader's INT4 converter bypassed) next to
        # INT4 B, to tell whether the converter costs the array its ninth cycle per eight steps
        if self.p.get("bench_set") == "int8b":
            for rep in range(2):
                for name, m, k, n in (("gu32", 2, 512, 6), ("gu64", 4, 512, 6), ("gu32_k128", 2, 128, 6)):
                    prev = gemm_task(f"{name}_i4_r{rep}", m, k, n, [prev])
                    prev = gemm_task(f"{name}_i8_r{rep}", m, k, n, [prev], int4_b=0)
            self.end_window([prev])
            self.run_checks([])
            return
        # params bench_set paired: the same shapes with INT4 B in the B-layout and in the paired layout (B walked
        # 64 bytes a pass, A's stride), and with INT8 B; the operands are the same bytes, read in each walk's order
        if self.p.get("bench_set") == "paired":
            for rep in range(2):
                for name, m, k, n in (("gu32", 2, 512, 6), ("dn32", 2, 88, 32), ("gu64", 4, 512, 6),
                                      ("gu32_k128", 2, 128, 6)):
                    prev = gemm_task(f"{name}_i4b_r{rep}", m, k, n, [prev])
                    prev = gemm_task(f"{name}_i4p_r{rep}", m, k, n, [prev], int4_b=M.INT4_B_LAYOUT_PAIRED)
                    prev = gemm_task(f"{name}_i8_r{rep}", m, k, n, [prev], int4_b=0)
            self.end_window([prev])
            self.run_checks([])
            return
        if self.p.get("bench_set") == "offsets":
            for rep in range(2):
                for o in OFFSETS:
                    prev = gemm_task(f"gu32_b{o}_r{rep}", 2, 512, 6, [prev], b_off=o)
                for o in OFFSETS[1:]:
                    prev = gemm_task(f"gu32_a{o}_r{rep}", 2, 512, 6, [prev], a_off=o)
            self.end_window([prev])
            self.run_checks([])
            return
        for rep in range(2):
            for name, m, k, n in SHAPES:
                prev = gemm_task(f"{name}_r{rep}", m, k, n, [prev])
        # contention: gate|up next to a DMA copy into the same L1 (64 KiB, twice), then next to a SIMD add
        # (32 KiB); both work in the upper half of D, which the 32-row gate|up does not write
        free = self.bd.view(64 * KIB)
        cp1 = self.node(R, 0, dm, "bench_dma1", "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(src(), free, 64 * KIB), [prev])
        g = gemm_task("gu32_dma", 2, 512, 6, [prev])
        cp2 = self.node(R, 0, dm, "bench_dma2", "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(src(64 * KIB), free, 64 * KIB), [cp1])
        sa = self.node(R, 0, simd, "bench_simd", "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdAddF16Args(free, free.view(32 * KIB), free, 32 * KIB // 64, 32), [g, cp2])
        g2 = gemm_task("gu32_simd", 2, 512, 6, [g, cp2])
        self.end_window([g2, sa])
        self.run_checks([])

    def finish(self, app_name, x_prefixes=("bench_",)):
        # the prefill's finish adds its buffer-hazard proof, over buffers this workload does not have
        M.cm.ChipletBuild.finish(self, app_name, x_prefixes)


def run():
    import argparse
    import hjson
    from bingo_platform import load_cluster_cfg, parse_platform_cfg
    ap = argparse.ArgumentParser(description="GEMM shapes in isolation")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_h", default=None)
    ap.add_argument("--output_offload_file_name", default="offload_bingo_hw.h")
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--hwcfg", required=True)
    ap.add_argument("--platformcfg", required=True)
    ap.add_argument("--dsv2_dir", required=True)
    args = ap.parse_args()
    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    S = GBench(args, p, parse_platform_cfg(args.platformcfg), load_cluster_cfg(args.hwcfg))
    S.build()
    S.finish("FFN GEMM shapes in isolation")
