# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaBuild's task helpers: one task of one cluster each."""

import numpy as np

from bingo_kernel_args import (
    INT4_B_LAYOUT_PAIRED, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelSimdAddF16Args,
    SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelSimdMulI8Args, simd_beats)


class FaOps:

    def gmm(self, cl, name, a, b, d, M, K, N, k_shift, deps):
        """One GEMM of cluster cl (gemm_full). Returns the task its output follows."""
        k, c = self.chip(cl), self.local(cl)
        args = SnaxBingoKernelGemmFullArgs(
            input_A_addr=a, input_B_addr=b, input_C_addr=0, output_D_addr=d, M=M, K=K, N=N, array_shape_idx=0,
            transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1,
            int4_b_enable=INT4_B_LAYOUT_PAIRED if self.wbits == 4 else 0, d_shift=k_shift)
        return self.node(k, c, self.roles["gemm"], name, "__snax_bingo_kernel_gemm_full", args, list(deps))

    def simd_mul(self, cl, name, a, b, out, nelems, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdMulF16Args(a, b, out, nelems // 32, 32), list(deps))

    def simd_add(self, cl, name, a, b, out, nelems, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdAddF16Args(a, b, out, nelems // 32, 32), list(deps))

    def mul_quant(self, cl, name, a, scale, dst, nelems, inv_bits, deps):
        """dst = quantise(a (.) scale): a dequantisation and the quantise after it in one SIMD task."""
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdMulI8Args(a, scale, dst, nelems // 32, 32, inv_bits), list(deps))

    def quant(self, cl, name, src, dst, nbytes, inv_bits, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_fp16_to_int8",
                         SnaxBingoKernelSimdFp16ToInt8Args(src, dst, beats=simd_beats(nbytes), rows=1,
                                                           inv_scale_f32bits=inv_bits), list(deps))

    def xd(self, cl, name, args, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["xdma"], name, "__snax_bingo_kernel_xdma_6d", args, list(deps))

    def dm_copy(self, cl, name, src, dst, nbytes, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["dm"], name, "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(src, dst, nbytes), list(deps))

    @staticmethod
    def fbits(inv):
        return int(np.asarray(inv, dtype=np.float32).reshape(()).view(np.uint32))
