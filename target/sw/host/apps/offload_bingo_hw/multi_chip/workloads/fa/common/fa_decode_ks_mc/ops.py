# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The task makers the steps share: a task on a cluster's core by role, L1 and L3 buffers, the GEMV, the SIMD passes,
the xDMA reshapes, the copies, and the flagged puts, waits and fetches between chips."""

from bingo_kernel_args import (SnaxBingoKernelGemvArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs,
                               SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
                               SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args,
                               SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelSimdMulI8Args, SnaxBingoKernelWaitFlagsArgs,
                               SnaxBingoKernelXdma6dArgs, SnaxBingoKernelXdmaMemsetArgs, simd_beats)
from bingo_mem_handle import BingoMemSymbol
from libs.blocks.collective import broadcast_put, chain_gather

from .consts import ATOM, BR, LANES, f32bits


class KsOps:
    """Task makers over ChipletBuild.node: `nd` names a task by its cluster (`<name>_c<c>`), the rest build on it."""

    def nd(self, k, c, role, name, kernel, args, deps=()):
        return self.node(k, c, self.roles[role], f"{name}_c{c}", kernel, args,
                         list(dict.fromkeys(t for t in deps if t is not None)))

    def l1(self, k, c, name, nb):
        return self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nb)

    def l3(self, name, off=0, chip=None):
        """An L3 symbol of this block at byte `off`, as chip `chip` names it (None: its own)."""
        return BingoMemSymbol(f"{self.TAG}_{name}", offset=off, chip_id=chip)

    def far(self, k):
        """How chip k names the root's L3."""
        return None if k == self.ROOT else self.ROOT

    def gemv(self, k, c, name, x, w, out, K, N, k_shift, deps):
        """x (a_row, K values) . w (paired INT4, K x N) into the FP16 row out."""
        return self.nd(k, c, "gemm", name, "__snax_bingo_kernel_gemv",
                       SnaxBingoKernelGemvArgs(x, w, out, K=K, N=N, d_shift=k_shift, a_blk=8, w4=True), deps)

    def mul(self, k, c, name, a, b, out, n, deps):
        return self.nd(k, c, "simd", name, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdMulF16Args(a, b, out, n // 32, 32), deps)

    def add(self, k, c, name, a, b, out, n, deps):
        return self.nd(k, c, "simd", name, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdAddF16Args(a, b, out, n // 32, 32), deps)

    def muli8(self, k, c, name, a, b, out, n, inv, deps):
        """out = quant_i8(a (.) b) over n FP16 values: a dequantisation and the quantiser in one SIMD pass."""
        return self.nd(k, c, "simd", name, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdMulI8Args(a, b, out, n // 32, 32, f32bits(inv)), deps)

    def quant(self, k, c, name, src, dst, nbytes, inv, deps):
        return self.nd(k, c, "simd", name, "__snax_bingo_kernel_simd_fp16_to_int8",
                       SnaxBingoKernelSimdFp16ToInt8Args(src, dst, beats=simd_beats(nbytes), rows=1,
                                                         inv_scale_f32bits=f32bits(inv)), deps)

    def xd(self, k, c, name, args, deps):
        return self.nd(k, c, "xdma", name, "__snax_bingo_kernel_xdma_6d", args, deps)

    def spread(self, k, c, name, src, dst, n, deps):
        """An FP16 row of n values into a_row's places, still FP16 (each 8-byte atom 16 B apart), on the xDMA; the
        buffer's other atoms stay as they are (zeroed before the window)."""
        if n % (4 * LANES):
            raise ValueError(f"spread: {n} values are not whole xDMA beats of {LANES} atoms")
        b = n // 4 // LANES
        return self.xd(k, c, name, SnaxBingoKernelXdma6dArgs(src, dst, ATOM, 2 * ATOM, [LANES * ATOM], [b],
                                                             [2 * LANES * ATOM], [b]), deps)

    def row_to_a(self, k, c, name, src, dst, cols, deps):
        """An FP16 row of `cols` values into row 0 of a 16-row FP16 A tile: its 8-byte atoms a 128-byte block apart."""
        blk, b = 4 * BR * 2, cols // 4 // LANES
        return self.xd(k, c, name, SnaxBingoKernelXdma6dArgs(src, dst, ATOM, blk, [LANES * ATOM], [b],
                                                             [LANES * blk], [b]), deps)

    def copy(self, k, c, name, src, dst, nb, deps):
        return self.nd(k, c, "dm", name, "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(src, dst, nb), deps)

    def copy2d(self, k, c, name, src, dst, size, sst, dstr, reps, deps):
        return self.nd(k, c, "dm", name, "__snax_bingo_kernel_idma_2d_copy",
                       SnaxBingoKernelIdma2dCopyArgs(src, dst, size, sst, dstr, reps), deps)

    def put(self, k, c, name, src, dst, nb, flag, deps, bcast=False, also_here=False):
        """A flagged iDMA put on chip k cluster c's DM core; bcast: a multicast into every chip
        (collective.broadcast_put)."""
        args = SnaxBingoKernelIdmaPutFlaggedArgs(src, dst, nb, flag, value=1, bcast=bcast, also_here=also_here)
        if bcast:
            return broadcast_put(lambda n, kern, a, d: self.nd(k, c, "dm", n, kern, a, d), name, args, deps)
        return self.nd(k, c, "dm", name, "__snax_bingo_kernel_idma_put_flagged", args, deps)

    def gather(self, k, c, name, args, deps):
        """A ChainGather collected on chip k cluster c, on its xDMA core (collective.chain_gather)."""
        return chain_gather(lambda n, kern, a, d: self.nd(k, c, "xdma", n, kern, a, d), name, args, deps)

    def _flagged(self, w, puts):
        """The puts a flag wait w follows: a dependency from a task of w's chip, an ordering edge from another's (the
        flag proves the landing)."""
        for t in puts:
            if t.assigned_chiplet_id == w.assigned_chiplet_id:
                if not self.dfg.has_edge(t, w):
                    self.edge(t, w)
            else:
                self.dfg.bingo_add_order_edge(t, w)
        return w

    def wait(self, k, c, name, flags, n, puts, deps=()):
        w = self.nd(k, c, "dm", name, "__snax_bingo_kernel_wait_flags", SnaxBingoKernelWaitFlagsArgs(flags, n, 1),
                    list(deps) + [t for t in puts if t.assigned_chiplet_id == k])
        return self._flagged(w, puts)

    def fetch(self, k, c, name, src, dst, nb, flags, n, puts, deps=()):
        w = self.nd(k, c, "dm", name, "__snax_bingo_kernel_idma_fetch_flagged",
                    SnaxBingoKernelIdmaFetchFlaggedArgs(src, dst, nb, flags, n, 1),
                    list(deps) + [t for t in puts if t.assigned_chiplet_id == k])
        return self._flagged(w, puts)

    def memset(self, k, c, name, buf, nb, deps=()):
        return self.nd(k, c, "xdma", name, "__snax_bingo_kernel_xdma_memset",
                       SnaxBingoKernelXdmaMemsetArgs(buf, nb, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), deps)
