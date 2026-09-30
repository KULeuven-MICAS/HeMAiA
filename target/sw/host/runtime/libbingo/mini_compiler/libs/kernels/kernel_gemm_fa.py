# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The FlashAttention GEMM pair, and the array's performance counters.

QK and PV are separate classes because they are different SHAPES of the same engine
program, not different kernels: QK writes a score tile the softmax reads, PV accumulates
into O. Keeping them apart is what lets the builder state which is which."""

from typing import Union, Dict, Optional
from bingo_mem_handle import BingoMemAlloc

from kernel_base import BingoKernelArgs


class _SnaxBingoKernelGemmFaArgs(BingoKernelArgs):
    """One FlashAttention matmul on the GEMM core. Subclasses bind KERNEL_NAME.

    Shapes are in ARRAY BLOCKS, as the streamer states them, not in elements:

        qk:  S^T = K.Q^T    M = Bc/meshRow, K = d/tileSize,  N = Br/meshCol
        pv:  O^T += V^T.P^T M = d/meshRow,  K = Bc/tileSize, N = Br/meshCol

    For `pv`, pass the SAME handle as input_C_addr and output_D_addr: the matmul then
    computes O += P.V in place, and the accumulation across KV tiles is the GEMM's own C
    input rather than a separate pass.

    These kernels emit the INTERLEAVED D layout the LANEWISE softmax depends on, which is
    what separates them from SnaxBingoKernelGemmFullArgs; see offload_hw_kernels/gemm_fa.h.
    """
    KERNEL_NAME: str = None

    # flags; must match GEMM_FA_* in offload_hw_kernels/gemm_fa.h. PV only.
    #   B_KMAJOR    B's 64-byte blocks are k-major, block (k, n) at (k*N + n)*64: the layout
    #               the softmax's P_INTERLEAVE mode writes P in.
    #   C_COLSCALE  scale C by the FP16 factor of its output column (corr_addr, 32 factors)
    #               on the way into the array: O = corr (.) O + V^T.P^T in one dispatch.
    #   D_FP16      the last tile: O leaves as FP16, RNE(O * 2^-d_shift), into a D buffer of
    #               its own, while C is still read as the INT32 accumulator.
    B_KMAJOR = 1
    C_COLSCALE = 2
    D_FP16 = 4
    # The widest output the C-path scaler covers, in meshCol blocks (its maxColBlocks).
    COLSCALE_MAX_N = 2
    # The D-port converter's largest power-of-two scale: the score tile is RNE(S * 2^-k),
    # k in 0..14. 2^-14 is FP16's smallest normal, so no integer lands in the subnormals.
    DSHIFT_MAX = 14
    B_TILE = 64

    def __init__(self,
                 input_A_addr: Union[BingoMemAlloc, int],
                 input_B_addr: Union[BingoMemAlloc, int],
                 input_C_addr: Union[BingoMemAlloc, int],
                 output_D_addr: Union[BingoMemAlloc, int],
                 M: int, K: int, N: int,
                 perf_addr: Union[BingoMemAlloc, int] = 0,
                 flags: int = 0,
                 corr_addr: Union[BingoMemAlloc, int] = 0,
                 d_shift: int = 0,
                 b_pitch: int = 0):
        is_pv = self.KERNEL_NAME == "__snax_bingo_kernel_gemm_fa_pv"
        if M <= 0 or K <= 0 or N <= 0:
            raise ValueError(f"M, K, N must be positive block counts, got {M}, {K}, {N}")
        if flags & ~(self.B_KMAJOR | self.C_COLSCALE | self.D_FP16):
            raise ValueError(f"flags={flags:#x} has bits outside GEMM_FA_*")
        if flags and not is_pv:
            raise ValueError("B_KMAJOR / C_COLSCALE / D_FP16 describe PV's operands; QK "
                             "takes none")
        # d_shift scales the converter's FP16 output: the score matmul's, or the last PV's.
        if d_shift and is_pv and not flags & self.D_FP16:
            raise ValueError("d_shift scales an FP16 output; an INT32 PV takes none (pass "
                             "D_FP16 on the last tile)")
        if flags & self.D_FP16 and input_C_addr and input_C_addr is output_D_addr:
            raise ValueError("D_FP16 ends the recurrence: C is the INT32 accumulator and D "
                             "a separate FP16 buffer, so the two may not be shared")
        if not 0 <= d_shift <= self.DSHIFT_MAX:
            raise ValueError(f"d_shift={d_shift} outside the converter's 0..{self.DSHIFT_MAX}")
        # b_pitch places PV's B = P^T blocks; QK's B is Q^T, dense.
        if b_pitch and not is_pv:
            raise ValueError("b_pitch places PV's P^T blocks; QK's B (Q^T) is dense")
        if b_pitch and (b_pitch < self.B_TILE or b_pitch % 8):
            raise ValueError(f"b_pitch={b_pitch} must be 0 (dense) or at least "
                             f"{self.B_TILE} and a multiple of 8")
        self.d_shift = int(d_shift)
        self.b_pitch = int(b_pitch)
        if flags & self.C_COLSCALE:
            if not corr_addr:
                raise ValueError("C_COLSCALE needs corr_addr: the 32 FP16 factors to apply")
            if not input_C_addr:
                raise ValueError("C_COLSCALE with a NULL C scales nothing: the channels are "
                                 "masked and the array sees zeros. Drop the flag on tile 0.")
            if N > self.COLSCALE_MAX_N:
                raise ValueError(f"C_COLSCALE covers {self.COLSCALE_MAX_N} column blocks "
                                 f"(Int32ColumnScale maxColBlocks), N={N}")
        self.flags = int(flags)
        self.corr_addr = corr_addr
        self.input_A_addr = input_A_addr
        self.input_B_addr = input_B_addr
        self.input_C_addr = input_C_addr
        self.output_D_addr = output_D_addr
        self.M = M
        self.K = K
        self.N = N
        # Optional five-word L1 accumulator for the array's own hardware counters. 0 (the
        # default) compiles to a kernel that does not even read the CSRs, so leaving it
        # unset costs nothing; see the struct comment in device_kernel_args.h.
        self.perf_addr = perf_addr

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_gemm_fa_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_A_addr, "input_A_addr", a, handle_name_map,
                           split_64bit=False)
        self._process_addr(self.input_B_addr, "input_B_addr", a, handle_name_map,
                           split_64bit=False)
        self._process_addr(self.input_C_addr, "input_C_addr", a, handle_name_map,
                           split_64bit=False)
        self._process_addr(self.output_D_addr, "output_D_addr", a, handle_name_map,
                           split_64bit=False)
        a["M"] = str(self.M)
        a["K"] = str(self.K)
        a["N"] = str(self.N)
        self._process_addr(self.perf_addr, "perf_addr", a, handle_name_map,
                           split_64bit=False)
        a["flags"] = str(self.flags)
        self._process_addr(self.corr_addr, "corr_addr", a, handle_name_map,
                           split_64bit=False)
        a["d_shift"] = str(self.d_shift)
        a["b_pitch"] = str(self.b_pitch)
        return a


class SnaxBingoKernelGemmPerfReportArgs(BingoKernelArgs):
    """Print one cluster's accumulated VersaCore counters, then clear them.

    Place it on the GEMM core after the last matmul: the counters reset on every config
    write, so they must be read per dispatch, but printing belongs outside the window the
    utilisation figure is divided by."""
    KERNEL_NAME = "__snax_bingo_kernel_gemm_perf_report"

    def __init__(self, perf_addr: Union[BingoMemAlloc, int], ideal_cc: int = 0):
        self.perf_addr = perf_addr
        self.ideal_cc = ideal_cc

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_gemm_perf_report_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.perf_addr, "perf_addr", a, handle_name_map,
                           split_64bit=False)
        a["ideal_cc"] = str(self.ideal_cc)
        return a


class SnaxBingoKernelGemmFaQkArgs(_SnaxBingoKernelGemmFaArgs):
    """S^T = K.Q^T, emitted as FP16 (the Int32ToFp16Converter on the D port is armed, so
    the score tile reaches TCDM in half the beats an INT32 tile would need). With d_shift =
    k the tile is RNE(S * 2^-k): full-range INT8 scores stay finite, and the softmax that
    reads it must be told a' = a * 2^k."""
    KERNEL_NAME = "__snax_bingo_kernel_gemm_fa_qk"


class SnaxBingoKernelGemmFaPvArgs(_SnaxBingoKernelGemmFaArgs):
    """O^T += V^T.P^T, accumulated in place in INT32 (converter off, C == D)."""
    KERNEL_NAME = "__snax_bingo_kernel_gemm_fa_pv"


