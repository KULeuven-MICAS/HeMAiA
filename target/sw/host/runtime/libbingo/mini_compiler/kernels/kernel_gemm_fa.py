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

    def __init__(self,
                 input_A_addr: Union[BingoMemAlloc, int],
                 input_B_addr: Union[BingoMemAlloc, int],
                 input_C_addr: Union[BingoMemAlloc, int],
                 output_D_addr: Union[BingoMemAlloc, int],
                 M: int, K: int, N: int,
                 perf_addr: Union[BingoMemAlloc, int] = 0):
        if M <= 0 or K <= 0 or N <= 0:
            raise ValueError(f"M, K, N must be positive block counts, got {M}, {K}, {N}")
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
    the score tile reaches TCDM in half the beats an INT32 tile would need)."""
    KERNEL_NAME = "__snax_bingo_kernel_gemm_fa_qk"


class SnaxBingoKernelGemmFaPvArgs(_SnaxBingoKernelGemmFaArgs):
    """O^T += V^T.P^T, accumulated in place in INT32 (converter off, C == D)."""
    KERNEL_NAME = "__snax_bingo_kernel_gemm_fa_pv"


