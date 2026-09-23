# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# SIMD FP16 RMSNorm — multi-row, ONE fused all-device kernel.
# out[r,:] = x[r,:] / sqrt(mean_j x[r,j]^2) over [rows, D] tiles (D = cols). The whole pipeline
# runs in a single SIMD-core kernel; the host only loads the input, stores the output, and checks it.
#
#   Load x[rows,D] -> RMSNorm __snax_bingo_kernel_simd_rmsnorm -> Store + Check(fp16 tol)
#   (also) the same kernel with out_i8=True -> int8 out + Check(int8 +-1)
#
# Args are HW-free: { input_addr, output_addr, rows, cols }; precision is in the kernel name.
# Inside the one kernel, all on the SIMD core: reduce(SUMSQ) -> broadcast carrying
# StreamMap(a=1/D, RSQRT) -> normalize (x * inv_rms) -> fused Fp16ToInt8.
#
# THE LUT BELOW IS STALE, and deliberately left in place to be re-measured. It was taken
# when inv_rms was computed ON THE CORE -- six serial `divu` through the integer
# sqrt_f16 + recip_f16 plus sixteen volatile stores a row, which at [32, 128] was 59% of
# the kernel. StreamMap's RSQRT func now does it in the datapath, on the broadcast pass
# that a per-row scalar needs anyway, so every entry here should fall and the rows slope
# should flatten. The snax reference app measures 7,717 -> 3,135 cc at [32, 128].
#
# Measured cost LUT -- PRE-RSQRT, re-measure
#     rows\cols     64     128     256
#        1           -      901     964
#        2         3207    1399    1599
#        4         1755    1961    2361
#        8         2676    3085    3886      ((1,64) is the warm-up config)
#
# NOTE the rows == 1 entries are no longer a distinct fast path: it existed only because a
# single row's scalar could be folded into a StreamMap immediate once the core had it in a
# register, and RSQRT keeps the scalar in the datapath where a CSR cannot reach it.

import os
import sys
import json
import argparse
import pathlib
import hjson
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(_THIS, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim")
for _p in [p for p in list(sys.path) if str(p).rstrip('/').endswith('util/sim')]:
    for _s in ('common', 'gemm', 'xdma', 'ara'):
        _sub = os.path.join(_p, _s)
        if _sub not in sys.path:
            sys.path.append(_sub)

import _bingo_paths  # noqa: F401,E402  (puts mini_compiler's grouped subdirs on sys.path)
from bingo_dfg import BingoDFG                            # noqa: E402
from bingo_platform import core_roles, guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode                          # noqa: E402
from bingo_mem_handle import BingoMemAlloc                     # noqa: E402
from bingo_data_staging import DataStaging                     # noqa: E402
from bingo_kernel_args import (                           # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelSimdRmsnormArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

# Core placement, from the generated map (snax_core_roles_defs.h).
_ROLES = core_roles()
SIMD_CORE = _ROLES["simd"]
DMA_CORE = _ROLES["dm"]
HOST_CORE = _ROLES["host"]
CHECK_FP16_TOL = 2
# NOTE: the fused FP16->INT8 rmsnorm variant (the same kernel with out_i8=True)
# is dropped from THIS CI sweep for the same L3-heap reason as simd_softmax_1cluster
# (48 host nodes don't fit the 128 KiB heap next to the 88 KiB device binary). The
# f16 chain preserves the full rows x cols cost LUT.


# (rows, cols); each row is a length-cols tile (cols a power of two). A rows x cols grid for the
# fused-kernel cost LUT (bilinear fit): rows {1,2,4,8} x cols {64,128,256} -- cols varies at EVERY
# row so the cols slope de-confounds from the rows slope. (The rows==1 fast path this grid
# was designed to separate is gone; see the header.)
_LUT_GRID = [(r, c) for r in (1, 2, 4, 8) for c in (64, 128, 256)]
CONFIGS = [{"rows": r, "cols": c} for (r, c) in _LUT_GRID]


def _rmsnorm_ref(rows, cols, i):
    """x and its golden. The golden is the TRUE 1/sqrt, which is what the device computes.

    The reduce accumulates in FP32 and narrows the scalar to FP16 before the rsqrt sees it,
    so this narrows too -- that step is in the datapath and is not an approximation the
    reference gets to skip. What follows it, though, is now a hardware table accurate to
    ~1 FP16 ULP, so the reference is the real 1/sqrt rather than a bit-exact model of the
    core's integer sqrt+reciprocal. Modelling the old path here would score the device
    against the 2-ULP error the RSQRT func was added to remove.
    """
    D = cols
    rng = np.random.RandomState(3300 + i)
    x_rows, y_rows = [], []
    for r in range(rows):
        x = rng.uniform(-4.0, 4.0, size=D).astype(np.float16)
        xf = x.astype(np.float32)
        ssq = np.float16(np.float32((xf ** 2).sum()))      # HW: fp32 accumulate -> fp16 scalar
        ssq_bits = int(np.float16(ssq).view(np.uint16))
        mean = np.float32(ssq) / np.float32(D)             # D is 2^k, so this is exact
        inv16 = np.float16(np.float32(1.0) / np.sqrt(mean))
        y = (xf * np.float32(inv16)).astype(np.float16)    # fp16 elementwise MUL
        x_rows.append(x)
        y_rows.append(y)
    return np.concatenate(x_rows), np.concatenate(y_rows)


class G:
    def __init__(self, dfg):
        self.dfg = dfg

    def l1(self, name, size):
        return BingoMemAlloc(name, size=size, mem_level="L1", chip_id=0, cluster_id=0)

    def node(self, name, core, kname, kargs, after):
        nd = BingoNode(assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=core,
                       node_name=name, kernel_name=kname, kernel_args=kargs)
        self.dfg.bingo_add_node(nd)
        if after is not None:
            self.dfg.bingo_add_edge(after, nd)
        return nd


def build_mempool(st):
    """Hand every config's arrays to the staging helper; return the handles.

    WHERE they land is the platform's business, not this workload's: a config with a
    memory chiplet gets a mempool.bin, one without gets C arrays in the host image. See
    util/sim/common/bingo_data_staging.py -- addressing a memory chiplet a config does
    not have reads unmapped memory rather than faulting.
    """
    meta = []

    for i in range(len(CONFIGS)):
        x, y = _rmsnorm_ref(CONFIGS[i]["rows"], CONFIGS[i]["cols"], i)
        oi = st.put(f"rmsnorm_in_{i}", "uint16_t", x.view(np.uint16))
        og = st.put(f"rmsnorm_golden_{i}", "uint16_t", y.view(np.uint16))
        meta.append((oi, og))
    return meta


# One shared L1/L3 buffer set reused across all serialized configs (see simd_silu_1cluster).
def build_config(g, i, meta, l1_x, l1_f16, l3_f, prev):
    rows  = CONFIGS[i]["rows"]
    D     = CONFIGS[i]["cols"]         # per-row width / reduction length (cols)
    n     = rows * D                   # total elements
    tot_b = rows * D * 2               # [rows, D] fp16 bytes
    off_in, off_golden = meta[i]

    # One fused kernel runs the whole rmsnorm on the DM core; the host loads the input
    # (from the memchip), stores the fp16 output, and checks it.
    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(off_in, l1_x, tot_b), prev)
    # fp16 output: reduce-SUMSQ, broadcast carrying StreamMap(1/D, RSQRT), normalize.
    rn_f = g.node(f"RMSNormF16_{i}", SIMD_CORE, "__snax_bingo_kernel_simd_rmsnorm",
                  SnaxBingoKernelSimdRmsnormArgs(l1_x, l1_f16, rows, D), load)
    st_f = g.node(f"StoreF16_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                  HostBingoKernelIdmaArgs(l1_f16, l3_f, tot_b), rn_f)
    ck_f = g.node(f"Check_rmsnorm_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                  HostBingoKernelCheckResultArgs(off_golden, l3_f,
                      name=f"rmsnorm_cfg{i}", check_type=CHECK_FP16_TOL, num_elements=n,
                      tolerance=0.03), st_f)
    return ck_f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    p.add_argument("--configs_out", type=pathlib.Path, default=None)
    args = p.parse_args()
    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    # The platform decides WHERE the arrays go -- a memory chiplet if this config
    # has one, the host image otherwise -- so it has to be parsed before staging.
    platform = parse_platform_cfg(args.platformcfg)
    st = DataStaging(platform)
    meta = build_mempool(st)
    if args.data_h is not None:
        n = st.emit(args.data_h, args.output_dir)
        print(f"Staged {n} B of inputs and goldens "
              f"{'on the memory chiplet' if st.on_memchip else 'in the host image'}")
    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": "rmsnorm", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    # Single-chip workload: build the DFG for chip 0x00 only (see simd_silu_1cluster).
    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[0x00])
    g = G(dfg)
    max_tot_b = max(r * c * 2 for (r, c) in _LUT_GRID)
    l1_x = g.l1("rms_x", max_tot_b)
    l1_f16 = g.l1("rms_f16", max_tot_b)
    l3_f = BingoMemAlloc("out_rmsnorm_f16", size=max_tot_b, mem_level="L3")
    prev = None
    for i in range(len(CONFIGS)):
        prev = build_config(g, i, meta, l1_x, l1_f16, l3_f, prev)
    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("xDMA rmsnorm (fused)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["rmsnorm_data.h"])
    print(f"Generated rmsnorm: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
