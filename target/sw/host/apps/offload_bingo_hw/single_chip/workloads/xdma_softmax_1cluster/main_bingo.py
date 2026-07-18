# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xDMA FP16 SOFTMAX — multi-row, ONE fused all-device kernel.
# out[r,:] = softmax(x[r,:]) over [rows, D] tiles (D = cols). The whole pipeline runs in a
# single DM-core kernel; the host only loads the input, stores the output, and checks it.
#
#   Load x[rows,D] (iDMA L3->L1)
#   Softmax  __snax_bingo_kernel_xdma_softmax_f16_f16   x -> out[rows,D] fp16
#   Store(out) + Check(fp16 tol)
#   (also) __snax_bingo_kernel_xdma_softmax_f16_i8      x -> out[rows,D] int8  + Check(int8 +-1)
#
# Precision is picked by KERNEL NAME (f16_f16 / f16_i8), not an arg -- the args are HW-free:
# { input_addr, output_addr, rows, cols }. Inside the one kernel (offload_hw_kernels/xdma.h) all
# on the DM core: reduce(MAX) -> negate -> sub-max -> merged EXP+Sexp -> integer reciprocal
# (rv32iM divu, no FPU) -> normalize-MUL, and the fused Fp16ToInt8 quant for the i8 variant.
#
# Measured cost LUT (single-chip RTL sweep, whole-kernel DM-core cycles)
#     rows\cols     64     128     256
#        1           -      834     926
#        2         3993    2048    2501
#        4         2292    2752    3658
#        8         3244    4159      -     ((1,64) is the warm-up config; (8,256) dropped: its
#                                           scratch overflows the L1 pool -> a fixed malloc spike)

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

from bingo_dfg import BingoDFG                            # noqa: E402
from bingo_helpers import chiplet_addr_transform_loc      # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode                          # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemFixedAddr  # noqa: E402
from bingo_kernel_args import (                           # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaSoftmaxF16F16Args,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE = 1
HOST_CORE = 2
CHECK_FP16_TOL = 2
# in/golden staged on the memchip (mempool.bin); see xdma_silu_1cluster.
MEMPOOL_LOC = (2, 0)
MEMPOOL_VADDR = 0x8000_0000
# NOTE: the fused FP16->INT8 softmax variant (__snax_bingo_kernel_xdma_softmax_f16_i8)
# is exercised by the standalone snax-xdma-softmax app. It is dropped from THIS CI
# sweep: keeping both the f16 and i8 check chains for all 12 configs needs 48 host
# nodes whose bingo scheduler metadata (~25 KiB) does not fit the 128 KiB L3 heap
# alongside the 88 KiB device binary. The f16 chain preserves the full rows x cols
# cost LUT.


# (rows, cols); each row is a length-cols tile (cols a multiple of 32). A rows x cols grid for the
# fused-kernel cost LUT (bilinear fit): rows {1,2,4,8} x cols {64,128,256} -- cols varies at EVERY
# row so the cols slope de-confounds from the rows==1 fast-path drop.
_LUT_GRID = [(r, c) for r in (1, 2, 4, 8) for c in (64, 128, 256)]
CONFIGS = [{"rows": r, "cols": c} for (r, c) in _LUT_GRID]


# ----- deterministic peaky softmax reference (mirrors the HW fp16 datapath) -----
def _softmax_ref(rows, cols, i):
    D = cols
    rng = np.random.RandomState(20240601 + i)
    x_rows, y_rows = [], []
    for r in range(rows):
        x = rng.uniform(-2.0, 2.0, size=D).astype(np.float32)
        for pos, val in ((0, 6.0), (D // 3, 5.0), (2 * D // 3, 4.5)):
            x[pos] = val
        x = x.astype(np.float16)
        xf = x.astype(np.float32)
        m = np.float16(xf.max())                                  # reduce(MAX) -> fp16
        xs = (xf - m.astype(np.float32)).astype(np.float16)       # ew(ADD): x + (-max), fp16
        e16 = np.exp(xs.astype(np.float32)).astype(np.float16)    # map(EXP), fp16
        s = np.float16(np.float32(e16.astype(np.float32).sum()))  # reduce(ADD) -> fp16
        inv16 = np.float16(np.float32(1.0) / s.astype(np.float32))  # host RECIP, fp16
        y = (e16.astype(np.float32) * inv16.astype(np.float32)).astype(np.float16)  # ew(MUL)
        x_rows.append(x)
        y_rows.append(y)
    return np.concatenate(x_rows), np.concatenate(y_rows)


# ----- tiny graph helpers -----
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


def build_mempool():
    """Concatenate every config's fp16 input + fp16 golden into the memchip image.

    Returns (blob: bytes, meta: [(off_in, off_golden), ...]); each array 64-B aligned.
    """
    blob = bytearray()
    meta = []

    def _pad():
        while len(blob) % 64:
            blob.append(0)

    for i in range(len(CONFIGS)):
        x, y = _softmax_ref(CONFIGS[i]["rows"], CONFIGS[i]["cols"], i)
        _pad(); oi = len(blob); blob += x.view(np.uint16).astype("<u2").tobytes()
        _pad(); og = len(blob); blob += y.view(np.uint16).astype("<u2").tobytes()
        meta.append((oi, og))
    return bytes(blob), meta


# One shared L1/L3 buffer set reused across all serialized configs (see xdma_silu_1cluster).
def build_config(g, i, base, meta, l1_x, l1_f16, l3_f, prev):
    rows  = CONFIGS[i]["rows"]
    D     = CONFIGS[i]["cols"]         # per-row length (cols)
    n     = rows * D                   # total elements
    tot_b = rows * D * 2               # [rows, D] fp16 bytes (packed)
    off_in, off_golden = meta[i]

    # One fused kernel runs the whole softmax on the DM core; the host only loads the
    # input (from the memchip), stores the fp16 output, and checks it.
    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(BingoMemFixedAddr(base + off_in), l1_x, tot_b), prev)
    # fp16 output: reduce-MAX, negate, sub-max, merged EXP+Sexp, integer reciprocal, normalize.
    sm_f = g.node(f"SoftmaxF16_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_softmax_f16_f16",
                  SnaxBingoKernelXdmaSoftmaxF16F16Args(l1_x, l1_f16, rows, D), load)
    st_f = g.node(f"StoreF16_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                  HostBingoKernelIdmaArgs(l1_f16, l3_f, tot_b), sm_f)
    ck_f = g.node(f"Check_softmax_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                  HostBingoKernelCheckResultArgs(BingoMemFixedAddr(base + off_golden), l3_f,
                      name=f"softmax_cfg{i}", check_type=CHECK_FP16_TOL,
                      num_elements=n, tolerance=0.02), st_f)
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

    blob, meta = build_mempool()
    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write("#include <stdint.h>\n")
        out_build = os.path.join(args.output_dir, "build")
        os.makedirs(out_build, exist_ok=True)
        with open(os.path.join(out_build, "mempool.bin"), "wb") as f:
            f.write(blob)
        print(f"Written data header: {args.data_h}")

    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": "softmax", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    # Single-chip workload: build the DFG for chip 0x00 only (see xdma_silu_1cluster).
    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[0x00])
    g = G(dfg)
    base = chiplet_addr_transform_loc(*MEMPOOL_LOC, MEMPOOL_VADDR)
    max_tot_b = max(r * c * 2 for (r, c) in _LUT_GRID)
    l1_x = g.l1("smx_x", max_tot_b)
    l1_f16 = g.l1("smx_f16", max_tot_b)
    l3_f = BingoMemAlloc("out_softmax_f16", size=max_tot_b, mem_level="L3")
    prev = None
    for i in range(len(CONFIGS)):
        prev = build_config(g, i, base, meta, l1_x, l1_f16, l3_f, prev)

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("xDMA softmax (fused)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["softmax_data.h"])
    print(f"Generated softmax: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
