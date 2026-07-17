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

from data_utils import format_vector_definition          # noqa: E402
from bingo_dfg import BingoDFG                            # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode                          # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa: E402
from bingo_kernel_args import (                           # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaSoftmaxF16F16Args,
    SnaxBingoKernelXdmaSoftmaxF16I8Args,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE = 1
HOST_CORE = 2
CHECK_FP16_TOL = 2
CHECK_INT8_TOL = 4      # int8 output: +-1 LSB tol (softmax exp is a HW LUT != np.exp, so a
                        # quantized golden differs by 1 LSB at rounding boundaries -- byte-exact
                        # would spuriously fail and abort the sweep; +-1 is the correct budget)


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


def gen_data(i):
    x, y = _softmax_ref(CONFIGS[i]["rows"], CONFIGS[i]["cols"], i)
    # int8 golden = quantize(softmax) at the kernel's baked 127.0 scale (softmax out in [0,1]).
    yq = np.clip(np.rint(y.astype(np.float32) * 127.0), -128, 127).astype(np.int8)
    return [format_vector_definition("uint16_t", f"in_{i}", x.view(np.uint16)),
            format_vector_definition("uint16_t", f"golden_{i}", y.view(np.uint16)),
            format_vector_definition("int8_t", f"golden_i8_{i}", yq)]


def build_config(g, i, prev):
    rows  = CONFIGS[i]["rows"]
    D     = CONFIGS[i]["cols"]         # per-row length (cols)
    n     = rows * D                   # total elements
    tot_b = rows * D * 2               # [rows, D] fp16 bytes (packed)

    # One fused kernel runs the whole softmax on the DM core; the host only loads the input,
    # stores the output, and checks it. The precision is picked by KERNEL NAME (f16_f16 /
    # f16_i8), not an arg; both take the same {input, output, rows, cols}. We exercise BOTH
    # off the same loaded input.
    l1_x   = g.l1(f"smx0x_{i}", tot_b)     # [rows,D] packed input
    l1_f16 = g.l1(f"smx1f16_{i}", tot_b)   # [rows,D] packed fp16 softmax output
    l1_i8  = g.l1(f"smx2i8_{i}", n)        # [rows,D] int8 softmax output

    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"in_{i}"), l1_x, tot_b), prev)
    # fp16 output: reduce-MAX, negate, sub-max, merged EXP+Sexp, integer reciprocal, normalize.
    sm_f = g.node(f"SoftmaxF16_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_softmax_f16_f16",
                  SnaxBingoKernelXdmaSoftmaxF16F16Args(l1_x, l1_f16, rows, D), load)
    l3_f = BingoMemAlloc(f"out_softmax_f16_{i}", size=tot_b, mem_level="L3")
    st_f = g.node(f"StoreF16_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                  HostBingoKernelIdmaArgs(l1_f16, l3_f, tot_b), sm_f)
    ck_f = g.node(f"Check_softmax_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                  HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_{i}"), l3_f,
                      name=f"softmax_cfg{i}", check_type=CHECK_FP16_TOL,
                      num_elements=n, tolerance=0.02), st_f)
    # int8 output: same pipeline, final pass fuses Fp16ToInt8. +-1 LSB check vs the quantized golden.
    sm_q = g.node(f"SoftmaxI8_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_softmax_f16_i8",
                  SnaxBingoKernelXdmaSoftmaxF16I8Args(l1_x, l1_i8, rows, D), ck_f)
    l3_q = BingoMemAlloc(f"out_softmax_i8_{i}", size=n, mem_level="L3")
    st_q = g.node(f"StoreI8_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                  HostBingoKernelIdmaArgs(l1_i8, l3_q, n), sm_q)
    ck_q = g.node(f"Check_softmax_i8_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                  HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_i8_{i}"), l3_q,
                      name=f"softmax_i8_cfg{i}", check_type=CHECK_INT8_TOL,
                      num_elements=n, tolerance=1.0), st_q)
    return ck_q


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

    if args.data_h is not None:
        defs = ["#include <stdint.h>"]
        for i in range(len(CONFIGS)):
            defs += gen_data(i)
        with open(args.data_h, "w") as f:
            f.write("\n\n".join(defs))
        print(f"Written data header: {args.data_h}")

    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": "softmax", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    dfg = BingoDFG(num_chiplets=platform["num_chiplets"],
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=platform["chiplet_ids"])
    g = G(dfg)
    prev = None
    for i in range(len(CONFIGS)):
        prev = build_config(g, i, prev)

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("xDMA softmax (fused)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["softmax_data.h"])
    print(f"Generated softmax: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
