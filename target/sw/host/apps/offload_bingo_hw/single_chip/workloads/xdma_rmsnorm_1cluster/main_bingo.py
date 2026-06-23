# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xDMA FP16 RMSNorm — explicit, self-contained workload graph (no xdma_ops_lib).
# Real device->host->device hand-off (host computes inv_rms = 1/sqrt(Sigma x^2 / N)):
#   Load x
#   T1  reduce(SUMSQ)                FULL    x       -> ssq          [dev]
#   H1  host RSQRT_MEAN(n=N)                 ssq     -> inv_rms(f32) [host]
#   T2  map(LINEAR, a=inv_rms)       STICKY  x       -> out          [dev]
#   Store(out) + Check(fp16 tol)                                     [host]
#   Tq  map(LINEAR, a=inv_rms)+quant STICKY  x       -> out_i8       [dev]

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
    SnaxBingoKernelXdmaStreamReduceArgs,
    SnaxBingoKernelXdmaStreamMapArgs,
    HostBingoKernelHostScalarArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_HOST_SCALAR_RSQRT_MEAN,
)

DMA_CORE, HOST_CORE = 1, 2
F_LINEAR = 0
R_SUMSQ = 2
CHECK_FP16_TOL = 2
INV_SCALE = int(np.float32(64.0).view(np.uint32))

CONFIGS = [{"beats": b} for b in (2, 8, 32, 128)]  # N = beats*32 (power of two)


def _rmsnorm_ref(beats, i):
    n = beats * 32
    rng = np.random.RandomState(3300 + i)
    x = rng.uniform(-4.0, 4.0, size=n).astype(np.float16)
    xf = x.astype(np.float32)
    ssq = np.float16(np.float32((xf ** 2).sum()))         # HW: fp32 accumulate -> fp16
    inv_rms = np.float32(1.0) / np.float32(np.sqrt(ssq.astype(np.float32) / np.float32(n)))
    y = (xf * inv_rms).astype(np.float16)
    return x, y


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
    x, y = _rmsnorm_ref(CONFIGS[i]["beats"], i)
    return [format_vector_definition("uint16_t", f"in_{i}", x.view(np.uint16)),
            format_vector_definition("uint16_t", f"golden_{i}", y.view(np.uint16))]


def build_config(g, i, prev):
    beats = CONFIGS[i]["beats"]
    n = beats * 32
    row_b = beats * 64
    l1_x      = g.l1(f"rms_x_{i}", row_b)
    l1_ssq    = g.l1(f"rms_ssq_{i}", 64)
    l1_invrms = g.l1(f"rms_invrms_{i}", 8)
    l1_out    = g.l1(f"rms_out_{i}", row_b)
    l1_i8     = g.l1(f"rms_i8_{i}", n)
    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"in_{i}"), l1_x, row_b), prev)
    t1 = g.node(f"SumSq_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_reduce",
                SnaxBingoKernelXdmaStreamReduceArgs(l1_x, l1_ssq, beats, op=R_SUMSQ,
                    red_tap=0, csr_mode=0, dst_bound0=1), load)
    h1 = g.node(f"HostRsqrt_{i}", HOST_CORE, "__host_bingo_kernel_host_scalar",
                HostBingoKernelHostScalarArgs(l1_ssq, l1_invrms,
                    op=BINGO_HOST_SCALAR_RSQRT_MEAN, n=n), t1)
    t2 = g.node(f"Scale_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map",
                SnaxBingoKernelXdmaStreamMapArgs(l1_x, l1_out, beats, func=F_LINEAR,
                    a_addr=l1_invrms, csr_mode=1, dst_bound0=beats), h1)
    l3_out = BingoMemAlloc(f"out_rmsnorm_{i}", size=row_b, mem_level="L3")
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, row_b), t2)
    chk = g.node(f"Check_rmsnorm_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_{i}"), l3_out,
                     name=f"rmsnorm_cfg{i}", check_type=CHECK_FP16_TOL, num_elements=n,
                     tolerance=0.03), store)
    g.node(f"Quant_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map",
           SnaxBingoKernelXdmaStreamMapArgs(l1_x, l1_i8, beats, func=F_LINEAR,
               a_addr=l1_invrms, csr_mode=1, dst_bound0=beats // 2,
               out_dtype=1, inv_scale_f32bits=INV_SCALE), chk)
    return chk


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
    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": "rmsnorm", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)
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
    dfg.bingo_compile_dfg("xDMA rmsnorm (explicit + host rsqrt)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["rmsnorm_data.h"])
    print(f"Generated rmsnorm: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
