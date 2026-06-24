# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xDMA FP16 RMSNorm — multi-row, explicit workload graph (no xdma_ops_lib).
# Per-row device->host->device hand-off over [rows, D] tiles (host computes the
# per-row inv_rms = 1/sqrt(Sigma x^2 / N) and broadcasts it):
#   Load x[rows,D]
#   T1  reduce(SUMSQ, rows)               x         -> ssq[rows]         [dev]
#   H1  host bcast(RSQRT_MEAN, N=D)       ssq[rows] -> inv_bcast[rows,D]  [host]
#   T2  elementwise(MUL, {x, inv_bcast})  x         -> out[rows,D]       [dev]
#   Store(out) + Check(fp16 tol)                                          [host]
#   Tq  elementwise(MUL)+quant            x         -> out_i8            [dev]
# The MUL reads x and inv_bcast as two SEPARATE L1 buffers via the stream_elementwise
# src_b mechanism; buffers are NAME-prefixed so inv_bcast allocates above x.

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
    SnaxBingoKernelXdmaStreamElementwiseArgs,
    HostBingoKernelHostScalarBcastArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_HOST_SCALAR_RSQRT_MEAN,
)

DMA_CORE, HOST_CORE = 1, 2
R_SUMSQ = 2
EW_MUL = 0
CHECK_FP16_TOL = 2
INV_SCALE = int(np.float32(64.0).view(np.uint32))

# (rows, beats); D = beats*32 = per-row reduction length, total elements = rows*D.
CONFIGS = [{"rows": 1, "beats": 2}, {"rows": 1, "beats": 8},
           {"rows": 4, "beats": 2}, {"rows": 8, "beats": 2}]


def _rmsnorm_ref(rows, beats, i):
    D = beats * 32
    rng = np.random.RandomState(3300 + i)
    x_rows, y_rows = [], []
    for r in range(rows):
        x = rng.uniform(-4.0, 4.0, size=D).astype(np.float16)
        xf = x.astype(np.float32)
        ssq = np.float16(np.float32((xf ** 2).sum()))      # HW: fp32 accumulate -> fp16 scalar
        inv = np.float32(1.0) / np.float32(np.sqrt(ssq.astype(np.float32) / np.float32(D)))
        inv16 = np.float16(inv)                            # host narrows to fp16 for the broadcast
        y = (xf * inv16.astype(np.float32)).astype(np.float16)  # fp16 elementwise MUL
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


def gen_data(i):
    x, y = _rmsnorm_ref(CONFIGS[i]["rows"], CONFIGS[i]["beats"], i)
    return [format_vector_definition("uint16_t", f"in_{i}", x.view(np.uint16)),
            format_vector_definition("uint16_t", f"golden_{i}", y.view(np.uint16))]


def build_config(g, i, prev):
    rows  = CONFIGS[i]["rows"]
    beats = CONFIGS[i]["beats"]
    D     = beats * 32                 # per-row width / reduction length
    n     = rows * D                   # total elements
    tot_b = rows * beats * 64          # [rows, D] fp16 bytes
    # Name prefixes 0x < 1ssq < 2invb < 3out < 4i8 => allocator places inv_bcast above x.
    l1_x    = g.l1(f"rms0x_{i}", tot_b)
    l1_ssq  = g.l1(f"rms1ssq_{i}", rows * 64)   # one splatted 64-B beat per row
    l1_invb = g.l1(f"rms2invb_{i}", tot_b)      # [rows, D] fp16 broadcast
    l1_out  = g.l1(f"rms3out_{i}", tot_b)
    l1_i8   = g.l1(f"rms4i8_{i}", n)
    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"in_{i}"), l1_x, tot_b), prev)
    t1 = g.node(f"SumSq_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_reduce",
                SnaxBingoKernelXdmaStreamReduceArgs(l1_x, l1_ssq, beats, op=R_SUMSQ,
                    rows=rows, csr_mode=0), load)
    h1 = g.node(f"HostRsqrt_{i}", HOST_CORE, "__host_bingo_kernel_host_scalar_bcast",
                HostBingoKernelHostScalarBcastArgs(l1_ssq, l1_invb,
                    op=BINGO_HOST_SCALAR_RSQRT_MEAN, rows=rows, D=D, N=D), t1)
    t2 = g.node(f"Scale_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
                SnaxBingoKernelXdmaStreamElementwiseArgs(l1_x, l1_out, beats, op=EW_MUL,
                    operand_count=2, rows=rows, src_b_addr=l1_invb, csr_mode=0,
                    dst_bound0=rows * beats), h1)
    l3_out = BingoMemAlloc(f"out_rmsnorm_{i}", size=tot_b, mem_level="L3")
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, tot_b), t2)
    chk = g.node(f"Check_rmsnorm_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_{i}"), l3_out,
                     name=f"rmsnorm_cfg{i}", check_type=CHECK_FP16_TOL, num_elements=n,
                     tolerance=0.03), store)
    g.node(f"Quant_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
           SnaxBingoKernelXdmaStreamElementwiseArgs(l1_x, l1_i8, beats, op=EW_MUL,
               operand_count=2, rows=rows, src_b_addr=l1_invb, csr_mode=1,
               dst_bound0=rows * beats // 2, out_dtype=1, inv_scale_f32bits=INV_SCALE), chk)
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
