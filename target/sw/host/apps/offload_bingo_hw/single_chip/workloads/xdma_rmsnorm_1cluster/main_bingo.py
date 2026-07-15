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
#   T1  reduce(SUMSQ, rows)               x         -> ssq_bt[rows]      [dev]
#   M1  map(LINEAR, a=1/N)                ssq_bt    -> mean_bt[rows]     [dev]
#   H1  sqrt       (GENERIC fp16 unary)   mean_bt   -> rms_bt[rows]      [host]
#   H2  reciprocal (GENERIC fp16 unary)   rms_bt    -> inv_bt[rows]      [host]
#   B1  xDMA broadcast (reader stride 0)  inv_bt    -> inv_bcast[rows,D] [dev]
#   T2  elementwise(MUL, {x, inv_bcast})  x         -> out[rows,D]       [dev]
#   Store(out) + Check(fp16 tol)                                          [host]
#   Tq  elementwise(MUL)+quant            x         -> out_i8            [dev]
#
# NO bespoke scalar-broadcast kernel. inv_rms = 1/sqrt(ssq/N) is assembled from ORDINARY ops:
# the /N is a generic xDMA StreamMap(LINEAR), and the sqrt + reciprocal are the ordinary host
# elementwise unaries at BINGO_PREC_FP16. That works because StreamReduce emits each row's
# scalar SPLATTED across all 32 lanes of its beat (StreamReduce.scala: "splatted across all
# lanes of one output beat"), so every step is a plain elementwise op over the rows*32 lanes
# and the result stays splatted -- the splat is free and no op needs to know about rows/beats.
#
# The [rows,D] operand still has to exist -- the StreamElementwise interleave is a single affine
# address stream and cannot hold inv_rms[r] fixed while x walks the row -- but the xDMA builds it
# (B1, reader stride 0), not the CVA6. The host therefore touches only rows*32 lanes, a cost
# independent of D; a host-side broadcast would be rows*D stores and dominate the makespan as D
# grows.
#
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
    SnaxBingoKernelXdma6dArgs,
    SnaxBingoKernelXdmaStreamReduceArgs,
    SnaxBingoKernelXdmaStreamMapArgs,
    SnaxBingoKernelXdmaStreamElementwiseArgs,
    HostBingoKernelAraSqrtF16Args,
    HostBingoKernelAraReciprocalF16Args,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE, HOST_CORE = 1, 2
R_SUMSQ = 2
EW_MUL = 0
F_LINEAR = 0       # StreamMap func: out = a*x + b
CHECK_FP16_TOL = 2
INV_SCALE = int(np.float32(64.0).view(np.uint32))
BEAT = 64          # xDMA bus beat in bytes
BEAT_FP16 = 32     # fp16 lanes per beat (StreamReduce splats the row scalar across all of them)

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


def bcast(g, name, src_beat, dst, beats, rows, dst_row_stride, after):
    """xDMA stride-0 BROADCAST: expand [rows] splatted scalar beats -> [rows, D].

    The reader's inner temporal dim has stride 0, so the row's single scalar beat is RE-READ
    `beats` times (the AGU's trip count comes from its own index counter, so a 0 step holds the
    address while the loop still runs); the outer dim steps one scalar beat per row.

    This is a plain strided copy -- no extension -- so the generic xdma_6d kernel expresses it.
    """
    return g.node(name, DMA_CORE, "__snax_bingo_kernel_xdma_6d",
                  SnaxBingoKernelXdma6dArgs(src_beat, dst, 8, 8,
                      [0, BEAT], [beats, rows],                # reader: stride-0 inner
                      [BEAT, dst_row_stride], [beats, rows]),  # writer: one row per scalar
                  after)


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
    row_b = beats * 64                 # row pitch of the [rows, D] operand
    lanes = rows * BEAT_FP16           # fp16 lanes in the `rows` splatted scalar beats
    inv_n = int(np.float32(1.0 / D).view(np.uint32))   # StreamMap `a` for the /N
    # Name prefixes 0x < 1ssq < 2mean < 3rms < 4inv < 5invbc < 6out < 7i8 => the allocator
    # places inv_bcast above x (the elementwise reader strides forward).
    l1_x     = g.l1(f"rms0x_{i}", tot_b)
    l1_ssqbt = g.l1(f"rms1ssqbt_{i}", rows * BEAT)   # reduce out: one SPLATTED beat per row
    l1_menbt = g.l1(f"rms2menbt_{i}", rows * BEAT)   # ssq/N
    l1_rmsbt = g.l1(f"rms3rmsbt_{i}", rows * BEAT)   # sqrt(ssq/N)
    l1_invbt = g.l1(f"rms4invbt_{i}", rows * BEAT)   # 1/sqrt(ssq/N)
    l1_invbc = g.l1(f"rms5invbc_{i}", tot_b)         # [rows, D] operand, built by the xDMA
    l1_out   = g.l1(f"rms6out_{i}", tot_b)
    l1_i8    = g.l1(f"rms7i8_{i}", n)
    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"in_{i}"), l1_x, tot_b), prev)
    t1 = g.node(f"SumSq_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_reduce",
                SnaxBingoKernelXdmaStreamReduceArgs(l1_x, l1_ssqbt, beats, op=R_SUMSQ,
                    rows=rows, csr_mode=0), load)
    # inv_rms = 1/sqrt(ssq/N), built WITHOUT any bespoke kernel. StreamReduce splats each row's
    # ssq across all 32 lanes of its beat, so every step below is a plain elementwise op over the
    # rows*32 lanes and the result stays splatted:
    #   M1  xDMA StreamMap(LINEAR, a=1/N)  -- the /N, on the device (generic map)
    #   H1  generic fp16 `sqrt`            -- host
    #   H2  generic fp16 `reciprocal`      -- host
    # (The two host ops are the ordinary elementwise unaries; they know nothing about rows/beats.)
    m1 = g.node(f"Mean_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map",
                SnaxBingoKernelXdmaStreamMapArgs(l1_ssqbt, l1_menbt, beats=rows, func=F_LINEAR,
                    a_f32bits=inv_n, b_f32bits=0, rows=1, csr_mode=0, dst_bound0=rows), t1)
    h1 = g.node(f"Sqrt_{i}", HOST_CORE, "__host_bingo_kernel_sqrt",
                HostBingoKernelAraSqrtF16Args(l1_menbt, l1_rmsbt, lanes), m1)
    h2 = g.node(f"Recip_{i}", HOST_CORE, "__host_bingo_kernel_reciprocal",
                HostBingoKernelAraReciprocalF16Args(l1_rmsbt, l1_invbt, lanes), h1)
    b1 = bcast(g, f"BcastInv_{i}", l1_invbt, l1_invbc, beats, rows, row_b, h2)
    t2 = g.node(f"Scale_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
                SnaxBingoKernelXdmaStreamElementwiseArgs(l1_x, l1_out, beats, op=EW_MUL,
                    operand_count=2, rows=rows, src_b_addr=l1_invbc, csr_mode=0,
                    dst_bound0=rows * beats), b1)
    l3_out = BingoMemAlloc(f"out_rmsnorm_{i}", size=tot_b, mem_level="L3")
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, tot_b), t2)
    chk = g.node(f"Check_rmsnorm_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_{i}"), l3_out,
                     name=f"rmsnorm_cfg{i}", check_type=CHECK_FP16_TOL, num_elements=n,
                     tolerance=0.03), store)
    g.node(f"Quant_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
           SnaxBingoKernelXdmaStreamElementwiseArgs(l1_x, l1_i8, beats, op=EW_MUL,
               operand_count=2, rows=rows, src_b_addr=l1_invbc, csr_mode=1,
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
