# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xDMA FP16 SOFTMAX — multi-row, explicit workload graph (no xdma_ops_lib).
# Per-row device->host->device hand-off over [rows, D] tiles: the device computes
# the per-row reductions on the xDMA, the CVA6 host computes & broadcasts the
# per-row scalars (-max, 1/Sigma exp), and the device's StreamElementwise reads the
# [rows, D] broadcast buffers via the src_b mechanism.
#
#   Load x[rows,D] (iDMA L3->L1)
#   T1  reduce(MAX, rows)                  x          -> max[rows]          [dev]
#   H1  host bcast(NEG)                    max[rows]  -> negmax_bc[rows,D]   [host]
#   T2  elementwise(ADD, {x, negmax_bc})   x          -> xs[rows,D]         [dev]
#   T3  map(EXP, a=1, b=0)                 xs         -> exp[rows,D]        [dev]
#   T4  reduce(ADD, rows)                  exp        -> sum[rows]          [dev]
#   H2  host bcast(RECIP)                  sum[rows]  -> recip_bc[rows,D]    [host]
#   T5  elementwise(MUL, {exp, recip_bc})  exp        -> out[rows,D]        [dev]
#   Store(out) + Check(fp16 tol)                                            [host]
#   Tq  elementwise(MUL)+Fp16ToInt8        exp        -> out_i8            [dev]
# Buffers are NAME-prefixed so each elementwise src_b operand allocates above its
# operand 0 (the reader strides forward).

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
    SnaxBingoKernelXdmaStreamElementwiseArgs,
    HostBingoKernelHostScalarBcastArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_HOST_SCALAR_NEG, BINGO_HOST_SCALAR_RECIP,
)

DMA_CORE = 1
HOST_CORE = 2
# StreamMap func / StreamReduce op / StreamElementwise op encodings.
F_EXP = 1
R_MAX, R_ADD = 0, 1
EW_MUL, EW_ADD = 0, 1
CHECK_FP16_TOL = 2
ONE_F32 = int(np.float32(1.0).view(np.uint32))
INV_SCALE = int(np.float32(127.0).view(np.uint32))

# (rows, beats); D = beats*32 = per-row length, total elements = rows*D.
CONFIGS = [{"rows": 1, "beats": 2}, {"rows": 1, "beats": 8},
           {"rows": 4, "beats": 2}, {"rows": 8, "beats": 2}]


# ----- deterministic peaky softmax reference (mirrors the HW fp16 datapath) -----
def _softmax_ref(rows, beats, i):
    D = beats * 32
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


# ----- tiny graph helpers (explicit graph; no xdma_ops_lib) -----
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
    x, y = _softmax_ref(CONFIGS[i]["rows"], CONFIGS[i]["beats"], i)
    return [format_vector_definition("uint16_t", f"in_{i}", x.view(np.uint16)),
            format_vector_definition("uint16_t", f"golden_{i}", y.view(np.uint16))]


def build_config(g, i, prev):
    rows  = CONFIGS[i]["rows"]
    beats = CONFIGS[i]["beats"]
    D     = beats * 32                 # per-row length
    n     = rows * D                   # total elements
    tot_b = rows * beats * 64          # [rows, D] fp16 bytes
    # Name prefixes 0x<1max<2negb<3xs<4expb<5sum<6recipb<7out<8i8 order the allocator
    # so each elementwise src_b operand (negb, recipb) sits above its operand 0.
    l1_x      = g.l1(f"smx0x_{i}", tot_b)
    l1_max    = g.l1(f"smx1max_{i}", rows * 64)
    l1_negb   = g.l1(f"smx2negb_{i}", tot_b)
    l1_xs     = g.l1(f"smx3xs_{i}", tot_b)
    l1_expb   = g.l1(f"smx4expb_{i}", tot_b)
    l1_sum    = g.l1(f"smx5sum_{i}", rows * 64)
    l1_recipb = g.l1(f"smx6recipb_{i}", tot_b)
    l1_out    = g.l1(f"smx7out_{i}", tot_b)
    l1_i8     = g.l1(f"smx8i8_{i}", n)

    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"in_{i}"), l1_x, tot_b), prev)
    t1 = g.node(f"Max_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_reduce",
                SnaxBingoKernelXdmaStreamReduceArgs(l1_x, l1_max, beats, op=R_MAX,
                    rows=rows, csr_mode=0), load)
    h1 = g.node(f"HostNeg_{i}", HOST_CORE, "__host_bingo_kernel_host_scalar_bcast",
                HostBingoKernelHostScalarBcastArgs(l1_max, l1_negb,
                    op=BINGO_HOST_SCALAR_NEG, rows=rows, D=D), t1)
    t2 = g.node(f"SubMax_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
                SnaxBingoKernelXdmaStreamElementwiseArgs(l1_x, l1_xs, beats, op=EW_ADD,
                    operand_count=2, rows=rows, src_b_addr=l1_negb, csr_mode=0,
                    dst_bound0=rows * beats), h1)
    t3 = g.node(f"Exp_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map",
                SnaxBingoKernelXdmaStreamMapArgs(l1_xs, l1_expb, beats, func=F_EXP,
                    a_f32bits=ONE_F32, b_f32bits=0, rows=rows, csr_mode=0,
                    dst_bound0=rows * beats), t2)
    t4 = g.node(f"Sum_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_reduce",
                SnaxBingoKernelXdmaStreamReduceArgs(l1_expb, l1_sum, beats, op=R_ADD,
                    rows=rows, csr_mode=0), t3)
    h2 = g.node(f"HostRecip_{i}", HOST_CORE, "__host_bingo_kernel_host_scalar_bcast",
                HostBingoKernelHostScalarBcastArgs(l1_sum, l1_recipb,
                    op=BINGO_HOST_SCALAR_RECIP, rows=rows, D=D), t4)
    t5 = g.node(f"Norm_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
                SnaxBingoKernelXdmaStreamElementwiseArgs(l1_expb, l1_out, beats, op=EW_MUL,
                    operand_count=2, rows=rows, src_b_addr=l1_recipb, csr_mode=0,
                    dst_bound0=rows * beats), h2)
    # Store + tolerant fp16 check of the softmax output.
    l3_out = BingoMemAlloc(f"out_softmax_{i}", size=tot_b, mem_level="L3")
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, tot_b), t5)
    chk = g.node(f"Check_softmax_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_{i}"), l3_out,
                     name=f"softmax_cfg{i}", check_type=CHECK_FP16_TOL,
                     num_elements=n, tolerance=0.02), store)
    # Tq fused FP16->INT8 quant — exercises the path (leaf, no numeric check).
    g.node(f"Quant_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
           SnaxBingoKernelXdmaStreamElementwiseArgs(l1_expb, l1_i8, beats, op=EW_MUL,
               operand_count=2, rows=rows, src_b_addr=l1_recipb, csr_mode=1,
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
    dfg.bingo_compile_dfg("xDMA softmax (explicit + host scalars)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["softmax_data.h"])
    print(f"Generated softmax: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
