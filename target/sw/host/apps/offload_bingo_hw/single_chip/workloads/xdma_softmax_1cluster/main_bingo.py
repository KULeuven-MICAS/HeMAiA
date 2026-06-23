# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xDMA FP16 SOFTMAX — explicit, self-contained workload graph (no xdma_ops_lib).
# Real device->host->device hand-off: the device computes the reductions on the
# xDMA, the CVA6 host computes the scalars (-max, 1/Sigma exp) from cluster L1,
# and the device's StreamMap reads them back from L1 (a_addr/b_addr).
#
#   Load x (iDMA L3->L1)
#   T1  reduce(MAX)                       FULL   x        -> max          [dev]
#   H1  host NEG                                 max      -> neg_max(f32) [host]
#   T2  map(EXP, b=neg_max) + reduce(ADD|TAP) STICKY x   -> exp + Sigma   [dev]
#   H2  host RECIP                               Sigma    -> inv_sum(f32) [host]
#   T3  map(LINEAR, a=inv_sum)            STICKY exp       -> out          [dev]
#   Store(out) + Check(fp16 tol)                                          [host]
#   Tq  map(LINEAR, a=inv_sum)+Fp16ToInt8 STICKY exp      -> out_i8       [dev]

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
    BINGO_HOST_SCALAR_NEG, BINGO_HOST_SCALAR_RECIP,
)

DMA_CORE = 1
HOST_CORE = 2
# StreamMap func / StreamReduce op encodings.
F_LINEAR, F_EXP = 0, 1
R_MAX, R_ADD, RED_TAP = 0, 1, 0x100
CHECK_FP16_TOL = 2

CONFIGS = [{"beats": b} for b in (2, 8, 32, 128)]  # N = beats*32 = 64/256/1024/4096


# ----- deterministic peaky softmax reference (mirrors the HW fp16 datapath) -----
def _softmax_ref(beats, i):
    n = beats * 32
    rng = np.random.RandomState(20240601 + i)
    x = rng.uniform(-2.0, 2.0, size=n).astype(np.float32)
    for pos, val in ((0, 6.0), (n // 3, 5.0), (2 * n // 3, 4.5)):
        x[pos] = val
    x = x.astype(np.float16)
    xf = x.astype(np.float32)
    m = np.float16(xf.max())
    e16 = np.exp(xf - np.float32(m)).astype(np.float16)
    s = np.float32(e16.astype(np.float32).sum())
    y = (e16.astype(np.float32) * (np.float32(1.0) / s)).astype(np.float16)
    return x, y


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
    x, y = _softmax_ref(CONFIGS[i]["beats"], i)
    return [format_vector_definition("uint16_t", f"in_{i}", x.view(np.uint16)),
            format_vector_definition("uint16_t", f"golden_{i}", y.view(np.uint16))]


def build_config(g, i, prev):
    beats = CONFIGS[i]["beats"]
    n = beats * 32
    row_b = beats * 64
    l1_x      = g.l1(f"smx_x_{i}", row_b)
    l1_max    = g.l1(f"smx_max_{i}", 64)
    l1_negmax = g.l1(f"smx_negmax_{i}", 8)
    l1_exp    = g.l1(f"smx_exp_{i}", (beats + 1) * 64)
    l1_invsum = g.l1(f"smx_invsum_{i}", 8)
    l1_out    = g.l1(f"smx_out_{i}", row_b)
    l1_i8     = g.l1(f"smx_i8_{i}", n)

    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"in_{i}"), l1_x, row_b), prev)
    t1 = g.node(f"Max_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_reduce",
                SnaxBingoKernelXdmaStreamReduceArgs(l1_x, l1_max, beats, op=R_MAX,
                    red_tap=0, csr_mode=0, dst_bound0=1), load)
    h1 = g.node(f"HostNeg_{i}", HOST_CORE, "__host_bingo_kernel_host_scalar",
                HostBingoKernelHostScalarArgs(l1_max, l1_negmax, op=BINGO_HOST_SCALAR_NEG), t1)
    t2 = g.node(f"ExpSum_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map",
                SnaxBingoKernelXdmaStreamMapArgs(l1_x, l1_exp, beats, func=F_EXP,
                    b_addr=l1_negmax, tap_reduce_op=(R_ADD | RED_TAP),
                    csr_mode=1, dst_bound0=beats + 1), h1)
    h2 = g.node(f"HostRecip_{i}", HOST_CORE, "__host_bingo_kernel_host_scalar",
                HostBingoKernelHostScalarArgs(l1_exp, l1_invsum, op=BINGO_HOST_SCALAR_RECIP,
                    in_offset=beats * 64), t2)
    t3 = g.node(f"Norm_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map",
                SnaxBingoKernelXdmaStreamMapArgs(l1_exp, l1_out, beats, func=F_LINEAR,
                    a_addr=l1_invsum, csr_mode=1, dst_bound0=beats), h2)
    # Store + tolerant fp16 check of the softmax output.
    l3_out = BingoMemAlloc(f"out_softmax_{i}", size=row_b, mem_level="L3")
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, row_b), t3)
    chk = g.node(f"Check_softmax_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_{i}"), l3_out,
                     name=f"softmax_cfg{i}", check_type=CHECK_FP16_TOL,
                     num_elements=n, tolerance=0.02), store)
    # Tq fused FP16->INT8 quant — exercises the path (leaf, no numeric check).
    g.node(f"Quant_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map",
           SnaxBingoKernelXdmaStreamMapArgs(l1_exp, l1_i8, beats, func=F_LINEAR,
               a_addr=l1_invsum, csr_mode=1, dst_bound0=beats // 2,
               out_dtype=1, inv_scale_f32bits=int(np.float32(127.0).view(np.uint32))), chk)
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
