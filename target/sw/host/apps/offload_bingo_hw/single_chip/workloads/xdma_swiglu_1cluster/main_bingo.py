# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xDMA FP16 SwiGLU — explicit, self-contained workload graph (no xdma_ops_lib).
# Fully on-device (no host scalar):
#   Load gate, up
#   T1  map(SILU)                         gate        -> sg              [dev]
#   T2  elementwise(MUL, {sg, up})        STICKY?FULL -> out             [dev]
#   Store(out) + Check(fp16 tol)
#   Tq  elementwise(MUL, {sg, up})+quant  STICKY      -> out_i8
# The MUL reads two SEPARATE L1 buffers (sg, up) via the stream_elementwise
# src_b mechanism. Buffers are NAME-prefixed so the allocator (name-sorted) puts
# `up` at a higher address than `sg` (the reader strides forward sg -> up).

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
    SnaxBingoKernelXdmaStreamMapArgs,
    SnaxBingoKernelXdmaStreamElementwiseArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE, HOST_CORE = 1, 2
F_SILU = 2
EW_MUL = 0
CHECK_FP16_TOL = 2
ONE_F32 = int(np.float32(1.0).view(np.uint32))
INV_SCALE = int(np.float32(16.0).view(np.uint32))

CONFIGS = [{"beats": b} for b in (2, 8, 32, 128)]


def _swiglu_ref(beats, i):
    n = beats * 32
    rng = np.random.RandomState(3400 + i)
    gate = rng.uniform(-8.0, 8.0, size=n).astype(np.float16)
    up = rng.uniform(-2.0, 2.0, size=n).astype(np.float16)
    g32 = gate.astype(np.float32)
    sg = (g32 / (1.0 + np.exp(-g32))).astype(np.float16)
    out = (sg.astype(np.float32) * up.astype(np.float32)).astype(np.float16)
    return gate, up, out


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
    gate, up, out = _swiglu_ref(CONFIGS[i]["beats"], i)
    return [format_vector_definition("uint16_t", f"gate_{i}", gate.view(np.uint16)),
            format_vector_definition("uint16_t", f"up_{i}", up.view(np.uint16)),
            format_vector_definition("uint16_t", f"golden_{i}", out.view(np.uint16))]


def build_config(g, i, prev):
    beats = CONFIGS[i]["beats"]
    n = beats * 32
    row_b = beats * 64
    # Name prefixes 0gate < 1sg < 2up < 3out < 4i8 => allocator places up after sg.
    l1_g   = g.l1(f"sw0gate_{i}", row_b)
    l1_sg  = g.l1(f"sw1sg_{i}", row_b)
    l1_up  = g.l1(f"sw2up_{i}", row_b)
    l1_out = g.l1(f"sw3out_{i}", row_b)
    l1_i8  = g.l1(f"sw4i8_{i}", n)
    lg = g.node(f"LoadGate_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"gate_{i}"), l1_g, row_b), prev)
    lu = g.node(f"LoadUp_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"up_{i}"), l1_up, row_b), lg)
    t1 = g.node(f"Silu_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map",
                SnaxBingoKernelXdmaStreamMapArgs(l1_g, l1_sg, beats, func=F_SILU,
                    a_f32bits=ONE_F32, b_f32bits=0, csr_mode=0, dst_bound0=beats), lu)
    t2 = g.node(f"Mul_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
                SnaxBingoKernelXdmaStreamElementwiseArgs(l1_sg, l1_out, beats, op=EW_MUL,
                    operand_count=2, src_b_addr=l1_up, csr_mode=0, dst_bound0=beats), t1)
    l3_out = BingoMemAlloc(f"out_swiglu_{i}", size=row_b, mem_level="L3")
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, row_b), t2)
    chk = g.node(f"Check_swiglu_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_{i}"), l3_out,
                     name=f"swiglu_cfg{i}", check_type=CHECK_FP16_TOL, num_elements=n,
                     tolerance=0.1), store)
    g.node(f"MulQuant_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
           SnaxBingoKernelXdmaStreamElementwiseArgs(l1_sg, l1_i8, beats, op=EW_MUL,
               operand_count=2, src_b_addr=l1_up, csr_mode=1, dst_bound0=beats // 2,
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
            json.dump({"op": "swiglu", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)
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
    dfg.bingo_compile_dfg("xDMA swiglu (explicit)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["swiglu_data.h"])
    print(f"Generated swiglu: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
