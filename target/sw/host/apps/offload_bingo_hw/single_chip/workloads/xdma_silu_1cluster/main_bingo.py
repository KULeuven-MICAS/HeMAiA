# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xDMA FP16 SiLU — explicit, self-contained workload graph (no xdma_ops_lib).
# Fully on-device (unary, no host scalar). Element-wise, so multi-row is just a
# larger flat outer bound rows*beats:
#   Load x[rows,D] -> T1 map(SILU, a=1, b=0) -> Store+Check(fp16) -> Tq map(SILU)+quant
#
# Silu IS one StreamMap pass, so this workload's fp16 span measures the shared xdma_streammap LUT.
# Measured cost
#     n         512    1024    2048    4096
#     cycles    420     544     800    1316      (n=256 is the warm-up config)

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
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE, HOST_CORE = 1, 2
F_SILU = 2
CHECK_FP16_TOL = 2
ONE_F32 = int(np.float32(1.0).view(np.uint32))
INV_SCALE = int(np.float32(16.0).view(np.uint32))

# n-linear sweep for the xdma_streammap cost LUT (op_fit "linear", keyed on element count
# n = rows*cols). Silu lowers to ONE StreamMap pass, so this workload's fp16 span IS an
# xdma_streammap measurement. The per-pass config intercept is rows-independent (one xDMA launch
# configures all rows via the ND descriptor's row loop), so we FIX rows past the rows==1 fast path
# and sweep cols to move n cleanly -> a clean intercept+slope. (cols = beats*32, D per row.)
_STREAMMAP_ROWS = 4
_STREAMMAP_COLS = [64, 128, 256, 512, 1024]           # n = 256, 512, 1024, 2048, 4096
CONFIGS = [{"rows": _STREAMMAP_ROWS, "beats": c // 32} for c in _STREAMMAP_COLS]


def _silu_ref(rows, beats, i):
    n = rows * beats * 32
    rng = np.random.RandomState(3200 + i)
    x = rng.uniform(-8.0, 8.0, size=n).astype(np.float16)
    xf = x.astype(np.float32)
    y = (xf / (1.0 + np.exp(-xf))).astype(np.float16)
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
    x, y = _silu_ref(CONFIGS[i]["rows"], CONFIGS[i]["beats"], i)
    return [format_vector_definition("uint16_t", f"in_{i}", x.view(np.uint16)),
            format_vector_definition("uint16_t", f"golden_{i}", y.view(np.uint16))]


def build_config(g, i, prev):
    rows  = CONFIGS[i]["rows"]
    beats = CONFIGS[i]["beats"]
    n     = rows * beats * 32          # total elements
    tot_b = rows * beats * 64          # [rows, D] fp16 bytes
    l1_x   = g.l1(f"silu_x_{i}", tot_b)
    l1_out = g.l1(f"silu_out_{i}", tot_b)
    l1_i8  = g.l1(f"silu_i8_{i}", n)
    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"in_{i}"), l1_x, tot_b), prev)
    t1 = g.node(f"Silu_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map",
                SnaxBingoKernelXdmaStreamMapArgs(l1_x, l1_out, beats, func=F_SILU,
                    a_f32bits=ONE_F32, b_f32bits=0, rows=rows, csr_mode=0,
                    dst_bound0=rows * beats), load)
    l3_out = BingoMemAlloc(f"out_silu_{i}", size=tot_b, mem_level="L3")
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, tot_b), t1)
    chk = g.node(f"Check_silu_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_{i}"), l3_out,
                     name=f"silu_cfg{i}", check_type=CHECK_FP16_TOL, num_elements=n,
                     tolerance=0.05), store)
    g.node(f"Quant_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map",
           SnaxBingoKernelXdmaStreamMapArgs(l1_x, l1_i8, beats, func=F_SILU,
               a_f32bits=ONE_F32, b_f32bits=0, rows=rows, csr_mode=1,
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
            json.dump({"op": "silu", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)
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
    dfg.bingo_compile_dfg("xDMA silu (explicit)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["silu_data.h"])
    print(f"Generated silu: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
