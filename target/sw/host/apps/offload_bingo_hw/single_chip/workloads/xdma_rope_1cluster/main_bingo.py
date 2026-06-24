# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xDMA FP16 RoPE — explicit, self-contained workload graph (no xdma_ops_lib).
# Zero new HW: 3 StreamElementwise passes over per-pair-duplicated tables.
#   inputs x, cos_full, xswap (adjacent-halfword swap of x), sin_signed (precomputed)
#   P1  elementwise(MUL, {x, cos})        -> tmp1     [dev]
#   P2  elementwise(MUL, {xswap, sin})    -> tmp2     [dev]
#   P3  elementwise(ADD, {tmp1, tmp2})    -> out      [dev]
#   Store(out) + Check(fp16 tol)
# Each pass reads two SEPARATE L1 buffers via the stream_elementwise src_b
# mechanism; buffers are NAME-prefixed so operand 1 always allocates above
# operand 0 (the reader strides forward). Each pass is FULL (distinct stride).

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
    SnaxBingoKernelXdmaStreamElementwiseArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE, HOST_CORE = 1, 2
EW_MUL, EW_ADD = 0, 1
CHECK_FP16_TOL = 2
ROPE_BASE = 10000.0
ROPE_POS = 1

# (rows, beats); D = beats*32 per-row width. Each row is a distinct token position
# (ROPE_POS + r), so cos/sin/xswap are per-row [rows, D] tables.
CONFIGS = [{"rows": 1, "beats": 2}, {"rows": 1, "beats": 8},
           {"rows": 4, "beats": 2}, {"rows": 8, "beats": 2}]


def _rope_row(rng, D, pos):
    half = D // 2
    inv_freq = ROPE_BASE ** (-(np.arange(half).astype(np.float64)) / half)
    ang = pos * inv_freq
    c = np.cos(ang).astype(np.float16).astype(np.float32)
    s = np.sin(ang).astype(np.float16).astype(np.float32)
    x = rng.uniform(-4.0, 4.0, size=D).astype(np.float16)
    cos_full = np.repeat(c, 2).astype(np.float16)
    sin_signed = np.empty(D, np.float32)
    sin_signed[0::2] = -s
    sin_signed[1::2] = +s
    sin_signed = sin_signed.astype(np.float16)
    xu = x.view(np.uint16)
    xsw = np.empty(D, np.uint16)
    xsw[0::2] = xu[1::2]
    xsw[1::2] = xu[0::2]
    xswap = xsw.view(np.float16)
    tmp1 = (x.astype(np.float32) * cos_full.astype(np.float32)).astype(np.float16)
    tmp2 = (xswap.astype(np.float32) * sin_signed.astype(np.float32)).astype(np.float16)
    out = (tmp1.astype(np.float32) + tmp2.astype(np.float32)).astype(np.float16)
    return x, cos_full, xswap, sin_signed, out


def _rope_ref(rows, beats, i):
    D = beats * 32
    rng = np.random.RandomState(3500 + i)
    cols = [[], [], [], [], []]
    for r in range(rows):
        for k, v in enumerate(_rope_row(rng, D, ROPE_POS + r)):
            cols[k].append(v)
    return tuple(np.concatenate(c) for c in cols)


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
    x, cos_full, xswap, sin_signed, out = _rope_ref(CONFIGS[i]["rows"], CONFIGS[i]["beats"], i)
    return [format_vector_definition("uint16_t", f"x_{i}", x.view(np.uint16)),
            format_vector_definition("uint16_t", f"cos_{i}", cos_full.view(np.uint16)),
            format_vector_definition("uint16_t", f"xswap_{i}", xswap.view(np.uint16)),
            format_vector_definition("uint16_t", f"sin_{i}", sin_signed.view(np.uint16)),
            format_vector_definition("uint16_t", f"golden_{i}", out.view(np.uint16))]


def build_config(g, i, prev):
    rows  = CONFIGS[i]["rows"]
    beats = CONFIGS[i]["beats"]
    n     = rows * beats * 32          # total elements
    tot_b = rows * beats * 64          # [rows, D] fp16 bytes
    # Name prefixes ensure operand1 > operand0 per pair: 0x<1cos, 2xswap<3sin, 4tmp1<5tmp2.
    l1_x     = g.l1(f"rp0x_{i}", tot_b)
    l1_cos   = g.l1(f"rp1cos_{i}", tot_b)
    l1_xswap = g.l1(f"rp2xswap_{i}", tot_b)
    l1_sin   = g.l1(f"rp3sin_{i}", tot_b)
    l1_tmp1  = g.l1(f"rp4tmp1_{i}", tot_b)
    l1_tmp2  = g.l1(f"rp5tmp2_{i}", tot_b)
    l1_out   = g.l1(f"rp6out_{i}", tot_b)
    lx = g.node(f"LoadX_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"x_{i}"), l1_x, tot_b), prev)
    lc = g.node(f"LoadCos_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"cos_{i}"), l1_cos, tot_b), lx)
    lxs = g.node(f"LoadXswap_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                 SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"xswap_{i}"), l1_xswap, tot_b), lc)
    ls = g.node(f"LoadSin_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"sin_{i}"), l1_sin, tot_b), lxs)
    p1 = g.node(f"MulCos_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
                SnaxBingoKernelXdmaStreamElementwiseArgs(l1_x, l1_tmp1, beats, op=EW_MUL,
                    operand_count=2, rows=rows, src_b_addr=l1_cos, csr_mode=0,
                    dst_bound0=rows * beats), ls)
    p2 = g.node(f"MulSin_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
                SnaxBingoKernelXdmaStreamElementwiseArgs(l1_xswap, l1_tmp2, beats, op=EW_MUL,
                    operand_count=2, rows=rows, src_b_addr=l1_sin, csr_mode=0,
                    dst_bound0=rows * beats), p1)
    p3 = g.node(f"Add_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
                SnaxBingoKernelXdmaStreamElementwiseArgs(l1_tmp1, l1_out, beats, op=EW_ADD,
                    operand_count=2, rows=rows, src_b_addr=l1_tmp2, csr_mode=0,
                    dst_bound0=rows * beats), p2)
    l3_out = BingoMemAlloc(f"out_rope_{i}", size=tot_b, mem_level="L3")
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, tot_b), p3)
    chk = g.node(f"Check_rope_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_{i}"), l3_out,
                     name=f"rope_cfg{i}", check_type=CHECK_FP16_TOL, num_elements=n,
                     tolerance=0.05), store)
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
            json.dump({"op": "rope", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)
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
    dfg.bingo_compile_dfg("xDMA rope (explicit)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["rope_data.h"])
    print(f"Generated rope: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
