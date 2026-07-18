# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xDMA FP16 RoPE — single fused kernel (__snax_bingo_kernel_xdma_rope).
# RoPE is all DMA-engine ops, so the whole flow is ONE offload node:
#   inputs x, cos_full, sin_signed (precomputed tables); xswap computed ON-DEVICE.
#   xswap = adjacent fp16-pair swap of x          (iDMA, inside the kernel)
#   P1 x*cos -> tmp1 ; P2 xswap*sin -> tmp2 ; P3 tmp1+tmp2 -> out  (xDMA, in-kernel)
#   Store(out) + Check(fp16 tol)
# The kernel allocates its own xswap/tmp1/tmp2 scratch from L1, so the DFG only
# loads x/cos/sin and provides out. xswap being derived inside the kernel rather
# than supplied as an input is what lets rope_q/rope_k run in-layer, on an x that
# only exists at run time.
#
# Measured cost LUT (single-chip RTL sweep, whole-kernel DM-core cycles)
#     rows\cols     64     128     256
#        1           -     1227    1675
#        2         1227    1674    2574
#        4         1674    2573    4367
#        8         2573    4366      -     ((1,64) is the warm-up config; (8,256) dropped: the
#                                           xswap/tmp1/tmp2 scratch overflows the L1 pool -> spike)

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
    SnaxBingoKernelXdmaRopeArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE, HOST_CORE = 1, 2
CHECK_FP16_TOL = 2
ROPE_BASE = 10000.0
ROPE_POS = 1
# x/cos/sin/golden staged on the memchip (mempool.bin); see xdma_silu_1cluster.
MEMPOOL_LOC = (2, 0)
MEMPOOL_VADDR = 0x8000_0000

# (rows, cols); D = cols. Each row is a distinct token position (ROPE_POS + r), so cos/sin/xswap
# are per-row [rows, D] tables. A rows x cols grid for the fused-kernel cost LUT (bilinear, keyed
# [rows, cols]): rows {1,2,4,8} x cols {64,128,256}, so the bilinear cols slope de-confounds from
# the rows==1 fast-path drop.
# cols {64,256} (not {64,128,256}): rope loads 3 inputs/config, so its generated
# scheduler code is the largest of the SIMD sweeps and the cols=128 column is
# dropped to keep the L3 heap comfortable. rows and both cols extremes are kept, so
# the bilinear rows/cols cost-LUT fit is preserved.
_LUT_GRID = [(r, c) for r in (1, 2, 4, 8) for c in (64, 256)]
CONFIGS = [{"rows": r, "cols": c} for (r, c) in _LUT_GRID]


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


def _rope_ref(rows, cols, i):
    D = cols
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


def build_mempool():
    """Concatenate every config's fp16 x + cos + sin + golden into the memchip image.

    xswap is computed on-device by the kernel, so it is NOT staged. Returns
    (blob: bytes, meta: [(off_x, off_cos, off_sin, off_golden), ...]); each array
    is 64-B aligned.
    """
    blob = bytearray()
    meta = []

    def _pad():
        while len(blob) % 64:
            blob.append(0)

    for i in range(len(CONFIGS)):
        x, cos_full, _xswap, sin_signed, out = _rope_ref(CONFIGS[i]["rows"], CONFIGS[i]["cols"], i)
        _pad(); ox = len(blob); blob += x.view(np.uint16).astype("<u2").tobytes()
        _pad(); oc = len(blob); blob += cos_full.view(np.uint16).astype("<u2").tobytes()
        _pad(); os_ = len(blob); blob += sin_signed.view(np.uint16).astype("<u2").tobytes()
        _pad(); ol = len(blob); blob += out.view(np.uint16).astype("<u2").tobytes()
        meta.append((ox, oc, os_, ol))
    return bytes(blob), meta


# Shared L1/L3 buffers reused across all serialized configs (see xdma_silu_1cluster).
def build_config(g, i, base, meta, l1_x, l1_cos, l1_sin, l1_out, l3_out, prev):
    rows  = CONFIGS[i]["rows"]
    D     = CONFIGS[i]["cols"]         # per-row fp16 length (cols)
    n     = rows * D                   # total fp16 elements
    tot_b = rows * D * 2               # [rows, D] fp16 bytes
    off_x, off_cos, off_sin, off_golden = meta[i]
    lx = g.node(f"LoadX_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(BingoMemFixedAddr(base + off_x), l1_x, tot_b), prev)
    lc = g.node(f"LoadCos_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(BingoMemFixedAddr(base + off_cos), l1_cos, tot_b), lx)
    ls = g.node(f"LoadSin_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(BingoMemFixedAddr(base + off_sin), l1_sin, tot_b), lc)
    rope = g.node(f"Rope_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_rope",
                  SnaxBingoKernelXdmaRopeArgs(l1_x, l1_cos, l1_sin, l1_out, D, rows), ls)
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, tot_b), rope)
    chk = g.node(f"Check_rope_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemFixedAddr(base + off_golden), l3_out,
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
    blob, meta = build_mempool()
    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write("#include <stdint.h>\n")
        out_build = os.path.join(args.output_dir, "build")
        os.makedirs(out_build, exist_ok=True)
        with open(os.path.join(out_build, "mempool.bin"), "wb") as f:
            f.write(blob)
    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": "rope", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)
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
    l1_x = g.l1("rp_x", max_tot_b)
    l1_cos = g.l1("rp_cos", max_tot_b)
    l1_sin = g.l1("rp_sin", max_tot_b)
    l1_out = g.l1("rp_out", max_tot_b)
    l3_out = BingoMemAlloc("out_rope", size=max_tot_b, mem_level="L3")
    prev = None
    for i in range(len(CONFIGS)):
        prev = build_config(g, i, base, meta, l1_x, l1_cos, l1_sin, l1_out, l3_out, prev)
    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("xDMA rope (explicit)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["rope_data.h"])
    print(f"Generated rope: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
