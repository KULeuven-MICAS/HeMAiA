# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# SIMD FP16 RoPE — TWO nodes, on the two engines that can each do their half.
#
#   Load x, cos_full, sin_signed into slots 0, 1, 3 of one 4-row operand block
#   Swap    slot 0 -> slot 2     (DM core, __snax_bingo_kernel_idma_pairwise_swap)
#   Rope    the whole rotation   (SIMD core, __snax_bingo_kernel_simd_rope, ONE task)
#   Store(out) + Check(fp16 tol)
#
# WHY TWO AND NOT ONE. The rotation is out = x*cos_full + xswap*sin_signed: two products
# and a sum, which is a combine on each side of the Map, so the arithmetic is one task over
# a four-deep operand axis. What cannot join it is xswap -- an adjacent fp16-pair
# permutation is a 2-byte reorder INSIDE one 8-byte TCDM word, and the reader's AGU places
# whole words. `lane_stride` moves channels, not halfwords.
#
# There used to be a self-contained kernel that did the swap with a word-rotate loop on the
# SIMD core. It was deleted, not kept as a fallback: the swap is rows*cols/2 element moves
# against a few hundred cycles for all the arithmetic it feeds, so it dominated, and doing
# it on the DM core's real byte-addressed DMA is strictly better. The LUT below was measured
# on that kernel and every entry should fall.
#
# Measured cost LUT -- PRE-SPLIT, re-measure (single-chip RTL sweep, SIMD-core cycles)
#     rows\cols     64     128     256
#        1           -     1227    1675
#        2         1227    1674    2574
#        4         1674    2573    4367
#        8         2573    4366      -     ((1,64) is the warm-up config; (8,256) dropped: the
#                                           xswap/tmp1/tmp2 scratch overflowed the L1 pool.
#                                           This kernel allocates NO scratch at all, so
#                                           that exclusion may no longer be needed.)

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

import _bingo_paths  # noqa: F401,E402  (puts mini_compiler's grouped subdirs on sys.path)
from bingo_dfg import BingoDFG                            # noqa: E402
from bingo_platform import core_roles, guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode                          # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView                     # noqa: E402
from bingo_data_staging import DataStaging                     # noqa: E402
from bingo_kernel_args import (                           # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelSimdRopeArgs,
    SnaxBingoKernelIdmaPairwiseSwapArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

# Core placement, from the generated map (snax_core_roles_defs.h).
_ROLES = core_roles()
SIMD_CORE = _ROLES["simd"]
DMA_CORE = _ROLES["dm"]
HOST_CORE = _ROLES["host"]
CHECK_FP16_TOL = 2
ROPE_BASE = 10000.0
ROPE_POS = 1

# (rows, cols); D = cols. Each row is a distinct token position (ROPE_POS + r), so cos/sin/xswap
# are per-row [rows, D] tables. A rows x cols grid for the cost LUT (bilinear, keyed
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


def build_mempool(st):
    """Hand every config's arrays to the staging helper; return the handles.

    WHERE they land is the platform's business, not this workload's: a config with a
    memory chiplet gets a mempool.bin, one without gets C arrays in the host image. See
    util/sim/common/bingo_data_staging.py -- addressing a memory chiplet a config does
    not have reads unmapped memory rather than faulting.
    """
    meta = []

    for i in range(len(CONFIGS)):
        x, cos_full, _xswap, sin_signed, out = _rope_ref(CONFIGS[i]["rows"], CONFIGS[i]["cols"], i)
        ox = st.put(f"rope_x_{i}", "uint16_t", x.view(np.uint16))
        oc = st.put(f"rope_cos_{i}", "uint16_t", cos_full.view(np.uint16))
        os_ = st.put(f"rope_sin_{i}", "uint16_t", sin_signed.view(np.uint16))
        ol = st.put(f"rope_golden_{i}", "uint16_t", out.view(np.uint16))
        meta.append((ox, oc, os_, ol))
    return meta


# Shared L1/L3 buffers reused across all serialized configs (see simd_silu_1cluster).
def build_config(g, i, meta, l1_ops, l1_out, l3_out, prev):
    rows  = CONFIGS[i]["rows"]
    D     = CONFIGS[i]["cols"]         # per-row fp16 length (cols)
    n     = rows * D                   # total fp16 elements
    tot_b = rows * D * 2               # [rows, D] fp16 bytes
    off_x, off_cos, off_sin, off_golden = meta[i]

    # THE FOUR OPERANDS ARE ONE BLOCK, slot k at k*tot_b -- the reader adds a single stride
    # per axis, so they have to be equally spaced, and the spacing is THIS config's row
    # size, not the shared allocation's. The block is sized for the largest config and each
    # config uses the first 4*tot_b of it.
    slot = lambda k: l1_ops if k == 0 else BingoMemAllocView(l1_ops, k * tot_b)

    lx = g.node(f"LoadX_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(off_x, slot(0), tot_b), prev)
    lc = g.node(f"LoadCos_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(off_cos, slot(1), tot_b), lx)
    ls = g.node(f"LoadSin_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(off_sin, slot(3), tot_b), lc)
    # The adjacent fp16-pair swap, on the DM core: it is a 2-byte reorder inside an 8-byte
    # TCDM word, which the SIMD reader's AGU cannot address at all. src and dst are 2*tot_b
    # apart and tot_b long, so the two strided copies cannot overlap.
    sw = g.node(f"Swap_{i}", DMA_CORE, "__snax_bingo_kernel_idma_pairwise_swap",
                SnaxBingoKernelIdmaPairwiseSwapArgs(slot(0), slot(2), n, 2), ls)
    rope = g.node(f"Rope_{i}", SIMD_CORE, "__snax_bingo_kernel_simd_rope",
                  SnaxBingoKernelSimdRopeArgs(l1_ops, l1_out, D, rows), sw)
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, tot_b), rope)
    chk = g.node(f"Check_rope_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(off_golden, l3_out,
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
    # The platform decides WHERE the arrays go -- a memory chiplet if this config
    # has one, the host image otherwise -- so it has to be parsed before staging.
    platform = parse_platform_cfg(args.platformcfg)
    st = DataStaging(platform)
    meta = build_mempool(st)
    if args.data_h is not None:
        n = st.emit(args.data_h, args.output_dir)
        print(f"Staged {n} B of inputs and goldens "
              f"{'on the memory chiplet' if st.on_memchip else 'in the host image'}")
    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": "rope", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    # Single-chip workload: build the DFG for chip 0x00 only (see simd_silu_1cluster).
    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[0x00])
    g = G(dfg)
    max_tot_b = max(r * c * 2 for (r, c) in _LUT_GRID)
    # ONE allocation for all four operands, sized for the largest config. Each config lays
    # its four rows out at its OWN tot_b spacing inside it, which is what the kernel
    # derives its operand stride from.
    l1_ops = g.l1("rp_ops", 4 * max_tot_b)
    l1_out = g.l1("rp_out", max_tot_b)
    l3_out = BingoMemAlloc("out_rope", size=max_tot_b, mem_level="L3")
    prev = None
    for i in range(len(CONFIGS)):
        prev = build_config(g, i, meta, l1_ops, l1_out, l3_out, prev)
    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("xDMA rope (explicit)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["rope_data.h"])
    print(f"Generated rope: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
