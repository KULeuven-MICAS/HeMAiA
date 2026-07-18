# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xDMA FP16 SiLU — one fused all-device kernel (__snax_bingo_kernel_xdma_silu_f16_f16).
# out[r,:] = silu(x[r,:]) = x*sigmoid(x) over [rows, cols] tiles. The whole op runs in a single
# DM-core kernel (one StreamMap pass); the host only loads the input, stores the output, checks it.
#
#   Load x[rows,cols] -> Silu __snax_bingo_kernel_xdma_silu_f16_f16 -> Store + Check(fp16 tol)
#
# Args are HW-free: { input_addr, output_addr, rows, cols }. The kernel derives everything else
# (the StreamMap SILU func, the 64-B beat count = cols/32) internally.

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
    SnaxBingoKernelXdmaSiluF16F16Args,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE, HOST_CORE = 1, 2
CHECK_FP16_TOL = 2
# Inputs + golden are staged on the memchip (mempool.bin) instead of baked into
# the host WIDE_SPM: 12 configs of fp16 in/golden (~26 KiB) would overflow the
# 128 KiB image once the 88 KiB device binary is included. The memchip base is
# the same one gemm_mem_chip_1cluster uses.
MEMPOOL_LOC = (2, 0)
MEMPOOL_VADDR = 0x8000_0000

# (rows, cols); each row is a length-cols tile (cols a multiple of 32). A rows x cols grid for the
# fused-kernel cost LUT (bilinear fit): rows {1,2,4,8} x cols {64,128,256} -- cols varies at EVERY
# row so the cols slope de-confounds from the rows==1 fast-path drop.
_LUT_GRID = [(r, c) for r in (1, 2, 4, 8) for c in (64, 128, 256)]
CONFIGS = [{"rows": r, "cols": c} for (r, c) in _LUT_GRID]


def _silu_ref(rows, cols, i):
    n = rows * cols
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


def build_mempool():
    """Concatenate every config's fp16 input + golden into the memchip image.

    Returns (blob: bytes, meta: [(off_in, off_golden), ...]). Each array is 64-B
    aligned so the iDMA / host loads land on aligned memchip words.
    """
    blob = bytearray()
    meta = []

    def _pad():
        while len(blob) % 64:
            blob.append(0)

    for i in range(len(CONFIGS)):
        x, y = _silu_ref(CONFIGS[i]["rows"], CONFIGS[i]["cols"], i)
        _pad(); oi = len(blob); blob += x.view(np.uint16).astype("<u2").tobytes()
        _pad(); og = len(blob); blob += y.view(np.uint16).astype("<u2").tobytes()
        meta.append((oi, og))
    return bytes(blob), meta


# One shared L1/L3 buffer set is reused across all 12 serialized configs (each
# config's Check depends on the previous config's Check, so the buffers are dead
# before reuse). Giving every config its OWN L3 out buffer emitted 12 separate
# never-freed bingo_l3_alloc calls and OOM'd the small 128 KiB L3 heap.
def build_config(g, i, base, meta, l1_x, l1_out, l3_out, prev):
    rows  = CONFIGS[i]["rows"]
    cols  = CONFIGS[i]["cols"]         # per-row length
    n     = rows * cols                # total elements
    tot_b = rows * cols * 2            # [rows, cols] fp16 bytes
    off_in, off_golden = meta[i]
    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(BingoMemFixedAddr(base + off_in), l1_x, tot_b), prev)
    silu = g.node(f"Silu_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_silu_f16_f16",
                  SnaxBingoKernelXdmaSiluF16F16Args(l1_x, l1_out, rows, cols), load)
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, tot_b), silu)
    chk = g.node(f"Check_silu_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemFixedAddr(base + off_golden), l3_out,
                     name=f"silu_cfg{i}", check_type=CHECK_FP16_TOL, num_elements=n,
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
        # Inputs + golden live on the memchip; the host data header is empty.
        with open(args.data_h, "w") as f:
            f.write("#include <stdint.h>\n")
        out_build = os.path.join(args.output_dir, "build")
        os.makedirs(out_build, exist_ok=True)
        with open(os.path.join(out_build, "mempool.bin"), "wb") as f:
            f.write(blob)
    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": "silu", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)
    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    # Single-chip workload: only chip 0x00 carries nodes. Building the DFG for
    # just chip 0x00 (instead of all 4 platform chiplets) drops the empty-chip
    # scheduler structures / global-task-id maps that otherwise waste ~7 KiB of
    # the small 128 KiB L3 heap.
    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[0x00])
    g = G(dfg)
    base = chiplet_addr_transform_loc(*MEMPOOL_LOC, MEMPOOL_VADDR)
    max_tot_b = max(r * c * 2 for (r, c) in _LUT_GRID)
    l1_x = g.l1("silu_x", max_tot_b)
    l1_out = g.l1("silu_out", max_tot_b)
    l3_out = BingoMemAlloc("out_silu", size=max_tot_b, mem_level="L3")
    prev = None
    for i in range(len(CONFIGS)):
        prev = build_config(g, i, base, meta, l1_x, l1_out, l3_out, prev)
    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("xDMA silu (fused)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["silu_data.h"])
    print(f"Generated silu: {len(CONFIGS)} configs")


if __name__ == "__main__":
    main()
