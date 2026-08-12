# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# MINIMAL 2-cluster SIMD xDMA softmax test (hemaia_tapeout_2c_simd).
#
# One small fused FP16 softmax per cluster, run concurrently:
#
#   cluster 0:  Load (memchip->L1) -> softmax [1, 64]  -> Store (L1->L3) -> Check
#   cluster 1:  Load (memchip->L1) -> softmax [2, 64]  -> Store (L1->L3) -> Check
#
# Same fused all-device kernel as xdma_softmax_1cluster (whole reduce(MAX) -> EXP ->
# integer-reciprocal -> normalize pipeline inside one DM-core kernel); the two chains are
# independent and never synchronize, so this covers both SIMD clusters running xDMA softmax
# at the same time against their own L1/L3 buffers. Host store/check nodes all sit on
# cluster 0 core 2 -- there is one host core per chiplet, and the bingo mini-compiler
# requires every host kernel to sit on it, even when the buffer it moves lives in another
# cluster's L1.
#
# Deliberately minimal: the wide-SPM L3 heap left over on this cfg is only ~18 KiB (the
# rest of the 128 KiB holds .data/.bss plus the embedded device binary). The full rows x
# cols cost-LUT sweep does NOT fit -- two 4 KiB output buffers plus 80 task descriptors
# OOM the heap. Take the softmax cycle LUT from xdma_softmax_1cluster (reproduced there);
# this workload only asserts that both clusters compute the right answer.

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
    SnaxBingoKernelXdmaSoftmaxF16F16Args,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE = 1
HOST_CORE = 2
HOST_CLUSTER = 0
CHECK_FP16_TOL = 2
# in/golden staged on the memchip (mempool.bin) so they cost no host .data; see
# xdma_silu_1cluster.
MEMPOOL_LOC = (2, 0)
MEMPOOL_VADDR = 0x8000_0000

# One config per cluster; CONFIGS[i] runs on cluster i. Keep these small -- see the L3
# heap note above. cols must be a multiple of 32.
CONFIGS = [
    {"rows": 1, "cols": 64},   # cluster 0: single-row fast path
    {"rows": 2, "cols": 64},   # cluster 1: multi-row path
]


# ----- deterministic peaky softmax reference (mirrors the HW fp16 datapath) -----
def _softmax_ref(rows, cols, i):
    D = cols
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


def build_mempool():
    """Concatenate every config's fp16 input + fp16 golden into the memchip image.

    Returns (blob: bytes, meta: [(off_in, off_golden), ...]); each array 64-B aligned.
    """
    blob = bytearray()
    meta = []

    def _pad():
        while len(blob) % 64:
            blob.append(0)

    for i in range(len(CONFIGS)):
        x, y = _softmax_ref(CONFIGS[i]["rows"], CONFIGS[i]["cols"], i)
        _pad(); oi = len(blob); blob += x.view(np.uint16).astype("<u2").tobytes()
        _pad(); og = len(blob); blob += y.view(np.uint16).astype("<u2").tobytes()
        meta.append((oi, og))
    return bytes(blob), meta


def build_chain(dfg, cluster, base, meta):
    """Load -> fused softmax -> store -> check, all buffers owned by `cluster`."""
    rows  = CONFIGS[cluster]["rows"]
    D     = CONFIGS[cluster]["cols"]   # per-row length (cols)
    n     = rows * D                   # total elements
    tot_b = n * 2                      # [rows, D] fp16 bytes (packed)
    off_in, off_golden = meta[cluster]

    l1_x = BingoMemAlloc(f"smx_x_c{cluster}", size=tot_b, mem_level="L1",
                         chip_id=0, cluster_id=cluster)
    l1_y = BingoMemAlloc(f"smx_y_c{cluster}", size=tot_b, mem_level="L1",
                         chip_id=0, cluster_id=cluster)
    l3_y = BingoMemAlloc(f"out_softmax_c{cluster}", size=tot_b, mem_level="L3")

    def node(name, cl, core, kname, kargs, after):
        nd = BingoNode(assigned_chiplet_id=0, assigned_cluster_id=cl, assigned_core_id=core,
                       node_name=name, kernel_name=kname, kernel_args=kargs)
        dfg.bingo_add_node(nd)
        if after is not None:
            dfg.bingo_add_edge(after, nd)
        return nd

    load = node(f"Load_c{cluster}", cluster, DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                SnaxBingoKernelIdma1dCopyArgs(BingoMemFixedAddr(base + off_in), l1_x, tot_b),
                None)
    # reduce-MAX, negate, sub-max, merged EXP+Sexp, integer reciprocal, normalize.
    sm = node(f"Softmax_c{cluster}", cluster, DMA_CORE,
              "__snax_bingo_kernel_xdma_softmax_f16_f16",
              SnaxBingoKernelXdmaSoftmaxF16F16Args(l1_x, l1_y, rows, D), load)
    st = node(f"Store_c{cluster}", HOST_CLUSTER, HOST_CORE, "__host_bingo_kernel_idma",
              HostBingoKernelIdmaArgs(l1_y, l3_y, tot_b), sm)
    node(f"Check_softmax_c{cluster}", HOST_CLUSTER, HOST_CORE,
         "__host_bingo_kernel_check_result",
         HostBingoKernelCheckResultArgs(BingoMemFixedAddr(base + off_golden), l3_y,
             name=f"softmax_c{cluster}", check_type=CHECK_FP16_TOL,
             num_elements=n, tolerance=0.02), st)


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
        print(f"Written data header: {args.data_h}")

    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": "softmax",
                       "configs": [dict(c, cluster=i) for i, c in enumerate(CONFIGS)]},
                      f, indent=2)

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    num_clusters = int(param["num_clusters"])
    if num_clusters != len(CONFIGS):
        raise ValueError(f"CONFIGS holds {len(CONFIGS)} entries (one per cluster) but "
                         f"params.hjson says num_clusters={num_clusters}")

    # Single-chip workload: build the DFG for chip 0x00 only (see xdma_silu_1cluster).
    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[0x00])
    base = chiplet_addr_transform_loc(*MEMPOOL_LOC, MEMPOOL_VADDR)
    for cluster in range(num_clusters):
        build_chain(dfg, cluster, base, meta)

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("xDMA softmax (minimal, 2-cluster SIMD)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["softmax_data.h"])
    print(f"Generated softmax: {len(CONFIGS)} configs, one per cluster")


if __name__ == "__main__":
    main()
