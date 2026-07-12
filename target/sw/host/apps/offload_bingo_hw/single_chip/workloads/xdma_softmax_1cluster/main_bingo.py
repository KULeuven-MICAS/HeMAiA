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
#   T1  reduce(MAX, rows)                  x          -> max_bt[rows]       [dev]
#   H1  neg      (GENERIC fp16 unary)      max_bt     -> negmax_bt[rows]    [host]
#   B1  xDMA broadcast (reader stride 0)   negmax_bt  -> negmax_bc[rows,D]  [dev]
#   T2  elementwise(ADD, {x, negmax_bc})   x          -> xs[rows,D]         [dev]
#   T34 map(EXP) -||> reduce(ADD, TAP)     xs         -> exp[rows,beats+1]  [dev]  <-- MERGED
#   G1  xDMA gather (strided)              exp's TAP  -> sum_bt[rows]       [dev]
#   H2  reciprocal (GENERIC fp16 unary)    sum_bt     -> inv_bt[rows]       [host]
#   B2  xDMA broadcast (reader stride 0)   inv_bt     -> inv_bc[rows,*]     [dev]
#   T5  elementwise(MUL, {exp, inv_bc})    exp        -> out[rows,D]        [dev]
#   Store(out) + Check(fp16 tol)                                            [host]
#   Tq  elementwise(MUL)+Fp16ToInt8        exp        -> out_i8            [dev]
#
# NO bespoke scalar-broadcast kernel. The host still does the fp16 scalar math, but through the
# ORDINARY elementwise unary kernels (__host_bingo_kernel_neg / _reciprocal at BINGO_PREC_FP16).
# That works because StreamReduce emits each row's scalar SPLATTED across all 32 lanes of its
# beat (see StreamReduce.scala: "splatted across all lanes of one output beat", and the TAP
# variant emits "one extra beat (the scalar, splatted)"). So running a plain elementwise unary
# over the rows*32 lanes transforms every lane and the result is STILL splatted -- the splat is
# free, and the host op is a generic array op that knows nothing about rows, beats or broadcast.
#
# The [rows,D] operand itself still has to exist: the StreamElementwise interleave is a single
# affine address stream, so it cannot hold the scalar's address fixed while x walks the row. But
# the xDMA builds it (B1/B2, reader stride 0), not the CVA6 -- so the host touches only rows*32
# lanes, independent of D. Two plain strided copies (bcast/gather) carry all the layout work.
#
# T34 is the map+reduce MERGE (the snax-xdma-softmax-fold "B3" fusion): StreamMap(EXP)
# chained into StreamReduce(ADD|TAP) in ONE xDMA task. It replaces the old T3 map(EXP) +
# T4 reduce(ADD) pair, whose reduce RE-READ the whole exp tensor -- so it removes a full
# pass over [rows,D] and one task setup (~600 cc of CSR orchestration).
#
# The cost of TAP is a PADDED exp: each row is [beats mapped beats | 1 scalar beat], i.e.
# (beats+1)*64 bytes, with the row's Sexp inline as the trailing beat. That ripples into
# the two consumers, which is why they take strides now:
#   H2 reads the per-row scalars as a strided column INSIDE exp -- exp.view(beats*64) with
#      in_row_stride=pad_row (there is no separate sum[] buffer any more).
#   B2 lays the recip broadcast down at exp's PADDED pitch, so recip_bc and exp share a row
#      stride and T5's interleave delta is constant.
#   T5/Tq read exp with src_row_stride=pad_row -> a 3D {operand, beat, row} reader that skips
#      each row's trailing scalar beat. Both operands must share the pitch, because the
#      interleave derives operand 1 as operand 0 + a CONSTANT delta. The OUTPUT is packed.
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
    SnaxBingoKernelXdma6dArgs,
    SnaxBingoKernelXdmaStreamReduceArgs,
    SnaxBingoKernelXdmaStreamMapReduceArgs,
    SnaxBingoKernelXdmaStreamElementwiseArgs,
    HostBingoKernelAraNegF16Args,
    HostBingoKernelAraReciprocalF16Args,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
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
BEAT = 64          # xDMA bus beat in bytes
BEAT_FP16 = 32     # fp16 lanes in a beat -> the host's per-row output width (one splatted beat)


def bcast(g, name, src_beat, dst, beats, rows, dst_row_stride, after):
    """xDMA stride-0 BROADCAST: expand [rows] splatted scalar beats -> [rows, D].

    The reader's inner temporal dim has stride 0, so the row's single scalar beat is RE-READ
    `beats` times (the AGU's trip count comes from its own index counter, so a 0 step holds the
    address while the loop still runs); the outer dim steps one scalar beat per row. The writer
    lays the row down at dst_row_stride, which is how the same pass serves both a flat operand
    (stride beats*64) and one that must line up with a TAP-padded tensor (stride (beats+1)*64).

    This is a plain strided copy -- no extension -- so the generic xdma_6d kernel expresses it.
    """
    return g.node(name, DMA_CORE, "__snax_bingo_kernel_xdma_6d",
                  SnaxBingoKernelXdma6dArgs(src_beat, dst, 8, 8,
                      [0, BEAT], [beats, rows],                # reader: stride-0 inner
                      [BEAT, dst_row_stride], [beats, rows]),  # writer: beats data beats/row
                  after)


def gather(g, name, src_view, dst, rows, src_row_stride, after):
    """xDMA strided gather: pull the `rows` per-row scalar beats out of a TAP-padded tensor
    (they sit src_row_stride bytes apart, one at the end of each padded row) into a CONTIGUOUS
    [rows] beat array. That is what lets the generic host unary below walk them as a plain
    fp16 array -- no strided-input support needed anywhere on the host.

    Also a plain strided copy, so again the generic xdma_6d kernel expresses it.
    """
    return g.node(name, DMA_CORE, "__snax_bingo_kernel_xdma_6d",
                  SnaxBingoKernelXdma6dArgs(src_view, dst, 8, 8,
                      [src_row_stride], [rows],   # reader: one scalar beat per padded row
                      [BEAT], [rows]),            # writer: pack them back-to-back
                  after)

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
    tot_b = rows * beats * 64          # [rows, D] fp16 bytes (packed)
    row_b = beats * 64                 # flat row pitch
    pad_row = (beats + 1) * 64         # TAP row: beats data beats + 1 trailing scalar beat
    pad_b   = rows * pad_row           # [rows, beats+1] bytes
    lanes = rows * BEAT_FP16           # fp16 elements in the `rows` splatted scalar beats
    # Name prefixes order the allocator so each elementwise src_b operand (negbc, invbc)
    # sits above its operand 0 (x, expb) -- the reader strides forward.
    # expb and invbc are PADDED (pad_row pitch) so the T5/Tq interleave sees a constant
    # operand delta; there is no separate sum[] buffer -- the merged map+reduce leaves each
    # row's Sexp inline as expb's trailing beat, and G1 gathers those out.
    # The *bt buffers are [rows] splatted 64-B scalar beats.
    l1_x      = g.l1(f"smx0x_{i}", tot_b)
    l1_maxbt  = g.l1(f"smx1maxbt_{i}", rows * BEAT)
    l1_negbt  = g.l1(f"smx2negbt_{i}", rows * BEAT)
    l1_negbc  = g.l1(f"smx3negbc_{i}", tot_b)
    l1_xs     = g.l1(f"smx4xs_{i}", tot_b)
    l1_expb   = g.l1(f"smx5expb_{i}", pad_b)
    l1_sumbt  = g.l1(f"smx6sumbt_{i}", rows * BEAT)
    l1_invbt  = g.l1(f"smx7invbt_{i}", rows * BEAT)
    l1_invbc  = g.l1(f"smx8invbc_{i}", pad_b)
    l1_out    = g.l1(f"smx9out_{i}", tot_b)
    l1_i8     = g.l1(f"smxAi8_{i}", n)

    load = g.node(f"Load_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"in_{i}"), l1_x, tot_b), prev)
    t1 = g.node(f"Max_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_reduce",
                SnaxBingoKernelXdmaStreamReduceArgs(l1_x, l1_maxbt, beats, op=R_MAX,
                    rows=rows, csr_mode=0), load)
    # The reduce splats each row's scalar across ALL 32 lanes of its beat, so a GENERIC
    # elementwise unary over the rows*32 fp16 lanes negates every lane and the result stays
    # splatted -- no scalar-broadcast kernel, just __host_bingo_kernel_neg at fp16.
    h1 = g.node(f"Neg_{i}", HOST_CORE, "__host_bingo_kernel_neg",
                HostBingoKernelAraNegF16Args(l1_maxbt, l1_negbt, lanes), t1)
    b1 = bcast(g, f"BcastNeg_{i}", l1_negbt, l1_negbc, beats, rows, row_b, h1)
    t2 = g.node(f"SubMax_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
                SnaxBingoKernelXdmaStreamElementwiseArgs(l1_x, l1_xs, beats, op=EW_ADD,
                    operand_count=2, rows=rows, src_b_addr=l1_negbc, csr_mode=0,
                    dst_bound0=rows * beats), b1)
    # MERGED: exp(xs) and Sexp in ONE task -- StreamMap(EXP) -||> StreamReduce(ADD|TAP).
    # Replaces the old Exp + Sum pair (the Sum re-read the whole exp tensor). The subtract
    # cannot fold into the map's `b` here: rows>1, and each row has its own max.
    t34 = g.node(f"ExpSum_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_map_reduce",
                 SnaxBingoKernelXdmaStreamMapReduceArgs(l1_xs, l1_expb, beats, func=F_EXP,
                     reduce_op=R_ADD, tap=True, a_f32bits=ONE_F32, b_f32bits=0,
                     rows=rows, csr_mode=0), t2)
    # Sexp is the trailing beat of each PADDED row -> gather the strided column into a
    # contiguous [rows] beat array so the generic host unary can read it as a flat fp16 array.
    g1 = gather(g, f"GatherSum_{i}", l1_expb.view(beats * BEAT), l1_sumbt, rows, pad_row, t34)
    h2 = g.node(f"Recip_{i}", HOST_CORE, "__host_bingo_kernel_reciprocal",
                HostBingoKernelAraReciprocalF16Args(l1_sumbt, l1_invbt, lanes), g1)
    # The broadcast lays 1/Sexp down at expb's PADDED pitch, so T5's interleave delta is constant.
    b2 = bcast(g, f"BcastRecip_{i}", l1_invbt, l1_invbc, beats, rows, pad_row, h2)
    t5 = g.node(f"Norm_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
                SnaxBingoKernelXdmaStreamElementwiseArgs(l1_expb, l1_out, beats, op=EW_MUL,
                    operand_count=2, rows=rows, src_b_addr=l1_invbc, csr_mode=0,
                    dst_bound0=rows * beats, src_row_stride=pad_row), b2)
    # Store + tolerant fp16 check of the softmax output.
    l3_out = BingoMemAlloc(f"out_softmax_{i}", size=tot_b, mem_level="L3")
    store = g.node(f"Store_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(l1_out, l3_out, tot_b), t5)
    chk = g.node(f"Check_softmax_cfg{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                 HostBingoKernelCheckResultArgs(BingoMemSymbol(f"golden_{i}"), l3_out,
                     name=f"softmax_cfg{i}", check_type=CHECK_FP16_TOL,
                     num_elements=n, tolerance=0.02), store)
    # Tq fused FP16->INT8 quant — exercises the path (leaf, no numeric check). Same padded
    # 3D read of expb as T5; only the writer differs (int8 packs 2 fp16 beats into 1).
    g.node(f"Quant_{i}", DMA_CORE, "__snax_bingo_kernel_xdma_stream_elementwise",
           SnaxBingoKernelXdmaStreamElementwiseArgs(l1_expb, l1_i8, beats, op=EW_MUL,
               operand_count=2, rows=rows, src_b_addr=l1_invbc, csr_mode=1,
               dst_bound0=rows * beats // 2, out_dtype=1, inv_scale_f32bits=INV_SCALE,
               src_row_stride=pad_row), chk)
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
