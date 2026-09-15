# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# SIMD PRIMITIVES -- one armed chain per kernel, the five building blocks the fused
# whole-op kernels are composed from. Each runs on the cluster's SIMD core.
#
# The fused workloads (softmax / rmsnorm / silu / swiglu / rope) each measure a whole
# pipeline, so none of them prices a single operator: a softmax cycle count folds a
# reduce, a map, the scalar work between them and a second map into one number. This
# workload arms ONE stage at a time, so the cost model has a per-operator curve and the
# difference between a fused kernel and the sum of its stages is visible as what the
# fusion actually bought.
#
# Per config, in this dispatch order (gather_simd_luts.py's OP_SPEC depends on it):
#
#   1 stream_map          out = 2*x                       StreamMap alone
#   2 stream_reduce       out = rowmax(x), splatted       StreamReduce alone
#   3 stream_elementwise  out = ea * eb                   EW1 alone
#   4 stream_map_reduce   out = rowmax(2*x), splatted     Map -||> Reduce, one task
#   5 fp16_to_int8        out = sat127(rne(q))            the quantiser alone
#
# Between them these cover every stage of the chain except EW0, the PRE-map elementwise,
# which by construction cannot be armed on its own: its whole purpose is to combine
# against a broadcast operand BEFORE a map, so a bare EW0 is just EW1 with a longer
# path. simd_softmax is what exercises it.
#
# EXACTNESS. Every golden here is computed to be bit-reproducible rather than merely
# close, because the check kernel offers a tolerance for fp16 but only byte-exact for
# int8:
#   * the map uses LINEAR a=2, which is an exponent increment -- exact in fp16;
#   * the reduce uses MAX, which is order-independent (ADD is not: the accumulator folds
#     in hardware order and fp16 addition does not associate);
#   * the quantiser gets inv_scale = 1.0 and integer-valued inputs, so the RNE round has
#     nothing to round and the check can be byte-exact. Two out-of-range values per row
#     make the saturation path part of the test rather than an untested branch.

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
from bingo_platform import core_roles, guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode                          # noqa: E402
from bingo_mem_handle import BingoMemAlloc                     # noqa: E402
from bingo_data_staging import DataStaging                     # noqa: E402
from bingo_kernel_args import (                           # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelSimdStreamMapArgs,
    SnaxBingoKernelSimdStreamReduceArgs,
    SnaxBingoKernelSimdStreamElementwiseArgs,
    SnaxBingoKernelSimdStreamMapReduceArgs,
    SnaxBingoKernelSimdFp16ToInt8Args,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

# Core placement, from the generated map (snax_core_roles_defs.h).
_ROLES = core_roles()
SIMD_CORE = _ROLES["simd"]
DMA_CORE = _ROLES["dm"]
HOST_CORE = _ROLES["host"]
CHECK_BYTE_EXACT = 0
CHECK_FP16_TOL = 2

# SIMD operator constants, mirroring snax_simd_lib.h.
FUNC_LINEAR = 0
RED_MAX = 0
EW_MUL = 0
BEAT_F16 = 32          # fp16 elements per 64-B beat
F32_TWO = 0x40000000   # FP32 bits of 2.0f
F32_ONE = 0x3F800000   # FP32 bits of 1.0f

# (rows, cols), cols a multiple of 32. The primitive costs are element-count keyed, so
# the grid only has to move n over a wide range; two row counts keep the per-row
# start/drain cost separable from the streaming cost.
_LUT_GRID = [(r, c) for r in (1, 4) for c in (64, 128, 256)]
CONFIGS = [{"rows": r, "cols": c} for (r, c) in _LUT_GRID]


def _ref(rows, cols, i):
    """Inputs and the five goldens for config *i*. See the EXACTNESS note above."""
    rng = np.random.RandomState(70100 + i)
    n = rows * cols

    # x: integer-valued and small, so `2*x` is exact and `x*1.0` needs no rounding.
    # Two values per row are pushed outside int8 range to exercise the saturation.
    x = rng.randint(-100, 101, size=n).astype(np.float16)
    x2 = x.reshape(rows, cols).copy()
    x2[:, 0] = np.float16(300.0)     # -> +127
    x2[:, 1] = np.float16(-300.0)    # -> -128 pre-clip, saturated to -127
    x = x2.reshape(-1)

    # ea/eb: a product that is exactly representable -- both operands are integers
    # small enough that ea*eb is an integer below 2048, where fp16 is exact.
    ea = rng.randint(-40, 41, size=n).astype(np.float16)
    eb = rng.randint(-40, 41, size=n).astype(np.float16)

    xf = x.astype(np.float32).reshape(rows, cols)

    # 1 map: LINEAR a=2, b=0
    g_map = (np.float32(2.0) * xf).astype(np.float16).reshape(-1)
    # 2 reduce: per-row MAX, emitted as one splatted 64-B beat per row
    g_red = np.repeat(xf.max(axis=1).astype(np.float16), BEAT_F16)
    # 3 elementwise: MUL
    g_ew = (ea.astype(np.float32) * eb.astype(np.float32)).astype(np.float16)
    # 4 map |> reduce: MAX over the mapped row. a=2 > 0 is monotonic, so this is
    #   exactly 2*rowmax and shares the reduce's order-independence.
    g_mr = np.repeat((np.float32(2.0) * xf).astype(np.float16).max(axis=1), BEAT_F16)
    # 5 quantise: the HW PE's model -- fp32 product, clamp to +/-128, RNE, symmetric
    #   saturate. With inv_scale = 1 and integer inputs the round is a no-op, so this
    #   check is byte-exact rather than tolerant.
    prod = np.clip(x.astype(np.float32) * np.float32(1.0),
                   np.float32(-128.0), np.float32(128.0))
    g_i8 = np.clip(np.rint(prod.astype(np.float64)), -127, 127).astype(np.int8)

    return x, ea, eb, g_map, g_red, g_ew, g_mr, g_i8


def build_mempool(st):
    """Hand every config's arrays to the staging helper; return the handles.

    WHERE they land is the platform's business, not this workload's: a config with a
    memory chiplet gets a mempool.bin, one without gets C arrays in the host image. See
    util/sim/common/bingo_data_staging.py -- addressing a memory chiplet a config does
    not have reads unmapped memory rather than faulting.
    """
    meta = []
    for i, c in enumerate(CONFIGS):
        x, ea, eb, g_map, g_red, g_ew, g_mr, g_i8 = _ref(c["rows"], c["cols"], i)
        meta.append({
            "x":     st.put(f"prims_x_{i}", "uint16_t", x.view(np.uint16)),
            "ea":    st.put(f"prims_ea_{i}", "uint16_t", ea.view(np.uint16)),
            "eb":    st.put(f"prims_eb_{i}", "uint16_t", eb.view(np.uint16)),
            "g_map": st.put(f"prims_g_map_{i}", "uint16_t", g_map.view(np.uint16)),
            "g_red": st.put(f"prims_g_red_{i}", "uint16_t", g_red.view(np.uint16)),
            "g_ew":  st.put(f"prims_g_ew_{i}", "uint16_t", g_ew.view(np.uint16)),
            "g_mr":  st.put(f"prims_g_mr_{i}", "uint16_t", g_mr.view(np.uint16)),
            "g_i8":  st.put(f"prims_g_i8_{i}", "int8_t", g_i8.astype(np.int8)),
        })
    return meta


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


def build_config(g, i, meta, buf, prev):
    """One config: load the inputs, run the five primitives, store and check each.

    The five are chained in dispatch order and every config is chained behind the
    previous config's last check, so one L1/L3 buffer set serves the whole sweep: each
    buffer is dead before it is reused. Twelve private buffer sets would be twelve
    never-freed bingo_l3_alloc calls.
    """
    rows = CONFIGS[i]["rows"]
    cols = CONFIGS[i]["cols"]
    n = rows * cols
    nb = n * 2                      # fp16 bytes of an [rows, cols] tile
    beats = cols // BEAT_F16        # 64-B beats per row
    m = meta[i]
    sb = rows * BEAT_F16 * 2        # bytes of the splatted per-row scalar beats

    def load(tag, handle, dst, nbytes, after):
        return g.node(f"Load_{tag}_{i}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                      SnaxBingoKernelIdma1dCopyArgs(handle, dst, nbytes), after)

    def check(tag, kern, l1_out, l3_out, nbytes, gold, num_el, ctype, tol):
        st = g.node(f"Store_{tag}_{i}", HOST_CORE, "__host_bingo_kernel_idma",
                    HostBingoKernelIdmaArgs(l1_out, l3_out, nbytes), kern)
        return g.node(f"Check_{tag}_{i}", HOST_CORE, "__host_bingo_kernel_check_result",
                      HostBingoKernelCheckResultArgs(
                          gold, l3_out,
                          name=f"{tag}_cfg{i}", check_type=ctype,
                          num_elements=num_el, tolerance=tol), st)

    ld_x = load("x", m["x"], buf["x"], nb, prev)
    ld_a = load("ea", m["ea"], buf["ea"], nb, ld_x)
    ld_b = load("eb", m["eb"], buf["eb"], nb, ld_a)

    # 1 -- StreamMap alone: out = LINEAR(2*x + 0)
    k1 = g.node(f"Map_{i}", SIMD_CORE, "__snax_bingo_kernel_simd_stream_map",
                SnaxBingoKernelSimdStreamMapArgs(
                    buf["x"], buf["o16"], beats=beats, func=FUNC_LINEAR,
                    a_f32bits=F32_TWO, b_f32bits=0, rows=rows), ld_b)
    c1 = check("map", k1, buf["o16"], buf["l3_16"], nb, m["g_map"], n, CHECK_FP16_TOL, 0.0)

    # 2 -- StreamReduce alone: one splatted MAX beat per row
    k2 = g.node(f"Reduce_{i}", SIMD_CORE, "__snax_bingo_kernel_simd_stream_reduce",
                SnaxBingoKernelSimdStreamReduceArgs(
                    buf["x"], buf["os"], beats=beats, op=RED_MAX, rows=rows), c1)
    c2 = check("reduce", k2, buf["os"], buf["l3_s"], sb, m["g_red"],
               rows * BEAT_F16, CHECK_FP16_TOL, 0.0)

    # 3 -- StreamElementwise (EW1) alone: out = ea * eb. Passing src_b_addr lets the two
    #      operands be separate allocations; the kernel derives the interleave stride.
    k3 = g.node(f"Elemwise_{i}", SIMD_CORE, "__snax_bingo_kernel_simd_stream_elementwise",
                SnaxBingoKernelSimdStreamElementwiseArgs(
                    buf["ea"], buf["o16"], beats=beats, op=EW_MUL, operand_count=2,
                    rows=rows, src_b_addr=buf["eb"]), c2)
    c3 = check("elemwise", k3, buf["o16"], buf["l3_16"], nb, m["g_ew"], n,
               CHECK_FP16_TOL, 0.0)

    # 4 -- Map -||> Reduce in ONE task, tap off: only the per-row scalars come out.
    k4 = g.node(f"MapReduce_{i}", SIMD_CORE, "__snax_bingo_kernel_simd_stream_map_reduce",
                SnaxBingoKernelSimdStreamMapReduceArgs(
                    buf["x"], buf["os"], beats=beats, func=FUNC_LINEAR,
                    reduce_op=RED_MAX, a_f32bits=F32_TWO, b_f32bits=0,
                    tap=False, rows=rows), c3)
    c4 = check("mapreduce", k4, buf["os"], buf["l3_s"], sb, m["g_mr"],
               rows * BEAT_F16, CHECK_FP16_TOL, 0.0)

    # 5 -- the quantiser alone, inv_scale = 1.0 (see the EXACTNESS note: byte-exact).
    k5 = g.node(f"Quant_{i}", SIMD_CORE, "__snax_bingo_kernel_simd_fp16_to_int8",
                SnaxBingoKernelSimdFp16ToInt8Args(
                    buf["x"], buf["o8"], beats=beats, rows=rows,
                    inv_scale_f32bits=F32_ONE), c4)
    c5 = check("quant", k5, buf["o8"], buf["l3_8"], n, m["g_i8"], n,
               CHECK_BYTE_EXACT, 0.0)
    return c5


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
            json.dump({"op": "prims", "configs": [dict(c) for c in CONFIGS]}, f, indent=2)

    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return

    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[0x00])
    g = G(dfg)

    max_n = max(r * c for (r, c) in _LUT_GRID)
    max_rows = max(r for (r, _) in _LUT_GRID)
    # Names are deliberately ordered so the elementwise operands land adjacent and in
    # ascending address order: the L1 allocator hands out addresses in name order, and
    # the elementwise reader bases at the lower operand and strides upward.
    buf = {
        "x":   g.l1("prims_a_x", max_n * 2),
        "ea":  g.l1("prims_b_ea", max_n * 2),
        "eb":  g.l1("prims_c_eb", max_n * 2),
        "o16": g.l1("prims_d_o16", max_n * 2),
        "os":  g.l1("prims_e_os", max_rows * BEAT_F16 * 2),
        "o8":  g.l1("prims_f_o8", max_n),
        "l3_16": BingoMemAlloc("out_prims_16", size=max_n * 2, mem_level="L3"),
        "l3_s":  BingoMemAlloc("out_prims_s", size=max_rows * BEAT_F16 * 2, mem_level="L3"),
        "l3_8":  BingoMemAlloc("out_prims_8", size=max_n, mem_level="L3"),
    }

    prev = None
    for i in range(len(CONFIGS)):
        prev = build_config(g, i, meta, buf, prev)

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg("SIMD primitives (one operator per kernel)", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=["prims_data.h"])
    print(f"Generated SIMD primitives: {len(CONFIGS)} configs x 5 kernels")


if __name__ == "__main__":
    main()
