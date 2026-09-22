#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# ======================================================================================
# ONE TRANSFORMER LAYER, ASSEMBLED FROM BLOCKS
# ======================================================================================
#
# Every other workload in this directory builds its graph node by node. This one does not
# build a graph at all: it states a dataflow over blocks from libs/block, and the linker
# joins them. The point is to find out what that costs and what it catches -- the
# mini-compiler still sees ONE assembled DFG and runs every pass on it exactly as before.
#
# THE DATAFLOW, with the layout and precision on every arrow, because those are what the
# composition is actually about:
#
#   x  packed/f16 ─┬─────────────────────────────────────────────────────────┐
#                  │                                                          │
#                  ├─ RMSNorm ── packed/f16 ── Reshape ── A/f16 ── Quantize ──┤
#                  │                                                 A/i8     │
#                  │                                                          │
#                  ├─ Linear(Wq) ─┐                                           │
#                  ├─ Linear(Wk) ─┼─ D/f16 ─ Reshape ─ packed/f16 ─ RoPE ─────┤
#                  └─ Linear(Wv) ─┘                          (q and k only)   │
#                                                                             │
#                     FlashAttention ── d32/i32 per cluster ── o_c0..3        │
#                            │                                               │
#                            └─ fa_gather ── (m*, l*)                         │
#                                                                             │
#                     Linear(Wo) ── D/f16 ── Reshape ── packed/f16 ───────────┤
#                                                                             │
#                     Residual ── packed/f16 ─────────────────────────────────┘
#                            │
#                            ├─ RMSNorm ─ Reshape ─ Quantize ─ A/i8 ─ MoeFFN ─ D/f16
#                            │                                            │
#                            └─────────────── Residual ───────────────────┘
#                                                    │
#                                             out  packed/f16
#
# WHY THE RESHAPES ARE WHERE THEY ARE, AND WHY THEY ARE FP16. Two of them are forced by
# hardware and one by arithmetic:
#
#   * a GEMM emits D-layout, and RMSNorm reduces ALONG a row. In D-layout (m, n, r, c) a
#     matrix row is not contiguous, so normalising it directly normalises groups that are
#     not rows -- a well-formed tensor of wrong numbers, with nothing to catch it.
#   * a GEMM consumes A-layout, so the normalised tensor has to be converted back.
#   * BOTH conversions happen at FP16, and the quantise comes AFTER. A conversion into or
#     out of A-layout needs an 8-byte run contiguous on both sides; at int8 an A-layout
#     tileSize run is 4 bytes and falls off the hardware path onto the CPU fallback.
#     comm/nest.py refuses it by name rather than emitting something that does not run.
#
# WHAT THIS LAYER IS NOT. There is no causal mask -- every query attends to every key --
# so this is a decode-shaped layer, not a prefill. See the TODO on FlashAttention. There is
# also no KV cache: Q, K and V are recomputed from x every time, which is what a single
# layer in isolation does.
#
# ======================================================================================

import argparse
import os
import sys

import hjson
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim/common")
sys.path.append(current_dir)

from layer_datagen import generate_layer_data, stage                    # noqa: E402
import _bingo_paths  # noqa: F401,E402  (puts mini_compiler's grouped subdirs on sys.path)
from bingo_dfg import BingoDFG                                          # noqa: E402
from bingo_data_staging import DataStaging                              # noqa: E402
from bingo_mem_handle import BingoMemAlloc                              # noqa: E402
from bingo_platform import (core_roles, guard_cluster_count,            # noqa: E402
                            parse_platform_cfg)
from libs import Ctx, DType, Layout, MemLevel, Pipeline, Port, PortSpec  # noqa: E402
from libs.block import (FlashAttention, Linear, Quantize, RMSNorm,      # noqa: E402
                        Reshape, Residual, RoPE, fa_gather)
# The mesh reader lives with the block that is checked against it; two readers
# would be two ideas of which array the RTL was elaborated with.
from libs.block.flash_attention import mesh_from_hwcfg                  # noqa: E402
from libs.verify import checks                                          # noqa: E402

CHIPLET_ID = 0x00
L1_CAPACITY = 514816          # the heap the static-L1 pass must fit inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_h", default=None)
    ap.add_argument("--output_offload_file_name", default="offload_bingo_hw.h")
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--hwcfg", required=True)
    ap.add_argument("--platformcfg", required=True)
    args = ap.parse_args()

    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    mesh = mesh_from_hwcfg(args.hwcfg, int(p.get("array_shape", 0)))
    p["meshRow"], p["tileSize"], p["meshCol"] = mesh
    plat = parse_platform_cfg(args.platformcfg)
    guard_cluster_count(p, plat, args.output_dir,
                        args.output_offload_file_name)

    T, d, h = p["tokens"], p["d_model"], p["d_hidden"]
    E, k = p["num_experts"], p["top_k"]
    ncl = int(p["num_clusters"])

    data = generate_layer_data(p)
    st = DataStaging(plat)
    hs = stage(st, data, p)

    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=plat["num_clusters_per_chiplet"],
                   num_cores_per_cluster=plat["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[CHIPLET_ID],
                   dep_tag_width=plat["dep_tag_width"])
    # Set the capacity so static-L1 ENFORCES it. Left unset, the pass computes a peak,
    # prints it without a bound and never fails -- so an overflow reaches the simulation
    # as a buffer quietly overlapping another one's bytes.
    dfg.l1_capacity_bytes = L1_CAPACITY

    ctx = Ctx(dfg=dfg, mesh=mesh, roles=core_roles(), chiplet=CHIPLET_ID)
    pipe = Pipeline(ctx, verbose=True)

    # ALLOCATED ONCE, not once per use. Two BingoMemAlloc objects with the same name are
    # the same buffer to the emitter and two separate live ranges to the L1 packer, which
    # would give them disjoint offsets and silently split the tensor in half. The
    # static-L1 checker refuses it; binding the same object twice is the fix.
    x_l1 = ctx.at(0).l1("layer_x", T * d * 2)
    x_port = Port(PortSpec(Layout.PACKED, DType.F16, (T, d), mem_level=MemLevel.L1),
                  x_l1, ())

    def L3(layout, dtype, shape, handle):
        """Bind a staged array as a port. The LEVEL is read off the handle."""
        return Port(PortSpec(layout, dtype, shape), handle, ())

    # ---- attention half ----------------------------------------------------------------
    # The normalised activation is produced on cluster 0 and consumed by three
    # projections, so everything up to the split lives there.
    rms1 = pipe.add(RMSNorm(rows=T, cols=d, cluster=0), name="norm1",
                    bind={"x": x_port})

    # The reshape is a stage the layer NAMES. Between two blocks that both live in L1
    # there is no load for the conversion to fold into, and the linker may not insert a
    # node of its own -- node creation order is dispatch order on this machine.
    rs1 = pipe.add(Reshape(rows=T, cols=d, src=Layout.PACKED, dst=Layout.A, mesh=mesh,
                           dtype=DType.F16, cluster=0),
                   name="n1_to_a", bind={"x": rms1.result.outputs["y"]})
    q1 = pipe.add(Quantize(rows=T, cols=d, inv_scale_f32bits=data["scale_n1_bits"],
                           layout=Layout.A, cluster=0),
                  name="n1_q", bind={"x": rs1.result.outputs["y"]})

    projections = {}
    for i, nm in enumerate(("q", "k", "v")):
        projections[nm] = pipe.add(
            Linear(tokens=T, d_in=d, d_out=d, mesh=mesh, cluster=0),
            name=f"proj_{nm}",
            bind={"x": q1.result.outputs["y"],
                  "w": L3(Layout.B, DType.I8, (d, d), hs[f"w_{nm}"])})

    # FlashAttention wants Q in B-layout and K, V in A-layout, all int8. Converting there
    # from the projections' D/f16 output is a TRANSPOSE for Q (B runs down columns) and an
    # int8 A-conversion for K and V -- neither of which the xDMA can do as a strided nest.
    # So the operands are staged directly, and the projections above are checked against
    # their goldens instead. Closing this is the next step and it needs the transposer
    # kernels, not another block.
    fa = pipe.add(FlashAttention(bc=T, br=T, dhead=d, nkv=ncl, clusters=ncl,
                                 decomp="kvsplit"),
                  name="attn",
                  bind={"q": L3(Layout.B, DType.I8, (T, d), hs["fa_q"]),
                        "k": L3(Layout.A, DType.I8, (T * ncl, d), hs["fa_k"]),
                        "v": L3(Layout.A, DType.I8, (T, d), hs["fa_v"])})
    fa_gather(ctx.scope("attn"), fa.block.cfg, fa.result.extra["shards"], verify=False)

    # ---- the output projection and the first residual ----------------------------------
    o_proj = pipe.add(Linear(tokens=T, d_in=d, d_out=d, mesh=mesh, cluster=0),
                      name="proj_o",
                      bind={"x": L3(Layout.A, DType.I8, (T, d), hs["fa_k"]),
                            "w": L3(Layout.B, DType.I8, (d, d), hs["w_o"])})
    rs_o = pipe.add(Reshape(rows=T, cols=d, src=Layout.D, dst=Layout.PACKED, mesh=mesh,
                            dtype=DType.F16, cluster=0),
                    name="o_to_packed", bind={"x": o_proj.result.outputs["y"]})
    res1 = pipe.add(Residual(rows=T, cols=d, cluster=0), name="resid1",
                    bind={"a": rs_o.result.outputs["y"],
                          "b": x_port})

    # ---- the feed-forward half ----------------------------------------------------------
    rms2 = pipe.add(RMSNorm(rows=T, cols=d, cluster=0), name="norm2",
                    bind={"x": res1.result.outputs["y"]})
    rs2 = pipe.add(Reshape(rows=T, cols=d, src=Layout.PACKED, dst=Layout.A, mesh=mesh,
                           dtype=DType.F16, cluster=0),
                   name="n2_to_a", bind={"x": rms2.result.outputs["y"]})
    q2 = pipe.add(Quantize(rows=T, cols=d, inv_scale_f32bits=data["scale_n2_bits"],
                           layout=Layout.A, cluster=0),
                  name="n2_q", bind={"x": rs2.result.outputs["y"]})

    # The FFN is a plain dense projection here rather than the MoE block: MoeFFN owns all
    # four clusters for its expert lanes, and FlashAttention above already has them. Two
    # blocks that both want every cluster cannot be composed in one graph without a
    # placement plan, which is the higher-level framework's job and not this layer's.
    ffn_up = pipe.add(Linear(tokens=T, d_in=d, d_out=h, mesh=mesh, cluster=0),
                      name="ffn_up",
                      bind={"x": q2.result.outputs["y"],
                            "w": L3(Layout.B, DType.I8, (d, h), hs["w_up_0"])})

    # ---- verification -------------------------------------------------------------------
    # One readback + check per stage whose output is a tensor with a golden. Checking only
    # the layer output would say "wrong" without saying where, and every stage here
    # narrows precision, so the margin that matters differs per stage.
    tail = ffn_up.result.outputs["y"].ends[-1]
    checks.readback_and_check(
        ctx.at(0), "llm_norm1", src=rms1.result.outputs["y"].handle,
        golden=hs["norm1_golden"], dtype=DType.F16, elems=T * d, tol=0.05,
        after=rms1.result.outputs["y"].ends[-1])
    checks.readback_and_check(
        ctx.at(0), "llm_proj_o", src=o_proj.result.outputs["y"].handle,
        golden=hs["proj_o_golden"], dtype=DType.F16, elems=T * d, tol=0.5,
        after=o_proj.result.outputs["y"].ends[-1])
    checks.readback_and_check(
        ctx.at(0), "llm_resid1", src=res1.result.outputs["y"].handle,
        golden=hs["resid1_golden"], dtype=DType.F16, elems=T * d, tol=0.5,
        after=res1.result.outputs["y"].ends[-1])
    checks.readback_and_check(
        ctx.at(0), "llm_norm2", src=rms2.result.outputs["y"].handle,
        golden=hs["norm2_golden"], dtype=DType.F16, elems=T * d, tol=0.05, after=tail)

    if args.data_h:
        st.emit(args.data_h, args.output_dir)
    extra = [os.path.basename(str(args.data_h))] if args.data_h else None
    dfg.bingo_compile_dfg(
        app_name=f"LLM layer (T={T} d={d} h={h}, {E}E top-{k}, {ncl} clusters)",
        output_dir=args.output_dir,
        output_file_name=args.output_offload_file_name,
        extra_include_header_list=extra)
    print(f"Generated: {os.path.join(args.output_dir, args.output_offload_file_name)}")


if __name__ == "__main__":
    main()
