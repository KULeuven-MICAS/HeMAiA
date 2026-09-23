#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# ======================================================================================
# ONE TRANSFORMER LAYER, END TO END, ON FOUR CLUSTERS
# ======================================================================================
#
# This is the layer as an APP, not as a rung of anything. It builds its own graph here,
# in one readable pass, and does not go through util/sim/llm/llm_layer_stages.py -- that
# module is the bring-up LADDER, whose job is to emit prefixes of the layer so a failure
# names the stage that broke. A ladder is a debugging instrument; this is the thing being
# debugged, and it should read like the layer rather than like a sweep harness.
#
# ======================================================================================
# THE DATAFLOW, WITH THE LAYOUT AND PRECISION ON EVERY ARROW
# ======================================================================================
#
#   x  [T, d] row_major/f16  in L3
#    |                                                                |
#    +----------------------------------------------------------------+   (the skip)
#    |                                                                |
#   Ld_layer_x ........................................ row_major/f16 L1
#    |                                                                |
#   RMSNorm ........................ row_major/f16 -> row_major/f16   |
#    |                                                                |
#   Reshape  n1_to_a ............... row_major/f16 -> A/f16           |
#   Quantize n1_q .................. A/f16         -> A/i8            |
#    |                                                                |
#   Linear   proj_q|k|v ............ A/i8 x B/i8   -> D/f16           |
#   Dequant  proj_*_dq ............. D/f16         -> D/f16           |
#    |                                                                |
#    :  (dead end -- see WHY THE PROJECTIONS DO NOT FEED ATTENTION)   |
#                                                                     |
#   FlashAttention ................. B/i8, A/i8, A/i8 -> d32/i32      |
#    |  over 4 clusters, KV-split, folded by an in-fabric gather      |
#    :  (its operands are STAGED, not produced above)                 |
#                                                                     |
#   Linear   proj_o ................ A/i8 x B/i8   -> D/f16           |
#   Dequant  proj_o_dq ............. D/f16         -> D/f16           |
#   Reshape  o_to_row_major ........ D/f16         -> row_major/f16   |
#    |                                                                |
#   Residual resid1 <-------------------------------------------------+
#    |  row_major/f16
#    +----------------------------------------------------------------+   (the skip)
#    |                                                                |
#   RMSNorm  norm2 ................. row_major/f16 -> row_major/f16   |
#   Reshape  n2_to_a ............... row_major/f16 -> A/f16           |
#   Quantize n2_q .................. A/f16         -> A/i8            |
#   Linear   ffn_up ................ A/i8 x B/i8   -> D/f16           |
#   Dequant  ffn_up_dq ............. D/f16         -> D/f16           |
#   Reshape  ffn_to_row_major ...... D/f16         -> row_major/f16   |
#    |                                                                |
#   Residual resid2 <-------------------------------------------------+
#    |
#   layer_out  [T, h] row_major/f16
#
# WHY THE RESHAPES SIT WHERE THEY DO, and why they are stages the layer NAMES rather than
# something the linker inserts. The array reads operand A as (m, k, r, s) and writes D as
# (m, n, r, c); everything else -- the SIMD block, a golden, the host -- works in plain
# row-major. So every crossing between the two is a real xDMA pass, and:
#
#   * it is FP16, always. A conversion into or out of A-layout needs an 8-byte run
#     contiguous on BOTH sides, which is four consecutive features at fp16 and only two
#     bytes at int8. That is why `quantise` comes AFTER `n1_to_a` and not before it --
#     reversing them makes the reshape inexpressible, and comm/nest.py refuses it by name.
#   * the linker may not insert it. Node creation order IS dispatch order on this machine,
#     so a linker that injected nodes would silently move the schedule. Between two blocks
#     that both live in L1 there is no load for the conversion to fold into, so it becomes
#     a stage written here, where its position in the order is visible.
#
# WHY EVERY GEMM IS FOLLOWED BY A DEQUANTISE. The array accumulates int8 x int8 in int32
# and the D port narrows to fp16, so a projection's output still carries BOTH operands'
# quantisation scales -- at d=128 it lands in the thousands while everything it has to
# rejoin is O(1). Two fp16 limits bracket the GEMM and neither reports being exceeded:
# the D-port narrow saturates over 65504, and RMSNorm's SUM(x^2) is itself narrowed to
# fp16 before the rsqrt sees it, so its input needs |x| < ~22 at d=128. A factor of a
# thousand sits between them and nothing else in the chain can absorb it.
#
# WHY THE PROJECTIONS DO NOT FEED ATTENTION. FlashAttention wants Q in B-layout and K/V
# in A-layout, all int8. Reaching that from the projections' D/f16 output is a TRANSPOSE
# for Q -- B runs down columns where row_major, A and D run along rows -- and an int8
# A-conversion for K and V. Neither is a strided nest: no pair of strides makes a common
# run when the two sides are contiguous along different axes, and at int8 the A-layout
# atom is 4 bytes against the xDMA's 8-byte lane. comm/nest.py refuses both BY NAME. So
# attention's operands are staged by the datagen and the projections are a CHECKED DEAD
# END -- they prove the projection arithmetic without pretending to feed the next stage.
# Closing that gap needs the transposer path, not another block; see THE TRANSPOSE
# QUESTION below.
#
# WHAT THIS LAYER IS NOT. No causal mask -- every query attends to every key -- so it is
# decode-shaped, not prefill. No KV cache: Q, K and V are staged rather than carried from
# a previous step. And the FFN is a dense up-projection, not the MoE: MoeFFN wants all
# four clusters for its expert lanes and FlashAttention above already has them, so the
# two need a placement plan that belongs above this file.
#
# ======================================================================================
# RMSNORM: WHICH KERNEL, AND WHY THE CHOICE IS WORTH A KNOB
# ======================================================================================
#
# The two RMSNorms measured 110 us each -- 40% of cluster 0's busy time, more than the
# GEMMs and the reshapes together. All of that was the ONE SCALAR PER ROW, and it has
# since been attacked twice.
#
#   1. THE SCALAR LEFT THE CORE. StreamMap grew an RSQRT func, so the per-row
#      1/sqrt(mean) is computed in the datapath on the broadcast pass that had to
#      replicate the scalar anyway. That deleted an integer sqrt + reciprocal (six serial
#      `divu` on an FPU-less core) and 16 volatile stores per row: 7,717 -> 3,135 cc at
#      [32, 128], and the answer got CLOSER to the true 1/sqrt (1 ULP against 2-3).
#
#   2. THE FOLD CAN GO TOO, IF THE TILE IS TRANSPOSED. StreamReduce carries one FP32
#      accumulator per lane and folds lane k into acc[k] every beat, so a reduction ALONG
#      BEATS is free while one ACROSS the lanes of a beat is a serialised log-depth fold
#      that stalls the reader ~35 cc per row. Row-major [T, d] scatters a row's terms
#      across all 32 lanes and must fold; transposed [d, T] puts one TOKEN PER LANE, so
#      acc[t] already holds token t's whole sum and SIMD_RED_LANEWISE just emits it. The
#      scale then rides back as a STICKY seed beat instead of a replicated [T, d] plane.
#      1,073 cc against 3,135 -- a further 2.9x on the SIMD, paid for with two xDMA block
#      transposes on the other engine.
#
# NORM_PATH below picks between them. It is pinned to "row_major" because the transposed
# arm is three new things at once -- a new kernel entry point, a seed-adjacency contract,
# and two xDMA transposer nodes per norm. simd_rmsnorm_t_1cluster is the workload that has
# to go green first: it runs both kernels over one [32, 128] tile with four checks, and it
# has not been on RTL yet. Flipping this afterwards is a clean single-variable A/B over an
# otherwise identical graph.
#
# ======================================================================================
# THE TRANSPOSE QUESTION, WHICH IS THE SAME QUESTION IN THREE PLACES
# ======================================================================================
#
# Three separate things in this layer are "a transpose", and they are NOT the same
# operation. Confusing them is how a day gets lost:
#
#   AXIS EXCHANGE, the same [r, c] tensor stored [c, r]. This is what RMSNorm and softmax
#   want, and it is an ordinary Layout -- `Layout.COL_MAJOR` -- because it IS a bijection
#   on a fixed shape: the tensor is still (rows, cols), only the address of element [r][c]
#   moves. It is closed by the xDMA's 8x8 block transposer, a real unit in the datapath,
#   and comm/transfer.py plans it like any other gap.
#
#   BLOCKED-LAYOUT TRANSPOSE, row_major -> B. B runs contiguously DOWN COLUMNS while
#   row_major, A and D run along rows, so no pair of strides expresses it. This is NOT
#   the axis exchange above -- row_major -> col_major is a layout pair comm/transfer.py
#   plans on its own; this one is not expressible at any shape. It is what blocks
#   proj_q -> FA above. nest.py refuses it by name and points at the transposer kernels.
#
#   ADJACENT-PAIR SWAP, x[i] <-> x[i^1]. RoPE's, and the odd one out: it is a 2-BYTE
#   permutation inside one 8-byte TCDM word, which is BELOW the granularity the reader's
#   AGU can address at all -- the AGU places whole words and `lane_stride` moves whole
#   channels. No stride, and no transposer mode, expresses it; only a real byte-addressed
#   DMA does. Hence RoPE is two nodes, one per engine.
#
# AND TRANSPOSING DOES NOT ALWAYS HELP. For RMSNorm and softmax it turns a cross-lane
# REDUCTION into an along-beat one, which the accumulators do for free -- that is the
# whole win. RoPE has no reduction; its coupling is between the halves of a pair, and
# transposing only moves that from adjacent lanes to adjacent beats, where the cos and
# sin tables then need exchanging between the two outputs of a pair. One materialisation
# traded for another.
#
# ROPE IS NOT BUILT HERE, and WITH_ROPE below says so in one word rather than leaving it
# unmentioned. The goldens are already staged (rope_cos, rope_sin, rope_q_golden) and the
# blocks exist; what stops it being on by default is that it would change a graph that
# passes today, and the projections it would hang off are already a dead end, so it adds
# a checked branch rather than completing the chain.
# ======================================================================================

import argparse
import os
import sys

import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim/common")   # bingo_data_staging
sys.path.append(f"{ROOT_DIR}/util/sim/llm")      # the datagen and its goldens

import _bingo_paths  # noqa: F401,E402
from bingo_data_staging import DataStaging                              # noqa: E402
from bingo_dfg import BingoDFG                                          # noqa: E402
from bingo_kernel_args import SnaxBingoKernelIdma1dCopyArgs             # noqa: E402
from bingo_platform import (core_roles, guard_cluster_count,            # noqa: E402
                            load_cluster_cfg, parse_platform_cfg)
from libs import Ctx, DType, Layout, MemLevel, Pipeline, Port, PortSpec  # noqa: E402
from libs.block import (Dequantize, FlashAttention, Linear, Quantize,   # noqa: E402
                        RMSNorm, Reshape, Residual, fa_gather)
from libs.block.flash_attention import mesh_from_hwcfg                  # noqa: E402
from libs.verify import checks                                          # noqa: E402
from llm_layer_data import generate_layer_data, stage                   # noqa: E402

CHIPLET_ID = 0x00
# The heap the static-L1 pass must fit inside. SET IT: left unset the pass computes a peak,
# prints it without a bound and never fails, so an overflow reaches the simulation as one
# buffer quietly overlapping another's bytes.
L1_CAPACITY = 514816

# Which RMSNorm kernel, in one word. See the long note in the header: "row_major" is the
# RSQRT path (one node, no layout change), "auto" adds the transposed path wherever
# rows == 32 (LANEWISE reduce + sticky scale, 2.9x on the SIMD, two xDMA transposes).
# Pinned until simd_rmsnorm_t_1cluster has run the transposed kernel green on RTL.
NORM_PATH = "row_major"

# RoPE on Q and K. Off because it would change a graph that passes, and because the
# projections it hangs off are a checked dead end either way -- see the header.
WITH_ROPE = False

# Per-stage absolute tolerance on the fp16 compare. THESE TRACK THE DEQUANTISE: every
# projection is scaled back to the activation domain, so the tensors are O(1)-O(11) rather
# than O(1000), and a tolerance sized for the old scale would accept anything.
#
#   norm1 / norm2          worst delta 0.002 on |values| <= 2.0   -> 0.05, 25x margin
#   proj_q / ffn_up /      worst delta 0.043 on |values| <= 3.9   -> 0.25, ~6x margin.
#   layer_out              These quantise a DEVICE-computed normalisation, so ~150 of 4096
#                          int8 values land the other side of a rounding boundary and each
#                          rides through a k=128 accumulation. That error is real and a
#                          correct kernel cannot avoid it.
#   proj_o / resid1        0.125. Their inputs are all STAGED arrays, so the accumulation
#                          and the D-port narrowing are exact and only the dequant multiply
#                          rounds -- about 12 ulps of headroom at |values| ~ 10.
#
# Every one is a few percent of full scale, so a wrong layout, a wrong weight or a dropped
# stage still misses by far more than the tolerance.
_TOL = {"norm1": 0.05, "proj_q": 0.25, "proj_o": 0.125, "resid1": 0.125,
        "norm2": 0.05, "ffn_up": 0.25, "layer_out": 0.25}


def build_layer(ctx, p, data, hs, *, verbose=True):
    """Emit the whole layer into ctx's DFG, with a check on every stage.

    CHECKS ARE PER-STAGE AND ORDERED BEFORE THE NEXT STAGE'S WORK. Checking only the layer
    output would say "wrong" without saying where, and every stage narrows precision, so
    the margin that matters differs per stage.
    """
    T, d, h = p["tokens"], p["d_model"], p["d_hidden"]
    if h != d:
        # The second residual adds the up-projection (T, h) to resid1 (T, d). That is only
        # a residual when h == d; a real FFN gets there through the DOWN-projection, which
        # this layer does not build yet. Refuse rather than add two different widths --
        # numpy would broadcast the golden and the device would read past the buffer.
        raise ValueError(
            f"this layer needs d_hidden == d_model (got h={h}, d={d}): the second residual "
            f"adds the up-projection to resid1, and without a down-projection those are "
            f"only the same shape when the two widths match.")
    ncl = int(p["num_clusters"])
    mesh = (p["meshRow"], p["tileSize"], p["meshCol"])
    pipe = Pipeline(ctx, verbose=verbose)

    def L3(layout, dtype, shape, handle):
        """Bind a staged array as a port; the level is read off the handle."""
        return Port(PortSpec(layout, dtype, shape), handle, ())

    def check(tag, port, golden_key, elems):
        """Read a stage's output back and compare. Ordered on the port's own producer."""
        checks.readback_and_check(
            ctx.at(0), f"llm_{tag}", src=port.handle, golden=hs[golden_key],
            dtype=DType.F16, elems=elems, tol=_TOL[tag], after=port.ends[-1])

    # ALLOCATED ONCE. Two BingoMemAlloc objects with the same name are one buffer to the
    # emitter and two live ranges to the L1 packer, which would split the tensor in half.
    x_l1 = ctx.at(0).l1("layer_x", T * d * 2)
    # AND LOADED. An L1 buffer nothing writes is not zero -- TCDM is SimInit "none", so a
    # never-written word reads X, and an X in a branch makes the Snitch PC go X and the
    # hart die while the UART stays silent. The layer input has to arrive from L3 like any
    # other operand, and `ends` carries that load so the residual orders against it too.
    ld_x = ctx.at(0).node("Ld_layer_x", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(hs["x"], x_l1, T * d * 2))
    x_port = Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (T, d), mem_level=MemLevel.L1),
                  x_l1, (ld_x,))

    # ---- 1. the first normalisation ----------------------------------------------------
    n1_y = pipe.add(RMSNorm(rows=T, cols=d, cluster=0, path=NORM_PATH), name="norm1",
                    bind={"x": x_port}).result.outputs["y"]
    check("norm1", n1_y, "norm1_golden", T * d)

    # ---- 2. the Q/K/V projections -------------------------------------------------------
    rs1 = pipe.add(Reshape(rows=T, cols=d, src=Layout.ROW_MAJOR, dst=Layout.A, mesh=mesh,
                           dtype=DType.F16, cluster=0),
                   name="n1_to_a", bind={"x": n1_y})
    q1 = pipe.add(Quantize(rows=T, cols=d, inv_scale_f32bits=data["scale_n1_bits"],
                           layout=Layout.A, cluster=0),
                  name="n1_q", bind={"x": rs1.result.outputs["y"]})
    proj_dq = {}
    for nm in ("q", "k", "v"):
        pr = pipe.add(Linear(tokens=T, d_in=d, d_out=d, mesh=mesh, cluster=0),
                      name=f"proj_{nm}",
                      bind={"x": q1.result.outputs["y"],
                            "w": L3(Layout.B, DType.I8, (d, d), hs[f"w_{nm}"])})
        proj_dq[nm] = pipe.add(
            Dequantize(rows=T, cols=d, scale_f32bits=data["dq_proj_bits"],
                       layout=Layout.D, cluster=0),
            name=f"proj_{nm}_dq", bind={"x": pr.result.outputs["y"]})
    # One projection is checked, not three: they are the same block on the same input with
    # different weights, so a second failing check would say nothing the first did not.
    check("proj_q", proj_dq["q"].result.outputs["y"], "proj_q_golden", T * d)

    # ---- 3. attention -------------------------------------------------------------------
    # Br is pinned to the SIMD beat at 32, so FA's tile does NOT follow `tokens`; the
    # datagen stages its operands at fa_tile for the same reason.
    fat = int(p.get("fa_tile", 32))
    fa = pipe.add(FlashAttention(bc=fat, br=fat, dhead=d, nkv=ncl, clusters=ncl,
                                 decomp="kvsplit"),
                  name="attn",
                  bind={"q": L3(Layout.B, DType.I8, (fat, d), hs["fa_q"]),
                        "k": L3(Layout.A, DType.I8, (fat * ncl, d), hs["fa_k"]),
                        "v": L3(Layout.A, DType.I8, (fat, d), hs["fa_v"])})
    fa_gather(ctx.scope("attn"), fa.block.cfg, fa.result.extra["shards"], verify=False)
    # FA's own output is d32/int32 per cluster and its partials are un-folded, so there is
    # no fp16 tensor here to compare. This stage proves attention BUILDS, DISPATCHES and
    # TERMINATES on four clusters; the arithmetic is covered by fa_decode_4cluster.

    # ---- 4. the output projection and the first residual --------------------------------
    # THE OPERAND IS STAGED, NOT FA'S OUTPUT. FA writes int32 in the D-port scatter layout,
    # per cluster and un-folded; reaching int8 A-layout from that is an unpermute-and-
    # requantise step this layer does not have. `attn_ctx` is the reference context
    # quantised and laid out by the datagen, so proj_o and everything after it are checked
    # against goldens computed from the SAME tensor.
    o_proj = pipe.add(Linear(tokens=T, d_in=d, d_out=d, mesh=mesh, cluster=0),
                      name="proj_o",
                      bind={"x": L3(Layout.A, DType.I8, (T, d), hs["attn_ctx"]),
                            "w": L3(Layout.B, DType.I8, (d, d), hs["w_o"])})
    o_dq = pipe.add(Dequantize(rows=T, cols=d, scale_f32bits=data["dq_proj_o_bits"],
                               layout=Layout.D, cluster=0),
                    name="proj_o_dq", bind={"x": o_proj.result.outputs["y"]})
    check("proj_o", o_dq.result.outputs["y"], "proj_o_golden", T * d)
    rs_o = pipe.add(Reshape(rows=T, cols=d, src=Layout.D, dst=Layout.ROW_MAJOR, mesh=mesh,
                            dtype=DType.F16, cluster=0),
                    name="o_to_row_major", bind={"x": o_dq.result.outputs["y"]})
    res1 = pipe.add(Residual(rows=T, cols=d, cluster=0), name="resid1",
                    bind={"a": rs_o.result.outputs["y"], "b": x_port})
    check("resid1", res1.result.outputs["y"], "resid1_golden", T * d)

    # ---- 5. the feed-forward half -------------------------------------------------------
    n2_y = pipe.add(RMSNorm(rows=T, cols=d, cluster=0, path=NORM_PATH), name="norm2",
                    bind={"x": res1.result.outputs["y"]}).result.outputs["y"]
    check("norm2", n2_y, "norm2_golden", T * d)
    rs2 = pipe.add(Reshape(rows=T, cols=d, src=Layout.ROW_MAJOR, dst=Layout.A, mesh=mesh,
                           dtype=DType.F16, cluster=0),
                   name="n2_to_a", bind={"x": n2_y})
    q2 = pipe.add(Quantize(rows=T, cols=d, inv_scale_f32bits=data["scale_n2_bits"],
                           layout=Layout.A, cluster=0),
                  name="n2_q", bind={"x": rs2.result.outputs["y"]})
    ffn = pipe.add(Linear(tokens=T, d_in=d, d_out=h, mesh=mesh, cluster=0),
                   name="ffn_up",
                   bind={"x": q2.result.outputs["y"],
                         "w": L3(Layout.B, DType.I8, (d, h), hs["w_up_0"])})
    ffn_dq = pipe.add(Dequantize(rows=T, cols=h, scale_f32bits=data["dq_ffn_bits"],
                                 layout=Layout.D, cluster=0),
                      name="ffn_up_dq", bind={"x": ffn.result.outputs["y"]})
    check("ffn_up", ffn_dq.result.outputs["y"], "ffn_up_golden", T * h)

    # ---- 6. the second residual, and the layer output ----------------------------------
    rs_f = pipe.add(Reshape(rows=T, cols=h, src=Layout.D, dst=Layout.ROW_MAJOR, mesh=mesh,
                            dtype=DType.F16, cluster=0),
                    name="ffn_to_row_major", bind={"x": ffn_dq.result.outputs["y"]})
    res2 = pipe.add(Residual(rows=T, cols=h, cluster=0), name="resid2",
                    bind={"a": rs_f.result.outputs["y"],
                          "b": res1.result.outputs["y"]})
    check("layer_out", res2.result.outputs["y"], "ladder_out_golden", T * h)
    return res2.result.outputs["y"]


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
    hw = load_cluster_cfg(args.hwcfg)
    mesh = mesh_from_hwcfg(args.hwcfg, int(p.get("array_shape", 0)))
    p["meshRow"], p["tileSize"], p["meshCol"] = mesh
    plat = parse_platform_cfg(args.platformcfg)
    guard_cluster_count(p, plat, args.output_dir, args.output_offload_file_name)

    data = generate_layer_data(p)
    st = DataStaging(plat)
    hs = stage(st, data, p)

    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=plat["num_clusters_per_chiplet"],
                   num_cores_per_cluster=plat["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[CHIPLET_ID],
                   dep_tag_width=plat["dep_tag_width"])
    dfg.l1_capacity_bytes = L1_CAPACITY
    # `hw` is the cluster cfg the RTL was elaborated from. Blocks derive the cfg-dependent
    # kernel constants from it -- an xDMA junction's id is its POSITION in that cfg's list,
    # so a literal is silently wrong on a cluster with a different set.
    ctx = Ctx(dfg=dfg, mesh=mesh, roles=core_roles(), chiplet=CHIPLET_ID, hw=hw)

    build_layer(ctx, p, data, hs)

    if args.data_h:
        st.emit(args.data_h, args.output_dir)
    extra = [os.path.basename(str(args.data_h))] if args.data_h else None
    dfg.bingo_compile_dfg(
        # "(layer)" used to be the ladder's stage NAME here. Kept the string identical
        # while porting so the generated header could be diffed byte for byte against the
        # ladder-built one; renamed once that passed.
        app_name=(f"LLM layer, 4 clusters -- "
                  f"T={p['tokens']} d={p['d_model']} h={p['d_hidden']}"),
        output_dir=args.output_dir,
        output_file_name=args.output_offload_file_name,
        extra_include_header_list=extra)
    print(f"Generated: {os.path.join(args.output_dir, args.output_offload_file_name)}")


if __name__ == "__main__":
    main()
