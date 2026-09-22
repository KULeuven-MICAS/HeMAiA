# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The transformer layer, built one stage at a time.

THE BRING-UP LADDER. `build(..., stages=N)` emits the first N stages and checks each
one's output against its own golden. Six workloads call it with N = 1..6, so the first
rung that fails on hardware names the stage that broke rather than the layer.

ONE BUILDER, NOT SIX. Every rung is literally a prefix of the same code, so a rung cannot
disagree with the full layer about how a stage is built -- which is the failure mode a
ladder of copy-pasted apps has, and the one that wastes the most time: rung 3 passes,
rung 6 fails, and the difference turns out to be in the apps rather than in the hardware.

CHECKS ARE PER-STAGE AND ORDERED BEFORE THE NEXT STAGE'S WORK. Checking only the layer
output would say "wrong" without saying where, and every stage here narrows precision, so
the margin that matters differs per stage.

    1 norm    RMSNorm(x)
    2 proj    + reshape -> quantise -> Wq, Wk, Wv
    3 attn    + FlashAttention over 4 clusters, and its fold
    4 resid   + Wo, reshape back, residual add
    5 ffn     + RMSNorm, reshape, quantise, the up-projection
    6 layer   + reshape back and the second residual: the layer output
"""

import sys

from bingo_kernel_args import SnaxBingoKernelIdma1dCopyArgs
from libs import Ctx, DType, Layout, MemLevel, Pipeline, Port, PortSpec
from libs.block import (Dequantize, FlashAttention, Linear, Quantize, RMSNorm,
                        Reshape, Residual, fa_gather)
from libs.verify import checks

STAGE_NAMES = {1: "norm", 2: "proj", 3: "attn", 4: "resid", 5: "ffn", 6: "layer"}
MAX_STAGE = 6

# Per-stage absolute tolerance on the fp16 compare. THESE TRACK THE DEQUANTISE: every
# projection is now scaled back to the activation domain, so the tensors are O(1)-O(11)
# rather than O(1000), and a tolerance sized for the old scale would accept anything.
# Re-measured on the current chain over 300 per-row perturbations at the device's 0.12%
# rsqrt error (see simd_rmsnorm_1cluster):
#
#   norm1 / norm2          worst delta 0.002 on |values| <= 2.0   -> 0.05, 25x margin
#   proj_q / ffn_up /      worst delta 0.043 on |values| <= 3.9   -> 0.25, ~6x margin.
#   layer_out              These quantise a DEVICE-computed normalisation, so ~150 of 4096
#                          int8 values land the other side of a rounding boundary and each
#                          rides through a k=128 accumulation. That error is real and a
#                          correct kernel cannot avoid it.
#   proj_o / resid1        0.125. Their inputs are all STAGED arrays, so the accumulation
#                          and the D-port narrowing are exact (_to_fp16 reproduces the
#                          narrowing bit for bit) and only the dequant multiply rounds --
#                          about 12 ulps of headroom at |values| ~ 10.
#
# Every one of these is a few percent of full scale, so a wrong layout, a wrong weight or
# a dropped stage still misses by far more than the tolerance.
_TOL = {"norm1": 0.05, "proj_q": 0.25, "proj_o": 0.125, "resid1": 0.125,
        "norm2": 0.05, "ffn_up": 0.25, "layer_out": 0.25}


def build(ctx: Ctx, p: dict, data: dict, hs: dict, *, stages: int = MAX_STAGE,
          verbose: bool = True):
    """Emit the first `stages` stages into ctx's DFG. Returns {stage: result}."""
    if not 1 <= stages <= MAX_STAGE:
        raise ValueError(f"stages={stages} must be 1..{MAX_STAGE}")
    T, d, h = p["tokens"], p["d_model"], p["d_hidden"]
    if stages >= MAX_STAGE and h != d:
        # The second residual adds the up-projection (T, h) to resid1 (T, d). That is only
        # a residual when h == d; a real FFN gets there through the DOWN-projection, which
        # this ladder does not build yet. Refuse rather than add two different widths --
        # numpy would broadcast the golden and the device would read past the buffer.
        raise ValueError(
            f"stage 6 needs d_hidden == d_model (got h={h}, d={d}): the second residual "
            f"adds the up-projection to resid1, and without a down-projection those are "
            f"only the same shape when the two widths match.")
    ncl = int(p["num_clusters"])
    mesh = (p["meshRow"], p["tileSize"], p["meshCol"])
    pipe = Pipeline(ctx, verbose=verbose)
    out = {}

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
    x_port = Port(PortSpec(Layout.PACKED, DType.F16, (T, d), mem_level=MemLevel.L1),
                  x_l1, (ld_x,))

    # ---- 1. the first normalisation ----------------------------------------------------
    rms1 = pipe.add(RMSNorm(rows=T, cols=d, cluster=0), name="norm1",
                    bind={"x": x_port})
    out[1] = rms1
    check("norm1", rms1.result.outputs["y"], "norm1_golden", T * d)
    if stages == 1:
        return out

    # ---- 2. the Q/K/V projections -------------------------------------------------------
    # Reshape is a STAGE the layer names: between two blocks that both live in L1 there is
    # no load for the conversion to fold into, and the linker may not insert a node --
    # node creation order is dispatch order on this machine.
    rs1 = pipe.add(Reshape(rows=T, cols=d, src=Layout.PACKED, dst=Layout.A, mesh=mesh,
                           dtype=DType.F16, cluster=0),
                   name="n1_to_a", bind={"x": rms1.result.outputs["y"]})
    q1 = pipe.add(Quantize(rows=T, cols=d, inv_scale_f32bits=data["scale_n1_bits"],
                           layout=Layout.A, cluster=0),
                  name="n1_q", bind={"x": rs1.result.outputs["y"]})
    # EVERY GEMM IS FOLLOWED BY A DEQUANTISE. The array accumulates int8 x int8 in int32
    # and the D port narrows that to fp16, so the output still carries both operands'
    # quantisation scales -- at d=128 it lands in the thousands while everything it has to
    # rejoin is O(1). The scale back is 1/(scale_x * W_SCALE); see Dequantize for the two
    # fp16 limits that make this its own pass rather than something folded elsewhere.
    proj, proj_dq = {}, {}
    for nm in ("q", "k", "v"):
        proj[nm] = pipe.add(Linear(tokens=T, d_in=d, d_out=d, mesh=mesh, cluster=0),
                            name=f"proj_{nm}",
                            bind={"x": q1.result.outputs["y"],
                                  "w": L3(Layout.B, DType.I8, (d, d), hs[f"w_{nm}"])})
        proj_dq[nm] = pipe.add(
            Dequantize(rows=T, cols=d, scale_f32bits=data["dq_proj_bits"],
                       layout=Layout.D, cluster=0),
            name=f"proj_{nm}_dq", bind={"x": proj[nm].result.outputs["y"]})
    out[2] = proj_dq
    # One projection is checked, not three: they are the same block on the same input with
    # different weights, so a second failing check would say nothing the first did not.
    check("proj_q", proj_dq["q"].result.outputs["y"], "proj_q_golden", T * d)
    if stages == 2:
        return out

    # ---- 3. attention -------------------------------------------------------------------
    # FA wants Q in B-layout and K/V in A-layout, all int8. Reaching that from the
    # projections' D/f16 output is a TRANSPOSE for Q (B runs down columns) and an int8
    # A-conversion for K/V -- neither is a strided nest, and comm/nest.py refuses both by
    # name. So FA's operands are staged directly and the projections are checked above
    # instead. Closing this needs the transposer kernels, not another block.
    fa = pipe.add(FlashAttention(bc=T, br=T, dhead=d, nkv=ncl, clusters=ncl,
                                 decomp="kvsplit"),
                  name="attn",
                  bind={"q": L3(Layout.B, DType.I8, (T, d), hs["fa_q"]),
                        "k": L3(Layout.A, DType.I8, (T * ncl, d), hs["fa_k"]),
                        "v": L3(Layout.A, DType.I8, (T, d), hs["fa_v"])})
    fa_gather(ctx.scope("attn"), fa.block.cfg, fa.result.extra["shards"], verify=False)
    out[3] = fa
    # FA's own output is d32/int32 per cluster and its partials are un-folded, so there is
    # no fp16 tensor here to compare. The rung still proves attention BUILDS, DISPATCHES
    # and TERMINATES on four clusters, which is what the stage is being brought up for;
    # the arithmetic is covered by fa_decode_4cluster's own checks.
    if stages == 3:
        return out

    # ---- 4. the output projection and the first residual --------------------------------
    # THE OPERAND IS STAGED, NOT FA'S OUTPUT. FA writes int32 in the D-port scatter
    # layout, per cluster and un-folded; reaching int8 A-layout from that is the
    # unpermute-and-requantise step stage 3 does not have yet. `attn_ctx` is the
    # reference context quantised and laid out by the datagen, so proj_o and everything
    # after it are checked against goldens computed from the SAME tensor. Binding some
    # other array here (the K projection, say) leaves every later rung failing against a
    # golden that describes a different computation.
    o_proj = pipe.add(Linear(tokens=T, d_in=d, d_out=d, mesh=mesh, cluster=0),
                      name="proj_o",
                      bind={"x": L3(Layout.A, DType.I8, (T, d), hs["attn_ctx"]),
                            "w": L3(Layout.B, DType.I8, (d, d), hs["w_o"])})
    o_dq = pipe.add(Dequantize(rows=T, cols=d, scale_f32bits=data["dq_proj_o_bits"],
                               layout=Layout.D, cluster=0),
                    name="proj_o_dq", bind={"x": o_proj.result.outputs["y"]})
    check("proj_o", o_dq.result.outputs["y"], "proj_o_golden", T * d)
    rs_o = pipe.add(Reshape(rows=T, cols=d, src=Layout.D, dst=Layout.PACKED, mesh=mesh,
                            dtype=DType.F16, cluster=0),
                    name="o_to_packed", bind={"x": o_dq.result.outputs["y"]})
    res1 = pipe.add(Residual(rows=T, cols=d, cluster=0), name="resid1",
                    bind={"a": rs_o.result.outputs["y"], "b": x_port})
    out[4] = res1
    check("resid1", res1.result.outputs["y"], "resid1_golden", T * d)
    if stages == 4:
        return out

    # ---- 5. the feed-forward half -------------------------------------------------------
    rms2 = pipe.add(RMSNorm(rows=T, cols=d, cluster=0), name="norm2",
                    bind={"x": res1.result.outputs["y"]})
    check("norm2", rms2.result.outputs["y"], "norm2_golden", T * d)
    rs2 = pipe.add(Reshape(rows=T, cols=d, src=Layout.PACKED, dst=Layout.A, mesh=mesh,
                           dtype=DType.F16, cluster=0),
                   name="n2_to_a", bind={"x": rms2.result.outputs["y"]})
    q2 = pipe.add(Quantize(rows=T, cols=d, inv_scale_f32bits=data["scale_n2_bits"],
                           layout=Layout.A, cluster=0),
                  name="n2_q", bind={"x": rs2.result.outputs["y"]})
    # A dense up-projection, not MoeFFN: MoeFFN owns all four clusters for its expert
    # lanes and FlashAttention above already has them. Two blocks that both want every
    # cluster need a placement plan, which is the framework's job above this layer.
    ffn = pipe.add(Linear(tokens=T, d_in=d, d_out=h, mesh=mesh, cluster=0),
                   name="ffn_up",
                   bind={"x": q2.result.outputs["y"],
                         "w": L3(Layout.B, DType.I8, (d, h), hs["w_up_0"])})
    ffn_dq = pipe.add(Dequantize(rows=T, cols=h, scale_f32bits=data["dq_ffn_bits"],
                                 layout=Layout.D, cluster=0),
                      name="ffn_up_dq", bind={"x": ffn.result.outputs["y"]})
    out[5] = ffn_dq
    check("ffn_up", ffn_dq.result.outputs["y"], "ffn_up_golden", T * h)
    if stages == 5:
        return out

    # ---- 6. the second residual, and the layer output ----------------------------------
    rs_f = pipe.add(Reshape(rows=T, cols=h, src=Layout.D, dst=Layout.PACKED, mesh=mesh,
                            dtype=DType.F16, cluster=0),
                    name="ffn_to_packed", bind={"x": ffn_dq.result.outputs["y"]})
    res2 = pipe.add(Residual(rows=T, cols=h, cluster=0), name="resid2",
                    bind={"a": rs_f.result.outputs["y"],
                          "b": res1.result.outputs["y"]})
    out[6] = res2
    check("layer_out", res2.result.outputs["y"], "ladder_out_golden", T * h)
    return out
