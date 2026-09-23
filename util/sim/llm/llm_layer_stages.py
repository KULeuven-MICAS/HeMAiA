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

from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs,
                              SnaxBingoKernelSimdRmsnormF16F16Args)
from libs import (Ctx, DType, Layout, MemLevel, Pipeline, Port, PortSpec,
                  at_offset)
from libs.block import (Dequantize, FlashAttention, Linear, Quantize, RMSNorm,
                        Reshape, Residual, fa_gather, shard_rows)
from libs.verify import checks

STAGE_NAMES = {1: "norm", 2: "proj", 3: "attn", 4: "resid", 5: "ffn", 6: "layer"}
# The LAST check each rung emits. Rung 3 adds no check of its own -- FlashAttention's
# output is un-folded d32/int32 and there is no fp16 tensor to compare -- so it keeps
# rung 2's. `verify="final"` emits only this one, which is what a timing run wants: a
# host compare costs ~3,370 us against ~400 us for the whole layer's compute, so leaving
# the per-stage checks in makes the trace 98% verification and measures the scaffold.
FINAL_CHECK = {1: "norm1", 2: "proj_q", 3: "proj_q",
               4: "resid1", 5: "ffn_up", 6: "layer_out"}
VERIFY_MODES = ("all", "final", "slices", "none")
MAX_STAGE = 6

# WHICH RMSNORM KERNEL THE LADDER RUNS. One word, because this is an A/B and the two arms
# have to be swappable over an otherwise identical graph.
#
#   "rowmajor"  reduce(SUMSQ) -> bcast carrying StreamMap(1/D, RSQRT) -> ew2(MUL). One
#               node, no layout change. The RSQRT func already took this path from
#               7,717 cc to 3,135 at [32, 128] by deleting the core's scalar epilogue.
#   "auto"      transposed wherever rows == 32: LANEWISE reduce, sticky scale, no
#               cross-lane fold and no broadcast plane -- 1,073 cc, a further 2.9x on the
#               SIMD -- at the cost of two xDMA block transposes around it.
#
# IT IS PINNED TO "rowmajor" ON PURPOSE. The transposed arm is three things at once that
# have never run on this machine: a new kernel entry point
# (__snax_bingo_kernel_simd_rmsnorm_t_f16_f16), a new seed-adjacency contract, and two new
# xDMA transposer nodes per norm. Turning it on here would change a graph that PASSES
# today, while the token-parallel arm is still being debugged, and any new failure would
# then have two candidate causes.
#
# WHAT HAS TO HAPPEN BEFORE FLIPPING IT to "auto". The transposed kernel has no test
# vehicle yet: simd_rmsnorm_1cluster sweeps rows {1, 2, 4, 8} and the transposed path needs
# rows == 32 exactly, so that workload cannot reach it at any of its twelve configs. It
# needs a rows == 32 arm -- checking the xDMA transpose against a reference, the norm
# against the exact 1/sqrt golden, and the round trip back to row-major -- which is what
# the snax reference app does and what proves the adjacency contract holds on silicon.
# After that this is a clean single-variable A/B over an otherwise identical graph.
NORM_PATH = "rowmajor"

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
          verify: str = "all", shard: str = "none", verbose: bool = True):
    """Emit the first `stages` stages into ctx's DFG. Returns {stage: result}.

    `verify` picks how much checking to emit:
      "all"   every stage's output is read back and compared -- the bring-up default.
      "final" only this rung's last check, so the run still proves it computed the right
              answer while the trace measures the layer instead of the compares.
      "none"  no checks at all. Only for a timing run whose correctness some other run
              has already established.
    """
    if not 1 <= stages <= MAX_STAGE:
        raise ValueError(f"stages={stages} must be 1..{MAX_STAGE}")
    if verify not in VERIFY_MODES:
        raise ValueError(f"verify={verify!r} must be one of {VERIFY_MODES}")
    if verify == "slices":
        raise ValueError(
            "verify='slices' is a token_parallel diagnostic: it checks each cluster's "
            "finished slice in place. The single-cluster ladder has one slice, which is "
            "what verify='final' already checks.")
    if shard not in ("none", "rows"):
        raise ValueError(f"shard={shard!r} must be 'none' or 'rows'")
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
        if verify == "none" or (verify == "final" and tag != FINAL_CHECK[stages]):
            return
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

    def rmsnorm(name, src, *, on_l1, after=()):
        """RMSNorm over `T` rows, on one cluster or split across all of them.

        WHY THIS ONE. Measured on RTL, the two RMSNorms cost 110 us each -- 40% of
        cluster 0's busy time and 26% of every engine's, more than the GEMMs and the
        reshapes together. The reason is in the kernel: the multi-row path takes an
        integer sqrt and reciprocal PER ROW and then splats the result across a 64-byte
        beat with 16 volatile stores, so ~60 us of each call is a scalar loop over 32
        rows on a core with no FPU. Rows are independent, so that loop divides.
        """
        if shard == "none":
            return pipe.add(RMSNorm(rows=T, cols=d, cluster=0, path=NORM_PATH), name=name,
                            bind={"x": src}).result.outputs["y"]

        def make(g, in_h, nrows, dep, out_h):
            return g.node(f"Rmsnorm_{name}", ctx.simd,
                          "__snax_bingo_kernel_simd_rmsnorm_f16_f16",
                          SnaxBingoKernelSimdRmsnormF16F16Args(
                              input_addr=in_h, output_addr=out_h,
                              rows=nrows, cols=d), dep)

        out, ends = shard_rows(ctx, src=src.handle if on_l1 else hs[src], rows=T, cols=d,
                               clusters=list(range(ncl)), make=make, root=0,
                               name=name, after=after, src_on_l1=on_l1)
        return Port(PortSpec(Layout.PACKED, DType.F16, (T, d), mem_level=MemLevel.L1),
                    out, tuple(ends), cluster=0, name="y")

    # ---- 1. the first normalisation ----------------------------------------------------
    # sharded, norm1 reads its slices straight from L3 -- no broadcast on the way in
    n1_y = rmsnorm("norm1", "x" if shard == "rows" else x_port,
                   on_l1=(shard != "rows"))
    out[1] = n1_y
    check("norm1", n1_y, "norm1_golden", T * d)
    if stages == 1:
        return out

    # ---- 2. the Q/K/V projections -------------------------------------------------------
    # Reshape is a STAGE the layer names: between two blocks that both live in L1 there is
    # no load for the conversion to fold into, and the linker may not insert a node --
    # node creation order is dispatch order on this machine.
    rs1 = pipe.add(Reshape(rows=T, cols=d, src=Layout.PACKED, dst=Layout.A, mesh=mesh,
                           dtype=DType.F16, cluster=0),
                   name="n1_to_a", bind={"x": n1_y})
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
    # norm2's input is an intermediate in cluster 0's L1, so a sharded build pulls the
    # slices across the fabric with each cluster's own xDMA rather than from L3
    # THE PULLS MUST WAIT FOR THE RESIDUAL. Each cluster reads its slice out of cluster
    # 0's L1, and nothing in the fabric would tell it the buffer is still being written.
    # Without this edge the sharded build races and the race is silent.
    n2_y = rmsnorm("norm2", res1.result.outputs["y"], on_l1=True,
                   after=tuple(res1.result.outputs["y"].ends))
    check("norm2", n2_y, "norm2_golden", T * d)
    rs2 = pipe.add(Reshape(rows=T, cols=d, src=Layout.PACKED, dst=Layout.A, mesh=mesh,
                           dtype=DType.F16, cluster=0),
                   name="n2_to_a", bind={"x": n2_y})
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


def token_parallel(ctx: Ctx, p: dict, data: dict, hs: dict, *, verify: str = "final",
                   verbose: bool = True):
    """The whole layer with the TOKENS split across the clusters, one slice each.

    WHY THIS DECOMPOSITION AND NOT ANOTHER. Measured, the single-cluster layer puts 543 us
    on cluster 0 and ~93 us on each of the other three; only FlashAttention uses the
    machine. Of the axes available:

      TOKENS (M)  every row is independent through norm, quantise, GEMM, dequantise and
                  the residual -- the whole chain except attention. The catch is
                  granularity: an A-layout tile is meshRow=16 rows, so T must be at least
                  16*clusters for the split to land on tile boundaries. T=64 over 4
                  clusters gives exactly one tile each.
      N           d/meshCol = 8 tiles would also divide, but every cluster then needs the
                  FULL activation, so it costs a broadcast per GEMM.
      K           divides too, and costs an all-reduce the library does not have.

    Tokens win because they need NO cross-cluster traffic in the middle: each cluster
    loads its own slice from L3 and the only transfer is the gather at the end, for the
    check. That also avoids the cluster-to-cluster pull path, which does not work.

    Attention is NOT token-split here. Its Br is pinned to 32 and its operands are staged,
    so it stays as it is -- already four-way over the KV axis.
    """
    T, d, h = p["tokens"], p["d_model"], p["d_hidden"]
    ncl = int(p["num_clusters"])
    mr = p["meshRow"]
    mesh = (mr, p["tileSize"], p["meshCol"])
    if verify == "all":
        raise ValueError(
            "token_parallel supports verify='final', 'slices' or 'none': a per-STAGE "
            "check would have to gather that stage's four slices first, and the gather "
            "is the one thing this decomposition exists to avoid doing in the middle. "
            "'slices' is the diagnostic in between -- it checks each cluster's finished "
            "slice where it lies, before the gather moves it.")
    if T % (mr * ncl):
        raise ValueError(
            f"tokens={T} over {ncl} clusters needs each slice to be whole A-layout tiles "
            f"of meshRow={mr} rows, so tokens must be a multiple of {mr * ncl}. At "
            f"tokens={T} the slice is {T // ncl} rows, which is {T / ncl / mr:.2f} tiles "
            f"and the reshape into A-layout has no valid nest.")
    NR = T // ncl
    pipe = Pipeline(ctx, verbose=verbose)

    def L3(layout, dtype, shape, handle):
        return Port(PortSpec(layout, dtype, shape), handle, ())

    # ATTENTION IS BUILT ONCE, NOT PER SLICE. It is already four-way over the KV axis and
    # its Br is pinned, so token-splitting it would mean four attentions each using four
    # clusters. It is here so this build is the same layer as the single-cluster one and
    # the two are comparable; its operands are staged, exactly as they are there.
    fat = int(p.get("fa_tile", 32))
    fa = pipe.add(FlashAttention(bc=fat, br=fat, dhead=d, nkv=ncl, clusters=ncl,
                                 decomp="kvsplit"),
                  name="attn",
                  bind={"q": L3(Layout.B, DType.I8, (fat, d), hs["fa_q"]),
                        "k": L3(Layout.A, DType.I8, (fat * ncl, d), hs["fa_k"]),
                        "v": L3(Layout.A, DType.I8, (fat, d), hs["fa_v"])})
    fa_gather(ctx.scope("attn"), fa.block.cfg, fa.result.extra["shards"], verify=False)

    # THE GATHER GOES TO L3, NOT TO THE ROOT'S L1, and each cluster uses its OWN iDMA.
    #
    # The obvious design -- every cluster pushes its slice into one buffer on cluster 0 --
    # is what this built first, and it WEDGED THE MACHINE. Writing another cluster's TCDM
    # opens a receive window on the destination, and three of those plus FlashAttention's
    # own gather into the same cluster is more than the adapter carries: the run died with
    # xdma_grant_manager, xdma_finish_manager and wide_send stall watchdogs firing on all
    # four clusters at once (16,384 cycles of no progress each). It is the same shape as
    # the known single-from_remote-context deadlock.
    #
    # An L1 -> L3 iDMA has none of that. It is an ordinary outbound write on a path every
    # readback already uses, the four are to disjoint L3 offsets so they need no ordering
    # between them, and they run on the DM cores, which are idle by then. The result also
    # ends up where the host wants it, so the check reads it in place and the separate
    # store-to-L3 node disappears -- one fewer host task per check.
    gathered = ctx.l3("layer_out", T * h * 2)
    ends = []

    for c in range(ncl):
        g = ctx.at(c)
        r0 = c * NR
        sfx = f"c{c}"
        # --- this cluster's slice of the layer input, straight from L3 ---
        x_l1 = g.l1(f"layer_x_{sfx}", NR * d * 2)
        ld_x = g.node(f"Ld_x_{sfx}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                      SnaxBingoKernelIdma1dCopyArgs(
                          at_offset(hs["x"], r0 * d * 2), x_l1, NR * d * 2))
        x_port = Port(PortSpec(Layout.PACKED, DType.F16, (NR, d), mem_level=MemLevel.L1),
                      x_l1, (ld_x,))

        # --- norm -> reshape -> quantise -> the three projections ---
        n1 = pipe.add(RMSNorm(rows=NR, cols=d, cluster=c, path=NORM_PATH), name=f"norm1_{sfx}",
                      bind={"x": x_port})
        a1 = pipe.add(Reshape(rows=NR, cols=d, src=Layout.PACKED, dst=Layout.A, mesh=mesh,
                              dtype=DType.F16, cluster=c),
                      name=f"n1_to_a_{sfx}", bind={"x": n1.result.outputs["y"]})
        q1 = pipe.add(Quantize(rows=NR, cols=d, inv_scale_f32bits=data["scale_n1_bits"],
                               layout=Layout.A, cluster=c),
                      name=f"n1_q_{sfx}", bind={"x": a1.result.outputs["y"]})
        for nm in ("q", "k", "v"):
            pr = pipe.add(Linear(tokens=NR, d_in=d, d_out=d, mesh=mesh, cluster=c),
                          name=f"proj_{nm}_{sfx}",
                          bind={"x": q1.result.outputs["y"],
                                "w": L3(Layout.B, DType.I8, (d, d), hs[f"w_{nm}"])})
            pipe.add(Dequantize(rows=NR, cols=d, scale_f32bits=data["dq_proj_bits"],
                                layout=Layout.D, cluster=c),
                     name=f"proj_{nm}_dq_{sfx}", bind={"x": pr.result.outputs["y"]})

        # --- output projection, from the staged context, and the first residual ---
        op = pipe.add(Linear(tokens=NR, d_in=d, d_out=d, mesh=mesh, cluster=c),
                      name=f"proj_o_{sfx}",
                      bind={"x": L3(Layout.A, DType.I8, (NR, d),
                                    at_offset(hs["attn_ctx"], r0 * d)),
                            "w": L3(Layout.B, DType.I8, (d, d), hs["w_o"])})
        odq = pipe.add(Dequantize(rows=NR, cols=d, scale_f32bits=data["dq_proj_o_bits"],
                                  layout=Layout.D, cluster=c),
                       name=f"proj_o_dq_{sfx}", bind={"x": op.result.outputs["y"]})
        orp = pipe.add(Reshape(rows=NR, cols=d, src=Layout.D, dst=Layout.PACKED, mesh=mesh,
                               dtype=DType.F16, cluster=c),
                       name=f"o_to_packed_{sfx}", bind={"x": odq.result.outputs["y"]})
        r1 = pipe.add(Residual(rows=NR, cols=d, cluster=c), name=f"resid1_{sfx}",
                      bind={"a": orp.result.outputs["y"], "b": x_port})

        # --- the feed-forward half ---
        n2 = pipe.add(RMSNorm(rows=NR, cols=d, cluster=c, path=NORM_PATH), name=f"norm2_{sfx}",
                      bind={"x": r1.result.outputs["y"]})
        a2 = pipe.add(Reshape(rows=NR, cols=d, src=Layout.PACKED, dst=Layout.A, mesh=mesh,
                              dtype=DType.F16, cluster=c),
                      name=f"n2_to_a_{sfx}", bind={"x": n2.result.outputs["y"]})
        q2 = pipe.add(Quantize(rows=NR, cols=d, inv_scale_f32bits=data["scale_n2_bits"],
                               layout=Layout.A, cluster=c),
                      name=f"n2_q_{sfx}", bind={"x": a2.result.outputs["y"]})
        up = pipe.add(Linear(tokens=NR, d_in=d, d_out=h, mesh=mesh, cluster=c),
                      name=f"ffn_up_{sfx}",
                      bind={"x": q2.result.outputs["y"],
                            "w": L3(Layout.B, DType.I8, (d, h), hs["w_up_0"])})
        udq = pipe.add(Dequantize(rows=NR, cols=h, scale_f32bits=data["dq_ffn_bits"],
                                  layout=Layout.D, cluster=c),
                       name=f"ffn_up_dq_{sfx}", bind={"x": up.result.outputs["y"]})
        urp = pipe.add(Reshape(rows=NR, cols=h, src=Layout.D, dst=Layout.PACKED, mesh=mesh,
                               dtype=DType.F16, cluster=c),
                       name=f"ffn_to_packed_{sfx}", bind={"x": udq.result.outputs["y"]})
        r2 = pipe.add(Residual(rows=NR, cols=h, cluster=c), name=f"resid2_{sfx}",
                      bind={"a": urp.result.outputs["y"], "b": r1.result.outputs["y"]})

        # --- this slice out to its place in the L3 result, on this cluster's own DM core ---
        slice_b = NR * h * 2
        last = r2.result.outputs["y"].ends[-1]
        st = g.node(f"St_out_{sfx}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(
                        r2.result.outputs["y"].handle,
                        at_offset(gathered, c * slice_b), slice_b), last)
        ends.append(st)

        if verify == "slices":
            # THE SLICE ON ITS OWN, which separates the two things a wrong layer output
            # cannot tell apart: a cluster that computed its rows wrongly, and a transfer
            # that reported success without moving anything. It costs no extra transfer --
            # the slice is already in L3 at a known offset, so this is a compare node and
            # nothing else.
            checks.check_out(
                ctx.at(0), f"Check_slice_{sfx}",
                golden=at_offset(hs["ladder_out_golden"], c * slice_b),
                got=at_offset(gathered, c * slice_b),
                dtype=DType.F16, elems=NR * h, tol=_TOL["layer_out"],
                after=st, label=f"slice_{sfx}")

    if verify != "none":
        # ORDERED ON EVERY SLICE, not just the last one. This reads the whole gathered
        # buffer, so it has to wait for all four writers; depending on one of them leaves
        # the other three racing and the check reads whatever was there.
        checks.check_out(
            ctx.at(0), "Check_llm_layer_out", golden=hs["ladder_out_golden"],
            got=gathered, dtype=DType.F16, elems=T * h, tol=_TOL["layer_out"],
            after=tuple(ends), label="llm_layer_out")
    return {"slices": ncl, "rows_each": NR, "out": gathered}
