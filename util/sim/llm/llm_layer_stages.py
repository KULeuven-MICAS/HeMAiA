# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The transformer layer as a BRING-UP LADDER -- a debugging instrument, not the app.

`build(..., stages=N)` emits the first N stages and checks each one against its own
golden, so the first rung that fails on hardware names the stage that broke rather than
the layer. `token_parallel(...)` is the other decomposition experiment: the same layer
with the TOKENS split four ways.

THIS IS NOT WHERE THE DELIVERABLE LAYER LIVES. That is
workloads/llm_layer_4cluster/main_bingo.py, which builds its own graph in one pass and
does not import this module. The two were verified byte-identical at the port -- same
generated header, same data header -- so this file is free to grow rungs and arms without
touching the app. The rung workloads (llm_s1_norm .. llm_s5_ffn) and the decomposition
arms (_shard, _tp, _t64, _perf) are gitignored for the same reason: they are regenerated
from here whenever this changes, so tracking them is pure churn.

ONE BUILDER, NOT SIX. Every rung is literally a prefix of the same code, so a rung cannot
disagree with the full layer about how a stage is built -- the failure mode a ladder of
copy-pasted apps has, and the one that wastes the most time: rung 3 passes, rung 6 fails,
and the difference turns out to be in the apps rather than in the hardware.

CHECKS ARE PER-STAGE AND ORDERED BEFORE THE NEXT STAGE'S WORK. Checking only the layer
output would say "wrong" without saying where, and every stage here narrows precision, so
the margin that matters differs per stage.

======================================================================================
THE DATAFLOW, AND WHAT EACH RUNG ADDS TO IT
======================================================================================

Layouts on every arrow, because none of them faults when wrong -- a mismatched one is a
permutation that computes a well-formed scrambled answer, and on random test data the
golden is scrambled identically and it PASSES.

  rung                                                            layout / precision
  ----  ------------------------------------------------  ----------------------------
   1    x -> Ld_layer_x -> RMSNorm norm1                   row_major/f16 throughout
   2       -> Reshape n1_to_a -> Quantize n1_q             row_major/f16 -> A/f16 -> A/i8
           -> Linear proj_q|k|v -> Dequantize              A/i8 x B/i8 -> D/f16
   3    FlashAttention, 4 clusters, KV-split + in-fabric   B/i8, A/i8, A/i8 -> d32/i32
         fold. Operands STAGED, not taken from rung 2.
   4    Linear proj_o -> Dequant -> Reshape o_to_row_major A/i8 -> D/f16 -> row_major/f16
         -> Residual resid1 (+ x, the skip)
   5    RMSNorm norm2 -> Reshape n2_to_a -> Quantize n2_q  row_major -> A/f16 -> A/i8
         -> Linear ffn_up -> Dequantize
   6    Reshape ffn_to_row_major -> Residual resid2        D/f16 -> row_major/f16

THE THREE THINGS THAT SHAPE IT, stated once here and argued in full in the app:

  RESHAPES ARE FP16 AND THEY ARE STAGES. Into or out of A-layout needs an 8-byte run
  contiguous on both sides -- four features at fp16, two bytes at int8 -- so `quantise`
  comes AFTER the reshape and never before. And the linker may not insert one: node
  creation order is dispatch order here, so an injected node would move the schedule.

  EVERY GEMM IS FOLLOWED BY A DEQUANTISE, because the D port narrows int32 to fp16 while
  the output still carries both operands' quantisation scales. Two fp16 limits bracket
  the GEMM -- the D-port narrow at 65504, and RMSNorm's SUM(x^2) which is itself narrowed
  to fp16 before the rsqrt sees it -- and a factor of a thousand sits between them.

  RUNG 2 IS A DEAD END BY CONSTRUCTION. FA wants Q in B-layout, and row_major -> B is a
  TRANSPOSE that no pair of strides expresses (B runs down columns; row_major, A and D run
  along rows); K/V want int8 A-layout, whose atom is 4 bytes against the xDMA's 8-byte
  lane. nest.py refuses both by name, so attention's operands are staged and rung 2 is
  checked rather than forwarded.

======================================================================================
WHAT CHANGED UNDER THIS LADDER, AND WHAT IT MEANS FOR THE RUNGS
======================================================================================

RMSNORM (rungs 1 and 5) lost its core round trip. StreamMap grew an RSQRT func, so the
per-row 1/sqrt(mean) rides the broadcast pass that had to replicate the scalar anyway --
7,717 -> 3,135 cc at [32, 128], and more accurate than the integer sqrt+reciprocal it
replaced. A transposed variant removes the cross-lane fold as well (1,073 cc), at the
cost of two xDMA block transposes; NORM_LAYOUT below selects between them.

ROPE IS STILL NOT IN THE CHAIN, and that is a layout fact rather than an omission. Its
partner permutation is x[i] <-> x[i^1] -- a 2-BYTE reorder inside one 8-byte TCDM word,
which is below the granularity the reader AGU can address at all. Only a real
byte-addressed DMA does it, so RoPE is two nodes (a DM-core swap, then one fused SIMD
task). The goldens are staged; wiring it in would hang a checked branch off rung 2's
dead end rather than complete anything.

THE WORD "TRANSPOSE" MEANS THREE DIFFERENT THINGS HERE. An AXIS EXCHANGE (the same
[r, c] tensor stored [c, r]) is what RMSNorm wants and is an ordinary Layout,
`Layout.COL_MAJOR`, closed by the xDMA block transposer. A BLOCKED-LAYOUT transpose
(row_major -> B) is what blocks rung 2. An
ADJACENT-PAIR SWAP is RoPE's and is below AGU granularity. And transposing only pays
where a cross-lane REDUCTION exists to convert -- RoPE has none, so it gains nothing.
"""

import sys

from bingo_kernel_args import (SnaxBingoKernelIdma1dCopyArgs,
                              SnaxBingoKernelSimdRmsnormArgs)
from libs import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Pipeline,
                  Port, PortSpec, at_offset)
from libs.block import (Dequantize, FlashAttention, Linear, Quantize, RMSNorm,
                        Reshape, Residual, fa_gather, shard_rows)
from libs.verify import checks

class _ShardedNorm(Block):
    """RMSNorm with its `rows` split across the clusters, one slice each.

    A BLOCK, so that it is emitted in the order it was added like every other stage. The
    pipeline resolves every boundary before it emits anything, so work built outside it
    lands wherever the application happened to call it -- and node creation order is
    dispatch order on this machine.

    Measured on RTL, the two RMSNorms cost 110 us each -- 40% of cluster 0's busy time and
    26% of every engine's, more than the GEMMs and the reshapes together. The reason is in
    the kernel: the multi-row path takes an integer sqrt and reciprocal PER ROW and then
    splats the result across a 64-byte beat with 16 volatile stores, so ~60 us of each
    call is a scalar loop over 32 rows on a core with no FPU. Rows are independent, so
    that loop divides.
    """

    name = "rmsnorm_sharded"

    def __init__(self, ctx, rows, cols, clusters, node_name, hs, *, on_l1, l3_key):
        self.ctx, self.rows, self.cols = ctx, rows, cols
        self.clusters, self.node_name, self.hs = clusters, node_name, hs
        self.on_l1, self.l3_key = on_l1, l3_key

    @property
    def _spec(self) -> PortSpec:
        return PortSpec(Layout.ROW_MAJOR, DType.F16, (self.rows, self.cols),
                        mem_level=MemLevel.L1)

    @property
    def inputs(self) -> dict:
        return {"x": self._spec} if self.on_l1 else {}

    @property
    def outputs(self) -> dict:
        return {"y": self._spec}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        d, name = self.cols, self.node_name

        def make(g, in_h, nrows, dep, out_h):
            return g.node(f"Rmsnorm_{name}", self.ctx.simd,
                          "__snax_bingo_kernel_simd_rmsnorm",
                          SnaxBingoKernelSimdRmsnormArgs(
                              input_addr=in_h, output_addr=out_h,
                              rows=nrows, cols=d), dep)

        src = bound["x"] if self.on_l1 else None
        # THE ORDERING GOES IN THROUGH `after`, not through the linker. shard_rows hangs
        # every cluster's load (or the root's scatter) off it, so the producer is already
        # an ancestor of each slice by the time the linker sees this block -- which is why
        # the input port below reports no ends: there is nothing left for it to join.
        out, ends = shard_rows(
            self.ctx, src=src.handle if self.on_l1 else self.hs[self.l3_key],
            rows=self.rows, cols=d, clusters=list(range(self.clusters)), make=make,
            root=0, name=name, after=tuple(src.ends) if self.on_l1 else (),
            src_on_l1=self.on_l1)
        port = Port(self._spec, out, tuple(ends), cluster=0, name="y")
        return BlockResult(
            outputs={"y": port},
            inputs={"x": Port(self._spec, src.handle, (), name="x")} if self.on_l1 else {},
            nodes=list(ends))


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

# WHICH RMSNORM KERNEL THE LADDER RUNS. Stating a layout PINS the boundary, so this one
# word decides the kernel over an otherwise identical graph -- which is what makes it an
# A/B. Leaving the norm's layouts unset instead lets libs/block/simd/norm.py pick.
#
#   row_major   reduce(SUMSQ) -> bcast carrying StreamMap(1/D, RSQRT) -> ew2(MUL). One
#               node, no layout change, 3,135 cc at [32, 128].
#   unpinned    the col_major kernel wherever rows == 32: LANEWISE reduce, sticky scale,
#               no cross-lane fold and no broadcast plane -- 1,073 cc, 2.9x on the SIMD --
#               at the cost of two xDMA block transposes around it.
#
# IT IS PINNED ON PURPOSE. The col_major arm is three things at once that have not run on
# this machine: a second kernel entry point (the col_major arm of
# __snax_bingo_kernel_simd_rmsnorm), a seed-adjacency contract, and two xDMA transposer
# nodes per norm. Unpinning would change a graph that PASSES while the token-parallel arm
# is still being debugged, and any new failure would then have two candidate causes.
#
# WHAT HAS TO HAPPEN BEFORE UNPINNING. simd_rmsnorm_t_1cluster is the test vehicle:
# [32, 128], both kernels over one tile, four checks -- the input transpose bit-exact, the
# norm against the exact 1/sqrt golden, the round trip back to row-major, and the
# row-major arm over the same x. It has NOT been run on RTL. Once it is green, dropping
# these two arguments is a clean single-variable A/B. (simd_rmsnorm_1cluster cannot reach
# the col_major path at all: it sweeps rows {1, 2, 4, 8} and LANEWISE needs rows == 32.)
NORM_LAYOUT = Layout.ROW_MAJOR

# Per-stage absolute tolerance on the fp16 compare. THESE TRACK THE DEQUANTISE: every
# projection is scaled back to the activation domain, so the tensors are O(1)-O(11), and a
# tolerance sized for an unscaled O(1000) tensor would accept anything.
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

    def check(tag, ref, golden_key, elems):
        """Read a stage's output back and compare. Ordered on the port's own producer.

        REGISTERED, NOT BUILT: the pipeline resolves the whole chain before it emits any
        of it, so `ref` has no buffer while the chain is still being written. raw() keeps
        this at the point it appears here, so the dispatch order is the order on the page.
        """
        if verify == "none" or (verify == "final" and tag != FINAL_CHECK[stages]):
            return
        def go():
            port = ref.port if hasattr(ref, "port") else ref
            checks.readback_and_check(
                ctx.at(0), f"llm_{tag}", src=port.handle, golden=hs[golden_key],
                dtype=DType.F16, elems=elems, tol=_TOL[tag], after=port.ends[-1])
        pipe.raw(go, f"check_{tag}")

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

    def rmsnorm(name, src, *, on_l1):
        """RMSNorm over `T` rows, on one cluster or split across all of them.

        WHY THIS ONE. Measured on RTL, the two RMSNorms cost 110 us each -- 40% of
        cluster 0's busy time and 26% of every engine's, more than the GEMMs and the
        reshapes together. The reason is in the kernel: the multi-row path takes an
        integer sqrt and reciprocal PER ROW and then splats the result across a 64-byte
        beat with 16 volatile stores, so ~60 us of each call is a scalar loop over 32
        rows on a core with no FPU. Rows are independent, so that loop divides.
        """
        if shard == "none":
            return pipe.add(RMSNorm(rows=T, cols=d, cluster=0, in_layout=NORM_LAYOUT,
                                    out_layout=NORM_LAYOUT, out_dtype=DType.F16),
                            name=name, bind={"x": src}).out("y")
        blk = _ShardedNorm(ctx, T, d, ncl, name, hs, on_l1=on_l1,
                           l3_key=None if on_l1 else src)
        if on_l1:
            return pipe.add(blk, name=name, bind={"x": src}).out("y")
        # Reading its slices straight from L3, it has no predecessor in the chain -- so it
        # is a SOURCE of the pipeline rather than a stage with an input, and registering
        # it as one is what keeps its nodes in the order they are written here.
        return pipe.source(name, blk.outputs["y"],
                           lambda: blk.build(ctx.scope(name), {}).outputs["y"]).out("y")

    # ---- 1. the first normalisation ----------------------------------------------------
    # sharded, norm1 reads its slices straight from L3 -- no broadcast on the way in
    n1_y = rmsnorm("norm1", "x" if shard == "rows" else x_port,
                   on_l1=(shard != "rows"))
    out[1] = n1_y
    check("norm1", n1_y, "norm1_golden", T * d)
    if stages == 1:
        pipe.run()
        return out

    # ---- 2. the Q/K/V projections -------------------------------------------------------
    # Reshape is a STAGE the layer names: between two blocks that both live in L1 there is
    # no load for the conversion to fold into, and the linker may not insert a node --
    # node creation order is dispatch order on this machine.
    rs1 = pipe.add(Reshape(rows=T, cols=d, src=Layout.ROW_MAJOR, dst=Layout.A, mesh=mesh,
                           dtype=DType.F16, cluster=0),
                   name="n1_to_a", bind={"x": n1_y})
    q1 = pipe.add(Quantize(rows=T, cols=d, inv_scale_f32bits=data["scale_n1_bits"],
                           layout=Layout.A, cluster=0),
                  name="n1_q", bind={"x": rs1.out("y")})
    # EVERY GEMM IS FOLLOWED BY A DEQUANTISE. The array accumulates int8 x int8 in int32
    # and the D port narrows that to fp16, so the output still carries both operands'
    # quantisation scales -- at d=128 it lands in the thousands while everything it has to
    # rejoin is O(1). The scale back is 1/(scale_x * W_SCALE); see Dequantize for the two
    # fp16 limits that make this its own pass rather than something folded elsewhere.
    proj, proj_dq = {}, {}
    for nm in ("q", "k", "v"):
        proj[nm] = pipe.add(Linear(tokens=T, d_in=d, d_out=d, mesh=mesh, cluster=0),
                            name=f"proj_{nm}",
                            bind={"x": q1.out("y"),
                                  "w": L3(Layout.B, DType.I8, (d, d), hs[f"w_{nm}"])})
        proj_dq[nm] = pipe.add(
            Dequantize(rows=T, cols=d, scale_f32bits=data["dq_proj_bits"],
                       layout=Layout.D, cluster=0),
            name=f"proj_{nm}_dq", bind={"x": proj[nm].out("y")})
    out[2] = proj_dq
    # One projection is checked, not three: they are the same block on the same input with
    # different weights, so a second failing check would say nothing the first did not.
    check("proj_q", proj_dq["q"].out("y"), "proj_q_golden", T * d)
    if stages == 2:
        pipe.run()
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
    pipe.raw(lambda: fa_gather(ctx.scope("attn"), fa.block.cfg,
                               fa.result.extra["shards"], verify=False), "fa_gather")
    out[3] = fa
    # FA's own output is d32/int32 per cluster and its partials are un-folded, so there is
    # no fp16 tensor here to compare. The rung still proves attention BUILDS, DISPATCHES
    # and TERMINATES on four clusters, which is what the stage is being brought up for;
    # the arithmetic is covered by fa_decode_4cluster's own checks.
    if stages == 3:
        pipe.run()
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
                    name="proj_o_dq", bind={"x": o_proj.out("y")})
    check("proj_o", o_dq.out("y"), "proj_o_golden", T * d)
    rs_o = pipe.add(Reshape(rows=T, cols=d, src=Layout.D, dst=Layout.ROW_MAJOR, mesh=mesh,
                            dtype=DType.F16, cluster=0),
                    name="o_to_row_major", bind={"x": o_dq.out("y")})
    res1 = pipe.add(Residual(rows=T, cols=d, cluster=0, layout=Layout.ROW_MAJOR), name="resid1",
                    bind={"a": rs_o.out("y"), "b": x_port})
    out[4] = res1
    check("resid1", res1.out("y"), "resid1_golden", T * d)
    if stages == 4:
        pipe.run()
        return out

    # ---- 5. the feed-forward half -------------------------------------------------------
    # norm2's input is an intermediate in cluster 0's L1, so a sharded build pulls the
    # slices across the fabric with each cluster's own xDMA rather than from L3
    # THE PULLS MUST WAIT FOR THE RESIDUAL. Each cluster reads its slice out of cluster
    # 0's L1, and nothing in the fabric would tell it the buffer is still being written.
    # Without this edge the sharded build races and the race is silent.
    n2_y = rmsnorm("norm2", res1.out("y"), on_l1=True)
    check("norm2", n2_y, "norm2_golden", T * d)
    rs2 = pipe.add(Reshape(rows=T, cols=d, src=Layout.ROW_MAJOR, dst=Layout.A, mesh=mesh,
                           dtype=DType.F16, cluster=0),
                   name="n2_to_a", bind={"x": n2_y})
    q2 = pipe.add(Quantize(rows=T, cols=d, inv_scale_f32bits=data["scale_n2_bits"],
                           layout=Layout.A, cluster=0),
                  name="n2_q", bind={"x": rs2.out("y")})
    # A dense up-projection, not MoeFFN: MoeFFN owns all four clusters for its expert
    # lanes and FlashAttention above already has them. Two blocks that both want every
    # cluster need a placement plan, which is the framework's job above this layer.
    ffn = pipe.add(Linear(tokens=T, d_in=d, d_out=h, mesh=mesh, cluster=0),
                   name="ffn_up",
                   bind={"x": q2.out("y"),
                         "w": L3(Layout.B, DType.I8, (d, h), hs["w_up_0"])})
    ffn_dq = pipe.add(Dequantize(rows=T, cols=h, scale_f32bits=data["dq_ffn_bits"],
                                 layout=Layout.D, cluster=0),
                      name="ffn_up_dq", bind={"x": ffn.out("y")})
    out[5] = ffn_dq
    check("ffn_up", ffn_dq.out("y"), "ffn_up_golden", T * h)
    if stages == 5:
        pipe.run()
        return out

    # ---- 6. the second residual, and the layer output ----------------------------------
    rs_f = pipe.add(Reshape(rows=T, cols=h, src=Layout.D, dst=Layout.ROW_MAJOR, mesh=mesh,
                            dtype=DType.F16, cluster=0),
                    name="ffn_to_row_major", bind={"x": ffn_dq.out("y")})
    res2 = pipe.add(Residual(rows=T, cols=h, cluster=0, layout=Layout.ROW_MAJOR), name="resid2",
                    bind={"a": rs_f.out("y"),
                          "b": res1.out("y")})
    out[6] = res2
    check("layer_out", res2.out("y"), "ladder_out_golden", T * h)
    pipe.run()                    # resolve every boundary, then build in the order above
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
    pipe.raw(lambda: fa_gather(ctx.scope("attn"), fa.block.cfg,
                               fa.result.extra["shards"], verify=False), "fa_gather")

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
        # A SOURCE OF THE PIPELINE, not a node built beside it. It sits between two
        # clusters' worth of stages, and node creation order is dispatch order, so
        # building it outside would put all four loads ahead of all four slices.
        def _load_x(g=g, sfx=sfx, r0=r0):
            x_l1 = g.l1(f"layer_x_{sfx}", NR * d * 2)
            ld_x = g.node(f"Ld_x_{sfx}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(
                              at_offset(hs["x"], r0 * d * 2), x_l1, NR * d * 2))
            return Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (NR, d),
                                 mem_level=MemLevel.L1), x_l1, (ld_x,))
        x_port = pipe.source(f"layer_x_{sfx}",
                             PortSpec(Layout.ROW_MAJOR, DType.F16, (NR, d),
                                      mem_level=MemLevel.L1), _load_x).out("y")

        # --- norm -> reshape -> quantise -> the three projections ---
        n1 = pipe.add(RMSNorm(rows=NR, cols=d, cluster=c, in_layout=NORM_LAYOUT,
                              out_layout=NORM_LAYOUT, out_dtype=DType.F16),
                      name=f"norm1_{sfx}",
                      bind={"x": x_port})
        a1 = pipe.add(Reshape(rows=NR, cols=d, src=Layout.ROW_MAJOR, dst=Layout.A, mesh=mesh,
                              dtype=DType.F16, cluster=c),
                      name=f"n1_to_a_{sfx}", bind={"x": n1.out("y")})
        q1 = pipe.add(Quantize(rows=NR, cols=d, inv_scale_f32bits=data["scale_n1_bits"],
                               layout=Layout.A, cluster=c),
                      name=f"n1_q_{sfx}", bind={"x": a1.out("y")})
        for nm in ("q", "k", "v"):
            pr = pipe.add(Linear(tokens=NR, d_in=d, d_out=d, mesh=mesh, cluster=c),
                          name=f"proj_{nm}_{sfx}",
                          bind={"x": q1.out("y"),
                                "w": L3(Layout.B, DType.I8, (d, d), hs[f"w_{nm}"])})
            pipe.add(Dequantize(rows=NR, cols=d, scale_f32bits=data["dq_proj_bits"],
                                layout=Layout.D, cluster=c),
                     name=f"proj_{nm}_dq_{sfx}", bind={"x": pr.out("y")})

        # --- output projection, from the staged context, and the first residual ---
        op = pipe.add(Linear(tokens=NR, d_in=d, d_out=d, mesh=mesh, cluster=c),
                      name=f"proj_o_{sfx}",
                      bind={"x": L3(Layout.A, DType.I8, (NR, d),
                                    at_offset(hs["attn_ctx"], r0 * d)),
                            "w": L3(Layout.B, DType.I8, (d, d), hs["w_o"])})
        odq = pipe.add(Dequantize(rows=NR, cols=d, scale_f32bits=data["dq_proj_o_bits"],
                                  layout=Layout.D, cluster=c),
                       name=f"proj_o_dq_{sfx}", bind={"x": op.out("y")})
        orp = pipe.add(Reshape(rows=NR, cols=d, src=Layout.D, dst=Layout.ROW_MAJOR, mesh=mesh,
                               dtype=DType.F16, cluster=c),
                       name=f"o_to_row_major_{sfx}", bind={"x": odq.out("y")})
        r1 = pipe.add(Residual(rows=NR, cols=d, cluster=c, layout=Layout.ROW_MAJOR), name=f"resid1_{sfx}",
                      bind={"a": orp.out("y"), "b": x_port})

        # --- the feed-forward half ---
        n2 = pipe.add(RMSNorm(rows=NR, cols=d, cluster=c, in_layout=NORM_LAYOUT,
                              out_layout=NORM_LAYOUT, out_dtype=DType.F16),
                      name=f"norm2_{sfx}",
                      bind={"x": r1.out("y")})
        a2 = pipe.add(Reshape(rows=NR, cols=d, src=Layout.ROW_MAJOR, dst=Layout.A, mesh=mesh,
                              dtype=DType.F16, cluster=c),
                      name=f"n2_to_a_{sfx}", bind={"x": n2.out("y")})
        q2 = pipe.add(Quantize(rows=NR, cols=d, inv_scale_f32bits=data["scale_n2_bits"],
                               layout=Layout.A, cluster=c),
                      name=f"n2_q_{sfx}", bind={"x": a2.out("y")})
        up = pipe.add(Linear(tokens=NR, d_in=d, d_out=h, mesh=mesh, cluster=c),
                      name=f"ffn_up_{sfx}",
                      bind={"x": q2.out("y"),
                            "w": L3(Layout.B, DType.I8, (d, h), hs["w_up_0"])})
        udq = pipe.add(Dequantize(rows=NR, cols=h, scale_f32bits=data["dq_ffn_bits"],
                                  layout=Layout.D, cluster=c),
                       name=f"ffn_up_dq_{sfx}", bind={"x": up.out("y")})
        urp = pipe.add(Reshape(rows=NR, cols=h, src=Layout.D, dst=Layout.ROW_MAJOR, mesh=mesh,
                               dtype=DType.F16, cluster=c),
                       name=f"ffn_to_row_major_{sfx}", bind={"x": udq.out("y")})
        r2 = pipe.add(Residual(rows=NR, cols=h, cluster=c, layout=Layout.ROW_MAJOR), name=f"resid2_{sfx}",
                      bind={"a": urp.out("y"), "b": r1.out("y")})

        # --- this slice out to its place in the L3 result, on this cluster's own DM core ---
        slice_b = NR * h * 2

        def _store(g=g, sfx=sfx, c=c, r2=r2, slice_b=slice_b):
            port = r2.out("y").port
            st = g.node(f"St_out_{sfx}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(
                            port.handle, at_offset(gathered, c * slice_b), slice_b),
                        port.ends[-1])
            ends.append(st)
        pipe.raw(_store, f"store_{sfx}")

        if verify == "slices":
            # THE SLICE ON ITS OWN, which separates the two things a wrong layer output
            # cannot tell apart: a cluster that computed its rows wrongly, and a transfer
            # that reported success without moving anything. It costs no extra transfer --
            # the slice is already in L3 at a known offset, so this is a compare node and
            # nothing else.
            def _slice_check(sfx=sfx, c=c, slice_b=slice_b):
                checks.check_out(
                    ctx.at(0), f"Check_slice_{sfx}",
                    golden=at_offset(hs["ladder_out_golden"], c * slice_b),
                    got=at_offset(gathered, c * slice_b),
                    dtype=DType.F16, elems=NR * h, tol=_TOL["layer_out"],
                    after=ends[-1], label=f"slice_{sfx}")
            pipe.raw(_slice_check, f"check_slice_{sfx}")

    if verify != "none":
        # ORDERED ON EVERY SLICE, not just the last one. This reads the whole gathered
        # buffer, so it has to wait for all four writers; depending on one of them leaves
        # the other three racing and the check reads whatever was there. `ends` is filled
        # by the per-slice stores, which are raw steps, so this one is too -- and being
        # registered last is what makes it depend on all of them.
        pipe.raw(lambda: checks.check_out(
            ctx.at(0), "Check_llm_layer_out", golden=hs["ladder_out_golden"],
            got=gathered, dtype=DType.F16, elems=T * h, tol=_TOL["layer_out"],
            after=tuple(ends), label="llm_layer_out"), "check_layer_out")
    pipe.run()                    # resolve every boundary, then build in the order above
    return {"slices": ncl, "rows_each": NR, "out": gathered}
