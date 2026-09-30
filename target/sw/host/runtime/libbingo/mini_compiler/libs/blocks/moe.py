# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The mixture-of-experts feed-forward block, as one DFG fragment.

Router -> top-k gate -> E expert lanes, each its own CERF group -> weighted combine. The
losing lanes are skipped by hardware: no dispatch, no weight traffic, no compute.

The block is compute only. Verification stays with the caller, because the goldens and the
tolerances are workload data. A caller that wants cross-stage L1 reuse hangs its readback
off the block's own last node, so the check sits inside this stage rather than between it
and the next one: liveness needs every user of one buffer to be an ancestor of every user
of the next, and a check ordered after the stage breaks that.
"""

from dataclasses import dataclass
from typing import Optional

from bingo_kernel_args import (
    HostBingoKernelAraSoftmaxF32Args,
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelSimdFp16ToInt8Args,
    SnaxBingoKernelSimdMoeCombineF16Args,
    SnaxBingoKernelSimdSwigluF16F16Args,
)

from ..comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                    PortSpec)
from ..comm.nest import d_to_a_args


@dataclass(frozen=True)
class MoeCfg:
    """Shapes in ELEMENTS. The mesh-tile counts a GEMM descriptor wants are derived, so a
    caller states the layer ("64-wide model, 32-wide hidden") and never the tiling."""
    num_experts: int
    top_k: int
    tokens: int
    d_model: int
    d_hidden: int
    mesh: tuple                       # (meshRow, tileSize, meshCol)
    array_shape_idx: int = 0
    transpose_a: int = 0
    transpose_b: int = 0
    int8_scale_bits: int = 0
    # WHERE THE LANES RUN, one cluster per expert, defaulting to one each in order.
    # `combine_cluster` is where the weighted sum lands and is a separate choice: the
    # combine reads every lane's pushed result, so it does not have to sit on any of them.
    clusters: Optional[tuple] = None
    combine_cluster: int = 0

    def __post_init__(self):
        object.__setattr__(self, "clusters",
                           tuple(range(self.num_experts)) if self.clusters is None
                           else tuple(self.clusters))
        if len(self.clusters) != self.num_experts:
            raise ValueError(
                f"MoeCfg: clusters={self.clusters} places {len(self.clusters)} lanes but "
                f"there are {self.num_experts} experts. Each expert is its own CERF "
                f"group and runs on its own cluster.")
        mr, ts, mc = self.mesh
        for nm, val, unit in (("tokens", self.tokens, mr),
                              ("d_model", self.d_model, ts), ("d_model", self.d_model, mc),
                              ("d_hidden", self.d_hidden, ts), ("d_hidden", self.d_hidden, mc)):
            if val % unit:
                raise ValueError(f"MoeCfg: {nm}={val} is not a multiple of {unit}; the "
                                 f"layer's widths have to tile the array exactly.")
        if self.top_k > self.num_experts:
            raise ValueError(f"MoeCfg: top_k={self.top_k} > num_experts={self.num_experts}.")

    # mesh-TILE counts, which is what a GEMM descriptor's M/K/N mean
    @property
    def M_T(self): return self.tokens // self.mesh[0]
    @property
    def K_up(self): return self.d_model // self.mesh[1]
    @property
    def N_up(self): return self.d_hidden // self.mesh[2]
    @property
    def K_down(self): return self.d_hidden // self.mesh[1]
    @property
    def N_down(self): return self.d_model // self.mesh[2]

    @property
    def sizes(self) -> dict:
        """Every buffer the block needs, in BYTES."""
        mr, ts, mc = self.mesh
        return {
            "x_a":      self.M_T * self.K_up * mr * ts,          # int8  A-layout
            "w_up":     self.K_up * self.N_up * ts * mc,         # int8  B-layout
            "w_down":   self.K_down * self.N_down * ts * mc,     # int8  B-layout
            "d_hidden": self.M_T * self.N_up * mr * mc * 2,      # fp16  D-layout
            "act_a":    self.M_T * self.K_down * mr * ts * 2,    # fp16  A-layout
            "act_i8":   self.M_T * self.K_down * mr * ts,        # int8  A-layout
            "y":        self.M_T * self.N_down * mr * mc * 2,    # fp16  D-layout
        }


def expert_lane(ctx: Ctx, cfg: MoeCfg, e: int, x_src, w, ybuf, after=None):
    """One expert's whole FFN, on its own cluster.

    Every node here goes into one branch of the fork, so the whole lane is one CERF group
    and is skipped or run as a unit. `w` is (up, gate, down) source handles.

    Returns (push_node, y_handle, [every node in the lane]).
    """
    sz = cfg.sizes
    w_up_src, w_gate_src, w_down_src = w

    l1_x = ctx.l1("moe_x", sz["x_a"])
    l1_up = ctx.l1("moe_w_up", sz["w_up"])
    l1_gate = ctx.l1("moe_w_gate", sz["w_up"])
    l1_down = ctx.l1("moe_w_down", sz["w_down"])
    d_up = ctx.l1("moe_d_up", sz["d_hidden"])
    d_gate = ctx.l1("moe_d_gate", sz["d_hidden"])
    d_act = ctx.l1("moe_act", sz["d_hidden"])
    a_act = ctx.l1("moe_act_a", sz["act_a"])
    a_i8 = ctx.l1("moe_act_i8", sz["act_i8"])
    l1_y = ctx.l1("moe_y", sz["y"])

    ld_x = ctx.node(f"LdX_e{e}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(x_src, l1_x, sz["x_a"]), after)
    ld_up = ctx.node(f"LdWup_e{e}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                     SnaxBingoKernelIdma1dCopyArgs(w_up_src, l1_up, sz["w_up"]))
    ld_gate = ctx.node(f"LdWgate_e{e}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(w_gate_src, l1_gate, sz["w_up"]))
    ld_down = ctx.node(f"LdWdown_e{e}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(w_down_src, l1_down, sz["w_down"]))

    def proj(name, A, B, C_D, K, N, after):
        # int32tofp16_enable moves the narrowing onto the GEMM's own D port: an int32 beat
        # carries half as many values as an fp16 one, so converting at the consumer would
        # double both the beats it reads and the L1 the tile occupies.
        #
        # input_C_addr=0, NOT C_D. Every VersaCore GEMM computes D = A*B + C, and
        # accumPrevC=0 selects where C comes from: a NON-ZERO address reads C from memory
        # and adds it, a ZERO address adds nothing. Passing the output buffer as C
        # therefore reads it before anything has written it -- uninitialised TCDM, which
        # simulates as X -- and D = A*B + X = X. That X flows through the whole lane and
        # finally reaches the host, where check_result loads it and the CVA6 issues on an
        # unknown operand. It never fails a check; it kills the host.
        return ctx.node(name, ctx.gemm, "__snax_bingo_kernel_gemm_full",
                        SnaxBingoKernelGemmFullArgs(
                            input_A_addr=A, input_B_addr=B, input_C_addr=0,
                            output_D_addr=C_D, M=cfg.M_T, K=K, N=N,
                            array_shape_idx=cfg.array_shape_idx,
                            transpose_A=cfg.transpose_a, transpose_B=cfg.transpose_b,
                            accumPrevC=0, int32tofp16_enable=1), after)

    gemm_up = proj(f"GemmUp_e{e}", l1_x, l1_up, d_up, cfg.K_up, cfg.N_up, [ld_x, ld_up])
    gemm_gate = proj(f"GemmGate_e{e}", l1_x, l1_gate, d_gate, cfg.K_up, cfg.N_up,
                     [ld_x, ld_gate])

    # Elementwise over the contiguous D-block buffer: no reshape needed in front of it.
    cols = cfg.sizes["d_hidden"] // 2
    swiglu = ctx.node(f"Swiglu_e{e}", ctx.simd,
                      "__snax_bingo_kernel_simd_swiglu_f16_f16",
                      SnaxBingoKernelSimdSwigluF16F16Args(d_gate, d_up, d_act,
                                                          rows=1, cols=cols),
                      [gemm_up, gemm_gate])
    reshape = ctx.node(f"Reshape_e{e}", ctx.xdma, "__snax_bingo_kernel_xdma_6d",
                       d_to_a_args(cfg.mesh, cfg.M_T, cfg.N_up, cfg.K_down,
                                   d_act, a_act), swiglu)
    quant = ctx.node(f"Quant_e{e}", ctx.simd, "__snax_bingo_kernel_simd_fp16_to_int8",
                     SnaxBingoKernelSimdFp16ToInt8Args(
                         a_act, a_i8, beats=sz["act_a"] // 64, rows=1,
                         inv_scale_f32bits=cfg.int8_scale_bits), reshape)
    gemm_down = proj(f"GemmDown_e{e}", a_i8, l1_down, l1_y, cfg.K_down, cfg.N_down,
                     [quant, ld_down])

    # The SIMD has no AXI port, so the combine can only read its OWN cluster's L1. Each
    # expert therefore pushes its result into a slot of one buffer on the combine cluster,
    # which is also what lets the combine address the operands as base + e * stride.
    push = ctx.node(f"PushY_e{e}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(l1_y, ybuf.view(e * sz["y"]), sz["y"]),
                    gemm_down)
    return push, l1_y, [ld_x, ld_up, ld_gate, ld_down, gemm_up, gemm_gate, swiglu,
                        reshape, quant, gemm_down, push]


class MoeFFN(Block):
    """Mixture-of-experts feed-forward: router -> top-k gate -> E lanes -> weighted combine.

    The losing lanes are skipped by hardware -- no dispatch, no weight traffic, no compute --
    which is the whole reason the block exists and why its input is read by every lane
    rather than gathered.

    PORTS
      in   x        the activation, A-layout int8, read by every expert lane
      out  y        the combined result, D-layout fp16, on the combine cluster

    WHY THIS ONE SPANS CLUSTERS when every other block runs on one. The lanes are not
    independent sub-graphs placed side by side: they are BRANCHES OF ONE CONDITIONAL FORK,
    each its own CERF group, recombined by that fork's weighted sum. The fork is a
    whole-graph construct, so the several clusters here are one thing and there is nothing
    to assemble them out of. `cfg.clusters` says where the lanes go and
    `cfg.combine_cluster` where the sum lands.

    `weights`, `logits` and `zero_src` are staged data, not ports: they come from the
    workload's own datagen and do not flow between blocks.
    """
    name = "moe_ffn"

    def __init__(self, cfg: MoeCfg, *, weights, logits, zero_src=None):
        self.cfg = cfg
        self.weights = weights
        self.logits = logits
        self.zero_src = zero_src

    @property
    def inputs(self) -> dict:
        c = self.cfg
        # This block does not override `needs`, so it fetches nothing: whatever is bound
        # has to be what its loads read, which is main memory.
        return {"x": PortSpec(Layout.A, DType.I8, (c.tokens, c.d_model),
                              mem_level=MemLevel.L3,
                              doc="activation, broadcast to every expert lane")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(Layout.D, DType.F16, (c.tokens, c.d_model),
                              mem_level=MemLevel.L1, cluster=c.combine_cluster,
                              doc="weighted sum over the selected experts")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        cfg = self.cfg
        E, k, sz = cfg.num_experts, cfg.top_k, cfg.sizes
        cc = cfg.combine_cluster
        g = ctx.at(cc)
        x_src = bound["x"].handle

        # ---- router --------------------------------------------------------------------
        # The softmax publishes p to its scratchpad, which is where the gating kernel reads
        # it from; nothing else has to hand it over.
        probs = g.l3("moe_probs", E * 4)
        router = g.node("Router", g.host, "__host_bingo_kernel_softmax",
                        HostBingoKernelAraSoftmaxF32Args(
                            input_addr=self.logits, output_addr=probs,
                            num_rows=1, row_length=E))

        # ---- the fork ------------------------------------------------------------------
        # One branch per expert, so each gets its own CERF group and is skipped alone.
        fork = ctx.dfg.bingo_conditional_fork(router, {"mode": "top_k", "k": k})
        ybuf = g.l1("moe_ybuf", E * sz["y"], cluster=cc)

        nodes, sources = [router], [router]
        zero = None
        if self.zero_src is not None:
            zero = g.node("ZeroYbuf", g.dm, "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(self.zero_src, ybuf, E * sz["y"]),
                          router, cluster=cc)
            nodes.append(zero)

        pushes, lanes, x_readers = [], [], []
        for e, cl in zip(range(E), cfg.clusters):
            push, _, lane = expert_lane(ctx.at(cl), cfg, e, x_src, self.weights[e], ybuf,
                                        after=(zero if zero is not None else router))
            pushes.append(push)
            lanes.append(lane)
            nodes += lane
            x_readers.append(lane[0])        # LdX: the first node that reads x
            sources += lane[1:4]             # the three weight loads have no predecessor
            fork.branch(lane)

        # ---- the combine ---------------------------------------------------------------
        out = g.l1("moe_out", sz["y"], cluster=cc)
        combine = g.node("Combine", g.simd, "__snax_bingo_kernel_simd_moe_combine_f16",
                         SnaxBingoKernelSimdMoeCombineF16Args(
                             output_addr=out, src_base_addr=ybuf, src_stride=sz["y"],
                             rows=1, cols=sz["y"] // 2), cluster=cc)
        # activation, weights and num_inputs are bound by the lowering, from the same gate.
        fork.combine(combine, inputs=pushes, kind="weighted_sum", weights=fork.weights)
        nodes.append(combine)

        return BlockResult(
            inputs={"x": Port(self.inputs["x"], x_src, tuple(x_readers), name="x")},
            outputs={"y": Port(self.outputs["y"], out, (combine,), name="y")},
            nodes=nodes, sources=sources,
            extra={"fork": fork, "ybuf": ybuf, "probs": probs, "pushes": pushes,
                   "lanes": lanes, "zero": zero, "combine": combine, "router": router})
