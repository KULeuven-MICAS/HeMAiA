#!/usr/bin/env python3
"""MoE4: one mixture-of-experts feed-forward layer, four experts, top-2 routing.

Four expert FFNs, one per cluster, each gated as its own CERF group, reconverging in a
weighted combine. This is the first workload where a data-dependent branch both SKIPS
work in hardware and comes back together, which is what makes it a layer rather than a
mechanism test.
"""

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
#
# ------------------------------------------------------------------------------
# WHAT A MIXTURE OF EXPERTS IS, AND WHY IT IS WORTH HARDWARE SUPPORT
#
# A dense feed-forward layer runs every parameter for every token. An MoE layer holds many
# expert FFNs but routes each token to only a few of them: a small router scores the experts,
# the top-k win, and only those run. Parameter count therefore grows with the number of
# experts while the compute per token grows with k, which is the entire reason the design
# exists -- capacity without proportional cost.
#
# The catch is that the work is DATA DEPENDENT. Which experts run is not known until the
# router has produced its scores, so the schedule cannot be fixed at compile time. On a
# machine that dispatches a static task list this is the hard part, and it is what the CERF
# / conditional-execution machinery is for: the graph contains every expert, and the gating
# task decides at run time which of them are allowed to execute. Work that is skipped costs
# no dispatch and no memory traffic.
#
# The dataflow of one MoE layer, end to end
#
# T tokens, model width d, E experts, k selected per token, expert hidden width h.
# Shapes are [rows, cols]; the two projections into the expert and the one out of it are
# what makes an expert an FFN rather than a matmul.
#
#   xn [T,d] ─┬─ router: xn @ Wr [d,E] ─→ logits [T,E] ─ softmax ─→ p [T,E] ─ top-k ─→ w
#             │                                                                        │
#             └─ for each SELECTED expert e only:                                      │
#                  up   = xn @ W_up[e]   [d,h] ─→ [T,h]                                │
#                  gate = xn @ W_gate[e] [d,h] ─→ [T,h]                                │
#                  a    = SwiGLU(gate, up)      ─→ [T,h]                               │
#                  y_e  = a  @ W_down[e] [h,d] ─→ [T,d]                                │
#                                                                                      │
#                  out = SUM over selected e of  w_e * y_e  ←────────────────────────────┘
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                        the COMBINE: fork.combine(kind='weighted_sum')
#
# The unselected experts contribute nothing: no dispatch, no weight traffic, no compute.
#
# A small example: E = 4, k = 2
#
#   router logits          [ -1.5,  0.3, -0.8,  2.1 ]
#   softmax p              [0.0219, 0.1325, 0.0441, 0.8015]   sums to 1 over ALL experts
#   top-2 winners           expert 3 (0.8015) and expert 1 (0.1325)
#   renormalise over the winners only:
#                           w_3 = 0.8015 / (0.8015 + 0.1325) = 0.858149
#                           w_1 = 0.1325 / (0.8015 + 0.1325) = 0.141851
#   out                    = 0.858149 * y_3 + 0.141851 * y_1
#
#   Experts 0 and 2 never run: their CERF groups stay inactive, so every task in their
#   lanes is skipped and their weights are never fetched. Note the renormalisation -- the
#   raw softmax sums to 1 across all four, but the two winners only carry 0.9340 of that
#   mass, so using the raw scores would silently scale the layer's output to 93.40%.
#
# ------------------------------------------------------------------------------
# HOW THE BRANCH IS ACTUALLY TAKEN: CERF, AND THE GATING NODE THE COMPILER INSERTS
#
# CERF is one 32-bit register in the quadrant controller -- one bit per conditional-execution
# group. Software can read and write it over the quad-ctrl regbus; the BINGO manager reads it
# combinationally. That single register is the entire branch state of the machine.
#
# Every task descriptor carries three fields for it, plus the 2-bit task type:
#
#   cond_exec_en       this task is guarded at all
#   cond_exec_group_id which of the 32 CERF bits guards it
#   cond_exec_invert   run when the bit is CLEAR instead of set
#   task_type          00 normal, 01 dummy, 10 gating, 11 reserved
#
# The skip decision is made per (core, cluster) at the head of the waiting queue, AFTER the
# task's dep-check has passed -- a skipped task still waits for its operands' dependencies,
# it just never dispatches (bingo_hw_manager_top.sv, the `cond_exec_skip` assign):
#
#   cerf_group_active = cerf_state[desc.cond_exec_group_id];
#   cond_exec_skip    = !queue_empty && desc.cond_exec_en &&
#                       (desc.cond_exec_invert ? cerf_group_active : !cerf_group_active);
#
# What fork.branch(lane) turns into
#
# The workload never names a CERF group. It declares the fork, and the four passes in
# bingo_dfg_conditional.py do the rest:
#
#   1. _collect_conditional_declarations   finds Router and the 44 tasks it gates
#   2. _insert_gating_node                 splices `__gating_Router` between Router and every
#                                          one of them. It lands on Router's OWN core -- the
#                                          chiplet's host core -- because that is the one place
#                                          guaranteed not to be CERF-skippable itself. A gating
#                                          node on a skippable cluster cannot run when that
#                                          cluster is inactive, and deadlocks everything it gates.
#   3. _assign_cerf_groups                 one group per branch (expert e -> group e here), sets
#                                          cond_exec_en / _group_id on all 11 nodes of each lane,
#                                          and records each node's BRANCH ordinal in
#                                          _cond_node_index. Branch ordinal, not group id: they
#                                          coincide only while no two branches share a group.
#   4. _build_gating_kernel_args           points the node at __host_bingo_kernel_cerf_gating and
#                                          allocates its three L3 arrays (below)
#
# The gating node is an ordinary task that RUNS on the host core. task_type=10 marks it as
# gating so the manager does not filter it like a dummy; the CERF write itself is a plain
# software store from inside the kernel, not something the manager does on completion.
#
# What __gating_Router does when it runs
#
#   in:   pred_scratchpad_addr -> Router's scratchpad. The softmax published its output pointer
#         and element count there, so the gate reads p[] without an explicit edge carrying it.
#   out:  three L3 arrays, ALL indexed by branch (= expert), never by group:
#
#         cerf_group_ids[e]   e -> which CERF bit guards expert e   { 0,1,2,3 }   read-only
#         cond_activation[e]  1 if expert e was selected                          written
#         cond_weight[e]      renormalised fp32 weight, 0.0f for losers           written
#
#         The last two are two views of ONE allocation (`__cond_sel_Router`, weights at +0,
#         flags at +4E), so a gate costs one L3 record rather than two.
#
#   then: for each of the k winners
#             cerf_write_mask   |= 1 << cerf_group_ids[winner]
#             cond_activation[winner] = 1
#             cond_weight[winner]     = p[winner]
#         renormalise the k weights so they sum to 1   <- the only float divide in the path
#         bingo_cerf_update(cerf_controlled_mask, cerf_write_mask)
#
# bingo_cerf_update is a read-modify-write: read the live CERF, clear the bits this gate owns,
# OR in the winners, store it back, pulse the write-enable. Owning a mask rather than writing
# the whole register is what lets two independent gates coexist.
#
# From that store onward every task in a losing lane hits `cond_exec_skip` and is dropped from
# the ready queue. Nothing in the lane's descriptors changed -- the same static task list runs
# a different subset depending on one register.
#
# ------------------------------------------------------------------------------
# HOW THE COMBINE IS BUILT, AND HOW IT READS THE GATE'S DECISION
#
# The landing area is the whole trick
#
# The SIMD block has no AXI port -- it can only reach its
# OWN cluster's L1 (BINGO_SIMD_REQUIRE_LOCAL) -- so the combine cannot read four results sitting
# on four clusters. Each expert therefore ends its lane with one cross-cluster iDMA push into a
# slot of a single buffer on the combine's cluster:
#
#     ybuf = g.l1("moe_ybuf", E * sz["y"], cluster=COMBINE_CLUSTER)   # E slots, one allocation
#     ...
#     push = g.node(f"PushY_e{e}", DMA_CORE, "...idma_1d_copy",
#                   SnaxBingoKernelIdma1dCopyArgs(l1_y, ybuf.view(e * sz["y"]), sz["y"]), ...)
#
# That is also why the kernel takes `src_base_addr` + `src_stride` + `num_inputs` rather than E
# pointers: the descriptor stays a fixed size whatever E is, and each expert's push destination
# is a plain view into the same allocation. (Same trade as xdma_elementwise_add.)
#
# The three lines that build it
#
#     combine = g.node("Combine", SIMD_CORE, "__snax_bingo_kernel_simd_moe_combine_f16",
#                      SnaxBingoKernelSimdMoeCombineF16Args(
#                          output_addr=out, src_base_addr=ybuf, src_stride=sz["y"],
#                          rows=1, cols=sz["y"] // 2),            # <- no mask, no weights here
#                      cluster=COMBINE_CLUSTER)
#     fork.combine(combine, inputs=pushes, kind="weighted_sum", weights=fork.weights)
#
# Note what is NOT passed: activation_addr, weight_addr and num_inputs are left unset. The
# fork fills them in during _lower_combines() through a one-method protocol --
#
#     if hasattr(node.kernel_args, "bind_combine"): node.kernel_args.bind_combine(...)
#
# -- so the compiler never has to know what this kernel calls its fields, and the combine is
# guaranteed to read the SAME decision record the gating kernel wrote. Binding it by hand would
# work, and would be one more place for the two to drift apart.
#
# How it pairs with the gating node
#
#            __gating_Router  (host core, RUNS)
#                 |  writes __cond_sel_Router:  [ float w[E] | uint8 act[E] ]
#                 |  writes CERF                -> hardware skips the losing lanes
#                 v
#     ybuf slot e <-- PushY_e   (in branch e; SKIPPED for a loser, so the slot is never written)
#                 |
#                 v
#            Combine  (ungated)  reads act[] to pick which slots to fold,
#                                reads w[]   as raw FP32 BITS -> StreamMap scale CSR
#
# The combine is deliberately NOT gated and NOT SW-guarded: cond_exec_en is False and its
# gating_sp_addr is emitted as 0, so BINGO_SW_GUARD_CHECK no-ops and it runs on every dispatch.
# Its dep-check waits on all four pushes -- and a SKIPPED push still fires its dep_set, because
# `cond_exec_skip` only feeds `ready_queue_filter_drop`, while `checkout_queue_data_in` is pushed
# either way with task_type forced to 2'b01 -- and checkout is what fires dep_set. (Both are in
# bingo_hw_manager_top.sv; grep the signal names rather than trusting a line number, the RTL is a
# pinned bender checkout that moves.) So the combine fires exactly once whatever the router
# chose. It reads the decision from the L3 record, never from a scratchpad, which is what keeps
# it independent of scratchpad slot liveness.
#
# What is actually in memory, for the small example above
#
# Four allocations carry the whole decision. Sizes here are byte-exact for E = 4; only the
# per-expert result slot Y scales with the layer shape (Y = 16,384 B at the shape in
# params.hjson today, so the landing area is 65,536 B).
#
#   L3, chiplet 0                                        written by        read by
#   ----------------------------------------------------------------------------------
#   moe_probs              16 B   float p[4]             Router            gate
#   __cerf_gids_Router      4 B   uint8 gid[4]           host, at setup    gate
#   __cond_sel_Router      20 B   the gate's verdict     GATE              COMBINE
#
#   L1, cluster 0 (the combine's own cluster -- the SIMD cannot reach any other)
#   ----------------------------------------------------------------------------------
#   moe_ybuf            4 x Y B   the landing area       ZeroYbuf, PushY   COMBINE
#   moe_out                 Y B   the layer output       COMBINE           StoreOut
#
# __cond_sel_Router after __gating_Router has run, byte for byte:
#
#      offset  field       value         meaning
#      ------  ----------  ------------  ------------------------------------------------
#        0..3  w[0]        0x00000000    0.0         expert 0 lost
#        4..7  w[1]        0x3e114168    0.141851    expert 1 WON
#       8..11  w[2]        0x00000000    0.0         expert 2 lost
#      12..15  w[3]        0x3f5bafa6    0.858149    expert 3 WON
#          16  act[0]      0x00          not selected
#          17  act[1]      0x01          selected
#          18  act[2]      0x00          not selected
#          19  act[3]      0x01          selected
#
# Those two words are the run's own, read back off the UART. They sum to exactly 1.0f, and
# they match a float32 recomputation of the renormalisation to within one ULP on w[1] and
# bit-exactly on w[3] -- which is the check `moe_w` performs.
#
# The gate also writes one thing that is NOT memory: the CERF register in the quadrant
# controller. It owns bits 0..3 (cerf_controlled_mask = 0x0000000f) because this fork has four
# branches, and since cerf_gids[e] == e here, the winners {1, 3} make the write
# cerf_write_mask = 0x0000000a. Every task carrying cond_exec_group_id 0 or 2 is then skipped.
#
# moe_ybuf when the combine starts:
#
#      +0*Y   expert 0 slot   all zeros      ZeroYbuf wrote it; PushY_e0 was skipped
#      +1*Y   expert 1 slot   y_1            PushY_e1 ran
#      +2*Y   expert 2 slot   all zeros      ZeroYbuf wrote it; PushY_e2 was skipped
#      +3*Y   expert 3 slot   y_3            PushY_e3 ran
#
# The combine reads act[] first, so it only ever touches slots 1 and 3; it never loads the
# zeroed ones. It reads w[1] and w[3] as raw 32-bit words and hands them to the StreamMap
# scale CSR without interpreting them, which is what makes this runnable on an FPU-less hart.
# The zeros exist for CheckYbuf, not for the combine -- see below.
#
# Why the compiler allows that edge at all
#
# `PushY_e (CERF group e) -> Combine (ungated)` is exactly the shape
# _validate_cerf_cross_group_edges refuses by default, because a consumer outside the group runs
# whether or not its producer did and would read a stale buffer. A DECLARED combine is the
# exemption -- and declaring it is also what lets the compiler check that EVERY branch of the
# fork reaches it. A combine that closes only three of four branches silently drops the fourth,
# so that is its own named refusal.
#
# What the kernel does with k winners
#
#     scan act[] -> a list of winner indices          (never reads a loser's slot)
#     k == 1:  simd_pass_map(src[a0] -> out,  scale = w[a0])                     1 pass
#     k >= 2:  map first winner into an accumulator, then for each remaining one:
#                  map it into a staging buffer, then StreamElementwise ADD
#              ping-ponging two accumulators so the LAST add lands in out_addr   2k-1 passes
#
# The ping-pong is not decoration: simd_pass_ew2 reads a 3-D interleave of its two operands and
# writes a packed 1-D stream, so dst == src is not a safe aliasing. Alternating also means the
# caller never has to ask where the answer ended up.
#
# The zeroing is for the check, not for the combine
#
# ZeroYbuf clears the landing area before any branch pushes into it. The combine does not need
# that -- it only ever touches slots act[] marks as live. It exists so ONE ungated check can
# cover the whole area afterwards: a winner's slot must hold its result, a loser's must still be
# zeros. That is the only check in the graph that would notice an expert running when the router
# did not pick it, and four per-expert checks inside the branches could not do it (they would
# also violate the one-CERF-group-per-core rule -- see the notes on the host core above).
#
# If you are adding your own combine
#
#   1. name fork.weights / fork.activation only AFTER the last fork.branch() call -- they are
#      sized by the branch count and allocated on first use
#   2. give your kernel-args class a bind_combine(activation, weights, num_inputs)
#   3. keep the combine ungated, and on a cluster whose own L1 holds every operand
#   4. pass every branch's producer in inputs=, or the build fails rather than dropping one
#
# ------------------------------------------------------------------------------
# WHERE THE WEIGHTS COME FROM
#
# No device core has an FPU -- every hart on snax_split_cluster is rv32ima -- so the
# renormalising divide cannot run on the device at all. It runs in the gating kernel on the
# CVA6, where the winner loop already is, and the result is published as float[E] next to
# the 0/1 activation mask in one L3 record. The combine then GATHERS those weights: it
# loads each winner's word and hands the FP32 BITS to the SIMD's StreamMap scale CSR
# without ever interpreting them as a number. The mask is what says which experts to read;
# the combine never re-derives top-k, so there is only one copy of that decision.
#
# ------------------------------------------------------------------------------
# WHAT THIS WORKLOAD BUILDS
#
#   router                        host softmax over the expert logits
#   __gating_router               inserted by the compiler from the fork's select=
#   for each expert e, on cluster e, all inside branch e:
#     ld_x, ld_wup, ld_wgate, ld_wdown    operands in                        (dm)
#     gemm_up, gemm_gate                  int8 x int8 -> fp16                (gemm)
#     swiglu                              silu(gate) * up, fp16              (simd)
#     reshape                             D-layout -> A-layout, fp16         (xdma)
#     quant                               fp16 -> int8                       (simd)
#     gemm_down                           int8 x int8 -> fp16 = y_e          (gemm)
#     push                                y_e -> cluster 0's landing slot    (dm)
#     Store_y / Check_y                   per-expert result check            (host)
#   combine                       SUM w_e * y_e over the winners             (simd, cluster 0)
#   Store_w / Check_w             the renormalised weights                   (host)
#   Store_out / Check_out         the layer output                           (host)
#
# The SwiGLU needs no reshape in front of it: it is elementwise, and rows/cols are only an
# AGU shape over a packed buffer, so it runs directly on the GEMM's D-block output. The
# reshape between the SwiGLU and the down projection is NOT optional -- A-layout (m,k,r,s)
# and D-layout (m,n,r,c) are a genuine permutation whenever meshRow > 1.
#
# END WORKLOAD DESCRIPTION AND TASK GRAPH

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR = os.path.dirname(current_dir)
sys.path.append(WORKLOADS_DIR)
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim/common")
sys.path.append(current_dir)

from moe_datagen import generate_moe_data, stage                      # noqa: E402
from bingo_dfg import BingoDFG                                          # noqa: E402
from bingo_platform import core_roles, guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode                                        # noqa: E402
from bingo_mem_handle import BingoMemAlloc                              # noqa: E402
from bingo_data_staging import DataStaging                              # noqa: E402
from bingo_kernel_args import (                                         # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelSimdSwigluF16F16Args,
    SnaxBingoKernelSimdFp16ToInt8Args,
    SnaxBingoKernelSimdMoeCombineF16Args,
    SnaxBingoKernelXdma6dArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelAraSoftmaxF32Args,
)

_ROLES = core_roles()
GEMM_CORE = _ROLES["gemm"]
SIMD_CORE = _ROLES["simd"]
DMA_CORE = _ROLES["dm"]
XDMA_CORE = _ROLES["xdma"]
HOST_CORE = _ROLES["host"]

CHECK_FP32_TOL = 1
CHECK_FP16_TOL = 2

CHIPLET_ID = 0x00
COMBINE_CLUSTER = 0          # where the experts' results land and are folded


class G:
    """Node and L1-handle factory, bound to one cluster.

    Every handle a cluster allocates carries its cluster id, so a handle from another
    cluster already resolves to a full (chip | cluster | offset) address -- which is what
    lets an expert's push name its destination on cluster 0 directly.
    """

    def __init__(self, dfg, cluster=0):
        self.dfg = dfg
        self.cluster = cluster

    def at(self, cluster):
        return G(self.dfg, cluster)

    def l1(self, name, size, cluster=None):
        # The cluster id is part of the NAME as well as the handle: allocation is keyed by
        # name, and four clusters each want their own `moe_x_a`.
        cl = self.cluster if cluster is None else cluster
        return BingoMemAlloc(f"{name}_cl{cl}", size=size, mem_level="L1", chip_id=0,
                             cluster_id=cl)

    def node(self, name, core, kname, kargs, after=(), cluster=None):
        cl = self.cluster if cluster is None else cluster
        nd = BingoNode(assigned_chiplet_id=CHIPLET_ID, assigned_cluster_id=cl,
                       assigned_core_id=core, node_name=f"{name}_cl{cl}",
                       kernel_name=kname, kernel_args=kargs)
        self.dfg.bingo_add_node(nd)
        for pred in (after if isinstance(after, (list, tuple)) else [after]):
            if pred is not None:
                self.dfg.bingo_add_edge(pred, nd)
        return nd


def get_args():
    p = argparse.ArgumentParser(description="MoE4 Workload")
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    return p.parse_args()


def load_hw_params(cfg_path, hwcfg_path):
    """Merge the workload's shapes with the array the RTL was actually elaborated with.

    The mesh is READ from the cluster cfg rather than assumed. The Makefile used to pass
    snax_versacore_to_cluster.hjson while the RTL is four snax_split_clusters, and
    array_shape 1 picked (1,16,32) out of it -- a cluster with no SIMD core and a mesh
    this workload never runs on. Every descriptor was built for that shape, silently.
    """
    with open(cfg_path) as f:
        param = hjson.loads(f.read())
    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    m = {**param, **hw}
    shape = m["array_shape"]
    acc = m["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = acc["snax_versacore_spatial_unrolling"][0]
    if shape >= len(unrolling):
        raise ValueError(
            f"{hwcfg_path} declares {len(unrolling)} array shape(s) but params.hjson "
            f"asks for index {shape}.")
    mr, ts, mc = (int(x) for x in unrolling[shape])
    if (mr, ts, mc) != (16, 4, 16):
        raise ValueError(
            f"{hwcfg_path} shape {shape} is (Mu, Ku, Nu) = {(mr, ts, mc)}, but this "
            f"workload is written for (16, 4, 16). Every tile count, every buffer size "
            f"and the reshape's strides all derive from the array, so the descriptors "
            f"and the goldens would both be wrong -- and neither would fault.")

    T, d, h = m["tokens"], m["d_model"], m["d_hidden"]
    for name, val, unit in (("tokens", T, mr), ("d_model", d, ts),
                            ("d_model", d, mc), ("d_hidden", h, ts),
                            ("d_hidden", h, mc)):
        if val % unit:
            raise ValueError(f"{name}={val} is not a multiple of {unit}; the layer's "
                             f"widths have to tile the array exactly.")
    return {
        "meshRow": mr, "tileSize": ts, "meshCol": mc, "arrayShapeIdx": shape,
        "tokens": T, "d_model": d, "d_hidden": h,
        "num_experts": m["num_experts"], "top_k": m["top_k"],
        # Mesh-TILE counts, which is what a GEMM descriptor's M/K/N mean.
        "M_T": T // mr,          # rows of x, in tiles
        "K_up": d // ts,         # the up/gate projections' contraction
        "N_up": h // mc,         # their output width
        "K_down": h // ts,       # the down projection's contraction
        "N_down": d // mc,       # its output width, back to the model width
        "transposeA": m.get("transposed_A", 0),
        "transposeB": m.get("transposed_B", 0),
    }


def sizes(hw):
    """Every buffer this layer needs, in BYTES, derived once."""
    mr, ts, mc = hw["meshRow"], hw["tileSize"], hw["meshCol"]
    return {
        "x_a":      hw["M_T"] * hw["K_up"] * mr * ts,            # int8  A-layout
        "w_up":     hw["K_up"] * hw["N_up"] * ts * mc,           # int8  B-layout
        "w_down":   hw["K_down"] * hw["N_down"] * ts * mc,       # int8  B-layout
        "d_hidden": hw["M_T"] * hw["N_up"] * mr * mc * 2,        # fp16  D-layout
        "act_a":    hw["M_T"] * hw["K_down"] * mr * ts * 2,      # fp16  A-layout
        "act_i8":   hw["M_T"] * hw["K_down"] * mr * ts,          # int8  A-layout
        "y":        hw["M_T"] * hw["N_down"] * mr * mc * 2,      # fp16  D-layout
    }


def reshape_6d(hw, src, dst):
    """D-layout (m, n, r, c) fp16 -> A-layout (m, k, r, s) fp16, on the xDMA AGU.

    The hidden index is n*meshCol + c on one side and k*tileSize + s on the other, so
    writing c = j*tileSize + s gives k = n*(meshCol/tileSize) + j and the innermost
    tileSize elements are contiguous on BOTH sides. That run is tileSize*2 = 8 bytes,
    which is the xDMA's own lane width -- the same 8-byte beat every xdma_layout_* kernel
    moves. In int8 it would be 4 bytes and fall off the hardware path onto the CPU
    fallback, which is why the reshape happens BEFORE the quantisation and not after.

    The 8 spatial lanes walk `r`, whose stride is constant on both sides.
    """
    mr, ts, mc = hw["meshRow"], hw["tileSize"], hw["meshCol"]
    eb = 2
    j_bound = mc // ts
    lanes = 8
    if mr % lanes:
        raise ValueError(f"meshRow={mr} is not a multiple of the xDMA's {lanes} lanes.")
    src_r, dst_r = mc * eb, ts * eb                 # one step of r
    # The M dimension is NOT optional, even though M_T == 1 hides it. D-layout is
    # (m, n, r, c) and A-layout is (m, k, r, s), so one step of m is a whole (n, r, c)
    # block on the source and a whole (k, r, s) block on the destination. Leaving it out
    # transfers only the FIRST m tile: at M_T=2 exactly half of `act_a` is never written,
    # stays uninitialised TCDM (= X), and the X reaches the host through quant -> down
    # GEMM -> push. It cost a full build+sim cycle to find, because at M_T=1 the bounds
    # are identical either way and everything passes.
    src_m = hw["N_up"] * mr * mc * eb
    dst_m = hw["K_down"] * mr * ts * eb

    # A layout conversion that does not move every byte leaves the rest of the destination
    # uninitialised, and uninitialised TCDM is X, which never fails a check -- it kills the
    # host instead. So count the bytes here rather than discovering it in a simulation.
    moved = lanes * (ts * eb) * (mr // lanes) * j_bound * hw["N_up"] * hw["M_T"]
    want = hw["M_T"] * hw["N_up"] * mr * mc * eb
    if moved != want:
        raise ValueError(
            f"reshape would move {moved} B but the tensor is {want} B. Every element of "
            f"the destination must be written; the shortfall stays uninitialised (X) and "
            f"reaches the host through quant -> down GEMM -> push. Check the temporal "
            f"dimension list against the layouts.")

    return SnaxBingoKernelXdma6dArgs(
        src, dst,
        spatial_stride_src=src_r, spatial_stride_dst=dst_r,
        #                r-group,           j,             n,             m
        temporal_strides_src=[lanes * src_r, ts * eb,       mr * mc * eb,  src_m],
        temporal_bounds_src=[mr // lanes,    j_bound,       hw["N_up"],    hw["M_T"]],
        temporal_strides_dst=[lanes * dst_r, mr * ts * eb,  j_bound * mr * ts * eb, dst_m],
        temporal_bounds_dst=[mr // lanes,    j_bound,       hw["N_up"],    hw["M_T"]])


def emit_expert_ffn(g, hw, sz, e, h, ybuf, inv_scale_bits, after=None):
    """One expert's whole FFN, on its own cluster. Returns (y_node, every node in it).

    Every node here goes into one branch of the fork, so the whole lane is one CERF group
    and is skipped or run as a unit.
    """
    l1_x = g.l1("moe_x", sz["x_a"])
    l1_up = g.l1("moe_w_up", sz["w_up"])
    l1_gate = g.l1("moe_w_gate", sz["w_up"])
    l1_down = g.l1("moe_w_down", sz["w_down"])
    d_up = g.l1("moe_d_up", sz["d_hidden"])
    d_gate = g.l1("moe_d_gate", sz["d_hidden"])
    d_act = g.l1("moe_act", sz["d_hidden"])
    a_act = g.l1("moe_act_a", sz["act_a"])
    a_i8 = g.l1("moe_act_i8", sz["act_i8"])
    l1_y = g.l1("moe_y", sz["y"])

    ld_x = g.node(f"LdX_e{e}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(h["x_a"], l1_x, sz["x_a"]), after)
    ld_up = g.node(f"LdWup_e{e}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                   SnaxBingoKernelIdma1dCopyArgs(h[f"w_up_{e}"], l1_up, sz["w_up"]))
    ld_gate = g.node(f"LdWgate_e{e}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                     SnaxBingoKernelIdma1dCopyArgs(h[f"w_gate_{e}"], l1_gate, sz["w_up"]))
    ld_down = g.node(f"LdWdown_e{e}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                     SnaxBingoKernelIdma1dCopyArgs(h[f"w_down_{e}"], l1_down, sz["w_down"]))

    def proj(name, A, B, C_D, K, N, after):
        # int32tofp16_enable moves the narrowing onto the GEMM's own D port: an int32 beat
        # carries half as many values as an fp16 one, so converting at the consumer would
        # double both the beats it reads and the L1 the tile occupies.
        #
        # input_C_addr=0, NOT C_D. Every VersaCore GEMM computes D = A*B + C, and
        # accumPrevC=0 selects where C comes from: a NON-ZERO address reads C from memory
        # and adds it (gemm.h:96), a ZERO address adds nothing. Passing the output buffer
        # as C therefore reads it before anything has written it -- uninitialised TCDM,
        # which simulates as X -- and D = A*B + X = X. That X then flows through the whole
        # lane (swiglu -> reshape -> quant -> down GEMM), gets DMA'd into the combine's
        # landing slot, and finally reaches the host, where check_result loads it and the
        # CVA6 issues on an unknown operand. Cost of the wrong value here: one X-poisoned
        # winner slot and a dead host. This is the same "null C" FA uses on its first PV.
        return g.node(name, GEMM_CORE, "__snax_bingo_kernel_gemm_full",
                      SnaxBingoKernelGemmFullArgs(
                          input_A_addr=A, input_B_addr=B, input_C_addr=0,
                          output_D_addr=C_D, M=hw["M_T"], K=K, N=N,
                          array_shape_idx=hw["arrayShapeIdx"],
                          transpose_A=hw["transposeA"], transpose_B=hw["transposeB"],
                          accumPrevC=0, int32tofp16_enable=1), after)

    gemm_up = proj(f"GemmUp_e{e}", l1_x, l1_up, d_up, hw["K_up"], hw["N_up"],
                   [ld_x, ld_up])
    gemm_gate = proj(f"GemmGate_e{e}", l1_x, l1_gate, d_gate, hw["K_up"], hw["N_up"],
                     [ld_x, ld_gate])

    # Elementwise over the packed D-block buffer: no reshape needed in front of it.
    cols = sz["d_hidden"] // 2
    swiglu = g.node(f"Swiglu_e{e}", SIMD_CORE,
                    "__snax_bingo_kernel_simd_swiglu_f16_f16",
                    SnaxBingoKernelSimdSwigluF16F16Args(d_gate, d_up, d_act,
                                                        rows=1, cols=cols),
                    [gemm_up, gemm_gate])
    reshape = g.node(f"Reshape_e{e}", XDMA_CORE, "__snax_bingo_kernel_xdma_6d",
                     reshape_6d(hw, d_act, a_act), swiglu)
    quant = g.node(f"Quant_e{e}", SIMD_CORE, "__snax_bingo_kernel_simd_fp16_to_int8",
                   SnaxBingoKernelSimdFp16ToInt8Args(
                       a_act, a_i8, beats=sz["act_a"] // 64, rows=1,
                       inv_scale_f32bits=inv_scale_bits), reshape)
    gemm_down = proj(f"GemmDown_e{e}", a_i8, l1_down, l1_y, hw["K_down"], hw["N_down"],
                     [quant, ld_down])

    # The SIMD has no AXI port, so the combine can only read its OWN cluster's L1. Each
    # expert therefore pushes its result into a slot of one buffer on cluster 0, which is
    # also what lets the combine address the operands as base + e * stride.
    push = g.node(f"PushY_e{e}", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(l1_y, ybuf.view(e * sz["y"]), sz["y"]),
                  gemm_down)
    return push, l1_y, [ld_x, ld_up, ld_gate, ld_down, gemm_up, gemm_gate, swiglu,
                        reshape, quant, gemm_down, push]


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir,
                               args.output_offload_file_name):
        return
    hw = load_hw_params(args.cfg, args.hwcfg)
    sz = sizes(hw)
    E, k = hw["num_experts"], hw["top_k"]
    if E > platform["num_clusters_per_chiplet"]:
        raise ValueError(
            f"{E} experts but only {platform['num_clusters_per_chiplet']} clusters; this "
            f"workload puts one expert per cluster so each lane is an independent branch.")

    data = generate_moe_data(hw)
    print(f"MoE4: {E} experts, top-{k}, T={hw['tokens']} d={hw['d_model']} "
          f"h={hw['d_hidden']} on mesh "
          f"({hw['meshRow']},{hw['tileSize']},{hw['meshCol']}), "
          f"int8 scale {data['int8_scale']}")
    st = DataStaging(platform)
    h = stage(st, data, hw)
    if args.data_h:
        st.emit(str(args.data_h), args.output_dir)

    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
                   num_cores_per_cluster=platform["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[CHIPLET_ID],
                   dep_tag_width=platform["dep_tag_width"])
    g = G(dfg, COMBINE_CLUSTER)

    # ---- router ------------------------------------------------------------------------
    # The softmax publishes p to its scratchpad, which is where the gating kernel reads it
    # from; nothing else has to hand it over.
    probs = BingoMemAlloc("moe_probs", size=E * 4, mem_level="L3", chip_id=CHIPLET_ID)
    router = g.node("Router", HOST_CORE, "__host_bingo_kernel_softmax",
                    HostBingoKernelAraSoftmaxF32Args(
                        input_addr=h["logits"], output_addr=probs,
                        num_rows=1, row_length=E))

    # ---- the fork ----------------------------------------------------------------------
    # One branch per expert, so each gets its own CERF group and is skipped independently.
    fork = dfg.bingo_conditional_fork(router, {"mode": "top_k", "k": k})
    ybuf = g.l1("moe_ybuf", E * sz["y"], cluster=COMBINE_CLUSTER)

    # Zero the landing area before anything pushes into it. This is what makes the
    # per-expert check below unconditional: a winner overwrites its slot, a loser's stays
    # zero, so ONE ungated check verifies both that the winners computed correctly and
    # that the losers really were skipped. Without it a loser's slot holds whatever the
    # previous dispatch left there and the check could only run on the winners.
    zero = g.node("ZeroYbuf", DMA_CORE, "__snax_bingo_kernel_idma_1d_copy",
                  SnaxBingoKernelIdma1dCopyArgs(h["ybuf_zero"], ybuf, E * sz["y"]),
                  router, cluster=COMBINE_CLUSTER)

    pushes, lanes = [], []
    for e in range(E):
        push, _, lane = emit_expert_ffn(g.at(e), hw, sz, e, h, ybuf,
                                        data["int8_scale_f32bits"], after=zero)
        pushes.append(push)
        lanes.append(lane)
        fork.branch(lane)

    # ---- the combine -------------------------------------------------------------------
    out = g.l1("moe_out", sz["y"], cluster=COMBINE_CLUSTER)
    combine = g.node("Combine", SIMD_CORE,
                     "__snax_bingo_kernel_simd_moe_combine_f16",
                     SnaxBingoKernelSimdMoeCombineF16Args(
                         output_addr=out, src_base_addr=ybuf, src_stride=sz["y"],
                         rows=1, cols=sz["y"] // 2),
                     cluster=COMBINE_CLUSTER)
    # activation, weights and num_inputs are bound by the lowering, from the same gate.
    fork.combine(combine, inputs=pushes, kind="weighted_sum", weights=fork.weights)

    # ---- every expert's own result, checked as ONE ungated node ------------------------
    # The whole landing area at once: a winner's slot must hold its y, a loser's must
    # still be the zeros ZeroYbuf wrote. That second half is the part worth having -- it
    # is the only check in the graph that would notice an expert running when the router
    # did not select it, which a correct-looking combined output would otherwise hide.
    #
    # This is deliberately NOT four gated host checks. Those deadlock: a CERF-skipped task
    # fires its dep_set immediately from the checkout queue, while a task that really runs
    # fires only after its done arrives, so skipping lets a producer's set overtake the
    # drain of an earlier consumer sharing the same reused dep tag. On a core carrying both
    # gated and ungated tasks that is reachable, and bingo_validate_no_hang does not model
    # it -- the CERF scenario sweep in bingo_sim_check is what catches it.
    yb = data["ybuf_golden"].astype(np.float16)
    l3_yb = BingoMemAlloc("moe_ybuf_l3", size=E * sz["y"], mem_level="L3",
                          chip_id=CHIPLET_ID)
    st_yb = g.node("StoreYbuf", HOST_CORE, "__host_bingo_kernel_idma",
                   HostBingoKernelIdmaArgs(ybuf, l3_yb, E * sz["y"]), combine)
    g.node("CheckYbuf", HOST_CORE, "__host_bingo_kernel_check_result",
           HostBingoKernelCheckResultArgs(
               h["ybuf_golden"], l3_yb, name="moe_y_per_expert",
               check_type=CHECK_FP16_TOL, num_elements=E * sz["y"] // 2,
               tolerance=float(8 * np.max(np.spacing(np.abs(yb) + 1e-3)))), st_yb)

    # ---- what the layer produced -------------------------------------------------------
    p = data["softmax_golden"]
    w = data["weight_golden"]
    o = data["out_golden"].astype(np.float16)

    l3_out = BingoMemAlloc("moe_out_l3", size=sz["y"], mem_level="L3", chip_id=CHIPLET_ID)
    st_out = g.node("StoreOut", HOST_CORE, "__host_bingo_kernel_idma",
                    HostBingoKernelIdmaArgs(out, l3_out, sz["y"]), combine)
    g.node("CheckOut", HOST_CORE, "__host_bingo_kernel_check_result",
           HostBingoKernelCheckResultArgs(
               h["out_golden"], l3_out, name="moe_out", check_type=CHECK_FP16_TOL,
               num_elements=sz["y"] // 2,
               tolerance=float(8 * np.max(np.spacing(np.abs(o) + 1e-3)))), st_out)

    # The router's own output. Ordered AFTER the combine, not after the router.
    #
    # The host core dispatches IN ORDER, so a check placed behind `router` sits between the
    # router and the gating node and delays every expert lane by its whole duration.
    # Measured at T=32/d=256/h=128: 95,016 ns of the 371,640 ns layer -- 25.6% of the
    # window spent printing a four-element comparison to the UART while four clusters
    # waited for a predicate that was already computable. `probs` is written once by the
    # router and read by nothing else, so checking it later is exactly as strong.
    g.node("CheckSoftmax", HOST_CORE, "__host_bingo_kernel_check_result",
           HostBingoKernelCheckResultArgs(
               h["softmax_golden"], probs, name="moe_softmax",
               check_type=CHECK_FP32_TOL, num_elements=E,
               tolerance=float(8 * np.max(np.spacing(p)))), combine)

    # The renormalised weights, which is the one thing the gating kernel produces that
    # nothing else would notice being wrong: a mis-scaled weight vector still yields a
    # plausible output. fork.weights is a view into the gate's own selection record.
    gate_w = BingoMemAlloc("moe_w_l3", size=E * 4, mem_level="L3", chip_id=CHIPLET_ID)
    st_w = g.node("StoreW", HOST_CORE, "__host_bingo_kernel_idma",
                  HostBingoKernelIdmaArgs(fork.weights, gate_w, E * 4), combine)
    g.node("CheckW", HOST_CORE, "__host_bingo_kernel_check_result",
           HostBingoKernelCheckResultArgs(
               h["weight_golden"], gate_w, name="moe_w", check_type=CHECK_FP32_TOL,
               num_elements=E, tolerance=float(8 * np.max(np.spacing(w + 1e-6)))), st_w)

    # Diagnostic only -- every actual check above is a node. This just makes a routing
    # surprise legible in the UART instead of showing up as a failed output check.
    # The C name comes from the handle the compiler allocated, not from a guess at what it
    # called the gate: fork.activation is a view into that record, so this cannot drift if
    # the node is renamed.
    act_c = f"ptr_{fork.activation.base.name} + {fork.activation.offset}"
    post = [
        "{",
        f"    uint8_t* __act = (uint8_t*)({act_c});",
        "    float* __w = (float*)ptr_moe_w_l3;",
        '    printf_safe("[Routing] selected:");',
        f"    for (int __i = 0; __i < {E}; __i++)",
        '        if (__act[__i]) printf_safe(" e%d(w=0x%08x)", __i, *(uint32_t*)&__w[__i]);',
        '    printf_safe("\\r\\n");',
        "}",
    ]

    extra = [os.path.basename(str(args.data_h))] if args.data_h else None
    dfg.bingo_compile_dfg(app_name=f"MoE4 ({E}E top-{k} FFN)",
                          output_dir=args.output_dir,
                          output_file_name=args.output_offload_file_name,
                          extra_include_header_list=extra,
                          post_execute_code=post)
    print(f"Generated: {os.path.join(args.output_dir, args.output_offload_file_name)}")


if __name__ == "__main__":
    main()
