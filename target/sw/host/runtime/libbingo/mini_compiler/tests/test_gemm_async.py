#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Tests for the queued GEMM dispatch: the start kernels' descriptors, the numbering and the start /
# wait pairs of GemmQueue, the core program order, and the FlashAttention sweep built on them.
# They check the compiler's side. Whether the array retires a queued dispatch when the wait says it
# did is the hardware's, and a simulation's to show.
#
#   python3 test_gemm_async.py

import os as _os
import sys as _sys

import networkx as nx

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402
from bingo_dfg import BingoDFG                                        # noqa: E402
from bingo_kernel_args import (SnaxBingoKernelGemmFaPvArgs, SnaxBingoKernelGemmFaQkArgs,   # noqa: E402
                               SnaxBingoKernelGemmFaPvStartArgs, SnaxBingoKernelGemmFaQkStartArgs,
                               SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdmaRingLoadArgs,
                               SnaxBingoKernelSimdFaSoftmaxArgs, set_simd_beat_bytes)
from libs import Ctx                                                  # noqa: E402
from libs.blocks import FaSweepCfg, GemmQueue, fa_sweep, fa_sweep_alloc   # noqa: E402

set_simd_beat_bytes(32)             # the paper cluster's SIMD: 16 FP16 lanes
FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        FAILED.append(name)


def refuses(name, fn, needle=""):
    try:
        fn()
    except ValueError as e:
        check(name, needle.lower() in str(e).lower(), f"raised, but not about {needle!r}: {e}")
        return
    except Exception as e:                                            # noqa: BLE001
        check(name, False, f"raised {type(e).__name__}, wanted ValueError: {e}")
        return
    check(name, False, "did not raise")


def new_ctx(chips=(0,)):
    d = BingoDFG(num_chiplets=len(chips), num_clusters_per_chiplet=2, num_cores_per_cluster=5,
                 is_host_as_acc=True, chiplet_ids=list(chips), dep_tag_width=5)
    return Ctx(dfg=d, mesh=(16, 4, 16), roles={"gemm": 0, "simd": 1, "xdma": 2, "dm": 3, "host": 4})


ROLES = {"gemm": 0, "simd": 1, "xdma": 2, "dm": 3}

print("\nthe start kernels' descriptors")
qk = SnaxBingoKernelGemmFaQkArgs(0x100, 0x200, 0, 0x300, 16, 144, 1, d_shift=3)
sq = SnaxBingoKernelGemmFaQkStartArgs(0x100, 0x200, 0, 0x300, M=16, K=144, N=1, d_shift=3, seq=5)
a = sq.get_c_field_assignments({})
check("a QK start names its own struct and carries seq",
      sq.get_struct_name() == "__snax_bingo_kernel_gemm_fa_start_args_t" and a["seq"] == "5"
      and sq.KERNEL_NAME == "__snax_bingo_kernel_gemm_fa_qk_start")
check("...with the blocking kernel's fields but no perf_addr",
      "perf_addr" not in a and a["M"] == "16" and a["K"] == "144" and a["d_shift"] == "3")
pv = SnaxBingoKernelGemmFaPvStartArgs(0x100, 0x200, 0x400, 0x400, M=32, K=64, N=1,
                                      flags=SnaxBingoKernelGemmFaPvArgs.B_KMAJOR
                                      | SnaxBingoKernelGemmFaPvArgs.C_COLSCALE, corr_addr=0x500, seq=6)
check("a PV start keeps PV's flags and corr address",
      pv.get_c_field_assignments({})["flags"] == "3" and pv.KERNEL_NAME.endswith("pv_start"))
refuses("a start numbers from 1", lambda: SnaxBingoKernelGemmFaQkStartArgs(1, 2, 0, 3, M=1, K=1, N=1, seq=0), "numbered")
refuses("a queued matmul takes no perf_addr",
        lambda: SnaxBingoKernelGemmFaQkStartArgs(1, 2, 0, 3, M=1, K=1, N=1, perf_addr=64, seq=1), "perf_addr")
refuses("a QK takes no PV flag, queued or not",
        lambda: SnaxBingoKernelGemmFaQkStartArgs(1, 2, 0, 3, M=1, K=1, N=1, flags=1, seq=1), "takes none")
rl = SnaxBingoKernelIdmaRingLoadArgs(0x10, 3, 0x20, 0x30, 4096, 0)
check("a ring load with release address 0 writes no release word",
      rl.get_c_field_assignments({})["release_addr"] == "(uint32_t)0")

print("\nthe queue: numbering, pairs, completion")
c = new_ctx((0, 1))
q = GemmQueue.of(c.dfg)
g00, g01, g10 = c.at(0), c.at(1), Ctx(**{**c.__dict__, "chiplet": 1}).at(0)
w1 = q.qk(g00, "qk0", qk)
w2 = q.pv(g00, "pv0", SnaxBingoKernelGemmFaPvArgs(1, 2, 0, 3, 32, 64, 1, flags=1))
w3 = q.qk(g01, "qk0b", qk)
w4 = q.dispatch(g00, "gemm", SnaxBingoKernelGemmFullArgs(
    input_A_addr=1, input_B_addr=2, input_C_addr=0, output_D_addr=3, M=1, K=4, N=1, array_shape_idx=0,
    transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1))
w5 = q.qk(g10, "qk0c", qk)
check("a cluster's dispatches are numbered 1, 2, 3 in creation order, per (chip, cluster)",
      [q.seq[(0, 0)], q.seq[(0, 1)], q.seq[(1, 0)]] == [3, 1, 1])
check("a dispatch returns its wait; the wait follows the start",
      w1.kernel_name == "__snax_bingo_kernel_gemm_wait" and c.dfg.has_edge(
          next(p for p in c.dfg.predecessors(w1)), w1))
check("the wait carries the dispatch's own number",
      [w.kernel_args.seq for w in (w1, w2, w4)] == [1, 2, 3] and w3.kernel_args.seq == 1)
check("gemm_full goes through the same queue as gemm_start",
      next(c.dfg.predecessors(w4)).kernel_name == "__snax_bingo_kernel_gemm_start")
refuses("a queue made one way is not reused the other",
        lambda: GemmQueue.of(c.dfg, enabled=False), "mixes")
b = new_ctx()
blk = GemmQueue.of(b.dfg, enabled=False).qk(b.at(0), "qk", qk)
check("disabled, the same call makes the blocking kernel", blk.kernel_name == "__snax_bingo_kernel_gemm_fa_qk")

print("\nthe core program order")


def program(n, deps=None):
    """n dispatches on one core, each after deps[i] (a list of earlier outputs); returns the DFG."""
    ctx = new_ctx()
    q = GemmQueue.of(ctx.dfg)
    g = ctx.at(0)
    ws = []
    for i in range(n):
        ws.append(q.qk(g, f"d{i}", qk, [ws[j] for j in (deps or {}).get(i, [])]))
    return ctx, q, ws


ctx, q, ws = program(3)
q.program_order(ctx.dfg, ROLES.values(), verbose=False)
core = sorted([t for t in ctx.dfg.node_list if t.assigned_core_id == 0], key=lambda t: t.node_id)
names = [t.node_name for t in core]
# the order the core is given: creation order with a wait behind the next start
order = list(nx.lexicographical_topological_sort(ctx.dfg, key=lambda t: t.node_id))
pos = {t.node_name: i for i, t in enumerate(order)}
check("start 1, start 2, wait 1: a wait trails the next start",
      pos["d1_cl0"] < pos["d0_w_cl0"] and pos["d2_cl0"] < pos["d1_w_cl0"])
check("...and the last wait stays last", pos["d2_w_cl0"] == max(pos.values()) and nx.is_directed_acyclic_graph(ctx.dfg))

ctx, q, ws = program(3, {1: [0]})          # dispatch 1 reads what dispatch 0 wrote: its start follows wait 0
q.program_order(ctx.dfg, ROLES.values(), verbose=False)
order = list(nx.lexicographical_topological_sort(ctx.dfg, key=lambda t: t.node_id))
pos = {t.node_name: i for i, t in enumerate(order)}
check("a start that depends on the wait keeps it in front (no cycle)",
      pos["d0_w_cl0"] < pos["d1_cl0"] and nx.is_directed_acyclic_graph(ctx.dfg))
check("...and the next pair still overlaps", pos["d2_cl0"] < pos["d1_w_cl0"])

print("\nthe FlashAttention sweep")


class Src:
    """A tile source over a pool of four slabs, in fetch order, like the weight ring's."""

    def __init__(self, g, nslab=4):
        self.g, self.n, self.i, self.last, self.log = g, nslab, 0, {}, []
        self.slab = [g.l1(f"slab{s}", 147456) for s in range(nslab)]

    def fetch(self, kind, j):
        s = self.i % self.n
        self.i += 1
        war = [self.last[s]] if s in self.last else []
        ld = self.g.node(f"ld_{kind}{j}", self.g.dm, "__snax_bingo_kernel_idma_ring_load",
                         SnaxBingoKernelIdmaRingLoadArgs(0x10, self.i, 0x20, self.slab[s], 1024, 0x40), war)
        self.log.append((kind, j, s))
        return ld, self.slab[s], s

    def release(self, token, task):
        self.last[token] = task


def build_sweep(nq, nt, async_, chunks=2, nq_cfg=None):
    ctx = new_ctx()
    g = ctx.at(0)
    cfg = FaSweepCfg(bc=256, nt=nt, nq=nq, k_s=3, k_o=5, a_exp=0.01, a_n_f32bits=0x3F800000,
                     gemm_async=async_)
    bufs = fa_sweep_alloc(lambda nm, nb: g.l1(nm, nb), cfg)
    src = Src(g)
    q8 = {i: (g.l1(f"q8_{i}", 9216), []) for i in range(nq)}
    prev, res = None, []
    for ch in range(chunks):
        prev = fa_sweep(g, cfg, bufs, src, q8, ROLES, prev=prev, name=f"c{ch}")
        res.append(prev)
    return ctx, g, cfg, src, res


ctx, g, cfg, src, res = build_sweep(2, 6, True)
nodes = ctx.dfg.node_list
kinds = lambda k: [t for t in nodes if t.kernel_name == k]
check("2 chunks x 2 query tiles x 6 tiles: 24 score and 24 PV starts, one wait each",
      len(kinds("__snax_bingo_kernel_gemm_fa_qk_start")) == 24
      and len(kinds("__snax_bingo_kernel_gemm_fa_pv_start")) == 24
      and len(kinds("__snax_bingo_kernel_gemm_wait")) == 48)
check("the cluster's dispatches are numbered 1..48 without a gap",
      sorted(t.kernel_args.seq for t in kinds("__snax_bingo_kernel_gemm_wait")) == list(range(1, 49)))
check("one softmax a tile and query tile, one normalise and transpose a query tile and sweep",
      len(kinds("__snax_bingo_kernel_simd_fa_softmax")) == 24
      and len(kinds("__snax_bingo_kernel_simd_mla_normalise")) == 4
      and len(kinds("__snax_bingo_kernel_xdma_transpose_2d")) == 4)
want = []
for ch in range(2):
    want += [x for j in range(6) for x in (("K", j), ("V", j))]
check("the tiles are fetched in the ring's order, K0 V0 K1 V1 ..., through the four slabs in turn",
      [(k, j) for k, j, _ in src.log] == want and [s for _, _, s in src.log] == [i % 4 for i in range(24)])
tile_slab = {}
for t in nodes:
    pass
q = GemmQueue.of(ctx.dfg)
q.program_order(ctx.dfg, ROLES.values(), verbose=False)
check("the graph with the cores' program order is acyclic", nx.is_directed_acyclic_graph(ctx.dfg))
order = list(nx.lexicographical_topological_sort(ctx.dfg, key=lambda t: t.node_id))
pos = {t.node_name: i for i, t in enumerate(order)}
# the GEMM core runs QK(j, *) and then PV(j - 1, *), with the next start before each wait
gc = [t.node_name for t in order if t.assigned_core_id == ROLES["gemm"]]
starts = [n for n in gc if "_w_" not in n]
check("the GEMM core issues the starts QK0 QK0 QK1 QK1 PV0 PV0 in that order",
      [n.split("_")[0] for n in starts[:6]] == ["c0qk0q0", "c0qk0q1", "c0qk1q0", "c0qk1q1", "c0pv0q0", "c0pv0q1"])
starts_before_wait = pos["c0qk1q0_cl0"] < pos["c0qk0q0_w_cl0"] + 4
check("a wait is not ahead of the next dispatch's start", starts_before_wait)
check("a tile's slab is not refilled before its last reader: K2 loads after both QK0",
      nx.has_path(ctx.dfg, next(t for t in nodes if t.node_name == "c0qk0q1_w_cl0"),
                  next(t for t in nodes if t.node_name == "ld_K2_cl0")))
check("...and V2 after both PV0 waits",
      nx.has_path(ctx.dfg, next(t for t in nodes if t.node_name == "c0pv0q1_w_cl0"),
                  next(t for t in nodes if t.node_name == "ld_V2_cl0")))
check("a PV start follows its softmax and the previous PV's wait (the accumulator chain)",
      nx.has_path(ctx.dfg, next(t for t in nodes if t.node_name == "c0pv0q0_w_cl0"),
                  next(t for t in nodes if t.node_name == "c0pv1q0_cl0")))
check("the next sweep's first softmax follows this one's normalise (the arena is read there)",
      nx.has_path(ctx.dfg, next(t for t in nodes if t.node_name == "c0norm0_cl0"),
                  next(t for t in nodes if t.node_name == "c1sm0q0_cl0")))
check("the last PV of a query tile leaves FP16 into its own buffer, once",
      sum(1 for t in kinds("__snax_bingo_kernel_gemm_fa_pv_start")
          if t.kernel_args.flags & SnaxBingoKernelGemmFaPvArgs.D_FP16) == 4)
check("the first PV of a sweep reads no C", sum(1 for t in kinds("__snax_bingo_kernel_gemm_fa_pv_start")
                                                if t.kernel_args.input_C_addr == 0) == 4)

ctx, g, cfg, src, res = build_sweep(2, 4, False, chunks=1)
check("with gemm_async off the same sweep makes the blocking kernels",
      len([t for t in ctx.dfg.node_list if t.kernel_name == "__snax_bingo_kernel_gemm_fa_qk"]) == 8
      and not [t for t in ctx.dfg.node_list if t.kernel_name == "__snax_bingo_kernel_gemm_wait"])
refuses("a one-tile sweep cannot fill the two-deep pipeline",
        lambda: FaSweepCfg(bc=256, nt=1, k_s=0, k_o=0, a_exp=1.0, a_n_f32bits=0), "tiles")

print("\na queued dispatch's operands live until its wait (static L1)")
# The shape of a fresh prompt's keys with two tiles a chip (fa_prefill_mc.py stage_kv): two dispatches
# read their own A, and tile 0's post-processing follows tile 0's wait only. The program order puts
# tile 1's start before tile 0's wait, so the post-processing follows tile 1's start; the array still
# streams tile 1's A until tile 1's wait, and the post-processing's buffer must not take its bytes.
from bingo_liveness import can_share, collect_handle_users, reachability   # noqa: E402
from bingo_kernel_args import SnaxBingoKernelIdma1dCopyArgs                  # noqa: E402
ctx = new_ctx()
q = GemmQueue.of(ctx.dfg)
g = ctx.at(0)
a0, a1, bw, d0, d1 = (g.l1(n, 4096) for n in ("a0", "a1", "b", "d0", "d1"))
post_buf = g.l1("post", 4096)


def full(a, d):
    return SnaxBingoKernelGemmFullArgs(input_A_addr=a, input_B_addr=bw, input_C_addr=0, output_D_addr=d, M=1, K=4,
                                       N=1, array_shape_idx=0, transpose_A=0, transpose_B=0, accumPrevC=0,
                                       int32tofp16_enable=1)


w0 = q.dispatch(g, "gkv_t0", full(a0, d0))
w1 = q.dispatch(g, "gkv_t1", full(a1, d1))
post = g.node("post_t0", g.xdma, "__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(d0, post_buf, 4096),
              [w0])
q.program_order(ctx.dfg, ROLES.values(), verbose=False)
s1 = next(ctx.dfg.predecessors(w1))
nodes = sorted(ctx.dfg.node_list, key=lambda t: t.node_id)
reach = reachability(ctx.dfg, nodes)
hu = collect_handle_users(nodes)
check("the program order puts tile 1's start before tile 0's post-processing", post in reach[s1])
check("tile 1's wait is a user of the A its start reads", w1 in hu[id(a1)][1])
check("tile 1's A and the post-processing's buffer may not share bytes",
      not can_share(hu[id(a1)][1], hu[id(post_buf)][1], reach))
check("tile 0's A, its wait before the post-processing, still may",
      can_share(hu[id(a0)][1], hu[id(post_buf)][1], reach))

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    raise SystemExit(1)
print("all passed")
