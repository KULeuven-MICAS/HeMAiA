#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for static L1 liveness and packing.
#
# These use synthetic graphs on purpose. FlashAttention exercises the happy path -- every
# buffer it has is long-lived and every scratchpad is single-node -- so running the packer on
# it proves almost nothing about the cases that would corrupt memory. What has to be tested is
# the REFUSALS: two buffers whose users are not ordered must never be given the same bytes, and
# the checker must say so even when the packer has been sabotaged into producing that layout.
#
#   python3 test_static_l1.py

import sys

import networkx as nx

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402  (groups the compiler's subdirs onto sys.path)
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView
from bingo_liveness import (collect_handle_users, check_handle_identity, reachability,
                            can_share, build_interference, extend_users_for_engine_drain)
from bingo_l1_packer import (pack, verify, pack_scratchpad_slots, verify_scratchpad_slots,
                             scratchpad_users, _align_up, check_emitted_arena_text)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


class FakeArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeNode:
    _ids = iter(range(10000))

    def __init__(self, name, cluster=0, core=0, args=None, chip=0):
        self.node_id = next(FakeNode._ids)
        self.node_name = name
        self.kernel_name = "__snax_test"
        self.kernel_args = args
        self.assigned_chiplet_id = chip
        self.assigned_cluster_id = cluster
        self.assigned_core_id = core
        self._gating_node = None
        self._pred_source_node = None

    def __repr__(self):
        return f"<{self.node_name}>"


def buf(name, size=1024, cluster=0, chip=0):
    return BingoMemAlloc(name, size, "L1", chip, cluster)


def graph(nodes, edges):
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n)
    for a, b in edges:
        g.add_edge(a, b)
    return g


# ---------------------------------------------------------------------------------------
print("\n1. ORDERED users may share; UNORDERED users may not")
A, B = buf("A"), buf("B")
n1 = FakeNode("writes_A", args=FakeArgs(dst=A))
n2 = FakeNode("reads_A", args=FakeArgs(src=A))
n3 = FakeNode("writes_B", args=FakeArgs(dst=B))
n4 = FakeNode("reads_B", args=FakeArgs(src=B))
g = graph([n1, n2, n3, n4], [(n1, n2), (n2, n3), (n3, n4)])   # a chain: A dies before B starts
hu = collect_handle_users([n1, n2, n3, n4])
desc = reachability(g, [n1, n2, n3, n4])
check("chain: A and B can share", can_share(hu[id(A)][1], hu[id(B)][1], desc))

g2 = graph([n1, n2, n3, n4], [(n1, n2), (n3, n4)])            # two independent chains
desc2 = reachability(g2, [n1, n2, n3, n4])
check("parallel: A and B canNOT share", not can_share(hu[id(A)][1], hu[id(B)][1], desc2))

# ---------------------------------------------------------------------------------------
print("\n2. A cross-cluster reader extends a buffer's life (the FA V-pull shape)")
V = buf("v8", 65536, cluster=0)
W = buf("other", 65536, cluster=0)
prod = FakeNode("cl0_loads_V", cluster=0, args=FakeArgs(dst=V))
pull1 = FakeNode("cl1_pulls_V", cluster=1, args=FakeArgs(src=V))
late = FakeNode("cl0_uses_other", cluster=0, args=FakeArgs(dst=W))
# cl0 -> cl1 pull, and independently cl0 runs a later node. The pull is NOT ordered against it.
gx = graph([prod, pull1, late], [(prod, pull1), (prod, late)])
hux = collect_handle_users([prod, pull1, late])
descx = reachability(gx, [prod, pull1, late])
check("V (read cross-cluster) canNOT share with an unordered same-cluster buffer",
      not can_share(hux[id(V)][1], hux[id(W)][1], descx))
# now order the pull before the later node -- sharing becomes legal
gy = graph([prod, pull1, late], [(prod, pull1), (pull1, late)])
descy = reachability(gy, [prod, pull1, late])
check("once the pull precedes it, they CAN share",
      can_share(hux[id(V)][1], hux[id(W)][1], descy))

# ---------------------------------------------------------------------------------------
print("\n3. Views count as uses of their base")
BASE = buf("base", 4096)
VIEW = BASE.view(64)
nb = FakeNode("writes_base", args=FakeArgs(dst=BASE))
nv = FakeNode("reads_view", args=FakeArgs(src=VIEW))
gv = graph([nb, nv], [(nb, nv)])
huv = collect_handle_users([nb, nv])
check("a view does not create a second buffer", len(huv) == 1)
check("the view's node is a user of the base", nv in huv[id(BASE)][1])

# ---------------------------------------------------------------------------------------
print("\n4. Handles inside lists are collected (multicast destinations)")
D0, D1 = buf("d0", 512, cluster=0), buf("d1", 512, cluster=1)
nm = FakeNode("multicast", args=FakeArgs(dsts=[D0, D1]))
hum = collect_handle_users([nm])
check("both list members collected", len(hum) == 2)

# ---------------------------------------------------------------------------------------
print("\n5. Identity check catches two objects denoting one buffer")
dup1, dup2 = buf("same", 256), buf("same", 256)
nd1 = FakeNode("n1", args=FakeArgs(a=dup1))
nd2 = FakeNode("n2", args=FakeArgs(a=dup2))
hud = collect_handle_users([nd1, nd2])
check("duplicate (name, cluster) reported", len(check_handle_identity(hud)) == 1)

# ---------------------------------------------------------------------------------------
print("\n6. Packer: parallel buffers get disjoint bytes, chained ones overlap")
nodes = [n1, n2, n3, n4]
pl_par, st_par = pack(collect_handle_users(nodes), desc2, "L1")
oA, oB = pl_par[id(A)][1], pl_par[id(B)][1]
check("parallel buffers do not overlap",
      not (oA < oB + _align_up(B.size) and oB < oA + _align_up(A.size)),
      f"A@{oA} B@{oB}")
pl_ch, st_ch = pack(collect_handle_users(nodes), desc, "L1")
check("chained buffers share the same offset",
      pl_ch[id(A)][1] == pl_ch[id(B)][1],
      f"A@{pl_ch[id(A)][1]} B@{pl_ch[id(B)][1]}")
check("chained peak is half of parallel peak",
      st_ch[(0, 0)]["peak"] * 2 == st_par[(0, 0)]["peak"],
      f"{st_ch[(0,0)]['peak']} vs {st_par[(0,0)]['peak']}")

# ---------------------------------------------------------------------------------------
print("\n7. THE CHECKER CATCHES A SABOTAGED PLACEMENT")
# Take the SAFE parallel placement and force the two buffers on top of each other, exactly
# what a liveness bug or a missing dependence edge would produce.
sabotaged = dict(pl_par)
sabotaged[id(B)] = (B, pl_par[id(A)][1])
probs = verify(sabotaged, collect_handle_users(nodes), desc2, None, "L1")
check("overlap of unordered buffers is reported",
      any("UNSAFE OVERLAP" in p for p in probs), f"got {probs}")
check("the safe placement reports nothing",
      verify(pl_par, collect_handle_users(nodes), desc2, None, "L1") == [])

# ---------------------------------------------------------------------------------------
print("\n8. Capacity is enforced")
big = buf("big", 600000)
nbig = FakeNode("uses_big", args=FakeArgs(a=big))
gb = graph([nbig], [])
hub = collect_handle_users([nbig])
descb = reachability(gb, [nbig])
plb, _ = pack(hub, descb, "L1")
check("a buffer past capacity is reported",
      any("past the" in p for p in verify(plb, hub, descb, 514816, "L1")))

# ---------------------------------------------------------------------------------------
print("\n9. Scratchpad slots: same core reuses, parallel cores do not")
c0a = FakeNode("c0a", cluster=0, core=0)
c0b = FakeNode("c0b", cluster=0, core=0)
c1a = FakeNode("c1a", cluster=0, core=1)
gs = graph([c0a, c0b, c1a], [(c0a, c0b)])       # same core ordered; core 1 independent
descs = reachability(gs, [c0a, c0b, c1a])
slots, users = pack_scratchpad_slots([c0a, c0b, c1a], descs)
check("same-core consecutive nodes share a slot", slots[c0a] == slots[c0b],
      f"{slots[c0a]} vs {slots[c0b]}")
check("a parallel core gets its own slot", slots[c1a] != slots[c0a])
check("slots are dense from 0", sorted(set(slots.values())) == list(range(len(set(slots.values())))))
check("slot verifier accepts it", verify_scratchpad_slots(slots, users, descs) == [])

print("\n10. A gating reader extends a scratchpad's life")
gate = FakeNode("gate", cluster=0, core=0)
gated = FakeNode("gated", cluster=0, core=1)
gated._gating_node = gate
after = FakeNode("after", cluster=0, core=0)
gg = graph([gate, gated, after], [(gate, gated), (gate, after)])
descg = reachability(gg, [gate, gated, after])
us = scratchpad_users([gate, gated, after])
check("the gated node is a user of the gating node's scratchpad", gated in us[gate])
sl2, us2 = pack_scratchpad_slots([gate, gated, after], descg)
check("gate and after do NOT share, because the gated reader is unordered against after",
      sl2[gate] != sl2[after], f"{sl2[gate]} vs {sl2[after]}")

# ---------------------------------------------------------------------------------------
print("\n11. The emitted arena puts the arg bump ABOVE the scratchpad slot block")
# This is the bug the RTL run would have hit and the unit tests could not see: buffers and
# slots are checked by separate oracles, and neither knows that device ARG structs are bump-
# allocated from the same arena. With the bump left at 0 they alias slot 0.
SP_A = "ALIGN_UP(sizeof(bingo_kernel_scratchpad_t), 64)"
def hdr(off_expr, reserved=5, top_slot=4):
    body = [f"uint64_t __bingo_l1_arena_chip00_cl0 = bingo_l1_alloc(0x00, 0,",
            f"    ALIGN_UP(sizeof(args_t), 64) + {reserved} * {SP_A});",
            f"uint64_t __bingo_l1_arena_chip00_cl0_off = {off_expr};"]
    for sl in range(top_slot + 1):
        body.append(f"x = (__bingo_l1_arena_chip00_cl0 + {sl} * {SP_A});")
    body.append("a = (__bingo_l1_arena_chip00_cl0 + __bingo_l1_arena_chip00_cl0_off);")
    return "\n".join(body)

bad = check_emitted_arena_text(hdr("0"))
check("bump at 0 over a 5-slot block is reported",
      any("would alias" in p for p in bad), f"got {bad}")
good = check_emitted_arena_text(hdr(f"5 * {SP_A}"))
check("bump above the block is accepted", good == [], f"got {good}")
short = check_emitted_arena_text(hdr(f"3 * {SP_A}", reserved=5, top_slot=4))
check("a bump that clears only part of the block is reported",
      any("would alias" in p for p in short), f"got {short}")
under = check_emitted_arena_text(hdr(f"5 * {SP_A}", reserved=3, top_slot=4))
check("an arena reserving fewer slots than are used is reported",
      any("reserves" in p for p in under), f"got {under}")
unpacked = check_emitted_arena_text(
    "uint64_t __bingo_l1_arena_chip00_cl0 = bingo_l1_alloc(0x00, 0, ALIGN_UP(sizeof(a), 64));\n"
    "uint64_t __bingo_l1_arena_chip00_cl0_off = 0;\n"
    "x = (__bingo_l1_arena_chip00_cl0 + __bingo_l1_arena_chip00_cl0_off);")
check("an unpacked header is not flagged", unpacked == [], f"got {unpacked}")

# ---------------------------------------------------------------------------------------
print("\n12. Every placed buffer keeps the runtime allocator's 256 B phase")
# The bug this guards: bingoHeapMalloc rounds fragments to 256 and returns frag+128, so every
# buffer it EVER produced shares one offset-mod-256. Packing at 128 is legal against the
# stated alignment and still broke FA, because Q landed on the other phase. See the comment on
# L1_ALIGNMENT.
import bingo_l1_packer as _pk
check("packer alignment is 256, not the promised 128", _pk.L1_ALIGNMENT == 256,
      f"got {_pk.L1_ALIGNMENT}")
sizes = [3000, 1024, 64, 100000, 17]
bufs = [buf(f"b{i}", z) for i, z in enumerate(sizes)]
nds = [FakeNode(f"n{i}", args=FakeArgs(a=b)) for i, b in enumerate(bufs)]
gph = nx.DiGraph()
for n in nds: gph.add_node(n)
for a, b in zip(nds, nds[1:]): gph.add_edge(a, b)   # a chain, so sharing is allowed
huP = collect_handle_users(nds)
plP, _ = pack(huP, reachability(gph, nds), "L1")
offs = [o for (_h, o) in plP.values()]
check("no buffer lands off the 256 B phase", all(o % 256 == 0 for o in offs),
      f"offsets {sorted(offs)}")

# ---------------------------------------------------------------------------------------
print("\n13. The engine-drain guard refuses the FA warm-up/K overlap")
# The shape that packing got wrong on RTL: a GEMM node writes a small buffer, a DMA on another
# core then fills a big buffer at the same address, and the GEMM core's NEXT node is what
# drains the first write. Graph ordering says the two buffers are disjoint in time; the
# hardware says otherwise, so the guard counts a node's same-core successor as a user too.
Wb = buf("warm", 1024)
Kb = buf("k8", 65536)
n_warm = FakeNode("WarmGemm", core=0, args=FakeArgs(d=Wb))
n_load = FakeNode("LoadK", core=2, args=FakeArgs(d=Kb))
n_qk = FakeNode("QK", core=0, args=FakeArgs(a=Kb))
gd = graph([n_warm, n_load, n_qk], [(n_warm, n_load), (n_load, n_qk), (n_warm, n_qk)])
hud2 = collect_handle_users([n_warm, n_load, n_qk])
descd = reachability(gd, [n_warm, n_load, n_qk])
check("without the guard the overlap is allowed (this is the bug)",
      can_share(hud2[id(Wb)][1], hud2[id(Kb)][1], descd))
hug = extend_users_for_engine_drain(hud2, gd)
check("with the guard it is refused",
      not can_share(hug[id(Wb)][1], hug[id(Kb)][1], descd))
check("the guard only ever lengthens a live range",
      all(hud2[h][1] <= hug[h][1] for h in hud2))
# and it must not refuse a genuinely safe pair on DIFFERENT cores with no same-core successor
check("a chain on one core still shares",
      can_share(*[extend_users_for_engine_drain(hu, g2)[i][1] for i in (id(A), id(B))], desc))

# ---------------------------------------------------------------------------------------
print("\n14. A kernel's declared placement order is collected and enforced")
# FA's SIMD softmax pairs operands across buffers -- it writes -m_new into s16's prefix beat
# and reads the row sum out of p8's trailing beat -- with the gap held in a uint32_t stride.
# The arena therefore has to sit below both. Breaking it does NOT corrupt the row maximum
# (that chain is arena-internal); it corrupts P and the row sum, and the run hangs before
# either is reported. So it has to be a build-time check, not something anyone remembers.
from bingo_l1_packer import collect_placement_order, check_placement_order
from bingo_kernel_args import SnaxBingoKernelSimdFaSoftmaxArgs as _FA

_ar = buf("fa_arena_0", 9856)
_s16 = buf("fa_s16_0", 32832)
_p8 = buf("fa_p8_0", 16448)
class _N:
    def __init__(self, a): self.kernel_args = a
_cons = collect_placement_order([_N(_FA(_s16, _p8, _ar, bc=512, dhead=128, tile_idx=0))])
check("both cross-buffer constraints are collected", len(_cons) == 2, f"got {_cons}")

_good = {id(_ar): (_ar, 0), id(_s16): (_s16, 195840), id(_p8): (_p8, 158464)}
_bad = {id(_ar): (_ar, 377856), id(_s16): (_s16, 262144), id(_p8): (_p8, 328192)}
check("the layout that passed 8/8 on RTL is accepted",
      check_placement_order(_good, _cons) == [])
check("the layout that hung on RTL is rejected",
      len(check_placement_order(_bad, _cons)) == 2)
check("a kernel that declares nothing is unaffected",
      collect_placement_order([_N(FakeArgs(a=_ar, b=_s16))]) == [])

print("\n15. The declared order GUIDES the packer, it does not only audit it")
# The point of declaring the constraint is that the allocator obeys it. Give the packer the
# worst possible heuristic for this workload -- size-descending, which sorts the 9,856 B arena
# last, behind s16 and p8 -- and it must still come out with the arena below both.
_k8 = buf("fa_k8_0", 65536)
_nodes2 = [_N(_FA(_s16, _p8, _ar, bc=512, dhead=128, tile_idx=0)), _N(FakeArgs(d=_k8))]
for _n, _id in zip(_nodes2, range(2)):
    _n.node_id = 900 + _id; _n.node_name = f"pl{_id}"; _n.kernel_name = "__snax"
    _n.assigned_chiplet_id = 0; _n.assigned_cluster_id = 0; _n.assigned_core_id = 0
    _n._gating_node = None; _n._pred_source_node = None
_g2 = graph(_nodes2, [(_nodes2[1], _nodes2[0])])
_hu2 = collect_handle_users(_nodes2)
_d2 = reachability(_g2, _nodes2)
_cons2 = collect_placement_order(_nodes2)

from bingo_l1_packer import StaticL1Options
_size_order = StaticL1Options(enable=True, order="size")
_pl_free, _ = pack(_hu2, _d2, "L1", opts=_size_order)                        # heuristic alone
_pl_guided, _ = pack(_hu2, _d2, "L1", constraints=_cons2, opts=_size_order)  # + the rule

check("size-descending alone violates the constraint (this is the bug)",
      len(check_placement_order(_pl_free, _cons2)) == 2)
check("the same heuristic with the rule satisfies it",
      check_placement_order(_pl_guided, _cons2) == [])
check("and it really is below both, not merely 'not flagged'",
      _pl_guided[id(_ar)][1] < _pl_guided[id(_s16)][1]
      and _pl_guided[id(_ar)][1] < _pl_guided[id(_p8)][1],
      f"arena@{_pl_guided[id(_ar)][1]} s16@{_pl_guided[id(_s16)][1]} p8@{_pl_guided[id(_p8)][1]}")

_cyc = [(_ar, _s16, "x"), (_s16, _ar, "y")]
try:
    pack(_hu2, _d2, "L1", constraints=_cyc)
    check("a circular constraint is refused", False, "no exception raised")
except ValueError as _e:
    check("a circular constraint is refused", "circular" in str(_e))

# ---------------------------------------------------------------------------------------
print("\n16. Static placement is an ARGUMENT, off by default, and ignores the environment")
# The interface matters as much as the algorithm here. A switch that changes generated code
# used to be an environment variable, and a container that forwarded no environment turned a
# "packed" build into an unpacked one that passed and looked like evidence. Off by default,
# passed in explicitly, and deaf to the environment is the whole point.
import os as _os3
check("the default is disabled", StaticL1Options().enable is False)
check("passing True enables it", StaticL1Options(enable=True).enable is True)
for _v in ("BINGO_L1_PACK", "BINGO_L1_ORDER", "BINGO_L1_GUARD", "BINGO_L1_ALIGN"):
    _os3.environ[_v] = "1" if _v.endswith("PACK") else "size" if _v.endswith("ORDER") else "512"
_env_opts = StaticL1Options()
check("the environment cannot enable it", _env_opts.enable is False)
check("the environment cannot change the order", _env_opts.order == "name")
check("the environment cannot change the alignment", _env_opts.alignment == 256)
check("the environment cannot add a guard", _env_opts.guard_bytes == 0)
for _v in ("BINGO_L1_PACK", "BINGO_L1_ORDER", "BINGO_L1_GUARD", "BINGO_L1_ALIGN"):
    _os3.environ.pop(_v, None)

# ---------------------------------------------------------------------------------------
print("\n17. The layout report names the node that is allowed to clobber each buffer")
# The debugging question this file exists to answer: data in buffer X went wrong -- who was
# allowed to write over it, and had they run yet? A row is only useful if `overwritten_by`
# names the FIRST node of the next occupant, not just any node that touches it.
import csv as _csv, tempfile as _tf, os as _osc
from bingo_l1_packer import write_layout_csv

_A, _B = buf("early", 1024), buf("late", 1024)
_w = FakeNode("WriteEarly", args=FakeArgs(d=_A))
_r = FakeNode("ReadEarly", args=FakeArgs(s=_A))
_w2 = FakeNode("WriteLate", args=FakeArgs(d=_B))
_r2 = FakeNode("ReadLate", args=FakeArgs(s=_B))
_ns = [_w, _r, _w2, _r2]
_gc = graph(_ns, [(_w, _r), (_r, _w2), (_w2, _r2)])       # a chain, so B may reuse A
_huc = collect_handle_users(_ns)
_dc = reachability(_gc, _ns)
_plc, _ = pack(_huc, _dc, "L1")
_rank = {n: i for i, n in enumerate(nx.topological_sort(_gc))}
_path = _osc.path.join(_tf.mkdtemp(), "static_l1_layout.csv")
_n = write_layout_csv(_path, _plc, _huc, _dc, _rank, StaticL1Options(enable=True))
check("a row per placed buffer", _n == 2, f"got {_n}")
_by = {r["buffer"]: r for r in _csv.DictReader(open(_path))}
check("the reused buffer names its successor's FIRST node",
      _by["early"]["overwritten_by"] == "WriteLate", f"got {_by['early']['overwritten_by']}")
check("and names the buffer that took the bytes",
      _by["early"]["reused_by_buffer"] == "late")
check("the survivor has no successor", _by["late"]["overwritten_by"] == "")
check("offsets and lifetimes are reported",
      _by["early"]["first_use"] == "WriteEarly" and _by["early"]["last_use"] == "ReadEarly"
      and _by["early"]["offset"] == _by["late"]["offset"],
      f"{_by['early']}")

# ---------------------------------------------------------------------------------------
print("\n18. The layout plot is a debugging aid and never breaks a build")
# It draws address against time, one rectangle per buffer. The contract that matters is that
# it is optional: matplotlib missing, a headless machine, an odd backend -- none of those may
# turn a working compile into a failing one.
from bingo_l1_packer import write_layout_plot
_png = _osc.path.join(_osc.path.dirname(_path), "static_l1_layout.png")
try:
    _got = write_layout_plot(_png, _plc, _huc, _dc, _rank, StaticL1Options(enable=True),
                             capacity=514816)
    _ok = _got is None or _osc.path.exists(_got)
    check("it writes a file, or declines cleanly when it cannot", _ok, f"returned {_got}")
except Exception as _e:
    check("it writes a file, or declines cleanly when it cannot", False, f"raised {_e!r}")
check("no buffers means no plot",
      write_layout_plot(_png, {}, {}, _dc, _rank, StaticL1Options(enable=True)) is None)

# ---------------------------------------------------------------------------------------
print()
if FAILED:
    print(f"{len(FAILED)} FAILURE(S): {', '.join(FAILED)}")
    sys.exit(1)
print("all static-L1 tests passed")
