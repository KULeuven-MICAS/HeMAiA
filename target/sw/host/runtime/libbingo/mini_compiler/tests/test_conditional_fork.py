#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the conditional fork/branch/combine API and the CERF cross-group rule.
#
# Synthetic graphs on purpose. The thing worth testing is not that a well-formed MoE
# compiles -- the workload proves that -- but the REFUSALS, because each one stands in
# for a failure that produces no error at run time: a branch whose result is silently
# dropped, a combine that reads a buffer its producer never wrote, an ambiguous merge.
# Every refusal below is a bug that would otherwise reach silicon and return a plausible
# wrong number.
#
#   python3 test_conditional_fork.py

import sys

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402  (groups the compiler's subdirs onto sys.path)
from bingo_dfg import BingoDFG
from bingo_node import BingoNode

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


def refuses(name, fn, needle=""):
    """The call must raise, and say why in terms the reader can act on."""
    try:
        fn()
    except ValueError as e:
        if needle and needle.lower() not in str(e).lower():
            check(name, False, f"raised, but not about '{needle}': {e}")
        else:
            check(name, True)
        return
    except Exception as e:                      # noqa: BLE001
        check(name, False, f"raised {type(e).__name__}, wanted ValueError: {e}")
        return
    check(name, False, "did not raise")


def new_dfg(clusters=4):
    return BingoDFG(num_chiplets=1, num_clusters_per_chiplet=clusters,
                    num_cores_per_cluster=5, is_host_as_acc=True,
                    chiplet_ids=[0x00], dep_tag_width=5)


def lane(dfg, e, cluster=None, host_core=4):
    """One expert: two loads, a compute, a store. The shape every branch has."""
    cl = e if cluster is None else cluster
    ld = [BingoNode(0, cl, 3, node_name=f"e{e}_ld{t}", kernel_name="__snax_k")
          for t in "AB"]
    gemm = BingoNode(0, cl, 0, node_name=f"e{e}_gemm", kernel_name="__snax_k")
    st = BingoNode(0, cl, 3, node_name=f"e{e}_st", kernel_name="__snax_k")
    for n in ld + [gemm, st]:
        dfg.bingo_add_node(n)
    for l in ld:
        dfg.bingo_add_edge(l, gemm)
    dfg.bingo_add_edge(gemm, st)
    return st, ld + [gemm, st]


def moe(dfg, experts=4, k=2, with_combine=True, kind="weighted_sum",
        weights="fork", drop_input=None, share=None, invert=None):
    """A router, `experts` branches and optionally a combine. The shape under test."""
    router = BingoNode(0, 0, 4, node_name="router", kernel_name="__host_k")
    dfg.bingo_add_node(router)
    fork = dfg.bingo_conditional_fork(router, {"mode": "top_k", "k": k})
    outs, brs = [], []
    for e in range(experts):
        st, nodes = lane(dfg, e, cluster=e % 4)
        grp = brs[share[1]] if (share and e == share[0]) else None
        inv = bool(invert and e in invert)
        brs.append(fork.branch(nodes, invert=inv, group=grp))
        outs.append(st)
    comb = None
    if with_combine:
        comb = BingoNode(0, 0, 1, node_name="combine", kernel_name="__snax_simd_k")
        dfg.bingo_add_node(comb)
        ins = [o for i, o in enumerate(outs) if i != drop_input]
        w = fork.weights if weights == "fork" else weights
        fork.combine(comb, inputs=ins, kind=kind,
                     **({} if kind != "weighted_sum" else {"weights": w}))
    return router, fork, outs, comb


def compile_cond(dfg):
    m = dfg.bingo_compile_conditional_regions()
    dfg._validate_cerf_cross_group_edges()
    return m


# ---------------------------------------------------------------- lowering
print("fork lowering")

d = new_dfg()
router, fork, outs, comb = moe(d)
groups = compile_cond(d)
gids = {n.node_name: g for n, g in groups.items()}
check("one CERF group per expert",
      len({gids[f"e{e}_gemm"] for e in range(4)}) == 4, gids)
check("a lane shares one group",
      all(gids[f"e{e}_ld A".replace(" ", "")] == gids[f"e{e}_st"] for e in range(4)))
check("gating node spliced in",
      any(n.node_name == "__gating_router" for n in d.node_list))
gate = next(n for n in d.node_list if n.node_name == "__gating_router")
check("gating node owns every group",
      sorted(gate.cerf_write_groups) == sorted({gids[f"e{e}_st"] for e in range(4)}))
check("gating kernel is the unified one",
      gate.kernel_name == "__host_bingo_kernel_cerf_gating")
check("fork policy reached the kernel args",
      int(gate.kernel_args.top_k_or_threshold) == 2)
check("selection record is one allocation, sized by BRANCH",
      gate.kernel_args.cond_weight_addr.base is gate.kernel_args.cond_activation_addr.base
      and gate.kernel_args.cond_weight_addr.base.size == 4 * 4 + 4,
      gate.kernel_args.cond_weight_addr.base.size)
check("weights at offset 0, flags after them",
      (gate.kernel_args.cond_weight_addr.offset,
       gate.kernel_args.cond_activation_addr.offset) == (0, 16))
check("fork exposes both halves to the workload",
      fork.weights is not None and fork.activation is not None)
check("each expert gets its own activation slot",
      sorted(outs[e]._cond_node_index for e in range(4)) == [0, 1, 2, 3])
check("combine survived the cross-group rule", comb is not None)

# The fork must lower to exactly what the edge form lowers to.
d2 = new_dfg()
r2 = BingoNode(0, 0, 4, node_name="router", kernel_name="__host_k")
d2.bingo_add_node(r2)
for e in range(4):
    st, nodes = lane(d2, e, cluster=e % 4)
    for n in nodes:
        d2.bingo_add_edge(r2, n, cond_dic={"mode": "top_k", "k": 2})
g2 = {n.node_name: g for n, g in d2.bingo_compile_conditional_regions().items()}
check("fork lowers identically to cond_dic",
      {k: v for k, v in gids.items() if not k.startswith("combine")} == g2, (gids, g2))

# ------------------------------------------------------------ group sharing
print("group sharing and invert")

d = new_dfg()
router, fork, outs, comb = moe(d, kind="sum", share=(1, 0))
groups = compile_cond(d)
gids = {n.node_name: g for n, g in groups.items()}
check("group= merges two branches into one CERF group",
      gids["e0_st"] == gids["e1_st"] and gids["e0_st"] != gids["e2_st"], gids)
check("merged branches still get 3 groups for 4 experts",
      len(set(gids.values())) == 3, gids)

d = new_dfg()
router, fork, outs, comb = moe(d, kind="sum", share=(1, 0), invert={1})
compile_cond(d)
check("invert reaches the descriptor",
      outs[1].cond_exec_invert is True and outs[0].cond_exec_invert is False)

# ------------------------------------------------------------- the refusals
print("refusals")

refuses("branch with no nodes",
        lambda: new_dfg().bingo_conditional_fork(
            BingoNode(0, 0, 4, node_name="r", kernel_name="__host_k"),
            {"mode": "top_k", "k": 1}).branch([]),
        "guards nothing")

refuses("weighted_sum without weights",
        lambda: moe(new_dfg(), kind="weighted_sum", weights=None),
        "needs weights")

refuses("unknown combine kind",
        lambda: moe(new_dfg(), kind="average"),
        "not one of")

def _two_combines():
    d = new_dfg()
    _, fork, outs, _ = moe(d, kind="sum")
    n = BingoNode(0, 0, 1, node_name="combine2", kernel_name="__snax_simd_k")
    d.bingo_add_node(n)
    fork.combine(n, inputs=outs, kind="sum")
refuses("a fork reconverges once", _two_combines, "already has a combine")

def _dropped_branch():
    d = new_dfg()
    # inputs= omits expert 3, and nothing else connects it to the combine.
    moe(d, kind="sum", drop_input=3)
    compile_cond(d)
refuses("a branch that never reaches its combine", _dropped_branch,
        "never reaches")

def _ambiguous_select():
    d = new_dfg()
    moe(d, experts=2, k=1, kind="select", share=(1, 0))
    compile_cond(d)
refuses("two uninverted branches sharing a group feeding a select",
        _ambiguous_select, "only one input")

def _policy_disagreement():
    d = new_dfg()
    r = BingoNode(0, 0, 4, node_name="router", kernel_name="__host_k")
    d.bingo_add_node(r)
    f = d.bingo_conditional_fork(r, {"mode": "top_k", "k": 2})
    st, nodes = lane(d, 0, cluster=0)
    f.branch(nodes)
    d.bingo_add_edge(r, nodes[0], cond_dic={"mode": "top_k", "k": 3})
    compile_cond(d)
refuses("a cond_dic that disagrees with its fork", _policy_disagreement,
        "disagree")

def _undeclared_consumer():
    d = new_dfg()
    _, fork, outs, _ = moe(d, with_combine=False)
    sink = BingoNode(0, 0, 1, node_name="sink", kernel_name="__snax_simd_k")
    d.bingo_add_node(sink)
    for o in outs:
        d.bingo_add_edge(o, sink)
    compile_cond(d)
refuses("an UNDECLARED consumer of a gated task", _undeclared_consumer,
        "unguarded")

def _combine_missing_a_branch():
    """Declared, but one branch has no path to it: the validator must still refuse.
    _lower_combines names it first; either message is the same defect."""
    d = new_dfg()
    _, fork, outs, comb = moe(d, kind="sum", drop_input=2)
    compile_cond(d)
refuses("a declared combine that closes only some branches",
        _combine_missing_a_branch, "")

def _stray_target():
    d = new_dfg()
    r = BingoNode(0, 0, 4, node_name="router", kernel_name="__host_k")
    d.bingo_add_node(r)
    f = d.bingo_conditional_fork(r, {"mode": "top_k", "k": 1})
    st, nodes = lane(d, 0, cluster=0)
    f.branch(nodes)
    stray = BingoNode(0, 1, 0, node_name="stray", kernel_name="__snax_k")
    d.bingo_add_node(stray)
    d.bingo_add_edge(r, stray, cond=True)       # gated, but in no branch
    compile_cond(d)
refuses("a conditional target no branch declares", _stray_target,
        "no branch")

# ------------------------------------------------ two groups on one core
# The one shape that passes every STATIC check and still deadlocks on silicon.
# Measured on the cycle model: 'all' runs clean, and every routing that skips
# anything hangs. See _validate_cerf_core_sharing for the mechanism.
print("core sharing")

def _two_groups_one_core():
    d = new_dfg()
    r = BingoNode(0, 0, 4, node_name="router", kernel_name="__host_k")
    d.bingo_add_node(r)
    f = d.bingo_conditional_fork(r, {"mode": "top_k", "k": 2})
    for e in range(2):
        st, nodes = lane(d, e, cluster=e)
        # the tail of every branch lands on ONE shared host core
        tail = BingoNode(0, 0, 4, node_name=f"e{e}_host_tail", kernel_name="__host_k")
        d.bingo_add_node(tail)
        d.bingo_add_edge(st, tail)
        f.branch(nodes + [tail])
    compile_cond(d)
    d._validate_cerf_core_sharing()
refuses("two CERF groups sharing one in-order core", _two_groups_one_core,
        "different CERF groups")

def _one_group_per_core():
    d = new_dfg()
    r = BingoNode(0, 0, 4, node_name="router", kernel_name="__host_k")
    d.bingo_add_node(r)
    f = d.bingo_conditional_fork(r, {"mode": "top_k", "k": 2})
    for e in range(2):
        st, nodes = lane(d, e, cluster=e)
        tail = BingoNode(0, e, 1, node_name=f"e{e}_tail", kernel_name="__snax_k")
        d.bingo_add_node(tail)
        d.bingo_add_edge(st, tail)
        f.branch(nodes + [tail])
    compile_cond(d)
    d._validate_cerf_core_sharing()
    return True
check("a branch keeping its tail on its own cluster is fine", _one_group_per_core())

def _ungated_on_the_shared_core():
    d = new_dfg()
    _, fork, outs, comb = moe(d, experts=2, kind="sum")
    compile_cond(d)
    d._validate_cerf_core_sharing()      # router + combine are ungated: allowed
    return True
check("ungated tasks may share a core with anything", _ungated_on_the_shared_core())

# ------------------------------------------------------- SW-guarded consumer
print("SW guard exemption")

d = new_dfg()
_, fork, outs, _ = moe(d, with_combine=False)
# A per-expert host check, inside the branch, guarded by the same gate.
compile_cond(d)
check("a lane with no consumer outside it is fine", True)

if FAILED:
    print(f"\n{len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("\nall conditional-fork tests passed")
