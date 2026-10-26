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
# On the cycle model it deadlocks for every routing that skips a branch; see
# _validate_cerf_core_sharing for the mechanism.
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
# A conditional graph whose lanes have no consumer outside them compiles.
try:
    compile_cond(d)
    _compiled, _why = True, ""
except Exception as e:                      # noqa: BLE001
    _compiled, _why = False, f"{type(e).__name__}: {e}"
check("a lane with no consumer outside it is fine", _compiled, _why)

# ------------------------------------------------------------- 128 CERF groups
print("CERF group count")

def _wide(groups, experts, chips=(0x00,)):
    d = BingoDFG(num_chiplets=len(chips), num_clusters_per_chiplet=4, num_cores_per_cluster=5,
                 is_host_as_acc=True, chiplet_ids=list(chips), dep_tag_width=5,
                 cerf_groups=groups)
    _, fork, outs, _ = moe(d, experts=experts, kind="sum")
    return d, compile_cond(d)

d, g = _wide(128, 40)
gids = {n.node_name: x for n, x in g.items()}
check("128 groups: 40 experts get 40 groups, no sharing",
      len({gids[f"e{e}_st"] for e in range(40)}) == 40, sorted(set(gids.values())))
check("128 groups: groups past 31 are used", max(gids.values()) >= 32)
check("128 groups: the descriptor field is 7 bits",
      dict(d.bingo_task_desc_layout())["cond_exec_group_id"] == 7)
gate = next(n for n in d.node_list if n.node_name == "__gating_router")
check("the gating args carry 4 mask words",
      "cerf_controlled_mask[3]" in gate.kernel_args.get_c_field_assignments({}))
d, g = _wide(32, 40)
check("32 groups: 40 experts share 32 groups", len(set(g.values())) == 32)
check("32 groups: the descriptor field stays 5 bits",
      dict(d.bingo_task_desc_layout())["cond_exec_group_id"] == 5)

# A cross-die branch takes a window group; a die-local one stays out of the window.
def _two_chip():
    d = BingoDFG(num_chiplets=2, num_clusters_per_chiplet=2, num_cores_per_cluster=5,
                 is_host_as_acc=True, chiplet_ids=[0x00, 0x01], dep_tag_width=5,
                 cerf_groups=128)
    r = BingoNode(0, 0, 4, node_name="router", kernel_name="__host_k")
    d.bingo_add_node(r)
    f = d.bingo_conditional_fork(r, {"mode": "static"})
    far = BingoNode(0x01, 0, 0, node_name="far", kernel_name="__snax_k")
    near = BingoNode(0x00, 1, 0, node_name="near", kernel_name="__snax_k")
    for n in (far, near):
        d.bingo_add_node(n)
        f.branch([n])
    comb = BingoNode(0, 0, 1, node_name="combine", kernel_name="__snax_k")
    d.bingo_add_node(comb)
    f.combine(comb, inputs=[far, near], kind="sum")
    return d, far, near

d, far, near = _two_chip()
compile_cond(d)
check("cross-die branch: a window group (< 8)", far.cond_exec_group_id < 8, far.cond_exec_group_id)
check("die-local branch beside it: out of the window", near.cond_exec_group_id >= 8,
      near.cond_exec_group_id)

# ------------------------------------------------------------------ tile fork
print("tile fork")

def _tiles(slots=2, tiles=4, one_core=True, chip_of=None, extra_edge=None):
    d = BingoDFG(num_chiplets=2, num_clusters_per_chiplet=2, num_cores_per_cluster=4,
                 is_host_as_acc=True, chiplet_ids=[0x00, 0x01], dep_tag_width=4,
                 cerf_groups=128)
    disp = BingoNode(0, 0, 3, node_name="dispatch", kernel_name="__snax_k")
    d.bingo_add_node(disp)
    f = d.bingo_conditional_fork(disp, {"mode": "tiles", "rows": 8, "counts": 0x1000})
    comb = BingoNode(0, 0, 2, node_name="combine", kernel_name="__snax_k")
    d.bingo_add_node(comb)
    ins, by = [], {}
    for s in range(slots):
        for t in range(tiles):
            chip = chip_of(s, t) if chip_of else 0
            g = BingoNode(chip, 0, 3, node_name=f"s{s}_t{t}_g", kernel_name="__snax_k")
            m = BingoNode(chip, 0, 0 if one_core else s, node_name=f"s{s}_t{t}_m",
                          kernel_name="__snax_k")
            d.bingo_add_node(g); d.bingo_add_node(m); d.bingo_add_edge(g, m)
            f.branch([g, m], slot=s, tile=t)
            ins.append(m); by[(s, t)] = (g, m)
    if extra_edge:
        a, b = extra_edge
        d.bingo_add_edge(by[a][1], by[b][0])
    f.combine(comb, inputs=ins, kind="sum")
    return d, f, by

d, f, by = _tiles()
compile_cond(d)
lay = f._tile_layout
check("tile layout: block 4, slots one after the other",
      (lay["block"], lay["slots"], lay["tiles"]) == (4, 2, 4), lay)
check("tile group = base + slot * block + tile",
      all(by[(s, t)][0].cond_exec_group_id == lay["base"] + s * 4 + t
          for s in range(2) for t in range(4)))
gate = next(n for n in d.node_list if n.node_name == "__gating_dispatch")
check("the tile gate is the device kernel on the source's core",
      gate.kernel_name == "__snax_bingo_kernel_cerf_tiles"
      and (gate.assigned_cluster_id, gate.assigned_core_id) == (0, 3))
check("the tile gate's args carry the layout",
      (gate.kernel_args.group_base, gate.kernel_args.group_block, gate.kernel_args.tiles,
       gate.kernel_args.num_slots) == (lay["base"], 4, 4, 2))
try:
    d._validate_cerf_core_sharing()
    _ok, _why = True, ""
except ValueError as e:
    _ok, _why = False, str(e)
check("a core may carry several tiles of one tile fork", _ok, _why)

def _later_tile_edge():
    d, f, by = _tiles(extra_edge=((0, 0), (0, 2)))
    compile_cond(d)
    return True
check("an edge from a tile to a later tile of its slot is allowed", _later_tile_edge())

def _earlier_tile_edge():
    d, f, by = _tiles(extra_edge=((0, 2), (0, 3)))
    d.bingo_add_edge(by[(0, 3)][1], by[(1, 0)][0])     # slot 0 -> slot 1: not a prefix
    compile_cond(d)
refuses("an edge from one slot's tile into another slot", _earlier_tile_edge, "unguarded")

def _dup_tile():
    d = new_dfg()
    disp = BingoNode(0, 0, 3, node_name="dispatch", kernel_name="__snax_k")
    d.bingo_add_node(disp)
    f = d.bingo_conditional_fork(disp, {"mode": "tiles", "rows": 8, "counts": 0})
    for _ in range(2):
        n = BingoNode(0, 1, 0, node_name=f"t{_}", kernel_name="__snax_k")
        d.bingo_add_node(n)
        f.branch([n], slot=0, tile=0)
refuses("a tile declared twice", _dup_tile, "twice")

def _tile_off_chip():
    d, f, by = _tiles(chip_of=lambda s, t: 0x01 if (s, t) == (1, 1) else 0x00)
    compile_cond(d)
refuses("a tile on another chiplet than its gate", _tile_off_chip, "own gate")

def _tile_branch_without_slot():
    d = new_dfg()
    disp = BingoNode(0, 0, 3, node_name="dispatch", kernel_name="__snax_k")
    d.bingo_add_node(disp)
    f = d.bingo_conditional_fork(disp, {"mode": "tiles", "rows": 8, "counts": 0})
    n = BingoNode(0, 1, 0, node_name="t", kernel_name="__snax_k")
    d.bingo_add_node(n)
    f.branch([n])
refuses("a tile branch without slot= and tile=", _tile_branch_without_slot, "slot=")


def _lower(d):
    """bingo_compile_dfg up to its hang checks, without emitting anything."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        d.bingo_transform_dfg_add_entry_node()
        d.bingo_transform_dfg_add_exit_nodes()
        d.bingo_compile_conditional_regions()
        d._validate_cerf_cross_group_edges()
        d._validate_cerf_core_sharing()
        d.bingo_transform_add_core_sequencing_edges()
        d.bingo_transform_prune_redundant_fanout()
        d.bingo_transform_keep_local_dep_set()
        d.bingo_transform_fit_dep_tag_budget(tag_width=d.dep_tag_width)
        d.bingo_transform_dfg_add_dummy_set_nodes()
        d.bingo_transform_dfg_add_dummy_check_nodes()
        d.bingo_assign_normal_node_dep_set_info()
        d.bingo_assign_normal_node_dep_check_info()
        d.bingo_transform_dfg_allocate_dep_tags(tag_width=d.dep_tag_width)
        return d.bingo_validate_no_hang(tag_width=d.dep_tag_width)

# The hang check refuses a remote task gated past the cross-die window.
d, far, near = _two_chip()
_lower(d)
far.cond_exec_group_id = 9
refuses("a remote gated task on group >= 8", lambda: d.bingo_validate_no_hang(),
        "cross a die")

# The doc's small case, on the cycle model: 2 chiplets, 1 slot each... here 2 slots of 4 tiles
# on shared cores, with the routing sweep (random prefixes, 0 and full included).
def _model_has_wide_cerf():
    try:
        import dataclasses
        from bingo_sim_check import _load_model
        _, SimConfig, *_ = _load_model()
        return "num_cerf_groups" in {f.name for f in dataclasses.fields(SimConfig)}
    except Exception:                       # noqa: BLE001
        return False

if _model_has_wide_cerf():
    from bingo_sim_check import simulate_for_hangs
    for one_core in (True, False):
        d, f, by = _tiles(slots=2, tiles=4, one_core=one_core)
        _lower(d)
        try:
            r = simulate_for_hangs(d, seeds=4, verbose=False)
            _ok, _why = "prefix0" in r["scenarios"], r["scenarios"]
        except ValueError as e:
            _ok, _why = False, str(e)[:300]
        check(f"tile fork survives the routing sweep ({'shared' if one_core else 'own'} "
              f"GEMM core)", _ok, _why)
else:
    print("  SKIP  tile routing sweep: the cycle model has no num_cerf_groups "
          "(set BINGO_MODEL_PATH to a bingo_hw_manager with it)")

if FAILED:
    print(f"\n{len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("\nall conditional-fork tests passed")
