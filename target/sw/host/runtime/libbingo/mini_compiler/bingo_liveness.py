# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# LIVENESS OF L1 BUFFERS, over the task graph.
#
# The point of this file is to decide when two buffers may occupy the same bytes. Everything
# else about static allocation is bookkeeping; this is the part that can silently corrupt a
# run, so the rules it implements are deliberately more conservative than they have to be.
#
# ---------------------------------------------------------------------------------------
# RULE 1: EVERY REFERENCE IS A USE. We do not try to tell a read from a write.
#
# There are 233 kernel-argument classes and none of them declares direction. The field names
# are conventional but not uniform -- src_addr, dst_addr, input_addr, output_addr, a_addr,
# x_addr, weight_addr, perf_addr, cz -- and several are genuinely both (the FA PV accumulator
# is read and written through the same pointer). Inferring direction across 233 classes would
# put a silent-corruption bug behind every naming exception.
#
# So a buffer is LIVE from the first node that mentions it to the last node that mentions it.
# This costs some packing opportunity -- a buffer written late and read early cannot exist, so
# nothing is actually lost there; what is lost is the ability to reuse a buffer's bytes
# between its own uses, which is not something this compiler wants to do anyway.
#
# It also means the analysis needs no per-kernel knowledge at all, so a new kernel cannot
# break it by forgetting an annotation.
#
# ---------------------------------------------------------------------------------------
# RULE 2: ORDERING COMES FROM THE GRAPH, NOT FROM A SCHEDULE.
#
# BINGO has no linear time axis. The hardware manager fires a task as soon as its incoming
# edges are satisfied, so two nodes adjacent in a topological sort may run concurrently or in
# the opposite order. Level numbers (ASAP/ALAP) are therefore NOT a safe basis for deciding
# that two buffers are disjoint in time: two nodes at different levels need not be ordered
# with respect to each other at all.
#
# The only sound test is reachability. Buffers b1 and b2 may share bytes iff
#
#     every user of b1 is a transitive ancestor of every user of b2   (or vice versa)
#
# which is exactly the statement "in every possible execution, everything that touches b1 has
# finished before anything touches b2".
#
# ---------------------------------------------------------------------------------------
# RULE 3: LIVENESS IS GLOBAL, NOT PER-CLUSTER.
#
# A buffer allocated on cluster 0 can be read by a node on cluster 1. This is not theoretical:
# fa_decode_4cluster's cross-cluster V pull has clusters 1..3 issuing iDMA reads whose SOURCE
# is cluster 0's v8 buffer. A per-cluster analysis would see cluster 0's own last use, free the
# buffer there, pack something else on top, and corrupt three clusters with no diagnostic.
#
# So users are collected across the whole graph. Only the PLACEMENT is per-cluster, because a
# cluster's TCDM is a separate address space.

from typing import Dict, List, Set, Tuple

import networkx as nx

from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView


def collect_handle_users(nodes) -> Dict[int, Tuple[BingoMemAlloc, Set]]:
    """handle id -> (handle, {nodes that mention it}).

    Walks kernel_args exactly the way _collect_memory_handles does -- scalars, lists, tuples,
    dicts and views -- because a use this walk cannot see is a use the packer will not know
    about, and that is the one failure mode with no symptom until the data is wrong.
    """
    users: Dict[int, Set] = {}
    handles: Dict[int, BingoMemAlloc] = {}

    def visit(value, node):
        if isinstance(value, BingoMemAlloc):
            handles[id(value)] = value
            users.setdefault(id(value), set()).add(node)
        elif isinstance(value, BingoMemAllocView):
            # A view is not its own allocation: touching the view touches the base.
            visit(value.base, node)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, node)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, node)

    for node in nodes:
        if getattr(node, "kernel_args", None) is None:
            continue
        for _attr, value in node.kernel_args.__dict__.items():
            visit(value, node)
    return {h: (handles[h], users[h]) for h in handles}


def check_handle_identity(handle_users: Dict[int, Tuple[BingoMemAlloc, Set]]) -> List[str]:
    """Two DISTINCT objects that denote the same buffer would break identity-keyed liveness.

    The compiler keys everything on object identity, so a workload that built two
    BingoMemAlloc objects with the same (name, cluster) would get two independent live ranges
    for one buffer and could have them packed on top of each other. Nothing forbids that
    today, so it is checked rather than assumed.
    """
    seen: Dict[Tuple[int, int, str, str], int] = {}
    problems = []
    for hid, (h, _u) in handle_users.items():
        key = (h.chip_id, h.cluster_id, h.mem_level, h.name)
        if key in seen and seen[key] != hid:
            problems.append(
                f"two distinct handle objects denote the same buffer {key}: "
                f"packing would give them separate live ranges"
            )
        seen[key] = hid
    return problems


def reachability(dfg, nodes) -> Dict[object, Set]:
    """node -> every node reachable from it (its transitive successors).

    Uses only real edges. A conditional edge (cond=True) still ORDERS its endpoints -- the
    consumer cannot start before the producer has been evaluated, whether or not it then runs
    -- so it is kept. An edge that did not order execution would have to be excluded here, and
    there is currently no such edge type.
    """
    desc: Dict[object, Set] = {}
    # reverse topological order, so a node's successors are already resolved
    for n in reversed(list(nx.topological_sort(dfg))):
        s: Set = set()
        for _u, v in dfg.out_edges(n):
            s.add(v)
            s |= desc.get(v, set())
        desc[n] = s
    for n in nodes:
        desc.setdefault(n, set())
    return desc


def can_share(users_a: Set, users_b: Set, desc: Dict[object, Set]) -> bool:
    """True iff every user of A precedes every user of B, or the reverse.

    This is the whole safety argument. If it holds one way, then in EVERY execution the last
    touch of A happens before the first touch of B, so their bytes may be the same bytes.
    """
    def all_before(x: Set, y: Set) -> bool:
        for a in x:
            da = desc.get(a, ())
            for b in y:
                if a is b or b not in da:
                    return False
        return True

    return all_before(users_a, users_b) or all_before(users_b, users_a)


def build_interference(handle_users, desc, level: str = "L1"):
    """(buffers, interference) for one memory level.

    interference[i] is the set of buffer indices that may be live at the same time as i, and
    therefore must not overlap in space.
    """
    items = [(h, u) for (h, u) in handle_users.values() if h.mem_level == level]
    items.sort(key=lambda hu: (hu[0].chip_id, hu[0].cluster_id, hu[0].name))
    n = len(items)
    interference = [set() for _ in range(n)]
    for i in range(n):
        hi, ui = items[i]
        for j in range(i + 1, n):
            hj, uj = items[j]
            # different address spaces cannot alias
            if (hi.chip_id, hi.cluster_id) != (hj.chip_id, hj.cluster_id):
                continue
            if not can_share(ui, uj, desc):
                interference[i].add(j)
                interference[j].add(i)
    return items, interference


def liveness_report(handle_users, desc, level: str = "L1") -> str:
    """What packing WOULD save, without changing any layout. Stage 1 of the plan."""
    items, interference = build_interference(handle_users, desc, level)
    per_cluster: Dict[Tuple[int, int], List[int]] = {}
    for idx, (h, _u) in enumerate(items):
        per_cluster.setdefault((h.chip_id, h.cluster_id), []).append(idx)

    lines = [f"  {level} liveness report ({len(items)} buffers)"]
    for key in sorted(per_cluster):
        idxs = per_cluster[key]
        total = sum(items[i][0].size for i in idxs)
        # a lower bound on what any packer can achieve: the weight of the heaviest set of
        # MUTUALLY interfering buffers. Greedy clique -- summing a buffer's neighbours is not
        # a bound, because those neighbours need not interfere with one another.
        lb = 0
        for seed in sorted(idxs, key=lambda i: -items[i][0].size):
            clique = [seed]
            for c in sorted(idxs, key=lambda i: -items[i][0].size):
                if c != seed and all(c in interference[m] for m in clique):
                    clique.append(c)
            lb = max(lb, sum(items[i][0].size for i in clique))
        shareable = sum(1 for i in idxs if len(interference[i]) < len(idxs) - 1)
        lines.append(
            f"    chip {key[0]:#04x} cluster {key[1]}: {len(idxs):3d} buffers, "
            f"sum {total:,} B, interference-bound >= {min(lb, total):,} B, "
            f"{shareable} buffer(s) share with someone"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# RULE 4 (OPTIONAL, BINGO_L1_DRAIN_GUARD): A BUFFER AN ACCELERATOR WROTE MAY OUTLIVE ITS NODE.
#
# Rules 1-3 end a buffer's life at its last node. That is the same assumption the runtime makes
# when it retires a task with a bare `csrw 0x5ff` and no fence: it says the node is done, not
# that every byte the node's engine will ever write has landed.
#
# For a core doing loads and stores that is nearly the same statement. For an accelerator it is
# not. VersaCore buffers its D output and is known to push and pop it against a separately
# managed C buffer, so the last write of a GEMM node can drain when the array is NEXT
# CONFIGURED rather than when the node retires -- which, on this graph, is thousands of cycles
# later and after a DMA has already refilled those bytes.
#
# The guard is to treat the next node on the same core as a user too: a buffer stays live until
# the engine that wrote it has been handed its next configuration. That is conservative (it can
# only lengthen a live range, never shorten one) and it costs almost nothing on FA, where the
# ranges are long already.

def extend_users_for_engine_drain(handle_users, dfg):
    """Add each user's same-core successors to the user set.

    Returns a new {id: (handle, users)} mapping; the input is not modified.
    """
    def same_core(a, b):
        return (getattr(a, "assigned_chiplet_id", None) == getattr(b, "assigned_chiplet_id", None)
                and getattr(a, "assigned_cluster_id", None) == getattr(b, "assigned_cluster_id", None)
                and getattr(a, "assigned_core_id", None) == getattr(b, "assigned_core_id", None))

    succ_same_core = {}
    for n in dfg.nodes():
        succ_same_core[n] = {v for _u, v in dfg.out_edges(n) if same_core(n, v)}

    out = {}
    for hid, (h, users) in handle_users.items():
        extended = set(users)
        for u in users:
            extended |= succ_same_core.get(u, set())
        out[hid] = (h, extended)
    return out
