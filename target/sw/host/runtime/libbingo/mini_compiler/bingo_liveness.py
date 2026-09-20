# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Liveness of L1 buffers over the task graph: when may two buffers occupy the same bytes.
#
# Three rules, all deliberately conservative -- this is the part of static allocation that can
# silently corrupt a run.
#
# 1. EVERY REFERENCE IS A USE. No kernel-argument class declares read/write direction and the
#    field names are not uniform, so a buffer is live from the first node that mentions it to
#    the last. Costs a little packing; needs no per-kernel knowledge, so a new kernel cannot
#    break it by forgetting an annotation.
#
# 2. ORDERING COMES FROM THE GRAPH, NOT A SCHEDULE. BINGO has no linear time axis -- the
#    manager fires a task as soon as its edges are satisfied -- so ASAP/ALAP levels do not
#    imply ordering. The sound test is reachability: b1 and b2 may share iff every user of one
#    is a transitive ancestor of every user of the other.
#
# 3. LIVENESS IS GLOBAL, PLACEMENT IS PER-CLUSTER. A buffer on one cluster can be read from
#    another (fa_decode_4cluster's cross-cluster V pull does). Collecting users per cluster
#    would free a buffer at its local last use and corrupt the remote readers.

from typing import Dict, List, Set, Tuple

import networkx as nx

from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView


def collect_handle_users(nodes) -> Dict[int, Tuple[BingoMemAlloc, Set]]:
    """handle id -> (handle, {nodes that mention it}).

    Walks kernel_args the way _collect_memory_handles does: scalars, lists, tuples, dicts and
    views. A use this walk misses is one the packer will not know about.
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
    """Two distinct objects denoting one buffer would get two independent live ranges.

    Everything is keyed on object identity, so nothing otherwise forbids packing them on top
    of each other. Checked rather than assumed.
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
    """node -> every node reachable from it.

    A conditional edge still orders its endpoints, so it counts; there is no edge type that
    does not.
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

    The whole safety argument: if it holds, then in every execution the last touch of A
    happens before the first touch of B.
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
        # lower bound for any packer: the heaviest MUTUALLY interfering set. Greedy clique --
        # summing one buffer's neighbours is not a bound, they need not interfere each other.
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
# RULE 4 (optional, StaticL1Options.drain_guard): a buffer an accelerator wrote may outlive
# its node. Rules 1-3 end a buffer's life at its last node, which is the same assumption the
# runtime makes when it retires a task with a fenceless `csrw 0x5ff`. VersaCore buffers its D
# output, so a GEMM's last write can drain when the array is next configured. The guard counts
# the next node on the same core as a user; it can only lengthen a live range.

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
