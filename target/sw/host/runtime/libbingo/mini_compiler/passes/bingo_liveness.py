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
#    break it by forgetting an annotation. A task whose engine keeps reading after it returns
#    names the task that waits for the engine (`operands_held_by`: a queued GEMM's start, whose
#    array streams the operands until the wait); that task is a user of everything it names.
#    A task that only writes addresses into an engine's configuration and starts nothing
#    (`config_only`: a split dispatch's configure part, bingo_dfg_schedule.py) is no user: its
#    start task, which names the same buffers, is.
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
from bingo_reach import reach_sets


def collect_handle_users(nodes) -> Dict[int, Tuple[BingoMemAlloc, Set]]:
    """handle id -> (handle, {nodes that mention it}).

    Walks kernel_args: handles, views, and lists, tuples, sets and dicts of them. A use this
    walk misses is one the packer will not know about.
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
        if getattr(node, "kernel_args", None) is None or getattr(node, "config_only", False):
            continue
        held_by = getattr(node, "operands_held_by", None)
        for _attr, value in node.kernel_args.__dict__.items():
            visit(value, node)
            if held_by is not None:
                visit(value, held_by)
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


def with_spin_order(g, order_edges, successors):
    """`g` with what an ordering-only edge (BingoDFG.bingo_add_order_edge) orders: its consumer spins until the
    producer's flag lands, so everything after the consumer -- its successors, `successors(c)` -- comes after the
    producer, but the consumer itself, which runs while it spins, does not. A copy where there are such edges."""
    if not order_edges:
        return g
    h = nx.DiGraph(g)
    for p, c in order_edges:
        for s_ in list(successors(c)):
            if s_ is not p:
                h.add_edge(p, s_)
    return h


def reachability(dfg, nodes) -> Dict[object, Set]:
    """node -> every node reachable from it, as a set to test membership in.

    A conditional edge still orders its endpoints, so it counts; there is no edge type that
    does not. An ordering-only edge orders its producer before the consumer's successors
    (with_spin_order). One bitset a node (bingo_reach): a Python set of every node's
    descendants grows with the square of the graph.
    """
    order = list(getattr(dfg, "order_edges", None) or ())
    g = with_spin_order(dfg, order, dfg.successors)
    desc = reach_sets(g)            # a ReachMap: the slot packer also asks it for concurrent nodes
    for n in nodes:
        desc.setdefault(n, frozenset())
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


def _column_summary(users, desc):
    """(the latest of the users' earliest reach on each column, the earliest user on each column, the users that are
    leaves) with desc's chain cover (bingo_reach.reach_sets, column form), else None. Every user of A precedes every
    column user of B exactly when A's first vector is at most B's second everywhere: a reaches b on b's column when its
    label there is at most b's position, and a user in both fails it (a node does not reach itself). A leaf of B has no
    column: _reach_leaf tests it."""
    import numpy as np
    L, where, pos = getattr(desc, "L", None), getattr(desc, "where", None), getattr(desc, "pos", None)
    if L is None:
        return None
    K = L.shape[1]
    late = np.full(K, -1, dtype=np.int64)
    first = np.full(K, np.iinfo(np.int64).max, dtype=np.int64)
    leaves = []
    for u in users:
        w = where.get(u)
        if w is None:
            return None                         # not in the graph: the pairwise test
        np.maximum(late, L[pos[u]], out=late)   # a leaf has a label row too
        if w[0] < 0:
            leaves.append(u)
        elif w[1] < first[w[0]]:
            first[w[0]] = w[1]
    return late, first, leaves


def _reach_leaf(users, upos, b, desc, memo):
    """Whether every one of `users` (a list, upos their rows in desc.L) reaches the leaf b: bingo_reach._reaches, b's
    ancestry walked once, the column test over all the users at once."""
    import numpy as np
    from bingo_reach import _leaf_ancestry
    if b not in memo:
        leaves, cols = _leaf_ancestry(desc.G, desc.where, b)
        memo[b] = (leaves | set(cols), np.array([desc.where[p][0] for p in cols], dtype=np.int64),
                   np.array([desc.where[p][1] for p in cols], dtype=np.int64))
    above, cc, ci = memo[b]
    if len(cc):
        ok = (desc.L[upos[:, None], cc[None, :]] <= ci[None, :]).any(axis=1)
    else:
        ok = np.zeros(len(users), dtype=bool)
    return all(k or (a is not b and a in above) for a, k in zip(users, ok))


def _bitset_interference(idxs, items, desc, interference):
    """build_interference for the buffers `idxs` of one address space where desc is in the bitset form
    (bingo_reach.Reach): every user of i precedes every user of j exactly when each of j's users lies in what every
    one of i's users reaches (a user of both does not: no node reaches itself). Over the users of these buffers, as
    one matrix product: entry (j, i) counts j's users that not all of i's reach. Returns the buffers it could not
    decide (a user without a bitset reach set), for can_share."""
    import numpy as np
    ok, rest = [], []
    for i in idxs:
        us = items[i][1]
        if us and all(hasattr(desc.get(u), "bits") and u in desc[u].pos for u in us):
            ok.append(i)
        else:
            rest.append(i)
    if len(ok) < 2:
        return rest + ok
    pos = desc[next(iter(items[ok[0]][1]))].pos
    cols = sorted({pos[u] for i in ok for u in items[i][1]})
    col = np.array(cols, dtype=np.int64)
    at = {p: k for k, p in enumerate(cols)}
    nbytes = (len(pos) + 7) // 8
    reached = np.zeros((len(ok), len(cols)), dtype=np.float32)      # what every user of the buffer reaches
    users = np.zeros((len(ok), len(cols)), dtype=np.float32)        # the buffer's users
    for r, i in enumerate(ok):
        bits = -1
        for u in items[i][1]:
            bits &= desc[u].bits
            users[r, at[pos[u]]] = 1
        row = np.unpackbits(np.frombuffer(bits.to_bytes(nbytes, "little"), dtype=np.uint8),
                            bitorder="little")
        reached[r] = row[col]
    missed = users @ (1 - reached).T                                  # (j, i): j's users not all of i's users reach
    before = missed.T == 0                                            # (i, j): every user of i precedes every user of j
    share = before | before.T
    for r, i in enumerate(ok):
        for c in np.flatnonzero(~share[r]):
            j = ok[c]
            if j != i:
                interference[i].add(j)
    return rest


def build_interference(handle_users, desc, level: str = "L1"):
    """(buffers, interference) for one memory level.

    interference[i] is the set of buffer indices that may be live at the same time as i, and
    therefore must not overlap in space. Buffers whose users all sit on the reach labels' columns are compared a whole
    row at a time (_column_summary); the others pair by pair (can_share).
    """
    import numpy as np
    items = [(h, u) for (h, u) in handle_users.values() if h.mem_level == level]
    items.sort(key=lambda hu: (hu[0].chip_id, hu[0].cluster_id, hu[0].name))
    n = len(items)
    interference = [set() for _ in range(n)]
    spaces = {}
    for i, (h, _u) in enumerate(items):
        spaces.setdefault((h.chip_id, h.cluster_id), []).append(i)      # different address spaces cannot alias
    for idxs in spaces.values():
        summ = {i: _column_summary(items[i][1], desc) for i in idxs}
        fast = [i for i in idxs if summ[i] is not None]
        if fast:
            late = np.stack([summ[i][0] for i in fast])
            first = np.stack([summ[i][1] for i in fast])
            memo, seen, rows = {}, {}, {}

            def before(x, y):
                """every user of x precedes every user of y, given that holds for y's column users"""
                if x not in rows:
                    us = list(items[x][1])
                    rows[x] = (us, np.array([desc.pos[u] for u in us], dtype=np.int64))
                for b in summ[y][2]:
                    k = (x, b)
                    if k not in seen:
                        seen[k] = _reach_leaf(*rows[x], b, desc, memo)
                    if not seen[k]:
                        return False
                return True
            leafy = np.array([bool(summ[i][2]) for i in fast])
            for r, i in enumerate(fast):
                fwd = np.all(late[r] <= first, axis=1)          # i's users precede j's column users
                bwd = np.all(late <= first[r], axis=1)          # j's users precede i's column users
                sure = (fwd & ~leafy) | (bwd & (not leafy[r]))  # no leaf on the later side: exact
                for c in np.nonzero(~sure)[0]:
                    j = fast[c]
                    if j == i or j in interference[i]:
                        continue
                    if not ((fwd[c] and before(i, j)) or (bwd[c] and before(j, i))):
                        interference[i].add(j)
                        interference[j].add(i)
        slow = [i for i in idxs if summ[i] is None]
        slow = _bitset_interference(slow, items, desc, interference)
        for i in slow:
            for j in idxs:
                if j != i and j not in interference[i] and not can_share(items[i][1], items[j][1], desc):
                    interference[i].add(j)
                    interference[j].add(i)
    return items, interference


def liveness_report(handle_users, desc, level: str = "L1") -> str:
    """What packing would save, without changing any layout."""
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
        by_size = sorted(idxs, key=lambda i: -items[i][0].size)
        for seed in by_size:
            weight, common = items[seed][0].size, set(interference[seed])    # interfere with the whole clique
            for c in by_size:
                if c in common:
                    weight += items[c][0].size
                    common &= interference[c]
            lb = max(lb, weight)
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
