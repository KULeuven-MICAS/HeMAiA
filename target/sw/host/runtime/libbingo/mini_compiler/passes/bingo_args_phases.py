# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Args phases cut along each cluster's program, with two alternating L1 regions (ping-pong).

A device task's args struct lives in its cluster's L1 (the compiler's args arena), so a long block (a 1,024-token
prompt's sixteen passes or eight chunks) holds thousands of them beside its static buffers. split_args_phases cuts
each cluster's device tasks, in the order they run, into phases of at most `budget` bytes of args; the compiler
(bingo_dfg_emit.py, ARGS PHASES with the DFG's args_pingpong) gives the even phases one region and the odd phases
another, phase 0's arrive at host init, and each later phase's by a DM task of the cluster (PhaseArgsLoadArgs), here
put in the DM core's program right before the phase's first DM task. Its dependencies: the last task of phase p - 2 on
every core of the cluster (the region it overwrites), unless the DM core's own program already orders it; every core's
first task of phase p depends on it, unless the DM core's program already does. It cuts along the programs, so it runs
once each core's program is fixed (the per-core chains are edges of the graph): bingo_compile_dfg runs it after the
schedule when the DFG's args_phases is set (ChipletBuild.finish sets it from params args_phase_bytes).
"""
import os
import re

import networkx as nx

from bingo_kernel_args import PhaseArgsLoadArgs, PhaseListLoadArgs
from bingo_reach import ChainLabels

_SIZES = None


def _struct_sizes():
    """{struct type name: bytes} of device_kernel_args.h's packed structs (the arena slices each to 64 bytes)."""
    global _SIZES
    if _SIZES is None:
        import bingo_kernel_args
        root = os.path.dirname(os.path.abspath(bingo_kernel_args.__file__))
        h = open(os.path.join(root, "..", "..", "..", "include", "libbingo", "device_kernel_args.h")).read()
        h = re.sub(r"//[^\n]*", "", h)
        # array sizes: the header's own macros, then the platform's (BINGO_CERF_WORDS)
        macros = {}
        plat = os.path.join(root, "..", "..", "..", "..", "..", "..", "shared", "platform", "generated", "occamy.h")
        for text in ([open(plat).read()] if os.path.exists(plat) else []) + [h]:
            macros.update({m: int(v, 0) for m, v in re.findall(r"#define\s+(\w+)\s+(0x[0-9a-fA-F]+|\d+)u?\b", text)})
        width = {"uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2, "uint32_t": 4, "int32_t": 4, "float": 4,
                 "uint64_t": 8, "int64_t": 8, "double": 8}
        _SIZES = {}
        for body, name in re.findall(r"__SNAX_KERNEL_ARGS_DEFINE\s+\w+\s*\{(.*?)\}\s*(\w+)\s*;", h, re.S):
            body = body.replace("BINGO_KERNEL_ARGS_TRAILER", "uint32_t a; uint32_t b; uint32_t c")
            n = 0
            for stmt in body.split(";"):
                m = re.match(r"\s*(\w+)\s+(.*)$", stmt, re.S)
                if not m or m.group(1) not in width:
                    continue
                for decl in m.group(2).split(","):        # `uint32_t a, b, c;` declares three
                    d = re.match(r"\s*\w+\s*(?:\[\s*(\w+)\s*\])?\s*$", decl)
                    if d is None:
                        raise ValueError(f"device_kernel_args.h {name}: cannot size the field `{decl.strip()}`")
                    dim = d.group(1)
                    if dim and not dim[0].isdigit() and dim not in macros:
                        raise ValueError(f"device_kernel_args.h {name}: no value for the array size {dim}")
                    n += width[m.group(1)] * (int(dim, 0) if dim and dim[0].isdigit() else macros[dim] if dim else 1)
            _SIZES[name] = (n + 3) // 4 * 4
    return _SIZES


def _args_bytes(t):
    """A task's args struct in the arena: its size 64-byte aligned (128 when its type is not found)."""
    ka = t.kernel_args
    name = ka.get_struct_name() if ka is not None else "__snax_bingo_kernel_exit_args_t"
    n = _struct_sizes().get(name)
    return 128 if n is None else (n + 63) // 64 * 64


def split_args_phases(S, budget, dev_cores):
    """The same cut on a generator S (dfg, roles, node()) whose generator ordered the cores itself."""
    dm = S.roles["dm"]
    return bingo_args_phases(S.dfg, budget, dev_cores, dm,
                             lambda k, c, name, kernel, args, deps: S.node(k, c, dm, name, kernel, args, deps))


from bingo_l1_heap import LIST_SLACK, MIN_PHASE, SLOTS_BOUND


def _list_mode(dfg):
    return ("paged" if getattr(dfg, "list_phases", False) else "gid" if getattr(dfg, "gid_lists", False)
            else "compact" if getattr(dfg, "compact_task_tables", False) else "full")


def _on_cluster(dfg, dev):
    on = {}
    for t in dfg.node_list:
        if t.assigned_core_id in dev:
            on.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id), []).append(t)
    return on


def _phase_load_bytes():
    return (_struct_sizes()[PhaseArgsLoadArgs(0, 0, 1).get_struct_name()] + 63) // 64 * 64


def static_capacity(dfg, counts, dev):
    """The static L1 plan's capacity: the least over the clusters of what the L1 heap (bingo_l1_heap.py) leaves beside
    the device's scratch pools, the scheduler's lists (counts: device tasks a cluster), the scratchpad slots and the
    args -- every args struct where they are few, else two phase regions of MIN_PHASE bytes and the resident args."""
    import bingo_l1_heap as hp
    from bingo_l1_heap import frag
    heap = hp.heap_bytes(getattr(dfg, "l1_tcdm_bytes", 1 << 20))
    mode, k = _list_mode(dfg), dfg.num_cores_per_cluster
    total = int(max(len(dfg.node_list), sum(counts.values())) * LIST_SLACK)
    load, on = _phase_load_bytes(), _on_cluster(dfg, dev)
    caps = []
    for key, cnt in counts.items():
        n, ts = int(cnt * LIST_SLACK), on.get(key, ())
        args = sum(_args_bytes(t) for t in ts if (t.kernel_name or "").startswith("__snax")
                   and not getattr(t, "args_resident", False)) + k * 128
        nph = args // MIN_PHASE + 1
        phased = 2 * MIN_PHASE + k * 128 + nph * load * (2 if mode == "paged" else 1)
        if mode == "paged":
            lists = (frag(8 * n) if args <= phased else 2 * frag(MIN_PHASE // 8)) \
                + frag(4 * (16 + (k + 1) * (nph + 1)) + 8 * (k + 2 * nph))
        else:
            lists = sum(frag(b) for b in hp.list_allocs(mode, n, total, k))
        pools = sum(frag(b) for b in hp.device_pools(ts).values())
        caps.append(heap - pools - lists - SLOTS_BOUND - frag(min(args, phased)) - 4 * frag(0))
    return min(caps) // 256 * 256 if caps else None


def auto_budgets(dfg, per_cl, dev):
    """{(chip, cluster): bytes of args a phase, or None where every args struct fits in L1 at once}: what is left of
    the cluster's L1 heap (bingo_l1_heap.py) beside the static plan the schedule fitted (l1_plan_hint), the device's
    scratch pools, the scheduler's lists and the scratchpad slots, for the args arena -- two phase regions and the
    resident args (the phase loads)."""
    import bingo_l1_heap as hp
    from bingo_l1_heap import frag
    heap = hp.heap_bytes(getattr(dfg, "l1_tcdm_bytes", 1 << 20))
    hint = getattr(dfg, "l1_plan_hint", None)
    peaks = {k: st["peak"] for k, st in hint[1].items()} if hint else {}
    cap = getattr(dfg, "l1_capacity_bytes", None) or 0
    mode, k = _list_mode(dfg), dfg.num_cores_per_cluster
    total = int(len(dfg.node_list) * LIST_SLACK)
    load, on = _phase_load_bytes(), _on_cluster(dfg, dev)
    out = {}
    for key, ts in per_cl.items():
        n = int(len(on.get(key, ())) * LIST_SLACK)
        fixed = (frag(peaks.get(key, cap)) + sum(frag(b) for b in hp.device_pools(on.get(key, ())).values())
                 + (0 if mode == "paged" else sum(frag(b) for b in hp.list_allocs(mode, n, total, k))))
        room = heap - fixed - SLOTS_BOUND - 4 * frag(0)
        need = sum(_args_bytes(t) for t in ts) + k * 128            # every args struct, the exit tasks
        # paged lists in one phase: every entry in window 0 (two words a task) and the header
        if need + (frag(8 * n) + frag(4 * (16 + 2 * (k + 1))) if mode == "paged" else 0) <= room:
            out[key] = None
            continue
        b = room // 2
        for _ in range(4):                  # the resident args and the paged lists follow from the phase count
            nph = need // max(b, 1) + 1
            resident = k * 128 + nph * load * (2 if mode == "paged" else 1)
            # paged: the header (the core bases of every phase, the resident entries)
            paged = frag(4 * (16 + (k + 1) * (nph + 1)) + 8 * (k + 2 * nph)) if mode == "paged" else 0
            b = int((room - resident - paged) / (2.25 if mode == "paged" else 2))     # paged: two list windows of b/8
        if b < MIN_PHASE:
            raise ValueError(f"args phases: chip 0x{key[0]:02x} cluster {key[1]} has {room:,} B of L1 for its args "
                             f"arena beside its static plan ({peaks.get(key, cap):,} B): too little for phases")
        out[key] = b
    return out


def bingo_args_phases(dfg, budget, dev_cores, dm, node):
    """budget: bytes of args a phase and cluster, or None: each cluster's own (auto_budgets), no phases where the
    args fit; dev_cores: the device core ids; dm: the DM core's id; node(chip, cluster, name, kernel, args, deps):
    makes a task on that cluster's DM core. Returns the number of phase loads made."""
    g = nx.DiGraph(dfg)
    g.add_edges_from(dfg.order_edges)
    order = nx.lexicographical_topological_sort(g, key=lambda t: t.node_id)
    pos = {t: i for i, t in enumerate(order)}
    dev = set(dev_cores)
    per_cl = {}
    for t in dfg.node_list:
        if t.assigned_core_id in dev and (t.kernel_name or "").startswith("__snax") \
                and not getattr(t, "args_resident", False):
            per_cl.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id), []).append(t)
    if budget is None:
        budgets = auto_budgets(dfg, per_cl, dev)
        if all(b is None for b in budgets.values()):
            return 0
        for key, ts in per_cl.items():
            if budgets[key] is None:
                for t in ts:
                    t.args_phase = 0            # every args struct in the one region (bingo_dfg_emit._args_group)
        per_cl = {key: ts for key, ts in per_cl.items() if budgets[key] is not None}
    else:
        budgets = {key: budget for key in per_cl}
    labels = ChainLabels(g)
    dfg.args_pingpong = True
    lists = bool(getattr(dfg, "list_phases", False))
    n_loads, n_edges, worst, n_lists = 0, 0, 0, 0
    for (k, c), ts in sorted(per_cl.items(), key=lambda kv: kv[0]):
        ts.sort(key=lambda t: pos[t])
        # the compiler's exit tasks are resident; every other task takes the phase of where it runs
        phase, acc, sizes = 0, 0, [0]
        for t in ts:
            b = _args_bytes(t)
            if acc + b > budgets[(k, c)] and acc:
                phase, acc = phase + 1, 0
                sizes.append(0)
            t.args_phase = phase
            acc += b
            sizes[-1] += b
        worst = max(worst, max(sizes[0::2]) + (max(sizes[1::2]) if len(sizes) > 1 else 0))
        cores = {}
        for t in ts:
            cores.setdefault(t.assigned_core_id, []).append(t)
        dms = [t for t in dfg.node_list if t.assigned_chiplet_id == k and t.assigned_cluster_id == c
               and t.assigned_core_id == dm]
        dms.sort(key=lambda t: pos[t])
        for p in range(1, phase + 1):
            first_dm = next((t for t in dms if getattr(t, "args_phase", -1) >= p
                             and not getattr(t, "args_resident", False)), None)
            prev_dm = None
            if first_dm is not None:
                i = dms.index(first_dm)
                prev_dm = dms[i - 1] if i else None
            deps = []
            for core, seq in cores.items():
                lastp = [t for t in seq if t.args_phase <= p - 2]
                # a phase load made here (the DM core has no task of the phase before) is not in the labels' graph
                if lastp and not (prev_dm is not None and (lastp[-1] is prev_dm or (prev_dm in g and
                                                                                  labels.reaches(lastp[-1], prev_dm)))):
                    deps.append(lastp[-1])
            if prev_dm is not None:
                deps.append(prev_dm)
            deps = sorted(set(deps), key=lambda t: t.node_id)
            if lists:
                # the DFG's list_phases: the phase's list entries come first, into the window of its parity, which
                # the same tasks as the args region free; the args load then depends on it alone
                lld = node(k, c, f"lists_p{p}", "__snax_bingo_kernel_idma_1d_copy", PhaseListLoadArgs(k, c, p), deps)
                lld.args_resident = True
                n_lists += 1
                deps = [lld]
            ld = node(k, c, f"args_p{p}", "__snax_bingo_kernel_idma_1d_copy", PhaseArgsLoadArgs(k, c, p), deps)
            ld.args_resident = True
            n_loads += 1
            n_edges += len(set(deps))
            if first_dm is not None:
                dfg.bingo_add_edge(ld, first_dm)
                n_edges += 1
            for core, seq in cores.items():
                f = next((t for t in seq if t.args_phase >= p), None)
                if f is None or f is first_dm:
                    continue
                if first_dm is not None and labels.reaches(first_dm, f):
                    continue
                dfg.bingo_add_edge(ld, f)
                n_edges += 1
            if first_dm is not None:
                dms.insert(dms.index(first_dm), ld)
    bs = sorted(budgets[key] for key in per_cl)
    print(f"[args phases] {n_loads} phase loads ({n_edges} edges) on {len(per_cl)} cluster(s), "
          + (f"budget {bs[0]:,} B a phase" if bs[0] == bs[-1] else f"budgets {bs[0]:,}-{bs[-1]:,} B a phase")
          + f"{'' if budget is not None else ' (each cluster what its L1 leaves)'}; the largest two regions of a "
          f"cluster {worst} B")
    if lists:
        print(f"[args phases] list_phases: {n_lists} list loads, one before each phase load")
    return n_loads
