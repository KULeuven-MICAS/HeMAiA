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
first task of phase p depends on it, unless the DM core's program already does. Call after the generator's program
order (the per-core chains are edges of the graph by then) and before the compile.
"""
import os
import re

import networkx as nx

from bingo_kernel_args import PhaseArgsLoadArgs
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
        macros = {m: int(v, 0) for m, v in re.findall(r"#define\s+(\w+)\s+(0x[0-9a-fA-F]+|\d+)u?\b", h)}
        width = {"uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2, "uint32_t": 4, "int32_t": 4, "float": 4,
                 "uint64_t": 8, "int64_t": 8, "double": 8}
        _SIZES = {}
        for body, name in re.findall(r"__SNAX_KERNEL_ARGS_DEFINE\s+\w+\s*\{(.*?)\}\s*(\w+)\s*;", h, re.S):
            body = body.replace("BINGO_KERNEL_ARGS_TRAILER", "uint32_t a; uint32_t b; uint32_t c")
            n = 0
            for ty, _, dim in re.findall(r"(\w+)\s+(\w+)\s*(?:\[\s*(\w+)\s*\])?\s*(?:[,;]|$)", body):
                if ty in width:
                    n += width[ty] * (int(dim, 0) if dim and dim[0].isdigit() else macros.get(dim, 1) if dim else 1)
            _SIZES[name] = (n + 3) // 4 * 4
    return _SIZES


def _args_bytes(t):
    """A task's args struct in the arena: its size 64-byte aligned (128 when its type is not found)."""
    ka = t.kernel_args
    name = ka.get_struct_name() if ka is not None else "__snax_bingo_kernel_exit_args_t"
    n = _struct_sizes().get(name)
    return 128 if n is None else (n + 63) // 64 * 64


def split_args_phases(S, budget, dev_cores):
    """S: the generator (dfg, roles, node()); budget: bytes of args a phase and cluster (estimate); dev_cores: the
    device core ids. Returns the number of phase loads made."""
    dfg = S.dfg
    dm = S.roles["dm"]
    g = nx.DiGraph(dfg)
    g.add_edges_from(dfg.order_edges)
    order = nx.lexicographical_topological_sort(g, key=lambda t: (getattr(t, "order_key", t.node_id), t.node_id))
    pos = {t: i for i, t in enumerate(order)}
    dev = set(dev_cores)
    per_cl = {}
    for t in dfg.node_list:
        if t.assigned_core_id in dev and (t.kernel_name or "").startswith("__snax") \
                and not getattr(t, "args_resident", False):
            per_cl.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id), []).append(t)
    labels = ChainLabels(g)
    dfg.args_pingpong = True
    n_loads, n_edges, worst = 0, 0, 0
    for (k, c), ts in sorted(per_cl.items(), key=lambda kv: kv[0]):
        ts.sort(key=lambda t: pos[t])
        # the compiler's exit tasks are resident; every other task takes the phase of where it runs
        phase, acc, sizes = 0, 0, [0]
        for t in ts:
            b = _args_bytes(t)
            if acc + b > budget and acc:
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
                if lastp and not (prev_dm is not None and (lastp[-1] is prev_dm or labels.reaches(lastp[-1], prev_dm))):
                    deps.append(lastp[-1])
            if prev_dm is not None:
                deps.append(prev_dm)
            ld = S.node(k, c, dm, f"args_p{p}", "__snax_bingo_kernel_idma_1d_copy", PhaseArgsLoadArgs(k, c, p),
                        sorted(set(deps), key=lambda t: t.node_id))
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
    print(f"[args phases] {n_loads} phase loads ({n_edges} edges), budget {budget} B a phase; the largest two "
          f"regions of a cluster {worst} B (estimate)")
    return n_loads
