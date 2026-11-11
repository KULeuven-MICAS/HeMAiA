# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for a cluster's L1 heap as the compiler counts it (passes/bingo_l1_heap.py) and the budgets it sizes from it
# (passes/bingo_args_phases.py): the args structs' sizes against the C compiler's, the heap's fragments, and the args
# phases and static capacity of a cluster that fits and of one that does not.
import os
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402  (groups the compiler's subdirs onto sys.path)
import bingo_l1_heap as hp  # noqa: E402
from bingo_args_phases import _struct_sizes, auto_budgets, static_capacity  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------------------------------
print("the args structs' sizes, against sizeof")
LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "include", "libbingo")
PLAT = os.path.join(hp._SW, "shared", "platform", "generated", "occamy.h")
sizes = _struct_sizes()
cc = shutil.which("gcc") or shutil.which("cc")
if cc is None:
    print("  (no C compiler here: skipped)")
else:
    words = re.search(r"#define\s+BINGO_CERF_WORDS\s+(\d+)", open(PLAT).read()).group(1)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "probe.c")
        with open(src, "w") as f:
            f.write('#include <stdio.h>\n#include "device_kernel_args.h"\nint main(void) {\n')
            for name in sorted(sizes):
                f.write(f'    printf("{name} %zu\\n", sizeof({name}));\n')
            f.write("    return 0;\n}\n")
        exe = os.path.join(d, "probe")
        r = subprocess.run([cc, f"-DBINGO_CERF_WORDS={words}", "-I", LIB, src, "-o", exe], capture_output=True,
                           text=True)
        check("the probe compiles", r.returncode == 0, r.stderr[:300])
        if r.returncode == 0:
            real = dict(line.split() for line in subprocess.run([exe], capture_output=True, text=True).stdout.split("\n")
                        if line.strip())
            bad = [n for n in sizes if int(real[n]) != sizes[n]]
            check(f"every one of the {len(sizes)} structs", not bad, f"differ: {bad[:5]}")

# ---------------------------------------------------------------------------------------
print("the heap")
check("a fragment is the request and a header, in 256-byte steps",
      [hp.frag(0), hp.frag(1), hp.frag(128), hp.frag(129), hp.frag(1000)] == [256, 256, 256, 512, 1280])
check("1 MiB of TCDM and 4 Snitch cores leave 1,038,720 B", hp.heap_bytes(1 << 20) == 1038720, hp.heap_bytes(1 << 20))
check("gid lists cost a cluster its own tasks only",
      hp.list_allocs("gid", 100, 10 ** 6, 4) == [400, 400, 432], hp.list_allocs("gid", 100, 10 ** 6, 4))
check("a compact map costs every task of the graph", hp.list_allocs("compact", 100, 1000, 4)[2] == 2008)

# ---------------------------------------------------------------------------------------
print("args phases and the static capacity")
GEMM = "__snax_bingo_kernel_gemm_full"
T = next(n for n in sizes if n.startswith("__snax_bingo_kernel_gemm_full"))


def node(i, cl, kernel=GEMM, core=0):
    return SimpleNamespace(node_id=i, assigned_chiplet_id=0, assigned_cluster_id=cl, assigned_core_id=core,
                           kernel_name=kernel, kernel_args=SimpleNamespace(get_struct_name=lambda: T))


def dfg(nodes, peaks):
    return SimpleNamespace(node_list=nodes, num_cores_per_cluster=4, l1_tcdm_bytes=1 << 20, gid_lists=True,
                           compact_task_tables=True, list_phases=False, l1_capacity_bytes=990000,
                           l1_plan_hint=({}, {k: {"peak": p} for k, p in peaks.items()}))


few = [node(i, 0) for i in range(100)]
many = [node(1000 + i, 1) for i in range(4000)]
d = dfg(few + many, {(0, 0): 900000, (0, 1): 900000})
b = auto_budgets(d, {(0, 0): few, (0, 1): many}, {0, 1, 2, 3})
check("a cluster whose args fit beside its static plan gets no phases", b[(0, 0)] is None, b)
per = (sizes[T] + 63) // 64 * 64
check("one whose args do not gets phases that leave it room",
      b[(0, 1)] is not None and hp.MIN_PHASE <= b[(0, 1)] < 4000 * per // 2, b)
room = hp.heap_bytes(1 << 20) - hp.frag(900000)
check("its two regions fit what its static plan leaves", b[(0, 1)] is not None and 2 * b[(0, 1)] < room, (b, room))
cap = static_capacity(d, {(0, 0): 100, (0, 1): 4000}, {0, 1, 2, 3})
check("the static capacity leaves the big cluster its lists and two minimal phases",
      cap is not None and cap <= hp.heap_bytes(1 << 20) - 2 * hp.MIN_PHASE - hp.SLOTS_BOUND, cap)
check("and is no less than what the heap holds without them",
      cap is not None and cap >= hp.heap_bytes(1 << 20) - 128 * 1024, cap)

# ---------------------------------------------------------------------------------------
print("the emitter's check of the args phases' order")
import networkx as nx  # noqa: E402
from bingo_dfg_emit import BingoDFGEmitMixin  # noqa: E402


class _G(nx.DiGraph):
    args_pingpong = True
    _phase_labels = None


class _T:
    def __init__(self, name, phase=None, load=None):
        self.node_name, self.assigned_cluster_id = name, 0
        if load is not None:          # a phase load: resident, its args name the phase
            self.kernel_name, self.args_resident = "__snax_bingo_kernel_idma_1d_copy", True
            self.kernel_args = SimpleNamespace(phase=load, cluster=0)
        else:
            self.kernel_name, self.kernel_args, self.args_phase = GEMM, None, phase


def phase_order(edges, extra=()):
    g = _G()
    t0, ld1, t1, t1b = _T("t0", 0), _T("args_p1", load=1), _T("t1", 1), _T("t1b", 1)
    nodes = {"t0": t0, "ld1": ld1, "t1": t1, "t1b": t1b}
    g.add_nodes_from(nodes.values())
    g.add_edges_from((nodes[a], nodes[b]) for a, b in edges)
    try:
        BingoDFGEmitMixin._check_args_phase_order(g, list(nodes.values()))
        return None
    except ValueError as e:
        return str(e)


check("phase 1's tasks after its load pass", phase_order([("t0", "ld1"), ("ld1", "t1"), ("ld1", "t1b")]) is None)
err = phase_order([("t0", "ld1"), ("ld1", "t1")])
check("a phase-1 task with no path from the load is refused", err is not None and "t1b" in err, err)

# ---------------------------------------------------------------------------------------
print()
if FAILED:
    print(f"{len(FAILED)} FAILURE(S): {', '.join(FAILED)}")
    sys.exit(1)
print("all L1 heap tests passed")
