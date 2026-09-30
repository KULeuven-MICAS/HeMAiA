#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the dependency-tag budget (bingo_transform_fit_dep_tag_budget).
#
# Synthetic graphs, because the workloads fit the hardware's 16 tags a cell on their own
# and so never exercise the pass. What is worth testing is that a graph needing more tags
# than the width gives LOWERS AND TAGS at that width once the pass has run -- the allocator
# and the hang check are the judges -- that the pass leaves a graph that already fits
# alone, and that a cell no ordering can fit is refused with the reason, not tagged wrong.
#
#   python3 test_tag_budget.py

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402  (groups the compiler's subdirs onto sys.path)
from bingo_dfg import BingoDFG  # noqa: E402
from bingo_node import BingoNode  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


def refuses(name, fn, needle=""):
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


def new_dfg():
    return BingoDFG(num_chiplets=1, num_clusters_per_chiplet=4, num_cores_per_cluster=5,
                    is_host_as_acc=True, chiplet_ids=[0x00], dep_tag_width=5)


def gather(dfg, per_cluster=4):
    """Four clusters' core-1 tasks each hand one result to its own task on cluster 0's
    core 3. The producers run ahead of the consumers and of each other, so every one of
    the 16 edges can be live at once: one (cluster 0, core 3, core 1) cell, 16 tags."""
    for i in range(per_cluster):
        for cl in range(4):
            p = BingoNode(0, cl, 1, node_name=f"p{cl}_{i}", kernel_name="__snax_k")
            c = BingoNode(0, 0, 3, node_name=f"c{cl}_{i}", kernel_name="__snax_k")
            dfg.bingo_add_node(p)
            dfg.bingo_add_node(c)
            dfg.bingo_add_edge(p, c)


def lower(dfg, tag_width):
    """bingo_compile_dfg's transforms and tag allocation, without the files it writes."""
    dfg.bingo_transform_dfg_add_entry_node()
    dfg.bingo_transform_dfg_add_exit_nodes()
    dfg.bingo_compile_conditional_regions()
    dfg.bingo_transform_add_core_sequencing_edges()
    dfg.bingo_transform_prune_redundant_fanout()
    added = dfg.bingo_transform_fit_dep_tag_budget(tag_width=tag_width)
    dfg.bingo_transform_dfg_add_dummy_set_nodes()
    dfg.bingo_transform_dfg_add_dummy_check_nodes()
    dfg.bingo_assign_normal_node_dep_set_info()
    dfg.bingo_assign_normal_node_dep_check_info()
    dfg.bingo_transform_dfg_allocate_dep_tags(tag_width=tag_width)
    hang = dfg.bingo_validate_no_hang(tag_width=tag_width)
    return added, hang


def quiet(fn, *a, **k):
    """The passes print a line per node; keep the test output to the verdicts."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


print("a 16-edge gather into one cell")
g = new_dfg()
gather(g)
refuses("without the budget, 4 tags are refused by the allocator",
        lambda: quiet(lambda: (
            g.bingo_transform_dfg_add_entry_node(), g.bingo_transform_dfg_add_exit_nodes(),
            g.bingo_compile_conditional_regions(), g.bingo_transform_add_core_sequencing_edges(),
            g.bingo_transform_dfg_add_dummy_set_nodes(), g.bingo_transform_dfg_add_dummy_check_nodes(),
            g.bingo_assign_normal_node_dep_set_info(), g.bingo_assign_normal_node_dep_check_info(),
            g.bingo_transform_dfg_allocate_dep_tags(tag_width=2))),
        "needs")

g = new_dfg()
gather(g)
added, hang = quiet(lower, g, 2)
check("with the budget, it lowers and tags at 4 tags a cell", hang["peak_tags_per_cell"] <= 4,
      f"peak {hang['peak_tags_per_cell']}")
check("...by adding ordering edges", added > 0, f"added {added}")
order_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("order_only")]
check("...each marked order_only", len(order_edges) == added,
      f"{len(order_edges)} marked of {added}")
check("...each from a cluster-0 consumer to a producer",
      all(u.assigned_cluster_id == 0 and u.assigned_core_id == 3 and v.assigned_core_id == 1
          for u, v in order_edges), str([(u.node_name, v.node_name) for u, v in order_edges]))

g = new_dfg()
gather(g)
added, hang = quiet(lower, g, 5)
check("at 32 tags the same graph is left alone", added == 0, f"added {added}")

print("a cell no ordering can fit")
# Two producers on clusters 1 and 2 both feed ONE consumer, so both edges are live until
# it checks, and each producer precedes it: neither edge can wait for the other.
g = new_dfg()
a = BingoNode(0, 1, 1, node_name="a", kernel_name="__snax_k")
b = BingoNode(0, 2, 1, node_name="b", kernel_name="__snax_k")
q = BingoNode(0, 0, 3, node_name="q", kernel_name="__snax_k")
for n in (a, b, q):
    g.bingo_add_node(n)
g.bingo_add_edge(a, q)
g.bingo_add_edge(b, q)
quiet(g.bingo_transform_add_core_sequencing_edges)
refuses("one tag for a two-producer fan-in is refused, with the cell",
        lambda: quiet(g.bingo_transform_fit_dep_tag_budget, tag_width=0), "cannot be ordered")

if FAILED:
    print(f"\n{len(FAILED)} tag-budget test(s) FAILED: {FAILED}")
    _sys.exit(1)
print("\nall tag-budget tests passed")
