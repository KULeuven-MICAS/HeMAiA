# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The compiler's order of the collectives (passes/bingo_dfg_collectives.py, libs/blocks/collective.py THE RULE): two
broadcasts ordered, unless one push-engine queue runs both; two gathers into overlapping bytes ordered; a broadcast
beside a gather, and gathers into disjoint bytes, left alone; a receiver's wait on a broadcast's data counts as it done;
an edge on one core is an ordering-only edge, across cores a dependency. Run: python3 tests/test_collectives.py."""

import os as _os
import sys as _sys

import networkx as nx

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402
from bingo_dfg import BingoDFG                                        # noqa: E402
from bingo_kernel_args import (SnaxBingoKernelIdmaPutFlaggedArgs,     # noqa: E402
                               SnaxBingoKernelXdmaChainGatherArgs)
from bingo_node import BingoNode                                      # noqa: E402
from libs.blocks.collective import mark_broadcast                     # noqa: E402

FAILED = []
GEMM, SIMD, XDMA, DM, HOST = 0, 1, 2, 3, 4
MASK40 = "0xFFFFFFFFFFULL"           # fabric.MASK40, as the drivers write it


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        FAILED.append(name)


def new_dfg():
    d = BingoDFG(num_chiplets=2, num_clusters_per_chiplet=2, num_cores_per_cluster=4, is_host_as_acc=True,
                 chiplet_ids=[0x00, 0x01], dep_tag_width=5)
    d.core_roles = {"gemm": GEMM, "simd": SIMD, "xdma": XDMA, "dm": DM, "host": HOST}
    return d


def task(d, core, name, kernel, args=None, after=(), cluster=0, chip=0):
    n = BingoNode(chip, cluster, core, node_name=name, kernel_name=kernel, kernel_args=args)
    d.bingo_add_node(n)
    for a in after:
        d.bingo_add_edge(a, n)
    return n


def bput(d, name, after=(), cluster=0):
    return task(d, DM, name, "__snax_bingo_kernel_idma_put_flagged",
                SnaxBingoKernelIdmaPutFlaggedArgs(0x100, 0x200, 64, 0x300, value=1, bcast=True, also_here=True),
                after, cluster)


def remote(chip, local):
    return f"(chiplet_addr_transform_full(0x{chip:02x}, ((uint64_t)({local}) & {MASK40})))"


def gather(d, name, dst, off, after=(), cluster=0):
    ga = SnaxBingoKernelXdmaChainGatherArgs(remote(0, "xch"), [remote(1, "xch"), remote(0, f"((uint64_t)({dst}) + {off})")],
                                            256, "0", 0, hops=[(1, cluster), (0, cluster)])
    return task(d, XDMA, name, "__snax_bingo_kernel_xdma_chain_gather", ga, after, cluster)


def stamp(d, name, after=(), core=DM, cluster=0):
    return task(d, core, name, "__snax_bingo_kernel_stamp", None, after, cluster)


def path(d, a, b):
    g = d.bingo_order_view()
    return nx.has_path(g, a, b)


print("\ntwo broadcasts on two cores, unordered: a dependency from the first to the second")
d = new_dfg()
s = stamp(d, "s")
p0 = bput(d, "p0", [s], cluster=0)
p1 = bput(d, "p1", [s], cluster=1)
added = d.bingo_order_collectives()
check("one edge added", len(added) == 1, added)
check("p0 now precedes p1 by a dependency", d.has_edge(p0, p1))

print("\ntwo broadcasts on one core: an ordering-only edge (the core's queue runs them in turn)")
d = new_dfg()
s = stamp(d, "s")
p0 = bput(d, "p0", [s])
p1 = bput(d, "p1", [s])
d.bingo_order_collectives()
check("no dependency edge", not d.has_edge(p0, p1) and not d.has_edge(p1, p0))
check("an ordering-only edge p0 -> p1", (p0, p1) in d.order_edges)

print("\na broadcast beside a gather: left alone")
d = new_dfg()
s = stamp(d, "s")
p0 = bput(d, "p0", [s])
g0 = gather(d, "g0", "ybuf", 0, [s])
check("no edge added", d.bingo_order_collectives() == [])

print("\ntwo gathers into overlapping bytes: ordered; into disjoint bytes: left alone")
d = new_dfg()
s = stamp(d, "s")
g0 = gather(d, "g0", "ybuf", 0, [s], cluster=0)
g1 = gather(d, "g1", "ybuf", 128, [s], cluster=1)
g2 = gather(d, "g2", "ybuf", 4096, [s], cluster=1)
added = d.bingo_order_collectives()
check("one edge, g0 -> g1 (bytes 128..255 shared)", [(a.node_name, b.node_name) for a, b in added] == [("g0", "g1")],
      [(a.node_name, b.node_name) for a, b in added])
check("g2 not ordered against g0 (disjoint)", not path(d, g0, g2) and not path(d, g2, g0))

print("\na receiver's wait on a broadcast's data counts as the broadcast done")
d = new_dfg()
s = stamp(d, "s")
p0 = bput(d, "p0", [s], cluster=0)
w = task(d, DM, "w", "__snax_bingo_kernel_wait_flags", None, [], cluster=0, chip=1)
d.bingo_add_order_edge(p0, w)
nxt = stamp(d, "after_w", [w], cluster=0)
p1 = bput(d, "p1", [nxt], cluster=1)
check("no edge added", d.bingo_order_collectives() == [])

print("\ntwo HBM casts that one push-engine queue runs: never overlap, no edge")
d = new_dfg()
s = stamp(d, "s", core=HOST)
c0 = task(d, HOST, "c0", "__host_bingo_kernel_engine_post", None, [s])
c1 = task(d, HOST, "c1", "__host_bingo_kernel_engine_post", None, [s])
mark_broadcast(d, c0, c0, queue=(0x20, 0), tasks=[c0])
mark_broadcast(d, c1, c1, queue=(0x20, 0), tasks=[c1])
check("same queue: no edge", d.bingo_order_collectives() == [])
d = new_dfg()
s = stamp(d, "s", core=HOST)
c0 = task(d, HOST, "c0", "__host_bingo_kernel_engine_post", None, [s])
c1 = task(d, HOST, "c1", "__host_bingo_kernel_engine_post", None, [s], chip=1)
mark_broadcast(d, c0, c0, queue=(0x20, 0), tasks=[c0])
mark_broadcast(d, c1, c1, queue=(0x21, 0), tasks=[c1])
check("two queues: one edge", len(d.bingo_order_collectives()) == 1)

print("\nreport only: the pairs are named, nothing is added")
d = new_dfg()
s = stamp(d, "s")
p0 = bput(d, "p0", [s], cluster=0)
p1 = bput(d, "p1", [s], cluster=1)
d.collectives_report = True
d.bingo_order_collectives()
check("no edge", not d.has_edge(p0, p1) and not d.order_edges)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    _sys.exit(1)
print("ALL PASS")
