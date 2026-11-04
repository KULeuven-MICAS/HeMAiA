#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Tests for the ChainGather as a configure, a start and a wait task: the tasks, the operand lifetimes,
# the crossing check and the teardown flags. They check the compiler's side. Whether a gather armed
# ahead runs as one armed in line is the hardware's, and a simulation's to show.
#
#   python3 test_xdma_chain_split.py

import os as _os
import sys as _sys

import networkx as nx

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402
from bingo_dfg import BingoDFG                                        # noqa: E402
from bingo_kernel_args import (SnaxBingoKernelXdma1dCopyArgs,         # noqa: E402
                               SnaxBingoKernelXdmaChainGatherArgs, SnaxBingoKernelXdmaChainStartArgs,
                               SnaxBingoKernelXdmaChainWaitArgs)
from bingo_liveness import collect_handle_users                       # noqa: E402
from bingo_mem_handle import BingoMemAlloc                            # noqa: E402
from libs import Ctx                                                  # noqa: E402
from libs.blocks import chain_gather, chain_gather_teardown           # noqa: E402

XDMA = 2
COPY = "__snax_bingo_kernel_xdma_1d_copy"


def graph():
    d = BingoDFG(num_chiplets=2, num_clusters_per_chiplet=2, num_cores_per_cluster=5, is_host_as_acc=True,
                 chiplet_ids=[0, 1], dep_tag_width=5)
    return Ctx(dfg=d, mesh=(16, 4, 16), roles={"gemm": 0, "simd": 1, "xdma": XDMA, "dm": 3, "host": 4})


def adder(ctx, cluster):
    g = ctx.at(cluster)
    return lambda name, kernel, args, deps: g.node(name, g.xdma, kernel, args, deps)


def short(n):
    return n.node_name.rsplit("_cl", 1)[0]


def copy_args():
    return SnaxBingoKernelXdma1dCopyArgs(0x100, 0x200, 64)


def gather(dst=0x1000, hops=((1, 0), (0, 0)), holds=()):
    return SnaxBingoKernelXdmaChainGatherArgs(0x100, [0x200 + i * 8 for i in range(len(hops) - 1)] + [dst], 64, "0",
                                              0, hops=list(hops), holds=holds)


def test_blocking_is_one_task():
    ctx = graph()
    t = chain_gather(adder(ctx, 0), "g", gather())
    assert t.kernel_name.endswith("xdma_chain_gather") and short(t) == "g"
    assert len(ctx.dfg.node_list) == 1


def test_split_tasks():
    ctx = graph()
    dfg = ctx.dfg
    dep = adder(ctx, 0)("dep", COPY, copy_args(), [])
    wt = chain_gather(adder(ctx, 0), "g", gather(), [dep], cfg_ahead=True)
    by = {short(n): n for n in dfg.node_list}
    cf, st = by["g_cfg"], by["g_st"]
    assert wt is by["g"] and wt.kernel_name.endswith("xdma_chain_wait")
    assert cf.kernel_name.endswith("xdma_chain_cfg") and st.kernel_name.endswith("xdma_chain_start")
    assert dfg.in_degree(cf) == 0, "the configure task waits for nothing"
    assert nx.has_path(dfg, cf, st) and nx.has_path(dfg, dep, st) and nx.has_path(dfg, st, wt)
    assert not nx.has_path(dfg, dep, cf)
    assert cf.config_only and st.operands_held_by is wt
    assert isinstance(st.kernel_args, SnaxBingoKernelXdmaChainStartArgs)
    assert isinstance(wt.kernel_args, SnaxBingoKernelXdmaChainWaitArgs) and wt.kernel_args.teardown


def test_configure_stays_behind_the_previous_gather():
    ctx = graph()
    add = adder(ctx, 0)
    w0 = chain_gather(add, "g0", gather(), cfg_ahead=True)
    chain_gather(add, "g1", gather(), cfg_ahead=True, cfg_after=[w0])
    by = {short(n): n for n in ctx.dfg.node_list}
    assert nx.has_path(ctx.dfg, by["g0_st"], by["g1_cfg"]), "a configure ahead of the previous start overwrites its CSRs"


def test_wait_holds_the_operands():
    ctx = graph()
    buf = BingoMemAlloc("src", size=64, mem_level="L1", chip_id=0, cluster_id=0)
    wt = chain_gather(adder(ctx, 0), "g", gather(holds=[buf]), cfg_ahead=True)
    users = collect_handle_users(ctx.dfg.node_list)[id(buf)][1]
    names = {short(n) for n in users}
    assert names == {"g_st", "g"}, f"the transfer reads it until the wait, and the configure task does not: {names}"
    assert wt in users


def test_crossing_edge_leaves_the_wait():
    # two gathers from different collectors sharing a middle hop: the later starts behind the earlier's WAIT
    ctx = graph()
    dfg = ctx.dfg
    mid = (1, 0)
    a = chain_gather(adder(ctx, 0), "a", gather(hops=((1, 1), mid, (0, 0))), cfg_ahead=True)
    b = chain_gather(adder(ctx, 1), "b", gather(hops=((1, 1), mid, (0, 1))), cfg_ahead=True)
    added = dfg.bingo_transform_order_crossed_chain_gathers()
    assert [(x.rsplit("_cl", 1)[0], y.rsplit("_cl", 1)[0]) for x, y in added] == [("a_st", "b_st")], added
    by = {short(n): n for n in dfg.node_list}
    assert dfg.has_edge(a, by["b_st"]), "a start released by the other's start would run beside it"
    assert not dfg.has_edge(by["a_st"], by["b_st"]) and b is by["b"]


def test_teardown_only_ahead_of_a_configure():
    ctx = graph()
    add = adder(ctx, 0)
    w0 = chain_gather(add, "g0", gather(), cfg_ahead=True)
    w1 = chain_gather(add, "g1", gather(), [w0], cfg_ahead=True)
    other = add("copy", COPY, copy_args(), [w1])
    w2 = chain_gather(add, "g2", gather(), [other], cfg_ahead=True)
    assert chain_gather_teardown(ctx.dfg, XDMA) == 1
    assert not w0.kernel_args.teardown, "a configure task follows"
    assert w1.kernel_args.teardown, "the copy expects clean CSRs"
    assert w2.kernel_args.teardown, "the last one leaves them clean"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(len(tests), "passed")
