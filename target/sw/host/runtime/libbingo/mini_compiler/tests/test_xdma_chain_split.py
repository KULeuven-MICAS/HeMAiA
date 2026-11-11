#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Tests for the ChainGather as a configure, a start and a wait task, the parts the compiler splits it into with
# async_exec (passes/bingo_dfg_schedule.py, tests/test_schedule.py for the split itself): the crossing check and the
# teardown rule on those parts. They check the compiler's side. Whether a gather armed ahead runs as one armed in line
# is the hardware's, and a simulation's to show.
#
#   python3 test_xdma_chain_split.py

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402
from bingo_dfg import BingoDFG                                        # noqa: E402
from bingo_kernel_args import (SnaxBingoKernelXdma1dCopyArgs,         # noqa: E402
                               SnaxBingoKernelXdmaChainGatherArgs, SnaxBingoKernelXdmaChainStartArgs,
                               SnaxBingoKernelXdmaChainWaitArgs)
from libs import Ctx                                                  # noqa: E402
from bingo_dfg_schedule import _chain_teardown                       # noqa: E402
from libs.blocks import chain_gather                                  # noqa: E402

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


def split(add, name, args, after=()):
    """The parts the compiler makes of a ChainGather: configure (config_only), start, wait (the gather's name)."""
    cf = add(f"{name}_cfg", "__snax_bingo_kernel_xdma_chain_cfg", args, [])
    cf.config_only = True
    start = SnaxBingoKernelXdmaChainStartArgs(args)
    st = add(f"{name}_st", start.KERNEL_NAME, start, list(after) + [cf])
    wait = SnaxBingoKernelXdmaChainWaitArgs(args)
    wt = add(name, wait.KERNEL_NAME, wait, [st])
    st.operands_held_by = wt
    return cf, st, wt


def test_blocking_is_one_task():
    ctx = graph()
    t = chain_gather(adder(ctx, 0), "g", gather())
    assert t.kernel_name.endswith("xdma_chain_gather") and short(t) == "g"
    assert len(ctx.dfg.node_list) == 1


def test_crossing_edge_leaves_the_wait():
    # two gathers from different collectors sharing a middle hop: the later starts behind the earlier's WAIT
    ctx = graph()
    dfg = ctx.dfg
    mid = (1, 0)
    _, a_st, a = split(adder(ctx, 0), "a", gather(hops=((1, 1), mid, (0, 0))))
    _, b_st, b = split(adder(ctx, 1), "b", gather(hops=((1, 1), mid, (0, 1))))
    added = dfg.bingo_transform_order_crossed_chain_gathers()
    assert [(x.rsplit("_cl", 1)[0], y.rsplit("_cl", 1)[0]) for x, y in added] == [("a_st", "b_st")], added
    assert dfg.has_edge(a, b_st), "a start released by the other's start would run beside it"
    assert not dfg.has_edge(a_st, b_st)


def test_teardown_only_ahead_of_a_configure():
    ctx = graph()
    add = adder(ctx, 0)
    c0, s0, w0 = split(add, "g0", gather())
    c1, s1, w1 = split(add, "g1", gather(), [w0])
    other = add("copy", COPY, copy_args(), [w1])
    c2, s2, w2 = split(add, "g2", gather(), [other])
    # a core's program as the compiler orders it
    assert _chain_teardown([c0, s0, w0, c1, s1, w1, other, c2, s2, w2]) == 1
    assert not w0.kernel_args.teardown, "a configure task follows"
    assert w1.kernel_args.teardown, "the copy expects clean CSRs"
    assert w2.kernel_args.teardown, "the last one leaves them clean"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(len(tests), "passed")
