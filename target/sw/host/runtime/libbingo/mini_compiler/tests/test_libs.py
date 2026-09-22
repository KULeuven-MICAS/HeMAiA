#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the block/port/linker layer.
#
# What is worth testing here is not that a well-formed pipeline assembles -- the workloads
# prove that -- but the REFUSALS and the JOIN. A contract that does not refuse is a comment,
# and a linker that silently fails to order two blocks produces a graph that compiles,
# passes the hang check, and reads a half-written buffer on silicon.
#
#   python3 test_libs.py

import sys

import networkx as nx

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402  (groups the compiler's subdirs onto sys.path)
from bingo_dfg import BingoDFG
from bingo_mem_handle import BingoMemAlloc
from bingo_liveness import collect_handle_users, reachability, can_share
from libs import Block, BlockResult, Ctx, Pipeline, Port, PortSpec, check_contract

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        FAILED.append(name)


def refuses(name, fn, needle=""):
    try:
        fn()
    except ValueError as e:
        check(name, needle.lower() in str(e).lower(), f"raised, but not about {needle!r}: {e}")
        return
    except Exception as e:                                        # noqa: BLE001
        check(name, False, f"raised {type(e).__name__}, wanted ValueError: {e}")
        return
    check(name, False, "did not raise")


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def new_ctx():
    d = BingoDFG(num_chiplets=1, num_clusters_per_chiplet=4, num_cores_per_cluster=4,
                 is_host_as_acc=True, chiplet_ids=[0], dep_tag_width=5)
    return Ctx(dfg=d, mesh=(16, 4, 16),
               roles={"gemm": 0, "simd": 1, "xdma": 2, "dm": 3, "host": 4})


class Producer(Block):
    """Writes `o`, and touches an internal temp that dies with the block."""
    name = "producer"

    @property
    def inputs(self): return {}

    @property
    def outputs(self):
        return {"o": PortSpec("A", "i8", (32, 128))}

    def build(self, ctx, bound):
        t = ctx.l1("temp", 64 * 1024)
        o = ctx.l1("out", 64 * 1024)
        ld = ctx.node("ld", ctx.dm, "__snax_k", Args(dst=t))
        wr = ctx.node("wr", ctx.gemm, "__snax_k", Args(src=t, dst=o), ld)
        return BlockResult(outputs={"o": Port(self.outputs["o"], o, (wr,), cluster=0)},
                           nodes=[ld, wr], sources=[ld], extra={"temp": t})


class Consumer(Block):
    """Reads `x`, plus a weight it loads from a node with NO predecessor."""
    name = "consumer"

    def __init__(self, layout="A", dtype="i8"):
        self._spec = PortSpec(layout, dtype, (32, 128))

    @property
    def inputs(self): return {"x": self._spec}

    @property
    def outputs(self): return {"y": PortSpec("D", "f16", (32, 128))}

    def build(self, ctx, bound):
        w = ctx.l1("wgt", 64 * 1024)
        y = ctx.l1("y", 64 * 1024)
        wld = ctx.node("wld", ctx.dm, "__snax_k", Args(dst=w))          # graph SOURCE
        rd = ctx.node("rd", ctx.dm, "__snax_k", Args(src=bound["x"].handle, dst=y))
        cmp_ = ctx.node("cmp", ctx.gemm, "__snax_k", Args(a=y, b=w, dst=y), [rd, wld])
        return BlockResult(
            inputs={"x": Port(self._spec, bound["x"].handle, (rd,))},
            outputs={"y": Port(self.outputs["y"], y, (cmp_,), cluster=0)},
            nodes=[wld, rd, cmp_], sources=[wld], extra={"wgt": w})


def assemble(gate_sources=True, consumer=None):
    ctx = new_ctx()
    pipe = Pipeline(ctx, gate_sources=gate_sources, verbose=False)
    a = pipe.add(Producer(), name="a")
    b = pipe.add(consumer or Consumer(), name="b", bind={"x": a.out("o")})
    return ctx, pipe, a, b


def shareable(ctx, x, y):
    nodes = sorted(ctx.dfg.node_list, key=lambda n: n.node_id)
    hu = collect_handle_users(nodes)
    desc = reachability(ctx.dfg, nodes)
    by = {v[0].name: v[1] for v in hu.values()}
    return can_share(by[x], by[y], desc)


# ---------------------------------------------------------------- the contract
print("the contract")

refuses("a dtype mismatch is refused, not converted",
        lambda: assemble(consumer=Consumer(dtype="f16")), "scale")
# A LAYOUT MISMATCH IS NOT ONE CASE, IT IS THREE, and the contract has to tell them apart:
# a conversion the hardware performs, one it cannot (too narrow a run at this precision),
# and one no pair of strides expresses at all (a transpose). Collapsing them into a single
# "insert a reshape" refusal was wrong in both directions -- it refused the conversion the
# xDMA does in one pass, and it promised one for the transpose, which needs another kernel.
refuses("a layout mismatch a block will not close is refused",
        lambda: assemble(consumer=Consumer(layout="D")), "does not close gaps")

closing = Consumer(layout="D")
closing.closes_gaps = True
refuses("an int8 reshape is refused for the RIGHT reason: the run is too narrow",
        lambda: assemble(consumer=closing), "8 B per lane")

# The transpose is checked on the plan directly: routed through assemble() it would hit
# the PRECISION refusal first, because this producer emits int8 and B-layout is an fp16
# conversion -- so the assertion would pass for the wrong reason.
from libs.comm import transfer as _staging                                     # noqa: E402
refuses("a transpose is refused as a transpose, not as a precision problem",
        lambda: _staging.plan(PortSpec("packed", "f16", (32, 128)),
                              PortSpec("B", "f16", (32, 128)),
                              mesh=(16, 4, 16), elem_bytes=2), "TRANSPOSE")
check("a reshape the xDMA can do is planned, not refused",
   _staging.plan(PortSpec("D", "f16", (32, 128)), PortSpec("A", "f16", (32, 128)),
                 mesh=(16, 4, 16), elem_bytes=2) != [])

bad_shape = Consumer()
bad_shape._spec = PortSpec("A", "i8", (64, 128))
refuses("a shape mismatch is refused", lambda: assemble(consumer=bad_shape), "shape")

refuses("an unbound input is refused",
        lambda: Pipeline(new_ctx(), verbose=False).add(Consumer(), name="b"),
        "not bound")


def _unknown():
    ctx = new_ctx()
    p = Pipeline(ctx, verbose=False)
    a = p.add(Producer(), name="a")
    p.add(Consumer(), name="b", bind={"x": a.out("o"), "z": a.out("o")})
refuses("binding an input the block does not have", _unknown, "inputs are")

ok = check_contract(Port(PortSpec("A", "i8", (32, 128)), None, ()),
                    PortSpec("A", "i8", (32, 128)), where="t")
check("a matching contract passes", ok is None, str(ok))

# ---------------------------------------------------------------- the join
print("\nthe join")

ctx, pipe, a, b = assemble()
g = ctx.dfg
wr = a.result.outputs["o"].ends[0]
rd = b.result.inputs["x"].ends[0]
check("the RAW edge exists, producer-of-o -> first-reader-of-o", g.has_edge(wr, rd))
check("the consumer's whole graph descends from it",
      all(n in (nx.descendants(g, wr) | {wr}) for n in b.result.nodes))

# the payoff: does static-L1 now see the stages as shareable?
check("producer's temp is reusable by the consumer's activation",
      shareable(ctx, "a.temp_cl0", "b.y_cl0"))
check("producer's temp is reusable by the consumer's WEIGHT (sources gated)",
      shareable(ctx, "a.temp_cl0", "b.wgt_cl0"))

ctx2, _, _, _ = assemble(gate_sources=False)
check("...and is NOT, when sources are left ungated",
      not shareable(ctx2, "a.temp_cl0", "b.wgt_cl0"))
check("the RAW path still shares without gating",
      shareable(ctx2, "a.temp_cl0", "b.y_cl0"))

# ---------------------------------------------------------------- namespacing
print("\nnamespacing")

names = {n.node_name for n in ctx.dfg.node_list}
check("each block's nodes carry its pipeline name", "a.wr_cl0" in names and "b.rd_cl0" in names,
      sorted(names))
handles = {v[0].name for v in collect_handle_users(
    sorted(ctx.dfg.node_list, key=lambda n: n.node_id)).values()}
check("and so do its handles", {"a.temp_cl0", "b.wgt_cl0"} <= handles, sorted(handles))
check("an empty scope is no scope", new_ctx().scope("") is not None
      and new_ctx().scope("").prefix == "")

# two instances of one block must not collide
ctx3 = new_ctx()
p3 = Pipeline(ctx3, verbose=False)
a3 = p3.add(Producer(), name="first")
b3 = p3.add(Producer(), name="second")
h3 = {v[0].name for v in collect_handle_users(
    sorted(ctx3.node_list if hasattr(ctx3, "node_list") else ctx3.dfg.node_list,
           key=lambda n: n.node_id)).values()}
check("two instances of one block get distinct handles",
      {"first.temp_cl0", "second.temp_cl0"} <= h3, sorted(h3))

# ---------------------------------------------------------------- the report
print("\nthe cut report")
blocked = pipe.report()
check("a clean pipeline reports no blocking node", blocked[("a", "b")] == [],
      [n.node_name for n in blocked[("a", "b")]])

if FAILED:
    print(f"\n{len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("\nall libs tests passed")
