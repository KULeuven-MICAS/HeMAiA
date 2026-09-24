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

from dataclasses import replace

import networkx as nx

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402  (groups the compiler's subdirs onto sys.path)
from bingo_dfg import BingoDFG
from bingo_mem_handle import BingoMemAlloc
from bingo_liveness import collect_handle_users, reachability, can_share
from libs import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Pipeline, Port,
                  PortSpec, check_contract)

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
        return {"o": PortSpec("A", "i8", (32, 128), mem_level="L1", cluster=0)}

    def build(self, ctx, bound):
        t = ctx.l1("temp", 64 * 1024)
        o = ctx.l1("out", 64 * 1024)
        ld = ctx.node("ld", ctx.dm, "__snax_k", Args(dst=t))
        wr = ctx.node("wr", ctx.gemm, "__snax_k", Args(src=t, dst=o), ld)
        return BlockResult(outputs={"o": Port(self.outputs["o"], o, (wr,))},
                           nodes=[ld, wr], sources=[ld], extra={"temp": t})


class Consumer(Block):
    """Reads `x`, plus a weight it loads from a node with NO predecessor."""
    name = "consumer"

    def __init__(self, layout="A", dtype="i8", mem_level="L1", needs=None, cluster=0):
        # Every port names the level it is handed at, and an L1 one names its cluster.
        # `needs` is how a block that FETCHES says what its own engines then read -- see
        # the `fetching` consumer below.
        self._spec = PortSpec(layout, dtype, (32, 128), mem_level=mem_level,
                              cluster=cluster if mem_level == "L1" else None)
        self._needs = needs

    @property
    def inputs(self): return {"x": self._spec}

    @property
    def needs(self): return self._needs or self.inputs

    @property
    def outputs(self): return {"y": PortSpec("D", "f16", (32, 128), mem_level="L1", cluster=0)}

    def build(self, ctx, bound):
        w = ctx.l1("wgt", 64 * 1024)
        y = ctx.l1("y", 64 * 1024)
        wld = ctx.node("wld", ctx.dm, "__snax_k", Args(dst=w))          # graph SOURCE
        rd = ctx.node("rd", ctx.dm, "__snax_k", Args(src=bound["x"].handle, dst=y))
        cmp_ = ctx.node("cmp", ctx.gemm, "__snax_k", Args(a=y, b=w, dst=y), [rd, wld])
        return BlockResult(
            inputs={"x": Port(self._spec, bound["x"].handle, (rd,))},
            outputs={"y": Port(self.outputs["y"], y, (cmp_,))},
            nodes=[wld, rd, cmp_], sources=[wld], extra={"wgt": w})


def assemble(gate_sources=True, consumer=None):
    ctx = new_ctx()
    pipe = Pipeline(ctx, gate_sources=gate_sources, verbose=False)
    a = pipe.add(Producer(), name="a")
    b = pipe.add(consumer or Consumer(), name="b", bind={"x": a.out("o")})
    pipe.run()                 # connect, resolve, build -- see libs/comm/link.py
    return ctx, pipe, a, b


def shareable(ctx, x, y):
    nodes = sorted(ctx.dfg.node_list, key=lambda n: n.node_id)
    hu = collect_handle_users(nodes)
    desc = reachability(ctx.dfg, nodes)
    by = {v[0].name: v[1] for v in hu.values()}
    return can_share(by[x], by[y], desc)


# ---------------------------------------------------------------- the contract
print("the contract")

from libs.block import Reshape as _Reshape_ctor                          # noqa: E402

refuses("a dtype mismatch is refused, not converted",
        lambda: assemble(consumer=Consumer(dtype="f16")), "scale")
# A LAYOUT GAP IS REFUSED BY NAME: no realisation of the consumer accepts what the
# producer emits. Matching asks the block, not whether the hardware COULD convert -- those
# are different questions, and a block reading A-layout bytes that arrived row-major
# computes a scrambled answer nothing catches. A block that converts says so with a
# variant.
refuses("a layout gap no realisation accepts is refused",
        lambda: assemble(consumer=Consumer(layout="D")), "but this port reads")

# THE HARDWARE'S REASONS LIVE IN THE BLOCK that owns the conversion, which is where the
# resolver reads them from when it prunes. A reshape asked for an int8 A-layout run is
# refused by the Reshape, naming the lane width.
refuses("an int8 reshape is refused for the RIGHT reason: the run is too narrow",
        lambda: _Reshape_ctor(rows=32, cols=128, mesh=(16, 4, 16), cluster=0,
                              src=Layout.ROW_MAJOR, dst=Layout.A, dtype=DType.I8),
        "8 B per lane")

# The transpose is checked on the plan directly: routed through assemble() it would hit
# the PRECISION refusal first, because this producer emits int8 and B-layout is an fp16
# conversion -- so the assertion would pass for the wrong reason.
#
# A -> B IS THE PAIR THAT STAYS REFUSED. A row_major side can pivot through the A/B
# identity and is planned as two steps; two BLOCKED layouts have nothing to pivot through,
# so the message must still name the transpose rather than a precision problem.
from libs.comm import transfer as _staging                                     # noqa: E402
refuses("a transpose is refused as a transpose, not as a precision problem",
        lambda: _staging.plan(PortSpec("A", "f16", (32, 128), mem_level="L1", cluster=0),
                              PortSpec("B", "f16", (32, 128), mem_level="L1", cluster=0),
                              mesh=(16, 4, 16), elem_bytes=2), "TRANSPOSE")
check("a reshape the xDMA can do is planned, not refused",
   _staging.plan(PortSpec("D", "f16", (32, 128), mem_level="L1", cluster=0),
                 PortSpec("A", "f16", (32, 128), mem_level="L1", cluster=0),
                 mesh=(16, 4, 16), elem_bytes=2) != [])

# EVERY PORT SAYS WHERE. A tensor is in some memory at every point in the graph, so a
# spec that declines to say is a hole the binding would silently fill.
from bingo_mem_handle import BingoMemSymbol as _Sym                           # noqa: E402
try:
    PortSpec("A", "i8", (32, 128))
    check("a port with no mem_level cannot be built", False, "it constructed")
except TypeError as _e:
    check("a port with no mem_level cannot be built", "mem_level" in str(_e), str(_e))
refuses("...and passing None says what to do instead",
        lambda: PortSpec("A", "i8", (32, 128), mem_level=None), "realisation per level")
# ...and L1 is the one memory where "which cluster" has an answer, so it is required
# there and refused everywhere else.
refuses("an L1 port must name its cluster",
        lambda: PortSpec("A", "i8", (32, 128), mem_level="L1"), "name the cluster")
refuses("a main-memory port must not",
        lambda: PortSpec("A", "i8", (32, 128), mem_level="L3", cluster=0),
        "only L1 is per-cluster")
# The vocabulary is a closed set: a typo is refused at construction, naming the valid
# values, rather than reaching a kernel as a layout nobody implements.
for _bad, _kind in ((("packd", "i8", None), "Layout"),
                    (("A", "fp16", None), "DType"),
                    (("A", "i8", "L5"), "MemLevel")):
    refuses(f"a typo'd {_kind} is refused with the valid values named",
            lambda b=_bad: PortSpec(b[0], b[1], (32, 128), mem_level=b[2]),
            "is not a valid")
_s = PortSpec("A", "i8", (32, 128), mem_level="L1", cluster=0)
check("a plain string is coerced to the enum member",
      (_s.layout is Layout.A and _s.dtype is DType.I8 and _s.mem_level is MemLevel.L1),
      f"got {_s.layout!r} {_s.dtype!r} {_s.mem_level!r}")
check("the two spellings make equal specs",
      _s == PortSpec(Layout.A, DType.I8, (32, 128), mem_level=MemLevel.L1, cluster=0))
check("a member still behaves as its string", f"{Layout.D32}" == "d32" and Layout.D32 == "d32")

check("a bound port agrees with its handle",
      Port(PortSpec("A", "i8", (32, 128), mem_level="L1", cluster=0),
           BingoMemAlloc("h", 4096, "L1"), ()).spec.mem_level == "L1")
check("...and a staged symbol is main memory",
      Port(PortSpec("A", "i8", (32, 128), mem_level="L3"), _Sym("staged"),
           ()).spec.mem_level == "L3")
# A FIXED ADDRESS records neither level nor cluster, so the spec is the only statement
# there is and it stands. Anything the handle DOES know is checked against instead: a
# staged symbol is main memory by construction, so calling it L4 is refused rather than
# carried downstream as fact.
from bingo_mem_handle import BingoMemFixedAddr as _Fixed                 # noqa: E402
_p = Port(PortSpec("A", "i8", (32, 128), mem_level="L4"), _Fixed(0x8000_0000), ())
check("a fixed address takes the level the spec states", _p.spec.mem_level == "L4",
      f"got {_p.spec.mem_level!r}")
refuses("a handle that contradicts its spec is refused",
        lambda: Port(PortSpec("A", "i8", (32, 128), mem_level="L4"), _Sym("staged"), ()),
        "is in L3")

bad_shape = Consumer()
bad_shape._spec = PortSpec("A", "i8", (64, 128), mem_level="L1", cluster=0)
refuses("a shape mismatch is refused", lambda: assemble(consumer=bad_shape), "shape")

def _unbound():
    p = Pipeline(new_ctx(), verbose=False)
    p.add(Consumer(), name="b")
    p.run()
refuses("an unbound input is refused", _unbound, "not bound")


def _unknown():
    ctx = new_ctx()
    p = Pipeline(ctx, verbose=False)
    a = p.add(Producer(), name="a")
    p.add(Consumer(), name="b", bind={"x": a.out("o"), "z": a.out("o")})
    p.run()
refuses("binding an input the block does not have", _unknown, "inputs are")

ok = check_contract(Port(PortSpec("A", "i8", (32, 128), mem_level="L1", cluster=0), None, ()),
                    PortSpec("A", "i8", (32, 128), mem_level="L1", cluster=0), where="t")
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
      shareable(ctx, "a_temp_cl0", "b_y_cl0"))
check("producer's temp is reusable by the consumer's WEIGHT (sources gated)",
      shareable(ctx, "a_temp_cl0", "b_wgt_cl0"))

ctx2, _, _, _ = assemble(gate_sources=False)
check("...and is NOT, when sources are left ungated",
      not shareable(ctx2, "a_temp_cl0", "b_wgt_cl0"))
check("the RAW path still shares without gating",
      shareable(ctx2, "a_temp_cl0", "b_y_cl0"))

# ---------------------------------------------------------------- namespacing
print("\nnamespacing")

names = {n.node_name for n in ctx.dfg.node_list}
check("each block's nodes carry its pipeline name", "a_wr_cl0" in names and "b_rd_cl0" in names,
      sorted(names))
handles = {v[0].name for v in collect_handle_users(
    sorted(ctx.dfg.node_list, key=lambda n: n.node_id)).values()}
check("and so do its handles", {"a_temp_cl0", "b_wgt_cl0"} <= handles, sorted(handles))
check("an empty scope is no scope", new_ctx().scope("") is not None
      and new_ctx().scope("").prefix == "")

# two instances of one block must not collide
ctx3 = new_ctx()
p3 = Pipeline(ctx3, verbose=False)
a3 = p3.add(Producer(), name="first")
b3 = p3.add(Producer(), name="second")
p3.run()
h3 = {v[0].name for v in collect_handle_users(
    sorted(ctx3.node_list if hasattr(ctx3, "node_list") else ctx3.dfg.node_list,
           key=lambda n: n.node_id)).values()}
check("two instances of one block get distinct handles",
      {"first_temp_cl0", "second_temp_cl0"} <= h3, sorted(h3))

# ---------------------------------------------------------------- the report
print("\nthe cut report")
blocked = pipe.report()
check("a clean pipeline reports no blocking node", blocked[("a", "b")] == [],
      [n.node_name for n in blocked[("a", "b")]])

# ------------------------------------------------------- the resolver
print("\nthe resolver: boundaries from the neighbours")
from libs.block import Reshape as _Reshape                               # noqa: E402
from libs.block.simd.norm import RMSNorm as _RMSNorm                     # noqa: E402

from libs.comm import variants_of                                        # noqa: E402

_T, _D, _M = 32, 128, (16, 4, 16)


def _chain(rows, pin=None, demand=None):
    """norm -> Reshape, the two-stage chain, with only what `pin` says decided."""
    c = new_ctx()
    g = c.at(0)
    x = Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (rows, _D), mem_level=MemLevel.L1, cluster=0),
             g.l1("x", rows * _D * 2), ())
    p = Pipeline(c, verbose=False)
    n = p.add(_RMSNorm(rows=rows, cols=_D, cluster=0, **(pin or {})), "norm", x=x)
    r = p.add(_Reshape(rows=rows, cols=_D, mesh=_M, cluster=0), "to_a", x=n.out())
    p.demand(r.out(), PortSpec(demand or Layout.A, DType.F16, (rows, _D),
                               mem_level=MemLevel.L1, cluster=0))
    p.run()
    return p, n, r


# LEGALITY IS BY CONSTRUCTION: a template offers every combination and the block's own
# constructor is what prunes it. At rows != 32 the col_major kernel does not exist, so
# four more realisations refuse -- and the refusals carry the block's own message.
_v32 = variants_of(_RMSNorm(rows=32, cols=_D, cluster=0, mesh=_M))
_v64 = variants_of(_RMSNorm(rows=64, cols=_D, cluster=0, mesh=_M))
check("a template enumerates its realisations", len(_v32) > 1, len(_v32))
check("...and fewer of them are legal off the col_major kernel's shape",
      len(_v64) < len(_v32), (len(_v32), len(_v64)))
check("a pinned block offers exactly one",
      len(variants_of(_RMSNorm(rows=32, cols=_D, cluster=0, in_layout=Layout.ROW_MAJOR,
                               out_layout=Layout.ROW_MAJOR, out_dtype=DType.F16,
                               in_level=MemLevel.L1))) == 1)
check("...and leaving only WHERE x arrives open offers one per level",
      len(variants_of(_RMSNorm(rows=32, cols=_D, cluster=0, in_layout=Layout.ROW_MAJOR,
                               out_layout=Layout.ROW_MAJOR,
                               out_dtype=DType.F16))) == 2)

# FOLDS RANK FIRST, so the resolver takes the col_major kernel even though it costs the
# chain more passes than the row_major arm -- 2,062 cycles off the busy engine for one
# extra transfer on an idle one.
_p, _n, _r = _chain(32)
check("rows=32: the chain comes out fold-free", _p.cost.folds == 0, str(_p.cost))
check("...on the col_major kernel", _n.block.col_major, _n.chosen.describe())
check("...even though it is NOT the fewest passes",
      _p.cost.xdma > min(v.cost.xdma for v in _n.variants), str(_p.cost))

# THE NEIGHBOUR FOLLOWS. Nothing told the Reshape what its source would be.
check("the Reshape took the layout the norm chose",
      _r.chosen.inputs["x"].layout == _n.chosen.outputs["y"].layout,
      (_n.chosen.describe(), _r.chosen.describe()))

# LEGALITY AGAIN, now through the chain: at rows=64 every col_major realisation refuses,
# so the resolver is left with the folding arm rather than proposing something unbuildable.
_p64, _n64, _ = _chain(64)
check("rows=64 falls back to the row_major kernel", not _n64.block.col_major,
      _n64.chosen.describe())
check("...and then it DOES pay a fold per row", _p64.cost.folds == 64, str(_p64.cost))

# A PINNED FIELD IS A CONSTRAINT, not a hint. The block does not overrule the layer.
_pp, _pn, _ = _chain(32, pin=dict(in_layout=Layout.ROW_MAJOR,
                                  out_layout=Layout.ROW_MAJOR, out_dtype=DType.F16))
check("a pinned boundary is honoured even when it costs folds",
      _pn.chosen.outputs["y"].layout == Layout.ROW_MAJOR and _pp.cost.folds == _T,
      (_pn.chosen.describe(), str(_pp.cost)))

# WHAT IT PICKS MUST BUILD. The whole point of a resolver is that its answer is realisable.
check("every stage of the resolved chain built", all(s.result is not None
                                                     for s in _p.stages))
check("...and the join is a real edge",
      _p.ctx.dfg.has_edge(_p.stages[0].result.outputs["y"].ends[-1],
                          _p.stages[1].result.inputs["x"].ends[0]))

# THE FAR END IS THE APPLICATION'S TO PIN. Without a demand the chain would stop wherever
# was cheapest, which is right for the graph and wrong for whatever reads the buffer.
_pd, _, _rd = _chain(32, demand=Layout.D)
check("demand() pins the chain's last layout",
      _rd.chosen.outputs["y"].layout == Layout.D, _rd.chosen.describe())
refuses("a demand nothing can produce is refused",
        lambda: _chain(32, demand=Layout.MONOID), "no set of realisations")

# x_in_place IS A PROMISE ABOUT ANOTHER BLOCK, so it is never a knob the resolver turns.
check("x_in_place is not among the realisations",
      not any("x_in_place" in v.params for v in _v32),
      [v.params for v in _v32])

# ONE REGISTERED KERNEL, dispatching on the layout arguments.
from bingo_kernel_args import SnaxBingoKernelSimdRmsnormArgs as _RNA   # noqa: E402
check("both arms emit the same kernel symbol",
      _RNA(0x100, 0x200, 32, 128).KERNEL_NAME
      == _RNA(0x140, 0x200, 32, 128, input_layout="col_major",
              output_layout="col_major", seed_addr=0x100).KERNEL_NAME
      == "__snax_bingo_kernel_simd_rmsnorm")
for _why, _kw in (("a transposing pair", dict(input_layout="col_major")),
                  ("col_major without a seed beat",
                   dict(input_layout="col_major", output_layout="col_major")),
                  ("col_major with int8 out",
                   dict(input_layout="col_major", output_layout="col_major",
                        seed_addr=0x100, out_i8=True)),
                  ("a blocked layout", dict(input_layout="A", output_layout="A"))):
    try:
        _RNA(0x140, 0x200, 32, 128, **_kw)
        check(f"the args refuse {_why}", False, "it constructed")
    except ValueError:
        check(f"the args refuse {_why}", True)

# x_in_place IS A PROMISE, and breaking it is refused rather than silently corrected.
_bad = _RMSNorm(rows=_T, cols=_D, cluster=0, in_layout=Layout.COL_MAJOR,
                out_layout=Layout.ROW_MAJOR, out_dtype=DType.F16, x_in_place=True,
                in_level=MemLevel.L1)
_g2 = new_ctx().at(0)
_bad.alloc(_g2)
try:
    _bad.build(_g2, {"x": Port(PortSpec(Layout.COL_MAJOR, DType.F16, (_T, _D),
                                        mem_level=MemLevel.L1, cluster=0),
                               _g2.l1("elsewhere", _T * _D * 2), (), name="x")})
    check("x_in_place=True on a foreign buffer is refused", False, "it built")
except ValueError:
    check("x_in_place=True on a foreign buffer is refused", True)

# ------------------------------------------------- the engine deduction
print("\nwhich engine moves the operand")
from libs.comm.ports import MemLevel as _ML                              # noqa: E402

_XDMA, _IDMA, _SIMD = _ctx_roles = 2, 3, 1


def _engines(in_lay, out_lay, level):
    c = new_ctx()
    g = c.at(0)
    b = _RMSNorm(rows=_T, cols=_D, cluster=0, in_layout=in_lay, out_layout=out_lay,
                 out_dtype=DType.F16, in_level=level)
    h = g.l1("x", _T * _D * 2) if level == _ML.L1 else g.l3("x_l3", _T * _D * 2)
    r = b.build(g, {"x": Port(PortSpec(in_lay, DType.F16, (_T, _D), mem_level=level,
                                       cluster=0 if level == _ML.L1 else None),
                              h, (), name="x")})
    return [(n.node_name.split("_cl")[0], n._assigned_core_id) for n in r.nodes], b


# EVERY NON-TRANSPOSING MOVE IS ON THE iDMA, and only transposes are on the xDMA.
for _in, _out, _lvl in ((Layout.ROW_MAJOR, Layout.ROW_MAJOR, _ML.L3),
                        (Layout.COL_MAJOR, Layout.COL_MAJOR, _ML.L1),
                        (Layout.COL_MAJOR, Layout.COL_MAJOR, _ML.L3),
                        (Layout.ROW_MAJOR, Layout.COL_MAJOR, _ML.L3)):
    _ns, _ = _engines(_in, _out, _lvl)
    _moves = [(n, core) for n, core in _ns if core != _SIMD]
    check(f"{_in}->{_out} from {_lvl}: no plain copy on the xDMA",
          all(core == _IDMA for n, core in _moves if not n.startswith("Xpose")), _ns)
    check(f"{_in}->{_out} from {_lvl}: every transpose IS on the xDMA",
          all(core == _XDMA for n, core in _moves if n.startswith("Xpose")), _ns)

# AN L3 OPERAND COSTS NO EXTRA NODE when the tile is wanted col_major: the load IS the
# staging copy, because the iDMA reaches main memory just as readily as L1.
_l1, _ = _engines(Layout.COL_MAJOR, Layout.COL_MAJOR, _ML.L1)
_l3, _ = _engines(Layout.COL_MAJOR, Layout.COL_MAJOR, _ML.L3)
check("an L3 col_major input costs no extra node", len(_l1) == len(_l3), (_l1, _l3))

# A ROW-MAJOR L3 OPERAND DOES cost one, and it lands on the iDMA, not the xDMA.
_l1r, _ = _engines(Layout.ROW_MAJOR, Layout.ROW_MAJOR, _ML.L1)
_l3r, _b = _engines(Layout.ROW_MAJOR, Layout.ROW_MAJOR, _ML.L3)
check("an L3 row_major input costs exactly one iDMA load",
      len(_l3r) == len(_l1r) + 1 and _l3r[0] == ("Load_x", _IDMA), _l3r)
check("...and does not change the xDMA count",
      _b.xdma_passes() == 0, _b.xdma_passes())
check("...while the L3 realisation prices it", _b.idma_passes() == 1, _b.idma_passes())

# L4 IS REFUSED, because the hoist is one move for the whole layer, not one per operator.
# In the CONSTRUCTOR, like every other refusal, so the realisation never enters the search.
refuses("an L4 operand is refused",
        lambda: _RMSNorm(rows=_T, cols=_D, cluster=0, in_layout=Layout.ROW_MAJOR,
                         out_layout=Layout.ROW_MAJOR, out_dtype=DType.F16,
                         in_level=_ML.L4), "in_level")

# ------------------------------------------------ RMSNorm straight into a GEMM operand
print("\nRMSNorm writing a GEMM operand (out_layout A / B)")


def _norm_out(in_lay, out_lay, dtype=DType.I8, rows=_T):
    c = new_ctx()
    g = c.at(0)
    b = _RMSNorm(rows=rows, cols=_D, cluster=0, in_layout=in_lay, out_layout=out_lay,
                 out_dtype=dtype, inv_scale_f32bits=0x42800000, mesh=_M,
                 in_level=_ML.L1)
    h = g.l1("x", rows * _D * 2)
    r = b.build(g, {"x": Port(PortSpec(in_lay, DType.F16, (rows, _D), mem_level=_ML.L1, cluster=0),
                              h, (), name="x")})
    names = [(n.node_name.split("_cl")[0], n._assigned_core_id) for n in r.nodes]
    simd = [n._kernel_args for n in r.nodes if n._assigned_core_id == _SIMD]
    return names, b, r, simd


# THE OUTPUT LAYOUT PICKS THE KERNEL: A is a row_major run, B a col_major one.
_ns, _b, _r, _k = _norm_out(Layout.ROW_MAJOR, Layout.A)
check("row_major -> A is ONE SIMD node, no transpose", _ns == [("Rmsnorm", _SIMD)], _ns)
check("...on the row_major kernel", not _b.col_major and _b.xdma_passes() == 0,
      (_b.col_major, _b.xdma_passes()))
check("...asking the kernel for A, int8, with the mesh and the scale",
      (_k[0].output_layout, _k[0].out_i8, _k[0].mesh, _k[0].inv_scale_f32bits)
      == ("A", True, (16, 4, 16), 0x42800000), vars(_k[0]))
check("...and its port says A/int8",
      (_r.outputs["y"].spec.layout, _r.outputs["y"].spec.dtype) == (Layout.A, DType.I8),
      _r.outputs["y"].spec)

_ns, _b, _r, _k = _norm_out(Layout.COL_MAJOR, Layout.B)
check("col_major -> B runs the col_major kernel with no output transpose",
      _b.col_major and _b.xdma_passes() == 0 and _ns[-1] == ("Rmsnorm_t", _SIMD), _ns)
check("...asking the kernel for B", _k[0].output_layout == "B", vars(_k[0]))

# THE CROSSED PAIRS COST EXACTLY ONE xDMA TRANSPOSE, IN FRONT -- never one behind.
_ns, _b, _, _k = _norm_out(Layout.COL_MAJOR, Layout.A)
check("col_major -> A: Xpose_in, then the row_major kernel writes A",
      [n for n, _ in _ns] == ["Xpose_in", "Rmsnorm"] and _k[0].output_layout == "A", _ns)
_ns, _b, _, _k = _norm_out(Layout.ROW_MAJOR, Layout.B)
check("row_major -> B: Xpose_in, then the col_major kernel writes B",
      [n for n, _ in _ns] == ["Xpose_in", "Rmsnorm_t"] and _k[0].output_layout == "B", _ns)

# fp16 blocked output is a kernel too, and an int8 row_major one stays on the row kernel.
_ns, _, _r, _k = _norm_out(Layout.ROW_MAJOR, Layout.A, dtype=DType.F16)
check("row_major -> A at fp16", _k[0].out_i8 is False
      and _r.outputs["y"].spec.dtype == DType.F16, vars(_k[0]))
_ns, _b, _, _k = _norm_out(Layout.COL_MAJOR, Layout.ROW_MAJOR)
check("int8 row_major out from a col_major input runs the ROW kernel (Xpose_in, Rmsnorm)",
      not _b.col_major and [n for n, _ in _ns] == ["Xpose_in", "Rmsnorm"], _ns)

for _why, _kw in (("B needs rows == 32", dict(in_lay=Layout.COL_MAJOR, out_lay=Layout.B,
                                                rows=64)),
                  ("an int8 col_major output is refused",
                   dict(in_lay=Layout.COL_MAJOR, out_lay=Layout.COL_MAJOR)),
                  ("D is not an operand", dict(in_lay=Layout.ROW_MAJOR, out_lay=Layout.D))):
    try:
        _norm_out(**_kw)
        check(_why, False, "it built")
    except ValueError:
        check(_why, True)

# THE ARGS CLASS REFUSES THE CROSSED PAIRS the block never emits, by name.
for _pair in (("row_major", "B"), ("col_major", "A")):
    try:
        _RNA(0, 0, _T, _D, input_layout=_pair[0], output_layout=_pair[1], seed_addr=64,
             mesh=(16, 4, 16))
        check(f"args refuse {_pair[0]} -> {_pair[1]}", False, "accepted")
    except ValueError:
        check(f"args refuse {_pair[0]} -> {_pair[1]}", True)

# A blocked output without the consuming GEMM's mesh cannot derive its read order.
try:
    _RMSNorm(rows=_T, cols=_D, cluster=0, in_layout=Layout.ROW_MAJOR,
             out_layout=Layout.A, out_dtype=DType.F16, in_level=_ML.L1)
    check("out_layout A without a mesh is refused", False, "built")
except ValueError:
    check("out_layout A without a mesh is refused", True)

# ------------------------------------------------ connecting the norm straight to a GEMM
print("\nno glue in between: the norm has to produce the operand itself")
from libs.block import Linear as _Linear                                 # noqa: E402
from libs.block import Quantize as _Quantize                             # noqa: E402


def _direct(rows, in_lay=Layout.ROW_MAJOR, glue=False):
    """norm -> [Reshape -> Quantize ->] Linear, with nothing pinned but the ends."""
    c = new_ctx()
    g = c.at(0)
    x = Port(PortSpec(in_lay, DType.F16, (rows, _D), mem_level=MemLevel.L1, cluster=0),
             g.l1("x", rows * _D * 2), ())
    w = Port(PortSpec(Layout.B, DType.I8, (_D, _D), mem_level=MemLevel.L1, cluster=0),
             g.l1("w", _D * _D), ())
    p = Pipeline(c, verbose=False)
    h = p.add(_RMSNorm(rows=rows, cols=_D, cluster=0, mesh=_M,
                       inv_scale_f32bits=0x42800000), "norm1", x=x).out()
    if glue:
        h = p.add(_Reshape(rows=rows, cols=_D, mesh=_M, cluster=0), "to_a", x=h).out()
        h = p.add(_Quantize(rows=rows, cols=_D, cluster=0,
                            inv_scale_f32bits=0x42800000), "q", x=h).out()
    p.add(_Linear(tokens=rows, d_in=_D, d_out=_D, mesh=_M, cluster=0), "qkv", x=h, w=w)
    p.run()
    return p


# WITH NOTHING BETWEEN THEM the norm is the only stage that can reach A/int8, so it must
# write the operand out of its own kernel -- which is the fused route, and the only legal
# one here. That is the block making a choice INSIDE its boundary, not the chain losing a
# stage: no Reshape and no Quantize were ever added.
_p = _direct(64)
_n = _p.stages[0].chosen
check("norm -> GEMM with no glue: the norm writes A/int8 itself",
      (_n.outputs["y"].layout, _n.outputs["y"].dtype) == (Layout.A, DType.I8),
      _n.describe())
check("...and that is the whole chain, two stages", len(_p.stages) == 2, len(_p.stages))

# THE SAME CHAIN WITH THE GLUE THE LAYER ASKED FOR still has it. A resolver that deleted
# stages it thought redundant would be overruling the layer's composition; the cost may be
# higher and that is the layer's business.
_p = _direct(64, glue=True)
check("the Reshape and Quantize the layer wrote are still built",
      [s.name for s in _p.stages] == ["norm1", "to_a", "q", "qkv"],
      [s.name for s in _p.stages])
check("...and the norm then stops at fp16, leaving them the work",
      _p.stages[0].chosen.outputs["y"].dtype == DType.F16,
      _p.stages[0].chosen.describe())

# AT rows == 32 the fold-free kernel exists, so the glued chain is fold-free too: the norm
# emits col_major and the Reshape adapts to a col_major source it was never told about.
_p = _direct(32, glue=True)
check("rows=32: the chain comes out fold-free", _p.cost.folds == 0, str(_p.cost))
check("...because the Reshape took whatever the norm chose",
      _p.stages[1].chosen.inputs["x"].layout == _p.stages[0].chosen.outputs["y"].layout,
      (_p.stages[0].chosen.describe(), _p.stages[1].chosen.describe()))

# ------------------------------------------------ the blocked nest, for any mesh
print("\nthe blocked nest: RMSNorm into the operand of ANY mesh")
from blocked_nest import blocked_nest as _bn, verify as _bn_verify   # noqa: E402

# EVERY SHAPE THE CLUSTER CFGS DECLARE, both outputs, both precisions: derived AND walked
# against the index map (blocked_nest verifies before it returns; the walk is repeated here
# so a derivation change cannot silently skip it).
_PR, _PC = (_D // 32 + 1) * 64 + 8, 2 * _T + 8
for _mesh, _want in (((16, 4, 16), "AAAA"), ((16, 8, 8), "AAAA"), ((16, 8, 16), "AAAA"),
                     ((1, 32, 32), "AAAA"), ((1, 16, 32), "AAA-"), ((32, 2, 32), "----")):
    _mu, _ku, _nu = _mesh
    _got = ""
    for _ob in (2, 1):
        for _args in ((_T, _D, _PR, _mu, _ku, _ob, True), (_D, _T, _PC, _nu, _ku, _ob, False)):
            try:
                _n = _bn(*_args)
                _bn_verify(_n, _args[0], _args[1], _args[3], _args[4], _ob)
                _got += "A" if _n.tasks <= 2 else "?"
            except ValueError:
                _got += "-"
    check(f"mesh {_mesh}: A f16, B f16, A i8, B i8 = {_want}", _got == _want, _got)

# (16, 4, 16) STILL DERIVES TO THE HAND-WRITTEN NEST that ran bit-exact on RTL.
_n = _bn(_T, _D, _PR, 16, 4, 1, True)
check("(16,4,16) int8 A is the RTL-validated nest",
      (_n.rd_lane, _n.rd, _n.wr_lane, _n.wr, _n.tasks)
      == (_PR, ((4, 8 * _PR), (32, 8)), 8, ((2, 2048), (32, 64)), 1), _n)

# THE REFUSALS SAY WHY, and they are hardware facts, not search limits.
for _why, _args, _needle in (
        ("tileSize 2 is below the 8 B lane", (_T, _D, _PR, 32, 2, 2, True), "tileSize 2"),
        ("an int8 atom a block apart needs a 2-D lane grid", (_D, _T, _PC, 32, 16, 1, False),
         "2-D lane grid"),
        ("a partial block is refused", (_T, _D, _PR, 64, 4, 2, True), "partial block")):
    try:
        _bn(*_args)
        check(_why, False, "derived")
    except ValueError as _e:
        check(_why, _needle in str(_e), str(_e))

# A REAL d_model DERIVES FAST: the search is vectorised, not a per-element Python walk.
import time as _time                                                   # noqa: E402
_t0 = _time.time()
_bn(32, 4096, (4096 // 32 + 1) * 64 + 8, 16, 8, 1, True)
check("[32, 4096] on (16, 8, 16) derives in under 10 s", _time.time() - _t0 < 10.0,
      _time.time() - _t0)

# THE ARGS CARRY IT: a non-(16,4,16) mesh now builds, and the refusals surface there too.
_a = _RNA(0, 0, _T, _D, output_layout="A", out_i8=True, mesh=(16, 8, 16))
_f = _a.get_c_field_assignments({})
check("args on (16,8,16) emit a descriptor", int(_f["blk_rd_lane"]) > 0
      and int(_f["blk_reps"]) >= 1 and int(_f["blk_pitch"]) == _PR, _f)
try:
    _RNA(0, 0, _T, _D, output_layout="A", mesh=(32, 2, 32))
    check("args refuse (32,2,32)", False, "accepted")
except ValueError:
    check("args refuse (32,2,32)", True)

# ------------------------------------------------ the A/B transpose identity
print("\nthe A/B identity, and the mesh it rests on")
import numpy as _np                                                      # noqa: E402
from libs.comm.nest import index_map as _imap                            # noqa: E402

_SQ, _NSQ = (16, 4, 16), (16, 4, 8)


def _lay(layout, X, mesh):
    r, c = X.shape
    m = _imap(layout, r, c, mesh)
    b = _np.empty(r * c, dtype=X.dtype)
    b[m.reshape(-1)] = X.reshape(-1)
    return b


# A(X) == B(X^T) IS THE WHOLE BASIS for planning a B pair. It is a property of the mesh,
# so it is checked on both a square array and a non-square one.
for _mesh, _want in ((_SQ, True), (_NSQ, False)):
    _X = _np.arange(32 * 128, dtype=_np.int32).reshape(32, 128)
    _got = _np.array_equal(_lay("A", _X, _mesh), _lay("B", _X.T.copy(), _mesh))
    check(f"A(X) == B(X^T) is {_want} on mesh {_mesh}", _got == _want, (_mesh, _got))

# ON A SQUARE ARRAY plan decomposes a B pair; on a non-square one it must refuse, because
# the identity it would rest on is false there.
_rm = PortSpec(Layout.ROW_MAJOR, DType.F16, (32, 128), mem_level=MemLevel.L1, cluster=0)
_b = PortSpec(Layout.B, DType.F16, (32, 128), mem_level=MemLevel.L1, cluster=0)
check("row_major -> B decomposes on a square array",
      [x.kind for x in _staging.plan(_rm, _b, mesh=_SQ, elem_bytes=2)]
      == ["transpose", "relayout"])
check("B -> row_major decomposes the other way round",
      [x.kind for x in _staging.plan(_b, _rm, mesh=_SQ, elem_bytes=2)]
      == ["relayout", "transpose"])
try:
    _staging.plan(_rm, _b, mesh=_NSQ, elem_bytes=2)
    check("row_major -> B is refused on a non-square array", False, "it planned")
except ValueError:
    check("row_major -> B is refused on a non-square array", True)

# THE DECOMPOSITION IS BYTE-EXACT, which no step count would show: transposing and then
# running the A nest on the swapped shape has to land on B's own index map.
for _R, _C in ((32, 128), (64, 64), (16, 64)):
    _X = _np.arange(_R * _C, dtype=_np.int32).reshape(_R, _C)
    check(f"[{_R},{_C}] A(X^T) is byte-exact B(X)",
          _np.array_equal(_lay("A", _X.T.copy(), _SQ), _lay("B", _X, _SQ)))

# A BLOCKED-TO-BLOCKED PAIR stays refused: it has no row_major side to pivot through.
_a = PortSpec(Layout.A, DType.F16, (32, 128), mem_level=MemLevel.L1, cluster=0)
try:
    _staging.plan(_a, _b, mesh=_SQ, elem_bytes=2)
    check("A -> B is still refused", False, "it planned")
except ValueError:
    check("A -> B is still refused", True)

# ONE CONVERTER KERNEL, dispatching on the pair.
from bingo_kernel_args import (SnaxBingoKernelXdmaLayoutConvertArgs as _LC,  # noqa: E402
                               xdma_conv_args as _fam)
check("every direction is the same symbol",
      len({_fam(f)(0, 0x100, 32, 128, _SQ, 2).KERNEL_NAME
           for f in ("xdma_row_major_to_a", "xdma_d_to_row_major")}) == 1)
# THE NEST IS DERIVED ON THE HOST, so an expressible conversion becomes an xdma_6d with
# strides that _verify_nest has already walked against both index maps -- and one that is
# not expressible falls back to the element loop, by name rather than silently.
check("an expressible conversion becomes a derived nest",
      _LC(0, 0x100, 32, 128, "row_major", "A", _SQ, 2).KERNEL_NAME
      == "__snax_bingo_kernel_xdma_6d")
check("...carrying strides, not a shape",
      "temporal_strides_src[0]" in
      _LC(0, 0x100, 32, 128, "row_major", "A", _SQ, 2).get_c_field_assignments({}))
check("an int8 A conversion has no nest and says so",
      _LC(0, 0x100, 32, 128, "row_major", "A", _SQ, 1).KERNEL_NAME
      == "__snax_bingo_kernel_xdma_layout_convert")
for _why, _kw in (("a pair with no row_major side", ("A", "B")),
                  ("a col_major pair", ("row_major", "col_major"))):
    try:
        _LC(0, 0x100, 32, 128, _kw[0], _kw[1], _SQ, 2)
        check(f"the converter refuses {_why}", False, "it constructed")
    except ValueError:
        check(f"the converter refuses {_why}", True)
try:
    _LC(0, 0x100, 30, 128, "row_major", "A", _SQ, 2)
    check("the converter refuses a shape that does not tile", False, "it constructed")
except ValueError:
    check("the converter refuses a shape that does not tile", True)

# ======================================================================================
# PLACEMENT: a kernel reads its own TCDM and nothing else
# ======================================================================================
# A remote handle is the dangerous case, because it does not fault. The transfer completes
# without writing and the destination keeps whatever it held -- X, on a buffer nothing else
# touched -- so it surfaces as a wrong answer or an assertion somewhere unrelated.

from libs.block import Gather, Quantize, RMSNorm, Scatter                # noqa: E402

_pc = new_ctx()
_l1_cl0 = _pc.at(0).l1("on_cl0", 32 * 128 * 2)
_l1_cl1 = _pc.at(1).l1("on_cl1", 32 * 128 * 2)
_l3_any = _pc.l3("in_l3", 32 * 128 * 2)
_want_cl0 = PortSpec(Layout.ROW_MAJOR, DType.F16, (32, 128), mem_level=MemLevel.L1,
                     cluster=0)

_want_cl1 = replace(_want_cl0, cluster=1)
_in_l3 = PortSpec(Layout.ROW_MAJOR, DType.F16, (32, 128), mem_level=MemLevel.L3)
check("an L1 port states the cluster it is in", Port(_want_cl1, _l1_cl1, ()).cluster == 1)
check("...and a main-memory one has none to state",
      Port(_in_l3, _l3_any, ()).cluster is None)
refuses("a spec that names the wrong cluster is refused at binding",
        lambda: Port(_want_cl0, _l1_cl1, ()), "allocated on cluster 1")
check("a producer on another cluster cannot feed this port",
      "cluster 1" in (check_contract(Port(_want_cl1, _l1_cl1, ()), _want_cl0,
                                     where="t") or ""))
check("...and the local one can",
      check_contract(Port(_want_cl0, _l1_cl0, ()), _want_cl0, where="t") is None)
# A BLOCK THAT READS FROM MORE THAN ONE LEVEL IS A FAMILY, not a port with a hole: each
# realisation states one level, and only the one the producer matches survives.
_norm_l1, _norm_l3 = [RMSNorm(rows=32, cols=128, cluster=0, in_layout=Layout.ROW_MAJOR,
                              out_layout=Layout.ROW_MAJOR, out_dtype=DType.F16,
                              in_level=lv).inputs["x"]
                      for lv in (MemLevel.L1, MemLevel.L3)]
check("the L3 realisation takes main memory",
      check_contract(Port(_in_l3, _l3_any, ()), _norm_l3, where="t") is None)
check("...and the L1 one does not",
      check_contract(Port(_in_l3, _l3_any, ()), _norm_l1, where="t") is not None)
check("...nor does the L1 one take a neighbour's TCDM",
      check_contract(Port(_want_cl1, _l1_cl1, ()), _norm_l1, where="t") is not None)
check("RMSNorm declares the cluster it reads on",
      RMSNorm(rows=32, cols=128, cluster=2, in_layout=Layout.ROW_MAJOR,
              out_layout=Layout.ROW_MAJOR, out_dtype=DType.F16,
              in_level=MemLevel.L1).inputs["x"].cluster == 2)
check("Quantize's output names its cluster too",
      Quantize(rows=32, cols=128, inv_scale_f32bits=0, layout=Layout.ROW_MAJOR,
               cluster=3).outputs["y"].cluster == 3)

# ======================================================================================
# SCATTER / GATHER: the two ends of a row split
# ======================================================================================

_CL = (0, 1, 2, 3)
_sc = Scatter(rows=32, cols=128, clusters=_CL, src_level=MemLevel.L3)
_ga = Gather(rows=32, cols=128, clusters=_CL)
check("Scatter names one output per cluster",
      sorted(_sc.outputs) == ["y_c0", "y_c1", "y_c2", "y_c3"])
check("...each an 8-row slice in that cluster's L1",
      all(_sc.outputs[f"y_c{c}"].shape == (8, 128)
          and _sc.outputs[f"y_c{c}"].cluster == c for c in _CL))
check("Gather names one input per cluster, and one output on the root",
      sorted(_ga.inputs) == ["x_c0", "x_c1", "x_c2", "x_c3"]
      and _ga.outputs["y"].shape == (32, 128) and _ga.outputs["y"].cluster == 0)
check("a Gather onto cluster 2 says so",
      Gather(rows=32, cols=128, clusters=_CL, root=2).outputs["y"].cluster == 2)

# WHERE THE TENSOR STARTS IS THE KNOB: two routes in, and they are different graphs.
_open = Scatter(rows=32, cols=128, clusters=_CL)
check("an unpinned Scatter offers both routes in", len(_open.variants()) == 2)
check("...the L3 one loads per cluster, the L1 one multicasts",
      _open.respec(src_level=MemLevel.L3).idma_passes() == 4
      and _open.respec(src_level=MemLevel.L1).xdma_passes() == 3)
check("a pinned Scatter offers exactly one", len(_sc.variants()) == 1)
refuses("a template Scatter refuses to build",
        lambda: _open.build(new_ctx(), {}), "template")

refuses("a ragged split is refused",
        lambda: Scatter(rows=30, cols=128, clusters=_CL), "does not divide")
refuses("...and so is a repeated cluster",
        lambda: Gather(rows=32, cols=128, clusters=(0, 1, 1)), "repeats")
refuses("a blocked layout cannot be row-sliced",
        lambda: Scatter(rows=32, cols=128, clusters=_CL, layout=Layout.A), "row_major")
refuses("the root has to be one of the clusters",
        lambda: Gather(rows=32, cols=128, clusters=(0, 1), root=3), "not among")

# The whole split, assembled: scatter -> one norm per cluster -> gather.
_sp = Pipeline(new_ctx(), verbose=False)
_src = Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (32, 128),
                          mem_level=MemLevel.L3),
            _sp.ctx.l3("split_x", 32 * 128 * 2), ())
_st_sc = _sp.add(Scatter(rows=32, cols=128, clusters=_CL), name="sc", bind={"x": _src})
_st_n = [_sp.add(RMSNorm(rows=8, cols=128, cluster=c, in_layout=Layout.ROW_MAJOR,
                         out_layout=Layout.ROW_MAJOR, out_dtype=DType.F16),
                 name=f"n{c}", bind={"x": _st_sc.out(f"y_c{c}")}) for c in _CL]
_st_ga = _sp.add(Gather(rows=32, cols=128, clusters=_CL), name="ga",
                 bind={f"x_c{c}": _st_n[c].out("y") for c in _CL})
_sp.run()
_g = _sp.ctx.dfg
check("a row split builds", _st_ga.out("y").port.handle is not None)
check("...with the L3 route chosen, so every cluster loads its own slice",
      _st_sc.chosen.block.src_level == MemLevel.L3)
check("...one norm on each cluster",
      sorted(nd.assigned_cluster_id for nd in _sp.ctx.dfg.nodes
             if "Rmsnorm" in nd.node_name) == [0, 1, 2, 3])
check("...and every slice is ordered between its load and its push",
      all(nx.has_path(_g, _st_sc.result.outputs[f"y_c{c}"].ends[-1],
                      _st_ga.result.inputs[f"x_c{c}"].ends[-1]) for c in _CL))

# A norm placed on the wrong cluster is a refusal, not a wrong answer.
_bad = Pipeline(new_ctx(), verbose=False)
_bsc = _bad.add(Scatter(rows=32, cols=128, clusters=(0, 1), src_level=MemLevel.L3),
                name="sc", bind={"x": Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (32, 128),
                          mem_level=MemLevel.L3),
                                           _bad.ctx.l3("bad_x", 32 * 128 * 2), ())})
_bad.add(RMSNorm(rows=16, cols=128, cluster=0, in_layout=Layout.ROW_MAJOR,
                 out_layout=Layout.ROW_MAJOR, out_dtype=DType.F16),
         name="n", bind={"x": _bsc.out("y_c1")})
refuses("a block bound to another cluster's slice is refused", _bad.run, "cluster")


# ======================================================================================
print("\nblock pictures: what the Pipeline records, and how a boundary edge is classed")
# The records are what every picture is drawn from, so they are what is checked. A node
# credited to the wrong block, or a gating edge counted as data, draws a confident picture
# of the wrong graph -- which is worse than no picture for someone hunting a missing edge.
from bingo_block_viz import _Scene, _World, render_blocks  # noqa: E402

_vctx = new_ctx()
_vp = Pipeline(_vctx, verbose=False)
_va = _vp.add(Producer(), name="a")
_vb = _vp.add(Consumer(), name="b", bind={"x": _va.out("o")})
_vp.raw(lambda: _vctx.node("peek", _vctx.dm, "__snax_k", Args(src=_vb.out("y").port.handle),
                           list(_vb.out("y").port.ends)), "peek")
_vp.run()
_recs = _vctx.dfg.block_records
_names = [[n.node_name for n in r.nodes] for r in _recs]
check("one record per stage and raw thunk, in build order",
      [(r.name, r.kind) for r in _recs] == [("a", "Producer"), ("b", "Consumer"),
                                            ("peek", "raw")])
check("...each holding exactly the nodes its build created, in creation order",
      _names == [["a_ld_cl0", "a_wr_cl0"], ["b_wld_cl0", "b_rd_cl0", "b_cmp_cl0"],
                 ["peek_cl0"]], str(_names))
check("...and every node of the graph is credited to exactly one of them",
      sorted(n for ns in _names for n in ns) == sorted(n.node_name for n in _vctx.dfg.nodes))
_pin = _recs[1].inputs[0]
check("an input names its producing port and that port's end nodes",
      _pin.peer == "a.o" and _pin.peer_ends == list(_va.out("o").port.ends)
      and _pin.cluster == 0)
_by = {n.node_name: n for n in _vctx.dfg.nodes}
_sc = _Scene(_recs[1], _World(_vctx.dfg, _recs))
check("link()'s gate on the weight load is COUNTED, not drawn as data",
      _sc.gated[_by["b_wld_cl0"]] == 1 and _sc.n_gate == 1
      and _sc.edge_class[(_by["a_wr_cl0"], _by["b_wld_cl0"])] == "gate x")
check("...while the RAW edge belongs to the input's card, as a declared reader",
      _sc.in_cards[0]["to"] == [_by["b_rd_cl0"]] and _sc.in_cards[0]["also"] == []
      and _sc.edge_class[(_by["a_wr_cl0"], _by["b_rd_cl0"])] == "port x")
check("...and the raw thunk that reads the output is a ghost, not a consumer",
      [xs for _l, xs, _i in _sc.ghost_out] == [[_by["peek_cl0"]]]
      and _sc.out_cards[0]["users"] == [])
check("depth inside the block is the longest path, not creation order",
      [_sc.depth[_by[n]] for n in ("b_wld_cl0", "b_rd_cl0", "b_cmp_cl0")] == [0, 0, 1])

try:
    import matplotlib  # noqa: F401
    _have_mpl = True
except ImportError:
    _have_mpl = False
if _have_mpl:
    import tempfile
    with tempfile.TemporaryDirectory() as _td:
        _bd = _os.path.join(_td, "block_dfg")
        _os.makedirs(_bd)
        open(_os.path.join(_bd, "99_renamed__Old.png"), "w").close()
        _before = (_vctx.dfg.number_of_nodes(), _vctx.dfg.number_of_edges())
        _res = render_blocks(_vctx.dfg, _td, "test")
        check("render writes a picture and a listing per stage, plus an index",
              sorted(_os.listdir(_bd)) == ["00_a__Producer.png", "00_a__Producer.txt",
                                           "01_b__Consumer.png", "01_b__Consumer.txt",
                                           "02_peek__raw.png", "02_peek__raw.txt",
                                           "README.md"], str(sorted(_os.listdir(_bd))))
        check("...clears a picture left behind by a stage that no longer exists",
              not _os.path.exists(_os.path.join(_bd, "99_renamed__Old.png")))
        check("...and leaves the graph it drew untouched",
              (_vctx.dfg.number_of_nodes(), _vctx.dfg.number_of_edges()) == _before)
else:
    print("  SKIP  rendering (no matplotlib)")


if FAILED:
    print(f"\n{len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("\nall libs tests passed")
