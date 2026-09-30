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
from libs.blocks import record_spec

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

from libs.blocks import Reshape as _Reshape_ctor                          # noqa: E402

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

# ------------------------------------------------ the memory chiplet's HBM
# A fixed address the staging helper placed in the HBM RECORDS that, so it is checked
# like an allocation: a spec calling it main memory is refused, and every offset into it
# stays in the HBM.
from libs.comm.ports import at_offset as _at, level_of as _level_of      # noqa: E402
from libs.comm.transfer import plan as _plan                              # noqa: E402
_hbm = _Fixed(0x1001_0000_0000, mem_level="HBM")
check("an HBM address carries its level", _level_of(_hbm) == "HBM", _level_of(_hbm))
check("...and a spec saying HBM binds to it",
      Port(PortSpec("B", "i8", (2048, 64), mem_level="HBM"), _hbm, ()).spec.mem_level
      == MemLevel.HBM)
refuses("...while a spec saying L3 is refused",
        lambda: Port(PortSpec("B", "i8", (2048, 64), mem_level="L3"), _hbm, ()), "is in HBM")
_hbm2 = _at(_hbm, 128 * 1024)
check("an offset into the HBM stays in the HBM",
      _hbm2.address == 0x1001_0002_0000 and _level_of(_hbm2) == "HBM", repr(_hbm2))
_lvl = PortSpec("B", "i8", (2048, 64), mem_level="HBM")
check("HBM -> L1 is a block's own load: nothing to hoist",
      _plan(_lvl, PortSpec("B", "i8", (2048, 64), mem_level="L1", cluster=0)) == [])
refuses("HBM -> L3 is refused: a streamed weight is never hoisted",
        lambda: _plan(_lvl, PortSpec("B", "i8", (2048, 64), mem_level="L3")), "HBM->L1")

from libs.comm import add_sim_paths as _add_sim_paths                     # noqa: E402
_add_sim_paths("common")
import tempfile as _tempfile                                              # noqa: E402
import numpy as _np                                                       # noqa: E402
from bingo_data_staging import DataStaging as _DS                         # noqa: E402
_plat = {"num_mem_chips": 1, "mem_chip_loc_x": 1, "mem_chip_loc_y": 0,
         "hbm_base": 0x1_0000_0000, "hbm_size": 0x4_0000_0000}
refuses("put_hbm on a platform without an HBM is refused",
        lambda: _DS(dict(_plat, hbm_size=0)).put_hbm("w", _np.zeros(64, _np.int8)),
        "no HBM")
_st = _DS(_plat, on_host=True)
_h0 = _st.put_hbm("w0", _np.arange(100, dtype=_np.int8))
_h1 = _st.put_hbm("w1", _np.ones(64, dtype=_np.int8))
check("the HBM base is the memory chiplet's, (1,0) -> chip 0x10",
      _h0.address == 0x1001_0000_0000 and _h0.mem_level == "HBM", repr(_h0))
check("...and every array starts on the 4 KiB interleave",
      _h1.address == 0x1001_0000_1000, repr(_h1))
check("on_host keeps put() in the host image, memory chiplet or not",
      type(_st.put("x", "int8_t", _np.zeros(64, _np.int8))).__name__ == "BingoMemSymbol")
with _tempfile.TemporaryDirectory() as _td:
    _st.emit(_os.path.join(_td, "d.h"), _td)
    _man = open(_os.path.join(_td, "build", "hbm", "manifest.txt")).read()
    _img = open(_os.path.join(_td, "build", "hbm", "hbm_image.bin"), "rb").read()
    check("emit writes the manifest the testharness loads",
          "0x0  hbm_image.bin" in _man, _man)
    check("...and the image holds each array at its offset",
          _img[:100] == bytes(range(100)) and _img[0x1000:0x1040] == b"\x01" * 64)

# Two memory chiplets, (1,0) and (3,0): each holds its own arrays.
_plat2 = dict(_plat, num_mem_chips=2, mem_chips=[
    {"id": 0x10, "hbm_base": 0x1_0000_0000, "hbm_size": 0x4_0000_0000},
    {"id": 0x30, "hbm_base": 0x1_0000_0000, "hbm_size": 0x4_0000_0000}])
_st2 = _DS(_plat2)
_a = _st2.put("a", "int8_t", _np.full(64, 7, _np.int8))
_b = _st2.put("b", "int8_t", _np.full(64, 9, _np.int8), mem_chip=(3, 0))
_w = _st2.put_hbm("w_far", _np.full(32, 5, _np.int8), mem_chip=0x30)
_v = _st2.put_hbm("w_home", _np.full(32, 6, _np.int8))
check("put(mem_chip=) addresses that memory chiplet, the rest the home one",
      _a.address == 0x1000_8000_0000 and _b.address == 0x3000_8000_0000,
      (hex(_a.address), hex(_b.address)))
check("...and so does put_hbm, by (x, y) or chip id",
      _w.address == 0x3001_0000_0000 and _v.address == 0x1001_0000_0000,
      (hex(_w.address), hex(_v.address)))
refuses("a memory chiplet the platform lacks is refused",
        lambda: _st2.put("c", "int8_t", _np.zeros(8, _np.int8), mem_chip=(2, 0)),
        "no memory chiplet at (2, 0)")
with _tempfile.TemporaryDirectory() as _td:
    _os.makedirs(_os.path.join(_td, "build"))
    open(_os.path.join(_td, "build", "mempool_chip_5_0.bin"), "wb").close()
    _st2.emit(_os.path.join(_td, "d.h"), _td)
    _bd = _os.path.join(_td, "build")
    _man = open(_os.path.join(_bd, "hbm", "manifest.txt")).read()
    check("each chip gets its own SRAM image; a stale one is removed",
          open(_os.path.join(_bd, "mempool.bin"), "rb").read() == b"\x07" * 64
          and open(_os.path.join(_bd, "mempool_chip_3_0.bin"), "rb").read() == b"\x09" * 64
          and not _os.path.exists(_os.path.join(_bd, "mempool_chip_5_0.bin")),
          sorted(_os.listdir(_bd)))
    check("with two chips every HBM image is tagged chip=<id>",
          "0x0  hbm_image.bin  chip=0x10" in _man
          and "0x0  hbm_image_chip_3_0.bin  chip=0x30" in _man, _man)

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
from libs.blocks import Reshape as _Reshape                               # noqa: E402
from libs.blocks.simd.norm import RMSNorm as _RMSNorm                     # noqa: E402

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
from libs.blocks import Linear as _Linear                                 # noqa: E402
from libs.blocks import Quantize as _Quantize                             # noqa: E402


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

from libs.blocks import Gather, Quantize, RMSNorm, Scatter                # noqa: E402

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


print("\nFlashAttention: the D-port shift and the P8 pitch")
import numpy as _np                                                         # noqa: E402
from libs.blocks import flash_attention as _fa                               # noqa: E402
from sim_golden_models import int32_to_fp16_golden as _i2h                  # noqa: E402
from bingo_kernel_args import (SnaxBingoKernelGemmFaQkArgs as _Qk,          # noqa: E402
                               SnaxBingoKernelGemmFaPvArgs as _Pv,
                               SnaxBingoKernelSimdFaSoftmaxArgs as _Sm)

_rs = _np.random.RandomState(7)
_xs = _np.concatenate([_rs.randint(-2**31 + 1, 2**31 - 1, 400, dtype=_np.int64),
                       _rs.randint(-3_000_000, 3_000_000, 400, dtype=_np.int64),
                       [0, 1, -1, 2047, 2049, 3001, 65504, 65519, 65520, 2_064_512]])
check("the block's converter model is the RTL's RNE(S * 2^-k), bit for bit, k = 0..14",
      all(int(_fa.d_port_f16(_xs, k).view(_np.uint16)[i]) == _i2h(int(x), k)
          for k in range(15) for i, x in enumerate(_xs)))
check("full-range INT8 at d = 128 needs k = 6: 128^2 * 128 is Inf at 5 and finite at 6",
      _fa.min_score_shift(128) == 6
      and not _fa.converted_score_finite(128 * 128 * 128, 5)
      and _fa.converted_score_finite(128 * 128 * 128, 6))
_cfg = _fa.FaCfg.from_shape(bc=64, br=32, dhead=128, nkv=4, clusters=(0,), decomp="kvsplit",
                            score_scale=0.003)
check("unset, the shift is the smallest safe one, and the softmax gets a' = a * 2^k",
      _cfg.dshift == 6 and _cfg.exp_scale == float(_np.float32(0.003) * _np.float32(64.0)))
check("...and synthetic operands go full range, where k = 0 had to shrink them to 5 bits",
      _cfg.qshift == 0 and replace(_cfg, score_shift=0).qshift == 3)
refuses("a shift past the converter's 14 is refused",
        lambda: replace(_cfg, score_shift=15).validate(), "0..14")

# THE SHIFT MOVES NO RESULT. On data that fits FP16 unshifted, k = 6 must give the same P
# (hence l and O) and m exactly 2^-6 times the unshifted one: a power of two only moves the
# exponent, and nothing the softmax does to the converted score is subnormal.
_c0 = replace(_cfg, score_shift=0, nkv=2)
_c6 = replace(_cfg, score_shift=6, nkv=2)
_q = (_rs.randint(-128, 128, 32 * 128) >> 3).astype(_np.int8)
_ks = [(_rs.randint(-128, 128, 64 * 128) >> 3).astype(_np.int8) for _ in range(2)]
_vs = [(_rs.randint(-128, 128, 64 * 128) >> 3).astype(_np.int8) for _ in range(2)]
_m0, _l0, _o0 = _fa.shard_golden(_c0, _q, _ks, _vs)
_m6, _l6, _o6 = _fa.shard_golden(_c6, _q, _ks, _vs)
check("the shift moves no result: m scales by exactly 2^-6, l and O are bit-identical",
      _np.array_equal(_m6.astype(_np.float64), _m0.astype(_np.float64) / 64.0)
      and _np.array_equal(_l6.view(_np.uint16), _l0.view(_np.uint16))
      and _np.array_equal(_o6, _o0))

# The P buffer on the pitch grid: P8 beats, then the row sum, then corr, one pitch apart.
_bc, _br = 512, 32
check("dense P8 is the old buffer: bc*br + row sum + corr",
      _Sm.p8_bytes(_bc, _br, _Sm.EXACT) == _bc * _br + 128
      and _Sm.corr_offset(_bc, _br) == _bc * _br + 64)
check("at 160 B the row sum is one pitch after the last P8 beat, corr one after that",
      _Sm.rowsum_offset(_bc, _br, 160) == 256 * 160
      and _Sm.corr_offset(_bc, _br, 160) == 257 * 160
      and _Sm.p8_bytes(_bc, _br, _Sm.EXACT, 160) == 257 * 160 + 64)
_len = _Sm.p8_bytes(_bc, _br, _Sm.EXACT, 160)
_b0 = {o + i for o in range(0, _len, 160) for i in range(64)}
_b1 = {_fa.P8_NEST + b for b in _b0}
check("two nested P buffers share no byte and fit the pair's region",
      not (_b0 & _b1) and max(_b1) < _fa.P8_NEST + _len)
refuses("nesting needs a pitch with room for the other buffer's block",
        lambda: replace(_cfg, p8_pitch=128).validate(), "p8_nest")

refuses("QK takes no B pitch (its B, Q^T, is dense)",
        lambda: _Qk(0, 0, 0, 0, 1, 1, 1, b_pitch=160), "dense")
refuses("PV takes no shift (it writes INT32)",
        lambda: _Pv(0, 0, 0, 0, 1, 1, 1, d_shift=6), "int32")
refuses("a QK shift past 14 is refused", lambda: _Qk(0, 0, 0, 0, 1, 1, 1, d_shift=15), "0..14")
refuses("a P8 pitch below one beat is refused",
        lambda: _Sm(0, 0, 0, bc=64, dhead=128, tile_idx=0, p8_pitch=32), "p8_pitch")


print("\nRMSNorm: the scratch the host passes in instead of the kernel allocating it")
from bingo_kernel_args import SnaxBingoKernelSimdRmsnormArgs as _Rn   # noqa: E402
_ra = _Rn(0x1000, 0x2000, rows=32, cols=128, input_layout="row_major", output_layout="A",
          out_i8=True, mesh=(16, 4, 16))
check("row_major -> A at [32, 128]: the 21,056 B simd.h's pool is sized for",
      _ra.scratch_bytes() == 21056)
check("...and 4x the rows need 4x the bytes, past the pool: why [128, 128] malloc'd",
      _Rn(0x1000, 0x2000, rows=128, cols=128, input_layout="row_major", output_layout="A",
          out_i8=True, mesh=(16, 4, 16)).scratch_bytes() > 24576)
_rc = _Rn(0x1040, 0x3000, rows=32, cols=128, input_layout="col_major",
          output_layout="col_major", seed_addr=0x1000)
check("col_major: one beat, with the slack to align it", _rc.scratch_bytes() == 127)
check("no scratch passed: scratch_bytes 0, so the device keeps its own pool",
      _rc.get_c_field_assignments({})["scratch_bytes"] == "0")
_rc.scratch_addr = 0x4000
check("a scratch passed: its size goes with it, for the device to check",
      _rc.get_c_field_assignments({})["scratch_bytes"] == "127")


print("\nLinear: the one-token GEMV, and weights streamed through double buffers")
from bingo_kernel_args import SnaxBingoKernelGemvArgs as _Gv               # noqa: E402
from libs.blocks import (Linear as _Lin, QuantizeARow as _QAR,                # noqa: E402
                        RMSNormRow as _RNR, ScaleCols as _SC)
from libs.blocks.linear import DEFAULT_W_CHUNK_BYTES as _WCB                  # noqa: E402

check("the D-port shift for each of the layer's depths",
      [_Gv.shift_for_depth(k) for k in (128, 512, 576, 1408, 2048, 2816)]
      == [5, 7, 8, 9, 9, 10])


def _gemv_chain(d_in=2048, d_out=768, w_level=MemLevel.HBM, **lin):
    """RMSNormRow -> QuantizeARow -> Linear(gemv) -> ScaleCols, the decode projection."""
    c = new_ctx()
    x = Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (1, d_in), mem_level=MemLevel.L3),
             _Sym("x"), ())
    w4 = lin.get("w_bits", 8) == 4
    w = Port(PortSpec(Layout.B_W4 if w4 else Layout.B, DType.I4 if w4 else DType.I8,
                      (d_in, d_out), mem_level=w_level),
             _Fixed(0x1001_0000_0000, mem_level="HBM") if w_level == MemLevel.HBM
             else _Sym("w"), ())
    s = Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (1, d_out), mem_level=MemLevel.L3),
             _Sym("s"), ())
    p = Pipeline(c, verbose=False, gate_sources=False)
    n = p.add(_RNR(cols=d_in, cluster=1), "norm", x=x)
    q = p.add(_QAR(cols=d_in, cluster=1, inv_scale_f32bits=0x42000000), "xa", x=n.out())
    y = p.add(_Lin(tokens=1, d_in=d_in, d_out=d_out, mesh=(16, 4, 16), cluster=1,
                   gemv=True, d_shift=9, **lin), "wq", x=q.out(), w=w)
    p.add(_SC(cols=d_out, cluster=1), "deq", x=y.out(), s=s)
    p.run()
    return c.dfg, p


_g, _p = _gemv_chain()
_lin = _p.stages[2].result
_ld, _ts = _lin.extra["loads"], _lin.extra["tasks"]
check("768 columns at K = 2,048 stream as twelve 64-column chunks",
      _lin.extra["chunks"] == [(64 * i, 64) for i in range(12)], _lin.extra["chunks"])
check("...through two buffers, one load and one GEMV per chunk",
      len(_lin.extra["w_bufs"]) == 2 and len(_ld) == 12 and len(_ts) == 12
      and all(t.kernel_name == "__snax_bingo_kernel_gemv" for t in _ts))
check("each GEMV waits for its own chunk's load (RAW)",
      all(_g.has_edge(_ld[i], _ts[i]) for i in range(12)))
check("...and for the GEMV before it: the last one implies them all",
      all(_g.has_edge(_ts[i - 1], _ts[i]) for i in range(1, 12)))
check("a load waits for the GEMV that last read its buffer (WAR)",
      all(_g.has_edge(_ts[i - 2], _ld[i]) for i in range(2, 12))
      and not any(_g.in_degree(_ld[i]) for i in range(2)))
check("the buffers alternate",
      [_ld[i].kernel_args.dst_addr is _lin.extra["w_bufs"][i % 2] for i in range(12)]
      == [True] * 12)
check("chunk i reads the weight 64 * 2,048 * i bytes in, still in the HBM",
      all(_ld[i].kernel_args.src_addr.address == 0x1001_0000_0000 + i * 64 * 2048
          and _ld[i].kernel_args.src_addr.mem_level == "HBM" for i in range(12)))
check("...and writes its 64 fp16 outputs 128 * i bytes into the row",
      all(_ts[i].kernel_args.output_D_addr.offset == 128 * i for i in range(12)))
check("each GEMV is K = 2,048 x N = 64 at the asked shift",
      all((t.kernel_args.kt, t.kernel_args.nb, t.kernel_args.d_shift) == (512, 4, 9)
          for t in _ts))
check("the quantised row feeds the FIRST GEMV",
      _g.has_edge(_p.stages[1].result.outputs["y"].ends[0], _ts[0]))
check("the dequantisation waits for the LAST GEMV",
      _g.has_edge(_ts[-1], _p.stages[3].result.nodes[-1]))
check("with the sources ungated the first two loads prefetch under the norm",
      all(_g.in_degree(_ld[i]) == 0 for i in range(2)))
_rn = _p.stages[0].result.nodes[-1].kernel_args
check("the norm's seed beat sits directly below its x",
      _rn.seed_addr is _rn.input_addr.base and _rn.input_addr.offset == 64)
_qa = _p.stages[1].result
check("the quantiser zeroes its 16-row operand on the xDMA first",
      _qa.nodes[0].kernel_name == "__snax_bingo_kernel_xdma_memset"
      and _qa.nodes[0].kernel_args.size == 16 * 2048
      and _g.has_edge(_qa.nodes[0], _qa.nodes[1]))
_mul = _p.stages[3].result.nodes[-1].kernel_args.get_c_field_assignments({})
check("the dequantisation is the elementwise kernel with op MUL", _mul["op"] == "0", _mul)
check("GEMV emits every field of its struct",
      set(_ts[0].kernel_args.get_c_field_assignments({})) ==
      {"input_A_addr", "input_B_addr", "output_D_addr", "kt", "nb", "groups", "a_step",
       "d_shift", "a_blk", "w4", "b_step", "d_step"})

_g3, _p3 = _gemv_chain(w_buffers=3)
_l3 = _p3.stages[2].result.extra
check("three buffers: a load waits for the GEMV three chunks back",
      len(_l3["w_bufs"]) == 3 and all(_g3.has_edge(_l3["tasks"][i - 3], _l3["loads"][i])
                                      for i in range(3, 12)))
_gs, _ps = _gemv_chain(d_out=64)
check("a weight that fits one chunk is loaded whole, one GEMV",
      [n.node_name for n in _ps.stages[2].result.nodes] ==
      ["wq_Ld_linear_w_cl1", "wq_Gemm_linear_cl1"], [n.node_name for n in
                                                     _ps.stages[2].result.nodes])
_gr, _pr = _gemv_chain(d_out=96, w_chunk_bytes=64 * 2048)
check("a width that does not divide the chunk ends in a narrower chunk",
      _pr.stages[2].result.extra["chunks"] == [(0, 64), (64, 32)])

# INT4 weights (w_bits=4): half the bytes, the same passes, the GEMV told so.
_g4, _p4 = _gemv_chain(w_bits=4)
_l4 = _p4.stages[2].result.extra
check("INT4: a 128 KiB chunk holds 128 columns at K = 2,048, so 768 stream as six",
      _l4["chunks"] == [(128 * i, 128) for i in range(6)], _l4["chunks"])
check("...chunk i reads 128 * 2,048 / 2 * i bytes in, 128 KiB each",
      all(_l4["loads"][i].kernel_args.src_addr.address == 0x1001_0000_0000 + i * 128 * 1024
          and _l4["loads"][i].kernel_args.size == 128 * 1024 for i in range(6)),
      [(l.kernel_args.src_addr.address, l.kernel_args.size) for l in _l4["loads"]])
check("...and every GEMV runs with w4 set, 128 columns (nb 8) at the same shift",
      all((t.kernel_args.w4, t.kernel_args.nb, t.kernel_args.d_shift) == (True, 8, 9)
          for t in _l4["tasks"]))
check("...while an INT8 GEMV leaves it clear", not _ts[0].kernel_args.w4)
check("...its weight port is b_w4 int4", _p4.stages[2].block.inputs["w"].layout == Layout.B_W4
      and _p4.stages[2].block.inputs["w"].dtype == DType.I4)
refuses("INT4 is the GEMV's: a GEMM with w_bits=4 is refused",
        lambda: _Lin(tokens=16, d_in=64, d_out=64, mesh=(16, 4, 16), w_bits=4), "w_bits=4")
refuses("INT4 comes in 32-column pairs: d_out = 48 is refused",
        lambda: _Lin(tokens=1, d_in=64, d_out=48, mesh=(16, 4, 16), gemv=True, w_bits=4),
        "pairs")
refuses("the weights are 8 or 4 bits", lambda: _Lin(tokens=1, d_in=64, d_out=64,
                                                    mesh=(16, 4, 16), gemv=True, w_bits=2),
        "w_bits")

# Several tokens on the one-row GEMV: one group per token over each shared chunk.
def _gemv_tokens(T=4, d_in=2048, d_out=768, w_bits=4):
    c = new_ctx()
    x = Port(PortSpec(Layout.A_ROW, DType.I8, (T, d_in), mem_level=MemLevel.L3), _Sym("x8"), ())
    w4 = w_bits == 4
    w = Port(PortSpec(Layout.B_W4 if w4 else Layout.B, DType.I4 if w4 else DType.I8,
                      (d_in, d_out), mem_level=MemLevel.HBM),
             _Fixed(0x1001_0000_0000, mem_level="HBM"), ())
    s = Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (1, d_out), mem_level=MemLevel.L3),
             _Sym("s"), ())
    p = Pipeline(c, verbose=False, gate_sources=False)
    y = p.add(_Lin(tokens=T, d_in=d_in, d_out=d_out, mesh=(16, 4, 16), cluster=1, gemv=True,
                   d_shift=9, x_layout=Layout.A_ROW, w_bits=w_bits), "wq", x=x, w=w)
    p.add(_SC(cols=d_out, cluster=1, rows=T), "deq", x=y.out(), s=s)
    p.run()
    return c.dfg, p


_gt, _pt = _gemv_tokens()
_lt = _pt.stages[0].result.extra
_ts4 = _lt["tasks"]
check("4 tokens: x is loaded whole, 4 a_rows of 4 KiB",
      _pt.stages[0].result.nodes[0].kernel_args.size == 4 * 2 * 2048)
check("...each chunk's GEMV runs 4 groups, a_row 4 KiB apart, the chunk shared (b_step 0)",
      all((t.kernel_args.groups, t.kernel_args.a_step, t.kernel_args.b_step, t.kernel_args.a_blk)
          == (4, 4096, 0, 8) for t in _ts4))
check("...and writes chunk i's 128 columns into each token's row: 256 i bytes in, rows 1,536 apart",
      all(t.kernel_args.output_D_addr.offset == 256 * i and t.kernel_args.d_step == 1536
          for i, t in enumerate(_ts4)))
_deq = [n for n in _pt.stages[1].result.nodes if n.kernel_name.endswith("stream_elementwise")]
check("the dequantisation is one task per token row, chained, against one factor load",
      len(_deq) == 4 and all(_gt.has_edge(_deq[i - 1], _deq[i]) for i in range(1, 4))
      and sum(1 for n in _pt.stages[1].result.nodes if "idma" in n.kernel_name) == 1)
refuses("several tokens need a_row (one aligned row each)",
        lambda: _Lin(tokens=4, d_in=64, d_out=64, mesh=(16, 4, 16), gemv=True), "a_row")
refuses("the GEMV's shape has one row: 16 tokens in the A layout",
        lambda: _Lin(tokens=16, d_in=128, d_out=64, mesh=(16, 4, 16), gemv=True), "one row")
refuses("the GEMV reads the (16, 4, 16) layouts",
        lambda: _Lin(tokens=1, d_in=128, d_out=64, mesh=(1, 4, 32), gemv=True), "layouts")
check("a weight may be handed over from the HBM",
      MemLevel.HBM in {v["w_level"] for v in _Lin(tokens=1, d_in=128, d_out=64,
                                                  mesh=(16, 4, 16), gemv=True).variants()})

# A multi-row GEMM streamed in chunks: the chunk's columns of every m-block are a separate
# D run, so each chunk is one task per m-block, chained.
_cg = new_ctx()
_xg = Port(PortSpec("A", "i8", (32, 2048), mem_level="L1", cluster=0),
           _cg.at(0).l1("xg", 32 * 2048), ())
_wg = Port(PortSpec("B", "i8", (2048, 256), mem_level="L3"), _Sym("wg"), ())
_rg = _Lin(tokens=32, d_in=2048, d_out=256, mesh=(16, 4, 16), cluster=0,
           x_level=MemLevel.L1, w_level=MemLevel.L3, d_shift=9).build(_cg, {"x": _xg, "w": _wg})
_tg = _rg.extra["tasks"]
check("a 2-m-block GEMM over 4 chunks is 8 gemm_full tasks, chained",
      len(_tg) == 8 and all(_cg.dfg.has_edge(_tg[i - 1], _tg[i]) for i in range(1, 8)))
check("...each one m-block of one chunk, at its D offset",
      [(t.kernel_args.M, t.kernel_args.N, t.kernel_args.output_D_addr.offset)
       for t in _tg[:2]] == [(1, 4, 0), (1, 4, 16 * 256 * 2)]
      and _tg[2].kernel_args.output_D_addr.offset == 4 * 512)
check("...and the shift rides every task", all(t.kernel_args.d_shift == 9 for t in _tg))

# NOT streamed, nothing changes: the graph an existing layer builds is the one it built.
_cn = new_ctx()
_xn = Port(PortSpec("A", "i8", (32, 128), mem_level="L1", cluster=0),
           _cn.at(0).l1("xn", 32 * 128), ())
_wn = Port(PortSpec("B", "i8", (128, 128), mem_level="L3"), _Sym("wn"), ())
_rn2 = _Lin(tokens=32, d_in=128, d_out=128, mesh=(16, 4, 16), cluster=0,
            x_level=MemLevel.L1, w_level=MemLevel.L3).build(_cn, {"x": _xn, "w": _wn})
check("a small weight: one load, one gemm_full over the whole matrix, the old names",
      [(n.node_name, getattr(n.kernel_args, "M", None)) for n in _rn2.nodes] ==
      [("Ld_linear_w_cl0", None), ("Gemm_linear_cl0", 2)],
      [(n.node_name, getattr(n.kernel_args, "M", None)) for n in _rn2.nodes])
check("...with k = 0 unless asked", _rn2.nodes[-1].kernel_args.d_shift == 0)


# ======================================================================================
# The whole-layer pieces: one stream per cluster, per-head groups, weights the router
# picks, MLA attention, the route record.
# ======================================================================================
from libs.blocks import (CacheAppend as _CA, LoadStream as _LS, MlaAttention as _MA,       # noqa: E402
                        MoeRoute as _MR, Pull as _Pull, QAssemble as _QA, SlotLoad as _SL,
                        expert_table as _etab, record_bytes as _rbytes)
import numpy as _np                                                                     # noqa: E402

_cs = new_ctx()
_ls = _LS(_cs, 0, nbytes=128 * 1024, nbuf=2)
_ps = Pipeline(_cs, verbose=False, gate_sources=True)
_xs = Port(PortSpec(Layout.A, DType.I8, (1, 2048), mem_level=MemLevel.L1, cluster=0),
           _cs.at(0).l1("xs", 16 * 2048), ())
_w1 = Port(PortSpec(Layout.B, DType.I8, (2048, 192), mem_level=MemLevel.HBM),
           _Fixed(0x1001_0000_0000, mem_level="HBM"), ())
_w2 = Port(PortSpec(Layout.B, DType.I8, (2048, 128), mem_level=MemLevel.HBM),
           _Fixed(0x1001_1000_0000, mem_level="HBM"), ())
_l1 = _ps.add(_Lin(tokens=1, d_in=2048, d_out=192, mesh=(16, 4, 16), cluster=0, gemv=True,
                   stream=_ls), "p1", x=_xs, w=_w1)
_l2 = _ps.add(_Lin(tokens=1, d_in=2048, d_out=128, mesh=(16, 4, 16), cluster=0, gemv=True,
                   stream=_ls), "p2", x=_xs, w=_w2)
_ps.run()
_e1, _e2 = _l1.result.extra, _l2.result.extra
check("a stream: two projections, 3 + 2 chunks, all through the stream's two slabs",
      len(_e1["loads"]) == 3 and len(_e2["loads"]) == 2 and
      [ld.kernel_args.dst_addr for ld in _e1["loads"] + _e2["loads"]] ==
      [_ls.slabs[i % 2] for i in range(5)])
check("...the next projection's first load waits for the task that last read ITS slab",
      _cs.dfg.has_edge(_e1["tasks"][1], _e2["loads"][0])
      and _cs.dfg.has_edge(_e1["tasks"][2], _e2["loads"][1]))
check("...and a stream reports no sources, so gating never holds it behind a producer",
      _l1.result.sources == [] and _l2.result.sources == []
      and _cs.dfg.in_degree(_e1["loads"][0]) == 0)
refuses("a chunk larger than the stream's slab is refused",
        lambda: _LS(_cs, 1, nbytes=64 * 1024).take(65 * 1024), "slab")

# groups: W_UK for 4 heads, 128 x 512 each, two heads a chunk
_cg2 = new_ctx()
_lsg = _LS(_cg2, 0)
_xg2 = Port(PortSpec(Layout.A, DType.I8, (1, 4 * 128), mem_level=MemLevel.L1, cluster=0),
            _cg2.at(0).l1("qn", 4 * 16 * 128), ())
_wg2 = Port(PortSpec(Layout.B, DType.I8, (128, 4 * 512), mem_level=MemLevel.HBM),
            _Fixed(0x1001_0000_0000, mem_level="HBM"), ())
_rg2 = _Lin(tokens=1, d_in=128, d_out=512, mesh=(16, 4, 16), cluster=0, gemv=True, groups=4,
            stream=_lsg, x_level=MemLevel.L1, w_level=MemLevel.HBM, d_shift=5).build(
                _cg2, {"x": _xg2, "w": _wg2})
_tk = [t.kernel_args for t in _rg2.extra["tasks"]]
check("groups: 4 heads of 64 KiB stream as 2 chunks of 2 heads",
      _rg2.extra["chunks"] == [(0, 1024), (1024, 1024)])
check("...each GEMV 2 groups, a_step one 16-row operand, A and D advanced by 2 heads",
      [(k.groups, k.a_step, k.nb, getattr(k.input_A_addr, "offset", 0), k.output_D_addr.offset)
       for k in _tk] == [(2, 2048, 32, 0, 0), (2, 2048, 32, 4096, 2048)])

# a slot's weights: the loads read the record, as a CHAIN after it
_cr = new_ctx()
_lsr = _LS(_cr, 1)
_rec = Port(record_spec(6, 1), _cr.at(1).l1("rec", 768), ())
_rp = _cr.node("RecPull", _cr.dm, "__snax_k", Args(dst=_rec.handle), cluster=1)
_rec = Port(record_spec(6, 1), _rec.handle, (_rp,))
_xr = Port(PortSpec(Layout.A, DType.I8, (1, 2048), mem_level=MemLevel.L1, cluster=1),
           _cr.at(1).l1("ha", 16 * 2048), ())
_rs = _Lin(tokens=1, d_in=2048, d_out=256, mesh=(16, 4, 16), cluster=1, gemv=True,
           stream=_lsr, w_slot=(3, "gu"), x_level=MemLevel.L1).build(_cr, {"x": _xr, "rec": _rec})
_lr = _rs.extra["loads"]
check("w_slot: every load is idma_copy_slot from slot 3's gate|up, at its chunk's offset",
      all(ld.kernel_name == "__snax_bingo_kernel_idma_copy_slot" for ld in _lr)
      and [(ld.kernel_args.slot, ld.kernel_args.field, ld.kernel_args.offset) for ld in _lr]
      == [(3, 0, 0), (3, 0, 64 * 2048), (3, 0, 128 * 2048), (3, 0, 192 * 2048)])
check("...the first waits for the record, the rest in a chain (no fan-out from it)",
      _cr.dfg.has_edge(_rp, _lr[0]) and not any(_cr.dfg.has_edge(_rp, ld) for ld in _lr[1:])
      and all(_cr.dfg.has_edge(_lr[i - 1], _lr[i]) for i in range(1, 4)))

# MLA attention on one cluster, 4 tiles of 64 keys
_ca = new_ctx()
_lsa = _LS(_ca, 1)
_pa = Pipeline(_ca, verbose=False, gate_sources=True)
_qt = Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (16, 512), mem_level=MemLevel.L1, cluster=1),
           _ca.at(1).l1("qt", 16 * 1024), ())
_qp = Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (16, 64), mem_level=MemLevel.L1, cluster=1),
           _ca.at(1).l1("qp", 16 * 128), ())
_c8 = Port(PortSpec(Layout.ROW_MAJOR, DType.I8, (1, 512), mem_level=MemLevel.L1, cluster=0),
           _ca.at(0).l1("c8", 512), ())
_kp = Port(PortSpec(Layout.ROW_MAJOR, DType.I8, (1, 64), mem_level=MemLevel.L1, cluster=0),
           _ca.at(0).l1("kp", 64), ())
_key = Port(PortSpec(Layout.A, DType.I8, (256, 576), mem_level=MemLevel.L3), _Sym("key"), ())
_val = Port(PortSpec(Layout.A, DType.I8, (512, 256), mem_level=MemLevel.L3), _Sym("val"), ())
_q8 = _pa.add(_QA(inv_qt_f32bits=0x3f800000, inv_qpe_f32bits=0x3f800000, cluster=1), "q8",
              qt=_qt, qpe=_qp)
_ap = _pa.add(_CA(pos=255, cap=256, cluster=0), "app", c8=_c8, kpe8=_kp, key=_key, val=_val)
_at = _pa.add(_MA(keys=256, cap=256, k_s=8, k_o=7, a_exp=0.03, a_n_f32bits=0x3f800000,
                  cluster=1, stream=_lsa), "att", q8=_q8.out(), key=_ap.out("key"),
              val=_ap.out("val"))
_pa.run()
_ax = _at.result.extra
_av = _ap.result.outputs["val"].ends[0]
check("attention: one QK, softmax and PV per tile",
      len(_ax["qk"]) == len(_ax["softmax"]) == len(_ax["pv"]) == 4)
_pvk = [_ax["pv"][j].kernel_args for j in range(4)]
check("PV: tile 0 reads no C; later tiles scale C by corr; the last leaves as FP16 at k_o",
      _pvk[0].input_C_addr == 0 and _pvk[0].flags == 1
      and all(k.flags == 3 for k in _pvk[1:3]) and _pvk[3].flags == 7
      and _pvk[3].d_shift == 7 and _pvk[3].output_D_addr is not _pvk[3].input_C_addr)
check("QK 0 and 1 wait for Q8 by an edge; the rest through the softmax two tiles back",
      all(_ca.dfg.has_edge(_q8.result.outputs["q8"].ends[0], _ax["qk"][j]) for j in (0, 1))
      and all(_ca.dfg.has_edge(_ax["softmax"][j - 2], _ax["qk"][j]) for j in (2, 3)))
_lds = [n for n in _at.result.nodes if n.node_name.startswith("att_Ld_")]
check("the append ends in ONE node, and the last tile's loads wait for it",
      _ap.result.outputs["key"].ends == (_av,) and
      all(_ca.dfg.has_edge(_av, n) for n in _lds if n.node_name[-5] == "3"))
check("a V tile is 32 runs of 16 Bc bytes, 16 cap apart",
      [(n.kernel_args.size, n.kernel_args.src_stride, n.kernel_args.reps) for n in _lds
       if "Ld_V" in n.node_name][:1] == [(1024, 16 * 256, 32)])

# the route record: the golden bytes the device's record is checked against
_tb = _etab(64, {5: (0x1001_0000_0000, 0x1001_0000_1000, 0x1001_0000_2000,
                     0x1001_0000_3000, 0x3f800000)})
_rb = _rbytes([5], [0x3555], _tb).view(_np.uint32)
check("record: id, the weight as FP32 and FP16 bits, then the expert's table entry",
      list(_rb[:3]) == [5, int(_np.array(0x3555, _np.uint16).view(_np.float16)
                                .astype(_np.float32).view(_np.uint32)), 0x3555]
      and list(_rb[16:18]) == [0x0000_0000, 0x1001] and _rb[24] == 0x3f800000)


if FAILED:
    print(f"\n{len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("\nall libs tests passed")
