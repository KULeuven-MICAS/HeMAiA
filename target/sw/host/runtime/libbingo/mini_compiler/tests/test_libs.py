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
        return {"o": PortSpec("A", "i8", (32, 128), mem_level="L1")}

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

    def __init__(self, layout="A", dtype="i8", mem_level="L1", needs=None):
        # A block that fetches nothing has to name the level it reads from: the contract
        # has nothing else to check the binding against. `needs` is how a block that DOES
        # fetch says so -- see the `fetching` consumer below.
        self._spec = PortSpec(layout, dtype, (32, 128), mem_level=mem_level)
        self._needs = needs

    @property
    def inputs(self): return {"x": self._spec}

    @property
    def needs(self): return self._needs or self.inputs

    @property
    def outputs(self): return {"y": PortSpec("D", "f16", (32, 128), mem_level="L1")}

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
        lambda: assemble(consumer=Consumer(layout="D")), "1 B/element")

# A block declares that it FETCHES by overriding `needs`; there is no separate flag. The
# contract is then checked against `needs`, so the gap is the block's to close -- and a
# conversion the hardware cannot do is still refused, with the hardware's reason.
closing = Consumer(layout="D", mem_level=None,
                   needs={"x": PortSpec("D", "i8", (32, 128), mem_level="L1")})
refuses("an int8 reshape is refused for the RIGHT reason: the run is too narrow",
        lambda: assemble(consumer=closing), "8 B per lane")

# The transpose is checked on the plan directly: routed through assemble() it would hit
# the PRECISION refusal first, because this producer emits int8 and B-layout is an fp16
# conversion -- so the assertion would pass for the wrong reason.
#
# A -> B IS THE PAIR THAT STAYS REFUSED. A row_major side can pivot through the A/B
# identity and is planned as two steps; two BLOCKED layouts have nothing to pivot through,
# so the message must still name the transpose rather than a precision problem.
from libs.comm import transfer as _staging                                     # noqa: E402
refuses("a transpose is refused as a transpose, not as a precision problem",
        lambda: _staging.plan(PortSpec("A", "f16", (32, 128), mem_level="L1"),
                              PortSpec("B", "f16", (32, 128), mem_level="L1"),
                              mesh=(16, 4, 16), elem_bytes=2), "TRANSPOSE")
check("a reshape the xDMA can do is planned, not refused",
   _staging.plan(PortSpec("D", "f16", (32, 128), mem_level="L1"),
                 PortSpec("A", "f16", (32, 128), mem_level="L1"),
                 mesh=(16, 4, 16), elem_bytes=2) != [])

refuses("a port with no mem_level, on a block that fetches nothing",
        lambda: assemble(consumer=Consumer(mem_level=None)), "neither the port nor")

# A bound port reads its level off the handle, so a caller may bind a block's own
# level-less input spec directly -- which is what makes "anywhere" usable.
from bingo_mem_handle import BingoMemSymbol as _Sym                           # noqa: E402
_p = Port(PortSpec("A", "i8", (32, 128)), BingoMemAlloc("h", 4096, "L1"), ())
# The vocabulary is a closed set: a typo is refused at construction, naming the valid
# values, rather than reaching a kernel as a layout nobody implements.
for _bad, _kind in ((("packd", "i8", None), "Layout"),
                    (("A", "fp16", None), "DType"),
                    (("A", "i8", "L5"), "MemLevel")):
    refuses(f"a typo'd {_kind} is refused with the valid values named",
            lambda b=_bad: PortSpec(b[0], b[1], (32, 128), mem_level=b[2]),
            "is not a valid")
_s = PortSpec("A", "i8", (32, 128), mem_level="L1")
check("a plain string is coerced to the enum member",
      (_s.layout is Layout.A and _s.dtype is DType.I8 and _s.mem_level is MemLevel.L1),
      f"got {_s.layout!r} {_s.dtype!r} {_s.mem_level!r}")
check("the two spellings make equal specs",
      _s == PortSpec(Layout.A, DType.I8, (32, 128), mem_level=MemLevel.L1))
check("a member still behaves as its string", f"{Layout.D32}" == "d32" and Layout.D32 == "d32")

check("a bound port takes its level from the handle", _p.spec.mem_level == "L1",
      f"got {_p.spec.mem_level!r}")
_p = Port(PortSpec("A", "i8", (32, 128)), _Sym("staged"), ())
check("a staged symbol resolves to L3", _p.spec.mem_level == "L3",
      f"got {_p.spec.mem_level!r}")
_p = Port(PortSpec("A", "i8", (32, 128), mem_level="L4"), _Sym("staged"), ())
check("an explicit level is not overwritten", _p.spec.mem_level == "L4",
      f"got {_p.spec.mem_level!r}")

bad_shape = Consumer()
bad_shape._spec = PortSpec("A", "i8", (64, 128), mem_level="L1")
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

ok = check_contract(Port(PortSpec("A", "i8", (32, 128), mem_level="L1"), None, ()),
                    PortSpec("A", "i8", (32, 128), mem_level="L1"), where="t")
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

# ------------------------------------------------------- the layout pass
print("\nthe layout pass")
from libs import LayoutStep, assign_layouts                              # noqa: E402
from libs.block import Reshape as _Reshape                               # noqa: E402
from libs.block.simd.norm import RMSNorm as _RMSNorm                     # noqa: E402

_T, _D, _M = 32, 128, (16, 4, 16)


def _chain(rows=_T):
    orient = []
    for i in (Layout.ROW_MAJOR, Layout.COL_MAJOR):
        for o in (Layout.ROW_MAJOR, Layout.COL_MAJOR):
            for ip in ((False, True) if i == Layout.COL_MAJOR else (False,)):
                orient.append({"in_layout": i, "out_layout": o, "x_in_place": ip})
    return [
        LayoutStep("norm", lambda **k: _RMSNorm(rows=rows, cols=_D, cluster=0, **k),
                   orient),
        LayoutStep("to_a", lambda **k: _Reshape(rows=rows, cols=_D, mesh=_M,
                                                dtype=DType.F16, cluster=0, **k),
                   [{"src": s, "dst": Layout.A}
                    for s in (Layout.ROW_MAJOR, Layout.COL_MAJOR)]),
    ]


def _run(start, rows=_T):
    return assign_layouts(_chain(rows), mesh=_M, elem_bytes=2, shape=(rows, _D),
                          start=start, end=Layout.A, verbose=False)


# THE BLOCK INFERS THE KERNEL FROM THE ENDS, so a row_major/row_major candidate really is
# the fold-paying one -- and the pass, ranking folds first, must reject it even though it
# is the candidate with the fewest passes.
_folds = {(c.src, c.dst): c.simd_folds for c in _run(Layout.ROW_MAJOR).candidates[0]}
check("row_major on both ends pays a fold per row",
      _folds[(Layout.ROW_MAJOR, Layout.ROW_MAJOR)] == _T, _folds)
check("any col_major end pays none",
      all(v == 0 for k, v in _folds.items() if Layout.COL_MAJOR in k), _folds)

r1 = _run(Layout.ROW_MAJOR)
check("the pass takes a fold-free arm", r1.simd_folds == 0, r1.simd_folds)
check("...even though it is NOT the fewest passes",
      r1.passes > min(c.xdma_passes for c in r1.candidates[0]), r1.passes)
check("the norm it picks runs the col_major kernel",
      _RMSNorm(rows=_T, cols=_D, cluster=0, **r1.chosen[0]).col_major, r1.chosen[0])

# A TRANSPOSED SOURCE IS STRICTLY BETTER, and only by one pass -- the SIMD is unchanged.
r2 = _run(Layout.COL_MAJOR)
check("staging x^T saves a pass", r2.passes == r1.passes - 1, (r1.passes, r2.passes))
check("...and does not change the SIMD work", r2.simd_folds == r1.simd_folds,
      (r1.simd_folds, r2.simd_folds))
check("...by consuming col_major directly, in place",
      r2.chosen[0]["in_layout"] == Layout.COL_MAJOR and r2.chosen[0]["x_in_place"],
      r2.chosen[0])

# LEGALITY COMES FROM THE BLOCKS. At rows != 32 every col_major candidate refuses, so the
# pass is left with the row_major arm rather than proposing something unbuildable.
r3 = _run(Layout.ROW_MAJOR, rows=64)
_n3 = _RMSNorm(rows=64, cols=_D, cluster=0, **r3.chosen[0])
check("an illegal shape falls back to the row_major kernel", not _n3.col_major,
      r3.chosen[0])
check("...and then it DOES pay a fold per row", _n3.simd_folds() == 64, _n3.simd_folds())

# WHAT IT PICKS MUST BUILD. The whole point of a planner is that its answer is realisable.
_c = new_ctx()
_blk = _RMSNorm(rows=_T, cols=_D, cluster=0, **r2.chosen[0])
_g = _c.at(0)
_slot = _blk.alloc(_g)
_res = _blk.build(_g, {"x": Port(PortSpec(Layout.COL_MAJOR, DType.F16, (_T, _D),
                                          mem_level=MemLevel.L1),
                                 _slot, (), cluster=0, name="x")})
check("the chosen norm builds, with no staging copy",
      not any("Stage_xt" in n.node_name for n in _res.nodes),
      [n.node_name for n in _res.nodes])
_rs = _Reshape(rows=_T, cols=_D, mesh=_M, dtype=DType.F16, cluster=0, **r2.chosen[1])
check("the chosen reshape builds", _rs.xdma_passes() >= 1, _rs.xdma_passes())

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
_bad = _RMSNorm(rows=_T, cols=_D, cluster=0,
                in_layout=Layout.COL_MAJOR, out_layout=Layout.ROW_MAJOR, x_in_place=True)
_g2 = new_ctx().at(0)
_bad.alloc(_g2)
try:
    _bad.build(_g2, {"x": Port(PortSpec(Layout.COL_MAJOR, DType.F16, (_T, _D),
                                        mem_level=MemLevel.L1),
                               _g2.l1("elsewhere", _T * _D * 2), (), cluster=0, name="x")})
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
    b = _RMSNorm(rows=_T, cols=_D, cluster=0, in_layout=in_lay, out_layout=out_lay)
    h = g.l1("x", _T * _D * 2) if level == _ML.L1 else g.l3("x_l3", _T * _D * 2)
    r = b.build(g, {"x": Port(PortSpec(in_lay, DType.F16, (_T, _D), mem_level=level),
                              h, (), cluster=0, name="x")})
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
check("...while idma_passes(in_l3=True) reports it", _b.idma_passes(in_l3=True) == 1,
      _b.idma_passes(in_l3=True))

# L4 IS REFUSED, because the hoist is one move for the whole layer, not one per operator.
try:
    _c4 = new_ctx()
    _b4 = _RMSNorm(rows=_T, cols=_D, cluster=0)
    _b4.build(_c4.at(0), {"x": Port(PortSpec(Layout.ROW_MAJOR, DType.F16, (_T, _D),
                                             mem_level=_ML.L4),
                                    _c4.at(0).l1("x", _T * _D * 2), (),
                                    cluster=0, name="x")})
    check("an L4 operand is refused", False, "it built")
except ValueError:
    check("an L4 operand is refused", True)

# ------------------------------------------------ RMSNorm straight into a GEMM operand
print("\nRMSNorm writing a GEMM operand (out_layout A / B)")


def _norm_out(in_lay, out_lay, dtype=DType.I8, rows=_T):
    c = new_ctx()
    g = c.at(0)
    b = _RMSNorm(rows=rows, cols=_D, cluster=0, in_layout=in_lay, out_layout=out_lay,
                 out_dtype=dtype, inv_scale_f32bits=0x42800000, mesh=_M)
    h = g.l1("x", rows * _D * 2)
    r = b.build(g, {"x": Port(PortSpec(in_lay, DType.F16, (rows, _D), mem_level=_ML.L1),
                              h, (), cluster=0, name="x")})
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
    _RMSNorm(rows=_T, cols=_D, cluster=0, out_layout=Layout.A)
    check("out_layout A without a mesh is refused", False, "built")
except ValueError:
    check("out_layout A without a mesh is refused", True)

# ------------------------------------------------ the planner sees fusion
print("\nthe layout pass, choosing between fused and unfused routes")
from libs.block import Quantize as _Quantize                             # noqa: E402


def _fuse_chain(rows, consumer):
    return [
        LayoutStep("norm", lambda **k: _RMSNorm(rows=rows, cols=_D, cluster=0, mesh=_M,
                                                 inv_scale_f32bits=0x42800000, **k),
                   _RMSNorm.options()),
        LayoutStep("to_op", lambda **k: _Reshape(rows=rows, cols=_D, mesh=_M,
                                                 dtype=DType.F16, cluster=0, **k),
                   [{"src": s, "dst": consumer} for s in (Layout.ROW_MAJOR, Layout.COL_MAJOR)],
                   optional=True),
        LayoutStep("quant", lambda **k: _Quantize(rows=rows, cols=_D, cluster=0,
                                                  inv_scale_f32bits=0x42800000, **k),
                   [{"layout": consumer}], optional=True),
    ]


def _fuse_run(rows, start, consumer):
    return assign_layouts(_fuse_chain(rows, consumer), mesh=_M, elem_bytes=2,
                          shape=(rows, _D), start=start, end=consumer,
                          end_dtype=DType.I8, verbose=False)


# rows = 64: no col_major kernel, so every route folds. The fused one (norm writes A/int8)
# saves the Reshape and the Quantize, and the planner must see that.
_f = _fuse_run(64, Layout.ROW_MAJOR, Layout.A)
check("rows=64 -> int8 A: the norm writes A/int8 itself",
      _f.chosen[0]["out_layout"] == Layout.A and _f.chosen[0]["out_dtype"] == DType.I8,
      _f.chosen[0])
check("...and the Reshape and Quantize are skipped", _f.chosen[1:] == [None, None],
      _f.chosen)
check("...costing 3 SIMD passes and no xDMA", (_f.simd_passes, _f.passes) == (3, 0),
      (_f.simd_passes, _f.passes))

# rows = 32, row_major in, A wanted: folds come FIRST, so the fold-free col_major kernel
# wins even though it costs a separate quantise and three transposes -- the measured trade
# (SIMD ~1,400 cc against the fused route's ~2,960).
_f = _fuse_run(32, Layout.ROW_MAJOR, Layout.A)
check("rows=32 -> int8 A: the fold-free chain wins", _f.simd_folds == 0, _f.simd_folds)
check("...with the quantise as its own step", _f.chosen[2] is not None, _f.chosen)

# col_major in, B wanted: col_major -> B is fold-free AND fused, so nothing else competes.
_f = _fuse_run(32, Layout.COL_MAJOR, Layout.B)
check("col_major -> int8 B: one kernel, no transposes, nothing after it",
      _f.chosen[0]["out_layout"] == Layout.B and _f.chosen[1:] == [None, None]
      and (_f.simd_folds, _f.passes) == (0, 0), (_f.chosen, _f.simd_folds, _f.passes))

# A link never changes precision: an int8 norm output cannot feed an fp16 Reshape.
_f = _fuse_run(64, Layout.ROW_MAJOR, Layout.A)
check("no chosen route links int8 into an fp16 step",
      not (_f.chosen[0]["out_dtype"] == DType.I8 and _f.chosen[1] is not None), _f.chosen)

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
_rm = PortSpec(Layout.ROW_MAJOR, DType.F16, (32, 128), mem_level=MemLevel.L1)
_b = PortSpec(Layout.B, DType.F16, (32, 128), mem_level=MemLevel.L1)
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
_a = PortSpec(Layout.A, DType.F16, (32, 128), mem_level=MemLevel.L1)
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

if FAILED:
    print(f"\n{len(FAILED)} FAILED: {', '.join(FAILED)}")
    sys.exit(1)
print("\nall libs tests passed")
