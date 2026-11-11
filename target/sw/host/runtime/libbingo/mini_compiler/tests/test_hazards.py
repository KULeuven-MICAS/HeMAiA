#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Tests for the memory-hazard check (passes/bingo_dfg_hazards.py): every two tasks of which one writes bytes the other
# touches must be ordered by the graph's edges, from what each kernel declares in accesses().
#
#   python3 test_hazards.py

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402
from bingo_dfg import BingoDFG                                        # noqa: E402
from bingo_kernel_args import (BingoKernelArgs, SnaxBingoKernelGemmFullArgs,   # noqa: E402
                               SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelWaitFlagsArgs, Read)
from bingo_mem_handle import BingoMemAlloc                            # noqa: E402
from bingo_node import BingoNode                                      # noqa: E402
from kernel_gemm import set_gemm_shapes                               # noqa: E402

set_gemm_shapes([(16, 4, 16)], (8, 8, 32, 32))   # the paper cluster's array: meshRow 16, tileSize 4, meshCol 16
FAILED = []
GEMM, DM, HOST = 0, 3, 4


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ""))
    if not cond:
        FAILED.append(name)


def new_dfg(enforce=True):
    d = BingoDFG(num_chiplets=1, num_clusters_per_chiplet=1, num_cores_per_cluster=4, is_host_as_acc=True,
                 chiplet_ids=[0], dep_tag_width=5)
    d.schedule_cores = enforce
    return d


def task(d, core, name, args, after=()):
    n = BingoNode(0, 0, core, node_name=name, kernel_args=args,
                  kernel_name=getattr(args, "KERNEL_NAME", None) or "__snax_bingo_kernel_idma_1d_copy")
    d.bingo_add_node(n)
    for a in after:
        d.bingo_add_edge(a, n)
    return n


def l1(name, nbytes):
    return BingoMemAlloc(name, size=nbytes, mem_level="L1", chip_id=0, cluster_id=0)


def gemm_reading(slab, out, half):
    """A GEMM whose B operand is one half of `slab` (M 1, K 64, N 1: 4 KiB of INT8 B) into its own output."""
    return SnaxBingoKernelGemmFullArgs(l1("a", 1024), slab.view(half * 4096), 0, out, M=1, K=64, N=1,
                                       array_shape_idx=0, transpose_A=0, transpose_B=0, accumPrevC=0)


def raises(fn, needle=""):
    try:
        fn()
    except ValueError as e:
        return needle in str(e), str(e)
    return False, "did not raise"


print("\nthe four-task example: L1 overwrites the slab G1 and G2 read")
d = new_dfg()
slab = l1("slab", 8192)
l0 = task(d, DM, "L0", SnaxBingoKernelIdma1dCopyArgs(l1("w0", 8192), slab, 8192))
g1 = task(d, GEMM, "G1", gemm_reading(slab, l1("y1", 8192), 0), [l0])
g2 = task(d, GEMM, "G2", gemm_reading(slab, l1("y2", 8192), 1), [l0])
l1_ = task(d, DM, "L1", SnaxBingoKernelIdma1dCopyArgs(l1("w1", 8192), slab, 8192), [g2])
ok, msg = raises(d.bingo_check_hazards, "G1")
check("without G1 -> L1 the compile fails, naming G1, L1 and the slab", ok and "L1" in msg and "slab" in msg, msg)
d.bingo_add_edge(g1, l1_)
check("with G1 -> L1 it passes", d.bingo_check_hazards() == [])

print("\nwhat is not a hazard")
d = new_dfg()
buf = l1("buf", 4096)
task(d, DM, "lo", SnaxBingoKernelIdma1dCopyArgs(l1("s0", 2048), buf.view(0), 2048))
task(d, DM, "hi", SnaxBingoKernelIdma1dCopyArgs(l1("s1", 2048), buf.view(2048), 2048))
check("two writes into disjoint halves, unordered", d.bingo_check_hazards() == [])
d = new_dfg()
src = l1("src", 4096)
task(d, DM, "r0", SnaxBingoKernelIdma1dCopyArgs(src, l1("d0", 4096), 4096))
task(d, DM, "r1", SnaxBingoKernelIdma1dCopyArgs(src, l1("d1", 4096), 4096))
check("two reads of one buffer, unordered", d.bingo_check_hazards() == [])
d = new_dfg()
flag = l1("flag", 64)
task(d, DM, "w0", SnaxBingoKernelWaitFlagsArgs(flag, 1))
task(d, DM, "w1", SnaxBingoKernelWaitFlagsArgs(flag, 1))
check("two polls of one flag (Sync), unordered", d.bingo_check_hazards() == [])

print("\na dispatch that reads until its wait")
d = new_dfg()
x = l1("x", 8192)
st = task(d, GEMM, "S", gemm_reading(x, l1("y", 8192), 0))
wt = task(d, GEMM, "W", SnaxBingoKernelWaitFlagsArgs(l1("f", 64), 1), [st])
st.operands_held_by = wt
ov = task(d, DM, "over", SnaxBingoKernelIdma1dCopyArgs(l1("z", 8192), x, 8192), [st])
ok, msg = raises(d.bingo_check_hazards, "over")
check("a write after the start but not after the wait is a hazard", ok, msg)
d.bingo_add_edge(wt, ov)
check("after the wait it is not", d.bingo_check_hazards() == [])

print("\nevery kernel declares its accesses")


class _NoDecl(BingoKernelArgs):
    def __init__(self, dst):
        self.dst_addr = dst

    def get_struct_name(self):
        return "x"

    def get_c_field_assignments(self, m):
        return {}


class _HalfDecl(_NoDecl):
    def __init__(self, src, dst):
        super().__init__(dst)
        self.src_addr = src

    def accesses(self):
        return [Read("src_addr", 64)]


d = new_dfg()
task(d, DM, "nd", _NoDecl(l1("b", 64)))
ok, msg = raises(d.bingo_check_hazards, "declares no accesses")
check("a kernel without accesses() is an error", ok, msg)
d = new_dfg()
task(d, DM, "hd", _HalfDecl(l1("s", 64), l1("b", 64)))
ok, msg = raises(d.bingo_check_hazards, "dst_addr")
check("a buffer field left out of accesses() is an error", ok, msg)

print("\na graph whose cores the compiler does not order is reported on, not refused")
d = new_dfg(enforce=False)
slab = l1("slab", 8192)
l0 = task(d, DM, "L0", SnaxBingoKernelIdma1dCopyArgs(l1("w0", 8192), slab, 8192))
task(d, GEMM, "G1", gemm_reading(slab, l1("y1", 8192), 0), [l0])
task(d, DM, "L1", SnaxBingoKernelIdma1dCopyArgs(l1("w1", 8192), slab, 8192), [l0])
bad = d.bingo_check_hazards()
check("no error, the unordered pair returned", len(bad) == 1)

print("\nthe args classes that declare no accesses() yet (to do: each fails the check once a graph uses it)")
import bingo_kernel_args as _K                                       # noqa: E402


def _subclasses(c):
    for s in c.__subclasses__():
        yield s
        yield from _subclasses(s)


_todo = sorted({c.__name__ for c in _subclasses(BingoKernelArgs)
                if not c.__name__.startswith("_") and c.__module__.startswith("kernel_")
                and all("accesses" not in vars(k) for k in c.__mro__[:c.__mro__.index(BingoKernelArgs)])})
print(f"  {len(_todo)}: {', '.join(_todo)}")

print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'}")
_sys.exit(1 if FAILED else 0)
