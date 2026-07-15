# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Shared builder for the per-precision GEMM cycle-sweep workloads
# (gemm_psweep_<prec>_1cluster). Lives in util/sim/ alongside gemm_sim_utils.py;
# each workload's main_bingo.py adds util/sim to sys.path and calls run("<prec>").
#
# Each workload runs ONE precision over a grid of (M, K, N, array_shape) sizes in
# a single DFG, using the clean precision-named wrapper kernels
# (offload_hw_kernels/gemm.h). Per config the DFG emits:
#     Load_A -> Load_B -> Gemm<prec> -> Check_D
# and the device GEMM kernel brackets its compute with BINGO_TRACE_GEMM_FULL_RUN_*
# markers, so target/sim/automation/sweep/gemm/gather_gemm_luts.py recovers one
# cycle count per config (paired with configs.json) — no host mcycle bracketing.
#
# Precision is split across workloads (one per precision) because it changes the
# generated data layout (int4 packing / quant int8 / fp16 output) and so cannot
# be looped inside one binary. Running the 5 workloads as a task list gives the
# parallel multi-precision sweep.
#
# Data + golden come from the validated gemm_sim_utils.generate_gemm_test_data
# (handles int4 packing, int8 requantization, int32->fp16) so this stays DRY.

import os
import sys
import json
import random
import argparse
import pathlib

import hjson

_THIS = os.path.dirname(os.path.abspath(__file__))         # .../util/sim/gemm
ROOT = os.path.normpath(os.path.join(_THIS, "../../.."))   # repo root
sys.path.append(os.path.join(ROOT, "target/sw/host/runtime/libbingo/mini_compiler"))
sys.path.append(os.path.join(ROOT, "util/sim/common"))     # data_utils
sys.path.append(_THIS)  # util/sim/gemm — gemm_sim_utils (sibling)

from bingo_dfg import BingoDFG                                       # noqa: E402
from bingo_node import BingoNode                                     # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol           # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg   # noqa: E402
from bingo_kernel_args import (                                      # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    HostBingoKernelCheckResultArgs,
)
import bingo_kernel_args as _bka                                     # noqa: E402
from gemm_sim_utils import (                                          # noqa: E402
    generate_gemm_test_data, _bytes_for_elements, _gemm_operand_widths,
)
from data_utils import format_vector_definition                     # noqa: E402

# ---------------------------------------------------------------------------
# Kernel-id resolution: ONE class per RUNNABLE kernel = (precision mode, array shape).
#
# The mini_compiler exposes one args class per (mode, shape), with ARRAY_SHAPE_IDX baked
# into the class rather than passed as a constructor argument -- that is what makes the
# bingo op id, the CSR the wrapper writes, the operand block geometry and the cost curve
# agree by construction. Each class also has a `Minimal` sibling: the REUSE kernel, which
# dispatches to the single __snax_bingo_kernel_gemm_minimal and reprograms only base
# addresses, inheriting the mode AND shape CSRs from the configure that ran before it.
#
# ---------------------------------------------------------------------------
_MODE_STEM = {
    "i8i8_i32": "I8I8I32", "i8i4_i32": "I8I4I32", "i4i4_i32": "I4I4I32",
    "i8i8_i8": "I8I8I8", "i8i4_f16": "I8I4F16", "i8i8_f16": "I8I8F16",
}


def shape_tag(mesh):
    """(mu, ku, nu) -> the M<mu>K<ku>N<nu> tag used in every kernel id and LUT name."""
    return "M{}K{}N{}".format(*mesh)


def op_id(mode, mesh, minimal=False):
    """The bingo op id, e.g. gemm_i8i8_i32_M32K2N32 / ..._minimal."""
    return f"gemm_{mode}_{shape_tag(mesh)}" + ("_minimal" if minimal else "")


def gemm_args_cls(mode, mesh, minimal=False):
    name = (f"SnaxBingoKernelGemm{_MODE_STEM[mode]}{shape_tag(mesh)}"
            f"{'Minimal' if minimal else ''}Args")
    cls = getattr(_bka, name, None)
    if cls is None:
        raise ValueError(f"mini_compiler has no {name} (mode={mode} mesh={mesh})")
    return cls


def meshes_of(hwcfg):
    """The array shapes the hwcfg exposes, as (mu, ku, nu) per array_shape index."""
    u = (hwcfg["snax_versacore_core_template"]["snax_acc_cfg"][0]
         ["snax_versacore_spatial_unrolling"][0])
    return [tuple(int(x) for x in s) for s in u]

# M/K/N are mesh-TILE counts: the element extents are M*meshRow, K*tileSize, N*meshCol.
#
# The grid these three lists span is what a GEMM cost model is fitted to, so it has to
# COVER the shapes the model will later be asked to price. This near-diagonal base is
# small and shallow in K: at shape 0 (tileSize=2), K=8 tiles is a K of just 16 ELEMENTS,
# while an llama3 layer reduces over K=4096 (Q/K/V/O, FFN gate/up) and 14336 (FFN down).
# On its own it would force the model to extrapolate by 2-3 orders of magnitude in K, and
# being diagonal it cannot tell an m*n term from an m*k*n one. The deep-K ladder and the
# off-diagonal (m,n) points below supply exactly those two missing directions.
LEGACY_CONFIGS = [(1, 1, 1), (1, 2, 1), (1, 4, 1), (1, 8, 1),
                  (2, 2, 2), (2, 4, 2), (4, 4, 4)]

# Deep-K ladder, run at (m,n)=(1,1) so K can go as deep as L1 allows. Stated in ELEMENTS,
# not tiles: a tile is 2 elements at shape 0 but 16 at shape 1, so a ladder in tiles walks
# the shapes to wildly different physical depths and lets the deepest one inflate the
# SHARED l1_B (which scales with tileSize*meshCol) until the other shapes are starved.
# Elements are also the units llama3's K is quoted in, which is what the grid must cover.
K_ELEM_LADDER = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Off-diagonal (m,n) points at a modest K: they are what separates an m*n term from an
# m*k*n term, which a grid with m==n cannot do.
MN_POINTS = [(1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (2, 4), (4, 2), (4, 4), (8, 8)]
MN_KS = [8, 64]


def _frag(x):
    """bingo_l1_alloc's per-allocation overhead + 256 B alignment."""
    return (x + 128 + 255) & ~255


# Every node placed on a CLUSTER core costs two L1 allocations that bingo_dfg makes up
# front and never frees: its bingo_kernel_scratchpad_t, and its kernel args struct. Both
# are far below the allocator's 256 B granularity, so each one occupies a whole slot
# regardless of its nominal size -- which is what makes them expensive in bulk.
L1_ALLOC_SLOT = _frag(1)                 # smallest slot bingo_l1_alloc can hand out
ALLOCS_PER_CLUSTER_NODE = 2              # scratchpad + kernel args

# Cluster nodes _create_dfg emits per config: Load_A, Load_B (dma core), Gemm, GemmMin
# (gemm core). The two Check nodes run on the HOST, whose scratchpad and args go to L3,
# so they cost no L1.
CLUSTER_NODES_PER_CONFIG = 4
EXIT_NODES = 2                           # bingo_dfg's own exit nodes, one per cluster core


def _node_alloc_bytes(n_configs):
    """L1 the per-node allocations take.

    These scale with the NUMBER of configs, not their size: at 4 cluster nodes a config,
    2 allocations a node and a 256 B slot each, a 92-config grid spends ~184 KiB here --
    over a third of the cluster's 504 KiB TCDM. Budgeting the operand buffers against the
    whole heap therefore overcommits it, and the grid dies at bingo_l1_alloc.
    """
    nodes = CLUSTER_NODES_PER_CONFIG * n_configs + EXIT_NODES
    return nodes * ALLOCS_PER_CLUSTER_NODE * L1_ALLOC_SLOT


def _operand_bytes(M, K, N, mesh, a_len, b_len, d_len):
    """A/B/D bytes for one config -- analytic, so the grid can be sized without
    generating the arrays."""
    mu, ku, nu = mesh
    a = _pad64(_bytes_for_elements(M * K * mu * ku, a_len))
    b = _bytes_for_elements(K * N * ku * nu, b_len)
    d = _bytes_for_elements(M * N * mu * nu, d_len)
    return a, b, d


def _l1_footprint(configs, meshes, a_len, b_len, d_len):
    """THE binding L1 constraint, and it is not per-config.

    _create_dfg allocates ONE l1_A / l1_B / l1_D for the whole binary, each sized at the
    MAX over every config (see `max_A, max_B, max_D` below), and bingo_dfg never emits a
    free -- so all three live simultaneously for the entire run. The budget is therefore
    max_A + max_B + max_D, NOT max over configs of (A+B+D). A grid whose configs each fit
    individually can still blow L1 and die at bingo_l1_alloc -> host_abort(BINGO_EXIT_OOM).

    On top of those three sit the per-node scratchpads, which grow with the NUMBER of
    configs rather than their size. They are the reason a grid cannot simply be extended
    until the operand buffers fill the heap: each config added costs its own scratchpads
    whether or not it enlarges any buffer.
    """
    if not configs:
        return 0, 0, 0, 0
    szs = [_operand_bytes(M, K, N, meshes[s], a_len, b_len, d_len) for (M, K, N, s) in configs]
    mA = max(_frag(s[0]) for s in szs)
    mB = max(_frag(s[1]) for s in szs)
    mD = max(_frag(s[2]) for s in szs)
    return mA + mB + mD + _node_alloc_bytes(len(configs)), mA, mB, mD


def build_configs(meshes, a_len, b_len, d_len, l1_size_kb, verbose=True):
    """Grow the grid greedily, keeping the whole L1 footprint inside the budget.

    The K ladder is climbed in LOCKSTEP across the array shapes -- one rung for every
    shape, then the next rung -- rather than one shape at a time. The shapes compete for a
    single budget (l1_A/l1_B/l1_D are sized at the max over ALL configs, whatever shape
    set it), so a shape-at-a-time walk lets whichever shape goes first climb until the
    budget is gone and leaves the rest with a K range too short to fit a curve through.
    Climbing in lockstep spends the budget on the rung every shape can still afford.

    How deep that gets differs per shape anyway, because the operands scale with different
    mesh dims (B with tileSize*meshCol: 512 at shape 1 against 64 at shape 0), so a shape
    simply stops when its next rung no longer fits.
    """
    budget = l1_size_kb * 1024
    nsh = len(meshes)
    cfgs = [(m, k, n, s) for s in range(nsh) for (m, k, n) in LEGACY_CONFIGS]

    def fits(extra):
        return _l1_footprint(cfgs + [extra], meshes, a_len, b_len, d_len)[0] <= budget

    kmax = {s: 0 for s in range(nsh)}
    live = set(range(nsh))                  # shapes whose ladder has not yet hit the wall
    for k_elem in K_ELEM_LADDER:
        for s in sorted(live):
            tile = meshes[s][1]             # tileSize: the K elements one K-tile holds
            k = k_elem // tile
            if k < 1 or k * tile != k_elem:  # this depth is not a whole number of tiles here
                continue
            if fits((1, k, 1, s)):
                cfgs.append((1, k, 1, s))
                kmax[s] = k
            else:
                live.discard(s)             # this shape is done; the others keep climbing
        if not live:
            break
    for s in range(nsh):
        for (m, n) in MN_POINTS:
            for k in MN_KS:
                if fits((m, k, n, s)):
                    cfgs.append((m, k, n, s))

    cfgs = sorted(set(cfgs))
    tot, mA, mB, mD = _l1_footprint(cfgs, meshes, a_len, b_len, d_len)
    if verbose:
        kel = {s: kmax[s] * meshes[s][1] for s in range(nsh)}
        print(f"[gemm_psweep] {len(cfgs)} configs | L1 {tot/1024:.1f}/{l1_size_kb} KiB "
              f"(A{mA//1024} B{mB//1024} D{mD//1024}) | max K per shape {kmax} "
              f"= {kel} ELEMENTS")
    return cfgs

_CTYPE_BYTES = {"int8_t": 1, "uint16_t": 2, "int32_t": 4}

# BINGO_SERIAL_C_D_WIDTH from gemm_shapes.h — the VersaCore C/D serializer beat
# width (bits). The D-extension (requant->int8 / int32->fp16) output stage has two
# kernel paths (offload_hw_kernels/gemm.h): when one narrowed output tile spans
# >= one beat (meshRow*meshCol*d_bits >= 1024 — array_shape 0 and 2 here) the
# streamer splits it into whole beats and works; when one tile is SMALLER than a
# beat (the narrow meshRow=1 array_shape 1) the "pack multiple tiles into one
# beat" path stalls the accelerator (observed: shape-1 hangs even when M*N fills
# whole beats, e.g. (2,2,2,1)). So a D-extension config is only emittable when one
# output tile already fills >= one serializer beat.
_SERIAL_C_D_WIDTH = 1024


def _d_extension_emittable(spec, gd):
    """Whether the D-extension output stage can emit this config without stalling.
    int32-output precisions have no D extension -> always emittable. For the
    requant-int8 / int32->fp16 precisions, only configs whose single narrowed
    output tile fills >= one serializer beat work (the small-tile packing path
    hangs); in this sweep that excludes every array_shape-1 config."""
    flags = spec["flags"]
    if not (flags.get("quantization_enable", 0) or flags.get("int32tofp16_enable", 0)):
        return True
    d_bits = _CTYPE_BYTES[spec["d_ctype"]] * 8
    one_output_tile_bits = gd["meshRow"] * gd["meshCol"] * d_bits
    return one_output_tile_bits >= _SERIAL_C_D_WIDTH


def make_gemm_args(mode, mesh, A, B, C, D, M, K, N, gd):
    """Args for the CONFIGURE kernel of (mode, mesh). The shape is baked into the class,
    so it is NOT passed; only the requantizing mode adds fields."""
    cls = gemm_args_cls(mode, mesh)
    kw = dict(input_A_addr=A, input_B_addr=B, input_C_addr=C, output_D_addr=D,
              M=M, K=K, N=N)
    if mode == "i8i8_i8":
        cfg = gd["config"]
        kw.update(shift_i=cfg["shift_i"], multiplier_i=cfg["multiplier_i"],
                  input_zp_i=cfg["input_zp_i"], output_zp_i=cfg["output_zp_i"])
    return cls(**kw)


def make_gemm_minimal_args(mode, mesh, A, B, C, D):
    """Args for the REUSE kernel: base addresses only. M/K/N and every mode/shape CSR are
    inherited from the configure that precedes it -- that is the whole point of the kernel,
    and why its cost is a different curve worth measuring."""
    return gemm_args_cls(mode, mesh, minimal=True)(
        input_A_addr=A, input_B_addr=B, input_C_addr=C, output_D_addr=D)


# Precision registry: name -> device wrapper symbol, datagen flags, D ctype.
# Names + flags mirror the GEMM_PRECISIONS test contract and the gemm.h wrappers.
# The args class is not listed here: it is resolved per (mode, shape) above.
PRECISIONS = {
    "i8i8_i32": dict(
        kernel="__snax_bingo_kernel_gemm_i8i8_i32",
        flags=dict(int4_a_enable=0, int4_b_enable=0, quantization_enable=0, int32tofp16_enable=0),
        d_ctype="int32_t"),
    "i8i4_i32": dict(
        kernel="__snax_bingo_kernel_gemm_i8i4_i32",
        flags=dict(int4_a_enable=0, int4_b_enable=1, quantization_enable=0, int32tofp16_enable=0),
        d_ctype="int32_t"),
    "i4i4_i32": dict(
        kernel="__snax_bingo_kernel_gemm_i4i4_i32",
        flags=dict(int4_a_enable=1, int4_b_enable=1, quantization_enable=0, int32tofp16_enable=0),
        d_ctype="int32_t"),
    "i8i8_i8": dict(
        kernel="__snax_bingo_kernel_gemm_i8i8_i8",
        flags=dict(int4_a_enable=0, int4_b_enable=0, quantization_enable=1, int32tofp16_enable=0),
        d_ctype="int8_t"),
    "i8i4_f16": dict(
        kernel="__snax_bingo_kernel_gemm_i8i4_f16",
        flags=dict(int4_a_enable=0, int4_b_enable=1, quantization_enable=0, int32tofp16_enable=1),
        d_ctype="uint16_t"),
    "i8i8_f16": dict(
        kernel="__snax_bingo_kernel_gemm_i8i8_f16",
        flags=dict(int4_a_enable=0, int4_b_enable=0, quantization_enable=0, int32tofp16_enable=1),
        d_ctype="uint16_t"),
}

# The reuse kernel every Minimal class dispatches to.
MINIMAL_KERNEL = "__snax_bingo_kernel_gemm_minimal"


def _pad64(n):
    return (n + 63) & ~63


def _gen_configs(configs, hwcfg, flags):
    """One generate_gemm_test_data() per config (deterministic). Returns the list
    of gemm_data dicts (A/B/D arrays + the config incl. any quant params)."""
    random.seed(320)  # make requant params reproducible across builds
    out = []
    for (M, K, N, shape) in configs:
        kwargs = {
            **hwcfg,
            "M": M, "K": K, "N": N, "array_shape": shape, "data_type": 0,
            "transposed_A": 0, "transposed_B": 0,
            "addNonZeroC": 0, "addZeroC": 1, "accumPrevC": 0,
            **flags,
        }
        out.append(generate_gemm_test_data(**kwargs))
    return out


def _emit_header(gen, d_ctype):
    lines = ["#include <stdint.h>"]
    for i, gd in enumerate(gen):
        lines.append("")
        lines.append(f"// cfg{i}: M={gd['M']} K={gd['K']} N={gd['N']} "
                     f"mesh={gd['meshRow']}x{gd['tileSize']}x{gd['meshCol']}")
        lines.append(format_vector_definition("int8_t", f"A_cfg{i}", gd["A"]))
        lines.append(format_vector_definition("int8_t", f"B_cfg{i}", gd["B"]))
        lines.append(format_vector_definition(d_ctype, f"D_cfg{i}", gd["D"]))
    return "\n".join(lines) + "\n"


def _create_dfg(configs, gen, spec, platform, mode, meshes):
    dbytes = _CTYPE_BYTES[spec["d_ctype"]]
    a_sz = [_pad64(len(gd["A"])) for gd in gen]   # xDMA-aligned
    b_sz = [len(gd["B"]) for gd in gen]
    d_sz = [len(gd["D"]) * dbytes for gd in gen]
    max_A, max_B, max_D = max(a_sz), max(b_sz), max(d_sz)

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )
    gemm_core, dma_core, host_core = 0, 1, 2
    l1_A = BingoMemAlloc("l1_A", size=max_A, mem_level="L1", chip_id=0, cluster_id=0)
    l1_B = BingoMemAlloc("l1_B", size=max_B, mem_level="L1", chip_id=0, cluster_id=0)
    l1_D = BingoMemAlloc("l1_D", size=max_D, mem_level="L1", chip_id=0, cluster_id=0)

    prev_check = None
    for i, (M, K, N, shape) in enumerate(configs):
        A_l3 = BingoMemSymbol(f"A_cfg{i}")
        B_l3 = BingoMemSymbol(f"B_cfg{i}")
        D_l3 = BingoMemSymbol(f"D_cfg{i}")
        load_A = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
            node_name=f"Load_A_cfg{i}", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(A_l3, l1_A, a_sz[i]))
        load_B = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
            node_name=f"Load_B_cfg{i}", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(B_l3, l1_B, b_sz[i]))
        mesh = meshes[shape]
        # CONFIGURE: programs every mode/shape CSR, then runs. -> GEMM_FULL_{CFG,RUN}
        gemm = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=gemm_core,
            node_name=f"Gemm_cfg{i}", kernel_name=spec["kernel"],
            kernel_args=make_gemm_args(mode, mesh, l1_A, l1_B, 0, l1_D, M, K, N, gen[i]))
        check = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
            node_name=f"Check_cfg{i}", kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                name=f"D{i}", golden_data_addr=D_l3, output_data_addr=l1_D,
                data_size=min(d_sz[i], 256)))
        # REUSE: same operands, same result, but reprograms ONLY the base addresses and
        # inherits the mode+shape CSRs this configure just wrote. Edged directly after it
        # on the same core, so what it inherits is unambiguous -- that ordering IS the
        # kernel's contract. It re-emits D identically, so the same golden checks it.
        # -> GEMM_MIN_{CFG,RUN}, the second cost curve of this (mode, shape).
        gemm_min = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=gemm_core,
            node_name=f"GemmMin_cfg{i}", kernel_name=MINIMAL_KERNEL,
            kernel_args=make_gemm_minimal_args(mode, mesh, l1_A, l1_B, 0, l1_D))
        check_min = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
            node_name=f"CheckMin_cfg{i}", kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                name=f"Dmin{i}", golden_data_addr=D_l3, output_data_addr=l1_D,
                data_size=min(d_sz[i], 256)))
        nodes = (load_A, load_B, gemm, check, gemm_min, check_min)
        on_cluster = [n for n in nodes if n.assigned_core_id != host_core]
        assert len(on_cluster) == CLUSTER_NODES_PER_CONFIG, (
            f"this config places {len(on_cluster)} nodes on a cluster core, but the L1 "
            f"budget is computed for CLUSTER_NODES_PER_CONFIG={CLUSTER_NODES_PER_CONFIG}. "
            f"Each of them costs {ALLOCS_PER_CLUSTER_NODE} x {L1_ALLOC_SLOT} B of L1 "
            f"(scratchpad + args) that is never freed, so the grid would be sized against "
            f"the wrong footprint and the sim would die at bingo_l1_alloc. Update "
            f"CLUSTER_NODES_PER_CONFIG.")
        for n in nodes:
            dfg.bingo_add_node(n)
        dfg.bingo_add_edge(load_A, load_B)
        dfg.bingo_add_edge(load_B, gemm)
        dfg.bingo_add_edge(gemm, check)
        dfg.bingo_add_edge(check, gemm_min)      # keeps the reuse strictly after its configure
        dfg.bingo_add_edge(gemm_min, check_min)
        if prev_check is not None:
            dfg.bingo_add_edge(prev_check, load_A)
        prev_check = check_min
    return dfg


def _check_fits_in_l1(gen, spec, l1_size_kb):
    """Assert the grid fits L1 -- against the constraint that ACTUALLY binds.

    The quantity to check is the PER-AXIS max, not the per-config total: _create_dfg
    allocates ONE l1_A/l1_B/l1_D sized at max_A / max_B / max_D over all configs, and
    nothing is ever freed, so the binary needs max_A + max_B + max_D live simultaneously.
    Checking each config in isolation (max over i of A_i+B_i+D_i) would let a grid whose
    configs each fit individually sail past this assert and then die at runtime in
    bingo_l1_alloc -> host_abort(BINGO_EXIT_OOM).
    """
    dbytes = _CTYPE_BYTES[spec["d_ctype"]]
    budget = l1_size_kb * 1024
    if not gen:
        return
    max_A = max(_frag(_pad64(len(gd["A"]))) for gd in gen)
    max_B = max(_frag(len(gd["B"])) for gd in gen)
    max_D = max(_frag(len(gd["D"]) * dbytes) for gd in gen)
    nodes = _node_alloc_bytes(len(gen))
    total = max_A + max_B + max_D + nodes
    assert total <= budget, (
        f"grid needs max_A+max_B+max_D+per-node = {max_A}+{max_B}+{max_D}+{nodes} "
        f"= {total} B > L1 budget {budget} B ({l1_size_kb} KiB). The three operand "
        f"buffers are ONE shared allocation each, sized at the max over all {len(gen)} "
        f"configs; on top of them every cluster node costs a scratchpad and an args "
        f"struct, {CLUSTER_NODES_PER_CONFIG} nodes x {ALLOCS_PER_CLUSTER_NODE} allocs x "
        f"{L1_ALLOC_SLOT} B per config. Nothing is freed, so all of it is live at once.")


def _get_args():
    p = argparse.ArgumentParser(description="per-precision GEMM cycle sweep")
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    p.add_argument("--configs_out", type=pathlib.Path, default=None)
    p.add_argument("--l1_size_kb", type=int, default=500)
    return p.parse_args()


def run(prec_name):
    """Entry point used by each gemm_psweep_<prec>_1cluster/main_bingo.py."""
    assert prec_name in PRECISIONS, f"unknown precision '{prec_name}'"
    spec = PRECISIONS[prec_name]
    args = _get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hwcfg = hjson.loads(f.read())

    # Build the grid for THIS precision, against the real L1 budget. It has to be
    # per-precision: how deep K can go depends on the operand bit widths (an i4 B operand
    # is half the bytes of i8, so it reaches twice the K), and the D buffer depends on the
    # output width. A single fixed config list shared by all precisions could not express
    # that.
    meshes = meshes_of(hwcfg)
    merged = dict(hwcfg)
    merged.update(spec["flags"])
    a_len, b_len, _c_len, d_len = _gemm_operand_widths(merged)
    print(f"[gemm_psweep:{prec_name}] operand widths a={a_len}b b={b_len}b d={d_len}b; "
          f"hwcfg exposes {len(meshes)} array shape(s)")
    configs = build_configs(meshes, a_len, b_len, d_len, args.l1_size_kb)

    gen = _gen_configs(configs, hwcfg, spec["flags"])

    # Drop configs the D-extension output serializer can't emit (would hang the
    # accelerator). Only affects the requant-int8 / int32->fp16 precisions at the
    # shapes/sizes whose narrowed output doesn't fill whole serializer beats
    # (here: shape 1 with M*N too small). Padding M/N to fill a beat would change
    # the measured GEMM, so these points are simply absent from the cycle LUT;
    # int32-output precisions keep every config. Filter configs+gen together so
    # configs.json (used by gather_gemm_luts.py) stays aligned with the DFG.
    keep = [_d_extension_emittable(spec, gd) for gd in gen]
    if not all(keep):
        dropped = [c for c, k in zip(configs, keep) if not k]
        print(f"[gemm_psweep:{prec_name}] dropped {len(dropped)} D-extension config(s) "
              f"whose narrowed output can't fill whole {_SERIAL_C_D_WIDTH}-bit serializer "
              f"beats (would hang): {dropped}")
        configs = [c for c, k in zip(configs, keep) if k]
        gen = [gd for gd, k in zip(gen, keep) if k]

    _check_fits_in_l1(gen, spec, args.l1_size_kb)

    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(_emit_header(gen, spec["d_ctype"]))
        print(f"Written data header: {args.data_h}")
    if args.configs_out is not None:
        # Carry the MESH per config, so gather_gemm_luts.py can build the op id
        # (gemm_<mode>_M<mu>K<ku>N<nu>[_minimal]) without re-reading the hwcfg -- the
        # sweep and the LUT then cannot disagree about which shape a number belongs to.
        with open(args.configs_out, "w") as f:
            json.dump({
                "precision": prec_name,
                "configs": configs,
                "meshes": {str(s): list(meshes[s]) for s in sorted({c[3] for c in configs})},
                "op_ids": [op_id(prec_name, meshes[c[3]]) for c in configs],
                "op_ids_minimal": [op_id(prec_name, meshes[c[3]], minimal=True) for c in configs],
            }, f, indent=2)
        print(f"Written configs list: {args.configs_out}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    dfg = _create_dfg(configs, gen, spec, platform, prec_name, meshes)
    dfg.bingo_compile_dfg(
        f"Single-Chip GEMM precision sweep ({prec_name})",
        args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["gemm_data.h"])
    ids = sorted({op_id(prec_name, meshes[c[3]]) for c in configs})
    print(f"Generated DFG: {prec_name} x {len(configs)} configs "
          f"-> {len(ids)} shape(s), each measured as configure + minimal: {ids}")
