#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Shared library for the per-op xDMA workloads. Each per-op workload dir
# (xdma_copy_1cluster, xdma_transpose_1cluster, ...) defines just an op name
# and a CONFIGS list of shapes, then calls run_op_workload(). For every config
# this builds a functional + cycle-characterization chain:
#
#     Load_input (iDMA L3->L1) -> xDMA op -> Store (L1->L3) -> Host check
#
# The xDMA op's BINGO_TRACE_XDMA_RUN_START/END markers give the per-config
# cycle count; after the sim, util/bingo_trace/bingo_trace.py emits one
# XDMA_RUN event per config (in CONFIGS order) which the LUT driver correlates
# with configs.json. Every config is also checked against a numpy golden, so a
# sweep doubles as the per-op functional test (one dir == one op).

import os
import sys
import json
import argparse
import pathlib
import hjson
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "../../"))
sys.path.append(f"{_ROOT}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{_ROOT}/util/sim")

from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from layout_convert import row_major_to_a, row_major_to_b, row_major_to_d  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdma6dArgs,
    SnaxBingoKernelXdmaTranspose2dArgs,
    SnaxBingoKernelXdmaSubmatrix2dArgs,
    SnaxBingoKernelXdmaExpand2dArgs,
    SnaxBingoKernelXdmaConcat2dArgs,
    SnaxBingoKernelXdmaPad2dArgs,
    SnaxBingoKernelXdmaGather2dArgs,
    SnaxBingoKernelXdmaElementwiseAddArgs,
    SnaxBingoKernelXdmaRowMajorToAArgs,
    SnaxBingoKernelXdmaAToRowMajorArgs,
    SnaxBingoKernelXdmaRowMajorToBArgs,
    SnaxBingoKernelXdmaBToRowMajorArgs,
    SnaxBingoKernelXdmaRowMajorToDArgs,
    SnaxBingoKernelXdmaDToRowMajorArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)

DMA_CORE = 1
HOST_CORE = 2

np.random.seed(42)


# ───────────────────────────── DFG builder helpers ────────────────────────
class Builder:
    """Per-workload helper that wires the recurring load/op/store/check chain.

    A per-op handler builds its op node(s) and calls store_check() to append
    the store + golden check. All nodes are serialized (each config after the
    previous one's check) so the XDMA_RUN events appear in CONFIGS order.
    """

    def __init__(self, dfg):
        self.dfg = dfg

    def l1(self, name, size):
        return BingoMemAlloc(name, size=size, mem_level="L1", chip_id=0, cluster_id=0)

    def sym(self, name, offset=0):
        return BingoMemSymbol(name, offset=offset)

    def idma_load(self, name, src, dst, size, after):
        node = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
            node_name=name, kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(src, dst, size))
        self.dfg.bingo_add_node(node)
        if after is not None:
            self.dfg.bingo_add_edge(after, node)
        return node

    def op(self, name, kernel_name, kernel_args, after):
        node = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
            node_name=name, kernel_name=kernel_name, kernel_args=kernel_args)
        self.dfg.bingo_add_node(node)
        if after is not None:
            self.dfg.bingo_add_edge(after, node)
        return node

    def store_check(self, name, l1_dst, op_node, golden_sym, out_bytes):
        l3_out = BingoMemAlloc(f"out_{name}", size=out_bytes, mem_level="L3")
        store = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
            node_name=f"Store_{name}", kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(l1_dst, l3_out, out_bytes))
        check = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
            node_name=f"Check_{name}", kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                self.sym(golden_sym), l3_out, out_bytes, name=name))
        self.dfg.bingo_add_node(store)
        self.dfg.bingo_add_node(check)
        self.dfg.bingo_add_edge(op_node, store)
        self.dfg.bingo_add_edge(store, check)
        return check


# ───────────────────────────── op handlers ────────────────────────────────
# Each handler: gen_data(cfg, i, ctx) -> list[str] of C defs (inputs+golden),
#               build(b, cfg, i, prev_chk) -> check node.
# Naming convention: input sym "in_<i>" (and "_b" for a 2nd input), golden
# sym "golden_<i>". elem-byte matrices use sequential uint8 values.

def _u8_matrix(rows, cols):
    return np.arange(rows * cols, dtype=np.uint8).reshape(rows, cols)


class CopyOp:
    name = "xdma_1d_copy"

    def in_bytes(self, c):
        return c["bytes"]

    def gen_data(self, c, i, ctx):
        n = c["bytes"]
        arr = (np.arange(n, dtype=np.uint8) & 0xFF)
        return [format_vector_definition("uint8_t", f"in_{i}", arr),
                format_vector_definition("uint8_t", f"golden_{i}", arr)]

    def build(self, b, c, i, prev):
        n = c["bytes"]
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", n)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_1d_copy",
                  SnaxBingoKernelXdma1dCopyArgs(l1s, l1d, n), load)
        return b.store_check(f"copy_cfg{i}", l1d, op, f"golden_{i}", n)


class Xdma6dOp:
    name = "xdma_6d"

    def gen_data(self, c, i, ctx):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        arr = (np.arange(rows * cols * elem, dtype=np.uint8) & 0xFF)
        return [format_vector_definition("uint8_t", f"in_{i}", arr),
                format_vector_definition("uint8_t", f"golden_{i}", arr)]

    def build(self, b, c, i, prev):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        n = rows * cols * elem
        row_bytes = cols * elem
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", n)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_6d",
                  SnaxBingoKernelXdma6dArgs(
                      l1s, l1d, row_bytes, row_bytes,
                      [8, row_bytes * 8], [row_bytes // 8, rows // 8],
                      [8, row_bytes * 8], [row_bytes // 8, rows // 8]), load)
        return b.store_check(f"xdma6d_cfg{i}", l1d, op, f"golden_{i}", n)


class TransposeOp:
    name = "xdma_transpose_2d"

    def gen_data(self, c, i, ctx):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        # elem>1 handled as elem consecutive bytes per element; build a byte
        # matrix of shape [rows, cols*elem] and transpose at element grain.
        base = (np.arange(rows * cols * elem, dtype=np.uint8) & 0xFF)
        mat = base.reshape(rows, cols, elem)
        golden = np.transpose(mat, (1, 0, 2)).reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", base),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        n = rows * cols * elem
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", n)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_transpose_2d",
                  SnaxBingoKernelXdmaTranspose2dArgs(l1s, l1d, rows, cols, elem), load)
        return b.store_check(f"transpose_cfg{i}", l1d, op, f"golden_{i}", n)


class SubmatrixOp:
    name = "xdma_submatrix_2d"

    def _slice(self, c):
        rows, cols = c["rows"], c["cols"]
        rs, re = c.get("rs", 0), c.get("re", 8)
        cs, ce = c.get("cs", 8), c.get("ce", cols)
        return rs, re, cs, ce

    def gen_data(self, c, i, ctx):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        rs, re, cs, ce = self._slice(c)
        mat = _u8_matrix(rows, cols * elem)
        golden = mat[rs:re, cs * elem:ce * elem].reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", mat.reshape(-1)),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        rs, re, cs, ce = self._slice(c)
        n = rows * cols * elem
        out = (re - rs) * (ce - cs) * elem
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", out)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_submatrix_2d",
                  SnaxBingoKernelXdmaSubmatrix2dArgs(
                      l1s, l1d, rows, cols, rs, re, cs, ce, elem), load)
        return b.store_check(f"submatrix_cfg{i}", l1d, op, f"golden_{i}", out)


class ExpandOp:
    name = "xdma_expand_2d"

    def gen_data(self, c, i, ctx):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        mat = _u8_matrix(rows, cols * elem)
        golden = np.tile(mat[0:1, :], (rows, 1)).reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", mat.reshape(-1)),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        n = rows * cols * elem
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", n)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_expand_2d",
                  SnaxBingoKernelXdmaExpand2dArgs(l1s, l1d, rows, cols, elem), load)
        return b.store_check(f"expand_cfg{i}", l1d, op, f"golden_{i}", n)


class ConcatOp:
    name = "xdma_concat_2d"

    def gen_data(self, c, i, ctx):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        mat = _u8_matrix(rows, cols * elem)  # reconstruct original via 2 halves
        return [format_vector_definition("uint8_t", f"in_{i}", mat.reshape(-1)),
                format_vector_definition("uint8_t", f"golden_{i}", mat.reshape(-1))]

    def build(self, b, c, i, prev):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        n = rows * cols * elem
        half = rows // 2
        half_b = half * cols * elem
        l1d = b.l1(f"l1d_{i}", n)
        # top half: read from in[0:half]
        l1_top = b.l1(f"l1top_{i}", half_b)
        load_t = b.idma_load(f"LoadTop_{i}", b.sym(f"in_{i}"), l1_top, half_b, prev)
        ctop = b.op(f"ConcatTop_{i}", "__snax_bingo_kernel_xdma_concat_2d",
                    SnaxBingoKernelXdmaConcat2dArgs(
                        l1_top, l1d, half, cols, rows, cols, 0, 0, elem), load_t)
        # bottom half: read from in[half:]
        l1_bot = b.l1(f"l1bot_{i}", half_b)
        load_b = b.idma_load(f"LoadBot_{i}", b.sym(f"in_{i}", offset=half_b),
                             l1_bot, half_b, ctop)
        cbot = b.op(f"ConcatBot_{i}", "__snax_bingo_kernel_xdma_concat_2d",
                    SnaxBingoKernelXdmaConcat2dArgs(
                        l1_bot, l1d, half, cols, rows, cols, 0, half, elem), load_b)
        return b.store_check(f"concat_cfg{i}", l1d, cbot, f"golden_{i}", n)


class PadOp:
    name = "xdma_pad_2d"

    def _pad(self, c):
        return (c.get("pt", 8), c.get("pb", 0), c.get("pl", 0), c.get("pr", 0))

    def gen_data(self, c, i, ctx):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        pt, pb, pl, pr = self._pad(c)
        mat = _u8_matrix(rows, cols * elem)
        golden = np.pad(mat, ((pt, pb), (pl * elem, pr * elem)),
                        mode="constant", constant_values=0).reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", mat.reshape(-1)),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        pt, pb, pl, pr = self._pad(c)
        n = rows * cols * elem
        out = (rows + pt + pb) * (cols + pl + pr) * elem
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", out)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_pad_2d",
                  SnaxBingoKernelXdmaPad2dArgs(l1s, l1d, rows, cols, pt, pb, pl, pr, elem), load)
        return b.store_check(f"pad_cfg{i}", l1d, op, f"golden_{i}", out)


class GatherOp:
    name = "xdma_gather_2d"

    def _g(self, c):
        return (c.get("start", 0), c.get("stride", 2))

    def gen_data(self, c, i, ctx):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        start, stride = self._g(c)
        mat = _u8_matrix(rows, cols * elem)
        golden = mat[start::stride, :].reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", mat.reshape(-1)),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        rows, cols, elem = c["rows"], c["cols"], c["elem"]
        start, stride = self._g(c)
        count = rows // stride
        n = rows * cols * elem
        out = count * cols * elem
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", out)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_gather_2d",
                  SnaxBingoKernelXdmaGather2dArgs(
                      l1s, l1d, rows, cols, count, start, stride, elem), load)
        return b.store_check(f"gather_cfg{i}", l1d, op, f"golden_{i}", out)


class ElementwiseAddOp:
    name = "xdma_elementwise_add"

    def gen_data(self, c, i, ctx):
        n_op, per = c["num_operands"], c["per_op"]
        ea = np.random.randint(-100000, 100000, size=(n_op, per), dtype=np.int32)
        golden = ea.sum(axis=0, dtype=np.int32)
        return [format_vector_definition("int32_t", f"in_{i}", ea.reshape(-1)),
                format_vector_definition("int32_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        n_op, per = c["num_operands"], c["per_op"]
        in_bytes = n_op * per * 4
        out_bytes = per * 4
        stride = per * 4
        l1s = b.l1(f"l1s_{i}", in_bytes)
        l1d = b.l1(f"l1d_{i}", out_bytes)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, in_bytes, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_elementwise_add",
                  SnaxBingoKernelXdmaElementwiseAddArgs(l1s, l1d, per, n_op, stride), load)
        return b.store_check(f"eltadd_cfg{i}", l1d, op, f"golden_{i}", out_bytes)


# ── VersaCore layout conversions (need meshRow/tileSize/meshCol from hwcfg) ──
def _mesh(ctx, c):
    array_shape = c.get("array_shape", ctx["array_shape"])
    u = ctx["hwcfg"]["snax_versacore_core_template"]["snax_acc_cfg"][0] \
            ["snax_versacore_spatial_unrolling"][0][array_shape]
    return int(u[0]), int(u[1]), int(u[2])


def _layout_dtype(elem):
    if elem == 1:
        return np.int8, -128, 127, "int8_t"
    if elem == 4:
        return np.int32, -1_000_000, 1_000_000, "int32_t"
    raise ValueError(f"layout conversion elem_bytes={elem} unsupported")


def make_layout_handlers():
    handlers = {}

    def reg(tag, name, kernel, args_cls, gen, sizes):
        h = type(f"Layout_{tag}", (), {})()
        h.tag = tag
        h.name = name
        h.kernel = kernel

        def gen_data(c, i, ctx, _gen=gen):
            elem = c["elem"]
            _, _, _, ctype = _layout_dtype(elem)
            src_arr, golden_arr = _gen(ctx, c)
            return [format_vector_definition(ctype, f"in_{i}", src_arr),
                    format_vector_definition(ctype, f"golden_{i}", golden_arr)]

        def build(b, c, i, prev, _sizes=sizes, _args=args_cls, _kernel=kernel, _tag=tag):
            n_src, n_dst, kwargs = _sizes(b, c)
            l1s = b.l1(f"l1s_{i}", n_src)
            l1d = b.l1(f"l1d_{i}", n_dst)
            load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n_src, prev)
            op = b.op(f"Op_{i}", _kernel, _args(src_addr=l1s, dst_addr=l1d, **kwargs), load)
            return b.store_check(f"{_tag}_cfg{i}", l1d, op, f"golden_{i}", n_dst)

        h.gen_data = gen_data
        h.build = build
        handlers[tag] = h

    return handlers, reg


def _register_layouts(handlers, reg, mesh_fn):
    # Each entry computes src/golden numpy arrays and the kernel kwargs/sizes.
    def dims(ctx, c):
        mR, tS, mC = mesh_fn(ctx, c)
        return c["M_T"], c["K_T"], c["N_T"], mR, tS, mC, c["elem"]

    # row_to_A
    def gen_row_to_a(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem)
        rm = np.random.randint(lo, hi, size=(M_T * mR, K_T * tS), dtype=dt)
        return rm.reshape(-1), row_major_to_a(rm, M_T, K_T, mR, tS)

    def sz_row_to_a(b, c):
        mR, tS, mC = _MESH_HOLDER["fn"](_MESH_HOLDER["ctx"], c)
        nbytes = c["M_T"] * mR * c["K_T"] * tS * c["elem"]
        return nbytes, nbytes, dict(M_T=c["M_T"], K_T=c["K_T"], meshRow=mR,
                                    tileSize=tS, elem_bytes=c["elem"])
    reg("row_to_a", "xdma_row_major_to_a", "__snax_bingo_kernel_xdma_row_major_to_a",
        SnaxBingoKernelXdmaRowMajorToAArgs, gen_row_to_a, sz_row_to_a)

    # A_to_row
    def gen_a_to_row(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem)
        rm = np.random.randint(lo, hi, size=(M_T * mR, K_T * tS), dtype=dt)
        return row_major_to_a(rm, M_T, K_T, mR, tS), rm.reshape(-1)

    def sz_a_to_row(b, c):
        mR, tS, mC = _MESH_HOLDER["fn"](_MESH_HOLDER["ctx"], c)
        nbytes = c["M_T"] * mR * c["K_T"] * tS * c["elem"]
        return nbytes, nbytes, dict(M_T=c["M_T"], K_T=c["K_T"], meshRow=mR,
                                    tileSize=tS, elem_bytes=c["elem"])
    reg("a_to_row", "xdma_a_to_row_major", "__snax_bingo_kernel_xdma_a_to_row_major",
        SnaxBingoKernelXdmaAToRowMajorArgs, gen_a_to_row, sz_a_to_row)

    # row_to_B
    def gen_row_to_b(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem)
        rm = np.random.randint(lo, hi, size=(K_T * tS, N_T * mC), dtype=dt)
        return rm.reshape(-1), row_major_to_b(rm, K_T, N_T, tS, mC)

    def sz_row_to_b(b, c):
        mR, tS, mC = _MESH_HOLDER["fn"](_MESH_HOLDER["ctx"], c)
        nbytes = c["K_T"] * tS * c["N_T"] * mC * c["elem"]
        return nbytes, nbytes, dict(K_T=c["K_T"], N_T=c["N_T"], tileSize=tS,
                                    meshCol=mC, elem_bytes=c["elem"])
    reg("row_to_b", "xdma_row_major_to_b", "__snax_bingo_kernel_xdma_row_major_to_b",
        SnaxBingoKernelXdmaRowMajorToBArgs, gen_row_to_b, sz_row_to_b)

    # B_to_row
    def gen_b_to_row(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem)
        rm = np.random.randint(lo, hi, size=(K_T * tS, N_T * mC), dtype=dt)
        return row_major_to_b(rm, K_T, N_T, tS, mC), rm.reshape(-1)

    def sz_b_to_row(b, c):
        mR, tS, mC = _MESH_HOLDER["fn"](_MESH_HOLDER["ctx"], c)
        nbytes = c["K_T"] * tS * c["N_T"] * mC * c["elem"]
        return nbytes, nbytes, dict(K_T=c["K_T"], N_T=c["N_T"], tileSize=tS,
                                    meshCol=mC, elem_bytes=c["elem"])
    reg("b_to_row", "xdma_b_to_row_major", "__snax_bingo_kernel_xdma_b_to_row_major",
        SnaxBingoKernelXdmaBToRowMajorArgs, gen_b_to_row, sz_b_to_row)

    # row_to_D
    def gen_row_to_d(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem)
        rm = np.random.randint(lo, hi, size=(M_T * mR, N_T * mC), dtype=dt)
        return rm.reshape(-1), row_major_to_d(rm, M_T, N_T, mR, mC)

    def sz_row_to_d(b, c):
        mR, tS, mC = _MESH_HOLDER["fn"](_MESH_HOLDER["ctx"], c)
        nbytes = c["M_T"] * mR * c["N_T"] * mC * c["elem"]
        return nbytes, nbytes, dict(M_T=c["M_T"], N_T=c["N_T"], meshRow=mR,
                                    meshCol=mC, elem_bytes=c["elem"])
    reg("row_to_d", "xdma_row_major_to_d", "__snax_bingo_kernel_xdma_row_major_to_d",
        SnaxBingoKernelXdmaRowMajorToDArgs, gen_row_to_d, sz_row_to_d)

    # D_to_row
    def gen_d_to_row(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem)
        rm = np.random.randint(lo, hi, size=(M_T * mR, N_T * mC), dtype=dt)
        return row_major_to_d(rm, M_T, N_T, mR, mC), rm.reshape(-1)

    def sz_d_to_row(b, c):
        mR, tS, mC = _MESH_HOLDER["fn"](_MESH_HOLDER["ctx"], c)
        nbytes = c["M_T"] * mR * c["N_T"] * mC * c["elem"]
        return nbytes, nbytes, dict(M_T=c["M_T"], N_T=c["N_T"], meshRow=mR,
                                    meshCol=mC, elem_bytes=c["elem"])
    reg("d_to_row", "xdma_d_to_row_major", "__snax_bingo_kernel_xdma_d_to_row_major",
        SnaxBingoKernelXdmaDToRowMajorArgs, gen_d_to_row, sz_d_to_row)


# Holder so the size-callbacks (which only get the Builder + cfg) can reach the
# mesh function + parsed hwcfg context resolved at run time.
_MESH_HOLDER = {"fn": None, "ctx": None}


def _build_registry():
    reg = {
        "copy": CopyOp(),
        "xdma_6d": Xdma6dOp(),
        "transpose": TransposeOp(),
        "submatrix": SubmatrixOp(),
        "expand": ExpandOp(),
        "concat": ConcatOp(),
        "pad": PadOp(),
        "gather": GatherOp(),
        "elementwise_add": ElementwiseAddOp(),
    }
    layout_handlers, layout_reg = make_layout_handlers()
    _register_layouts(layout_handlers, layout_reg, _mesh)
    reg.update(layout_handlers)
    return reg


REGISTRY = _build_registry()


# ───────────────────────────── runner ─────────────────────────────────────
def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    p.add_argument("--configs_out", type=pathlib.Path, default=None)
    return p.parse_args()


def run_op_workload(op, configs):
    """Build the single-op sweep workload (functional check + cycle markers)."""
    if op not in REGISTRY:
        raise ValueError(f"unknown op '{op}'. known: {sorted(REGISTRY)}")
    handler = REGISTRY[op]
    args = _get_args()
    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hwcfg = hjson.loads(f.read())
    ctx = {"hwcfg": hwcfg, "param": param,
           "array_shape": int(param.get("array_shape", 0))}
    _MESH_HOLDER["fn"] = _mesh
    _MESH_HOLDER["ctx"] = ctx

    # Emit data header (inputs + goldens for every config).
    if args.data_h is not None:
        defs = ["#include <stdint.h>"]
        for i, c in enumerate(configs):
            defs += handler.gen_data(c, i, ctx)
        with open(args.data_h, "w") as f:
            f.write("\n\n".join(defs))
        print(f"Written data header: {args.data_h}")

    # Dump configs.json for the LUT driver (CONFIGS order == XDMA_RUN order).
    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"op": handler.name,
                       "configs": [dict(c) for c in configs]}, f, indent=2)
        print(f"Written configs: {args.configs_out}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir,
                               args.output_offload_file_name):
        return
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"])

    b = Builder(dfg)
    prev = None
    for i, c in enumerate(configs):
        prev = handler.build(b, c, i, prev)

    os.makedirs(args.output_dir, exist_ok=True)
    dfg.bingo_compile_dfg(f"xDMA {op} sweep", args.output_dir,
                          args.output_offload_file_name,
                          extra_include_header_list=[f"{op}_data.h"])
    print(f"Generated {op} sweep: {len(configs)} configs")
