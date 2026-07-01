#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Shared library for the per-op xDMA workloads. Each per-op workload dir
# (xdma_transpose_1cluster, xdma_pad_1cluster, ...) defines just an op name
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
_ROOT = os.path.normpath(os.path.join(_THIS, "../../../"))  # util/sim/xdma -> repo root
sys.path.append(f"{_ROOT}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{_ROOT}/util/sim")

import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from layout_convert import row_major_to_a, row_major_to_b, row_major_to_d  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdmaPairwiseSwapArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdma6dArgs,
    SnaxBingoKernelXdmaTranspose2dArgs,
    SnaxBingoKernelXdmaSubmatrix2dArgs,
    SnaxBingoKernelXdmaExpand2dArgs,
    SnaxBingoKernelXdmaConcat2dArgs,
    SnaxBingoKernelXdmaPad2dArgs,
    SnaxBingoKernelXdmaGather2dArgs,
    SnaxBingoKernelXdmaElementwiseAddArgs,
    SnaxBingoKernelXdmaStreamReduceArgs,
    SnaxBingoKernelXdmaStreamMapArgs,
    SnaxBingoKernelXdmaStreamElementwiseArgs,
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
# sym "golden_<i>". elem_bytes-byte matrices use sequential uint8 values.

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
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        arr = (np.arange(rows * cols * elem_bytes, dtype=np.uint8) & 0xFF)
        return [format_vector_definition("uint8_t", f"in_{i}", arr),
                format_vector_definition("uint8_t", f"golden_{i}", arr)]

    def build(self, b, c, i, prev):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        n = rows * cols * elem_bytes
        row_bytes = cols * elem_bytes
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
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        # elem_bytes>1 handled as elem_bytes consecutive bytes per element; build a byte
        # matrix of shape [rows, cols*elem_bytes] and transpose at element grain.
        base = (np.arange(rows * cols * elem_bytes, dtype=np.uint8) & 0xFF)
        mat = base.reshape(rows, cols, elem_bytes)
        golden = np.transpose(mat, (1, 0, 2)).reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", base),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        n = rows * cols * elem_bytes
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", n)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_transpose_2d",
                  SnaxBingoKernelXdmaTranspose2dArgs(l1s, l1d, rows, cols, elem_bytes), load)
        return b.store_check(f"transpose_cfg{i}", l1d, op, f"golden_{i}", n)


class SwapOp:
    name = "idma_pairwise_swap"

    def gen_data(self, c, i, ctx):
        # Adjacent element-pair swap: dst[i] = src[i^1]. Build a byte matrix of
        # shape [num_pairs, 2, elem_bytes] and reverse the size-2 (pair) axis.
        num_elems, elem_bytes = c["num_elems"], c["elem_bytes"]
        base = (np.arange(num_elems * elem_bytes, dtype=np.uint8) & 0xFF)
        mat = base.reshape(num_elems // 2, 2, elem_bytes)
        golden = mat[:, ::-1, :].reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", base),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        num_elems, elem_bytes = c["num_elems"], c["elem_bytes"]
        n = num_elems * elem_bytes
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", n)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_idma_pairwise_swap",
                  SnaxBingoKernelIdmaPairwiseSwapArgs(l1s, l1d, num_elems, elem_bytes), load)
        return b.store_check(f"swap_cfg{i}", l1d, op, f"golden_{i}", n)


class SubmatrixOp:
    name = "xdma_submatrix_2d"

    def _slice(self, c):
        rows, cols = c["rows"], c["cols"]
        rs, re = c.get("rs", 0), c.get("re", 8)
        cs, ce = c.get("cs", 8), c.get("ce", cols)
        return rs, re, cs, ce

    def gen_data(self, c, i, ctx):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        rs, re, cs, ce = self._slice(c)
        mat = _u8_matrix(rows, cols * elem_bytes)
        golden = mat[rs:re, cs * elem_bytes:ce * elem_bytes].reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", mat.reshape(-1)),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        rs, re, cs, ce = self._slice(c)
        n = rows * cols * elem_bytes
        out = (re - rs) * (ce - cs) * elem_bytes
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", out)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_submatrix_2d",
                  SnaxBingoKernelXdmaSubmatrix2dArgs(
                      l1s, l1d, rows, cols, rs, re, cs, ce, elem_bytes), load)
        return b.store_check(f"submatrix_cfg{i}", l1d, op, f"golden_{i}", out)


class ExpandOp:
    name = "xdma_expand_2d"

    def gen_data(self, c, i, ctx):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        mat = _u8_matrix(rows, cols * elem_bytes)
        golden = np.tile(mat[0:1, :], (rows, 1)).reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", mat.reshape(-1)),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        n = rows * cols * elem_bytes
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", n)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_expand_2d",
                  SnaxBingoKernelXdmaExpand2dArgs(l1s, l1d, rows, cols, elem_bytes), load)
        return b.store_check(f"expand_cfg{i}", l1d, op, f"golden_{i}", n)


class ConcatOp:
    name = "xdma_concat_2d"

    def gen_data(self, c, i, ctx):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        mat = _u8_matrix(rows, cols * elem_bytes)  # reconstruct original via 2 halves
        return [format_vector_definition("uint8_t", f"in_{i}", mat.reshape(-1)),
                format_vector_definition("uint8_t", f"golden_{i}", mat.reshape(-1))]

    def build(self, b, c, i, prev):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        n = rows * cols * elem_bytes
        half = rows // 2
        half_b = half * cols * elem_bytes
        l1d = b.l1(f"l1d_{i}", n)
        # top half: read from in[0:half]
        l1_top = b.l1(f"l1top_{i}", half_b)
        load_t = b.idma_load(f"LoadTop_{i}", b.sym(f"in_{i}"), l1_top, half_b, prev)
        ctop = b.op(f"ConcatTop_{i}", "__snax_bingo_kernel_xdma_concat_2d",
                    SnaxBingoKernelXdmaConcat2dArgs(
                        l1_top, l1d, half, cols, rows, cols, 0, 0, elem_bytes), load_t)
        # bottom half: read from in[half:]
        l1_bot = b.l1(f"l1bot_{i}", half_b)
        load_b = b.idma_load(f"LoadBot_{i}", b.sym(f"in_{i}", offset=half_b),
                             l1_bot, half_b, ctop)
        cbot = b.op(f"ConcatBot_{i}", "__snax_bingo_kernel_xdma_concat_2d",
                    SnaxBingoKernelXdmaConcat2dArgs(
                        l1_bot, l1d, half, cols, rows, cols, 0, half, elem_bytes), load_b)
        return b.store_check(f"concat_cfg{i}", l1d, cbot, f"golden_{i}", n)


class PadOp:
    name = "xdma_pad_2d"

    def _pad(self, c):
        return (c.get("pt", 8), c.get("pb", 0), c.get("pl", 0), c.get("pr", 0))

    def gen_data(self, c, i, ctx):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        pt, pb, pl, pr = self._pad(c)
        mat = _u8_matrix(rows, cols * elem_bytes)
        golden = np.pad(mat, ((pt, pb), (pl * elem_bytes, pr * elem_bytes)),
                        mode="constant", constant_values=0).reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", mat.reshape(-1)),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        pt, pb, pl, pr = self._pad(c)
        n = rows * cols * elem_bytes
        out = (rows + pt + pb) * (cols + pl + pr) * elem_bytes
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", out)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_pad_2d",
                  SnaxBingoKernelXdmaPad2dArgs(l1s, l1d, rows, cols, pt, pb, pl, pr, elem_bytes), load)
        return b.store_check(f"pad_cfg{i}", l1d, op, f"golden_{i}", out)


class GatherOp:
    name = "xdma_gather_2d"

    def _g(self, c):
        return (c.get("start", 0), c.get("stride", 2))

    def gen_data(self, c, i, ctx):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        start, stride = self._g(c)
        mat = _u8_matrix(rows, cols * elem_bytes)
        golden = mat[start::stride, :].reshape(-1)
        return [format_vector_definition("uint8_t", f"in_{i}", mat.reshape(-1)),
                format_vector_definition("uint8_t", f"golden_{i}", golden)]

    def build(self, b, c, i, prev):
        rows, cols, elem_bytes = c["rows"], c["cols"], c["elem_bytes"]
        start, stride = self._g(c)
        count = rows // stride
        n = rows * cols * elem_bytes
        out = count * cols * elem_bytes
        l1s = b.l1(f"l1s_{i}", n)
        l1d = b.l1(f"l1d_{i}", out)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1s, n, prev)
        op = b.op(f"Op_{i}", "__snax_bingo_kernel_xdma_gather_2d",
                  SnaxBingoKernelXdmaGather2dArgs(
                      l1s, l1d, rows, cols, count, start, stride, elem_bytes), load)
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


# ── FP16 streaming-SIMD primitive: softmax (reduce + map+tap + map[+quant]) ──
# Decomposes one softmax row onto the 3 new stream kernels and validates the
# fp16 output to tolerance. Exercises ALL the new capabilities in one chain:
#   T1  reduce(MAX)                       FULL   -> establishes the shared AGU
#   T2  map(EXP, b=-max) + reduce(ADD|TAP) STICKY -> exp row + Σexp, 2-ext fusion
#   T3  map(LINEAR, a=inv_sum)            STICKY -> softmax probs (checked, fp16)
#   Tq  map(LINEAR) + Fp16ToInt8          STICKY -> fused FP16->INT8 (path exercised)
# The host scalars (-max, inv_sum) are baked into the args (as the standalone
# snax-xdma-softmax app precomputes them in data.h); on real HeMAiA they come
# from host nodes — that wiring is the bingo_framework next phase. int8 numeric
# correctness is covered by the standalone app (no int8-tolerance check here).

def _f32bits(x):
    return int(np.float32(x).view(np.uint32))


def _softmax_ref(beats, i):
    """Deterministic per-config softmax reference, mimicking the HW fp16 datapath.
    Returns (x_fp16, y_fp16, yq_int8, neg_max_bits, inv_sum_bits, inv_scale_bits).

    The input is PEAKY (a few large entries on a small-uniform background) so the
    dominant probabilities stay O(0.1-0.7) regardless of N — otherwise large-N
    softmax flattens to ~1/N and an absolute-tolerance check passes even garbage.
    """
    n = beats * 32  # 64 B beat = 32 fp16
    rng = np.random.RandomState(20240601 + i)
    x = rng.uniform(-2.0, 2.0, size=n).astype(np.float32)
    for pos, val in ((0, 6.0), (n // 3, 5.0), (2 * n // 3, 4.5)):
        x[pos] = val
    x = x.astype(np.float16)
    xf = x.astype(np.float32)
    m = np.float16(xf.max())                                   # HW T1: fp16 row max
    e16 = np.exp(xf - np.float32(m)).astype(np.float16)        # HW T2: fp16 exp
    s = np.float32(e16.astype(np.float32).sum())               # HW T2 tap: Σ over fp16
    inv_sum = np.float32(1.0) / s                              # host reciprocal
    y = (e16.astype(np.float32) * inv_sum).astype(np.float16)  # HW T3: normalize
    inv_scale = np.float32(127.0)                              # probs in [0,1] -> int8
    yq = np.clip(np.rint(y.astype(np.float32) * inv_scale), -128, 127).astype(np.int8)
    return x, y, yq, _f32bits(-np.float32(m)), _f32bits(inv_sum), _f32bits(inv_scale)


class SoftmaxOp:
    name = "softmax"

    def gen_data(self, c, i, ctx):
        x, y, _yq, _nm, _is, _isc = _softmax_ref(c["beats"], i)
        return [format_vector_definition("uint16_t", f"in_{i}", x.view(np.uint16)),
                format_vector_definition("uint16_t", f"golden_{i}", y.view(np.uint16))]

    def _store_check_fp16(self, b, name, l1_dst, op_node, golden_sym, n_elems, tol):
        out_bytes = n_elems * 2
        l3 = BingoMemAlloc(f"out_{name}", size=out_bytes, mem_level="L3")
        store = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
            node_name=f"Store_{name}", kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(l1_dst, l3, out_bytes))
        check = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
            node_name=f"Check_{name}", kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                b.sym(golden_sym), l3, name=name,
                check_type=2, num_elements=n_elems, tolerance=tol))  # 2 = FP16_TOL
        b.dfg.bingo_add_node(store)
        b.dfg.bingo_add_node(check)
        b.dfg.bingo_add_edge(op_node, store)
        b.dfg.bingo_add_edge(store, check)
        return check

    def build(self, b, c, i, prev):
        beats = c["beats"]
        n = beats * 32
        row_b = beats * 64
        x, y, yq, neg_max_bits, inv_sum_bits, inv_scale_bits = _softmax_ref(beats, i)
        l1_x   = b.l1(f"smx_x_{i}", row_b)
        l1_max = b.l1(f"smx_max_{i}", 64)
        l1_exp = b.l1(f"smx_exp_{i}", (beats + 1) * 64)
        l1_out = b.l1(f"smx_out_{i}", row_b)
        l1_i8  = b.l1(f"smx_i8_{i}", n)
        load = b.idma_load(f"Load_{i}", b.sym(f"in_{i}"), l1_x, row_b, prev)
        # T1 max — FULL config establishes the AGU shape the STICKY ops reuse.
        t1 = b.op(f"Max_{i}", "__snax_bingo_kernel_xdma_stream_reduce",
                  SnaxBingoKernelXdmaStreamReduceArgs(l1_x, l1_max, beats, op=0,
                      red_tap=0, csr_mode=0, dst_bound0=1), load)
        # T2 exp(x-max) + Σexp fused (StreamMap EXP -||> StreamReduce ADD|TAP), STICKY.
        t2 = b.op(f"ExpSum_{i}", "__snax_bingo_kernel_xdma_stream_map",
                  SnaxBingoKernelXdmaStreamMapArgs(l1_x, l1_exp, beats, func=1,
                      b_f32bits=neg_max_bits, tap_reduce_op=(1 | 0x100),
                      csr_mode=1, dst_bound0=beats + 1), t1)
        # T3 out = inv_sum * exp (StreamMap LINEAR), STICKY.
        t3 = b.op(f"Norm_{i}", "__snax_bingo_kernel_xdma_stream_map",
                  SnaxBingoKernelXdmaStreamMapArgs(l1_exp, l1_out, beats, func=0,
                      a_f32bits=inv_sum_bits, csr_mode=1, dst_bound0=beats), t2)
        chk = self._store_check_fp16(b, f"softmax_cfg{i}", l1_out, t3, f"golden_{i}", n, 0.02)
        # Tq fused FP16->INT8 quant — exercises the Fp16ToInt8 datapath (leaf, no check).
        tq = b.op(f"Quant_{i}", "__snax_bingo_kernel_xdma_stream_map",
                  SnaxBingoKernelXdmaStreamMapArgs(l1_exp, l1_i8, beats, func=0,
                      a_f32bits=inv_sum_bits, csr_mode=1, dst_bound0=beats // 2,
                      out_dtype=1, inv_scale_f32bits=inv_scale_bits), chk)
        return tq


# ── VersaCore layout conversions (need meshRow/tileSize/meshCol from hwcfg) ──
def _mesh(ctx, c):
    array_shape = c.get("array_shape", ctx["array_shape"])
    u = ctx["hwcfg"]["snax_versacore_core_template"]["snax_acc_cfg"][0] \
            ["snax_versacore_spatial_unrolling"][0][array_shape]
    return int(u[0]), int(u[1]), int(u[2])


def _layout_dtype(elem_bytes):
    if elem_bytes == 1:
        return np.int8, -128, 127, "int8_t"
    if elem_bytes == 4:
        return np.int32, -1_000_000, 1_000_000, "int32_t"
    raise ValueError(f"layout conversion elem_bytes={elem_bytes} unsupported")


def make_layout_handlers():
    handlers = {}

    def reg(tag, name, kernel, args_cls, gen, sizes):
        h = type(f"Layout_{tag}", (), {})()
        h.tag = tag
        h.name = name
        h.kernel = kernel

        def gen_data(c, i, ctx, _gen=gen):
            elem_bytes = c["elem_bytes"]
            _, _, _, ctype = _layout_dtype(elem_bytes)
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


def _register_layouts(handlers, reg):
    # Each entry computes src/golden numpy arrays and the kernel kwargs/sizes.
    def dims(ctx, c):
        mR, tS, mC = _mesh(ctx, c)
        return c["M_T"], c["K_T"], c["N_T"], mR, tS, mC, c["elem_bytes"]

    # row_to_A
    def gen_row_to_a(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem_bytes = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem_bytes)
        rm = np.random.randint(lo, hi, size=(M_T * mR, K_T * tS), dtype=dt)
        return rm.reshape(-1), row_major_to_a(rm, M_T, K_T, mR, tS)

    def sz_row_to_a(b, c):
        mR, tS, mC = _mesh(_MESH_HOLDER["ctx"], c)
        nbytes = c["M_T"] * mR * c["K_T"] * tS * c["elem_bytes"]
        return nbytes, nbytes, dict(M_T=c["M_T"], K_T=c["K_T"], meshRow=mR,
                                    tileSize=tS, elem_bytes=c["elem_bytes"])
    reg("row_to_a", "xdma_row_major_to_a", "__snax_bingo_kernel_xdma_row_major_to_a",
        SnaxBingoKernelXdmaRowMajorToAArgs, gen_row_to_a, sz_row_to_a)

    # A_to_row
    def gen_a_to_row(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem_bytes = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem_bytes)
        rm = np.random.randint(lo, hi, size=(M_T * mR, K_T * tS), dtype=dt)
        return row_major_to_a(rm, M_T, K_T, mR, tS), rm.reshape(-1)

    def sz_a_to_row(b, c):
        mR, tS, mC = _mesh(_MESH_HOLDER["ctx"], c)
        nbytes = c["M_T"] * mR * c["K_T"] * tS * c["elem_bytes"]
        return nbytes, nbytes, dict(M_T=c["M_T"], K_T=c["K_T"], meshRow=mR,
                                    tileSize=tS, elem_bytes=c["elem_bytes"])
    reg("a_to_row", "xdma_a_to_row_major", "__snax_bingo_kernel_xdma_a_to_row_major",
        SnaxBingoKernelXdmaAToRowMajorArgs, gen_a_to_row, sz_a_to_row)

    # row_to_B
    def gen_row_to_b(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem_bytes = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem_bytes)
        rm = np.random.randint(lo, hi, size=(K_T * tS, N_T * mC), dtype=dt)
        return rm.reshape(-1), row_major_to_b(rm, K_T, N_T, tS, mC)

    def sz_row_to_b(b, c):
        mR, tS, mC = _mesh(_MESH_HOLDER["ctx"], c)
        nbytes = c["K_T"] * tS * c["N_T"] * mC * c["elem_bytes"]
        return nbytes, nbytes, dict(K_T=c["K_T"], N_T=c["N_T"], tileSize=tS,
                                    meshCol=mC, elem_bytes=c["elem_bytes"])
    reg("row_to_b", "xdma_row_major_to_b", "__snax_bingo_kernel_xdma_row_major_to_b",
        SnaxBingoKernelXdmaRowMajorToBArgs, gen_row_to_b, sz_row_to_b)

    # B_to_row
    def gen_b_to_row(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem_bytes = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem_bytes)
        rm = np.random.randint(lo, hi, size=(K_T * tS, N_T * mC), dtype=dt)
        return row_major_to_b(rm, K_T, N_T, tS, mC), rm.reshape(-1)

    def sz_b_to_row(b, c):
        mR, tS, mC = _mesh(_MESH_HOLDER["ctx"], c)
        nbytes = c["K_T"] * tS * c["N_T"] * mC * c["elem_bytes"]
        return nbytes, nbytes, dict(K_T=c["K_T"], N_T=c["N_T"], tileSize=tS,
                                    meshCol=mC, elem_bytes=c["elem_bytes"])
    reg("b_to_row", "xdma_b_to_row_major", "__snax_bingo_kernel_xdma_b_to_row_major",
        SnaxBingoKernelXdmaBToRowMajorArgs, gen_b_to_row, sz_b_to_row)

    # row_to_D
    def gen_row_to_d(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem_bytes = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem_bytes)
        rm = np.random.randint(lo, hi, size=(M_T * mR, N_T * mC), dtype=dt)
        return rm.reshape(-1), row_major_to_d(rm, M_T, N_T, mR, mC)

    def sz_row_to_d(b, c):
        mR, tS, mC = _mesh(_MESH_HOLDER["ctx"], c)
        nbytes = c["M_T"] * mR * c["N_T"] * mC * c["elem_bytes"]
        return nbytes, nbytes, dict(M_T=c["M_T"], N_T=c["N_T"], meshRow=mR,
                                    meshCol=mC, elem_bytes=c["elem_bytes"])
    reg("row_to_d", "xdma_row_major_to_d", "__snax_bingo_kernel_xdma_row_major_to_d",
        SnaxBingoKernelXdmaRowMajorToDArgs, gen_row_to_d, sz_row_to_d)

    # D_to_row
    def gen_d_to_row(ctx, c):
        M_T, K_T, N_T, mR, tS, mC, elem_bytes = dims(ctx, c)
        dt, lo, hi, _ = _layout_dtype(elem_bytes)
        rm = np.random.randint(lo, hi, size=(M_T * mR, N_T * mC), dtype=dt)
        return row_major_to_d(rm, M_T, N_T, mR, mC), rm.reshape(-1)

    def sz_d_to_row(b, c):
        mR, tS, mC = _mesh(_MESH_HOLDER["ctx"], c)
        nbytes = c["M_T"] * mR * c["N_T"] * mC * c["elem_bytes"]
        return nbytes, nbytes, dict(M_T=c["M_T"], N_T=c["N_T"], meshRow=mR,
                                    meshCol=mC, elem_bytes=c["elem_bytes"])
    reg("d_to_row", "xdma_d_to_row_major", "__snax_bingo_kernel_xdma_d_to_row_major",
        SnaxBingoKernelXdmaDToRowMajorArgs, gen_d_to_row, sz_d_to_row)


# Holder so the size-callbacks (which only get the Builder + cfg) can reach the
# parsed hwcfg context resolved at run time.
_MESH_HOLDER = {"ctx": None}


def _build_registry():
    reg = {
        "copy": CopyOp(),
        "xdma_6d": Xdma6dOp(),
        "transpose": TransposeOp(),
        "swap": SwapOp(),
        "submatrix": SubmatrixOp(),
        "expand": ExpandOp(),
        "concat": ConcatOp(),
        "pad": PadOp(),
        "gather": GatherOp(),
        "elementwise_add": ElementwiseAddOp(),
        "softmax": SoftmaxOp(),
    }
    layout_handlers, layout_reg = make_layout_handlers()
    _register_layouts(layout_handlers, layout_reg)
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
