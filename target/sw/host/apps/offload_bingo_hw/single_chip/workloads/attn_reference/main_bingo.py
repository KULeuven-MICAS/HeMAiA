#!/usr/bin/env python3
"""
=============================================================================
attn_reference — Clean Reference Attention Workload for HeMAiA
=============================================================================

This is the hand-crafted reference implementation of a single attention layer
built from scratch using the proven patterns from the attn_test_* test suite.
It is intentionally modular so that a future higher-level code emitter
(ONNX/IR → main_bingo.py) can produce near-identical output by calling the
same pattern helpers.

Attention computation:

    X  --quantize--> X_int8
    X_int8 --proj_q--> Q_int32 --dequant--> Q_fp32
    X_int8 --proj_k--> K_int32 --dequant--> K_fp32
    X_int8 --proj_v--> V_int32 --dequant--> V_fp32
    (Q_fp32, K_fp32) --qk_matmul--> QK_int32 --dequant(·1/√d)--> QK_scaled_fp32
    QK_scaled_fp32 --softmax-->                                   attn_weights_fp32
    (attn_weights_fp32, V_fp32) --av_matmul--> AV_int32 --dequant--> AV_fp32
    AV_fp32 --proj_o--> Y_int32 --dequant--> Y_fp32

Each arrow is one of 3 patterns:
  - emit_quantize_pattern     (fp32 → int8 on host)
  - emit_ksplit_gemm_pattern  (int8 x int8 → int32 across 4 clusters,
                               int32_add reduction, int32 → fp32 dequant,
                               xdma_d_to_row_major reshape)
  - emit_softmax_pattern      (row-wise fp32 softmax on host)

Layout note: VersaCore writes D in block layout [M_T, N_T, meshRow, meshCol].
For N_T > 1 (e.g. qk_matmul) this differs from row-major, so the K-split
GEMM pattern emits an xdma_d_to_row_major kernel after dequantize to
produce a row-major buffer that matches Python goldens and serves as the
correct input for downstream ops like softmax. With the generalized
layout-convert kernels (snax_kernel_lib.h), this reshape runs on the
xDMA AGU for every VersaCore array_shape (the canonical 32x4x32 INT32
case takes Path 2 — meshRow split as r_outerx8). The L3 fp32 src is
auto-IDMA-staged to local L1 by `xdma_layout_stage_in` (snax_xdma_lib.h);
no host-side prefetch is needed.

Structure of this file:
  SECTION 1: Imports + HW params
  SECTION 2: Dataclasses (HwParams, KSplitGemmSpec)
  SECTION 3: Pattern helpers (quantize, K-split GEMM, softmax)
  SECTION 4: Attention spec factory
  SECTION 5: main() composition

In DEBUG mode every pattern emits check_result nodes that compare runtime
buffers against goldens from attention_datagen.py. In RUN mode only the
final checks are emitted (smaller DFG, faster simulation).
"""

from __future__ import annotations

# ─── SECTION 1: Imports + HW config ───────────────────────────────────────
import os
import sys
import re
import argparse
import pathlib
import hjson
from dataclasses import dataclass
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from attention_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelXdmaDToRowMajorArgs,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelFp32QuantizeArgs,
    HostBingoKernelInt32DequantizeArgs,
    HostBingoKernelFp32SoftmaxArgs,
    HostBingoKernelInt32AddArgs,
    BINGO_CHECK_TYPE_BYTE_EXACT,
    BINGO_CHECK_TYPE_FP32_TOL,
)

# Core IDs within a cluster (fixed by the HeMAiA architecture).
GEMM_CORE = 0  # VersaCore (int8 GEMM accelerator)
DMA_CORE  = 1  # iDMA (L3 ↔ L1 transfers)
HOST_CORE = 2  # CVA6 + Ara (fp32 ops, quantize/dequant/softmax, check, add)

# Cap the amount of data each check_result compares, for faster simulation.
# The GEMMs still compute full-size buffers; only the HOST check_result loop
# is shortened (which also bounds the number of mismatch printf lines the
# UART emits on failure — a major sim-time contributor).
#   byte-exact mode:  min(full_bytes, CHECK_BYTES_LIMIT) bytes compared
#   fp32-tol mode:    min(full_fp32_elems, CHECK_BYTES_LIMIT // 4) elements
#   fp16-tol mode:    min(full_fp16_elems, CHECK_BYTES_LIMIT // 2) elements
# Set to None (or a very large int) to verify the full buffer.
CHECK_BYTES_LIMIT = 128


def _capped_bytes(full_bytes: int) -> int:
    """Byte-exact check_result data_size, capped by CHECK_BYTES_LIMIT."""
    if CHECK_BYTES_LIMIT is None:
        return full_bytes
    return min(full_bytes, CHECK_BYTES_LIMIT)


def _capped_elems(full_elems: int, elem_bytes: int) -> int:
    """FP check_result num_elements, capped so total bytes <= CHECK_BYTES_LIMIT."""
    if CHECK_BYTES_LIMIT is None:
        return full_elems
    max_elems = max(1, CHECK_BYTES_LIMIT // elem_bytes)
    return min(full_elems, max_elems)


# ─── SECTION 2: Dataclasses ────────────────────────────────────────────────

@dataclass
class HwParams:
    """All dimensions + platform info needed to emit the attention workload."""
    # Spatial unrolling (from hwcfg)
    meshRow: int
    tileSize: int
    meshCol: int
    array_shape_idx: int
    # Attention shape (from params.hjson)
    seq_len: int
    d_model: int
    d_head: int
    # Platform (from occamy.h + RTL config)
    num_chiplets: int
    num_clusters_per_chiplet: int
    num_cores_per_cluster: int
    chiplet_ids: list


@dataclass
class KSplitGemmSpec:
    """Specification for one K-split GEMM stage.

    The emit_ksplit_gemm_pattern() helper reads this spec and produces a
    complete DFG sub-graph (loads, GEMMs, stores, int32_add reduction,
    dequantize, checks). All buffer/symbol names follow the `name`-based
    convention so goldens from attention_datagen.py resolve by construction.

    Fields:
        name:              logical GEMM name ('proj_q', 'qk_matmul', ...).
        M_T, K_T, N_T:     full tile dims BEFORE K-splitting.
        k_split:           total number of K-chunks.
        chunks_per_cluster: chunks_per_cluster[cid] = list of k-chunk indices
                           assigned to cluster cid. For k_split == num_clusters,
                           each cluster gets one chunk. For k_split > num_clusters
                           (e.g. proj_o with k_split=8 on 4 clusters), each
                           cluster handles multiple chunks and accumulates
                           locally via accumPrevC on subsequent GEMMs.
        scale_sym_name:    C symbol name of the combined_scale[1] array used
                           by dequantize.
        check_tolerance:   fp32 absolute-tolerance for the final dequant check.
    """
    name: str
    M_T: int
    K_T: int
    N_T: int
    k_split: int
    chunks_per_cluster: list
    scale_sym_name: str
    check_tolerance: float = 1e-5


# ─── SECTION 3: Pattern helpers ────────────────────────────────────────────
#
# Each helper takes (dfg, mem, hw, ...) and:
#   1. Allocates the memory handles it needs (adds them to `mem` dict).
#   2. Creates BingoNodes and wires internal edges.
#   3. Wires external input-dependency edges (`input_dep` param).
#   4. Returns the terminal node for chaining by the caller.
# ───────────────────────────────────────────────────────────────────────────


def _sym(mem: dict, name: str) -> BingoMemSymbol:
    """Register-or-fetch a BingoMemSymbol by name (data-header golden symbol)."""
    if name not in mem:
        mem[name] = BingoMemSymbol(name)
    return mem[name]


def _l3(mem: dict, name: str, size: int) -> BingoMemAlloc:
    """Register-or-fetch an L3 runtime buffer allocation."""
    if name not in mem:
        mem[name] = BingoMemAlloc(name, size=size, mem_level="L3")
    return mem[name]


def _l1(mem: dict, name: str, size: int, cluster_id: int) -> BingoMemAlloc:
    """Register-or-fetch an L1 runtime buffer (per-cluster)."""
    if name not in mem:
        mem[name] = BingoMemAlloc(name, size=size, mem_level="L1",
                                  chip_id=0, cluster_id=cluster_id)
    return mem[name]


def _add_edge_list(dfg, sources, dest):
    """Add edges from each source to dest. `sources` may be None, a single
    BingoNode, or a list of BingoNodes."""
    if sources is None:
        return
    if not isinstance(sources, (list, tuple)):
        sources = [sources]
    for s in sources:
        if s is not None:
            dfg.bingo_add_edge(s, dest)


# ─── Pattern A: Quantize (FP32 → INT8) ────────────────────────────────────

def emit_quantize_pattern(dfg: BingoDFG, mem: dict, hw: HwParams, *,
                          input_fp32_sym: str = "fp32_X",
                          output_int8_buf: str = "X_int8_buf",
                          scale_out_buf: str = "scale_X_l3",
                          golden_int8_sym: str = "golden_int8_X_rm",
                          num_elements: Optional[int] = None,
                          check_name: str = "quantize_X",
                          debug: bool = True,
                          input_dep=None):
    """Emit FP32 → INT8 quantize on the CVA6 host.

    DFG: [optional input_dep] → Quantize → (debug: Check_int8)

    Allocates:
      - output L3 buffer (`output_int8_buf`) of size `num_elements` bytes
      - scale output L3 buffer (`scale_out_buf`) of 4 bytes

    Returns the terminal node (check node if debug, else quantize node).
    """
    if num_elements is None:
        num_elements = hw.seq_len * hw.d_model

    fp32_in  = _sym(mem, input_fp32_sym)
    int8_out = _l3(mem, output_int8_buf, size=num_elements)  # int8 → 1 byte each
    scale_out = _l3(mem, scale_out_buf, size=4)

    node_q = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Quantize_X",
        kernel_name="__host_bingo_kernel_fp32_quantize",
        kernel_args=HostBingoKernelFp32QuantizeArgs(
            input_addr=fp32_in, output_addr=int8_out,
            scale_out_addr=scale_out, num_elements=num_elements,
        ),
    )
    dfg.bingo_add_node(node_q)
    _add_edge_list(dfg, input_dep, node_q)

    if not debug:
        return node_q

    node_check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name=f"Check_{check_name}",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=_sym(mem, golden_int8_sym),
            output_data_addr=int8_out,
            # int8 → 1 byte per element → num_elements IS bytes
            data_size=_capped_bytes(num_elements),
            name=check_name,
            check_type=BINGO_CHECK_TYPE_BYTE_EXACT,
        ),
    )
    dfg.bingo_add_node(node_check)
    dfg.bingo_add_edge(node_q, node_check)
    return node_check


# ─── Pattern B: K-split GEMM (int8 x int8 → int32 → fp32) ─────────────────

def emit_ksplit_gemm_pattern(dfg: BingoDFG, mem: dict, hw: HwParams,
                             spec: KSplitGemmSpec, *,
                             debug: bool = True,
                             input_dep=None):
    """Emit a full K-split GEMM stage. The produced DFG sub-graph is:

        For each cluster cid in spec.chunks_per_cluster:
          For each k_idx in chunks_per_cluster[cid]:
            Load_A_k{k_idx} → Load_B_k{k_idx} → GEMM_cid_chunk{k_idx}
          Store_D_c{cid}
          (debug) Check_D_partial_c{cid}

        Then on host / DMA cores, serially:
          int32_add c0+c1 → Sum_c0_c1 → (debug) Check
          int32_add +c2   → Sum_c0_c1_c2 → (debug) Check
          ...
          Dequantize (host)      → {name}_fp32_l3      (BLOCK layout)
          d_to_row_major (DMA)   → {name}_fp32_rm_l3   (ROW-MAJOR layout)
          (debug) Check_fp32 with FP32_TOL tolerance on row-major buffer

    BASELINE constraint: exactly ONE K-chunk per cluster. Asserted at
    runtime. No HW accumulation modes (accumPrevC, addNonZeroC) are used —
    every GEMM is a straight D = A*B (C_addr=0, accumPrevC=0). If multi-
    chunk accumulation is ever needed, emit separate per-chunk L1_D buffers
    and reduce on the host via int32_add (same pattern as inter-cluster
    reduction below) — NOT via the HW's addNonZeroC mode.

    The xdma_d_to_row_major reshape is essential for N_T>1 GEMMs (e.g.
    qk_matmul) where block layout ≠ row-major layout; for N_T=1 it's
    a uniform step that effectively copies the data.

    Returns the terminal node (check_fp32 in debug, reshape otherwise).
    The returned node's output is {name}_fp32_rm_l3 (row-major).
    """
    name = spec.name

    # ─── Per-tile and full byte sizes ─────────────────────────────────────
    M_T, N_T = spec.M_T, spec.N_T
    K_T_tile = spec.K_T // spec.k_split
    A_tile_bytes = M_T * K_T_tile * hw.meshRow * hw.tileSize   # int8
    B_tile_bytes = K_T_tile * N_T * hw.tileSize * hw.meshCol    # int8
    D_bytes = M_T * N_T * hw.meshRow * hw.meshCol * 4           # int32
    D_num_elements = M_T * N_T * hw.meshRow * hw.meshCol
    FP32_bytes = D_bytes  # same element count as int32

    # ─── Allocate per-cluster memory handles ──────────────────────────────
    sorted_cids = sorted(spec.chunks_per_cluster.keys())
    for cid in sorted_cids:
        _l1(mem, f"l1_A_{name}_c{cid}", A_tile_bytes, cid)
        _l1(mem, f"l1_B_{name}_c{cid}", B_tile_bytes, cid)
        _l1(mem, f"l1_D_{name}_c{cid}", D_bytes, cid)
        _l3(mem, f"{name}_D_partial_c{cid}_l3", D_bytes)

    # Reduction-sum L3 buffers (one per step)
    for i in range(1, len(sorted_cids)):
        label = "_".join(f"c{c}" for c in sorted_cids[:i + 1])
        _l3(mem, f"{name}_sum_{label}_l3", D_bytes)

    # Final dequantized fp32 output (BLOCK layout — direct from HW dequant)
    _l3(mem, f"{name}_fp32_l3", FP32_bytes)
    # Row-major fp32 output (after xdma_d_to_row_major reshape).
    # This is the "logically correct" layout that the check and downstream
    # ops (softmax, next stage's A) consume.
    _l3(mem, f"{name}_fp32_rm_l3", FP32_bytes)

    # ─── Per-cluster: load → gemm[s] → store  [→ check_partial] ──────────
    store_nodes = {}   # cid → store node (for wiring reduction)
    check_partial_nodes = {}   # cid → check (for optional wiring)

    # BASELINE: exactly ONE K-chunk per cluster — asserted here so future
    # specs with multi-chunk-per-cluster fail loudly rather than silently
    # producing wrong results. If multi-chunk is ever needed, emit the chunks
    # as separate per-chunk L1_D buffers and reduce them on the host via
    # int32_add — NOT via the HW's addNonZeroC/accumPrevC modes.
    for cid, kchunks in spec.chunks_per_cluster.items():
        assert len(kchunks) == 1, (
            f"Baseline model supports exactly 1 K-chunk per cluster; "
            f"got {len(kchunks)} chunks on cluster {cid} for GEMM '{name}'."
        )

    for cid in sorted_cids:
        k_idx = spec.chunks_per_cluster[cid][0]  # exactly one chunk per cluster
        l1_A = mem[f"l1_A_{name}_c{cid}"]
        l1_B = mem[f"l1_B_{name}_c{cid}"]
        l1_D = mem[f"l1_D_{name}_c{cid}"]
        gold_A = _sym(mem, f"golden_{name}_A_k{k_idx}")
        gold_B = _sym(mem, f"golden_{name}_B_k{k_idx}")

        load_A = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=cid, assigned_core_id=DMA_CORE,
            node_name=f"Load_{name}_c{cid}_k{k_idx}_A",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=gold_A, dst_addr=l1_A, size=A_tile_bytes,
            ),
        )
        load_B = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=cid, assigned_core_id=DMA_CORE,
            node_name=f"Load_{name}_c{cid}_k{k_idx}_B",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=gold_B, dst_addr=l1_B, size=B_tile_bytes,
            ),
        )

        # Fresh GEMM: D = A*B (C_addr=0, accumPrevC=0 → addNonZeroC=0).
        # No HW accumulation modes used anywhere.
        gemm = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=cid, assigned_core_id=GEMM_CORE,
            node_name=f"Gemm_{name}_c{cid}_k{k_idx}",
            kernel_name="__snax_bingo_kernel_gemm_full",
            kernel_args=SnaxBingoKernelGemmFullArgs(
                input_A_addr=l1_A, input_B_addr=l1_B,
                input_C_addr=0, output_D_addr=l1_D,
                M=M_T, K=K_T_tile, N=N_T,
                array_shape_idx=hw.array_shape_idx,
                transpose_A=0, transpose_B=0, accumPrevC=0,
            ),
        )
        for n in (load_A, load_B, gemm):
            dfg.bingo_add_node(n)
        dfg.bingo_add_edge(load_A, load_B)
        dfg.bingo_add_edge(load_B, gemm)

        # External input dependency on first load_A of this cluster
        _add_edge_list(dfg, input_dep, load_A)

        last_gemm = gemm  # (kept as last_gemm so the store code below links correctly)

        # Store after the GEMM on this cluster
        store = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=cid, assigned_core_id=DMA_CORE,
            node_name=f"Store_{name}_c{cid}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=l1_D, dst_addr=mem[f"{name}_D_partial_c{cid}_l3"],
                size=D_bytes,
            ),
        )
        dfg.bingo_add_node(store)
        dfg.bingo_add_edge(last_gemm, store)
        store_nodes[cid] = store

        if debug:
            check_p = BingoNode(
                assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
                node_name=f"Check_{name}_D_partial_c{cid}",
                kernel_name="__host_bingo_kernel_check_result",
                kernel_args=HostBingoKernelCheckResultArgs(
                    golden_data_addr=_sym(mem, f"golden_{name}_D_partial_c{cid}"),
                    output_data_addr=mem[f"{name}_D_partial_c{cid}_l3"],
                    data_size=_capped_bytes(D_bytes),
                    name=f"{name}_D_partial_c{cid}",
                    check_type=BINGO_CHECK_TYPE_BYTE_EXACT,
                ),
            )
            dfg.bingo_add_node(check_p)
            dfg.bingo_add_edge(store, check_p)
            check_partial_nodes[cid] = check_p

    # ─── Host-side reduction chain ────────────────────────────────────────
    # running_buf holds the most recent sum (L3 buffer ref)
    running_buf = mem[f"{name}_D_partial_c{sorted_cids[0]}_l3"]
    last_reduc_node = store_nodes[sorted_cids[0]]  # after the first store

    for i in range(1, len(sorted_cids)):
        cid = sorted_cids[i]
        label = "_".join(f"c{c}" for c in sorted_cids[:i + 1])
        out_buf = mem[f"{name}_sum_{label}_l3"]
        add_node = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
            node_name=f"Add_{name}_{label}",
            kernel_name="__host_bingo_kernel_int32_add",
            kernel_args=HostBingoKernelInt32AddArgs(
                input_a_addr=running_buf,
                input_b_addr=mem[f"{name}_D_partial_c{cid}_l3"],
                output_addr=out_buf,
                num_elements=D_num_elements,
            ),
        )
        dfg.bingo_add_node(add_node)
        # Reduction requires both incoming partial stores to be complete
        dfg.bingo_add_edge(last_reduc_node, add_node)
        dfg.bingo_add_edge(store_nodes[cid], add_node)

        next_tail = add_node
        if debug:
            check_s = BingoNode(
                assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
                node_name=f"Check_{name}_sum_{label}",
                kernel_name="__host_bingo_kernel_check_result",
                kernel_args=HostBingoKernelCheckResultArgs(
                    golden_data_addr=_sym(mem, f"golden_{name}_sum_{label}"),
                    output_data_addr=out_buf,
                    data_size=_capped_bytes(D_bytes),
                    name=f"{name}_sum_{label}",
                    check_type=BINGO_CHECK_TYPE_BYTE_EXACT,
                ),
            )
            dfg.bingo_add_node(check_s)
            dfg.bingo_add_edge(add_node, check_s)
            next_tail = check_s

        running_buf = out_buf
        last_reduc_node = next_tail

    # ─── Dequantize → reshape (block → row-major) → final fp32 check ──────
    # HW dequantize preserves the VersaCore D_block layout [M_T, N_T, meshRow, meshCol].
    # For N_T > 1 (qk_matmul) the block layout ≠ row-major, so we MUST emit an
    # xdma_d_to_row_major reshape to get the logically-correct row-major
    # output that downstream ops (softmax, next GEMM's A) and the check
    # golden (also row-major) expect. For N_T = 1 this reshape is a
    # no-op-equivalent memcpy but we keep the step uniform.
    #
    # Path note: __snax_bingo_kernel_xdma_d_to_row_major picks the xDMA HW
    # path for every VersaCore unrolling (see the coverage matrix in
    # snax_kernel_lib.h). For array_shape 0 (32x4x32 with elem=4) it takes
    # Path 2 (meshRow=32 split as r_outer=4 x spatial r_inner=8). The src
    # buffer is in L3 — the kernel's stage-in helper auto-IDMAs it to local
    # L1 before the xDMA AGU runs.
    fp32_dblk_buf = mem[f"{name}_fp32_l3"]         # block layout
    fp32_rm_buf   = mem[f"{name}_fp32_rm_l3"]      # row-major layout
    dequant = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name=f"Dequant_{name}",
        kernel_name="__host_bingo_kernel_int32_dequantize",
        kernel_args=HostBingoKernelInt32DequantizeArgs(
            input_addr=running_buf, output_addr=fp32_dblk_buf,
            scale_addr=_sym(mem, spec.scale_sym_name),
            num_elements=D_num_elements,
        ),
    )
    dfg.bingo_add_node(dequant)
    dfg.bingo_add_edge(last_reduc_node, dequant)

    # Reshape step: D-block layout → row-major, fp32 (elem_bytes=4).
    # Runs on the DMA core (core 1) — the kernel asserts this requirement.
    reshape = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name=f"Reshape_{name}_d2rm",
        kernel_name="__snax_bingo_kernel_xdma_d_to_row_major",
        kernel_args=SnaxBingoKernelXdmaDToRowMajorArgs(
            src_addr=fp32_dblk_buf, dst_addr=fp32_rm_buf,
            M_T=M_T, N_T=N_T, meshRow=hw.meshRow, meshCol=hw.meshCol,
            elem_bytes=4,
        ),
    )
    dfg.bingo_add_node(reshape)
    dfg.bingo_add_edge(dequant, reshape)

    if not debug:
        return reshape

    check_fp32 = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name=f"Check_{name}_fp32",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=_sym(mem, f"golden_{name}_fp32"),
            output_data_addr=fp32_rm_buf,   # read the ROW-MAJOR buffer
            # FP32_TOL: pass num_elements (fp32 count), capped by CHECK_BYTES_LIMIT.
            num_elements=_capped_elems(D_num_elements, elem_bytes=4),
            name=f"{name}_fp32",
            check_type=BINGO_CHECK_TYPE_FP32_TOL,
            tolerance=spec.check_tolerance,
        ),
    )
    dfg.bingo_add_node(check_fp32)
    dfg.bingo_add_edge(reshape, check_fp32)
    return check_fp32


# ─── Pattern C: Row-wise FP32 Softmax ─────────────────────────────────────

def emit_softmax_pattern(dfg: BingoDFG, mem: dict, hw: HwParams, *,
                         input_buf_name: str,
                         output_buf_name: str = "attn_weights_fp32_l3",
                         golden_sym_name: str = "golden_attn_weights_fp32",
                         num_rows: Optional[int] = None,
                         row_length: Optional[int] = None,
                         check_tolerance: float = 1e-6,
                         check_name: str = "softmax",
                         debug: bool = True,
                         input_dep=None):
    """Emit row-wise fp32 softmax on the CVA6 host + (debug) check.

    Allocates the output L3 buffer if not present. Returns terminal node.
    """
    if num_rows is None:
        num_rows = hw.seq_len
    if row_length is None:
        row_length = hw.seq_len

    in_buf = mem.get(input_buf_name)
    if in_buf is None:
        raise KeyError(f"softmax input buffer '{input_buf_name}' not allocated")

    out_size = num_rows * row_length * 4
    out_buf = _l3(mem, output_buf_name, out_size)

    node_sm = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Softmax",
        kernel_name="__host_bingo_kernel_fp32_softmax",
        kernel_args=HostBingoKernelFp32SoftmaxArgs(
            input_addr=in_buf, output_addr=out_buf,
            num_rows=num_rows, row_length=row_length,
        ),
    )
    dfg.bingo_add_node(node_sm)
    _add_edge_list(dfg, input_dep, node_sm)

    if not debug:
        return node_sm

    node_check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name=f"Check_{check_name}",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=_sym(mem, golden_sym_name),
            output_data_addr=out_buf,
            # FP32_TOL: fp32 element count, capped by CHECK_BYTES_LIMIT.
            num_elements=_capped_elems(num_rows * row_length, elem_bytes=4),
            name=check_name,
            check_type=BINGO_CHECK_TYPE_FP32_TOL,
            tolerance=check_tolerance,
        ),
    )
    dfg.bingo_add_node(node_check)
    dfg.bingo_add_edge(node_sm, node_check)
    return node_check


# ─── SECTION 4: Attention spec factory ────────────────────────────────────

def build_attention_specs(hw: HwParams) -> dict:
    """Derive all 6 GEMM specs for single-layer attention.

    BASELINE MODEL: all GEMMs use k_split=4 with exactly ONE K-chunk per
    cluster. No multi-chunk-per-cluster accumulation, no in-HW
    addNonZeroC=1 mode — every GEMM is a straight D = A*B with C_addr=0
    and accumPrevC=0. This is the simplest, most portable pattern.

    For seq_len=64, d_model=32, d_head=32 on array_shape=0 (32x4x32):
      proj_q/k/v:  M=2, K=8,  N=1,  K_tile=2  (k_split=4, 1 chunk/cluster)
      qk_matmul:   M=2, K=8,  N=2,  K_tile=2
      av_matmul:   M=2, K=16, N=1,  K_tile=4
      proj_o:      M=2, K=8,  N=1,  K_tile=2
    """
    mR, tS, mC = hw.meshRow, hw.tileSize, hw.meshCol

    def one_chunk_per_cluster(k_split):
        return {cid: [cid] for cid in range(k_split)}

    return {
        'proj_q': KSplitGemmSpec(
            name='proj_q',
            M_T=hw.seq_len // mR, K_T=hw.d_model // tS, N_T=hw.d_head // mC,
            k_split=4, chunks_per_cluster=one_chunk_per_cluster(4),
            scale_sym_name='combined_scale_Q',
        ),
        'proj_k': KSplitGemmSpec(
            name='proj_k',
            M_T=hw.seq_len // mR, K_T=hw.d_model // tS, N_T=hw.d_head // mC,
            k_split=4, chunks_per_cluster=one_chunk_per_cluster(4),
            scale_sym_name='combined_scale_K',
        ),
        'proj_v': KSplitGemmSpec(
            name='proj_v',
            M_T=hw.seq_len // mR, K_T=hw.d_model // tS, N_T=hw.d_head // mC,
            k_split=4, chunks_per_cluster=one_chunk_per_cluster(4),
            scale_sym_name='combined_scale_V',
        ),
        'qk_matmul': KSplitGemmSpec(
            name='qk_matmul',
            M_T=hw.seq_len // mR, K_T=hw.d_head // tS, N_T=hw.seq_len // mC,
            k_split=4, chunks_per_cluster=one_chunk_per_cluster(4),
            # Pre-multiplied by 1/sqrt(d_head) in attention_datagen.
            scale_sym_name='combined_scale_QK_scaled',
        ),
        'av_matmul': KSplitGemmSpec(
            name='av_matmul',
            M_T=hw.seq_len // mR, K_T=hw.seq_len // tS, N_T=hw.d_head // mC,
            k_split=4, chunks_per_cluster=one_chunk_per_cluster(4),
            scale_sym_name='combined_scale_attn_out',
        ),
        'proj_o': KSplitGemmSpec(
            name='proj_o',
            M_T=hw.seq_len // mR, K_T=hw.d_head // tS, N_T=hw.d_model // mC,
            # BASELINE: k_split=4 (was 8 with multi-chunk accumulation).
            # Keeps all GEMMs on the same simple "one chunk per cluster" path.
            k_split=4, chunks_per_cluster=one_chunk_per_cluster(4),
            scale_sym_name='combined_scale_Y',
        ),
    }


# ─── SECTION 5: Platform config + main ─────────────────────────────────────

def parse_platform_cfg(occamy_h_path, rtlcfg_path):
    """Parse HW platform params from generated occamy.h and RTL config hjson.
    Chiplet IDs are coordinate-encoded: (x, y) → (x << 4) | y.
    """
    defines = {}
    with open(occamy_h_path) as f:
        for line in f:
            m = re.match(r'#define\s+(\w+)\s+(\d+)', line)
            if m:
                defines[m.group(1)] = int(m.group(2))
    with open(rtlcfg_path) as f:
        rtlcfg = hjson.loads(f.read())
    multichip = rtlcfg["hemaia_multichip"]
    if multichip["single_chip"]:
        chiplet_ids = [0x00]
    else:
        chiplet_ids = [(c["coordinate"][0] << 4) | c["coordinate"][1]
                       for c in multichip["testbench_cfg"]["hemaia_compute_chip"]]
    return {
        "num_chiplets": defines["N_CHIPLETS"],
        "num_clusters_per_chiplet": defines["N_CLUSTERS_PER_CHIPLET"],
        "num_cores_per_cluster": defines["N_CORES_PER_CLUSTER"],
        "chiplet_ids": chiplet_ids,
    }


def get_args():
    p = argparse.ArgumentParser(description="attn_reference (modular attention workload)")
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--rtlcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    p.add_argument("--debug-mode", choices=["debug", "run"], default="debug")
    return p.parse_args()


def main():
    args = get_args()
    out_dir = args.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load configs
    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hwcfg = hjson.loads(f.read())
    merged = {**param, **hwcfg}

    # Emit data header (goldens)
    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(**merged))
        print(f"Written data header: {args.data_h}")

    # Derive HW params
    array_shape = merged["array_shape"]
    unrolling = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]["snax_versacore_spatial_unrolling"][0][array_shape]
    platform = parse_platform_cfg(args.platformcfg, args.rtlcfg)
    hw = HwParams(
        meshRow=unrolling[0], tileSize=unrolling[1], meshCol=unrolling[2],
        array_shape_idx=array_shape,
        seq_len=merged["seq_len"], d_model=merged["d_model"], d_head=merged["d_head"],
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        chiplet_ids=platform["chiplet_ids"],
    )
    debug = (args.debug_mode == "debug")

    # Build DFG
    dfg = BingoDFG(
        num_chiplets=hw.num_chiplets,
        num_clusters_per_chiplet=hw.num_clusters_per_chiplet,
        num_cores_per_cluster=hw.num_cores_per_cluster,
        is_host_as_acc=True,
        chiplet_ids=hw.chiplet_ids,
    )
    mem: dict = {}
    specs = build_attention_specs(hw)

    # ═══ Compose the attention chain ════════════════════════════════════════
    # In debug mode the datagen provides "atomic" golden tiles for EVERY GEMM
    # input (via attention_datagen's tile-level golden section). This means
    # each GEMM's load_A/load_B read GOLDEN data, not the runtime output of
    # the upstream stage. The check_result nodes verify each stage in isolation.
    # Data-flow edges (input_dep) between stages keep scheduling sane and
    # let future "run mode" (no goldens) work by switching the loads to
    # read runtime buffers instead of golden symbols.
    # ═══════════════════════════════════════════════════════════════════════

    # Quantize X (FP32 → INT8)
    tail_quant = emit_quantize_pattern(dfg, mem, hw, debug=debug)

    # Q/K/V projections — all depend on quantize completing
    tail_proj_q = emit_ksplit_gemm_pattern(dfg, mem, hw, specs['proj_q'],
                                           debug=debug, input_dep=tail_quant)
    tail_proj_k = emit_ksplit_gemm_pattern(dfg, mem, hw, specs['proj_k'],
                                           debug=debug, input_dep=tail_quant)
    tail_proj_v = emit_ksplit_gemm_pattern(dfg, mem, hw, specs['proj_v'],
                                           debug=debug, input_dep=tail_quant)

    # QK matmul — depends on Q and K being ready
    tail_qk = emit_ksplit_gemm_pattern(dfg, mem, hw, specs['qk_matmul'],
                                       debug=debug,
                                       input_dep=[tail_proj_q, tail_proj_k])

    # Softmax on the ROW-MAJOR qk_matmul output. The emit_ksplit_gemm_pattern
    # emits an xdma_d_to_row_major reshape after dequant, populating
    # `qk_matmul_fp32_rm_l3` with a logical (seq_len x seq_len) row-major
    # attention-score matrix — the correct input for row-wise softmax.
    tail_sm = emit_softmax_pattern(dfg, mem, hw,
                                   input_buf_name="qk_matmul_fp32_rm_l3",
                                   debug=debug,
                                   input_dep=tail_qk)

    # AV matmul — depends on softmax (A) and proj_v (V)
    tail_av = emit_ksplit_gemm_pattern(dfg, mem, hw, specs['av_matmul'],
                                       debug=debug,
                                       input_dep=[tail_sm, tail_proj_v])

    # Output projection
    tail_o = emit_ksplit_gemm_pattern(dfg, mem, hw, specs['proj_o'],
                                      debug=debug, input_dep=tail_av)

    # Summary
    n_nodes = len(dfg._graph.nodes) if hasattr(dfg, "_graph") else "?"
    print(f"Built attention DFG ({'debug' if debug else 'run'} mode): "
          f"~{n_nodes} nodes composed from 3 pattern types")
    print(f"  Quantize: 1 stage; K-split GEMM: 6 stages (proj_q/k/v, qk, av, o); "
          f"Softmax: 1 stage")
    print(f"  Final terminal node: Check_proj_o_fp32")
    _ = tail_o  # silence unused-var lint

    dfg.bingo_compile_dfg(
        "attn_reference", out_dir, args.output_offload_file_name,
        extra_include_header_list=["attention_datagen_data.h"],
    )


if __name__ == "__main__":
    main()
