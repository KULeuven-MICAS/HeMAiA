#!/usr/bin/env python3
"""
Golden data generator for gemm_nsplit_accumprevc_4cluster.

Design C: the GEMM is parallelized along N across `num_clusters` clusters; each
cluster owns one output tile (M=1, N_per_cluster=1) and reduces over K in
`k_chunks` chunks ON-CHIP via accumPrevC -- no host reduction.

Data is emitted as embedded C arrays (BingoMemSymbol-backed), NOT a mem-chip
mempool.bin, because the default single-chip RTL config has no mem chip.

Emitted symbols (all in the generated data header):
  - A_k{j}        int8 : the j-th K-chunk of A (M=1). Shared/broadcast to every
                         cluster (each cluster DMAs its own copy into L1).
  - B_c{c}_k{j}   int8 : cluster c's B operand for K-chunk j (its N-tile, K-chunk j)
  - golden_D_c{c} int32: cluster c's full-K output tile = sum over j of A_k{j}*B_c{c}_k{j}

Byte-layout contract (must match the VersaCore streamer AND block_gemm_golden_model):
  A is A-layout  flat (m, k, meshRow,  tileSize)   <- golden does a.reshape(m,k,row,size)
  B is B-layout  flat (n, k, meshCol,  tileSize)   <- golden does b.reshape(n,k,col,size)
  D is D-layout  flat (m, n, meshRow,  meshCol)
So A_full is generated as (M, K, meshRow, tileSize) and B_full as
(num_clusters, K, meshCol, tileSize); slicing those gives the exact bytes both the
device and the golden model consume.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from sim_golden_models import block_gemm_golden_model  # noqa E402

np.random.seed(42)


def emit_header_file(**kwargs):
    lines = ["#include <stdint.h>"]

    array_shape = kwargs["array_shape"]
    data_type = kwargs.get("data_type", 0)  # int8
    num_clusters = kwargs["num_clusters"]
    M = kwargs["M"]
    N_per_cluster = kwargs["N_per_cluster"]
    k_chunks = kwargs["k_chunks"]
    K_per_chunk = kwargs["K_per_chunk"]

    # accumPrevC tile constraint: the on-chip accumulator holds exactly one
    # output tile, so every GEMM call must produce a single tile.
    if M != 1 or N_per_cluster != 1:
        raise ValueError(
            f"accumPrevC needs M==1 and N_per_cluster==1, got M={M} N_per_cluster={N_per_cluster}"
        )
    if k_chunks < 1:
        raise ValueError(f"k_chunks ({k_chunks}) must be >= 1")

    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, tileSize, meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    K = k_chunks * K_per_chunk          # global K-tile count
    N = num_clusters * N_per_cluster    # global N-tile count (one tile per cluster)
    D_size = M * N_per_cluster * meshRow * meshCol  # int32 elems per cluster tile

    lines.append(
        f"// Design-C N-split GEMM: M={M} K={K} N={N} (num_clusters={num_clusters}, "
        f"N_per_cluster={N_per_cluster}, k_chunks={k_chunks}, K_per_chunk={K_per_chunk})"
    )
    lines.append(f"// mesh={meshRow}x{tileSize}x{meshCol}  (one D-tile = {D_size} int32)")

    A_MIN, A_MAX = -128, 127
    B_MIN, B_MAX = -128, 127

    # A-layout (m,k,row,size); B-layout (n,k,col,size). One N-tile per cluster.
    A_full = np.random.randint(A_MIN, A_MAX, size=(M, K, meshRow, tileSize)).astype(np.int8)
    B_full = np.random.randint(B_MIN, B_MAX, size=(num_clusters, K, meshCol, tileSize)).astype(np.int8)

    C_zero = np.zeros(D_size, dtype=np.int32)

    def kslice(j):
        return slice(j * K_per_chunk, (j + 1) * K_per_chunk)

    # --- A per K-chunk (broadcast to all clusters) ---
    for j in range(k_chunks):
        A_kj = A_full[:, kslice(j), :, :].reshape(-1).copy().astype(np.int8)
        lines.append(format_vector_definition("int8_t", f"A_k{j}", A_kj))

    # --- B per cluster per K-chunk + per-cluster golden output tile ---
    for c in range(num_clusters):
        for j in range(k_chunks):
            B_cj = B_full[c, kslice(j), :, :].reshape(-1).copy().astype(np.int8)
            lines.append(format_vector_definition("int8_t", f"B_c{c}_k{j}", B_cj))

        # Full-K golden for cluster c's tile = sum_j A_k{j} * B_c{c}_k{j}.
        golden_D = block_gemm_golden_model(
            M, K, N_per_cluster, meshRow, tileSize, meshCol,
            A_full.reshape(-1).copy(), B_full[c].reshape(-1).copy(), 0, 0, C_zero.copy(),
        ).astype(np.int32)

        # Sanity: the accumPrevC chain sums per-chunk partials. Confirm that
        # equals the full-K golden (linearity of the block GEMM over K).
        acc = np.zeros(D_size, dtype=np.int64)
        for j in range(k_chunks):
            partial = block_gemm_golden_model(
                M, K_per_chunk, N_per_cluster, meshRow, tileSize, meshCol,
                A_full[:, kslice(j), :, :].reshape(-1).copy(),
                B_full[c, kslice(j), :, :].reshape(-1).copy(), 0, 0, C_zero.copy(),
            ).astype(np.int64)
            acc += partial
        assert np.array_equal(acc.astype(np.int32), golden_D), (
            f"chunk-sum != full golden for cluster {c}"
        )

        lines.append(format_vector_definition("int32_t", f"golden_D_c{c}", golden_D))

    # Informative scalars (not required by the runtime; main_bingo recomputes sizes).
    for name, val in [
        ("nsplit_num_clusters", num_clusters),
        ("nsplit_M", M),
        ("nsplit_K", K),
        ("nsplit_N", N),
        ("nsplit_k_chunks", k_chunks),
        ("nsplit_K_per_chunk", K_per_chunk),
        ("nsplit_meshRow", meshRow),
        ("nsplit_tileSize", tileSize),
        ("nsplit_meshCol", meshCol),
    ]:
        lines.append(format_scalar_definition("uint32_t", name, val))

    return "\n\n".join(lines) + "\n"
