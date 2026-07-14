#!/usr/bin/env python3
r"""
gemm_nsplit_accumprevc_4cluster

Parallelize a GEMM along N across the 4 clusters (each owns one disjoint output
tile) and do the K reduction ON-CHIP inside each cluster with the VersaCore
`accumPrevC` feature -- so the host core never sums anything.

Why accumPrevC works here (and not across a parallel K-split): the accumulator
register is private to ONE cluster's VersaCore and is never cleared at kernel
end (Accumulator.scala: it only updates on adder.out.fire). Two GEMM nodes placed
on the same cluster's GEMM core run strictly back-to-back -- bingo_dfg.py's
bingo_transform_add_core_sequencing_edges groups nodes by (chiplet,cluster,core)
and serializes them -- so the second call (accumPrevC=1) accumulates onto the
register the first call (accumPrevC=0) left behind. `input_C_addr=0` on every
call: with accumPrevC no C is streamed (the "+C" comes from the register).

Constraint: accumPrevC needs M=1 and N=1 per call (the register is one tile), so
each cluster owns exactly one output tile and N_global = num_clusters.

================================================================================
WHO HOLDS WHAT  --  the GEMM is split along N; each cluster owns ONE output tile.

  Global GEMM (tile counts M=1, K=k_chunks, N=num_clusters):   D = A . B

  A is SHARED by every cluster (M=1 row-tile), split over K into chunks:
        A = [ A_k0 | A_k1 | ... ]            (one meshRow x tileSize tile per chunk)

  B is split by N (one column-block per cluster), each block also split over K:
                 cluster0    cluster1    cluster2    cluster3
               +----------+----------+----------+----------+
        chunk0 |  B0_k0   |  B1_k0   |  B2_k0   |  B3_k0   |
        chunk1 |  B0_k1   |  B1_k1   |  B2_k1   |  B3_k1   |
               +----------+----------+----------+----------+

  D is those tiles laid side by side along N  --  NO host reduction:
               +----------+----------+----------+----------+
         D  =  |    D0    |    D1    |    D2    |    D3    |
               | cluster0 | cluster1 | cluster2 | cluster3 |
               +----------+----------+----------+----------+
                 \____________ concatenation, no add node ____________/

  Each cluster c builds its OWN tile by accumulating over K on-chip:
        Dc = A_k0 . Bc_k0  +  A_k1 . Bc_k1  + ...
             |__ Gemm k0 __|  |__ Gemm k1 (accumPrevC=1) __|
              reg = P0          reg += P1  ->  reg = Dc   (register kept call-to-call)

================================================================================
NUMERICAL WALKTHROUGH  (logical 2x2 tiles so it fits; the real workload uses mesh
32x2x32, M=1, N=4, K=2).  Worked for cluster 0; clusters 1-3 are identical with
their own B column-block.

  Shared A (2 rows x 4 cols) and its two K-chunks:
        A    = [ 1  2 | 3  4 ]       A_k0 = [ 1  2 ]      A_k1 = [ 3  4 ]
               [ 5  6 | 7  8 ]              [ 5  6 ]             [ 7  8 ]

  Cluster 0's B column-block (4 rows x 2 cols) and its two K-chunks:
        B0   = [ 1  0 ]              B0_k0 = [ 1  0 ]     B0_k1 = [ 2  1 ]
               [ 0  1 ]                      [ 0  1 ]             [ 1  2 ]
               [ 2  1 ]
               [ 1  2 ]

  Gemm k0 (accumPrevC=0):  reg = A_k0 . B0_k0 = [  1   2 ]  = P0
                                                [  5   6 ]
  Gemm k1 (accumPrevC=1):  P1  = A_k1 . B0_k1 = [ 10  11 ]
                                                [ 22  23 ]
                           reg = P0 + P1       = [ 11  13 ]  = D0   (cluster 0's tile)
                                                [ 27  29 ]

  Assembled D  --  each cluster's 2x2 tile sits in the N-band it owns:
                cluster0    cluster1    cluster2    cluster3
              +----------+----------+----------+----------+
         D =  | 11    13 |  D1[0,:] |  D2[0,:] |  D3[0,:] |
              | 27    29 |  D1[1,:] |  D2[1,:] |  D3[1,:] |
              +----------+----------+----------+----------+
                ^ computed entirely on cluster 0, no other cluster touches it

================================================================================
Per-cluster DFG (cluster c; DOUBLE-BUFFERED: one A/B buffer per K-chunk, k_chunks=2):

  Load_A_c{c}_k0 \                              Load_A_c{c}_k1 \
                  Gemm_c{c}_k0 ---------------->                Gemm_c{c}_k1 -> Store -> Check
  Load_B_c{c}_k0 /  (accumPrevC=0, C=0)         Load_B_c{c}_k1 /  (accumPrevC=1, C=0)
                    reg = A0*Bc0                                  reg += A1*Bc1 = Dc

  Chunk 1's loads write their OWN buffers, so they do NOT wait on Gemm k0 (no WAR
  edge): the DMA core prefetches chunk 1 while the GEMM core runs chunk 0. The
  only ordering kept is Gemm k0 -> Gemm k1 -- the accumPrevC register must survive
  between the two calls.

--------------------------------------------------------------------------------
Gantt (all 4 clusters in parallel; DOUBLE-BUFFERED => NO GEMM bubble: the chunk-1
load overlaps Gemm k0, so Gemm k0 and Gemm k1 run back-to-back). DMA core = loads
+ store, GEMM core = matmul, HOST = checks:

  time ->   t0      t1      t2       t3       t4        t5
  DMA c0:  LdA k0  LdB k0  LdA k1   LdB k1            Store D0
  GEMM c0:                 Gemm k0  Gemm k1                       reg: 0->P0->P0+P1=D0
  DMA c1:  LdA k0  LdB k0  LdA k1   LdB k1            Store D1
  GEMM c1:                 Gemm k0  Gemm k1
  DMA c2:  LdA k0  LdB k0  LdA k1   LdB k1            Store D2
  GEMM c2:                 Gemm k0  Gemm k1
  DMA c3:  LdA k0  LdB k0  LdA k1   LdB k1            Store D3
  GEMM c3:                 Gemm k0  Gemm k1
  HOST:                                              Check D0|D1|D2|D3

  LdA/LdB k1 land in the chunk-1 buffers DURING Gemm k0, so the moment Gemm k0
  retires, Gemm k1 starts -- the bubble between the two GEMMs (present with a
  single reused buffer) is gone. Still no host-reduction row: D = [D0|D1|D2|D3].
"""

import os
import sys
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR = os.path.dirname(current_dir)
sys.path.append(WORKLOADS_DIR)
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nsplit_accumprevc_gemm_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    HostBingoKernelCheckResultArgs,
)

DATA_HEADER = "nsplit_accumprevc_gemm_data.h"

# Core assignment within a cluster (matches the other offload_bingo_hw workloads)
GEMM_CORE = 0
DMA_CORE = 1
HOST_CORE = 2


def get_args():
    parser = argparse.ArgumentParser(description="gemm_nsplit_accumprevc_4cluster")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

    # Emit the data header (embedded C arrays).
    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(**merged))
        print(f"Written data header: {args.data_h}")

    # Derive params.
    array_shape = merged["array_shape"]
    data_type = merged.get("data_type", 0)
    num_clusters = merged["num_clusters"]
    M = merged["M"]
    N_per_cluster = merged["N_per_cluster"]
    k_chunks = merged["k_chunks"]
    K_per_chunk = merged["K_per_chunk"]
    transpose_A = merged.get("transposed_A", 0)
    transpose_B = merged.get("transposed_B", 0)

    if M != 1 or N_per_cluster != 1:
        raise ValueError(f"accumPrevC needs M==1 and N_per_cluster==1, got M={M} N_per_cluster={N_per_cluster}")

    snax_acc_cfg = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, tileSize, meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    # Byte sizes of the per-call operands (int8 inputs, int32 output tile).
    A_chunk_bytes = M * K_per_chunk * meshRow * tileSize          # int8
    B_chunk_bytes = K_per_chunk * N_per_cluster * tileSize * meshCol  # int8
    D_bytes = M * N_per_cluster * meshRow * meshCol * 4           # int32

    # Platform.
    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    # Embedded-symbol inputs/golden (live in the data header, no mem chip).
    sym_A = {j: BingoMemSymbol(f"A_k{j}") for j in range(k_chunks)}
    sym_B = {(c, j): BingoMemSymbol(f"B_c{c}_k{j}")
             for c in range(num_clusters) for j in range(k_chunks)}
    sym_golden = {c: BingoMemSymbol(f"golden_D_c{c}") for c in range(num_clusters)}

    # Per-cluster working buffers. DOUBLE-BUFFERED: a SEPARATE A/B buffer per
    # (cluster, K-chunk) so the DMA core can prefetch chunk j+1 while the GEMM
    # core still computes chunk j -- this removes the bubble the single reused
    # buffer caused (no WAR stall waiting to reload). One D tile buffer per
    # cluster (accumPrevC accumulates in place across chunks), plus an L3 landing
    # buffer for the store. The per-chunk buffers are tiny (64 B each here), so
    # the extra L1 footprint is negligible.
    l1_A = {(c, j): BingoMemAlloc(f"l1_A_c{c}_k{j}", size=A_chunk_bytes, mem_level="L1", chip_id=0, cluster_id=c)
            for c in range(num_clusters) for j in range(k_chunks)}
    l1_B = {(c, j): BingoMemAlloc(f"l1_B_c{c}_k{j}", size=B_chunk_bytes, mem_level="L1", chip_id=0, cluster_id=c)
            for c in range(num_clusters) for j in range(k_chunks)}
    l1_D = {c: BingoMemAlloc(f"l1_D_c{c}", size=D_bytes, mem_level="L1", chip_id=0, cluster_id=c)
            for c in range(num_clusters)}
    l3_D = {c: BingoMemAlloc(f"l3_D_c{c}", size=D_bytes, mem_level="L3", chip_id=0)
            for c in range(num_clusters)}

    for c in range(num_clusters):
        prev_gemm = None
        for j in range(k_chunks):
            load_A = BingoNode(
                assigned_chiplet_id=0, assigned_cluster_id=c, assigned_core_id=DMA_CORE,
                node_name=f"Load_A_c{c}_k{j}",
                kernel_name="__snax_bingo_kernel_idma_1d_copy",
                kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                    src_addr=sym_A[j], dst_addr=l1_A[(c, j)], size=A_chunk_bytes,
                ),
            )
            load_B = BingoNode(
                assigned_chiplet_id=0, assigned_cluster_id=c, assigned_core_id=DMA_CORE,
                node_name=f"Load_B_c{c}_k{j}",
                kernel_name="__snax_bingo_kernel_idma_1d_copy",
                kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                    src_addr=sym_B[(c, j)], dst_addr=l1_B[(c, j)], size=B_chunk_bytes,
                ),
            )
            # Chunk 0 seeds the register (accumPrevC=0); chunks >=1 accumulate.
            gemm = BingoNode(
                assigned_chiplet_id=0, assigned_cluster_id=c, assigned_core_id=GEMM_CORE,
                node_name=f"Gemm_c{c}_k{j}",
                kernel_name="__snax_bingo_kernel_gemm_full",
                kernel_args=SnaxBingoKernelGemmFullArgs(
                    input_A_addr=l1_A[(c, j)], input_B_addr=l1_B[(c, j)],
                    input_C_addr=0, output_D_addr=l1_D[c],
                    M=M, K=K_per_chunk, N=N_per_cluster,
                    array_shape_idx=array_shape,
                    transpose_A=transpose_A, transpose_B=transpose_B,
                    accumPrevC=(0 if j == 0 else 1),
                ),
            )
            for n in (load_A, load_B, gemm):
                dfg.bingo_add_node(n)
            dfg.bingo_add_edge(load_A, gemm)
            dfg.bingo_add_edge(load_B, gemm)
            # Double-buffered: this chunk's loads write their OWN buffer, so there
            # is NO WAR edge from the prior GEMM -- the DMA core prefetches chunk
            # j+1 while the GEMM core runs chunk j (squeezes out the bubble). The
            # only ordering kept is the accumPrevC chain itself: the on-chip
            # register must survive from one GEMM call to the next.
            if prev_gemm is not None:
                dfg.bingo_add_edge(prev_gemm, gemm)
            prev_gemm = gemm

        store = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=c, assigned_core_id=DMA_CORE,
            node_name=f"Store_D_c{c}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=l1_D[c], dst_addr=l3_D[c], size=D_bytes,
            ),
        )
        check = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
            node_name=f"Check_D_c{c}",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=sym_golden[c],
                output_data_addr=l3_D[c],
                data_size=D_bytes,
                name=f"D_c{c}",
            ),
        )
        dfg.bingo_add_node(store)
        dfg.bingo_add_node(check)
        dfg.bingo_add_edge(prev_gemm, store)
        dfg.bingo_add_edge(store, check)

    total_nodes = num_clusters * (k_chunks * 3 + 2)
    print(f"Built DFG: {total_nodes} nodes "
          f"({num_clusters} parallel clusters x ({k_chunks} K-chunks + store + check), no reduction)")
    print(f"  M={M}, N_per_cluster={N_per_cluster}, k_chunks={k_chunks}, K_per_chunk={K_per_chunk}, "
          f"mesh={meshRow}x{tileSize}x{meshCol}")

    dfg.bingo_compile_dfg(
        "gemm_nsplit_accumprevc_4cluster", output_dir, args.output_offload_file_name,
        extra_include_header_list=[DATA_HEADER],
    )


if __name__ == "__main__":
    main()
