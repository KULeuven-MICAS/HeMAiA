#!/usr/bin/env python3
"""
Attention test: K-split=4 GEMM across 4 clusters with int32_add reduction
and dequantize.

Per cluster i (0..3, in parallel):
  Load_A_k{i} (DMA,c{i}) -> Load_B_k{i} (DMA,c{i}) -> GEMM_full (GEMM,c{i})
  -> Store_D_c{i} (DMA,c{i}) -> Check_D_partial_c{i} (Host,c0)

Then sequential on host:
  Add_c0_c1 (Host,c0) -> Check_sum_c0_c1 (Host,c0)
  -> Add_c0_c1_c2 (Host,c0) -> Check_sum_c0_c1_c2 (Host,c0)
  -> Add_c0_c1_c2_c3 (Host,c0) -> Check_sum_final (Host,c0)
  -> Dequant (Host,c0) -> Check_fp32 (Host,c0)

Total: ~28 nodes
"""

import os
import sys
import re
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ksplit_gemm_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelInt32AddArgs,
    HostBingoKernelInt32DequantizeArgs,
)


def parse_platform_cfg(occamy_h_path, rtlcfg_path):
    """Parse HW platform parameters from generated occamy.h and RTL config."""
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
        chiplet_ids = []
        for chip in multichip["testbench_cfg"]["hemaia_compute_chip"]:
            x, y = chip["coordinate"]
            chiplet_ids.append((x << 4) | y)
    return {
        "num_chiplets": defines["N_CHIPLETS"],
        "num_clusters_per_chiplet": defines["N_CLUSTERS_PER_CHIPLET"],
        "num_cores_per_cluster": defines["N_CORES_PER_CLUSTER"],
        "chiplet_ids": chiplet_ids,
    }


def get_args():
    parser = argparse.ArgumentParser(description="attn_test_ksplit_gemm")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--rtlcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load configs
    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

    # Emit data header
    if args.data_h is not None:
        content = emit_header_file(**merged)
        with open(args.data_h, "w") as f:
            f.write(content)
        print(f"Written data header: {args.data_h}")

    # Parse HW platform parameters
    platform = parse_platform_cfg(args.platformcfg, args.rtlcfg)

    # Derive params
    array_shape = merged["array_shape"]
    seq_len = merged["seq_len"]
    d_model = merged["d_model"]
    d_head = merged["d_head"]
    k_split = merged["k_split"]
    transpose_A = merged.get("transposed_A", 0)
    transpose_B = merged.get("transposed_B", 0)
    accumPrevC = merged.get("accumPrevC", 0)

    data_type = 0  # int8
    snax_acc_cfg = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, tileSize, meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    M_T = seq_len // meshRow
    K_T_full = d_model // tileSize
    N_T = d_head // meshCol
    K_T_tile = K_T_full // k_split

    A_tile_bytes = M_T * K_T_tile * meshRow * tileSize   # int8 bytes per K-chunk
    B_tile_bytes = K_T_tile * N_T * tileSize * meshCol    # int8 bytes per K-chunk
    D_num_elements = M_T * N_T * meshRow * meshCol        # int32 element count
    D_bytes = D_num_elements * 4                          # int32 bytes

    # Core IDs
    GEMM_CORE = 0
    DMA_CORE = 1
    HOST_CORE = 2

    # ── Memory handles ──────────────────────────────────────────────

    # L3 symbols for golden tile inputs (from ksplit_gemm_data.h)
    sym_A = {}
    sym_B = {}
    sym_D_partial = {}
    for i in range(k_split):
        sym_A[i] = BingoMemSymbol(f"A_k{i}")
        sym_B[i] = BingoMemSymbol(f"B_k{i}")
        sym_D_partial[i] = BingoMemSymbol(f"D_partial_c{i}")

    # Golden reduction sums
    sym_sum_c0_c1 = BingoMemSymbol("golden_sum_c0_c1")
    sym_sum_c0_c1_c2 = BingoMemSymbol("golden_sum_c0_c1_c2")
    sym_sum_final = BingoMemSymbol("golden_sum_c0_c1_c2_c3")
    sym_golden_fp32 = BingoMemSymbol("golden_fp32_D")
    sym_combined_scale = BingoMemSymbol("combined_scale")

    # L1 buffers per cluster (for GEMM compute)
    l1_A = {}
    l1_B = {}
    l1_D = {}
    for i in range(k_split):
        l1_A[i] = BingoMemAlloc(f"l1_A_c{i}", size=A_tile_bytes, mem_level="L1",
                                chip_id=0, cluster_id=i)
        l1_B[i] = BingoMemAlloc(f"l1_B_c{i}", size=B_tile_bytes, mem_level="L1",
                                chip_id=0, cluster_id=i)
        l1_D[i] = BingoMemAlloc(f"l1_D_c{i}", size=D_bytes, mem_level="L1",
                                chip_id=0, cluster_id=i)

    # L3 buffers for partial D outputs (stored from each cluster)
    l3_D_partial = {}
    for i in range(k_split):
        l3_D_partial[i] = BingoMemAlloc(f"D_partial_l3_c{i}", size=D_bytes, mem_level="L3")

    # L3 buffers for reduction sums
    l3_sum_c0_c1 = BingoMemAlloc("sum_c0_c1_l3", size=D_bytes, mem_level="L3")
    l3_sum_c0_c1_c2 = BingoMemAlloc("sum_c0_c1_c2_l3", size=D_bytes, mem_level="L3")
    l3_sum_final = BingoMemAlloc("sum_final_l3", size=D_bytes, mem_level="L3")

    # L3 buffer for dequantized FP32 output
    l3_fp32_D = BingoMemAlloc("fp32_D_l3", size=D_num_elements * 4, mem_level="L3")

    # ── DFG ─────────────────────────────────────────────────────────
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    # ── Per-cluster paths (parallel across clusters) ────────────────
    # For each cluster i: Load_A -> Load_B -> GEMM -> Store_D -> Check_D
    node_load_A = {}
    node_load_B = {}
    node_gemm = {}
    node_store_D = {}
    node_check_D = {}

    for i in range(k_split):
        # Load A tile: L3 -> L1 via device iDMA
        node_load_A[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Load_A_k{i}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=sym_A[i], dst_addr=l1_A[i], size=A_tile_bytes,
            ),
        )

        # Load B tile: L3 -> L1 via device iDMA
        node_load_B[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Load_B_k{i}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=sym_B[i], dst_addr=l1_B[i], size=B_tile_bytes,
            ),
        )

        # GEMM on cluster i
        node_gemm[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=GEMM_CORE,
            node_name=f"Gemm_c{i}",
            kernel_name="__snax_bingo_kernel_gemm_full",
            kernel_args=SnaxBingoKernelGemmFullArgs(
                input_A_addr=l1_A[i], input_B_addr=l1_B[i],
                input_C_addr=0, output_D_addr=l1_D[i],
                M=M_T, K=K_T_tile, N=N_T,
                array_shape_idx=array_shape,
                transpose_A=transpose_A, transpose_B=transpose_B,
                accumPrevC=accumPrevC,
            ),
        )

        # Store D: L1 -> L3
        node_store_D[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=i, assigned_core_id=DMA_CORE,
            node_name=f"Store_D_c{i}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=l1_D[i], dst_addr=l3_D_partial[i], size=D_bytes,
            ),
        )

        # Check partial D against golden
        node_check_D[i] = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
            node_name=f"Check_D_partial_c{i}",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=sym_D_partial[i],
                output_data_addr=l3_D_partial[i],
                data_size=D_bytes,
                name=f"D_partial_c{i}",
            ),
        )

    # Add all per-cluster nodes
    for i in range(k_split):
        for n in [node_load_A[i], node_load_B[i], node_gemm[i],
                  node_store_D[i], node_check_D[i]]:
            dfg.bingo_add_node(n)

    # Per-cluster edges: Load_A -> Load_B -> GEMM -> Store_D -> Check_D
    for i in range(k_split):
        dfg.bingo_add_edge(node_load_A[i], node_load_B[i])
        dfg.bingo_add_edge(node_load_B[i], node_gemm[i])
        dfg.bingo_add_edge(node_gemm[i], node_store_D[i])
        dfg.bingo_add_edge(node_store_D[i], node_check_D[i])

    # ── Reduction phase (serial on host) ────────────────────────────
    # Add_c0_c1: D_partial_c0 + D_partial_c1 -> sum_c0_c1
    node_add_01 = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Add_c0_c1",
        kernel_name="__host_bingo_kernel_int32_add",
        kernel_args=HostBingoKernelInt32AddArgs(
            input_a_addr=l3_D_partial[0],
            input_b_addr=l3_D_partial[1],
            output_addr=l3_sum_c0_c1,
            num_elements=D_num_elements,
        ),
    )

    node_check_sum_01 = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_sum_c0_c1",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=sym_sum_c0_c1,
            output_data_addr=l3_sum_c0_c1,
            data_size=D_bytes,
            name="sum_c0_c1",
        ),
    )

    # Add_c0_c1_c2: sum_c0_c1 + D_partial_c2 -> sum_c0_c1_c2
    node_add_012 = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Add_c0_c1_c2",
        kernel_name="__host_bingo_kernel_int32_add",
        kernel_args=HostBingoKernelInt32AddArgs(
            input_a_addr=l3_sum_c0_c1,
            input_b_addr=l3_D_partial[2],
            output_addr=l3_sum_c0_c1_c2,
            num_elements=D_num_elements,
        ),
    )

    node_check_sum_012 = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_sum_c0_c1_c2",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=sym_sum_c0_c1_c2,
            output_data_addr=l3_sum_c0_c1_c2,
            data_size=D_bytes,
            name="sum_c0_c1_c2",
        ),
    )

    # Add_c0_c1_c2_c3: sum_c0_c1_c2 + D_partial_c3 -> sum_final
    node_add_final = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Add_c0_c1_c2_c3",
        kernel_name="__host_bingo_kernel_int32_add",
        kernel_args=HostBingoKernelInt32AddArgs(
            input_a_addr=l3_sum_c0_c1_c2,
            input_b_addr=l3_D_partial[3],
            output_addr=l3_sum_final,
            num_elements=D_num_elements,
        ),
    )

    node_check_sum_final = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_sum_final",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=sym_sum_final,
            output_data_addr=l3_sum_final,
            data_size=D_bytes,
            name="sum_final",
        ),
    )

    # ── Dequantize + final check ────────────────────────────────────
    node_dequant = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Dequant",
        kernel_name="__host_bingo_kernel_int32_dequantize",
        kernel_args=HostBingoKernelInt32DequantizeArgs(
            input_addr=l3_sum_final,
            output_addr=l3_fp32_D,
            scale_addr=sym_combined_scale,
            num_elements=D_num_elements,
        ),
    )

    node_check_fp32 = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_fp32",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=sym_golden_fp32,
            output_data_addr=l3_fp32_D,
            data_size=D_num_elements * 4,
            name="fp32_D",
        ),
    )

    # Add reduction + dequant nodes
    for n in [node_add_01, node_check_sum_01,
              node_add_012, node_check_sum_012,
              node_add_final, node_check_sum_final,
              node_dequant, node_check_fp32]:
        dfg.bingo_add_node(n)

    # ── Reduction edges ─────────────────────────────────────────────
    # All partial D checks must complete before reduction starts
    # Add_c0_c1 needs D_partial_c0 and D_partial_c1 to be in L3
    dfg.bingo_add_edge(node_check_D[0], node_add_01)
    dfg.bingo_add_edge(node_check_D[1], node_add_01)
    dfg.bingo_add_edge(node_add_01, node_check_sum_01)

    # Add_c0_c1_c2 needs sum_c0_c1 check + D_partial_c2
    dfg.bingo_add_edge(node_check_sum_01, node_add_012)
    dfg.bingo_add_edge(node_check_D[2], node_add_012)
    dfg.bingo_add_edge(node_add_012, node_check_sum_012)

    # Add_c0_c1_c2_c3 needs sum_c0_c1_c2 check + D_partial_c3
    dfg.bingo_add_edge(node_check_sum_012, node_add_final)
    dfg.bingo_add_edge(node_check_D[3], node_add_final)
    dfg.bingo_add_edge(node_add_final, node_check_sum_final)

    # Dequant after final sum check
    dfg.bingo_add_edge(node_check_sum_final, node_dequant)
    dfg.bingo_add_edge(node_dequant, node_check_fp32)

    total_nodes = k_split * 5 + 8  # 5 per cluster + 6 reduction/check + dequant + check_fp32
    print(f"Built DFG: {total_nodes} nodes ({k_split} parallel cluster paths + reduction chain)")
    print(f"  M_T={M_T}, K_T_tile={K_T_tile}, N_T={N_T}, k_split={k_split}")
    print(f"  A_tile_bytes={A_tile_bytes}, B_tile_bytes={B_tile_bytes}, D_bytes={D_bytes}")

    dfg.bingo_compile_dfg(
        "attn_test_ksplit_gemm", output_dir, args.output_offload_file_name,
        extra_include_header_list=["ksplit_gemm_data.h"],
    )


if __name__ == "__main__":
    main()
