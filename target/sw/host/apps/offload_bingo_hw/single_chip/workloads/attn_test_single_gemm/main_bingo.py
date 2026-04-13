#!/usr/bin/env python3
"""
Attention test: single GEMM on one cluster.

DFG: Load_A → Load_B → GEMM_full → Store_D → Check_D

Follows the exact pattern of gemm_stacked/main_bingo.py but simplified
to a single GEMM. Every BingoNode has explicit kernel_args.
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

from gemm_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    HostBingoKernelCheckResultArgs,
)


def parse_platform_cfg(occamy_h_path, rtlcfg_path):
    """Parse HW platform parameters from generated occamy.h and RTL config.

    - occamy.h provides: N_CHIPLETS, N_CLUSTERS_PER_CHIPLET, N_CORES_PER_CLUSTER
    - RTL config (hemaia_*.hjson) provides: chiplet coordinates → chiplet_ids

    Chiplet IDs are coordinate-encoded: (x, y) → (x << 4) | y
    E.g. 2×2 grid: (0,0)=0x00, (0,1)=0x01, (1,0)=0x10, (1,1)=0x11
    """
    # Parse occamy.h #defines
    defines = {}
    with open(occamy_h_path) as f:
        for line in f:
            m = re.match(r'#define\s+(\w+)\s+(\d+)', line)
            if m:
                defines[m.group(1)] = int(m.group(2))

    # Parse RTL config for chiplet coordinates
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
    parser = argparse.ArgumentParser(description="attn_test_single_gemm")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True,
                        help="Path to generated occamy.h with HW platform defines")
    parser.add_argument("--rtlcfg", type=pathlib.Path, required=True,
                        help="Path to hemaia RTL config hjson (e.g. hemaia_bingo.hjson)")
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

    # Parse HW platform parameters from generated occamy.h + RTL config
    platform = parse_platform_cfg(args.platformcfg, args.rtlcfg)

    # Derive params
    array_shape = merged["array_shape"]
    data_type = 0  # int8
    snax_acc_cfg = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, tileSize, meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    M = merged["seq_len"] // meshRow
    K = merged["d_model"] // tileSize
    N = merged["d_head"] // meshCol

    A_size = M * K * meshRow * tileSize       # int8 bytes
    B_size = K * N * tileSize * meshCol        # int8 bytes
    D_size = M * N * meshRow * meshCol * 4     # int32 bytes

    transpose_A = merged.get("transposed_A", 0)
    transpose_B = merged.get("transposed_B", 0)
    accumPrevC = merged.get("accumPrevC", 0)

    # Core IDs
    GEMM_CORE = 0
    DMA_CORE = 1
    HOST_CORE = 2

    # ── Memory handles ──
    # L3 symbols (golden data from gemm_data.h)
    sym_A = BingoMemSymbol("A")
    sym_B = BingoMemSymbol("B")
    sym_D = BingoMemSymbol("D")

    # L1 buffers (runtime allocation on cluster 0)
    l1_A = BingoMemAlloc("A_l1", size=A_size, mem_level="L1", chip_id=0, cluster_id=0)
    l1_B = BingoMemAlloc("B_l1", size=B_size, mem_level="L1", chip_id=0, cluster_id=0)
    l1_D = BingoMemAlloc("D_l1", size=D_size, mem_level="L1", chip_id=0, cluster_id=0)

    # L3 output buffer (runtime)
    l3_D = BingoMemAlloc("D_l3", size=D_size, mem_level="L3")

    # ── DFG ──
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    # Load A: L3 → L1 via device iDMA
    node_load_A = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Load_A",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=sym_A, dst_addr=l1_A, size=A_size,
        ),
    )

    # Load B: L3 → L1 via device iDMA
    node_load_B = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Load_B",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=sym_B, dst_addr=l1_B, size=B_size,
        ),
    )

    # GEMM: A @ B → D
    node_gemm = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=GEMM_CORE,
        node_name="Gemm",
        kernel_name="__snax_bingo_kernel_gemm_full",
        kernel_args=SnaxBingoKernelGemmFullArgs(
            input_A_addr=l1_A, input_B_addr=l1_B,
            input_C_addr=0, output_D_addr=l1_D,
            M=M, K=K, N=N,
            array_shape_idx=array_shape,
            transpose_A=transpose_A, transpose_B=transpose_B,
            accumPrevC=accumPrevC,
        ),
    )

    # Store D: L1 → L3
    node_store_D = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Store_D",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=l1_D, dst_addr=l3_D, size=D_size,
        ),
    )

    # Check D: compare L3 output vs golden
    node_check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_D",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=sym_D, output_data_addr=l3_D,
            data_size=D_size,
            name="D",
        ),
    )

    # Add nodes
    for n in [node_load_A, node_load_B, node_gemm, node_store_D, node_check]:
        dfg.bingo_add_node(n)

    # Edges: Load_A → Load_B (serialize DMA on same core)
    #         Load_B → GEMM (data ready)
    #         GEMM → Store_D (compute done)
    #         Store_D → Check_D (data in L3)
    dfg.bingo_add_edge(node_load_A, node_load_B)
    dfg.bingo_add_edge(node_load_B, node_gemm)
    dfg.bingo_add_edge(node_gemm, node_store_D)
    dfg.bingo_add_edge(node_store_D, node_check)

    print(f"Built DFG: 5 nodes (Load_A → Load_B → GEMM → Store_D → Check_D)")
    print(f"  M={M}, K={K}, N={N}, array_shape={array_shape}")
    print(f"  A_size={A_size}, B_size={B_size}, D_size={D_size}")

    dfg.bingo_compile_dfg(
        "attn_test_single_gemm", output_dir, args.output_offload_file_name,
        extra_include_header_list=["gemm_data.h"],
    )


if __name__ == "__main__":
    main()
