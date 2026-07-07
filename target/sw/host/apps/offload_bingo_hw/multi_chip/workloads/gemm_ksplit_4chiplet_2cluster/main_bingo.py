#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""
K-split GEMM test across 4 chiplets with two clusters per chiplet.
We split the K dimension into 8 chunks, each chunk is computed by one cluster.
A1,B1 to cluster 0 of chiplet 0, A2,B2 to cluster 1 of chiplet 0,
A3,B3 to cluster 0 of chiplet 1, A4,B4 to cluster 1 of chiplet 1, and so on.

Each cluster computes one K chunk:
  Load A_i, load B_i, GEMM.

Partial D_i buffers are reduced on cluster 0 of chiplet 0. Remote partials are
staged into chiplet-local L3 symbols, then copied directly into chiplet 0
cluster 0 TCDM receive slots. The graph only checks the final INT32 reduction
result and the final dequantized FP32 output.

Remote partials are staged from L1/TCDM back to chiplet-local L3 because the L1
buffers are dynamic BingoMemAlloc objects. Their generated C pointer variables
only exist inside the remote chiplet's kernel_execution block, so chip00 cannot
compile a direct reference to them. The L3 staging buffer is a static symbol, so
chip00 can form a valid remote address with chiplet_addr_transform_full().

Task dependency graph:

  Per-cluster compute lanes:
    k0 chip00 c0: Load_A0 -> Load_B0 -> Gemm_k0
    k1 chip00 c1: Load_A1 -> Load_B1 -> Gemm_k1 -> Copy_D_k1_to_chip00_c0_tcdm
    k2 chip01 c0: Load_A2 -> Load_B2 -> Gemm_k2 -> Store_D_k2_to_chip01_l3
                  -> Pull_D_k2_to_chip00_c0_tcdm
    k3 chip01 c1: Load_A3 -> Load_B3 -> Gemm_k3 -> Store_D_k3_to_chip01_l3
                  -> Pull_D_k3_to_chip00_c0_tcdm
    k4 chip10 c0: Load_A4 -> Load_B4 -> Gemm_k4 -> Store_D_k4_to_chip10_l3
                  -> Pull_D_k4_to_chip00_c0_tcdm
    k5 chip10 c1: Load_A5 -> Load_B5 -> Gemm_k5 -> Store_D_k5_to_chip10_l3
                  -> Pull_D_k5_to_chip00_c0_tcdm
    k6 chip11 c0: Load_A6 -> Load_B6 -> Gemm_k6 -> Store_D_k6_to_chip11_l3
                  -> Pull_D_k6_to_chip00_c0_tcdm
    k7 chip11 c1: Load_A7 -> Load_B7 -> Gemm_k7 -> Store_D_k7_to_chip11_l3
                  -> Pull_D_k7_to_chip00_c0_tcdm

  Chip00 c0 reduction and checks:
    Gemm_k0 + Copy_D_k1_to_chip00_c0_tcdm -> Reduce_Add_k0_to_k1
    Reduce_Add_k0_to_k1 + Pull_D_k2_to_chip00_c0_tcdm -> Reduce_Add_k0_to_k2
    Reduce_Add_k0_to_k2 + Pull_D_k3_to_chip00_c0_tcdm -> Reduce_Add_k0_to_k3
    Reduce_Add_k0_to_k3 + Pull_D_k4_to_chip00_c0_tcdm -> Reduce_Add_k0_to_k4
    Reduce_Add_k0_to_k4 + Pull_D_k5_to_chip00_c0_tcdm -> Reduce_Add_k0_to_k5
    Reduce_Add_k0_to_k5 + Pull_D_k6_to_chip00_c0_tcdm -> Reduce_Add_k0_to_k6
    Reduce_Add_k0_to_k6 + Pull_D_k7_to_chip00_c0_tcdm -> Reduce_Add_k0_to_k7
    Reduce_Add_k0_to_k7 -> Load_Golden_Final_i32_D -> Check_Final_i32_D
    Check_Final_i32_D -> Dequant_Final_i32_to_fp32
    Dequant_Final_i32_to_fp32 -> Load_Golden_fp32_D -> Check_fp32_D
"""

import argparse
import os
import pathlib
import sys

import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from bingo_dfg import BingoDFG  # noqa E402
from bingo_helpers import chiplet_addr_transform_loc  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelAraAddI32Args,
    HostBingoKernelAraDequantizeI32F32Args,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelIdma1dCopyArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemFixedAddr, BingoMemSymbol  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_chiplet_count, guard_cluster_count, parse_platform_cfg  # noqa E402
from ksplit_gemm_multi_chiplet_datagen import emit_header_file  # noqa E402


WORKLOAD_NAME = "gemm_ksplit_4chiplet_2cluster"
APP_NAME = "Multi-Chip GEMM K-Split 4 Chiplets 2 Clusters"
EXPECTED_CHIPLETS = [0x00, 0x01, 0x10, 0x11]
CLUSTER_IDS = [0, 1]

GEMM_CORE = 0
DMA_CORE = 1
XDMA_WIDTH_BYTES = 64
MEMPOOL_LOC_X = 2
MEMPOOL_LOC_Y = 0
MEMPOOL_BASE_LOCAL = 0x8000_0000
LOW_40_BIT_ADDR_MASK = "0x000000ffffffffffULL"


def get_args():
    parser = argparse.ArgumentParser(description=WORKLOAD_NAME)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def chip_hex(chiplet_id):
    return f"{chiplet_id:02x}"


def data_index(chiplet_pos, cluster_id):
    return chiplet_pos * len(CLUSTER_IDS) + cluster_id


def chiplet_symbol_expr(chiplet_id, symbol_name, byte_offset=0):
    local_addr_expr = f"((uint64_t)((uintptr_t){symbol_name}) & {LOW_40_BIT_ADDR_MASK})"
    if byte_offset:
        local_addr_expr = f"({local_addr_expr} + {byte_offset})"
    return f"(chiplet_addr_transform_full(0x{chiplet_id:02x}, {local_addr_expr}))"


def load_hjson(path):
    with path.open() as f:
        return hjson.loads(f.read())


def validate_platform(platform):
    chiplets = set(platform["chiplet_ids"])
    missing = [chip for chip in EXPECTED_CHIPLETS if chip not in chiplets]
    if missing:
        raise ValueError(
            f"{WORKLOAD_NAME} expects chiplets "
            f"{[chip_hex(c) for c in EXPECTED_CHIPLETS]}, "
            f"but platform is missing {[chip_hex(c) for c in missing]}"
        )


def define_workload_params(cfg_path, hwcfg_path):
    param = load_hjson(cfg_path)
    hw = load_hjson(hwcfg_path)
    merged = {**param, **hw}

    expected_k_split = len(EXPECTED_CHIPLETS) * len(CLUSTER_IDS)
    k_split = int(merged["k_split"])
    if k_split != expected_k_split:
        raise ValueError(f"{WORKLOAD_NAME} expects k_split={expected_k_split}, got {k_split}")

    array_shape = merged["array_shape"]
    data_type = int(merged.get("data_type", 0))
    snax_acc_cfg = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]
    mesh_row, tile_size, mesh_col = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    m = int(merged["M"])
    k = int(merged["K"])
    n = int(merged["N"])
    if k % k_split != 0:
        raise ValueError(f"K ({k}) must be divisible by k_split ({k_split})")

    k_tile = k // k_split
    a_tile_bytes = m * k_tile * mesh_row * tile_size
    b_tile_bytes = k_tile * n * tile_size * mesh_col
    d_num_elements = m * n * mesh_row * mesh_col
    d_bytes = d_num_elements * 4
    if d_bytes % XDMA_WIDTH_BYTES != 0:
        raise ValueError(
            f"D_bytes ({d_bytes}) must be {XDMA_WIDTH_BYTES}-byte aligned "
            "for cross-chiplet partial transfers"
        )

    mempool_base = chiplet_addr_transform_loc(
        MEMPOOL_LOC_X,
        MEMPOOL_LOC_Y,
        MEMPOOL_BASE_LOCAL,
    )
    a_mp_base = mempool_base
    b_mp_base = a_mp_base + k_split * a_tile_bytes
    golden_d_mp_base = b_mp_base + k_split * b_tile_bytes
    golden_sum_mp_base = golden_d_mp_base + k_split * d_bytes
    golden_fp32_mp_base = golden_sum_mp_base + (k_split - 1) * d_bytes
    combined_scale_mp_base = golden_fp32_mp_base + d_bytes

    params = {
        "app_name": APP_NAME,
        "M": m,
        "K": k,
        "N": n,
        "K_tile": k_tile,
        "k_split": k_split,
        "meshRow": mesh_row,
        "tileSize": tile_size,
        "meshCol": mesh_col,
        "array_shape": array_shape,
        "data_type": data_type,
        "transpose_A": merged.get("transposed_A", 0),
        "transpose_B": merged.get("transposed_B", 0),
        "accumPrevC": merged.get("accumPrevC", 0),
        "A_tile_bytes": a_tile_bytes,
        "B_tile_bytes": b_tile_bytes,
        "D_num_elements": d_num_elements,
        "D_bytes": d_bytes,
        "fp32_D_bytes": d_bytes,
        "mempool_base": mempool_base,
        "A_mp_base": a_mp_base,
        "B_mp_base": b_mp_base,
        "golden_sum_final_mp": golden_sum_mp_base + (k_split - 2) * d_bytes,
        "golden_fp32_mp": golden_fp32_mp_base,
        "combined_scale_mp": combined_scale_mp_base,
    }
    return params, merged


def define_memory_handles(params):
    mem = {
        "A_mp": {},
        "B_mp": {},
        "A_l1": {},
        "B_l1": {},
        "D_l1": {},
        "D_reduce_l1": {},
        "D_remote_l3": {},
        "D_remote_l3_full": {},
    }

    for idx in range(params["k_split"]):
        mem["A_mp"][idx] = BingoMemFixedAddr(
            params["A_mp_base"] + idx * params["A_tile_bytes"]
        )
        mem["B_mp"][idx] = BingoMemFixedAddr(
            params["B_mp_base"] + idx * params["B_tile_bytes"]
        )

    for chiplet_pos, chiplet in enumerate(EXPECTED_CHIPLETS):
        h = chip_hex(chiplet)
        for cluster_id in CLUSTER_IDS:
            idx = data_index(chiplet_pos, cluster_id)
            suffix = f"k{idx}_chip{h}_c{cluster_id}"
            key = (chiplet, cluster_id)
            mem["A_l1"][key] = BingoMemAlloc(
                f"A_l1_{suffix}",
                size=params["A_tile_bytes"],
                mem_level="L1",
                chip_id=chiplet,
                cluster_id=cluster_id,
            )
            mem["B_l1"][key] = BingoMemAlloc(
                f"B_l1_{suffix}",
                size=params["B_tile_bytes"],
                mem_level="L1",
                chip_id=chiplet,
                cluster_id=cluster_id,
            )
            mem["D_l1"][key] = BingoMemAlloc(
                f"D_l1_{suffix}",
                size=params["D_bytes"],
                mem_level="L1",
                chip_id=chiplet,
                cluster_id=cluster_id,
            )
            if chiplet != 0x00:
                l3_offset = cluster_id * params["D_bytes"]
                mem["D_remote_l3"][key] = BingoMemSymbol(
                    "D_partial_local_l3",
                    offset=l3_offset,
                )
                mem["D_remote_l3_full"][key] = chiplet_symbol_expr(
                    chiplet,
                    "D_partial_local_l3",
                    l3_offset,
                )

    mem["D_reduce_l1"][0] = mem["D_l1"][(0x00, 0)]
    for idx in range(1, params["k_split"]):
        mem["D_reduce_l1"][idx] = BingoMemAlloc(
            f"D_reduce_k{idx}_chip00_c0_l1",
            size=params["D_bytes"],
            mem_level="L1",
            chip_id=0x00,
            cluster_id=0,
        )

    mem["golden_sum_final_mp"] = BingoMemFixedAddr(params["golden_sum_final_mp"])
    mem["golden_fp32_mp"] = BingoMemFixedAddr(params["golden_fp32_mp"])
    mem["combined_scale_mp"] = BingoMemFixedAddr(params["combined_scale_mp"])
    mem["golden_sum_final_l3"] = BingoMemAlloc(
        "golden_sum_final_l3",
        size=params["D_bytes"],
        mem_level="L3",
        chip_id=0x00,
    )
    mem["fp32_D_l3"] = BingoMemAlloc(
        "fp32_D_l3",
        size=params["fp32_D_bytes"],
        mem_level="L3",
        chip_id=0x00,
    )
    mem["golden_fp32_l3"] = BingoMemAlloc(
        "golden_fp32_D_l3",
        size=params["fp32_D_bytes"],
        mem_level="L3",
        chip_id=0x00,
    )

    return mem


def add_node(dfg, node):
    dfg.bingo_add_node(node)
    return node


def make_load_node(dfg, mem, params, chiplet, cluster_id, idx, kind):
    h = chip_hex(chiplet)
    src_key = f"{kind}_mp"
    dst_key = f"{kind}_l1"
    size_key = f"{kind}_tile_bytes"
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=cluster_id,
            assigned_core_id=DMA_CORE,
            node_name=f"Load_{kind}_k{idx}_Chip{h}_C{cluster_id}_TCDM",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem[src_key][idx],
                dst_addr=mem[dst_key][(chiplet, cluster_id)],
                size=params[size_key],
            ),
        ),
    )


def make_gemm_node(dfg, mem, params, chiplet, cluster_id, idx):
    h = chip_hex(chiplet)
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=cluster_id,
            assigned_core_id=GEMM_CORE,
            node_name=f"Gemm_k{idx}_Chip{h}_C{cluster_id}",
            kernel_name="__snax_bingo_kernel_gemm_full",
            kernel_args=SnaxBingoKernelGemmFullArgs(
                input_A_addr=mem["A_l1"][(chiplet, cluster_id)],
                input_B_addr=mem["B_l1"][(chiplet, cluster_id)],
                input_C_addr=0,
                output_D_addr=mem["D_l1"][(chiplet, cluster_id)],
                M=params["M"],
                K=params["K_tile"],
                N=params["N"],
                array_shape_idx=params["array_shape"],
                transpose_A=params["transpose_A"],
                transpose_B=params["transpose_B"],
                accumPrevC=params["accumPrevC"],
            ),
        ),
    )


def make_local_partial_copy_node(dfg, mem, params, cluster_id, idx):
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=cluster_id,
            assigned_core_id=DMA_CORE,
            node_name=f"Copy_D_k{idx}_Chip00_C{cluster_id}_to_Chip00_C0_TCDM",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem["D_l1"][(0x00, cluster_id)],
                dst_addr=mem["D_reduce_l1"][idx],
                size=params["D_bytes"],
            ),
        ),
    )


def make_remote_partial_store_node(dfg, mem, params, chiplet, cluster_id, idx):
    # Stage remote partial D into a static L3 symbol before chip00 pulls it.
    # Chip00 cannot reference the remote chiplet's dynamic L1 allocation
    # variable, but it can address this symbol through chiplet_addr_transform_full.
    h = chip_hex(chiplet)
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=cluster_id,
            assigned_core_id=DMA_CORE,
            node_name=f"Store_D_k{idx}_Chip{h}_C{cluster_id}_TCDM_to_Local_L3",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem["D_l1"][(chiplet, cluster_id)],
                dst_addr=mem["D_remote_l3"][(chiplet, cluster_id)],
                size=params["D_bytes"],
            ),
        ),
    )


def make_remote_partial_copy_node(dfg, mem, params, chiplet, cluster_id, idx, host_core_id):
    h = chip_hex(chiplet)
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name=f"Pull_D_k{idx}_Chip{h}_C{cluster_id}_to_Chip00_C0_TCDM",
            kernel_name="__host_bingo_kernel_xdma_1d_copy",
            kernel_args=HostBingoKernelXdma1dCopyArgs(
                src_addr=mem["D_remote_l3_full"][(chiplet, cluster_id)],
                dst_addr=mem["D_reduce_l1"][idx],
                size=params["D_bytes"],
            ),
        ),
    )


def create_dfg(params, mem, platform):
    host_core_id = platform["num_cores_per_cluster"]
    if host_core_id <= DMA_CORE:
        raise ValueError(
            f"{WORKLOAD_NAME} expects at least {DMA_CORE + 1} SNAX cores per cluster "
            f"before the host core, got {host_core_id}"
        )

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    reduce_ready = {}
    copy_nodes = {}

    for chiplet_pos, chiplet in enumerate(EXPECTED_CHIPLETS):
        for cluster_id in CLUSTER_IDS:
            idx = data_index(chiplet_pos, cluster_id)
            load_a = make_load_node(dfg, mem, params, chiplet, cluster_id, idx, "A")
            load_b = make_load_node(dfg, mem, params, chiplet, cluster_id, idx, "B")
            gemm = make_gemm_node(dfg, mem, params, chiplet, cluster_id, idx)

            dfg.bingo_add_edge(load_a, load_b)
            dfg.bingo_add_edge(load_b, gemm)

            if chiplet == 0x00 and cluster_id == 0:
                reduce_ready[idx] = gemm
            elif chiplet == 0x00:
                copy_nodes[idx] = make_local_partial_copy_node(
                    dfg,
                    mem,
                    params,
                    cluster_id,
                    idx,
                )
                dfg.bingo_add_edge(gemm, copy_nodes[idx])
                reduce_ready[idx] = copy_nodes[idx]
            else:
                store = make_remote_partial_store_node(
                    dfg,
                    mem,
                    params,
                    chiplet,
                    cluster_id,
                    idx,
                )
                copy_nodes[idx] = make_remote_partial_copy_node(
                    dfg,
                    mem,
                    params,
                    chiplet,
                    cluster_id,
                    idx,
                    host_core_id,
                )
                dfg.bingo_add_edge(gemm, store)
                dfg.bingo_add_edge(store, copy_nodes[idx])
                reduce_ready[idx] = copy_nodes[idx]

    prev_sum = mem["D_reduce_l1"][0]
    prev_ready = reduce_ready[0]
    for idx in range(1, params["k_split"]):
        add = add_node(
            dfg,
            BingoNode(
                assigned_chiplet_id=0x00,
                assigned_cluster_id=0,
                assigned_core_id=host_core_id,
                node_name=f"Reduce_Add_k0_to_k{idx}",
                kernel_name="__host_bingo_kernel_add_i32",
                kernel_args=HostBingoKernelAraAddI32Args(
                    input_a_addr=prev_sum,
                    input_b_addr=mem["D_reduce_l1"][idx],
                    output_addr=prev_sum,
                    num_elements=params["D_num_elements"],
                ),
            ),
        )
        dfg.bingo_add_edge(prev_ready, add)
        dfg.bingo_add_edge(reduce_ready[idx], add)
        prev_ready = add

    load_golden_sum = add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Load_Golden_Final_i32_D",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem["golden_sum_final_mp"],
                dst_addr=mem["golden_sum_final_l3"],
                size=params["D_bytes"],
            ),
        ),
    )
    check_final_i32 = add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_Final_i32_D",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem["golden_sum_final_l3"],
                output_data_addr=prev_sum,
                data_size=params["D_bytes"],
                name="final_i32_D",
            ),
        ),
    )
    dequant = add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Dequant_Final_i32_to_fp32",
            kernel_name="__host_bingo_kernel_dequantize_i32f32",
            kernel_args=HostBingoKernelAraDequantizeI32F32Args(
                input_addr=prev_sum,
                output_addr=mem["fp32_D_l3"],
                scale_addr=mem["combined_scale_mp"],
                num_elements=params["D_num_elements"],
            ),
        ),
    )
    load_golden_fp32 = add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Load_Golden_fp32_D",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem["golden_fp32_mp"],
                dst_addr=mem["golden_fp32_l3"],
                size=params["fp32_D_bytes"],
            ),
        ),
    )
    check_fp32 = add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Check_fp32_D",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem["golden_fp32_l3"],
                output_data_addr=mem["fp32_D_l3"],
                data_size=params["fp32_D_bytes"],
                name="fp32_D",
            ),
        ),
    )

    dfg.bingo_add_edge(prev_ready, load_golden_sum)
    dfg.bingo_add_edge(load_golden_sum, check_final_i32)
    dfg.bingo_add_edge(check_final_i32, dequant)
    dfg.bingo_add_edge(dequant, load_golden_fp32)
    dfg.bingo_add_edge(load_golden_fp32, check_fp32)

    return dfg, check_fp32


def main():
    args = get_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    params, merged_config = define_workload_params(args.cfg, args.hwcfg)

    if args.data_h is not None:
        build_dir = os.path.join(output_dir, "build")
        data_h_content = emit_header_file(**merged_config, out_dir=build_dir)
        with args.data_h.open("w") as f:
            f.write(data_h_content)
        print(f"Written data header: {args.data_h}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_chiplet_count(merged_config, platform, output_dir, args.output_offload_file_name):
        return
    if not guard_cluster_count(merged_config, platform, output_dir, args.output_offload_file_name):
        return
    validate_platform(platform)

    mem = define_memory_handles(params)
    dfg, check_fp32 = create_dfg(params, mem, platform)

    total_compute_nodes = params["k_split"] * 3
    total_remote_store_nodes = (len(EXPECTED_CHIPLETS) - 1) * len(CLUSTER_IDS)
    total_copy_nodes = params["k_split"] - 1
    total_reduce_nodes = params["k_split"] - 1
    total_final_nodes = 5
    total_nodes = (
        total_compute_nodes
        + total_remote_store_nodes
        + total_copy_nodes
        + total_reduce_nodes
        + total_final_nodes
    )
    print(
        "Built DFG: A/B load + GEMM on 8 clusters, stage remote partials, "
        "copy partials to chip00 C0 TCDM, reduce, final INT32 check, "
        "dequantize, FP32 check"
    )
    print(f"  active_chiplets={[chip_hex(c) for c in EXPECTED_CHIPLETS]}")
    print(f"  active_clusters={CLUSTER_IDS}")
    print(f"  nodes_before_entry_exit={total_nodes}")
    print(f"  M={params['M']}, K_tile={params['K_tile']}, N={params['N']}, k_split={params['k_split']}")
    print(
        f"  A_tile_bytes={params['A_tile_bytes']}, "
        f"B_tile_bytes={params['B_tile_bytes']}, D_bytes={params['D_bytes']}"
    )
    print(f"  mempool_base=0x{params['mempool_base']:x}")

    dfg.bingo_compile_dfg(
        params["app_name"],
        output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["ksplit_gemm_data.h"],
    )

    return check_fp32


if __name__ == "__main__":
    main()
