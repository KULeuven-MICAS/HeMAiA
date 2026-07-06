#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""
Four-chiplet, two-cluster GEMM M-split workload.

A is split into A1..A8 and B is stored in the memory chip.  The tile mapping is:
  chiplet 00 cluster0/1 -> A1/A2 -> D1/D2
  chiplet 01 cluster0/1 -> A3/A4 -> D3/D4
  chiplet 10 cluster0/1 -> A5/A6 -> D5/D6
  chiplet 11 cluster0/1 -> A7/A8 -> D7/D8

Flow:
  1. Load A1..A8 from the memory chip into local TCDM.
  2. Load B from the memory chip into chiplet 00 cluster 0 TCDM.
  3. Fan out B to the remote chiplets' cluster 0 TCDM with point-to-point D2D
     copies, then copy B from cluster 0 to cluster 1 inside each chiplet.
  4. Run GEMM on every active cluster and store each D tile from TCDM to the
     chiplet-local L3.
  5. Check each local L3 D tile against the corresponding golden D tile.
"""

import argparse
import os
import pathlib
import sys

import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from bingo_dfg import BingoDFG  # noqa E402
from bingo_helpers import chiplet_addr_transform_loc  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelCheckResultArgs,
    HostBingoKernelIdmaArgs,
    SnaxBingoKernelGemmFullArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemFixedAddr, BingoMemSymbol  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_chiplet_count, guard_cluster_count, parse_platform_cfg  # noqa E402
from gemm_multi_chiplet_datagen import emit_header_file  # noqa E402


WORKLOAD_NAME = "gemm_msplit_4chiplet_2cluster"
APP_NAME = "Multi-Chip GEMM M-Split 4 Chiplets 2 Clusters"
EXPECTED_CHIPLETS = [0x00, 0x01, 0x10, 0x11]
CLUSTER_IDS = [0, 1]

GEMM_CORE = 0
DMA_CORE = 1
XDMA_WIDTH_BYTES = 64
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


def align_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def chiplet_alloc_expr(chiplet_id, alloc):
    local_addr_expr = f"((uint64_t){alloc.get_c_var_name()} & {LOW_40_BIT_ADDR_MASK})"
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

    data_type = int(merged.get("data_type", 0))
    array_shape = merged["array_shape"]
    snax_acc_cfg = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]
    mesh_row, tile_size, mesh_col = unrolling

    m = merged["M"]
    k = merged["K"]
    n = merged["N"]
    tile_count = len(EXPECTED_CHIPLETS) * len(CLUSTER_IDS)
    raw_a_size = m * k * mesh_row * tile_size
    a_tile_size = align_up(raw_a_size, XDMA_WIDTH_BYTES)

    main_mem_base_addr = 0x8000_0000
    mem_pool_base_addr = chiplet_addr_transform_loc(2, 0, main_mem_base_addr)

    params = {
        "app_name": APP_NAME,
        "M": m,
        "K": k,
        "N": n,
        "meshRow": mesh_row,
        "tileSize": tile_size,
        "meshCol": mesh_col,
        "arrayShapeIdx": array_shape,
        "transposeA": merged.get("transposed_A", 0),
        "transposeB": merged.get("transposed_B", 0),
        "accumPrevC": merged.get("accumPrevC", 0),
        "Atile_size": a_tile_size,
        "B_size": k * n * tile_size * mesh_col,
        "C_size": m * n * mesh_row * mesh_col * 4,
        "D_size": m * n * mesh_row * mesh_col * 4,
        "tile_count": tile_count,
        "mem_pool_base_addr": mem_pool_base_addr,
        "B_mp_base_addr": mem_pool_base_addr + tile_count * a_tile_size,
    }
    return params, merged


def define_memory_handles(params):
    mem = {
        "A_mp": {},
        "A_golden_l3": {},
        "D_golden_l3": {},
        "A_l1": {},
        "B_l1": {},
        "D_l1": {},
        "D_l3": {},
    }

    for idx in range(params["tile_count"]):
        mem["A_mp"][idx] = BingoMemFixedAddr(
            params["mem_pool_base_addr"] + idx * params["Atile_size"]
        )
        mem["A_golden_l3"][idx] = BingoMemSymbol(
            "A_data_L3", offset=idx * params["Atile_size"]
        )
        mem["D_golden_l3"][idx] = BingoMemSymbol(
            "D_data_L3", offset=idx * params["D_size"]
        )
    mem["B_mp"] = BingoMemFixedAddr(params["B_mp_base_addr"])
    mem["B_golden_l3"] = BingoMemSymbol("B_data_L3")

    for chiplet in EXPECTED_CHIPLETS:
        h = chip_hex(chiplet)
        for cluster_id in CLUSTER_IDS:
            suffix = f"chip{h}_c{cluster_id}"
            key = (chiplet, cluster_id)
            mem["A_l1"][key] = BingoMemAlloc(
                f"A_l1_{suffix}",
                size=params["Atile_size"],
                mem_level="L1",
                chip_id=chiplet,
                cluster_id=cluster_id,
            )
            mem["B_l1"][key] = BingoMemAlloc(
                f"B_l1_{suffix}",
                size=params["B_size"],
                mem_level="L1",
                chip_id=chiplet,
                cluster_id=cluster_id,
            )
            mem["D_l1"][key] = BingoMemAlloc(
                f"D_l1_{suffix}",
                size=params["D_size"],
                mem_level="L1",
                chip_id=chiplet,
                cluster_id=cluster_id,
            )
            mem["D_l3"][key] = BingoMemAlloc(
                f"D_l3_{suffix}",
                size=params["D_size"],
                mem_level="L3",
                chip_id=chiplet,
            )

    return mem


def add_node(dfg, node):
    dfg.bingo_add_node(node)
    return node


def make_a_load_node(dfg, mem, params, chiplet, cluster_id, idx, host_core_id):
    h = chip_hex(chiplet)
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name=f"Load_A{idx + 1}_MemChip_to_Chip{h}_C{cluster_id}_TCDM",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem["A_mp"][idx],
                dst_addr=mem["A_l1"][(chiplet, cluster_id)],
                size=params["Atile_size"],
            ),
        ),
    )


def make_load_b_node(dfg, mem, params, host_core_id):
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name="Load_B_MemChip_to_Chip00_C0_TCDM",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem["B_mp"],
                dst_addr=mem["B_l1"][(0x00, 0)],
                size=params["B_size"],
            ),
        ),
    )


def make_fanout_b_node(dfg, mem, params, chiplet, host_core_id):
    h = chip_hex(chiplet)
    # The copy is emitted in chip00's host branch, where only chip00 allocation
    # variables are in scope.  The B buffers are allocated in the same order on
    # every chiplet, so retarget chip00's local B address to the remote chiplet.
    dst_addr = chiplet_alloc_expr(chiplet, mem["B_l1"][(0x00, 0)])
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=0x00,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name=f"Copy_B_Chip00_C0_to_Chip{h}_C0_TCDM",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem["B_l1"][(0x00, 0)],
                dst_addr=dst_addr,
                size=params["B_size"],
            ),
        ),
    )


def make_copy_b_to_cluster1_node(dfg, mem, params, chiplet, host_core_id):
    h = chip_hex(chiplet)
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name=f"Copy_B_Chip{h}_C0_to_C1_TCDM",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem["B_l1"][(chiplet, 0)],
                dst_addr=mem["B_l1"][(chiplet, 1)],
                size=params["B_size"],
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
            node_name=f"Gemm_A{idx + 1}_B_Chip{h}_C{cluster_id}",
            kernel_name="__snax_bingo_kernel_gemm_full",
            kernel_args=SnaxBingoKernelGemmFullArgs(
                input_A_addr=mem["A_l1"][(chiplet, cluster_id)],
                input_B_addr=mem["B_l1"][(chiplet, cluster_id)],
                input_C_addr=0,
                output_D_addr=mem["D_l1"][(chiplet, cluster_id)],
                M=params["M"],
                K=params["K"],
                N=params["N"],
                array_shape_idx=params["arrayShapeIdx"],
                transpose_A=params["transposeA"],
                transpose_B=params["transposeB"],
                accumPrevC=params["accumPrevC"],
            ),
        ),
    )


def make_store_d_node(dfg, mem, params, chiplet, cluster_id, idx, host_core_id):
    h = chip_hex(chiplet)
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name=f"Store_D{idx + 1}_Chip{h}_C{cluster_id}_TCDM_to_L3",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem["D_l1"][(chiplet, cluster_id)],
                dst_addr=mem["D_l3"][(chiplet, cluster_id)],
                size=params["D_size"],
            ),
        ),
    )


def make_check_d_node(dfg, mem, params, chiplet, cluster_id, idx, host_core_id):
    h = chip_hex(chiplet)
    return add_node(
        dfg,
        BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,
            node_name=f"Check_D{idx + 1}_Chip{h}_C{cluster_id}_L3",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                name=f"D{idx + 1}_chip{h}_c{cluster_id}",
                golden_data_addr=mem["D_golden_l3"][idx],
                output_data_addr=mem["D_l3"][(chiplet, cluster_id)],
                data_size=params["D_size"],
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

    load_a = {}
    ordered_loads = []
    for chiplet_pos, chiplet in enumerate(EXPECTED_CHIPLETS):
        for cluster_id in CLUSTER_IDS:
            idx = data_index(chiplet_pos, cluster_id)
            load_a[(chiplet, cluster_id)] = make_a_load_node(
                dfg, mem, params, chiplet, cluster_id, idx, host_core_id
            )
            ordered_loads.append(load_a[(chiplet, cluster_id)])

    load_b = make_load_b_node(dfg, mem, params, host_core_id)
    for prev_node, next_node in zip(ordered_loads, ordered_loads[1:]):
        dfg.bingo_add_edge(prev_node, next_node)
    dfg.bingo_add_edge(ordered_loads[-1], load_b)

    # The D2D broadcast address includes the source chiplet and causes the
    # source write leg to wait indefinitely. Use explicit remote copies instead;
    # chip00 cluster0 already has B from load_b.
    prev_node = load_b
    for chiplet in EXPECTED_CHIPLETS[1:]:
        fanout_b = make_fanout_b_node(dfg, mem, params, chiplet, host_core_id)
        dfg.bingo_add_edge(prev_node, fanout_b)
        prev_node = fanout_b

    copy_b_to_c1 = {}
    for chiplet in EXPECTED_CHIPLETS:
        copy_b_to_c1[chiplet] = make_copy_b_to_cluster1_node(
            dfg, mem, params, chiplet, host_core_id
        )
        dfg.bingo_add_edge(prev_node, copy_b_to_c1[chiplet])
        prev_node = copy_b_to_c1[chiplet]

    for chiplet_pos, chiplet in enumerate(EXPECTED_CHIPLETS):
        for cluster_id in CLUSTER_IDS:
            idx = data_index(chiplet_pos, cluster_id)
            gemm = make_gemm_node(dfg, mem, params, chiplet, cluster_id, idx)
            store = make_store_d_node(
                dfg, mem, params, chiplet, cluster_id, idx, host_core_id
            )
            check = make_check_d_node(
                dfg, mem, params, chiplet, cluster_id, idx, host_core_id
            )

            dfg.bingo_add_edge(prev_node, gemm)
            dfg.bingo_add_edge(gemm, store)
            dfg.bingo_add_edge(store, check)
            prev_node = check

    return dfg


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
    dfg = create_dfg(params, mem, platform)

    print("Built DFG: load A1..A8, fan out B to cluster0, copy B to cluster1, GEMM/check D1..D8")
    print(f"  active_chiplets={[chip_hex(c) for c in EXPECTED_CHIPLETS]}")
    print(f"  active_clusters={CLUSTER_IDS}")
    print(
        f"  Atile_size={params['Atile_size']}, B_size={params['B_size']}, "
        f"D_size={params['D_size']}"
    )
    print(f"  mempool_base=0x{params['mem_pool_base_addr']:x}")
    print(f"  B_mempool_addr=0x{params['B_mp_base_addr']:x}")

    dfg.bingo_compile_dfg(
        params["app_name"],
        output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["gemm_data.h"],
    )


if __name__ == "__main__":
    main()
