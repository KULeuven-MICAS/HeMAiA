#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>
import argparse
import os
import pathlib
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
APP_NAME = "Single-Chip GEMM Memory Chip"

cur_chiplet_id = 0x00
cur_cluster_id = 0

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim")
sys.path.append(current_dir)

from bingo_dfg import BingoDFG  # noqa E402
from bingo_helpers import chiplet_addr_transform_loc  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelCheckResultArgs,
    HostBingoKernelIdmaArgs,
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelIdma1dCopyArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemFixedAddr  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from gemm_mem_chip_datagen import emit_header_file  # noqa E402
from gemm_sim_utils import (  # noqa E402
    _bytes_for_elements,
    _gemm_operand_widths,
    define_gemm_workload_params,
)


def get_args():
    parser = argparse.ArgumentParser(description="Bingo HW Manager")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument(
        "--output_offload_file_name",
        type=str,
        default="offload_bingo_hw.h",
    )
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def define_workload_params(cfg_path, hwcfg_path):
    params, merged = define_gemm_workload_params(cfg_path, hwcfg_path)
    a_len, _, _, _ = _gemm_operand_widths(merged)
    a_elements = params["M"] * params["K"] * params["meshRow"] * params["tileSize"]
    params["A_mem_size"] = _bytes_for_elements(a_elements + (-a_elements) % 64, a_len)
    mempool_base = chiplet_addr_transform_loc(2, 0, 0x8000_0000)
    params["A_mp_base_addr"] = mempool_base
    params["B_mp_base_addr"] = mempool_base + params["A_mem_size"]
    params["D_mp_base_addr"] = params["B_mp_base_addr"] + params["B_size"]
    return params, merged


def define_memory_handles(params):
    mem_handles = {}
    mem_handles["A_mp"] = BingoMemFixedAddr(params["A_mp_base_addr"])
    mem_handles["B_mp"] = BingoMemFixedAddr(params["B_mp_base_addr"])
    mem_handles["D_mp"] = BingoMemFixedAddr(params["D_mp_base_addr"])
    mem_handles["l1_buf_A"] = BingoMemAlloc(
        "l1_buf_A",
        size=params["A_mem_size"],
        mem_level="L1",
        chip_id=cur_chiplet_id,
        cluster_id=cur_cluster_id,
    )
    mem_handles["l1_buf_B"] = BingoMemAlloc(
        "l1_buf_B",
        size=params["B_size"],
        mem_level="L1",
        chip_id=cur_chiplet_id,
        cluster_id=cur_cluster_id,
    )
    mem_handles["l1_buf_D"] = BingoMemAlloc(
        "l1_buf_D",
        size=params["D_size"],
        mem_level="L1",
        chip_id=cur_chiplet_id,
        cluster_id=cur_cluster_id,
    )
    mem_handles["l3_buf_D"] = BingoMemAlloc(
        "l3_buf_D", size=params["D_size"], mem_level="L3", chip_id=cur_chiplet_id
    )
    mem_handles["l3_golden_D"] = BingoMemAlloc(
        "l3_golden_D", size=params["D_size"], mem_level="L3", chip_id=cur_chiplet_id
    )
    return mem_handles


def create_dfg(params, mem_handles, platform):
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    gemm_core_id = 0
    dma_core_id = 1
    host_core_id = 2

    load_A = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=dma_core_id,
        node_name="Load_A_From_MemChip",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["A_mp"],
            dst_addr=mem_handles["l1_buf_A"],
            size=params["A_mem_size"],
        ),
    )
    load_B = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=host_core_id,
        node_name="Load_B_From_MemChip",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=mem_handles["B_mp"],
            dst_addr=mem_handles["l1_buf_B"],
            size=params["B_size"],
        ),
    )
    gemm = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=gemm_core_id,
        node_name="Gemm_A_B",
        kernel_name="__snax_bingo_kernel_gemm_full",
        kernel_args=SnaxBingoKernelGemmFullArgs(
            input_A_addr=mem_handles["l1_buf_A"],
            input_B_addr=mem_handles["l1_buf_B"],
            input_C_addr=0,
            output_D_addr=mem_handles["l1_buf_D"],
            M=params["M"],
            K=params["K"],
            N=params["N"],
            array_shape_idx=params["arrayShapeIdx"],
            transpose_A=params["transposeA"],
            transpose_B=params["transposeB"],
            accumPrevC=params["accumPrevC"],
            quantization_enable=params["quantization_enable"],
            shift_i=params["shift_i"],
            multiplier_i=params["multiplier_i"],
            input_zp_i=params["input_zp_i"],
            output_zp_i=params["output_zp_i"],
            int32tofp16_enable=params["int32tofp16_enable"],
            int4_a_enable=params["int4_a_enable"],
            int4_b_enable=params["int4_b_enable"],
        ),
    )
    store_D = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=dma_core_id,
        node_name="Store_D",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["l1_buf_D"],
            dst_addr=mem_handles["l3_buf_D"],
            size=params["D_size"],
        ),
    )
    load_golden_D = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=host_core_id,
        node_name="Load_Golden_D_From_MemChip",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=mem_handles["D_mp"],
            dst_addr=mem_handles["l3_golden_D"],
            size=params["D_size"],
        ),
    )
    check_D = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=host_core_id,
        node_name="Check_D",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            name="D",
            golden_data_addr=mem_handles["l3_golden_D"],
            output_data_addr=mem_handles["l3_buf_D"],
            data_size=min(64, params["D_size"]),
        ),
    )

    for node in (load_A, load_B, gemm, store_D, load_golden_D, check_D):
        dfg.bingo_add_node(node)

    dfg.bingo_add_edge(load_A, gemm)
    dfg.bingo_add_edge(load_B, gemm)
    dfg.bingo_add_edge(gemm, store_D)
    dfg.bingo_add_edge(store_D, check_D)
    dfg.bingo_add_edge(load_golden_D, check_D)
    return dfg


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    params, merged_config = define_workload_params(args.cfg, args.hwcfg)
    if args.data_h is not None:
        data_h_content = emit_header_file(
            **merged_config, out_dir=os.path.join(args.output_dir, "build")
        )
        with open(args.data_h, "w") as f:
            f.write(data_h_content)
        print(f"Written data header: {args.data_h}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(
        merged_config, platform, args.output_dir, args.output_offload_file_name
    ):
        return

    mem_handles = define_memory_handles(params)
    dfg = create_dfg(params, mem_handles, platform)
    dfg.bingo_compile_dfg(
        APP_NAME,
        args.output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["gemm_data.h"],
    )


if __name__ == "__main__":
    main()
