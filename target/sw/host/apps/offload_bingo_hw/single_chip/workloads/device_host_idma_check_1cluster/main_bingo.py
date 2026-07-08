#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""
Regression workload for Bingo HW dependency generation.

The critical chain is on chiplet 0:
  device iDMA copy -> host iDMA golden load -> host check

The check must wait for the host iDMA node, not merely for the preceding device
node. If dependency tags are reused incorrectly, the check can observe stale or
uninitialized L3 golden data.
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
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelCheckResultArgs,
    HostBingoKernelIdmaArgs,
    SnaxBingoKernelIdma1dCopyArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from device_host_idma_check_datagen import emit_header_file  # noqa E402


DMA_CORE = 1
HOST_CORE = 2
APP_NAME = "device_host_idma_check_1cluster"


def get_args():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    size = int(param["size"])
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    if args.data_h is not None:
        with args.data_h.open("w") as f:
            f.write(emit_header_file(size))
        print(f"Written data header: {args.data_h}")

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

    l1_output = BingoMemAlloc("l1_output", size=size, mem_level="L1",
                              chip_id=0, cluster_id=0)
    l3_golden = BingoMemAlloc("l3_golden", size=size, mem_level="L3", chip_id=0)

    device_copy = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=DMA_CORE,
        node_name="Device_Copy_Input_to_L1",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=BingoMemSymbol("input_data"),
            dst_addr=l1_output,
            size=size,
        ),
    )
    host_load_golden = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=HOST_CORE,
        node_name="Host_Load_Golden_to_L3",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=BingoMemSymbol("golden_data"),
            dst_addr=l3_golden,
            size=size,
        ),
    )
    host_check = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=HOST_CORE,
        node_name="Host_Check_Device_Output",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=l3_golden,
            output_data_addr=l1_output,
            data_size=size,
            name="device_host_idma_check",
        ),
    )

    for node in (device_copy, host_load_golden, host_check):
        dfg.bingo_add_node(node)

    dfg.bingo_add_edge(device_copy, host_load_golden)
    dfg.bingo_add_edge(host_load_golden, host_check)

    dfg.bingo_compile_dfg(
        APP_NAME,
        args.output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["device_host_idma_check_data.h"],
    )
    print(f"Generated {APP_NAME} DFG")


if __name__ == "__main__":
    main()
