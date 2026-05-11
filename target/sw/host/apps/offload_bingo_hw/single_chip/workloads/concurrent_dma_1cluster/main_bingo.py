# Xiaoling Yi <xiaoling.yi@kuleuven.be>

# This file is the main entry point for the bingo offload application
# Users will create the dfg in this file
# And then the mini-compiler will emit the WORKLOAD.h file
import os
import sys
import argparse
import pathlib

current_dir = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR = os.path.dirname(current_dir)
sys.path.append(WORKLOADS_DIR)
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)
APP_NAME = "Single-Chip Concurrent DMA"

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim")

from bingo_dfg import BingoDFG
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    HostBingoKernelCheckResultArgs,
)
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402


def get_args():
    parser = argparse.ArgumentParser(description="Bingo HW Manager")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--output_offload_file_name",
        type=str,
        default="offload_bingo_hw.h",
        help="Output filename for the offload header file",
    )
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file (params.hjson)",
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        required=True,
        help="Select hardware config file",
    )
    parser.add_argument(
        "--platformcfg",
        type=pathlib.Path,
        required=True,
        help="Path to generated occamy.h with HW platform defines",
    )
    parser.add_argument(
        "--data_h",
        type=pathlib.Path,
        default=None,
        help="Output path for the generated data header (e.g. gemm_data.h). If omitted, data header is not written.",
    )
    return parser.parse_args()


def define_workload_params(cfg_path, _hwcfg_path):
    params = {}
    with open(cfg_path) as f:
        for line in f:
            line = line.split("//", 1)[0].split("#", 1)[0].strip()
            if not line or line in ("{", "}"):
                continue
            key, sep, value = line.partition(":")
            if not sep:
                continue
            product = 1
            value = value.strip().rstrip(",")
            for factor in value.split("*"):
                product *= int(factor.strip(), 0)
            params[key.strip()] = product

    for name in ("num_clusters", "A_size"):
        if name not in params:
            raise KeyError(f"params.hjson must define {name}")

    # Rule: one configured A_size is reused for both A1 and A2 DMA inputs.
    params["A1_size"] = params["A_size"]
    params["A2_size"] = params["A_size"]
    return params, dict(params)


def emit_header_file(**kwargs):
    a1_size = kwargs["A1_size"]
    a2_size = kwargs["A2_size"]
    a1_data = [idx & 0xFF for idx in range(a1_size)]
    a2_data = [(idx + 17) & 0xFF for idx in range(a2_size)]

    data_str = [
        "#include <stdint.h>",
        format_scalar_definition("uint32_t", "A1_size", a1_size),
        format_scalar_definition("uint32_t", "A2_size", a2_size),
        format_vector_definition("uint8_t", "A1_l3", a1_data),
        format_vector_definition("uint8_t", "A2_l3", a2_data),
    ]
    return "\n\n".join(data_str) + "\n"

# Chip 0, Cluster 0
cur_chiplet_id = 0
cur_cluster_id = 0

def define_memory_handles(params):
    """Defines memory symbols and handles."""
    mem_handles = {}

    # 1. Define Memory Symbols (Existing C variables)
    # The MemSymbol are the variables defined in the data.h file which the memory location is already known at compile time

    mem_handles['A1_data_L3_symbol'] = BingoMemSymbol("A1_l3",offset=0)
    mem_handles['A2_data_L3_symbol'] = BingoMemSymbol("A2_l3",offset=0)

    # 2. Define Memory Handles (Dynamic Allocations)
    # The MemHandles are the buffers that need to be allocated at runtime by the bingo runtime
    # L3 Buffers, None here

    # L1 Buffers
    mem_handles['A1_L1_buf'] = BingoMemAlloc(f"A1_L1_buf", size=params['A1_size'], mem_level="L1", chip_id=cur_chiplet_id, cluster_id=cur_cluster_id)
    mem_handles['A2_L1_buf'] = BingoMemAlloc(f"A2_L1_buf", size=params['A2_size'], mem_level="L1", chip_id=cur_chiplet_id, cluster_id=cur_cluster_id)

    return mem_handles

def create_dfg(params, mem_handles, platform):
    """Creates the Bingo Data Flow Graph with nodes and dependencies."""

    # id abstraction aligned with the cmd processor hw
    gemm_core_id = 0  # Core 0 for Compute
    dma_core_id = 1  # Core 1 for Load
    host_core_id = 2  # Core 2 for Host DMA Store

    # 1. Initialize DFG using HW params derived from occamy.h + RTL config
    bingo_dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    # 2. Define Nodes
    # Node: Copy A1 (Static A1 matrix, loaded once)
    task_copy_A1 = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=dma_core_id,
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=mem_handles['A1_data_L3_symbol'],
            dst_addr=mem_handles['A1_L1_buf'],
            size=params['A1_size']
        )
    )

    task_copy_A2 = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=dma_core_id,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles['A2_data_L3_symbol'],
            dst_addr=mem_handles['A2_L1_buf'],
            size=params['A2_size']
        )
    )

    task_check_result_A1 = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=host_core_id,
        kernel_name="__snax_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=mem_handles['A1_data_L3_symbol'],
            output_data_addr=mem_handles['A1_L1_buf'],
            data_size=params['A1_size']
        )
    )

    task_check_result_A2 = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=host_core_id,
        kernel_name="__snax_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=mem_handles['A2_data_L3_symbol'],
            output_data_addr=mem_handles['A2_L1_buf'],
            data_size=params['A2_size']
        )
    )

    # 3. Add Nodes to DFG
    bingo_dfg.bingo_add_node(task_copy_A1)
    bingo_dfg.bingo_add_node(task_copy_A2)
    bingo_dfg.bingo_add_node(task_check_result_A1)
    bingo_dfg.bingo_add_node(task_check_result_A2)
    # 4. Define Dependencies
    # The two copy tasks can run in parallel as they are independent
    # The check result tasks depend on their respective copy tasks
    bingo_dfg.bingo_add_edge(task_copy_A1, task_check_result_A1)
    bingo_dfg.bingo_add_edge(task_copy_A2, task_check_result_A2)

    return bingo_dfg

def main():
    args = get_args()
    output_dir = args.output_dir
    output_file_name = args.output_offload_file_name
    print(f"Output DIR: {output_dir}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Execute Pipeline
    params, merged_config = define_workload_params(args.cfg, args.hwcfg)

    # Emit the static L3 source data used by the two DMA copy tasks.
    if args.data_h is not None:
        data_h_content = emit_header_file(**merged_config)
        with open(args.data_h, "w") as f:
            f.write(data_h_content)
        print(f"Written data header: {args.data_h}")

    mem_handles = define_memory_handles(params)
    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(merged_config, platform, args.output_dir, args.output_offload_file_name):
        return
    dfg = create_dfg(params, mem_handles, platform)
    data_header = os.path.basename(args.data_h) if args.data_h is not None else "dma_data.h"
    dfg.bingo_compile_dfg(APP_NAME, output_dir, output_file_name, extra_include_header_list=[data_header])

if __name__ == "__main__":
    main()
