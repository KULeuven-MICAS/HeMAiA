# Fanchen Kong <fanchen.kong@kuleuven.be>
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

# This file is the main entry point for the bingo offload application
# Users will create the dfg in this file
# And then the mini-compiler will emit the WORKLOAD.h file

# Conduct D=A*B in a single cluster with separate load and compute
# issue the load commands for A and B in parallel using the host DMA and device DMA respectively, 
# to show the improvement of the bingo hw manager in scheduling and issuing commands to different engines in parallel

import os
import sys
import argparse
import pathlib

current_dir = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR = os.path.dirname(current_dir)
sys.path.append(WORKLOADS_DIR)
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)
APP_NAME = "Single-Chip GEMM Seperate Load and Compute Parallel"

cur_chiplet_id = 0x00
cur_cluster_id = 0

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim")

# Import emit_matmul_data from gemm_datagen to derive hardware-specific params
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gemm_datagen import emit_header_file  # noqa E402
from gemm_sim_utils import define_gemm_workload_params  # noqa E402

from bingo_dfg import BingoDFG
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol
from bingo_kernel_args import SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelGemmFullArgs, HostBingoKernelCheckResultArgs, HostBingoKernelIdmaArgs


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

define_workload_params = define_gemm_workload_params

def define_memory_handles(params):
    """Defines memory handles used in the DFG."""
    # Here we only have A, B, D in L3
    mem_handles = {}

    mem_handles['A_L3'] = BingoMemSymbol('A')
    mem_handles['B_L3'] = BingoMemSymbol('B')
    mem_handles['D_L3'] = BingoMemSymbol('D')

    # L1 buffers
    mem_handles['l1_buf_A'] = BingoMemAlloc('l1_buf_A',size=params['A_size'], mem_level="L1", chip_id=cur_chiplet_id, cluster_id=cur_cluster_id)
    mem_handles['l1_buf_B'] = BingoMemAlloc('l1_buf_B',size=params['B_size'], mem_level="L1", chip_id=cur_chiplet_id, cluster_id=cur_cluster_id)
    mem_handles['l1_buf_D'] = BingoMemAlloc('l1_buf_D',size=params['D_size'], mem_level="L1", chip_id=cur_chiplet_id, cluster_id=cur_cluster_id)
    # L3 buffers
    mem_handles['l3_buf_D'] = BingoMemAlloc('l3_buf_D',size=params['D_size'], mem_level="L3", chip_id=cur_chiplet_id)
    return mem_handles

def create_dfg(params, mem_handles, platform):
    # 1. Initialize DFG using HW params derived from occamy.h + RTL config
    bingo_dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    gemm_core_id = 0  # Core 0 for Compute
    dma_core_id = 1  # Core 1 for Load
    host_core_id = 2  # Core 2 for Host DMA Store

    # 2. Define Nodes
    # Dev IDMA1D Copy A
    task_copy_A = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=dma_core_id,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles['A_L3'],
            dst_addr=mem_handles['l1_buf_A'],
            size=params['A_size']
        )
    )
    # Host IDMA1D Copy B
    task_copy_B = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=host_core_id,
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=mem_handles['B_L3'],
            dst_addr=mem_handles['l1_buf_B'],
            size=params['B_size']
        )
    )
    # Gemm Full Compute
    task_gemm_full = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=gemm_core_id,
        kernel_name="__snax_bingo_kernel_gemm_full",
        kernel_args=SnaxBingoKernelGemmFullArgs(
            input_A_addr=mem_handles['l1_buf_A'],
            input_B_addr=mem_handles['l1_buf_B'],
            input_C_addr=0,
            output_D_addr=mem_handles['l1_buf_D'],
            M=params['M'],
            K=params['K'],
            N=params['N'],
            array_shape_idx=params['arrayShapeIdx'],
            transpose_A=params['transposeA'],
            transpose_B=params['transposeB'],
            accumPrevC=params['accumPrevC']
        )
    )
    # Dev IDMA1D Copy D
    task_copy_D = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=dma_core_id,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles['l1_buf_D'],
            dst_addr=mem_handles['l3_buf_D'],
            size=params['D_size']
        )
    )
    # Host check result
    task_check_result = BingoNode(
            assigned_chiplet_id=cur_chiplet_id,
            assigned_cluster_id=cur_cluster_id,
            assigned_core_id=host_core_id,
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles['D_L3'], 
                output_data_addr=mem_handles['l3_buf_D'],         
                data_size=64                        
            )
    )
    # Check Result
    # 3. Add Nodes to DFG
    bingo_dfg.bingo_add_node(task_copy_A)
    bingo_dfg.bingo_add_node(task_copy_B)
    bingo_dfg.bingo_add_node(task_gemm_full)
    bingo_dfg.bingo_add_node(task_copy_D)
    bingo_dfg.bingo_add_node(task_check_result)
    # 4. Define Edges
    # since we are doing this in parallel with 2 dma engines, we don't need to add edges between copy A and copy B
    bingo_dfg.bingo_add_edge(task_copy_A, task_gemm_full)
    bingo_dfg.bingo_add_edge(task_copy_B, task_gemm_full)
    bingo_dfg.bingo_add_edge(task_gemm_full, task_copy_D)
    bingo_dfg.bingo_add_edge(task_copy_D, task_check_result)
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

    # Emit gemm_data.h (same as running gemm_datagen.py separately)
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
    dfg.bingo_compile_dfg(APP_NAME, output_dir, output_file_name, extra_include_header_list=["gemm_data.h"])



if __name__ == "__main__":
    main()
