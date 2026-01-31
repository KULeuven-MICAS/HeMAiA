import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")

# The stacked GeMM workload does the following things:
# 1. D1 =  A1 X B1 (A1 B1 are int8, D1 is int32)
# 2. D2 =  D1 X B2 (D1 will be treated as int8 input, B2 is int8, D2 is int32)

from bingo_dfg import BingoDFG
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol, BingoMemFixedAddr
from bingo_kernel_args import (
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelCheckResultArgs,
)
from bingo_helpers import chiplet_addr_transform_loc


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
    return parser.parse_args()
def define_workload_params():
    """Defines the workload parameters."""
    # Stacked GEMM workload parameters
    params = {
        "M1": 2,
        "K1": 4,
        "N1": 2,
        "M2": 2,
        "K2": 16,
        "N2": 4,
        "meshRow": 1,
        "tileSize": 8,
        "meshCol": 64,
        "arrayShapeIdx": 1,
        "transposeA": 0,
        "transposeB": 0,
        "accumPrevC": 0,
    }
    params["app_name"] = "Single-Chip GEMM Stacked"
    # Derived sizes
    params["A1_size"] = (
        params["M1"] * params["K1"] * params["meshRow"] * params["tileSize"] * 1
    )  # int8
    params["B1_size"] = (
        params["K1"] * params["N1"] * params["meshCol"] * params["tileSize"] * 1
    )  # int8
    params["D1_size"] = (
        params["M1"] * params["N1"] * params["meshRow"] * params["meshCol"] * 4
    )  # int32
    params["B2_size"] = (
        params["K2"] * params["N2"] * params["meshCol"] * params["tileSize"] * 1
    )  # int8
    params["D2_size"] = (
        params["M2"] * params["N2"] * params["meshRow"] * params["meshCol"] * 4
    )  # int32
    return params

def define_memory_handles(params):
    """Defines memory symbols and handles."""
    mem_handles = {}
    # 1. Define Memory Fixed Addresses

    # 2. Define Memory Symbols
    # The MemSymbol are the variables defined in the data.h file which the memory location is already known at compile time
    mem_handles["A1_symbol_l3"] = BingoMemSymbol("A1", offset=0)
    mem_handles["B1_symbol_l3"] = BingoMemSymbol("B1", offset=0)
    mem_handles["B2_symbol_l3"] = BingoMemSymbol("B2", offset=0)
    mem_handles["D2_symbol_l3"] = BingoMemSymbol("D2", offset=0)

    # 3. Define Memory Handles (Dynamic Allocations)
    # Prepare L1 buffers for A1, B1, D1, B2, D2 and L3 for D2
    mem_handles["A1_l1"] = BingoMemAlloc(
        "A1_l1",
        size=params["A1_size"],
        mem_level="L1",
        chip_id=0x00,
        cluster_id=0,
    )
    mem_handles["B1_l1"] = BingoMemAlloc(
        "B1_l1", size=params["B1_size"], mem_level="L1", chip_id=0x00, cluster_id=0
    )
    mem_handles["D1_l1"] = BingoMemAlloc(
        "D1_l1", size=params["D1_size"], mem_level="L1", chip_id=0x00, cluster_id=0
    )
    mem_handles["B2_l1"] = BingoMemAlloc(
        "B2_l1", size=params["B2_size"], mem_level="L1", chip_id=0x00, cluster_id=0
    )
    mem_handles["D2_l1"] = BingoMemAlloc(
        "D2_l1", size=params["D2_size"], mem_level="L1", chip_id=0x00, cluster_id=0
    )
    mem_handles["D2_result_l3"] = BingoMemAlloc(
        "D2_l3", size=params["D2_size"], mem_level="L3", chip_id=0x00
    )

    return mem_handles

def create_dfg(params, mem_handles):
    """Creates the Bingo Data Flow Graph with nodes and dependencies."""
    # 1. Initialize DFG
    num_chiplets = 1
    num_clusters_per_chiplet = 1
    num_cores_per_cluster = 2
    is_host_as_acc = True
    chiplet_ids = [0x00]
    bingo_dfg = BingoDFG(
        num_chiplets,
        num_clusters_per_chiplet,
        num_cores_per_cluster,
        is_host_as_acc,
        chiplet_ids,
    )
    gemm_core_id = 0  # dev core for GEMM
    dma_core_id = 1   # dev core for IDMA
    host_core_id = 2  # host core
    # 2. Define Nodes
    # Load A1 using host IDMA
    node_load_A1 = BingoNode(
        assigned_chiplet_id=0x00,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,  # host IDMA
        node_name="Load_A1",
        kernel_name="__host_bingo_kernel_idma",
        kernel_args=HostBingoKernelIdmaArgs(
            src_addr=mem_handles["A1_symbol_l3"],
            dst_addr=mem_handles["A1_l1"],
            size=params["A1_size"],
        ),
    )
    # Load B1 using dev IDMA
    node_load_B1 = BingoNode(
        assigned_chiplet_id=0x00,
        assigned_cluster_id=0,
        assigned_core_id=dma_core_id,  # dev IDMA
        node_name="Load_B1",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["B1_symbol_l3"],
            dst_addr=mem_handles["B1_l1"],
            size=params["B1_size"],
        ),
    )
    # Load B2 using dev IDMA
    node_load_B2 = BingoNode(
        assigned_chiplet_id=0x00,
        assigned_cluster_id=0,
        assigned_core_id=dma_core_id,  # dev IDMA
        node_name="Load_B2",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["B2_symbol_l3"],
            dst_addr=mem_handles["B2_l1"],
            size=params["B2_size"],
        ),
    )
    # Compute GEMM D1 using dev
    node_gemm_D1 = BingoNode(
        assigned_chiplet_id=0x00,
        assigned_cluster_id=0,
        assigned_core_id=gemm_core_id,  # dev
        node_name="Gemm_D1",
        kernel_name="__snax_bingo_kernel_gemm_full",
        kernel_args=SnaxBingoKernelGemmFullArgs(
            input_A_addr=mem_handles["A1_l1"],
            input_B_addr=mem_handles["B1_l1"],
            input_C_addr=0,
            output_D_addr=mem_handles["D1_l1"],
            M=params["M1"],
            K=params["K1"],
            N=params["N1"],
            array_shape_idx=params["arrayShapeIdx"],
            transpose_A=params["transposeA"],
            transpose_B=params["transposeB"],
            accumPrevC=params["accumPrevC"],
        ),
    )
    # Compute GEMM D2 using dev
    node_gemm_D2 = BingoNode(
        assigned_chiplet_id=0x00,
        assigned_cluster_id=0,
        assigned_core_id=gemm_core_id,  # dev
        node_name="Gemm_D2",
        kernel_name="__snax_bingo_kernel_gemm_full",
        kernel_args=SnaxBingoKernelGemmFullArgs(
            input_A_addr=mem_handles["D1_l1"],  # D1 as input A for second GEMM
            input_B_addr=mem_handles["B2_l1"],
            input_C_addr=0,
            output_D_addr=mem_handles["D2_l1"],
            M=params["M2"],
            K=params["K2"],
            N=params["N2"],
            array_shape_idx=params["arrayShapeIdx"],
            transpose_A=params["transposeA"],
            transpose_B=params["transposeB"],
            accumPrevC=params["accumPrevC"],
        ),
    )
    # Store D2 using dev IDMA
    node_store_D2 = BingoNode(
        assigned_chiplet_id=0x00,
        assigned_cluster_id=0,
        assigned_core_id=dma_core_id,  # dev IDMA
        node_name="Store_D2",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles["D2_l1"],
            dst_addr=mem_handles["D2_result_l3"],
            size=params["D2_size"],
        ),
    )
    # Check D2 using Host
    node_check_D2 = BingoNode(
        assigned_chiplet_id=0x00,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,  # host
        node_name="Check_D2",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=mem_handles["D2_symbol_l3"],
            output_data_addr=mem_handles["D2_result_l3"],
            data_size=params["D2_size"],
        ),
    )
    # 3. Add Nodes to DFG
    bingo_dfg.bingo_add_node(node_load_A1)
    bingo_dfg.bingo_add_node(node_load_B1)
    bingo_dfg.bingo_add_node(node_load_B2)
    bingo_dfg.bingo_add_node(node_gemm_D1)
    bingo_dfg.bingo_add_node(node_gemm_D2)
    bingo_dfg.bingo_add_node(node_store_D2)
    bingo_dfg.bingo_add_node(node_check_D2)

    # 4. Define Dependencies
    bingo_dfg.add_edge(node_load_A1, node_gemm_D1)
    bingo_dfg.add_edge(node_load_B1, node_gemm_D1)
    bingo_dfg.add_edge(node_gemm_D1, node_gemm_D2)
    bingo_dfg.add_edge(node_load_B2, node_gemm_D2)
    bingo_dfg.add_edge(node_gemm_D2, node_store_D2)
    bingo_dfg.add_edge(node_store_D2, node_check_D2)
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
    params = define_workload_params()
    mem_handles = define_memory_handles(params)
    dfg = create_dfg(params, mem_handles)
    dfg.bingo_compile_dfg(params["app_name"], output_dir, output_file_name, extra_include_header_list=["gemm_data.h"])

if __name__ == "__main__":
    main()