import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")

from bingo_dfg import BingoDFG
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol, BingoMemFixedAddr
from bingo_kernel_args import (
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelIdmaArgs
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
    # GeMM workload parameters
    params = {
        "M": 2,
        "K": 4,
        "N": 1,
        "meshRow": 1,
        "tileSize": 8,
        "meshCol": 64,
        "arrayShapeIdx": 1,
        "transposeA": 0,
        "transposeB": 0,
        "accumPrevC": 0,
    }
    params["app_name"] = "Multi-Chip GEMM Seperate Load and Compute Parallel"
    # Derived sizes
    params["Atile_size"] = (
        params["M"] * params["K"] * params["meshRow"] * params["tileSize"] * 1
    )  # int8
    params["B_size"] = (
        params["K"] * params["N"] * params["meshCol"] * params["tileSize"] * 1
    )  # int8
    params["C_size"] = (
        params["M"] * params["N"] * params["meshRow"] * params["meshCol"] * 4
    )  # int32
    params["D_size"] = (
        params["M"] * params["N"] * params["meshRow"] * params["meshCol"] * 4
    )  # int32
    
    # The hardcoded MemPool address for A1, A2, A3, A4, B
    main_mem_base_addr = 0x8000_0000
    mem_pool_chip_loc = {"x": 2, "y": 0}  # Chiplet at (2,0)
    mem_pool_base_addr = chiplet_addr_transform_loc(
        mem_pool_chip_loc["x"], mem_pool_chip_loc["y"], main_mem_base_addr
    )
    params["A1_mp_base_addr"] = mem_pool_base_addr + 0 * params["Atile_size"]
    params["A2_mp_base_addr"] = mem_pool_base_addr + 1 * params["Atile_size"]
    params["A3_mp_base_addr"] = mem_pool_base_addr + 2 * params["Atile_size"]
    params["A4_mp_base_addr"] = mem_pool_base_addr + 3 * params["Atile_size"]
    params["B_mp_base_addr"]  = mem_pool_base_addr + 4 * params["Atile_size"]
    return params



def define_memory_handles(params):
    """Defines memory symbols and handles."""
    mem_handles = {}
    # 1. Define Memory Fixed Addresses
    # The MemFixedAddr are the hardcoded MemPool addresses for A1, A2, A3, A4, B
    
    mem_handles["A1_mp"] = BingoMemFixedAddr(params["A1_mp_base_addr"])
    mem_handles["A2_mp"] = BingoMemFixedAddr(params["A2_mp_base_addr"])
    mem_handles["A3_mp"] = BingoMemFixedAddr(params["A3_mp_base_addr"])
    mem_handles["A4_mp"] = BingoMemFixedAddr(params["A4_mp_base_addr"])
    mem_handles["B_mp"] = BingoMemFixedAddr(params["B_mp_base_addr"])
    
    # 2. Define Memory Symbols
    # The MemSymbol are the variables defined in the data.h file which the memory location is already known at compile time
    mem_handles["A1_golden_l3"] = BingoMemSymbol("A_data_L3", offset=0)
    mem_handles["A2_golden_l3"] = BingoMemSymbol("A_data_L3", offset=1*params["Atile_size"])
    mem_handles["A3_golden_l3"] = BingoMemSymbol("A_data_L3", offset=2*params["Atile_size"])
    mem_handles["A4_golden_l3"] = BingoMemSymbol("A_data_L3", offset=3*params["Atile_size"])
    mem_handles["B_golden_l3"] = BingoMemSymbol("B_data_L3")

    mem_handles["D1_golden_l3"] = BingoMemSymbol("D_data_L3", offset=0)
    mem_handles["D2_golden_l3"] = BingoMemSymbol("D_data_L3", offset=1*params["D_size"])
    mem_handles["D3_golden_l3"] = BingoMemSymbol("D_data_L3", offset=2*params["D_size"])
    mem_handles["D4_golden_l3"] = BingoMemSymbol("D_data_L3", offset=3*params["D_size"])

    # 3. Define Memory Handles (Dynamic Allocations)
    # Prepare all the A, B, D L1 buffers and final D L3 buffer for all chpiplets
    mem_handles["A_l1_chip00"] = BingoMemAlloc(
        "A_l1_chip00",
        size=params["Atile_size"],
        mem_level="L1",
        chip_id=0x00,
        cluster_id=0,
    )
    mem_handles["B_l1_chip00"] = BingoMemAlloc(
        "B_l1_chip00", size=params["B_size"], mem_level="L1", chip_id=0x00, cluster_id=0
    )
    mem_handles["D_l1_chip00"] = BingoMemAlloc(
        "D_l1_chip00", size=params["D_size"], mem_level="L1", chip_id=0x00, cluster_id=0
    )
    mem_handles["D_l3_chip00"] = BingoMemAlloc(
        "D_l3_chip00", size=params["D_size"], mem_level="L3", chip_id=0x00
    )

    mem_handles["A_l1_chip01"] = BingoMemAlloc(
        "A_l1_chip01",
        size=params["Atile_size"],
        mem_level="L1",
        chip_id=0x01,
        cluster_id=0,
    )
    mem_handles["B_l1_chip01"] = BingoMemAlloc(
        "B_l1_chip01", size=params["B_size"], mem_level="L1", chip_id=0x01, cluster_id=0
    )
    mem_handles["D_l1_chip01"] = BingoMemAlloc(
        "D_l1_chip01", size=params["D_size"], mem_level="L1", chip_id=0x01, cluster_id=0
    )
    mem_handles["D_l3_chip01"] = BingoMemAlloc(
        "D_l3_chip01", size=params["D_size"], mem_level="L3", chip_id=0x01
    )

    mem_handles["A_l1_chip10"] = BingoMemAlloc(
        "A_l1_chip10",
        size=params["Atile_size"],
        mem_level="L1",
        chip_id=0x10,
        cluster_id=0,
    )
    mem_handles["B_l1_chip10"] = BingoMemAlloc(
        "B_l1_chip10", size=params["B_size"], mem_level="L1", chip_id=0x10, cluster_id=0
    )
    mem_handles["D_l1_chip10"] = BingoMemAlloc(
        "D_l1_chip10", size=params["D_size"], mem_level="L1", chip_id=0x10, cluster_id=0
    )
    mem_handles["D_l3_chip10"] = BingoMemAlloc(
        "D_l3_chip10", size=params["D_size"], mem_level="L3", chip_id=0x10
    )

    mem_handles["A_l1_chip11"] = BingoMemAlloc(
        "A_l1_chip11",
        size=params["Atile_size"],
        mem_level="L1",
        chip_id=0x11,
        cluster_id=0,
    )
    mem_handles["B_l1_chip11"] = BingoMemAlloc(
        "B_l1_chip11", size=params["B_size"], mem_level="L1", chip_id=0x11, cluster_id=0
    )
    mem_handles["D_l1_chip11"] = BingoMemAlloc(
        "D_l1_chip11", size=params["D_size"], mem_level="L1", chip_id=0x11, cluster_id=0
    )
    mem_handles["D_l3_chip11"] = BingoMemAlloc(
        "D_l3_chip11", size=params["D_size"], mem_level="L3", chip_id=0x11
    )

    return mem_handles

def create_dfg(params, mem_handles):
    """Creates the Bingo Data Flow Graph with nodes and dependencies."""
    # 1. Initialize DFG
    num_chiplets = 4
    num_clusters_per_chiplet = 1
    num_cores_per_cluster = 2
    is_host_as_acc = True
    chiplet_ids = [0x00, 0x01, 0x10, 0x11]
    bingo_dfg = BingoDFG(
        num_chiplets,
        num_clusters_per_chiplet,
        num_cores_per_cluster,
        is_host_as_acc,
        chiplet_ids,
    )
    gemm_core_id = 0  # Core 0 for Compute
    dma_core_id = 1  # Core 1 for Load
    host_core_id = 2  # Core 2 for Host DMA Store
    # 2. Define Nodes
    chiplets = [0x00, 0x01, 0x10, 0x11]
    a_indices = {0x00: 1, 0x01: 2, 0x10: 3, 0x11: 4}
    d_indices = {0x00: 1, 0x01: 2, 0x10: 3, 0x11: 4}

    for chiplet in chiplets:
        a_idx = a_indices[chiplet]
        d_idx = d_indices[chiplet]
        chiplet_hex = f"{chiplet:02x}"

        # Load Ai using dev IDMA
        load_A = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,  # dev IDMA
            node_name=f"Load_A{a_idx}_Chip{chiplet_hex}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles[f"A{a_idx}_mp"],
                dst_addr=mem_handles[f"A_l1_chip{chiplet_hex}"],
                size=params["Atile_size"],
            ),
        )
        # Load B using host IDMA
        load_B = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,  # host
            node_name=f"Load_B_Chip{chiplet_hex}",
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=mem_handles["B_mp"],
                dst_addr=mem_handles[f"B_l1_chip{chiplet_hex}"],
                size=params["B_size"],
            ),
        )
        # Compute GeMM using dev
        gemm = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=gemm_core_id,  # dev
            node_name=f"Gemm_A{a_idx}_B_Chip{chiplet_hex}",
            kernel_name="__snax_bingo_kernel_gemm_full",
            kernel_args=SnaxBingoKernelGemmFullArgs(
                input_A_addr=mem_handles[f"A_l1_chip{chiplet_hex}"],
                input_B_addr=mem_handles[f"B_l1_chip{chiplet_hex}"],
                input_C_addr=0,
                output_D_addr=mem_handles[f"D_l1_chip{chiplet_hex}"],
                M=params["M"],
                K=params["K"],
                N=params["N"],
                array_shape_idx=params["arrayShapeIdx"],
                transpose_A=params["transposeA"],
                transpose_B=params["transposeB"],
                accumPrevC=params["accumPrevC"],
            ),
        )
        # Store Di using dev IDMA
        store_D = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,  # dev IDMA
            node_name=f"Store_D{d_idx}_Chip{chiplet_hex}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles[f"D_l1_chip{chiplet_hex}"],
                dst_addr=mem_handles[f"D_l3_chip{chiplet_hex}"],
                size=params["D_size"],
            ),
        )
        # Check Di using Host
        check_D = BingoNode(
            assigned_chiplet_id=chiplet,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id,  # host
            node_name=f"Check_D{d_idx}_Chip{chiplet_hex}",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles[f"D{d_idx}_golden_l3"],
                output_data_addr=mem_handles[f"D_l3_chip{chiplet_hex}"],
                data_size=params["D_size"],
            ),
        )

        # Add nodes to DFG
        bingo_dfg.bingo_add_node(load_A)
        bingo_dfg.bingo_add_node(load_B)
        bingo_dfg.bingo_add_node(gemm)
        bingo_dfg.bingo_add_node(store_D)
        bingo_dfg.bingo_add_node(check_D)

        # Add edges
        bingo_dfg.add_edge(load_A, gemm)
        bingo_dfg.add_edge(load_B, gemm)
        bingo_dfg.add_edge(gemm, store_D)
        bingo_dfg.add_edge(store_D, check_D)
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
