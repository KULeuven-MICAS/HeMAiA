# Fanchen Kong <fanchen.kong@kuleuven.be>

# Workload: build a single-cluster GEMM DFG that keeps B in L1, streams tiled A
# through ping-pong buffers, streams D tiles back to L3, and checks the results. Tile A among M dim.

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
APP_NAME = "Single-Chip GEMM Double Buffer"

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim")

# Import emit_header_file from gemm_datagen to emit gemm_data.h directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gemm_datagen import emit_header_file  # noqa E402
from snax_utils import define_gemm_workload_params  # noqa E402

from bingo_dfg import BingoDFG
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol
from bingo_kernel_args import SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelGemmFullArgs, HostBingoKernelCheckResultArgs, HostBingoKernelIdmaArgs, SnaxBingoKernelGemmMinimalArgs


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


def define_workload_params(cfg_path, hwcfg_path):
    params, merged = define_gemm_workload_params(cfg_path, hwcfg_path)
    num_double_buffers = merged["num_double_buffers"]
    print(f"M: {params['M']}, K: {params['K']}, N: {params['N']}, meshRow: {params['meshRow']}, tileSize: {params['tileSize']}, meshCol: {params['meshCol']}")
    print(f"Calculated A size: {params['A_size']//1024} kbytes, B size: {params['B_size']//1024} kbytes, C size: {params['C_size']//1024} kbytes, D size: {params['D_size']//1024} kbytes")
    params.update(
        num_double_buffers=num_double_buffers,
        A_tile_size=params['A_size'] // num_double_buffers,
        D_tile_size=params['D_size'] // num_double_buffers,
    )
    return params, merged

# Chip 0, Cluster 0
cur_chiplet_id = 0
cur_cluster_id = 0

def define_memory_handles(params):
    """Defines memory symbols and handles."""
    mem_handles = {}

    # 1. Define Memory Symbols (Existing C variables)
    # The MemSymbol are the variables defined in the data.h file which the memory location is already known at compile time
    # The A tiles
    for i in range(params['num_double_buffers']):
        mem_handles[f'A{i}_data_L3_symbol'] = BingoMemSymbol("A",offset=i*params['A_tile_size'])
    # B is not tiled, only one symbol
    mem_handles['B_data_L3_symbol'] = BingoMemSymbol("B")
    # The D tiles
    for i in range(params['num_double_buffers']):
        mem_handles[f'D{i}_data_L3_symbol'] = BingoMemSymbol("D",offset=i*params['D_tile_size'])
    # C is not used

    # 2. Define Memory Handles (Dynamic Allocations)
    # The MemHandles are the buffers that need to be allocated at runtime by the bingo runtime
    # L3 Buffers
    for i in range(params['num_double_buffers']):
        mem_handles[f'D{i}_L3_buf'] = BingoMemAlloc(f"D{i}_L3_buf", size=params['D_tile_size'], mem_level="L3", chip_id=cur_chiplet_id)

    # L1 Buffers
    # For the double buffering, we need to calculate the number of L1 buffers needed
    # 1. For Matrix B:
    #    Since B does not change across tiles of A, we just need one buffer for B.
    #    Size = B_size
    mem_handles['l1_buf_B'] = BingoMemAlloc("l1_buf_B", size=params['B_size'], mem_level="L1", chip_id=cur_chiplet_id, cluster_id=cur_cluster_id)

    # 2. For Matrix A:
    #    We tile A into num_double_buffers tiles. We use double buffering for A to overlap loading A[i+1] with computing A[i].
    #    We need 2 buffers (Ping/Pong).
    #    Size = A_tile_size
    mem_handles['l1_buf_A_ping'] = BingoMemAlloc("l1_buf_A_ping", size=params['A_tile_size'], mem_level="L1", chip_id=cur_chiplet_id, cluster_id=cur_cluster_id)
    mem_handles['l1_buf_A_pong'] = BingoMemAlloc("l1_buf_A_pong", size=params['A_tile_size'], mem_level="L1", chip_id=cur_chiplet_id, cluster_id=cur_cluster_id)

    # 3. For Matrix D (Output):
    #    D is tiled corresponding to A's tiling.
    #    We need double buffering for D to overlap storing D[i-1] with computing D[i].
    #    Size = D_tile_size
    mem_handles['l1_buf_D_ping'] = BingoMemAlloc("l1_buf_D_ping", size=params['D_tile_size'], mem_level="L1", chip_id=cur_chiplet_id, cluster_id=cur_cluster_id)
    mem_handles['l1_buf_D_pong'] = BingoMemAlloc("l1_buf_D_pong", size=params['D_tile_size'], mem_level="L1", chip_id=cur_chiplet_id, cluster_id=cur_cluster_id)

    # Computation Flow (num_double_buffers Tiles):
    # We use separate DMA cores/channels for Load and Store to maximize overlap.
    # -------------------------------------------------------------------------------------------------------
    # Time Step | Core 1 (DMA Load)             | Core 0 (Compute)              | Core 2 (Host DMA Store)
    # -------------------------------------------------------------------------------------------------------
    # Prologue  | Load B                        |                               |
    #           | Load A0 -> A_ping             |                               |
    # -------------------------------------------------------------------------------------------------------
    # Iter 0    | Load A1 -> A_pong             | Comp D0 (A_ping, B) -> D_ping |
    # -------------------------------------------------------------------------------------------------------
    # Iter 1    | Load A2 -> A_ping             | Comp D1 (A_pong, B) -> D_pong | Store D0 <- D_ping
    # -------------------------------------------------------------------------------------------------------
    # Iter 2    | Load A3 -> A_pong             | Comp D2 (A_ping, B) -> D_ping | Store D1 <- D_pong
    # -------------------------------------------------------------------------------------------------------
    # Iter 3    |                               | Comp D3 (A_pong, B) -> D_pong | Store D2 <- D_ping
    # -------------------------------------------------------------------------------------------------------
    # Epilogue  |                               |                               | Store D3 <- D_pong
    # -------------------------------------------------------------------------------------------------------

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
    # Node: Copy B (Static B matrix, loaded once)
    task_copy_B = BingoNode(
        assigned_chiplet_id=cur_chiplet_id,
        assigned_cluster_id=cur_cluster_id,
        assigned_core_id=dma_core_id,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles['B_data_L3_symbol'],
            dst_addr=mem_handles['l1_buf_B'],
            size=params['B_size']
        )
    )

    # Note: Other nodes (Copy A, GEMM, Copy D) are defined in the loop below

    # 3. Add Nodes to DFG
    bingo_dfg.bingo_add_node(task_copy_B)

    # 4. Loop to create nodes for each tile
    task_copy_A_nodes = []
    task_gemm_nodes = []
    task_copy_D_nodes = []

    for i in range(params['num_double_buffers']):
        # Determine current buffer (Ping for even, Pong for odd)
        current_l1_A = mem_handles['l1_buf_A_ping'] if i % 2 == 0 else mem_handles['l1_buf_A_pong']
        current_l1_D = mem_handles['l1_buf_D_ping'] if i % 2 == 0 else mem_handles['l1_buf_D_pong']

        # --- Node: Copy A[i] ---
        node_copy_A = BingoNode(
            assigned_chiplet_id=cur_chiplet_id,
            assigned_cluster_id=cur_cluster_id,
            assigned_core_id=dma_core_id, # Core 1 for Load
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles[f'A{i}_data_L3_symbol'], # A0..Anum_double_buffers
                dst_addr=current_l1_A,
                size=params['A_tile_size'] # 4KB
            )
        )
        task_copy_A_nodes.append(node_copy_A)
        bingo_dfg.bingo_add_node(node_copy_A)

        # --- Node: GEMM[i] ---
        # Note: M in GEMM kernel args refers to the tile height
        M_tile = params['M'] // params['num_double_buffers']

        if i == 0:
            node_gemm = BingoNode(
                assigned_chiplet_id=cur_chiplet_id,
                assigned_cluster_id=cur_cluster_id,
                assigned_core_id=gemm_core_id, # Core 0 for Compute
                kernel_name="__snax_bingo_kernel_gemm_full",
                kernel_args=SnaxBingoKernelGemmFullArgs(
                    input_A_addr=current_l1_A,
                    input_B_addr=mem_handles['l1_buf_B'],
                    input_C_addr=0, # Not used
                    output_D_addr=current_l1_D,
                    M=M_tile,
                    K=params['K'],
                    N=params['N'],
                    array_shape_idx=params['arrayShapeIdx'],
                    transpose_A=params['transposeA'],
                    transpose_B=params['transposeB'],
                    accumPrevC=params['accumPrevC']
                )
            )
        else:
            node_gemm = BingoNode(
                assigned_chiplet_id=cur_chiplet_id,
                assigned_cluster_id=cur_cluster_id,
                assigned_core_id=gemm_core_id, # Core 0 for Compute
                kernel_name="__snax_bingo_kernel_gemm_minimal",
                kernel_args=SnaxBingoKernelGemmMinimalArgs(
                    input_A_addr=current_l1_A,
                    input_B_addr=mem_handles['l1_buf_B'],
                    input_C_addr=0, # Not used
                    output_D_addr=current_l1_D,
                )
            )

        task_gemm_nodes.append(node_gemm)
        bingo_dfg.bingo_add_node(node_gemm)

        # --- Node: Copy D[i] ---
        node_copy_D = BingoNode(
            assigned_chiplet_id=cur_chiplet_id,
            assigned_cluster_id=cur_cluster_id,
            assigned_core_id=host_core_id, # Core 2 for Store
            kernel_name="__host_bingo_kernel_idma",
            kernel_args=HostBingoKernelIdmaArgs(
                src_addr=current_l1_D,
                dst_addr=mem_handles[f'D{i}_L3_buf'], # D0..D3
                size=params['D_tile_size']
            )
        )
        task_copy_D_nodes.append(node_copy_D)
        bingo_dfg.bingo_add_node(node_copy_D)

    # Node: Check Result (Host Side)
    # only check the first 64B of each D tile to save time
    # should be sufficient to verify correctness
    task_check_result_nodes = []
    for i in range(params['num_double_buffers']):
        task_check_result = BingoNode(
            assigned_chiplet_id=cur_chiplet_id,
            assigned_cluster_id=cur_cluster_id,
            assigned_core_id=host_core_id, # Core 2 for Host
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_handles[f'D{i}_data_L3_symbol'],
                output_data_addr=mem_handles[f'D{i}_L3_buf'],
                data_size=64
            )
        )
        task_check_result_nodes.append(task_check_result)
        bingo_dfg.bingo_add_node(task_check_result)

    # 4. Define dependencies
    # Note: all the tasks in one execution core queue (compute core, dma core, host core) must have an order specified by the user, otherwise the execution order is not guaranteed, which may cause deadlock.

    # --- A. Data Validity (Producer -> Consumer) ---
    # These dependencies enforce the fundamental data flow of the application.

    # A task consuming data must wait for the task producing it to finish.
    for i in range(params['num_double_buffers']):
        # 1. Load A[i] (DMA) must complete before Compute[i] (GEMM) starts.
        #    The GEMM kernel reads matrix A from the L1 SPM buffer populated by this load.
        bingo_dfg.bingo_add_edge(task_copy_A_nodes[i], task_gemm_nodes[i])

        # 2. Compute[i] (GEMM) must complete before Store D[i] (DMA) starts.
        #    The Store task reads the result matrix D from L1 SPM to move it to L3.
        bingo_dfg.bingo_add_edge(task_gemm_nodes[i], task_copy_D_nodes[i])

    # --- B. Core Sequencing (Sequential execution on same core) ---
    # Although tasks on the same core will naturally execute sequentially if queued,
    # adding explicit edges makes the execution order deterministic and easier to visualize.

    # 1. DMA Load Queue (Core 1):
    #    First load Matrix B (static for all GEMMs), then Load A[0], then Load A[1], etc.
    bingo_dfg.bingo_add_edge(task_copy_B, task_copy_A_nodes[0])
    for i in range(params['num_double_buffers'] - 1):
        bingo_dfg.bingo_add_edge(task_copy_A_nodes[i], task_copy_A_nodes[i+1])

    # 2. Compute Queue (Core 0):
    #    Perform GEMM operations strictly in order: GEMM 0 -> GEMM 1 -> ...
    for i in range(params['num_double_buffers'] - 1):
        bingo_dfg.bingo_add_edge(task_gemm_nodes[i], task_gemm_nodes[i+1])

    # 3. DMA Store Queue (Core 2):
    #    Store results strictly in order: Store D[0] -> Store D[1] -> ...
    for i in range(params['num_double_buffers'] - 1):
        bingo_dfg.bingo_add_edge(task_copy_D_nodes[i], task_copy_D_nodes[i+1])

    # 4. Result Check Sequencing:
    #    Ensure checks run in order.
    for i in range(params['num_double_buffers'] - 1):
        bingo_dfg.bingo_add_edge(task_check_result_nodes[i], task_check_result_nodes[i+1])

    # 5. Start Verification:
    #    Start checking results only after the *last* Store D is complete.
    #    This represents a barrier: Compute All -> Store All -> Verify All.
    bingo_dfg.bingo_add_edge(task_copy_D_nodes[-1], task_check_result_nodes[0])

    # --- C. Buffer Safety (Double Buffering / Resource Management) ---
    # Due to limited L1 memory, we reuse a set of buffers (Ping/Pong scheme).
    # We must add synchronization edges to prevent data hazards (Read-after-Write, Write-after-Read).

    # Buffer Usage Map:
    # Buffer A_Ping: Used by Load A[0], Compute[0], Load A[2], Compute[2], ... (Even indices)
    # Buffer A_Pong: Used by Load A[1], Compute[1], Load A[3], Compute[3], ... (Odd indices)
    # Buffer D_Ping: Used by Compute[0], Store D[0], Compute[2], Store D[2], ... (Even indices)
    # Buffer D_Pong: Used by Compute[1], Store D[1], Compute[3], Store D[3], ... (Odd indices)

    # 1. Prevent Overwriting Input Buffers (Write-after-Read Hazard):
    #    We cannot start loading A[i+2] (overwriting Ping) until Compute[i] is finished reading Ping.
    #    Load A[i+2] depends on Compute[i].
    for i in range(params['num_double_buffers'] - 2):
        bingo_dfg.bingo_add_edge(task_gemm_nodes[i], task_copy_A_nodes[i+2])

    # 2. Prevent Overwriting Output Buffers (Write-after-Read Hazard):
    #    We cannot start computing GEMM[i+2] (writing D into Ping) until Store D[i] is finished reading Ping.
    #    Compute[i+2] depends on Store D[i].
    for i in range(params['num_double_buffers'] - 2):
        bingo_dfg.bingo_add_edge(task_copy_D_nodes[i], task_gemm_nodes[i+2])

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
