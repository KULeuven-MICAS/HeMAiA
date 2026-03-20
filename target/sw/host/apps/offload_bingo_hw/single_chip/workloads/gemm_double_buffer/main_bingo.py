# Fanchen Kong <fanchen.kong@kuleuven.be>

# This file is the main entry point for the bingo offload application
# Users will create the dfg in this file
# And then the mini-compiler will emit the WORKLOAD.h file
import os
import sys
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

print(f"ROOT_DIR: {ROOT_DIR}")
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")

# Import emit_header_file from gemm_datagen to emit gemm_data.h directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gemm_datagen import emit_header_file  # noqa E402

from bingo_dfg import BingoDFG
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
        "--emit_mini_golden",
        action="store_true",
        help="Emit mini golden data for verification",
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
        "--data_h",
        type=pathlib.Path,
        default=None,
        help="Output path for the generated data header (e.g. gemm_data.h). If omitted, data header is not written.",
    )
    parser.add_argument(
        "--num_double_buffers",
        type=int,
        default=8,
        help="Number of double buffers (default: 8)",
    )
    return parser.parse_args()
def define_workload_params(cfg_path, hwcfg_path, emit_mini_golden=False, num_double_buffers=8):
    """Load workload params from hjson config files.
    M, K, N and array_shape come from params.hjson;
    meshRow/tileSize/meshCol are derived from the hw config.
    Returns (params dict, merged_config dict).
    """
    with open(cfg_path) as f:
        param = hjson.loads(f.read())
    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw,
              "emit_mini_golden": emit_mini_golden,
              "num_double_buffers": num_double_buffers}

    # Derive meshRow/tileSize/meshCol from the hw config
    data_type = 0  # int8
    array_shape = merged["array_shape"]
    snax_acc_cfg = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]
    meshRow  = unrolling[0]
    tileSize = unrolling[1]
    meshCol  = unrolling[2]

    M = merged["M"]
    K = merged["K"]
    N = merged["N"]
    print(f"Loaded M: {M}, K: {K}, N: {N} from config (meshRow={meshRow}, tileSize={tileSize}, meshCol={meshCol})")

    params = {
        'M': M,
        'K': K,
        'N': N,
        'meshRow':  meshRow,
        'tileSize': tileSize,
        'meshCol':  meshCol,
        'arrayShapeIdx': array_shape,
        'transposeA': merged.get("transposed_A", 0),
        'transposeB': merged.get("transposed_B", 0),
        'accumPrevC': merged.get("accumPrevC",  0),
    }
    params["app_name"] = "Single-Chip GEMM Double Buffer"
    # Derived sizes
    params['A_size'] = M * K * meshRow * tileSize * 1  # int8
    params['B_size'] = K * N * tileSize * meshCol * 1  # int8
    params['C_size'] = M * N * meshRow * meshCol * 4   # int32
    params['D_size'] = M * N * meshRow * meshCol * 4   # int32
    print(f"Calculated A size: {params['A_size']//1024} kbytes, B size: {params['B_size']//1024} kbytes, C size: {params['C_size']//1024} kbytes, D size: {params['D_size']//1024} kbytes")

    # double buffer number
    params['num_double_buffers'] = num_double_buffers
    params['A_tile_size'] = params['A_size'] // num_double_buffers
    params['D_tile_size'] = params['D_size'] // num_double_buffers
    params['emit_mini_golden'] = emit_mini_golden
    return params, merged

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
    # If emit_mini_golden is True, the D array is compacted to only contain the verification data (64B per tile)
    golden_stride = 64 if params.get('emit_mini_golden', False) else params['D_tile_size']
    for i in range(params['num_double_buffers']):
        mem_handles[f'D{i}_data_L3_symbol'] = BingoMemSymbol("D",offset=i*golden_stride)
    # C is not used

    # 2. Define Memory Handles (Dynamic Allocations)
    # The MemHandles are the buffers that need to be allocated at runtime by the bingo runtime
    # L3 Buffers
    for i in range(params['num_double_buffers']):
        mem_handles[f'D{i}_L3_buf'] = BingoMemAlloc(f"D{i}_L3_buf", size=params['D_tile_size'], mem_level="L3")
    # L1 Buffers
    # Chip 0, Cluster 0
    chip_id = 0
    cluster_id = 0

    # For the double buffering, we need to calculate the number of L1 buffers needed
    # 1. For Matrix B:
    #    Since B does not change across tiles of A, we just need one buffer for B.
    #    Size = B_size (4KB)
    mem_handles['l1_buf_B'] = BingoMemAlloc("l1_buf_B", size=params['B_size'], mem_level="L1", chip_id=chip_id, cluster_id=cluster_id)

    # 2. For Matrix A:
    #    We tile A into 4 tiles. We use double buffering for A to overlap loading A[i+1] with computing A[i].
    #    We need 2 buffers (Ping/Pong).
    #    Size = A_tile_size
    mem_handles['l1_buf_A_ping'] = BingoMemAlloc("l1_buf_A_ping", size=params['A_tile_size'], mem_level="L1", chip_id=chip_id, cluster_id=cluster_id)
    mem_handles['l1_buf_A_pong'] = BingoMemAlloc("l1_buf_A_pong", size=params['A_tile_size'], mem_level="L1", chip_id=chip_id, cluster_id=cluster_id)

    # 3. For Matrix D (Output):
    #    D is tiled corresponding to A's tiling.
    #    We need double buffering for D to overlap storing D[i-1] with computing D[i].
    #    Size = D_tile_size
    mem_handles['l1_buf_D_ping'] = BingoMemAlloc("l1_buf_D_ping", size=params['D_tile_size'], mem_level="L1", chip_id=chip_id, cluster_id=cluster_id)
    mem_handles['l1_buf_D_pong'] = BingoMemAlloc("l1_buf_D_pong", size=params['D_tile_size'], mem_level="L1", chip_id=chip_id, cluster_id=cluster_id)

    # Computation Flow (4 Tiles):
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
    cur_chiplet_id = 0
    cur_cluster_id = 0
    gemm_core_id = 0  # Core 0 for Compute
    dma_core_id = 1  # Core 1 for Load
    host_core_id = 2  # Core 2 for Host DMA Store
    # 2. Define Nodes
    
    # Node: Copy B (Static B matrix, loaded once)
    task_copy_B = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
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
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id, # Core 1 for Load
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_handles[f'A{i}_data_L3_symbol'], # A0..A3
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
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
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
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
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
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
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
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
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
    
    # 3. Flow Control / Window Size Control:
    #    This edge limits the "lookahead" of the loader.
    #    Load A[i+3] depends on Store D[i].
    #    This ensures we don't have too many active tasks in flight, keeping the pipeline balanced.
    #    It effectively regulates the distance between the production of results and new inputs.
    for i in range(params['num_double_buffers'] - 3):
        bingo_dfg.bingo_add_edge(task_copy_D_nodes[i], task_copy_A_nodes[i+3])

    # 4. Pipeline Staggering:
    #    Store D[i] depends on Load A[i+1].
    #    This artificially delays the store of the current iteration until the load of the next iteration is launched.
    for i in range(params['num_double_buffers'] - 1):
        bingo_dfg.bingo_add_edge(task_copy_A_nodes[i+1], task_copy_D_nodes[i])
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
    params, merged_config = define_workload_params(
        args.cfg, args.hwcfg,
        emit_mini_golden=args.emit_mini_golden,
        num_double_buffers=args.num_double_buffers
    )

    # Emit gemm_data.h (same as running gemm_datagen.py separately)
    if args.data_h is not None:
        data_h_content = emit_header_file(**merged_config)
        with open(args.data_h, "w") as f:
            f.write(data_h_content)
        print(f"Written data header: {args.data_h}")

    mem_handles = define_memory_handles(params)
    dfg = create_dfg(params, mem_handles)
    dfg.bingo_compile_dfg(params["app_name"], output_dir, output_file_name, extra_include_header_list=["gemm_data.h"])

if __name__ == "__main__":
    main()
