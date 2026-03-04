# Fanchen Kong <fanchen.kong@kuleuven.be>

# This file is the main entry point for the bingo offload application
# Users will create the dfg in this file
# And then the mini-compiler will emit the WORKLOAD.h file
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
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol
from bingo_kernel_args import SnaxBingoKernelDummyArgs, HostBingoKernelDummyArgs

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
    """Defines the GeMM workload parameters."""
    num_double_buffers = 8
    params = {}
    params["num_double_buffers"] = num_double_buffers
    params["dummy_input"] = 1
    params["app_name"] = "DUMMY CI MIMIC GeMM Double Buffer"
    return params

def define_memory_handles(params):
    """Defines memory symbols and handles."""
    mem_handles = {}
    
    return mem_handles

def create_dfg(params, mem_handles):
    """Creates the Bingo Data Flow Graph with nodes and dependencies."""
    
    # 1. Initialize DFG
    num_chiplets = 1
    num_clusters_per_chiplet = 4
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
    gemm_core_id = 0  # Core 0 for Compute
    dma_core_id = 1  # Core 1 for Load
    host_core_id = 2  # Core 2 for Host DMA Store
    # 2. Define Nodes
    
    # Node: Copy B (Static B matrix, loaded once)
    task_copy_B = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=dma_core_id,
        kernel_name="__snax_bingo_kernel_dummy",
        kernel_args=SnaxBingoKernelDummyArgs(
            dummy_input = params["dummy_input"]
        )
    )
    params["dummy_input"] +=1
    # Note: Other nodes (Copy A, GEMM, Copy D) are defined in the loop below

    # 3. Add Nodes to DFG
    bingo_dfg.bingo_add_node(task_copy_B)

    # 4. Loop to create nodes for each tile
    task_copy_A_nodes = []
    task_gemm_nodes = []
    task_copy_D_nodes = []
    
    for i in range(params['num_double_buffers']):

        # --- Node: Copy A[i] ---
        node_copy_A = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=dma_core_id,
            kernel_name="__snax_bingo_kernel_dummy",
            kernel_args=SnaxBingoKernelDummyArgs(
                dummy_input = params["dummy_input"]
            )
        )
        params["dummy_input"] +=1
        task_copy_A_nodes.append(node_copy_A)
        bingo_dfg.bingo_add_node(node_copy_A)
        
        # --- Node: GEMM[i] ---
        
        node_gemm = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=gemm_core_id, # Core 0 for Compute
            kernel_name="__snax_bingo_kernel_dummy",
            kernel_args=SnaxBingoKernelDummyArgs(
                dummy_input = params["dummy_input"]
            )
        )
        params["dummy_input"] +=1
        task_gemm_nodes.append(node_gemm)
        bingo_dfg.bingo_add_node(node_gemm)
        
        # --- Node: Copy D[i] ---
        node_copy_D = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id, # Core 2 for Store
            kernel_name="__host_bingo_kernel_dummy",
            kernel_args=HostBingoKernelDummyArgs(
                dummy_input = params["dummy_input"]
            )
        )
        params["dummy_input"] +=1
        task_copy_D_nodes.append(node_copy_D)
        bingo_dfg.bingo_add_node(node_copy_D)

    # Node: Check Result (Host Side)
    task_check_result_nodes = []
    for i in range(params['num_double_buffers']):
        task_check_result = BingoNode(
            assigned_chiplet_id=0,
            assigned_cluster_id=0,
            assigned_core_id=host_core_id, # Core 2 for Host
            kernel_name="__host_bingo_kernel_dummy",
            kernel_args=HostBingoKernelDummyArgs(
                dummy_input = params["dummy_input"]
            )
        )
        params["dummy_input"] +=1
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
    params = define_workload_params()
    mem_handles = define_memory_handles(params)
    dfg = create_dfg(params, mem_handles)
    dfg.bingo_compile_dfg(params["app_name"], output_dir, output_file_name, extra_include_header_list=[])

if __name__ == "__main__":
    main()
