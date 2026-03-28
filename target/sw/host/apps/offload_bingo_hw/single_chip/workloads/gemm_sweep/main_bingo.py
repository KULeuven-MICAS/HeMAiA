# Fanchen Kong <fanchen.kong@kuleuven.be>

# This file is the main entry point for the gemm_sweep workload.
# M, N, K, array_shape are configurable from the Makefile via CLI args.
# The DFG is minimal: load A & B from L3 to L1, compute GEMM,
# then compare the D result in L1 with the golden D in L3.

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
from bingo_kernel_args import SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelGemmFullArgs, HostBingoKernelCheckResultArgs


def get_args():
    parser = argparse.ArgumentParser(description="Bingo HW Manager - GEMM Sweep")
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
        "--data_h",
        type=pathlib.Path,
        default=None,
        help="Output path for the generated data header (e.g. gemm_data.h). If omitted, data header is not written.",
    )
    # Configurable GEMM parameters (override params.hjson from Makefile)
    parser.add_argument("--M", type=int, default=None, help="GEMM M dimension (overrides params.hjson)")
    parser.add_argument("--N", type=int, default=None, help="GEMM N dimension (overrides params.hjson)")
    parser.add_argument("--K", type=int, default=None, help="GEMM K dimension (overrides params.hjson)")
    parser.add_argument("--array_shape", type=int, default=None, help="Array shape index (overrides params.hjson)")
    parser.add_argument("--l1_size_kb", type=int, default=512, help="L1 storage size in kB (default: 512)")
    return parser.parse_args()

def _o1heap_fragment_size(amount, alignment=128):
    """Model the actual o1heap allocation cost for a given request.
    Mirrors o1heap64.c: fragment_size = roundUpToPowerOf2(amount + O1HEAP_ALIGNMENT).
    """
    fragment_size_min = alignment * 2
    raw = amount + alignment
    p = 1
    while p < raw:
        p <<= 1
    return max(p, fragment_size_min)


def _o1heap_capacity(heap_size, alignment=128, num_bins=64):
    """Compute usable o1heap capacity from a raw heap region size.
    Subtracts the padded O1HeapInstance struct and aligns down.
    """
    # O1HeapInstance: bins[num_bins]*8 + nonempty_bin_mask(8) + O1HeapDiagnostics(5*8)
    instance_size = num_bins * 8 + 8 + 5 * 8
    instance_size_padded = (instance_size + alignment - 1) & ~(alignment - 1)
    fragment_size_min = alignment * 2
    cap = heap_size - instance_size_padded
    cap = cap - (cap % fragment_size_min)
    return cap


def define_workload_params(cfg_path, hwcfg_path, cli_overrides=None, l1_size_kb=512):
    """Load workload params from hjson config files.
    meshRow/tileSize/meshCol are derived from the hw config.
    CLI overrides (M, N, K, array_shape) take precedence over params.hjson.
    Returns (params dict, merged_config dict).
    """
    with open(cfg_path) as f:
        param = hjson.loads(f.read())

    # Apply CLI overrides before merging
    if cli_overrides:
        for key in ['M', 'N', 'K', 'array_shape']:
            if cli_overrides.get(key) is not None:
                param[key] = cli_overrides[key]

    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

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

    params = {
        'M': M,
        'K': K,
        'N': N,
        'addZeroC':    merged.get("addZeroC",    1),
        'addNonZeroC': merged.get("addNonZeroC", 0),
        'accumPrevC':  merged.get("accumPrevC",  0),
        'meshRow':  meshRow,
        'tileSize': tileSize,
        'meshCol':  meshCol,
        'arrayShapeIdx': array_shape,
        'transposeA': merged.get("transposed_A", 0),
        'transposeB': merged.get("transposed_B", 0),
    }
    params["app_name"] = "Single-Chip GEMM Sweep"
    # Derived sizes
    params['A_size'] = M * K * meshRow * tileSize * 1  # int8
    params['B_size'] = K * N * tileSize * meshCol * 1  # int8
    params['D_size'] = M * N * meshRow * meshCol * 4   # int32

    # L1 storage size check (accounting for o1heap buddy allocator overhead)
    # o1heap rounds each allocation to next_power_of_2(size + O1HEAP_ALIGNMENT=128)
    l1_size = l1_size_kb * 1024
    l1_capacity = _o1heap_capacity(l1_size)
    A_frag = _o1heap_fragment_size(params['A_size'])
    B_frag = _o1heap_fragment_size(params['B_size'])
    D_frag = _o1heap_fragment_size(params['D_size'])
    total_l1_frag = A_frag + B_frag + D_frag
    # Each individual allocation must fit (needs one contiguous power-of-2 block)
    for name, raw, frag in [('A', params['A_size'], A_frag), ('B', params['B_size'], B_frag), ('D', params['D_size'], D_frag)]:
        assert frag <= l1_capacity, (
            f"GEMM {name} allocation ({raw}B) requires o1heap fragment of {frag}B "
            f"(next_pow2({raw}+128)={frag}), exceeding L1 heap capacity ({l1_capacity}B from {l1_size_kb}KB TCDM). "
            f"Reduce M={M}, K={K}, or N={N}."
        )
    assert total_l1_frag <= l1_capacity, (
        f"GEMM data exceeds L1 heap! "
        f"o1heap fragments: A({A_frag}B) + B({B_frag}B) + D({D_frag}B) = {total_l1_frag}B > capacity {l1_capacity}B. "
        f"(o1heap rounds each alloc to next_pow2(size+128)). "
        f"Reduce M={M}, K={K}, or N={N}."
    )
    A_kb = params['A_size'] / 1024
    B_kb = params['B_size'] / 1024
    D_kb = params['D_size'] / 1024
    print(f"L1 usage: A({A_kb:.1f}KB) + B({B_kb:.1f}KB) + D({D_kb:.1f}KB) = {(A_frag+B_frag+D_frag)/1024:.1f}KB (o1heap) / {l1_capacity/1024:.1f}KB capacity ({100*total_l1_frag/l1_capacity:.1f}%)")

    # Pass l1_size_kb into merged config so gemm_datagen can use it
    merged["l1_size_kb"] = l1_size_kb
    return params, merged

def define_memory_handles(params):
    """Defines memory handles used in the DFG."""
    mem_handles = {}
    # L3 symbols (compile-time known locations from gemm_data.h)
    mem_handles['A_L3'] = BingoMemSymbol('A')
    mem_handles['B_L3'] = BingoMemSymbol('B')
    mem_handles['D_L3'] = BingoMemSymbol('D')
    # L1 buffers (runtime allocated)
    chip_id = 0
    cluster_id = 0
    mem_handles['l1_buf_A'] = BingoMemAlloc('l1_buf_A', size=params['A_size'], mem_level="L1", chip_id=chip_id, cluster_id=cluster_id)
    mem_handles['l1_buf_B'] = BingoMemAlloc('l1_buf_B', size=params['B_size'], mem_level="L1", chip_id=chip_id, cluster_id=cluster_id)
    mem_handles['l1_buf_D'] = BingoMemAlloc('l1_buf_D', size=params['D_size'], mem_level="L1", chip_id=chip_id, cluster_id=cluster_id)
    return mem_handles

def create_dfg(params, mem_handles):
    """Creates the minimal DFG: Load A & B from L3, compute GEMM, check D in L1 vs golden D in L3."""
    # 1. Initialize DFG
    bingo_dfg = BingoDFG(
        num_chiplets=1,
        num_clusters_per_chiplet=1,
        num_cores_per_cluster=2,
        is_host_as_acc=True,
        chiplet_ids=[0x00],
    )
    gemm_core_id = 0  # Core 0 for Compute
    dma_core_id = 1   # Core 1 for DMA
    host_core_id = 2  # Core 2 for Host

    # 2. Define Nodes
    # Load A from L3 to L1
    task_copy_A = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=dma_core_id,
        node_name="Load_A",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles['A_L3'],
            dst_addr=mem_handles['l1_buf_A'],
            size=params['A_size']
        )
    )
    # Load B from L3 to L1
    task_copy_B = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=dma_core_id,
        node_name="Load_B",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_handles['B_L3'],
            dst_addr=mem_handles['l1_buf_B'],
            size=params['B_size']
        )
    )
    # GEMM Full Compute
    task_gemm_full = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=gemm_core_id,
        node_name="Gemm_Full",
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
    # Check result: compare D in L1 with golden D in L3
    # only check the first 256 bytes for simplicity
    task_check_result = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=0,
        assigned_core_id=host_core_id,
        node_name="Check_D",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=mem_handles['D_L3'],
            output_data_addr=mem_handles['l1_buf_D'],
            data_size=256
        )
    )

    # 3. Add Nodes to DFG
    bingo_dfg.bingo_add_node(task_copy_A)
    bingo_dfg.bingo_add_node(task_copy_B)
    bingo_dfg.bingo_add_node(task_gemm_full)
    bingo_dfg.bingo_add_node(task_check_result)

    # 4. Define Edges
    # A and B must be loaded before GEMM (serial DMA: A before B)
    bingo_dfg.bingo_add_edge(task_copy_A, task_copy_B)
    bingo_dfg.bingo_add_edge(task_copy_B, task_gemm_full)
    # Check after GEMM is done
    bingo_dfg.bingo_add_edge(task_gemm_full, task_check_result)
    return bingo_dfg


def main():
    args = get_args()
    output_dir = args.output_dir
    output_file_name = args.output_offload_file_name
    print(f"Output DIR: {output_dir}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect CLI overrides for GEMM params
    cli_overrides = {
        'M': args.M,
        'N': args.N,
        'K': args.K,
        'array_shape': args.array_shape,
    }

    # Execute Pipeline
    params, merged_config = define_workload_params(args.cfg, args.hwcfg, cli_overrides, l1_size_kb=args.l1_size_kb)

    # Emit gemm_data.h
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
