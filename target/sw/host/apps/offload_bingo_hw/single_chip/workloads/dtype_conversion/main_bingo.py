# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Data type conversion round-trip test:
#   FP32 -> quantize INT8 -> GEMM (INT8xINT8->INT32) -> dequantize FP32
#
# Tests the full mixed-precision data path needed for TinyLlama inference:
#   CVA6 FP32 ops -> quantize -> VersaCore GEMM -> dequantize -> CVA6 FP32 ops

import os
import sys
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dtype_conv_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelFp32QuantizeArgs,
    HostBingoKernelInt32DequantizeArgs,
)


def get_args():
    parser = argparse.ArgumentParser(description="Dtype conversion round-trip test")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def define_workload_params(cfg_path, hwcfg_path):
    with open(cfg_path) as f:
        param = hjson.loads(f.read())
    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

    data_type = 0  # int8
    array_shape = merged["array_shape"]
    snax_acc_cfg = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]
    meshRow = unrolling[0]
    tileSize = unrolling[1]
    meshCol = unrolling[2]

    M = merged["M"]
    K = merged["K"]
    N = merged["N"]

    num_A = M * K * meshRow * tileSize
    num_B = K * N * tileSize * meshCol
    num_D = M * N * meshRow * meshCol

    # Padded A size for xDMA (multiple of 64)
    A_int8_padded = num_A + ((-num_A) % 64)

    params = {
        'M': M, 'K': K, 'N': N,
        'meshRow': meshRow, 'tileSize': tileSize, 'meshCol': meshCol,
        'arrayShapeIdx': array_shape,
        'num_A': num_A,
        'num_B': num_B,
        'num_D': num_D,
        'A_fp32_size': num_A * 4,
        'B_fp32_size': num_B * 4,
        'A_int8_size': A_int8_padded,
        'B_int8_size': num_B,
        'D_int32_size': num_D * 4,
        'D_fp32_size': num_D * 4,
        'scale_size': 4,  # sizeof(float)
        'app_name': 'Single-Chip Dtype Conversion Round-Trip',
    }
    return params, merged


def define_memory_handles(params):
    mem = {}
    chip_id = 0
    cluster_id = 0

    # L3 symbols (from data header)
    mem['fp32_A_L3'] = BingoMemSymbol('fp32_input_A')
    mem['fp32_B_L3'] = BingoMemSymbol('fp32_input_B')
    mem['golden_int8_A'] = BingoMemSymbol('golden_int8_A')
    mem['golden_int8_B'] = BingoMemSymbol('golden_int8_B')
    mem['golden_int32_D'] = BingoMemSymbol('golden_int32_D')
    mem['golden_fp32_D'] = BingoMemSymbol('golden_fp32_D')
    mem['combined_scale'] = BingoMemSymbol('combined_scale')

    # L3 buffers for quantized outputs and intermediate results
    mem['int8_A_buf'] = BingoMemAlloc('int8_A_buf', size=params['A_int8_size'], mem_level="L3")
    mem['int8_B_buf'] = BingoMemAlloc('int8_B_buf', size=params['B_int8_size'], mem_level="L3")
    mem['scale_A_buf'] = BingoMemAlloc('scale_A_buf', size=params['scale_size'], mem_level="L3")
    mem['scale_B_buf'] = BingoMemAlloc('scale_B_buf', size=params['scale_size'], mem_level="L3")
    mem['int32_D_buf'] = BingoMemAlloc('int32_D_buf', size=params['D_int32_size'], mem_level="L3")
    mem['fp32_D_buf'] = BingoMemAlloc('fp32_D_buf', size=params['D_fp32_size'], mem_level="L3")

    # L1 buffers for GEMM
    mem['l1_A'] = BingoMemAlloc('l1_A', size=params['A_int8_size'], mem_level="L1",
                                chip_id=chip_id, cluster_id=cluster_id)
    mem['l1_B'] = BingoMemAlloc('l1_B', size=params['B_int8_size'], mem_level="L1",
                                chip_id=chip_id, cluster_id=cluster_id)
    mem['l1_D'] = BingoMemAlloc('l1_D', size=params['D_int32_size'], mem_level="L1",
                                chip_id=chip_id, cluster_id=cluster_id)
    return mem


def create_dfg(params, mem):
    dfg = BingoDFG(
        num_chiplets=1,
        num_clusters_per_chiplet=1,
        num_cores_per_cluster=2,
        is_host_as_acc=True,
        chiplet_ids=[0x00],
    )
    gemm_core = 0
    dma_core = 1
    host_core = 2

    # ── 1. Quantize FP32 -> INT8 (on CVA6 host) ──────────────────
    quant_A = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
        kernel_name="__host_bingo_kernel_fp32_quantize",
        kernel_args=HostBingoKernelFp32QuantizeArgs(
            input_addr=mem['fp32_A_L3'],
            output_addr=mem['int8_A_buf'],
            scale_out_addr=mem['scale_A_buf'],
            num_elements=params['num_A'],
        ),
    )
    quant_B = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
        kernel_name="__host_bingo_kernel_fp32_quantize",
        kernel_args=HostBingoKernelFp32QuantizeArgs(
            input_addr=mem['fp32_B_L3'],
            output_addr=mem['int8_B_buf'],
            scale_out_addr=mem['scale_B_buf'],
            num_elements=params['num_B'],
        ),
    )

    # ── 2. Check quantized A against golden (byte-exact) ──────────
    check_quant_A = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=mem['golden_int8_A'],
            output_data_addr=mem['int8_A_buf'],
            data_size=params['A_int8_size'],
        ),
    )
    check_quant_B = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=mem['golden_int8_B'],
            output_data_addr=mem['int8_B_buf'],
            data_size=params['B_int8_size'],
        ),
    )

    # ── 3. Load INT8 A, B to L1 (DMA on cluster) ─────────────────
    load_A = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem['int8_A_buf'],
            dst_addr=mem['l1_A'],
            size=params['A_int8_size'],
        ),
    )
    load_B = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem['int8_B_buf'],
            dst_addr=mem['l1_B'],
            size=params['B_int8_size'],
        ),
    )

    # ── 4. GEMM: INT8 x INT8 -> INT32 (VersaCore) ────────────────
    gemm = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=gemm_core,
        kernel_name="__snax_bingo_kernel_gemm_full",
        kernel_args=SnaxBingoKernelGemmFullArgs(
            input_A_addr=mem['l1_A'],
            input_B_addr=mem['l1_B'],
            input_C_addr=0,
            output_D_addr=mem['l1_D'],
            M=params['M'], K=params['K'], N=params['N'],
            array_shape_idx=params['arrayShapeIdx'],
            transpose_A=0, transpose_B=0, accumPrevC=0,
        ),
    )

    # ── 5. Store INT32 D from L1 to L3 ────────────────────────────
    store_D = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem['l1_D'],
            dst_addr=mem['int32_D_buf'],
            size=params['D_int32_size'],
        ),
    )

    # ── 6. Check GEMM result (byte-exact INT32) ───────────────────
    check_gemm = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=mem['golden_int32_D'],
            output_data_addr=mem['int32_D_buf'],
            data_size=params['D_int32_size'],
        ),
    )

    # ── 7. Dequantize INT32 -> FP32 (on CVA6 host) ───────────────
    dequant_D = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
        kernel_name="__host_bingo_kernel_int32_dequantize",
        kernel_args=HostBingoKernelInt32DequantizeArgs(
            input_addr=mem['int32_D_buf'],
            output_addr=mem['fp32_D_buf'],
            scale_addr=mem['combined_scale'],
            num_elements=params['num_D'],
        ),
    )

    # ── 8. Check dequantized FP32 result ──────────────────────────
    check_dequant = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=mem['golden_fp32_D'],
            output_data_addr=mem['fp32_D_buf'],
            data_size=params['D_fp32_size'],
        ),
    )

    # ── Add nodes ─────────────────────────────────────────────────
    for node in [quant_A, quant_B, check_quant_A, check_quant_B,
                 load_A, load_B, gemm, store_D, check_gemm,
                 dequant_D, check_dequant]:
        dfg.bingo_add_node(node)

    # ── Add edges (serial data flow) ─────────────────────────────
    # Quantize A/B can run in parallel, but host is single-threaded
    dfg.bingo_add_edge(quant_A, quant_B)
    dfg.bingo_add_edge(quant_A, check_quant_A)
    dfg.bingo_add_edge(quant_B, check_quant_B)
    # After quantize checks pass, load to L1
    dfg.bingo_add_edge(check_quant_A, load_A)
    dfg.bingo_add_edge(check_quant_B, load_B)
    # DMA loads are on same core, serialize
    dfg.bingo_add_edge(load_A, load_B)
    # GEMM after both loads complete
    dfg.bingo_add_edge(load_B, gemm)
    # Store result
    dfg.bingo_add_edge(gemm, store_D)
    # Check GEMM result
    dfg.bingo_add_edge(store_D, check_gemm)
    # Dequantize
    dfg.bingo_add_edge(check_gemm, dequant_D)
    # Check dequantized result
    dfg.bingo_add_edge(dequant_D, check_dequant)

    return dfg


def main():
    args = get_args()
    output_dir = args.output_dir
    print(f"Output DIR: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    params, merged_config = define_workload_params(args.cfg, args.hwcfg)

    if args.data_h is not None:
        data_h_content = emit_header_file(**merged_config)
        with open(args.data_h, "w") as f:
            f.write(data_h_content)
        print(f"Written data header: {args.data_h}")

    mem_handles = define_memory_handles(params)
    dfg = create_dfg(params, mem_handles)
    dfg.bingo_compile_dfg(
        params["app_name"], output_dir, args.output_offload_file_name,
        extra_include_header_list=["dtype_conv_data.h"],
    )


if __name__ == "__main__":
    main()
