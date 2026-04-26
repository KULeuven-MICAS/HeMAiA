#!/usr/bin/env python3
"""MoE4: 4 experts, top-2 — tests normal CERF (1 group per expert, no sharing).

The user only specifies cond_dic on edges. The compiler auto-inserts the
gating node, assigns CERF groups, and populates all kernel args.
"""

import os
import sys
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR = os.path.dirname(current_dir)
sys.path.append(WORKLOADS_DIR)
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from moe_datagen import generate_moe_data, emit_header_file
from bingo_dfg import BingoDFG
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol
from bingo_kernel_args import (
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    HostBingoKernelFp32SoftmaxArgs,
)

CHIPLET_ID = 0x00
GEMM_CORE = 0
DMA_CORE = 1
HOST_CORE = 2
NUM_CLUSTERS = 4


def get_args():
    parser = argparse.ArgumentParser(description="MoE4 Workload")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def load_hw_params(cfg_path, hwcfg_path):
    with open(cfg_path) as f:
        param = hjson.loads(f.read())
    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}
    data_type = 0
    array_shape = merged["array_shape"]
    snax_acc_cfg = merged["snax_versacore_core_template"]["snax_acc_cfg"][0]
    unrolling = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]
    return {
        'M': merged["M"], 'K': merged["K"], 'N': merged["N"],
        'meshRow': unrolling[0], 'tileSize': unrolling[1], 'meshCol': unrolling[2],
        'arrayShapeIdx': array_shape,
        'transposeA': merged.get("transposed_A", 0),
        'transposeB': merged.get("transposed_B", 0),
        'accumPrevC': merged.get("accumPrevC", 0),
        'num_experts': merged["num_experts"],
        'top_k': merged["top_k"],
    }


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    hw = load_hw_params(args.cfg, args.hwcfg)

    num_experts = hw['num_experts']
    top_k = hw['top_k']
    A_size = hw['M'] * hw['K'] * hw['meshRow'] * hw['tileSize']
    B_size = hw['K'] * hw['N'] * hw['meshCol'] * hw['tileSize']
    D_size = hw['M'] * hw['N'] * hw['meshRow'] * hw['meshCol'] * 4

    print(f"MoE4: {num_experts} experts, top-{top_k}")

    # Data header
    params = {**hw, 'A_size': A_size, 'B_size': B_size, 'D_size': D_size}
    if args.data_h:
        data = generate_moe_data(params)
        emit_header_file(str(args.data_h), params, data)

    # DFG
    dfg = BingoDFG(num_chiplets=1, num_clusters_per_chiplet=NUM_CLUSTERS,
                   num_cores_per_cluster=2, is_host_as_acc=True, chiplet_ids=[CHIPLET_ID])

    logits = BingoMemAlloc("logits", num_experts * 4, "L3")
    router = BingoNode(CHIPLET_ID, 0, HOST_CORE, node_name="router",
                       kernel_name="__host_bingo_kernel_fp32_softmax",
                       kernel_args=HostBingoKernelFp32SoftmaxArgs(
                           input_addr=BingoMemSymbol("router_logits", offset=0),
                           output_addr=logits, num_rows=1, row_length=num_experts))
    dfg.bingo_add_node(router)

    for i in range(num_experts):
        cl = i % NUM_CLUSTERS
        l1_A = BingoMemAlloc(f"l1_A_{i}", A_size, "L1", chip_id=0, cluster_id=cl)
        l1_B = BingoMemAlloc(f"l1_B_{i}", B_size, "L1", chip_id=0, cluster_id=cl)
        l1_D = BingoMemAlloc(f"l1_D_{i}", D_size, "L1", chip_id=0, cluster_id=cl)
        D_L3 = BingoMemAlloc(f"D_L3_{i}", D_size, "L3")

        ldA = BingoNode(CHIPLET_ID, cl, DMA_CORE, node_name=f"e{i}_ldA",
                        kernel_name="__snax_bingo_kernel_idma_1d_copy",
                        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                            src_addr=BingoMemSymbol("input_A", offset=0), dst_addr=l1_A, size=A_size))
        ldB = BingoNode(CHIPLET_ID, cl, DMA_CORE, node_name=f"e{i}_ldB",
                        kernel_name="__snax_bingo_kernel_idma_1d_copy",
                        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                            src_addr=BingoMemSymbol(f"expert_{i}_B", offset=0), dst_addr=l1_B, size=B_size))
        gemm = BingoNode(CHIPLET_ID, cl, GEMM_CORE, node_name=f"e{i}_gemm",
                         kernel_name="__snax_bingo_kernel_gemm_full",
                         kernel_args=SnaxBingoKernelGemmFullArgs(
                             input_A_addr=l1_A, input_B_addr=l1_B, input_C_addr=l1_D, output_D_addr=l1_D,
                             M=hw['M'], K=hw['K'], N=hw['N'], array_shape_idx=hw['arrayShapeIdx'],
                             transpose_A=hw['transposeA'], transpose_B=hw['transposeB'],
                             accumPrevC=hw['accumPrevC']))
        stD = BingoNode(CHIPLET_ID, cl, DMA_CORE, node_name=f"e{i}_stD",
                        kernel_name="__snax_bingo_kernel_idma_1d_copy",
                        kernel_args=SnaxBingoKernelIdma1dCopyArgs(src_addr=l1_D, dst_addr=D_L3, size=D_size))
        for n in [ldA, ldB, gemm, stD]:
            dfg.bingo_add_node(n)

        dfg.bingo_add_edge(router, ldA, cond_dic={'mode': 'top_k', 'k': top_k})
        dfg.bingo_add_edge(router, ldB, cond_dic={'mode': 'top_k', 'k': top_k})
        dfg.bingo_add_edge(router, gemm, cond_dic={'mode': 'top_k', 'k': top_k})
        dfg.bingo_add_edge(router, stD, cond_dic={'mode': 'top_k', 'k': top_k})
        dfg.bingo_add_edge(ldA, gemm)
        dfg.bingo_add_edge(ldB, gemm)
        dfg.bingo_add_edge(gemm, stD)

    # Post-execution check: runs after scheduler completes, reads cond_activation
    # to determine which experts were selected, then verifies their D_L3 buffers.
    golden_arr = ", ".join(f"(uint8_t*)(uintptr_t)chiplet_addr_transform((uint64_t)(uintptr_t)expert_{i}_D_golden)" for i in range(num_experts))
    output_arr = ", ".join(f"(uint8_t*)ptr_D_L3_{i}" for i in range(num_experts))
    post_check = [
        "{",
        f"    // Check softmax output against golden",
        f"    float* __logits = (float*)ptr_logits;",
        f"    float* __sm_golden = (float*)(uintptr_t)chiplet_addr_transform((uint64_t)(uintptr_t)softmax_golden);",
        f"    uint32_t __sm_err = 0;",
        f"    for (int __i = 0; __i < {num_experts}; __i++) {{",
        f"        float __diff = __logits[__i] - __sm_golden[__i];",
        f"        if (__diff < 0) __diff = -__diff;",
        f"        int __ok = (__diff < 1e-5f);",
        f'        printf_safe("[Softmax] [%d] got=0x%08x golden=0x%08x %s\\r\\n",',
        f'            __i, *(uint32_t*)&__logits[__i], *(uint32_t*)&__sm_golden[__i],',
        f'            __ok ? "OK" : "MISMATCH");',
        f"        if (!__ok) __sm_err++;",
        f"    }}",
        f'    if (__sm_err) printf_safe("[Softmax] FAIL: %d mismatches\\r\\n", __sm_err);',
        f'    else printf_safe("[Softmax] PASS\\r\\n");',
        f"    uint8_t* __cond_act = (uint8_t*)ptr___cond_act_router;",
        f'    printf_safe("[Routing] selected experts:");',
        f"    for (int __i = 0; __i < {num_experts}; __i++)",
        f'        if (__cond_act[__i]) printf_safe(" %d", __i);',
        f'    printf_safe("\\r\\n");',
        f"    // Check expert D buffers",
        f"    uint8_t* __golden[] = {{{golden_arr}}};",
        f"    uint8_t* __output[] = {{{output_arr}}};",
        f"    for (int __i = 0; __i < {num_experts}; __i++) {{",
        f"        if (!__cond_act[__i]) {{",
        f'            printf_safe("[expert_%d_D] SKIPPED\\r\\n", __i);',
        f"            continue;",
        f"        }}",
        f"        uint32_t __err = 0;",
        f"        for (uint32_t __j = 0; __j < {D_size}; __j++) {{",
        f"            if (__golden[__i][__j] != __output[__i][__j]) __err++;",
        f"        }}",
        f'        if (__err) printf_safe("[expert_%d_D] FAIL: %d mismatches\\r\\n", __i, __err);',
        f'        else printf_safe("[expert_%d_D] PASS\\r\\n", __i);',
        f"    }}",
        "}",
    ]

    extra = [os.path.basename(str(args.data_h))] if args.data_h else None
    dfg.bingo_compile_dfg(app_name=f"MoE4 ({num_experts}E top-{top_k})",
                          output_dir=args.output_dir,
                          output_file_name=args.output_offload_file_name,
                          extra_include_header_list=extra,
                          post_execute_code=post_check)
    print(f"Generated: {os.path.join(args.output_dir, args.output_offload_file_name)}")


if __name__ == "__main__":
    main()
