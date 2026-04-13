#!/usr/bin/env python3
"""
Attention test: FP32 quantize kernel in isolation.

DFG: Quantize_X (Host,c0) -> Check_int8_X (Host,c0)

Tests __host_bingo_kernel_fp32_quantize by quantizing a FP32 input
and checking the INT8 output against a golden reference.
"""

import os
import sys
import re
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantize_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelCheckResultArgs,
    HostBingoKernelFp32QuantizeArgs,
)


def parse_platform_cfg(occamy_h_path, rtlcfg_path):
    """Parse HW platform parameters from generated occamy.h and RTL config."""
    defines = {}
    with open(occamy_h_path) as f:
        for line in f:
            m = re.match(r'#define\s+(\w+)\s+(\d+)', line)
            if m:
                defines[m.group(1)] = int(m.group(2))
    with open(rtlcfg_path) as f:
        rtlcfg = hjson.loads(f.read())
    multichip = rtlcfg["hemaia_multichip"]
    if multichip["single_chip"]:
        chiplet_ids = [0x00]
    else:
        chiplet_ids = []
        for chip in multichip["testbench_cfg"]["hemaia_compute_chip"]:
            x, y = chip["coordinate"]
            chiplet_ids.append((x << 4) | y)
    return {
        "num_chiplets": defines["N_CHIPLETS"],
        "num_clusters_per_chiplet": defines["N_CLUSTERS_PER_CHIPLET"],
        "num_cores_per_cluster": defines["N_CORES_PER_CLUSTER"],
        "chiplet_ids": chiplet_ids,
    }


def get_args():
    parser = argparse.ArgumentParser(description="attn_test_quantize")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--rtlcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load configs
    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

    # Emit data header
    if args.data_h is not None:
        content = emit_header_file(**merged)
        with open(args.data_h, "w") as f:
            f.write(content)
        print(f"Written data header: {args.data_h}")

    # Parse HW platform parameters
    platform = parse_platform_cfg(args.platformcfg, args.rtlcfg)

    # Derive params
    seq_len = merged["seq_len"]
    d_model = merged["d_model"]
    num_elements = seq_len * d_model

    # Core IDs
    HOST_CORE = 2

    # -- Memory handles --
    # L3 symbols (golden data from quantize_data.h)
    sym_fp32_X = BingoMemSymbol("fp32_X")
    sym_golden_int8_X = BingoMemSymbol("golden_int8_X")

    # L3 buffers (runtime allocation)
    int8_X_buf = BingoMemAlloc("int8_X_buf", size=num_elements, mem_level="L3")
    scale_X_buf = BingoMemAlloc("scale_X_buf", size=4, mem_level="L3")

    # -- DFG --
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    # Node 1: Quantize FP32 -> INT8
    node_quantize = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Quantize_X",
        kernel_name="__host_bingo_kernel_fp32_quantize",
        kernel_args=HostBingoKernelFp32QuantizeArgs(
            input_addr=sym_fp32_X,
            output_addr=int8_X_buf,
            scale_out_addr=scale_X_buf,
            num_elements=num_elements,
        ),
    )

    # Node 2: Check INT8 output against golden
    node_check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_int8_X",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=sym_golden_int8_X,
            output_data_addr=int8_X_buf,
            data_size=num_elements,
            name="int8_X",
        ),
    )

    # Add nodes
    for n in [node_quantize, node_check]:
        dfg.bingo_add_node(n)

    # Edges: Quantize -> Check
    dfg.bingo_add_edge(node_quantize, node_check)

    print(f"Built DFG: 2 nodes (Quantize_X -> Check_int8_X)")
    print(f"  num_elements={num_elements}")

    dfg.bingo_compile_dfg(
        "attn_test_quantize", output_dir, args.output_offload_file_name,
        extra_include_header_list=["quantize_data.h"],
    )


if __name__ == "__main__":
    main()
