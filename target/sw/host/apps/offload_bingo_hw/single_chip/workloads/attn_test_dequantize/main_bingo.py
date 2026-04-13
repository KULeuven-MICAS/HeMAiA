#!/usr/bin/env python3
"""
Attention test: INT32 dequantize kernel in isolation.

DFG: Dequantize (Host,c0) -> Check_fp32 (Host,c0)

Tests __host_bingo_kernel_int32_dequantize by dequantizing an INT32 input
with a known scale and checking the FP32 output against a golden reference.
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

from dequant_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelCheckResultArgs,
    HostBingoKernelInt32DequantizeArgs,
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
    parser = argparse.ArgumentParser(description="attn_test_dequantize")
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
    d_head = merged["d_head"]
    num_elements = seq_len * d_head

    # Core IDs
    HOST_CORE = 2

    # -- Memory handles --
    # L3 symbols (golden data from dequant_data.h)
    sym_int32_D = BingoMemSymbol("int32_D_input")
    sym_golden_fp32 = BingoMemSymbol("golden_fp32_D")
    sym_scale = BingoMemSymbol("combined_scale")

    # L3 buffer (runtime allocation) - FP32 output = num_elements * 4 bytes
    fp32_D_buf = BingoMemAlloc("fp32_D_buf", size=num_elements * 4, mem_level="L3")

    # -- DFG --
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    # Node 1: Dequantize INT32 -> FP32
    node_dequant = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Dequantize",
        kernel_name="__host_bingo_kernel_int32_dequantize",
        kernel_args=HostBingoKernelInt32DequantizeArgs(
            input_addr=sym_int32_D,
            output_addr=fp32_D_buf,
            scale_addr=sym_scale,
            num_elements=num_elements,
        ),
    )

    # Node 2: Check FP32 output against golden
    node_check = BingoNode(
        assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_fp32",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=sym_golden_fp32,
            output_data_addr=fp32_D_buf,
            data_size=num_elements * 4,
            name="fp32_D",
        ),
    )

    # Add nodes
    for n in [node_dequant, node_check]:
        dfg.bingo_add_node(n)

    # Edges: Dequantize -> Check
    dfg.bingo_add_edge(node_dequant, node_check)

    print(f"Built DFG: 2 nodes (Dequantize -> Check_fp32)")
    print(f"  num_elements={num_elements}")

    dfg.bingo_compile_dfg(
        "attn_test_dequantize", output_dir, args.output_offload_file_name,
        extra_include_header_list=["dequant_data.h"],
    )


if __name__ == "__main__":
    main()
