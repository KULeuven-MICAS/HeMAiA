#!/usr/bin/env python3
# Attention test DFG for __host_bingo_kernel_int32_add.
# Chains 3 add calls: ((p0 + p1) + p2) + p3 → golden_sum_final.
# Validates the kernel needed for inter-cluster partial-D accumulation
# in K-split GEMM schemes (used by attn_test_ksplit_gemm).

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

from int32_add_datagen import emit_header_file  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelCheckResultArgs,
    HostBingoKernelInt32AddArgs,
)

HOST_CORE = 2


def parse_platform_cfg(occamy_h_path, rtlcfg_path):
    """Parse HW platform parameters from generated occamy.h and RTL config.

    - occamy.h provides: N_CHIPLETS, N_CLUSTERS_PER_CHIPLET, N_CORES_PER_CLUSTER
    - RTL config (lru.hjson/hemaia_*.hjson) provides: chiplet coordinates → chiplet_ids

    Chiplet IDs are coordinate-encoded: (x, y) → (x << 4) | y
    E.g. 2×2 grid: (0,0)=0x00, (0,1)=0x01, (1,0)=0x10, (1,1)=0x11
    """
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
    p = argparse.ArgumentParser(description="attn_test_int32_add DFG")
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True,
                   help="Path to generated occamy.h with HW platform defines")
    p.add_argument("--rtlcfg", type=pathlib.Path, required=True,
                   help="Path to hemaia RTL config hjson (e.g. lru.hjson)")
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    return p.parse_args()


def main():
    args = get_args()
    out_dir = args.output_dir
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    with open(args.cfg) as f: param = hjson.loads(f.read())
    with open(args.hwcfg) as f: hw = hjson.loads(f.read())
    merged = {**param, **hw}
    num_elements = merged["num_elements"]
    array_bytes = num_elements * 4

    if args.data_h is not None:
        content = emit_header_file(**merged)
        with open(args.data_h, "w") as f: f.write(content)

    # Parse HW platform parameters from generated occamy.h + RTL config
    platform = parse_platform_cfg(args.platformcfg, args.rtlcfg)

    mem = {}
    # L3 symbols (golden data)
    for sym in ("golden_partial_0", "golden_partial_1", "golden_partial_2", "golden_partial_3",
                "golden_sum_01", "golden_sum_012", "golden_sum_final"):
        mem[sym] = BingoMemSymbol(sym)
    # L3 buffers (intermediate + final results)
    mem["sum_01_l3"] = BingoMemAlloc("sum_01_l3", size=array_bytes, mem_level="L3")
    mem["sum_012_l3"] = BingoMemAlloc("sum_012_l3", size=array_bytes, mem_level="L3")
    mem["sum_final_l3"] = BingoMemAlloc("sum_final_l3", size=array_bytes, mem_level="L3")

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    def add_node(name, a_sym, b_sym, out_buf):
        return BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
            node_name=name, kernel_name="__host_bingo_kernel_int32_add",
            kernel_args=HostBingoKernelInt32AddArgs(
                input_a_addr=a_sym, input_b_addr=b_sym,
                output_addr=out_buf, num_elements=num_elements))

    def check_node(name, golden_sym, output_buf):
        return BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
            node_name=name, kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=golden_sym, output_data_addr=output_buf,
                data_size=array_bytes, name=name))

    # Chain: ((p0 + p1) + p2) + p3
    add_01 = add_node("Add_01", mem["golden_partial_0"], mem["golden_partial_1"], mem["sum_01_l3"])
    check_01 = check_node("Check_01", mem["golden_sum_01"], mem["sum_01_l3"])
    add_012 = add_node("Add_012", mem["sum_01_l3"], mem["golden_partial_2"], mem["sum_012_l3"])
    check_012 = check_node("Check_012", mem["golden_sum_012"], mem["sum_012_l3"])
    add_final = add_node("Add_Final", mem["sum_012_l3"], mem["golden_partial_3"], mem["sum_final_l3"])
    check_final = check_node("Check_Final", mem["golden_sum_final"], mem["sum_final_l3"])

    for n in [add_01, check_01, add_012, check_012, add_final, check_final]:
        dfg.bingo_add_node(n)
    dfg.bingo_add_edge(add_01, check_01)
    dfg.bingo_add_edge(check_01, add_012)
    dfg.bingo_add_edge(add_012, check_012)
    dfg.bingo_add_edge(check_012, add_final)
    dfg.bingo_add_edge(add_final, check_final)

    dfg.bingo_compile_dfg("attn_test_int32_add", out_dir, args.output_offload_file_name,
                          extra_include_header_list=["int32_add_datagen_data.h"])


if __name__ == "__main__":
    main()
