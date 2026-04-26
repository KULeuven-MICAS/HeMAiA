# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# xdma_sweep: RTL cycle characterization for xDMA data-movement kernels.
#
# Single DFG that runs many (op x size) configs back-to-back.  Each config
# performs:
#   1. Load_Input (iDMA L3 → L1)
#   2. the xDMA op (whose BINGO_TRACE_XDMA_RUN_START/END markers give the
#      cycle count we want)
#
# After the RTL sim, `make bingo-vis-traces` produces bin/logs/bingo_trace.json
# containing one XDMA_RUN event per config (in the order configs are listed
# below).  The shell driver correlates events with configs in that order.

import os
import sys
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR = os.path.dirname(current_dir)
sys.path.append(WORKLOADS_DIR)
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from xdma_sweep_datagen import emit_header_file, compute_in_bytes  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdmaTranspose2dArgs,
    SnaxBingoKernelXdmaSubmatrix2dArgs,
)

# ─────────────────────────────────────────────────────────────────────────
#  Sweep configurations: (op, size, rows, cols, elem_bytes)
#    - op ∈ {"copy_1d", "transpose_2d", "submatrix_2d"}
#    - size: total bytes for copy_1d (0 for 2D ops)
#    - rows, cols, elem_bytes: dims for 2D ops (0 for copy_1d)
# ─────────────────────────────────────────────────────────────────────────
CONFIGS = [
    # copy_1d sweep: total bytes (64-byte aligned)
    ("copy_1d", 64,    0, 0, 1),
    ("copy_1d", 256,   0, 0, 1),
    ("copy_1d", 1024,  0, 0, 1),
    ("copy_1d", 4096,  0, 0, 1),
    ("copy_1d", 8192,  0, 0, 1),
    # transpose_2d sweep: NxN with elem_bytes=1 and 4
    ("transpose_2d", 0,  8,  8, 1),
    ("transpose_2d", 0, 16, 16, 1),
    ("transpose_2d", 0, 32, 32, 1),
    ("transpose_2d", 0,  8,  8, 4),
    ("transpose_2d", 0, 16, 16, 4),
    # submatrix_2d sweep
    ("submatrix_2d", 0,  8,  8, 1),
    ("submatrix_2d", 0, 16, 16, 1),
    ("submatrix_2d", 0, 32, 32, 1),
]

OP_NAME_MAP = {
    "copy_1d": "xdma_1d_copy",
    "transpose_2d": "xdma_transpose_2d",
    "submatrix_2d": "xdma_submatrix_2d",
}


def get_args():
    parser = argparse.ArgumentParser(description="xdma_sweep — RTL cycle characterization")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True,
                        help="Path to generated occamy.h with HW platform defines")
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    parser.add_argument("--configs_out", type=pathlib.Path, default=None,
                        help="Optional: dump CONFIGS as JSON for the shell "
                             "driver to correlate trace events with configs.")
    return parser.parse_args()


def out_bytes_for(op: str, rows: int, cols: int, elem: int, in_bytes: int) -> int:
    """Output size of a given xDMA op (copy_1d, transpose_2d are same-size;
    submatrix_2d extracts the top-left quadrant)."""
    if op == "submatrix_2d":
        sub_rows = max(1, rows // 2)
        sub_cols = max(1, cols // 2)
        return sub_rows * sub_cols * elem
    return in_bytes


def build_xdma_node(op: str, src, dst, size: int, rows: int, cols: int,
                    elem_bytes: int, core_id: int, name: str) -> BingoNode:
    if op == "copy_1d":
        return BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=core_id,
            node_name=name, kernel_name="__snax_bingo_kernel_xdma_1d_copy",
            kernel_args=SnaxBingoKernelXdma1dCopyArgs(src, dst, size),
        )
    if op == "transpose_2d":
        return BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=core_id,
            node_name=name, kernel_name="__snax_bingo_kernel_xdma_transpose_2d",
            kernel_args=SnaxBingoKernelXdmaTranspose2dArgs(
                src, dst, M=rows, N=cols, elem_bytes=elem_bytes),
        )
    if op == "submatrix_2d":
        sub_rows = max(1, rows // 2)
        sub_cols = max(1, cols // 2)
        return BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=core_id,
            node_name=name, kernel_name="__snax_bingo_kernel_xdma_submatrix_2d",
            kernel_args=SnaxBingoKernelXdmaSubmatrix2dArgs(
                src, dst, src_rows=rows, src_cols=cols,
                row_start=0, row_end=sub_rows,
                col_start=0, col_end=sub_cols,
                elem_bytes=elem_bytes),
        )
    raise ValueError(f"unknown op: {op}")


def create_dfg(platform: dict):
    """Build a DFG that sweeps every xDMA config."""
    # Determine shared L1 buffer sizes
    max_bytes = 0
    for op, size, rows, cols, elem in CONFIGS:
        max_bytes = max(max_bytes, compute_in_bytes(op, size, rows, cols, elem))

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )
    dma_core = 1

    l1_src = BingoMemAlloc("l1_src", size=max_bytes, mem_level="L1",
                           chip_id=0, cluster_id=0)
    l1_dst = BingoMemAlloc("l1_dst", size=max_bytes, mem_level="L1",
                           chip_id=0, cluster_id=0)

    prev = None
    for i, (op, size, rows, cols, elem) in enumerate(CONFIGS):
        in_bytes = compute_in_bytes(op, size, rows, cols, elem)
        # Load input[i] -> L1
        load = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
            node_name=f"Load_cfg{i}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                BingoMemSymbol(f"input_cfg{i}"), l1_src, in_bytes),
        )
        xdma_node = build_xdma_node(
            op, l1_src, l1_dst, size, rows, cols, elem,
            core_id=dma_core, name=f"Xdma_cfg{i}",
        )
        dfg.bingo_add_node(load)
        dfg.bingo_add_node(xdma_node)
        dfg.bingo_add_edge(load, xdma_node)
        if prev is not None:
            dfg.bingo_add_edge(prev, load)
        prev = xdma_node

    return dfg


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.cfg) as f:
        param = hjson.loads(f.read())

    if args.data_h is not None:
        header = emit_header_file(CONFIGS)
        with open(args.data_h, "w") as f:
            f.write(header)
        print(f"Written data header: {args.data_h}")

    if args.configs_out is not None:
        import json
        # Dump configs with normalized op names (match RTL LUT naming)
        dumped = [
            {
                "op": OP_NAME_MAP[op],
                "in_bytes": compute_in_bytes(op, size, rows, cols, elem),
                "size": size, "rows": rows, "cols": cols, "elem": elem,
            }
            for (op, size, rows, cols, elem) in CONFIGS
        ]
        with open(args.configs_out, "w") as f:
            json.dump({"configs": dumped}, f, indent=2)
        print(f"Written configs list: {args.configs_out}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    dfg = create_dfg(platform)
    dfg.bingo_compile_dfg(
        "Single-Chip xDMA Sweep",
        args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["xdma_sweep_data.h"],
    )
    print(f"Generated DFG: {len(CONFIGS)} xDMA configs")


if __name__ == "__main__":
    main()
