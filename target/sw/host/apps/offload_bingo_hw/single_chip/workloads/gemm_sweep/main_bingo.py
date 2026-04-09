# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# gemm_sweep: RTL cycle characterization for VersaCore GEMM.
#
# Single DFG that runs MANY (M,K,N,shape) configs back-to-back.  For each
# config, two GEMMs are executed on the same data:
#   1. gemm_full   — configures the streamers for this (M,K,N,shape) then runs
#   2. gemm_min    — reuses the streamers the full GEMM just set up, only
#                    reconfigures base addresses (measures the steady-state
#                    GEMM cost without reconfiguration overhead)
#
# After both GEMMs the host compares D against its golden.  All timing is
# captured via the BINGO_TRACE_GEMM_FULL_RUN_* and BINGO_TRACE_GEMM_MIN_RUN_*
# markers already embedded in the device kernels, so `make bingo-vis-traces`
# produces bin/logs/bingo_trace.json with one GEMM_FULL_RUN + one GEMM_MIN_RUN
# event per config (in the order configs are listed below).

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

from gemm_datagen import emit_header_file, config_sizes, mesh_dims  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelGemmMinimalArgs,
    HostBingoKernelCheckResultArgs,
)

# ─────────────────────────────────────────────────────────────────────────
#  Sweep configurations: (M, K, N, array_shape)
#
#  Values are in mesh tiles (per-config mesh dims come from the hw hjson).
#  Chosen to:
#    - Cover shapes 0..4 (all VersaCore array configurations)
#    - Vary M, K, N independently so the 3-parameter linear fit is well-
#      conditioned (cycles = a + b·M·K·N + c·max(M,K,N))
#    - Keep the A+B+D footprint inside 500 KB TCDM per config
# ─────────────────────────────────────────────────────────────────────────
CONFIGS = [
    # shape 0 (meshRow=32, tileSize=4, meshCol=32)
    (1, 1, 1, 0), (1, 2, 1, 0), (1, 4, 1, 0), (1, 8, 1, 0),
    (2, 2, 2, 0), (2, 4, 2, 0), (4, 4, 4, 0),
    # shape 1 (meshRow=1, tileSize=8, meshCol=64)
    (1, 1, 1, 1), (1, 2, 1, 1), (1, 4, 1, 1), (1, 8, 1, 1),
    (2, 2, 2, 1), (2, 4, 2, 1), (4, 4, 4, 1),
    # shape 2 (meshRow=4, tileSize=8, meshCol=64)
    (1, 1, 1, 2), (1, 2, 1, 2), (1, 4, 1, 2),
    (2, 2, 2, 2), (2, 4, 2, 2), (4, 4, 4, 2),
    # shape 3 (meshRow=8, tileSize=8, meshCol=64)
    (1, 1, 1, 3), (1, 2, 1, 3), (1, 4, 1, 3),
    (2, 2, 2, 3), (4, 4, 4, 3),
    # shape 4 (meshRow=8, tileSize=32, meshCol=8)
    (1, 1, 1, 4), (1, 2, 1, 4), (1, 4, 1, 4),
    (2, 2, 2, 4), (4, 4, 4, 4),
]


def get_args():
    parser = argparse.ArgumentParser(description="gemm_sweep — RTL cycle characterization")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    parser.add_argument("--l1_size_kb", type=int, default=500)
    parser.add_argument("--configs_out", type=pathlib.Path, default=None,
                        help="Optional: dump CONFIGS as JSON for the shell "
                             "driver to correlate trace events with configs.")
    return parser.parse_args()


def check_fits_in_l1(hwcfg: dict, l1_size_kb: int) -> None:
    """Raise if any config's A+B+D exceeds L1 budget (with allocator overhead)."""
    l1_bytes = l1_size_kb * 1024
    for M, K, N, shape in CONFIGS:
        A_sz, B_sz, D_sz = config_sizes(hwcfg, M, K, N, shape)
        # Rough allocator overhead: each allocation padded to 256 B
        frag = lambda x: (x + 128 + 255) & ~255
        total = frag(A_sz) + frag(B_sz) + frag(D_sz)
        assert total <= l1_bytes, (
            f"Config (M={M},K={K},N={N},shape={shape}) needs {total} B "
            f"(A={A_sz} + B={B_sz} + D={D_sz}) — exceeds L1 budget {l1_bytes}"
        )


def max_l1_sizes(hwcfg: dict) -> tuple[int, int, int]:
    """Return (max_A, max_B, max_D) across all configs."""
    max_A = max_B = max_D = 0
    for M, K, N, shape in CONFIGS:
        A_sz, B_sz, D_sz = config_sizes(hwcfg, M, K, N, shape)
        max_A = max(max_A, A_sz)
        max_B = max(max_B, B_sz)
        max_D = max(max_D, D_sz)
    return max_A, max_B, max_D


def create_dfg(hwcfg: dict):
    """Build a DFG that sweeps every config × {gemm_full, gemm_min}."""
    # Determine shared L1 buffer sizes (large enough for the biggest config)
    max_A, max_B, max_D = max_l1_sizes(hwcfg)
    # Pad A to 64 B for xDMA alignment
    max_A = (max_A + 63) & ~63

    dfg = BingoDFG(
        num_chiplets=1, num_clusters_per_chiplet=1, num_cores_per_cluster=2,
        is_host_as_acc=True, chiplet_ids=[0x00],
    )
    gemm_core = 0
    dma_core = 1
    host_core = 2

    # Shared L1 buffers (reused across configs — each config loads fresh A/B)
    l1_A = BingoMemAlloc("l1_A", size=max_A, mem_level="L1",
                        chip_id=0, cluster_id=0)
    l1_B = BingoMemAlloc("l1_B", size=max_B, mem_level="L1",
                        chip_id=0, cluster_id=0)
    l1_D = BingoMemAlloc("l1_D", size=max_D, mem_level="L1",
                        chip_id=0, cluster_id=0)

    prev_check = None
    for i, (M, K, N, shape) in enumerate(CONFIGS):
        A_sz, B_sz, D_sz = config_sizes(hwcfg, M, K, N, shape)
        A_sz_padded = (A_sz + 63) & ~63  # xDMA alignment

        # L3 symbols generated by gemm_datagen.py
        A_l3 = BingoMemSymbol(f"A_cfg{i}")
        B_l3 = BingoMemSymbol(f"B_cfg{i}")
        D_l3_golden = BingoMemSymbol(f"D_cfg{i}")

        # Load A[i] -> L1
        load_A = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
            node_name=f"Load_A_cfg{i}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(A_l3, l1_A, A_sz_padded),
        )
        # Load B[i] -> L1
        load_B = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
            node_name=f"Load_B_cfg{i}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(B_l3, l1_B, B_sz),
        )
        # gemm_full: configures streamers for (M,K,N,shape) and runs
        gemm_full = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=gemm_core,
            node_name=f"GemmFull_cfg{i}",
            kernel_name="__snax_bingo_kernel_gemm_full",
            kernel_args=SnaxBingoKernelGemmFullArgs(
                input_A_addr=l1_A, input_B_addr=l1_B, input_C_addr=0,
                output_D_addr=l1_D,
                M=M, K=K, N=N, array_shape_idx=shape,
                transpose_A=0, transpose_B=0, accumPrevC=0,
            ),
        )
        # gemm_min: reuses streamers from gemm_full, measures steady-state cost
        gemm_min = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=gemm_core,
            node_name=f"GemmMin_cfg{i}",
            kernel_name="__snax_bingo_kernel_gemm_minimal",
            kernel_args=SnaxBingoKernelGemmMinimalArgs(
                input_A_addr=l1_A, input_B_addr=l1_B, input_C_addr=0,
                output_D_addr=l1_D,
            ),
        )
        # Check result (small prefix of D to keep host I/O cheap)
        check_bytes = min(D_sz, 256)
        check = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
            node_name=f"Check_cfg{i}",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=D_l3_golden, output_data_addr=l1_D,
                data_size=check_bytes,
            ),
        )
        for n in [load_A, load_B, gemm_full, gemm_min, check]:
            dfg.bingo_add_node(n)
        # Edges: serialize loads, gemm_full after loads, gemm_min after full,
        # check after gemm_min. Serial with previous config's check.
        dfg.bingo_add_edge(load_A, load_B)
        dfg.bingo_add_edge(load_B, gemm_full)
        dfg.bingo_add_edge(gemm_full, gemm_min)
        dfg.bingo_add_edge(gemm_min, check)
        if prev_check is not None:
            dfg.bingo_add_edge(prev_check, load_A)
        prev_check = check

    return dfg


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.hwcfg) as f:
        hwcfg = hjson.loads(f.read())

    check_fits_in_l1(hwcfg, args.l1_size_kb)

    # Emit data header with per-config A/B/D arrays
    if args.data_h is not None:
        header = emit_header_file(CONFIGS, hwcfg)
        with open(args.data_h, "w") as f:
            f.write(header)
        print(f"Written data header: {args.data_h}")

    # Optional: dump CONFIGS as JSON so the shell driver can correlate
    # trace events (GEMM_FULL_RUN #i / GEMM_MIN_RUN #i) with (M,K,N,shape).
    if args.configs_out is not None:
        import json
        with open(args.configs_out, "w") as f:
            json.dump({"configs": CONFIGS}, f, indent=2)
        print(f"Written configs list: {args.configs_out}")

    dfg = create_dfg(hwcfg)
    dfg.bingo_compile_dfg(
        "Single-Chip GEMM Sweep",
        args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["gemm_data.h"],
    )
    print(f"Generated DFG: {len(CONFIGS)} configs × (gemm_full + gemm_min)")


if __name__ == "__main__":
    main()
