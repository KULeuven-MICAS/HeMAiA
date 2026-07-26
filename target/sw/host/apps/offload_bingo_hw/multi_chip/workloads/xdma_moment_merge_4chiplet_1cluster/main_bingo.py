#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Gate A -- CROSS-CHIPLET in-fabric moment-merge. The P=4 cross-cluster point
# (xdma_moment_merge_p4_infabric) lifted to four COMPUTE CHIPLETS: each of chiplets
# 00/01/10/11 holds its own (m_i,l_i) in cluster-0 L1; the assembler chiplet 00
# holds its own value locally and REMOTE-reads (over the die-to-die link) chiplets
# 01/10/11's (m,l) into its staging beat, then pushes the assembled 4-lane beat to
# the merger chiplet 01 with the writer-side UnifiedMonoidMergeRt (MOMENT mode)
# armed (nvalid=4). The writerExtCfg rides the cross-chiplet push over the D2D link,
# so chiplet 01's writer folds the arriving (m,l) stream in-fabric -- the same
# mechanism as the cross-cluster point, now proven across dies.
#
# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Task dependency graph:
#   Init_own (per chiplet 00/01/10/11): mm_mc_m/l_vals[i] -> own_l1[0/4]
#   Init_own_c00 -> Copy_lane0_m, Copy_lane0_l              (chiplet 00's own value -> beat lane 0)
#   Init_own_c01 -> Gather_lane1_m, Gather_lane1_l          (D2D read chiplet 01 -> beat lane 1)
#   Init_own_c10 -> Gather_lane2_m, Gather_lane2_l          (D2D read chiplet 10 -> beat lane 2)
#   Init_own_c11 -> Gather_lane3_m, Gather_lane3_l          (D2D read chiplet 11 -> beat lane 3)
#   {all 8 lane copies} -> MergePush (chiplet 00 -> chiplet 01, ext armed, nvalid=4)
#   MergePush -> Store_result -> Check_result               (on merger chiplet 01)
# END WORKLOAD DESCRIPTION AND TASK GRAPH

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

from momentmerge_multichip_datagen import emit_header_file, CHECK_TOLERANCE  # noqa: E402
from bingo_dfg import BingoDFG  # noqa: E402
from bingo_helpers import chiplet_addr_transform_loc  # noqa: E402
from bingo_platform import guard_chiplet_count, guard_cluster_count, parse_platform_cfg  # noqa: E402
from bingo_node import BingoNode  # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol, BingoMemFixedAddr  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdmaMomentMergePushArgs,
    HostBingoKernelCheckResultArgs,
    BINGO_CHECK_TYPE_FP32_TOL,
)

APP_NAME = "xdma_moment_merge_4chiplet_1cluster"
REQUIRED_CHIPLETS = [0x00, 0x01, 0x10, 0x11]
ASSEMBLER_CHIPLET = 0x00   # holds its own value + gathers the others + issues the push
MERGER_CHIPLET = 0x01      # the cross-chiplet push destination -- its writer folds in-fabric
DMA_CORE = 1
HOST_CORE = 2
BEAT_BYTES = 64
OWN_BYTES = 8  # a chiplet's own (m,l), held contiguously in cluster-0 L1
LOW_40_BIT_ADDR_MASK = "0x000000ffffffffffULL"


def lane_m(k):
    return 4 * k


def lane_l(k):
    return 32 + 4 * k


def chip_hex(chiplet_id):
    return f"{chiplet_id:02x}"


def chiplet_full_addr_from_c_expr(chiplet_id, c_expr):
    local_addr_expr = f"((uint64_t)({c_expr}) & {LOW_40_BIT_ADDR_MASK})"
    return f"(chiplet_addr_transform_full(0x{chiplet_id:02x}, {local_addr_expr}))"


def get_args():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.cfg) as f:
        param = hjson.loads(f.read())

    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(**param))
        print(f"Written data header: {args.data_h}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_chiplet_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return

    missing = [c for c in REQUIRED_CHIPLETS if c not in platform["chiplet_ids"]]
    if missing:
        have = ", ".join(chip_hex(c) for c in platform["chiplet_ids"])
        want = ", ".join(chip_hex(c) for c in missing)
        raise ValueError(f"{APP_NAME} requires chiplets {want}; platform has [{have}]")

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"])

    # Two SHARED-name buffers. The codegen lays out each chip's L1 alloc block
    # ALPHABETICALLY BY HANDLE NAME (bingo_dfg _collect_memory_handles), emitting only
    # the buffers a chip references. For the cross-chip address transforms to land on the
    # right offset, a buffer must sit at the SAME cluster-local offset on every chip that
    # references it -- so the names are chosen so the SHARED buffer sorts first everywhere:
    #   * own_l1 ("...own_l1")  -- sorts 1st => base on ALL chips (00 D2D-reads 01/10/11).
    #   * stage_beat ("...stage_beat") -- sorts 2nd; referenced only on 00 (staging beat,
    #     push SRC) and 01 (fold dest, push DST + check). Both have own(8) first, so the
    #     beat is at the SAME offset on 00 and 01 => 00 can name 01's beat via the transform.
    # (A name that sorted BEFORE own_l1 would put the beat first on 00/01 but leave own at
    #  base on 10/11 -- a silent offset mismatch that breaks the 10/11 gather.)
    l1_own = {
        c: BingoMemAlloc("mm_mc_own_l1", size=OWN_BYTES, mem_level="L1", chip_id=c, cluster_id=0)
        for c in REQUIRED_CHIPLETS
    }
    l1_beat = {
        c: BingoMemAlloc("mm_mc_stage_beat", size=BEAT_BYTES, mem_level="L1", chip_id=c, cluster_id=0)
        for c in REQUIRED_CHIPLETS
    }
    l1_stage = l1_beat[ASSEMBLER_CHIPLET]  # the assembled 4-lane beat on chiplet 00 (push SRC)
    # The fold destination MUST be the MERGER chiplet's CLUSTER memory (L1/TCDM) -- that is where the
    # writer-side UnifiedMonoidMergeRt lives. Main memory has no writer extension, so the fold can only
    # happen at a cluster writer. So chiplet 00 pushes the assembled beat ACROSS the D2D link into
    # chiplet 01's cluster L1; chiplet 01's writer folds the arriving (m,l) lanes in-transit (nvalid=4)
    # and the writerExtCfg rides the cross-chiplet push to configure it.
    l1_dest = l1_beat[MERGER_CHIPLET]  # the in-fabric fold destination on chiplet 01 (cluster L1)

    nodes = []
    edges = []

    # Each chiplet loads its own (m_i,l_i) -- one contiguous 8-byte record -- from the
    # device-binary symbol into own_l1 with a SINGLE idma copy. One root per chiplet is
    # what the cross-chiplet start-broadcast can arm (one (cluster,core) cell each).
    init_own = {}
    for i, c in enumerate(REQUIRED_CHIPLETS):
        h = chip_hex(c)
        init_own[c] = BingoNode(
            assigned_chiplet_id=c, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
            node_name=f"Init_own_c{h}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=BingoMemSymbol("mm_mc_ml_vals", offset=8 * i),
                dst_addr=l1_own[c].view(0), size=OWN_BYTES))
        nodes.append(init_own[c])

    # Shared cluster-local c-var of own_l1 (same offset on every chiplet).
    own_c_var = l1_own[ASSEMBLER_CHIPLET].get_c_var_name()

    # Assembler chiplet 00: its own value copies locally into beat lane 0.
    lane_srcs = []  # (node) that must complete before the push
    a_own = init_own[ASSEMBLER_CHIPLET]
    copy0_m = BingoNode(
        assigned_chiplet_id=ASSEMBLER_CHIPLET, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Copy_lane0_m",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=l1_own[ASSEMBLER_CHIPLET].view(0), dst_addr=l1_stage.view(lane_m(0)), size=4))
    copy0_l = BingoNode(
        assigned_chiplet_id=ASSEMBLER_CHIPLET, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="Copy_lane0_l",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=l1_own[ASSEMBLER_CHIPLET].view(4), dst_addr=l1_stage.view(lane_l(0)), size=4))
    nodes += [copy0_m, copy0_l]
    edges += [(a_own, copy0_m), (a_own, copy0_l)]
    lane_srcs += [copy0_m, copy0_l]

    # Assembler chiplet 00 REMOTE-reads (D2D) chiplets 01/10/11's own (m,l) into lanes 1,2,3.
    for k in (1, 2, 3):
        c = REQUIRED_CHIPLETS[k]
        h = chip_hex(c)
        p_own = init_own[c]
        rem_m = chiplet_full_addr_from_c_expr(c, own_c_var)
        rem_l = chiplet_full_addr_from_c_expr(c, f"((uintptr_t){own_c_var} + 4)")
        g_m = BingoNode(
            assigned_chiplet_id=ASSEMBLER_CHIPLET, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
            node_name=f"Gather_lane{k}_m_c{h}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=rem_m, dst_addr=l1_stage.view(lane_m(k)), size=4))
        g_l = BingoNode(
            assigned_chiplet_id=ASSEMBLER_CHIPLET, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
            node_name=f"Gather_lane{k}_l_c{h}",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=rem_l, dst_addr=l1_stage.view(lane_l(k)), size=4))
        nodes += [g_m, g_l]
        edges += [(p_own, g_m), (p_own, g_l)]
        lane_srcs += [g_m, g_l]

    # The cross-chiplet in-fabric fold: chiplet 00 pushes the assembled beat ACROSS the D2D link to
    # chiplet 01's cluster L1; 01's writer-side UnifiedMonoidMergeRt folds the 4 lanes (nvalid=4) as the
    # beat lands. dst = 01's mm_mc_stage_beat, named from the ASSEMBLER's own c-var (shared name => same
    # cluster-local offset on 00 and 01).
    dest_full = chiplet_full_addr_from_c_expr(MERGER_CHIPLET, l1_beat[ASSEMBLER_CHIPLET].get_c_var_name())
    merge_push = BingoNode(
        assigned_chiplet_id=ASSEMBLER_CHIPLET, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
        node_name="MergePush",
        kernel_name="__snax_bingo_kernel_xdma_moment_merge_push",
        kernel_args=SnaxBingoKernelXdmaMomentMergePushArgs(
            src_addr=l1_stage, dst_addr=dest_full, nvalid=4))
    nodes.append(merge_push)
    for src in lane_srcs:
        edges.append((src, merge_push))

    # Check on the merger chiplet 01 (host), reading the folded (m*,l*) from its cluster L1.
    check_result = BingoNode(
        assigned_chiplet_id=MERGER_CHIPLET, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Check_result",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=BingoMemSymbol("mm_mc_golden_pair"),
            output_data_addr=l1_dest,
            check_type=BINGO_CHECK_TYPE_FP32_TOL,
            tolerance=CHECK_TOLERANCE,
            num_elements=2,
            name="mm_mc_4chiplet"))
    nodes += [check_result]
    edges += [(merge_push, check_result)]

    for n in nodes:
        dfg.bingo_add_node(n)
    for u, v in edges:
        dfg.bingo_add_edge(u, v)

    print(f"Built DFG: {len(nodes)} nodes, {len(edges)} edges "
          f"(4 own inits x2 + 2 local lane0 + 6 D2D gather + merge push + store + check)")
    print(f"  chiplets={[chip_hex(c) for c in REQUIRED_CHIPLETS]}, "
          f"assembler={chip_hex(ASSEMBLER_CHIPLET)}, merger={chip_hex(MERGER_CHIPLET)}")

    dfg.bingo_compile_dfg(
        APP_NAME, args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["momentmerge_multichip_data.h"])
    print(f"Generated {APP_NAME} DFG")


if __name__ == "__main__":
    main()
