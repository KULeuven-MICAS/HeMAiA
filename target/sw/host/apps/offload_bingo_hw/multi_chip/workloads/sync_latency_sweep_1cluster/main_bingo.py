#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# ARM D -- cross-chip synchronization latency of the BINGO *hardware* manager.
#
# Deliberately the same graph as arm C's sync_latency_sweep_1cluster (the software
# manager) so the two are directly comparable: same phases, same rectangles, same
# round-trip measurement, same printed quantities. Only the scheduler differs.
#
#   A0 --mids(phase 0)--> A1 --mids(phase 1)--> A2 -- ... --> An
#   ^^                    ^^
#   anchor tasks, all on chip 0x00, each stamps mcycle into stamp_buf[i]
#
# Phase i's latency is stamp[i+1] - stamp[i]: out to that phase's participants and back.
# mcycle is per-core and NOT synchronized across chiplets, so a one-way remote timing is
# not a measurable quantity; anchoring both ends on chip 0x00 is what makes it real. The
# round trip covers TWO edges.
#
# Phases, in order:
#   0            untimed WARM-UP: one-to-all over the largest rectangle. Not optional --
#                a chiplet's first touch costs ~1000 cc more than later ones, which broke
#                arm C's P=2 identity check until this was added.
#   1..2n        per rectangle: one-to-one (0x00 -> furthest -> back) then one-to-all
#                (0x00 -> every other participant -> back)
#   last         local control: the same chain entirely on chip 0x00, i.e. the manager's
#                own per-edge bookkeeping with zero cross-chip distance. Deliberately
#                LAST: a chip00-only phase in the middle lets remote hosts go idle, and
#                the next remote phase then pays a re-engagement cost.
# END WORKLOAD DESCRIPTION AND TASK GRAPH

import argparse
import csv
import os
import pathlib
import sys

import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)

from bingo_dfg import BingoDFG  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelSyncReportArgs,
    SnaxBingoKernelSyncProbeArgs,
)
from bingo_mem_handle import BingoMemAlloc  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_cluster_count, parse_platform_cfg  # noqa E402

APP_NAME = "sync_latency_sweep_1cluster"
SRC_CHIP = 0x00
DMA_CORE = 1
HOST_CORE = 2

# Rectangle bottom-right per P; also the furthest chiplet from 0x00 in that rectangle,
# since chip_id = (x << 4) | y and both coordinates are maximal there.
#   P=2 -> 0x01 (1x2), 1 hop     P=8  -> 0x13 (2x4), 4 hops
#   P=4 -> 0x11 (2x2), 2 hops    P=16 -> 0x33 (4x4), 6 hops
RECT_BR = [0x01, 0x11, 0x13, 0x33]

KIND_WARMUP, KIND_LOCAL, KIND_1TO1, KIND_1TOALL = 0, 1, 2, 3


def get_args():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=False)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def rect_chips(br):
    return [(x << 4) | y
            for x in range(0, (br >> 4) + 1)
            for y in range(0, (br & 0x0F) + 1)]


def hops(chip):
    return (chip >> 4) + (chip & 0x0F)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    max_p = int(param.get("max_p", 16))

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    available = set(platform["chiplet_ids"])

    # Rectangles this platform can express, up to the cap.
    rects = [br for br in RECT_BR
             if len(rect_chips(br)) <= max_p and set(rect_chips(br)) <= available]
    if not rects:
        raise ValueError(f"{APP_NAME}: no rectangle fits platform {sorted(available)}")
    warm_br = rects[-1]

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    # One stamp per anchor, on chip 0x00. mcycle is only comparable within a chiplet, so
    # every timestamp in the sweep has to come from here.
    num_phases = 1 + 2 * len(rects) + 1              # warm-up + 2/rect + local control
    stamp_buf = BingoMemAlloc("A_sync_stamps", size=4 * (num_phases + 1),
                              mem_level="L2", chip_id=SRC_CHIP, cluster_id=0)

    phases = []          # (kind, P, hops, edges) in emission order
    slot = [0]
    nodes = []

    def anchor():
        n = BingoNode(
            assigned_chiplet_id=SRC_CHIP, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
            node_name=f"Anchor_{slot[0]}",
            kernel_name="__snax_bingo_kernel_sync_probe",
            kernel_args=SnaxBingoKernelSyncProbeArgs(stamp_buf=stamp_buf, slot=slot[0]),
        )
        slot[0] += 1
        dfg.bingo_add_node(n)
        nodes.append(n)
        return n

    def mid(chip, tag):
        # stamp_buf=0: these are never timed directly, only bracketed by the anchors.
        n = BingoNode(
            assigned_chiplet_id=chip, assigned_cluster_id=0, assigned_core_id=DMA_CORE,
            node_name=f"Mid_{tag}_chip{chip:02x}",
            kernel_name="__snax_bingo_kernel_sync_probe",
            kernel_args=SnaxBingoKernelSyncProbeArgs(stamp_buf=0, slot=0),
        )
        dfg.bingo_add_node(n)
        nodes.append(n)
        return n

    def emit_phase(kind, chips, p, hop, tag):
        nonlocal cur
        nxt = anchor()
        for c in chips:
            m = mid(c, tag)
            dfg.bingo_add_edge(cur, m)
            dfg.bingo_add_edge(m, nxt)
        edges = 0 if kind == KIND_WARMUP else 2 * len(chips)
        phases.append((kind, p, hop, edges))
        cur = nxt

    cur = anchor()

    # phase 0: untimed warm-up over the largest rectangle -- every participating chiplet
    # touched once, so no later phase pays a first-touch cost.
    warm_chips = [c for c in rect_chips(warm_br) if c != SRC_CHIP]
    emit_phase(KIND_WARMUP, warm_chips, len(rect_chips(warm_br)), 0, "warmup")

    for br in rects:
        p = len(rect_chips(br))
        emit_phase(KIND_1TO1, [br], p, hops(br), f"1to1_p{p}")
        others = [c for c in rect_chips(br) if c != SRC_CHIP]
        emit_phase(KIND_1TOALL, others, p, hops(br), f"1toall_p{p}")

    # local control, last (see the header note on phase ordering)
    local_phase = len(phases)
    emit_phase(KIND_LOCAL, [SRC_CHIP], 1, 0, "local")

    # Reporting is a HOST node at the very end, so it never sits on a measured path.
    report = BingoNode(
        assigned_chiplet_id=SRC_CHIP, assigned_cluster_id=0, assigned_core_id=HOST_CORE,
        node_name="Sync_Report",
        kernel_name="__host_bingo_kernel_sync_report",
        kernel_args=HostBingoKernelSyncReportArgs(
            stamp_buf=stamp_buf, num_phases=len(phases), local_phase=local_phase),
    )
    dfg.bingo_add_node(report)
    dfg.bingo_add_edge(cur, report)

    # The phase table: the device prints latency by phase index, and this says what each
    # index means. A DFG memory handle only reserves storage, so it cannot be preloaded.
    csv_path = os.path.join(args.output_dir, "sync_phases.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "kind", "P", "hops", "edges"])
        names = {KIND_WARMUP: "warmup", KIND_LOCAL: "local",
                 KIND_1TO1: "1to1", KIND_1TOALL: "1toall"}
        for i, (kind, p, hop, edges) in enumerate(phases):
            w.writerow([i, names[kind], p, hop, edges])

    print(f"Built DFG: {len(nodes) + 1} nodes, {len(phases)} phases, "
          f"rects={[hex(b) for b in rects]}, max_p={max_p}")
    print(f"  phase table -> {csv_path}")

    dfg.bingo_compile_dfg(APP_NAME, args.output_dir, args.output_offload_file_name,
                          extra_include_header_list=[])


if __name__ == "__main__":
    main()
