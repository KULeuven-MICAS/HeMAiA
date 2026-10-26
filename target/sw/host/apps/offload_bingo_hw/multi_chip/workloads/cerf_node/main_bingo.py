#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Decode, one node: what it costs to change one node of the graph at run time
# (docs/cerf_128bit_todo.md, section 2). Empty kernels; only the control path is timed.
#
#   join -> [cerf  x R] -> [skip  x R] -> [cross x R] -> hub -> host arm (every chiplet)
#
#   cerf   src (host 0x00) -> gate (static, on) -> node (DM 0x00, gated) -> after (DM 0x00)
#   skip   the same with the gate off: the node is skipped, `after` is released by the skip
#   cross  src (host 0x00) -> gate (on) -> node (DM of the far chiplet) -> after (far DM),
#          the CERF window riding the dep-set (cerf_carry)
#   dev    as cerf, the gate on a device core: src and gate on 0x00's DM core, node and after on core 0
#   tiles  the tile skip of a prefill expert slot, on 0x00: a device gate (the 'tiles' fork)
#          reads tile_counts and enables tiles 0 .. ceil(count / tile_rows) - 1 of each of
#          tile_slots slots of tile_count tiles, every tile its own CERF group (slot s, tile t on
#          cluster 1, core s); the tasks are empty, the point is the device CERF write and
#          which tiles start
#   cost   0x00's host times the pieces of a host CERF write, then a whole-CERF update several ways
#          (cerf_node_kernels.h); then the same ways from a device core (__snax_bingo_kernel_cerf_cost)
#   host   a node on node_chip that the graph skips (its gate is off); then on every chiplet a
#          host task that, R times, rebuilds the node's descriptor (node chiplet), synchronizes
#          the hosts (a chip barrier over every chiplet) and relaunches the node
#          (bingo_hw_scheduler_launch) -- see cerf_node_kernels.h.
#
# Every round is a chain, so rounds never overlap. The DFG holds its exit tasks back
# (hold_exits): the host arm launches them at its end, so between the launches every core is
# free to take the relaunched node.
#
# Timing comes from the simulation, one time base for host and devices (bingo_progress.log):
# [BINGO CERF] a CERF write landing, [BINGO LAUNCH] a START write landing, [BINGO EV] each
# task's release (R), start (S) and done (D). The host arm also prints its own host cycles.
# report.py joins them through cerf_node_map.csv, which names every task of every round.
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

import _bingo_paths  # noqa: F401,E402  (puts mini_compiler's grouped subdirs on sys.path)
from typing import Dict  # noqa E402
from bingo_dfg import BingoDFG  # noqa E402
from bingo_kernel_args import (  # noqa E402
    BingoKernelArgs,
    HostBingoKernelChipBarrierArgs,
    SnaxBingoKernelCerfStaticArgs,
    SnaxBingoKernelSyncProbeArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import core_roles, guard_cluster_count, parse_platform_cfg  # noqa E402

APP_NAME = "cerf_node"
SRC_CHIP = 0x00
_ROLES = core_roles()
DM_CORE = _ROLES["dm"]
HOST_CORE = _ROLES["host"]


class HostCerfNodeNopArgs(BingoKernelArgs):
    """__host_bingo_kernel_cerf_node_nop (cerf_node_kernels.h): a host task doing nothing."""

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_cerf_node_nop_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        return {}


class HostCerfNodeCostArgs(HostCerfNodeNopArgs):
    """__host_bingo_kernel_cerf_node_cost (cerf_node_kernels.h): a host CERF write, piece by piece."""

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_cerf_node_cost_args_t"


class HostCerfNodeHostArmArgs(BingoKernelArgs):
    """__host_bingo_kernel_cerf_node_host_arm (cerf_node_kernels.h)."""

    def __init__(self, node_chip, node_task_id, rounds, flag, top_left, bottom_right):
        self.node_chip = int(node_chip)
        self.node_task_id = int(node_task_id)
        self.rounds = int(rounds)
        self.flag = flag
        self.top_left = int(top_left)
        self.bottom_right = int(bottom_right)

    def get_struct_name(self) -> str:
        return "__host_bingo_kernel_cerf_node_host_arm_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {"node_chip": str(self.node_chip), "node_task_id": str(self.node_task_id),
             "rounds": str(self.rounds), "top_left": str(self.top_left),
             "bottom_right": str(self.bottom_right)}
        self._process_addr(self.flag, "flag_addr", a, handle_name_map,
                           split_64bit=False, as_64bit=True)
        return a


def get_args():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=False)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    rounds = int(param.get("rounds", 20))
    if not 1 <= rounds <= 64:
        raise ValueError(f"{APP_NAME}: rounds={rounds}, 1..64 (CERF_NODE_MAX_ROUNDS)")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    chips = sorted(platform["chiplet_ids"])
    far_chip = int(str(param.get("far_chip", chips[-1])), 0)
    node_chip = int(str(param.get("node_chip", SRC_CHIP)), 0)
    if far_chip not in chips or node_chip not in chips:
        raise ValueError(f"{APP_NAME}: far_chip 0x{far_chip:02x} / node_chip 0x{node_chip:02x} "
                         f"not among the compute chiplets {[hex(c) for c in chips]}")

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )
    dfg.hold_exits = True

    def node(chip, cluster, core, name, kernel, kargs):
        n = BingoNode(assigned_chiplet_id=chip, assigned_cluster_id=cluster,
                      assigned_core_id=core, node_name=name, kernel_name=kernel,
                      kernel_args=kargs)
        dfg.bingo_add_node(n)
        return n

    def nop(chip, name):
        return node(chip, 0, HOST_CORE, name, "__host_bingo_kernel_cerf_node_nop",
                    HostCerfNodeNopArgs())

    def probe(chip, name, stamp=0):
        return node(chip, 0, DM_CORE, name, "__snax_bingo_kernel_sync_probe",
                    SnaxBingoKernelSyncProbeArgs(stamp_buf=stamp, slot=0))

    # Every chiplet clears its chip-barrier state before any chiplet's first barrier: all the
    # inits before one join on 0x00, which starts everything else.
    join = nop(SRC_CHIP, "join")
    for c in chips:
        init = node(c, 0, HOST_CORE, f"barrier_init_{c:02x}", HostBingoKernelChipBarrierArgs.KERNEL_NAME,
                    HostBingoKernelChipBarrierArgs(op=HostBingoKernelChipBarrierArgs.OP_INIT))
        dfg.bingo_add_edge(init, join)

    rows = []           # (arm, round, src, node, after)
    cur = join

    def gated_round(arm, r, on, chip, device=False):
        nonlocal cur
        # the gate sits on its source's core: the host, or (device) chip 0x00's DM core of cluster 0, whose
        # gated node and consumer then run on core 0 of the same cluster
        src = probe(SRC_CHIP, f"{arm}_src_{r}") if device else nop(SRC_CHIP, f"{arm}_src_{r}")
        dfg.bingo_add_edge(cur, src)
        fork = dfg.bingo_conditional_fork(src, {"mode": "static", "write_mask": 1 if on else 0})
        core = 0 if device else DM_CORE
        n = node(chip, 0, core, f"{arm}_node_{r}", "__snax_bingo_kernel_sync_probe",
                 SnaxBingoKernelSyncProbeArgs(stamp_buf=0, slot=0))
        fork.branch([n])
        after = node(chip, 0, core, f"{arm}_after_{r}", "__snax_bingo_kernel_sync_probe",
                     SnaxBingoKernelSyncProbeArgs(stamp_buf=0, slot=0))
        fork.combine(after, inputs=[n], kind="sum")
        rows.append((arm, r, src, n, after))
        cur = after

    for arm, on, chip, device in (("cerf", True, SRC_CHIP, False), ("skip", False, SRC_CHIP, False),
                                  ("cross", True, far_chip, False), ("dev", True, SRC_CHIP, True)):
        for r in range(rounds):
            gated_round(arm, r, on, chip, device)

    # The tile arm: one device gate, every slot on its own core of cluster 1. The counts are a
    # static array of the program image (cerf_node_data.h), read by the gate on its chiplet.
    tile_rows = int(param.get("tile_rows", 8))
    tile_count = int(param.get("tile_count", 32))
    counts = [int(c) for c in param.get("tile_counts", [0, 37, 256, 1])]
    if len(counts) > DM_CORE + 1:
        raise ValueError(f"{APP_NAME}: {len(counts)} tile slots, one a core of cluster 1")
    with open(os.path.join(args.output_dir, "cerf_node_data.h"), "w") as f:
        f.write("// Generated by main_bingo.py: the tile arm's rows per slot.\n#pragma once\n"
                "#include <stdint.h>\n"
                f"static const uint32_t cerf_node_tile_counts[{len(counts)}] "
                f"__attribute__((aligned(8))) = {{{', '.join(str(c) for c in counts)}}};\n")
    tsrc = probe(SRC_CHIP, "tiles_src")
    dfg.bingo_add_edge(cur, tsrc)
    tfork = dfg.bingo_conditional_fork(
        tsrc, {"mode": "tiles", "rows": tile_rows,
               "counts": BingoMemSymbol("cerf_node_tile_counts")})
    tiles = []
    for s_ in range(len(counts)):
        for t in range(tile_count):
            tile = node(SRC_CHIP, 1, s_, f"tile_s{s_}_t{t}", "__snax_bingo_kernel_sync_probe",
                        SnaxBingoKernelSyncProbeArgs(stamp_buf=0, slot=0))
            tfork.branch([tile], slot=s_, tile=t)
            tiles.append((s_, t, tile))
    tafter = node(SRC_CHIP, 1, DM_CORE, "tiles_after", "__snax_bingo_kernel_sync_probe",
                  SnaxBingoKernelSyncProbeArgs(stamp_buf=0, slot=0))
    tfork.combine(tafter, inputs=[t for _, _, t in tiles], kind="sum")
    cur = tafter

    # A CERF update, piece by piece and several ways, from 0x00's host and then from the device gate's core.
    # After the tiles: nothing waits on a CERF group meanwhile, and both give the CERF back as they found it.
    cost = node(SRC_CHIP, 0, HOST_CORE, "cerf_cost", "__host_bingo_kernel_cerf_node_cost",
                HostCerfNodeCostArgs())
    dfg.bingo_add_edge(cur, cost)
    words = dfg.cerf_words
    ctrl = sum(0x00FFFF00 << (32 * w) for w in range(words))
    state = sum((0xA5A5A5A5 if w & 1 else 0x5A5A5A5A) << (32 * w) for w in range(words)) & ctrl
    dcost = node(SRC_CHIP, 0, DM_CORE, "cerf_cost_dev", "__snax_bingo_kernel_cerf_cost",
                 SnaxBingoKernelCerfStaticArgs(ctrl, state, words))
    dfg.bingo_add_edge(cost, dcost)
    cur = dcost

    # The host arm's node: in the graph so its kernel and arguments are in the device tables,
    # skipped by the graph itself (gate off), run only by the relaunches.
    hub = nop(SRC_CHIP, "hub")
    dfg.bingo_add_edge(cur, hub)
    flag = BingoMemAlloc("cerf_node_flag", size=64, mem_level="L1", chip_id=node_chip,
                         cluster_id=0)
    hsrc = nop(node_chip, "host_src")
    dfg.bingo_add_edge(hub, hsrc)
    hfork = dfg.bingo_conditional_fork(hsrc, {"mode": "static", "write_mask": 0})
    hnode = probe(node_chip, "host_node", stamp=flag)
    hfork.branch([hnode])
    arms = {}
    for c in chips:
        arms[c] = node(c, 0, HOST_CORE, f"host_arm_{c:02x}",
                       "__host_bingo_kernel_cerf_node_host_arm",
                       HostCerfNodeHostArmArgs(node_chip, hnode.node_id, rounds,
                                               flag if c == node_chip else 0,
                                               chips[0], chips[-1]))
        dfg.bingo_add_edge(hub, arms[c])
    hfork.combine(arms[node_chip], inputs=[hnode], kind="sum")

    print(f"Built DFG: {len(dfg.node_list)} nodes, {rounds} rounds a arm, far chiplet "
          f"0x{far_chip:02x}, host-arm node on 0x{node_chip:02x}")
    dfg.bingo_compile_dfg(APP_NAME, args.output_dir, args.output_offload_file_name,
                          extra_include_header_list=["cerf_node_data.h", "cerf_node_kernels.h"])

    # Every task of every round, by id, for report.py. The gate is the node the compiler
    # spliced in after the source.
    gate_of = {n.node_name: n for n in dfg.node_list if n.node_type == "gating"}
    with open(os.path.join(args.output_dir, "cerf_node_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "round", "gate_chip", "node_chip", "src", "gate", "node", "after",
                    "group"])
        for arm, r, src, n, after in rows:
            g = gate_of[f"__gating_{src.node_name}"]
            w.writerow([arm, r, f"{g.assigned_chiplet_id:02x}", f"{n.assigned_chiplet_id:02x}",
                        src.node_id, g.node_id, n.node_id, after.node_id, n.cond_exec_group_id])
        g = gate_of[f"__gating_{hsrc.node_name}"]
        w.writerow(["host", -1, f"{g.assigned_chiplet_id:02x}", f"{node_chip:02x}",
                    hsrc.node_id, g.node_id, hnode.node_id, arms[node_chip].node_id,
                    hnode.cond_exec_group_id])
        # one row a tile: round = slot * tile_count + tile; enabled = 1 if the gate runs it
        g = gate_of[f"__gating_{tsrc.node_name}"]
        for s_, t, tile in tiles:
            on = t < min(-(-counts[s_] // tile_rows), tile_count)
            w.writerow([f"tile{int(on)}", s_ * tile_count + t, f"{g.assigned_chiplet_id:02x}",
                        f"{SRC_CHIP:02x}", tsrc.node_id, g.node_id, tile.node_id,
                        tafter.node_id, tile.cond_exec_group_id])


if __name__ == "__main__":
    main()
