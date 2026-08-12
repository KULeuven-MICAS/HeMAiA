#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Cross-chiplet CLUSTER-xDMA remote write into BOTH clusters of a remote chiplet:
# chiplet 00 cluster 0 pushes the same 1 KiB payload out of its own L1 into
# chiplet 10 cluster 0's L1 and then into chiplet 10 cluster 1's L1, and
# chiplet 10 checks both.
#
# Task dependency graph:
#
#   Load_Payload_MemPool_to_Chip00_C0_L1 -> XDMA_Remote_Write_Chip00_C0_to_Chip10_C0
#   Poison_Recv_Buffer_Chip10_C0_L1      -> XDMA_Remote_Write_Chip00_C0_to_Chip10_C0
#   XDMA_Remote_Write_Chip00_C0_to_Chip10_C0 -> Check_Payload_Received_On_Chip10_C0
#   XDMA_Remote_Write_Chip00_C0_to_Chip10_C0 -> XDMA_Remote_Write_Chip00_C0_to_Chip10_C1
#   Poison_Recv_Buffer_Chip10_C1_L1      -> XDMA_Remote_Write_Chip00_C0_to_Chip10_C1
#   XDMA_Remote_Write_Chip00_C0_to_Chip10_C1 -> Check_Payload_Received_On_Chip10_C1
# END WORKLOAD DESCRIPTION AND TASK GRAPH

"""Extended `xdma_remote_write_4chiplet_1cluster`: one sender, two remote clusters.

What this adds over the 1-cluster workload
------------------------------------------
`xdma_remote_write_4chiplet_1cluster` proves the cross-chiplet xDMA cfg/grant
handshake closes for ONE destination cluster (chip00 c0 -> chip01 c0). It cannot
say anything about which cluster on the far side was addressed, because there is
only one.

This workload sends to BOTH clusters of chiplet 10 from the same sender. That
matters because the destination cluster is not a field in the transfer -- it is
implied by the destination ADDRESS. The sending xDMA derives the receiving
cluster's control MMIO (cfg/grant/finish, the top 12 KiB of that cluster's
window) from the address it was handed, so a cluster-1 destination exercises a
different control window than a cluster-0 one. Everything downstream of that
derivation -- the narrow-only routing of the control region, the grant coming
back from the right cluster, the finish matching the right task -- is re-run
against a second window.

A collision is caught by construction: both destinations are poisoned first and
both are checked, so if the cluster-1 address resolved back onto cluster 0 (or
vice versa), the neglected cluster stays poisoned and its check fails.

Addressing the second cluster
-----------------------------
Chiplet 00 must name a buffer that lives on chiplet 10, in a cluster it is not
running on. Two facts make that constructible:

  * Bingo emits every named L1 handle BEFORE any scratchpad or kernel-arg
    allocation, and each (chip, cluster) has its own allocator starting at the
    same cluster-local base. This workload puts exactly ONE named L1 handle of
    the same size in each of the three (chip, cluster) pairs it uses, so all
    three are that allocator's first allocation and land at the same
    cluster-local offset.
  * Clusters are a replicated address space: cluster N's window is cluster 0's
    plus `N * cluster_offset` (`cluster_offset` comes from
    occamy_memory_map.h, which occamygen asserts is common to all clusters).

So the sender takes its OWN `ptr_A_xfer_l1_c0`, masks off the chip tag, adds
`cluster_id * cluster_offset`, and re-tags with chiplet 10's ID.

The handles are named per DESTINATION CLUSTER rather than sharing one name, and
that is deliberate. `_collect_memory_handles` emits one C variable `ptr_<name>`
per handle in a single per-chip scope, so two same-named handles on the SAME
chip -- which is exactly what "one name for both of chiplet 10's clusters" would
produce -- emit a duplicate definition and the app does not compile. (The
1-cluster workload gets away with one shared name only because its two handles
sit on different chips, hence in different scopes.) Sharing a name is also not
what makes the offsets line up here; being each cluster's sole named allocation
is.

Note a `BingoMemAlloc` no node's `kernel_args` references is never collected and
therefore never allocated -- adding a padding buffer to line offsets up silently
does nothing. All three handles here are referenced.
"""

import argparse
import os
import pathlib
import sys

import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(current_dir)
# Share the payload/poison/golden generator with the 1-cluster workload so the
# two cannot drift apart. The Makefile depends on that file too.
sys.path.append(os.path.join(current_dir, "..", "xdma_remote_write_4chiplet_1cluster"))

from bingo_dfg import BingoDFG  # noqa E402
from bingo_helpers import chiplet_addr_transform_loc  # noqa E402
from bingo_kernel_args import (  # noqa E402
    HostBingoKernelCheckResultArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemFixedAddr, BingoMemSymbol  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_platform import guard_chiplet_count, guard_cluster_count, parse_platform_cfg  # noqa E402
from xdma_remote_write_datagen import emit_header_file  # noqa E402


APP_NAME = "xdma_remote_write_4chiplet_2cluster"
# The platform this runs on; only SRC and DST actually do any work.
REQUIRED_CHIPLETS = [0x00, 0x01, 0x10, 0x11]
SRC_CHIPLET = 0x00
SRC_CLUSTER = 0
DST_CHIPLET = 0x10
DST_CLUSTERS = [0, 1]
DMA_CORE = 1
# Host kernels must sit on the chiplet-local host core, which the mini-compiler
# fixes at cluster 0 -- even when the buffer they check lives in cluster 1's L1.
HOST_CLUSTER = 0
LOW_40_BIT_ADDR_MASK = "0x000000ffffffffffULL"


def get_args():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()


def chip_hex(chiplet_id):
    return f"{chiplet_id:02x}"


def remote_cluster_l1_expr(chiplet_id, cluster_id, c_expr):
    """C expression for `c_expr`'s cluster-local twin on another chip/cluster.

    `c_expr` is a cluster-0 L1 pointer on the sender. Strip the chip tag, walk
    `cluster_id` cluster windows up the replicated address space, re-tag with the
    destination chip.
    """
    local_addr_expr = f"((uint64_t)({c_expr}) & {LOW_40_BIT_ADDR_MASK})"
    if cluster_id:
        local_addr_expr = f"({local_addr_expr} + (uint64_t)cluster_offset * {cluster_id})"
    return f"(chiplet_addr_transform_full(0x{chiplet_id:02x}, {local_addr_expr}))"


def get_payload_bytes(param):
    payload_bytes = int(param.get("payload_bytes", 1024))
    if payload_bytes <= 0:
        raise ValueError(f"payload_bytes must be positive, got {payload_bytes}")
    if payload_bytes % 64 != 0:
        raise ValueError(f"payload_bytes ({payload_bytes}) must be 64-byte aligned for xDMA")
    return payload_bytes


def main():
    args = get_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

    if args.data_h is not None:
        content = emit_header_file(**merged, out_dir=os.path.join(output_dir, "build"))
        with args.data_h.open("w") as f:
            f.write(content)
        print(f"Written data header: {args.data_h}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_chiplet_count(param, platform, output_dir, args.output_offload_file_name):
        return
    if not guard_cluster_count(param, platform, output_dir, args.output_offload_file_name):
        return

    missing = [c for c in REQUIRED_CHIPLETS if c not in platform["chiplet_ids"]]
    if missing:
        available = ", ".join(chip_hex(c) for c in platform["chiplet_ids"])
        raise ValueError(
            f"{APP_NAME} requires chiplets {', '.join(chip_hex(c) for c in missing)}; "
            f"platform has [{available}]"
        )

    # The host core is the last core of the cluster once the host is exposed as an
    # accelerator (BingoDFG adds it on top of N_CORES_PER_CLUSTER).
    host_core = platform["num_cores_per_cluster"]

    payload_bytes = get_payload_bytes(param)

    # MemPool layout: [payload | poison].
    mempool_base = chiplet_addr_transform_loc(2, 0, 0x8000_0000)
    mem_payload = BingoMemFixedAddr(mempool_base)
    mem_poison = BingoMemFixedAddr(mempool_base + payload_bytes)
    # Golden lives in each chiplet's own L3, so the checker does no cross-chip read.
    mem_golden = BingoMemSymbol("xdma_rw_golden_l3")

    # Exactly one named L1 handle, of the same size, per (chip, cluster) used ->
    # each is its allocator's first allocation and therefore sits at the same
    # cluster-local offset, which is what makes the remote addresses below
    # constructible from the sender's own pointer. Names differ per cluster
    # because same-named handles on the SAME chip collide into one C variable.
    def xfer_buf(chip, cluster):
        return BingoMemAlloc(
            f"A_xfer_l1_c{cluster}",
            size=payload_bytes,
            mem_level="L1",
            chip_id=chip,
            cluster_id=cluster,
        )

    l1_src = xfer_buf(SRC_CHIPLET, SRC_CLUSTER)
    l1_dst = {cluster: xfer_buf(DST_CHIPLET, cluster) for cluster in DST_CLUSTERS}

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    load_payload = BingoNode(
        assigned_chiplet_id=SRC_CHIPLET,
        assigned_cluster_id=SRC_CLUSTER,
        assigned_core_id=DMA_CORE,
        node_name=f"Load_Payload_MemPool_to_Chip{chip_hex(SRC_CHIPLET)}_C{SRC_CLUSTER}_L1",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_payload,
            dst_addr=l1_src,
            size=payload_bytes,
        ),
    )
    dfg.bingo_add_node(load_payload)

    prev_write = None
    for cluster in DST_CLUSTERS:
        h = chip_hex(DST_CHIPLET)

        poison = BingoNode(
            assigned_chiplet_id=DST_CHIPLET,
            assigned_cluster_id=cluster,
            assigned_core_id=DMA_CORE,
            node_name=f"Poison_Recv_Buffer_Chip{h}_C{cluster}_L1",
            kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                src_addr=mem_poison,
                dst_addr=l1_dst[cluster],
                size=payload_bytes,
            ),
        )

        # The transfer under test. src is local L1 (the xDMA is cluster-local and
        # requires one endpoint here); dst is the same cluster-local offset in the
        # destination chiplet's cluster `cluster`.
        remote_write = BingoNode(
            assigned_chiplet_id=SRC_CHIPLET,
            assigned_cluster_id=SRC_CLUSTER,
            assigned_core_id=DMA_CORE,
            node_name=(
                f"XDMA_Remote_Write_Chip{chip_hex(SRC_CHIPLET)}_C{SRC_CLUSTER}"
                f"_to_Chip{h}_C{cluster}"
            ),
            kernel_name="__snax_bingo_kernel_xdma_1d_copy",
            kernel_args=SnaxBingoKernelXdma1dCopyArgs(
                src_addr=l1_src,
                dst_addr=remote_cluster_l1_expr(
                    DST_CHIPLET, cluster, l1_src.get_c_var_name()
                ),
                size=payload_bytes,
            ),
        )

        check = BingoNode(
            assigned_chiplet_id=DST_CHIPLET,
            assigned_cluster_id=HOST_CLUSTER,
            assigned_core_id=host_core,
            node_name=f"Check_Payload_Received_On_Chip{h}_C{cluster}",
            kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                golden_data_addr=mem_golden,
                output_data_addr=l1_dst[cluster],
                data_size=payload_bytes,
                name=f"payload_recv_chip{h}_c{cluster}",
            ),
        )

        for node in (poison, remote_write, check):
            dfg.bingo_add_node(node)

        # The poison must land BEFORE the write, or it would overwrite it.
        dfg.bingo_add_edge(poison, remote_write)
        dfg.bingo_add_edge(remote_write, check)
        if prev_write is None:
            dfg.bingo_add_edge(load_payload, remote_write)
        else:
            # Serialize the two writes so the sender issues them in a known order;
            # the payload dependency comes along transitively.
            dfg.bingo_add_edge(prev_write, remote_write)
        prev_write = remote_write

    print(f"Built DFG: {1 + 3 * len(DST_CLUSTERS)} nodes "
          f"(payload load, then poison/write/check per destination cluster)")
    print(f"  src=chip{chip_hex(SRC_CHIPLET)} c{SRC_CLUSTER} -> "
          f"dst=chip{chip_hex(DST_CHIPLET)} clusters {DST_CLUSTERS}")
    print(f"  payload_bytes={payload_bytes}, mempool_base=0x{mempool_base:x}")

    dfg.bingo_compile_dfg(
        APP_NAME,
        output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["xdma_remote_write_data.h"],
    )


if __name__ == "__main__":
    main()
