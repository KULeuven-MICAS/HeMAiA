#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Minimal cross-chiplet CLUSTER-xDMA remote write: chiplet 00 cluster 0 pushes a
# 1 KiB payload straight from its own L1 into chiplet 01 cluster 0's L1, and
# chiplet 01 checks what landed.
#
# Task dependency graph:
#
#   Load_Payload_MemPool_to_Chip00_L1 -> XDMA_Remote_Write_Chip00_to_Chip01
#   Poison_Recv_Buffer_Chip01_L1      -> XDMA_Remote_Write_Chip00_to_Chip01
#   XDMA_Remote_Write_Chip00_to_Chip01 -> Check_Payload_Received_On_Chip01
# END WORKLOAD DESCRIPTION AND TASK GRAPH

"""Smallest workload that exercises a cross-chiplet xDMA remote WRITE.

Why this workload exists
------------------------
The other multi-chip tests move data between chiplets with the *iDMA*
(`dma_*`/`xdma_int32_add_*`, which only uses xDMA for the local add) or with the
*host* xDMA (`dma_write_from_other_chip_*`). Neither touches the path this
workload targets: the SNAX **cluster** xDMA writing into a *remote chiplet's*
cluster L1.

That path is special because it is not a plain AXI write. The sending xDMA first
configures the *receiving* cluster's writer through the xDMA cross-cluster
control MMIO -- the top 12 KiB of the destination cluster window
(cfg/grant/finish, 4 KiB each, per xdma_axi_adapter's MMIO{Cfg,Grant,Finish}Offset)
-- and then waits for a grant before any data moves. Those control writes are
NARROW. Across chiplets they are upsized to WIDE to cross the D2D link, so the
address map must keep that top 12 KiB narrow-only on the receiver; otherwise the
control write lands on the cluster's wide slave, which has no cfg demux, is
dropped, no grant ever returns, and the sender hangs. See the
`quad_narrow_cluster_<i>_xdma_ctrl` leaf in util/occamygen/occamy.py.

So: a PASS here means the cross-chiplet xDMA cfg/grant handshake completed and
the data arrived. A hang means it did not.

Flow
----
  1. Chiplet 00 iDMA-loads the payload from the MemPool chip into its L1.
  2. Chiplet 01 iDMA-loads a poison pattern into its own L1 receive buffer, so a
     PASS cannot come from stale or coincidentally-correct L1 content.
  3. Chiplet 00's cluster xDMA writes its L1 buffer directly into chiplet 01's
     L1 buffer -- the transfer under test.
  4. Chiplet 01's host core checks the received buffer against a golden array in
     its own L3.

Cross-chip address trick
------------------------
Chiplet 00 must name a buffer that lives on chiplet 01. Bingo emits L1
allocations per chiplet in alphabetical order of handle name
(BingoDFG._collect_memory_handles sorts by `h.name`), so two handles that share a
name -- and are preceded by the same set of names on both chiplets -- land at the
same cluster-local offset. Here there is exactly ONE L1 buffer name,
`A_xfer_l1`, allocated on both chiplets, so both sit at the same offset by
construction. Chiplet 00 therefore takes its OWN `ptr_A_xfer_l1`, masks off the
chip tag, and re-tags it with chiplet 01's ID.

Note the buffer must be referenced by some node's kernel_args to be allocated at
all: an unreferenced BingoMemAlloc is never collected, which would silently skew
the offsets. Both handles here are referenced.
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


APP_NAME = "xdma_remote_write_4chiplet_1cluster"
# The platform this runs on; only SRC and DST actually do any work.
REQUIRED_CHIPLETS = [0x00, 0x01, 0x10, 0x11]
SRC_CHIPLET = 0x00
DST_CHIPLET = 0x01
DMA_CORE = 1
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


def chiplet_full_addr_from_c_expr(chiplet_id, c_expr):
    """Re-tag a cluster-local address with another chiplet's ID, as a C expression."""
    local_addr_expr = f"((uint64_t)({c_expr}) & {LOW_40_BIT_ADDR_MASK})"
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

    # The ONLY L1 buffer name, allocated on both chiplets -> identical
    # cluster-local offset on both, which is what makes the remote address below
    # constructible from the sender's own pointer.
    l1_xfer = {
        chiplet: BingoMemAlloc(
            "A_xfer_l1",
            size=payload_bytes,
            mem_level="L1",
            chip_id=chiplet,
            cluster_id=0,
        )
        for chiplet in (SRC_CHIPLET, DST_CHIPLET)
    }

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    load_payload = BingoNode(
        assigned_chiplet_id=SRC_CHIPLET,
        assigned_cluster_id=0,
        assigned_core_id=DMA_CORE,
        node_name=f"Load_Payload_MemPool_to_Chip{chip_hex(SRC_CHIPLET)}_L1",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_payload,
            dst_addr=l1_xfer[SRC_CHIPLET],
            size=payload_bytes,
        ),
    )

    poison_recv = BingoNode(
        assigned_chiplet_id=DST_CHIPLET,
        assigned_cluster_id=0,
        assigned_core_id=DMA_CORE,
        node_name=f"Poison_Recv_Buffer_Chip{chip_hex(DST_CHIPLET)}_L1",
        kernel_name="__snax_bingo_kernel_idma_1d_copy",
        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
            src_addr=mem_poison,
            dst_addr=l1_xfer[DST_CHIPLET],
            size=payload_bytes,
        ),
    )

    # The transfer under test. src is local L1 (the xDMA is cluster-local and
    # requires one endpoint here); dst is the SAME cluster-local offset on the
    # destination chiplet, re-tagged with its chip ID.
    remote_dst = chiplet_full_addr_from_c_expr(
        DST_CHIPLET, l1_xfer[SRC_CHIPLET].get_c_var_name()
    )
    remote_write = BingoNode(
        assigned_chiplet_id=SRC_CHIPLET,
        assigned_cluster_id=0,
        assigned_core_id=DMA_CORE,
        node_name=(
            f"XDMA_Remote_Write_Chip{chip_hex(SRC_CHIPLET)}"
            f"_to_Chip{chip_hex(DST_CHIPLET)}"
        ),
        kernel_name="__snax_bingo_kernel_xdma_1d_copy",
        kernel_args=SnaxBingoKernelXdma1dCopyArgs(
            src_addr=l1_xfer[SRC_CHIPLET],
            dst_addr=remote_dst,
            size=payload_bytes,
        ),
    )

    check = BingoNode(
        assigned_chiplet_id=DST_CHIPLET,
        assigned_cluster_id=0,
        assigned_core_id=host_core,
        node_name=f"Check_Payload_Received_On_Chip{chip_hex(DST_CHIPLET)}",
        kernel_name="__host_bingo_kernel_check_result",
        kernel_args=HostBingoKernelCheckResultArgs(
            golden_data_addr=mem_golden,
            output_data_addr=l1_xfer[DST_CHIPLET],
            data_size=payload_bytes,
            name=f"payload_recv_chip{chip_hex(DST_CHIPLET)}",
        ),
    )

    for node in (load_payload, poison_recv, remote_write, check):
        dfg.bingo_add_node(node)

    # The poison must land BEFORE the remote write, or it would overwrite it.
    dfg.bingo_add_edge(load_payload, remote_write)
    dfg.bingo_add_edge(poison_recv, remote_write)
    dfg.bingo_add_edge(remote_write, check)

    print("Built DFG: 4 nodes (payload load, poison, cross-chiplet xDMA write, check)")
    print(f"  src_chiplet={chip_hex(SRC_CHIPLET)} -> dst_chiplet={chip_hex(DST_CHIPLET)}")
    print(f"  payload_bytes={payload_bytes}, mempool_base=0x{mempool_base:x}")

    dfg.bingo_compile_dfg(
        APP_NAME,
        output_dir,
        args.output_offload_file_name,
        extra_include_header_list=["xdma_remote_write_data.h"],
    )

    return check


if __name__ == "__main__":
    main()
