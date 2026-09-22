# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# BEGIN WORKLOAD DESCRIPTION AND TASK GRAPH
# Main-memory bandwidth, measured in three phases so the read-only rate, the write-only rate
# and the concurrent rate can be compared in ONE run.
#
# WHY IT EXISTS. Every other workload on this chip is read-dominated against main memory --
# FA streams K/V in and writes (m, l, O) back only in its epilogue, and the tiled GEMMs put
# all four Store_D at the end -- so none of them can tell whether the L3 front end serves a
# read and a write in the same cycle or serialises them. That is exactly the property the
# front end's read/write split changes (docs/l3_frontend.md), so it needs its own benchmark.
#
# Task dependency graph (NCL clusters, REPS transfers each, one iDMA engine per cluster so
# a cluster's own transfers chain; the clusters run concurrently):
#
#   phase R   c0..c3:  Rd_c{c}_0 -> Rd_c{c}_1 -> ... -> Rd_c{c}_{REPS-1}     L3 src -> L1
#                      all tails -> Bar_R (host)
#   phase W   c0..c3:  Bar_R -> Wr_c{c}_0 -> ... -> Wr_c{c}_{REPS-1}         L1 -> L3 dst_c
#                      all tails -> Bar_W (host)
#   phase M   c0, c1:  Bar_W -> Mx_c{c}_0 -> ...                             L3 src -> L1
#             c2, c3:  Bar_W -> Mx_c{c}_0 -> ...                             L1 -> L3 dst_c
#                      all tails -> Check_c{c} (host), chained
#
# The host barriers are what keep the phases apart: without them the manager would start a
# write while reads are still in flight and the three rates would all measure the same mix.
#
# READING THE RESULT. Each phase's rate is (NCL * REPS * xfer_bytes) / (phase span), with the
# span taken from the trace. Read-only and write-only give each direction on its own; the
# mixed phase moves half as much in each direction at once. If the front end serialises the
# two directions, mixed lands at the same aggregate as either single phase; if it serves them
# from separate ports, mixed is higher.
# END WORKLOAD DESCRIPTION AND TASK GRAPH

import os
import sys
import argparse
import pathlib

current_dir = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR = os.path.dirname(current_dir)
sys.path.append(WORKLOADS_DIR)
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)
APP_NAME = "Single-Chip L3 Bandwidth"

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim")
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)

import _bingo_paths  # noqa: F401,E402  (puts mini_compiler's grouped subdirs on sys.path)
from bingo_dfg import BingoDFG  # noqa E402
from bingo_platform import core_roles, guard_cluster_count, parse_platform_cfg  # noqa E402
from bingo_node import BingoNode  # noqa E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol  # noqa E402
from bingo_kernel_args import (  # noqa E402
    SnaxBingoKernelIdma1dCopyArgs,
    HostBingoKernelCheckResultArgs,
)
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

CHIP = 0


def get_args():
    p = argparse.ArgumentParser(description="Bingo HW Manager")
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    p.add_argument("--configs_out", type=pathlib.Path, default=None)
    p.add_argument("--static-l1", action="store_true",
                   help="let the compiler place L1 buffers statically (default: off)")
    return p.parse_args()


def define_workload_params(cfg_path, _hwcfg_path):
    params = {}
    with open(cfg_path) as f:
        for line in f:
            line = line.split("//", 1)[0].split("#", 1)[0].strip()
            if not line or line in ("{", "}"):
                continue
            key, sep, value = line.partition(":")
            if not sep:
                continue
            product = 1
            for factor in value.strip().rstrip(",").split("*"):
                product *= int(factor.strip(), 0)
            params[key.strip()] = product
    for name in ("num_clusters", "xfer_bytes", "reps"):
        if name not in params:
            raise KeyError(f"params.hjson must define {name}")
    return params, dict(params)


def emit_header_file(**kw):
    n = kw["xfer_bytes"]
    # A byte pattern rather than zeros: a copy that never ran would leave the destination
    # at whatever L3 held, and against zeros that can pass by accident.
    src = [(idx * 7 + 13) & 0xFF for idx in range(n)]
    return "\n\n".join([
        "#include <stdint.h>",
        format_scalar_definition("uint32_t", "L3BW_xfer_bytes", n),
        format_vector_definition("uint8_t", "L3BW_src", src),
    ]) + "\n"


def define_memory_handles(params):
    ncl = params["num_clusters"]
    n = params["xfer_bytes"]
    h = {"src": BingoMemSymbol("L3BW_src", offset=0)}
    for c in range(ncl):
        h[f"l1_{c}"] = BingoMemAlloc(f"l3bw_l1_{c}", size=n, mem_level="L1",
                                     chip_id=CHIP, cluster_id=c)
        # One destination per cluster: a shared one would make the four writers race and
        # the final check would only prove that SOMEBODY wrote it. No cluster_id -- an L3
        # allocation belongs to the chip, and the compiler rejects one that claims a cluster.
        h[f"l3_{c}"] = BingoMemAlloc(f"l3bw_l3_{c}", size=n, mem_level="L3", chip_id=CHIP)
    return h


def create_dfg(params, h, platform):
    roles = core_roles(platform)
    dma_core = roles["dm"]
    host_core = roles["host"]
    ncl = params["num_clusters"]
    reps = params["reps"]
    n = params["xfer_bytes"]

    g = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    def copy(name, c, src, dst, after):
        node = BingoNode(
            assigned_chiplet_id=CHIP, assigned_cluster_id=c, assigned_core_id=dma_core,
            node_name=name, kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(src_addr=src, dst_addr=dst, size=n),
        )
        g.bingo_add_node(node)
        for a in after:
            g.bingo_add_edge(a, node)
        return node

    def chain(prefix, clusters, direction, after):
        """One serial chain of `reps` transfers per cluster; the chains run concurrently.

        Chained rather than left independent because there is one iDMA engine per cluster:
        they would serialise on it anyway, and stating the order keeps the manager's ready
        set small."""
        tails = []
        for c in clusters:
            prev = list(after)
            for i in range(reps):
                if direction == "read":
                    node = copy(f"{prefix}_c{c}_{i}", c, h["src"], h[f"l1_{c}"], prev)
                else:
                    node = copy(f"{prefix}_c{c}_{i}", c, h[f"l1_{c}"], h[f"l3_{c}"], prev)
                prev = [node]
            tails.append(prev[0])
        return tails

    def barrier(name, after):
        # A host node with no work of its own. The check kernel is the only host kernel that
        # needs no destination, so it doubles as the barrier: it re-verifies cluster 0's L1
        # against the golden, which is true at every point it is used here.
        node = BingoNode(
            assigned_chiplet_id=CHIP, assigned_cluster_id=0, assigned_core_id=host_core,
            node_name=name, kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                name=name, golden_data_addr=h["src"],
                output_data_addr=h["l1_0"], data_size=64,
            ),
        )
        g.bingo_add_node(node)
        for a in after:
            g.bingo_add_edge(a, node)
        return node

    # phase R: every cluster reads main memory
    bar_r = barrier("Bar_R", chain("Rd", range(ncl), "read", []))
    # phase W: every cluster writes main memory
    bar_w = barrier("Bar_W", chain("Wr", range(ncl), "write", [bar_r]))
    # phase M: half read while the other half writes -- the case the split is for
    half = ncl // 2
    tails = chain("Mx", range(half), "read", [bar_w])
    tails += chain("Mx", range(half, ncl), "write", [bar_w])

    # The checks prove every destination really holds the source bytes. Chained so the four
    # of them do not contend on the host, which would blur the end of the mixed phase.
    prev = tails
    for c in range(ncl):
        node = BingoNode(
            assigned_chiplet_id=CHIP, assigned_cluster_id=0, assigned_core_id=host_core,
            node_name=f"Check_c{c}", kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                name=f"L3_c{c}", golden_data_addr=h["src"],
                output_data_addr=h[f"l3_{c}"], data_size=n,
            ),
        )
        g.bingo_add_node(node)
        for a in prev:
            g.bingo_add_edge(a, node)
        prev = [node]
    return g


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    params, merged = define_workload_params(args.cfg, args.hwcfg)

    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(**merged))

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(merged, platform, args.output_dir,
                               args.output_offload_file_name):
        return
    h = define_memory_handles(params)
    dfg = create_dfg(params, h, platform)
    data_header = os.path.basename(args.data_h) if args.data_h is not None else "l3bw_data.h"
    dfg.bingo_compile_dfg(APP_NAME, args.output_dir, args.output_offload_file_name,
                          extra_include_header_list=[data_header],
                          static_l1=args.static_l1)
    moved = params["num_clusters"] * params["reps"] * params["xfer_bytes"]
    print(f"L3 bandwidth: {params['num_clusters']} clusters x {params['reps']} x "
          f"{params['xfer_bytes']} B = {moved} B per phase, three phases")


if __name__ == "__main__":
    main()
