# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The main() every rung of the LLM bring-up ladder shares.

A rung's own main_bingo.py is one call into here with its stage count. Everything that
could differ between rungs and should not -- the cfg parsing, the mesh read, the staging,
the L1 capacity, the compile -- lives here once.
"""

import argparse
import os

import hjson

from bingo_dfg import BingoDFG
from bingo_data_staging import DataStaging
from bingo_platform import (core_roles, guard_cluster_count, load_cluster_cfg,
                            parse_platform_cfg)
from libs import Ctx
from libs.block.flash_attention import mesh_from_hwcfg

from llm_layer_data import generate_layer_data, stage
from llm_layer_stages import MAX_STAGE, STAGE_NAMES, build, token_parallel

CHIPLET_ID = 0x00
# The heap the static-L1 pass must fit inside. SET IT: left unset the pass computes a peak,
# prints it without a bound and never fails, so an overflow reaches the simulation as one
# buffer quietly overlapping another's bytes.
L1_CAPACITY = 514816


def run(stages=MAX_STAGE, verify="all", shard="none", decomp="single"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_h", default=None)
    ap.add_argument("--output_offload_file_name", default="offload_bingo_hw.h")
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--hwcfg", required=True)
    ap.add_argument("--platformcfg", required=True)
    args = ap.parse_args()

    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    hw = load_cluster_cfg(args.hwcfg)
    mesh = mesh_from_hwcfg(args.hwcfg, int(p.get("array_shape", 0)))
    p["meshRow"], p["tileSize"], p["meshCol"] = mesh
    plat = parse_platform_cfg(args.platformcfg)
    guard_cluster_count(p, plat, args.output_dir, args.output_offload_file_name)

    data = generate_layer_data(p)
    st = DataStaging(plat)
    hs = stage(st, data, p)

    dfg = BingoDFG(num_chiplets=1,
                   num_clusters_per_chiplet=plat["num_clusters_per_chiplet"],
                   num_cores_per_cluster=plat["num_cores_per_cluster"],
                   is_host_as_acc=True, chiplet_ids=[CHIPLET_ID],
                   dep_tag_width=plat["dep_tag_width"])
    dfg.l1_capacity_bytes = L1_CAPACITY
    # `hw` is the cluster cfg the RTL was elaborated from. Blocks derive the
    # cfg-dependent kernel constants from it -- an xDMA junction's id is its
    # position in that cfg's list, so a literal is silently wrong elsewhere.
    ctx = Ctx(dfg=dfg, mesh=mesh, roles=core_roles(), chiplet=CHIPLET_ID, hw=hw)

    if decomp == "tokens":
        token_parallel(ctx, p, data, hs, verify=verify)
    else:
        build(ctx, p, data, hs, stages=stages, verify=verify, shard=shard)

    if args.data_h:
        st.emit(args.data_h, args.output_dir)
    extra = [os.path.basename(str(args.data_h))] if args.data_h else None
    name = "token-parallel" if decomp == "tokens" else STAGE_NAMES[stages]
    dfg.bingo_compile_dfg(
        app_name=(f"LLM layer ({name}) "
                  f"T={p['tokens']} d={p['d_model']} h={p['d_hidden']}"),
        output_dir=args.output_dir,
        output_file_name=args.output_offload_file_name,
        extra_include_header_list=extra)
    print(f"Generated: {os.path.join(args.output_dir, args.output_offload_file_name)}")
