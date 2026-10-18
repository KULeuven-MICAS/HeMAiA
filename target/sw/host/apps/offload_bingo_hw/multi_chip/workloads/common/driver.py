# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""What the multi-chiplet LLM workload drivers (workloads/ffn, moe, fa) share: the checkout's root, the
command line every main_bingo.py takes, and the body of a `run()`:

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))     # workloads/common
    from driver import repo_root, run_driver
    REPO = repo_root(__file__)

    def _app_name(S):
        return f"MoE, {S.T} tokens"

    def run():
        run_driver("moe workload generator", MBuild, _app_name)

run_driver parses the command line, loads the params (-c), the platform and the cluster configuration,
builds the workload (`build()`), and finishes it (`finish(app_name)`: checks, emission, compile).

A driver imports this module before it puts mini_compiler on sys.path (repo_root gives the path), so the
compiler's modules are imported inside run_driver.
"""

import argparse
import os

DSV2_DIR_ARG = {"required": True,
                "help": "snax_cluster's target/snitch_cluster/sw/apps/dsv2 (the numerics models)"}


def repo_root(file):
    """The checkout `file` (a driver's __file__) is in: the path before /target/sw/host/."""
    return os.path.dirname(os.path.abspath(file)).split("/target/sw/host/")[0]


def parse_args(description, dsv2_dir=DSV2_DIR_ARG):
    """The command line of a main_bingo.py. `dsv2_dir`: the keywords of --dsv2_dir, for a driver
    that does not need the numerics models."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_h", default=None)
    ap.add_argument("--output_offload_file_name", default="offload_bingo_hw.h")
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--hwcfg", required=True)
    ap.add_argument("--platformcfg", required=True)
    ap.add_argument("--dsv2_dir", **dsv2_dir)
    return ap.parse_args()


def run_driver(description, build_cls, app_name, dsv2_dir=DSV2_DIR_ARG, **finish_kw):
    """Build and emit one workload: `build_cls(args, params, platform, hw)`, then `build()`, then
    `finish(app_name(S), **finish_kw)`. `app_name` takes the built workload and returns the name the
    report and the emitted header carry."""
    import hjson
    from bingo_platform import load_cluster_cfg, parse_platform_cfg
    args = parse_args(description, dsv2_dir)
    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    plat = parse_platform_cfg(args.platformcfg)
    hw = load_cluster_cfg(args.hwcfg)
    S = build_cls(args, p, plat, hw)
    S.build()
    S.finish(app_name(S), **finish_kw)
