#!/usr/bin/env python3
# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>
# Ryan Antonio <ryan.antonio@kuleuven.be>

# TODO: commenting some so that they will be added later
# from mako.lookup import TemplateLookup
# from mako.template import Template
from jsonref import JsonRef
import hjson
import argparse
import os
import subprocess
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from resolve_cluster_cfg import bingo_scheduler_enabled  # noqa: E402


# Extract json file
def get_config(cfg_path: str):
    with open(cfg_path, "r") as jsonf:
        srcfull = jsonf.read()

    # Format hjson file
    cfg = hjson.loads(srcfull, use_decimal=True)
    cfg = JsonRef.replace_refs(cfg)
    return cfg

def hemaia_util():
    # Parse all arguments
    parser = argparse.ArgumentParser(
        description="The util collection to \
            retrieve information from the hemaia json file"
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="../../target/rtl/cfg/lru.hjson",
        help="Path to the hemaia json file",
    )

    parser.add_argument(
        "--print_clusters",
        action="store_true",
        help="Print out the containing cluster names",
    )

    parser.add_argument(
        "--get_bender_targets",
        action="store_true",
        help="Get the bender targets for the whole system",
    )

    parser.add_argument("--outdir",
                        "-o",
                        type=pathlib.Path,
                        help="Target directory.")

    parser.add_argument("--snax-path",
                        type=pathlib.Path,
                        help="Path to the SNAX cluster repo.")

    parser.add_argument("--cluster-flist",
                        action="store_true",
                        help="Flag for generating for generating \
                            cluster specific flists only.")

    parsed_args = parser.parse_args()

    # Parse the occamy_cfg file
    occamy_cfg = get_config(parsed_args.cfg_path)
    snax_path = parsed_args.snax_path or pathlib.Path(__file__).resolve().parents[2] / "deps/snitch_cluster"
    cluster_cfg_paths = [
        snax_path / "target/snitch_cluster/cfg" / f"{cluster}.hjson"
        for cluster in occamy_cfg.get("clusters", [])
    ]
    cluster_cfgs = [get_config(path) for path in cluster_cfg_paths]
    occamy_cfg['clusters'] = cluster_cfgs

    # The remaining part is the region for the util functions
    # Available variables:
    # - occamy_cfg: The main configuration file, with parsed cluster configurations
    # - cluster_cfg_paths: The paths to the cluster configurations in a list

    # For printing out the cluster names and generate targets
    if parsed_args.print_clusters:
        for cluster_cfg in occamy_cfg['clusters']:
            print(cluster_cfg['cluster']['name'] + " ", end="")
        print()
        return
    
    # For generating filelists for each cluster
    # These filelists are specific for synthesis only
    if parsed_args.cluster_flist:

        def generate_cluster_syn_flist(cfg, snax_path, outdir):
            config_name = os.path.splitext(os.path.basename(os.path.normpath(cfg)))[0]
            subprocess.call(f"make -C {snax_path}/target/snitch_cluster \
                            CFG_OVERRIDE={cfg} \
                            MEM_TYPE=exclude_tcsram \
                            SYN_FLIST={config_name}.tcl \
                            SYN_BUILDDIR={outdir} \
                            gen-syn", shell=True)
        
        print("Generate filelist for each cluster only.")
        for cluster_cfg_path in cluster_cfg_paths:
            generate_cluster_syn_flist(cluster_cfg_path,
                                    snax_path,
                                    parsed_args.outdir)
        return
    
    # For generating all bender targets
    if parsed_args.get_bender_targets:
        def get_bender_targets(cfg):
            targets = []
            # If cfg is dictionary, then first check if it has bender_target, then iterate over the rest
            if isinstance(cfg, dict):
                for name,content in cfg.items():
                    if name == "bender_target":
                        targets.extend(content)
                    else:
                        targets.extend(get_bender_targets(content))
            # If cfg is a list, then iterate over the list
            elif isinstance(cfg, list):
                for item in cfg:
                    targets.extend(get_bender_targets(item))
            return list(dict.fromkeys(targets))
        
        bender_targets = get_bender_targets(occamy_cfg)
        if cluster_cfgs:
            if "hemaia_clusters" not in bender_targets:
                bender_targets.append("hemaia_clusters")
        else:
            bender_targets = [target for target in bender_targets
                              if target not in ("hemaia_clusters", "occamy_noc")]
        if cluster_cfgs or bingo_scheduler_enabled(occamy_cfg):
            bender_targets.append("hemaia_bingo_regs")
        if bingo_scheduler_enabled(occamy_cfg):
            bender_targets.append("hemaia_bingo_scheduler")
        for i in bender_targets:
            print(" -t "+ i,end="")
        print()
        return


if __name__ == "__main__":
    hemaia_util()
