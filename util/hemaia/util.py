#!/usr/bin/env python3
# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>

from mako.lookup import TemplateLookup
from mako.template import Template
from jsonref import JsonRef
import hjson
import argparse
import os


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
        description="The util collection to retrieve information from the hemaia json file"
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="target/rtl/cfg/occamy_cfg/lru.hjson",
        help="Path to the hemaia json file",
    )

    parser.add_argument(
        "--print_clusters",
        action="store_true",
        help="Print out the containing cluster names",
    )

    parsed_args = parser.parse_args()

    # Parse the occamy_cfg file
    occamy_cfg = get_config(parsed_args.cfg_path)
    if 'clusters' in occamy_cfg:
        clusters = occamy_cfg['clusters']
        cluster_cfgs = []
        for cluster in clusters:
            cluster_cfg_path = os.path.dirname(parsed_args.cfg_path) + "/../cluster_cfg/" + cluster + ".hjson"
            cluster_cfgs.append(get_config(cluster_cfg_path))
    else:
        raise Exception("No clusters found in the hemaia json file")
    
    if cluster_cfgs.__len__() == 0:
        raise Exception("The number of cluster is 0")
            
    # The remaining part is related to different functions
    # Available variables:
    # - occamy_cfg: The main configuration file
    # - cluster_cfgs: The parsed cluster configurations in a list

    if parsed_args.print_clusters:
        for cluster_cfg in cluster_cfgs:
            print(cluster_cfg['cluster']['name'] + " ", end="")
        print()
        return

if __name__ == "__main__":
    hemaia_util()
