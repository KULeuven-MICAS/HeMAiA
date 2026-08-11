#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the ChainGather bring-up (single-chip 4-cluster).
# Each cluster owns one 64 B partial (PARTIAL_ELEMS fp32). The collector (cluster 0)
# gathers the other clusters' partials, folding them element-wise (ElementwiseJunction
# ADD, FP32) plus its own, so the golden is the element-wise sum over all clusters.
# Values are small integers so the fp32 sum is exact and the on-device check can use ==.

import argparse
import pathlib
import hjson
import numpy as np

np.random.seed(320)


def emit_header_file(**kwargs):
    num_clusters = kwargs["num_clusters"]
    partial_elems = kwargs["partial_elems"]

    # Small integer-valued fp32 partials -> exact fp32 sums.
    data = np.random.randint(0, 64, size=(num_clusters, partial_elems)).astype(np.float32)
    golden = np.sum(data, axis=0).astype(np.float32)
    # P=2 cross-die golden: collector (CHIPS idx 0 = 0x00) + adjacent source (CHIPS idx 2 = 0x10).
    # A length-2 gather chain has NO middle node, so it exercises only the (working) root->head grant.
    golden_p2 = (data[0] + data[2]).astype(np.float32)

    def fvec(name, arr):
        vals = ", ".join(f"{v:.1f}f" for v in arr.reshape(-1))
        return f"float {name}[{arr.size}] = {{{vals}}};"

    lines = [
        "#include <stdint.h>",
        f"#define NUM_CLUSTERS {num_clusters}",
        f"#define PARTIAL_ELEMS {partial_elems}",
        fvec("chain_gather_data", data),
        fvec("chain_gather_golden", golden),
        fvec("chain_gather_golden_p2", golden_p2),
    ]
    return "\n\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generating data for ChainGather")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True,
                        help="Select param config file")
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
