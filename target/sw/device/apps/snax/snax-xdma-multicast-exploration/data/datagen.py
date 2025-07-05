#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import re

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_scalar_define, format_vector_definition  # noqa E402

np.random.seed(320)

# Add stdint.h header


def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_random_data(**kwargs)
    emit_str += emit_multicast_pointers(**kwargs)
    return "\n\n".join(emit_str)


def emit_random_data(**kwargs):
    data_size = kwargs["size"]
    padded_data_size = (data_size + 7) // 8 * 8

    data = np.zeros((padded_data_size), dtype=np.uint64)
    data[:data_size] = np.random.randint(
        low=0, high=1 << 8, size=(data_size), dtype=np.uint64)

    # Emit data
    emit_str = [
        format_scalar_definition(
            "uint32_t",
            "data_size",
            data.size)]
    emit_str += [format_vector_definition("uint8_t",
                                          "data", data)]
    return emit_str

def emit_multicast_pointers(**kwargs):
    noc_x_size = kwargs["noc_dimensions"][0]
    noc_y_size = kwargs["noc_dimensions"][1]
    pointers = []
    for i in range(noc_x_size):
        for j in range(noc_y_size):
            if i % 2:
                j = noc_y_size - 1 - j
            pointers.append(i * noc_y_size + j)
    
    pointers = pointers[1:]
    
    if kwargs["multicast_num"] > len(pointers):
        raise ValueError(
            f"Multicast number {kwargs['multicast_num']} exceeds the number of pointers {len(pointers)}.")
    else:
        # Randomly drop elements to match multicast_num
        selected_indices = np.sort(
            np.random.choice(
            len(pointers),
            size=kwargs["multicast_num"],
            replace=False
            )
        )
        pointers = [pointers[i] for i in selected_indices]

    emit_str = [
        format_scalar_define(
            "MULTICAST_NUM",
            len(pointers))
    ]

    emit_str += [format_vector_definition(
        "uint8_t",
        "multicast_pointers_optimized",
        np.array(pointers, dtype=np.uint8))
    ]

    # Randomly permute the pointers before passing to the function
    pointers = np.random.permutation(pointers)

    emit_str += [format_vector_definition(
        "uint8_t",
        "multicast_pointers_randomized",
        np.array(pointers, dtype=np.uint8))
    ]

    return emit_str

def main():
    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generating data for kernels")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    # Emit header file
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
