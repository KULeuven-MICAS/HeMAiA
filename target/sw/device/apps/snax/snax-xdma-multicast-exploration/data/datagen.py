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

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_scalar_define, format_vector_definition  # noqa E402

np.random.seed(320)

# Optimise pointer order for multicast
def optimise_pointer_order(pointers, noc_x, noc_y):
    # helper ------------------------------------------------------------
    def coords(idx): return divmod(idx, noc_y)

    def xy_edges(a, b):
        ax, ay = coords(a);  bx, by = coords(b)
        edges = []
        step = 1 if bx > ax else -1
        for x in range(ax, bx, step):
            edges.append(((x, ay), (x+step, ay)))
        step = 1 if by > ay else -1
        for y in range(ay, by, step):
            edges.append(((bx, y), (bx, y+step)))
        return edges

    # greedy ------------------------------------------------------------
    remaining = set(pointers)
    start = min(remaining, key=lambda p: p)        # closest to core 0
    order = [start]; remaining.remove(start)
    used = set(xy_edges(0, start))                 # path from core to first

    while remaining:
        best = None; best_hops = noc_x + noc_y     # upper bound
        for cand in remaining:
            path = xy_edges(order[-1], cand)
            hops = len(path)
            if not used.intersection(path) and hops < best_hops:
                best, best_hops, best_path = cand, hops, path
        if best is None:                           # no edge-disjoint choice
            # fall back to pure shortest hop
            best = min(remaining, key=lambda c: len(xy_edges(order[-1], c)))
            best_path = xy_edges(order[-1], best)
        order.append(best)
        used.update(best_path)
        remaining.remove(best)
    return order

def total_xy_hops(order, noc_x, noc_y):
    """
    Return the total number of XY-routing hops needed to visit
    all destinations in *order*, starting from node 0.

    Parameters
    ----------
    order : Sequence[int]
        Ordered list of mesh node indices (e.g. multicast_pointers_optimized).
    noc_x, noc_y : int
        Mesh dimensions: x-size (rows) and y-size (columns).

    Returns
    -------
    int
        Sum of hops along the entire traversal 0 → order[0] → order[1] → …
        when every leg obeys deterministic XY routing
        (move in X first, then in Y).
    """

    def coords(idx):
        # row-major index → (x, y)
        return divmod(idx, noc_y)

    def xy_hops(a, b):
        ax, ay = coords(a)
        bx, by = coords(b)
        return abs(bx - ax) + abs(by - ay)   # Manhattan distance

    hops = 0
    current = 0          # start at core 0
    for dst in order:
        hops += xy_hops(current, dst)
        current = dst
    return hops

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
            # if i % 2:
            #     j = noc_y_size - 1 - j
            pointers.append(i * noc_y_size + j)
    
    pointers = pointers[1:]
    
    if kwargs["multicast_num"] > len(pointers):
        raise ValueError(
            f"Multicast number {kwargs['multicast_num']} exceeds the number of pointers {len(pointers)}.")
    else:
        # Randomly drop elements to match multicast_num
        pointers = np.random.choice(pointers, size=kwargs["multicast_num"], replace=False)
        pointers = np.sort(pointers)

    emit_str = [
        format_scalar_define(
            "MULTICAST_NUM",
            len(pointers))
    ]

    emit_str += [format_vector_definition(
        "uint8_t",
        "multicast_pointers",
        np.array(pointers, dtype=np.uint8))
    ]

    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "multicast_total_hops",
            total_xy_hops(pointers, noc_x_size, noc_y_size)),
    ]

    # Do the optimization on the pointer order
    pointers = optimise_pointer_order(pointers.tolist(), noc_x_size, noc_y_size)

    emit_str += [format_vector_definition(
        "uint8_t",
        "multicast_pointers_optimized",
        np.array(pointers, dtype=np.uint8))
    ]

    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "multicast_total_hops_optimized",
            total_xy_hops(pointers, noc_x_size, noc_y_size)),
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
