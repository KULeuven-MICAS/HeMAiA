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
import time

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_scalar_define, format_vector_definition  # noqa E402

np.random.seed((int(time.time() * 1e9) ^ os.getpid()) % (2**32))

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

# Optimise pointer order for multicast
def optimise_pointer_order_improved(pointers, noc_x, noc_y):
    """
    Given a collection of destination nodes on a 2-D mesh NoC, compute an order
    in which to visit all destinations such that the sum of XY-routing hops
    between consecutive destinations is minimised.  In addition, ensure that
    no physical mesh link (edge) is used more than once across the entire
    multicast route.  The starting destination is chosen to be the one
    numerically closest to the source (core 0), matching the previous greedy
    implementation.

    This routine attempts to build an optimal solution using the CP-SAT solver
    from Google's OR-Tools.  If OR-Tools is unavailable, or if the solver
    fails to find a solution within a reasonable time, it falls back to the
    original greedy heuristic.  The input and output signatures are kept
    identical to the previous function.

    Parameters
    ----------
    pointers : Sequence[int]
        The indices of mesh nodes to visit.  Indices are in row-major order.
    noc_x, noc_y : int
        The mesh dimensions (rows and columns).

    Returns
    -------
    list[int]
        A permutation of *pointers* representing the visitation order.
    """

    # -------------------------------------------------------------------
    # Helper functions local to the optimisation routine.  These mirror the
    # helpers used in the greedy implementation and are defined here to
    # avoid namespace pollution.
    def coords(idx):
        """Convert a row-major node index into (x, y) coordinates."""
        return divmod(idx, noc_y)

    def xy_edges(a, b):
        """
        Produce the ordered list of mesh edges traversed by deterministic
        XY-routing from node *a* to node *b*.  The returned list contains
        tuples of ((x0, y0), (x1, y1)) representing directed edges between
        neighbouring routers.  The order is always X-then-Y, which matches
        the behaviour of the existing simulator.
        """
        ax, ay = coords(a)
        bx, by = coords(b)
        edges = []
        # move along the X dimension first
        if bx != ax:
            step = 1 if bx > ax else -1
            for x in range(ax, bx, step):
                edges.append(((x, ay), (x + step, ay)))
        # then move along the Y dimension
        if by != ay:
            step = 1 if by > ay else -1
            for y in range(ay, by, step):
                edges.append(((bx, y), (bx, y + step)))
        return edges

    # If there are zero or one destinations there is nothing to optimise.
    if not pointers or len(pointers) <= 1:
        return list(pointers)

    # Identify the starting destination as the pointer closest to the source
    # (core 0).  In case of ties, choose the smallest index.
    start_pointer = min(pointers, key=lambda p: p)

    # Precompute the set of edges used by the initial path from the core to
    # the first destination.  These edges must never be reused later on.
    start_path_edges = set(xy_edges(0, start_pointer))

    # Attempt to import OR-Tools and use it to solve the optimisation.  If
    # anything fails, we fall back to the greedy heuristic defined below.
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        # OR-Tools is not available. Using greedy optimization
        cp_model = None

    # Only build and solve a CP-SAT model when OR-Tools is available.
    if cp_model is not None:
        # Create a new CP-SAT model.
        model = cp_model.CpModel()

        # Normalise the destination list into a sequence of indices 0..n-1.
        dest_list = list(pointers)
        n = len(dest_list)
        dest_index = {p: i for i, p in enumerate(dest_list)}
        s_idx = dest_index[start_pointer]

        # Compute the set of XY edges and hop counts for every ordered pair.
        # Skip pairs that would intersect the path from core 0 to the start.
        # path_edges[(i, j)] = set of directed edges used between dest_list[i] and dest_list[j]
        # path_cost[(i, j)] = Manhattan distance between dest_list[i] and dest_list[j]
        path_edges = {}
        path_cost = {}
        # Additionally build a map from each physical edge to the boolean
        # variables that could take it.
        edge_to_pairs = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a = dest_list[i]
                b = dest_list[j]
                edges = xy_edges(a, b)
                # Exclude pairs whose path reuses an edge from the core->start path.
                if set(edges) & start_path_edges:
                    continue
                path_edges[(i, j)] = set(edges)
                path_cost[(i, j)] = len(edges)
                # Update the reverse mapping to allow edge uniqueness constraints
                for e in edges:
                    edge_to_pairs.setdefault(e, []).append((i, j))

        # Create integer variables representing the order (position) of each destination.
        order_vars = [model.NewIntVar(0, n - 1, f'order_{i}') for i in range(n)]
        model.AddAllDifferent(order_vars)
        # Fix the starting destination to position 0.
        model.Add(order_vars[s_idx] == 0)

        # Boolean decision variables: x[i, j] == 1 if j is the successor of i.
        x_vars = {}
        for (i, j) in path_edges:
            x_vars[(i, j)] = model.NewBoolVar(f'x_{i}_{j}')
        # Link successor variables to order positions: x[i,j] => order_j == order_i + 1
        # and ensure that if a node has a successor then it cannot be the last.
        for (i, j), var in x_vars.items():
            # Enforce that taking arc i→j implies j's position is immediately after i's.
            model.Add(order_vars[j] == order_vars[i] + 1).OnlyEnforceIf(var)
            # Also ensure that i is not the last node when it has a successor.
            model.Add(order_vars[i] <= n - 2).OnlyEnforceIf(var)

        # Each destination except the starting one must have exactly one predecessor.
        for j in range(n):
            # Gather all incoming arcs for this node
            incoming = [x_vars[(i, j)] for i in range(n) if (i, j) in x_vars]
            if j == s_idx:
                # The start has no predecessors
                model.Add(sum(incoming) == 0)
            else:
                model.Add(sum(incoming) == 1)

        # Each destination must have exactly one successor, except the one that is last.
        # Introduce boolean indicators for the last position.
        is_last = [model.NewBoolVar(f'is_last_{i}') for i in range(n)]
        for i in range(n):
            # Define is_last[i] ↔ (order_vars[i] == n - 1)
            model.Add(order_vars[i] == n - 1).OnlyEnforceIf(is_last[i])
            model.Add(order_vars[i] != n - 1).OnlyEnforceIf(is_last[i].Not())
        # There must be exactly one last node.
        model.Add(sum(is_last) == 1)
        # Constrain outgoing arcs based on whether the node is last or not.
        for i in range(n):
            outgoing = [x_vars[(i, j)] for j in range(n) if (i, j) in x_vars]
            # If this node is marked as last, it should have no successor.
            model.Add(sum(outgoing) == 0).OnlyEnforceIf(is_last[i])
            # Otherwise it must have exactly one successor.
            model.Add(sum(outgoing) == 1).OnlyEnforceIf(is_last[i].Not())

        # Enforce that no physical mesh edge is used more than once across the route.
        for edge, pairs in edge_to_pairs.items():
            model.Add(sum(x_vars[p] for p in pairs) <= 1)

        # Objective: minimise the total hop count across all chosen arcs.
        objective_terms = []
        for (i, j), var in x_vars.items():
            objective_terms.append(path_cost[(i, j)] * var)
        model.Minimize(sum(objective_terms))

        # Create a solver and set a reasonable time limit.  The limit can be
        # adjusted if larger multicast sets are expected, but in practice
        # multicast counts tend to be small.
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        # Enable multithreading if available
        solver.parameters.num_search_workers = max(1, os.cpu_count() or 1)

        try:
            status = solver.Solve(model)
        except Exception:
            status = cp_model.UNKNOWN

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract the order by sorting destinations according to their
            # computed positions.  The positions are 0-based integers.
            ordered = [None] * n
            for i in range(n):
                pos = solver.Value(order_vars[i])
                ordered[pos] = dest_list[i]
            return ordered
        # If no solution was found, fall back to the greedy heuristic below.

    # -------------------------------------------------------------------
    # Greedy fallback: replicate the original heuristic behaviour when
    # optimisation cannot be performed.
    remaining = set(pointers)
    order = [start_pointer]
    remaining.remove(start_pointer)
    used = set(start_path_edges)
    # Continue selecting the closest edge-disjoint candidate until all
    # destinations have been placed.
    while remaining:
        best = None
        best_hops = noc_x + noc_y  # An upper bound on Manhattan distance
        best_path = None
        for cand in remaining:
            path = xy_edges(order[-1], cand)
            hops = len(path)
            # Choose the candidate with the smallest hop count that does not
            # intersect the set of already used edges.
            if not used.intersection(path) and hops < best_hops:
                best = cand
                best_hops = hops
                best_path = path
        if best is None:
            # If no edge-disjoint choice is possible, fall back to the
            # pure shortest path ignoring edge reuse.
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

# Compute broadcast hops by counting each XY-edge only once
def total_xy_hops_broadcast(dests, noc_x, noc_y):
    def coords(idx):
        return divmod(idx, noc_y)
    def xy_edges(a, b):
        ax, ay = coords(a)
        bx, by = coords(b)
        edges = []
        # X‐direction
        step = 1 if bx > ax else -1
        for x in range(ax, bx, step):
            edges.append(((x, ay), (x+step, ay)))
        # Y‐direction
        step = 1 if by > ay else -1
        for y in range(ay, by, step):
            edges.append(((bx, y), (bx, y+step)))
        return edges

    used_edges = set()
    for d in dests:
        used_edges.update(xy_edges(0, d))
    return len(used_edges)


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

    unicast_total_hops = sum(
        total_xy_hops([p], noc_x_size, noc_y_size) for p in pointers
    )
    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "unicast_total_hops",
            unicast_total_hops),
    ]

    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "multicast_total_hops",
            total_xy_hops(pointers, noc_x_size, noc_y_size)),
    ]

    # Do the optimization on the pointer order
    if kwargs["better_solver"] is True:
        pointers = optimise_pointer_order_improved(pointers.tolist(), noc_x_size, noc_y_size)
    else:
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

    # The total hops for broadcast
    broadcast_total_hops = total_xy_hops_broadcast(pointers,
                                                   noc_x_size,
                                                   noc_y_size)

    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "broadcast_total_hops",
            broadcast_total_hops
        )
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
