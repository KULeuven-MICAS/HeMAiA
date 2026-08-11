#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for the MonoidJunction (nonlinear online-softmax moment-merge) ChainGather.
# Each cluster owns ONE (m, l) softmax normalizer pair. The collector gathers all clusters'
# pairs, folding them in-fabric along the chain with the MonoidJunction MOMENT combine
#   (m1,l1) (+) (m2,l2) = (max(m1,m2), l1*exp(m1-m*) + l2*exp(m2-m*))
# so the golden is the online-softmax merge over all clusters.
#
# MonoidJunction PAIRED layout (MOMENT, pairSlots = dataWidth/(2*accWidth) = 8): field0 (m) at
# lane k, field1 (l) at lane pairSlots+k. With nValid=1 only slot 0 is live => m at lane 0,
# l at lane 8; the fold result lands m* at lane 0, l* at lane 8.

import argparse
import pathlib
import hjson
import numpy as np

# Same operands as the Gate-A moment-merge, so results are comparable. m* = 3.0, l* ~= 1.193497.
M_VALS = [1.0, 3.0, 2.0, -1.0]
L_VALS = [2.0, 0.5, 1.0, 3.0]
PAIRSLOTS = 8   # dataWidth/(2*accWidth) = 512/64


def _f(v):
    # repr(float) always carries a decimal point / exponent, so "3.0f" not the invalid "3f",
    # and enough digits to round-trip the exact fp32 value.
    return repr(float(v)) + "f"


def emit_header_file(**kwargs):
    num_clusters = kwargs["num_clusters"]
    partial_elems = kwargs["partial_elems"]   # 16 fp32 = one 64 B beat
    assert num_clusters <= len(M_VALS)

    m = np.array(M_VALS[:num_clusters], dtype=np.float64)
    l = np.array(L_VALS[:num_clusters], dtype=np.float64)
    m_star = float(np.max(m))
    l_star = float(np.sum(l * np.exp(m - m_star)))

    # Per-cluster beat: m at lane 0, l at lane PAIRSLOTS, rest 0 (masked to identity by nValid=1).
    data = np.zeros((num_clusters, partial_elems), dtype=np.float32)
    for i in range(num_clusters):
        data[i, 0] = np.float32(M_VALS[i])
        data[i, PAIRSLOTS] = np.float32(L_VALS[i])
    golden = np.zeros((partial_elems,), dtype=np.float32)
    golden[0] = np.float32(m_star)
    golden[PAIRSLOTS] = np.float32(l_star)

    # P=2 cross-die golden: collector (CHIPS idx 0 = 0x00) merged with adjacent source
    # (CHIPS idx 2 = 0x10). A length-2 chain has no middle node (only the working root->head grant).
    m2 = np.array([M_VALS[0], M_VALS[2]], dtype=np.float64)
    l2 = np.array([L_VALS[0], L_VALS[2]], dtype=np.float64)
    m_star2 = float(np.max(m2))
    l_star2 = float(np.sum(l2 * np.exp(m2 - m_star2)))
    golden_p2 = np.zeros((partial_elems,), dtype=np.float32)
    golden_p2[0] = np.float32(m_star2)
    golden_p2[PAIRSLOTS] = np.float32(l_star2)

    def fvec(name, arr):
        vals = ", ".join(_f(v) for v in arr.reshape(-1))
        return f"float {name}[{arr.size}] = {{{vals}}};"

    lines = [
        "#include <stdint.h>",
        f"#define NUM_CLUSTERS {num_clusters}",
        f"#define PARTIAL_ELEMS {partial_elems}",
        f"#define MOMENT_M_LANE 0",
        f"#define MOMENT_L_LANE {PAIRSLOTS}",
        fvec("chain_gather_data", data),
        fvec("chain_gather_golden", golden),
        fvec("chain_gather_golden_p2", golden_p2),
    ]
    return "\n\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MonoidJunction ChainGather data")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
