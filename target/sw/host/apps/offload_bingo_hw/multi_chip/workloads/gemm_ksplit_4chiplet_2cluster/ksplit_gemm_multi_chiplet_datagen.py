#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""Golden-data wrapper for gemm_ksplit_4chiplet_2cluster."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)
from gemm_sim_utils import emit_ksplit_gemm_header_file, get_gemm_mesh_dims  # noqa E402


WORKLOAD_NAME = "gemm_ksplit_4chiplet_2cluster"
EXPECTED_K_SPLIT = 8
CLUSTERS_PER_CHIPLET = 2


def emit_header_file(**kwargs):
    content = emit_ksplit_gemm_header_file(
        WORKLOAD_NAME,
        expected_k_split=EXPECTED_K_SPLIT,
        **kwargs,
    )

    mesh_row, _, mesh_col = get_gemm_mesh_dims(kwargs)
    d_num_elements = kwargs["M"] * kwargs["N"] * mesh_row * mesh_col

    local_l3_buffers = [
        "",
        "// Per-chiplet L3 staging slots for remote partial-D pulls.",
        f"int32_t D_partial_local_l3[{CLUSTERS_PER_CHIPLET * d_num_elements}] "
        "__attribute__ ((aligned (64)));",
    ]

    return content + "\n".join(local_l3_buffers) + "\n"
