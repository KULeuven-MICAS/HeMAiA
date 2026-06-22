#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""Golden-data wrapper for gemm_ksplit_4chiplet_1cluster."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)
from gemm_sim_utils import emit_ksplit_gemm_header_file, get_gemm_mesh_dims  # noqa E402


def emit_header_file(**kwargs):
    content = emit_ksplit_gemm_header_file("gemm_ksplit_4chiplet_1cluster", **kwargs)

    meshRow, _, meshCol = get_gemm_mesh_dims(kwargs)
    d_num_elements = kwargs["M"] * kwargs["N"] * meshRow * meshCol

    direct_l3_buffers = [
        "",
        "// Shared chiplet-0 L3 target for direct remote partial-D stores.",
        f'int32_t D_partial_remote_chip00_l3[{d_num_elements}] '
        '__attribute__ ((aligned (64)));',
    ]

    return content + "\n".join(direct_l3_buffers) + "\n"
