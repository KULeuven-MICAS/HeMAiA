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
from gemm_sim_utils import emit_ksplit_gemm_header_file  # noqa E402


def emit_header_file(**kwargs):
    content = emit_ksplit_gemm_header_file("gemm_ksplit_4chiplet_1cluster", **kwargs)

    array_shape = kwargs["array_shape"]
    data_type = kwargs.get("data_type", 0)
    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, _, meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]
    d_num_elements = kwargs["M"] * kwargs["N"] * meshRow * meshCol

    direct_l3_buffers = [
        "",
        "// Shared chiplet-0 L3 target for direct remote partial-D stores.",
        f'int32_t D_partial_remote_chip00_l3[{d_num_elements}] '
        '__attribute__ ((aligned (64)));',
    ]

    return content + "\n".join(direct_l3_buffers) + "\n"
