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
    return emit_ksplit_gemm_header_file("gemm_ksplit_4chiplet_1cluster", **kwargs)
