#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for the VersaCore GEMM test parameter schema.

``testing_workload_gen.py`` (which writes the workload CSV) and
``testing_launch.py`` (which reads it back to launch each test) must agree on the
exact ``params.hjson`` field order. Defining it once here keeps the writer and
reader from drifting; both derive their CSV column lists from this.
"""

PARAM_FIELDS = [
    "num_clusters",
    "array_shape",
    "data_type",
    "transposed_A",
    "transposed_B",
    "int4_a_enable",
    "int4_b_enable",
    "K",
    "N",
    "M",
    "addNonZeroC",
    "addZeroC",
    "accumPrevC",
    "quantization_enable",
    "int32tofp16_enable",
]
