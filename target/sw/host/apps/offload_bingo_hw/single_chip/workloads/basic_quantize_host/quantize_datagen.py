#!/usr/bin/env python3
"""
Golden data generator for basic_quantize_host.
Tests the FP32->INT8 quantize host kernel in isolation.

Generates FP32 input X, quantizes to INT8 with symmetric per-tensor quantization,
and emits the golden INT8 output for verification.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
# __usg__ grouped util/sim: make common/gemm/xdma/ara importable
import os as _usg_os, sys as _usg_sys
for _usg_p in [p for p in list(_usg_sys.path) if str(p).rstrip('/').endswith('util/sim')]:
    for _usg_s in ('common', 'gemm', 'xdma', 'ara'):
        _usg_sub = _usg_os.path.join(_usg_p, _usg_s)
        if _usg_sub not in _usg_sys.path:
            _usg_sys.path.append(_usg_sub)
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

np.random.seed(42)


def emit_header_file(**kwargs):
    lines = ["#include <stdint.h>"]

    num_elements = kwargs["num_elements"]

    # Generate FP32 input X, uniform(-1, 1)
    X = np.random.uniform(-1.0, 1.0, size=num_elements).astype(np.float32)

    # Quantize: scale = max(abs(X)) / 127; int8_X = clip(round(X / scale), -128, 127)
    abs_max = float(np.max(np.abs(X)))
    if abs_max < 1e-10:
        abs_max = 1e-10
    scale = abs_max / 127.0
    golden_int8_X = np.clip(np.round(X.reshape(-1) / scale), -128, 127).astype(np.int8)

    lines.append(f"// Quantize test: {num_elements} elements")
    lines.append(format_vector_definition("float", "fp32_X", X))
    lines.append(format_vector_definition("int8_t", "golden_int8_X", golden_int8_X))
    lines.append(format_scalar_definition("uint32_t", "X_num_elements", num_elements))

    return "\n\n".join(lines) + "\n"
