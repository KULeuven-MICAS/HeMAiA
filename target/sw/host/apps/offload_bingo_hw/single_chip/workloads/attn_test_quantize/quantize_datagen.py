#!/usr/bin/env python3
"""
Golden data generator for attn_test_quantize.
Tests __host_bingo_kernel_fp32_quantize in isolation.

Generates FP32 input X, quantizes to INT8 with symmetric per-tensor quantization,
and emits the golden INT8 output for verification.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

np.random.seed(42)


def emit_header_file(**kwargs):
    lines = ["#include <stdint.h>"]

    seq_len = kwargs["seq_len"]
    d_model = kwargs["d_model"]
    num_elements = seq_len * d_model

    # Generate FP32 input X, shape (seq_len, d_model), uniform(-1, 1)
    X = np.random.uniform(-1.0, 1.0, size=(seq_len, d_model)).astype(np.float32)

    # Quantize: scale = max(abs(X)) / 127; int8_X = clip(round(X / scale), -128, 127)
    abs_max = float(np.max(np.abs(X)))
    if abs_max < 1e-10:
        abs_max = 1e-10
    scale = abs_max / 127.0
    golden_int8_X = np.clip(np.round(X.reshape(-1) / scale), -128, 127).astype(np.int8)

    lines.append(f"// Quantize test: {seq_len}x{d_model} = {num_elements} elements")
    lines.append(format_vector_definition("float", "fp32_X", X.reshape(-1)))
    lines.append(format_vector_definition("int8_t", "golden_int8_X", golden_int8_X))
    lines.append(format_scalar_definition("uint32_t", "X_num_elements", num_elements))

    return "\n\n".join(lines) + "\n"
