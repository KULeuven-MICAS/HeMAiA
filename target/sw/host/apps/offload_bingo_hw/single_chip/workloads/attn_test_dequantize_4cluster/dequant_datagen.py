#!/usr/bin/env python3
"""
Golden data generator for attn_test_dequantize.
Tests __host_bingo_kernel_int32_dequantize in isolation.

Generates random INT32 data (simulating GEMM output), a known combined_scale,
and the expected FP32 dequantized output for verification.
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
    d_head = kwargs["d_head"]
    num_elements = seq_len * d_head

    # Generate random INT32 D array (simulating GEMM output)
    int32_D = np.random.randint(-10000, 10000, size=num_elements).astype(np.int32)

    # Define combined_scale (arbitrary known value for testing)
    # Must be a numpy float32 to match hardware arithmetic exactly.
    combined_scale = np.float32(0.0001)

    # Compute golden FP32 output — mirror HW kernel arithmetic exactly:
    #   1. int32 -> fp32 (vfcvt_f_x_v_f32m1)
    #   2. fp32 * fp32 scale (vfmul_vf_f32m1) -> fp32
    # Using fp64 intermediate would produce different LSBs due to different rounding.
    fp32_input = int32_D.astype(np.float32)
    golden_fp32_D = (fp32_input * combined_scale).astype(np.float32)

    lines.append(f"// Dequantize test: {num_elements} elements, scale={combined_scale}")
    lines.append(format_vector_definition("int32_t", "int32_D_input", int32_D))
    lines.append(format_vector_definition("float", "golden_fp32_D", golden_fp32_D))
    # NOTE: declare as 1-element array so array-to-pointer decay gives the
    # address when passed as scale_addr. A scalar `float combined_scale = ...`
    # would be cast by value (truncated to 0) at the kernel arg assignment.
    lines.append(f"float combined_scale[1] __attribute__((aligned(8))) = {{ {float(combined_scale):.10e}f }};")
    lines.append(format_scalar_definition("uint32_t", "num_elements", num_elements))

    return "\n\n".join(lines) + "\n"
