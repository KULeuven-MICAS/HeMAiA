#!/usr/bin/env python3
"""MoE data generator — produces weight/input/golden data for testing.

Generates:
  - Input activation (INT8)
  - Per-expert weight matrices (INT8)
  - Router logits (FP32, one per expert)
  - CERF group ID mapping (uint8_t, one per expert)
  - Golden output for verification
"""

import numpy as np
import struct
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from sim_golden_models import block_gemm_golden_model  # noqa: E402


def generate_moe_data(params):
    """Generate all data arrays for the MoE workload.

    Args:
        params: dict with keys M, K, N, meshRow, tileSize, meshCol,
                num_experts, top_k, arrayShapeIdx

    Returns:
        dict of {name: numpy_array} for each data buffer
    """
    num_experts = params['num_experts']
    top_k = params['top_k']
    M = params['M']
    K = params['K']
    N = params['N']
    meshRow = params['meshRow']
    tileSize = params['tileSize']
    meshCol = params['meshCol']

    rng = np.random.default_rng(seed=42)

    data = {}

    # Input activation in block layout (shared across all experts)
    data['input_A'] = rng.integers(
        -8, 8, size=(M, K, meshRow, tileSize), dtype=np.int8
    ).reshape(-1)

    # Per-expert weights in block layout
    for i in range(num_experts):
        data[f'expert_{i}_B'] = rng.integers(
            -8, 8, size=(K, N, tileSize, meshCol), dtype=np.int8
        ).reshape(-1)

    # Router logits (FP32) — simulate that experts 0 and 2 have highest logits
    logits = np.array([-1.5, 0.3, -0.8, 2.1], dtype=np.float32)[:num_experts]
    data['router_logits'] = logits

    # Golden softmax output
    exp_logits = np.exp(logits - np.max(logits))
    data['softmax_golden'] = (exp_logits / np.sum(exp_logits)).astype(np.float32)

    # CERF group IDs (expert i -> CERF group i)
    data['cerf_group_ids'] = np.arange(num_experts, dtype=np.uint8)

    # Golden output: use block_gemm_golden_model (matches HW accelerator)
    for i in range(num_experts):
        C = np.zeros(M * N * meshRow * meshCol, dtype=np.int32)
        D = block_gemm_golden_model(
            M, K, N, meshRow, tileSize, meshCol,
            data['input_A'], data[f'expert_{i}_B'],
            0, 0, C,
        )
        data[f'expert_{i}_D_golden'] = D

    return data


def emit_header_file(output_path, params, data):
    """Emit a C header file with all MoE data arrays.

    Args:
        output_path: Path to output .h file
        params: workload parameters
        data: dict from generate_moe_data()
    """
    num_experts = params['num_experts']
    M = params['M']
    K = params['K']
    N = params['N']
    meshRow = params['meshRow']
    tileSize = params['tileSize']
    meshCol = params['meshCol']

    A_size = M * K * meshRow * tileSize  # int8
    B_size = K * N * tileSize * meshCol  # int8
    D_size = M * N * meshRow * meshCol   # int32 elements

    with open(output_path, 'w') as f:
        f.write(f"// Auto-generated MoE data — do not edit\n")
        f.write(f"// Experts: {num_experts}, Top-K: {params['top_k']}\n")
        f.write(f"// GEMM per expert: ({M*meshRow}x{K*tileSize}) * ({K*tileSize}x{N*meshCol})\n")
        f.write(f"#pragma once\n")
        f.write(f"#include <stdint.h>\n\n")

        # Dimensions
        f.write(f"#define MOE_NUM_EXPERTS {num_experts}\n")
        f.write(f"#define MOE_TOP_K {params['top_k']}\n")
        f.write(f"#define MOE_A_SIZE {A_size}\n")
        f.write(f"#define MOE_B_SIZE {B_size}\n")
        f.write(f"#define MOE_D_SIZE_BYTES {D_size * 4}\n")
        f.write(f"#define MOE_D_SIZE_ELEMS {D_size}\n\n")

        # Input activation
        _emit_array(f, "input_A", data['input_A'].flatten(), "int8_t")

        # Per-expert weights
        for i in range(num_experts):
            _emit_array(f, f"expert_{i}_B", data[f'expert_{i}_B'].flatten(), "int8_t")

        # Per-expert golden outputs
        for i in range(num_experts):
            _emit_array(f, f"expert_{i}_D_golden", data[f'expert_{i}_D_golden'].flatten(), "int32_t")

        # Router logits (FP32)
        logits = data['router_logits']
        f.write(f"// Router logits (FP32)\n")
        f.write(f"float router_logits[{len(logits)}] __attribute__((section(\".wide_spm\"))) = {{\n")
        for i, val in enumerate(logits):
            f.write(f"    {float(val):.4f}f{',' if i < len(logits)-1 else ''}\n")
        f.write(f"}};\n\n")

        # Golden softmax output
        sm = data['softmax_golden']
        f.write(f"// Golden softmax output (FP32)\n")
        f.write(f"float softmax_golden[{len(sm)}] __attribute__((section(\".wide_spm\"))) = {{\n")
        for i, val in enumerate(sm):
            f.write(f"    {float(val):.8f}f{',' if i < len(sm)-1 else ''}\n")
        f.write(f"}};\n\n")

        # CERF group IDs
        ids = data['cerf_group_ids']
        f.write(f"uint8_t cerf_group_ids[{len(ids)}] __attribute__((section(\".wide_spm\"))) = {{")
        f.write(", ".join(str(x) for x in ids))
        f.write(f"}};\n\n")


def _emit_array(f, name, arr, dtype):
    """Emit a C array to file."""
    f.write(f"{dtype} {name}[{len(arr)}] __attribute__((section(\".wide_spm\"))) = {{\n")
    for i in range(0, len(arr), 16):
        chunk = arr[i:i+16]
        vals = ", ".join(str(int(x)) for x in chunk)
        f.write(f"    {vals},\n")
    f.write(f"}};\n\n")
