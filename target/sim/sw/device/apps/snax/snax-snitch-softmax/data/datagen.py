#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Generate inputs and bit-exact golden data for fixed-point Snitch softmax."""

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/")
)
from data_utils import format_vector_definition  # noqa: E402


UINT32_MAX = (1 << 32) - 1
LUT_SIZE = 256  # int8 max - int8 min spans [0, 255]
TCDM_ALIGNMENT = 64


def align_up(value, alignment=TCDM_ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def validate_params(param):
    rows = int(param["softmax_rows"])
    cols = int(param["softmax_cols"])
    input_frac_bits = int(param["input_frac_bits"])
    exp_frac_bits = int(param["exp_frac_bits"])
    output_scale = int(param["output_scale"])
    input_min = int(param["input_min"])
    input_max = int(param["input_max"])

    if rows <= 0 or cols <= 0:
        raise ValueError("softmax_rows and softmax_cols must be positive")
    if not 0 <= input_frac_bits <= 15:
        raise ValueError("input_frac_bits must be in [0, 15]")
    if not 1 <= exp_frac_bits <= 15:
        raise ValueError("exp_frac_bits must be in [1, 15] for a uint16_t LUT")
    if not 1 <= output_scale <= 255:
        raise ValueError("output_scale must be in [1, 255] for uint8_t output")
    if not -128 <= input_min <= input_max <= 127:
        raise ValueError("input_min/input_max must fit in int8_t")
    if cols * (1 << exp_frac_bits) > UINT32_MAX:
        raise ValueError("softmax_cols is too large for the uint32_t exponent sum")


def generate_input(param):
    rows = int(param["softmax_rows"])
    cols = int(param["softmax_cols"])
    input_min = int(param["input_min"])
    input_max = int(param["input_max"])
    rng = np.random.default_rng(int(param["seed"]))

    values = rng.integers(
        input_min, input_max + 1, size=(rows, cols), dtype=np.int16
    ).astype(np.int8)

    # Deterministic corner cases in addition to random rows.
    if rows >= 1:
        values[0, :] = 0  # uniform distribution
    if rows >= 2:
        values[1, :] = input_min
        values[1, 0] = input_max  # strongly peaked distribution
    if rows >= 3:
        values[2, :] = np.linspace(input_min, input_max, cols, dtype=np.int16)

    return values


def generate_exp_lut(input_frac_bits, exp_frac_bits):
    delta = np.arange(LUT_SIZE, dtype=np.float64)
    real_delta = delta / float(1 << input_frac_bits)
    lut = np.rint(np.exp(-real_delta) * (1 << exp_frac_bits))
    return np.clip(lut, 0, (1 << 16) - 1).astype(np.uint16)


def fixed_softmax_model(values, exp_lut, output_scale):
    rows, cols = values.shape
    output = np.empty((rows, cols), dtype=np.uint8)

    for row in range(rows):
        row_i16 = values[row].astype(np.int16)
        delta = int(np.max(row_i16)) - row_i16
        exp_values = exp_lut[delta].astype(np.uint32)
        exp_sum = int(np.sum(exp_values, dtype=np.uint64))
        if exp_sum == 0:
            raise RuntimeError("invalid zero exponent sum")

        # Round to the nearest output quantum. This intentionally does not force
        # the independently rounded row entries to sum to output_scale.
        numerator = exp_values.astype(np.uint64) * output_scale + exp_sum // 2
        output[row] = (numerator // exp_sum).astype(np.uint8)

    return output


def floating_reference(values, input_frac_bits, output_scale):
    real_values = values.astype(np.float64) / float(1 << input_frac_bits)
    real_values -= np.max(real_values, axis=1, keepdims=True)
    probabilities = np.exp(real_values)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return np.rint(probabilities * output_scale).astype(np.uint8)


def emit_header(param):
    validate_params(param)

    rows = int(param["softmax_rows"])
    cols = int(param["softmax_cols"])
    elements = rows * cols
    input_frac_bits = int(param["input_frac_bits"])
    exp_frac_bits = int(param["exp_frac_bits"])
    output_scale = int(param["output_scale"])

    values = generate_input(param)
    exp_lut = generate_exp_lut(input_frac_bits, exp_frac_bits)
    golden = fixed_softmax_model(values, exp_lut, output_scale)
    reference = floating_reference(values, input_frac_bits, output_scale)
    expected_max_error = int(
        np.max(np.abs(golden.astype(np.int16) - reference.astype(np.int16)))
    )

    input_offset = 0
    output_offset = align_up(input_offset + elements)
    golden_offset = align_up(output_offset + elements)
    reference_offset = align_up(golden_offset + elements)
    lut_offset = align_up(reference_offset + elements)
    error_offset = align_up(lut_offset + LUT_SIZE * np.dtype(np.uint16).itemsize)
    tcdm_bytes = error_offset + np.dtype(np.uint32).itemsize

    defines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define SOFTMAX_ROWS {rows}",
        f"#define SOFTMAX_COLS {cols}",
        f"#define SOFTMAX_ELEMENTS {elements}",
        f"#define SOFTMAX_INPUT_FRAC_BITS {input_frac_bits}",
        f"#define SOFTMAX_EXP_FRAC_BITS {exp_frac_bits}",
        f"#define SOFTMAX_OUTPUT_SCALE {output_scale}",
        f"#define SOFTMAX_REFERENCE_TOLERANCE {int(param['reference_tolerance'])}",
        f"#define SOFTMAX_EXPECTED_MAX_ERROR {expected_max_error}",
        f"#define SOFTMAX_TARGET_CLUSTER {int(param['target_cluster'])}",
        f"#define SOFTMAX_LUT_SIZE {LUT_SIZE}",
        "",
        f"#define SOFTMAX_INPUT_OFFSET {input_offset}",
        f"#define SOFTMAX_OUTPUT_OFFSET {output_offset}",
        f"#define SOFTMAX_GOLDEN_OFFSET {golden_offset}",
        f"#define SOFTMAX_REFERENCE_OFFSET {reference_offset}",
        f"#define SOFTMAX_LUT_OFFSET {lut_offset}",
        f"#define SOFTMAX_ERROR_OFFSET {error_offset}",
        f"#define SOFTMAX_TCDM_BYTES {tcdm_bytes}",
        "",
    ]

    arrays = [
        format_vector_definition("int8_t", "softmax_input", values.reshape(-1)),
        format_vector_definition("uint16_t", "softmax_exp_lut", exp_lut),
        format_vector_definition("uint8_t", "softmax_golden", golden.reshape(-1)),
        format_vector_definition(
            "uint8_t", "softmax_reference", reference.reshape(-1)
        ),
    ]

    return "\n".join(defines) + "\n\n" + "\n\n".join(arrays) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a fixed-point softmax test for Snitch cores"
    )
    parser.add_argument(
        "-c", "--cfg", type=pathlib.Path, required=True, help="HJSON parameter file"
    )
    args = parser.parse_args()

    with args.cfg.open() as config_file:
        param = hjson.loads(config_file.read())

    print(emit_header(param), end="")


if __name__ == "__main__":
    main()
