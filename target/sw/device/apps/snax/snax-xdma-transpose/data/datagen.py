#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import re

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

np.random.seed(320)

# Add stdint.h header


def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_transposer_data(**kwargs)
    return "\n\n".join(emit_str)


def emit_transposer_data(**kwargs):
    tile_width = None
    element_width = kwargs["BIT_WIDTH"]
    if element_width == 8 or element_width == 16:
        tile_width = 8
    else:
        raise ValueError(
            f"Invalid BIT_WIDTH: {element_width}, only 8 and 16 are supported")
    data_type = None
    if element_width == 8:
        data_type = "uint8_t"
    elif element_width == 16:
        data_type = "uint16_t"
    else:
        raise ValueError(
            f"Invalid BIT_WIDTH: {element_width}, only 8 and 16 are supported")
    transfer_per_transpose = None
    if element_width == 8:
        transfer_per_transpose = 1
    elif element_width == 16:
        transfer_per_transpose = 2
    else:
        raise ValueError(
            f"Invalid BIT_WIDTH: {element_width}, only 8 and 16 are supported")

    emit_str = []
    padded_M = None
    padded_N = None
    if kwargs["input_layout"] == "MN":
        padded_M = (kwargs["M"] + tile_width - 1) // tile_width * tile_width
        padded_N = (kwargs["N"] + tile_width - 1) // tile_width * tile_width
    else:
        match = re.search(r'MNM(\d+)N(\d+)', kwargs["input_layout"])
        if match:
            m, n = match.groups()
            m, n = int(m), int(n)
            padded_M = (kwargs["M"] + m - 1) // m * m
            padded_N = (kwargs["N"] + n - 1) // n * n
        else:
            raise ValueError(f"Invalid input layout: {kwargs['input_layout']}")

    matrix_data = np.zeros((padded_M, padded_N), dtype=np.uint64)
    matrix_data[:kwargs["M"], :kwargs["N"]] = np.random.randint(
        low=0, high=1 << element_width, size=(kwargs["M"], kwargs["N"]), dtype=np.uint64)
    input_matrix = matrix_data
    if kwargs["input_layout"] == "MN":
        input_matrix = input_matrix.ravel()
    else:
        match = re.search(r'MNM(\d+)N(\d+)', kwargs["input_layout"])
        if match:
            m, n = match.groups()
            m, n = int(m), int(n)
            if m % tile_width != 0 or n % tile_width != 0:
                raise ValueError(
                    f"Invalid input layout: {kwargs['input_layout']}. \
                        The tile size should be a multiple of {tile_width}.")
            input_matrix = input_matrix.reshape(
                input_matrix.shape[0] // m,
                m,
                input_matrix.shape[1] // n,
                n).swapaxes(
                1,
                2).ravel()
        else:
            raise ValueError(f"Invalid input layout: {kwargs['input_layout']}")

    # Emit input matrix
    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "matrix_size",
            matrix_data.size)]
    emit_str += [format_vector_definition(data_type,
                                          "input_matrix", input_matrix)]

    # Emit output matrix
    output_matrix = matrix_data
    if kwargs["enable_transpose"] is True:
        output_matrix = output_matrix.T
    if kwargs["output_layout"] == "MN":
        output_matrix = output_matrix.ravel()
    else:
        match = re.search(r'MNM(\d+)N(\d+)', kwargs["output_layout"])
        if match:
            m, n = match.groups()
            m, n = int(m), int(n)
            if m % tile_width != 0 or n % tile_width != 0:
                raise ValueError(
                    f"Invalid output layout: {kwargs['output_layout']}. \
                        The tile size should be a multiple of {tile_width}.")
            output_matrix = output_matrix.reshape(
                output_matrix.shape[0] // m,
                m,
                output_matrix.shape[1] // n,
                n).swapaxes(
                1,
                2).ravel()
        else:
            raise ValueError(
                f"Invalid output layout: {kwargs['output_layout']}")
    emit_str += [format_vector_definition(data_type,
                                          "golden_output_matrix",
                                          output_matrix)]

    # Emit the configuration for XDMA
    spatial_stride_src = None
    spatial_stride_dst = None
    temporal_strides_src = []
    temporal_strides_dst = []
    temporal_bounds_src = []
    temporal_bounds_dst = []

    # Input Side (Reader)
    if kwargs["input_layout"] == "MN":
        spatial_stride_src = matrix_data.shape[1] * element_width // 8
        temporal_bounds_src = [transfer_per_transpose, matrix_data.shape[1] //
                               tile_width, matrix_data.shape[0] // tile_width]
        temporal_strides_src = [
            8,
            tile_width *
            element_width //
            8,
            matrix_data.shape[1] *
            tile_width *
            element_width //
            8]
    else:
        match = re.search(r'MNM(\d+)N(\d+)', kwargs["input_layout"])
        m, n = match.groups()
        m, n = int(m), int(n)
        spatial_stride_src = n * element_width // 8
        temporal_bounds_src = [
            transfer_per_transpose,
            n // tile_width,
            matrix_data.shape[1] // n,
            m // tile_width,
            matrix_data.shape[0] // m]
        temporal_strides_src = [
            8,
            tile_width *
            element_width //
            8,
            m *
            n *
            element_width //
            8,
            n *
            tile_width *
            element_width //
            8,
            matrix_data.shape[1] *
            m *
            element_width //
            8]

    # Output Side (Writer)
    if kwargs["enable_transpose"] is True:
        if kwargs["output_layout"] == "MN":
            spatial_stride_dst = matrix_data.shape[0] * element_width // 8
            temporal_bounds_dst = [
                transfer_per_transpose,
                matrix_data.shape[1] // tile_width,
                matrix_data.shape[0] // tile_width]
            temporal_strides_dst = [
                8,
                matrix_data.shape[0] *
                tile_width *
                element_width //
                8,
                tile_width *
                element_width //
                8]
        else:
            match = re.search(r'MNM(\d+)N(\d+)', kwargs["output_layout"])
            m, n = match.groups()
            m, n = int(m), int(n)
            spatial_stride_dst = n * element_width // 8
            temporal_bounds_dst = [
                transfer_per_transpose,
                m // tile_width,
                matrix_data.shape[1] // m,
                n // tile_width,
                matrix_data.shape[0] // n]
            temporal_strides_dst = [
                8,
                n * tile_width * element_width // 8,
                m * matrix_data.shape[0] * element_width // 8,
                tile_width * element_width // 8,
                m * n * element_width // 8]
    else:
        if kwargs["output_layout"] == "MN":
            spatial_stride_dst = matrix_data.shape[1] * element_width // 8
            temporal_bounds_dst = [
                transfer_per_transpose,
                matrix_data.shape[1] // tile_width,
                matrix_data.shape[0] // tile_width]
            temporal_strides_dst = [
                8,
                tile_width *
                element_width //
                8,
                matrix_data.shape[1] *
                tile_width *
                element_width //
                8]
        else:
            match = re.search(r'MNM(\d+)N(\d+)', kwargs["output_layout"])
            m, n = match.groups()
            m, n = int(m), int(n)
            spatial_stride_dst = n * element_width // 8
            temporal_bounds_dst = [
                transfer_per_transpose,
                n // tile_width,
                matrix_data.shape[1] // n,
                m // tile_width,
                matrix_data.shape[0] // m]
            temporal_strides_dst = [
                8,
                tile_width * element_width // 8,
                m * n * element_width // 8,
                n * tile_width * element_width // 8,
                matrix_data.shape[1] * m * element_width // 8]

    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "spatial_stride_src",
            spatial_stride_src)]
    emit_str += [
        format_scalar_definition(
            "uint32_t",
            "spatial_stride_dst",
            spatial_stride_dst)]
    emit_str += [format_vector_definition("uint32_t",
                                          "temporal_bounds_src",
                                          temporal_bounds_src)]
    emit_str += [format_vector_definition("uint32_t",
                                          "temporal_bounds_dst",
                                          temporal_bounds_dst)]
    emit_str += [format_vector_definition("uint32_t",
                                          "temporal_strides_src",
                                          temporal_strides_src)]
    emit_str += [format_vector_definition("uint32_t",
                                          "temporal_strides_dst",
                                          temporal_strides_dst)]
    emit_str += [format_scalar_definition("uint32_t",
                                          "temporal_dimension_src",
                                          len(temporal_bounds_src))]
    emit_str += [format_scalar_definition("uint32_t",
                                          "temporal_dimension_dst",
                                          len(temporal_bounds_dst))]

    emit_str += [format_scalar_definition("uint8_t",
                                          "enable_transpose",
                                          1 if kwargs["enable_transpose"]
                                          else 0)]
    emit_str += [format_vector_definition("uint32_t",
                                          "transposer_param",
                                          [0 if element_width == 8 else 1])]
    return emit_str


def main():
    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generating data for kernels")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    # Emit header file
    print(emit_header_file(**param))


if __name__ == "__main__":
    main()
