#!/usr/bin/env python3

# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Ryan Antonio <ryan.antonio@esat.kuleuven.be>

import sys
import os
import numpy as np

# -----------------------
# Add data utility path
# -----------------------
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/")
)
from data_utils import format_scalar_definition, format_vector_definition  # noqa: E402


def binarize(vec):
    return (vec >= 0).astype(int)


def pack_to_32b(bin_vec):
    # Each element of bin_vec is a single bit; pack groups of 32 into one uint32
    packed = bin_vec.reshape(-1, 32)
    powers = np.uint32(1) << np.arange(32, dtype=np.uint32)
    return np.array(
        [np.bitwise_or.reduce(row.astype(np.uint32) * powers) for row in packed],
        dtype=np.uint32,
    )


def main():

    # Working parameters
    class_size = 100
    data_size = 512

    assert data_size % 32 == 0, f"data_size ({data_size}) must be a multiple of 32"

    # Generating sample query hypervector
    data_vec = np.random.randint(low=-128, high=128, size=data_size)
    bin_data_vec = binarize(data_vec)
    bin_data_vec_32b = pack_to_32b(bin_data_vec)

    # Generating class array
    class_array = np.zeros((class_size, data_size))
    bin_class_array = np.zeros((class_size, data_size))
    bin_class_array_32b = np.zeros((class_size, data_size // 32), dtype=np.uint32)
    for i in range(class_size):
        class_array[i] = np.random.randint(low=-128, high=128, size=data_size)
        bin_class_array[i] = binarize(class_array[i])
        bin_class_array_32b[i] = pack_to_32b(bin_class_array[i])
    bin_class_array_32b_flat = bin_class_array_32b.flatten()

    # Calculating hamming distance between query and classes
    hamming_distances = np.zeros(class_size, dtype=np.uint32)
    for i in range(class_size):
        xor_result = np.bitwise_xor(bin_data_vec_32b, bin_class_array_32b[i])
        hamming_distances[i] = np.sum(np.unpackbits(xor_result.view(np.uint8)))

    # Format output data
    class_size_str = format_scalar_definition("uint32_t", "class_size", class_size)
    data_size_str = format_scalar_definition("uint32_t", "data_size", data_size)
    data_vec_str = format_vector_definition("int8_t", "data_vec", data_vec)
    bin_data_vec_32b_str = format_vector_definition(
        "uint32_t", "bin_data_vec_32b", bin_data_vec_32b
    )
    bin_class_array_32b_str = format_vector_definition(
        "uint32_t", "bin_class_array_32b", bin_class_array_32b_flat
    )
    hamming_distances_str = format_vector_definition(
        "uint32_t", "hamming_distances", hamming_distances
    )

    f_str = "\n\n".join(
        [
            class_size_str,
            data_size_str,
            data_vec_str,
            bin_data_vec_32b_str,
            bin_class_array_32b_str,
            hamming_distances_str,
        ]
    )
    f_str += "\n"

    print(f_str)


if __name__ == "__main__":
    sys.exit(main())
