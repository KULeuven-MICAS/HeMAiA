#!/usr/bin/env python3

# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Ryan Antonio <ryan.antonio@esat.kuleuven.be>

import sys
import numpy as np
import os
import subprocess

# -----------------------
# Add data utility path
# -----------------------
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/")
)
from data_utils import format_scalar_definition, format_vector_definition  # noqa: E402


# Main function to generate data
def main():

    # Generate sample matrix size
    M = 32
    K = 32
    N = 32

    matrix_a = np.random.randint(low=-128, high=127, size=(M, K))
    vec_a = matrix_a.flatten().tolist()

    matrix_b = np.random.randint(low=-128, high=127, size=(K, N))
    vec_b = matrix_b.flatten().tolist()

    matrix_o = np.matmul(matrix_a, matrix_b)
    vec_o = matrix_o.flatten().tolist()

    # Golden list of data
    M_str = format_scalar_definition("int32_t", "M", M)
    K_str = format_scalar_definition("int32_t", "K", K)
    N_str = format_scalar_definition("int32_t", "N", N)
    vec_a_str = format_vector_definition("int8_t", "vec_a", vec_a)
    vec_b_str = format_vector_definition("int8_t", "vec_b", vec_b)
    vec_o_str = format_vector_definition("int32_t", "vec_o", vec_o)
        
    # Preparing string to load
    f_str = "\n\n".join(
        [
            M_str,
            K_str,
            N_str,
            vec_a_str,
            vec_b_str,
            vec_o_str
        ]
    )
    f_str += "\n"

    # Write to stdout
    print(f_str)


if __name__ == "__main__":
    sys.exit(main())

