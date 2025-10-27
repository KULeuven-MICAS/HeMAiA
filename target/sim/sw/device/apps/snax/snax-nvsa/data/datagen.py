#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Ryan Antonio <ryan.antonio@esat.kuleuven.be>

import sys
import numpy as np
import os
import subprocess

# Set seed
np.random.seed(42)

# -----------------------
# Add data utility path
# -----------------------
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/")
)
from data_utils import (
    format_vector_definition,
    format_scalar_definition
)  # noqa: E402

# -----------------------
# Add hypercorex utility paths
# -----------------------
bender_command = subprocess.run(
    ["bender", "path", "hypercorex"], capture_output=True, text=True
)
hypercorex_path = bender_command.stdout.strip()

sys.path.append(hypercorex_path + "/hdc_exp/")
sys.path.append(hypercorex_path + "/sw/")

from hdc_util import (  # noqa: E402
    reshape_hv_list,
    binarize_hv,
)

from system_regression import (  # noqa: E402
    data_ortho_im_only,
)

from hypercorex_compiler import compile_hypercorex_asm  # noqa: E402

# Paths for the hypercorex "asm"
current_directory = os.path.dirname(os.path.abspath(__file__))
am_search_asm_path = current_directory + "/am_search.txt"

snax_hypercorex_parameters = {
    "seed_size": 32,
    "hv_dim": 512,
    "num_total_im": 1024,
    "num_per_im_bank": 128,
    "base_seeds": [
        1103779247,
        2391206478,
        3074675908,
        2850820469,
        811160829,
        4032445525,
        2525737372,
        2535149661,
    ],
    "gen_seed": True,
    "ca90_mode": "ca90_hier",
}

# Loading data first
ortho_im, _ = data_ortho_im_only(
    seed_size=snax_hypercorex_parameters["seed_size"],
    hv_dim=snax_hypercorex_parameters["hv_dim"],
    num_total_im=snax_hypercorex_parameters["num_total_im"],
    num_per_im_bank=snax_hypercorex_parameters["num_per_im_bank"],
    base_seeds=snax_hypercorex_parameters["base_seeds"],
    gen_seed=snax_hypercorex_parameters["gen_seed"],
    ca90_mode=snax_hypercorex_parameters["ca90_mode"],
)

# -----------------------
# Some useful functions
# -----------------------

# Convert from list to binary value
def hvlist2num(hv_list):
    # Bring back into an integer itself!
    # Sad workaround is to convert to str
    # The convert to integer
    hv_num = "".join(hv_list.astype(str))
    hv_num = int(hv_num, 2)

    return hv_num


def reorder_list(input_list, reorder_chunk_size, num_iterations):
    new_list = []
    for i in range(num_iterations):
        sub_list = \
            input_list[i * reorder_chunk_size:(i + 1) * reorder_chunk_size]
        # Reverse the sublist
        sub_list_reversed = sub_list[::-1]
        # Append the reversed sublist to the new list
        new_list.extend(sub_list_reversed)
    return new_list

def pack_int8_to_int32_shift(v: np.ndarray, little_endian: bool = True) -> np.ndarray:
    v = np.asarray(v, dtype=np.int8)
    if v.size % 4 != 0:
        raise ValueError("Length must be a multiple of 4")

    # Convert to uint8 so we can pack raw byte values
    b = v.view(np.uint8).reshape(-1, 4)  # (N, 4) groups

    if not little_endian:
        b = b[:, ::-1]  # reverse byte order for big-endian

    u32 = np.zeros(b.shape[0], dtype=np.uint32)
    for k in range(4):
        u32 |= (b[:, k].astype(np.uint32) << (8 * k))

    return u32.view(np.int32)


# Main function to generate data
def main():

    HV_DIM = 512
    VEC_ROWS = 10
    CUT_WIDTH = 32

    # Specifics from cluster
    NARROW_DATA_WIDTH = 64
    WIDE_DATA_WIDTH = 512

    # Number of elements per wide
    NUM_CUT_IN_WIDE_ELEM = WIDE_DATA_WIDTH // CUT_WIDTH

    # Number of targetted wide data
    NUM_AM_SEARCH = 100

    # Total number of data to check
    TOTAL_NUM_AM_DATA = NUM_AM_SEARCH * NUM_CUT_IN_WIDE_ELEM

    #----------------------------------------
    # Preparing AM data
    #----------------------------------------
    # Extract instructions for training and testing
    am_search_code_list, am_search_control_code_list = compile_hypercorex_asm(
        am_search_asm_path
    )

    # Convert each instruction to integers for input
    for i in range(len(am_search_code_list)):
        am_search_code_list[i] = hvlist2num(np.array(am_search_code_list[i]))

    # Convert ortho im to reshape
    # Data a is the first TOTAL_NUM_AM_DATA
    ortho_im_reshape = reshape_hv_list(ortho_im, CUT_WIDTH)
    ortho_im_list = []
    for i in range(TOTAL_NUM_AM_DATA):
        ortho_im_list.append(hvlist2num(np.array(ortho_im_reshape[i])))

    #----------------------------------------
    # Preparing data to be compressed
    #----------------------------------------
    vec_list = []
    binarized_vec_list = []

    # Generate data
    for i in range(VEC_ROWS):
        # Generate a random vector of size HV_DIM
        vec = np.random.randint(-128, 128, size=HV_DIM, dtype=np.int8)
        binarized_vec = binarize_hv(vec, 0)
        vec_list.append(vec)
        binarized_vec_list.append(binarized_vec)

    # Re-pack so it's 32 bits packed
    vec_reshape_list = pack_int8_to_int32_shift(vec_list, little_endian=True)

    # Fix the 512b vector alignments
    binarized_vec_reshape = reshape_hv_list(binarized_vec_list, CUT_WIDTH)

    binarized_vec_list = []
    for i in range(len(binarized_vec_reshape)):
        binarized_vec_list.append(
            hvlist2num(np.array(binarized_vec_reshape[i])))

    binarized_vec_list = reorder_list(binarized_vec_list, 16, VEC_ROWS)

    #----------------------------------------
    # Generating strings to print
    #----------------------------------------
    num_am_search_str = format_scalar_definition(
        "uint32_t", "num_am_search", NUM_AM_SEARCH
    )
    am_search_code_str = format_vector_definition(
        "uint32_t", "am_search_code", am_search_code_list
    )
    am_list_str = format_vector_definition(
        "uint32_t", "am_list", ortho_im_list
    )
    vec_num_str = format_scalar_definition(
        "uint32_t", "vec_num", VEC_ROWS
    )
    # Generating strings to print
    vec_list_str = format_vector_definition(
        "int32_t", "vec_list", vec_reshape_list
    )
    # Generate string for ortho_im_reshape
    binarized_vec_list_str = format_vector_definition(
        "uint32_t", "binarized_vec_list", binarized_vec_list
    )
    # Preparing string to load
    f_str = "\n\n".join(
        [
            num_am_search_str,
            vec_num_str,
            am_search_code_str,
            am_list_str,
            vec_list_str,
            binarized_vec_list_str,
        ]
    )
    f_str += "\n"

    # Write to stdout
    print(f_str)


if __name__ == "__main__":
    sys.exit(main())
