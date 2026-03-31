#!/usr/bin/env python3

# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

import sys
import os
import numpy as np


# -----------------------
# Add data utility path
# -----------------------
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/")
)
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

def main():

    # Working parameters
    num_32b = 20000
    num_bytes = num_32b * 4

    # Generating sample query hypervector
    data_set = np.frombuffer(np.random.bytes(num_32b*4), dtype=np.uint32)

   
    # Format output data
    num_32b_str = format_scalar_definition("uint32_t", "num_32b", num_32b)
    num_bytes_str = format_scalar_definition("uint32_t", "num_bytes", num_bytes)
    data_set_str = format_vector_definition("uint32_t", "data_set", data_set)
   

    f_str = "\n\n".join(
        [
            num_32b_str,
            num_bytes_str,
            data_set_str,
        ]
    )
    f_str += "\n"

    print(f_str)


if __name__ == "__main__":
    sys.exit(main())
