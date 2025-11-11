#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../sim/apps/"))

from bin2preload import bin2preload  # noqa E402

if __name__ == "__main__":
    input_binary_file = os.path.join(
        os.path.dirname(__file__),
        "./matmul_data.bin"
    )
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../../../sim/bin/app_chip_2_0/"
    )

    # Convert binary to hex
    bin2preload(input_binary_file, output_dir)
