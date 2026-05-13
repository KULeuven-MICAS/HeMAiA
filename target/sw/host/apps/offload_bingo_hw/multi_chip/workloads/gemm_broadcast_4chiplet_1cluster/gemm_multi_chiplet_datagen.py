#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

import os
import sys

SIM_UTIL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim")
)
if SIM_UTIL_DIR not in sys.path:
    sys.path.append(SIM_UTIL_DIR)

from snax_utils import (  # noqa E402
    emit_multichip_gemm_header_file,
    run_multichip_gemm_datagen,
)


def emit_header_file(**kwargs):
    return emit_multichip_gemm_header_file(**kwargs)


def main():
    run_multichip_gemm_datagen()


if __name__ == "__main__":
    main()
