#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
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

# __usg__ grouped util/sim: make common/gemm/xdma/ara importable
import os as _usg_os, sys as _usg_sys
for _usg_p in [p for p in list(_usg_sys.path) if str(p).rstrip('/').endswith('util/sim')]:
    for _usg_s in ('common', 'gemm', 'xdma', 'ara'):
        _usg_sub = _usg_os.path.join(_usg_p, _usg_s)
        if _usg_sub not in _usg_sys.path:
            _usg_sys.path.append(_usg_sub)
from gemm_sim_utils import (  # noqa E402
    emit_multichip_gemm_header_file,
    run_multichip_gemm_datagen,
)


def emit_header_file(**kwargs):
    return emit_multichip_gemm_header_file(**kwargs)


def main():
    run_multichip_gemm_datagen()


if __name__ == "__main__":
    main()
