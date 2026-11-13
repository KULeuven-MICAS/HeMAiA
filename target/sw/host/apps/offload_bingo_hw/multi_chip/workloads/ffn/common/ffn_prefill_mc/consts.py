# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The block's constants (the package's __init__ has put util/sim/multichip and the mini_compiler on sys.path)."""

import chiplet_mc as cm
from driver import repo_root

REPO = repo_root(__file__)
TILE = 16 * 16 * 2                         # one D-layout FP16 tile
BLK = 16                                   # a GEMM column block (meshCol)
EW_F16_ADD = cm.EW_F16_ADD
