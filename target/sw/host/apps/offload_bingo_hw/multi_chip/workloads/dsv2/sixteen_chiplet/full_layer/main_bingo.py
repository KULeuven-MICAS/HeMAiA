#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1 on sixteen chiplets: the whole layer, stages 1-8.

The layer's code is the stage chain (../stage8_experts builds on stages 1..7); this directory
holds only the layer's tuned settings (params.hjson).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_mc import load_stage, run  # noqa: E402

LAST = load_stage(__file__, "stage8_experts")
STAGE = LAST.STAGE
build, checks = LAST.build, LAST.checks

if __name__ == "__main__":
    run(sys.modules[__name__])
