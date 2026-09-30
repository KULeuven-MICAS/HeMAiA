#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1, the full layer: x -> MLA attention -> MoE -> out, every stage.

Checked: every stage's values, then the layer output. The layer's code is the stage chain
(../stage1_rmsnorm ... ../stage8_experts): this app builds all of it, with the whole layer's
tuned settings (params.hjson, one token; params_t4.hjson, a pass of four). Run this to run the
layer; work on one stage in its own directory.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_staged import load_stage, run  # noqa: E402

LAST = load_stage(__file__, "stage8_experts")
STAGE = LAST.STAGE
build, checks = LAST.build, LAST.checks

if __name__ == "__main__":
    run(sys.modules[__name__])
