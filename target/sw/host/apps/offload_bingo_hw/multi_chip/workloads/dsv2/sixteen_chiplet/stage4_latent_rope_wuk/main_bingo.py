#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1 on sixteen chiplets, stage 4: the latent's RMSNorm, RoPE, W_UK absorbed.

The stage's code is common/mc_stages/stage4_latent_rope_wuk.py (shared with six_chiplet); this
directory holds the Makefile and the params.hjson.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_mc import load_stage, run  # noqa: E402

_S = load_stage(__file__, "stage4_latent_rope_wuk")
STAGE, build, checks = _S.STAGE, _S.build, _S.checks

if __name__ == "__main__":
    run(sys.modules[__name__])
