#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The FFN ablation ladder at 64 tokens, rung bcast: DeepSeek-V2-Lite's dense FFN (layer 0), one pass on 16 chiplets, X from HBM
(params.hjson); the driver is ../../common/ffn_stages_mc.py."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from ffn_stages_mc import run  # noqa: E402

if __name__ == "__main__":
    run()
