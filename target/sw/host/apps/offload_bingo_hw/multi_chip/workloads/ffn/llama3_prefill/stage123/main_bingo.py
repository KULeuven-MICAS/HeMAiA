#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's FFN (synthetic weights) as a 256-token prefill on sixteen chiplets, stages 1-3 of one pass: X in L1 before the window; gate|up over the weight stream, dequantise and SwiGLU, the reshape, the quantise and the A8 exchange (params.hjson); the driver is
../../common/ffn_stages_mc.py."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "llama3", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import llama3_data  # noqa: E402
from ffn_stages_mc import run  # noqa: E402

if __name__ == "__main__":
    llama3_data.register_ffn()
    run()
