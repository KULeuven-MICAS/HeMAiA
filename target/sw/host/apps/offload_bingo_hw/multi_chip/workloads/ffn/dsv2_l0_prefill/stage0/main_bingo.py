#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's dense FFN (layer 0) as a 256-token prefill on sixteen chiplets, stage 0: X of every pass from the root memory chip's HBM into every chip's L3 (its push engine's multicast, a flag a tile) and the first pass's X into every cluster's L1 (params.hjson); the driver is
../../common/ffn_stages_mc.py."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from ffn_stages_mc import run  # noqa: E402

if __name__ == "__main__":
    run()
