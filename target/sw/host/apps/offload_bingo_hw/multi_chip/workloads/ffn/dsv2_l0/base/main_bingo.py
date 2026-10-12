#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's dense FFN (layer 0) on sixteen chiplets, variant base: params.hjson says which
mechanisms it uses; the driver is ../../common/ffn_mc.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from ffn_mc import run  # noqa: E402

if __name__ == "__main__":
    run()
