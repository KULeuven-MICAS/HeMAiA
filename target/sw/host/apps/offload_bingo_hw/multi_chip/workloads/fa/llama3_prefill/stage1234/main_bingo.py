#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's attention with synthetic weights over a fresh 256-token prompt on sixteen chiplets, stages 1-4 alone (x, the projections, the K/V trade): the driver is
../../../llama3/common/llama3_stages_mc.py; params.hjson says the rest."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "llama3", "common"))
from llama3_stages_mc import run  # noqa: E402

if __name__ == "__main__":
    run()
