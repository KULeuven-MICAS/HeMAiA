#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""One Llama-3-8B layer with synthetic weights over a fresh 1,024-token prompt on sixteen chiplets, as one graph, every
pass: the driver is ../common/llama3_layer_mc.py; params.hjson says the rest."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from llama3_layer_mc import run  # noqa: E402

if __name__ == "__main__":
    run()
