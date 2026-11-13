#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B decoder layer with synthetic weights, one decode token over a 1,024-position KV cache on sixteen chiplets,
the decode blocks in one graph: the driver is ../../common/llama3_decode_layer_mc.py; params.hjson says the rest."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from llama3_decode_layer_mc import run  # noqa: E402

if __name__ == "__main__":
    run()
