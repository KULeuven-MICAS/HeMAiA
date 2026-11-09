#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Llama-3-8B's whole decode attention block, one token over an 8,192-position KV cache on sixteen chiplets, the attention
split by key, rung all of the FA decode ablation, its cache copies behind q and W_O pushed during the sweep (cache_order after_q, wo_push before_chains): the driver is ../../../llama3/common/llama3_decode_ks_mc.py; params.hjson
says the rest."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "llama3", "common"))
from llama3_decode_ks_mc import run  # noqa: E402

if __name__ == "__main__":
    run()
