#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# ======================================================================================
# ONE TRANSFORMER LAYER, ASSEMBLED FROM BLOCKS
# ======================================================================================
#
# The top of the bring-up ladder: every stage, with a check on each. Rungs 1..5 are
# llm_s1_norm_4cluster .. llm_s5_ffn_4cluster, and each is a PREFIX of the same builder,
# so the first rung that fails on hardware names the stage rather than the layer.
#
# The dataflow, the layouts on every arrow, and why the reshapes sit where they do are in
# util/sim/common/llm_layer_stages.py, which is where the layer is actually built. This
# file exists so the full layer is a workload like any other.
#
# WHAT THIS LAYER IS NOT. There is no causal mask -- every query attends to every key --
# so it is decode-shaped, not prefill. See the TODO on FlashAttention. There is also no KV
# cache: Q, K and V come from the staged operands rather than from a previous step.
# ======================================================================================

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim/common")

import _bingo_paths  # noqa: F401,E402
from llm_layer_app import run  # noqa: E402
from llm_layer_stages import MAX_STAGE  # noqa: E402

if __name__ == "__main__":
    run(stages=MAX_STAGE)
