#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Rung 5 of the LLM bring-up ladder: the layer's first 5 stage(s), with a check on
# each. The build itself is util/sim/common/llm_layer_stages.py -- every rung is a PREFIX
# of the same code, so a rung cannot disagree with the full layer about how a stage is
# built. See that module for the ladder and what each rung adds.

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(current_dir, "../../../../../../../../"))
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(f"{ROOT_DIR}/util/sim/common")

import _bingo_paths  # noqa: F401,E402
from llm_layer_app import run  # noqa: E402

if __name__ == "__main__":
    run(stages=5)
