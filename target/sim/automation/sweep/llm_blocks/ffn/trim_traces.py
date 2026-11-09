#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Moved to sweep/common/trim_traces.py (9 Nov). This forwarder stays only while a run launched before the move is
alive: its watchdog loaded the profile's on_stop_cmd, this path, when it started (stg_layer0_prefill_full_1,
stg_sweep64_base_n4_1, stg_sweep64_all_n4_1). Delete it once none of them runs."""
import os
import runpy
import sys

sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common", "trim_traces.py")
runpy.run_path(sys.argv[0], run_name="__main__")
