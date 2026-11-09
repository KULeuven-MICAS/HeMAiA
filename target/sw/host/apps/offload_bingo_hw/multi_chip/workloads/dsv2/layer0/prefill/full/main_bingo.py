#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 0 as a prefill of a fresh 1,024-token prompt on sixteen chiplets, the whole layer: all eight
attention chunks and the dense FFN in eight passes of 128 rows (params.hjson); the driver is
../../../common/dsv2_layer_prefill_mc.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
from dsv2_layer_prefill_mc import run  # noqa: E402

if __name__ == "__main__":
    run()
