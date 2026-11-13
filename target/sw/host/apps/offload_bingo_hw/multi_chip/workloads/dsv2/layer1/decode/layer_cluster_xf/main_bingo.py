#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""dsv2/layer1/decode/layer_cluster_xf (cluster-posted weight rings, the root memory chip's other chips' attention rings
after the input norm's HBM read: params weight_ring.mode cluster, layer.x_load_first): DeepSeek-V2-Lite's layer 1 (MLA
attention + MoE) as one decode step over a 1,024-token cache, the attention split by key and every weight multiply on
the GEMV, the blocks' decode mappings in one graph (params.hjson); the driver is ../../../common/dsv2_decode_mc.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
from dsv2_decode_mc import run  # noqa: E402

if __name__ == "__main__":
    run()
