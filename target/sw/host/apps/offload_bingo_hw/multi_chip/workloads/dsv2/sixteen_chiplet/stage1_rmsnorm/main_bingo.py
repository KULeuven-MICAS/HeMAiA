#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1 on sixteen chiplets, stage 1: x -> RMSNorm + quantiser -> x8 on EVERY chip.

Checked: x8, on every chip. Each compute chip normalises the token itself, on its cluster 0,
from its own memory chiplet's HBM, and packs the INT8 row as a_row into its own L3 -- the
operand every later GEMV on that chip reads. The twelve norms run in parallel and save a
broadcast. The shared driver (../../common/dsv2_mc.py) stages the data and emits the
headers; this file holds only what stage 1 adds.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_mc import D, F16, HBM, RM, run  # noqa: E402

from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs import PortSpec  # noqa: E402
from libs.blocks import ARowPack, RMSNormRow  # noqa: E402

STAGE = 1


def build(S):
    """Per chip: the norm on its cluster 0, the a_row pack into its L3 (S.x8[chip])."""
    S.x8 = {}
    for k in S.chips:
        g = S.gs(k)[0]
        xt = S.staged(PortSpec(RM, F16, (1, D), mem_level=HBM), S.h_xh[S.mem[k]])
        nrm = S.add(RMSNormRow(cols=D, cluster=0, in_level=HBM, quant_inv=S.inv["x"]),
                    f"norm_k{k:02x}", g, x=xt)
        S.x8[k] = S.add(ARowPack(cols=D, cluster=0, dst=S.h_x8), f"pack_x8_k{k:02x}", g,
                        x=nrm.out())
        S.packs.append(S.x8[k])


def checks(S):
    """x8 is already in each chip's L3: chip 0's host reads every copy."""
    for k in S.chips:
        S.stash.append((f"x8_k{k:02x}", S.x8[k], S.h_gx8,
                        BingoMemSymbol("dsv2_x8", chip_id=k), 2 * D))


if __name__ == "__main__":
    run(sys.modules[__name__])
