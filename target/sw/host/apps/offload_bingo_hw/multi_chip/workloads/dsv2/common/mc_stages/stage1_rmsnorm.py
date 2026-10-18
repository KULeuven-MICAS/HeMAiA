# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1, multi-chiplet stage 1: x -> RMSNorm + quantiser -> x8 on EVERY chip.

Checked: x8, on every chip. Each compute chip normalises the token itself, on its cluster 0,
from its own memory chiplet's HBM, and packs the INT8 row as a_row into its own L3 -- the
operand every later GEMV on that chip reads. The per-chip norms run in parallel and save a
broadcast. The shared driver (../dsv2_mc.py) stages the data and emits the headers; this
file holds only what stage 1 adds.
"""

# Loaded by dsv2_mc.load_stage: dsv2_mc, and with it the import paths, is already in place.
from dsv2_mc import D, F16, HBM, RM

from bingo_mem_handle import BingoMemSymbol
from libs import PortSpec
from libs.blocks import ARowPack, RMSNormRow

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

