#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite layer 1, stage 1: x (HBM) -> RMSNorm + quantiser -> x8, a_row in L3.

Checked: x8. The first stage: every later one starts from the x8 row this one leaves in L3.
The shared driver (../../common/dsv2_staged.py) stages the data and emits the headers; this
file holds only what stage 1 adds.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from dsv2_staged import D, F16, HBM, I8, L3, RM, run  # noqa: E402

from libs import Layout, PortSpec, at_offset  # noqa: E402
from libs.blocks import ARowPack, Join, RMSNormRow  # noqa: E402

STAGE = 1


def build(S):
    """One token per cluster (token t on cluster tc[t]): x, FP16 in the HBM, normalised and
    quantised in one kernel, its INT8 row packed into x8 in L3 as a_row -- the operand every
    later GEMV reads. A pass of several tokens joins their rows into one [T, D] a_row."""
    T, pipe = S.T, S.pipe
    for t in range(T if S.FROM == 1 else 0):
        sfx = "" if T == 1 else f"_t{t}"
        xt = S.staged(PortSpec(RM, F16, (1, D), mem_level=HBM), at_offset(S.h_xh, 2 * D * t))
        nrm = pipe.add(RMSNormRow(cols=D, cluster=S.tc[t], in_level=HBM, quant_inv=S.inv["x"]),
                       f"norm{sfx}", x=xt)
        S.packs.append(pipe.add(ARowPack(cols=D, cluster=S.tc[t],
                                         dst=at_offset(S.h_x8, 2 * D * t)),
                                f"pack_x8{sfx}", x=nrm.out()))
    packs = S.packs
    S.x8p = None if not packs else packs[0] if T == 1 else pipe.add(
        Join(parts=[PortSpec(Layout.A_ROW, I8, (1, D), mem_level=L3)] * T,
             dst=PortSpec(Layout.A_ROW, I8, (T, D), mem_level=L3)), "x8",
        **{f"x{t}": p_.out() for t, p_ in enumerate(packs)})


def checks(S):
    """x8 needs no stash: it is already in L3, and the driver checks it first of all."""


if __name__ == "__main__":
    run(sys.modules[__name__])
