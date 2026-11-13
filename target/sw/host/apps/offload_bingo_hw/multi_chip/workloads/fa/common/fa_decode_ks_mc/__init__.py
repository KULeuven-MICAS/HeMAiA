# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MLA attention as ONE DECODE STEP, split by key, every weight multiply on the one-token
GEMV: the attention block of the whole-layer decode (../../../dsv2/common/dsv2_decode_mc.py, AttnKsPart). README.md
explains the mapping, the steps, the orders, the collectives and the parameters. The modules:

    consts.py    the sizes, the constants' layouts, a_row
    golden.py    KsGolden, the device's arithmetic of the step
    params.py    the parameters and the mapping (FaDecodeKs.__init__)
    memory.py    the data, the staging, the ring and its loads, the exchange buffers (mem_handles)
    ops.py       the task makers the steps share
    build.py     build(): the steps in order, the preloads, the hooks a layer provides
    query.py     x into L1, W_KVA's columns, the query rows, the query tile
    new_key.py   the new key and its append
    sweep.py     pass 1, m*, pass 2
    output.py    the partials' chain, o~, W_UV, W_O
    report.py    the D2D traffic
"""

import os
import sys

_ROOT = os.path.abspath(__file__).split("/target/sw/host/")[0]
sys.path.insert(0, os.path.join(_ROOT, "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(_ROOT, "util", "sim", "common"))
sys.path.insert(0, f"{_ROOT}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # fa/common: fa_decode_data

import _bingo_paths  # noqa: F401,E402
import chiplet_mc as cm  # noqa: E402

from .build import KsBuild  # noqa: E402
from .consts import (ARENA, ATOM, BC, BEAT, BR, C0_BYTES, C1_BYTES, D_QK, D_V, EW_F16_MAX, FA, HIDDEN,  # noqa: E402
                     KB, KV_RANK, LANES, LAT, LINK_BW, LRUN, MRUN, N_KVA, OAR, OB, PB, PK, PVB, PVK, Q_HEAD, Q_NOPE,
                     QB, QC, QROW, ROPE, SB, V_HEAD, VB, WCH, WQ_ORDER, XB, XFER, a_row, f32bits)
from .golden import KsGolden  # noqa: E402
from .memory import KsMemory  # noqa: E402
from .new_key import KsNewKey  # noqa: E402
from .ops import KsOps  # noqa: E402
from .output import KsOutput  # noqa: E402
from .params import KsParams  # noqa: E402
from .query import KsQuery  # noqa: E402
from .report import KsReport  # noqa: E402
from .sweep import KsSweep  # noqa: E402


__all__ = ["ARENA", "ATOM", "BC", "BEAT", "BR", "C0_BYTES", "C1_BYTES", "D_QK", "D_V", "EW_F16_MAX", "FA", "HIDDEN", "KB",
           "KV_RANK", "LANES", "LAT", "LINK_BW", "LRUN", "MRUN", "N_KVA", "OAR", "OB", "PB", "PK", "PVB", "PVK", "Q_HEAD",
           "Q_NOPE", "QB", "QC", "QROW", "ROPE", "SB", "V_HEAD", "VB", "WCH", "WQ_ORDER", "XB", "XFER", "a_row", "f32bits",
           "FaDecodeKs", "KsGolden"]


class FaDecodeKs(KsParams, KsMemory, KsOps, KsBuild, KsQuery, KsNewKey, KsSweep, KsOutput, KsReport,
                 cm.ChipletBuild):
    """The key-split MLA decode step (README.md), a block of a layer: __init__ (params.py), mem_handles()
    (memory.py), build() (build.py and the steps' modules). The layer subclass provides the window (setup,
    end_window), x (x_multicast), y's destination (stage_store) and the checks."""
    TAG = "fk"
    QC, KC = 0, 1                    # every chip's query / attention cluster, and its W_KVA cluster
