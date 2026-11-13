# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The block's shapes and the mapping's arithmetic (check)."""

import os
import sys

_ROOT = os.path.abspath(__file__).split("/target/sw/host/")[0]
sys.path.insert(0, os.path.join(_ROOT, "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(_ROOT, "util", "sim", "common"))
from driver import repo_root  # noqa: E402

REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # fa/common

import _bingo_paths  # noqa: F401,E402
import chiplet_mc as cm  # noqa: E402

BR = 16                       # query rows a tile: the SIMD's FP16 lanes
D_QK, D_V, KV_RANK, ROPE, Q_NOPE, Q_HEAD, HIDDEN = 576, 512, 512, 64, 128, 192, 2048
TILE16 = 16 * 16 * 2          # one D-layout FP16 block
EW_F16_ADD = cm.EW_F16_ADD
KB, VB = 16 * D_QK, D_V * 16  # a 16-token piece of the keys: its key rows, its value columns (INT8)
KVP = KB + VB


def check(p):
    """The mapping's arithmetic; raises ValueError where the parameters cannot be laid out. The keys are the prompt's
    own tokens (a fresh prompt)."""
    tokens, chunk, bc = (int(p[k]) for k in ("tokens", "chunk", "bc"))
    rows = chunk // 2
    if tokens % chunk or chunk % 2 or rows % BR:
        raise ValueError(f"tokens {tokens} / chunk {chunk}: a cluster takes chunk / 2 rows, a whole number of "
                         f"{BR}-row query tiles")
    if tokens % bc or bc % 16 or tokens // bc < 4:
        raise ValueError(f"tokens {tokens}: not whole key tiles of bc {bc} (a multiple of 16), at least four")
    if chunk % bc:
        raise ValueError(f"chunk {chunk}: whole key tiles of bc {bc} (a chunk's two row blocks sweep every tile up to "
                         f"its end)")
    wq_block = HIDDEN * 64 * int(p.get("wbits", 8)) // 8
    if int(p["w_chunk_bytes"]) < max(bc * D_QK, D_V * bc, wq_block):
        raise ValueError(f"w_chunk_bytes {p['w_chunk_bytes']}: a K tile is {bc * D_QK} B, a W_q block "
                         f"{wq_block} B")
    return dict(chunks=tokens // chunk, query_tiles=rows // BR, tiles=tokens // bc, keys=tokens)
