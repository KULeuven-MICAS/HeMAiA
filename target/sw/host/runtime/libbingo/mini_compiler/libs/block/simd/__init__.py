# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The per-element and per-row operators a layer is glued together with.

RMSNorm, RoPE, quantise, dequantise and the residual add. Each is one fused kernel, so each
is one node, and the block around it exists to state the layouts -- which is the whole
difficulty, because these do NOT all care about layout in the same way. That difference is
what the files are cut on:

  pointwise.py  ELEMENTWISE -- quantise, dequantise, residual add. They touch each value
                independently, so a permutation of the inputs is the same permutation of
                the output and ANY layout works. They declare one for readability and pass
                it through; MoE runs its SwiGLU straight over a D-layout block for exactly
                this reason.

  norm.py       PER-ROW, AND REDUCING -- RMSNorm. It reduces ALONG a row, which is both a
                contiguity requirement and, on this block, a 3x cost difference depending
                on orientation. That file is mostly about the difference.

  rope.py       PER-ROW, NOT REDUCING -- RoPE. It rotates along a row, so it needs the row
                contiguous but has no reduction for a transposed layout to make cheap.

  common.py     the tile description and the beat it has to tile to.

WHY THE DISTINCTION IS NOT A DETAIL. In D-layout (m, n, r, c) a matrix row is NOT
contiguous: consecutive columns of one row are `meshCol` apart across n-blocks. Handing
D-layout to a per-row operator normalises or rotates groups that are not rows. It does not
fault, it does not go out of range, and the answer is a well-formed tensor of wrong
numbers. So their ports say `row_major` and mean it.

THE ORDER A LAYER HAS TO USE, and it is forced by hardware rather than taste:

    GEMM (D/f16) -> reshape to row_major -> RMSNorm (row_major/f16) -> reshape to A (f16)
                 -> quantise (A/i8) -> GEMM

The two reshapes are FP16 because a conversion into or out of A-layout needs an 8-byte run
contiguous on both sides, and at int8 an A-layout tileSize run is 4 bytes and falls off the
hardware path. Quantising before the reshape would make the reshape impossible. See
comm/nest.py, which refuses it by name rather than emitting something that does not run.

The same 8-byte-run argument is why RMSNorm's transposed form has to be undone before that
second reshape, and it is worked through in norm.py.
"""

from .common import BEAT_BYTES, LANES_PER_BEAT, RowCfg, check_pow2, check_row
from .norm import NormCfg, RMSNorm
from .pointwise import Dequantize, Quantize, Residual
from .rope import RoPE

__all__ = ["BEAT_BYTES", "LANES_PER_BEAT", "RowCfg", "check_pow2", "check_row",
           "NormCfg", "RMSNorm", "Dequantize", "Quantize", "Residual", "RoPE"]
