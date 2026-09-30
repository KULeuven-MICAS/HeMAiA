# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""What every SIMD block shares: the tile description, and the beat it has to tile to.

Small on purpose. It holds the two facts that are properties of the HARDWARE rather than
of any one operator, so that a new block picks them up instead of restating them -- and so
that when the beat width changes with the cfg there is one place to change.
"""

from dataclasses import dataclass

# One 512-bit SIMD beat is 64 B = 32 FP16 lanes, and every fused kernel here works a whole
# beat at a time. A row that is not a multiple of this has a partial final beat, which the
# kernels do not handle.
LANES_PER_BEAT = 32
BEAT_BYTES = 64


def check_row(cols: int, who: str) -> None:
    if cols % LANES_PER_BEAT:
        raise ValueError(
            f"{who}: cols={cols} must be a multiple of {LANES_PER_BEAT} -- one SIMD beat "
            f"is {BEAT_BYTES} B = {LANES_PER_BEAT} fp16 lanes, and a partial final beat is "
            f"not handled.")


def check_pow2(cols: int, who: str) -> None:
    """For the reductions that divide by the row length.

    The mean in a normalisation is an EXPONENT SUBTRACT -- on the core it was
    `(E - log2D) << 10`, and in the datapath it is a 1/D immediate the RSQRT func
    multiplies by. Both are exact at 2^k and only at 2^k. At any other width the kernel's
    `for (t = cols; t > 1; t >>= 1) log2D++` truncates, the scale comes out a power of two
    wrong, and the result is a well-formed tensor scaled by the wrong constant.
    """
    if cols <= 0 or (cols & (cols - 1)):
        raise ValueError(
            f"{who}: cols={cols} must be a POWER OF TWO. The division by the row length is "
            f"an exponent subtract, exact at 2^k and not otherwise; a non-power-of-two "
            f"silently normalises by the wrong constant.")


@dataclass(frozen=True)
class RowCfg:
    """A [rows, cols] FP16 tensor, one row per independent token position."""

    rows: int
    cols: int
    cluster: int = 0
