# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The decode's parameters (README.md, PARAMETERS) and its constants."""

ROWS = 16                                  # the GEMM shape's row granule: the prefill's sizes for one row tile
ATOM, LANES = 8, 8                         # the xDMA's 8-byte atom (4 FP16 values), atoms a beat


class DParams:
    """DBuild.__init__: the parameters; no data, memory or task yet."""

    def __init__(self, args, p, plat, hw, shared=None):
        """The decode's refusals of what needs more than one pass, row tile or post block, its switches (y_dequant,
        chains_after_stream), then PBuild.__init__ with the params of one pass of one 16-row tile (its sizes); the
        decode's token count is one."""
        if int(p.get("tokens", 1)) != 1:
            raise ValueError(f"params tokens={p['tokens']}: the decode runs one token")
        bad = [k for k in ("l3_overlay", "y_scatter", "split_last", "split_first_dn", "restream")
               if p.get(k, False)]
        bad += [k for k, v in (("rows", ROWS), ("post_blocks", 1), ("x_blocks", 1)) if int(p.get(k, v)) != v]
        bad += [f"{k} {p[k]}" for k, v in (("sync", "edge"), ("reduce", "chain"), ("x_cast", "multicast"))
                if str(p.get(k, v)) != v]
        if bad:
            raise ValueError(f"params {bad}: the decode is one pass of one {ROWS}-row tile, one post block, X from HBM "
                             f"by one multicast, the ChainGather reduce, edge sync")
        gone = [k for k in ("x_row", "dn_gemv", "gu_gemv") if k in p]
        if gone:
            raise ValueError(f"params {gone}: not taken (every weight multiply of the decode is a GEMV, X the token's "
                             f"row in a_row)")
        # y_dequant: the root's SIMD multiplies Y_raw's row by s_dn (one task, before the HBM write), so the HBM holds
        # Y16; default false: Y_raw, as the prefill leaves it
        self.y_dequant = bool(p.get("y_dequant", False))
        # chains_after_stream: the gathers are made after every down GEMV, the first behind every cluster's last one,
        # which follows the cluster's last ring load: every gather runs after every chunk of every memory chip's push
        # has landed (no gather beside a push). No slot is then reused before its gather, so out_slots holds every slice
        # of a cluster
        self.chains_after_stream = bool(p.get("chains_after_stream", False))
        # the prefill's builder runs the tile as one pass of ROWS tokens
        super().__init__(args, dict(p, tokens=ROWS, rows=ROWS, post_blocks=1, x_blocks=1), plat, hw, shared=shared)
        self.T = 1
