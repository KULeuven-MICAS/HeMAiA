# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DBuild's parameters (README.md, PARAMETERS)."""

from moe_prefill_mc.common import chain_path_param

# the prefill's switches the decode does not take
PREFILL_ONLY = ("x_cast", "barrier_at", "args_phases", "x_blocks", "expert_rows", "slot_order", "shared_after")


class Params:
    """The parameters; no data, memory or task yet."""

    def __init__(self, args, p, plat, hw, shared=None):
        """ChipletBuild.__init__ first: the platform's and the graph's switches; then the decode's: the reduce's chain
        path and the gate blocks a gate|up chunk (the placement's routed_split and the slice are read with the data)."""
        super().__init__(args, p, plat, hw, shared=shared)
        if self.T != 1:
            raise ValueError(f"params tokens={self.T}: the decode runs one token")
        if self.bcast_impl != "bcast_l1" or self.reduce_impl != "chain" or self.sync != "edge":
            raise ValueError("this driver builds bcast bcast_l1, reduce chain, sync edge")
        for k in PREFILL_ONLY:
            if k in p:
                raise ValueError(f"params {k}: a prefill switch the decode does not take")
        self.chain_path = chain_path_param(p, self.chips, self.ROOT)
        self.BPC = int(p.get("blocks", 3))                 # at most this many column blocks a gate|up chunk
