# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""MBuild's parameters: every switch of the block parsed and checked in one place (README.md, PARAMETERS)."""

import moe_prefill_data as md

from .common import chain_path_param


class Params:
    """The parameters; no data, memory or task yet."""

    def __init__(self, args, p, plat, hw, shared=None):
        """ChipletBuild.__init__ first: the platform's and the graph's switches (bcast, reduce, sync among them), the
        chips, each one's memory chip and its push engine there (with `shared`, a layer's graph and data staging). Then
        the block's own."""
        super().__init__(args, p, plat, hw, shared=shared)
        # THE ABLATION (workloads/moe/dsv2_l1_prefill_ablation): x_cast, reduce and sync take the paper's three
        # mechanisms off one at a time, on the same mapping
        #   x_cast     multicast: X and the router's weights from HBM by one multicast each, the route table by one
        #              multicast a row block; unicast: the memory chip's push engine writes X and the router's weights
        #              into each chip's L3 in turn (__host_bingo_kernel_hbm_ucast), and the root's DM core puts each
        #              row block of the table into each chip's L3 in turn; the same addresses and flags
        #   reduce     chain: ChainGathers; root: each root cluster pulls every other chip's partial of a slice into
        #              its two p1c slots and adds it into its own on the SIMD, in the fold order (fold_order_c)
        #   sync       edge: every cross-chip dependency a BINGO hardware edge; barrier: chip barriers instead, on each
        #              chip's cluster 0 xDMA core (its DM core's queue holds the weight loads, which wait for slabs
        #              that only the GEMMs behind a barrier free)
        #   barrier_at kernel: a barrier after every kernel, fencing every core of the chip: the route; the gate|up (the
        #              shared MLP and the experts, their SwiGLU and quantise); the A8 exchange; then the down projection
        #              and the reduce in turns, as many slices as the output slots hold. cross: the tuned schedule
        #              with a barrier before each slice's reduce, the reduce running under the next slices, and only
        #              the tasks a barrier names waiting for it
        if self.bcast_impl != "bcast_l1":
            raise ValueError("params bcast: bcast_l1 (the multicasts and their unicast counterparts: params x_cast)")
        if self.reduce_impl not in ("chain", "root"):
            raise ValueError(f"params reduce={self.reduce_impl!r}: chain or root")
        self.x_cast = str(p.get("x_cast", "multicast"))
        if self.x_cast not in ("multicast", "unicast"):
            raise ValueError(f"params x_cast={self.x_cast!r}: multicast or unicast")
        self.barrier_at = str(p.get("barrier_at", "kernel")) if self.sync == "barrier" else None
        if self.barrier_at not in (None, "kernel", "cross"):
            raise ValueError(f"params barrier_at={self.barrier_at!r}: kernel or cross")
        self.per_kernel = self.barrier_at == "kernel"
        self.cross = self.barrier_at == "cross"
        if self.reduce_impl == "root" and not self.per_kernel:
            raise ValueError("params reduce root: with sync barrier, barrier_at kernel")
        self.barrier_fence = self.per_kernel             # kernel: every core of a chip waits at a barrier
        self.barrier_role = "xdma"
        # resid (a block of a whole layer, dsv2_layer_prefill_mc.py): {chip, src, piece, h16}: the residual (h16, FP16
        # [T, H], in L3 at src in the output's slice layout) added into that chip's partial of every slice, `piece`
        # bytes at a time
        self.resid = getattr(self, "resid", None)
        if self.sync == "barrier" and self.resid is not None:
            raise ValueError("params sync barrier: the block alone, without a residual")
        nch = len(self.chips)
        if md.N_EXP % nch:
            raise ValueError(f"{md.N_EXP} experts over {nch} chips")
        self.EPC = md.N_EXP // nch
        self.BPC = int(p.get("blocks", 3))                 # gate blocks a gate|up chunk
        self.SW = int(p.get("slice", 128))                 # output columns a down slice
        self.XB = int(p.get("x_blocks", 4))                # row blocks of X (the shared MLP's passes)
        if self.T % (8 * self.XB):
            raise ValueError(f"params x_blocks={self.XB}: whole 8-row tiles of {self.T} tokens a block")
        self.XR = self.T // self.XB
        self.NXT = self.T // 8                             # X's multicast tiles, 8 tokens each
        self.SL = self.T * self.SW * 2                     # one output slice, FP16
        # params chain_path: the chips every ChainGather passes, in data order, as hex chip ids, the root last
        # (default the fold cycle). Pushes run along the rows from the memory chips, so the path's steps between
        # columns are the ones that share a link with them.
        self.chain_path = chain_path_param(p, self.chips, self.ROOT)
        # params args_phases (the block alone): args phases as a layer has them (bingo_dfg_emit.py ARGS PHASES), one
        # boundary before the down projection: a cluster's args of each phase share one L1 region, loaded by a DM task
        # after every task of the previous phase on the chip
        self.args_phases = bool(p.get("args_phases", False))
        self.phase_starts, self.phase_names, self.loaders = [], ["MoE route and gate|up"], {}
        # params shared_after: the shared MLP's gate|up one weight chunk at a time over every X row block, the first
        # chunk as X arrives and chunk j >= 1 after expert slot shared_after[j - 1]'s gate|up (X's blocks copied again
        # from the chip's L3), so no stretch of the shared MLP holds both weight slabs while the ring fills up behind it
        sa = p.get("shared_after")
        self.shared_after = None if sa is None else [int(x) for x in sa]
        # params expert_rows (default 0: off): an expert slot of more rows runs its gate|up in row blocks of at most this
        # many (whole 8-row tiles, as even as they come), each its own pass: its rows gathered into the next row buffer
        # (xe), the expert's gate|up chunks streamed again, its rows of A8 quantised; the down projection still takes
        # the slot's rows at once. The row buffers then hold a block, not the busiest expert's rows
        self.expert_rows = int(p.get("expert_rows", 0))
        if self.expert_rows < 0 or self.expert_rows % 8:
            raise ValueError(f"params expert_rows={self.expert_rows}: whole 8-row tiles")
        # params slot_order: the order of a chip's expert slots, as a list of ranks: slot i holds the chip's expert of
        # rank slot_order[i], ranks counted from the fewest padded rows (ties by expert id), e.g. [1, 2, 3, 0] puts the
        # expert with the fewest rows last. Default: the placement's order (most padded rows first)
        so = p.get("slot_order")
        self.slot_order = None if so is None else [int(x) for x in so]
        if self.slot_order is not None and sorted(self.slot_order) != list(range(self.EPC)):
            raise ValueError(f"params slot_order={so}: a permutation of 0..{self.EPC - 1}")
