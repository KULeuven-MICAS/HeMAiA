# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The block's parameters (README.md, PARAMETERS): every switch parsed and checked in one place."""

from bingo_kernel_args import INT4_B_LAYOUT_PAIRED

from .consts import BLK


class PParams:
    """PBuild.__init__: the parameters; no data, memory or task yet."""

    def __init__(self, args, p, plat, hw, shared=None):
        """ChipletBuild.__init__ first: the platform's and the graph's switches, the chips, each one's memory chip and its
        push engine there (with `shared`, a layer's graph and data staging). Then the block's: the ablation arms, the rows
        a pass and the passes, the X and post blocks, the gate|up chunks' and down slices' widths, the root's columns, the
        reduce's output slots and chain groups."""
        super().__init__(args, p, plat, hw, shared=shared)
        if self.bcast_impl != "bcast_l1":
            raise ValueError("params bcast: bcast_l1 (X reaches the chips through the memory chip, params x_cast)")
        # the ablation's arms (workloads/ffn/dsv2_l0_prefill_ablation), on the same mapping:
        #   x_cast     multicast: X from HBM by one multicast; unicast: the memory chip's push engine writes a copy into
        #              each chip's L3 in turn, the same tiles and flags
        #   reduce     chain: ChainGathers; root: the root's clusters pull every chip's partial of a slice into two L1
        #              slots and fold it into their own on the SIMD, in the fold order (sync barrier)
        #   sync       edge: every cross-chip dependency a BINGO hardware edge; barrier: a chip barrier after every
        #              kernel, fencing every core of the chip, kernels never overlapping (kernel.py)
        self.x_cast = str(p.get("x_cast", "multicast"))
        if self.x_cast not in ("multicast", "unicast"):
            raise ValueError(f"params x_cast={self.x_cast!r}: multicast or unicast")
        self.per_kernel = self.sync == "barrier"
        if self.reduce_impl == "root" and (not self.per_kernel or p.get("chain_alt", False)):
            raise ValueError("params reduce root: with sync barrier and no chain_alt (one fold order)")
        self.barrier_fence = self.per_kernel
        R = self.ROWS = int(p.get("rows", 32))
        if R % 16 or self.T % R:
            raise ValueError(f"params tokens={self.T} rows={R}: rows a multiple of 16 dividing T")
        self.NI, self.MT = self.T // R, R // 16
        # deq_per_row: the dequantise one task a row tile, each with the chunk's one scale tile
        self.deq_per_row = bool(p.get("deq_per_row", False))
        # x_blocks: X of the first pass fetched in this many row blocks, and the first gate|up chunk run as one GEMM per
        # block, so the arrays start when the first block has landed
        self.x_blocks = int(p.get("x_blocks", 1))
        # post_blocks: the reshape and quantise of A16 in this many row blocks, so the SIMD quantises a block while the
        # xDMA reshapes the next (the gap between gate|up and down shrinks)
        self.post_blocks = int(p.get("post_blocks", 1))
        # split_last (with post_blocks): the last gate|up chunk as one GEMM per post block, each block's dequantise and
        # SwiGLU, its reshape and the previous block's quantise right behind it, so only the last block's work is left
        # between the last gate|up GEMM and the first down GEMM
        self.split_last = bool(p.get("split_last", False))
        if self.split_last and (self.post_blocks < 2 or not self.deq_per_row):
            raise ValueError("params split_last needs post_blocks > 1 and deq_per_row")
        if self.post_blocks < self.MIN_POST_BLOCKS or self.MT % self.post_blocks:
            raise ValueError(f"params post_blocks={self.post_blocks}: at least {self.MIN_POST_BLOCKS}, dividing {self.MT} "
                             f"row tiles")
        if self.MT % self.x_blocks:
            raise ValueError(f"params x_blocks={self.x_blocks}: dividing {self.MT} row tiles")
        # split_first_dn: each cluster's first down slice as one GEMM per post block, each behind its block's quantise
        # and A8 copy only, so the end of the post-processing runs under down GEMMs
        self.split_first_dn = bool(p.get("split_first_dn", False))
        # restream: every pass streams its weight chunks from the memory chip again (a ring load each), instead of
        # copying them from the L3 slots the first pass filled; for weights larger than the ring
        self.restream = bool(p.get("restream", False))
        self.BPC = int(p.get("blocks", 3))          # 16-column gate blocks per gate|up chunk
        # first_blocks: the first gate|up chunk of every cluster this many gate blocks wide (0 = blocks): a narrow first
        # chunk is pushed sooner, so the first GEMM starts sooner
        self.first_blocks = int(p.get("first_blocks", 0))
        self.DNC = int(p.get("dn_cols", 512))       # output columns per down chunk
        # every INT4 weight chunk in the paired layout (blocks of 32 columns, layout.py to_b_pairs), which the B reader
        # walks 64 bytes a pass, as A: every GEMM's column tiles, and every down slice, come in whole pairs
        self.int4_b = INT4_B_LAYOUT_PAIRED
        self.dn_unit = 2 * BLK
        if self.DNC % self.dn_unit:
            raise ValueError(f"params dn_cols={self.DNC}: every down GEMM whole {self.dn_unit}-column units (pairs of "
                             f"column tiles)")
        # the intermediate columns split over the clusters in whole GEMM column blocks, the root chip's clusters taking
        # the smaller share, since its cluster 0 also runs every gather and stores Y. root_cols: the root chip's clusters
        # get exactly this many columns (one number for both, or one per cluster; the rest is split over the others), so
        # its extra duties (every gather and store) cost no window time
        rc = p.get("root_cols", 0)
        self.root_cols = [int(v) for v in rc] if isinstance(rc, (list, tuple)) else [int(rc)] * self.NC
        # dn_balance: each cluster's share of the outputs set by the array steps (memory.py _dn_out_split)
        self.dn_balance = bool(p.get("dn_balance", False))
        # chain_alt: cluster 1's ChainGathers run the chips in the reverse order, so a gather's data follows the previous
        # gather's finish back along the ring instead of meeting it mid-chain (a collector retires before its finish has
        # walked back to the first hop)
        self.chain_alt = bool(p.get("chain_alt", False))
        # l3_overlay: Y_raw of a pass is stored over that pass's X buffers in L3, one region a pass ([source | landing],
        # as long as Y_raw), so the image holds Y beside neither; pass i's stores follow every reader of X_i (the root's
        # put and its clusters' fetches) through the pass's GEMMs
        self.l3_overlay = bool(p.get("l3_overlay", False))
        # y_scatter: the root stores each slice of Y_raw row tile by row tile (16 rows) straight into the L3 of the chip
        # that owns the tile (tile q on the q-th chip, as a layer's tiles), behind a flag a slice there, instead of into
        # its own L3: a tile's whole row of Y_raw is then in its owner's L3 (D layout) once the pass's slices are stored,
        # and the block ends with one wait a chip on its tiles' flags
        self.y_scatter = bool(p.get("y_scatter", False))
        # out_slots (default 2): the slots of each cluster's exchange buffer the down slices go through (slice j into slot
        # j mod out_slots), so the down GEMMs run that many slices ahead of the serial gathers; the root's fold buffer
        # keeps two
        self.out_slots = int(p.get("out_slots", 2))
        if self.out_slots < 2:
            raise ValueError(f"params out_slots={self.out_slots}: at least 2")
        # chain_group (default 1): one ChainGather carries this many consecutive down slices of a cluster, the output
        # slots they sit in back to back, into a fold slot of that many slices at the root, stored to L3 by one copy (or
        # row tile by row tile with y_scatter)
        self.chain_group = int(p.get("chain_group", 1))
        # start_per_chip: a chip's scale loads one after the other, cluster 0's last, so the start waits for one task a
        # chip (on the root, the load ahead of it on its own core) instead of one a cluster
        self.start_per_chip = bool(p.get("start_per_chip", False))
        # gather_relay_only: a dep-tag cell of the xDMA tasks (the gathers) that is over budget relays its leftover edges
        # instead of ordering them, so no down GEMM waits for an earlier gather
        self.gather_relay_only = bool(p.get("gather_relay_only", False))
        if self.per_kernel:
            # kernel by kernel: a kernel's tasks never run inside another kernel (mem_handles sets the output slots)
            bad = [k for k in ("split_last", "split_first_dn", "y_scatter") if p.get(k, False)]
            if bad or "out_slots" in p:
                raise ValueError(f"params sync barrier: no {bad or 'out_slots'} (a kernel's tasks never run inside "
                                 f"another kernel)")
            if self.chain_group != 1 and self.reduce_impl != "chain":
                raise ValueError("params chain_group: with reduce chain")
        self.NTL = -(-self.T // 16 // len(self.chips))
