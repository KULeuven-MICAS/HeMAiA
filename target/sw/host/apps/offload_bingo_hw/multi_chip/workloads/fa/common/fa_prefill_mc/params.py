# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaBuild's parameters: the shapes, the head of each chip and every switch, read and checked in one place."""

import os

from bingo_kernel_args import simd_fp16_lanes

from .consts import BR, D_QK, HIDDEN, REPO, check


class FaParams:
    """FaBuild.__init__: the parameters; no data, memory or task yet."""

    # the root cluster whose DM core multicasts the keys' pieces (stage 3)
    x_put_cluster = 1

    def __init__(self, args, p, plat, hw, shared=None):
        """ChipletBuild.__init__ first: the platform's and the graph's switches, the chips, each one's memory chip and
        its push engine there, the empty DFG, the data staging (self.st) and the stamp buffer (a layer's block shares
        the layer's). Then the block's: the chunks, their query and key tiles (check), every switch with its checks,
        the head of each chip and the fold order, the data's inputs and the sizes of a chunk's x and of a cluster's
        partial y."""
        super().__init__(args, p, plat, hw, shared=shared)
        if (self.bcast_impl, self.reduce_impl, self.sync) != ("bcast_l1", "chain", "edge"):
            raise ValueError("the FA prefill builds bcast bcast_l1, reduce chain, sync edge")
        m = check(p)
        self.NCH, self.NQ, self.NT = m["chunks"], m["query_tiles"], m["tiles"]
        self.C = int(p["chunk"])
        self.ROWS = self.C // self.NC                  # query rows a cluster
        self.keys, self.bc = m["keys"], int(p["bc"])
        # params row_swap: in every odd chunk the clusters take the other halves (causal: the late half sweeps more
        # tiles)
        self.row_swap = bool(p.get("row_swap", False))
        # params post_tiles: o_h's relayout and quantisation a query tile at a time, each behind its own W_UV GEMM, so
        # they run under the next tiles' GEMMs and W_O starts behind the last tile's
        self.post_tiles = bool(p.get("post_tiles", False))
        # params skip_masked: no key tile whose every score is masked (fa_prefill_data's tile plan)
        self.skip_masked = bool(p.get("skip_masked", False))
        # chip chips[q % 16] makes the keys of tokens 16 q .. 16 q + 15, T / 256 tiles a chip 16 tiles apart; kv_put
        # {tile: the put of its piece into the root's L3}
        self.kv_put = {}
        nch = len(self.chips)
        if self.T % (16 * nch):
            raise ValueError(f"{self.T} tokens; each of the {nch} chips makes whole 16-token tiles")
        self.kv_tiles = {k: [q for q in range(self.T // 16) if self.chips[q % nch] == k] for k in self.chips}
        # params wbits (8 | 4): the projection weights INT8 or INT4 in the paired layout (half the bytes pushed, half
        # the L1 slab)
        self.wbits = int(p.get("wbits", 8))
        if self.wbits not in (8, 4):
            raise ValueError(f"params wbits={self.wbits}: 8 or 4")
        # W_KVA by columns: chip j streams only its 16-column granules [36 j / 16, 36 (j + 1) / 16) of W_KVA's 36 (two
        # or three: one ring chunk) and makes them for every token; each tile's slice goes into the L3 of the tile's
        # owner behind a flag of its own, and the owner finishes the tile's keys on the gathered rows (INT4 streams
        # whole pairs of granules: the paired layout's blocks of 32 columns)
        pg = 2 if self.wbits == 4 else 1
        ng = D_QK // 16 // pg
        self.kvg = {k: (pg * (j * ng // nch), pg * ((j + 1) * ng // nch)) for j, k in enumerate(self.chips)}
        self.kv_owner = {q: k for k in self.chips for q in self.kv_tiles[k]}
        self.KVD = 16 * D_QK * 2                     # a tile's kv, D layout [16, 576] FP16
        self.KVR_TILES = max(len(t) for t in self.kv_tiles.values())
        if simd_fp16_lanes() != BR:
            raise ValueError(f"the SIMD has {simd_fp16_lanes()} FP16 lanes; the softmax takes one query "
                             f"a lane and the tiles are {BR} rows")
        # params stop (full | q8): the whole block, or the chunks up to their query tiles Q8 (a stage-4 window)
        self.stop = str(p.get("stop", "full"))
        if self.stop not in ("full", "q8"):
            raise ValueError(f"params stop={self.stop!r}: full or q8")
        # params kv_slabs (split | split_kb): the sweep's K tiles in slab 0 and its V tiles in slabs 1 and 2
        # alternately, or (split_kb) the odd K tiles in a one-tile buffer of each cluster instead of slab 0
        self.kv_slabs = str(p.get("kv_slabs", "split"))
        if self.kv_slabs not in ("split", "split_kb") or int(p.get("w_buffers", 3)) != 3:
            raise ValueError(f"params kv_slabs={self.kv_slabs!r}: split or split_kb, with w_buffers 3")
        self._kbuf, self._kb_reader = {}, {}
        # params wo_split_rows: W_O as GEMMs of two 16-row tiles each instead of one of all the cluster's rows
        self.wo_split_rows = bool(p.get("wo_split_rows", False))
        # params wo_row_pipe (post_tiles, wo_split_rows): each W_O GEMM of two 16-row tiles waits for those two tiles'
        # o8 only, not for the last tile's, so W_O starts while the last tiles are still relaid and quantised
        self.wo_row_pipe = bool(p.get("wo_row_pipe", False))
        if self.wo_row_pipe and not (self.post_tiles and self.wo_split_rows):
            raise ValueError("params wo_row_pipe: needs post_tiles and wo_split_rows")
        self._o8_tiles = {}
        # params y_roots: the chips that collect and store y, one a W_O block in turn (chunk i's block h:
        # y_roots[(2 i + h) % len]), so the root is not every chain's collector and store; each chain folds the cycle
        # from its own collector (default: the root collects every block)
        self.y_roots = [int(str(k), 16) for k in p.get("y_roots", [])]
        self._wo_blk = {}
        self._last_chain = None
        self._chains = []                           # every reduce chain, in creation order
        # params chain_concurrent (default true): cluster 0's and cluster 1's ChainGathers run at the same time
        # through every chip. False: every ChainGather after the previous one of either cluster, so a chain's finish
        # never meets the other cluster's chain data in a chip's D2D framer (the MoE's deadlock with a memory chip's
        # push through a hop chip). The last chunk's two clusters' chains of a W_O block run at the same time, once
        # every chip's ring has landed its last chunk (after_pushes_landed: no memory-chip push then transits a hop
        # chip, the third party of the concurrent-chain deadlock); the blocks stay in series
        self.chain_concurrent = bool(p.get("chain_concurrent", True))
        self._pair_prev = None
        # params checks (y | stages): the checks; stages adds the query tiles and o~ of the chip in the middle of the
        # fold order
        self.checks_kind = str(p.get("checks", "y"))
        # head h(k): a memory chip's row holds four consecutive heads, nearest the memory chip last
        by_row = sorted(self.chips, key=lambda k: (k & 15, k >> 4))
        self.head = {k: i for i, k in enumerate(by_row)}
        if len(self.head) != 16:
            raise ValueError(f"{len(self.head)} compute chips: the layer has 16 heads, one a chip")
        # the fold order of the 16 partials: the chips' order from the base class, as heads
        self.fold_heads = [self.head[k] for k in self.fold_order()]
        # the data's inputs: the cache of hidden states, the checkpoint's weights, the calibration tokens
        self.kv_path = os.path.join(REPO, p.get("kv_cache", "target/sim/automation/sweep/llm_blocks/fa/cache/"
                                                            "dsv2_l1_kv_8193.npz"))
        self.w_cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        self.n_calib = int(p.get("n_calib", 32))
        self.XB = self.C * HIDDEN                       # a chunk's x, INT8
        self.YB = self.ROWS * HIDDEN * 2                # a cluster's partial y of a chunk, FP16
