# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaStages' parameters: the block builder's switches, read and checked in one place."""

from fa_prefill_mc import HIDDEN

X_XDMA_SHARE = 0.3       # stage 1: the xDMA's share of each tile of x into L1, the iDMA the rest


class FaStagesParams:
    """FaStages.__init__: the parameters; no data, memory or task yet."""

    def __init__(self, args, p, plat, hw, shared=None):
        """FaBuild's parameters, then the block builder's switches with their checks."""
        if "stage" in p:
            raise ValueError("params stage: not taken (the block is built whole; each stage alone is its own app, "
                             "fa/dsv2_l1_prefill/stage<N>)")
        super().__init__(args, p, plat, hw, shared=shared)
        self.XT = 16 * HIDDEN                          # a tile of x: 16 tokens, A layout, INT8
        self.NX = self.T // 16
        # stage 1 loads every tile of x into L1 by the iDMA and the xDMA at once (whole 512-byte blocks), which reach L3
        # through two different wide ports
        self.x_xdma = int(round(self.XT * X_XDMA_SHARE / 512)) * 512
        # params x_pf_split: the root's W_KVA ring runs ahead of the multicast in its memory chip's engine queue and the
        # rest of its ring after it, by boundaries of the root's ring (_cluster_x_order), so the root's keys do not wait
        # for the multicast and the multicast does not wait for the root's stage-4 weights
        self.pf_split = bool(p.get("x_pf_split", False))
        # params chains_after_sweep: the root clusters' chains and stores of chunk i - 1 are made after
        # their sweep of chunk i, so the root's in-order xDMA and DM cores take the sweep's tasks (its softmax geometry,
        # its key and value tiles) first and the chains run under the sweep's computation
        self.chains_after_sweep = bool(p.get("chains_after_sweep", False))
        # params yr_store_tile t (y_roots): a collector cluster's store of chunk i's y block goes behind its own load of
        # chunk i + 1's key tile t's values in its DM core's program. The chains and stores are made while the root's
        # tasks are, so on a collector they would sit ahead of its chunk i + 1 front, which would wait for the chain
        self.yr_store_tile = p.get("yr_store_tile")
        # params kx_mid_double (keys_by_chunk): the key tiles of the later chunks made inside the window take two x and
        # two output buffers in turn, like chunk 0's, so a tile's load and GEMM wait for the tile two before, not the
        # one before
        self.kx_mid_double = bool(p.get("kx_mid_double", False))
        # params kc_xs_late (keys_by_chunk): a helping chip's tiles after the window take a copy of the helped pair's
        # W_KVA of their own, as its own tiles do of W_KVA, so the window's copy is free after the window's last helped
        # tile
        self.kc_xs_late = bool(p.get("kc_xs_late", False))
        # params kx_per_chunk (keys_by_chunk): the key tiles of every chunk after the first take x and output buffers,
        # a W_KVA copy and a helped-pair copy of that chunk's own (both copies from the chip's L3, filled from HBM before
        # the window), and the loads that start them depend on the cluster's last q projection of the chunk before, so
        # nothing of the key loop lives from chunk to chunk and the static L1 planner can lay each chunk's over the
        # chunk before's x rows. From three chunks on, one shared set beside a collector's held y block and the next
        # chunk's x rows overflows its L1
        self.kx_per_chunk = bool(p.get("kx_per_chunk", False))
        # params kx_after_chains (kx_per_chunk, y_roots, yr_store_tile): on every cluster, chunk h's key tasks (h >= 2)
        # behind its load of chunk h - 1's key tile yr_store_tile's values (as a collector's, behind its store), and
        # their first loads behind the last reduce chain of chunk h - 2, so the key loop's transfers (its HBM copies,
        # its puts to the owners) never run beside a chain: a chain must not meet another chip's traffic on a hop
        self.kx_after_chains = bool(p.get("kx_after_chains", False))
        # params yr_rel1 (y_roots): on each chip that collected chunk i - 1, chunk i's ring chunks are released by
        # cluster 1's read: a collector's cluster 1 runs its chain and store last, so it reads them last, and cluster
        # 0's read, the default releaser, would wait for it
        self.yr_rel1 = bool(p.get("yr_rel1", False))
        # params full_kv_all (default true): every chip waits for every chunk's keys at the end of stage
        # 3; false: the sweep waits for its own tiles only (the wait would hold cluster 0's in-order DM core ahead of
        # stage 4)
        self.full_kv_all = bool(p.get("full_kv_all", True))
        # params keys_by_chunk (x_pf_split, full_kv_all false): each chunk's keys just ahead of its queries
        # instead of every key first. On every device core, the key tasks (stages 1-3) of chunk h follow the core's
        # cluster's stage-4 tasks of chunk h - 1; those of the chunks after the last one made, everything of the window.
        # The root's host multicasts chunk 0's x, then pushes chunk 0's stage-4 weights, then multicasts the rest of x,
        # then pushes the rest of the weights
        self.keys_by_chunk = bool(p.get("keys_by_chunk", False))
        if self.keys_by_chunk and (not self.pf_split or self.full_kv_all):
            raise ValueError("params keys_by_chunk: needs x_pf_split and full_kv_all false")
        self._key_runs = []     # keys_by_chunk: (first node id, key chunk or None) of each run of key tasks, in order
        # params x_rest_early (keys_by_chunk): the rest of x (chunks 2 on) right after chunk 0's key multicast, ahead of
        # the rest of the root's ring, not behind its last run: behind it the rest lands only once the chips have
        # consumed the last chunk's weights, which needs chunk 2's keys and so its x: a full block's cycle. Only the
        # root chip's own ring waits for the rest; chunk 1's key multicast and every reduce chain follow it (one
        # multicast source at a time)
        self.x_rest_early = bool(p.get("x_rest_early", False))
        if self.x_rest_early and not self.keys_by_chunk:
            raise ValueError("params x_rest_early: needs keys_by_chunk")
        if self.kx_per_chunk and not self.keys_by_chunk:
            raise ValueError("params kx_per_chunk: needs keys_by_chunk")
        if self.kx_after_chains and not (self.kx_per_chunk and p.get("y_roots") and p.get("yr_store_tile") is not None):
            raise ValueError("params kx_after_chains: needs kx_per_chunk, y_roots and yr_store_tile")
        # params s4_q8_pairs n: the SIMD core quantises the query tiles tile by tile, q~ then q_pe, so the
        # sweep's first QK waits for one tile's two quantisations, not every tile's q_pe then q~ tile 0
        self.s4_q8_pairs = p.get("s4_q8_pairs")
        if self.s4_q8_pairs is not None and not 0 <= int(self.s4_q8_pairs) < self.NQ:
            raise ValueError("params s4_q8_pairs: a query tile index")
