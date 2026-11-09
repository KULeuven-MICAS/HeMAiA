# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The attention of one Llama-3-8B layer (grouped-query, 32 query heads, 8 key/value heads of 128) over a fresh prompt,
causal, on the 16 + 4 chiplet platform, every GEMM on the (16, 4, 16) shape with INT4 weights. The platform, the weight
rings, the stamps and the checks are chiplet_mc.py's; the data and the golden are llama3_data.py's (LlamaAttnData); the
sweep is libs/blocks/fa_sweep.py's.

THE SPLIT. Cluster g runs query head h(g). A key/value group's four query heads sit on two chips of one compute column,
(x, 2 y) and (x, 2 y + 1); on each of them cluster 0 projects the group's K and cluster 1 its V (so the group's K and V
are made once on each of its chips), and the two clusters of a chip trade them through the chip's L3. Nothing of the
attention leaves a chip until the output projection.

THE PASSES. The prompt runs in passes of `rows` tokens, one after the other; pass i on every cluster:

    x          the root multicasts the pass's rows (INT8, A layout) into every chip's L3; each cluster fetches them
    proj       per weight chunk of 32 columns: x . W chunk (D layout, FP16), dequantised; q and k rotated (RoPE: the
               pairs swapped on the iDMA, x cos + swap(x) sin_signed on the SIMD), reshaped D -> A on the xDMA
    q8, k8     q quantised one query tile at a time; k into the cluster's key tiles at the pass's place
    v8         v reshaped D -> row-major, each tile of `bc` keys transposed and reshaped into V^T's A layout, quantised
               into the value tiles at the pass's place
    trade      cluster 0's new key tiles and cluster 1's new value tiles through the chip's L3 into the other cluster
    sweep      fa_sweep over key tiles 0 .. the pass's last: S^T = K Q^T, the causal mask added where a key is after a
               query, the online softmax, O^T += V^T P^T; then o~ = O / l, one query a row
    o8         o~ reshaped row -> A and quantised into the chip's o8 at the cluster's head; the other cluster's head
               comes through the chip's L3, so each cluster holds both of the chip's heads
    wo         per slice of `dn_cols` outputs of the cluster's half: o8 (both heads) . W_O slice (K = 256) into a slot
               of the cluster's symmetric buffer: the chip's partial of those outputs
    reduce     each slice's partials folded into the root's cluster c by one ChainGather through every chip (one at a
               time over the whole layer), and stored into the root's L3 slice by slice

THE ORDERS. Every device core runs its tasks in creation order, a queued GEMM's wait behind the next start of its phase
(GemmQueue.program_order). The weights of a cluster land once in its ring and later passes copy them from their slots.
X of pass i + 1 is multicast before pass i's first gather, so no multicast meets a gather. No cluster writes another's
L1: the trades go through the chip's L3.

PARAMETERS (params.hjson): tokens, rows, passes (the first passes only), bc, dn_cols (outputs a W_O slice), w_chunk_bytes, w_buffers, weight_ring, seed,
w4_paired, the platform's and the graph's switches (static_l1, l1_capacity, compact_id_maps, gid_lists, per_chip_image,
desc_in_image, d2d_ddr), checks (y | stages); cfg_reuse and gemm_shape_l1 (configuration costs, default false);
gemm_reuse, gemm_cfg_ahead, gemm_cfg_hoist, fa_cfg_ahead and mask_c (the FA block's stage-2 switches, fa_prefill_mc.py;
default false);
kv_push (with kv_split): each chip's halves pushed into its mate's L3 instead of read from there.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "common"))
from driver import repo_root, run_driver  # noqa: E402
REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    INT4_B_LAYOUT_B, INT4_B_LAYOUT_PAIRED, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelGemmShapeInitArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelIdmaPairwiseSwapArgs,
    SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args,
    SnaxBingoKernelWaitFlagsArgs,
    SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelXdma1dCopyArgs, SnaxBingoKernelXdmaChainGatherArgs,
    SnaxBingoKernelXdmaTranspose2dArgs, SnaxBingoKernelXdma6dArgs,
    simd_beats, simd_fp16_lanes)
from bingo_cfg_reuse import prime_fa_softmax, reuse_gemm_fa_config, reuse_gemm_start_config  # noqa: E402
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs.blocks import FaSweepCfg, GemmQueue, fa_sweep, fa_sweep_alloc  # noqa: E402
from libs.comm.nest import convert_args, d_to_a_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import llama3_data  # noqa: E402

BLK = 16                      # a GEMM column block (meshCol)
BR = 16                       # query rows a tile: the SIMD's FP16 lanes
QC = 32                       # columns a projection weight chunk
TILE16 = 16 * 16 * 2          # one D-layout FP16 block
EW_F16_ADD = cm.EW_F16_ADD


class L1Tiles:
    """fa_sweep's tile source over a cluster's L1-resident key and value tiles: tile j is ready after the task that
    put it there, and nothing is ever released (the tiles stay for the later passes)."""

    def __init__(self, ab, g):
        self.ab, self.g = ab, g

    def fetch(self, kind, j):
        ab = self.ab
        buf = ab.kbuf[self.g] if kind == "K" else ab.vbuf[self.g]
        return ab.tile_ready[(self.g, kind, j)], buf.view(j * ab.TILE_B), None

    def release(self, token, task):
        return None


class ABuild(cm.ChipletBuild):
    TAG = "la"

    def __init__(self, args, p, plat, hw, shared=None):
        super().__init__(args, p, plat, hw, shared=shared)
        if simd_fp16_lanes() != BR:
            raise ValueError(f"the SIMD has {simd_fp16_lanes()} FP16 lanes; the softmax takes one query a lane")
        self.R = int(p.get("rows", 64))
        self.bc = int(p.get("bc", 32))
        self.SLC = int(p.get("dn_cols", 256))
        # params wo_cols: W_O by output columns instead of by heads: every head's o8 into the root's L3 (each cluster's
        # own, behind a flag; the pass's x landing area holds them, dead by then), one multicast of all of them from the
        # root, and each cluster's GEMMs make its 128 columns of y over the full depth, in slices of 32, stored straight
        # into the root's y: no reduce
        self.wo_cols = bool(p.get("wo_cols", False))
        if self.wo_cols:
            # params wo_slice (32 | 128): the width of y's slices in the root's L3; 128, a cluster's columns in one slice,
            # gives the glue's row loads runs of 128 columns instead of 32
            self.SLC = int(p.get("wo_slice", 32))
            if self.SLC not in (32, 128):
                raise ValueError(f"params wo_slice={self.SLC}: 32 or 128")
        # params y_scatter (wo_cols, wo_slice 128): each cluster's y columns of a pass go, row tile by row tile, straight
        # into the L3 of the chip that owns the tile (tile q on the q-th chip, as a layer's tiles), behind a flag there,
        # instead of into the root's y: a tile's whole row of y is then in its owner's L3 (D layout) when the pass ends,
        # and the block ends with one wait a chip on its own tiles' flags
        self.y_scatter = bool(p.get("y_scatter", False))
        if self.y_scatter and not (self.wo_cols and self.SLC == 128):
            raise ValueError("params y_scatter: needs wo_cols with wo_slice 128 (a cluster's columns in one slice)")
        # params wo_early (y_scatter): the root multicasts every head's o8 before its own gathers of them (no y store lands
        # over them), and each cluster loads its first W_O chunk before it waits for the multicast
        self.wo_early = bool(p.get("wo_early", False))
        if self.wo_early and not self.y_scatter:
            raise ValueError("params wo_early: needs y_scatter (else the y stores land over the o8 the root still reads)")
        # params rope_pairs: swap (each 32-column chunk of q and k holds whole pairs (2j, 2j + 1); RoPE swaps them on the
        # iDMA) or chunks (chunk 2t holds the first dimension of 32 pairs, chunk 2t + 1 the second, in the same order:
        # RoPE of the two chunks reads each from the other at the same offsets, with no swap). q and k columns take the
        # same order, so S = K Q^T, the sweep and y are unchanged
        self.rope_pairs = str(p.get("rope_pairs", "swap"))
        if self.rope_pairs not in ("swap", "chunks"):
            raise ValueError(f"params rope_pairs={self.rope_pairs!r}: swap or chunks")
        # params kv_split (rope_pairs chunks): the two chips of a key/value group each project half of its K (cluster 0) and
        # half of its V (cluster 1), whole chunk pairs of RoPE, into their own L3, and every cluster of the group takes the
        # halves it lacks from there (the other chip's by a remote read), instead of both chips projecting all of K and V
        self.kv_split = bool(p.get("kv_split", False))
        # params cfg_reuse: the sweep's tasks skip the accelerator CSRs their core's previous task left in place
        # (bingo_cfg_reuse, after the program order); gemm_shape_l1: every gemm_start reads the array shape's
        # parameters from an L1 copy a cluster (gemm_shape_init) instead of L3
        self.cfg_reuse = bool(p.get("cfg_reuse", False))
        self.gemm_shape_l1 = bool(p.get("gemm_shape_l1", False))
        self.gsh = {}
        # params gemm_reuse, gemm_cfg_ahead, fa_cfg_ahead, mask_c: as the FA block's (fa_prefill_mc.py). gemm_reuse: a
        # gemm_start that repeats its cluster's previous dispatch writes only its addresses; gemm_cfg_ahead / fa_cfg_ahead:
        # a dispatch as a configure task right before a start task (blocks/gemm_async.py cfg_ahead); mask_c: each masked
        # score tile's mask as its QK matmul's INT32 C (fa_sweep mask_c) instead of a SIMD add
        self.gemm_reuse = bool(p.get("gemm_reuse", False))
        self.gemm_cfg_ahead = bool(p.get("gemm_cfg_ahead", False))
        self.fa_cfg_ahead = bool(p.get("fa_cfg_ahead", False))
        self.gemm_cfg_hoist = bool(p.get("gemm_cfg_hoist", False))
        if self.gemm_cfg_hoist and not self.gemm_cfg_ahead:
            raise ValueError("params gemm_cfg_hoist: needs gemm_cfg_ahead")
        self.mask_c = bool(p.get("mask_c", False))
        # params ot8_fused: the sweep's normalisation quantises o~ straight into the A layout of each 16-row query tile
        # (fa_sweep ot8, simd_mla_normalise_q8), so neither the sweep's transpose nor stage_post's relayout and
        # quantisation run; stage_post then copies each tile into the head's o8 on the xDMA
        self.ot8_fused = bool(p.get("ot8_fused", False))
        # params sm_geom_xdma: each softmax arena's geometry built on the xDMA core ahead of the arena's first softmax
        # (fa_sweep geom_core), off the SIMD core
        self.sm_geom = "xdma" if bool(p.get("sm_geom_xdma", False)) else None
        # params sm_batch (mask_c): a key tile's softmaxes of every query tile as one SIMD task (fa_sweep sm_batch,
        # simd_fa_softmax_n): one dispatch a key tile instead of one a query tile
        self.sm_batch = bool(p.get("sm_batch", False))
        # params sm_async: each softmax of the sweeps fires and returns, a wait task retires it (fa_sweep sm_async)
        self.sm_async = bool(p.get("sm_async", False))
        if self.sm_batch and (self.sm_geom or not self.mask_c):
            raise ValueError("params sm_batch: needs mask_c, and not sm_geom_xdma")
        if self.kv_split and self.rope_pairs != "chunks":
            raise ValueError("params kv_split: needs rope_pairs chunks (a half of a head's columns then holds whole pairs)")
        # params wo_under_proj (wo_cols): pass i + 1's x fetch and projections go between pass i's o8 puts and the root's
        # wait for them, so the array runs them while the o8 are gathered and multicast; pass i's gather of every head's
        # o8 then lands in pass i + 1's x buffer, once those projections are done with it (no L1 beside)
        self.wo_under_proj = bool(p.get("wo_under_proj", False))
        # params wo_load_ahead (wo_cols): each W_O chunk is loaded right after the GEMM before it, ahead of that GEMM's y
        # store on the DM core, so the next GEMM's weights do not wait behind the store
        self.wo_load_ahead = bool(p.get("wo_load_ahead", False))
        # params o_get_xdma (wo_cols): each cluster's gather of every head's o8 into L1 on its xDMA core instead of its DM
        # core, so the DM core's multicast (on the root) and weight loads run beside it
        self.o_get_xdma = bool(p.get("o_get_xdma", False))
        # params o_overlap (y_scatter, wo_early): pass i's front on each DM core (its o8 put, its first W_O chunk, the
        # root's wait and multicast, the others' wait) goes ahead of pass i - 1's y stores, and the gather no longer
        # waits for pass i - 1's GEMMs: the static planner keeps the two passes' o8 apart, so pass i's gather and
        # multicast run under pass i - 1's last GEMMs
        self.o_overlap = bool(p.get("o_overlap", False))
        # params x_ahead (stage1234): pass i's x fetch, RoPE table and first weight chunk go ahead of pass i - 1's K/V trade
        # in each DM core's program. x still lands over pass i - 1's once its projections are done (two passes' x do not
        # fit beside the weight slabs); the RoPE table no longer waits for pass i - 1's RoPE (the planner keeps the two
        # apart), so the fetch starts when pass i - 1's last GEMM ends, not behind the trade
        self.x_ahead = bool(p.get("x_ahead", False))
        if self.o_overlap and not (self.wo_early and self.wo_cols):
            raise ValueError("params o_overlap: needs wo_early (and so y_scatter, whose stores follow a pass's last GEMM)")
        if self.wo_under_proj and not self.wo_cols:
            raise ValueError("params wo_under_proj: needs wo_cols")
        # params x_early (wo_under_proj): pass i + 1's x fetch goes before pass i's sweep in the DM core's program, not
        # after pass i's o8 puts, so x is in L1 when the next pass's projections start
        self.x_early = bool(p.get("x_early", False))
        # params kv_fence (kv_split): every reader of a half in L3 waits for the next iDMA task on its writer's DM core,
        # which drains that core's earlier transfers to their write responses before it starts: an iDMA task is done
        # once its last write beat has left, and a read from another master may pass a write still on its way to L3
        self.kv_fence = bool(p.get("kv_fence", False))
        if self.kv_fence and not self.kv_split:
            raise ValueError("params kv_fence: needs kv_split")
        # params kv_push (kv_split): GQA's sharing of a group's K and V between the pair's two chips as a push. Each chip
        # puts its half of the pass's key tiles and of each value tile into its mate's L3 as well (a flagged unicast put
        # from L1), and every cluster copies the mate's halves from its own chip's L3 once their flags are there, instead
        # of reading them from the mate's L3: no remote read, and the flag is written after the data, so no fence
        self.kv_push = bool(p.get("kv_push", False))
        if self.kv_push and (not self.kv_split or self.kv_fence):
            raise ValueError("params kv_push: needs kv_split, and replaces kv_fence")
        if self.x_early and not self.wo_under_proj:
            raise ValueError("params x_early: needs wo_under_proj")
        # params o_mc_first (wo_under_proj, wo_early): the root's wait for a pass's o8 and its multicast go before the
        # next pass's front in the root's DM core's program (stage_wo_cols)
        self.o_mc_first = bool(p.get("o_mc_first", False))
        if self.o_mc_first and not (self.wo_under_proj and self.wo_early):
            raise ValueError("params o_mc_first: needs wo_under_proj and wo_early")
        # params x_after_sm (x_early): pass i + 1's x fetch waits for pass i's last softmax on the cluster and follows the
        # sweep's tasks on the DM core, so its copy into L1 runs beside the sweep's tail, not beside its softmaxes
        self.x_after_sm = bool(p.get("x_after_sm", False))
        if self.x_after_sm and not self.x_early:
            raise ValueError("params x_after_sm: needs x_early")
        # params o_get_ahead (wo_under_proj, o_get_xdma): pass i's gather of every head's o8 goes right behind pass i + 1's
        # last projection GEMM in the xDMA core's program, ahead of that pass's layout conversions made after it
        self.o_get_ahead = bool(p.get("o_get_ahead", False))
        if self.o_get_ahead and not (self.wo_under_proj and self.o_get_xdma):
            raise ValueError("params o_get_ahead: needs wo_under_proj and o_get_xdma")
        # params trade_ahead (wo_under_proj): pass i + 1's K/V trade goes ahead of pass i's y stores in each DM core's
        # program, so a cluster's keys and values leave once they are made, not once its W_O GEMMs are done
        self.trade_ahead = bool(p.get("trade_ahead", False))
        if self.trade_ahead and not (self.wo_under_proj and self.y_scatter):
            raise ValueError("params trade_ahead: needs wo_under_proj and y_scatter")
        # params front_first (wo_under_proj): pass i + 1's RoPE table and first two weight chunks go ahead of the root's
        # wait for pass i's o8 and its multicast ("root", needs o_mc_first), or ahead of pass i's o8 put on every DM core
        # ("all"), or around pass i + 1's x fetch on every DM core ("x", with x_early: the first chunk right ahead of the
        # fetch, the RoPE table and the second chunk right behind it, so the first GEMM's operands come first),
        # so no cluster's next projections wait behind the o8 round trip
        self.front_first = p.get("front_first", False)
        if self.front_first not in (False, "root", "all", "x"):
            raise ValueError(f"params front_first: false, \"root\", \"all\" or \"x\", not {self.front_first!r}")
        if self.front_first and not (self.wo_under_proj and (self.o_mc_first or self.front_first != "root")):
            raise ValueError("params front_first: needs wo_under_proj (\"root\": and o_mc_first)")
        if self.front_first == "x" and not self.x_early:
            raise ValueError("params front_first \"x\": needs x_early (the next pass's x fetch is made before the front)")
        # params front_after_sm (front_first "x", x_after_sm): the next pass's first weight chunk waits for this pass's
        # last softmax as its x does, so no load into L1 runs beside the softmaxes
        self.front_after_sm = bool(p.get("front_after_sm", False))
        if self.front_after_sm and not (self.front_first == "x" and self.x_after_sm):
            raise ValueError("params front_after_sm: needs front_first \"x\" and x_after_sm")
        self.x_device = bool(p.get("x_device", False))
        # x comes from HBM, as DSv2's block: the root memory chip's push engine multicasts each pass's x from its HBM,
        # configured by the root's host (one host task a pass, all of them at the start, ahead of the root's weight
        # prefetcher, which drives the same engine); or (x_device) a device task makes it in C00's L3 and C00 cluster 0's
        # iDMA multicasts it. x is never put in L3 before the window. x_tile_bytes: the multicast's tile (a flag after
        # each); x_pf_split: the root's prefetcher in two parts, its first x_pf_chunks chunks of each ring before the
        # multicasts, the rest after (host_kernel_lib.h weight_prefetch_range)
        if p.get("x_bcast", "memchip") != "memchip":
            raise ValueError(f"params x_bcast={p.get('x_bcast')!r}: x comes from HBM (memchip), never preloaded in L3")
        self.x_mem = not self.x_device
        self.XT = int(p.get("x_tile_bytes", 32768))
        self.pf_split = bool(p.get("x_pf_split", False))
        if self.pf_split and not self.x_mem:
            raise ValueError("params x_pf_split: needs x from HBM, not from a device task (x_device)")
        # params x_split_frac: each pass's x into L1 by the iDMA and the xDMA at once, the xDMA's part x_split_frac of it
        # (whole 512-byte blocks), both behind one wait for the multicast's last tile on the DM core
        self.x_split_frac = float(p.get("x_split_frac", 0))
        if not 0 <= self.x_split_frac < 1:
            raise ValueError(f"params x_split_frac={self.x_split_frac}: the xDMA's part, 0 (off) to below 1")
        if self.x_split_frac and not self.x_mem:
            raise ValueError("params x_split_frac: needs x from HBM (x_device's root fetch is its own multicast's)")
        # params x_row_tiles: each pass's x fetched a row tile (16 rows) at a time, each behind its own multicast tiles'
        # flags, and the pass's first projection GEMM a row tile at a time, each behind its own row tile, so the GEMMs
        # start once the first row tile is in L1 and the rest of the copy runs beside them
        self.x_row_tiles = bool(p.get("x_row_tiles", False))
        if self.x_row_tiles and (self.x_split_frac or not self.x_mem):
            raise ValueError("params x_row_tiles: needs x from HBM and not x_split_frac")
        # params x_tiles_xdma (x_row_tiles, x_after_sm): from the second pass on, each row tile of x a wait for its flags
        # and a copy on the xDMA core, not a fetch on the DM core, ahead of the previous pass's o8 copies in that core's
        # program: the xDMA's TCDM ports are arbitrated (a policy can put them last), the iDMA's wide port is not. The
        # first pass's x stays on the DM core, ahead of its ring loads (check_core_order)
        self.x_tiles_xdma = bool(p.get("x_tiles_xdma", False))
        if self.x_tiles_xdma and not (self.x_row_tiles and p.get("x_after_sm", False)):
            raise ValueError("params x_tiles_xdma: needs x_row_tiles and x_after_sm")
        # params x_xdma_free (x_tiles_xdma): the row tiles do not wait for the pass's last softmax, so they run beside the
        # sweep once the xDMA core has made the sweep's shapes
        self.x_xdma_free = bool(p.get("x_xdma_free", False))
        if self.x_xdma_free and not self.x_tiles_xdma:
            raise ValueError("params x_xdma_free: needs x_tiles_xdma")
        # params o_row_tiles (o_get_xdma, wo_cols): each cluster's gather of every head's o8 a row tile (16 rows) at a
        # time, and the pass's first W_O GEMM a row tile at a time, each behind its own row tile, so the W_O GEMMs start
        # once the first row tile is in L1 and the rest of the gather runs beside them
        self.o_row_tiles = bool(p.get("o_row_tiles", False))
        if self.o_row_tiles and not (self.o_get_xdma and self.wo_cols and not p.get("o_get_split_frac", 0)):
            raise ValueError("params o_row_tiles: needs o_get_xdma and wo_cols, not o_get_split_frac")
        # params o_get_split_frac (o_get_xdma): each cluster's gather of every head's o8 by the xDMA and the iDMA at once,
        # the iDMA's part o_get_split_frac of the heads
        self.o_get_split = float(p.get("o_get_split_frac", 0))
        if not 0 <= self.o_get_split < 1:
            raise ValueError(f"params o_get_split_frac={self.o_get_split}: the iDMA's part of the heads, 0 (off) to below 1")
        self.x_ready = {}
        # params restream: every pass streams its weight chunks from the memory chip again instead of copying them from the
        # L3 slots the first pass filled (the ring then need not hold all of a cluster's chunks)
        self.restream = bool(p.get("restream", False))
        # params load_ahead: a projection chunk's weights are loaded right after the GEMM before it is made, ahead of that
        # GEMM's RoPE swap on the DM core, so the load runs under the GEMM instead of between two
        self.load_ahead = bool(p.get("load_ahead", False))
        # params l3_overlay: a pass's y is stored over that pass's x source and landing area in L3 ([x src | x land], as
        # long as y), so the image holds y beside neither; pass i's stores follow every reader of x_i (the multicast and
        # the fetches, through the pass's GEMMs)
        self.l3_overlay = bool(p.get("l3_overlay", False))
        # THE ABLATION (workloads/fa/llama3_prefill_ablation): params x_cast and sync take the paper's broadcast and
        # per-edge synchronisation off, on the same mapping (a head-parallel prefill has no partials to reduce).
        #   x_cast     multicast: each pass's x by one multicast of the memory chip and every head's o8 by one multicast
        #              of the root; unicast: a copy a chip in turn, x by the memory chip's push engine (the same tiles and
        #              flags, __host_bingo_kernel_hbm_ucast), o8 by a flagged put a chip on the root's DM core
        #   sync       edge; barrier: a chip barrier (on cluster 0's xDMA core) after every kernel of a pass -- x and the
        #              projections, the K/V trade, the sweep, the gather and broadcast of the o8, W_O and the y stores --
        #              fencing every core of the chip, so no two kernels overlap and no dependency edge crosses chips
        #              inside the window
        self.x_cast = str(p.get("x_cast", "multicast"))
        if self.x_cast not in ("multicast", "unicast"):
            raise ValueError(f"params x_cast={self.x_cast!r}: multicast or unicast")
        if self.x_cast == "unicast" and not (self.x_mem and self.wo_cols):
            raise ValueError("params x_cast unicast: needs x from HBM and wo_cols (the o8 broadcast it replaces)")
        self.per_kernel = self.sync == "barrier"
        if self.per_kernel:
            if p.get("barrier_at", "kernel") != "kernel":
                raise ValueError("params barrier_at: kernel only (a barrier after every kernel of a pass)")
            overlap = [n for n in ("x_early", "wo_under_proj", "o_mc_first", "o_get_ahead", "trade_ahead", "front_first",
                                   "x_after_sm", "front_after_sm", "x_tiles_xdma", "x_xdma_free", "wo_early", "o_row_tiles",
                                   "o_overlap", "x_ahead") if p.get(n, False)]
            if overlap:
                raise ValueError(f"params sync barrier: no two kernels overlap, so none of {overlap}")
            if not (self.wo_cols and self.y_scatter):
                raise ValueError("params sync barrier: needs wo_cols and y_scatter")
            self.barrier_role = "xdma"
        self.checks_kind = str(p.get("checks", "y"))
        self.w4_paired = bool(p.get("w4_paired", True))
        self.int4_b = INT4_B_LAYOUT_PAIRED if self.w4_paired else INT4_B_LAYOUT_B
        d = self.d = llama3_data.LlamaAttnData(tokens=self.T, rows=self.R, bc=self.bc,
                                               seed=int(p.get("seed", 1)), wbits=self.wbits,
                                               mask_c=bool(p.get("mask_c", False)))
        self.lay, self.mesh = d.layout, d.mesh
        self.NP, self.HD, self.H = d.NP, d.HD, d.H
        # params passes: build only the prompt's first `passes` passes (a proof of concept of a long prompt in a short
        # window). The data and its scales, and the L1 buffers of every key and value tile, stay the whole prompt's; the
        # L3 and HBM buffers a pass needs are made for the built passes only, and the checks read those passes' y
        self.NP_T = d.NP
        self.NP = min(d.NP, int(p.get("passes", d.NP)))
        self.NQ = self.R // BR                          # query tiles a pass
        self.TPP = self.R // self.bc                    # key tiles a pass
        self.NQC = self.HD // QC                        # weight chunks a projection
        # the head's dimensions in the order the q and k chunks hold them (rope_pairs)
        if self.rope_pairs == "chunks":
            if self.NQC % 2:
                raise ValueError(f"params rope_pairs chunks: {self.NQC} chunks a head, not whole chunk pairs")
            first = np.arange(QC) * 2
            self.perm = np.concatenate([np.concatenate([t * 2 * QC + first, t * 2 * QC + first + 1])
                                        for t in range(self.NQC // 2)])
        else:
            self.perm = np.arange(self.HD)
        self.HALF = self.H // self.NC                   # outputs a cluster's W_O
        if self.HALF % self.SLC or self.SLC % (2 * BLK):
            raise ValueError(f"params dn_cols={self.SLC}: whole 32-column pairs dividing a cluster's {self.HALF} outputs")
        self.NS = self.HALF // self.SLC                 # W_O slices a cluster a pass
        self.WOC = self.H // (len(self.chips) * self.NC)  # wo_cols: y's columns a cluster
        self.NWS = self.WOC // 32                       # wo_cols: its GEMMs of 32 columns
        if len(self.chips) * self.NC != d.NH:
            raise ValueError(f"{len(self.chips) * self.NC} clusters: the attention has {d.NH} query heads, one a cluster")
        # ---- the split: key/value groups on chip pairs of a column, a query head a cluster ---------------------------
        pairs = sorted({(k, k + 1) for k in self.chips if (k & 15) % 2 == 0 and k + 1 in self.chips})
        if len(pairs) != d.NKV:
            raise ValueError(f"{len(pairs)} chip pairs (x, 2y), (x, 2y + 1): the attention has {d.NKV} key/value groups")
        self.group, self.head = {}, {}
        for m, (ka, kb) in enumerate(pairs):
            for n, (k, c) in enumerate(((ka, 0), (ka, 1), (kb, 0), (kb, 1))):
                g = self.chips.index(k) * self.NC + c
                self.group[k] = m
                self.head[g] = m * d.GQ + n
        self.kind = {g: "K" if self.c(g) == 0 else "V" for g in range(self.G)}     # the projection besides q
        self.kv_half, self.kv_mate = {}, {}                                         # kv_split: a chip's half, its pair
        for ka, kb in pairs:
            self.kv_half[ka], self.kv_half[kb] = 0, 1
            self.kv_mate[ka], self.kv_mate[kb] = kb, ka
        if self.kv_split and self.NQC % 4:
            raise ValueError(f"params kv_split: {self.NQC} chunks a head, not two halves of whole chunk pairs")
        self.NKVC = self.NQC // 2 if self.kv_split else self.NQC   # the k or v chunks a cluster projects
        self.WO0 = self.NQC + self.NKVC                             # its first W_O chunk in its weight stream
        self.chip_heads = {k: [self.head[g] for g in self.gs(k)] for k in self.chips}
        # ---- the golden of the reduce: the chips' partials folded in the chain's order -------------------------------
        self.gold_y = (d.part(list(range(d.NH))) if self.wo_cols else
                       d.fold([d.part(self.chip_heads[k]) for k in self.fold_order()]))
        for line in d.report(d.y16(self.gold_y)):
            print(f"[{self.TAG}] {line}")
        print(f"[{self.TAG}] {self.T} tokens in {self.NP_T} passes of {self.R} ({self.NP} built): {self.NQ} query tiles, {self.TPP} key tiles "
              f"of {self.bc} a pass; W_O " + (f"by columns, {self.WOC} a cluster in slices of {self.SLC}" if self.wo_cols else
              f"in {self.NS} slices of {self.SLC} a cluster") + "; fold order "
              f"{[f'{k:02x}' for k in self.order]}; key/value groups {[(f'{a:02x}', f'{b:02x}') for a, b in pairs]}")
        self.XB = self.R * self.H                       # a pass's x, INT8
        self.YB = self.R * self.H * 2                   # a pass's y, FP16
        self.SLB = self.R * self.SLC * 2                # one reduced slice, FP16
        self.TILE_B = self.bc * self.HD                 # a key or value tile, INT8
        self.HKB = self.TILE_B * self.NKVC // self.NQC  # the part of one a cluster projects (kv_split: half)
        self.YT_B = BR * self.H * 2                     # y_scatter: a 16-row tile of y, D layout, FP16
        self.NTL = -(-self.T // BR // len(self.chips))  # y_scatter: the tiles a chip owns
        self._stage()
        self.make_streams()
        self.make_rings()
        self._buffers()

    def pass_start(self, i):
        """Where the tasks of pass i (i >= 1) begin, in the order they are made: nothing here; a layer starts an args
        phase (llama3_prefill_mc.py)."""

    def xsrc(self, off=0, chip_id=None):
        """The root's x source (INT8, A layout a pass) at byte `off` of all passes' x: what a layer's input norm writes
        (params l3_overlay: each pass's in its region of the overlay symbol)."""
        if self.l3_overlay:
            return BingoMemSymbol(f"{self.TAG}_xy", offset=off // self.XB * self.YB + off % self.XB, chip_id=chip_id)
        return BingoMemSymbol(f"{self.TAG}_xsrc", offset=off, chip_id=chip_id)

    def xland(self, off=0):
        """Where x lands in every chip's L3, at byte `off`; with params l3_overlay after its pass's source."""
        if self.l3_overlay:
            return BingoMemSymbol(f"{self.TAG}_xy", offset=off // self.XB * self.YB + self.XB + off % self.XB)
        return BingoMemSymbol(f"{self.TAG}_xland", offset=off)

    def y_sym(self, off=0, chip_id=None):
        """The reduced y in the root's L3 at byte `off`: per pass, per cluster half, slice by slice, D layout (params
        l3_overlay: over the x buffers, at the same offset)."""
        return BingoMemSymbol(f"{self.TAG}_{'xy' if self.l3_overlay else 'y'}", offset=off, chip_id=chip_id)

    def y_owner(self, q):
        """y_scatter: (chip, its tile index) of 16-row tile q of y."""
        return self.chips[q % len(self.chips)], q // len(self.chips)

    def yt_sym(self, q, off=0, chip_id=None):
        """y_scatter: tile q of y in its owner's L3 at byte `off` (chip_id: as another chip names it)."""
        return BingoMemSymbol(f"{self.TAG}_yt", offset=self.y_owner(q)[1] * self.YT_B + off, chip_id=chip_id)

    # ---- data -------------------------------------------------------------------------------------------------------
    def _pack_b(self, q):
        lay = self.lay
        return lay.pack_int4(lay.to_b_pairs(q, self.mesh) if self.w4_paired else lay.to_b(q, self.mesh))

    def _d_rows(self, vec, rows):
        """A per-column vector on `rows` rows in the D layout, FP16."""
        return self.lay.to_d(np.tile(np.asarray(vec, dtype=np.float16), (rows, 1)), self.mesh).astype(np.float16)

    def _stage(self):
        st, d, T, lay, R, HD = self.st, self.d, self.TAG, self.lay, self.R, self.HD
        # the prompt's rows, A layout a pass: the root's source and every chip's landing
        xs = [lay.to_a(d.hq[i * R:(i + 1) * R], self.mesh).astype(np.int8) for i in range(self.NP)]
        if self.l3_overlay and (self.YB != 2 * self.XB):
            raise ValueError("params l3_overlay: a pass's y must be as long as its x source and landing area")
        if self.x_mem:
            m = self.mem[self.ROOT]
            self.h_xhbm = st.put_hbm(f"{T}_xhbm", np.concatenate(xs), mem_chip=(m >> 4, m & 15))
            if self.XB % self.XT:
                raise ValueError(f"params x_tile_bytes={self.XT}: a pass's x ({self.XB} B) in whole tiles")
            self.NXT = self.XB // self.XT
            st.put_zeros(f"{T}_xtflag", "uint8_t", self.NP * self.NXT * 64)
            if getattr(self, "x_cast", "multicast") == "unicast":                # the root's progress flags, one a (chip, tile) of every pass
                st.put_zeros(f"{T}_xuprog", "uint8_t", self.NP * len(self.chips) * self.NXT * 64)
        if self.l3_overlay:
            st.put_zeros(f"{T}_xy", "int8_t", self.NP * self.YB, preload=True)
        else:
            # x from HBM needs no source in L3, and y_scatter stores y in its owners' tiles (yt), never in y: a long
            # prompt's L3 then holds only every pass's x landing area
            if not self.x_mem:
                st.put_zeros(f"{T}_xsrc", "int8_t", self.NP * self.XB, preload=True)
            st.put_zeros(f"{T}_xland", "int8_t", self.NP * self.XB, preload=True)
            if not self.y_scatter:
                st.put_zeros(f"{T}_y", "uint8_t", self.NP * self.YB, preload=True)
        st.put_zeros(f"{T}_xflag", "uint8_t", self.NP * 64)
        # the trades through each chip's L3: per pass the new key tiles, the new value tiles, and both heads' o8
        self.KVX = 2 * self.TPP * self.HKB
        self.OX = R * HD
        st.put_zeros(f"{T}_kvx", "uint8_t", self.NP * self.KVX)
        if self.kv_push:
            st.put_zeros(f"{T}_kvp", "uint8_t", self.NP * self.KVX)                # the mate's halves, pushed here
            st.put_zeros(f"{T}_kvpflag", "uint8_t", self.NP * (1 + self.TPP) * 64)   # its key half's, each value half's
        if self.wo_cols:
            st.put_zeros(f"{T}_oflag", "uint8_t", self.NP * self.d.NH * 64)    # a head's o8 landed on the root
            st.put_zeros(f"{T}_omflag", "uint8_t", self.NP * 64)               # the root's multicast of them landed
            st.put_zeros(f"{T}_yflag", "uint8_t", self.NP * self.G * 64)       # a cluster's slices of y landed
            st.put_zeros(f"{T}_ydum", "uint8_t", self.G * 64)                  # the flag puts' 64 bytes (wo_slice 128)
            if self.y_scatter:
                st.put_zeros(f"{T}_yt", "uint8_t", self.NTL * self.YT_B, preload=True)  # the chip's tiles of y
                st.put_zeros(f"{T}_ytflag", "uint8_t", self.NTL * self.G * 64)     # a cluster's columns of one landed
        else:
            st.put_zeros(f"{T}_ox", "uint8_t", self.NP * self.NC * self.OX)
        # RoPE a pass: per chunk of 32 columns, cos then sin_signed of the pass's rows, D layout
        rope = []
        for i in range(self.NP):
            for c in range(self.NQC):
                cols = self.perm[c * QC:(c + 1) * QC]
                rope += [lay.to_d(d.cos_rep[i * R:(i + 1) * R][:, cols], self.mesh),
                         lay.to_d(d.sin_sgn[i * R:(i + 1) * R][:, cols], self.mesh)]
        self.ROPE_B = 2 * self.NQC * R * QC * 2         # a pass's tables
        st.put(f"{T}_rope", "uint16_t", np.concatenate(rope).astype(np.float16).view(np.uint16))
        # per cluster, in its memory chip's HBM: its weights in the order it streams them (W_q's chunks, then W_k's or
        # W_v's, then W_O's slices of both of the chip's heads for its half of the outputs), and its constants (the
        # dequantisation factors of every chunk on R rows, D layout; the mask patterns, S^T layout)
        self.img, self.chunk_off, self.cst = {}, {}, {}
        for g in range(self.G):
            k, c, h = self.chip(g), self.c(g), self.head[g]
            m = self.group[k]
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            wkv, fkv = (d.wk_q, d.k_f[m]) if self.kind[g] == "K" else (d.wv_q, d.v_f[m])
            # q's and k's columns in the order rope_pairs gives them (v's as they are)
            qcol = lambda cc: self.perm[cc * QC:(cc + 1) * QC]
            kcol = qcol if self.kind[g] == "K" else (lambda cc: np.arange(cc * QC, (cc + 1) * QC))
            parts = [self._pack_b(d.wq_q[:, h * HD + qcol(cc)]) for cc in range(self.NQC)]
            parts += [self._pack_b(wkv[:, m * HD + kcol(cc)]) for cc in self.kv_cc(g)]
            rows = np.concatenate([np.arange(hh * HD, (hh + 1) * HD) for hh in self.chip_heads[k]])
            if self.wo_cols:
                for s in range(self.NWS):
                    o0 = self.WOC * g + s * 32
                    parts.append(self._pack_b(d.wo_q[:, o0:o0 + 32]))
            else:
                for s in range(self.NS):
                    o0 = c * self.HALF + s * self.SLC
                    parts.append(self._pack_b(d.wo_q[rows, o0:o0 + self.SLC]))
            offs, off = [], 0
            for b in parts:
                if b.size > self.chunk:
                    raise ValueError(f"a weight chunk of {b.size} B; params w_chunk_bytes is {self.chunk}")
                offs.append((off, b.size))
                off += b.size
            self.img[g] = st.put_hbm(f"{T}_w_g{g}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[g] = offs
            sc = [self._d_rows(d.q_f[h][qcol(cc)], R) for cc in range(self.NQC)]
            sc += [self._d_rows(fkv[kcol(cc)], R) for cc in self.kv_cc(g)]
            masks = d.mask_bias_patterns() if self.mask_c else [np.asarray(a, dtype=np.float16) for a in d.mask_patterns]
            blob = np.concatenate([np.asarray(a, dtype=np.float16).reshape(-1).view(np.int8) for a in sc]
                                  + [m.reshape(-1).view(np.int8) for m in masks])
            self.cst[g] = (st.put_hbm(f"{T}_c_g{g}", blob, mem_chip=loc), blob.nbytes)
        self.SC_B = R * QC * 2                          # one chunk's factors
        self.C_MASK = (self.NQC + self.NKVC) * self.SC_B
        self.MASK_B = self.bc * BR * (4 if self.mask_c else 2)    # FP16 patterns, or mask_c's INT32 biases

    def _buffers(self):
        d, R = self.d, self.R
        self.sym_zero = False
        self.sym_bytes = 2 * self.SLB
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_sym", self.sym_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", self.sym_bytes) for c in range(self.NC)}
        self.kbuf, self.vbuf, self.consts, self.sweep_bufs, self.o8 = {}, {}, {}, {}, {}
        self.ofull, self.ywo, self.ywo_b = {}, {}, {}
        for g in range(self.G):
            l1 = self.l1_of(g)
            self.kbuf[g] = l1("kt", self.NP_T * self.TPP * self.TILE_B)    # every key tile of the prompt, A layout
            self.vbuf[g] = l1("vt", self.NP_T * self.TPP * self.TILE_B)    # every value tile, V^T in the A layout
            self.consts[g] = l1("consts", self.cst[g][1])
            self.o8[g] = l1("o8", R * (1 if self.wo_cols else self.NC) * self.HD)   # the chip's heads' o8, A layout
            if self.wo_cols:
                self.ywo[g] = l1("ywo", self.R * self.WOC * 2)            # the cluster's columns of y
                if self.o_overlap:
                    # the odd passes' (o_overlap): a pass's first GEMM then waits for the stores of the pass two back
                    self.ywo_b[g] = l1("ywo_b", self.R * self.WOC * 2)
            self.sweep_bufs[g] = fa_sweep_alloc(lambda name, nb, l1=l1: l1(name, nb), self.sweep_cfg(1))
        self._pb = {}

    def _yw(self, g, i):
        """Cluster g's W_O output buffer for pass i: one, or (o_overlap) one for the even passes and one for the odd."""
        return self.ywo_b[g] if self.o_overlap and i % 2 else self.ywo[g]

    def _ywk(self, g, s, i):
        """The key of _ywo_war for GEMM s of pass i on cluster g: the stores that read the buffer it writes."""
        return (g, s, i % 2 if self.o_overlap else 0)

    def l1_of(self, g):
        k, c = self.chip(g), self.c(g)
        return lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nb)

    def pb(self, g, i):
        """Pass i's own buffers on cluster g: the static L1 planner lays those of different passes over each other."""
        if (g, i) in self._pb:
            return self._pb[g, i]
        R, l1 = self.R, self.l1_of(g)
        b = dict(x=l1(f"x_i{i}", self.XB), rope=l1(f"rope_i{i}", self.ROPE_B),
                 a16q=l1(f"a16q_i{i}", R * self.HD * 2), q8=[l1(f"q8_{u}_i{i}", BR * self.HD) for u in range(self.NQ)],
                 a16o=l1(f"a16o_i{i}", R * self.HD * 2))
        if self.wo_cols and not (self.wo_under_proj and i + 1 < self.NP):
            # every head's o8 of the pass, A layout [R, H]; named to sort right after the pass's x, whose bytes it takes
            # (the static packer places by name, first fit); wo_under_proj lands it in the next pass's x instead
            b["ofull"] = l1(f"x_i{i}o", R * self.H)
        for side, ccs in (("q", range(self.NQC)), ("kv", self.kv_cc(g))):
            for cc in ccs:
                b[f"{side}d{cc}"] = l1(f"{side}d{cc}_i{i}", R * QC * 2)
                b[f"{side}s{cc}"] = l1(f"{side}s{cc}_i{i}", R * QC * 2)       # the swapped pairs, then x sin
                b[f"{side}t{cc}"] = l1(f"{side}t{cc}_i{i}", R * QC * 2)       # x cos
        if self.kind[g] == "K":
            b["a16k"] = l1(f"a16k_i{i}", R * self.NKVC * QC * 2)
            if self.kv_split:
                b["k8h"] = l1(f"k8h_i{i}", R * self.NKVC * QC)                # the chip's half of the new key tiles
        else:
            if self.kv_fence:
                b["kvf"] = l1(f"kvf_i{i}", 64)                                # the value puts' fence copy lands here
            for cc in self.kv_cc(g):
                b[f"vr{cc}"] = l1(f"vr{cc}_i{i}", R * QC * 2)                 # row-major v
                for t in range(self.TPP):
                    b[f"vt{cc}{t}"] = l1(f"vt{cc}{t}_i{i}", QC * self.bc * 2)  # its tile t transposed
                    b[f"va{cc}{t}"] = l1(f"va{cc}{t}_i{i}", QC * self.bc * 2)  # in the A layout
        self._pb[g, i] = b
        return b

    def sweep_cfg(self, i_plus_1):
        d = self.d
        return FaSweepCfg(bc=self.bc, nt=i_plus_1 * self.TPP, nq=self.NQ, br=BR, k_s=d.k_s, k_o=d.k_o,
                          a_exp=d.a_exp, a_n_f32bits=d.a_n_bits, d_qk=self.HD, d_v=self.HD, gemm_async=True,
                          phase=f"fa{i_plus_1 - 1}")

    # ---- helpers ----------------------------------------------------------------------------------------------------
    def _load(self, g, i, j, deps=()):
        """Weight chunk j of cluster g into the next slab: a ring load in pass 0, a copy from its slot later."""
        k, c, dm = self.chip(g), self.c(g), self.roles["dm"]
        ls = self.ls[g]
        off, nb = self.chunk_off[g][j]
        si, slab, war = ls.take(nb)
        if i == 0 or self.restream:
            ld, sl = ls.rings.load_chunk(
                lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
                f"Ld_w{j}_i{i}_g{g}", src=self.img[g].address + off, nbytes=nb, slab=slab, war=war, deps=deps, dm=dm)
            self.slot[(g, j)] = (sl, ld)
        else:
            sl, first = self.slot[(g, j)]
            ld = self.node(k, c, dm, f"Ld_w{j}_i{i}_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(sl, slab, nb), list(dict.fromkeys(war + [first] + list(deps))))
        return ld, si, slab

    def gmm(self, g, name, a, b, dst, M, K, N, k_shift, deps, phase):
        k, c = self.chip(g), self.c(g)
        args = SnaxBingoKernelGemmFullArgs(
            input_A_addr=a, input_B_addr=b, input_C_addr=0, output_D_addr=dst, M=M, K=K, N=N, array_shape_idx=0,
            transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1, int4_b_enable=self.int4_b, d_shift=k_shift)
        shape_addr = 0
        if self.gemm_shape_l1:
            if (k, c) not in self.gsh:
                buf = self.cx_chip(k).at(c).l1(f"{self.TAG}_gsh", SnaxBingoKernelGemmShapeInitArgs.BYTES)
                self.node(k, c, self.roles["gemm"], "gsh_init", "__snax_bingo_kernel_gemm_shape_init",
                          SnaxBingoKernelGemmShapeInitArgs(buf, 0), [])
                self.gsh[(k, c)] = buf
            shape_addr = self.gsh[(k, c)]
        q = GemmQueue.of(self.dfg, True)
        q.cfg_ahead, q.cfg_ahead_fa, q.cfg_hoist = self.gemm_cfg_ahead, self.fa_cfg_ahead, self.gemm_cfg_hoist
        return q.dispatch(self.at(k, c), name, args, list(deps), phase, shape_addr=shape_addr)

    def nd(self, g, role, name, kernel, args, deps):
        return self.node(self.chip(g), self.c(g), self.roles[role], name, kernel, args, list(dict.fromkeys(deps)))

    def mul(self, g, name, a, b, out, n, deps):
        return self.nd(g, "simd", name, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdMulF16Args(a, b, out, n // 32, 32), deps)

    def add(self, g, name, a, b, out, n, deps):
        return self.nd(g, "simd", name, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdAddF16Args(a, b, out, n // 32, 32), deps)

    def quant(self, g, name, src, dst, nbytes, inv, deps):
        bits = int(np.asarray(inv, dtype=np.float32).reshape(()).view(np.uint32))
        return self.nd(g, "simd", name, "__snax_bingo_kernel_simd_fp16_to_int8",
                       SnaxBingoKernelSimdFp16ToInt8Args(src, dst, beats=simd_beats(nbytes), rows=1, inv_scale_f32bits=bits),
                       deps)

    def xd(self, g, name, args, deps):
        return self.nd(g, "xdma", name, "__snax_bingo_kernel_xdma_6d", args, deps)

    def copy(self, g, name, src, dst, nb, deps):
        return self.nd(g, "dm", name, "__snax_bingo_kernel_idma_1d_copy", SnaxBingoKernelIdma1dCopyArgs(src, dst, nb), deps)

    def copy2d(self, g, name, src, dst, size, sst, dstr, reps, deps):
        return self.nd(g, "dm", name, "__snax_bingo_kernel_idma_2d_copy",
                       SnaxBingoKernelIdma2dCopyArgs(src, dst, size, sst, dstr, reps), deps)

    # ---- the stages of one pass ---------------------------------------------------------------------------------------
    @staticmethod
    def _dl(t):
        """A task, or a list of them (a fetch or gather split over two engines), as a list of dependencies."""
        return list(t) if isinstance(t, (list, tuple)) else [t]

    def stage_x_put(self, i, after):
        T, R = self.TAG, self.ROOT
        if self.x_mem and getattr(self, "x_cast", "multicast") == "unicast":
            # params x_cast unicast: a copy a chip in turn, in the fold order, the same tiles and flags
            from bingo_kernel_args import HostBingoKernelHbmUcastArgs
            m = self.mem[R]
            return self.cx_chip(R).host_node(
                f"x_hbm_ucast_i{i}", "__host_bingo_kernel_hbm_ucast",
                HostBingoKernelHbmUcastArgs(self.h_xhbm.address + i * self.XB, self.xland(i * self.XB),
                                            BingoMemSymbol(f"{T}_xtflag", offset=i * self.NXT * 64),
                                            BingoMemSymbol(f"{T}_xuprog", offset=i * len(self.chips) * self.NXT * 64),
                                            self.h_seq[m].address + 64, 1, self.XT, self.NXT, m, self.engine[R],
                                            self.order),
                after=list(after))
        if self.x_mem:
            from bingo_kernel_args import HostBingoKernelHbmBcastArgs
            m = self.mem[R]
            return self.cx_chip(R).host_node(
                f"x_hbm_bcast_i{i}", "__host_bingo_kernel_hbm_bcast",
                HostBingoKernelHbmBcastArgs(self.h_xhbm.address + i * self.XB, self.xland(i * self.XB),
                                            BingoMemSymbol(f"{T}_xtflag", offset=i * self.NXT * 64),
                                            self.h_seq[m].address + 64, 1, self.XT, self.NXT, m, self.engine[R]),
                after=list(after))
        return self.node(R, 0, self.roles["dm"], f"x_bput_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                         SnaxBingoKernelIdmaPutFlaggedArgs(
                             self.xsrc(i * self.XB), self.xland(i * self.XB),
                             self.XB, BingoMemSymbol(f"{T}_xflag", offset=i * 64), value=1, bcast=True, also_here=True),
                         list(after) + list(self.x_ready.get(i, [])))

    def x_puts_up_front(self, go):
        """The first pass's multicast; with x from HBM every pass's, one after the other on the root's host, ahead of
        its weight prefetcher (x_pf_split: behind the prefetcher's first part)."""
        R = self.ROOT
        first = [go[R]]
        if self.pf_split:
            first.append(self.split_root_prefetch(go))
        puts = [self.stage_x_put(0, first)]
        if self.x_mem:
            for i in range(1, self.NP):
                puts.append(self.stage_x_put(i, [go[R]] + puts[-1:]))
            self.edge(puts[-1], self.prefetch[R])
            # every o8 multicast of the root's waits for the last of these: two multicast sources at once stall the
            # fabric, and the task ends only once its last flag has landed
            self._x_put_last = puts[-1]
        return puts

    def split_root_prefetch(self, go):
        """x_pf_split: the root's prefetcher's first x_pf_chunks chunks of each ring in a task of their own behind the
        start; the prefetcher then starts at the next chunk. Returns the first part."""
        from bingo_kernel_args import HostBingoKernelWeightPrefetchRangeArgs
        R = self.ROOT
        n = int(self.p.get("x_pf_chunks", self.WO0))
        if not 0 < n <= self.n_slots:
            raise ValueError(f"params x_pf_chunks={n}: 1..{self.n_slots} (the ring's slots: the first part never waits)")
        t, b = self.prefetch[R], self.prefetch[R].kernel_args
        part = lambda lo, hi: HostBingoKernelWeightPrefetchRangeArgs(
            b.sched_addr, b.ring_addr, b.flag_addr, b.release_addr, b.seq_addr, b.rec_addr, b.rec_slots, b.n_rings,
            b.n_slots, b.slot_bytes, b.memchip_id, batch=b.batch, policy=b.policy, engine=b.engine,
            batch_routed=b.batch_routed, queued=b.queued, queued_inflight_routed=b.queued_inflight_routed, head=b.head,
            batch_head=b.batch_head, chunk_lo=lo, chunk_hi=hi)
        t.kernel_args = part(n, 0)
        t.kernel_name = t.kernel_args.KERNEL_NAME
        return self.cx_chip(R).host_node(f"weight_prefetch_k{R:02x}_head", t.kernel_name, part(0, n), after=[go[R]])

    def stage_x(self, g, i, put, deps):
        T, k = self.TAG, self.chip(g)
        # the root's iDMA multicast is a task of the root's: its own fetch depends on it; the memory chip's (a host task)
        # orders every fetch, the flag proves the landing
        own = k == self.ROOT and not self.x_mem
        flag = (BingoMemSymbol(f"{T}_xtflag", offset=(i * self.NXT + self.NXT - 1) * 64) if self.x_mem else
                BingoMemSymbol(f"{T}_xflag", offset=i * 64))
        xx = int(round(self.XB * self.x_split_frac / 512)) * 512
        if self.x_row_tiles:
            # x_row_tiles: one fetch a row tile, each polling its own multicast tiles' flags; the A layout keeps a row
            # tile's bytes together (row tile outermost), so each is one contiguous copy
            if getattr(self, "x_ahead", False):
                raise ValueError("params x_row_tiles: not with x_ahead (it orders the fetch as one task)")
            mt = self.R // 16
            rb = self.XB // mt
            if rb % self.XT:
                raise ValueError(f"params x_row_tiles: a row tile ({rb} B) is not whole multicast tiles of {self.XT} B")
            tpr = rb // self.XT
            out, xd = [], self.x_tiles_xdma and i > 0
            for t in range(mt):
                f = BingoMemSymbol(f"{T}_xtflag", offset=(i * self.NXT + t * tpr) * 64)
                if xd:
                    w = self.nd(g, "xdma", f"x_xw_i{i}t{t}_g{g}", "__snax_bingo_kernel_wait_flags",
                                SnaxBingoKernelWaitFlagsArgs(f, tpr, 1), list(deps))
                    self.dfg.bingo_add_order_edge(put, w)
                    if not out:
                        self._x_first[g, i] = w
                    out.append(self.nd(g, "xdma", f"x_xcp_i{i}t{t}_g{g}", "__snax_bingo_kernel_xdma_1d_copy",
                                       SnaxBingoKernelXdma1dCopyArgs(self.xland(i * self.XB + t * rb),
                                                                     self.pb(g, i)["x"].view(t * rb), rb), [w]))
                    self._x_xdma.setdefault((g, i), []).extend([w, out[-1]])
                    continue
                w = self.nd(g, "dm", f"x_wait_i{i}t{t}_g{g}", "__snax_bingo_kernel_idma_fetch_flagged",
                            SnaxBingoKernelIdmaFetchFlaggedArgs(
                                self.xland(i * self.XB + t * rb), self.pb(g, i)["x"].view(t * rb), rb, f, tpr, 1),
                            list(deps))
                self.dfg.bingo_add_order_edge(put, w)
                out.append(w)
            if not xd:
                self._x_first[g, i] = out[0]
            self._x_last[g, i] = out[-1]
            return out
        if xx:
            # x_split_frac: one wait, then the iDMA's part and the xDMA's at once; the consumers wait for both
            if getattr(self, "x_ahead", False):
                raise ValueError("params x_split_frac: not with x_ahead (it orders the fetch as one task)")
            w = self.nd(g, "dm", f"x_wait_i{i}_g{g}", "__snax_bingo_kernel_wait_flags",
                        SnaxBingoKernelWaitFlagsArgs(flag, 1, 1), list(deps))
            self.dfg.bingo_add_order_edge(put, w)
            self._x_first[g, i] = self._x_last[g, i] = w
            a = self.XB - xx
            dst = self.pb(g, i)["x"]
            return [self.nd(g, "dm", f"x_cp_i{i}_g{g}", "__snax_bingo_kernel_idma_1d_copy",
                            SnaxBingoKernelIdma1dCopyArgs(self.xland(i * self.XB), dst, a), [w]),
                    self.nd(g, "xdma", f"x_xcp_i{i}_g{g}", "__snax_bingo_kernel_xdma_1d_copy",
                            SnaxBingoKernelXdma1dCopyArgs(self.xland(i * self.XB + a), dst.view(a), xx), [w])]
        nd = self.nd(g, "dm", f"x_wait_i{i}_g{g}", "__snax_bingo_kernel_idma_fetch_flagged",
                     SnaxBingoKernelIdmaFetchFlaggedArgs(
                         self.xland(i * self.XB), self.pb(g, i)["x"], self.XB, flag, 1, 1),
                     list(deps) + ([put] if own else []))
        if not own:
            self.dfg.bingo_add_order_edge(put, nd)
        self._x_first[g, i] = self._x_last[g, i] = nd
        return nd

    def _rope(self, g, i, side, cc, src, after):
        """RoPE of a [R, 32] D-layout chunk in place: x cos + swap(x) sin_signed."""
        b, n = self.pb(g, i), self.R * QC
        cos = b["rope"].view(2 * cc * self.SC_B)
        sin = b["rope"].view((2 * cc + 1) * self.SC_B)
        sw = self.nd(g, "dm", f"{side}sw{cc}_i{i}", "__snax_bingo_kernel_idma_pairwise_swap",
                     SnaxBingoKernelIdmaPairwiseSwapArgs(src, b[f"{side}s{cc}"], n, 2), after)
        m1 = self.mul(g, f"{side}rc{cc}_i{i}", src, cos, b[f"{side}t{cc}"], n, after + [self.rope_ld[g, i]])
        m2 = self.mul(g, f"{side}rs{cc}_i{i}", b[f"{side}s{cc}"], sin, b[f"{side}s{cc}"], n, [sw, self.rope_ld[g, i]])
        return self.add(g, f"{side}ra{cc}_i{i}", b[f"{side}t{cc}"], b[f"{side}s{cc}"], src, n, [m1, m2, sw])

    def _rope_pair(self, g, i, side, ca, cb, after):
        """rope_pairs chunks: RoPE of chunks ca (the pairs' first dimensions) and cb (their second), [R, 32] D layout
        each, in place: a = a cos_a + b sin_a, b = b cos_b + a sin_b (sin signed). Returns the two adds."""
        b, n = self.pb(g, i), self.R * QC
        tab = lambda cc, k: b["rope"].view((2 * cc + k) * self.SC_B)
        xa, xb = b[f"{side}d{ca}"], b[f"{side}d{cb}"]
        dep = list(after) + [self.rope_ld[g, i]]
        ta = self.mul(g, f"{side}rc{ca}_i{i}", xa, tab(ca, 0), b[f"{side}t{ca}"], n, dep)
        sa = self.mul(g, f"{side}rs{ca}_i{i}", xb, tab(ca, 1), b[f"{side}s{ca}"], n, dep)
        tb = self.mul(g, f"{side}rc{cb}_i{i}", xb, tab(cb, 0), b[f"{side}t{cb}"], n, dep)
        sb = self.mul(g, f"{side}rs{cb}_i{i}", xa, tab(cb, 1), b[f"{side}s{cb}"], n, dep)
        ra = self.add(g, f"{side}ra{ca}_i{i}", b[f"{side}t{ca}"], b[f"{side}s{ca}"], xa, n, [ta, sa, sb])
        rb = self.add(g, f"{side}ra{cb}_i{i}", b[f"{side}t{cb}"], b[f"{side}s{cb}"], xb, n, [tb, sb, sa])
        return ra, rb

    def stage_proj(self, g, i, x):
        """Pass i's q (into its query tiles) and k or v (into the cluster's tiles). Returns ({u: (q8 handle, ready)},
        the projection's last GEMM, the task after which the cluster's own new tiles are there)."""
        d, b, R, HD = self.d, self.pb(g, i), self.R, self.HD
        mt, kb = R // 16, HD // 4                     # row tiles; 4-column blocks of a head in the A layout
        self.rope_ld[g, i] = self.copy(g, f"rope_ld_i{i}", BingoMemSymbol(f"{self.TAG}_rope", offset=i * self.ROPE_B),
                                       b["rope"], self.ROPE_B, [] if self.x_ahead else self._rope_war.get(g, []))
        cst = self.consts[g]
        last_gm, d2a = None, {"q": [], "kv": []}
        vq = []
        # (side, its index in the cluster's weight stream and constants, chunk)
        items = ([("q", cc, cc) for cc in range(self.NQC)]
                 + [("kv", self.NQC + n, cc) for n, cc in enumerate(self.kv_cc(g))])
        lds = {}

        def load(n):
            lds[n] = self._load(g, i, items[n][1])

        for n, (side, wi, cc) in enumerate(items):
            if n not in lds:
                load(n)
            if n == 0:
                self._first_ld[g, i] = lds[0][0]
            ld, si, slab = lds[n]
            dd = b[f"{side}d{cc}"]
            if n == 0 and self.x_row_tiles:
                # x_row_tiles: the first GEMM a row tile at a time, each behind its row tile of x (A and D both keep a
                # row tile's bytes together)
                rb, db = self.XB // mt, QC // BLK * 16 * 16 * 2
                gms = [self.gmm(g, f"g{side}{cc}_i{i}r{t}", b["x"].view(t * rb), slab, dd.view(t * db), 1, self.H // 4,
                                QC // BLK, d.k_x, [ld, x[t]], f"p{i}") for t in range(mt)]
                self.ls[g].release(si, gms[-1])
                gm = gms
                last_gm = gms[-1]
            else:
                gm = self.gmm(g, f"g{side}{cc}_i{i}", b["x"], slab, dd, mt, self.H // 4, QC // BLK, d.k_x,
                              [ld] + self._dl(x), f"p{i}")
                self.ls[g].release(si, gm)
                last_gm = gm
            if self.load_ahead and n + 1 < len(items):
                load(n + 1)
            dq = self.mul(g, f"dq{side}{cc}_i{i}", dd, cst.view(wi * self.SC_B), dd, R * QC, self._dl(gm))
            if side == "q" or self.kind[g] == "K":
                dst = b["a16q"] if side == "q" else b["a16k"]
                # the chunk's place in the A layout: q's of the head; k's of the columns the cluster projects
                lo, kbw = (0, kb) if side == "q" else (self.kv_cc(g)[0], self.NKVC * QC // 4)
                to_a = lambda c_, rt_: self.xd(g, f"d2a{side}{c_}_i{i}",
                                               d_to_a_args(self.mesh, mt, QC // BLK, kbw, b[f"{side}d{c_}"],
                                                           dst.view((c_ - lo) * (QC // 4) * 16 * 4 * 2)), [rt_])
                if self.rope_pairs == "swap":
                    d2a[side].append(to_a(cc, self._rope(g, i, side, cc, dd, [dq])))
                elif cc % 2 == 0:
                    first_dq = dq                     # the pair's first chunk waits for its second
                else:
                    ra, rb = self._rope_pair(g, i, side, cc - 1, cc, [first_dq, dq])
                    d2a[side] += [to_a(cc - 1, ra), to_a(cc, rb)]
            else:
                # v: D -> row-major, then each key tile transposed into V^T's A layout and quantised into its place
                vr = self.xd(g, f"vrm{cc}_i{i}", convert_args("D", "row_major", R, QC, self.mesh, 2, dd, b[f"vr{cc}"]),
                             [dq])
                for t in range(self.TPP):
                    tr = self.nd(g, "xdma", f"vtr{cc}{t}_i{i}", "__snax_bingo_kernel_xdma_transpose_2d",
                                 SnaxBingoKernelXdmaTranspose2dArgs(b[f"vr{cc}"].view(t * self.bc * QC * 2),
                                                                    b[f"vt{cc}{t}"], self.bc, QC, 2), [vr])
                    va = self.xd(g, f"vta{cc}{t}_i{i}",
                                 convert_args("row_major", "A", QC, self.bc, self.mesh, 2, b[f"vt{cc}{t}"],
                                              b[f"va{cc}{t}"]), [tr])
                    j = i * self.TPP + t
                    vq.append(self.quant(g, f"vq{cc}{t}_i{i}", b[f"va{cc}{t}"],
                                         self.vbuf[g].view(j * self.TILE_B + cc * QC * self.bc),
                                         QC * self.bc * 2, d.s_v.inv, [va]))
        # q8: one query tile at a time
        q8 = {}
        for u in range(self.NQ):
            qa = self.quant(g, f"q8_{u}_i{i}", b["a16q"].view(u * BR * HD * 2), b["q8"][u], BR * HD * 2, d.s_q.inv,
                            d2a["q"])
            q8[u] = (b["q8"][u], [qa])
        if self.kind[g] == "K" and self.kv_split:
            own = self.quant(g, f"k8_i{i}", b["a16k"], b["k8h"], R * self.NKVC * QC * 2, d.s_k.inv, d2a["kv"])
        elif self.kind[g] == "K":
            own = self.quant(g, f"k8_i{i}", b["a16k"], self.kbuf[g].view(i * self.TPP * self.TILE_B), R * HD * 2,
                             d.s_k.inv, d2a["kv"])
        else:
            own = vq[-1]
        self._rope_war[g] = [d2a["q"][-1]] + (d2a["kv"][-1:] if d2a["kv"] else [])
        self._proj_lds[g, i] = [lds[n][0] for n in range(len(items))]
        return q8, last_gm, own

    def stage_trade_kv(self, k, i, own):
        """Cluster 0's new key tiles and cluster 1's new value tiles through chip k's L3 into the other cluster."""
        T, n = self.TAG, self.TPP * self.TILE_B
        g0, g1 = self.gs(k)
        base = i * self.KVX
        out0 = self.copy(g0, f"kx_put_i{i}", self.kbuf[g0].view(i * n), BingoMemSymbol(f"{T}_kvx", offset=base), n, [own[g0]])
        out1 = self.copy(g1, f"vx_put_i{i}", self.vbuf[g1].view(i * n), BingoMemSymbol(f"{T}_kvx", offset=base + n), n,
                         [own[g1]])
        in1 = self.copy(g1, f"kx_get_i{i}", BingoMemSymbol(f"{T}_kvx", offset=base), self.kbuf[g1].view(i * n), n, [out0])
        in0 = self.copy(g0, f"vx_get_i{i}", BingoMemSymbol(f"{T}_kvx", offset=base + n), self.vbuf[g0].view(i * n), n, [out1])
        for t in range(self.TPP):
            j = i * self.TPP + t
            self.tile_ready[(g0, "K", j)], self.tile_ready[(g1, "K", j)] = own[g0], in1
            self.tile_ready[(g1, "V", j)], self.tile_ready[(g0, "V", j)] = own[g1], in0

    def kv_cc(self, g):
        """The k or v chunks cluster g projects: all of them, or (kv_split) its chip's half."""
        h = self.kv_half[self.chip(g)] if self.kv_split else 0
        return range(h * self.NKVC, (h + 1) * self.NKVC)

    def stage_trade_kv_split(self, i, own):
        """params kv_split: each chip's half of the pass's new key tiles (cluster 0, compact) and value tiles (cluster 1,
        in place) into its own L3; every cluster of the group then copies the halves it lacks into its tiles, from its
        own chip's L3 and, by a remote read, from the other chip's (a half is only ever written where it is made). A key
        tile's half is a run of each 16-row tile of its A layout; a value tile's (V^T) is contiguous."""
        T, HD, TPP, bc = self.TAG, self.HD, self.TPP, self.bc
        hk = self.NKVC * QC * BR                # a key half's run in a 16-row tile (INT8 A layout)
        hv = self.NKVC * QC * bc                # a value half of one tile
        rt = TPP * bc // BR                     # the pass's 16-row tiles of keys
        base, voff = i * self.KVX, TPP * self.HKB
        kvx = lambda off, chip=None: BingoMemSymbol(f"{T}_kvx", offset=base + off, chip_id=chip)
        kput, vput, kpl, kf, vf = {}, {}, {}, {}, {}
        for k in self.chips:
            g0, g1 = self.gs(k)
            h = self.kv_half[k]
            kput[k] = self.copy(g0, f"kx_put_i{i}", self.pb(g0, i)["k8h"], kvx(0), TPP * self.HKB, [own[g0]])
            kpl[k] = self.copy2d(g0, f"kx_pl_i{i}", self.pb(g0, i)["k8h"],
                                 self.kbuf[g0].view(i * TPP * self.TILE_B + h * hk), hk, hk, HD * BR, rt, [own[g0]])
            vput[k] = [self.copy(g1, f"vx_put{t}_i{i}", self.vbuf[g1].view((i * TPP + t) * self.TILE_B + h * hv),
                                 kvx(voff + t * hv), hv, [own[g1]]) for t in range(TPP)]
            # kv_fence: the key half's fence is the placement right behind its put on the same core; the value halves'
            # a 64-byte copy behind their puts
            kf[k] = [kpl[k]] if self.kv_fence else []
            vf[k] = [self.copy(g1, f"vx_fence_i{i}", self.vbuf[g1].view(i * TPP * self.TILE_B + h * hv),
                               self.pb(g1, i)["kvf"], 64, vput[k])] if self.kv_fence else []
        if self.kv_push:
            self._trade_kv_push(i, own, kput, kpl, vput, kvx, hk, hv, rt)
            return
        # sync barrier: the halves' puts are a kernel of their own, and the barrier after it, not an edge from the mate's
        # puts, lets the remote reads start
        self.kernel_barrier()
        mate = (lambda ts: []) if getattr(self, "per_kernel", False) else (lambda ts: list(ts))
        for k in self.chips:
            g0, g1 = self.gs(k)
            h, m = self.kv_half[k], self.kv_mate[k]
            kdst = lambda g, hh: self.kbuf[g].view(i * TPP * self.TILE_B + hh * hk)
            vdst = lambda g, t, hh: self.vbuf[g].view((i * TPP + t) * self.TILE_B + hh * hv)
            k1 = self.copy2d(g1, f"kx_get_i{i}", kvx(0), kdst(g1, h), hk, hk, HD * BR, rt, [kput[k]] + kf[k])
            k0 = self.copy2d(g0, f"kx_rget_i{i}", kvx(0, m), kdst(g0, 1 - h), hk, hk, HD * BR, rt,
                             mate([kput[m]]) + [kpl[k]] + mate(kf[m]))
            k1 = self.copy2d(g1, f"kx_rget_i{i}", kvx(0, m), kdst(g1, 1 - h), hk, hk, HD * BR, rt,
                             mate([kput[m]]) + [k1] + mate(kf[m]))
            v0, v1 = None, own[g1]
            for t in range(TPP):
                v0 = self.copy(g0, f"vx_get{t}_i{i}", kvx(voff + t * hv), vdst(g0, t, h), hv,
                               [vput[k][t]] + vf[k] + ([v0] if v0 is not None else []))
                v0 = self.copy(g0, f"vx_rget{t}_i{i}", kvx(voff + t * hv, m), vdst(g0, t, 1 - h), hv,
                               mate([vput[m][t]]) + [v0] + mate(vf[m]))
                v1 = self.copy(g1, f"vx_rget{t}_i{i}", kvx(voff + t * hv, m), vdst(g1, t, 1 - h), hv,
                               mate([vput[m][t]]) + [v1] + mate(vf[m]))
            for t in range(TPP):
                j = i * TPP + t
                self.tile_ready[(g0, "K", j)], self.tile_ready[(g1, "K", j)] = k0, k1
                self.tile_ready[(g0, "V", j)], self.tile_ready[(g1, "V", j)] = v0, v1

    def _trade_kv_push(self, i, own, kput, kpl, vput, kvx, hk, hv, rt):
        """params kv_push: the mate's halves pushed into this chip's L3 (a flagged unicast put each), then copied from
        there; this chip's own halves as kv_split takes them (its L3 copy for the other cluster, the placement in L1)."""
        T, HD, TPP = self.TAG, self.HD, self.TPP
        base, voff = i * self.KVX, TPP * self.HKB
        kvp = lambda off, chip=None: BingoMemSymbol(f"{T}_kvp", offset=base + off, chip_id=chip)
        flg = lambda n, chip=None: BingoMemSymbol(f"{T}_kvpflag", offset=(i * (1 + TPP) + n) * 64, chip_id=chip)
        kpush, vpush = {}, {}
        for k in self.chips:
            g0, g1 = self.gs(k)
            h, m = self.kv_half[k], self.kv_mate[k]
            kpush[k] = self.nd(g0, "dm", f"kx_push_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                               SnaxBingoKernelIdmaPutFlaggedArgs(self.pb(g0, i)["k8h"], kvp(0, m), TPP * self.HKB,
                                                                 flg(0, m), value=1), [own[g0], kput[k]])
            vpush[k] = [self.nd(g1, "dm", f"vx_push{t}_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                                SnaxBingoKernelIdmaPutFlaggedArgs(
                                    self.vbuf[g1].view((i * TPP + t) * self.TILE_B + h * hv), kvp(voff + t * hv, m), hv,
                                    flg(1 + t, m), value=1), [own[g1], vput[k][t]]) for t in range(TPP)]
        for k in self.chips:
            g0, g1 = self.gs(k)
            h, m = self.kv_half[k], self.kv_mate[k]
            kdst = lambda g, hh: self.kbuf[g].view(i * TPP * self.TILE_B + hh * hk)
            vdst = lambda g, t, hh: self.vbuf[g].view((i * TPP + t) * self.TILE_B + hh * hv)
            # each cluster waits for all of the mate's flags behind its own pushes (its DM core is in order)
            w0 = self.nd(g0, "dm", f"kvp_wait_i{i}", "__snax_bingo_kernel_wait_flags",
                         SnaxBingoKernelWaitFlagsArgs(flg(0), 1 + TPP, 1), [kpush[k]])
            w1 = self.nd(g1, "dm", f"kvp_wait_i{i}", "__snax_bingo_kernel_wait_flags",
                         SnaxBingoKernelWaitFlagsArgs(flg(0), 1 + TPP, 1), [vpush[k][-1]])
            for w in (w0, w1):
                self.dfg.bingo_add_order_edge(kpush[m], w)
                for t in range(TPP):
                    self.dfg.bingo_add_order_edge(vpush[m][t], w)
            k1 = self.copy2d(g1, f"kx_get_i{i}", kvx(0), kdst(g1, h), hk, hk, HD * BR, rt, [kput[k]])
            k0 = self.copy2d(g0, f"kx_pget_i{i}", kvp(0), kdst(g0, 1 - h), hk, hk, HD * BR, rt, [w0, kpl[k]])
            k1 = self.copy2d(g1, f"kx_pget_i{i}", kvp(0), kdst(g1, 1 - h), hk, hk, HD * BR, rt, [w1, k1])
            v0, v1 = None, own[g1]
            for t in range(TPP):
                v0 = self.copy(g0, f"vx_get{t}_i{i}", kvx(voff + t * hv), vdst(g0, t, h), hv,
                               [vput[k][t]] + ([v0] if v0 is not None else []))
                v0 = self.copy(g0, f"vx_pget{t}_i{i}", kvp(voff + t * hv), vdst(g0, t, 1 - h), hv, [w0, v0])
                v1 = self.copy(g1, f"vx_pget{t}_i{i}", kvp(voff + t * hv), vdst(g1, t, 1 - h), hv, [w1, v1])
            for t in range(TPP):
                j = i * TPP + t
                self.tile_ready[(g0, "K", j)], self.tile_ready[(g1, "K", j)] = k0, k1
                self.tile_ready[(g0, "V", j)], self.tile_ready[(g1, "V", j)] = v0, v1

    def mask(self, i):
        """The sweep's mask of pass i: key tile j, query tile u -> the pattern's handle in the cluster's constants."""
        d = self.d
        first = i * self.TPP

        def m(g):
            def f(j, u):
                if j < first:
                    return None
                pi = d.mask_tiles[(j - first, u)]
                return None if pi is None else self.consts[g].view(self.C_MASK + pi * self.MASK_B)
            return f
        return m

    def ot8_bits(self):
        """fa_sweep's ot8: the o8 quantiser's scale (FP32 bits) under params ot8_fused, else None."""
        if not self.ot8_fused:
            return None
        return int(np.asarray(self.d.s_ot.inv, dtype=np.float32).reshape(()).view(np.uint32))

    def stage_post(self, g, i, sw):
        """o~ of pass i into the cluster's head of the chip's o8. Returns (the readers of the sweep's tiles, the last
        task writing o8)."""
        d, b, HD = self.d, self.pb(g, i), self.HD
        c = self.c(g)
        last = None
        cv = []
        if self.ot8_fused:
            # each tile is o8 already (A layout of 16 rows): one copy a tile into its place in o8
            for u in range(self.NQ):
                dst = (self.o8[g].view(u * BR * HD) if self.wo_cols else
                       self.o8[g].view(u * BR * self.NC * HD + c * BR * HD))
                last = self.nd(g, "xdma", f"o8cp{u}_i{i}", "__snax_bingo_kernel_xdma_1d_copy",
                               SnaxBingoKernelXdma1dCopyArgs(sw["xt"][u], dst, BR * HD),
                               [sw["tr"][u]] + self._o8_war.get(g, []))
                cv.append(last)
            return cv, last
        for u in range(self.NQ):
            cv.append(self.xd(g, f"o2a{u}_i{i}", convert_args("row_major", "A", BR, HD, self.mesh, 2, sw["xt"][u],
                                                              b["a16o"].view(u * BR * HD * 2)), [sw["tr"][u]]))
            dst = (self.o8[g].view(u * BR * HD) if self.wo_cols else
                   self.o8[g].view(u * BR * self.NC * HD + c * BR * HD))
            last = self.quant(g, f"o8_{u}_i{i}", b["a16o"].view(u * BR * HD * 2), dst, BR * HD * 2, d.s_ot.inv,
                              [cv[-1]] + self._o8_war.get(g, []))
        return cv, last

    def stage_trade_o(self, k, i, own):
        """Each cluster's head of o8 through chip k's L3 into the other cluster's o8, so both hold the chip's two heads."""
        T, HD, NC = self.TAG, self.HD, self.NC
        run, pitch = BR * HD, BR * NC * HD
        got = {}
        for g in self.gs(k):
            c = self.c(g)
            put = self.copy2d(g, f"ox_put_i{i}", self.o8[g].view(c * run), BingoMemSymbol(f"{T}_ox", offset=(i * NC + c) * self.OX),
                              run, pitch, run, self.NQ, [own[g]])
            other = self.gs(k)[1 - c]
            got[other] = self.copy2d(other, f"ox_get_i{i}", BingoMemSymbol(f"{T}_ox", offset=(i * NC + c) * self.OX),
                                     self.o8[other].view(c * run), run, run, pitch, self.NQ,
                                     [put] + self._o8_war.get(other, []))
        return got

    def stage_wo_cols(self, i, post_last, x, stores, between=None):
        """Pass i's W_O by columns (params wo_cols): every head's o8 into the root's L3, the root's multicast of them,
        each cluster's gather into its L1 (A layout), its GEMMs over the full depth and its slices stored into the root's
        y. `between()` (wo_under_proj) makes the next pass's work that goes after the puts and before the root's wait; it
        returns {cluster: the last task reading its x buffer}, where the gather then lands."""
        T, R, HD, NH, G = self.TAG, self.ROOT, self.HD, self.d.NH, self.G
        OB, base = self.R * HD, i * self.XB              # a head's o8; pass i's x landing area, where they land
        on_root = lambda sym, k: sym if k == R else BingoMemSymbol(sym.symbol_name, offset=sym.offset, chip_id=R)
        rg = [g for g in range(G) if self.chip(g) == R]
        puts = {}
        for g in range(G):
            k, h = self.chip(g), self.head[g]
            # into the root's landing area of x_i, after the root's fetches of x_i (sync barrier: the kernels' barriers
            # order them)
            puts[g] = self.nd(g, "dm", f"o_put_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                              SnaxBingoKernelIdmaPutFlaggedArgs(
                                  self.o8[g], on_root(self.xland(base + h * OB), k), OB,
                                  on_root(BingoMemSymbol(f"{T}_oflag", offset=(i * NH + h) * 64), k), value=1),
                              [post_last[g]] + ([] if getattr(self, "per_kernel", False) else [t for r in rg for t in self._dl(x[r])]))
            self._o8_war[g] = [puts[g]]
        def nwait():
            w = self.node(R, 0, self.roles["dm"], f"o_nwait_i{i}", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_oflag", offset=i * NH * 64), NH, 1),
                          [puts[g] for g in rg])
            for g in range(G):
                if self.chip(g) != R:
                    self.dfg.bingo_add_order_edge(puts[g], w)
            return w
        xl = [self._x_put_last] if getattr(self, "_x_put_last", None) is not None else []
        uput = {}                                         # x_cast unicast: the root's put into each other chip

        def mc_of(deps):
            if getattr(self, "x_cast", "multicast") == "unicast":
                # a flagged put a chip, one after the other on the root's DM core; no multicast meets another
                prev = list(deps)
                for k in self.order:
                    if k == R:
                        continue
                    uput[k] = self.node(R, 0, self.roles["dm"], f"o_uput_i{i}_k{k:02x}",
                                        "__snax_bingo_kernel_idma_put_flagged",
                                        SnaxBingoKernelIdmaPutFlaggedArgs(
                                            self.xland(base), BingoMemSymbol(self.xland(base).symbol_name,
                                                                             offset=self.xland(base).offset, chip_id=k),
                                            NH * OB, BingoMemSymbol(f"{T}_omflag", offset=i * 64, chip_id=k), value=1),
                                        prev)
                    prev = [uput[k]]
                return prev[0]
            return self.node(R, 0, self.roles["dm"], f"o_bput_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                             SnaxBingoKernelIdmaPutFlaggedArgs(self.xland(base), self.xland(base), NH * OB,
                                                               BingoMemSymbol(f"{T}_omflag", offset=i * 64),
                                                               value=1, bcast=True), deps + xl)
        # o_mc_first: the root's wait for every head's o8 and its multicast go before the next pass's front in its DM
        # core's program, not behind that front's weight loads (which follow the root's own projection GEMMs)
        early = self.o_mc_first and between is not None
        w = nwait() if early else None
        mc = mc_of([w]) if early else None
        nxt = between() if between is not None else None
        if self.front_first == "x" and nxt is not None:
            # front_first "x": the first weight chunk right ahead of the next pass's x fetch, the RoPE table and the second
            # chunk right behind its last task; front_after_sm: the chunk behind this pass's last softmax
            for g in range(G):
                ld0, ld1 = self._proj_lds[g, i + 1][:2]
                x0, xl = self._x_first[g, i + 1], self._x_last[g, i + 1]
                ld0.order_key = x0.node_id - 0.5
                self.rope_ld[g, i + 1].order_key = xl.node_id + 0.3
                ld1.order_key = xl.node_id + 0.6
                if self.front_after_sm:
                    self.dfg.bingo_add_edge(self._last_sm[g, i], ld0)
        elif self.front_first and nxt is not None:
            # front_first: the next pass's RoPE table and first two weight chunks ahead of the o8 put ("all") or of the
            # root's wait and multicast ("root") in the DM core's program; the rest of its loads wait for slabs anyway
            for g in range(G):
                if self.front_first == "all":
                    at = puts[g]
                elif early and self.chip(g) == R and self.c(g) == 0:
                    at = w
                else:
                    continue
                front = [self.rope_ld[g, i + 1]] + self._proj_lds[g, i + 1][:2]
                for j, t in enumerate(front):
                    t.order_key = at.node_id - 1 + (j + 1) / (len(front) + 1)
        obuf = (lambda g: self.pb(g, i + 1)["x"]) if nxt is not None else (lambda g: self.pb(g, i)["ofull"])
        # wo_early: the first W_O chunk of each cluster, loaded right behind its put
        first_ld = {g: self._load(g, i, self.WO0) for g in range(G)} if self.wo_early and nxt is None else {}
        if w is None:
            w = nwait()
        # the gather into L1, A layout [R, H]: head h's [R, 128] (row tiles of 16) to columns 128 h .. of every row tile
        def gather(g, src, deps):
            deps = list(deps) + ([] if self.o_overlap else self._ofull_war.get(g, [])) + ([nxt[g]] if nxt else [])
            run = HD // 4 * 64                            # a head's row tile, contiguous on both sides
            if self.o_get_xdma:
                # o_get_split_frac: the last hd heads by the iDMA, beside the xDMA's
                hd = int(round(NH * self.o_get_split))
                hx = NH - hd
                if self.o_row_tiles:
                    # o_row_tiles: a row tile of every head a task, into its 64 KB of the A layout
                    ts = []
                    for r in range(self.R // 16):
                        t = self.nd(g, "xdma", f"o_get_i{i}r{r}", "__snax_bingo_kernel_xdma_6d",
                                    SnaxBingoKernelXdma6dArgs(
                                        BingoMemSymbol(src.symbol_name, offset=src.offset + r * run, chip_id=src.chip_id),
                                        obuf(g).view(r * self.H // 4 * 64), 8, 8,
                                        [64, run, OB], [run // 64, 1, hx], [64, self.H // 4 * 64, run], [run // 64, 1, hx]),
                                    deps)
                        if self.o_get_ahead and nxt:
                            t.order_key = nxt[g].node_id + 0.5 + 0.01 * r
                        ts.append(t)
                    return ts
                # one pass: 64-byte beats (the 8 lanes on consecutive 8-byte atoms) along a run, the row tiles, the heads
                t = self.nd(g, "xdma", f"o_get_i{i}", "__snax_bingo_kernel_xdma_6d",
                            SnaxBingoKernelXdma6dArgs(src, obuf(g), 8, 8,
                                                      [64, run, OB], [run // 64, self.R // 16, hx],
                                                      [64, self.H // 4 * 64, run], [run // 64, self.R // 16, hx]),
                            deps)
                if self.o_get_ahead and nxt:
                    # o_get_ahead: right behind the GEMM it waits for, ahead of the conversions made after that GEMM
                    t.order_key = nxt[g].node_id + 0.5
                if not hd:
                    return t
                src_d = BingoMemSymbol(src.symbol_name, offset=src.offset + hx * OB, chip_id=src.chip_id)
                td = self.nd(g, "dm", f"o_dget_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                             SnaxBingoKernelIdma2dCopyArgs(src_d, obuf(g).view(hx * run), run, run, self.H // 4 * 64,
                                                           self.R // 16, outer=hd, src_outer=OB, dst_outer=run), deps)
                return [t, td]
            return self.nd(g, "dm", f"o_get_i{i}", "__snax_bingo_kernel_idma_2d_copy",
                           SnaxBingoKernelIdma2dCopyArgs(src, obuf(g), run, run, self.H // 4 * 64, self.R // 16,
                                                         outer=NH, src_outer=OB, dst_outer=run), deps)
        if mc is not None:
            gets = {g: gather(g, self.xland(base), [w]) for g in rg}
        elif self.wo_early:
            mc = mc_of([w])
            gets = {g: gather(g, self.xland(base), [w]) for g in rg}
        else:
            gets = {g: gather(g, self.xland(base), [w]) for g in rg}
            # the multicast after the root's own gathers on its DM core (the stores of y that land over this area follow
            # it); with the gathers on the xDMA core, beside them: both only read the area, and the stores follow both
            mc = mc_of([w] + ([] if self.o_get_xdma else [gets[g] for g in rg]))
        ows = {}
        for g in range(G):
            if self.chip(g) == R:
                continue
            ow = ows[g] = self.nd(g, "dm", f"o_wait_i{i}", "__snax_bingo_kernel_wait_flags",
                                  SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{T}_omflag", offset=i * 64), 1, 1), [])
            self.dfg.bingo_add_order_edge(uput.get(self.chip(g), mc), ow)
            gets[g] = gather(g, self.xland(base), [ow])
        if self.o_overlap:
            # the pass's front on each DM core, in the order made, ahead of the previous pass's first y store there
            for g in range(G):
                prev = self._y_first.get(g)
                if prev is None:
                    continue
                front = [puts[g]] + ([first_ld[g][0]] if g in first_ld else []) + ([ows[g]] if g in ows else [])
                if self.chip(g) == R and self.c(g) == 0:
                    front += [w, mc]
                front.sort(key=lambda t: t.node_id)
                for j, t in enumerate(front):
                    t.order_key = prev.node_id - 1 + (j + 1) / (len(front) + 1)
        self.kernel_barrier()                             # sync barrier: the gather and broadcast of the o8 end
        for g in range(G):
            k = self.chip(g)
            gm, gms = None, []
            GB = self.R * 32 * 2                          # one GEMM's [R, 32] of y, FP16, D layout
            lds = {}

            def wo_load(s, g=g):
                lds[s] = first_ld[g] if (s == 0 and g in first_ld) else self._load(g, i, self.WO0 + s)

            for s in range(self.NWS):
                if s not in lds:
                    wo_load(s)
                ld, si, slab = lds[s]
                if s == 0 and self.o_row_tiles:
                    # o_row_tiles: the first chunk's GEMM a row tile at a time, each behind its row tile of o8
                    mt = self.R // 16
                    rt = [self.gmm(g, f"gwo{s}_i{i}r{r}", obuf(g).view(r * self.H // 4 * 64), slab,
                                   self._yw(g, i).view(s * GB + r * GB // mt), 1, self.H // 4, 32 // BLK, self.d.k_wo,
                                   [ld, gets[g][r]] + self._ywo_war.get(self._ywk(g, s, i), []), f"o{i}")
                          for r in range(mt)]
                    gm = rt[-1]
                    gms.extend(rt[:-1])
                else:
                    gm = self.gmm(g, f"gwo{s}_i{i}", obuf(g), slab, self._yw(g, i).view(s * GB), self.R // 16,
                                  self.H // 4, 32 // BLK, self.d.k_wo,
                                  [ld] + self._dl(gets[g]) + self._ywo_war.get(self._ywk(g, s, i), []), f"o{i}")
                self.ls[g].release(si, gm)
                if self.wo_load_ahead and s + 1 < self.NWS:
                    wo_load(s + 1)
                gms.append(gm)
                after = [gm] + ([mc] if k == R else [])
                if self.SLC == 32:
                    # a slice a GEMM: one copy, the last behind a flag in the root's L3 (the cluster's earlier stores
                    # ran before it on its in-order DM core, so the flag says all have landed)
                    dst = self.y_sym(i * self.YB + (self.WOC * g + 32 * s) * self.R * 2, chip_id=None if k == R else R)
                    if s + 1 < self.NWS:
                        st = self.nd(g, "dm", f"y_store_i{i}s{s}", "__snax_bingo_kernel_idma_1d_copy",
                                     SnaxBingoKernelIdma1dCopyArgs(self.ywo[g].view(s * GB), dst, GB), after)
                    else:
                        st = self.nd(g, "dm", f"y_store_i{i}s{s}", "__snax_bingo_kernel_idma_put_flagged",
                                     SnaxBingoKernelIdmaPutFlaggedArgs(
                                         self.ywo[g].view(s * GB), dst, GB,
                                         on_root(BingoMemSymbol(f"{T}_yflag", offset=(i * G + g) * 64), k), value=1),
                                     after)
                        self._y_last.append(st)
                elif self.y_scatter:
                    if s + 1 == self.NWS:
                        self._y_scatter_stores(g, i, k, gms)
                    continue
                else:
                    # the cluster's one slice of 128 columns (D layout, 8 column tiles a row tile): this GEMM's 2 column
                    # tiles of each row tile into their place
                    dst = self.y_sym(i * self.YB + self.WOC * g * self.R * 2 + s * 2 * TILE16, chip_id=None if k == R else R)
                    st = self.nd(g, "dm", f"y_store_i{i}s{s}", "__snax_bingo_kernel_idma_2d_copy",
                                 SnaxBingoKernelIdma2dCopyArgs(self.ywo[g].view(s * GB), dst, 2 * TILE16, 2 * TILE16,
                                                               (self.SLC // BLK) * TILE16, self.R // 16), after)
                    if s + 1 == self.NWS:
                        self._y_last.append(self.nd(g, "dm", f"y_flag_i{i}", "__snax_bingo_kernel_idma_put_flagged",
                                                    SnaxBingoKernelIdmaPutFlaggedArgs(
                                                        self.ywo[g], on_root(BingoMemSymbol(f"{T}_ydum", offset=g * 64), k), 64,
                                                        on_root(BingoMemSymbol(f"{T}_yflag", offset=(i * G + g) * 64), k),
                                                        value=1), [st]))
                self._ywo_war[self._ywk(g, s, i)] = [st]
                stores.append(st)
            self._ofull_war[g] = [gm]

    def kernel_barrier(self):
        """sync barrier: one chip barrier on every chip after the kernel whose tasks were made before it (the fence ties
        every device task to the barriers around it, in the order made)."""
        if getattr(self, "per_kernel", False):
            self.barrier({k: [] for k in self.chips})

    def _y_scatter_stores(self, g, i, k, gms):
        """y_scatter: cluster g's [R, 128] of pass i (ywo, the W_O GEMMs' [R, 32] D blocks one after the other), row tile by
        row tile into the tile's owner: the GEMMs' 2 column tiles of that row tile, 4 KiB apart, as 8 consecutive column
        tiles at the cluster's place in the tile's row (D layout); then a 64-byte put that raises the cluster's flag there
        (the in-order DM core runs it after the copy)."""
        T, GB = self.TAG, self.R * 32 * 2
        mine = []
        for m in range(self.R // BR):
            q = i * (self.R // BR) + m
            own, lt = self.y_owner(q)
            far = None if own == k else own
            cp = self.nd(g, "dm", f"y_sc_i{i}m{m}", "__snax_bingo_kernel_idma_2d_copy",
                         SnaxBingoKernelIdma2dCopyArgs(self._yw(g, i).view(m * 2 * TILE16),
                                                       self.yt_sym(q, g * self.WOC * BR * 2, chip_id=far),
                                                       2 * TILE16, GB, 2 * TILE16, self.NWS),
                         list(gms))
            fl = self.nd(g, "dm", f"y_sf_i{i}m{m}", "__snax_bingo_kernel_idma_put_flagged",
                         SnaxBingoKernelIdmaPutFlaggedArgs(
                             self._yw(g, i), BingoMemSymbol(f"{T}_ydum", offset=g * 64, chip_id=far), 64,
                             BingoMemSymbol(f"{T}_ytflag", offset=(lt * self.G + g) * 64, chip_id=far), value=1), [cp])
            self._yt_puts.setdefault(own, []).append(fl)
            mine.append(cp)
        for s in range(self.NWS):
            self._ywo_war[self._ywk(g, s, i)] = list(mine)
        self._y_first[g] = mine[0]

    def _trade_ahead_of_y(self, mark):
        """trade_ahead: the DM tasks made from node `mark` on (the next pass's K/V trade), in the order made, ahead of each
        cluster's first y store of the pass before."""
        dm, moved = self.roles["dm"], {}
        for t in self.dfg.node_list:
            if t.node_id >= mark and t.assigned_core_id == dm:
                g = self.chips.index(t.assigned_chiplet_id) * self.NC + t.assigned_cluster_id
                moved.setdefault(g, []).append(t)
        for g, ts in moved.items():
            at = self._y_first[g]
            ts.sort(key=lambda t: t.node_id)
            for j, t in enumerate(ts):
                t.order_key = at.node_id - 1 + (j + 1) / (len(ts) + 1)
        if moved:
            print(f"[{self.TAG}] trade_ahead: {sum(len(ts) for ts in moved.values())} K/V trade tasks on {len(moved)} "
                  f"clusters ahead of the pass before's y stores")

    def _y_scatter_waits(self):
        """y_scatter: on each chip's cluster 0, one wait on its tiles' flags (every cluster's columns of each)."""
        waits = []
        for k in self.chips:
            puts = self._yt_puts.get(k, [])
            w = self.node(k, 0, self.roles["dm"], "yt_wait", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_ytflag"), len(puts), 1),
                          [t for t in puts if t.assigned_chiplet_id == k])
            for t in puts:
                if t.assigned_chiplet_id != k:
                    self.dfg.bingo_add_order_edge(t, w)
            waits.append(w)
        return waits

    # ---- the attention ------------------------------------------------------------------------------------------------
    def build(self):
        if self.o_overlap:
            raise ValueError("params o_overlap: the stage-678 app only. In the block, pass i + 1's o8 waits for that pass's x "
                             "fetch, which follows pass i's y stores on the DM core: the order would close a cycle")
        R, dm, xdma = self.ROOT, self.roles["dm"], self.roles["xdma"]
        self.slot, self.tile_ready, self.rope_ld, self._rope_war, self._o8_war = {}, {}, {}, {}, {}
        self._first_ld, self._proj_lds, self._x_first, self._x_last, self._last_sm = {}, {}, {}, {}, {}
        self._post_first, self._x_xdma = {}, {}
        self._ofull_war, self._ywo_war, self._y_last, self._yt_puts, self._y_first = {}, {}, [], {}, {}
        for ls in self.ls:
            ls.strict = True
        pre = {k: [] for k in self.chips}
        for k in self.chips:
            for c in reversed(range(self.NC)):
                g = self.chips.index(k) * self.NC + c
                h, nb = self.cst[g]
                pre[k] = [self.copy(g, f"const_ld_g{g}", h.address, self.consts[g], nb, pre[k])]
        go = self.setup(extra=pre)
        self.dfg.relay_one_consumer_cores = [xdma]
        others = self.order[1:]
        p1x = {c: f"((uint64_t)({self.p1c[c].get_c_var_name()}))" for c in range(self.NC)}
        sym_reader, p1_reader, stores = {}, {}, []
        last_red, prev_sw, xt_rd = None, {}, {}
        puts = self.x_puts_up_front(go)
        last_proj = {}
        xpre = {}                                   # x_early: pass i's x fetch, made before pass i - 1's sweep

        def fetch_x(i, extra=None):
            # x of pass i into the cluster's buffer once the previous pass's projections are done reading theirs
            # (wo_cols: and once the previous pass's W_O GEMMs are done with its gathered o8, which the static planner then
            # lays the rows over; the in-order GEMM core runs those GEMMs before this pass's projections anyway)
            return {g: self.stage_x(g, i, puts[i], [go[self.chip(g)]] + ([last_proj[g]] if g in last_proj else [])
                                    + (self._ofull_war.get(g, []) if self.wo_cols else [])
                                    + (extra[g] if extra else []))
                    for g in range(self.G)}

        def front(i):
            """Pass i's x fetch and projections (and the multicast of the next pass's x); returns x, q8, own."""
            x = xpre.pop(i) if i in xpre else fetch_x(i)
            q8, own = {}, {}
            for g in range(self.G):
                q8[g], last_proj[g], own[g] = self.stage_proj(g, i, x[g])
            if i + 1 < self.NP and not self.x_mem:
                puts.append(self.stage_x_put(i + 1, [go[R]] + puts[-1:]))
            return x, q8, own

        def trade(i, own):
            if self.kv_split:
                self.stage_trade_kv_split(i, own)
            else:
                for k in self.chips:
                    self.stage_trade_kv(k, i, own)

        ahead = {}                                  # wo_under_proj: pass i's front, made under pass i - 1's W_O
        for i in range(self.NP):
            if i:
                self.pass_start(i)
            if i in ahead:
                x, q8, own = ahead.pop(i)
            else:
                x, q8, own = front(i)
                self.kernel_barrier()               # sync barrier: x and the projections end
                trade(i, own)
                self.kernel_barrier()               # the K/V trade ends
            if self.x_early and not self.x_after_sm and i + 1 < self.NP:
                xpre[i + 1] = fetch_x(i + 1)
            sweeps, post_last = {}, {}
            for g in range(self.G):
                k, c = self.chip(g), self.c(g)
                sweeps[g] = fa_sweep(self.at(k, c), self.sweep_cfg(i + 1), self.sweep_bufs[g], L1Tiles(self, g), q8[g],
                                     self.roles, prev=prev_sw.get(g), oacc_free=xt_rd.get(g), name=f"s{i}_", early=1,
                                     mask=self.mask(i)(g), mask_c=self.mask_c, ot8=self.ot8_bits(),
                                     geom_core=self.sm_geom, sm_batch=self.sm_batch,
                                     sm_async=self.sm_async)
                cv, post_last[g] = self.stage_post(g, i, sweeps[g])
                self._post_first[g, i] = cv[0]
                xt_rd[g] = {u: [cv[u]] for u in range(self.NQ)}
                prev_sw[g] = sweeps[g]
                self._last_sm[g, i] = sweeps[g]["sm"][self.NQ - 1][-1]
            self.kernel_barrier()                   # sync barrier: the sweep ends
            if self.x_after_sm and i + 1 < self.NP:
                # x_after_sm: made after the sweep (its place on the DM core), behind the cluster's last softmax
                xpre[i + 1] = fetch_x(i + 1, None if self.x_xdma_free else
                                      {g: [sweeps[g]["sm"][self.NQ - 1][-1]] for g in range(self.G)})
                if self.x_tiles_xdma:
                    # x_tiles_xdma: ahead of the pass's o8 copies (made before them on the xDMA core) in its program
                    for g in range(self.G):
                        at = self._post_first[g, i].node_id - 0.5
                        for j, t in enumerate(self._x_xdma[g, i + 1]):
                            t.order_key = at + 0.001 * j
            if self.wo_cols:
                between = None
                if self.wo_under_proj and i + 1 < self.NP:
                    def between(i=i):
                        ahead[i + 1] = front(i + 1)
                        return {g: last_proj[g] for g in range(self.G)}
                self.stage_wo_cols(i, post_last, x, stores, between)
                self.kernel_barrier()               # sync barrier: W_O and the y stores end
                if i + 1 in ahead:
                    mark = max(t.node_id for t in self.dfg.node_list) + 1
                    trade(i + 1, ahead[i + 1][2])
                    if self.trade_ahead:
                        self._trade_ahead_of_y(mark)
                continue
            got = {}
            for k in self.chips:
                got.update(self.stage_trade_o(k, i, post_last))
            # W_O slice by slice, each slice's ChainGather through every chip one at a time, the root's store behind it
            for s in range(self.NS):
                slot = s % 2
                for g in range(self.G):
                    k, c = self.chip(g), self.c(g)
                    ld, si, slab = self._load(g, i, self.WO0 + s)
                    gm = self.gmm(g, f"gwo{s}_i{i}", self.o8[g], slab, self.sym[(k, c)].view(slot * self.SLB), self.R // 16,
                                  self.NC * self.HD // 4, self.SLC // BLK, self.d.k_wo,
                                  [ld, post_last[g], got[g]] + sym_reader.get((k, c, slot), []), f"o{i}")
                    self.ls[g].release(si, gm)
                    self._wo_last = gm
                    sym_reader.setdefault(("wo", g), []).append(gm)
                for c in range(self.NC):
                    ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                        cm.remote, R, c, others, src=self.sx(c, slot * self.SLB), dst=f"({p1x[c]} + {slot * self.SLB})",
                        size=self.SLB, junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[self.p1c[c]])
                    first = [last_red] if last_red is not None else []
                    if s == 0 and c == 0 and i + 1 < self.NP:
                        first.append(puts[i + 1])
                    wo_tasks = [sym_reader[("wo", g)][-1] for g in range(self.G) if self.c(g) == c]
                    red = self.node(R, c, xdma, f"y_chain_i{i}s{s}c{c}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                    wo_tasks + p1_reader.get((c, slot), []) + first)
                    last_red = red
                    for k in self.chips:
                        sym_reader[(k, c, slot)] = [red]
                    o0 = c * self.HALF + s * self.SLC
                    st = self.node(R, c, dm, f"y_store_i{i}s{s}c{c}", "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(self.p1c[c].view(slot * self.SLB),
                                                                 self.y_sym(i * self.YB + o0 * self.R * 2),
                                                                 self.SLB), [red])
                    p1_reader[(c, slot)] = [st]
                    stores.append(st)
            # the next pass's o8 writes follow this pass's W_O GEMMs (they read both heads)
            self._o8_war = {g: [sym_reader[("wo", g)][-1]] for g in range(self.G)}
        if not self.restream:
            # a pass after the first copies every chunk from the slot it landed in: each ring holds all of its chunks
            for k in self.chips:
                for r, e in enumerate(self.rings[k].entries[:self.NC]):
                    if len(e) > self.n_slots:
                        raise ValueError(f"params weight_ring slots={self.n_slots}: chip {k:02x} ring {r} takes {len(e)} "
                                         f"chunks, and the passes after the first read each from its slot")
        if self.y_scatter:
            # each chip's tiles of y have landed: one wait a chip
            self.end_window(self._y_scatter_waits())
        elif self.wo_cols:
            # every cluster's slices of every pass have landed: one wait on the root, on their flags
            yw = self.node(R, 0, dm, "y_wait", "__snax_bingo_kernel_wait_flags",
                           SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_yflag"), self.NP * self.G, 1),
                           [t for t in self._y_last if t.assigned_chiplet_id == R])
            for t in self._y_last:
                if t.assigned_chiplet_id != R:
                    self.dfg.bingo_add_order_edge(t, yw)
            self.end_window([yw])
        else:
            self.end_window(stores[-self.NC:])
        self._checks()

    # ---- checks ---------------------------------------------------------------------------------------------------------
    def y_image(self, i):
        """Pass i's reduced y as the root stores it: per cluster half, slice by slice, each slice D layout, FP16."""
        rows = self.gold_y[i * self.R:(i + 1) * self.R]
        out = []
        for c in range(self.NC):
            for s in range(self.NS):
                o0 = c * self.HALF + s * self.SLC
                out.append(self.lay.to_d(rows[:, o0:o0 + self.SLC], self.mesh).astype(np.float16))
        return np.concatenate([a.reshape(-1) for a in out]).view(np.uint8)

    def _checks(self):
        R, n = self.ROOT, 2048
        checks = []
        if self.y_scatter:
            # the first, a middle and the last tile of y in their owners' L3: head, middle and tail of each
            nt = self.NP * self.R // BR
            for q in (0, nt // 2, nt - 1):
                img = self.lay.to_d(self.gold_y[q * BR:(q + 1) * BR], self.mesh).astype(np.float16).reshape(-1).view(np.uint8)
                for nm, off in (("a", 0), ("m", self.YT_B // 2 - n // 2), ("b", self.YT_B - n)):
                    checks.append((f"yt_q{q}{nm}", self.yt_sym(q, off, chip_id=self.y_owner(q)[0]), img[off:off + n], n))
            self.run_checks(checks)
            return
        for i in range(self.NP):
            img = self.y_image(i)
            for nm, off in (("a", 0), ("m", self.YB // 2 - n // 2), ("b", self.YB - n)):
                checks.append((f"y_i{i}{nm}", self.y_sym(i * self.YB + off, chip_id=R),
                               img[off:off + n], n))
        if self.checks_kind == "stages":
            # the first pass's first query tile and key tiles of the root's cluster 0, before the y checks
            g, d = 0, self.d
            h = self.head[g]
            q8 = self.lay.to_a(d.q8[h][:BR][:, self.perm], self.mesh).astype(np.int8)       # in rope_pairs' order
            m = self.group[self.chip(g)]
            kt = self.lay.to_a(d.k8[m][:self.R][:, self.perm], self.mesh).astype(np.int8)
            stage = [("q8", self.stash("q8", self.chip(g), 0, self.pb(g, 0)["q8"][0], q8.size, []), q8.view(np.uint8), q8.size),
                     ("k8", self.stash("k8", self.chip(g), 0, self.kbuf[g], kt.size, []), kt.view(np.uint8), kt.size)]
            checks = stage + checks
        elif self.checks_kind == "kvstages":
            # pass 0's key and value tiles (all R keys) on the root's and its K/V partner's clusters, values first, then
            # the root's cluster 1's query tile, before the y checks: the run's first failing check names the tile
            d, R0 = self.d, self.ROOT
            stage = []
            for k in (R0, self.kv_mate.get(R0, R0)):
                m = self.group[k]
                kt = self.lay.to_a(d.k8[m][:self.R][:, self.perm], self.mesh).astype(np.int8)
                vt = self.lay.to_a(d.v8[m][:self.R].T, self.mesh).astype(np.int8)
                for c in range(self.NC):
                    g = self.chips.index(k) * self.NC + c
                    tag = f"{k:02x}c{c}"
                    stage.append((f"v8_{tag}", self.stash(f"v8_{tag}", k, c, self.vbuf[g], vt.size, []),
                                  vt.view(np.uint8), vt.size))
                    if (k, c) != (R0, 0):
                        stage.append((f"k8_{tag}", self.stash(f"k8_{tag}", k, c, self.kbuf[g], kt.size, []),
                                      kt.view(np.uint8), kt.size))
            g = self.chips.index(R0) * self.NC + 1
            q8 = self.lay.to_a(d.q8[self.head[g]][:BR][:, self.perm], self.mesh).astype(np.int8)
            stage.append(("q8_c1", self.stash("q8_c1", R0, 1, self.pb(g, 0)["q8"][0], q8.size, []), q8.view(np.uint8),
                          q8.size))
            checks = stage + checks
        self.run_checks(checks)

    # ---- emission -------------------------------------------------------------------------------------------------------
    def chain_program_order(self):
        dev = [self.roles[r] for r in ("gemm", "simd", "xdma", "dm")]
        return GemmQueue.of(self.dfg, True).program_order(self.dfg, dev, tag=self.TAG)

    def check_window_local(self):
        """sync barrier: no dependency edge crosses chips between the start barrier and the end stamp; only the
        barriers synchronise the chips there (data a chip waits for by its flag is an order edge, not a dependency).
        Returns the offending edges."""
        import networkx as nx
        pre = set().union(*(nx.ancestors(self.dfg, t) for t in self.go.values()))
        post = nx.descendants(self.dfg, self.end) | {self.end}
        return [(u, v) for u, v in self.dfg.edges()
                if u.assigned_chiplet_id != v.assigned_chiplet_id and u not in pre and u not in post and v not in post]

    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_", "x_copy_")):
        if getattr(self, "per_kernel", False):
            if self.barrier_fence:                  # the fences before the args phases' loads exist
                self.fence_barriers()
                self.barrier_fence = False
            cross = self.check_window_local()
            if cross:
                raise ValueError(f"sync barrier: {len(cross)} dependency edges cross chips inside the window, e.g. "
                                 + "; ".join(f"{u.node_name} -> {v.node_name}" for u, v in cross[:4]))
            print(f"[{self.TAG}] sync barrier: {self.n_barrier} barriers a chip, no edge crosses chips inside the window")
        self.chain_program_order()
        if self.p.get("args_phase_bytes"):
            # params args_phase_bytes: each cluster's device args in phases of at most this many bytes (estimate), two
            # alternating L1 regions (mini_compiler passes/bingo_args_phases.py); a long block's args beside its static buffers
            # overflow L1 otherwise
            from bingo_args_phases import split_args_phases
            split_args_phases(self, int(self.p["args_phase_bytes"]), [self.roles[r] for r in ("gemm", "simd", "xdma", "dm")])
        if self.cfg_reuse:
            n = reuse_gemm_fa_config(self.dfg, [self.roles["gemm"]])
            primed, csr = prime_fa_softmax(self.dfg, [self.roles["simd"]])
            print(f"[{self.TAG}] cfg_reuse: {n} FA matmuls REUSE, softmax {primed} PRIMED + {csr} CSR_PRIMED")
        if self.gemm_reuse:
            n = reuse_gemm_start_config(self.dfg, [self.roles["gemm"]])
            n_cfg = sum(t.kernel_name.endswith("gemm_start") and t.kernel_args.stage != 2 for t in self.dfg.node_list)
            print(f"[{self.TAG}] gemm_reuse: {n} of {n_cfg} gemm_start configurations reuse")
        self.dfg.split_chip_functions = True
        super().finish(app_name, x_prefixes)


def _app_name(S):
    arms = "" if (S.x_cast, S.sync) == ("multicast", "edge") else f", x_cast={S.x_cast} sync={S.sync}"
    return (f"Llama-3-8B attention (synthetic weights), fresh {S.T}-token prompt, {len(S.chips)} chips x {S.NC} clusters, "
            f"passes of {S.R}" + arms)


def run():
    run_driver("llama3 attention workload generator", ABuild, _app_name)
