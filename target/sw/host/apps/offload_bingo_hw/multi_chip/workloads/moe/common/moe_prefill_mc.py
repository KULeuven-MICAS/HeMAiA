# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the moe prefill workloads (workloads/moe/<model>_prefill/<variant>/):
DeepSeek-V2-Lite's layer-1 MoE over a prompt of T tokens on the 16 + 4 chiplet platform, every
GEMM on VersaCore's (8, 8, 16) shape (array shape 3) with INT4 weights. The platform, the weight
rings and the anchors are chiplet_mc.py's; the data and the golden moe_prefill_data.py's.

THE ROUTE runs once, on the root's cluster 0, row block by row block of X: the router GEMM, its
column factors, the softmax and the top 6 of every token (moe_topk_rows, in tasks of TOPK_CHUNK
tokens, each on the cluster's core a model of the route frees first), into a ROUTE TABLE that goes
to every chip. Or (params route dist) every cluster routes its own 8-token tile of X and puts the
tile's entries into the root's table, which the root sends on to every chip row block by row block. Each cluster turns it into the
row lists of its chip's experts (moe_dispatch). The root chip's shared-MLP blocks run on its
cluster 1, and the placement counts the route as the root cluster 0's work.

THE PLACEMENT is static: 4 experts a chip, most padded rows first onto the chip with the least
work so far (the root starting with the router's), each expert's intermediate columns split
over the chip's two clusters by 16-column blocks (cluster 0 an even number), moved until the
clusters' work is even. The shared MLP: each chip a run of its column blocks, split the same
way. Every expert's row count (its tokens, padded to whole 8-row tiles) is fixed here, from the
generator's own route; the device's route is checked against it by moe_dispatch.

PER CLUSTER, in the order the arrays run it:
    shared gate|up   per row block of X (multicast into the chip's L3, copied into L1 at a padded
                     row pitch): one GEMM per weight chunk, its dequantisation and SwiGLU
    experts gate|up  per expert slot: its rows of X gathered from L3 (idma_gather_rows), one
                     GEMM per weight chunk, dequantisation and SwiGLU
    quantise         every row times its route weight (the shared MLP's: 1), into the chip's
                     row-major A8 of the MLP (the cluster's columns at their offset); then each
                     cluster copies its columns into the other cluster's A8
    down             per output slice of the cluster's half of the outputs: the shared MLP's GEMM
                     over the chip's shared columns writes the slice, dequantised; then each
                     expert's GEMM over all its columns, dequantised, and its rows added into the
                     rows of their tokens (simd_scatter_add_rows)
    reduce           one ChainGather a slice through every chip's same cluster into the root's
                     cluster, after its own cluster's previous gather, so cluster 0's and cluster 1's
                     run at the same time (params chain_concurrent false: after the previous gather of
                     either cluster); the root stores the slice
TIMING: the start anchor after every chip's setup; the end anchor after the last store.
"""

import argparse
import os
import sys

import hjson
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE.split("/target/sw/host/")[0]
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "ffn", "common"))

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    INT4_B_LAYOUT_B, INT4_B_LAYOUT_PAIRED, INT4_B_LAYOUT_PARITY, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelGemmShapeInitArgs,
    SnaxBingoKernelGemmStartArgs, SnaxBingoKernelGemmWaitArgs, SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs,
    SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelIdmaGatherRowsArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelIdmaRingLoadArgs, SnaxBingoKernelMoeDispatchArgs, SnaxBingoKernelMoeTopkRowsArgs,
    SnaxBingoKernelSimdDeqSwigluArgs, SnaxBingoKernelSimdMulTilesArgs, SnaxBingoKernelSimdQuantRowsArgs,
    SnaxBingoKernelSimdScatterAddRowsArgs, SnaxBingoKernelSimdSoftmaxF16F16Args, SnaxBingoKernelWaitFlagsArgs,
    SnaxBingoKernelXdmaChainGatherArgs)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from bingo_platform import load_cluster_cfg, parse_platform_cfg  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import moe_prefill_data as md  # noqa: E402

BLK = 16                      # a GEMM column block (Nu)
TOPK_CORES = ("xdma", "dm", "gemm", "simd")   # the root cluster 0 cores the top 6 is spread over
TOPK_US_PER_TOKEN = 1.65      # the top 6 of one token on one core, for the route's model
TOPK_CHUNK = 16               # tokens a top 6 task
TOPK_TAKE = 4                 # params topk_shared: tokens a core takes from its row block's counter at a time
# The route's model on the root's cluster 0, in us a row block of X: its put and the root's own copy
# (DM core), the router GEMM (GEMM core), its factors and softmax (SIMD core, per SOFTMAX_ROWS rows).
# It decides which core each top 6 task runs on and the route's share of the placement.
RT_MODEL = dict(x_put=21.0, x_copy=6.0, router=24.0, deq=6.0, softmax=6.0)
D_BLOCK = 256                 # one D-layout block: 8 rows x 16 FP16
SHAPE3 = 3                    # VersaCore array shape (8, 8, 16)
MESH3 = md.MESH3
ROUTE_B = 32                  # one token's entry of the route table
SOFTMAX_ROWS = 32             # the softmax's rows a task (its scratch fits the SIMD pool)
B_PARITY_OFF = 8              # a parity-layout weight chunk starts one word into its slab: A's rows
                              # start 64-B aligned, so B is an odd number of words after A
EW_F16_ADD = cm.EW_F16_ADD
# params gemm_async: the GEMMs whose waits may trail the next start (same phase), by name prefix. The
# router's own phase: its wait releases the route, so it must not wait behind a shared GEMM's start,
# which waits for that GEMM's weight chunk to stream in. The experts' gate|up phase is per expert slot and
# the down projection's per output slice (gemm_phase): a slot's last wait releases its last SwiGLU and slab,
# while the next slot's first start waits for its row gather; a slice's last wait releases its last
# scatter-add and so its ChainGather, while the next slice's first start waits for the ChainGather two
# slices back to free the output buffer
GEMM_PHASE = {"rt": "route", "sg": "x", "gu": "gate_up", "sd": "down", "dn": "down"}


def gemm_phase(name):
    """The async phase of a GEMM by its name: GEMM_PHASE, the experts' gate|up per slot (gu_s0c1) and the
    down projection per slice (sd_j3, dn_j3s0)."""
    head = name.split("_")[0]
    ph = GEMM_PHASE.get(head, name)
    if ph == "gate_up":
        ph += "_" + name.split("_")[1].split("c")[0]
    elif ph == "down":
        ph += "_" + name.split("_")[1].split("s")[0]
    return ph


def row_pitch(nbytes):
    """A row-major operand's row pitch: at least a row and 16 bytes (an odd quantise writes 16
    values past it), beat-aligned, and 32 mod 64 so the 8 rows of a GEMM pass sit in 8 banks."""
    p = int(nbytes) + 16
    p += (32 - p % 64) % 64
    return p


class HalfSlabs:
    """A cluster's load stream in half slabs (params down_half_slabs): a down projection chunk fits
    half a slab, so the same L1 holds twice as many loads in flight. take() and release() as the
    stream's own; each half's first load waits for whoever last read its whole slab."""

    def __init__(self, ls):
        self.ls, self.rings, self.slabs = ls, ls.rings, ls.slabs
        self.half = ls.slab_bytes // 2 // 64 * 64
        n = len(ls.slabs)
        # slab 0's halves, then slab 1's, ...: consecutive loads alternate between the slabs
        self.order = [(i % n, i // n) for i in range(2 * n)]
        self._next = 0
        self._reader = [ls._reader[s] for s, _ in self.order]

    def take(self, nbytes):
        if nbytes > self.half:
            raise ValueError(f"params down_half_slabs: a {nbytes} B load does not fit half a slab ({self.half} B)")
        i = self._next
        self._next = (i + 1) % len(self.order)
        war = [self._reader[i]] if self._reader[i] is not None else []
        s, h = self.order[i]
        return i, self.slabs[s].view(h * self.half), war

    def release(self, i, reader):
        self._reader[i] = reader


class MBuild(cm.ChipletBuild):
    TAG = "mp"

    def __init__(self, args, p, plat, hw):
        super().__init__(args, p, plat, hw)
        if self.crest:
            raise ValueError("params weight_crest: the prefill streams plain INT4")
        if self.bcast_impl != "bcast_l1" or self.reduce_impl != "chain" or self.sync != "edge":
            raise ValueError("this driver builds bcast bcast_l1, reduce chain, sync edge")
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        d = self.d = md.MoePrefillData(cache, args.dsv2_dir, hw=hw, seed=int(p.get("seed", 1)),
                                       n_calib=int(p.get("n_calib", 8)), n_cal_act=int(p.get("n_cal_act", 32)),
                                       tokens=self.T, inter_e=p.get("inter_e"), inter_sh=p.get("inter_sh"))
        self.lay = d.layout
        self.H, self.IE, self.ISH = d.H, d.I_e, d.I_sh
        self.EB, self.SHB = self.IE // BLK, self.ISH // BLK
        nch = len(self.chips)
        if md.N_EXP % nch:
            raise ValueError(f"{md.N_EXP} experts over {nch} chips")
        self.EPC = md.N_EXP // nch
        # the shared MLP's column blocks per chip (self.shk) and each chip's first block (self.sho):
        # an even split, or (params root_shared_blocks) the root's count set and the rest spread over
        # the other chips, a block more on the first ones -- the root's cluster 0 runs the route
        rsb = p.get("root_shared_blocks")
        if rsb is None:
            if self.SHB % nch:
                raise ValueError(f"{self.SHB} shared blocks over {nch} chips")
            self.shk = {k: self.SHB // nch for k in self.chips}
        else:
            rest = self.SHB - int(rsb)
            if not 0 <= int(rsb) <= self.SHB:
                raise ValueError(f"params root_shared_blocks={rsb}: 0..{self.SHB}")
            others = [k for k in self.chips if k != self.ROOT]
            q, r = divmod(rest, len(others))
            self.shk = {self.ROOT: int(rsb)}
            self.shk.update({k: q + (1 if i < r else 0) for i, k in enumerate(others)})
        self.sho, acc = {}, 0
        for k in self.chips:
            self.sho[k], acc = acc, acc + self.shk[k]
        self.BPC = int(p.get("blocks", 3))                 # gate blocks a gate|up chunk
        self.SW = int(p.get("slice", 128))                 # output columns a down slice
        if (self.H // self.NC) % self.SW or self.SW % BLK:
            raise ValueError(f"params slice={self.SW}: whole column blocks dividing a cluster's {self.H // self.NC} outputs")
        self.NSLC = self.H // self.NC // self.SW           # slices a cluster
        self.XB = int(p.get("x_blocks", 4))                # row blocks of X (shared MLP, router)
        if self.T % (8 * self.XB):
            raise ValueError(f"params x_blocks={self.XB}: whole 8-row tiles of {self.T} tokens a block")
        self.XR = self.T // self.XB
        self.PX = row_pitch(self.H)
        self.PE = row_pitch(self.IE)
        self.PS = row_pitch(BLK * max(self.shk.values()))
        self.SL = self.T * self.SW * 2                     # one output slice, FP16
        self.stop = str(p.get("stop", "full"))
        if self.stop not in ("route", "a8", "full"):
            raise ValueError(f"params stop={self.stop!r}: route, a8 or full")
        # Every ChainGather runs the ring the same way: two chains crossing a D2D link in opposite
        # directions at the same time can deadlock.
        if "chain_alt" in p:
            raise ValueError("params chain_alt: removed, every ChainGather runs the ring the same way")
        # params chain_concurrent (default true): a slice's ChainGather waits for its own cluster's
        # previous gather only, so cluster 0's and cluster 1's gathers run at the same time through
        # every chip. Each hop then shares the chip's wide AXI paths with the other cluster's hop,
        # which needs the xDMA's whole-burst remote writes (xdma_axi_adapter RemoteWriteWholeBurst).
        # False: every gather after the previous one of either cluster.
        self.chain_concurrent = bool(p.get("chain_concurrent", True))
        # params store_lag (0, 1 or 2; default 0): the root's store of slice j's reduced output (p1c -> L3) is
        # emitted after slice j + store_lag's weight loads in its DM core's program, so the root's next slices'
        # loads do not queue behind chain j. p1c is double-buffered: chain j + 2 waits for the store of slice j,
        # which bounds the lag at 2.
        self.store_lag = int(p.get("store_lag", 0))
        if self.store_lag not in (0, 1, 2):
            raise ValueError(f"params store_lag={self.store_lag}: 0, 1 or 2 (p1c holds two slices)")
        # params down_half_slabs: the down projection's weight chunks load into half slabs (HalfSlabs), so
        # w_buffers slabs keep 2 * w_buffers of them in flight
        self.down_half = bool(p.get("down_half_slabs", False))
        # params dtmp_bufs: the expert down GEMMs' output buffers a cluster rotates through (each read
        # by its dequantisation and scatter-add)
        self.ndt = int(p.get("dtmp_bufs", 2))
        if self.ndt < 1:
            raise ValueError(f"params dtmp_bufs={self.ndt}: at least one")
        # params gemm_async: every GEMM is a gemm_start / gemm_wait pair, and the GEMM core's program
        # issues a start before the previous GEMM's wait where the graph allows (chain_program_order),
        # so it programs the next dispatch while the array runs (VersaCore and its streamer queue a
        # configuration)
        self.gemm_async = bool(p.get("gemm_async", False))
        self.gseq, self.gwait, self.gstart = {}, {}, {}
        # params gemm_shape_l1 (with gemm_async): each cluster's GEMM core first copies the array shape's
        # parameters into its L1 (gemm_shape_init), and every gemm_start reads them there instead of L3
        self.gemm_shape_l1 = bool(p.get("gemm_shape_l1", False))
        self.gsh = {}
        # params b_parity: the INT4 weights in the parity layout (gemm.h), one word into their slab,
        # so the B reader and the row-major A reader keep to banks of opposite parity
        self.b_parity = bool(p.get("b_parity", False))
        # params w4_paired: the INT4 weights in the paired layout (blocks of 32 columns, util/layout.py
        # to_b_pairs), which the B reader walks 64 bytes a pass, as A; every GEMM's column tiles come in
        # whole pairs
        self.w4_paired = bool(p.get("w4_paired", False))
        if self.b_parity and self.w4_paired:
            raise ValueError("params b_parity, w4_paired: one INT4 layout")
        # params topk_shared: every root cluster 0 core gets a top 6 task for each row block, and the
        # four share the block's tokens through a counter (whoever is free takes TOPK_TAKE tokens)
        self.topk_shared = bool(p.get("topk_shared", False))
        # params warm_root: a small GEMM (and on cluster 0 a scale and a softmax) on each root cluster
        # right after the start, so the router's and the shared MLP's first tasks find their code cached
        self.warm_root = bool(p.get("warm_root", False))
        # params x_put_c1: the root's cluster 1 DM core puts X to every chip, and both root clusters copy
        # their row blocks from the root's own source without waiting for the multicast, so the router
        # starts as soon as its first block is copied
        self.x_put_c1 = bool(p.get("x_put_c1", False))
        # params x_local: both root clusters copy their row blocks from the root's own source without waiting
        # for the multicast, the puts staying on cluster 0's DM core, each block's copy just ahead of its put
        self.x_local = bool(p.get("x_local", False))
        # params dispatch_blocks: the route table goes out row block by row block as each block's top 6 is
        # done, and every cluster appends a block's tokens to its row lists as it arrives
        self.dispatch_blocks = bool(p.get("dispatch_blocks", False))
        # params route: root (the router, its softmax and the top 6 on the root's cluster 0) or dist (every
        # cluster routes its own 8-token tile of X once that row block lands: the router GEMM, its factors,
        # the softmax and the top 6 on the cluster's xDMA core; it puts the tile's table entries and a flag
        # into the root's landing area, and the root sends each row block's table on to every chip once the
        # block's flags are in -- one multicast source, as with root)
        self.route = str(p.get("route", "root"))
        if self.route not in ("root", "dist"):
            raise ValueError(f"params route={self.route!r}: root or dist")
        self.route_dist = self.route == "dist"
        if self.route_dist and not self.dispatch_blocks:
            raise ValueError("params route dist: needs dispatch_blocks")
        # params route_root_light (route dist): the root's cluster 0, which collects the tiles and sends each
        # row block's table on, runs no shared-MLP blocks and copies only the row block its own tile is in,
        # so no shared weight load or X copy holds its DM core ahead of the route
        self.route_root_light = bool(p.get("route_root_light", False))
        # params route_root_last (route dist): the root's clusters route the last tiles, so their uploads -- behind
        # the root's X puts and copies on its DM cores -- do not hold the first row blocks' tables
        self.route_root_last = bool(p.get("route_root_last", False))
        if self.route_root_last and not self.route_dist:
            raise ValueError("params route_root_last: needs route dist")
        # params l3_overlay: X's source (read on the root in the head), every chip's X landing (read by the row
        # gathers of the experts' gate|up) and the output Y (written by the root's stores in the down phase) share
        # one L3 region, Y over the first two; each root cluster's first store waits for the other root cluster's
        # last row gather and the last X put (its own come before it on its DM core)
        self.l3_overlay = bool(p.get("l3_overlay", False))
        if self.route_root_light and not self.route_dist:
            raise ValueError("params route_root_light: needs route dist")
        # params x_puts_first: the root's cluster 1 DM core puts all X row blocks before anything else it runs
        # (its own copies and weight loads after), and both root clusters copy their blocks from the root's
        # own source; the root's cluster 0 DM core is left the router's copies and the top 6
        self.x_puts_first = bool(p.get("x_puts_first", False))
        # params prefetch_after_x: every chip's weight prefetcher starts once the root's last X row block
        # multicast is done (landed on every chip), so the multicast has the half-duplex D2D links to itself
        # instead of crossing the memory chips' pushes
        self.prefetch_after_x = bool(p.get("prefetch_after_x", False))
        # params shared_after: the shared MLP's gate|up one weight chunk at a time over every X row block, the
        # first chunk as X arrives and chunk j >= 1 after expert slot shared_after[j - 1]'s gate|up (X's blocks
        # copied again from the chip's L3), so no stretch of the shared MLP holds both weight slabs while the
        # ring fills up behind it
        # params fuse_deq_scatter: an expert's down output dequantised by its scatter-add task, one SIMD task
        # instead of two
        self.fuse_deq_scatter = bool(p.get("fuse_deq_scatter", False))
        sa = p.get("shared_after")
        self.shared_after = None if sa is None else [int(x) for x in sa]
        # params x_in_xe: X's row blocks land in the expert row buffers (xe) instead of their own two buffers;
        # the first experts' row gathers follow the last row blocks' GEMMs
        self.x_in_xe = bool(p.get("x_in_xe", False))
        # params route_tile_buf (route dist): each cluster copies its 8-token tile of X from the chip's L3 into
        # its own buffer as soon as the tile's row block lands, ahead of the row blocks' copies, which wait
        # for the shared GEMMs to free a block buffer; its upload follows the second row block's copy
        self.route_tile_buf = bool(p.get("route_tile_buf", False))
        # params x_put_once (x_puts_first): the root multicasts all of X in one put with one landed flag, so the
        # row blocks stream back to back instead of each put waiting for the whole multicast tree to answer
        self.x_put_once = bool(p.get("x_put_once", False))
        # params x_puts_split (x_puts_first): the root's two DM cores put alternate row blocks of X, so two
        # multicasts are in flight instead of one
        self.x_puts_split = bool(p.get("x_puts_split", False))
        if self.x_puts_split and (self.x_put_once or not p.get("x_puts_first", False)):
            raise ValueError("params x_puts_split: needs x_puts_first, not x_put_once")
        # params route_early (route_tile_buf, dispatch_blocks): the root's waits for the tiles and its block
        # multicasts, and every cluster's fetches of the table, follow the cluster's upload on its DM core, ahead
        # of the row blocks the shared GEMMs have not freed a buffer for
        self.route_early = bool(p.get("route_early", False))
        # params slot_order: the order of a chip's expert slots. "rows" (default): most padded rows first, as the
        # placement holds them. "alternate": smallest, largest, second smallest, second largest, ...: the first
        # slot drains the rings the shared MLP's first pass filled, and every array-bound slot (many rows a weight
        # chunk) is followed by a stream-bound one, so the ring absorbs the difference instead of filling up.
        # A list of ranks: slot i holds the chip's expert of rank slot_order[i], ranks counted from the fewest padded
        # rows (ties by expert id), e.g. [1, 2, 3, 0] puts the expert with the fewest rows last
        so = p.get("slot_order", "rows")
        self.slot_order = [int(x) for x in so] if isinstance(so, list) else str(so)
        if isinstance(self.slot_order, list):
            if sorted(self.slot_order) != list(range(self.EPC)):
                raise ValueError(f"params slot_order={so}: a permutation of 0..{self.EPC - 1}")
        elif self.slot_order not in ("rows", "alternate"):
            raise ValueError(f"params slot_order={self.slot_order}: rows, alternate or a list of ranks")
        if self.route_early and not (self.route_tile_buf and p.get("dispatch_blocks", False)):
            raise ValueError("params route_early: needs route_tile_buf and dispatch_blocks")
        if self.x_put_once and not p.get("x_puts_first", False):
            raise ValueError("params x_put_once: needs x_puts_first")
        if self.route_tile_buf and not (self.route_dist and p.get("x_puts_first", False)):
            raise ValueError("params route_tile_buf: needs route dist and x_puts_first")
        if self.x_local + self.x_put_c1 + self.x_puts_first > 1:
            raise ValueError("params x_local, x_put_c1, x_puts_first: one of them")
        # params weight_ring.trailer: every chunk ends in a 64-B record that says it has landed
        # (chunk_trailer), so the prefetcher pushes no flag after it -- one launch a run, not two
        self.plain_trailer = bool(p.get("weight_ring", {}).get("trailer", False))
        self._place()
        if self.slot_order == "alternate":
            for k in self.chips:
                es = sorted(self.held[k], key=lambda e: (md.pad8(d.cnt[e]), e))
                self.held[k] = [es[(i // 2) if i % 2 == 0 else len(es) - 1 - i // 2] for i in range(len(es))]
        elif isinstance(self.slot_order, list):
            for k in self.chips:
                es = sorted(self.held[k], key=lambda e: (md.pad8(d.cnt[e]), e))
                self.held[k] = [es[r] for r in self.slot_order]
        self._split_all()
        print(f"[mp] placement (chip: experts, padded rows): " + "; ".join(
            f"{k:02x}: {self.held[k]} {sum(md.pad8(d.cnt[e]) for e in self.held[k])}" for k in self.chips))
        print(f"[mp] slot rows (params slot_order {self.slot_order}): " + "; ".join(
            f"{k:02x}: {[md.pad8(d.cnt[e]) for e in self.held[k]]}" for k in self.chips))
        self._golden()
        self._stage()
        self.make_streams()
        # The shared gate|up keeps each of its weight chunks in an L1 slab for every row block of X
        # (loaded at the first block, released after the last), so a cluster needs a slab per shared
        # chunk: with fewer, a later chunk's load overwrites one still to be read.
        for g in range(self.G):
            n_sh = len(self.chunks(self.ch(self.chip(g), self.c(g))[0][1]))
            if n_sh > len(self.ls[g].slabs):
                raise ValueError(f"cluster {g} holds {n_sh} shared gate|up chunks in "
                                 f"{len(self.ls[g].slabs)} slabs: params w_buffers >= {n_sh}")
        self.make_rings({})
        self._buffers()

    # ---- the placement ------------------------------------------------------------------------
    def _place(self):
        """4 experts a chip, most padded rows first onto the chip with the least work so far among
        those holding fewer than 4; the root starts with the router's work."""
        cnt = self.d.cnt
        router = self.route_rb()
        # the shared MLP's work too: the same on every chip with the even split
        load = {k: (router if k == self.ROOT else 0.0) + self.T * self.shk[k] for k in self.chips}
        held = {k: [] for k in self.chips}
        for e in sorted(range(md.N_EXP), key=lambda e: (-md.pad8(cnt[e]), e)):
            k = min((k for k in self.chips if len(held[k]) < self.EPC), key=lambda k: (load[k], self.chips.index(k)))
            held[k].append(e)
            load[k] += md.pad8(cnt[e]) * self.EB
        self.held = held

    def _split_all(self):
        """Per chip: the shared blocks (s0, s1) and per expert (b0, b1) of its two clusters, cluster 0
        an even number, blocks moved two at a time from the busier cluster while that evens them."""
        cnt, T = self.d.cnt, self.T
        router = self.route_rb()
        self.shb, self.eb = {}, {}
        for k in self.chips:
            S = self.shk[k]
            # the root's cluster 0 runs the route on all its cores: its shared blocks go to cluster 1
            s0 = 0 if k == self.ROOT and (not self.route_dist or self.route_root_light) else S // 2 + (S // 2) % 2
            sh = [s0, S - s0]
            eb = {}
            for e in self.held[k]:
                b0 = self.EB // 2 - (self.EB // 2) % 2
                eb[e] = [b0, self.EB - b0]

            def loads():
                L = [sh[c] * T + sum(eb[e][c] * md.pad8(cnt[e]) for e in eb) for c in (0, 1)]
                if k == self.ROOT:
                    L[0] += router
                return L
            for _ in range(4 * self.EB):
                L = loads()
                hi = 0 if L[0] > L[1] else 1
                gap = abs(L[0] - L[1])
                best = None
                for e in eb:
                    mv = 4 * md.pad8(cnt[e])
                    if eb[e][hi] >= 4 and abs(gap - mv) < gap and (best is None or abs(gap - mv) < best[0]):
                        best = (abs(gap - mv), e)
                if best is None:
                    break
                eb[best[1]][hi] -= 2
                eb[best[1]][1 - hi] += 2
            self.shb[k] = sh
            self.eb[k] = eb

    def topk_plan(self):
        """The top 6 in TOPK_CHUNK-token tasks, in row-block order, each on the root cluster 0 core the
        modelled route (RT_MODEL) frees first: the xDMA core idle from the start, the DM core after its X
        puts and copies, the GEMM core after the router, the SIMD core after the last softmax. Returns
        [(first token, core role)], the modelled end of the router and of the last top 6 task (us)."""
        m = RT_MODEL
        dm = gemm = simd = 0.0
        ready = []
        for _ in range(self.XB):
            dm += m["x_put"] + m["x_copy"]
            gemm = max(gemm, dm) + m["router"]
            simd = max(simd, gemm) + m["deq"] + self.XR // SOFTMAX_ROWS * m["softmax"]
            ready.append(simd)
        free = {"xdma": 0.0, "dm": dm, "gemm": gemm, "simd": simd}
        plan = []
        for b in range(self.XB):
            for t0 in range(b * self.XR, (b + 1) * self.XR, TOPK_CHUNK):
                r = min(TOPK_CORES, key=lambda c: max(free[c], ready[b]))
                free[r] = max(free[r], ready[b]) + min(TOPK_CHUNK, (b + 1) * self.XR - t0) * TOPK_US_PER_TOKEN
                plan.append((t0, r))
        return plan, gemm, max(free.values())

    def route_rb(self):
        """The route's work on the root's cluster 0 in row-blocks (one row against one 16-column block of
        gate, up and down): the router GEMM, and the arrays' idle time from the router's end to the last
        top 6 task's, in the route's model (at 1,024 MAC a cycle, 500 MHz)."""
        if self.route_dist:
            return 0.0
        gemm = self.T * 64 / (BLK * 3)
        _, rt_end, tk_end = self.topk_plan()
        idle = max(0.0, tk_end - rt_end) * 1e-6 * 5e8 * 1024 / (BLK * 3 * self.H)
        return gemm + idle

    def ch(self, k, c):
        """Cluster c of chip k's column ranges: shared (first global shared column, columns) and per
        expert (first column within the expert, columns)."""
        i = self.chips.index(k)
        s0, s1 = self.shb[k]
        sh = (BLK * (self.sho[k] + (0 if c == 0 else s0)), BLK * (s0 if c == 0 else s1))
        ex = {e: ((0, BLK * b0) if c == 0 else (BLK * b0, BLK * b1)) for e, (b0, b1) in self.eb[k].items()}
        return sh, ex

    def chunks(self, n_cols):
        """[(first column, columns)] of a run of n_cols intermediate columns in gate|up chunks."""
        nb = n_cols // BLK
        return [(BLK * b, BLK * min(self.BPC, nb - b)) for b in range(0, nb, self.BPC)]

    def out_slices(self, c):
        """Cluster c's output slices: [(first output column, columns)]."""
        h = self.H // self.NC
        return [(c * h + j * self.SW, self.SW) for j in range(self.NSLC)]

    # ---- the golden ---------------------------------------------------------------------------
    def chain_order(self, c):
        """The chips a ChainGather of cluster c passes, the root last; both clusters' the same way."""
        return self.order[1:] + [self.ROOT]

    def _golden(self):
        d, add16 = self.d, self.d.fp.add16
        y = np.zeros((self.T, self.H), dtype=np.float16)
        for c in range(self.NC):
            for c0, w in self.out_slices(c):
                parts = {}
                for k in self.chips:
                    i = self.chips.index(k)
                    ks0 = BLK * self.sho[k]
                    parts[k] = d.chip_partial((ks0, ks0 + BLK * self.shk[k]), self.held[k], c0, w)
                order = self.chain_order(c)
                acc = parts[order[0]]
                for k in order[1:]:
                    acc = add16(acc, parts[k])
                y[:, c0:c0 + w] = acc
        self.y_raw = y
        for line in d.report(y):
            print(f"[mp] {line}")

    # ---- data --------------------------------------------------------------------------------
    def _pack_b(self, q):
        if self.w4_paired:
            if q.shape[1] % (2 * MESH3[2]):
                raise ValueError(f"params w4_paired: a weight chunk of {q.shape[1]} columns is not whole pairs of "
                                 f"{MESH3[2]}-column tiles")
            return self.lay.pack_int4(self.lay.to_b_pairs(q, MESH3))
        b = self.lay.pack_int4(self.lay.to_b(q, MESH3))
        if not self.b_parity:
            return b
        # the parity layout: each pair of passes (64 B each, eight words) interleaved word by word
        K, N = q.shape
        if (K // MESH3[1]) % 2:
            raise ValueError(f"params b_parity: {K // MESH3[1]} passes of K is odd")
        return np.ascontiguousarray(np.asarray(b).reshape(N // MESH3[2], K // MESH3[1] // 2, 2, 8, 8)
                                    .transpose(0, 1, 3, 2, 4)).reshape(-1)

    def _scale_tile(self, f):
        return self.lay.to_d(np.tile(np.asarray(f, dtype=np.float16), (8, 1)), MESH3).astype(np.float16)

    def _chunk(self, w_bytes, f):
        """A weight chunk as it is streamed: the INT4 weights, then (64-B aligned) the scale tile of
        its output columns. Returns (bytes, offset of the scale tile)."""
        wb = np.asarray(w_bytes, dtype=np.int8).reshape(-1)
        if self.b_parity:
            wb = np.concatenate([np.zeros(B_PARITY_OFF, dtype=np.int8), wb])
        off = -(-wb.size // 64) * 64
        st = self._scale_tile(f).view(np.int8)
        parts = [wb, np.zeros(off - wb.size, dtype=np.int8), st]
        if self.plain_trailer:
            parts.append(self.chunk_trailer())
        return np.concatenate(parts), off

    def _stage(self):
        st, d, T, R = self.st, self.d, self.TAG, self.ROOT
        hq = np.asarray(d.hq, dtype=np.int8)
        if self.l3_overlay:
            # X's source, then every chip's X landing (the prompt's rows and the zero row the experts' padding rows
            # read); Y lies over the first T * H * 2 bytes
            assert self.T * self.H * 2 <= (2 * self.T + 1) * self.H
            st.put(f"{T}_xy", "int8_t", np.concatenate([hq.reshape(-1), np.zeros((self.T + 1) * self.H, np.int8)]))
        else:
            st.put(f"{T}_xsrc", "int8_t", hq.reshape(-1))
            # every chip's X: the prompt's rows and the zero row the experts' padding rows read
            st.put_zeros(f"{T}_xland", "int8_t", (self.T + 1) * self.H, preload=True)
        st.put_zeros(f"{T}_xflag", "uint8_t", 64 * self.XB)
        st.put_zeros(f"{T}_rland", "uint8_t", self.T * ROUTE_B, preload=True)
        st.put_zeros(f"{T}_rflag", "uint8_t", 64 * (self.XB if self.dispatch_blocks else 1))
        if self.route_dist:
            st.put_zeros(f"{T}_rup", "uint8_t", 64 * (self.T // 8))   # the root's: one flag a tile
        if self.topk_shared or self.warm_root:
            st.put_zeros(f"{T}_zero1k", "uint8_t", 1024)
        if not self.l3_overlay:
            st.put_zeros(f"{T}_y", "uint8_t", self.T * self.H * 2, preload=True)
        rw = self.lay.to_b(np.asarray(d.wr_q, dtype=np.int8), MESH3).astype(np.int8)
        self.h_rw = st.put(f"{T}_rw", "int8_t", rw)
        self.h_rws = st.put(f"{T}_rws", "int8_t", self._scale_tile(d.wr_f).view(np.int8))
        # each cluster's weights in the order it loads them: its shared gate|up chunks, its experts'
        # gate|up chunks slot by slot, then per output slice the shared down chunk and each slot's
        self.img, self.chunk_off = {}, {}
        sh_mp = d.sh
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            (sc0, sn), ex = self.ch(k, c)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts, offs = [], []

            def add(blob, soff):
                off0 = sum(b.size for b in parts)
                parts.append(blob)
                offs.append((off0, blob.size, soff))

            for b0, n in self.chunks(sn):
                cols = np.concatenate([np.arange(sc0 + b0, sc0 + b0 + n), self.ISH + np.arange(sc0 + b0, sc0 + b0 + n)])
                add(*self._chunk(self._pack_b(sh_mp["gu_q"][:, cols]), sh_mp["gu_f"][cols]))
            for e in self.held[k]:
                mp = d.expert(e)
                e0, en = ex[e]
                for b0, n in self.chunks(en):
                    cols = np.concatenate([np.arange(e0 + b0, e0 + b0 + n), self.IE + np.arange(e0 + b0, e0 + b0 + n)])
                    add(*self._chunk(self._pack_b(mp["gu_q"][:, cols]), mp["gu_f"][cols]))
            i = self.chips.index(k)
            ks0 = BLK * self.sho[k]
            for o0, w in self.out_slices(c):
                add(*self._chunk(self._pack_b(sh_mp["dn_q"][ks0:ks0 + BLK * self.shk[k], o0:o0 + w]),
                                 sh_mp["dn_f"][o0:o0 + w]))
                for e in self.held[k]:
                    mp = d.expert(e)
                    add(*self._chunk(self._pack_b(mp["dn_q"][:, o0:o0 + w]), mp["dn_f"][o0:o0 + w]))
            for b in parts:
                if b.size > self.chunk:
                    raise ValueError(f"a weight chunk of {b.size} B; params w_chunk_bytes is {self.chunk}")
            self.img[g] = st.put_hbm(f"{T}_w_g{g}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[g] = offs

    def _buffers(self):
        d, T = self.d, self.T
        # route dist: the 8-token tiles each cluster routes, spread evenly over the clusters
        ntile = T // 8
        self.rt_tiles = {g: [i for i in range(ntile) if i * self.G // ntile == g] for g in range(self.G)}
        if self.route_root_last:
            # the same spread, rotated by one chip so the root's clusters get the last tiles
            self.rt_tiles = {g: [] for g in range(self.G)}
            for i in range(ntile):
                self.rt_tiles[(i * self.G // ntile + self.NC) % self.G].append(i)
        if self.route_dist and max(len(v) for v in self.rt_tiles.values()) > 1:
            raise ValueError(f"params route dist: {ntile} tiles over {self.G} clusters, at most one a cluster")
        self.sym = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_sym", 2 * self.SL)
                    for k in self.chips for c in range(self.NC)}
        self.sym_zero = False                    # the shared down GEMM writes each slot whole first
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", 2 * self.SL) for c in range(self.NC)}
        B = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nb)
            (_, sn), ex = self.ch(k, c)
            rows = [md.pad8(d.cnt[e]) for e in self.held[k]]
            mrow = max(rows)
            gt = max(self.XR, mrow) // 8 * 2 * self.BPC * D_BLOCK
            nmax = max(n for _, n in ex.values()) // BLK
            b = B[g] = dict(
                xblk=None if self.x_in_xe else [l1(f"xblk{i}", self.XR * self.PX) for i in range(2)],
                # params shared_after, x_in_xe: X's row blocks are copied into these too
                xe=[l1(f"xe{i}", (max(mrow, self.XR) if (self.shared_after or self.x_in_xe) else mrow) * self.PX)
                    for i in range(2)],
                gbuf=[l1(f"g{i}", gt) for i in range(2)],
                sg=l1("sg", gt),
                # one A16 for every expert: the SIMD core quantises an expert before its next SwiGLU
                a16e=l1("a16e", mrow // 8 * nmax * D_BLOCK + D_BLOCK),
                a8sh=l1("a8sh", T * self.PS),
                a8e=[l1(f"a8e{s}", r * self.PE) for s, r in enumerate(rows)],
                dtmp=[l1(f"dtmp{i}", mrow * self.SW * 2) for i in range(self.ndt)],
                rt=l1("rt", T * ROUTE_B),
                lists=l1("lists", SnaxBingoKernelMoeDispatchArgs.list_bytes(rows)))
            if self.x_in_xe:
                b["xblk"] = b["xe"]
            if self.dispatch_blocks:
                b["dst"] = l1("dst", 64)                       # the row lists' counts between blocks
            if sn:
                b["a16sh"] = l1("a16sh", T // 8 * (sn // BLK) * D_BLOCK + D_BLOCK)
            if self.route_dist and self.rt_tiles[g]:       # route dist: the cluster's tile
                if self.route_tile_buf:
                    b["rx"] = l1("rx", 8 * self.PX)
                b["rw"] = l1("rw", self.H * 64)
                b["rws"] = l1("rws", 4 * D_BLOCK)
                b["lg"] = l1("lg", 8 * 64 * 2)
                b["pin"] = l1("pin", 8 * 128)
                b["pp"] = l1("pp", 8 * 128)
            if g == 0 and not self.route_dist:             # the root's cluster 0: the router
                b["rw"] = l1("rw", self.H * 64)
                b["rws"] = l1("rws", 4 * D_BLOCK)
                b["lg"] = [l1(f"lg{i}", self.XR * 64 * 2) for i in range(2)]
                b["pin"] = l1("pin", T * 128)
                b["pp"] = l1("pp", T * 128)
                if self.topk_shared:
                    b["tkc"] = l1("tkc", 64)               # one word a row block
            if self.warm_root and (k == self.ROOT or self.route_dist):
                b["wz"] = l1("wz", 1024)
        self.B = B

    # ---- helpers ----------------------------------------------------------------------------
    def _load(self, g, j, deps=()):
        """Weight chunk j of cluster g from its ring into the next slab. Returns (load, slab index,
        slab, offset of its scale tile)."""
        k, c, dm = self.chip(g), self.c(g), self.roles["dm"]
        ls = self.ls[g]
        off, nb, soff = self.chunk_off[g][j]
        si, slab, war = ls.take(nb)
        rings = ls.rings
        sl, fl, rl, seq, kind, _ = rings.take_ex(c, src=self.img[g].address + off, nbytes=nb)
        prev = [rings.last[c]] if rings.last.get(c) is not None else []
        ld = self.node(k, c, dm, f"Ld_w{j}_g{g}", "__snax_bingo_kernel_idma_ring_load",
                       SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, slab, nb, rl, trailer=bool(kind & 2)),
                       list(dict.fromkeys(war + prev + list(deps))))
        rings.last[c] = ld
        ld.exec_chunk = rings.last_take
        return ld, si, slab, soff

    def gemm(self, k, c, name, a, a_pitch, b, dst, M, K, N, k_shift, deps, int4=1):
        """One GEMM; returns the task its output is ready after (params gemm_async: the wait)."""
        lay = 0
        if int4:
            lay = INT4_B_LAYOUT_PARITY if self.b_parity else INT4_B_LAYOUT_PAIRED if self.w4_paired else INT4_B_LAYOUT_B
            if self.b_parity:
                b = b.view(B_PARITY_OFF)
            if self.w4_paired and N % 2:
                raise ValueError(f"params w4_paired: GEMM {name} has {N} column tiles, not whole pairs")
        kw = dict(input_A_addr=a, input_B_addr=b, input_C_addr=0, output_D_addr=dst, M=M, K=K, N=N,
                  array_shape_idx=SHAPE3, transpose_A=0, transpose_B=0, accumPrevC=0,
                  int32tofp16_enable=1, int4_b_enable=lay, d_shift=k_shift, a_pitch=a_pitch)
        gc = self.roles["gemm"]
        if not self.gemm_async:
            return self.node(k, c, gc, name, "__snax_bingo_kernel_gemm_full", SnaxBingoKernelGemmFullArgs(**kw),
                             list(deps))
        seq = self.gseq[(k, c)] = self.gseq.get((k, c), 0) + 1
        shape_addr = 0
        if self.gemm_shape_l1:
            if (k, c) not in self.gsh:
                buf = self.cx_chip(k).at(c).l1(f"{self.TAG}_gsh", SnaxBingoKernelGemmShapeInitArgs.BYTES)
                self.node(k, c, gc, "gsh_init", "__snax_bingo_kernel_gemm_shape_init",
                          SnaxBingoKernelGemmShapeInitArgs(buf, SHAPE3))
                self.gsh[(k, c)] = buf
            shape_addr = self.gsh[(k, c)]
        st = self.node(k, c, gc, name, "__snax_bingo_kernel_gemm_start",
                       SnaxBingoKernelGemmStartArgs(**kw, seq=seq, shape_addr=shape_addr), list(deps))
        wt = self.node(k, c, gc, f"{name}_w", "__snax_bingo_kernel_gemm_wait", SnaxBingoKernelGemmWaitArgs(seq), [st])
        # the phase a start opens: a wait moves behind the next start of its own phase only
        self.gwait[wt] = self.gstart[st] = gemm_phase(name)
        return wt

    # ---- the prefill --------------------------------------------------------------------------
    def build(self):
        d, R, T, Tg = self.d, self.ROOT, self.T, self.TAG
        dm, simd, xdma = self.roles["dm"], self.roles["simd"], self.roles["xdma"]
        if self.p.get("relay_per_gather", True):
            self.dfg.relay_one_consumer_cores = [xdma]
        B0 = self.B[0]
        pre = {k: [] for k in self.chips}
        zero = BingoMemSymbol(f"{Tg}_zero1k")
        self.rw_ld = {}
        if self.route_dist:
            # every routing cluster's router weights and their factors, ahead of the window on its DM core; its
            # router GEMM and factors wait for them on the chip (only the root's setup holds the start anchor,
            # whose dependency cell would otherwise collect every chip's)
            for g in range(self.G):
                if self.rt_tiles[g]:
                    k, c, bg = self.chip(g), self.c(g), self.B[g]
                    self.rw_ld[g] = (self.node(k, c, dm, "rw_load", "__snax_bingo_kernel_idma_1d_copy",
                                               SnaxBingoKernelIdma1dCopyArgs(self.h_rw, bg["rw"], self.H * 64)),
                                     self.node(k, c, dm, "rws_load", "__snax_bingo_kernel_idma_1d_copy",
                                               SnaxBingoKernelIdma1dCopyArgs(self.h_rws, bg["rws"], 4 * D_BLOCK)))
                    if k == R:
                        pre[k] += list(self.rw_ld[g])
        else:
            pre[R] = [self.node(R, 0, dm, "rw_load", "__snax_bingo_kernel_idma_1d_copy",
                                SnaxBingoKernelIdma1dCopyArgs(self.h_rw, B0["rw"], self.H * 64)),
                      self.node(R, 0, dm, "rws_load", "__snax_bingo_kernel_idma_1d_copy",
                                SnaxBingoKernelIdma1dCopyArgs(self.h_rws, B0["rws"], 4 * D_BLOCK))]
            if self.topk_shared:
                pre[R].append(self.node(R, 0, dm, "tkc_init", "__snax_bingo_kernel_idma_1d_copy",
                                        SnaxBingoKernelIdma1dCopyArgs(zero, B0["tkc"], 64)))
        if self.warm_root:
            # the code of the first tasks into the instruction cache, on zeros, before the start (the X
            # puts would otherwise starve its fetches): A 8 rows of pitch 32 at 0, B at 256, D at 512, the
            # SIMD's output at 768. The root's clusters, or (route dist) every chip's, which all route
            for k in (self.chips if self.route_dist else [R]):
                for c in range(self.NC):
                    wz = self.B[self.gs(k)[c]]["wz"]
                    wz_init = self.node(k, c, dm, "wz_init", "__snax_bingo_kernel_idma_1d_copy",
                                        SnaxBingoKernelIdma1dCopyArgs(zero, wz, 1024))
                    # one column tile: INT8 B with the paired layout, which takes whole pairs only
                    # the root's warm-up holds the start anchor; another chip's only runs first on its cores
                    warm = pre[k] if k == R else []
                    warm.append(self.gemm(k, c, "warm_gemm", wz, 32, wz.view(256), wz.view(512), 1, 2, 1, 0,
                                          [wz_init], int4=0 if self.w4_paired else 1))
                    if c == 0 or self.route_dist:
                        warm.append(self.node(k, c, simd, "warm_scale", "__snax_bingo_kernel_simd_mul_tiles",
                                              SnaxBingoKernelSimdMulTilesArgs(wz.view(512), wz.view(512),
                                                                              wz.view(768), 1, 1), [wz_init]))
                        warm.append(self.node(k, c, simd, "warm_softmax", "__snax_bingo_kernel_simd_softmax_f16_f16",
                                              SnaxBingoKernelSimdSoftmaxF16F16Args(wz, wz.view(768), 2, 64),
                                              [warm[-1]]))
        go = self.setup(extra=pre)
        G, XR = self.G, self.XR

        # ---- X into every chip, row block by row block; the router and the shared gate|up ----
        # each block's put is created with the block, so the root's DM core copies a block for the router
        # right after putting it, before the next put
        puts = []
        xblk_reader = {g: [[], []] for g in range(G)}
        gb_reader = {g: [None, None] for g in range(G)}
        self.gcount = {g: 0 for g in range(G)}
        sh_ld = {}
        last_sw = {g: None for g in range(G)}
        lg_reader, rtr_mul = [None, None], []
        rt_sm, blk_sm = [], []
        rt_tile = {}                                        # route dist: tile -> (cluster, its top 6)

        def next_g(g):
            i = self.gcount[g] % 2
            self.gcount[g] += 1
            return i, self.B[g]["gbuf"][i]

        def root_copy(g, b):
            """The root's own copy of row block b into cluster g's block buffer, from its source."""
            c = self.c(g)
            return self.node(R, c, dm, f"x_copy_b{b}_g{g}", "__snax_bingo_kernel_idma_2d_copy",
                             SnaxBingoKernelIdma2dCopyArgs(self.xsrc(b * XR * self.H),
                                                           self.B[g]["xblk"][b % 2], self.H, self.H, self.PX, XR),
                             [go[R]] + xblk_reader[g][b % 2])

        def put(b, nblk=1):
            pc = (b % 2) if self.x_puts_split else (1 if (self.x_put_c1 or self.x_puts_first) else 0)
            x_last[pc] = self.node(R, pc, dm, f"x_bput_b{b}", "__snax_bingo_kernel_idma_put_flagged",
                                   SnaxBingoKernelIdmaPutFlaggedArgs(
                                       self.xsrc(b * XR * self.H), self.xland(b * XR * self.H), nblk * XR * self.H,
                                       BingoMemSymbol(f"{Tg}_xflag", offset=xf(b)), value=1, bcast=True,
                                       also_here=True), [go[R]] + ([x_last[pc]] if pc in x_last else []))
            return x_last[pc]

        def xf(b):
            """row block b's landed flag (params x_put_once: the one flag of the whole put)"""
            return 0 if self.x_put_once else b * 64

        def route_tile(g, i, a, deps):
            """route dist: cluster g's tile i (8 tokens) from its rows at `a`: the router GEMM, its factors, the
            softmax and the top 6 into the cluster's route table"""
            k, c = self.chip(g), self.c(g)
            bg = self.B[g]
            rg = self.gemm(k, c, f"rt_t{i}", a, self.PX, bg["rw"], bg["lg"], 1, self.H // 8, 4, d.k_x,
                           list(deps) + [self.rw_ld[g][0]], int4=0)
            mt = self.node(k, c, simd, f"rt_deq_t{i}", "__snax_bingo_kernel_simd_mul_tiles",
                           SnaxBingoKernelSimdMulTilesArgs(bg["lg"], bg["rws"], bg["pin"], 1, 4, dst_pitch=128),
                           [rg, self.rw_ld[g][1]])
            sm = self.node(k, c, simd, f"rt_softmax_t{i}", "__snax_bingo_kernel_simd_softmax_f16_f16",
                           SnaxBingoKernelSimdSoftmaxF16F16Args(bg["pin"], bg["pp"], 8, 64), [mt])
            rt_tile[i] = (g, self.node(k, c, xdma, f"rt_topk_t{i}", "__snax_bingo_kernel_moe_topk_rows",
                                       SnaxBingoKernelMoeTopkRowsArgs(bg["pp"], 128, 64, md.TOP_K, 0, 8,
                                                                      bg["rt"].view(8 * i * ROUTE_B)), [sm]))
            return rg

        def route_up(i):
            """route dist: tile i's entries and flag into the root's landing area"""
            g, tk = rt_tile[i]
            k, c, o = self.chip(g), self.c(g), 8 * i * ROUTE_B
            rt_ups[i] = self.node(k, c, dm, f"rt_up_t{i}", "__snax_bingo_kernel_idma_put_flagged",
                                  SnaxBingoKernelIdmaPutFlaggedArgs(
                                      self.B[g]["rt"].view(o), BingoMemSymbol(f"{Tg}_rland", offset=o, chip_id=R),
                                      8 * ROUTE_B, BingoMemSymbol(f"{Tg}_rup", offset=64 * i, chip_id=R), value=1),
                                  [tk])
            w = rt_wait.get(8 * i // XR)
            if w is not None:                               # params route_early: the root's wait already exists
                if k == R:
                    self.edge(rt_ups[i], w)
                else:
                    self.dfg.bingo_add_order_edge(rt_ups[i], w)

        def root_collect():
            """route dist: per row block, the root waits for the block's tile flags and sends the block's table on
            to every chip (and its own flag)"""
            for b in range(self.XB):
                tl = [i for i in range(T // 8) if 8 * i // XR == b]
                local = [rt_ups[i] for i in tl if i in rt_ups and self.chip(rt_tile[i][0]) == R]
                w = self.node(R, 0, dm, f"rt_wait_b{b}", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_rup", offset=64 * tl[0]), len(tl), 1),
                              [go[R]] + local + rputs[-1:])
                for i in tl:
                    if i in rt_ups and self.chip(rt_tile[i][0]) != R:
                        self.dfg.bingo_add_order_edge(rt_ups[i], w)
                rt_wait[b] = w
                o = b * XR * ROUTE_B
                rputs.append(self.node(R, 0, dm, f"rt_bput_b{b}", "__snax_bingo_kernel_idma_put_flagged",
                                       SnaxBingoKernelIdmaPutFlaggedArgs(
                                           BingoMemSymbol(f"{Tg}_rland", offset=o), BingoMemSymbol(f"{Tg}_rland", offset=o),
                                           XR * ROUTE_B, BingoMemSymbol(f"{Tg}_rflag", offset=64 * b), value=1,
                                           bcast=True, also_here=True), [w]))

        def fetch_dispatch(g):
            """dispatch_blocks: cluster g fetches each block's table as the root sends it and turns it into its
            chip's row lists"""
            k, c = self.chip(g), self.c(g)
            bg = self.B[g]
            caps = [md.pad8(d.cnt[e]) for e in self.held[k]]
            cnts = [int(d.cnt[e]) for e in self.held[k]]
            prev = []
            for b in range(self.XB):
                o = b * XR * ROUTE_B
                f = self.node(k, c, dm, f"rt_fetch_b{b}_g{g}", "__snax_bingo_kernel_idma_fetch_flagged",
                              SnaxBingoKernelIdmaFetchFlaggedArgs(
                                  BingoMemSymbol(f"{Tg}_rland", offset=o), bg["rt"].view(o), XR * ROUTE_B,
                                  BingoMemSymbol(f"{Tg}_rflag", offset=64 * b), 1, 1),
                              [rputs[b]] if k == R else [go[k]])
                if k != R:
                    self.dfg.bingo_add_order_edge(rputs[b], f)
                prev = [self.node(k, c, xdma, f"rt_dispatch_b{b}_g{g}", "__snax_bingo_kernel_moe_dispatch",
                                  SnaxBingoKernelMoeDispatchArgs(bg["rt"], T, md.TOP_K, self.held[k], bg["lists"],
                                                                 caps, cnts, pad_row=T, t0=b * XR, t1=(b + 1) * XR,
                                                                 state_addr=bg["dst"]), [f] + prev)]
            disp_early[g] = prev[0]

        rt_ups, rt_wait, disp_early = {}, {}, {}
        x_last = {}                                         # each root DM core's last X put
        rputs, root_disp = [], []                           # dispatch_blocks: each block's put, the root's lists
        if self.x_puts_first and self.x_put_once:
            puts += [put(0, self.XB)] * self.XB
        elif self.x_puts_first:
            for b in range(self.XB):
                puts.append(put(b))
        if self.route_tile_buf:
            # every cluster's tile first: its rows straight from the chip's X (the root: its source)
            for g in range(G):
                k, c = self.chip(g), self.c(g)
                for i in self.rt_tiles[g]:
                    bt = 8 * i // XR
                    if k == R:
                        src, deps = self.xsrc(8 * i * self.H), [go[R]]
                    else:
                        w = self.node(k, c, dm, f"rt_xwait_t{i}", "__snax_bingo_kernel_wait_flags",
                                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xflag", offset=xf(bt)), 1, 1),
                                      [go[k]])
                        self.dfg.bingo_add_order_edge(puts[bt], w)
                        src, deps = self.xland(8 * i * self.H), [w]
                    xc = self.node(k, c, dm, f"rt_xcopy_t{i}", "__snax_bingo_kernel_idma_2d_copy",
                                   SnaxBingoKernelIdma2dCopyArgs(src, self.B[g]["rx"], self.H, self.H, self.PX, 8), deps)
                    route_tile(g, i, self.B[g]["rx"], [xc])
        for b in range(self.XB):
            early = {0: root_copy(0, b)} if self.x_local else {}
            if not self.x_puts_first:
                puts.append(put(b))
            for g in range(G):
                k, c = self.chip(g), self.c(g)
                bg = self.B[g]
                xb = bg["xblk"][b % 2]
                if (self.route_root_light and g == 0 and not self.ch(k, c)[0][1]
                        and not any(8 * i // XR == b for i in self.rt_tiles[g])):
                    continue                                # route_root_light: a block the root cluster 0 does not use
                if k == R and (self.x_put_c1 or self.x_local or self.x_puts_first):
                    xc = early[g] if g in early else root_copy(g, b)
                else:
                    w = self.node(k, c, dm, f"x_wait_b{b}_g{g}", "__snax_bingo_kernel_wait_flags",
                                  SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xflag", offset=xf(b)), 1, 1),
                                  [go[k]] + ([puts[b]] if k == R else []))
                    if k != R:
                        self.dfg.bingo_add_order_edge(puts[b], w)
                    xc = self.node(k, c, dm, f"x_copy_b{b}_g{g}", "__snax_bingo_kernel_idma_2d_copy",
                                   SnaxBingoKernelIdma2dCopyArgs(self.xland(b * XR * self.H),
                                                                 xb, self.H, self.H, self.PX, XR),
                                   [w] + xblk_reader[g][b % 2])
                xblk_reader[g][b % 2] = []
                (sc0, sn), _ = self.ch(k, c)
                if b == 0:                                  # the shared gate|up chunks, held for every block
                    nsj = len(self.chunks(sn)) if self.shared_after is None else min(1, len(self.chunks(sn)))
                    sh_ld[g] = [self._load(g, j) for j in range(nsj)]
                if self.route_tile_buf:
                    if b == min(1, self.XB - 1):            # the cluster's upload, behind its first row blocks' copies
                        for i in self.rt_tiles[g]:
                            route_up(i)
                        if self.route_early:
                            if g == 0:
                                root_collect()
                            fetch_dispatch(g)
                for i in (self.rt_tiles[g] if self.route_dist and not self.route_tile_buf else ()):
                    if 8 * i // XR != b:
                        continue
                    # route dist: this cluster's tile, rows 8 i.. of X, in block b
                    rg = route_tile(g, i, xb.view((8 * i - b * XR) * self.PX), [xc])
                    xblk_reader[g][b % 2].append(rg)
                if g == 0 and not self.route_dist:          # the router: logits, their factors, row-major
                    li = b % 2
                    lg = B0["lg"][li]
                    rg = self.gemm(k, c, f"rt_b{b}", xb, self.PX, B0["rw"], lg, XR // 8, self.H // 8, 4, d.k_x,
                                   [xc] + ([lg_reader[li]] if lg_reader[li] is not None else []), int4=0)
                    xblk_reader[g][b % 2].append(rg)
                    mt = self.node(k, c, simd, f"rt_deq_b{b}", "__snax_bingo_kernel_simd_mul_tiles",
                                   SnaxBingoKernelSimdMulTilesArgs(lg, B0["rws"], B0["pin"].view(b * XR * 128),
                                                                   XR // 8, 4, dst_pitch=128), [rg])
                    lg_reader[li] = mt
                    rtr_mul.append(mt)
                    for r0 in range(b * XR, (b + 1) * XR, SOFTMAX_ROWS):
                        rt_sm.append(self.node(k, c, simd, f"rt_softmax_r{r0}", "__snax_bingo_kernel_simd_softmax_f16_f16",
                                               SnaxBingoKernelSimdSoftmaxF16F16Args(B0["pin"].view(r0 * 128),
                                                                                    B0["pp"].view(r0 * 128),
                                                                                    SOFTMAX_ROWS, 64),
                                               [mt] + rt_sm[-1:]))
                    blk_sm.append(rt_sm[-1])
                for j, (b0, n) in enumerate(self.chunks(sn)[:len(sh_ld[g])]):
                    ld, si, slab, soff = sh_ld[g][j]
                    gi, gb = next_g(g)
                    gm = self.gemm(k, c, f"sg_c{j}_b{b}_g{g}", xb, self.PX, slab, gb, XR // 8, self.H // 8,
                                   2 * n // BLK, d.k_x,
                                   [xc, ld] + ([gb_reader[g][gi]] if gb_reader[g][gi] is not None else []))
                    xblk_reader[g][b % 2].append(gm)
                    nsh = sn // BLK
                    sw = self.node(k, c, simd, f"ssw_c{j}_b{b}_g{g}", "__snax_bingo_kernel_simd_deq_swiglu",
                                   SnaxBingoKernelSimdDeqSwigluArgs(
                                       gb, slab.view(soff), self.B[g]["sg"],
                                       self.B[g]["a16sh"].view(b * (XR // 8) * nsh * D_BLOCK + (b0 // BLK) * D_BLOCK),
                                       XR // 8, n // BLK, nsh * D_BLOCK), [gm] + ([last_sw[g]] if last_sw[g] else []))
                    gb_reader[g][gi] = sw
                    last_sw[g] = sw
                    if b == self.XB - 1:
                        self.ls[g].release(si, sw)

        if self.prefetch_after_x:
            if not puts:
                raise ValueError("params prefetch_after_x: needs the X multicast (bcast_l1)")
            for k in self.chips:
                for t in x_last.values():
                    self.edge(t, self.prefetch[k])

        # ---- the route: the top 6, the table to every chip, each cluster's row lists ---------------
        # after every core's X-stage tasks in each core's order, as topk_plan assumes
        topks = []
        topk_b = {b: [] for b in range(self.XB)}            # each row block's top 6 tasks

        def root_block(b):
            """dispatch_blocks: row block b's table put and the root cluster 0's lists for it, created right
            after the block's top 6 so the root's DM and xDMA cores do them before the next block's."""
            o = b * XR * ROUTE_B
            rputs.append(self.node(R, 0, dm, f"rt_bput_b{b}", "__snax_bingo_kernel_idma_put_flagged",
                                   SnaxBingoKernelIdmaPutFlaggedArgs(
                                       B0["rt"].view(o), BingoMemSymbol(f"{Tg}_rland", offset=o), XR * ROUTE_B,
                                       BingoMemSymbol(f"{Tg}_rflag", offset=64 * b), value=1, bcast=True,
                                       also_here=True), topk_b[b] + rputs[-1:]))
            caps = [md.pad8(d.cnt[e]) for e in self.held[R]]
            cnts = [int(d.cnt[e]) for e in self.held[R]]
            root_disp.append(self.node(R, 0, xdma, f"rt_dispatch_b{b}_g0", "__snax_bingo_kernel_moe_dispatch",
                                       SnaxBingoKernelMoeDispatchArgs(B0["rt"], T, md.TOP_K, self.held[R], B0["lists"],
                                                                      caps, cnts, pad_row=T, t0=b * XR,
                                                                      t1=(b + 1) * XR, state_addr=B0["dst"]),
                                       topk_b[b] + root_disp[-1:]))

        if self.route_dist:
            # each tile's entries and flag into the root's landing area; per row block, the root waits for
            # the block's flags and sends the block's table on to every chip (and its own flag)
            for i in sorted(rt_tile):
                if i not in rt_ups:
                    route_up(i)
            if not self.route_early:
                root_collect()
        for b in (range(self.XB) if self.topk_shared and not self.route_dist else ()):
            for r in TOPK_CORES:
                topks.append(self.node(R, 0, self.roles[r], f"rt_topk_b{b}_{r}", "__snax_bingo_kernel_moe_topk_rows",
                                       SnaxBingoKernelMoeTopkRowsArgs(B0["pp"], 128, 64, md.TOP_K, b * XR, (b + 1) * XR,
                                                                      B0["rt"], ctr_addr=B0["tkc"].view(4 * b),
                                                                      chunk=TOPK_TAKE),
                                       [blk_sm[b]]))
                topk_b[b].append(topks[-1])
            if self.dispatch_blocks:
                root_block(b)
        for t0, r in (() if self.topk_shared or self.route_dist else self.topk_plan()[0]):
            b = t0 // XR
            topks.append(self.node(R, 0, self.roles[r], f"rt_topk_t{t0}", "__snax_bingo_kernel_moe_topk_rows",
                                   SnaxBingoKernelMoeTopkRowsArgs(B0["pp"], 128, 64, md.TOP_K, t0,
                                                                  min(t0 + TOPK_CHUNK, (b + 1) * XR), B0["rt"]),
                                   [blk_sm[b]]))
            topk_b[b].append(topks[-1])
        disp = {}
        if self.dispatch_blocks:
            if not self.topk_shared and not self.route_dist:
                for b in range(self.XB):
                    root_block(b)
            self.route_put = rputs[-1]
            if not self.route_dist:
                disp[0] = root_disp[-1]
            for g in range(0 if self.route_dist else 1, G):
                if g not in disp_early:
                    fetch_dispatch(g)
                disp[g] = disp_early[g]
        rput = None if self.dispatch_blocks else self.node(
            R, 0, dm, "rt_bput", "__snax_bingo_kernel_idma_put_flagged",
            SnaxBingoKernelIdmaPutFlaggedArgs(B0["rt"], BingoMemSymbol(f"{Tg}_rland"), T * ROUTE_B,
                                              BingoMemSymbol(f"{Tg}_rflag"), value=1, bcast=True,
                                              also_here=True), topks)
        if rput is not None:
            self.route_put = rput
        for g in (() if self.dispatch_blocks else range(G)):
            k, c = self.chip(g), self.c(g)
            bg = self.B[g]
            if g == 0:
                have = topks
            else:
                have = self.node(k, c, dm, f"rt_fetch_g{g}", "__snax_bingo_kernel_idma_fetch_flagged",
                                 SnaxBingoKernelIdmaFetchFlaggedArgs(BingoMemSymbol(f"{Tg}_rland"), bg["rt"], T * ROUTE_B,
                                                                     BingoMemSymbol(f"{Tg}_rflag"), 1, 1),
                                 [rput] if k == R else [go[k]])
                if k != R:
                    self.dfg.bingo_add_order_edge(rput, have)
            caps = [md.pad8(d.cnt[e]) for e in self.held[k]]
            cnts = [int(d.cnt[e]) for e in self.held[k]]
            disp[g] = self.node(k, c, xdma, f"rt_dispatch_g{g}", "__snax_bingo_kernel_moe_dispatch",
                                SnaxBingoKernelMoeDispatchArgs(bg["rt"], T, md.TOP_K, self.held[k], bg["lists"],
                                                               caps, cnts, pad_row=T), have if g == 0 else [have])
        if self.stop == "route":
            self.end_window(list(disp.values()))
            self._checks()
            return

        # ---- the shared MLP's quantise; the experts' gate|up and quantise; the A8 exchange -------
        qa = {}                                             # (g, slot or "sh") -> quantise
        ex_copy = {}                                        # (g, slot or "sh") -> the copy into the other cluster
        a16_reader = {g: None for g in range(G)}
        root_gat = {}
        # params x_in_xe: the row blocks' last readers (one GEMM core, in order) hold the buffers first
        xe_reader = {g: [(xblk_reader[g][i] or [None])[-1] if self.x_in_xe else None for i in range(2)]
                     for g in range(G)}
        pend = {g: [] for g in range(G)}                    # exchange copies created one slot later

        def quant_sh(g):
            k, c = self.chip(g), self.c(g)
            (sc0, sn), _ = self.ch(k, c)
            if not sn:
                return None
            bg = self.B[g]
            coff = 0 if c == 0 else BLK * self.shb[k][0]
            q = self.node(k, c, simd, f"q_sh_g{g}", "__snax_bingo_kernel_simd_quant_rows",
                          SnaxBingoKernelSimdQuantRowsArgs(bg["a16sh"], T, sn // BLK, (sn // BLK) * D_BLOCK,
                                                           d.sh["inv_a"], bg["a8sh"].view(coff), self.PS),
                          [last_sw[g]])
            qa[(g, "sh")] = q
            return q, coff, sn

        def exchange(g, key, src, coff, ncols, rows, pitch, q):
            k, c = self.chip(g), self.c(g)
            other = self.gs(k)[1 - c]
            dst = (self.B[other]["a8sh"] if key == "sh" else self.B[other]["a8e"][key]).view(coff)
            ex_copy[(g, key)] = self.node(k, c, dm, f"a8_copy_{key}_g{g}", "__snax_bingo_kernel_idma_2d_copy",
                                          SnaxBingoKernelIdma2dCopyArgs(src.view(coff), dst, ncols, pitch, pitch, rows),
                                          [q])

        def finish_sh(g):
            r = quant_sh(g)
            if r is not None:
                q, coff, sn = r
                pend[g].append(("sh", self.B[g]["a8sh"], coff, sn, T, self.PS, q))

        def shared_pass(g, j):
            """params shared_after: shared gate|up chunk j of cluster g over every X row block, the blocks
            copied again from the chip's X landing (the root's from its source) into the expert row buffers,
            which the slots before have finished with and the slots after gather into behind it"""
            k, c = self.chip(g), self.c(g)
            bg = self.B[g]
            sn = self.ch(k, c)[0][1]
            b0, n = self.chunks(sn)[j]
            nsh = sn // BLK
            ld, si, slab, soff = self._load(g, j)
            sw = None
            for b in range(self.XB):
                xb = bg["xe"][b % 2]
                src = self.xsrc(b * XR * self.H) if k == R else self.xland(b * XR * self.H)
                xc = self.node(k, c, dm, f"x_copy_p{j}_b{b}_g{g}", "__snax_bingo_kernel_idma_2d_copy",
                               SnaxBingoKernelIdma2dCopyArgs(src, xb, self.H, self.H, self.PX, XR),
                               [xe_reader[g][b % 2]] if xe_reader[g][b % 2] is not None else [])
                if k == R:
                    root_gat[c] = xc                        # l3_overlay: the root cluster's last read of X
                gi, gb = next_g(g)
                gm = self.gemm(k, c, f"sg_c{j}_b{b}_g{g}", xb, self.PX, slab, gb, XR // 8, self.H // 8,
                               2 * n // BLK, d.k_x,
                               [xc, ld] + ([gb_reader[g][gi]] if gb_reader[g][gi] is not None else []))
                xe_reader[g][b % 2] = gm
                sw = self.node(k, c, simd, f"ssw_c{j}_b{b}_g{g}", "__snax_bingo_kernel_simd_deq_swiglu",
                               SnaxBingoKernelSimdDeqSwigluArgs(
                                   gb, slab.view(soff), bg["sg"],
                                   bg["a16sh"].view(b * (XR // 8) * nsh * D_BLOCK + (b0 // BLK) * D_BLOCK),
                                   XR // 8, n // BLK, nsh * D_BLOCK), [gm] + ([last_sw[g]] if last_sw[g] else []))
                gb_reader[g][gi] = sw
                last_sw[g] = sw
            self.ls[g].release(si, sw)

        def sh_passes(g):
            """{slot: [shared chunks whose pass follows that slot's gate|up]} of cluster g"""
            nj = len(self.chunks(self.ch(self.chip(g), self.c(g))[0][1]))
            if self.shared_after is None or nj <= 1:
                return {}
            if len(self.shared_after) < nj - 1 or any(not 0 <= x < self.EPC for x in self.shared_after):
                raise ValueError(f"params shared_after={self.shared_after}: {nj - 1} expert slots in 0..{self.EPC - 1}")
            out = {}
            for j in range(1, nj):
                out.setdefault(self.shared_after[j - 1], []).append(j)
            return out

        passes = {g: sh_passes(g) for g in range(G)}
        for g in range(G):
            if not passes[g]:
                finish_sh(g)
        for s in range(self.EPC):
            for g in range(G):
                k, c = self.chip(g), self.c(g)
                bg = self.B[g]
                e = self.held[k][s]
                rows = md.pad8(d.cnt[e])
                _, ex = self.ch(k, c)
                e0, en = ex[e]
                lists = bg["lists"]
                caps = [md.pad8(d.cnt[x]) for x in self.held[k]]
                xe = bg["xe"][s % 2]
                gat = self.node(k, c, dm, f"gather_s{s}_g{g}", "__snax_bingo_kernel_idma_gather_rows",
                                SnaxBingoKernelIdmaGatherRowsArgs(self.xland(), self.H,
                                                                  lists.view(SnaxBingoKernelMoeDispatchArgs.rows_offset(caps, s)),
                                                                  rows, xe, self.PX),
                                [disp[g]] + ([xe_reader[g][s % 2]] if xe_reader[g][s % 2] is not None else [])
                                # the root's clusters never wait for the X flag: its X has landed after the last put
                                + (list(x_last.values()) if k == R and (self.x_put_c1 or self.x_local or self.x_puts_first) else []))
                if k == R:
                    root_gat[c] = gat                       # l3_overlay: the root cluster's last row gather
                a16 = bg["a16e"]
                sw, gm = None, None
                for j, (b0, n) in enumerate(self.chunks(en)):
                    jj = len(self.chunks(self.ch(k, c)[0][1])) + sum(len(self.chunks(ex[x][1])) for x in self.held[k][:s]) + j
                    ld, si, slab, soff = self._load(g, jj)
                    gi, gb = next_g(g)
                    gm = self.gemm(k, c, f"gu_s{s}c{j}_g{g}", xe, self.PX, slab, gb, rows // 8, self.H // 8,
                                   2 * n // BLK, d.k_x,
                                   [gat, ld] + ([gb_reader[g][gi]] if gb_reader[g][gi] is not None else []))
                    ne = en // BLK
                    sw = self.node(k, c, simd, f"esw_s{s}c{j}_g{g}", "__snax_bingo_kernel_simd_deq_swiglu",
                                   SnaxBingoKernelSimdDeqSwigluArgs(gb, slab.view(soff), bg["sg"],
                                                                    a16.view((b0 // BLK) * D_BLOCK),
                                                                    rows // 8, n // BLK, ne * D_BLOCK),
                                   [gm] + ([a16_reader[g]] if (j == 0 and a16_reader[g] is not None) else [])
                                   + ([last_sw[g]] if last_sw[g] else []))
                    gb_reader[g][gi] = sw
                    last_sw[g] = sw
                    self.ls[g].release(si, sw)
                xe_reader[g][s % 2] = gm
                coff = 0 if c == 0 else e0
                q = self.node(k, c, simd, f"q_s{s}_g{g}", "__snax_bingo_kernel_simd_quant_rows",
                              SnaxBingoKernelSimdQuantRowsArgs(a16, rows, en // BLK, (en // BLK) * D_BLOCK,
                                                               d.expert(e)["inv_a"], bg["a8e"][s].view(coff), self.PE,
                                                               w_addr=lists.view(SnaxBingoKernelMoeDispatchArgs.w_offset(caps, s))),
                              [sw])
                a16_reader[g] = q
                qa[(g, s)] = q
                # the exchange of the previous MLP, behind this slot's loads in the DM core's program
                for item in pend[g]:
                    exchange(g, *item)
                pend[g] = [(s, bg["a8e"][s], coff, en, rows, self.PE, q)]
                for j in passes[g].get(s, []):
                    shared_pass(g, j)
                if passes[g] and s == max(passes[g]):
                    finish_sh(g)
        for g in range(G):
            for item in pend[g]:
                exchange(g, *item)
        if self.stop == "a8":
            self.end_window([t for t in ex_copy.values()])
            self._checks()
            return

        # ---- the down projection by output slice; one ChainGather a slice -------------------------
        sym_reader = {g: [None, None] for g in range(G)}
        dt_reader = {g: [None] * self.ndt for g in range(G)}
        ndt_used = {g: 0 for g in range(G)}
        p1_reader = {c: [None, None] for c in range(self.NC)}
        if self.down_half:
            for g in range(G):
                self.ls[g] = HalfSlabs(self.ls[g])
        nt = self.SW // BLK
        jbase = {}
        for g in range(G):
            k, c = self.chip(g), self.c(g)
            (_, sn), ex = self.ch(k, c)
            jbase[g] = len(self.chunks(sn)) + sum(len(self.chunks(ex[x][1])) for x in self.held[k])
        stores, last_red, last_any = [], {c: None for c in range(self.NC)}, None
        copies_in = {c: [ex_copy[(g, key)] for (g, key) in ex_copy if self.c(g) != c] for c in range(self.NC)}
        pend_st = []                                        # params store_lag: (c, slice, its chain) not yet stored

        def store(c, jl, red):
            """the root's store of cluster c's reduced slice jl from p1c to L3, after the slice's chain"""
            s2 = jl % 2
            # l3_overlay: Y lies over X's source and the root's X landing
            over = ([root_gat[1 - c]] + list(x_last.values()) if self.l3_overlay and jl == 0 else [])
            st = self.node(R, c, dm, f"y_store_c{c}j{jl}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(self.p1c[c].view(s2 * self.SL),
                                                         self.ysym(self.y_off(c, jl)), self.SL),
                           [red] + over)
            p1_reader[c][s2] = st
            stores.append(st)

        for jl in range(self.NSLC):
            s2 = jl % 2
            last_add = {}
            for g in range(G):
                k, c = self.chip(g), self.c(g)
                bg = self.B[g]
                other = self.gs(k)[1 - c]
                y = self.sym[(k, c)].view(s2 * self.SL)
                jj = jbase[g] + jl * (1 + self.EPC)
                ld, si, slab, soff = self._load(g, jj)
                dep_sh = [qa[(g, "sh")]] if (g, "sh") in qa else []
                dep_sh += [ex_copy[(other, "sh")]] if (other, "sh") in ex_copy else []
                gm = self.gemm(k, c, f"sd_j{jl}_g{g}", bg["a8sh"], self.PS, slab, y, T // 8, BLK * self.shk[k] // 8,
                               nt, d.k_dn_sh, [ld] + dep_sh + ([sym_reader[g][s2]] if sym_reader[g][s2] else []))
                prev = self.node(k, c, simd, f"sd_deq_j{jl}_g{g}", "__snax_bingo_kernel_simd_mul_tiles",
                                 SnaxBingoKernelSimdMulTilesArgs(y, slab.view(soff), y, T // 8, nt), [gm])
                self.ls[g].release(si, prev)
                caps = [md.pad8(d.cnt[x]) for x in self.held[k]]
                for s, e in enumerate(self.held[k]):
                    rows = md.pad8(d.cnt[e])
                    ld, si, slab, soff = self._load(g, jj + 1 + s)
                    di = ndt_used[g] % self.ndt
                    ndt_used[g] += 1
                    dt = bg["dtmp"][di]
                    gm = self.gemm(k, c, f"dn_j{jl}s{s}_g{g}", bg["a8e"][s], self.PE, slab, dt, rows // 8, self.IE // 8,
                                   nt, d.k_dn_e,
                                   [ld, qa[(g, s)], ex_copy[(other, s)]] + ([dt_reader[g][di]] if dt_reader[g][di] else []))
                    ids = bg["lists"].view(SnaxBingoKernelMoeDispatchArgs.rows_offset(caps, s))
                    if self.fuse_deq_scatter:
                        prev = self.node(k, c, simd, f"scat_j{jl}s{s}_g{g}", "__snax_bingo_kernel_simd_scatter_add_rows",
                                         SnaxBingoKernelSimdScatterAddRowsArgs(y, dt, nt, ids, int(d.cnt[e]),
                                                                               s_addr=slab.view(soff)), [gm, prev])
                        self.ls[g].release(si, prev)
                    else:
                        mt = self.node(k, c, simd, f"dn_deq_j{jl}s{s}_g{g}", "__snax_bingo_kernel_simd_mul_tiles",
                                       SnaxBingoKernelSimdMulTilesArgs(dt, slab.view(soff), dt, rows // 8, nt), [gm])
                        self.ls[g].release(si, mt)
                        prev = self.node(k, c, simd, f"scat_j{jl}s{s}_g{g}", "__snax_bingo_kernel_simd_scatter_add_rows",
                                         SnaxBingoKernelSimdScatterAddRowsArgs(y, dt, nt, ids, int(d.cnt[e])), [mt, prev])
                    dt_reader[g][di] = prev
                last_add[g] = prev
            # params store_lag: the stores of slice jl - store_lag follow this slice's loads in the root's DM
            # core's program, and come before chain jl, which waits for the store of slice jl - 2
            while pend_st and pend_st[0][1] <= jl - self.store_lag:
                store(*pend_st.pop(0))
            for c in range(self.NC):
                ordr = self.chain_order(c)[:-1]
                p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
                chain = [cm.remote(k, self.sx(c, s2 * self.SL)) for k in ordr] + [cm.remote(R, f"({p1x} + {s2 * self.SL})")]
                ga = SnaxBingoKernelXdmaChainGatherArgs(
                    local_src=cm.remote(R, self.sx(c, s2 * self.SL)), chain=chain, size=self.SL,
                    junction=str(self.jct["Elementwise"]), jct_csr0=EW_F16_ADD,
                    hops=[(k, c) for k in ordr] + [(R, c)])
                ga.holds = [self.p1c[c]]
                first = list(copies_in[c]) if jl == 0 else []
                before = last_red[c] if self.chain_concurrent else last_any
                red = self.node(R, c, xdma, f"y_chain_c{c}j{jl}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                [last_add[g] for g in range(G) if self.c(g) == c] + first
                                + ([before] if before is not None else [])
                                + ([p1_reader[c][s2]] if p1_reader[c][s2] is not None else []))
                last_red[c] = last_any = red
                for g in range(G):
                    if self.c(g) == c:
                        sym_reader[g][s2] = red
                if self.store_lag:
                    pend_st.append((c, jl, red))
                else:
                    store(c, jl, red)
        for item in pend_st:
            store(*item)
        self.end_window(stores[-self.NC:])
        self._checks()

    def xsrc(self, off=0):
        return BingoMemSymbol(f"{self.TAG}_xy" if self.l3_overlay else f"{self.TAG}_xsrc", offset=off)

    def xland(self, off=0):
        return (BingoMemSymbol(f"{self.TAG}_xy", offset=self.T * self.H + off) if self.l3_overlay
                else BingoMemSymbol(f"{self.TAG}_xland", offset=off))

    def ysym(self, off=0, chip_id=None):
        return BingoMemSymbol(f"{self.TAG}_xy" if self.l3_overlay else f"{self.TAG}_y", offset=off, chip_id=chip_id)

    def y_off(self, c, jl):
        return (c * self.NSLC + jl) * self.SL

    # ---- checks -------------------------------------------------------------------------------
    def _route_bytes(self):
        d = self.d
        out = np.zeros((self.T, ROUTE_B), dtype=np.uint8)
        out[:, :8] = 0xFF
        out[:, :md.TOP_K] = d.ids.astype(np.uint8)
        w = np.zeros((self.T, 8), dtype=np.float16)
        w[:, :md.TOP_K] = d.w16
        out[:, 8:24] = w.view(np.uint8).reshape(self.T, 16)
        return out.reshape(-1)

    def _lists_bytes(self, k):
        d, T = self.d, self.T
        parts = []
        for e in self.held[k]:
            rows, w = d.rows_of(e)
            parts += [np.array(rows, dtype=np.uint16).view(np.uint8), np.asarray(w, dtype=np.float16).view(np.uint8)]
        return np.concatenate(parts)

    def _checks(self):
        R, T, Tg = self.ROOT, self.T, self.TAG
        checks = []
        n = 2048
        # params route dist: the whole table is the root's landing area in its L3, so the root cluster 0's
        # copy need not stay live to the end
        rt = (BingoMemSymbol(f"{Tg}_rland", chip_id=R) if self.route_dist
              else self.stash("rt_root", R, 0, self.B[0]["rt"], T * ROUTE_B, []))
        checks.append(("route", rt, self._route_bytes(), T * ROUTE_B))
        lb = self._lists_bytes(R)
        got = self.stash("lists_root_c1", R, 1, self.B[1]["lists"], lb.size, [])
        checks.append(("lists", got, lb, lb.size))
        if self.stop in ("a8", "full"):
            e = self.held[R][0]
            a8 = self.d.a8_expert(e)
            img = np.zeros((a8.shape[0], self.PE), dtype=np.int8)
            img[:, :self.IE] = a8
            # row 0's columns only: the rest of a row's pitch is never written, and a DMA write of
            # unwritten (X) TCDM fails mem_wide_narrow_mux's DMAWriteDataCorrect in simulation
            nb = min(n, self.IE)
            got = self.stash("a8_root_s0", R, 0, self.B[0]["a8e"][0], nb, [])
            checks.append(("a8_s0", got, img.reshape(-1)[:nb].copy(), nb))
        if self.stop == "full":
            img = np.concatenate([self.lay.to_d(self.y_raw[:, o0:o0 + w], MESH3).astype(np.float16)
                                  for c in range(self.NC) for o0, w in self.out_slices(c)]).view(np.uint8)
            for name, off in (("y_a", 0), ("y_mid", img.size // 2), ("y_b", img.size - n)):
                checks.append((name, self.ysym(off, chip_id=R), img[off:off + n], n))
        self.run_checks(checks)

    # ---- emission -----------------------------------------------------------------------------
    def chain_program_order(self):
        """Each device core runs its tasks in creation order, except (params gemm_async) that a GEMM's
        wait trails the next GEMM's start of the same phase when nothing else sits between them on
        the GEMM core and the program stays acyclic with every dependency and order edge."""
        import networkx as nx
        dev = {self.roles[r] for r in ("gemm", "simd", "xdma", "dm")}
        groups = {}
        for t in self.dfg.node_list:
            if t.assigned_core_id in dev:
                groups.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id, t.assigned_core_id), []).append(t)
        for ts in groups.values():
            ts.sort(key=lambda t: t.node_id)
        movable = set()
        if self.gemm_async:
            for ts in groups.values():
                for a, b in zip(ts, ts[1:]):
                    if a in self.gwait and self.gstart.get(b) == self.gwait[a]:
                        movable.add(a)

        def sequence(ts, moved):
            out, held = [], None
            for t in ts:
                if held is not None and t is not held:
                    out.append(t)
                    out.append(held)
                    held = None
                    continue
                if t in moved:
                    held = t
                    continue
                out.append(t)
            if held is not None:
                out.append(held)
            return out

        def program_edges(moved):
            edges = []
            for ts in groups.values():
                seq = sequence(ts, moved)
                edges += list(zip(seq, seq[1:]))
            return edges

        moved = set(movable)
        while True:
            g = nx.DiGraph(self.dfg)
            g.add_edges_from(self.dfg.order_edges)
            g.add_edges_from(program_edges(moved))
            try:
                cyc = nx.find_cycle(g)
            except nx.NetworkXNoCycle:
                break
            back = [u for u, _ in cyc if u in moved]
            if not back:
                raise ValueError(f"the program order closes a cycle: {[(a.node_name, b.node_name) for a, b in cyc[:6]]}")
            moved.discard(back[0])
        added = 0
        for a, b in program_edges(moved):
            if not self.dfg.has_edge(a, b):
                self.dfg.bingo_add_edge(a, b)
                added += 1
        if not nx.is_directed_acyclic_graph(self.dfg):
            cyc = nx.find_cycle(self.dfg)
            raise ValueError(f"the program order closes a cycle: {[(a.node_name, b.node_name) for a, b in cyc[:6]]}")
        print(f"[{self.TAG}] program order: {added} core-order edges"
              + (f"; {len(moved)} of {len(self.gwait)} GEMM waits behind the next start "
                 f"({len(movable) - len(moved)} kept in place by a cycle)" if self.gemm_async else ""))

    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_", "x_copy_")):
        self.chain_program_order()
        self.dfg.split_chip_functions = True
        super().finish(app_name, x_prefixes)


def run():
    ap = argparse.ArgumentParser(description="moe prefill workload generator")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_h", default=None)
    ap.add_argument("--output_offload_file_name", default="offload_bingo_hw.h")
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--hwcfg", required=True)
    ap.add_argument("--platformcfg", required=True)
    ap.add_argument("--dsv2_dir", required=True,
                    help="snax_cluster's target/snitch_cluster/sw/apps/dsv2 (the numerics models)")
    args = ap.parse_args()
    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    plat = parse_platform_cfg(args.platformcfg)
    hw = load_cluster_cfg(args.hwcfg)
    S = MBuild(args, p, plat, hw)
    S.build()
    S.finish(f"DeepSeek-V2-Lite layer-1 MoE prefill, {S.T} tokens, {len(S.chips)} chips x {S.NC} clusters, "
             f"bcast={S.bcast_impl} reduce={S.reduce_impl} sync={S.sync}")
