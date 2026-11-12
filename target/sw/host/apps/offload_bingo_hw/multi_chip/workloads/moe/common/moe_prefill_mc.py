# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the moe prefill workloads (workloads/moe/<model>_prefill/<variant>/):
DeepSeek-V2-Lite's layer-1 MoE over a prompt of T tokens on the 16 + 4 chiplet platform, every
GEMM on VersaCore's (8, 8, 16) shape (array shape 3) with INT4 weights. The platform, the weight
rings and the stamps are chiplet_mc.py's; the data and the golden moe_prefill_data.py's.

THE ROUTE runs once, on the root's cluster 0, row block by row block of X: the router GEMM, its
column factors, the softmax and the top 6 of every token (moe_topk_rows, in tasks of TOPK_CHUNK
tokens, each on the cluster's core a model of the route frees first), into a ROUTE TABLE that goes
to every chip. Or (params route dist) every cluster routes its own 8-token tiles of X (T / 256 of them, as
one) and puts each tile's entries into the root's table, which the root sends on to every chip row block by row block. Each cluster turns it into the
row lists of its chip's experts (moe_dispatch). The root chip's shared-MLP blocks run on its
cluster 1, and the placement counts the route as the root cluster 0's work.

THE PLACEMENT is static: 4 experts a chip, most padded rows first onto the chip with the least
work so far (the root starting with the router's), each expert's intermediate columns split
over the chip's two clusters by 16-column blocks (cluster 0 an even number), moved until the
clusters' work is even. The shared MLP: each chip a run of its column blocks, split the same
way. Every expert's row count (its tokens, padded to whole 8-row tiles) is fixed here, from the
generator's own route; the device's route is checked against it by moe_dispatch.

X AND THE ROUTER'S WEIGHTS come from HBM (params x_hbm): they sit in the HBM of the root's memory chip, whose
push engine multicasts them into every chip's L3 inside the window (configured by the root's host,
__host_bingo_kernel_hbm_bcast): the router's weights behind one flag, then X a tile of 8 tokens at a time, each
behind a flag of its own, so a cluster starts on its tile as soon as it lands. In a whole layer (x_device) X is made
on the root and the root multicasts it from its L3.

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
TIMING: the start stamp after every chip's setup; the end stamp after the last store.
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
    HostBingoKernelHbmBcastArgs, HostBingoKernelHbmUcastArgs, PhaseArgsLoadArgs, INT4_B_LAYOUT_B, INT4_B_LAYOUT_PAIRED, INT4_B_LAYOUT_PARITY, SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs,
    SnaxBingoKernelIdmaFetchFlaggedArgs, SnaxBingoKernelIdmaGatherRowsArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelMoeDispatchArgs, SnaxBingoKernelMoeTopkRowsArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdDeqSwigluArgs, SnaxBingoKernelSimdMulTilesArgs,
    SnaxBingoKernelSimdQuantRowsArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelSimdScatterAddRowsArgs, SnaxBingoKernelSimdSoftmaxF16F16Args, SnaxBingoKernelWaitFlagsArgs,
    SnaxBingoKernelXdmaChainGatherArgs)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402

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
SIMD_BEAT = 32                # one SIMD beat: a SIMD task writes whole ones
SHAPE3 = 3                    # VersaCore array shape (8, 8, 16)
MESH3 = md.MESH3
ROUTE_B = 32                  # one token's entry of the route table
SOFTMAX_ROWS = 32             # the softmax's rows a task (its scratch fits the SIMD pool)
B_PARITY_OFF = 8              # a parity-layout weight chunk starts one word into its slab: A's rows
                              # start 64-B aligned, so B is an odd number of words after A
EW_F16_ADD = cm.EW_F16_ADD


def chain_path_param(p, chips, root, whole=True):
    """params chain_path: the chips a ChainGather passes in data order, as hex chip ids, the root last (None:
    the default fold cycle). whole: the path must pass every chip."""
    cp = p.get("chain_path")
    if cp is None:
        return None
    path = [int(str(x), 16) for x in cp]
    if path[-1] != root or len(set(path)) != len(path) or not set(path) <= set(chips):
        raise ValueError(f"params chain_path={cp}: distinct chips of the platform, the root {root:02x} last")
    if whole and set(path) != set(chips):
        raise ValueError(f"params chain_path={cp}: must pass every chip")
    return path


def row_pitch(nbytes):
    """A row-major operand's row pitch: at least a row and 16 bytes (an odd quantise writes 16
    values past it), beat-aligned, and 32 mod 64 so the 8 rows of a GEMM pass sit in 8 banks."""
    p = int(nbytes) + 16
    p += (32 - p % 64) % 64
    return p


class Scale:
    """A weight chunk's scale tile as its dequantisation reads it: the address, and with params scale_out the
    copy task that put it there and the index of its buffer."""

    def __init__(self, addr, copy=None, buf=None):
        self.addr, self.copy, self.buf = addr, copy, buf

    @property
    def deps(self):
        return [self.copy] if self.copy is not None else []


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
    """The MoE prefill block (the module docstring). run_driver makes the program in four steps: __init__ (the
    parameters), mem_handles() (the data, the placement, the golden and the memory), build() (the task graph: every task
    and edge) and finish() (the compile into the C header and the data image, in which the compiler orders every core's
    tasks)."""
    TAG = "mp"

    def __init__(self, args, p, plat, hw, shared=None):
        """The parameters; no data, memory or task yet. ChipletBuild.__init__ first: the platform's and the graph's
        switches, the chips, each one's memory chip and its push engine there (with `shared`, a layer's graph and data
        staging). Then the block's switches: the ablation arms, where X comes from, the experts a chip, the gate|up
        chunks and down slices, the row blocks of X, the route, the placement's, the reduce's and the schedule's."""
        super().__init__(args, p, plat, hw, shared=shared)
        # THE ABLATION (workloads/moe/dsv2_l1_prefill_ablation): params x_cast, reduce and sync take the paper's three
        # mechanisms off one at a time, on the same mapping, as ffn_prefill_mc.py's do for the FFN.
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
        if self.sync == "barrier" and "barrier_fence" in p and bool(p["barrier_fence"]) != self.per_kernel:
            raise ValueError("params barrier_fence: barrier_at sets it (kernel: every core waits; cross: the named tasks)")
        self.barrier_fence = self.per_kernel
        self.barrier_role = "xdma"
        # x_device (a block of a whole layer, dsv2_layer_prefill_mc.py): X's source on the root is made on the device,
        # so it is staged as zeros. resid: {chip, src, piece, h16}: the residual (h16, FP16 [T, H], in L3 at src in
        # the output's slice layout) added into that chip's partial of every slice, `piece` bytes at a time
        self.x_device = bool(p.get("x_device", False))
        # x_hbm (default unless x_device): X and the router's weights from the HBM of the root's memory chip (the
        # module docstring). The prompt preloaded in the root's L3 is not an input: X comes from HBM or the device
        self.x_hbm = bool(p.get("x_hbm", not self.x_device))
        if self.x_hbm == self.x_device:
            raise ValueError("params x_hbm, x_device: X comes from HBM (x_hbm) or is made on the device (x_device), "
                             "exactly one of them")
        if self.x_hbm and any(p.get(k) for k in ("x_puts_first", "x_put_once", "x_puts_split", "x_local", "x_put_c1")):
            raise ValueError("params x_hbm: the memory chip multicasts X; x_puts_first, x_put_once, x_puts_split, "
                             "x_local and x_put_c1 are the root's puts from its L3 (x_device)")
        # params route_after_x (x_hbm): every cluster's route waits for all of X instead of its own tile (no overlap
        # of the multicast with the route; the reference for it)
        self.route_after_x = bool(p.get("route_after_x", False))
        self.resid = getattr(self, "resid", None)
        nch = len(self.chips)
        if md.N_EXP % nch:
            raise ValueError(f"{md.N_EXP} experts over {nch} chips")
        self.EPC = md.N_EXP // nch
        # params root_shared_blocks: the root's count of the shared MLP's column blocks (mem_handles splits them)
        self.root_shared_blocks = p.get("root_shared_blocks")
        self.BPC = int(p.get("blocks", 3))                 # gate blocks a gate|up chunk
        # params a16_pairs: an expert's SwiGLU output two gate|up chunks at a time, in a buffer of two chunks' blocks,
        # quantised into its A8 behind each pair (an even number of blocks: the quantisation writes 16 values past an
        # odd one), instead of one buffer of the expert's rows by all its columns, quantised at its end: the L1 a busy
        # expert's rows take shrinks by its columns over two chunks'. One buffer: the SwiGLU and the quantisation run
        # on the one in-order SIMD core, so a second would not let them overlap
        self.a16_pairs = bool(p.get("a16_pairs", False))
        # params sh_quant_blocks: the shared MLP's SwiGLU output quantised into its A8 per row block of X and chunk, right
        # after the block's SwiGLU, in a buffer of one block's rows, instead of one buffer of every token's rows
        # quantised after the last pass (the same scale: identical A8). A chunk of an odd number of blocks writes 16
        # values past its columns, so the other cluster's copy of its columns into this A8 follows the last quantise
        self.sh_qb = bool(p.get("sh_quant_blocks", False))
        self.SW = int(p.get("slice", 128))                 # output columns a down slice
        self.XB = int(p.get("x_blocks", 4))                # row blocks of X (shared MLP, router)
        if self.T % (8 * self.XB):
            raise ValueError(f"params x_blocks={self.XB}: whole 8-row tiles of {self.T} tokens a block")
        self.XR = self.T // self.XB
        self.NXT = self.T // 8                             # x_hbm: X's multicast tiles, 8 tokens each
        self.SL = self.T * self.SW * 2                     # one output slice, FP16
        self.stop = str(p.get("stop", "full"))
        # params stop: where the window ends. route: X and the route (no shared MLP); shared: + the shared MLP's gate|up;
        # gateup: + the experts' gate|up (no quantise); a8: + the quantise and the A8 exchange; full: the block
        if self.stop not in ("route", "shared", "gateup", "a8", "full"):
            raise ValueError(f"params stop={self.stop!r}: route, shared, gateup, a8 or full")
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
        # params chain_path: the chips every ChainGather passes, in data order, as hex chip ids, the root last
        # (default the fold cycle). Pushes run along the rows from the memory chips, so the path's steps between
        # columns are the ones that share a link with them.
        self.chain_path = chain_path_param(p, self.chips, self.ROOT)
        # params chain_halves {west: [...], east: [...]}: each slice's reduce of a cluster as two ChainGathers at once
        # over disjoint chips -- west (hex chip ids in data order, ending at the root) into p1c, and east (ending at its
        # collector) into the collector's own slot of the exchange buffer (the same local address on every
        # chip, so the root's tasks can name it), its own partial first copied aside -- then one hop that adds the
        # east sum to p1c into the root's own slot, which the root stores. No gather runs in place: the chain kernel
        # marks its destination with a sentinel before it arms, which in place would overwrite the collector's own
        # first operand. The next chain through any chip waits for that hop, so no chip carries two chains' traffic
        # at once (with chain_tail, the last slice's as before: after every push has landed)
        ch = p.get("chain_halves")
        self.halves = None
        if ch is not None:
            west, east = [[int(str(x), 16) for x in ch[h]] for h in ("west", "east")]
            if (west[-1] != self.ROOT or set(west) & set(east) or set(west) | set(east) != set(self.chips)
                    or len(set(west)) != len(west) or len(set(east)) != len(east) or self.chain_path):
                raise ValueError(f"params chain_halves={ch}: disjoint paths over every chip, west ending at the root,"
                                 " without chain_path")
            self.halves = (west, east)
        # params down_half_slabs: the down projection's weight chunks load into half slabs (HalfSlabs), so
        # w_buffers slabs keep 2 * w_buffers of them in flight
        self.down_half = bool(p.get("down_half_slabs", False))
        # params dtmp_bufs: the expert down GEMMs' output buffers a cluster rotates through (each read
        # by its dequantisation and scatter-add)
        self.ndt = int(p.get("dtmp_bufs", 2))
        if self.ndt < 1:
            raise ValueError(f"params dtmp_bufs={self.ndt}: at least one")
        # params b_parity: the INT4 weights in the parity layout (gemm.h), one word into their slab,
        # so the B reader and the row-major A reader keep to banks of opposite parity
        self.b_parity = bool(p.get("b_parity", False))
        # params w4_paired: the INT4 weights in the paired layout (blocks of 32 columns, layout.py
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
        # cluster routes its own 8-token tiles of X once their row block lands: the router GEMM, its factors,
        # the softmax and the top 6 on the cluster's xDMA core; it puts each tile's table entries and a flag
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
        # params place_root_last: the placement's ties go to the root last. With route dist the root starts with no
        # router work, so it would tie with every chip and take the busiest expert, beside the route's buffers
        self.place_root_last = bool(p.get("place_root_last", False))
        # params place_hot_light: after the placement, the chip of the busiest expert trades its other experts for the
        # least busy ones of other chips (not the root's): its L1 holds that expert's rows, so its other experts' rows
        # are what it can give up
        self.place_hot_light = bool(p.get("place_hot_light", False))
        # params place_refine: after the placement, pairwise swaps that lower the busiest chip's load (_place)
        self.place_refine = bool(p.get("place_refine", False))
        # params down_phases (in a layer with args phases): the down slices in that many args phases, so a cluster's
        # args region holds a part of the down phase's, not all of it (args_phase)
        self.down_phases = int(p.get("down_phases", 1))
        # params args_phases (the block alone): args phases as a layer has them (bingo_dfg_emit.py ARGS PHASES), one
        # boundary before the down projection and one at each of down_phases' boundaries: a cluster's args of each
        # phase share one L1 region, loaded by a DM task after every task of the previous phase on the chip
        self.args_phases = bool(p.get("args_phases", False))
        self.phase_starts, self.phase_names, self.loaders = [], ["MoE route and gate|up"], {}
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
        # params prefetch_hold_rows (with prefetch_after_x): only the chips in these grid rows hold their prefetchers
        # for X. The broadcast from the root runs along row 0, then down every column, so only row 0's pushes share
        # links with it; the other rows' prefetchers start with the window
        hr = p.get("prefetch_hold_rows")
        self.prefetch_hold_rows = None if hr is None else {int(r) for r in hr}
        # params chain_first_alt: the cluster whose ChainGather of a slice runs first alternates from slice to slice
        # (with serial chains the second cluster's next slice waits for the later chain)
        self.chain_first_alt = bool(p.get("chain_first_alt", False))
        # params chain_tail (0 or 1): the last slice's two ChainGathers run at the same time, each after its own
        # cluster's previous gather. On every chip each cluster's last scatter-add also waits for the other
        # cluster's last weight load, so a last-slice chain starts only once every push has landed: no weight push
        # is in flight while two chains are
        self.chain_tail = int(p.get("chain_tail", 0))
        if self.chain_tail not in (0, 1):
            raise ValueError(f"params chain_tail={self.chain_tail}: 0 or 1")
        # params shared_after: the shared MLP's gate|up one weight chunk at a time over every X row block, the
        # first chunk as X arrives and chunk j >= 1 after expert slot shared_after[j - 1]'s gate|up (X's blocks
        # copied again from the chip's L3), so no stretch of the shared MLP holds both weight slabs while the
        # ring fills up behind it
        # params fuse_deq_scatter: an expert's down output dequantised by its scatter-add task, one SIMD task
        # instead of two
        self.fuse_deq_scatter = bool(p.get("fuse_deq_scatter", False))
        # params scale_out: each weight chunk's scale tile is copied from its slab into one of scale_bufs small
        # buffers right after its load (DM core), the dequantisation reads the copy, and the slab is free once the
        # chunk's GEMMs are done instead of after its dequantisation. Off: the dequantisation reads the slab, and
        # the next load into the slab waits for it.
        self.scale_out = bool(p.get("scale_out", False))
        self.n_sc = int(p.get("scale_bufs", 4))
        if self.scale_out and (self.n_sc < 2 or p.get("down_half_slabs", False)):
            raise ValueError("params scale_out: scale_bufs >= 2, without down_half_slabs")
        sa = p.get("shared_after")
        self.shared_after = None if sa is None else [int(x) for x in sa]
        # params shared_down_early (an expert slot): the shared MLP's down projection of every output slice runs right
        # after that slot's gate|up (where the arrays wait for the stream), through the exchange buffer's slots,
        # dequantised and parked in the chip's L3; each slice of the down phase then starts from a copy of its part
        # instead of a GEMM. Its weight chunks are taken from the ring there, in that order
        sde = p.get("shared_down_early")
        self.sd_early = None if sde is None else int(sde)
        # params out_slots (default 2): the slots of the exchange buffer the down phase's slices rotate through,
        # so slice j + out_slots waits for slice j's ChainGather (and the root's store) to have read its slot. The
        # root's p1c keeps two slots: the serial chains read them in turn
        self.NOUT = int(p.get("out_slots", 2))
        if self.NOUT < 2 or (self.NOUT != 2 and self.sd_early is not None):
            raise ValueError(f"params out_slots={self.NOUT}: at least 2, and 2 with shared_down_early")
        # params chain_group (default 1): one ChainGather carries this many consecutive output slices of a cluster, the
        # output slots they sit in back to back, into a fold slot of that many slices at the root, stored to L3 by one
        # copy (Y is slice by slice)
        self.chain_group = int(p.get("chain_group", 1))
        # params esrc_early (chain_halves): the east collector's copy of its own partial waits for its slice and its own
        # cluster's previous reduce (the last to read esrc and to use the collector's xDMA), not for the chain before it,
        # so it is done before its east chain may start. The copy stays inside the collector's L1
        self.esrc_early = bool(p.get("esrc_early", False))
        if self.esrc_early and p.get("chain_halves") is None:
            raise ValueError("params esrc_early: needs chain_halves")
        # params expert_rows (default 0: off): an expert slot of more rows runs its gate|up in row blocks of at most this
        # many (whole 8-row tiles, as even as they come), each its own pass: its rows gathered into the next row buffer
        # (xe), the expert's gate|up chunks streamed again, its rows of A8 quantised; the down projection still takes
        # the slot's rows at once. The row buffers then hold a block, not the busiest expert's rows
        self.expert_rows = int(p.get("expert_rows", 0))
        if self.expert_rows < 0 or self.expert_rows % 8:
            raise ValueError(f"params expert_rows={self.expert_rows}: whole 8-row tiles")
        # params x_in_xe: X's row blocks land in the expert row buffers (xe) instead of their own two buffers;
        # the first experts' row gathers follow the last row blocks' GEMMs
        self.x_in_xe = bool(p.get("x_in_xe", False))
        # params route_tile_buf (route dist): each cluster copies its 8-token tiles of X from the chip's L3 into
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
        if self.route_tile_buf and not (self.route_dist and (p.get("x_puts_first", False) or self.x_hbm)):
            raise ValueError("params route_tile_buf: needs route dist and x_puts_first or x_hbm")
        if self.route_after_x and not (self.x_hbm and self.route_tile_buf):
            raise ValueError("params route_after_x: needs x_hbm and route_tile_buf")
        if self.stop == "route" and not (self.x_hbm and self.route_tile_buf):
            raise ValueError("params stop route: needs x_hbm and route_tile_buf (the route alone, no shared MLP)")
        if self.x_local + self.x_put_c1 + self.x_puts_first > 1:
            raise ValueError("params x_local, x_put_c1, x_puts_first: one of them")
        # params weight_ring.trailer: every chunk ends in a 64-B record that says it has landed
        # (chunk_trailer), so the prefetcher pushes no flag after it -- one launch a run, not two
        if self.x_cast == "unicast" and not self.x_hbm:
            raise ValueError("params x_cast unicast: X from HBM (x_hbm)")
        if self.sync == "barrier" and (self.halves or self.chain_group != 1 or self.NOUT != 2 or self.sd_early is not None
                                       or self.resid is not None or self.stop != "full" or not self.prefetch_after_x
                                       or not (self.x_hbm and self.route_dist and self.route_tile_buf
                                               and self.dispatch_blocks and self.route_early)):
            raise ValueError("params sync barrier: X from HBM, route dist with route_tile_buf, dispatch_blocks and "
                             "route_early, prefetch_after_x, two output slots, stop full; no chain_halves, chain_group, "
                             "shared_down_early or residual")
        self.plain_trailer = bool(p.get("weight_ring", {}).get("trailer", False))

    def mem_handles(self):
        """The data, the placement, the golden and the memory; the only tasks made are the prefetchers. The MoE's data
        (_data: the experts, the shared MLP, the router, X and the route) and the sizes it gives: the shared MLP's
        column blocks a chip (shk, from params root_shared_blocks) and each chip's first (sho), the slices a cluster,
        the row pitches; the placement of the experts on the chips (_place, params slot_order) and their split over
        each chip's clusters (_split_all); the golden (_golden); then _stage(): X, the router's weights and every
        cluster's weight chunks and factors into HBM, every L3 landing area, table and flag; make_streams(): each
        cluster's L1 weight slabs; make_rings(): each chip's weight rings in its L3 and its host's prefetcher task
        (weight_prefetch_k<chip>); _buffers(): every cluster's L1 buffers."""
        d = self.d = self._data(self.args, self.p, self.hw)
        self.lay = d.layout
        self.H, self.IE, self.ISH = d.H, d.I_e, d.I_sh
        self.EB, self.SHB = self.IE // BLK, self.ISH // BLK
        nch = len(self.chips)
        # the shared MLP's column blocks per chip (self.shk) and each chip's first block (self.sho):
        # an even split, or (params root_shared_blocks) the root's count set and the rest spread over
        # the other chips, a block more on the first ones -- the root's cluster 0 runs the route
        rsb = self.root_shared_blocks
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
        if (self.H // self.NC) % self.SW or self.SW % BLK:
            raise ValueError(f"params slice={self.SW}: whole column blocks dividing a cluster's {self.H // self.NC} outputs")
        self.NSLC = self.H // self.NC // self.SW           # slices a cluster
        self.XT = 8 * self.H                               # x_hbm: a multicast tile of X, 8 tokens
        self.PX = row_pitch(self.H)
        self.PE = row_pitch(self.IE)
        self.PS = row_pitch(BLK * max(self.shk.values()))
        if self.down_phases < 1 or self.NSLC % self.down_phases:
            raise ValueError(f"params down_phases={self.down_phases}: dividing the {self.NSLC} slices a cluster")
        if self.chain_group > 1 and (self.halves or self.sd_early is not None or self.NSLC % self.chain_group
                                     or self.NOUT % self.chain_group):
            raise ValueError(f"params chain_group={self.chain_group}: not with chain_halves or shared_down_early, and "
                             f"must divide the {self.NSLC} slices a cluster and out_slots={self.NOUT}")
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
        for cl in range(self.N_CL):
            n_sh = len(self.chunks(self.ch(self.chip(cl), self.local(cl))[0][1]))
            if n_sh > len(self.ls[cl].slabs):
                raise ValueError(f"cluster {cl} holds {n_sh} shared gate|up chunks in "
                                 f"{len(self.ls[cl].slabs)} slabs: params w_buffers >= {n_sh}")
        self.make_rings()
        self._buffers()

    def _data(self, args, p, hw):
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        return md.MoePrefillData(cache, hw=hw, seed=int(p.get("seed", 1)),
                                 n_calib=int(p.get("n_calib", 8)), n_cal_act=int(p.get("n_cal_act", 32)),
                                 tokens=self.T, inter_e=p.get("inter_e"), inter_sh=p.get("inter_sh"))

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
            k = min((k for k in self.chips if len(held[k]) < self.EPC),
                    key=lambda k: (load[k], self.place_root_last and k == self.ROOT, self.chips.index(k)))
            held[k].append(e)
            load[k] += md.pad8(cnt[e]) * self.EB
        if self.place_hot_light:
            rows = lambda e: md.pad8(cnt[e])
            hot = max(range(md.N_EXP), key=lambda e: (rows(e), -e))
            kh = next(k for k in self.chips if hot in held[k])
            for e in sorted((x for x in held[kh] if x != hot), key=lambda x: (-rows(x), x)):
                cands = [(rows(x), load[k2], self.chips.index(k2), x, k2) for k2 in self.chips
                         if k2 not in (kh, self.ROOT) for x in held[k2] if rows(x) < rows(e)]
                if not cands:
                    break
                _, _, _, x, k2 = min(cands)
                held[kh][held[kh].index(e)], held[k2][held[k2].index(x)] = x, e
                load[kh] += (rows(x) - rows(e)) * self.EB
                load[k2] += (rows(e) - rows(x)) * self.EB
        if self.place_refine:
            # the busiest chip swaps one of its experts for a lighter one of another chip while that lowers the larger
            # of the two loads; the root takes no heavier expert (its L1 holds the route's buffers too)
            rows = lambda e: md.pad8(cnt[e]) * self.EB
            for _ in range(md.N_EXP * len(self.chips)):
                kh = max(self.chips, key=lambda k: (load[k], -self.chips.index(k)))
                best = None
                for e in held[kh]:
                    for k2 in self.chips:
                        if k2 in (kh, self.ROOT):
                            continue
                        for x in held[k2]:
                            dl = rows(e) - rows(x)
                            if dl > 0 and load[k2] + dl < load[kh]:
                                cand = (max(load[kh] - dl, load[k2] + dl), self.chips.index(k2), e, x, k2, dl)
                                best = cand if best is None or cand < best else best
                if best is None:
                    break
                _, _, e, x, k2, dl = best
                held[kh][held[kh].index(e)], held[k2][held[k2].index(x)] = x, e
                load[kh] -= dl
                load[k2] += dl
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

    def sh_stage_pitch(self, k, c):
        """sh_quant_blocks: the row pitch of cluster c's staging rows on chip k, or 0 when every one of its shared
        chunks starts on a SIMD beat of its A8 rows. The quantise writes whole 32-byte beats, so a chunk whose first
        column sits mid-beat is quantised into staging rows and copied into place by the DM core."""
        sn = self.ch(k, c)[0][1]
        coff = 0 if c == 0 else BLK * self.shb[k][0]
        odd = [n for b0, n in self.chunks(sn) if (coff + b0) % SIMD_BEAT]
        return -(-(max(odd) + BLK) // SIMD_BEAT) * SIMD_BEAT if odd else 0

    @staticmethod
    def beat_aligned(off, what):
        """A SIMD task's output offset inside its buffer: the kernels write whole beats."""
        if off % SIMD_BEAT:
            raise ValueError(f"{what}: output at byte {off} of its buffer, not on a {SIMD_BEAT}-byte SIMD beat")
        return off

    def chunks(self, n_cols):
        """[(first column, columns)] of a run of n_cols intermediate columns in gate|up chunks."""
        nb = n_cols // BLK
        return [(BLK * b, BLK * min(self.BPC, nb - b)) for b in range(0, nb, self.BPC)]

    def row_blocks(self, rows):
        """An expert slot's padded rows as its gate|up passes take them (params expert_rows): [(first row, rows)]."""
        mb = self.expert_rows
        if not mb or rows <= mb:
            return [(0, rows)]
        nb, tiles = -(-rows // mb), rows // 8
        out, r0 = [], 0
        for i in range(nb):
            n = 8 * (tiles // nb + (1 if i < tiles % nb else 0))
            out.append((r0, n))
            r0 += n
        return out

    def out_slices(self, c):
        """Cluster c's output slices: [(first output column, columns)]."""
        h = self.H // self.NC
        return [(c * h + j * self.SW, self.SW) for j in range(self.NSLC)]

    # ---- the golden ---------------------------------------------------------------------------
    def chain_order(self, c):
        """The chips a ChainGather of cluster c passes, the root last; both clusters' the same way."""
        return list(self.chain_path) if self.chain_path else self.order[1:] + [self.ROOT]

    def fold_order_c(self, c):
        """The chips in the order cluster c's reduce adds their partials (FP16 addition does not associate, so the
        golden adds in it): a ChainGather's (chain_order); the root's pulls (reduce root): its own partial first, then
        every other chip's in the chain's order."""
        ch = self.chain_order(c)
        return ch if self.reduce_impl == "chain" else [self.ROOT] + [k for k in ch if k != self.ROOT]

    def _golden(self):
        d, add16 = self.d, self.d.fp.add16
        y = np.zeros((self.T, self.H), dtype=np.float16)
        for c in range(self.NC):
            for c0, w in self.out_slices(c):
                parts = {}
                for k in self.chips:
                    i = self.chips.index(k)
                    ks0 = BLK * self.sho[k]
                    init = (self.resid["h16"][:, c0:c0 + w] if self.resid is not None and k == self.resid["chip"]
                            else None)
                    parts[k] = d.chip_partial((ks0, ks0 + BLK * self.shk[k]), self.held[k], c0, w, init=init)
                sums = []
                for order in (self.halves if self.halves else [self.fold_order_c(c)]):
                    acc = parts[order[0]]
                    for k in order[1:]:
                        acc = add16(acc, parts[k])
                    sums.append(acc)
                y[:, c0:c0 + w] = sums[0] if len(sums) == 1 else add16(sums[0], sums[1])
        self.y_raw = y
        for line in d.report(y, resid=None if self.resid is None else self.resid["h16"]):
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
            # X's source (x_device: the root's, made on the device), then every chip's X landing (the prompt's rows
            # and the zero row the experts' padding rows read); Y lies over the first T * H * 2 bytes
            assert self.T * self.H * 2 <= (2 * self.T + 1) * self.H
            st.put_zeros(f"{T}_xy", "int8_t", (2 * self.T + 1) * self.H, preload=True)
        elif self.x_device:
            st.put_zeros(f"{T}_xsrc", "int8_t", self.T * self.H, preload=True)
        else:
            # every chip's X: the prompt's rows and the zero row the experts' padding rows read
            st.put_zeros(f"{T}_xland", "int8_t", (self.T + 1) * self.H, preload=True)
        m = self.mem[R]
        if self.x_hbm:
            self.h_xhbm = st.put_hbm(f"{T}_xhbm", hq.reshape(-1), mem_chip=(m >> 4, m & 15))
            st.put_zeros(f"{T}_xtflag", "uint8_t", 64 * self.NXT)
            if self.x_cast == "unicast":                    # the root's progress flags, one a (chip, tile)
                st.put_zeros(f"{T}_xuprog", "uint8_t", 64 * self.NXT * len(self.chips))
        st.put_zeros(f"{T}_xflag", "uint8_t", 64 * self.XB)
        st.put_zeros(f"{T}_rland", "uint8_t", self.T * ROUTE_B, preload=True)
        st.put_zeros(f"{T}_rflag", "uint8_t", 64 * (self.XB if self.dispatch_blocks else 1))
        if self.route_dist:
            st.put_zeros(f"{T}_rup", "uint8_t", 64 * (self.T // 8))   # the root's: one flag a tile
        if self.topk_shared or self.warm_root:
            st.put_zeros(f"{T}_zero1k", "uint8_t", 1024)
        if not self.l3_overlay:
            st.put_zeros(f"{T}_y", "uint8_t", self.T * self.H * 2, preload=True)
        if self.sd_early is not None:                       # every cluster's parked shared down outputs, on every chip
            st.put_zeros(f"{T}_sdpark", "uint8_t", self.NC * self.NSLC * self.SL, preload=True)
        rw = self.lay.to_b(np.asarray(d.wr_q, dtype=np.int8), MESH3).astype(np.int8)
        rws = self._scale_tile(d.wr_f).view(np.int8)
        if self.x_hbm:
            # the router's weights and factors in the HBM of the root's memory chip; the multicast lands them in
            # every chip's L3 behind one flag
            blob = np.concatenate([rw.reshape(-1), rws.reshape(-1)])
            self.rw_bytes = blob.size
            self.h_rwhbm = st.put_hbm(f"{T}_rwhbm", blob, mem_chip=(m >> 4, m & 15))
            st.put_zeros(f"{T}_rwland", "int8_t", blob.size, preload=True)
            st.put_zeros(f"{T}_rwflag", "uint8_t", 64)
            if self.x_cast == "unicast":
                st.put_zeros(f"{T}_rwuprog", "uint8_t", 64 * len(self.chips))
            self.h_rw, self.h_rws = BingoMemSymbol(f"{T}_rwland"), BingoMemSymbol(f"{T}_rwland", offset=rw.size)
        else:
            self.h_rw = st.put(f"{T}_rw", "int8_t", rw)
            self.h_rws = st.put(f"{T}_rws", "int8_t", rws)
        if self.stop == "route":
            self.img, self.chunk_off = {}, {}
            return                                          # the route alone loads no weight chunk
        # each cluster's weights in the order it loads them: its shared gate|up chunks, its experts'
        # gate|up chunks slot by slot, then per output slice the shared down chunk and each slot's
        self.img, self.chunk_off = {}, {}
        sh_mp = d.sh
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
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
            self.img[cl] = st.put_hbm(f"{T}_w_cl{cl}", np.concatenate(parts).astype(np.int8), mem_chip=loc)
            self.chunk_off[cl] = offs

    def _buffers(self):
        d, T = self.d, self.T
        # route dist: the 8-token tiles each cluster routes, spread evenly over the clusters (each cluster's a run of
        # consecutive tiles in one row block of X, routed as one, at most SOFTMAX_ROWS rows)
        ntile = T // 8
        self.rt_tiles = {cl: [i for i in range(ntile) if i * self.N_CL // ntile == cl] for cl in range(self.N_CL)}
        if self.route_root_last:
            # the same spread, rotated by one chip so the root's clusters get the last tiles
            self.rt_tiles = {cl: [] for cl in range(self.N_CL)}
            for i in range(ntile):
                self.rt_tiles[(i * self.N_CL // ntile + self.NC) % self.N_CL].append(i)
        if self.route_dist and (max(len(v) for v in self.rt_tiles.values()) * 8 > SOFTMAX_ROWS or any(
                v and (v != list(range(v[0], v[0] + len(v))) or 8 * v[0] // self.XR != 8 * v[-1] // self.XR)
                for v in self.rt_tiles.values())):
            raise ValueError(f"params route dist: {ntile} tiles over {self.N_CL} clusters, each cluster's consecutive, "
                             f"in one row block of X, {SOFTMAX_ROWS // 8} at most")
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.NOUT * self.SL)
                    for k in self.chips for c in range(self.NC)}
        self.xch_zero = False                    # the shared down GEMM writes each slot whole first
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", 2 * self.chain_group * self.SL)
                    for c in range(self.NC)}
        if self.halves:                                     # params chain_halves: the east collector's partial, set aside
            E = self.halves[1][-1]
            self.esrc = {c: self.cx_chip(E).at(c).l1(f"{self.TAG}_esrc", self.SL) for c in range(self.NC)}
        B = {}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nb)
            (_, sn), ex = self.ch(k, c)
            rows = [md.pad8(d.cnt[e]) for e in self.held[k]]
            mrow = max(rows)
            # the gate|up passes' rows (params expert_rows: an expert's row blocks), pass p into xe[p % 2]
            prow = [n for r in rows for _, n in self.row_blocks(r)]
            brow = max(prow)
            gt = max(self.XR, brow) // 8 * 2 * self.BPC * D_BLOCK
            nmax = max(n for _, n in ex.values()) // BLK
            b = B[cl] = dict(
                xblk=None if self.x_in_xe else [l1(f"xblk{i}", self.XR * self.PX) for i in range(2)],
                # slot s gathers its rows into xe[s % 2], so each holds the most rows of its slots; params
                # shared_after, x_in_xe: X's row blocks are copied into both too
                xe=[l1(f"xe{i}", (max([r for s, r in enumerate(prow) if s % 2 == i] + [0])
                                  if not (self.shared_after or self.x_in_xe) else
                                  max([r for s, r in enumerate(prow) if s % 2 == i] + [self.XR])) * self.PX)
                    for i in range(2)],
                gbuf=[l1(f"g{i}", gt) for i in range(2)],
                sg=l1("sg", gt),
                # one A16 for every expert: the SIMD core quantises an expert before its next SwiGLU (a16_pairs: one
                # of two chunks' blocks)
                a16e=None if self.a16_pairs else l1("a16e", brow // 8 * nmax * D_BLOCK + D_BLOCK),
                a16c=([l1("a16c0", brow // 8 * 2 * self.BPC * D_BLOCK + D_BLOCK)] if self.a16_pairs else None),
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
                b["a16sh"] = l1("a16sh", (self.XR if self.sh_qb else T) // 8 * (sn // BLK) * D_BLOCK + D_BLOCK)
                if self.sh_qb and self.sh_stage_pitch(k, c):
                    b["shq_st"] = l1("shq_st", T * self.sh_stage_pitch(k, c))
            if self.route_dist and self.rt_tiles[cl]:      # route dist: the cluster's tiles
                nr = 8 * len(self.rt_tiles[cl])
                if self.route_tile_buf:
                    b["rx"] = l1("rx", nr * self.PX)
                b["rw"] = l1("rw", self.H * 64)
                b["rws"] = l1("rws", 4 * D_BLOCK)
                b["lg"] = l1("lg", nr * 64 * 2)
                b["pin"] = l1("pin", nr * 128)
                b["pp"] = l1("pp", nr * 128)
            if cl == 0 and not self.route_dist:            # the root's cluster 0: the router
                b["rw"] = l1("rw", self.H * 64)
                b["rws"] = l1("rws", 4 * D_BLOCK)
                b["lg"] = [l1(f"lg{i}", self.XR * 64 * 2) for i in range(2)]
                b["pin"] = l1("pin", T * 128)
                b["pp"] = l1("pp", T * 128)
                if self.topk_shared:
                    b["tkc"] = l1("tkc", 64)               # one word a row block
            if self.warm_root and (k == self.ROOT or self.route_dist):
                b["wz"] = l1("wz", 1024)
            if self.resid is not None and k == self.resid["chip"]:
                b["hres"] = l1("hres", self.resid["piece"])
            if self.scale_out:                              # an 8-row D-layout tile of a chunk's widest columns
                # named to sort after the weight slabs: the static packer places buffers in name order, and these
                # live for the whole run, so placed before the long-lived row buffers they would split their space
                b["sc"] = [self.cx_chip(k).at(c).l1(f"zy_{self.TAG}_sc{i}", 16 * max(self.SW, 2 * BLK * self.BPC))
                           for i in range(self.n_sc)]
        self.B = B
        self._hres_reader = {}
        self.sc_next = {cl: 0 for cl in range(self.N_CL)}
        self.sc_reader = {cl: [None] * self.n_sc for cl in range(self.N_CL)}

    # ---- helpers ----------------------------------------------------------------------------
    def args_phase(self, name):
        """An args phase boundary (params down_phases): in a layer, the layer's (MoePart); the block alone, with params
        args_phases, its own: each cluster's DM core gets the task that loads the next phase's args, after every task
        of the phase on the chip (a task may write the other cluster's L1, so a chip's clusters leave a phase
        together): after each of the phase's tasks on the chip that no other one of them follows. The tasks made from
        here on are of the next phase."""
        if not self.args_phases:
            raise ValueError("params down_phases: the block alone has args phases only with params args_phases")
        lo = self.phase_starts[-1] if self.phase_starts else 0
        mine = {}
        for n in self.dfg.node_list:
            if n.node_id >= lo and self._phase_task(n):
                mine.setdefault(n.assigned_chiplet_id, set()).add(n)
        ph = len(self.phase_starts) + 1
        for k in self.chips:
            ts = mine.get(k, set())
            deps = sorted((t for t in ts if not any(v in ts for v in self.dfg.successors(t))), key=lambda t: t.node_id)
            for c in range(self.NC):
                ld = self.node(k, c, self.roles["dm"], f"args_p{ph}", "__snax_bingo_kernel_idma_1d_copy",
                               PhaseArgsLoadArgs(k, c, ph), deps)
                ld.args_resident = True
                self.loaders[(ph, k, c)] = ld
        self.phase_starts.append(max(n.node_id for n in self.dfg.node_list) + 1)
        self.phase_names.append(name)

    @staticmethod
    def _phase_task(n):
        """A device task whose args sit in its phase's region (not the compiler's resident ones)."""
        return (n.kernel_name or "").startswith("__snax") and not getattr(n, "args_resident", False)

    def _assign_phases(self):
        """params args_phases: every device task its args phase; each task of a phase that no task of the phase on its
        cluster precedes waits for its cluster's load."""
        phase_of = {}
        for n in self.dfg.node_list:
            if self._phase_task(n):
                phase_of[n] = n.args_phase = sum(1 for s in self.phase_starts if n.node_id >= s)
        cl = lambda t: (t.assigned_chiplet_id, t.assigned_cluster_id)
        for n, ph in phase_of.items():
            if not ph:
                continue
            ld = self.loaders[(ph, *cl(n))]
            if any(phase_of.get(u) == ph and cl(u) == cl(n) for u in self.dfg.predecessors(n)):
                continue
            if not self.dfg.has_edge(ld, n):
                self.edge(ld, n)
        counts = {}
        for n in self.dfg.node_list:
            if self._phase_task(n) and n.assigned_chiplet_id == self.ROOT and n.assigned_cluster_id == 0:
                counts[n.args_phase] = counts.get(n.args_phase, 0) + 1
        print(f"[{self.TAG}] args phases: " + "; ".join(f"{q} {nm}" for q, nm in enumerate(self.phase_names))
              + f"; root cluster 0 tasks a phase {dict(sorted(counts.items(), key=str))}")

    def _resid_add(self, cl, jl, y, after):
        """params resid: the residual's slice jl of cluster cl's outputs added into the chip's partial y (after the
        shared down projection's dequantisation), a piece at a time through the cluster's hres buffer. Returns
        the last add."""
        k, c = self.chip(cl), self.local(cl)
        dm, simd = self.roles["dm"], self.roles["simd"]
        pc, hb = self.resid["piece"], self.B[cl]["hres"]
        prev = after
        for o in range(0, self.SL, pc):
            war = [self._hres_reader[cl]] if self._hres_reader.get(cl) is not None else []
            src = self.resid["src"]
            ld = self.node(k, c, dm, f"res_ld_j{jl}o{o}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(src.symbol_name,
                                                                        offset=src.offset + self.y_off(c, jl) + o,
                                                                        chip_id=getattr(src, "chip_id", None)),
                                                         hb, pc), war)
            prev = self.node(k, c, simd, f"res_add_j{jl}o{o}", "__snax_bingo_kernel_simd_stream_elementwise",
                             SnaxBingoKernelSimdAddF16Args(y.view(o), hb, y.view(o), pc // 64, 32), [ld, prev])
            self._hres_reader[cl] = prev
        return prev

    def _load(self, cl, j, deps=(), tag=""):
        """Weight chunk j of cluster cl from its ring into the next slab (tag: in the names of a chunk loaded again).
        Returns (load, slab index, slab, its scale tile as the dequantisation reads it)."""
        k, c = self.chip(cl), self.local(cl)
        ls = self.ls[cl]
        off, nb, soff = self.chunk_off[cl][j]
        si, slab, war = ls.take(nb)
        ld, _ = ls.rings.load_chunk(
            lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
            f"Ld_w{j}{tag}", src=self.img[cl].address + off, nbytes=nb, slab=slab, war=war,
            deps=deps, dm=self.roles["dm"], trailer=True)
        if not self.scale_out:
            return ld, si, slab, Scale(slab.view(soff))
        i = self.sc_next[cl]
        self.sc_next[cl] = (i + 1) % self.n_sc
        buf, rd = self.B[cl]["sc"][i], self.sc_reader[cl][i]
        cp = self.node(k, c, self.roles["dm"], f"sc_w{j}{tag}", "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(slab.view(soff), buf, nb - soff - (64 if self.plain_trailer else 0)),
                       [ld] + ([rd] if rd is not None else []))
        return ld, si, slab, Scale(buf, cp, i)

    def _scale_read(self, cl, sc, reader):
        """`reader` is the last task that reads scale tile `sc` (params scale_out: its buffer's next copy waits for it)."""
        if sc.buf is not None:
            self.sc_reader[cl][sc.buf] = reader

    def _release(self, cl, si, gemm, deq, sc):
        """Slab si is free after its readers: the dequantisation `deq`, which reads the scale tile from the slab and
        follows its GEMM, or with params scale_out the chunk's GEMM `gemm` and the copy of its scale tile. `gemm` a
        list: the chunk's GEMMs of every row block, each a reader."""
        gms = list(gemm) if isinstance(gemm, (list, tuple)) else [gemm]
        if sc.copy is None:
            self.ls[cl].release(si, [deq] + gms if isinstance(gemm, (list, tuple)) else deq)
        else:
            self.ls[cl].release(si, gms)
            self.ls[cl]._extra[si].append(sc.copy)

    def gemm(self, k, c, name, a, a_pitch, b, dst, M, K, N, k_shift, deps, int4=1):
        """One GEMM (gemm_full); returns the task its output is ready after."""
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
        return self.node(k, c, self.roles["gemm"], name, "__snax_bingo_kernel_gemm_full",
                         SnaxBingoKernelGemmFullArgs(**kw), list(deps))

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
            # router GEMM and factors wait for them on the chip (only the root's setup holds the start stamp,
            # whose dependency cell would otherwise collect every chip's). x_hbm: inside the window (below)
            for cl in (() if self.x_hbm else range(self.N_CL)):
                if self.rt_tiles[cl]:
                    k, c, bg = self.chip(cl), self.local(cl), self.B[cl]
                    self.rw_ld[cl] = (self.node(k, c, dm, "rw_load", "__snax_bingo_kernel_idma_1d_copy",
                                               SnaxBingoKernelIdma1dCopyArgs(self.h_rw, bg["rw"], self.H * 64)),
                                     self.node(k, c, dm, "rws_load", "__snax_bingo_kernel_idma_1d_copy",
                                               SnaxBingoKernelIdma1dCopyArgs(self.h_rws, bg["rws"], 4 * D_BLOCK)))
                    if k == R:
                        pre[k] += list(self.rw_ld[cl])
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
                    wz = self.B[self.clusters(k)[c]]["wz"]
                    wz_init = self.node(k, c, dm, "wz_init", "__snax_bingo_kernel_idma_1d_copy",
                                        SnaxBingoKernelIdma1dCopyArgs(zero, wz, 1024))
                    # one column tile: INT8 B with the paired layout, which takes whole pairs only
                    # the root's warm-up holds the start stamp; another chip's only runs first on its cores
                    warm = pre[k] if k == R else []
                    warm.append(self.gemm(k, c, "warm_gemm", wz, 32, wz.view(256), wz.view(512), 1, 2, 1, 0,
                                          [wz_init], int4=0 if self.w4_paired else 1))
                    if c == 0 or self.route_dist:
                        warm.append(self.node(k, c, simd, "warm_scale", "__snax_bingo_kernel_simd_mul_tiles",
                                              SnaxBingoKernelSimdMulTilesArgs(wz.view(256), wz.view(256),
                                                                              wz.view(768), 1, 1), [wz_init]))
                        warm.append(self.node(k, c, simd, "warm_softmax", "__snax_bingo_kernel_simd_softmax_f16_f16",
                                              SnaxBingoKernelSimdSoftmaxF16F16Args(wz, wz.view(768), 2, 64),
                                              [warm[-1]]))
        go = self.setup(extra=pre)
        N_CL, XR = self.N_CL, self.XR
        if self.x_hbm:
            if not self.route_dist:
                raise ValueError("params x_hbm: needs route dist")
            self.rw_bcast, self.x_bcast = self.hbm_bcasts(go)
            # every routing cluster's router weights and factors into L1 once the multicast has landed them
            for cl in range(N_CL):
                if self.rt_tiles[cl]:
                    k, c, bg = self.chip(cl), self.local(cl), self.B[cl]
                    w = self.node(k, c, dm, "rw_wait", "__snax_bingo_kernel_wait_flags",
                                  SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_rwflag"), 1, 1), [go[k]])
                    self.dfg.bingo_add_order_edge(self.rw_bcast, w)
                    self.rw_ld[cl] = (self.node(k, c, dm, "rw_load", "__snax_bingo_kernel_idma_1d_copy",
                                               SnaxBingoKernelIdma1dCopyArgs(self.h_rw, bg["rw"], self.H * 64), [w]),
                                     self.node(k, c, dm, "rws_load", "__snax_bingo_kernel_idma_1d_copy",
                                               SnaxBingoKernelIdma1dCopyArgs(self.h_rws, bg["rws"], 4 * D_BLOCK), [w]))
            if self.sync == "barrier":
                # params sync barrier: no prefetcher waits for the root's multicast across chips; each chip's waits for
                # X's last tile in its own L3 (a flag), on its cluster 0 xDMA core, which is idle until its top 6
                self.x_landed = {}
                for k in self.chips:
                    w = self.node(k, 0, xdma, "x_landed", "__snax_bingo_kernel_wait_flags",
                                  SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xtflag", offset=64 * (self.NXT - 1)),
                                                               1, 1), [go[k]])
                    self.dfg.bingo_add_order_edge(self.x_bcast, w)
                    self.x_landed[k] = w

        # ---- X into every chip, row block by row block; the router and the shared gate|up ----
        # each block's put is created with the block, so the root's DM core copies a block for the router
        # right after putting it, before the next put
        puts = []
        xblk_reader = {cl: [[], []] for cl in range(N_CL)}
        sg_gms = {}                                         # (cl, chunk) -> its shared gate|up GEMMs, every X block
        gb_reader = {cl: [None, None] for cl in range(N_CL)}
        self.gcount = {cl: 0 for cl in range(N_CL)}
        sh_ld = {}
        last_sw = {cl: None for cl in range(N_CL)}
        lg_reader, rtr_mul = [None, None], []
        rt_sm, blk_sm = [], []
        rt_tile = {}                                        # route dist: tile -> (cluster, its top 6)

        def next_g(cl):
            i = self.gcount[cl] % 2
            self.gcount[cl] += 1
            return i, self.B[cl]["gbuf"][i]

        def root_copy(cl, b):
            """The root's own copy of row block b into cluster cl's block buffer, from its source."""
            c = self.local(cl)
            return self.node(R, c, dm, f"x_copy_b{b}", "__snax_bingo_kernel_idma_2d_copy",
                             SnaxBingoKernelIdma2dCopyArgs(self.xsrc(b * XR * self.H),
                                                           self.B[cl]["xblk"][b % 2], self.H, self.H, self.PX, XR),
                             [go[R]] + xblk_reader[cl][b % 2])

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

        def route_tile(cl, ts, a, deps):
            """route dist: cluster cl's tiles ts (consecutive, 8 tokens each, named by the first) from their rows at
            `a`: the router GEMM, its factors, the softmax and the top 6 into the cluster's route table"""
            k, c = self.chip(cl), self.local(cl)
            bg = self.B[cl]
            i, n = ts[0], len(ts)
            rg = self.gemm(k, c, f"rt_t{i}", a, self.PX, bg["rw"], bg["lg"], n, self.H // 8, 4, d.k_x,
                           list(deps) + [self.rw_ld[cl][0]], int4=0)
            mt = self.node(k, c, simd, f"rt_deq_t{i}", "__snax_bingo_kernel_simd_mul_tiles",
                           SnaxBingoKernelSimdMulTilesArgs(bg["lg"], bg["rws"], bg["pin"], n, 4, dst_pitch=128),
                           [rg, self.rw_ld[cl][1]])
            sm = self.node(k, c, simd, f"rt_softmax_t{i}", "__snax_bingo_kernel_simd_softmax_f16_f16",
                           SnaxBingoKernelSimdSoftmaxF16F16Args(bg["pin"], bg["pp"], 8 * n, 64), [mt])
            tk = self.node(k, c, xdma, f"rt_topk_t{i}", "__snax_bingo_kernel_moe_topk_rows",
                           SnaxBingoKernelMoeTopkRowsArgs(bg["pp"], 128, 64, md.TOP_K, 0, 8 * n,
                                                          bg["rt"].view(8 * i * ROUTE_B)), [sm])
            for t in ts:
                rt_tile[t] = (cl, tk)
            return rg

        def route_up(i):
            """route dist: tile i's entries and flag into the root's landing area"""
            cl, tk = rt_tile[i]
            k, c, o = self.chip(cl), self.local(cl), 8 * i * ROUTE_B
            rt_ups[i] = self.node(k, c, dm, f"rt_up_t{i}", "__snax_bingo_kernel_idma_put_flagged",
                                  SnaxBingoKernelIdmaPutFlaggedArgs(
                                      self.B[cl]["rt"].view(o), BingoMemSymbol(f"{Tg}_rland", offset=o, chip_id=R),
                                      8 * ROUTE_B, BingoMemSymbol(f"{Tg}_rup", offset=64 * i, chip_id=R), value=1),
                                  [tk])
            w = rt_wait.get(8 * i // XR)
            if w is not None:                               # params route_early: the root's wait already exists
                if k == R:
                    self.edge(rt_ups[i], w)
                else:
                    self.dfg.bingo_add_order_edge(rt_ups[i], w)

        rc_xdone = []

        def root_collect():
            """route dist: per row block, the root waits for the block's tile flags and sends
            the block's table on to every chip (and its own flag)"""
            xdone = rc_xdone
            if self.x_hbm and not xdone:
                # one multicast source at a time: the table goes out once X's last tile has landed here
                xdone.append(self.node(R, 0, dm, "rt_xdone", "__snax_bingo_kernel_wait_flags",
                                       SnaxBingoKernelWaitFlagsArgs(
                                           BingoMemSymbol(f"{Tg}_xtflag", offset=64 * (self.NXT - 1)), 1, 1), [go[R]]))
                self.dfg.bingo_add_order_edge(self.x_bcast, xdone[0])
            for b in range(self.XB):
                tl = [i for i in range(T // 8) if 8 * i // XR == b]
                local = [rt_ups[i] for i in tl if i in rt_ups and self.chip(rt_tile[i][0]) == R]
                w = self.node(R, 0, dm, f"rt_wait_b{b}", "__snax_bingo_kernel_wait_flags",
                              SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_rup", offset=64 * tl[0]), len(tl), 1),
                              [go[R]] + local + rputs[-1:] + (xdone if b == 0 else []))
                for i in tl:
                    if i in rt_ups and self.chip(rt_tile[i][0]) != R:
                        self.dfg.bingo_add_order_edge(rt_ups[i], w)
                rt_wait[b] = w
                o = b * XR * ROUTE_B
                if self.x_cast == "unicast":
                    # params x_cast unicast: the block's table into each chip's L3 in turn (self.order: the root's own
                    # first, a copy onto itself that sets its flag), each with its flag there
                    prev = [w]
                    for kk in self.order:
                        there = None if kk == R else kk
                        rput_to[(b, kk)] = self.node(
                            R, 0, dm, f"rt_uput_b{b}_k{kk:02x}", "__snax_bingo_kernel_idma_put_flagged",
                            SnaxBingoKernelIdmaPutFlaggedArgs(
                                BingoMemSymbol(f"{Tg}_rland", offset=o), BingoMemSymbol(f"{Tg}_rland", offset=o, chip_id=there),
                                XR * ROUTE_B, BingoMemSymbol(f"{Tg}_rflag", offset=64 * b, chip_id=there), value=1),
                            prev)
                        prev = [rput_to[(b, kk)]]
                    rputs.append(rput_to[(b, self.order[-1])])
                    continue
                rputs.append(self.node(R, 0, dm, f"rt_bput_b{b}", "__snax_bingo_kernel_idma_put_flagged",
                                       SnaxBingoKernelIdmaPutFlaggedArgs(
                                           BingoMemSymbol(f"{Tg}_rland", offset=o), BingoMemSymbol(f"{Tg}_rland", offset=o),
                                           XR * ROUTE_B, BingoMemSymbol(f"{Tg}_rflag", offset=64 * b), value=1,
                                           bcast=True, also_here=True), [w]))

        fd_prev = {}
        rput_to = {}                                        # params x_cast unicast: (block, chip) -> its put

        def fetch_dispatch(cl):
            """dispatch_blocks: cluster cl fetches each block's table as the root sends it and
            turns it into its chip's row lists"""
            k, c = self.chip(cl), self.local(cl)
            bg = self.B[cl]
            caps = [md.pad8(d.cnt[e]) for e in self.held[k]]
            cnts = [int(d.cnt[e]) for e in self.held[k]]
            prev = fd_prev.get(cl, [])
            for b in range(self.XB):
                o = b * XR * ROUTE_B
                f = self.node(k, c, dm, f"rt_fetch_b{b}", "__snax_bingo_kernel_idma_fetch_flagged",
                              SnaxBingoKernelIdmaFetchFlaggedArgs(
                                  BingoMemSymbol(f"{Tg}_rland", offset=o), bg["rt"].view(o), XR * ROUTE_B,
                                  BingoMemSymbol(f"{Tg}_rflag", offset=64 * b), 1, 1),
                              [rput_to.get((b, k), rputs[b])] if k == R else [go[k]])
                if k != R:
                    self.dfg.bingo_add_order_edge(rput_to.get((b, k), rputs[b]), f)
                prev = [self.node(k, c, xdma, f"rt_dispatch_b{b}", "__snax_bingo_kernel_moe_dispatch",
                                  SnaxBingoKernelMoeDispatchArgs(bg["rt"], T, md.TOP_K, self.held[k], bg["lists"],
                                                                 caps, cnts, pad_row=T, t0=b * XR, t1=(b + 1) * XR,
                                                                 state_addr=bg["dst"]), [f] + prev)]
            fd_prev[cl] = prev
            disp_early[cl] = prev[0]

        sh_qlast = {}                                       # sh_quant_blocks: each cluster's last shared quantise
        sh_cp = {}                                          # sh_quant_blocks: each cluster's last staged chunk's copy

        def sh_a16(cl, b, b0):
            """Where block b's SwiGLU output of the chunk at column b0 goes (sh_quant_blocks: one block's buffer)."""
            nsh = self.ch(self.chip(cl), self.local(cl))[0][1] // BLK
            return self.B[cl]["a16sh"].view(((0 if self.sh_qb else b * (XR // 8) * nsh) + b0 // BLK) * D_BLOCK)

        def sh_quant_block(cl, j, b, b0, n, sw):
            """sh_quant_blocks: block b's columns b0 .. b0 + n of the cluster's shared MLP into its A8; the next SwiGLU
            into the buffer follows it. Returns the task that block's next writer waits for."""
            if not self.sh_qb:
                return sw
            k, c = self.chip(cl), self.local(cl)
            sn = self.ch(k, c)[0][1]
            coff = 0 if c == 0 else BLK * self.shb[k][0]
            after = [sh_cp[cl]] if cl in sh_cp else []    # the staging rows read, and every copy before this one done
            pq = self.sh_stage_pitch(k, c)
            if (coff + b0) % SIMD_BEAT and pq:
                dst, pitch = self.B[cl]["shq_st"].view(b * XR * pq), pq
            else:
                dst, pitch = self.B[cl]["a8sh"].view(self.beat_aligned(b * XR * self.PS + coff + b0, "q_sh")), self.PS
            q = self.node(k, c, simd, f"q_sh_c{j}_b{b}", "__snax_bingo_kernel_simd_quant_rows",
                          SnaxBingoKernelSimdQuantRowsArgs(sh_a16(cl, b, b0), XR, n // BLK, (sn // BLK) * D_BLOCK,
                                                           d.sh["inv_a"], dst, pitch), [sw] + after)
            sh_qlast[cl] = q
            return q

        def sh_quant_chunk_end(cl, j, b0, n):
            """sh_quant_blocks: a chunk quantised into the staging rows, every row block of it, copied into its
            columns of the A8 by the DM core; the copy then stands for every quantise of the cluster so far."""
            k, c = self.chip(cl), self.local(cl)
            coff = 0 if c == 0 else BLK * self.shb[k][0]
            pq = self.sh_stage_pitch(k, c) if self.sh_qb else 0
            if not pq or not (coff + b0) % SIMD_BEAT:
                return
            sh_cp[cl] = sh_qlast[cl] = self.node(k, c, dm, f"q_sh_cp_c{j}", "__snax_bingo_kernel_idma_2d_copy",
                                               SnaxBingoKernelIdma2dCopyArgs(self.B[cl]["shq_st"],
                                                                             self.B[cl]["a8sh"].view(coff + b0),
                                                                             n, pq, self.PS, T), [sh_qlast[cl]])

        rt_ups, rt_wait, disp_early = {}, {}, {}
        x_last = {}                                         # each root DM core's last X put
        rputs, root_disp = [], []                           # dispatch_blocks: each block's put, the root's lists
        if self.x_hbm:
            puts += [self.x_bcast] * self.XB
        elif self.x_puts_first and self.x_put_once:
            puts += [put(0, self.XB)] * self.XB
        elif self.x_puts_first:
            for b in range(self.XB):
                puts.append(put(b))
        if self.route_tile_buf:
            # every cluster's tile first: its rows straight from the chip's X (the root: its source)
            for cl in range(N_CL):
                k, c = self.chip(cl), self.local(cl)
                ts = self.rt_tiles[cl]
                for i in ts[:1]:
                    bt = 8 * i // XR
                    if self.x_hbm:
                        # the cluster's tiles' flags (route_after_x: every tile's)
                        t0, nt = (0, self.NXT) if self.route_after_x else (i, len(ts))
                        w = self.node(k, c, dm, f"rt_xwait_t{i}", "__snax_bingo_kernel_wait_flags",
                                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xtflag", offset=64 * t0), nt, 1),
                                      [go[k]])
                        self.dfg.bingo_add_order_edge(self.x_bcast, w)
                        src, deps = self.xland(8 * i * self.H), [w]
                    elif k == R:
                        src, deps = self.xsrc(8 * i * self.H), [go[R]]
                    else:
                        w = self.node(k, c, dm, f"rt_xwait_t{i}", "__snax_bingo_kernel_wait_flags",
                                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xflag", offset=xf(bt)), 1, 1),
                                      [go[k]])
                        self.dfg.bingo_add_order_edge(puts[bt], w)
                        src, deps = self.xland(8 * i * self.H), [w]
                    xc = self.node(k, c, dm, f"rt_xcopy_t{i}", "__snax_bingo_kernel_idma_2d_copy",
                                   SnaxBingoKernelIdma2dCopyArgs(src, self.B[cl]["rx"], self.H, self.H, self.PX,
                                                                 8 * len(ts)),
                                   deps)
                    route_tile(cl, ts, self.B[cl]["rx"], [xc])
        if self.per_kernel:
            # params barrier_at kernel, K1: the route whole -- every cluster's upload, the root's table out to every
            # chip, every cluster's row lists -- before any row block of X
            for cl in range(N_CL):
                for i in self.rt_tiles[cl]:
                    route_up(i)
            root_collect()
            for cl in range(N_CL):
                fetch_dispatch(cl)
            self.barrier({k: [] for k in self.chips})
        for b in range(self.XB):
            early = {0: root_copy(0, b)} if self.x_local else {}
            if not (self.x_puts_first or self.x_hbm):
                puts.append(put(b))
            for cl in range(N_CL):
                k, c = self.chip(cl), self.local(cl)
                bg = self.B[cl]
                xb = bg["xblk"][b % 2]
                if self.stop == "route":
                    # the route alone: each cluster's upload and the table, no row block, no shared MLP
                    if b == min(1, self.XB - 1):
                        for i in self.rt_tiles[cl]:
                            route_up(i)
                        if self.route_early:
                            if cl == 0:
                                root_collect()
                            fetch_dispatch(cl)
                    continue
                if (self.route_root_light and cl == 0 and not self.ch(k, c)[0][1]
                        and not any(8 * i // XR == b for i in self.rt_tiles[cl])):
                    continue                                # route_root_light: a block the root cluster 0 does not use
                if k == R and (self.x_put_c1 or self.x_local or self.x_puts_first):
                    xc = early[cl] if cl in early else root_copy(cl, b)
                else:
                    if self.x_hbm:                          # the block's tiles' flags
                        w = self.node(k, c, dm, f"x_wait_b{b}", "__snax_bingo_kernel_wait_flags",
                                      SnaxBingoKernelWaitFlagsArgs(
                                          BingoMemSymbol(f"{Tg}_xtflag", offset=64 * (b * XR // 8)), XR // 8, 1), [go[k]])
                    else:
                        w = self.node(k, c, dm, f"x_wait_b{b}", "__snax_bingo_kernel_wait_flags",
                                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xflag", offset=xf(b)), 1, 1),
                                      [go[k]] + ([puts[b]] if k == R else []))
                    if k != R or self.x_hbm:
                        self.dfg.bingo_add_order_edge(puts[b], w)
                    xc = self.node(k, c, dm, f"x_copy_b{b}", "__snax_bingo_kernel_idma_2d_copy",
                                   SnaxBingoKernelIdma2dCopyArgs(self.xland(b * XR * self.H),
                                                                 xb, self.H, self.H, self.PX, XR),
                                   [w] + xblk_reader[cl][b % 2])
                xblk_reader[cl][b % 2] = []
                (sc0, sn), _ = self.ch(k, c)
                if b == 0:                                  # the shared gate|up chunks, held for every block
                    nsj = len(self.chunks(sn)) if self.shared_after is None else min(1, len(self.chunks(sn)))
                    sh_ld[cl] = [self._load(cl, j) for j in range(nsj)]
                if self.route_tile_buf and not self.per_kernel:
                    b1 = min(1, self.XB - 1)
                    if b == b1:                             # the cluster's upload, behind its first row blocks' copies
                        for i in self.rt_tiles[cl]:
                            route_up(i)
                        if self.route_early:
                            if cl == 0:
                                root_collect()
                            fetch_dispatch(cl)
                for i in (self.rt_tiles[cl][:1] if self.route_dist and not self.route_tile_buf else ()):
                    if 8 * i // XR != b:
                        continue
                    # route dist: this cluster's tiles, rows 8 i.. of X, in block b
                    rg = route_tile(cl, self.rt_tiles[cl], xb.view((8 * i - b * XR) * self.PX), [xc])
                    xblk_reader[cl][b % 2].append(rg)
                if cl == 0 and not self.route_dist:         # the router: logits, their factors, row-major
                    li = b % 2
                    lg = B0["lg"][li]
                    rg = self.gemm(k, c, f"rt_b{b}", xb, self.PX, B0["rw"], lg, XR // 8, self.H // 8, 4, d.k_x,
                                   [xc] + ([lg_reader[li]] if lg_reader[li] is not None else []), int4=0)
                    xblk_reader[cl][b % 2].append(rg)
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
                for j, (b0, n) in enumerate(self.chunks(sn)[:len(sh_ld[cl])]):
                    ld, si, slab, sc = sh_ld[cl][j]
                    gi, gb = next_g(cl)
                    gm = self.gemm(k, c, f"sg_c{j}_b{b}", xb, self.PX, slab, gb, XR // 8, self.H // 8,
                                   2 * n // BLK, d.k_x,
                                   [xc, ld] + ([gb_reader[cl][gi]] if gb_reader[cl][gi] is not None else []))
                    xblk_reader[cl][b % 2].append(gm)
                    sg_gms.setdefault((cl, j), []).append(gm)
                    nsh = sn // BLK
                    sw = self.node(k, c, simd, f"ssw_c{j}_b{b}", "__snax_bingo_kernel_simd_deq_swiglu",
                                   SnaxBingoKernelSimdDeqSwigluArgs(
                                       gb, sc.addr, self.B[cl]["sg"], sh_a16(cl, b, b0),
                                       XR // 8, n // BLK, nsh * D_BLOCK),
                                       [gm] + sc.deps + ([last_sw[cl]] if last_sw[cl] else []))
                    gb_reader[cl][gi] = sw
                    last_sw[cl] = sh_quant_block(cl, j, b, b0, n, sw)
                    self._scale_read(cl, sc, sw)
                    if b == self.XB - 1:
                        self._release(cl, si, sg_gms[(cl, j)], sw, sc)

        if self.x_hbm:
            # the root's prefetcher drives the multicast's engine: it pushes once X has landed
            x_last = {0: self.x_bcast}
            self.edge(self.x_bcast, self.prefetch[R])
        if self.prefetch_after_x:
            if not puts:
                raise ValueError("params prefetch_after_x: needs the X multicast (bcast_l1)")
            for k in self.chips:
                if self.prefetch_hold_rows is not None and (k & 15) not in self.prefetch_hold_rows:
                    continue
                if self.sync == "barrier":
                    self.edge(self.x_landed[k], self.prefetch[k])
                    continue
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
            if not self.route_early and not self.per_kernel:
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
            for cl in range(0 if self.route_dist else 1, N_CL):
                if cl not in disp_early:
                    fetch_dispatch(cl)
                disp[cl] = disp_early[cl]
        rput = None if self.dispatch_blocks else self.node(
            R, 0, dm, "rt_bput", "__snax_bingo_kernel_idma_put_flagged",
            SnaxBingoKernelIdmaPutFlaggedArgs(B0["rt"], BingoMemSymbol(f"{Tg}_rland"), T * ROUTE_B,
                                              BingoMemSymbol(f"{Tg}_rflag"), value=1, bcast=True,
                                              also_here=True), topks)
        if rput is not None:
            self.route_put = rput
        for cl in (() if self.dispatch_blocks else range(N_CL)):
            k, c = self.chip(cl), self.local(cl)
            bg = self.B[cl]
            if cl == 0:
                have = topks
            else:
                have = self.node(k, c, dm, f"rt_fetch", "__snax_bingo_kernel_idma_fetch_flagged",
                                 SnaxBingoKernelIdmaFetchFlaggedArgs(BingoMemSymbol(f"{Tg}_rland"), bg["rt"], T * ROUTE_B,
                                                                     BingoMemSymbol(f"{Tg}_rflag"), 1, 1),
                                 [rput] if k == R else [go[k]])
                if k != R:
                    self.dfg.bingo_add_order_edge(rput, have)
            caps = [md.pad8(d.cnt[e]) for e in self.held[k]]
            cnts = [int(d.cnt[e]) for e in self.held[k]]
            disp[cl] = self.node(k, c, xdma, f"rt_dispatch", "__snax_bingo_kernel_moe_dispatch",
                                SnaxBingoKernelMoeDispatchArgs(bg["rt"], T, md.TOP_K, self.held[k], bg["lists"],
                                                               caps, cnts, pad_row=T), have if cl == 0 else [have])
        if self.stop == "route":
            self.end_window(list(disp.values()))
            self._checks()
            return

        # ---- the shared MLP's quantise; the experts' gate|up and quantise; the A8 exchange -------
        qa = {}                                             # (cl, slot or "sh") -> quantise
        ex_copy = {}                                        # (cl, slot or "sh") -> the copy into the other cluster
        a16_reader = {cl: ([None, None] if self.a16_pairs else None) for cl in range(N_CL)}
        a16_used = {cl: 0 for cl in range(N_CL)}          # a16_pairs: pairs so far, the buffer in turn
        root_gat = {}
        # params x_in_xe: the row blocks' last readers (one GEMM core, in order) hold the buffers first
        xe_reader = {cl: [list(xblk_reader[cl][i]) if self.x_in_xe else [] for i in range(2)] for cl in range(N_CL)}
        pend = {cl: [] for cl in range(N_CL)}               # exchange copies created one slot later
        held_ex = {cl: [] for cl in range(N_CL)}            # barrier_at kernel: every exchange, after the gate|up

        def quant_sh(cl):
            k, c = self.chip(cl), self.local(cl)
            (sc0, sn), _ = self.ch(k, c)
            if not sn:
                return None
            bg = self.B[cl]
            coff = 0 if c == 0 else BLK * self.shb[k][0]
            if self.sh_qb:                                  # quantised block by block already
                qa[(cl, "sh")] = sh_qlast[cl]
                return sh_qlast[cl], coff, sn
            q = self.node(k, c, simd, f"q_sh", "__snax_bingo_kernel_simd_quant_rows",
                          SnaxBingoKernelSimdQuantRowsArgs(bg["a16sh"], T, sn // BLK, (sn // BLK) * D_BLOCK,
                                                           d.sh["inv_a"], bg["a8sh"].view(self.beat_aligned(coff, "q_sh")), self.PS),
                          [last_sw[cl]])
            qa[(cl, "sh")] = q
            return q, coff, sn

        def exchange(cl, key, src, coff, ncols, rows, pitch, q):
            k, c = self.chip(cl), self.local(cl)
            other = self.clusters(k)[1 - c]
            dst = (self.B[other]["a8sh"] if key == "sh" else self.B[other]["a8e"][key]).view(coff)
            # sh_quant_blocks: after the other cluster's last quantise, whose odd chunk writes past its columns
            after = [sh_qlast[other]] if key == "sh" and self.sh_qb and other in sh_qlast else []
            ex_copy[(cl, key)] = self.node(k, c, dm, f"a8_copy_{key}", "__snax_bingo_kernel_idma_2d_copy",
                                          SnaxBingoKernelIdma2dCopyArgs(src.view(coff), dst, ncols, pitch, pitch, rows),
                                          [q] + after)

        def finish_sh(cl):
            r = quant_sh(cl)
            if r is not None:
                q, coff, sn = r
                pend[cl].append(("sh", self.B[cl]["a8sh"], coff, sn, T, self.PS, q))

        def shared_pass(cl, j):
            """params shared_after: shared gate|up chunk j of cluster cl over every X row block, the blocks
            copied again from the chip's X landing (the root's from its source) into the expert row buffers,
            which the slots before have finished with and the slots after gather into behind it"""
            k, c = self.chip(cl), self.local(cl)
            bg = self.B[cl]
            sn = self.ch(k, c)[0][1]
            b0, n = self.chunks(sn)[j]
            nsh = sn // BLK
            ld, si, slab, sc = self._load(cl, j)
            sw = gm = None
            gms = []                                        # every X block's GEMM reads the chunk's slab
            for b in range(self.XB):
                xb = bg["xe"][b % 2]
                # x_hbm: the root's X is its landing too (the multicast fills every chip's); its source holds X only
                # when the root starts with it
                src = self.xsrc(b * XR * self.H) if k == R and not self.x_hbm else self.xland(b * XR * self.H)
                xc = self.node(k, c, dm, f"x_copy_p{j}_b{b}", "__snax_bingo_kernel_idma_2d_copy",
                               SnaxBingoKernelIdma2dCopyArgs(src, xb, self.H, self.H, self.PX, XR),
                               list(xe_reader[cl][b % 2]))
                if k == R:
                    root_gat[c] = xc                        # l3_overlay: the root cluster's last read of X
                gi, gb = next_g(cl)
                gm = self.gemm(k, c, f"sg_c{j}_b{b}", xb, self.PX, slab, gb, XR // 8, self.H // 8,
                               2 * n // BLK, d.k_x,
                               [xc, ld] + ([gb_reader[cl][gi]] if gb_reader[cl][gi] is not None else []))
                xe_reader[cl][b % 2] = [gm]
                gms.append(gm)
                sw = self.node(k, c, simd, f"ssw_c{j}_b{b}", "__snax_bingo_kernel_simd_deq_swiglu",
                               SnaxBingoKernelSimdDeqSwigluArgs(
                                   gb, sc.addr, bg["sg"], sh_a16(cl, b, b0),
                                   XR // 8, n // BLK, nsh * D_BLOCK),
                                   [gm] + sc.deps + ([last_sw[cl]] if last_sw[cl] else []))
                gb_reader[cl][gi] = sw
                last_sw[cl] = sh_quant_block(cl, j, b, b0, n, sw)
            sh_quant_chunk_end(cl, j, b0, n)
            self._scale_read(cl, sc, sw)
            self._release(cl, si, gms, sw, sc)

        def sh_passes(cl):
            """{slot: [shared chunks whose pass follows that slot's gate|up]} of cluster cl"""
            nj = len(self.chunks(self.ch(self.chip(cl), self.local(cl))[0][1]))
            if self.shared_after is None or nj <= 1:
                return {}
            if len(self.shared_after) < nj - 1 or any(not 0 <= x < self.EPC for x in self.shared_after):
                raise ValueError(f"params shared_after={self.shared_after}: {nj - 1} expert slots in 0..{self.EPC - 1}")
            out = {}
            for j in range(1, nj):
                out.setdefault(self.shared_after[j - 1], []).append(j)
            return out

        sde_last = {}                                       # params shared_down_early: each cluster's last two parks

        def sd_early_pass(cl):
            """params shared_down_early: cluster cl's shared down GEMM of every output slice, through the exchange
            buffer's two slots, dequantised, parked in L3"""
            k, c = self.chip(cl), self.local(cl)
            other = self.clusters(k)[1 - c]
            if (cl, "sh") not in qa or (other, "sh") not in ex_copy:
                raise ValueError(f"params shared_down_early={self.sd_early}: the shared MLP's A8 of chip {k:02x} is not "
                                 "complete by then (its exchange follows the slot after its last gate|up chunk)")
            bg, nt = self.B[cl], self.SW // BLK
            (_, sn), ex = self.ch(k, c)
            base = len(self.chunks(sn)) + sum(len(self.chunks(ex[x][1])) for x in self.held[k])
            st = {}
            for jl in range(self.NSLC):
                y = self.xch[(k, c)].view((jl % 2) * self.SL)
                ld, si, slab, sc = self._load(cl, base + jl * (1 + self.EPC))
                gm = self.gemm(k, c, f"sde_j{jl}", bg["a8sh"], self.PS, slab, y, T // 8, BLK * self.shk[k] // 8,
                               nt, d.k_dn_sh, [ld, qa[(cl, "sh")], ex_copy[(other, "sh")]]
                               + ([st[jl - 2]] if jl >= 2 else []))
                dq = self.node(k, c, simd, f"sde_deq_j{jl}", "__snax_bingo_kernel_simd_mul_tiles",
                               SnaxBingoKernelSimdMulTilesArgs(y, sc.addr, y, T // 8, nt), [gm] + sc.deps)
                self._scale_read(cl, sc, dq)
                self._release(cl, si, gm, dq, sc)
                st[jl] = self.node(k, c, dm, f"sde_st_j{jl}", "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(y, self.sdpark(c, jl), self.SL), [dq])
            sde_last[cl] = [st[jl] for jl in range(self.NSLC)]

        if self.stop == "shared":
            # the shared MLP's gate|up alone after the route: every chunk's pass now, no expert slot between them
            for cl in range(N_CL):
                for _, js in sorted(sh_passes(cl).items()):
                    for j in js:
                        shared_pass(cl, j)
            self.end_window([last_sw[cl] for cl in range(N_CL) if last_sw[cl] is not None] + list(disp.values()))
            self._checks()
            return
        quant = self.stop != "gateup"                       # stop gateup: no quantise, no exchange
        passes = {cl: sh_passes(cl) for cl in range(N_CL)}
        for cl in range(N_CL):
            if not passes[cl] and quant:
                finish_sh(cl)
        npass = {cl: 0 for cl in range(N_CL)}               # each cluster's expert gate|up passes: pass p in xe[p % 2]
        for s in range(self.EPC):
            for cl in range(N_CL):
                k, c = self.chip(cl), self.local(cl)
                bg = self.B[cl]
                e = self.held[k][s]
                rows = md.pad8(d.cnt[e])
                _, ex = self.ch(k, c)
                e0, en = ex[e]
                lists = bg["lists"]
                caps = [md.pad8(d.cnt[x]) for x in self.held[k]]
                a16 = bg["a16e"]
                sw, gm, q = None, None, None
                chs = self.chunks(en)
                coff = 0 if c == 0 else e0
                ids0 = SnaxBingoKernelMoeDispatchArgs.rows_offset(caps, s)
                w0 = SnaxBingoKernelMoeDispatchArgs.w_offset(caps, s)
                blocks = self.row_blocks(rows)
                for r, (r0, mr) in enumerate(blocks):
                    # params expert_rows: rows r0 .. r0 + mr - 1 of the slot, its names and gate|up phase its own (r)
                    rs = "" if len(blocks) == 1 else f"r{r}"
                    xi = npass[cl] % 2
                    npass[cl] += 1
                    xe = bg["xe"][xi]
                    gat = self.node(k, c, dm, f"gather_s{s}{rs}", "__snax_bingo_kernel_idma_gather_rows",
                                    SnaxBingoKernelIdmaGatherRowsArgs(self.xland(), self.H, lists.view(ids0 + 2 * r0), mr, xe,
                                                                      self.PX),
                                    [disp[cl]] + list(xe_reader[cl][xi])
                                    # the root's clusters never wait for the X flag: its X has landed after the last put
                                    + (list(x_last.values()) if k == R and (self.x_put_c1 or self.x_local or self.x_puts_first)
                                       else []))
                    if k == R:
                        root_gat[c] = gat                   # l3_overlay: the root cluster's last row gather
                    w_rows = lists.view(w0 + 2 * r0)
                    pass_gms = []
                    for j, (b0, n) in enumerate(chs):
                        jj = len(self.chunks(self.ch(k, c)[0][1])) + sum(len(self.chunks(ex[x][1])) for x in self.held[k][:s]) + j
                        ld, si, slab, sc = self._load(cl, jj, tag=f"r{r}" if r else "")
                        gi, gb = next_g(cl)
                        gm = self.gemm(k, c, f"gu_s{s}{rs}c{j}", xe, self.PX, slab, gb, mr // 8, self.H // 8,
                                       2 * n // BLK, d.k_x,
                                       [gat, ld] + ([gb_reader[cl][gi]] if gb_reader[cl][gi] is not None else []))
                        pass_gms.append(gm)
                        ne = en // BLK
                        if self.a16_pairs:
                            # the pair's first column, its buffer
                            p0, bi = chs[j - j % 2][0], a16_used[cl] % len(bg["a16c"])
                            dst, stride = bg["a16c"][bi].view(((b0 - p0) // BLK) * D_BLOCK), 2 * self.BPC * D_BLOCK
                            war = [a16_reader[cl][bi]] if (j % 2 == 0 and a16_reader[cl][bi] is not None) else []
                        else:
                            dst, stride = a16.view((b0 // BLK) * D_BLOCK), ne * D_BLOCK
                            war = [a16_reader[cl]] if (j == 0 and a16_reader[cl] is not None) else []
                        sw = self.node(k, c, simd, f"esw_s{s}{rs}c{j}", "__snax_bingo_kernel_simd_deq_swiglu",
                                       SnaxBingoKernelSimdDeqSwigluArgs(gb, sc.addr, bg["sg"], dst, mr // 8, n // BLK, stride),
                                       [gm] + sc.deps + war + ([last_sw[cl]] if last_sw[cl] else []))
                        gb_reader[cl][gi] = sw
                        last_sw[cl] = sw
                        self._scale_read(cl, sc, sw)
                        self._release(cl, si, gm, sw, sc)
                        if quant and self.a16_pairs and (j % 2 == 1 or j == len(chs) - 1):
                            # the pair into its columns of A8, after the pair before it (the last one implies every one)
                            q = self.node(k, c, simd, f"q_s{s}{rs}p{j // 2}", "__snax_bingo_kernel_simd_quant_rows",
                                          SnaxBingoKernelSimdQuantRowsArgs(bg["a16c"][bi], mr, (b0 + n - p0) // BLK, stride,
                                                                           d.expert(e)["inv_a"],
                                                                           bg["a8e"][s].view(self.beat_aligned(r0 * self.PE + coff + p0, f"q_s{s}")),
                                                                           self.PE, w_addr=w_rows),
                                          [sw] + ([q] if q is not None else []))
                            a16_reader[cl][bi] = q
                            a16_used[cl] += 1
                    xe_reader[cl][xi] = pass_gms
                    if quant and not self.a16_pairs:
                        q = self.node(k, c, simd, f"q_s{s}{rs}", "__snax_bingo_kernel_simd_quant_rows",
                                      SnaxBingoKernelSimdQuantRowsArgs(a16, mr, en // BLK, (en // BLK) * D_BLOCK,
                                                                       d.expert(e)["inv_a"],
                                                                       bg["a8e"][s].view(self.beat_aligned(r0 * self.PE + coff, f"q_s{s}")), self.PE,
                                                                       w_addr=w_rows),
                                      [sw] + ([q] if q is not None else []))
                        a16_reader[cl] = q
                qa[(cl, s)] = q
                # the exchange of the previous MLP, behind this slot's loads in the DM core's program
                if self.per_kernel:
                    held_ex[cl] += pend[cl]
                else:
                    for item in pend[cl]:
                        exchange(cl, *item)
                pend[cl] = [(s, bg["a8e"][s], coff, en, rows, self.PE, q)] if quant else []
                for j in passes[cl].get(s, []):
                    shared_pass(cl, j)
                if passes[cl] and s == max(passes[cl]) and quant:
                    finish_sh(cl)
            if self.sd_early == s:
                for cl in range(N_CL):
                    sd_early_pass(cl)
        if self.per_kernel:
            # params barrier_at kernel: K2 ends with the gate|up; K3, the A8 exchange
            self.barrier({k: [] for k in self.chips})
            for cl in range(N_CL):
                pend[cl] = held_ex[cl] + pend[cl]
        for cl in range(N_CL):
            for item in pend[cl]:
                exchange(cl, *item)
        if self.per_kernel:
            self.barrier({k: [] for k in self.chips})
        if self.stop == "gateup":
            self.end_window([last_sw[cl] for cl in range(N_CL) if last_sw[cl] is not None])
            self._checks()
            return
        if self.stop == "a8":
            self.end_window([t for t in ex_copy.values()])
            self._checks()
            return

        # ---- the down projection by output slice; one ChainGather a slice -------------------------
        if self.sd_early is not None and len(sde_last) != N_CL:
            raise ValueError(f"params shared_down_early={self.sd_early}: not an expert slot (0..{self.EPC - 1})")
        xch_reader = {cl: (sde_last[cl][-2:] if cl in sde_last else [None] * self.NOUT) for cl in range(N_CL)}
        dt_reader = {cl: [None] * self.ndt for cl in range(N_CL)}
        ndt_used = {cl: 0 for cl in range(N_CL)}
        p1_reader = {c: [None, None] for c in range(self.NC)}
        if self.down_half:
            for cl in range(N_CL):
                self.ls[cl] = HalfSlabs(self.ls[cl])
        nt = self.SW // BLK
        jbase = {}
        for cl in range(N_CL):
            k, c = self.chip(cl), self.local(cl)
            (_, sn), ex = self.ch(k, c)
            jbase[cl] = len(self.chunks(sn)) + sum(len(self.chunks(ex[x][1])) for x in self.held[k])
        stores, last_red, last_any = [], {c: None for c in range(self.NC)}, None
        copies_in = {c: [ex_copy[(cl, key)] for (cl, key) in ex_copy if self.local(cl) != c] for c in range(self.NC)}
        last_ld = {}                                        # params chain_tail: every cluster's last weight load

        GS = self.chain_group

        def store(c, jl, red):
            """the root's store of cluster c's reduced slices jl .. jl + chain_group - 1 to L3, after their chain: from
            p1c, or with params chain_halves from the root's own output slot, where the last hop leaves the sum"""
            s2, so = (jl // GS) % 2, jl % self.NOUT
            # l3_overlay: Y lies over X's source and the root's X landing
            over = ([root_gat[1 - c]] + ([] if self.x_hbm else list(x_last.values())) if self.l3_overlay and jl == 0 else [])
            src = self.xch[(R, c)].view(so * self.SL) if self.halves else self.p1c[c].view(s2 * GS * self.SL)
            st = self.node(R, c, dm, f"y_store_c{c}j{jl}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(src, self.ysym(self.y_off(c, jl)), GS * self.SL),
                           [red] + over)
            if self.halves:
                for cl in range(N_CL):
                    if self.chip(cl) == R and self.local(cl) == c:
                        xch_reader[cl][so] = st
            else:
                p1_reader[c][s2] = st
            stores.append(st)

        prev_add = {}

        def down_cl(cl, jl, so, last_add):
            """Cluster cl's down projection of output slice jl into its output slot so: the shared MLP's down GEMM and
            its dequantise, then each expert's down GEMM, dequantise and scatter-add into the slice (last_add[cl])."""
            k, c = self.chip(cl), self.local(cl)
            bg = self.B[cl]
            other = self.clusters(k)[1 - c]
            y = self.xch[(k, c)].view(so * self.SL)
            jj = jbase[cl] + jl * (1 + self.EPC)
            if self.sd_early is not None:
                # the shared MLP's part of the slice, parked in the gate|up phase
                prev = self.node(k, c, dm, f"sd_cp_j{jl}", "__snax_bingo_kernel_idma_1d_copy",
                                 SnaxBingoKernelIdma1dCopyArgs(self.sdpark(c, jl), y, self.SL),
                                 [sde_last[cl][jl]] + ([xch_reader[cl][so]] if xch_reader[cl][so] else []))
            else:
                ld, si, slab, sc = self._load(cl, jj)
                dep_sh = [qa[(cl, "sh")]] if (cl, "sh") in qa else []
                dep_sh += [ex_copy[(other, "sh")]] if (other, "sh") in ex_copy else []
                gm = self.gemm(k, c, f"sd_j{jl}", bg["a8sh"], self.PS, slab, y, T // 8, BLK * self.shk[k] // 8,
                               nt, d.k_dn_sh, [ld] + dep_sh + ([xch_reader[cl][so]] if xch_reader[cl][so] else []))
                prev = self.node(k, c, simd, f"sd_deq_j{jl}", "__snax_bingo_kernel_simd_mul_tiles",
                                 SnaxBingoKernelSimdMulTilesArgs(y, sc.addr, y, T // 8, nt), [gm] + sc.deps)
                self._scale_read(cl, sc, prev)
                self._release(cl, si, gm, prev, sc)
            if self.resid is not None and k == self.resid["chip"]:
                prev = self._resid_add(cl, jl, y, prev)
            caps = [md.pad8(d.cnt[x]) for x in self.held[k]]
            for s, e in enumerate(self.held[k]):
                rows = md.pad8(d.cnt[e])
                ld, si, slab, sc = self._load(cl, jj + 1 + s)
                last_ld[cl] = ld
                di = ndt_used[cl] % self.ndt
                ndt_used[cl] += 1
                dt = bg["dtmp"][di]
                gm = self.gemm(k, c, f"dn_j{jl}s{s}", bg["a8e"][s], self.PE, slab, dt, rows // 8, self.IE // 8,
                               nt, d.k_dn_e,
                               [ld, qa[(cl, s)],
                                ex_copy[(other, s)]] + ([dt_reader[cl][di]] if dt_reader[cl][di] else []))
                ids = bg["lists"].view(SnaxBingoKernelMoeDispatchArgs.rows_offset(caps, s))
                if self.fuse_deq_scatter:
                    prev = self.node(k, c, simd, f"scat_j{jl}s{s}", "__snax_bingo_kernel_simd_scatter_add_rows",
                                     SnaxBingoKernelSimdScatterAddRowsArgs(y, dt, nt, ids, int(d.cnt[e]),
                                                                           s_addr=sc.addr, y_bytes=self.SL),
                                     [gm, prev] + sc.deps)
                    self._scale_read(cl, sc, prev)
                    self._release(cl, si, gm, prev, sc)
                else:
                    mt = self.node(k, c, simd, f"dn_deq_j{jl}s{s}", "__snax_bingo_kernel_simd_mul_tiles",
                                   SnaxBingoKernelSimdMulTilesArgs(dt, sc.addr, dt, rows // 8, nt), [gm] + sc.deps)
                    self._scale_read(cl, sc, mt)
                    self._release(cl, si, gm, mt, sc)
                    prev = self.node(k, c, simd, f"scat_j{jl}s{s}", "__snax_bingo_kernel_simd_scatter_add_rows",
                                     SnaxBingoKernelSimdScatterAddRowsArgs(y, dt, nt, ids, int(d.cnt[e]), y_bytes=self.SL),
                                     [mt, prev])
                dt_reader[cl][di] = prev
            last_add[cl] = prev
            if jl % GS:
                # params chain_group: a gather waits for its group's last slice on each chip only; the slices'
                # last adds go into the core's in-order queue in slice order
                self.dfg.bingo_add_order_edge(prev_add[cl], prev)
            prev_add[cl] = prev

        def fold_store(c, jl, acc, last):
            """reduce root: root cluster c's folded slice jl from its own output slot into L3 (l3_overlay: Y lies over
            X's landing, so the first store follows the other root cluster's last read of X)"""
            over = [root_gat[1 - c]] if self.l3_overlay and jl == 0 else []
            stores.append(self.node(R, c, dm, f"y_store_c{c}j{jl}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(acc, self.ysym(self.y_off(c, jl)), self.SL),
                                    [last] + over))

        if self.args_phases:
            self.args_phase("MoE down")
        if self.per_kernel:
            # params barrier_at kernel: K4 and K5 in turns -- the down projection of as many slices as the output slots
            # hold, then their reduce into the root and its stores -- each ending at a barrier. reduce chain: the
            # ChainGathers one at a time into the root's two p1c slots, each slice stored after its chain. reduce root:
            # each root cluster pulls every other chip's partial of its slice (one pull in flight, its two p1c slots)
            # and its SIMD adds it into the cluster's own, in fold_order_c, then stores the slice
            pull_last, add_hist = {c: None for c in range(self.NC)}, {c: [] for c in range(self.NC)}
            for j0 in range(0, self.NSLC, self.NOUT):
                grp = list(range(j0, min(j0 + self.NOUT, self.NSLC)))
                for jl in grp:
                    if jl and jl % (self.NSLC // self.down_phases) == 0:
                        self.args_phase(f"MoE down from slice {jl}")
                    last_add = {}
                    for cl in range(N_CL):
                        down_cl(cl, jl, jl % self.NOUT, last_add)
                self.barrier({k: [] for k in self.chips})
                for jl in grp:
                    so, s2 = jl % self.NOUT, jl % 2
                    for c in (list(range(self.NC))[::-1] if (self.chain_first_alt and jl % 2) else list(range(self.NC))):
                        if self.reduce_impl == "chain":
                            p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
                            ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                                cm.remote, R, c, self.chain_order(c)[:-1], src=self.xch_addr(c, so * self.SL),
                                dst=f"({p1x} + {s2 * self.SL})", size=self.SL, junction=self.jct["Elementwise"],
                                jct_csr0=EW_F16_ADD, holds=[self.p1c[c]])
                            p1 = [p1_reader[c][s2]] if p1_reader[c][s2] is not None else []
                            red = self.node(R, c, xdma, f"y_chain_c{c}j{jl}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                            ([last_any] if last_any is not None else []) + p1)
                            last_red[c] = last_any = red
                            store(c, jl, red)
                            continue
                        acc = self.xch[(R, c)].view(so * self.SL)
                        for n_, k in enumerate(self.fold_order_c(c)[1:]):
                            q = len(add_hist[c])
                            slot = self.p1c[c].view((q % 2) * self.SL)
                            # one pull in flight; the slot free once the add two pulls back has read it
                            pl = self.node(R, c, dm, f"y_pull_c{c}j{jl}n{n_}", "__snax_bingo_kernel_idma_1d_copy",
                                           SnaxBingoKernelIdma1dCopyArgs(cm.remote(k, self.xch_addr(c, so * self.SL)),
                                                                         slot, self.SL),
                                           ([pull_last[c]] if pull_last[c] is not None else []) + add_hist[c][q - 2:q - 1])
                            pull_last[c] = pl
                            add_hist[c].append(self.node(R, c, simd, f"y_add_c{c}j{jl}n{n_}",
                                                         "__snax_bingo_kernel_simd_stream_elementwise",
                                                         SnaxBingoKernelSimdAddF16Args(acc, slot, acc, self.SL // 64, 32),
                                                         [pl] + add_hist[c][-1:]))
                        fold_store(c, jl, acc, add_hist[c][-1])
                if grp[-1] + 1 < self.NSLC:
                    self.barrier({k: [] for k in self.chips})
            self.end_window(stores[-self.NC:])
            self._checks()
            return
        freed = []                                          # barrier_at cross: (cluster, slot) the next barrier frees
        for jl in range(self.NSLC):
            if jl and jl % (self.NSLC // self.down_phases) == 0:
                self.args_phase(f"MoE down from slice {jl}")
            so = jl % self.NOUT
            last_add = {}
            for cl in range(N_CL):
                down_cl(cl, jl, so, last_add)
            if (jl + 1) % GS:
                continue
            j0 = jl + 1 - GS                                # the group's first slice
            s2, so0 = (j0 // GS) % 2, j0 % self.NOUT
            tail = jl >= self.NSLC - self.chain_tail
            corder = list(range(self.NC))[::-1] if (self.chain_first_alt and (j0 // GS) % 2) else list(range(self.NC))
            bar = None
            if self.cross:
                # params barrier_at cross: the slice's barrier takes every chip's last adds of the slice (before slice 0
                # also its A8 copies, which no chain may meet) and the root's reduces of the slice before; the output
                # slots those read are free once it is passed
                bar = self.barrier({k: [last_add[cl] for cl in self.clusters(k)]
                                    + ([t for (cl, _), t in ex_copy.items() if self.chip(cl) == k] if j0 == 0 else [])
                                    + ([last_red[c] for c in range(self.NC) if last_red[c] is not None] if k == R else [])
                                    for k in self.chips})
                for cl, o in freed:
                    xch_reader[cl][o] = bar[self.chip(cl)]
                freed = []
            for c in corder:
                ordr = self.chain_order(c)[:-1]
                p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, ordr, src=self.xch_addr(c, so0 * self.SL), dst=f"({p1x} + {s2 * GS * self.SL})",
                    size=GS * self.SL, junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD,
                    holds=[self.p1c[c]])
                first = list(copies_in[c]) if j0 == 0 and not self.cross else []
                before = last_red[c] if (self.chain_concurrent or tail) else last_any
                after = first + ([before] if before is not None else [])
                p1 = [p1_reader[c][s2]] if p1_reader[c][s2] is not None else []
                if self.halves:                             # params chain_halves: west and east at once, then one hop
                    west, east = self.halves
                    E = east[-1]
                    add_of = {self.chip(cl): last_add[cl] for cl in range(N_CL) if self.local(cl) == c}
                    slot = self.xch_addr(c, so * self.SL)          # a slot of the exchange buffer, on any chip
                    p1s = f"({p1x} + {s2 * self.SL})"
                    kern = "__snax_bingo_kernel_xdma_chain_gather"
                    gw = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                        cm.remote, R, c, west[:-1], src=slot, dst=p1s, size=self.SL,
                        junction=self.jct["Elementwise"], jct_csr0=EW_F16_ADD, holds=[self.p1c[c]])
                    wch = self.node(R, c, xdma, f"y_chw_c{c}j{jl}", kern, gw, [add_of[k] for k in west] + after + p1)
                    # the east collector's partial copied aside, so its chain lands in its slot and not in place; on
                    # the xDMA core that runs the chain, so the two are ordered by the core. params esrc_early: after
                    # this cluster's previous reduce only (its east chain read esrc, its last hop used this xDMA)
                    e_after = (first + ([last_red[c]] if last_red[c] is not None else [])) if self.esrc_early else after
                    ecp = self.node(E, c, xdma, f"y_esrc_c{c}j{jl}", "__snax_bingo_kernel_xdma_1d_copy",
                                    SnaxBingoKernelXdma1dCopyArgs(self.xch[(E, c)].view(so * self.SL), self.esrc[c],
                                                                  self.SL),
                                    [add_of[E]] + e_after)
                    ge = SnaxBingoKernelXdmaChainGatherArgs(
                        local_src=cm.remote(E, f"((uint64_t)({self.esrc[c].get_c_var_name()}))"),
                        chain=[cm.remote(k, slot) for k in east[:-1]] + [cm.remote(E, slot)], size=self.SL,
                        junction=str(self.jct["Elementwise"]), jct_csr0=EW_F16_ADD,
                        hops=[(k, c) for k in east], holds=[self.esrc[c]])
                    ech = self.node(E, c, xdma, f"y_che_c{c}j{jl}", kern, ge, [add_of[k] for k in east[:-1]] + [ecp] + after)
                    # the east sum (chip E's slot) added to the west sum (p1c) into the root's own slot, which the
                    # west chain has read already
                    gc = SnaxBingoKernelXdmaChainGatherArgs(
                        local_src=cm.remote(R, p1s), chain=[cm.remote(E, slot), cm.remote(R, slot)], size=self.SL,
                        junction=str(self.jct["Elementwise"]), jct_csr0=EW_F16_ADD, hops=[(E, c), (R, c)],
                        holds=[self.p1c[c]])
                    red = self.node(R, c, xdma, f"y_chain_c{c}j{jl}", kern, gc, [wch, ech])
                    p1_reader[c][s2] = red
                    for cl in range(N_CL):
                        if self.local(cl) == c:
                            # chip E's slot holds the east sum until the hop has read it; the root's (set by its store)
                            # the result until it is stored
                            k = self.chip(cl)
                            xch_reader[cl][so] = red if k in (E, R) else (wch if k in west else ech)
                else:
                    ins = [bar[R]] if self.cross else [last_add[cl] for cl in range(N_CL) if self.local(cl) == c]
                    red = self.node(R, c, xdma, f"y_chain_c{c}j{j0}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                    ins + after + p1)
                    for cl in range(N_CL):
                        if self.local(cl) == c:
                            for o in range(so0, so0 + GS):
                                if self.cross and self.chip(cl) != R:
                                    freed.append((cl, o))
                                else:
                                    xch_reader[cl][o] = red
                last_red[c] = last_any = red
                store(c, j0, red)
        if self.chain_tail:                                 # every push has landed before a last-slice chain
            for cl in range(N_CL):
                for o in self.clusters(self.chip(cl)):
                    if o != cl:
                        self.edge(last_ld[o], last_add[cl])
        self.end_window(stores[-self.NC:])
        self._checks()

    def hbm_bcasts(self, go):
        """x_hbm: the root's host queues its memory chip's push engine (the root's prefetch engine, which pushes the
        root's weights only after these): the router's weights behind one flag, then X a tile at a time, each tile
        and its flag multicast into every chip's L3. Returns the two host tasks; the second ends when X's last flag
        has landed on the root."""
        R, T = self.ROOT, self.TAG
        m = self.mem[R]
        one = self.h_seq[m].address + 64                    # a word holding 1 in the memory chip's HBM: the flags' value
        if self.x_cast == "unicast":
            # params x_cast unicast: the same tiles and flags written into each chip's L3 in turn (self.order: the root
            # first), with a progress flag a (chip, tile) on the root; each task ends once its last progress flag has
            # landed
            rw = self.cx_chip(R).host_node(
                f"rw_hbm_ucast_k{R:02x}", "__host_bingo_kernel_hbm_ucast",
                HostBingoKernelHbmUcastArgs(self.h_rwhbm.address, BingoMemSymbol(f"{T}_rwland"),
                                            BingoMemSymbol(f"{T}_rwflag"), BingoMemSymbol(f"{T}_rwuprog"), one, 1,
                                            self.rw_bytes, 1, m, self.engine[R], self.order),
                after=[go[R]])
            x = self.cx_chip(R).host_node(
                f"x_hbm_ucast_k{R:02x}", "__host_bingo_kernel_hbm_ucast",
                HostBingoKernelHbmUcastArgs(self.h_xhbm.address, self.xland(), BingoMemSymbol(f"{T}_xtflag"),
                                            BingoMemSymbol(f"{T}_xuprog"), one, 1, self.XT, self.NXT, m, self.engine[R],
                                            self.order),
                after=[rw])
            return rw, x
        rw = self.cx_chip(R).host_node(
            f"rw_hbm_bcast_k{R:02x}", "__host_bingo_kernel_hbm_bcast",
            HostBingoKernelHbmBcastArgs(self.h_rwhbm.address, BingoMemSymbol(f"{T}_rwland"),
                                        BingoMemSymbol(f"{T}_rwflag"), one, 1, self.rw_bytes, 1, m, self.engine[R]),
            after=[go[R]])
        x = self.cx_chip(R).host_node(
            f"x_hbm_bcast_k{R:02x}", "__host_bingo_kernel_hbm_bcast",
            HostBingoKernelHbmBcastArgs(self.h_xhbm.address, self.xland(), BingoMemSymbol(f"{T}_xtflag"), one, 1,
                                        self.XT, self.NXT, m, self.engine[R]),
            after=[rw])
        return rw, x

    def xsrc(self, off=0):
        return BingoMemSymbol(f"{self.TAG}_xy" if self.l3_overlay else f"{self.TAG}_xsrc", offset=off)

    def xland(self, off=0, chip_id=None):
        return (BingoMemSymbol(f"{self.TAG}_xy", offset=self.T * self.H + off, chip_id=chip_id) if self.l3_overlay
                else BingoMemSymbol(f"{self.TAG}_xland", offset=off, chip_id=chip_id))

    def ysym(self, off=0, chip_id=None):
        return BingoMemSymbol(f"{self.TAG}_xy" if self.l3_overlay else f"{self.TAG}_y", offset=off, chip_id=chip_id)

    def sdpark(self, c, jl):
        """params shared_down_early: where cluster c parks the shared MLP's part of its output slice jl (its own chip's
        L3)"""
        return BingoMemSymbol(f"{self.TAG}_sdpark", offset=(c * self.NSLC + jl) * self.SL)

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
        if self.x_hbm and self.stop == "route":
            # X's landing on the root, a chip in the middle and the last chip: the first, a middle and the last tile
            hq = np.asarray(self.d.hq, dtype=np.int8).reshape(-1)
            for k in dict.fromkeys((R, self.chips[len(self.chips) // 2], self.chips[-1])):
                for t in (0, self.NXT // 2, self.NXT - 1):
                    off = t * self.XT + (self.XT - n if t == self.NXT - 1 else 0)
                    checks.append((f"x_k{k:02x}_t{t}", self.xland(off, None if k == R else k),
                                   hq[off:off + n].view(np.uint8), n))
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
            # the shared MLP's A8 on the root's cluster 0, row 0: its own columns and, after the exchange, cluster 1's
            ns = BLK * self.shk[R]
            sh = np.ascontiguousarray(self.d.a8_shared()[0, BLK * self.sho[R]:BLK * self.sho[R] + ns]).astype(np.int8)
            got = self.stash("a8sh_root", R, 0, self.B[0]["a8sh"], ns, [])
            checks.append(("a8sh_r0", got, sh.view(np.uint8), ns))
        if self.stop == "full":
            img = np.concatenate([self.lay.to_d(self.y_raw[:, o0:o0 + w], MESH3).astype(np.float16)
                                  for c in range(self.NC) for o0, w in self.out_slices(c)]).view(np.uint8)
            for name, off in (("y_a", 0), ("y_mid", img.size // 2), ("y_b", img.size - n)):
                checks.append((name, self.ysym(off, chip_id=R), img[off:off + n], n))
        self.run_checks(checks)

    # ---- emission -----------------------------------------------------------------------------
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
        if self.sync == "barrier":
            if self.barrier_fence:                          # the fences before the args phases' loads exist
                self.fence_barriers()
                self.barrier_fence = False
            cross = self.check_window_local()
            if cross:
                raise ValueError(f"sync barrier: {len(cross)} dependency edges cross chips inside the window, e.g. "
                                 + "; ".join(f"{u.node_name} -> {v.node_name}" for u, v in cross[:4]))
            print(f"[{self.TAG}] sync barrier ({self.barrier_at}): {self.n_barrier} barriers a chip, no edge crosses "
                  f"chips inside the window")
        if self.args_phases:
            self._assign_phases()
        super().finish(app_name, x_prefixes)


def _app_name(S):
    arms = "" if (S.x_cast, S.reduce_impl, S.sync) == ("multicast", "chain", "edge") else \
        f" x_cast={S.x_cast}" + (f" barrier_at={S.barrier_at}" if S.barrier_at else "")
    return (f"DeepSeek-V2-Lite layer-1 MoE prefill, {S.T} tokens, {len(S.chips)} chips x {S.NC} clusters, "
            f"bcast={S.bcast_impl} reduce={S.reduce_impl} sync={S.sync}" + arms)


def run():
    run_driver("moe prefill workload generator", MBuild, _app_name)
