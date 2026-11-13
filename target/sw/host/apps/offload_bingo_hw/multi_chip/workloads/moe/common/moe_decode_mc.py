# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""DeepSeek-V2-Lite's layer-1 MoE for ONE token (a decode step) on the 16 + 4 chiplet platform, on the prefill's
machinery (moe_prefill_mc.MBuild): the router on the device, a static placement the device's route is checked against,
the weight rings, the A8 exchange between a chip's two clusters, the down projection by output halves and the
ChainGather reduce into the root, every GEMM on VersaCore's (8, 8, 16) shape with INT4 weights (the router INT8). The
data and the golden are moe_prefill_data.py's at one token.

THE TOKEN'S ROW TILE. The arrays take whole 8-row tiles, so the token is row 0 of one tile whose rows 1 to 7 are zero:
X's landing in every chip's L3 holds the token and seven zero rows. The zero rows stay zero through every step (their
GEMMs, silu(0) * 0, their route weight 0), and only row 0 is reduced into y's row and checked.

ONLY THE TOKEN'S EXPERTS ARE STREAMED. The prefill gives every expert at least one 8-row tile, which for one token would
stream all 64; here the 6 experts of the token's route (the generator's own, which the device's is checked against) and
the shared MLP are the only weights streamed, and no expert runs a pad tile.

THE PLACEMENT spreads every MLP over many chips so that each memory chip streams the same share and every cluster works.
The prefill's per-chip placement of whole experts would, for one token, put the 6 experts on 2 chips and leave the other
chips' links idle. Instead:
    shared MLP   a run of its column blocks on every chip, SHB / 16 of them (as the prefill's)
    a routed     its column blocks in params routed_split runs (pieces), each on another chip; every chip then holds
      expert     TOP_K * routed_split / 16 pieces of distinct experts: piece i of slot s on chip (s * routed_split + i)
                 mod 16 in the platform's order, so with routed_split 8 the chips of rows 0 and 1 hold slots 0, 2, 4 and
                 those of rows 2 and 3 slots 1, 3, 5
Every chip holds the same number of intermediate columns (the shared run and its pieces: 44 blocks at the model's size)
and streams the same bytes: per column, its gate and up columns over every input and its row of the down projection
over every output. A memory chip feeds the four chips of its row, two over each of its links, so every memory chip and
every link carries the same share (the root's link one chunk more: the router's). Each routed slot on 2 chips and the
shared expert on 4 would balance the streams too, but 12 chips would then have nothing to stream until the route is
known; here every chip streams its shared run meanwhile.

THE CLUSTERS. Each piece's columns are split over the chip's two clusters by 16-column blocks, cluster 0 an even number
(cluster 1's columns then start on a SIMD beat of the A8 rows), moved two at a time until the two clusters hold the
same number. Per piece, each cluster runs:
    gate|up      per weight chunk of params blocks column blocks: the GEMM over the token's tile (in L1 once: every
                 expert's one row is the token, so there is nothing to gather), its dequantisation and SwiGLU into the
                 piece's A16
    quantise     the A16 into the piece's row-major A8 at the cluster's columns, a routed piece's times the token's
                 route weight from the device's row lists (the shared MLP's: 1)
    exchange     its columns copied into the other cluster's A8 of the piece (the prefill's), so each cluster has all
                 the chip's columns of every piece
    down         per output slice of the cluster's half of the outputs: the shared piece's GEMM over the chip's shared
                 columns writes the slice, dequantised; each routed piece's GEMM over the chip's columns of its expert,
                 dequantised, its token row added into the slice's row 0 (simd_scatter_add_rows, by the device's row
                 list). The shared MLP's down runs right after its exchange, ahead of the routed pieces, since its
                 chunks are pushed before the route is known.

THE ROUTE runs on the root's cluster 0. The router's weights and factors are the first chunk of that cluster's weight
ring, pushed by the root's prefetcher from the HBM of the root's memory chip: only the root routes, so they cross the
links to the root only, not the multicast into every chip's L3 the prefill's route (every cluster routing its tiles)
needs. The router GEMM over the token's tile, its factors, the softmax and the top 6 into the route table's entry, which
the root multicasts to every chip's L3 behind a flag. Every cluster fetches it and turns it into its chip's row lists
(moe_dispatch), which fails when a held expert's count is not the placement's: the check of the device's route against
the generator's. The row lists feed the quantise (the route weight) and the down's scatter-add (the token's row).

ROUTE BEFORE THE ROUTED PUSHES. Each chip's weight prefetcher runs as two host tasks over the ring schedule
(host_kernel_lib.h weight_prefetch_range): the first pushes each ring's first n chunks, those no route decides (the
router's chunk, the shared MLP's gate|up and down) and starts with the window once X has landed (the prefill's
prefetch_after_x: an HBM multicast into an L3 beside a push into it can deadlock); the second pushes the rest, the
routed experts' chunks, and depends on both of the chip's moe_dispatch tasks. So no routed byte leaves a memory chip
before the device's route has reached that chip and passed its check, while the shared MLP's stream covers the route's
latency. n is the fewer such chunks of the chip's two rings, so a chunk past it that no route decides (the root's cluster
0 ring has the router's chunk extra) waits with the routed ones. The HBM image of each ring has a 64-byte gap before
chunk n: no run of the prefetcher crosses the split, so the ring places the compiler fixed are the ones the two tasks
push into. With weight_ring.mode cluster (each cluster posts its ring's runs) chunk n starts a run of its own
(WeightRings.boundary), whose post depends on the chip's moe_dispatch tasks: the route's wait is a graph edge.

X FROM HBM. The root's host queues its memory chip's push engine, which multicasts the token's row from HBM into every
chip's L3 behind one flag (the prefill's x_hbm, one tile of one row); every cluster waits for the flag and copies the
tile into L1.

THE REDUCE: one ChainGather per output slice of each cluster through every chip's same cluster into the root's (params
chain_path). params slice (default: H / 2, one slice a cluster): a chain costs about the same whatever its payload, so
more slices add chains without shortening the last. The chains of a slice run one at a time except the last slice's two,
which run at once once every push has landed (the prefill's chain_tail: on every chip each cluster's last scatter-add
also waits for the other cluster's last weight load).

Y TO HBM. After both clusters' last chains (no push and no chain is in flight then), each root cluster copies row 0 of
its reduced slices into a row-major FP16 row (a 2D copy inside its L1) and writes it to the HBM of the root's memory chip:
y [H] FP16, row-major.

CHECKS (the root's host, byte for byte): the route table's entry on the root, the root cluster 1's row lists, the root's
first routed piece's A8 and its shared A8 (row 0, after the exchange), and y in HBM.

THE GOLDEN, the device's arithmetic: each chip's partial over an output slice is the shared MLP's down over the chip's
shared columns, then each of its routed pieces' down over its columns added in piece order (the D port and the
dequantisation per piece, FP16 adds); the partials are folded in the chain's order.

TIMING: the start stamp after every chip's setup; the end stamp after both writes of y to HBM.

THE GEMV MODE (params gemv). Every weight multiply, the router's included, on VersaCore's one-token GEMV ((1, 4, 32),
__snax_bingo_kernel_gemv, 128 weights a pass: 128 INT8 bytes, or 64 bytes of INT4 nibbles through B's converter), its
operands in the layouts of shapes 0..2 (the generator's images: the (16, 4, 16) B blocks, INT4 in their paired form), A
the token's a_row; no 8-row tile anywhere, every activation one row. The placement, the cluster split, the rings and their
two prefetcher tasks, the route's path, the exchange and the reduce are the ones above. Per step:
    X          the token's INT8 row in a_row (THE X LAYOUT below), multicast from HBM into every chip's L3 behind one
               flag as above; every cluster copies it into L1
    route      one INT8 GEMV of the router's chunk (W_r in the B layout, then its factors' row) into the logits' row,
               times the factors into row 0 of the softmax's 8 rows (rows 1 to 7 zeroed before the window), then the
               softmax, the top 6, the table's multicast and moe_dispatch as above
    gate|up    the chunks of a cluster's run of a piece: at most `blocks` column blocks each, as even in size as whole
               blocks allow. Per chunk, one INT4 GEMV of X into the chunk's part of the piece's G row, then times the
               chunk's factor row in place. The chunk's columns are interleaved by 16-column block (a block's gate
               columns, then its up columns), so each pair of blocks the GEMV reads is one block's gate and up, and every
               block's 64 bytes of G hold its gate values, then its up values. After the piece's last chunk: SiLU over
               the G row, then each block's SiLU'd gate times its up (the stream kernel's padded-row read, one beat a
               block, the rows 64 bytes apart) into the piece's A16 row, the cluster's columns in order
    quantise   the A16 row spread on the xDMA into the quantiser's input: atom t (4 values, 8 bytes) at 256 (t / 2) +
               16 (t % 2), so each 32-byte beat, 256 bytes apart as simd_quant_rows reads a D-layout row, holds two
               atoms with zeros after each; then simd_quant_rows of that one row times the route weight (the device's
               row lists; the shared MLP's 1) into the cluster's run of the piece's A8 row in a_row (2 bytes a column).
               Fp16ToInt8 keeps every value's place, so the zeros become each a_row word's free bytes. The quantiser's
               input and the A16 row's padding are zeroed before the window and only the atoms are ever written
    exchange   the cluster's run of the A8 row copied into the other cluster's (one 1-D copy)
    down       per output slice and piece: one INT4 GEMV of the piece's A8 row (both clusters' columns) into the
               slice's FP16 row; the shared piece's into the chip's partial, times the factor row in place; a routed
               piece's into a row buffer, times its factor row, then added into the partial (stream add)
    reduce     the ChainGathers above, one row a slice
    y          each root cluster's fold row is its half of y in column order and is written to HBM as it is
Every chunk's factors stream with it as one FP16 row (2 bytes a column), not as an 8-row scale tile. Every SIMD task is a
32-byte-beat stream kernel (stream_elementwise, silu_f16_f16, quant_rows, softmax_f16_f16); none of simd_row.h's. There is
no GEMM: no gemm_full can follow a GEMV on a GEMM core (or be rewritten into a gemm_minimal behind one), which the build
checks. The golden is the one above: the GEMV's D port and every SIMD step are the tile path's arithmetic on row 0, so the
route, the row lists, the A8 values and y are the same bytes.

THE X LAYOUT (gemv) a producer delivers into md_xland, then md_xtflag = 1: the token's INT8 row in a_row, 2 x hidden
bytes, value c at byte (c // 4) * 8 + c % 4; the GEMV reads the first 4 bytes of each 8-byte word, so the other 4 are free
(zero here). A producer holding the FP16 row (the layer's glue on C00, after the RMSNorm and the post-attention gain)
makes it with two existing kernels: an xDMA spread of the row's 8-byte atoms 16 bytes apart into an FP16 buffer zeroed
once (the gaps stay zero across uses), then simd_fp16_to_int8 over its 4 x hidden bytes at the static scale s_h (a_row
out). A contiguous INT8 row has no cheap conversion (4-byte runs; the xDMA moves 8-byte atoms).

CHECKS (gemv): as above, the A8 rows in a_row (both clusters' columns, the free bytes zero).

params: tokens 1; routed_split (default 8); slice; blocks (column blocks a gate|up chunk; gemv: at most); w_chunk_bytes
(at least the largest chunk); w_buffers; weight_ring; chain_path; chain_tail (0 or 1, default 1); chain_first_alt;
warm_root; scale_out; gemv (the GEMV mode; not with b_parity, w4_paired, scale_out); inter_e
/ inter_sh (the experts' and the shared MLP's first columns, as the prefill's proof of concept); and the platform and
graph switches every block shares (moe_prefill_mc.py, chiplet_mc.py).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moe_prefill_mc as mp  # noqa: E402  (puts util/sim/multichip and mini_compiler on sys.path)
from driver import run_driver  # noqa: E402
from bingo_kernel_args import (  # noqa: E402
    HostBingoKernelHbmBcastArgs, HostBingoKernelWeightPrefetchRangeArgs, SnaxBingoKernelGemvArgs,
    SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaFetchFlaggedArgs,
    SnaxBingoKernelIdmaPutFlaggedArgs, SnaxBingoKernelMoeDispatchArgs, SnaxBingoKernelMoeTopkRowsArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdDeqSwigluArgs, SnaxBingoKernelSimdMulF16Args,
    SnaxBingoKernelSimdMulTilesArgs, SnaxBingoKernelSimdQuantRowsArgs, SnaxBingoKernelSimdScatterAddRowsArgs,
    SnaxBingoKernelSimdSiluF16F16Args, SnaxBingoKernelSimdSoftmaxF16F16Args, SnaxBingoKernelWaitFlagsArgs,
    SnaxBingoKernelXdma6dArgs, SnaxBingoKernelXdmaChainGatherArgs, SnaxBingoKernelXdmaMemsetArgs)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs import at_offset  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import moe_prefill_data as md  # noqa: E402

BLK, D_BLOCK, MESH3, ROUTE_B, SIMD_BEAT = mp.BLK, mp.D_BLOCK, mp.MESH3, mp.ROUTE_B, mp.SIMD_BEAT
ROWS = md.TILE_ROWS           # the token's row tile: the token, then zero rows
GAP = 64                      # the HBM image's gap before a ring's first chunk of the second prefetcher task
GV_MESH = (16, 4, 16)         # gemv: the layouts of VersaCore's shapes 0..2, which the GEMV reads
ATOM, LANES = 8, 8            # gemv: the xDMA's 8-byte atom (4 FP16 values) and its lanes a beat
SPREAD_ATOMS = 2 * LANES      # gemv: the atoms of the A16 row one step of the spread's outer loop moves


def a_row_bytes(row):
    """An INT8 row as the GEMV's a_row: value c at byte (c // 4) * 8 + c % 4, the other 4 bytes of each word zero."""
    row = np.asarray(row, dtype=np.int8).reshape(-1)
    c = np.arange(row.size)
    out = np.zeros(2 * row.size, dtype=np.int8)
    out[(c // 4) * 8 + c % 4] = row
    return out


def spread_steps(cols):
    """gemv: the outer steps of the spread of an A16 row of `cols` values (SPREAD_ATOMS atoms a step)."""
    return -(-(cols // 4) // SPREAD_ATOMS)


def spread_args(src, dst, cols):
    """gemv: the xDMA pass that spreads an FP16 row of `cols` values (src) into simd_quant_rows' input (dst): atom t
    (values 4t .. 4t + 3) to byte 256 (t // 2) + 16 (t % 2), so the 32-byte beat m, 256 bytes on from beat m - 1, holds
    atoms 2m and 2m + 1, each followed by 8 bytes the pass never writes. The lanes take atoms 2 apart (16 bytes in the
    row, 256 in dst), the inner loop the atom of the pair, the outer loop SPREAD_ATOMS atoms at a time; the last step
    moves whole lanes, so src is read and dst written up to spread_steps(cols) steps (128 and 2048 bytes each). The
    nest is walked here and compared with that mapping."""
    steps = spread_steps(cols)
    args = SnaxBingoKernelXdma6dArgs(src, dst, 2 * ATOM, D_BLOCK, [ATOM, SPREAD_ATOMS * ATOM], [2, steps],
                                     [2 * ATOM, LANES * D_BLOCK], [2, steps])
    j, b, lane = np.meshgrid(np.arange(2), np.arange(steps), np.arange(LANES), indexing="ij")
    so = (j * ATOM + b * SPREAD_ATOMS * ATOM + lane * 2 * ATOM).reshape(-1)
    do = (j * 2 * ATOM + b * LANES * D_BLOCK + lane * D_BLOCK).reshape(-1)
    t = so // ATOM
    if so.min() < 0 or np.any(so % ATOM) or not np.array_equal(np.sort(t), np.arange(steps * SPREAD_ATOMS)) \
            or not np.array_equal(do, D_BLOCK * (t // 2) + 2 * ATOM * (t % 2)):
        raise ValueError("the spread's xDMA nest does not make the quantiser's input")
    return args


class SimdMulRowsF16Args(SnaxBingoKernelSimdMulF16Args):
    """out = a (.) b over `rows` rows of `cols` FP16 values, each operand's rows `row_stride` bytes apart and out packed:
    the stream kernel's padded-row read (src_row_stride, device_kernel_args.h), both operands at the same stride."""

    def __init__(self, a_addr, b_addr, out_addr, rows: int, cols: int, row_stride: int):
        super().__init__(a_addr, b_addr, out_addr, rows, cols)
        if row_stride % SIMD_BEAT or row_stride < 2 * cols:
            raise ValueError(f"row_stride={row_stride}: whole SIMD beats, at least a row of {cols} values")
        self.row_stride = int(row_stride)

    def get_c_field_assignments(self, handle_name_map):
        a = super().get_c_field_assignments(handle_name_map)
        a["src_row_stride"] = str(self.row_stride)
        return a


class Piece:
    """A run of one MLP's intermediate columns on a chip: e (None: the shared MLP), its first column k0 within the MLP,
    n columns, b0 of its column blocks on cluster 0 (the rest on cluster 1)."""

    def __init__(self, e, k0, n):
        self.e, self.k0, self.n, self.b0 = e, int(k0), int(n), 0

    @property
    def nb(self):
        return self.n // BLK

    def cols(self, c):
        """Cluster c's columns of the piece: (first column within the piece, columns)."""
        return (0, BLK * self.b0) if c == 0 else (BLK * self.b0, self.n - BLK * self.b0)

    def __repr__(self):
        return f"{'sh' if self.e is None else f'e{self.e}'}[{self.k0}:{self.k0 + self.n}]"


class DBuild(mp.MBuild):
    """The MoE decode block (the module docstring), on the prefill's helpers. run_driver makes the program in four
    steps: __init__ (the parameters), mem_handles() (the data, the placement, the golden and the memory), build() (the
    task graph: every task and edge) and finish() (the compile into the C header and the data image)."""
    TAG = "md"

    def __init__(self, args, p, plat, hw, shared=None):
        """The parameters; no data, memory or task yet. ChipletBuild.__init__, not MBuild's: the decode refuses the
        prefill's switches it does not take and sets the ones the prefill's helpers it reuses read (gemm, _load,
        _release, _pack_b, _chunk, chain_order, finish) at what it builds; then its own (gemv, the reduce's chains, the
        gate blocks a gate|up chunk)."""
        cm.ChipletBuild.__init__(self, args, p, plat, hw, shared=shared)
        if self.T != 1:
            raise ValueError(f"params tokens={self.T}: the decode runs one token")
        if self.bcast_impl != "bcast_l1" or self.reduce_impl != "chain" or self.sync != "edge":
            raise ValueError("this driver builds bcast bcast_l1, reduce chain, sync edge")
        for k in ("x_device", "stop", "route", "expert_rows", "x_blocks", "args_phases", "chain_halves", "out_slots"):
            if k in p:
                raise ValueError(f"params {k}: a prefill switch the decode does not take")
        # the switches of the prefill's helpers, at what the decode builds
        self.x_hbm, self.x_device, self.l3_overlay, self.args_phases = True, False, False, False
        self.plain_trailer = bool(p.get("weight_ring", {}).get("trailer", False))
        if self.plain_trailer:
            raise ValueError("params weight_ring.trailer: the decode's rings push flags (queued)")
        self.resid, self.phase_starts, self.phase_names, self.loaders = None, [], [], {}
        self.b_parity = bool(p.get("b_parity", False))
        self.w4_paired = bool(p.get("w4_paired", False))
        if self.b_parity and self.w4_paired:
            raise ValueError("params b_parity, w4_paired: one INT4 layout")
        self.warm_root = bool(p.get("warm_root", False))
        self.scale_out = bool(p.get("scale_out", False))
        self.n_sc = int(p.get("scale_bufs", 4))
        if self.scale_out and self.n_sc < 2:
            raise ValueError("params scale_out: scale_bufs >= 2")
        self.chain_path = mp.chain_path_param(p, self.chips, self.ROOT)
        self.chain_tail = int(p.get("chain_tail", 1))
        if self.chain_tail not in (0, 1):
            raise ValueError(f"params chain_tail={self.chain_tail}: 0 or 1")
        self.chain_first_alt = bool(p.get("chain_first_alt", False))
        self.ndt = int(p.get("dtmp_bufs", 2))
        if self.ndt < 1:
            raise ValueError(f"params dtmp_bufs={self.ndt}: at least one")
        # params gemv: every weight multiply on the one-token GEMV (the module docstring's GEMV MODE)
        self.gemv = bool(p.get("gemv", False))
        if self.gemv:
            bad = [k for k in ("b_parity", "w4_paired", "scale_out") if p.get(k, False)]
            if bad:
                raise ValueError(f"params gemv with {bad}: the GEMV reads its own paired INT4 layout, its factor rows "
                                 f"from the slab, and is a blocking task (no GEMM to queue)")
        self.BPC = int(p.get("blocks", 3))

    def mem_handles(self):
        """The data, the placement, the golden and the memory; the only tasks made are the prefetchers. The decode's
        own, not MBuild's (which sizes its buffers by a prompt's 8-token tiles and places whole experts by their
        tokens): the token's data (_data) and the sizes it gives (the output slices, params slice, its default a
        cluster's outputs; X's multicast tile); the pieces of the token's experts and of the shared MLP on the chips
        (_place, params routed_split); the golden (_golden); then _stage(): X and Y's row in the HBM of the root's memory
        chip, every chip's landing areas and flags, every cluster's weight chunks and factors in its memory chip's HBM;
        make_streams(): each cluster's L1 weight slabs; make_rings(): each chip's weight rings in its L3 and its host's
        prefetcher task; _buffers(): every cluster's L1 buffers."""
        d = self.d = self._data(self.args, self.p, self.hw)
        self.lay = d.layout
        self.H, self.IE, self.ISH = d.H, d.I_e, d.I_sh
        self.EB, self.SHB = self.IE // BLK, self.ISH // BLK
        self.SW = int(self.p.get("slice", self.H // self.NC))
        if (self.H // self.NC) % self.SW or self.SW % BLK:
            raise ValueError(f"params slice={self.SW}: whole column blocks dividing a cluster's {self.H // self.NC} "
                             f"outputs")
        self.NSLC = self.H // self.NC // self.SW
        self.SL = ROWS * self.SW * 2                      # one output slice of the token's tile, FP16
        self.PX = mp.row_pitch(self.H)
        self.XT, self.NXT = self.H, 1                     # X's multicast: one tile, the token's row
        if self.gemv:
            self.SL = self.SW * 2                         # gemv: one output slice is the token's row
            self.XT = 2 * self.H                          # gemv: X's multicast is the token's a_row

        self._place(self.p)
        self._golden()
        self._stage()
        self.make_streams()
        self.make_rings()
        self._buffers()

    # ---- the placement ----------------------------------------------------------------------------------------------
    def _place(self, p):
        """self.pieces {chip: [shared piece, routed pieces]}, self.held {chip: the routed pieces' experts}, and each
        piece's split over the chip's clusters."""
        d, nch = self.d, len(self.chips)
        self.route_ids = [int(e) for e in d.ids[0]]       # the generator's route of the token, slot order
        if len(set(self.route_ids)) != md.TOP_K or any(int(d.cnt[e]) != 1 for e in self.route_ids):
            raise ValueError(f"the token's route {self.route_ids}: {md.TOP_K} distinct experts")
        if self.SHB % nch:
            raise ValueError(f"{self.SHB} shared blocks over {nch} chips")
        self.shk = {k: self.SHB // nch for k in self.chips}
        self.sho = {k: i * self.SHB // nch for i, k in enumerate(self.chips)}
        R = self.routed_split = int(p.get("routed_split", 8))
        if R < 1 or self.EB % R or (md.TOP_K * R) % nch:
            raise ValueError(f"params routed_split={R}: dividing an expert's {self.EB} column blocks, {md.TOP_K} x it "
                             f"a multiple of the {nch} chips")
        self.PB = self.EB // R                            # column blocks a routed piece
        self.pieces = {k: [Piece(None, BLK * self.sho[k], BLK * self.shk[k])] for k in self.chips}
        for s, e in enumerate(self.route_ids):
            for i in range(R):
                k = self.chips[(s * R + i) % nch]
                self.pieces[k].append(Piece(e, BLK * i * self.PB, BLK * self.PB))
        self.held = {}
        for k in self.chips:
            es = [pc.e for pc in self.pieces[k][1:]]
            if len(set(es)) != len(es) or not 0 < len(es) <= SnaxBingoKernelMoeDispatchArgs.MAX_SLOTS:
                raise ValueError(f"chip {k:02x} holds pieces of experts {es}: distinct, "
                                 f"{SnaxBingoKernelMoeDispatchArgs.MAX_SLOTS} at most (moe_dispatch's slots)")
            self.held[k] = es
            self._split(self.pieces[k], 1 if k == self.ROOT else 0)
        for k in self.chips:
            print(f"[md] chip {k:02x}: " + ", ".join(f"{pc} ({pc.b0}+{pc.nb - pc.b0} blocks)" for pc in self.pieces[k]))

    def _split(self, pieces, extra0):
        """Each piece's blocks on cluster 0 (b0): an even number, at least 2, leaving cluster 1 at least one. For each
        such split of the shared piece, the routed pieces start at half (rounded down to even) and take moves of two
        blocks from the busier cluster, each to the piece it leaves the most even, while that evens the chip's two
        clusters. The split kept: the two clusters' blocks the most even, then the most route-free chunks in the
        fewer of the two rings (cluster 0's has `extra0` more: the router's), then the shared piece the most even."""
        for pc in pieces:
            if pc.nb < 3:
                raise ValueError(f"a piece of {pc.nb} column blocks: at least 3 (cluster 0 an even number, cluster 1 "
                                 f"at least one)")
        sh, routed = pieces[0], pieces[1:]
        nchunks = lambda nb: len(self.chunks(BLK * nb))
        best = None
        for s0 in range(2, sh.nb, 2):
            b0 = [max(2, pc.nb // 2 - (pc.nb // 2) % 2) for pc in routed]
            for _ in range(sum(pc.nb for pc in routed)):
                n0 = s0 + sum(b0)
                gap = n0 - (sh.nb + sum(pc.nb for pc in routed) - n0)
                step = -2 if gap > 0 else 2
                cand = [i for i, pc in enumerate(routed) if 2 <= b0[i] + step <= pc.nb - 1]
                if gap == 0 or not cand or abs(gap + 2 * step) >= abs(gap):
                    break
                i = min(cand, key=lambda i: (abs(2 * (b0[i] + step) - routed[i].nb), i))
                b0[i] += step
            n0 = s0 + sum(b0)
            imbalance = abs(2 * n0 - sum(pc.nb for pc in pieces))
            free = min(extra0 + nchunks(s0), nchunks(sh.nb - s0)) + self.NSLC
            key = (imbalance, -free, abs(2 * s0 - sh.nb))
            if best is None or key < best[0]:
                best = (key, s0, b0)
        sh.b0 = best[1]
        for pc, b in zip(routed, best[2]):
            pc.b0 = b

    def chunks(self, n_cols):
        """[(first column, columns)] of a run of n_cols intermediate columns in gate|up chunks: the prefill's (params
        blocks column blocks each, the last the rest); gemv: as few chunks of at most `blocks` blocks as cover the run,
        their sizes as even as whole blocks allow."""
        if not self.gemv:
            return super().chunks(n_cols)
        nb = n_cols // BLK
        n = -(-nb // self.BPC)
        out, b = [], 0
        for i in range(n):
            m = nb // n + (1 if i < nb % n else 0)
            out.append((BLK * b, BLK * m))
            b += m
        return out

    # ---- the golden -------------------------------------------------------------------------------------------------
    def _golden(self):
        d, add16 = self.d, self.d.fp.add16
        a8sh = d.a8_shared()                              # [1, I_sh]
        y = np.zeros(self.H, dtype=np.float16)
        for c in range(self.NC):
            for c0, w in self.out_slices(c):
                parts = {}
                for k in self.chips:
                    sh = self.pieces[k][0]
                    yk = d.d16(d.sh, a8sh, sh.k0, sh.k0 + sh.n, c0, w)[0].copy()
                    for pc in self.pieces[k][1:]:
                        yk = add16(yk, d.d16(d.expert(pc.e), d.a8_expert(pc.e), pc.k0, pc.k0 + pc.n, c0, w)[0])
                    parts[k] = yk
                order = self.chain_order(c)
                acc = parts[order[0]]
                for k in order[1:]:
                    acc = add16(acc, parts[k])
                y[c0:c0 + w] = acc
        self.y_row = y
        print(f"[md] {d.repo}@{d.revision[:12]} layer {d.layer} MoE, one token: experts {self.route_ids} (slot order), "
              f"route weights {[round(float(x), 4) for x in d.w16[0]]}; experts of {self.IE}, shared {self.ISH}")
        for line in d.report(y[None, :])[1:]:              # the D-port shifts, the float layer's error
            print(f"[md] {line}")

    # ---- data -------------------------------------------------------------------------------------------------------
    def _stage(self):
        if self.gemv:
            return self._stage_gv()
        st, d, T, R = self.st, self.d, self.TAG, self.ROOT
        m = self.mem[R]
        loc_r = (m >> 4, m & 15)
        # X: the token's row in the HBM of the root's memory chip; every chip's landing, the token and the zero rows
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", np.asarray(d.hq, dtype=np.int8).reshape(-1), mem_chip=loc_r)
        st.put_zeros(f"{T}_xland", "int8_t", ROWS * self.H, preload=True)
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64)
        st.put_zeros(f"{T}_rland", "uint8_t", 64, preload=True)
        st.put_zeros(f"{T}_rflag", "uint8_t", 64)
        if self.warm_root:
            st.put_zeros(f"{T}_zero1k", "uint8_t", 1024)
        # y: the block's output row, written back to the HBM of the root's memory chip
        self.h_yhbm = st.put_hbm(f"{T}_yhbm", np.zeros(2 * self.H, dtype=np.uint8), mem_chip=loc_r)
        rw = self.lay.to_b(np.asarray(d.wr_q, dtype=np.int8), MESH3).astype(np.int8).reshape(-1)
        rws = self._scale_tile(d.wr_f).view(np.int8).reshape(-1)
        self.rw_bytes = rw.size
        # each cluster's weights in the order it loads them, keyed, and the index of each ring's first chunk a route
        # decides (or that follows one)
        free, blobs = {}, {}
        sh_mp = d.sh
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            sh = self.pieces[k][0]
            items = []                                    # (key, blob, scale offset)
            if cl == 0:                                   # the router's weights and factors, plain INT8 B layout
                items.append((("rw",), np.concatenate([rw, rws]), rw.size))
            e0, en = sh.cols(c)
            for j, (b0, n) in enumerate(self.chunks(en)):
                a = sh.k0 + e0 + b0
                cols = np.concatenate([np.arange(a, a + n), self.ISH + np.arange(a, a + n)])
                items.append((("sg", j),) + self._chunk(self._pack_b(sh_mp["gu_q"][:, cols]), sh_mp["gu_f"][cols]))
            for jl, (o0, w) in enumerate(self.out_slices(c)):
                items.append((("sd", jl),) + self._chunk(self._pack_b(sh_mp["dn_q"][sh.k0:sh.k0 + sh.n, o0:o0 + w]),
                                                         sh_mp["dn_f"][o0:o0 + w]))
            free[cl] = len(items)
            for pi, pc in enumerate(self.pieces[k][1:], start=1):
                emp = d.expert(pc.e)
                e0, en = pc.cols(c)
                for j, (b0, n) in enumerate(self.chunks(en)):
                    a = pc.k0 + e0 + b0
                    cols = np.concatenate([np.arange(a, a + n), self.IE + np.arange(a, a + n)])
                    items.append((("gu", pi, j),) + self._chunk(self._pack_b(emp["gu_q"][:, cols]), emp["gu_f"][cols]))
            for jl, (o0, w) in enumerate(self.out_slices(c)):
                for pi, pc in enumerate(self.pieces[k][1:], start=1):
                    emp = d.expert(pc.e)
                    items.append((("dn", jl, pi),) + self._chunk(self._pack_b(emp["dn_q"][pc.k0:pc.k0 + pc.n, o0:o0 + w]),
                                                                 emp["dn_f"][o0:o0 + w]))
            blobs[cl] = items
        self._images(blobs, free)

    def _images(self, blobs, free):
        """Each cluster's weight image in its memory chip's HBM: blobs {cl: [(key, chunk bytes, factors' offset)]} in
        the order the cluster loads them, free {cl: its first chunks no route decides}; the gap before the first chunk
        of each chip's second prefetcher task; then the stream report."""
        st, T = self.st, self.TAG
        for cl, items in blobs.items():
            for key, b, _ in items:
                if b.size > self.chunk:
                    raise ValueError(f"cluster {cl}: weight chunk {key} of {b.size} B; params w_chunk_bytes is "
                                     f"{self.chunk}")
        self.img, self.chunk_off, self.cidx = {}, {}, {}
        # the split between each chip's two prefetcher tasks: the fewer route-free chunks of its two rings
        self.n_free = {k: min(free[cl] for cl in self.clusters(k)) for k in self.chips}
        for cl in range(self.N_CL):
            k = self.chip(cl)
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            parts, offs, keys = [], [], {}
            off = 0
            for i, (key, b, soff) in enumerate(blobs[cl]):
                if i == self.n_free[k] and i:
                    parts.append(np.zeros(GAP, dtype=np.int8))
                    off += GAP
                keys[key] = len(offs)
                parts.append(np.asarray(b, dtype=np.int8))
                offs.append((off, b.size, soff))
                off += b.size
            self.img[cl] = st.put_hbm(f"{T}_w_cl{cl}", np.concatenate(parts), mem_chip=loc)
            self.chunk_off[cl], self.cidx[cl] = offs, keys
        self._stream_report()

    # ---- gemv: the data ---------------------------------------------------------------------------------------------
    def _pack_gv(self, q):
        """INT4 weights [K, N] as the GEMV reads them: the paired layout of the (16, 4, 16) B blocks, nibble-packed."""
        return self.lay.pack_int4(self.lay.to_b_pairs(np.asarray(q), GV_MESH))

    @staticmethod
    def _chunk_gv(w_bytes, f):
        """A GEMV weight chunk as it is streamed: the weights, then (64-B aligned) its output columns' factors as one
        FP16 row. Returns (bytes, offset of the factor row)."""
        wb = np.asarray(w_bytes, dtype=np.int8).reshape(-1)
        off = -(-wb.size // 64) * 64
        row = np.ascontiguousarray(np.asarray(f, dtype=np.float16)).view(np.int8)
        return np.concatenate([wb, np.zeros(off - wb.size, dtype=np.int8), row]), off

    @staticmethod
    def _gu_cols_gv(inter, a, n):
        """The gate|up columns of the n intermediate columns from `a` of an MLP of `inter`, as a GEMV chunk takes them:
        each 16-column block's gate columns, then its up columns."""
        return np.concatenate([np.r_[a + b + np.arange(BLK), inter + a + b + np.arange(BLK)] for b in range(0, n, BLK)])

    def _stage_gv(self):
        """gemv: X as the token's a_row; the router's INT8 chunk in the B layout with its factor row; every gate|up and
        down chunk in the GEMV's paired INT4 layout with its factor row, in the tile path's order."""
        st, d, T, R = self.st, self.d, self.TAG, self.ROOT
        m = self.mem[R]
        loc_r = (m >> 4, m & 15)
        # X: the token's a_row in the HBM of the root's memory chip; every chip's landing of it
        self.h_xhbm = st.put_hbm(f"{T}_xhbm", a_row_bytes(d.hq[0]), mem_chip=loc_r)
        st.put_zeros(f"{T}_xland", "int8_t", self.XT, preload=True)
        st.put_zeros(f"{T}_xtflag", "uint8_t", 64)
        st.put_zeros(f"{T}_rland", "uint8_t", 64, preload=True)
        st.put_zeros(f"{T}_rflag", "uint8_t", 64)
        if self.warm_root:
            st.put_zeros(f"{T}_zero1k", "uint8_t", 1024)
        self.h_yhbm = st.put_hbm(f"{T}_yhbm", np.zeros(2 * self.H, dtype=np.uint8), mem_chip=loc_r)
        rw = self.lay.to_b(np.asarray(d.wr_q, dtype=np.int8), GV_MESH).astype(np.int8).reshape(-1)
        self.rw_bytes = rw.size
        free, blobs = {}, {}
        sh_mp = d.sh
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            sh = self.pieces[k][0]
            items = []                                    # (key, blob, factor row offset)
            if cl == 0:                                   # the router's INT8 weights, then its factors
                items.append((("rw",),) + self._chunk_gv(rw, d.wr_f))
            e0, en = sh.cols(c)
            for j, (b0, n) in enumerate(self.chunks(en)):
                cols = self._gu_cols_gv(self.ISH, sh.k0 + e0 + b0, n)
                items.append((("sg", j),) + self._chunk_gv(self._pack_gv(sh_mp["gu_q"][:, cols]), sh_mp["gu_f"][cols]))
            for jl, (o0, w) in enumerate(self.out_slices(c)):
                items.append((("sd", jl),) + self._chunk_gv(self._pack_gv(sh_mp["dn_q"][sh.k0:sh.k0 + sh.n, o0:o0 + w]),
                                                            sh_mp["dn_f"][o0:o0 + w]))
            free[cl] = len(items)
            for pi, pc in enumerate(self.pieces[k][1:], start=1):
                emp = d.expert(pc.e)
                e0, en = pc.cols(c)
                for j, (b0, n) in enumerate(self.chunks(en)):
                    cols = self._gu_cols_gv(self.IE, pc.k0 + e0 + b0, n)
                    items.append((("gu", pi, j),) + self._chunk_gv(self._pack_gv(emp["gu_q"][:, cols]),
                                                                   emp["gu_f"][cols]))
            for jl, (o0, w) in enumerate(self.out_slices(c)):
                for pi, pc in enumerate(self.pieces[k][1:], start=1):
                    emp = d.expert(pc.e)
                    items.append((("dn", jl, pi),) + self._chunk_gv(
                        self._pack_gv(emp["dn_q"][pc.k0:pc.k0 + pc.n, o0:o0 + w]), emp["dn_f"][o0:o0 + w]))
            blobs[cl] = items
        self._images(blobs, free)

    def _stream_report(self):
        """The bytes a token streams: per memory chip, per link (a memory chip's link carries the chips behind it), and
        per chip the route-free chunks, pushed before the route is known (the first prefetcher task, or a cluster ring's
        runs before its boundary), which cover the route's latency."""
        R = self.ROOT
        chip_b = {k: sum(sum(nb for _, nb, _ in self.chunk_off[cl]) for cl in self.clusters(k)) for k in self.chips}
        free_b = {k: sum(sum(nb for _, nb, _ in self.chunk_off[cl][:self.n_free[k]]) for cl in self.clusters(k))
                  for k in self.chips}
        mem_b = {m: sum(b for k, b in chip_b.items() if self.mem[k] == m) for m in sorted(set(self.mem.values()))}
        mem_b[self.mem[R]] += self.XT                     # X's multicast
        link_b = {}
        for k, b in chip_b.items():
            link_b[self.link[k]] = link_b.get(self.link[k], 0) + b
        total = sum(mem_b.values())
        w_only = sum(3 * self.H * pc.n for k in self.chips for pc in self.pieces[k]) // 2
        scales = "factor rows" if self.gemv else "scale tiles"
        print(f"[md] stream: {total:,} B a token ({w_only:,} B of INT4 weights, the rest their {scales}, the "
              f"router's INT8 chunk and X); per memory chip " + ", ".join(f"{m:02x} {b:,}" for m, b in mem_b.items()))
        print("[md] stream per link: " + ", ".join(f"{m:02x}{'+' if s > 0 else '-'} {b:,}"
                                                  for (m, s), b in sorted(link_b.items())))
        print("[md] pushed before the route (route-free chunks a ring, bytes a chip): "
              + ", ".join(f"{k:02x} {self.n_free[k]} {free_b[k]:,}" for k in self.chips))

    def _buffers(self):
        if self.gemv:
            return self._buffers_gv()
        B = {}
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.NSLC * self.SL)
                    for k in self.chips for c in range(self.NC)}
        self.xch_zero = False                             # the shared down GEMM writes each slot whole first
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", self.NSLC * self.SL) for c in range(self.NC)}
        self.yrow = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_yrow", self.H // self.NC * 2)
                     for c in range(self.NC)}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nb)
            pcs = self.pieces[k]
            gt = ROWS // 8 * 2 * self.BPC * D_BLOCK
            b = B[cl] = dict(
                x=l1("x", ROWS * self.PX),
                gbuf=[l1(f"g{i}", gt) for i in range(2)],
                sg=l1("sg", gt),
                a16=[l1(f"a16p{i}", pc.cols(c)[1] // BLK * D_BLOCK + D_BLOCK) for i, pc in enumerate(pcs)],
                a8=[l1(f"a8p{i}", ROWS * mp.row_pitch(pc.n)) for i, pc in enumerate(pcs)],
                dtmp=[l1(f"dtmp{i}", ROWS * self.SW * 2) for i in range(self.ndt)],
                rt=l1("rt", 64),
                lists=l1("lists", SnaxBingoKernelMoeDispatchArgs.list_bytes([ROWS] * len(self.held[k]))))
            if cl == 0:                                   # the root's cluster 0: the router
                b["lg"] = l1("lg", ROWS // 8 * 4 * D_BLOCK)
                b["pin"] = l1("pin", ROWS * 128)
                b["pp"] = l1("pp", ROWS * 128)
            if self.warm_root:
                b["wz"] = l1("wz", 1024)
            if self.scale_out:                            # named to sort after the slabs, as the prefill's
                b["sc"] = [self.cx_chip(k).at(c).l1(f"zy_{self.TAG}_sc{i}", 16 * max(self.SW, 2 * BLK * self.BPC))
                           for i in range(self.n_sc)]
        self.B = B
        self.sc_next = {cl: 0 for cl in range(self.N_CL)}
        self.sc_reader = {cl: [None] * self.n_sc for cl in range(self.N_CL)}

    def _buffers_gv(self):
        """gemv: every activation a row. The exchange buffer and the root's fold buffer one row a slice; the fold rows
        are y's halves (yrow). Per cluster: X (a_row); per piece its G row (64 B a block of the cluster's columns), its
        SiLU, its A16 row and the quantiser's input, and its A8 row (a_row, both clusters' columns); the routed downs'
        row buffers, the route table, the row lists; on the root's cluster 0 the logits' row and the softmax's 8 rows.
        The quantisers' inputs, the A16 rows (padded to whole spread steps) and the softmax's rows share one buffer the
        xDMA zeroes before the window ("rowz"): the spread writes only the atoms, the SiLU'd product only the A16 row's
        own beats and the router's factors only row 0, so the rest stays zero."""
        B = {}
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.NSLC * self.SL)
                    for k in self.chips for c in range(self.NC)}
        self.xch_zero = False                             # the shared down GEMV writes each slot whole first
        self.p1c = {c: self.cx_chip(self.ROOT).at(c).l1(f"{self.TAG}_p1c", self.NSLC * self.SL) for c in range(self.NC)}
        self.yrow = self.p1c                              # slice after slice: the cluster's half of y in column order
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nb)
            pcs = self.pieces[k]
            ens = [pc.cols(c)[1] for pc in pcs]
            steps = [spread_steps(en) for en in ens]
            zb = sum(LANES * D_BLOCK * s + SPREAD_ATOMS * ATOM * s for s in steps) + (ROWS * 128 if cl == 0 else 0)
            z = l1("rowz", zb)
            off, sq, a16 = 0, [], []
            for s in steps:                               # the quantisers' inputs
                sq.append(z.view(off))
                off += LANES * D_BLOCK * s
            for s in steps:                               # the A16 rows
                a16.append(z.view(off))
                off += SPREAD_ATOMS * ATOM * s
            b = B[cl] = dict(
                x=l1("x", self.XT),
                rowz=(z, zb),
                gr=[l1(f"gr{i}", 4 * en) for i, en in enumerate(ens)],
                sgr=[l1(f"sgr{i}", 4 * en) for i, en in enumerate(ens)],
                a16=a16, sq=sq,
                a8=[l1(f"a8p{i}", 2 * pc.n) for i, pc in enumerate(pcs)],
                dtmp=[l1(f"dtmp{i}", self.SL) for i in range(self.ndt)],
                rt=l1("rt", 64),
                lists=l1("lists", SnaxBingoKernelMoeDispatchArgs.list_bytes([ROWS] * len(self.held[k]))))
            if cl == 0:                                   # the root's cluster 0: the router
                b["lg"] = l1("lg", 2 * md.N_EXP)
                b["pin"] = z.view(off)                    # the softmax's 8 rows of 64, 128 B apart
                b["pp"] = l1("pp", ROWS * 128)
            if self.warm_root:
                b["wz"] = l1("wz", 1024)
        self.B = B
        self.sc_next = {cl: 0 for cl in range(self.N_CL)}
        self.sc_reader = {cl: [None] * self.n_sc for cl in range(self.N_CL)}

    # ---- helpers ------------------------------------------------------------------------------------------------------
    def _ld(self, cl, key):
        """The ring load of cluster cl's chunk `key`: loads are created in the image's order, so a ring's entries are
        its image's chunks and the first prefetcher task's range is the image's first chunks."""
        j = self.cidx[cl][key]
        k, c = self.chip(cl), self.local(cl)
        if len(self.rings[k].entries[c]) != j:
            raise RuntimeError(f"cluster {cl}: chunk {key} (image index {j}) loaded as ring entry "
                               f"{len(self.rings[k].entries[c])}")
        return self._load(cl, j)

    def gemv_task(self, k, c, name, a, b, dst, K, N, k_shift, deps, w4=True):
        """One GEMV on chip k's cluster c, a blocking task on its GEMM core: the a_row a [K] times b [K, N] (w4: the
        paired INT4 layout; else INT8, the B layout) into the FP16 row dst [N], at D-port shift k_shift."""
        return self.node(k, c, self.roles["gemm"], name, "__snax_bingo_kernel_gemv",
                         SnaxBingoKernelGemvArgs(a, b, dst, K=K, N=N, d_shift=k_shift, a_blk=8, w4=w4), list(deps))

    def _warm(self, pre):
        """params warm_root: a small GEMM and a scale on every cluster (and the softmax on the root's cluster 0) before
        the start, so the first tasks find their code cached; the root's hold the start stamp. gemv: a small GEMV and
        a stream multiply instead."""
        R, dm, simd = self.ROOT, self.roles["dm"], self.roles["simd"]
        zero = BingoMemSymbol(f"{self.TAG}_zero1k")
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            wz = self.B[cl]["wz"]
            wz_init = self.node(k, c, dm, "wz_init", "__snax_bingo_kernel_idma_1d_copy",
                                SnaxBingoKernelIdma1dCopyArgs(zero, wz, 1024))
            warm = pre[k] if k == R else []
            if self.gemv:                                 # the kernels the GEMV mode runs: a GEMV and a stream multiply
                warm.append(self.gemv_task(k, c, "warm_gemv", wz, wz.view(256), wz.view(512), 4, 2 * BLK, 0, [wz_init]))
                warm.append(self.node(k, c, simd, "warm_scale", "__snax_bingo_kernel_simd_stream_elementwise",
                                      # reads the zeroed head (the GEMV reads it too), not the GEMV's output
                                      SnaxBingoKernelSimdMulF16Args(wz, wz.view(128), wz.view(768), 1, 64),
                                      [wz_init]))
            else:
                warm.append(self.gemm(k, c, "warm_gemm", wz, 32, wz.view(256), wz.view(512), 1, 2, 1, 0, [wz_init],
                                      int4=0 if self.w4_paired else 1))
                warm.append(self.node(k, c, simd, "warm_scale", "__snax_bingo_kernel_simd_mul_tiles",
                                      SnaxBingoKernelSimdMulTilesArgs(wz.view(256), wz.view(256), wz.view(768), 1, 1),
                                      [wz_init]))
            if cl == 0:
                warm.append(self.node(k, c, simd, "warm_softmax", "__snax_bingo_kernel_simd_softmax_f16_f16",
                                      SnaxBingoKernelSimdSoftmaxF16F16Args(wz, wz.view(768), 2, 64), [warm[-1]]))

    def _split_prefetch(self, go, after):
        """Each chip's prefetcher as two host tasks: its rings' first n_free chunks, after the start and `after`; then
        the rest, the original task (the routed chunks), which the caller makes wait for the route (_after_route).
        weight_ring.mode cluster: each ring's first post after `after` too, and its chunk n_free starts a run of its
        own. Returns the first tasks (none in cluster mode)."""
        first = {}
        if self.ring_mode == "cluster":
            for k in self.chips:
                for t in [go[k]] + list(after):
                    self.rings[k].start_after(t)
                for r in range(self.rings[k].n_rings):
                    self.rings[k].boundary(r, self.n_free[k])
            return first
        for k in self.chips:
            t, b = self.prefetch[k], self.prefetch[k].kernel_args
            n = self.n_free[k]
            if not 0 < n <= self.n_slots:
                raise ValueError(f"chip {k:02x}: {n} route-free ring chunks, 1..{self.n_slots} (the ring's slots: the "
                                 f"first task never waits for a release)")
            part = lambda lo, hi, b=b: HostBingoKernelWeightPrefetchRangeArgs(
                b.sched_addr, b.ring_addr, b.flag_addr, b.release_addr, b.seq_addr, b.rec_addr, b.rec_slots, b.n_rings,
                b.n_slots, b.slot_bytes, b.memchip_id, batch=b.batch, policy=b.policy, engine=b.engine,
                batch_routed=b.batch_routed, queued=b.queued, queued_inflight_routed=b.queued_inflight_routed,
                head=b.head, batch_head=b.batch_head, chunk_lo=lo, chunk_hi=hi)
            t.kernel_args = part(n, 0)
            t.kernel_name = t.kernel_args.KERNEL_NAME
            first[k] = self.cx_chip(k).host_node(f"weight_prefetch_k{k:02x}_free", t.kernel_name, part(0, n),
                                                 after=[go[k]] + list(after))
            self.edge(first[k], t)
        return first

    def _after_route(self, disp):
        """The routed chunks are pushed only once the chip has the device's route and its check passed: each chip's
        second prefetcher task after its clusters' moe_dispatch tasks (disp {cluster: task}); weight_ring.mode cluster:
        the post of each ring's run from chunk n_free on."""
        for k in self.chips:
            ds = [disp[cl] for cl in self.clusters(k)]
            if self.ring_mode == "cluster":
                for r in range(self.rings[k].n_rings):
                    self.rings[k].post_after(r, self.n_free[k], ds)
                continue
            for d in ds:
                self.edge(d, self.prefetch[k])

    # ---- the decode -------------------------------------------------------------------------------------------------
    def build(self):
        if self.gemv:
            return self._build_gv()
        d, R, Tg, N_CL = self.d, self.ROOT, self.TAG, self.N_CL
        dm, simd, xdma = self.roles["dm"], self.roles["simd"], self.roles["xdma"]
        if self.p.get("relay_per_gather", True):
            self.dfg.relay_one_consumer_cores = [xdma]
        pre = {k: [] for k in self.chips}
        if self.warm_root:
            self._warm(pre)
        go = self.setup(extra=pre)
        # ---- X from HBM into every chip's L3, then the prefetchers' first tasks ----
        self.x_bcast = self.x_multicast(go)
        self._split_prefetch(go, [self.done_of(self.x_bcast)])
        xc = {}
        for cl in range(N_CL):
            k, c = self.chip(cl), self.local(cl)
            w = self.node(k, c, dm, f"x_wait", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xtflag"), 1, 1), [go[k]])
            self.dfg.bingo_add_order_edge(self.queued_of(self.x_bcast), w)
            xc[cl] = self.node(k, c, dm, f"x_copy", "__snax_bingo_kernel_idma_2d_copy",
                              SnaxBingoKernelIdma2dCopyArgs(self.xland(), self.B[cl]["x"], self.H, self.H, self.PX,
                                                            ROWS),
                              [w])

        # ---- the route on the root's cluster 0, the table to every chip, each cluster's row lists ----
        b0 = self.B[0]
        ld, si, slab, sc = self._ld(0, ("rw",))
        rg = self.gemm(R, 0, "rt_t0", b0["x"], self.PX, slab, b0["lg"], ROWS // 8, self.H // 8, 4, d.k_x, [xc[0], ld],
                       int4=0)
        mt = self.node(R, 0, simd, "rt_deq_t0", "__snax_bingo_kernel_simd_mul_tiles",
                       SnaxBingoKernelSimdMulTilesArgs(b0["lg"], sc.addr, b0["pin"], ROWS // 8, 4, dst_pitch=128),
                       [rg] + sc.deps)
        self._scale_read(0, sc, mt)
        self._release(0, si, rg, mt, sc)
        sm = self.node(R, 0, simd, "rt_softmax_t0", "__snax_bingo_kernel_simd_softmax_f16_f16",
                       SnaxBingoKernelSimdSoftmaxF16F16Args(b0["pin"], b0["pp"], ROWS, 64), [mt])
        tk = self.node(R, 0, xdma, "rt_topk_t0", "__snax_bingo_kernel_moe_topk_rows",
                       SnaxBingoKernelMoeTopkRowsArgs(b0["pp"], 128, 64, md.TOP_K, 0, self.T, b0["rt"]), [sm])
        # one multicast source at a time: X's multicast has landed on the root before the router ran
        rput = self.node(R, 0, dm, "rt_bput", "__snax_bingo_kernel_idma_put_flagged",
                         SnaxBingoKernelIdmaPutFlaggedArgs(b0["rt"], BingoMemSymbol(f"{Tg}_rland"), ROUTE_B,
                                                           BingoMemSymbol(f"{Tg}_rflag"), value=1, bcast=True,
                                                           also_here=True), [tk])
        disp = {}
        for cl in range(N_CL):
            k, c = self.chip(cl), self.local(cl)
            bg = self.B[cl]
            f = self.node(k, c, dm, f"rt_fetch", "__snax_bingo_kernel_idma_fetch_flagged",
                          SnaxBingoKernelIdmaFetchFlaggedArgs(BingoMemSymbol(f"{Tg}_rland"), bg["rt"], ROUTE_B,
                                                              BingoMemSymbol(f"{Tg}_rflag"), 1, 1),
                          [rput] if k == R else [go[k]])
            if k != R:
                self.dfg.bingo_add_order_edge(rput, f)
            n = len(self.held[k])
            disp[cl] = self.node(k, c, xdma, f"rt_dispatch", "__snax_bingo_kernel_moe_dispatch",
                                SnaxBingoKernelMoeDispatchArgs(bg["rt"], self.T, md.TOP_K, self.held[k], bg["lists"],
                                                               [ROWS] * n, [1] * n, pad_row=self.T), [f])
        # the routed chunks are pushed only once the chip has the device's route and its check passed
        self._after_route(disp)

        # ---- gate|up and quantise per piece; the A8 exchange ----
        gb_reader = {cl: [None, None] for cl in range(N_CL)}
        gcount = {cl: 0 for cl in range(N_CL)}
        last_sw = {cl: None for cl in range(N_CL)}
        qa, ex = {}, {}

        def gate_up(cl, pi):
            """Cluster cl's columns of its chip's piece pi: per chunk the GEMM, dequantisation and SwiGLU into the
            piece's A16, then the quantise into the piece's A8 at the cluster's columns."""
            k, c = self.chip(cl), self.local(cl)
            bg, pc = self.B[cl], self.pieces[k][pi]
            e0, en = pc.cols(c)
            nb = en // BLK
            sw = None
            for j, (cb, n) in enumerate(self.chunks(en)):
                ld, si, slab, sc = self._ld(cl, ("sg", j) if pc.e is None else ("gu", pi, j))
                gi = gcount[cl] % 2
                gcount[cl] += 1
                gb = bg["gbuf"][gi]
                name = f"sg_c{j}" if pc.e is None else f"gu_p{pi}c{j}"
                gm = self.gemm(k, c, name, bg["x"], self.PX, slab, gb, ROWS // 8, self.H // 8, 2 * n // BLK, d.k_x,
                               [xc[cl], ld] + ([gb_reader[cl][gi]] if gb_reader[cl][gi] is not None else []))
                sw = self.node(k, c, simd, (f"ssw_c{j}" if pc.e is None else f"esw_p{pi}c{j}"),
                               "__snax_bingo_kernel_simd_deq_swiglu",
                               SnaxBingoKernelSimdDeqSwigluArgs(gb, sc.addr, bg["sg"], bg["a16"][pi].view(cb // BLK * D_BLOCK),
                                                                ROWS // 8, n // BLK, nb * D_BLOCK),
                               [gm] + sc.deps + ([last_sw[cl]] if last_sw[cl] is not None else []))
                gb_reader[cl][gi] = last_sw[cl] = sw
                self._scale_read(cl, sc, sw)
                self._release(cl, si, gm, sw, sc)
            if pc.e is None:
                inv_a, w_addr, deps = d.sh["inv_a"], 0, [sw]
            else:
                s = self.held[k].index(pc.e)
                inv_a = d.expert(pc.e)["inv_a"]
                w_addr = bg["lists"].view(SnaxBingoKernelMoeDispatchArgs.w_offset([ROWS] * len(self.held[k]), s))
                deps = [sw, disp[cl]]
            qa[(cl, pi)] = self.node(k, c, simd, f"q_p{pi}", "__snax_bingo_kernel_simd_quant_rows",
                                    SnaxBingoKernelSimdQuantRowsArgs(
                                        bg["a16"][pi], ROWS, nb, nb * D_BLOCK, inv_a,
                                        bg["a8"][pi].view(self.beat_aligned(e0, f"q_p{pi}")), mp.row_pitch(pc.n),
                                        w_addr=w_addr), deps)

        def exchange(cl, pi):
            k, c = self.chip(cl), self.local(cl)
            pc = self.pieces[k][pi]
            e0, en = pc.cols(c)
            other = self.clusters(k)[1 - c]
            pitch = mp.row_pitch(pc.n)
            ex[(cl, pi)] = self.node(k, c, dm, f"a8_copy_p{pi}", "__snax_bingo_kernel_idma_2d_copy",
                                    SnaxBingoKernelIdma2dCopyArgs(self.B[cl]["a8"][pi].view(e0),
                                                                  self.B[other]["a8"][pi].view(e0),
                                                                  en, pitch, pitch, ROWS), [qa[(cl, pi)]])

        # the shared MLP: gate|up, quantise, exchange, then its down projection of every output slice
        for cl in range(N_CL):
            gate_up(cl, 0)
        for cl in range(N_CL):
            exchange(cl, 0)
        nt = self.SW // BLK
        xch_add = {}                                      # (cl, slice) -> the last task that wrote the slice
        for cl in range(N_CL):
            k, c = self.chip(cl), self.local(cl)
            other = self.clusters(k)[1 - c]
            sh = self.pieces[k][0]
            for jl in range(self.NSLC):
                ld, si, slab, sc = self._ld(cl, ("sd", jl))
                y = self.xch[(k, c)].view(jl * self.SL)
                gm = self.gemm(k, c, f"sd_j{jl}", self.B[cl]["a8"][0], mp.row_pitch(sh.n), slab, y, ROWS // 8,
                               sh.n // 8, nt, d.k_dn_sh, [ld, qa[(cl, 0)], ex[(other, 0)]])
                dq = self.node(k, c, simd, f"sd_deq_j{jl}", "__snax_bingo_kernel_simd_mul_tiles",
                               SnaxBingoKernelSimdMulTilesArgs(y, sc.addr, y, ROWS // 8, nt), [gm] + sc.deps)
                self._scale_read(cl, sc, dq)
                self._release(cl, si, gm, dq, sc)
                xch_add[(cl, jl)] = dq
        # the routed pieces: each piece's exchange behind the next piece's loads in the DM core's program
        npc = len(self.pieces[self.chips[0]])
        for pi in range(1, npc):
            for cl in range(N_CL):
                gate_up(cl, pi)
                if pi > 1:
                    exchange(cl, pi - 1)
        for cl in range(N_CL):
            exchange(cl, npc - 1)

        # ---- the routed pieces' down projection by output slice; one ChainGather a slice ----
        dt_reader = {cl: [None] * self.ndt for cl in range(N_CL)}
        ndt_used = {cl: 0 for cl in range(N_CL)}
        last_ld = {}
        for jl in range(self.NSLC):
            for cl in range(N_CL):
                k, c = self.chip(cl), self.local(cl)
                bg = self.B[cl]
                other = self.clusters(k)[1 - c]
                y = self.xch[(k, c)].view(jl * self.SL)
                caps = [ROWS] * len(self.held[k])
                prev = xch_add[(cl, jl)]
                for pi in range(1, npc):
                    pc = self.pieces[k][pi]
                    s = self.held[k].index(pc.e)
                    ld, si, slab, sc = self._ld(cl, ("dn", jl, pi))
                    last_ld[cl] = ld
                    di = ndt_used[cl] % self.ndt
                    ndt_used[cl] += 1
                    dt = bg["dtmp"][di]
                    gm = self.gemm(k, c, f"dn_j{jl}s{pi}", bg["a8"][pi], mp.row_pitch(pc.n), slab, dt, ROWS // 8,
                                   pc.n // 8, nt, d.k_dn_e,
                                   [ld, qa[(cl, pi)],
                                    ex[(other, pi)]] + ([dt_reader[cl][di]] if dt_reader[cl][di] else []))
                    mt = self.node(k, c, simd, f"dn_deq_j{jl}s{pi}", "__snax_bingo_kernel_simd_mul_tiles",
                                   SnaxBingoKernelSimdMulTilesArgs(dt, sc.addr, dt, ROWS // 8, nt), [gm] + sc.deps)
                    self._scale_read(cl, sc, mt)
                    self._release(cl, si, gm, mt, sc)
                    ids = bg["lists"].view(SnaxBingoKernelMoeDispatchArgs.rows_offset(caps, s))
                    prev = self.node(k, c, simd, f"scat_j{jl}s{pi}", "__snax_bingo_kernel_simd_scatter_add_rows",
                                     SnaxBingoKernelSimdScatterAddRowsArgs(y, dt, nt, ids, 1), [mt, prev])
                    dt_reader[cl][di] = prev
                xch_add[(cl, jl)] = prev
        if self.chain_tail:                               # every push has landed before a last-slice chain
            for cl in range(N_CL):
                for o in self.clusters(self.chip(cl)):
                    if o != cl:
                        self.edge(last_ld[o], xch_add[(cl, self.NSLC - 1)])
        copies_in = {c: [t for (cl, _), t in ex.items() if self.local(cl) != c] for c in range(self.NC)}
        last_red, last_any = {c: None for c in range(self.NC)}, None
        for jl in range(self.NSLC):
            tail = jl >= self.NSLC - self.chain_tail
            corder = [1, 0] if (self.chain_first_alt and jl % 2) else [0, 1]
            for c in corder:
                p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, self.chain_order(c)[:-1], src=self.xch_addr(c, jl * self.SL),
                    dst=f"({p1x} + {jl * self.SL})", size=self.SL, junction=self.jct["Elementwise"],
                    jct_csr0=mp.EW_F16_ADD, holds=[self.p1c[c]])
                before = last_red[c] if tail else last_any
                red = self.node(R, c, xdma, f"y_chain_c{c}j{jl}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                [xch_add[(cl, jl)] for cl in range(N_CL) if self.local(cl) == c]
                                + (copies_in[c] if jl == 0 else []) + ([before] if before is not None else []))
                last_red[c] = last_any = red

        # ---- y's row to HBM: row 0 of each root cluster's reduced slices, once no chain is in flight ----
        hc = self.H // self.NC
        rows = {c: self.node(R, c, dm, f"y_row_c{c}", "__snax_bingo_kernel_idma_2d_copy",
                             SnaxBingoKernelIdma2dCopyArgs(self.p1c[c], self.yrow[c], SIMD_BEAT, D_BLOCK, SIMD_BEAT,
                                                           hc // BLK), [last_red[c]])
                for c in range(self.NC)}
        self.end_window(self.y_out(rows, last_red))
        self._checks()

    # ---- gemv: the decode -------------------------------------------------------------------------------------------
    def _build_gv(self):
        """The decode with every weight multiply on the GEMV (the module docstring's GEMV MODE). The tasks are made in
        the tile path's order, so every core's program is the tile path's with the GEMV's tasks in place of its."""
        d, R, Tg, N_CL = self.d, self.ROOT, self.TAG, self.N_CL
        dm, simd, xdma = self.roles["dm"], self.roles["simd"], self.roles["xdma"]
        first = max((t.node_id for t in self.dfg.node_list), default=-1) + 1
        if self.p.get("relay_per_gather", True):
            self.dfg.relay_one_consumer_cores = [xdma]
        pre = {k: [] for k in self.chips}
        if self.warm_root:
            self._warm(pre)
        # the zeroed rows (the quantisers' inputs, the A16 rows' padding, the softmax's rows 1 to 7) before the window:
        # a chip's two clusters one after the other, the start behind the second (one dependency a chip)
        for k in self.chips:
            prev = []
            for cl in self.clusters(k):
                z, zb = self.B[cl]["rowz"]
                zero = SnaxBingoKernelXdmaMemsetArgs(z, zb, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO)
                prev = [self.node(k, self.local(cl), xdma, f"row_zero", "__snax_bingo_kernel_xdma_memset", zero, prev)]
            pre[k] += prev
        go = self.setup(extra=pre)
        # ---- X from HBM into every chip's L3, then the prefetchers' first tasks ----
        self.x_bcast = self.x_multicast(go)
        self._split_prefetch(go, [self.done_of(self.x_bcast)])
        xc = {}
        for cl in range(N_CL):
            k, c = self.chip(cl), self.local(cl)
            w = self.node(k, c, dm, f"x_wait", "__snax_bingo_kernel_wait_flags",
                          SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{Tg}_xtflag"), 1, 1), [go[k]])
            self.dfg.bingo_add_order_edge(self.queued_of(self.x_bcast), w)
            xc[cl] = self.node(k, c, dm, f"x_copy", "__snax_bingo_kernel_idma_1d_copy",
                              SnaxBingoKernelIdma1dCopyArgs(self.xland(), self.B[cl]["x"], self.XT), [w])

        # ---- the route on the root's cluster 0 (an INT8 GEMV), the table to every chip, each cluster's row lists ----
        b0 = self.B[0]
        ld, si, slab, sc = self._ld(0, ("rw",))
        rg = self.gemv_task(R, 0, "rt_t0", b0["x"], slab, b0["lg"], self.H, md.N_EXP, d.k_x, [xc[0], ld], w4=False)
        mt = self.node(R, 0, simd, "rt_deq_t0", "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdMulF16Args(b0["lg"], sc.addr, b0["pin"], 1, md.N_EXP), [rg] + sc.deps)
        self._scale_read(0, sc, mt)
        self._release(0, si, rg, mt, sc)
        sm = self.node(R, 0, simd, "rt_softmax_t0", "__snax_bingo_kernel_simd_softmax_f16_f16",
                       SnaxBingoKernelSimdSoftmaxF16F16Args(b0["pin"], b0["pp"], ROWS, 64), [mt])
        tk = self.node(R, 0, xdma, "rt_topk_t0", "__snax_bingo_kernel_moe_topk_rows",
                       SnaxBingoKernelMoeTopkRowsArgs(b0["pp"], 128, 64, md.TOP_K, 0, self.T, b0["rt"]), [sm])
        # one multicast source at a time: X's multicast has landed on the root before the router ran
        rput = self.node(R, 0, dm, "rt_bput", "__snax_bingo_kernel_idma_put_flagged",
                         SnaxBingoKernelIdmaPutFlaggedArgs(b0["rt"], BingoMemSymbol(f"{Tg}_rland"), ROUTE_B,
                                                           BingoMemSymbol(f"{Tg}_rflag"), value=1, bcast=True,
                                                           also_here=True), [tk])
        disp = {}
        for cl in range(N_CL):
            k, c = self.chip(cl), self.local(cl)
            bg = self.B[cl]
            f = self.node(k, c, dm, f"rt_fetch", "__snax_bingo_kernel_idma_fetch_flagged",
                          SnaxBingoKernelIdmaFetchFlaggedArgs(BingoMemSymbol(f"{Tg}_rland"), bg["rt"], ROUTE_B,
                                                              BingoMemSymbol(f"{Tg}_rflag"), 1, 1),
                          [rput] if k == R else [go[k]])
            if k != R:
                self.dfg.bingo_add_order_edge(rput, f)
            n = len(self.held[k])
            disp[cl] = self.node(k, c, xdma, f"rt_dispatch", "__snax_bingo_kernel_moe_dispatch",
                                SnaxBingoKernelMoeDispatchArgs(bg["rt"], self.T, md.TOP_K, self.held[k], bg["lists"],
                                                               [ROWS] * n, [1] * n, pad_row=self.T), [f])
        # the routed chunks are pushed only once the chip has the device's route and its check passed
        self._after_route(disp)

        # ---- gate|up and quantise per piece; the A8 exchange ----
        qa, ex = {}, {}

        def gate_up(cl, pi):
            """Cluster cl's columns of its chip's piece pi: per chunk the GEMV into the piece's G row and its factors;
            then SiLU and the gate-times-up into the A16 row, the spread and the quantise into the piece's A8 row."""
            k, c = self.chip(cl), self.local(cl)
            bg, pc = self.B[cl], self.pieces[k][pi]
            e0, en = pc.cols(c)
            nb = en // BLK
            gr = bg["gr"][pi]
            dq = None
            for j, (cb, n) in enumerate(self.chunks(en)):
                ld, si, slab, sc = self._ld(cl, ("sg", j) if pc.e is None else ("gu", pi, j))
                row = gr.view(4 * cb)                     # 64 B a block: its gate values, then its up values
                name = f"sg_c{j}" if pc.e is None else f"gu_p{pi}c{j}"
                gm = self.gemv_task(k, c, name, bg["x"], slab, row, self.H, 2 * n, d.k_x, [xc[cl], ld])
                dq = self.node(k, c, simd, (f"sdq_c{j}" if pc.e is None else f"edq_p{pi}c{j}"),
                               "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdMulF16Args(row, sc.addr, row, 1, 2 * n),
                               [gm] + sc.deps + ([dq] if dq is not None else []))   # in turn: the last stands for all
                self._scale_read(cl, sc, dq)
                self._release(cl, si, gm, dq, sc)
            sl = self.node(k, c, simd, f"silu_p{pi}", "__snax_bingo_kernel_simd_silu_f16_f16",
                           SnaxBingoKernelSimdSiluF16F16Args(gr, bg["sgr"][pi], 1, 2 * en), [dq])
            sw = self.node(k, c, simd, f"swi_p{pi}", "__snax_bingo_kernel_simd_stream_elementwise",
                           SimdMulRowsF16Args(bg["sgr"][pi], gr.view(2 * BLK), bg["a16"][pi], nb, BLK, 4 * BLK), [sl])
            sp = self.node(k, c, xdma, f"a2q_p{pi}", "__snax_bingo_kernel_xdma_6d",
                           spread_args(bg["a16"][pi], bg["sq"][pi], en), [sw])
            if pc.e is None:
                inv_a, w_addr, deps = d.sh["inv_a"], 0, [sp]
            else:
                s = self.held[k].index(pc.e)
                inv_a = d.expert(pc.e)["inv_a"]
                w_addr = bg["lists"].view(SnaxBingoKernelMoeDispatchArgs.w_offset([ROWS] * len(self.held[k]), s))
                deps = [sp, disp[cl]]
            # one row: its en / 8 beats 256 B apart, two atoms each, into the cluster's run of the A8 row (2 B a column)
            qa[(cl, pi)] = self.node(k, c, simd, f"q_p{pi}", "__snax_bingo_kernel_simd_quant_rows",
                                    SnaxBingoKernelSimdQuantRowsArgs(
                                        bg["sq"][pi], 1, en // 8, D_BLOCK, inv_a,
                                        bg["a8"][pi].view(self.beat_aligned(2 * e0, f"q_p{pi}")), 2 * en,
                                        w_addr=w_addr), deps)

        def exchange(cl, pi):
            k, c = self.chip(cl), self.local(cl)
            pc = self.pieces[k][pi]
            e0, en = pc.cols(c)
            other = self.clusters(k)[1 - c]
            ex[(cl, pi)] = self.node(k, c, dm, f"a8_copy_p{pi}", "__snax_bingo_kernel_idma_1d_copy",
                                    SnaxBingoKernelIdma1dCopyArgs(self.B[cl]["a8"][pi].view(2 * e0),
                                                                  self.B[other]["a8"][pi].view(2 * e0), 2 * en),
                                    [qa[(cl, pi)]])

        # the shared MLP: gate|up, quantise, exchange, then its down projection of every output slice
        for cl in range(N_CL):
            gate_up(cl, 0)
        for cl in range(N_CL):
            exchange(cl, 0)
        xch_add = {}                                      # (cl, slice) -> the last task that wrote the slice
        for cl in range(N_CL):
            k, c = self.chip(cl), self.local(cl)
            other = self.clusters(k)[1 - c]
            sh = self.pieces[k][0]
            for jl in range(self.NSLC):
                ld, si, slab, sc = self._ld(cl, ("sd", jl))
                y = self.xch[(k, c)].view(jl * self.SL)
                gm = self.gemv_task(k, c, f"sd_j{jl}", self.B[cl]["a8"][0], slab, y, sh.n, self.SW, d.k_dn_sh,
                                    [ld, qa[(cl, 0)], ex[(other, 0)]])
                dq = self.node(k, c, simd, f"sd_deq_j{jl}", "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdMulF16Args(y, sc.addr, y, 1, self.SW), [gm] + sc.deps)
                self._scale_read(cl, sc, dq)
                self._release(cl, si, gm, dq, sc)
                xch_add[(cl, jl)] = dq
        # the routed pieces: each piece's exchange behind the next piece's loads in the DM core's program
        npc = len(self.pieces[self.chips[0]])
        for pi in range(1, npc):
            for cl in range(N_CL):
                gate_up(cl, pi)
                if pi > 1:
                    exchange(cl, pi - 1)
        for cl in range(N_CL):
            exchange(cl, npc - 1)

        # ---- the routed pieces' down projection by output slice; one ChainGather a slice ----
        dt_reader = {cl: [None] * self.ndt for cl in range(N_CL)}
        ndt_used = {cl: 0 for cl in range(N_CL)}
        last_ld = {}
        for jl in range(self.NSLC):
            for cl in range(N_CL):
                k, c = self.chip(cl), self.local(cl)
                bg = self.B[cl]
                other = self.clusters(k)[1 - c]
                y = self.xch[(k, c)].view(jl * self.SL)
                prev = xch_add[(cl, jl)]
                for pi in range(1, npc):
                    pc = self.pieces[k][pi]
                    ld, si, slab, sc = self._ld(cl, ("dn", jl, pi))
                    last_ld[cl] = ld
                    di = ndt_used[cl] % self.ndt
                    ndt_used[cl] += 1
                    dt = bg["dtmp"][di]
                    gm = self.gemv_task(k, c, f"dn_j{jl}s{pi}", bg["a8"][pi], slab, dt, pc.n, self.SW, d.k_dn_e,
                                        [ld, qa[(cl, pi)], ex[(other, pi)]]
                                        + ([dt_reader[cl][di]] if dt_reader[cl][di] else []))
                    mt = self.node(k, c, simd, f"dn_deq_j{jl}s{pi}", "__snax_bingo_kernel_simd_stream_elementwise",
                                   SnaxBingoKernelSimdMulF16Args(dt, sc.addr, dt, 1, self.SW), [gm] + sc.deps)
                    self._scale_read(cl, sc, mt)
                    self._release(cl, si, gm, mt, sc)
                    prev = self.node(k, c, simd, f"dn_add_j{jl}s{pi}",
                                     "__snax_bingo_kernel_simd_stream_elementwise",
                                     SnaxBingoKernelSimdAddF16Args(y, dt, y, 1, self.SW), [mt, prev])
                    dt_reader[cl][di] = prev
                xch_add[(cl, jl)] = prev
        if self.chain_tail:                               # every push has landed before a last-slice chain
            for cl in range(N_CL):
                for o in self.clusters(self.chip(cl)):
                    if o != cl:
                        self.edge(last_ld[o], xch_add[(cl, self.NSLC - 1)])
        copies_in = {c: [t for (cl, _), t in ex.items() if self.local(cl) != c] for c in range(self.NC)}
        last_red, last_any = {c: None for c in range(self.NC)}, None
        for jl in range(self.NSLC):
            tail = jl >= self.NSLC - self.chain_tail
            corder = [1, 0] if (self.chain_first_alt and jl % 2) else [0, 1]
            for c in corder:
                p1x = f"((uint64_t)({self.p1c[c].get_c_var_name()}))"
                ga = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                    cm.remote, R, c, self.chain_order(c)[:-1], src=self.xch_addr(c, jl * self.SL),
                    dst=f"({p1x} + {jl * self.SL})", size=self.SL, junction=self.jct["Elementwise"],
                    jct_csr0=mp.EW_F16_ADD, holds=[self.p1c[c]])
                before = last_red[c] if tail else last_any
                red = self.node(R, c, xdma, f"y_chain_c{c}j{jl}", "__snax_bingo_kernel_xdma_chain_gather", ga,
                                [xch_add[(cl, jl)] for cl in range(N_CL) if self.local(cl) == c]
                                + (copies_in[c] if jl == 0 else []) + ([before] if before is not None else []))
                last_red[c] = last_any = red

        # ---- y's row to HBM: each root cluster's fold row (yrow) once no chain is in flight ----
        self.end_window(self.y_out(dict(last_red), last_red))
        self._check_gemv_only(first)
        self._checks()

    def _check_gemv_only(self, first):
        """gemv: every array task this block made (node ids from `first`) is a GEMV, so no gemm_full follows a GEMV on a
        GEMM core."""
        gemm = self.roles["gemm"]
        arr = [t for t in self.dfg.node_list if t.node_id >= first and t.assigned_core_id == gemm
               and (t.kernel_name or "").startswith("__snax")]
        bad = [t.node_name for t in arr if t.kernel_name != "__snax_bingo_kernel_gemv"]
        if bad:
            raise ValueError(f"params gemv: array tasks that are not GEMVs: {bad[:4]}")
        print(f"[{self.TAG}] gemv: {len(arr)} array tasks, every one a GEMV (the router INT8, gate|up and down INT4); "
              f"X a_row ({self.XT} B), factor rows, one row a slice ({self.SL} B a chain)")

    def x_multicast(self, go):
        """X into every chip's L3 behind one flag (md_xland, md_xtflag): the root's host queues its memory chip's push
        engine, which multicasts the token's row from HBM (ChipletBuild.hbm_cast). Returns the HbmCast the X waits and
        the first prefetcher tasks follow. A whole layer (dsv2/common/dsv2_decode_mc.py) multicasts its glue's row here
        instead."""
        R, m = self.ROOT, self.mem[self.ROOT]
        one = self.h_seq[m].address + 64                  # a word holding 1 in the memory chip's HBM: the flag's value
        return self.hbm_cast(
            R, f"x_hbm_bcast_k{R:02x}",
            HostBingoKernelHbmBcastArgs(self.h_xhbm.address, self.xland(), BingoMemSymbol(f"{self.TAG}_xtflag"), one, 1,
                                        self.XT, self.NXT, m, self.engine[R]), after=[go[R]])

    def y_out(self, rows, last_red):
        """Each root cluster writes its half of y's row (yrow, made by rows[c]) into the HBM of the root's memory chip once
        both root clusters' last chains (last_red) are done. Returns the tasks the end follows. A whole layer
        (dsv2/common/dsv2_decode_mc.py) adds its residual here instead."""
        R, dm, hc = self.ROOT, self.roles["dm"], self.H // self.NC
        return [self.node(R, c, dm, f"y_hbm_c{c}", "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(self.yrow[c], at_offset(self.h_yhbm, 2 * hc * c), 2 * hc),
                          [rows[c]] + list(last_red.values()))
                for c in range(self.NC)]

    # ---- checks -------------------------------------------------------------------------------------------------------
    def _checks(self):
        R, Tg = self.ROOT, self.TAG
        checks = [("route", BingoMemSymbol(f"{Tg}_rland", chip_id=R), self._route_bytes(), ROUTE_B)]
        lb = self._lists_bytes(R)
        checks.append(("lists", self.stash("lists_root_c1", R, 1, self.B[1]["lists"], lb.size, []), lb, lb.size))
        # row 0 of the root cluster 0's A8 of its shared piece and of its first routed piece: both clusters' columns,
        # cluster 1's after the exchange
        for pi, name in ((0, "a8sh"), (1, "a8_p1")):
            gold = self.a8_gold(R, pi)
            checks.append((name, self.stash(f"{name}_root", R, 0, self.B[0]["a8"][pi], gold.size, []), gold, gold.size))
        checks.append(("y_hbm", self.h_yhbm, self.y_row.view(np.uint8), 2 * self.H))
        self.run_checks(checks)

    def a8_gold(self, k, pi):
        """The bytes of the token's A8 row of chip k's piece pi as its clusters hold it (both clusters' columns): row 0
        of the row-major A8; gemv: the A8 row in a_row (2 B a column, the free bytes zero)."""
        pc = self.pieces[k][pi]
        a8 = self.d.a8_shared() if pc.e is None else self.d.a8_expert(pc.e)
        row = np.ascontiguousarray(a8[0, pc.k0:pc.k0 + pc.n]).astype(np.int8)
        return (a_row_bytes(row) if self.gemv else row).view(np.uint8)


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer-1 MoE decode, one token, {len(S.chips)} chips x {S.NC} clusters, routed experts "
            f"over {S.routed_split} chips each, X from HBM, y to HBM"
            + (", every weight multiply on the GEMV" if S.gemv else ""))


def run():
    run_driver("moe decode workload generator", DBuild, _app_name)
