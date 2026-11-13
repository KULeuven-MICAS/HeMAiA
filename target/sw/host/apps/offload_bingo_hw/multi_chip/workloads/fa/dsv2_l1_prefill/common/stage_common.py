# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""What every DSv2 FA stage app (../stage<N>/main_bingo.py) is built on: StageBase, a ChipletBuild with the block's
data, memory, weight ring, task helpers and x's multicast from HBM. Each stage app adds its own staging, its
preloads, its tasks, its checks and its build().

DeepSeek-V2-Lite's layer-1 MLA attention over a fresh 256-token prompt, head parallel on the 16 + 4 chiplet platform:
chip k computes head h(k); its two clusters take the two halves of each 128-token chunk. A stage app stages every
input of the whole block (the same image as the block's), so a stage's window runs on the memory the block would
leave it, its own inputs (the stage before's result) added in HBM and copied into place before the window.

A stage app's class sets:
    STAGE          its name in the app name (0, 1, 2, 3, "0123", 4, 5, "678")
    X_HBM          x multicast inside the window from the HBM of the root's memory chip (stages 0 - 3); else x is
                   already in every chip's L3 (x_resident)
    X_XDMA         the key stages' loads of x split between the iDMA and the xDMA (X_XDMA_SHARE of each tile)
    KVA_IN_RING    the chip's W_KVA slice is the ring's first chunk (stages 0 - 3)
    PER_CHUNK      ring chunks a token chunk takes (W_q x 3, W_UK, W_UV, W_O x 2)
    ROW_SWAP       in every odd chunk the clusters take the other halves (the data and the RoPE tables follow it)
    SKIP_MASKED    no key tile whose every score is masked (the data's tile plan)
    KV_BALANCE_WAYS  the INT4 W_KVA balance (kv_helpers): None, or the runs the tiles are cut into"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(__file__).split("/target/sw/host/")[0]
sys.path.insert(0, os.path.join(REPO, "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(REPO, "util", "sim", "common"))
sys.path.insert(0, os.path.join(REPO, "target", "sw", "host", "runtime", "libbingo", "mini_compiler"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))          # fa/common: fa_prefill_data
import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    INT4_B_LAYOUT_PAIRED, HostBingoKernelHbmBcastArgs, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdmaRingLoadArgs, SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args,
    SnaxBingoKernelSimdMulF16Args, SnaxBingoKernelSimdMulI8Args, SnaxBingoKernelWaitFlagsArgs, simd_beats,
    simd_fp16_lanes)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs.blocks import FaSweepCfg, WeightRings, fa_sweep_alloc  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import fa_prefill_data  # noqa: E402

BR = 16                       # query rows a tile: the SIMD's FP16 lanes
D_QK, D_V, KV_RANK, ROPE, Q_NOPE, Q_HEAD, HIDDEN = 576, 512, 512, 64, 128, 192, 2048
TILE16 = 16 * 16 * 2          # one D-layout FP16 block
EW_F16_ADD = cm.EW_F16_ADD
KB, VB = 16 * D_QK, D_V * 16  # a 16-token piece of the keys: its key rows, its value columns (INT8)
KVP = KB + VB
PG = 2                        # granules an INT4 pair of W_KVA's columns
X_XDMA_SHARE = 0.3            # the key stages: the xDMA's share of each tile of x into L1, the iDMA the rest
KV_CACHE = os.path.join(REPO, "target/sim/automation/sweep/llm_blocks/fa/cache/dsv2_l1_kv_8193.npz")
WEIGHTS = os.path.join(REPO, "target/sim/automation/sweep/common/weights/cache")
N_CALIB = 32                  # calibration tokens


def check(p):
    """The mapping's arithmetic; raises ValueError where the parameters cannot be laid out. The keys are the prompt's
    own tokens (a fresh prompt)."""
    tokens, chunk, bc = (int(p[k]) for k in ("tokens", "chunk", "bc"))
    rows = chunk // 2
    if tokens % chunk or chunk % 2 or rows % BR:
        raise ValueError(f"tokens {tokens} / chunk {chunk}: a cluster takes chunk / 2 rows, a whole number of "
                         f"{BR}-row query tiles")
    if tokens % bc or bc % 16 or tokens // bc < 4:
        raise ValueError(f"tokens {tokens}: not whole key tiles of bc {bc} (a multiple of 16), at least four")
    if chunk % bc:
        raise ValueError(f"chunk {chunk}: whole key tiles of bc {bc} (a chunk's two row blocks sweep every tile up to "
                         f"its end)")
    wq_block = HIDDEN * 64 * int(p.get("wbits", 8)) // 8
    if int(p["w_chunk_bytes"]) < max(bc * D_QK, D_V * bc, wq_block):
        raise ValueError(f"w_chunk_bytes {p['w_chunk_bytes']}: a K tile is {bc * D_QK} B, a W_q block "
                         f"{wq_block} B")
    return dict(chunks=tokens // chunk, query_tiles=rows // BR, tiles=tokens // bc, keys=tokens)


class StageBase(cm.ChipletBuild):
    """The block's data, memory, ring and task helpers for a stage app (the module docstring), in ChipletBuild's
    four steps: __init__ (the parameters), mem_handles() (the data, the memory: the block's image and the stage's
    inputs, stage_inputs()), build() (the app's) and finish()."""
    TAG = "fp"
    x_put_cluster = 1             # the root cluster whose DM core multicasts the keys' pieces (stage 3)
    PER_CHUNK_HEAD, PER_CHUNK_TAIL = 4, 3          # W_q x 3 and W_UK; W_UV and W_O x 2

    STAGE = None
    X_HBM = True
    X_XDMA = False
    KVA_IN_RING = 1
    PER_CHUNK = PER_CHUNK_HEAD + PER_CHUNK_TAIL
    ROW_SWAP = False
    SKIP_MASKED = False
    KV_BALANCE_WAYS = None

    def __init__(self, args, p, plat, hw, shared=None):
        """ChipletBuild.__init__ first: the platform's and the graph's switches, the chips, each one's memory chip and
        its push engine there, the empty DFG, the data staging (self.st) and the stamp buffer. Then the block's
        shapes (check), the head of each chip and the fold order, and the sizes of a tile and a chunk of x and of a
        cluster's partial y."""
        self.x_hbm = self.X_HBM
        self.x_resident = not self.x_hbm
        super().__init__(args, p, plat, hw, shared=shared)
        if (self.bcast_impl, self.reduce_impl, self.sync) != ("bcast_l1", "chain", "edge"):
            raise ValueError("the FA prefill builds bcast bcast_l1, reduce chain, sync edge")
        m = check(p)
        self.NCH, self.NQ, self.NT = m["chunks"], m["query_tiles"], m["tiles"]
        self.C = int(p["chunk"])
        self.ROWS = self.C // self.NC                  # query rows a cluster
        self.keys, self.bc = m["keys"], int(p["bc"])
        self.row_swap = self.ROW_SWAP
        # chip chips[q % 16] makes the keys of tokens 16 q .. 16 q + 15, T / 256 tiles a chip 16 tiles apart; kv_put
        # {tile: the put of its piece into the root's L3}
        self.kv_put = {}
        nch = len(self.chips)
        if self.T % (16 * nch):
            raise ValueError(f"{self.T} tokens; each of the {nch} chips makes whole 16-token tiles")
        self.kv_tiles = {k: [q for q in range(self.T // 16) if self.chips[q % nch] == k] for k in self.chips}
        # params wbits (8 | 4): the projection weights INT8 or INT4 in the paired layout
        self.wbits = int(p.get("wbits", 8))
        if self.wbits not in (8, 4):
            raise ValueError(f"params wbits={self.wbits}: 8 or 4")
        # W_KVA by columns: chip j streams only its 16-column granules [36 j / 16, 36 (j + 1) / 16) of W_KVA's 36 (INT4
        # streams whole pairs of granules: the paired layout's blocks of 32 columns)
        pg = 2 if self.wbits == 4 else 1
        ng = D_QK // 16 // pg
        self.kvg = {k: (pg * (j * ng // nch), pg * ((j + 1) * ng // nch)) for j, k in enumerate(self.chips)}
        self.kv_owner = {q: k for k in self.chips for q in self.kv_tiles[k]}
        self.KVD = 16 * D_QK * 2                     # a tile's kv, D layout [16, 576] FP16
        self.KVR_TILES = max(len(t) for t in self.kv_tiles.values())
        if simd_fp16_lanes() != BR:
            raise ValueError(f"the SIMD has {simd_fp16_lanes()} FP16 lanes; the softmax takes one query "
                             f"a lane and the tiles are {BR} rows")
        if int(p.get("w_buffers", 3)) != 3:
            raise ValueError("params w_buffers: 3 (the sweep's K tiles in slab 0, its V tiles in slabs 1 and 2)")
        self.stop = "full"
        # head h(k): a memory chip's row holds four consecutive heads, nearest the memory chip last
        by_row = sorted(self.chips, key=lambda k: (k & 15, k >> 4))
        self.head = {k: i for i, k in enumerate(by_row)}
        if len(self.head) != 16:
            raise ValueError(f"{len(self.head)} compute chips: the layer has 16 heads, one a chip")
        # the fold order of the 16 partials: the chips' order from the base class, as heads
        self.fold_heads = [self.head[k] for k in self.fold_order()]
        self.XB = self.C * HIDDEN                       # a chunk's x, INT8
        self.YB = self.ROWS * HIDDEN * 2                # a cluster's partial y of a chunk, FP16
        self.XT = 16 * HIDDEN                          # a tile of x: 16 tokens, A layout, INT8
        self.NX = self.T // 16
        self.x_xdma = int(round(self.XT * X_XDMA_SHARE / 512)) * 512 if self.X_XDMA else 0
        self._first_ld = set()
        self._key_runs = []

    # ---- data and memory ----------------------------------------------------------------------------------------
    def mem_handles(self):
        """The data and the memory; no task yet: the data and its golden (fa_prefill_data.FaPrefillData) and their
        report; _stage (every input staged in L3 and HBM, then the stage's own); the load streams' L1 slabs; one
        weight ring a chip; the L1 buffers."""
        d = self.d = fa_prefill_data.FaPrefillData(KV_CACHE, WEIGHTS, tokens=self.T, keys=self.keys, bc=self.bc,
                                                   n_calib=N_CALIB, fresh=True, chunk=self.C, rope_perm=True,
                                                   mask_c=True, wbits=self.wbits, skip_masked=self.SKIP_MASKED)
        self.lay = d.layout
        for line in d.report(self.fold_heads):
            print(f"[fp] {line}")
        print(f"[fp] {self.T} tokens in {self.NCH} chunks of {self.C}: {self.NQ} query tiles of {BR} rows a "
              f"cluster, {self.NT} K/V tiles of {self.bc} keys; fold order "
              f"{[f'{k:02x}' for k in self.order]}")
        self._stage()
        self.make_streams()
        self._make_rings()
        self._buffers()

    def flag_dep(self, t, k, c):
        """Whether task t enters a flag wait on chip k cluster c as a dependency (a task of that chip), else as an
        ordering edge (the flag proves the landing)."""
        return t.assigned_chiplet_id == k

    def xh(self, i, c):
        """The half of chunk i's rows that cluster c takes."""
        return c ^ (i & 1) if self.row_swap else c

    def r0(self, i, c):
        """The first row of chunk i that cluster c takes."""
        return i * self.C + self.xh(i, c) * self.ROWS

    def _stage(self):
        """The block's image (_stage_block), the INT4 balance's helpers (kv_helpers), the stage's own inputs
        (stage_inputs), then the helpers' slices' flags."""
        self._stage_block()
        self.kv_helpers()
        self.stage_inputs()
        if self.kvx:
            # the helpers' slices' flags: per owned tile, one a two-pair chip
            self.st.put_zeros(f"{self.TAG}_kvxflag", "uint8_t", self.KVR_TILES * len(self.kv_two) * 64)

    def stage_inputs(self):
        """The stage's own inputs in HBM (the stage before's result), staged after the block's: none here."""

    def _stage_block(self):
        st, d, T = self.st, self.d, self.TAG
        # the prompt's rows, A layout per chunk: the root's source and every chip's landing (x_hbm: every chip's copy
        # in L3 starts as zeros for a multicast inside the window to fill, a tile of 16 tokens at a time behind its own
        # flag, from the prompt's rows in the HBM of the root's memory chip)
        self.xsrc = np.concatenate([d.x_chunk_a(i * self.C, self.C) for i in range(self.NCH)]).astype(np.int8)
        self.h_xsrc = (st.put_zeros(f"{T}_xsrc", "int8_t", self.xsrc.size, preload=True) if self.x_hbm
                       else st.put(f"{T}_xsrc", "int8_t", self.xsrc))
        if self.x_hbm:
            m = self.mem[self.ROOT]
            self.h_xhbm = st.put_hbm(f"{T}_xhbm", self.xsrc, mem_chip=(m >> 4, m & 15))
            st.put_zeros(f"{T}_xtflag", "uint8_t", (self.T // 16) * 64)
        st.put_zeros(f"{T}_xland", "int8_t", self.NCH * self.XB, preload=True)
        st.put_zeros(f"{T}_xflag", "uint8_t", self.NCH * 64)
        # the root's y lives in its x buffers (y of chunk i, cluster 0 in the landing buffer of chunk i, cluster 1 in
        # the source of chunk i)
        if self.NC != 2 or self.XB != self.YB:
            raise ValueError("y over x needs two clusters and a chunk's x the size of a cluster's y")
        # RoPE tables, per chunk and cluster: cos then sin_signed, D layout [rows, 64]
        rope = []
        for i in range(self.NCH):
            for c in range(self.NC):
                cs, sn = d.rope_tables(self.r0(i, c), self.ROWS)
                rope += [cs, sn]
        self.h_rope = st.put(f"{T}_rope", "uint16_t", np.concatenate(rope).astype(np.float16).view(np.uint16))
        self.ROPE_B = self.ROWS * ROPE * 2
        self.h_w, self.h_c, self.h_wkv, self.h_kc = {}, {}, {}, {}
        # W_KVA whole in each memory chip's HBM (no task reads it: each chip streams its own slice, wkvc); the 16-token
        # pieces and their landed flags in every chip's L3
        wkv = np.concatenate(d.kva_chunks()).astype(np.int8)
        for m in sorted(set(self.mem.values())):
            self.h_wkv[m] = st.put_hbm(f"{T}_wkv_m{m:02x}", wkv, mem_chip=(m >> 4, m & 15))
        st.put_zeros(f"{T}_kv", "int8_t", (self.T // 16) * KVP, preload=True)
        st.put_zeros(f"{T}_kvflag", "uint8_t", self.NCH * 64)          # a chunk's pieces landed (multicast)
        st.put_zeros(f"{T}_kvnflag", "uint8_t", (self.T // 16) * 64)   # a piece landed on the root
        # each chip's W_KVA slice in its memory chip's HBM; per owned tile the gathered kv and a flag a chip
        self.h_wkvc = {k: st.put_hbm(f"{T}_wkvc_k{k:02x}", d.kva_cols(16 * self.kvg[k][0], 16 * self.kvg[k][1]),
                                     mem_chip=(self.mem[k] >> 4, self.mem[k] & 15)) for k in self.chips}
        st.put_zeros(f"{T}_kvraw", "uint8_t", self.KVR_TILES * self.KVD)
        st.put_zeros(f"{T}_kvrflag", "uint8_t", self.KVR_TILES * len(self.chips) * 64)
        # per chip: W_KVA's dequantisation (D layout [16, 576]), then each of its tiles' RoPE tables (cos, sin_signed),
        # KC_TILE bytes a tile
        kva = d.scale_tile(d.kva_scale()).view(np.uint8)
        for k in self.chips:
            tabs = []
            for q in self.kv_tiles[k]:
                cs, sn = d.rope_tables(16 * q, 16)
                tabs += [np.asarray(t, dtype=np.float16).view(np.uint8) for t in (cs, sn)]
            img = np.concatenate([kva] + tabs)
            self.h_kc[k] = st.put_hbm(f"{T}_kc_k{k:02x}", img.astype(np.uint8),
                                      mem_chip=(self.mem[k] >> 4, self.mem[k] & 15))
        self.KC_ROPE, self.KC_TILE = kva.size, 2 * 16 * ROPE * 2
        self.KC_BYTES = kva.size + len(self.kv_tiles[self.ROOT]) * self.KC_TILE
        # per head: the weights as the ring streams them, and the dequantisation constants: q_nope's 128 scales and
        # o's in the A layout of the cluster's rows, q~'s of one query tile, where the fused tasks multiply the
        # reshaped values; q_pe's stay D (RoPE reads them dequantised); then the causal masks as the score matmul's
        # INT32 [bc, 16] biases
        self.w_off = {}
        for k in self.chips:
            h = self.head[k]
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            chunks = d.weight_chunks(h, wo_block=None)
            self.w_off[k], off = [], 0
            for nm, b in chunks:
                self.w_off[k].append((nm, off, b.size))
                off += b.size
            self.h_w[k] = st.put_hbm(f"{T}_w_k{k:02x}", np.concatenate([b for _, b in chunks]), mem_chip=loc)
            qs = d.q_scale(h)
            sq = np.concatenate([d.scale_tile_a(qs[:Q_NOPE], self.ROWS).view(np.uint8),
                                 np.tile(d.scale_tile(qs[Q_NOPE:]), self.ROWS // 16).view(np.uint8)])
            sqt, suv = d.scale_tile_a(d.qt_scale(h), 16), d.scale_tile_a(d.uv_scale(h), self.ROWS)
            masks = d.mask_bias_patterns()
            const = np.concatenate([sq.view(np.uint8), sqt.view(np.uint8), suv.view(np.uint8),
                                    np.concatenate([m.reshape(-1) for m in masks]).view(np.uint8)])
            self.h_c[k] = st.put_hbm(f"{T}_c_k{k:02x}", const.astype(np.uint8), mem_chip=loc)
        self.SQ_B = (self.ROWS // 16) * 2048           # one W_q block's scales: 16 rows x 64 FP16 a row tile
        self.C_SQ, self.C_QT = 0, 3 * self.SQ_B
        self.C_UV = self.C_QT + 16384
        self.C_MASK = self.C_UV + self.ROWS * Q_NOPE * 2
        self.MASK_B = self.bc * BR * 4
        self.C_BYTES = self.C_MASK + len(d.mask_patterns) * self.MASK_B

    def kv_helpers(self):
        """KV_BALANCE_WAYS (INT4): W_KVA's 18 INT4 pairs over 16 chips leave two chips with two pairs, twice the keys'
        GEMM work. The tiles are cut into KV_BALANCE_WAYS equal runs: each two-pair chip makes its second pair for the
        first run only; for every other run a helper, a one-pair chip near the same memory chip (not the root), makes
        it from the pair's copy in that chip's HBM slab (the paired layout keeps each pair whole: the second pair is
        the slab's second half). Sets kvx {helper: (two-pair chip, the pair's first granule, flag slot, its tiles)},
        kv_two and kv_half (the tiles the two-pair chips leave to helpers)."""
        self.kvx, self.kv_two, self.kv_half = {}, [], set()
        ways = self.KV_BALANCE_WAYS
        if not ways:
            return
        if self.wbits != 4:
            raise ValueError("the W_KVA balance needs wbits 4")
        nx = self.T // 16
        if ways < 2 or nx % ways:
            raise ValueError(f"KV_BALANCE_WAYS={ways}: at least 2, dividing the {nx} tiles")
        run = nx // ways
        width = {k: self.kvg[k][1] - self.kvg[k][0] for k in self.chips}
        if set(width.values()) - {PG, 2 * PG}:
            raise ValueError(f"the W_KVA balance: chips with one or two INT4 pairs only, not "
                             f"{sorted(set(width.values()))}")
        dist = lambda a, b: abs((a >> 4) - (b >> 4)) + abs((a & 15) - (b & 15))     # noqa: E731
        self.kv_two = [k for k in self.chips if width[k] == 2 * PG]
        free = [k for k in self.chips if width[k] == PG and k != self.ROOT]
        if len(free) < len(self.kv_two) * (ways - 1):
            raise ValueError(f"KV_BALANCE_WAYS={ways}: {len(self.kv_two) * (ways - 1)} helpers, {len(free)} chips")
        for e, a in enumerate(self.kv_two):
            for w in range(1, ways):
                h = min(free, key=lambda k: (self.mem[k] != self.mem[a], dist(k, self.mem[a]), k))
                free.remove(h)
                self.kvx[h] = (a, self.kvg[a][0] + PG, e, set(range(w * run, (w + 1) * run)))
        self.kv_half = set(range(run, nx))
        print("[fp] kv_balance: " + ", ".join(f"chip {h:02x} makes chip {a:02x}'s second pair for tiles "
                                                f"{min(ts)}-{max(ts)}" for h, (a, _, _, ts) in self.kvx.items()))

    def _buffers(self):
        d = self.d
        self.xch_bytes = self.YB
        self.xch_zero = False
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.xch_bytes)
                    for k in self.chips for c in range(self.NC)}
        self.sweep_cfg = FaSweepCfg(bc=self.bc, nt=self.NT, nq=self.NQ, br=BR, k_s=d.k_s, k_o=d.k_o,
                                    a_exp=d.a_exp, a_n_f32bits=d.a_n_bits)
        self.consts = {cl: self.cx(cl).l1(f"{self.TAG}_consts", self.C_BYTES)
                       for cl in range(self.N_CL)}
        # W_KVA's dequantisation and the chip's tiles' RoPE tables, on cluster 0 (the tiles' keys)
        self.kvc = {(k, 0): self.cx_chip(k).at(0).l1(f"{self.TAG}_kvc", self.KC_BYTES) for k in self.chips}
        self._bufs, self._dst = {}, {}

    def bufs(self, cl, i):
        """The buffers of chunk i on cluster cl. A chunk has its own, so the static L1 planner overlays the ones whose
        lifetimes cannot meet."""
        if (cl, i) in self._bufs:
            return self._bufs[cl, i]
        k, c = self.chip(cl), self.local(cl)
        rows, nq = self.ROWS, self.NQ
        l1 = lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}_i{i}", nb)    # noqa: E731
        b = {}
        b["xrows"] = l1("xrows", rows * HIDDEN)
        # q: three D regions of 64 columns (the last is q_pe), rows x 64 FP16 each; simd_rope's operand block
        # [x | cos | xswap | sin], ROPE_B each: q_pe's GEMM and dequantisation write slot 0, the tables slots 1 and 3,
        # the pair swap slot 2
        b["rope4"] = l1("rope4", 4 * self.ROPE_B)
        b["q"] = [l1(f"q{r}", rows * 64 * 2) for r in range(2)] + [b["rope4"]]
        b["a16n"] = l1("a16n", rows * Q_NOPE * 2)             # q_nope, A layout FP16
        b["qn8"] = l1("qn8", rows * Q_NOPE)
        b["qt"] = l1("qt", rows * KV_RANK * 2)                # q~, D layout FP16
        b["a16q"] = l1("a16q", rows * KV_RANK * 2)
        b["pe"] = l1("pe", rows * ROPE * 2)
        b["a16p"] = l1("a16p", rows * ROPE * 2)
        b["q8"] = [l1(f"q8_{q}", D_QK * BR) for q in range(nq)]
        b["uv"] = l1("uv", rows * Q_NOPE * 2)                 # o_h, D layout FP16
        b["a16u"] = l1("a16u", rows * Q_NOPE * 2)
        b["o8"] = l1("o8", rows * Q_NOPE)
        # the sweep's names keep their sort order inside a chunk (the softmax's strides need it)
        b["swp"] = fa_sweep_alloc(lambda name, nb: self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}_i{i}", nb),
                                  self.sweep_cfg)
        self._bufs[cl, i] = b
        return b

    # ---- the weight ring ----------------------------------------------------------------------------------------
    def kva_in_ring(self):
        """Ring chunks ahead of the token chunks: the chip's slice of W_KVA."""
        return self.KVA_IN_RING

    def per_chunk(self):
        """Ring chunks a token chunk takes."""
        return self.PER_CHUNK

    def chunk_index(self, i, part):
        """The ring index of token chunk i's first chunk of `part`: head (W_q, W_UK) or tail (W_UV, W_O)."""
        return self.kva_in_ring() + i * self.per_chunk() + {"head": 0, "tail": self.PER_CHUNK_HEAD}[part]

    def _make_rings(self):
        """One ring a chip, shared by its two clusters: the per-chip schedule is the same for both. The cluster that
        first asks for a chunk posts its run (WeightRings.post_chunk); params weight_ring (cm.RING_KEYS) as
        ChipletBuild.make_rings reads them."""
        p, st, T = self.p, self.st, self.TAG
        wr = cm.ring_params(p)
        self.ring_queue = int(wr.get("queue", 16))
        self.n_slots = int(wr.get("slots", 16))
        self.ring_batch = int(wr.get("batch", 4))
        self.ring_head = int(wr.get("head", 0))
        self.ring_batch_head = int(wr.get("batch_head", 0))
        seq = np.zeros((1024, 8), dtype=np.uint64)
        seq[:, 0] = np.arange(1024, dtype=np.uint64)
        mems = sorted(set(self.mem.values()))
        self.h_seq = {m: st.put_hbm(f"{T}_ring_seq_m{m:02x}", seq.view(np.int8).reshape(-1),
                                    mem_chip=(m >> 4, m & 15)) for m in mems}
        self.h_flags = st.put_zeros(f"{T}_ring_flags", "uint8_t", self.n_slots * 64)
        self.h_wslots = st.put_zeros(f"{T}_wring_slots", "uint8_t", self.n_slots * self.slot_bytes, preload=True)
        self.rings, self.ring_take, self.ring_pos = {}, {}, {}
        for k in self.chips:
            self.rings[k] = WeightRings(
                slots=self.h_wslots, flags=self.h_flags, n_rings=1, n_slots=self.n_slots, slot_bytes=self.slot_bytes,
                batch=self.ring_batch, head=self.ring_head, batch_head=self.ring_batch_head)
            self.ring_take[k] = []
            self.ring_pos[k] = {}
            self.rings[k].push_from(memchip=self.mem[k], engine=self.engine[k], seq=self.h_seq[self.mem[k]],
                                    edge=self.edge, depth=self.ring_queue)
            self.queue_poster((self.mem[k], self.engine[k]), f"{T} ring, chip {k:#04x}", self.ring_queue,
                              self.rings[k].queue_span)
        self.ld_node = {}
        # (chip, ring chunk) whose last reader is cluster 1's read, after cluster 0's; every other chunk's last reader
        # is cluster 0's read
        self.ring_rel1 = set()

    def _src(self, k, n):
        """(HBM address, bytes) of chip k's ring chunk n."""
        if n < self.kva_in_ring():                  # the chip's W_KVA slice
            gr0, gr1 = self.kvg[k]
            return self.h_wkvc[k].address, HIDDEN * 16 * (gr1 - gr0) * self.wbits // 8
        nm, off, size = self.w_off[k][(n - self.kva_in_ring()) % self.per_chunk()]    # W_q x 3, W_UK, W_UV, W_O x 2
        return self.h_w[k].address + off, size

    def load(self, cl, n, name, deps=(), slab=None, dst=None):
        """Ring chunk n of chip(cl) into cluster cl's next slab: cluster 1 reads it, cluster 0 reads it after
        cluster 1's read, the chunk's last reader. Returns (load task, slab index, slab). dst: into that L1
        buffer instead, outside the cluster's load stream (slab index None, no WAR edge)."""
        k, c, dm = self.chip(cl), self.local(cl), self.roles["dm"]
        takes, pos = self.ring_take[k], self.ring_pos[k]
        if n not in pos:
            # a chunk's place in the ring is the order the program first asks for it
            if n != len(takes):
                raise ValueError(f"ring chunk {n} of chip {k:#04x} asked for before {len(takes)}")
            src, nb = self._src(k, n)
            pos[n] = len(takes)
            # the run's post on this cluster's DM core; the post that reuses the slot follows the chunk's last reader
            # (read_chunk below)
            tup, post = self.rings[k].post_chunk(
                lambda nm, core, kern, args, after: self.node(k, c, core, nm, kern, args, after), 0, name,
                src=src, nbytes=nb, dm=dm)
            takes.append(tup + (self.rings[k].last_take, post))
        sl, fl, seq, nb, ex, post = takes[pos[n]]
        ls = self.ls[cl]
        if dst is not None:
            si, slab, war = None, dst, []
        else:
            si, slab, war = ls.take(nb) if slab is None else ls.take_at(slab, nb)
        prev = [self.last_ld[cl]] if cl in self.last_ld else []
        # the chunk's last reader is one cluster's read (cluster 0's, or cluster 1's for a chunk in ring_rel1), which
        # follows every other cluster's read of it
        rel = 1 if (k, n) in self.ring_rel1 else 0
        readers = [self.ld_node[(k, c2, n)] for c2 in range(self.NC) if c2 != rel and (k, c2, n) in self.ld_node] \
            if c == rel else []
        ld = self.node(k, c, dm, f"{name}", "__snax_bingo_kernel_idma_ring_load",
                       SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, slab, nb),
                       list(dict.fromkeys(war + prev + list(deps) + readers + [post])))
        self.rings[k].read_chunk(0, seq - 1, ld, last=c == rel)
        if c != rel and (k, rel, n) in self.ld_node:
            self.dfg.bingo_add_edge(ld, self.ld_node[(k, rel, n)])
        self.last_ld[cl] = ld
        ld.exec_chunk = ex
        self.ld_node[(k, c, n)] = ld
        return ld, si, slab

    # ---- one task of one cluster --------------------------------------------------------------------------------
    def gmm(self, cl, name, a, b, d, M, K, N, k_shift, deps):
        """One GEMM of cluster cl (gemm_full). Returns the task its output follows."""
        k, c = self.chip(cl), self.local(cl)
        args = SnaxBingoKernelGemmFullArgs(
            input_A_addr=a, input_B_addr=b, input_C_addr=0, output_D_addr=d, M=M, K=K, N=N, array_shape_idx=0,
            transpose_A=0, transpose_B=0, accumPrevC=0, int32tofp16_enable=1,
            int4_b_enable=INT4_B_LAYOUT_PAIRED if self.wbits == 4 else 0, d_shift=k_shift)
        return self.node(k, c, self.roles["gemm"], name, "__snax_bingo_kernel_gemm_full", args, list(deps))

    def simd_mul(self, cl, name, a, b, out, nelems, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdMulF16Args(a, b, out, nelems // 32, 32), list(deps))

    def simd_add(self, cl, name, a, b, out, nelems, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdAddF16Args(a, b, out, nelems // 32, 32), list(deps))

    def mul_quant(self, cl, name, a, scale, dst, nelems, inv_bits, deps):
        """dst = quantise(a (.) scale): a dequantisation and the quantise after it in one SIMD task."""
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_stream_elementwise",
                         SnaxBingoKernelSimdMulI8Args(a, scale, dst, nelems // 32, 32, inv_bits), list(deps))

    def quant(self, cl, name, src, dst, nbytes, inv_bits, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["simd"], name, "__snax_bingo_kernel_simd_fp16_to_int8",
                         SnaxBingoKernelSimdFp16ToInt8Args(src, dst, beats=simd_beats(nbytes), rows=1,
                                                           inv_scale_f32bits=inv_bits), list(deps))

    def xd(self, cl, name, args, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["xdma"], name, "__snax_bingo_kernel_xdma_6d", args, list(deps))

    def dm_copy(self, cl, name, src, dst, nbytes, deps):
        k, c = self.chip(cl), self.local(cl)
        return self.node(k, c, self.roles["dm"], name, "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(src, dst, nbytes), list(deps))

    @staticmethod
    def fbits(inv):
        return int(np.asarray(inv, dtype=np.float32).reshape(()).view(np.uint32))

    def relays(self):
        """Every dependency-tag relay of an xDMA task serves that one task."""
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]

    # ---- x from HBM into every chip's L3 --------------------------------------------------------------------------
    def cl_tiles(self, c):
        """The tiles of x cluster c makes keys for and waits for."""
        return [t for t in range(self.NX) if t % self.NC == c]

    def stage_x_bcast(self, go):
        """x's multicast from HBM (ChipletBuild.hbm_cast: its posts and its waits), queued by the root's host on its
        memory chip's push engine, ahead of the root's ring runs."""
        R, T = self.ROOT, self.TAG
        m = self.mem[R]
        return [self.hbm_cast(
            R, f"x_hbm_bcast_k{R:02x}",
            HostBingoKernelHbmBcastArgs(self.h_xhbm.address, BingoMemSymbol(f"{T}_xsrc"), BingoMemSymbol(f"{T}_xtflag"),
                                        self.h_seq[m].address + 64, 1, self.XT, self.NX, m, self.engine[R]),
            after=[go[R]], ahead_of_rings=True)]

    def x_wait(self, k, c, t, go, name=None):
        """Chip k cluster c waits for tile t's landed flag in its chip's L3 (the flag proves the multicast's data; the
        task that queues the tile is an ordering edge)."""
        w = self.node(k, c, self.roles["dm"], name or f"x_wait_t{t}", "__snax_bingo_kernel_wait_flags",
                      SnaxBingoKernelWaitFlagsArgs(BingoMemSymbol(f"{self.TAG}_xtflag", offset=t * 64), 1, 1), [go[k]])
        self.dfg.bingo_add_order_edge(self.queued_of(self.x_puts[0]), w)
        return w

    # ---- checks -------------------------------------------------------------------------------------------------
    def run_checks(self, checks):
        """The key checks read what every owner's puts wrote into its L3: the checks also wait for every put."""
        puts = list(getattr(self, "kv_put", {}).values()) + list(getattr(self, "kvp_put", {}).values())
        self.check_copies = list(self.check_copies) + [t for t in puts if t not in self.check_copies]
        return super().run_checks(checks)

    def window_end(self, last):
        """One probe a chip behind its last tasks, the root's behind x's multicast too; then the end stamp."""
        R = self.ROOT
        ends = [self.probe(k, f"fin{k:02x}", last[k]) for k in self.chips if k != R]
        ends.append(self.probe(R, f"fin{R:02x}", last[R] + [self.done_of(p) for p in getattr(self, "x_puts", [])[-1:]]))
        self.end_window(ends)


def app_name(S):
    """The program's name: its stage, its tokens and its platform."""
    return (f"DeepSeek-V2-Lite layer 1 MLA attention prefill, stage {S.STAGE}, {S.T} tokens, {len(S.chips)} chips x "
            f"{S.NC} clusters, x from HBM" + (", split loads" if S.x_xdma else ""))
