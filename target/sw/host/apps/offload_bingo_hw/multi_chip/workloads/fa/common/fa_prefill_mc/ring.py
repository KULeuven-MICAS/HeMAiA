# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaBuild's weight ring (one a chip, both clusters read it) and the sweep's key and value tiles from L3."""

import numpy as np

import chiplet_mc as cm
from bingo_kernel_args import SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaRingLoadArgs
from bingo_mem_handle import BingoMemSymbol
from libs.blocks import WeightRings

from .consts import D_QK, D_V, HIDDEN, KB, KVP


class FreshTileSource:
    """fa_sweep's tiles: key and value tile j of chunk i from the chip's own L3, where the keys' multicasts leave the
    16-token pieces [key rows | value columns] side by side, behind one wait a sweep on the flags of the pieces it
    reads. A key tile is its pieces' key rows one after the other (a 2D copy); a value tile is the A layout of
    [512, bc], each piece's 256-byte runs one 16-row block apart (a 3D copy). K tiles in slab 0 (split) or slab 0 and a
    one-tile buffer alternately (split_kb), V tiles in slabs 1 and 2 alternately."""

    def __init__(self, fb, cl, i, nt):
        self.fb, self.cl, self.i, self.nt = fb, cl, i, nt
        self.wait = None

    def fetch(self, kind, j):
        fb = self.fb
        k, c = fb.chip(self.cl), fb.local(self.cl)
        if self.wait is None:
            self.wait = fb.kv_wait(self.cl, self.i)
        n = fb.bc // 16
        src = BingoMemSymbol(f"{fb.TAG}_kv", offset=j * n * KVP + (KB if kind == "V" else 0))
        if kind == "K" and fb.kv_slabs == "split_kb" and j % 2:
            # split_kb: the odd key tiles in a buffer of one key tile of their own, free once the last QK of the
            # odd tile before has read it
            si, slab = ("kb", self.cl), fb.kbuf(self.cl)
            war = list(fb._kb_reader.get(self.cl, []))
            args = SnaxBingoKernelIdma2dCopyArgs(src, slab, KB, KVP, KB, n)
        elif kind == "K":
            si, slab, war = fb.ls[self.cl].take_at(0, fb.bc * D_QK)
            args = SnaxBingoKernelIdma2dCopyArgs(src, slab, KB, KVP, KB, n)
        else:
            si, slab, war = fb.ls[self.cl].take_at(1 + j % 2, D_V * fb.bc)
            args = SnaxBingoKernelIdma2dCopyArgs(src, slab, 256, 256, 16 * fb.bc, D_V // 16, outer=n, src_outer=KVP,
                                                 dst_outer=256)
        ld = fb.node(k, c, fb.roles["dm"], f"Ld_{kind}{j}_i{self.i}", "__snax_bingo_kernel_idma_2d_copy", args,
                     war + [self.wait])
        return ld, slab, si

    def release(self, token, task):
        if isinstance(token, tuple) and token[0] == "kb":
            self.fb._kb_reader[token[1]] = list(task) if isinstance(task, (list, tuple)) else [task]
        else:
            self.fb.ls[self.cl].release(token, task)


class FaRing:
    """One ring a chip: W_KVA's slice of the chip first, then per token chunk W_q (3), W_UK, W_UV and W_O (2)."""

    PER_CHUNK_HEAD, PER_CHUNK_TAIL = 4, 3          # W_q x 3 and W_UK; W_UV and W_O x 2

    def kva_in_ring(self):
        """Ring chunks ahead of the token chunks: the chip's slice of W_KVA."""
        return 1

    def per_chunk(self):
        """Ring chunks a token chunk takes: a run that stops at Q8 takes only the head."""
        return self.PER_CHUNK_HEAD if self.stop == "q8" else self.PER_CHUNK_HEAD + self.PER_CHUNK_TAIL

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
