# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""CREST-compressed weight streams for WeightRings (params weight_crest).

The memory chiplet pushes each weight chunk COMPRESSED (snax_cluster hw/chisel/doc/
crest_decompressor.md), so the half-duplex D2D link -- the layer's bottleneck -- carries
fewer bytes; the cluster's xDMA reads the chunk from its L3 ring slot and the
writer's CrestDecompressor expands it into the L1 slab (__snax_bingo_kernel_xdma_ring_load).

A chunk as stored and pushed (a RECORD): padding, the stream, then a 64-B TAIL beat -- bytes
8..11 the stream's 64-B words W (0: the chunk is stored plain, its N beats instead of a
stream), 12..15 the output beats N, and in trailer mode bytes 0..7 the trailer magic. The DM
core's ring load copies the record into the END of the L1 slab, so the tail is the slab's last
beat whatever the padding, and the xDMA expands it into the slab's first N beats in place
(__snax_bingo_kernel_xdma_crest_expand). That is safe only if no output beat lands on an input
word the decompressor has not read yet; record() checks it for the slab and stores the chunk
plain when it does not hold. Lengths are known here, when a chunk is taken, so the ring places
it by its record size; ADDRESSES are fixed before the graph is built -- the regions are
reserved in HBM at their uncompressed size, which the expert table and every golden that holds
an address already name -- and the bytes are written into them after the build (finish).

Dense chunks (kind 0) go into their ring's region back to back in take order, so a ring's
consecutive chunks stay adjacent and still merge into one push. A routed chunk (kind 1, an
expert field the router picks at run time) has one length per expert: every expert's copy is
padded to the longest, so the chunk's offset and pushed size do not depend on the choice --
only its header does, which the load reads after the flag.
"""

import atexit
import hashlib
import json
import os

import numpy as np

LANES = {0: 128, 1: 64, 2: 32}       # CREST mode -> lanes per 64-B beat
EXTRA = {0: 0, 1: 0, 2: 8}          # raw bits per lane after the code (bf16)
G = 64                              # beats per group (snax_split_cluster groupBeats)
# trailer mode (weight_ring.trailer): the LAST 64-B beat of every record carries this 64-bit
# word, so the push itself says when a chunk has landed (same transfer, writes in order) and
# the run needs no flag transfer; the load polls it, then clears it before it frees the slot
TRAILER_MAGIC = 0x5EED7A11C0DEF1A6


# export_only: a record's size and whether it expands in place depend only on the chunk's bytes,
# so they are cached by content (BINGO_CREST_CACHE, default ~/.cache/hemaia/crest_sizes.json) and
# a size-only CrestRings returns a stream of the right length without encoding it. The bytes are
# placeholders: an export-only build is for the bingo DSE, never for the RTL.
_SIZES = None


def _sizes():
    global _SIZES
    if _SIZES is None:
        path = os.environ.get("BINGO_CREST_CACHE",
                              os.path.expanduser("~/.cache/hemaia/crest_sizes.json"))
        try:
            with open(path) as f:
                _SIZES = {"path": path, "map": json.load(f), "dirty": False}
        except (OSError, ValueError):
            _SIZES = {"path": path, "map": {}, "dirty": False}
        atexit.register(_save_sizes)
    return _SIZES


def _save_sizes():
    c = _SIZES
    if c is None or not c["dirty"]:
        return
    os.makedirs(os.path.dirname(c["path"]), exist_ok=True)
    try:                                      # merge with what another export wrote meanwhile
        with open(c["path"]) as f:
            c["map"] = {**json.load(f), **c["map"]}
    except (OSError, ValueError):
        pass
    tmp = f"{c['path']}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(c["map"], f)
    os.replace(tmp, c["path"])


class CrestRings:
    """One chip's compressed weight chunks.

    codec      crest_codec (snax_cluster hw/chisel/doc/crest_decompressor/crest_codec.py)
    dense      {ring: (uncompressed base address, uncompressed bytes, reserved handle,
                reserved bytes)}: the ring's dense weights and its compressed region
    routed     {field: ({expert: uncompressed bytes}, {expert: reserved handle})}: the chip's
               blob of that field per expert; a ring's slice starts at slices[ring][field]
    slices     {ring: {field: (first byte, end byte)}}: each ring's slice of a routed blob
    """

    FIELDS = ("gu", "dn")

    def __init__(self, codec, dense, routed, slices, slab_bytes, modes=(0, 1), trailer=False,
                 size_only=False):
        self.codec, self.dense_map, self.routed_map = codec, dense, routed
        self.size_only = bool(size_only)     # export_only: placeholder streams, sizes cached
        self.slices, self.modes, self.trailer = slices, tuple(modes), bool(trailer)
        if slab_bytes % 64:
            raise ValueError(f"CREST: slab of {slab_bytes} B is not whole 64-B beats")
        self.cap = slab_bytes // 64          # the L1 slab the records expand in, in words
        self.cur = {r: 0 for r in dense}
        self.rcur = {(r, f): slices[r][f][0] for r in slices for f in slices[r]}
        self.writes = []                     # (handle, offset, bytes)
        self.placed = {}                     # routed (ring, field, offset, bytes) -> (at, n)
        self.stats = {"dense": [0, 0], "routed": [0, 0]}    # [uncompressed, pushed]
        self.plain = 0                       # chunks stored plain

    def _in_place_ok(self, words: np.ndarray, beats: int) -> bool:
        """Whether the stream, ending right below the tail of a self.cap-word slab, expands
        into the slab's first `beats` beats without an output beat landing on an input word
        not yet read. Input word i sits at word cap - 1 - W + i; before it writes beat j the
        decompressor has read at least the stream header, every earlier group and this
        group's escape words and plane up to beat j (it reads ahead of that, which only
        helps). So for every j: W - read(j) < cap - 1 - j."""
        w = np.asarray(words, np.uint8).reshape(-1, 64)
        W = w.shape[0]
        sh = int(w[0, :4].view("<u4")[0])
        mode, p = sh & 3, (sh >> 11) & 7
        L, lb = LANES[mode], p + EXTRA[mode]
        pos, done = 1, 0                     # words read so far (whole groups), beats so far
        while done < beats:
            gb = min(G, beats - done)
            hdr = int(w[pos, :4].view("<u4")[0])
            j = np.arange(gb)
            if hdr & (1 << 15):              # raw: the header, then the beats
                need = pos + 1 + j + 1
                pos += 1 + gb
            else:
                e = hdr & 0x7FFF
                need = pos + e + -(-(j + 1) * L * lb // 512)
                pos += e + -(-gb * L * lb // 512)
            if np.any(W - need >= self.cap - 1 - (done + j)):
                return False
            done += gb
        return pos == W

    def encode(self, data):
        """(stream bytes or None, W): the shortest of the modes, None = store the chunk plain
        (it does not get shorter, or it would not expand in place in the slab)."""
        data = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
        if data.size % 64:
            raise ValueError(f"CREST chunk of {data.size} B: whole 64-B beats only")
        beats = data.size // 64
        if beats + 1 > self.cap:
            raise ValueError(f"CREST: a {beats}-beat chunk and its tail exceed the "
                             f"{self.cap}-word slab")
        key = None
        if self.size_only:
            key = f"{hashlib.sha1(data.tobytes()).hexdigest()}:{self.modes}:{self.cap}"
            hit = _sizes()["map"].get(key)
            if hit is not None:
                words, nbytes = hit
                if words:
                    return np.zeros(nbytes, np.uint8), words
                self.plain += 1
                return None, 0
        best = None
        for m in self.modes:
            words, info = self.codec.encode(data.tobytes(), m)
            if best is None or info["words"] < best[1]:
                best = (np.asarray(words, np.uint8), info["words"])
        ok = best[1] < beats and self._in_place_ok(best[0], beats)
        if key is not None:
            c = _sizes()
            c["map"][key] = [int(best[1]), int(best[0].size)] if ok else [0, 0]
            c["dirty"] = True
        if ok:
            return best[0], best[1]
        self.plain += 1
        return None, 0

    def tail(self, words: int, beats: int) -> np.ndarray:
        t = np.zeros(64, np.uint8)
        if self.trailer:
            t[:8] = np.array([TRAILER_MAGIC], dtype="<u8").view(np.uint8)
        t[8:16] = np.array([words, beats], dtype="<u4").view(np.uint8)
        return t

    def record(self, data, size: int = None) -> np.ndarray:
        """The chunk as pushed: padding up to `size` bytes (default none), the stream (or the
        plain data), the tail."""
        data = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
        stream, W = self.encode(data)
        body = data if stream is None else stream
        rec = np.concatenate([body, self.tail(W, data.size // 64)])
        if size is not None and size > rec.size:
            rec = np.concatenate([np.zeros(size - rec.size, np.uint8), rec])
        return rec

    def dense(self, ring: int, src: int, nbytes: int):
        """(compressed source address, pushed bytes) of a dense chunk."""
        base, data, h, cap = self.dense_map[ring]
        off = int(src) - int(base)
        if off < 0 or off + nbytes > data.size:
            raise ValueError(f"CREST: ring {ring}'s chunk at {src:#x} (+{nbytes}) is outside "
                             f"its dense image")
        rec = self.record(data[off:off + nbytes])
        at = self.cur[ring]
        if at + rec.size > cap:
            raise ValueError(f"CREST: ring {ring}'s compressed region ({cap} B) is full")
        self.cur[ring] = at + rec.size
        self.writes.append((h, at, rec))
        self.stats["dense"][0] += nbytes
        self.stats["dense"][1] += rec.size
        return int(h.address) + at, rec.size

    def routed(self, ring: int, field: str, offset: int, nbytes: int):
        """(compressed offset into the field's blob, pushed bytes) of a routed chunk: every
        expert's copy padded to the longest. Every expert slot loads the same chunks of its
        own expert, so a chunk is placed once and the other slots' takes find it here."""
        key = (ring, field, int(offset), int(nbytes))
        if key in self.placed:
            return self.placed[key]
        data, handles = self.routed_map[field]
        recs = {e: self.record(d[offset:offset + nbytes]) for e, d in data.items()}
        # every expert's record padded at the FRONT to the longest: the tail (and the trailer)
        # stays the last beat, and the stream ends right below it, whichever expert it is
        n = max(r.size for r in recs.values())
        at = self.rcur[(ring, field)]
        end = self.slices[ring][field][1]
        if at + n > end:
            raise ValueError(f"CREST: ring {ring}'s slice of {field} ends at {end}; a chunk "
                             f"of {n} B at {at} runs past it")
        self.rcur[(ring, field)] = at + n
        for e, r in recs.items():
            self.writes.append((handles[e], at, np.concatenate([np.zeros(n - r.size, np.uint8),
                                                                r])))
        self.stats["routed"][0] += nbytes
        self.stats["routed"][1] += n
        self.placed[key] = (at, n)
        return at, n

    def finish(self, fill):
        """Write every compressed chunk into its reserved region (fill(handle, bytes))."""
        from .comm.ports import at_offset
        for h, at, rec in self.writes:
            fill(at_offset(h, at), rec)
        self.writes = []
        return {k: (u, p, (u / p) if p else 0.0) for k, (u, p) in self.stats.items()}
