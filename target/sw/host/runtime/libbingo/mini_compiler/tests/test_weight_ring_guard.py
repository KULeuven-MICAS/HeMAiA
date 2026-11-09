#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Tests for the weight prefetcher's byte guard (host runtime libbingo/weight_ring_guard.h): the C
# guard, built for the workstation, under a model of the prefetcher's push loop, on the ring
# places WeightRings gives. A run's chunks sit back to back from the run's first slot, so a chunk
# shorter than a slot can start in a lower slot than its own; the slot rule alone (a run waits for
# the chunks n_slots before its own) then lets the next wrap overwrite a chunk not read yet. The
# stream is one cluster of the 4-chip DeepSeek-V2-Lite FFN (workloads/ffn/scaling/base_n4): 43
# gate|up chunks of 64 KiB, then 16 down chunks of 86.5 KiB, one HBM range, 16 slots of 88 KiB.
#
#   python3 test_weight_ring_guard.py

import ctypes
import os as _os
import random
import subprocess
import sys as _sys
import tempfile

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bingo_paths  # noqa: F401,E402
from bingo_mem_handle import BingoMemSymbol                          # noqa: E402
from libs.blocks import WeightRings                                  # noqa: E402

GUARD_H = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
                        "weight_ring_guard.h")
NS, SB = 16, 90112
FFN_4CHIP = [65536] * 43 + [88576] * 16            # one cluster's chunks, in ring order


class Guard:
    """The C guard (weight_ring_guard.h), compiled for this machine."""
    _lib = None

    def __init__(self, pushed=0, span=4):
        if Guard._lib is None:
            Guard._lib = _build()
        self.g = (ctypes.c_int64 * 32)()
        Guard._lib.g_init(self.g, pushed, NS, span)

    def free(self, s0, nbytes, cons):
        return bool(Guard._lib.g_free(self.g, s0, nbytes, SB, cons))

    def mark(self, s0, off, size, k):
        Guard._lib.g_mark(self.g, s0, off, size, SB, k)


def _build():
    shim = ("#include \"%s\"\n"
            "void g_init(int64_t *g, uint64_t p, uint64_t ns, uint64_t span) {"
            " weight_ring_guard_init((weight_ring_guard_t *)g, p, ns, span); }\n"
            "int g_free(const int64_t *g, uint64_t s0, uint64_t b, uint64_t sb, uint64_t c) {"
            " return weight_ring_guard_free((const weight_ring_guard_t *)g, s0, b, sb, c); }\n"
            "void g_mark(int64_t *g, uint64_t s0, uint64_t o, uint64_t sz, uint64_t sb, uint64_t k) {"
            " weight_ring_guard_mark((weight_ring_guard_t *)g, s0, o, sz, sb, k); }\n") % GUARD_H
    d = tempfile.mkdtemp()
    src, so = _os.path.join(d, "shim.c"), _os.path.join(d, "shim.so")
    open(src, "w").write(shim)
    subprocess.run(["gcc", "-O1", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "-o", so, src], check=True)
    lib = ctypes.CDLL(so)
    lib.g_init.argtypes = [ctypes.c_void_p] + [ctypes.c_uint64] * 3
    lib.g_free.argtypes = [ctypes.c_void_p] + [ctypes.c_uint64] * 4
    lib.g_mark.argtypes = [ctypes.c_void_p] + [ctypes.c_uint64] * 5
    return lib


def places(sizes, batch=4, head=4, batch_head=1):
    """[(place in the ring, bytes)] of each chunk and [(first chunk, chunks, first slot)] of each run, as
    WeightRings fixes them (the prefetcher forms the same runs)."""
    wr = WeightRings(slots=BingoMemSymbol("slots"), flags=BingoMemSymbol("flags"),
                     releases=BingoMemSymbol("rel"), n_rings=1, n_slots=NS,
                     slot_bytes=SB, batch=batch, head=head, batch_head=batch_head)
    out, runs, src = [], [], 0x8000_0000
    for k, nb in enumerate(sizes):
        wr.take(0, src, nb)
        src += nb
        r0, _, filled = wr._run[0]
        out.append((r0 * SB + filled - nb, nb))
        if runs and runs[-1][2] == r0 and filled > nb:
            runs[-1][1] += 1
        else:
            runs.append([k, 1, r0])
    return out, [tuple(r) for r in runs]


def replay(sizes, guarded, stalls, lo=0, seed=0):
    """The push loop of one ring against a cluster that reads chunk j at the first step where it has
    landed and its stall (stalls[j] steps) has passed. A run is pushed at the first step the slot rule
    allows (every chunk of it at most n_slots ahead of the reads) and, guarded, the guard too. Starts
    with chunks [0, lo) pushed and read up to lo - n_slots (a later part of the prefetcher). Returns the
    overwrites of a chunk not read yet: (unread chunk, pushed chunk)."""
    pl, runs = places(sizes)
    runs = [r for r in runs if r[0] >= lo]
    rng = random.Random(seed)
    g = Guard(pushed=lo) if guarded else None
    pushed, cons, bad = lo, max(0, lo - NS), []
    landed = set(range(lo))
    wait = {j: stalls.get(j, rng.randint(0, 2)) for j in range(len(sizes))}
    ri = 0
    for _ in range(100000):
        if cons >= len(sizes):
            break
        # the cluster: reads the next chunk once it has landed and its stall is over
        if cons in landed:
            if wait[cons] > 0:
                wait[cons] -= 1
            else:
                cons += 1
        # the prefetcher: the next run, if the rules allow it now
        if ri < len(runs):
            k0, m, s0 = runs[ri]
            nbytes = sum(sizes[k0:k0 + m])
            ok = k0 + m - 1 < cons + NS
            if ok and guarded:
                ok = g.free(s0, nbytes, cons)
            if ok:
                lo_b, hi_b = s0 * SB, s0 * SB + nbytes
                for j in range(cons, pushed):           # landed, not read yet
                    pj, nj = pl[j]
                    if pj < hi_b and lo_b < pj + nj:
                        bad.append((j, k0 + m - 1))
                off = 0
                for k in range(k0, k0 + m):
                    if guarded:
                        g.mark(s0, off, sizes[k], k)
                    off += sizes[k]
                    landed.add(k)
                pushed, ri = k0 + m, ri + 1
    return bad


def test_ffn_places():
    """The places of the 4-chip FFN's ring: chunk 43 (the first down chunk, in the run of the last gate|up
    chunks) at slot 10.18, chunk 58 one wrap later at 9.97, over chunk 43's first 69,120 bytes."""
    pl, _ = places(FFN_4CHIP)
    assert pl[43] == (int(10.181818181818183 * SB), 88576), pl[43]
    assert pl[58][0] == pl[43][0] - 19456, pl[58]
    assert pl[58][0] + pl[58][1] - pl[43][0] == 69120


def test_slot_rule_alone_overwrites():
    """Without the guard, the cluster stalling before its first down chunk (the barrier after the gate|up
    phase) lets the push of chunk 58 land on chunk 43."""
    bad = replay(FFN_4CHIP, guarded=False, stalls={43: 200})
    assert (43, 58) in bad or any(j == 43 for j, _ in bad), bad


def test_guard_never_overwrites():
    """With the guard, no stall pattern lets a push land on a chunk not read yet."""
    for seed in range(200):
        rng = random.Random(1000 + seed)
        stalls = {j: rng.choice([0, 0, 0, 1, 5, 50, 200]) for j in range(len(FFN_4CHIP))}
        assert replay(FFN_4CHIP, guarded=True, stalls=stalls, seed=seed) == [], seed


def test_guard_later_part():
    """A prefetcher part that starts mid-ring (earlier chunks pushed by another part, their runs unknown)
    holds the slots those chunks may fill."""
    for lo in (8, 30, 40, 44, 50):
        for seed in range(50):
            rng = random.Random(seed)
            stalls = {j: rng.choice([0, 1, 50, 200]) for j in range(len(FFN_4CHIP))}
            assert replay(FFN_4CHIP, guarded=True, stalls=stalls, lo=lo, seed=seed) == [], (lo, seed)


def test_equal_chunks_unchanged():
    """Chunks of one slot each never straddle: the guard allows exactly what the slot rule allows."""
    sizes = [SB] * 60
    pl, runs = places(sizes)
    g = Guard()
    for k0, m, s0 in runs:
        for cons in range(max(0, k0 + m - NS), k0 + m):
            assert g.free(s0, m * SB, cons) == (k0 + m - 1 < cons + NS), (k0, cons)
        for k in range(k0, k0 + m):
            g.mark(s0, (k - k0) * SB, SB, k)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
