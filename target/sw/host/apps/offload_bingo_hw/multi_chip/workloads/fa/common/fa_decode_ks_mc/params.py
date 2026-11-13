# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The step's parameters and its mapping (README.md, PARAMETERS and THE MAPPING): the cache's length, the checks, the
data's inputs, the head of each chip, the attention's units and W_KVA's chunks a chip."""

import os

from bingo_kernel_args import simd_fp16_lanes
from driver import repo_root

import chiplet_mc as cm

from .consts import BC, BR, KB, N_KVA

REPO = repo_root(__file__)
CHECKS = ("y", "stages")             # params checks: the layer's checks of the attention (y), or with the steps' too


class KsParams:
    """FaDecodeKs.__init__: ChipletBuild's (the platform's and the graph's switches, the chips, the empty DFG, the data
    staging, a layer's block sharing the layer's), then the step's parameters and its mapping. No data, memory or task
    yet."""

    def __init__(self, args, p, plat, hw, shared=None):
        cm.ChipletBuild.__init__(self, args, p, plat, hw, shared=shared)
        if self.T != 1:
            raise ValueError("params tokens: the decode runs one token")
        if BR != simd_fp16_lanes():
            raise ValueError(f"the softmax takes one query per FP16 lane: {simd_fp16_lanes()} on this cluster, BR {BR}")
        if self.wbits != 4:
            raise ValueError("params wbits 4: the GEMV streams paired INT4")
        for key, want in (("bcast", "bcast_l1"), ("reduce", "chain"), ("sync", "edge")):
            if key in p and p[key] != want:
                raise ValueError(f"params {key}={p[key]!r}: the key-split decode builds {want!r}")
        if self.chunk < KB:
            raise ValueError(f"params w_chunk_bytes={self.chunk}: a K tile is {KB} B")
        self.P = int(p.get("cache", 1024))
        nchip = len(self.chips)
        if self.P % (BC * nchip):
            raise ValueError(f"params cache={self.P}: whole tiles of {BC} keys, the same number on each of {nchip} "
                             f"chips")
        self.NTC = self.P // BC                    # the cached tiles; the new tile is tile NTC
        self.checks_kind = str(p.get("checks", "y"))
        if self.checks_kind not in CHECKS:
            raise ValueError(f"params checks={self.checks_kind!r}: y or stages")
        # the data's inputs: the cache of hidden states, the checkpoint's weights, the calibration tokens
        self.kv_path = os.path.join(REPO, p.get("kv_cache", "target/sim/automation/sweep/llm_blocks/fa/cache/"
                                                            "dsv2_l1_kv_8193.npz"))
        self.w_cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        self.n_calib = int(p.get("n_calib", 32))
        R = self.ROOT
        self.head = {k: i for i, k in enumerate(self.chips)}
        self.tiles = {k: [i + nchip * t for t in range(self.NTC // nchip)] for i, k in enumerate(self.chips)}
        self.units = [(k, self.QC, self.tiles[k]) for k in self.chips] + [(R, self.KC, [self.NTC])]
        # W_KVA's chunks: chunk n on chip n's cluster 1, the rest on one chip a memory link (_kva_extra)
        extra = self._kva_extra(N_KVA - nchip)
        if len(extra) != N_KVA - nchip:
            raise ValueError(f"W_KVA's chunks {nchip}..{N_KVA - 1}: {len(extra)} memory links for them")
        self.kva_extra = extra
        self.kva = {k: [i] for i, k in enumerate(self.chips)}
        for n, k in enumerate(extra, start=nchip):
            self.kva[k].append(n)

    def _kva_extra(self, n):
        """The chips whose cluster 1 takes W_KVA's chunks past one a chip: one chip a memory link, the links of the
        memory chips other than the root's (whose links also carry x and the root's multicasts) in link order, each
        the chip nearest to its memory chip."""
        R, out = self.ROOT, []
        for ln in sorted(set(self.link.values())):
            if ln[0] == self.mem[R] or len(out) == n:
                continue
            out.append(min((k for k in self.chips if self.link[k] == ln), key=lambda k: (cm.dist(k, self.mem[k]), k)))
        return out
