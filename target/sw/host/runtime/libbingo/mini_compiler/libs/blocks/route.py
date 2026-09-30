# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""A mixture of experts whose experts the router picks AT RUN TIME, as data.

    MoeRoute    the top k of the router's probabilities, as an EXPERT-SLOT RECORD: per
                chosen expert its id, its weight, and its entry of the expert table (the
                addresses of its weights and dequant factors, and its SwiGLU scale)
    SlotLoad    one of a slot's tensors (its dequant factors) into L1, from the address
                the record names

A slot's GEMVs are ordinary Linear blocks with w_slot=(slot, field): their chunk loads read
the record too. So the graph is the same whatever the router decides -- nothing is skipped,
nothing branches on an id -- and a model with 64 experts costs the graph of its 6 slots,
not of 64 branches (compare MoeFFN, whose experts are branches of a conditional fork).
This is the snax reference's design (snax-dsv2-moe.h, "EXPERT SLOTS, NOT BRANCHES").

THE TABLE. n entries of 64 bytes in L3 (or any memory the iDMA reaches), entry e:
    words 0-7   gate|up weights, their factors, down weights, their factors -- four
                64-bit addresses, (lo, hi) each
    word  8     the SwiGLU output's inv scale, FP32 bits
    words 9-15  0
An expert whose weights were never staged has an all-zero entry: the route refuses it.
"""

from dataclasses import dataclass, field

import numpy as np

from bingo_kernel_args import (SnaxBingoKernelIdmaCopySlotArgs, SnaxBingoKernelIdmaRingLoadArgs,
                               SnaxBingoKernelMoeRouteArgs)

from ..comm import Block, BlockResult, Ctx, DType, Layout, MemLevel, Port, PortSpec
from .linear import REC_SLOT_BYTES, record_spec

TABLE_ENTRY_BYTES = SnaxBingoKernelMoeRouteArgs.TABLE_ENTRY_BYTES


def expert_table(n: int, entries: dict) -> np.ndarray:
    """The expert table as bytes: `entries` maps an expert id to (gu, gu_s, dn, dn_s,
    inv_a_f32bits), four addresses and a scale; every other expert gets a zero entry."""
    t = np.zeros((n, TABLE_ENTRY_BYTES // 4), dtype=np.uint32)
    for e, (gu, gus, dn, dns, inv) in entries.items():
        for i, a in enumerate((gu, gus, dn, dns)):
            t[e, 2 * i] = int(a) & 0xFFFFFFFF
            t[e, 2 * i + 1] = (int(a) >> 32) & 0xFFFFFFFF
        t[e, 8] = int(inv) & 0xFFFFFFFF
    return t.view(np.int8).reshape(-1)


def pass_record_bytes(id_lists, w16_lists, table: np.ndarray) -> np.ndarray:
    """The records moe_route writes for a pass: the union of the tokens' ids (token 0's in
    order, then each later token's new ones), one record per token with its own weights and
    0 for the experts it did not pick."""
    order = []
    for ids in id_lists:
        order += [int(e) for e in ids if int(e) not in order]
    recs = []
    for ids, ws in zip(id_lists, w16_lists):
        w_of = {int(e): int(w) & 0xFFFF for e, w in zip(ids, ws)}
        recs.append(record_bytes(order, [w_of.get(e, 0) for e in order], table))
    return np.concatenate(recs), order


def record_bytes(ids, w16_bits, table: np.ndarray) -> np.ndarray:
    """The record moe_route writes for the chosen `ids` (their FP16 weights as bits): what a
    host check compares it against, byte for byte."""
    tab = np.asarray(table, dtype=np.int8).reshape(-1, TABLE_ENTRY_BYTES)
    rec = np.zeros((len(ids), REC_SLOT_BYTES // 4), dtype=np.uint32)
    for s, (e, w) in enumerate(zip(ids, w16_bits)):
        w = int(w) & 0xFFFF
        rec[s, 0] = int(e)
        rec[s, 1] = int(np.array(w, dtype=np.uint16).view(np.float16).astype(np.float32)
                        .view(np.uint32))
        rec[s, 2] = w
        rec[s, 16:32] = tab[int(e)].view(np.uint32)
    return rec.view(np.int8).reshape(-1)


@dataclass(frozen=True)
class MoeRouteCfg:
    n: int = 64                # experts
    k: int = 6                 # chosen per token
    cluster: int = 0
    # A pass of `tokens`: p holds one row per token, and the slots are the union of their
    # top k -- exactly `union` of them (moe_route.h) -- with one record per token
    tokens: int = 1
    union: int = 0

    @property
    def slots(self) -> int:
        return self.union if self.tokens > 1 else self.k


class MoeRoute(Block):
    """The router's top k as an expert-slot record (offload_hw_kernels/moe_route.h).

      in   p      [1, n] fp16, L1: the router's probabilities
           table  [n, 64] bytes, L3: the expert table (module doc)
      out  rec    [k, 128] bytes, L1: the record, one slot per chosen expert, in order

    On the DM core: core code for the top k (FP16 bit patterns compared as integers, ties to
    the lower index, hwmodel.top_k16), then one iDMA copy per slot for its table entry.
    """

    name = "moe_route"

    def __init__(self, cfg: MoeRouteCfg = None, **params):
        self.cfg = cfg if cfg is not None else MoeRouteCfg(**params)
        c = self.cfg
        SnaxBingoKernelMoeRouteArgs(0, c.n, c.k, 0, 0, tokens=c.tokens, p_pitch=2 * c.n,
                                    union_n=c.union)                   # validates

    def idma_passes(self) -> int:
        return self.cfg.slots

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"p": PortSpec(Layout.ROW_MAJOR, DType.F16, (c.tokens, c.n),
                              mem_level=MemLevel.L1, cluster=c.cluster,
                              doc="router probabilities, a row per token"),
                "table": PortSpec(Layout.ROW_MAJOR, DType.I8, (c.n, TABLE_ENTRY_BYTES),
                                  mem_level=MemLevel.L3, doc="expert table")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"rec": record_spec(c.tokens * c.slots, c.cluster)}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        rec = g.l1(f"{self.name}_rec",
                   SnaxBingoKernelMoeRouteArgs.record_bytes(c.tokens * c.slots))
        nd = g.node("Route", ctx.dm, "__snax_bingo_kernel_moe_route",
                    SnaxBingoKernelMoeRouteArgs(bound["p"].handle, c.n, c.k,
                                                bound["table"].handle, rec, tokens=c.tokens,
                                                p_pitch=2 * c.n, union_n=c.union),
                    list(bound["p"].ends))
        return BlockResult(
            outputs={"rec": Port(self.outputs["rec"], rec, (nd,), name="rec")},
            inputs={"p": Port(self.inputs["p"], bound["p"].handle, (nd,), name="p"),
                    "table": Port(self.inputs["table"], bound["table"].handle, (nd,),
                                  name="table")},
            nodes=[nd])


@dataclass(frozen=True)
class SlotLoadCfg:
    slot: int
    field: str                 # gu_s or dn_s (or gu / dn)
    cols: int                  # fp16 values
    cluster: int = 0
    rec_slots: int = 6
    # The cluster's LoadStream: with weight rings, the tensor is PUSHED through the ring
    # with the expert's weights (right after them, in take order) and the load is a copy
    # out of L3 -- instead of a PULL across the half-duplex link, which waits for the push
    # stream to drain (~25 us for 5.6 KiB, measured) while the in-order DM core holds
    # every later task behind it.
    stream: object = field(default=None, compare=False, repr=False)


class SlotLoad(Block):
    """One tensor of a routed expert into this cluster's L1, from the address its record
    slot names (idma_copy_slot): the dequant factors of its gate|up or down.

      in   rec  the expert-slot record, in this L1
      out  y    [1, cols] fp16, L1
    """

    name = "slot_load"

    def __init__(self, cfg: SlotLoadCfg = None, **params):
        self.cfg = cfg if cfg is not None else SlotLoadCfg(**params)
        if self.cfg.field not in SnaxBingoKernelIdmaCopySlotArgs.FIELDS:
            raise ValueError(f"SlotLoad: field {self.cfg.field!r}.")

    def idma_passes(self) -> int:
        return 1

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"rec": record_spec(c.rec_slots, c.cluster)}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(Layout.ROW_MAJOR, DType.F16, (1, c.cols), mem_level=MemLevel.L1,
                              cluster=c.cluster, doc=f"slot {c.slot}'s {c.field}")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        y = g.l1(f"{self.name}_y", 2 * c.cols)
        rec = bound["rec"]
        rings = getattr(c.stream, "rings", None) if c.stream is not None else None
        if rings is not None:
            from .linear import WeightRings
            sl, fl, rl, seq = rings.take(c.cluster, nbytes=2 * c.cols,
                                         src=(c.slot << 8) | WeightRings.FIELDS[c.field],
                                         kind=1, offset=0)
            prev = rings.last.get(c.cluster)
            nd = g.node("LdSlot", ctx.dm, "__snax_bingo_kernel_idma_ring_load",
                        SnaxBingoKernelIdmaRingLoadArgs(fl, seq, sl, y, 2 * c.cols, rl),
                        list(rec.ends) + ([prev] if prev is not None else []))
            rings.last[c.cluster] = nd
        else:
            nd = g.node("LdSlot", ctx.dm, "__snax_bingo_kernel_idma_copy_slot",
                        SnaxBingoKernelIdmaCopySlotArgs(rec.handle, c.slot, c.field, 0, y,
                                                        2 * c.cols),
                        list(rec.ends))
        return BlockResult(outputs={"y": Port(self.outputs["y"], y, (nd,), name="y")},
                           inputs={"rec": Port(self.inputs["rec"], rec.handle, (nd,),
                                               name="rec")},
                           nodes=[nd])
