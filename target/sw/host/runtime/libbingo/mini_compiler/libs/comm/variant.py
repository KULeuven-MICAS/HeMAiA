# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""What a block COULD be, before the graph decides what it will be.

A block is a FAMILY, not one implementation. It is constructed with the tile and the
semantics; every layout or precision it does NOT pin is a knob it owns.

    variants()   the parameter dicts this block could be realised with
    respec(**p)  build one of them
    the cost hooks price it

The resolver in link.py picks one variant per block so that every boundary agrees, and the
block then emits whatever sub-DFG its choice implies. The application says which blocks
there are and pins the boundaries it cares about; the rest follow from the neighbours.

======================================================================================
LEGALITY IS BY CONSTRUCTION
======================================================================================

A variant is REALISED in order to be priced. So a configuration the block refuses -- an
RMSNorm asking for the col_major kernel at rows != 32, a Reshape whose conversion is not a
strided nest -- raises in its own constructor, with its own message, and drops out of the
search. There is no second copy of the rules here to drift out of step with the first.
This is the one property to protect when adding a block: state the refusal in __init__,
and the resolver inherits it for free.

======================================================================================
THE COST ORDER IS THE MACHINE'S, AND IT IS A CHOICE
======================================================================================

Cost compares LEXICOGRAPHICALLY in engine order: serialised cross-lane folds, then SIMD
passes, then xDMA passes, then iDMA. The SIMD core is this machine's constraint; the xDMA
is next because the relayouts of the whole chain queue on it; the iDMA is last because the
DM core is idle while the other two work, so a plain copy put there is very nearly free.

Minimising TOTAL passes would be a different objective and a worse one. The layer's first
RMSNorm is the case that separates them: the row-major arm is one pass and the col-major
arm is two, and the col-major arm is the one worth having because it takes ~2,000 cycles
off the busy engine to put one extra transfer on an idle one.

THESE ARE COUNTS, NOT CYCLES, DELIBERATELY. A cycle number is a constant correct for one
shape on one RTL build, with nothing to stop it rotting into a model nobody re-measures.
A fold count derived from the shape cannot rot. It does not predict a runtime; it ORDERS
the arms, and ordering is all a resolver needs.
"""

from dataclasses import dataclass

_HOOKS = (("folds", "simd_folds"), ("simd", "simd_passes"),
          ("xdma", "xdma_passes"), ("idma", "idma_passes"))


@dataclass(frozen=True, order=True)
class Cost:
    """What one realisation costs, per engine. Ordered lexicographically, field by field.

    `order=True` on the dataclass IS the comparison: folds first, then SIMD passes, then
    xDMA, then iDMA. Reordering the fields reorders the objective, which is why they are
    declared in engine order and not alphabetically.
    """

    folds: int = 0
    simd: int = 0
    xdma: int = 0
    idma: int = 0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self.folds + other.folds, self.simd + other.simd,
                    self.xdma + other.xdma, self.idma + other.idma)

    def __str__(self) -> str:
        return f"{self.folds}f/{self.simd}s/{self.xdma}x/{self.idma}i"


def cost_of(blk) -> Cost:
    """Price a realised block from whatever hooks it implements.

    A block that implements none costs nothing and is ranked on its neighbours alone --
    which is right: a block with one variant has nothing to choose between, so its cost
    cannot change the answer.
    """
    vals = {}
    for field_name, hook in _HOOKS:
        fn = getattr(blk, hook, None)
        vals[field_name] = int(fn() or 0) if callable(fn) else 0
    return Cost(**vals)


@dataclass(frozen=True)
class Variant:
    """One realisable configuration of a block, and the boundary it would then present."""

    params: dict            # what respec() was called with; {} is the block as constructed
    block: object           # the realised Block -- built, so its refusals have already run
    cost: Cost

    @property
    def inputs(self) -> dict:
        return self.block.inputs

    @property
    def outputs(self) -> dict:
        return self.block.outputs

    @property
    def needs(self) -> dict:
        return getattr(self.block, "needs", self.block.inputs)

    def describe(self) -> str:
        ins = ", ".join(f"{k}:{v.describe()}" for k, v in self.inputs.items())
        outs = ", ".join(f"{k}:{v.describe()}" for k, v in self.outputs.items())
        return f"{ins} -> {outs}  [{self.cost}]"


def variants_of(template, *, on_refusal=None) -> list:
    """Every legal realisation of a block template, priced.

    `on_refusal(params, exc)` is called for each configuration the block rejects, so a
    caller that ends up with nothing can report the block's OWN reasons rather than a bare
    "no legal variant".
    """
    out, seen = [], set()
    for params in template.variants():
        try:
            blk = template.respec(**params) if params else template
        except (ValueError, KeyError, TypeError) as e:
            if on_refusal is not None:
                on_refusal(params, e)
            continue
        # Two different parameter dicts can name the same realisation -- a free field that
        # the block pins itself, for instance. Keep the first; the later ones would only
        # widen the search without widening the answer.
        try:
            key = (tuple(sorted((k, str(v.layout), str(v.dtype), tuple(v.shape),
                                 str(v.mem_level)) for k, v in blk.inputs.items())),
                   tuple(sorted((k, str(v.layout), str(v.dtype), tuple(v.shape),
                                 str(v.mem_level)) for k, v in blk.outputs.items())))
        except (ValueError, KeyError, TypeError) as e:
            if on_refusal is not None:
                on_refusal(params, e)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(Variant(params, blk, cost_of(blk)))
    return out


def free_fields(cfg, names) -> list:
    """Which of `names` this cfg left unset. The block's knobs, as opposed to its contract.

    A field the caller pinned is a CONSTRAINT and must survive into every variant; a field
    left None is the block's to choose. Keeping that distinction in one helper is what
    stops a block from quietly overruling something the caller asked for.
    """
    return [n for n in names if getattr(cfg, n) is None]


