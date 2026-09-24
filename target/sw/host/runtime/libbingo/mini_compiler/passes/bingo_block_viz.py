# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""One picture per block: the sub-DFG each libs block built, drawn at the node level.

final_dfg.png draws the whole LOWERED graph, which answers a different question. Most of
its nodes are the compiler's own -- llm_chain_4cluster lowers 181 task nodes into 663, 482
of them dummy set/check nodes -- and its labels are bare ids, so one block's work cannot be
picked out of it. A block's author needs something narrower: which nodes did THIS block
create, on which engine, in what dependency order, handing which buffers to which kernel,
and how is it joined to the blocks around it.

WHAT IS DRAWN is the graph as the Pipeline left it, before bingo_compile_dfg adds the
entry, exit and dummy nodes: the dependencies a block states, not the handshakes they are
lowered to. The task ids are the final ones -- an id is handed out once, at creation, and
never renumbered -- so a node here is the same row of final_dfg.csv and the same task id a
core reads from CSR 0x5fe in a trace.

THE LAYOUT, one picture per block:

    rows     one lane per (cluster, core) the block uses, grouped by cluster. The lane
             names the engine and so does the node's colour, because a node on the wrong
             hart does not fault: it programs that hart's accelerator and reports success.
    columns  dependency depth INSIDE the block. A node sits one column right of its
             deepest predecessor, so every arrow points right and a column holds work that
             could run concurrently. Nodes of one lane at one depth stack inside the lane,
             and a node keeps the row of its predecessor on the same core, so a chain
             stays on one line. A graph deeper than a band wraps onto the next band, and
             an edge that crosses bands ends in a stub naming the node at the other end.
    left     the input ports -- spec, buffer, and which stage produced it -- with a dashed
             arrow to each node that reads it first.
    right    the output ports -- spec, buffer, and which stages consume it.
    ghosts   every other edge across the block's boundary (a check reading an output, a
             node the application built by hand, a dependency a block added on another
             block's node) as a dashed box naming the node and the stage it belongs to. An
             edge from outside is where a missing or surprising dependency hides, so none
             is left out.

GATING EDGES ARE COUNTED, NOT DRAWN. link() holds every graph source of a block behind
the last node of each of its producers, so a block with twelve inputs and eight sources
has ninety-six of them; drawn, they bury everything else. Each port card says how many
nodes its producer gates, each gated node carries a "gated by N" tag, and the .txt lists
every one.

Edges inside the block are drawn after transitive reduction: an edge implied by a longer
path orders nothing new, and drawing it only hides the ones that do. The header counts
what was left out.

A kernel's buffers are printed under their ARGUMENT NAMES (src=, dst=, A=, D=) rather
than sorted into reads and writes, because no kernel-args class declares a direction and
the field names are not uniform (see bingo_liveness). A guessed direction that was wrong
would mislead exactly the reader who is hunting a race.

NEVER ON THE BUILD PATH. The caller catches everything: a missing plotting library or a
bug in here must not fail a compile.
"""

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from bingo_mem_handle import (BingoMemAlloc, BingoMemAllocView, BingoMemFixedAddr,
                              BingoMemSymbol)

# Where the pictures go, relative to the app's output directory. The directory is owned by
# this module and emptied of earlier pictures on every run: a stage that was renamed or
# removed would otherwise leave behind a picture that no longer matches the app.
SUBDIR = "block_dfg"


# ======================================================================================
# What the Pipeline records
# ======================================================================================

@dataclass
class PortRecord:
    """One port of a built block, reduced to what the pictures need."""
    name: str
    spec: str                    # PortSpec.describe(): "A/i8 (256, 128) in L1 on cl0"
    buffer: str                  # the handle the port names
    ends: list                   # this block's nodes at the port: first readers / last writers
    peer: str = ""               # inputs: the producing "stage.port", or where it was staged
    peer_ends: list = field(default_factory=list)   # inputs: the producer's end nodes
    cluster: Optional[int] = None    # whose L1 holds the buffer; None outside L1


@dataclass
class BlockRecord:
    """What one Pipeline item built. Filled in by Pipeline.run(), drawn by render_blocks()."""
    index: int                   # build order, which is node creation order
    name: str                    # the stage name -- also the prefix of its nodes and handles
    kind: str                    # the block class, "source", or "raw"
    nodes: list                  # every node it created, in creation order
    inputs: list = field(default_factory=list)      # PortRecord
    outputs: list = field(default_factory=list)     # PortRecord
    sources: list = field(default_factory=list)     # BlockResult.sources: what link() gates
    params: str = ""             # the realisation the resolver picked
    cost: str = ""
    doc: str = ""


# ======================================================================================
# Presentation constants
# ======================================================================================

# role -> (label, colour). Okabe-Ito hues: they stay distinct under the common forms of
# colour blindness, and none of them is the vermillion reserved for cross-cluster edges.
_ENGINES = {
    "gemm": ("GEMM", "#E69F00"),
    "simd": ("SIMD", "#009E73"),
    "xdma": ("xDMA", "#0072B2"),
    "dm": ("iDMA", "#CC79A7"),
    "host": ("host", "#6E6E6E"),
}
_ENGINE_ORDER = ["gemm", "simd", "xdma", "dm", "host"]

_EDGE_SAME_LANE = "#8A8A8A"
_EDGE_CROSS_CORE = "#2B2B2B"
_EDGE_CROSS_CLUSTER = "#D55E00"
_EDGE_PORT = "#A87900"
_EDGE_GHOST = "#8F8F8F"
_CARD_FILL = "#FFF6DB"
_GHOST_FILL = "#F3F3F3"

# Geometry, in inches: the axes are laid out one data unit to the inch, so a font size in
# points means the same thing on every picture however large the block is.
_BOX_W = 2.6             # a node
_PITCH_X = 3.25          # column pitch: the box plus room for arrowheads
_LINE = 0.148            # one body line
_TITLE_LINE = 0.19       # the node's first line
_LANE_PAD = 0.14         # above and below the boxes inside a lane
_ROW_GAP = 0.14          # between two stacked boxes of one lane
_LABEL_W = 1.55          # the lane-label strip
_MARGIN_W = 3.3          # the port / ghost column on either side
_CARD_W = 2.85           # a port card or ghost box
_CARD_LINE = 0.142
_BAND_GAP = 0.5          # between two bands of one picture
_DEPTH_HDR = 0.28        # the "depth n" row above each band
_MAX_BAND_COLS = 10      # depths per band before the picture wraps
_CHARS = 44              # body characters that fit one box line at the body font size
_CARD_CHARS = 48

_FS_TITLE = 13.5
_FS_HEAD = 8.6
_FS_NODE_TITLE = 7.8
_FS_BODY = 6.5
_FS_LANE = 7.4
_FS_CARD = 6.7
_MONO = "DejaVu Sans Mono"


# ======================================================================================
# Text
# ======================================================================================

_KERNEL_PREFIXES = ("__snax_bingo_kernel_", "__host_bingo_kernel_", "__snax_kernel_",
                    "__host_kernel_")

# Argument names shortened for the box. Anything not listed loses a trailing "_addr".
_ARG_SHORT = {
    "input_A_addr": "A", "input_B_addr": "B", "input_C_addr": "C", "output_D_addr": "D",
    "src_addr": "src", "dst_addr": "dst", "input_addr": "in", "output_addr": "out",
    "weight_addr": "w", "dst_list": "dst",
}


def _tint(colour: str, a: float = 0.2) -> str:
    """`colour` at strength `a` over white, OPAQUE. A translucent fill lets an arrow that
    passes behind a box show through its text, where it reads as strikethrough."""
    rgb = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 - (255 - c) * a):02x}" for c in rgb)


def _fit(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _kernel_short(kname: str) -> str:
    for p in _KERNEL_PREFIXES:
        if kname.startswith(p):
            return kname[len(p):]
    return kname


def _common_scope(names) -> str:
    """The longest `xxx_` prefix every name shares: the ctx.scope() of a raw thunk."""
    names = [n for n in names if n]
    if not names:
        return ""
    p = os.path.commonprefix(names)
    cut = p.rfind("_")
    return p[:cut + 1] if cut > 0 else ""


def _short_name(node, prefix: str) -> str:
    """The node's name without its block's prefix and without the `_clN` every name ends in."""
    nm = node.node_name or ""
    if prefix and nm.startswith(prefix):
        nm = nm[len(prefix):]
    return re.sub(r"_cl\d+$", "", nm) or nm


def _is_handle(v) -> bool:
    return isinstance(v, (BingoMemAlloc, BingoMemAllocView, BingoMemSymbol, BingoMemFixedAddr))


def _handle_text(h, prefix: str) -> Optional[str]:
    """How a buffer is written in a box: its handle name, where it is and any offset."""
    def strip(nm):
        return nm[len(prefix):] if prefix and nm.startswith(prefix) else nm

    if isinstance(h, BingoMemAllocView):
        base = _handle_text(h.base, prefix)
        return f"{base}+{h.offset}" if h.offset else base
    if isinstance(h, BingoMemAlloc):
        where = "" if h.mem_level == "L1" else f"@{h.mem_level}"
        off = f"+{h.offset}" if h.offset else ""
        return f"{strip(h.name)}{off}{where}"
    if isinstance(h, BingoMemSymbol):
        off = f"+{h.offset}" if h.offset else ""
        return f"{h.symbol_name}{off}@L3"
    if isinstance(h, BingoMemFixedAddr):
        return f"0x{h.address:x}"
    return None


def _args_of(node, prefix: str):
    """(buffer args, scalar args) of a node, as ("name", "text") pairs in declaration order."""
    args = getattr(node, "kernel_args", None)
    if args is None:
        return [], []
    bufs, scalars = [], []
    for k, v in vars(args).items():
        if k.startswith("_"):
            continue
        short = _ARG_SHORT.get(k, k[:-5] if k.endswith("_addr") else k)
        if _is_handle(v):
            bufs.append((short, _handle_text(v, prefix)))
        elif isinstance(v, (list, tuple)) and v and all(_is_handle(x) for x in v):
            bufs.append((short, "[" + ",".join(_handle_text(x, prefix) for x in v) + "]"))
        elif isinstance(v, bool):
            scalars.append((k, str(int(v))))
        elif isinstance(v, int):
            # A packed register value reads better in hex; a size or a count in decimal.
            scalars.append((k, f"0x{v:x}" if v >= (1 << 20) and "size" not in k else str(v)))
        elif isinstance(v, float):
            scalars.append((k, f"{v:g}"))
        elif isinstance(v, str) and len(v) <= 24:
            scalars.append((k, v))
        elif isinstance(v, (list, tuple)) and len(v) <= 6 and all(isinstance(x, int) for x in v):
            scalars.append((k, "[" + ",".join(str(x) for x in v) + "]"))
    return bufs, scalars


def _wrap_pairs(pairs, width: int, max_lines: int) -> List[str]:
    """Pack "k=v" items onto at most `max_lines` lines of `width` characters."""
    lines, cur = [], ""
    for k, v in pairs:
        item = f"{k}={v}"
        if not cur:
            cur = item
        elif len(cur) + 2 + len(item) <= width:
            cur += "  " + item
        else:
            lines.append(cur)
            cur = item
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _fit(lines[-1] + "  …", width)
    return [_fit(ln, width) for ln in lines]


# ======================================================================================
# What one picture shows, before any geometry
# ======================================================================================

class _Machine:
    """What a lane is called and coloured, from the core-role map the Pipeline was built with."""

    def __init__(self, roles: Optional[dict], dfg):
        self.role_of = {v: k for k, v in (roles or {}).items()}
        self.multi_chip = len(getattr(dfg, "chiplet_ids", [0]) or [0]) > 1

    def engine(self, core: int):
        role = self.role_of.get(core)
        if role in _ENGINES:
            return role, _ENGINES[role][0], _ENGINES[role][1]
        return None, f"core {core}", "#999999"

    def where(self, n) -> str:
        role, eng, _c = self.engine(n.assigned_core_id)
        return "host" if role == "host" else f"cl{n.assigned_cluster_id} {eng}"

    def lane_label(self, lane) -> str:
        chip, cl, core = lane
        role, eng, _ = self.engine(core)
        if role == "host":
            return "host" + (f" 0x{chip:02x}" if self.multi_chip else "")
        return f"c{core} · {eng}"

    def cluster_label(self, lane) -> str:
        chip, cl, core = lane
        if self.role_of.get(core) == "host":
            return "HOST"
        return (f"0x{chip:02x} " if self.multi_chip else "") + f"CLUSTER {cl}"


def _lane(n):
    return (n.assigned_chiplet_id, n.assigned_cluster_id, n.assigned_core_id)


class _World:
    """What every picture of one graph shares: who owns each node, who consumes each port."""

    def __init__(self, dfg, records):
        self.dfg = dfg
        self.records = records
        self.machine = _Machine(getattr(dfg, "block_roles", None), dfg)
        self.owner = {n: r for r in records for n in r.nodes}
        self.scope = {r.name: (f"{r.name}_" if r.kind != "raw"
                               else _common_scope([n.node_name for n in r.nodes]))
                      for r in records}
        # (producer stage, port) -> [(consumer stage, port, the consumer's first readers)]
        by_name = {r.name: r for r in records}
        direct = defaultdict(list)
        for r in records:
            for p in r.inputs:
                if "." in p.peer:
                    s, q = p.peer.split(".", 1)
                    direct[(s, q)].append((r.name, p.name, list(p.ends)))

        def resolve(key, seen=()):
            # A view (Slice, TransposeView) builds no node, so "consumed by the view" says
            # nothing about who reads the bytes. Follow it to the stage that does.
            out = []
            for (s, q, cends) in direct.get(key, []):
                r = by_name.get(s)
                if r is not None and not r.nodes and r.outputs and s not in seen:
                    for o in r.outputs:
                        for (s2, q2, e2) in resolve((s, o.name), seen + (s,)):
                            out.append((f"{s}→{s2}", q2, e2))
                else:
                    out.append((s, q, cends))
            return out
        self.consumers = {k: resolve(k) for k in list(direct)}

    def owner_name(self, n) -> str:
        r = self.owner.get(n)
        return r.name if r is not None else "the application"

    def name(self, n) -> str:
        r = self.owner.get(n)
        return _short_name(n, self.scope.get(r.name, "") if r is not None else "")


class _Scene:
    """One block's picture, as content: nodes, depths, rows, bands, and its boundary."""

    def __init__(self, rec: BlockRecord, w: _World):
        self.rec, self.w, self.machine = rec, w, w.machine
        dfg = w.dfg
        self.prefix = w.scope.get(rec.name, "")
        self.nodes = list(rec.nodes)
        self.inside = set(self.nodes)
        # A plain DiGraph, not dfg.subgraph(): networkx builds views and reductions by
        # calling the graph's own class with no arguments, which BingoDFG does not accept.
        sub = nx.DiGraph()
        sub.add_nodes_from(self.nodes)
        sub.add_edges_from((u, v) for u in self.nodes for v in dfg.successors(u)
                           if v in self.inside)
        self.edges = list(sub.edges())
        try:
            red = nx.transitive_reduction(sub)
            self.drawn = [e for e in self.edges if red.has_edge(*e)]
        except Exception:          # not a DAG: draw everything and let the header say so
            self.drawn = list(self.edges)
        self.implied = len(self.edges) - len(self.drawn)

        # ---- depth, rows, bands -----------------------------------------------------------
        depth = {}
        for n in nx.lexicographical_topological_sort(sub, key=lambda n: n.node_id):
            depth[n] = max((depth[p] + 1 for p in sub.predecessors(n)), default=0)
        self.depth = depth
        role_of = self.machine.role_of
        self.lanes = sorted({_lane(n) for n in self.nodes},
                            key=lambda ln: (ln[0], role_of.get(ln[2]) == "host", ln[1], ln[2]))
        # A node takes the row of its latest predecessor on the same core when that row is
        # free at its depth, so a chain on one engine reads as one straight line.
        self.row, taken = {}, defaultdict(set)
        for n in sorted(self.nodes, key=lambda n: (depth[n], n.node_id)):
            ln, used = _lane(n), taken[(_lane(n), depth[n])]
            prefs = [self.row[p] for p in sorted(sub.predecessors(n),
                                                 key=lambda p: (-depth[p], -p.node_id))
                     if _lane(p) == ln]
            r = next((s for s in prefs if s not in used), None)
            if r is None:
                r = 0
                while r in used:
                    r += 1
            self.row[n] = r
            used.add(r)
        self.nrows = {ln: 1 + max(self.row[n] for n in self.nodes if _lane(n) == ln)
                      for ln in self.lanes}
        ndepth = max(depth.values()) + 1 if depth else 1
        nb = math.ceil(ndepth / _MAX_BAND_COLS)
        per = math.ceil(ndepth / nb)
        self.bands = [list(range(s, min(s + per, ndepth))) for s in range(0, ndepth, per)]
        self.band_of = {d: b for b, ds in enumerate(self.bands) for d in ds}
        self.cols = per if nb > 1 else ndepth

        # ---- the text in each box ---------------------------------------------------------
        many = len(self.nodes) > 80
        self.body = {}
        for n in self.nodes:
            bufs, scalars = _args_of(n, self.prefix)
            lines = [_fit(_kernel_short(n.kernel_name or "?"), _CHARS)]
            lines += _wrap_pairs(bufs, _CHARS, 1 if many else 2)
            if not many:
                lines += _wrap_pairs(scalars, _CHARS, 1)
            self.body[n] = lines
        nbody = max([len(v) for v in self.body.values()] or [1])
        self.box_h = _TITLE_LINE + nbody * _LINE + 0.12

        # ---- the boundary -----------------------------------------------------------------
        self.ext_in = [(u, v) for v in self.nodes for u in dfg.predecessors(v)
                       if u not in self.inside]
        self.ext_out = [(u, v) for u in self.nodes for v in dfg.successors(u)
                        if v not in self.inside]
        self.edge_class = {}               # boundary edge -> "port <p>" | "gate <p>" | "other"
        self.gated = Counter()             # inside node -> gating edges into it
        sources = set(rec.sources)

        # An input edge belongs to the card of the port whose producer it comes from. From
        # that producer's LAST node to one of this block's sources it is link()'s gate;
        # anything else from it is data, and is drawn from the card.
        self.in_cards = []
        for p in rec.inputs:
            stage = p.peer.split(".", 1)[0] if "." in p.peer else None
            self.in_cards.append({"port": p, "stage": stage, "gates": 0, "also": [],
                                  "to": [n for n in p.ends if n in self.inside]})
        ghost_in = []
        for (u, v) in self.ext_in:
            c = next((c for c in self.in_cards if u in c["port"].peer_ends
                      or (c["stage"] and w.owner_name(u) == c["stage"])), None)
            if c is None:
                ghost_in.append((u, v))
                self.edge_class[(u, v)] = "other"
                continue
            p = c["port"]
            if (v in sources and p.peer_ends and u is p.peer_ends[-1]
                    and v not in p.ends):
                c["gates"] += 1
                self.gated[v] += 1
                self.edge_class[(u, v)] = f"gate {p.name}"
            else:
                # Not a declared reader, but ordered after the producer all the same: the
                # block added the edge. Drawn from the card, in a lighter style.
                if v not in c["to"] and v not in c["also"]:
                    c["also"].append(v)
                self.edge_class[(u, v)] = (f"port {p.name}" if v in c["to"]
                                           else f"after {p.name}'s producer")

        # An output edge belongs to the card when it leaves one of the port's last writers
        # for a stage bound to that port.
        self.out_cards = []
        for p in rec.outputs:
            users = w.consumers.get((rec.name, p.name), [])
            uends = {n for (_s, _q, e) in users for n in e}
            stages = {s.split("→")[-1] for (s, _q, _e) in users}
            self.out_cards.append({"port": p, "users": users, "uends": uends,
                                   "stages": stages, "gates": Counter(),
                                   "from": [n for n in p.ends if n in self.inside]})
        ghost_out = []
        for (u, v) in self.ext_out:
            c = next((c for c in self.out_cards if u in c["port"].ends
                      and (v in c["uends"] or w.owner_name(v) in c["stages"])), None)
            if c is None:
                ghost_out.append((u, v))
                self.edge_class[(u, v)] = "other"
                continue
            own = w.owner.get(v)
            if (own is not None and v in set(own.sources) and v not in c["uends"]
                    and u is c["port"].ends[-1]):
                c["gates"][own.name] += 1
                self.edge_class[(u, v)] = f"gate {c['port'].name}"
            else:
                self.edge_class[(u, v)] = f"port {c['port'].name}"
        self.n_gate = sum(1 for k in self.edge_class.values() if k.startswith("gate"))
        self.ghost_in = self._ghosts(ghost_in, 0)
        self.ghost_out = self._ghosts(ghost_out, 1)

    def _ghosts(self, edges, far: int):
        """Group boundary edges by the outside node; fold a stage's crowd into one box.

        Returns [(label lines, outside nodes, inside nodes)]. `far` is 0 for predecessors
        (the outside node is the edge's source) and 1 for successors."""
        by_node = defaultdict(set)
        for e in edges:
            by_node[e[far]].add(e[1 - far])
        by_owner = defaultdict(list)
        for x in sorted(by_node, key=lambda n: n.node_id):
            by_owner[self.w.owner_name(x)].append(x)
        out = []
        for own, xs in by_owner.items():
            if len(xs) > 3:
                inner = set().union(*(by_node[x] for x in xs))
                out.append(([f"{len(xs)} nodes of {own}",
                             f"#{xs[0].node_id}…#{xs[-1].node_id}"], xs, inner))
                continue
            for x in xs:
                out.append(([_fit(f"#{x.node_id} {self.w.name(x)}", _CARD_CHARS),
                             _fit(f"{own} · {self.machine.where(x)}", _CARD_CHARS)],
                            [x], by_node[x]))
        return out


# ======================================================================================
# Geometry
# ======================================================================================

def _in_card_lines(c) -> List[str]:
    p = c["port"]
    ids = ",".join(f"#{n.node_id}" for n in p.peer_ends[:4])
    if len(p.peer_ends) > 4:
        ids += ",…"
    lines = [f"IN  {p.name}",
             _fit(p.spec, _CARD_CHARS),
             _fit(f"buffer {p.buffer}", _CARD_CHARS),
             _fit(f"from {p.peer or '?'}" + (f" ({ids})" if ids else ""), _CARD_CHARS)]
    if c["gates"]:
        lines.append(_fit(f"its producer gates {_plural(c['gates'], 'source')} here",
                          _CARD_CHARS))
    return lines


def _out_card_lines(c) -> List[str]:
    p = c["port"]
    lines = [f"OUT {p.name}",
             _fit(p.spec, _CARD_CHARS),
             _fit(f"buffer {p.buffer}", _CARD_CHARS)]
    names = [f"{s}.{q}" for (s, q, _e) in c["users"]]
    if names:
        lines.append(_fit("to " + ", ".join(names[:3])
                          + (f" +{len(names) - 3} more" if len(names) > 3 else ""),
                          _CARD_CHARS))
    else:
        lines.append("bound to no later stage")
    if c["gates"]:
        lines.append(_fit("gates " + ", ".join(f"{_plural(k, 'source')} of {s}"
                                                for s, k in c["gates"].items()), _CARD_CHARS))
    return lines


def _layout(sc: _Scene) -> None:
    """Node centres, band extents and the margin items, in inches down from the top."""
    rec = sc.rec
    sc.head_h = 1.3 if (rec.doc and (rec.params or rec.cost)) else 1.08
    sc.x_nodes = _LABEL_W + _MARGIN_W
    sc.band_w = sc.cols * _PITCH_X
    sc.fig_w = max(sc.x_nodes + sc.band_w + _MARGIN_W + 0.25, 13.0)
    sc.lane_h = {ln: 2 * _LANE_PAD + sc.nrows[ln] * sc.box_h + (sc.nrows[ln] - 1) * _ROW_GAP
                 for ln in sc.lanes}
    sc.lane_t, t = {}, 0.0
    for ln in sc.lanes:
        sc.lane_t[ln] = t
        t += sc.lane_h[ln]
    sc.lanes_h = t
    sc.band_top, t = [], sc.head_h
    for b in range(len(sc.bands)):
        sc.band_top.append(t)
        t += _DEPTH_HDR + sc.lanes_h + (_BAND_GAP if b < len(sc.bands) - 1 else 0)
    body_bottom = t

    sc.pos = {}
    for n in sc.nodes:
        d = sc.depth[n]
        b = sc.band_of[d]
        x = sc.x_nodes + (d - sc.bands[b][0] + 0.5) * _PITCH_X
        t = (sc.band_top[b] + _DEPTH_HDR + sc.lane_t[_lane(n)] + _LANE_PAD
             + sc.row[n] * (sc.box_h + _ROW_GAP) + sc.box_h / 2)
        sc.pos[n] = (x, t)

    def card_h(lines):
        return 0.12 + len(lines) * _CARD_LINE

    sc.in_items = [("card", _in_card_lines(c), c["to"] + c["also"], c) for c in sc.in_cards]
    sc.in_items += [("ghost", lines, list(inner), xs) for lines, xs, inner in sc.ghost_in]
    sc.out_items = [("card", _out_card_lines(c), c["from"], c) for c in sc.out_cards]
    sc.out_items += [("ghost", lines, list(inner), xs) for lines, xs, inner in sc.ghost_out]

    def place(items):
        # Each item as near as it can get to the nodes it connects to, top to bottom,
        # never overlapping the one above it.
        floor = sc.head_h + _DEPTH_HDR
        want = []
        for (_k, lines, ends, _o) in items:
            h = card_h(lines)
            ts = [sc.pos[e][1] for e in ends if e in sc.pos]
            want.append(((sum(ts) / len(ts)) if ts else floor + h / 2, h))
        out = [None] * len(items)
        for i in sorted(range(len(items)), key=lambda i: want[i][0]):
            c, h = want[i]
            top = max(c - h / 2, floor)
            out[i] = (top + h / 2, h)
            floor = top + h + 0.12
        return out, floor

    sc.in_place, bot_l = place(sc.in_items)
    sc.out_place, bot_r = place(sc.out_items)
    legend_h = 0.95
    sc.fig_h = max(body_bottom, bot_l, bot_r) + 0.2 + legend_h
    sc.legend_t = sc.fig_h - legend_h
    sc.x_in = _LABEL_W + 0.12
    sc.x_out = sc.x_nodes + sc.band_w + 0.3


# ======================================================================================
# Drawing
# ======================================================================================

def _draw(sc: _Scene, path: str, dpi: int = 100) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

    _layout(sc)
    W, H = sc.fig_w, sc.fig_h
    fig = plt.figure(figsize=(W, H), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def Y(t):
        return H - t

    rec, m = sc.rec, sc.machine
    half_w, half_h = _BOX_W / 2, sc.box_h / 2

    # ---- header ------------------------------------------------------------------------
    ax.add_patch(Rectangle((0, Y(sc.head_h - 0.1)), W, sc.head_h - 0.1, color="#F2F2F2",
                           zorder=0, lw=0))
    title = ax.text(0.25, Y(0.36), f"[{rec.index:02d}]  {rec.name}", fontsize=_FS_TITLE,
                    fontweight="bold", va="center")
    fig.canvas.draw()
    bb = title.get_window_extent().transformed(ax.transData.inverted())
    ax.text(bb.x1 + 0.2, Y(0.36), f"·  {rec.kind}", fontsize=_FS_TITLE, va="center",
            color="#444444")
    eng = Counter(m.engine(n.assigned_core_id)[1] for n in sc.nodes)
    clusters = sorted({n.assigned_cluster_id for n in sc.nodes
                       if m.role_of.get(n.assigned_core_id) != "host"})
    ids = sorted(n.node_id for n in sc.nodes)
    parts = [f"{_plural(len(sc.nodes), 'node')} ("
             + ", ".join(f"{c} {e}" for e, c in eng.most_common()) + ")"]
    if clusters:
        parts.append(f"cluster{'s' if len(clusters) > 1 else ''} "
                     + ",".join(map(str, clusters)))
    parts.append(f"task ids #{ids[0]}…#{ids[-1]}")
    parts.append(f"{_plural(len(sc.edges), 'edge')} inside"
                 + (f" ({sc.implied} implied by a longer path, not drawn)" if sc.implied else ""))
    parts.append(f"{len(sc.ext_in)} in from outside, {len(sc.ext_out)} out"
                 + (f" ({sc.n_gate} of them gating)" if sc.n_gate else ""))
    ax.text(0.25, Y(0.68), "  ·  ".join(parts), fontsize=_FS_HEAD, va="center")
    t = 0.9
    line3 = []
    if rec.params:
        line3.append(f"realisation: {rec.params}")
    if rec.cost:
        line3.append(f"cost: {rec.cost}")
    if line3:
        ax.text(0.25, Y(t), "   ·   ".join(line3), fontsize=_FS_HEAD, va="center",
                color="#333333")
        t += 0.22
    if rec.doc:
        ax.text(0.25, Y(t), _fit(rec.doc, 200), fontsize=_FS_HEAD, va="center",
                color="#555555", style="italic")

    # ---- bands, lanes, depth columns ---------------------------------------------------
    for b, ds in enumerate(sc.bands):
        top = sc.band_top[b]
        lanes_top = top + _DEPTH_HDR
        for d in ds:
            x = sc.x_nodes + (d - ds[0] + 0.5) * _PITCH_X
            ax.text(x, Y(top + _DEPTH_HDR / 2), f"depth {d}", fontsize=7, color="#8A8A8A",
                    va="center", ha="center")
        for i, ln in enumerate(sc.lanes):
            lt = lanes_top + sc.lane_t[ln]
            lh = sc.lane_h[ln]
            ax.add_patch(Rectangle((_LABEL_W, Y(lt + lh)), W - _LABEL_W - 0.1, lh,
                                   color="#F6F6F6" if i % 2 == 0 else "#FFFFFF", lw=0,
                                   zorder=0))
            role, engname, colr = m.engine(ln[2])
            ax.add_patch(Rectangle((_LABEL_W - 0.1, Y(lt + lh) + 0.05), 0.07, lh - 0.1,
                                   color=colr, lw=0, zorder=1))
            ax.text(_LABEL_W - 0.18, Y(lt + lh / 2), m.lane_label(ln), fontsize=_FS_LANE,
                    ha="right", va="center")
        # cluster groups: a rule above each and the name down the left
        groups = []
        for ln in sc.lanes:
            lab = m.cluster_label(ln)
            if not groups or groups[-1][0] != lab:
                groups.append([lab, sc.lane_t[ln], 0.0])
            groups[-1][2] += sc.lane_h[ln]
        for k, (lab, gt, gh) in enumerate(groups):
            if k:
                ax.plot([0.1, W - 0.1], [Y(lanes_top + gt)] * 2, color="#9A9A9A", lw=1.1,
                        zorder=1)
            ax.text(0.16, Y(lanes_top + gt + gh / 2), lab, fontsize=7.6, fontweight="bold",
                    rotation=90, va="center", ha="center", color="#555555")
        if len(sc.bands) > 1:
            ax.text(W - 0.2, Y(top + _DEPTH_HDR / 2), f"band {b + 1} of {len(sc.bands)}",
                    fontsize=7, ha="right", color="#8A8A8A", va="center")

    # ---- edges inside the block --------------------------------------------------------
    def arrow(p0, p1, color, lw, ls="-", rad=0.0, alpha=1.0, z=2):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, color=color,
                                     lw=lw, linestyle=ls, alpha=alpha, zorder=z,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                                     shrinkB=0))

    kinds_used = set()
    for u, v in sc.drawn:
        (xu, tu), (xv, tv) = sc.pos[u], sc.pos[v]
        if (u.assigned_chiplet_id, u.assigned_cluster_id) != \
                (v.assigned_chiplet_id, v.assigned_cluster_id):
            color, lw, kind = _EDGE_CROSS_CLUSTER, 1.9, "xcl"
        elif _lane(u) == _lane(v):
            color, lw, kind = _EDGE_SAME_LANE, 1.1, "lane"
        else:
            color, lw, kind = _EDGE_CROSS_CORE, 1.25, "core"
        kinds_used.add(kind)
        bu, bv = sc.band_of[sc.depth[u]], sc.band_of[sc.depth[v]]
        if bu != bv:
            # Across bands: a stub out of the source and one into the destination, each
            # naming the node at the other end, instead of a line through the bands between.
            kinds_used.add("stub")
            xe = sc.x_nodes + sc.band_w + 0.05
            arrow((xu + half_w, Y(tu)), (xe, Y(tu)), color, lw, ls=(0, (4, 2)))
            ax.text(xe + 0.05, Y(tu), f"#{v.node_id}", fontsize=6.6, color=color, va="center")
            xs = sc.x_nodes - 0.15
            arrow((xs, Y(tv)), (xv - half_w, Y(tv)), color, lw, ls=(0, (4, 2)))
            ax.text(xs - 0.05, Y(tv), f"#{u.node_id}", fontsize=6.6, color=color,
                    va="center", ha="right")
            continue
        rad = 0.0
        if _lane(u) == _lane(v) and sc.row[u] == sc.row[v]:
            # a same-row edge that would run through a box in between arcs over it
            if any(_lane(n) == _lane(u) and sc.row[n] == sc.row[u]
                   and sc.band_of[sc.depth[n]] == bu and xu < sc.pos[n][0] < xv
                   for n in sc.nodes):
                rad = -0.25
        arrow((xu + half_w, Y(tu)), (xv - half_w, Y(tv)), color, lw, rad=rad)

    # ---- nodes -------------------------------------------------------------------------
    for n in sc.nodes:
        x, t = sc.pos[n]
        role, engname, colr = m.engine(n.assigned_core_id)
        ax.add_patch(FancyBboxPatch((x - half_w, Y(t + half_h)), _BOX_W, sc.box_h,
                                    boxstyle="round,pad=0.0,rounding_size=0.07",
                                    facecolor=_tint(colr), edgecolor=colr, lw=1.5, zorder=3))
        ax.text(x - half_w + 0.09, Y(t - half_h + 0.13),
                _fit(f"#{n.node_id}  {_short_name(n, sc.prefix)}", 30),
                fontsize=_FS_NODE_TITLE, fontweight="bold", va="center", zorder=4)
        if sc.gated[n]:
            ax.text(x + half_w - 0.06, Y(t - half_h + 0.13), f"gated by {sc.gated[n]}",
                    fontsize=5.9, ha="right", va="center", zorder=5, color="#5C4300",
                    bbox=dict(boxstyle="round,pad=0.18", fc=_CARD_FILL, ec=_EDGE_PORT,
                              lw=0.7))
        for k, line in enumerate(sc.body[n]):
            ax.text(x - half_w + 0.09, Y(t - half_h + _TITLE_LINE + 0.07 + (k + 0.5) * _LINE),
                    line, fontsize=_FS_BODY, family=_MONO, va="center", zorder=4,
                    color="#000000" if k == 0 else "#262626")

    # ---- the margins -------------------------------------------------------------------
    def card(x, tc, h, lines, ghost):
        ax.add_patch(FancyBboxPatch((x, Y(tc + h / 2)), _CARD_W, h,
                                    boxstyle="round,pad=0.0,rounding_size=0.06",
                                    facecolor=_GHOST_FILL if ghost else _CARD_FILL,
                                    edgecolor="#9A9A9A" if ghost else _EDGE_PORT,
                                    lw=1.0 if ghost else 1.3,
                                    linestyle=(0, (3, 2)) if ghost else "-", zorder=3))
        for k, line in enumerate(lines):
            ax.text(x + 0.08, Y(tc - h / 2 + 0.06 + (k + 0.5) * _CARD_LINE), line,
                    fontsize=_FS_CARD, va="center", zorder=4,
                    family=None if k == 0 else _MONO,
                    fontweight="bold" if k == 0 and not ghost else "normal",
                    color="#555555" if ghost else "#1A1A1A")

    def margin_colour(kind, obj, n):
        # A port whose buffer sits in ANOTHER cluster's L1 than the node touching it is a
        # remote access. A pull reading it is fine; a plain write to it completes without
        # writing anything, so it is drawn in the cross-cluster colour to be looked at.
        if kind == "ghost":
            return _EDGE_GHOST, 1.0
        cl = obj["port"].cluster
        if cl is not None and n.assigned_cluster_id != cl:
            kinds_used.add("remote")
            return _EDGE_CROSS_CLUSTER, 1.3
        return _EDGE_PORT, 1.0

    for (kind, lines, ends, obj), (tc, h) in zip(sc.in_items, sc.in_place):
        card(sc.x_in, tc, h, lines, kind == "ghost")
        for n in ends:
            x, t = sc.pos[n]
            colour, lw = margin_colour(kind, obj, n)
            also = kind == "card" and n in obj["also"]
            if also:
                kinds_used.add("also")
            arrow((sc.x_in + _CARD_W, Y(tc)), (x - half_w, Y(t)), colour,
                  lw * (0.8 if also else 1.0), ls=(0, (1, 2)) if also else (0, (4, 2)),
                  alpha=0.85, z=1)
    for (kind, lines, ends, obj), (tc, h) in zip(sc.out_items, sc.out_place):
        card(sc.x_out, tc, h, lines, kind == "ghost")
        for n in ends:
            x, t = sc.pos[n]
            colour, lw = margin_colour(kind, obj, n)
            arrow((x + half_w, Y(t)), (sc.x_out, Y(tc)), colour, lw,
                  ls=(0, (4, 2)), alpha=0.85, z=1)

    # ---- legend ------------------------------------------------------------------------
    handles = []
    present = {m.role_of.get(n.assigned_core_id) for n in sc.nodes}
    for role in _ENGINE_ORDER:
        if role in present:
            lab, colr = _ENGINES[role]
            handles.append(Patch(facecolor=_tint(colr), edgecolor=colr, label=lab))
    for kind, color, lw, lab in (("lane", _EDGE_SAME_LANE, 1.1, "same core"),
                                 ("core", _EDGE_CROSS_CORE, 1.25, "other core, same cluster"),
                                 ("xcl", _EDGE_CROSS_CLUSTER, 1.9, "other cluster")):
        if kind in kinds_used:
            handles.append(Line2D([0], [0], color=color, lw=lw, label=lab))
    if sc.in_cards or sc.out_cards:
        handles.append(Line2D([0], [0], color=_EDGE_PORT, lw=1.0, ls=(0, (4, 2)),
                              label="port binding"))
    if "also" in kinds_used:
        handles.append(Line2D([0], [0], color=_EDGE_PORT, lw=0.8, ls=(0, (1, 2)),
                              label="ordered after an input's producer, not a declared reader"))
    if "remote" in kinds_used:
        handles.append(Line2D([0], [0], color=_EDGE_CROSS_CLUSTER, lw=1.3, ls=(0, (4, 2)),
                              label="port in another cluster's L1"))
    if sc.ghost_in or sc.ghost_out:
        handles.append(Line2D([0], [0], color=_EDGE_GHOST, lw=1.0, ls=(0, (4, 2)),
                              label="other edge across the boundary"))
    leg = ax.legend(handles=handles, loc="center left", ncol=len(handles), fontsize=7.3,
                    frameon=False, bbox_to_anchor=(0.12 / W, (H - sc.legend_t - 0.2) / H),
                    handlelength=2.2, columnspacing=1.6)
    leg.set_zorder(5)
    notes = ["#id = task id (final_dfg.csv, runtime trace)",
             "columns = dependency depth inside the block; one column may run concurrently",
             "buffers under their kernel-argument names; the .txt has every argument and edge"]
    if sc.n_gate:
        notes.append("'gated by N': held until the last node of N producers (link gate_sources)")
    if "stub" in kinds_used:
        notes.append("'#id' at a band edge: the edge continues at that node")
    import textwrap
    for k, line in enumerate(textwrap.wrap("  ·  ".join(notes), width=int((W - 0.5) * 19))[:3]):
        ax.text(0.25, Y(sc.legend_t + 0.5 + k * 0.17), line, fontsize=6.9, color="#555555",
                va="center")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ======================================================================================
# The listing beside each picture
# ======================================================================================

def _listing(sc: _Scene) -> str:
    rec, m, w = sc.rec, sc.machine, sc.w
    out = [f"[{rec.index:02d}] {rec.name} · {rec.kind}"]
    if rec.doc:
        out.append(f"    {rec.doc}")
    if rec.params:
        out.append(f"realisation: {rec.params}")
    if rec.cost:
        out.append(f"cost: {rec.cost}")
    out.append("")

    def nid(n):
        return f"#{n.node_id} {w.name(n)}"

    def ext(n):
        return f"{nid(n)} [{w.owner_name(n)}]"

    out.append("PORTS")
    for c in sc.in_cards:
        p = c["port"]
        out.append(f"  in  {p.name:<10} {p.spec}")
        out.append(f"      buffer {p.buffer}   from {p.peer}"
                   + (f" ({', '.join(nid(n) for n in p.peer_ends)})" if p.peer_ends else ""))
        out.append(f"      first read by: {', '.join(nid(n) for n in c['to']) or '-'}")
        if c["also"]:
            out.append(f"      also ordered after its producer: "
                       f"{', '.join(nid(n) for n in c['also'])}")
        if c["gates"]:
            out.append(f"      its producer gates {_plural(c['gates'], 'source')} of this block")
    for c in sc.out_cards:
        p = c["port"]
        out.append(f"  out {p.name:<10} {p.spec}")
        out.append(f"      buffer {p.buffer}   last written by: "
                   f"{', '.join(nid(n) for n in c['from']) or '-'}")
        for (s, q, cends) in c["users"]:
            out.append(f"      consumed by {s}.{q}: {', '.join(nid(n) for n in cends) or '-'}")
        for s, k in c["gates"].items():
            out.append(f"      gates {_plural(k, 'source')} of {s}")
    if not rec.inputs and not rec.outputs:
        out.append("  (none: this is not a block)")
    out.append("")

    out.append("NODES (creation order)")
    for n in sc.nodes:
        role, eng, _c = m.engine(n.assigned_core_id)
        where = "host" if role == "host" else f"cl{n.assigned_cluster_id} c{n.assigned_core_id} {eng}"
        out.append(f"  #{n.node_id:<5} {_short_name(n, sc.prefix):<28} {where:<14} "
                   f"depth {sc.depth[n]:<3} {n.kernel_name}")
        args = getattr(n, "kernel_args", None)
        if args is not None:
            for k, v in vars(args).items():
                if k.startswith("_"):
                    continue
                if _is_handle(v):
                    txt = _handle_text(v, "")
                    if isinstance(v, BingoMemAlloc):
                        txt += f"  ({v.mem_level}, {v.size} B)"
                elif isinstance(v, (list, tuple)) and v and all(_is_handle(x) for x in v):
                    txt = "[" + ", ".join(_handle_text(x, "") for x in v) + "]"
                else:
                    txt = repr(v)
                    if len(txt) > 160:
                        txt = txt[:157] + "..."
                out.append(f"          {k:<22} = {txt}")
        preds = sorted(w.dfg.predecessors(n), key=lambda x: x.node_id)
        succs = sorted(w.dfg.successors(n), key=lambda x: x.node_id)
        out.append("          after  " + (", ".join(nid(x) if x in sc.inside else ext(x)
                                                    for x in preds) or "(nothing)"))
        out.append("          before " + (", ".join(nid(x) if x in sc.inside else ext(x)
                                                    for x in succs) or "(nothing)"))
    out.append("")
    out.append(f"EDGES INSIDE ({len(sc.edges)}; {sc.implied} implied by a longer path, "
               f"not drawn)")
    drawn = set(sc.drawn)
    for u, v in sorted(sc.edges, key=lambda e: (e[0].node_id, e[1].node_id)):
        out.append(f"  {nid(u)} -> {nid(v)}" + ("" if (u, v) in drawn else "   (implied)"))
    out.append("")
    out.append(f"EDGES ACROSS THE BOUNDARY ({len(sc.ext_in)} in, {len(sc.ext_out)} out; "
               f"{sc.n_gate} gating)")
    for u, v in sorted(sc.ext_in, key=lambda e: (e[0].node_id, e[1].node_id)):
        out.append(f"  in   {ext(u)} -> {nid(v)}   {sc.edge_class.get((u, v), '')}")
    for u, v in sorted(sc.ext_out, key=lambda e: (e[0].node_id, e[1].node_id)):
        out.append(f"  out  {nid(u)} -> {ext(v)}   {sc.edge_class.get((u, v), '')}")
    return "\n".join(out) + "\n"


# ======================================================================================
# Entry point
# ======================================================================================

# The pictures are independent, so they are drawn in parallel. Each worker is a FORK of
# the compiler process and finds the graph here rather than having it pickled over.
_JOBS: list = []


def _render_job(i: int):
    world, rec, png, txt = _JOBS[i]
    sc = _Scene(rec, world)
    _draw(sc, png)
    with open(txt, "w") as f:
        f.write(_listing(sc))


def _render_all(n_jobs: int) -> None:
    import concurrent.futures as cf
    import multiprocessing as mp

    workers = min(8, os.cpu_count() or 1, n_jobs)
    if workers > 1 and "fork" in mp.get_all_start_methods():
        try:
            with cf.ProcessPoolExecutor(max_workers=workers,
                                        mp_context=mp.get_context("fork")) as ex:
                list(ex.map(_render_job, range(n_jobs)))
            return
        except (OSError, cf.BrokenExecutor) as e:
            # No room for workers. A bug in the drawing is NOT caught here: it re-raises
            # as itself, and bingo_visualize_blocks reports it.
            print(f"[block_dfg] no worker pool ({type(e).__name__}: {e}); drawing serially")
    for i in range(n_jobs):
        _render_job(i)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def render_blocks(dfg, output_dir: str, app_name: str = None) -> Dict[str, object]:
    """Write one picture and one listing per recorded block, plus an index, under
    `output_dir`/block_dfg. Returns {"dir": ..., "pictures": n, "blocks": n}."""
    global _JOBS
    records: List[BlockRecord] = list(getattr(dfg, "block_records", None) or [])
    out_dir = os.path.join(output_dir, SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith((".png", ".txt", ".md")):
            os.remove(os.path.join(out_dir, f))
    world = _World(dfg, records)
    width = max(2, len(str(len(records) - 1)))
    rows, jobs = [], []
    for r in records:
        stem = f"{r.index:0{width}d}_{_safe(r.name)}__{_safe(r.kind)}"
        if not r.nodes:
            rows.append((r, None, None, "view: builds no node"))
            continue
        png, txt = f"{stem}.png", f"{stem}.txt"
        jobs.append((world, r, os.path.join(out_dir, png), os.path.join(out_dir, txt)))
        eng = Counter(world.machine.engine(n.assigned_core_id)[1] for n in r.nodes)
        rows.append((r, png, txt, ", ".join(f"{c} {e}" for e, c in eng.most_common())))
    _JOBS = jobs
    try:
        _render_all(len(jobs))
    finally:
        _JOBS = []
    _index(out_dir, rows, app_name, dfg, records)
    return {"dir": out_dir, "pictures": len(jobs), "blocks": len(records)}


def _index(out_dir, rows, app_name, dfg, records) -> None:
    recorded = sum(len(r.nodes) for r in records)
    lines = [f"# Block sub-DFGs{': ' + app_name if app_name else ''}", "",
             "One picture per stage of the Pipeline, in build order, which is node creation",
             "order. A picture shows the nodes its stage created: one lane per (cluster, core),",
             "columns by dependency depth inside the stage, the ports as cards on the left and",
             "right, and every other edge across the boundary as a dashed ghost box. The `.txt`",
             "beside each picture lists every argument and every edge.", "",
             f"The stages below built {recorded} of the graph's {dfg.number_of_nodes()} nodes;"
             " any others were added by the application outside the Pipeline.", "",
             "| # | stage | block | nodes | engines | picture | listing |",
             "|---|---|---|---|---|---|---|"]
    for r, png, txt, eng in rows:
        lines.append(f"| {r.index} | `{r.name}` | {r.kind} | {len(r.nodes)} | {eng} | "
                     f"{f'[png]({png})' if png else '-'} | {f'[txt]({txt})' if txt else '-'} |")
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
