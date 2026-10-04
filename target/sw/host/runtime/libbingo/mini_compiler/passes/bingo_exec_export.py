"""Export a compiled BingoDFG as an Execution IR (the bingo framework's `ir/exec.py`) in JSON.

The bingo framework's chiplet DSE simulates the layer from this file instead of re-deriving its
structure: every node the RTL runs (dummy dep-tag nodes included), its core, kernel and arguments,
the dependency edges, each chip's descriptor stream in dispatch order, and the weight rings with
their chunks (pushed and expanded bytes, routed, trailer) and the prefetcher's policy. The framework's
`frontend/parser_hemaia` maps it onto its cost model.

    dfg.skip_viz = True                      # optional: the pictures are slow on large graphs
    dfg.bingo_compile_dfg(...)
    bingo_export_exec_ir(dfg, "exec_ir.json", ring_tables=[...], prefetch={...}, meta={...})

Arguments are exported generically: a numeric field as a number, a memory handle as its level, chip
and cluster (where it is), so the framework can tell an L1, L3, HBM or cross-chip copy apart.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

ROLE = {0: "gemm", 1: "simd", 2: "xdma", 3: "dm"}


def _handle(h) -> Optional[dict]:
    cls = type(h).__name__
    if cls == "BingoMemAllocView":
        d = _handle(h.base)
        if d is not None:
            d["offset"] = int(getattr(h, "offset", 0)) + int(d.get("offset", 0))
        return d
    if cls == "BingoMemAlloc":
        return {"h": "alloc", "name": h.name, "level": h.mem_level, "chip": h.chip_id,
                "cluster": h.cluster_id, "size": int(h.size), "offset": int(h.offset)}
    if cls == "BingoMemSymbol":
        return {"h": "symbol", "name": h.symbol_name, "level": "L3", "chip": h.chip_id,
                "offset": int(h.offset)}
    if cls == "BingoMemFixedAddr":
        lvl = h.mem_level or ("HBM" if int(h.address) >= 0x1_0000_0000 else "L3")
        return {"h": "fixed", "level": lvl, "addr": int(h.address)}
    return None


def _args(args) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if args is None:
        return out
    for k, v in vars(args).items():
        if k.startswith("_") or isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = v
        else:
            h = _handle(v)
            if h is not None:
                out[k] = h
    return out


def bingo_export_exec_ir(dfg, path: str, ring_tables: List[dict] = None, prefetch: dict = None,
                         meta: dict = None) -> dict:
    """Write `path` (JSON) and return the dict. Call after `bingo_compile_dfg` (the stream order and
    the dummy nodes are the compiled graph's)."""
    order = dfg.bingo_stream_order()
    tasks, per_chip = [], {}
    for n in order:
        core_id = n.assigned_core_id
        chip = n.assigned_chiplet_id
        if core_id in ROLE:
            core = ["c", chip, n.assigned_cluster_id, ROLE[core_id]]
        else:
            core = ["h", chip]
        args = _args(n.kernel_args)
        chunk = getattr(n, "exec_chunk", None)
        tasks.append({
            "tid": n.node_id, "name": n.node_name, "core": core, "kernel": n.kernel_name or "",
            "size": {k: v for k, v in args.items() if not isinstance(v, dict)},
            "deps": [p.node_id for p, _ in dfg.in_edges(n)], "op": "",
            "kind": "ring_load" if chunk is not None else "task",
            "chunk": [chunk[0], chunk[1]] if chunk is not None else None,
            "meta": {"type": n.node_type,
                     "handles": {k: v for k, v in args.items() if isinstance(v, dict)}}})
        per_chip.setdefault(chip, []).append(n.node_id)
    doc = {"tasks": tasks, "rings": [], "route_tasks": [], "notes": [], "desc_in_l3": False,
           "prefetch": prefetch, "order": {str(k): v for k, v in per_chip.items()},
           "meta": dict(meta or {}, source="hemaia", nodes=len(tasks)),
           "ring_tables": ring_tables or []}
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    return doc


def ring_table(rings, chip: int, mem: int, engine: int) -> dict:
    """A chip's WeightRings as plain data: per ring its chunks (pushed bytes, expanded bytes, routed,
    trailer, compressed, source stream)."""
    out = {"chip": chip, "mem": mem, "engine": engine, "n_slots": rings.n_slots,
           "slot_bytes": rings.slot_bytes, "split": bool(getattr(rings, "split", False)),
           "n_clusters": getattr(rings, "n_clusters", rings.n_rings), "rings": []}
    for r, entries in enumerate(rings.entries):
        metas = rings.entry_meta[r] if hasattr(rings, "entry_meta") else [{}] * len(entries)
        chunks, sid, prev = [], 0, None
        for (src, kind, off, nbytes), m in zip(entries, metas):
            routed = bool(kind & 1)
            # the prefetcher's run rule (host_kernel_lib.h weight_prefetch_run_of): a run goes on
            # only within one source stream -- the same kind, and the next byte of the same range
            # (dense) or the same record field at the next offset (routed). A new stream id where
            # it breaks; the simulator's runs never cross one.
            if prev is not None:
                ps, pk, po, pn = prev
                cont = kind == pk and ((src == ps and off == po + pn) if routed
                                       else src + off == ps + po + pn)
                sid += 0 if cont else 1
            prev = (src, kind, off, nbytes)
            chunks.append({"nbytes": int(nbytes), "raw_bytes": int(m.get("raw", nbytes)),
                           "routed": routed, "trailer": bool(kind & 2),
                           "compressed": bool(m.get("compressed", False)),
                           "stream": [sid, int(kind)]})
        n_cl = out["n_clusters"]
        out["rings"].append({"rid": r, "cluster": r % n_cl,
                             "kind": ("routed" if r >= n_cl else "dense") if out["split"] else "all",
                             "chunks": chunks})
    return out
