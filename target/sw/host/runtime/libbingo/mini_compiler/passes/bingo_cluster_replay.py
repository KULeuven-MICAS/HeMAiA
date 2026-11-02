"""One cluster of a compiled BingoDFG as a standalone program for the SNAX cluster simulator.

`emit_cluster_replay` writes a C header that a standalone SNAX app (the snax_cluster repository's
sw/apps/hemaia/hemaia-replay) runs on one cluster: the cluster's tasks inside the window, each core's in
the order the hardware streams them, each with the same kernel and the same arguments the HeMAiA build
gives it (the args classes' own C emission), and a dependency between two of them wherever the graph has
one inside the cluster (through the dummy dep-tag nodes). What lies outside the cluster is taken as done
when the program starts: the tasks before the window (the preloads), the other clusters' and chips'
tasks, the host's. So a cluster's window here is its own work with its inputs ready; the external
waits HeMAiA adds (another cluster's data, a multicast, a ring push) are not in it.

Addresses: the cluster's L1 buffers at the static planner's offsets (static_l1_layout.csv) from the
cluster's TCDM base; another cluster's L1 into a sink; every L3 symbol, on any chip, into an array of
its name in the cluster's DRAM, and every HBM address into one more (rp_hbm). Arguments go into TCDM
behind the planned buffers, as HeMAiA keeps them in L1; the scratchpads into DRAM.

Gates (`gates`): a task can wait, besides its dependencies inside the cluster, for a time after one of the cluster's
own tasks ends: [task name regex, anchor name regex, cycles]. That stands for what comes from outside the cluster
with a delay measured on HeMAiA (another cluster's data, a multicast) relative to the cluster's own matching event.
Setup (`include_pre`): the cluster's own tasks before the window run too, first, for state they leave in L1 that the
window's tasks read (a GEMM shape table, zeroed buffers); without it they count as done.
Durations (`durations`): a task can be held to at least a number of cycles: [task name regex, cycles], for a task whose
work the standalone cluster does only in part (a ChainGather's hops through the other chips: the app then runs its last
hop's landing alone and holds the task for the chain's measured time).

    dfg.bingo_compile_dfg(..., static_l1=True)
    emit_cluster_replay(dfg, chip, cluster, "<out>/static_l1_layout.csv", "<out>/replay.h")
"""
from __future__ import annotations

import csv
from typing import Dict, List

ROLE = {0: "gemm", 1: "simd", 2: "xdma", 3: "dm"}
SKIP_KERNELS = ("exit",)


def _window(dfg):
    """(the tasks before the window, the tasks after it): the start anchor's ancestors, the end anchor's
    descendants."""
    nodes = list(dfg.node_list)
    a0 = [n for n in nodes if (n.node_name or "").startswith("A0_start")]
    a1 = [n for n in nodes if (n.node_name or "").startswith("A1_end")]
    pre, post = set(), set()
    stack = list(a0)
    while stack:
        n = stack.pop()
        for p, _ in dfg.in_edges(n):
            if p not in pre:
                pre.add(p)
                stack.append(p)
    stack = list(a1)
    while stack:
        n = stack.pop()
        for s, _ in dfg.out_edges(n):
            if s not in post:
                post.add(s)
                stack.append(s)
    return pre | set(a0), post | set(a1)


def _real_preds(dfg, n, keep, memo):
    """The tasks in `keep` that n depends on, through every node not in it (the dummy dep-tag nodes)."""
    if n in memo:
        return memo[n]
    out = set()
    for p, _ in dfg.in_edges(n):
        if p in keep:
            out.add(p)
        else:
            out |= _real_preds(dfg, p, keep, memo)
    memo[n] = out
    return out


def emit_cluster_replay(dfg, chip: int, cluster: int, layout_csv: str, out_path: str, gates=None,
                        durations=None, include_pre: bool = False) -> dict:
    import re
    from bingo_mem_handle import BingoMemAlloc, BingoMemFixedAddr, BingoMemSymbol
    pre, post = _window(dfg)
    order = dfg.bingo_stream_order()
    mine = [n for n in order
            if n.assigned_chiplet_id == chip and n.assigned_cluster_id == cluster and n.assigned_core_id in ROLE
            and n.kernel_name and not n.kernel_name.endswith(SKIP_KERNELS) and (include_pre or n not in pre)
            and n not in post]
    keep = set(mine)
    idx = {n: i for i, n in enumerate(mine)}
    memo: Dict = {}
    deps = [sorted(idx[p] for p in _real_preds(dfg, n, keep, memo)) for n in mine]
    # gates: each matching task waits `cycles` after the end of the cluster's latest earlier task matching the anchor
    gate_anc, gate_dly, n_gated = [0xFFFF] * len(mine), [0] * len(mine), 0
    for pat, anc, cyc in (gates or []):
        rt, ra = re.compile(pat), re.compile(anc)
        for i, n in enumerate(mine):
            if rt.search(n.node_name or ""):
                a = [j for j in range(len(mine)) if ra.search(mine[j].node_name or "") and j != i
                     and mine[j].node_id < n.node_id]
                if not a:
                    raise ValueError(f"replay gate: no anchor {anc!r} before {n.node_name}")
                gate_anc[i], gate_dly[i] = max(a, key=lambda j: mine[j].node_id), int(cyc)
                n_gated += 1
    min_dur = [0] * len(mine)
    for pat, cyc in (durations or []):
        rt = re.compile(pat)
        for i, n in enumerate(mine):
            if rt.search(n.node_name or ""):
                min_dur[i] = int(cyc)
    # the planned L1 offsets of the cluster's buffers
    off, end = {}, 0
    for r in csv.DictReader(open(layout_csv)):
        if int(r["chip"], 16) == chip and int(r["cluster"]) == cluster:
            off[r["buffer"]] = int(r["offset"])
            end = max(end, int(r["end"]))
    # handles -> C expressions; L3 symbols keep their names (arrays below); HBM -> rp_hbm
    hmap: Dict = {}
    syms: Dict[str, int] = {}
    hbm_slot: Dict[int, int] = {}
    HBM_SLOT = 1 << 21

    # 64 bit, as HeMAiA's chiplet addresses are: the args take the high word with a shift by 32
    def alloc_expr(h):
        if h.mem_level == "L1" and h.chip_id == chip and h.cluster_id == cluster:
            if h.name not in off:
                raise ValueError(f"replay: L1 buffer {h.name} has no planned offset (static_l1 off?)")
            return f"((uint64_t)(RP_L1 + {off[h.name]}u))"
        if h.mem_level == "L1":
            return "((uint64_t)RP_SINK)"
        syms.setdefault(f"rp_l3_{h.name}", 0)
        syms[f"rp_l3_{h.name}"] = max(syms[f"rp_l3_{h.name}"], int(h.size))
        return f"((uint64_t)(uintptr_t)rp_l3_{h.name})"

    # each symbol's lowest and highest offset an argument names: the app hashes from the one to 64 KiB past the other
    touched: Dict[str, tuple] = {}
    lines_init: List[str] = []
    types: List[str] = []
    for i, n in enumerate(mine):
        a = n.kernel_args
        swapped = {}
        for k, v in list(vars(a).items()):
            if isinstance(v, BingoMemFixedAddr):
                base = int(v.address) >> 20
                if base not in hbm_slot:
                    hbm_slot[base] = len(hbm_slot) * HBM_SLOT
                swapped[k] = v
                setattr(a, k, BingoMemSymbol("rp_hbm", offset=hbm_slot[base] + (int(v.address) & 0xFFFFF)))
        for k, v in vars(a).items():
            hs = [v] if not isinstance(v, (list, tuple)) else list(v)
            for h in hs:
                if isinstance(h, BingoMemAlloc):
                    hmap[h] = alloc_expr(h)
                elif hasattr(h, "base") and isinstance(getattr(h, "base", None), BingoMemAlloc):
                    hmap[h.base] = alloc_expr(h.base)
                if isinstance(h, BingoMemSymbol) and h.symbol_name != "rp_hbm":
                    need = int(h.offset) + (1 << 20)
                    syms[h.symbol_name] = max(syms.get(h.symbol_name, 0), need)
                    lo, hi = touched.get(h.symbol_name, (1 << 62, 0))
                    touched[h.symbol_name] = (min(lo, int(h.offset)), max(hi, int(h.offset)))
        a._scratchpad_c_expr = f"(uint32_t)(uintptr_t)&rp_sp[{i}]"
        try:
            fields = a.get_c_field_assignments_with_scratchpad(hmap)
        finally:
            for k, v in swapped.items():
                setattr(a, k, v)
        t = a.get_struct_name()
        types.append(t)
        lines_init.append(f"    {{ {t} *a = ({t} *)(RP_ARGS + o); rp_args[{i}] = a; o += (sizeof({t}) + 7u) & ~7u;")
        fields.setdefault("gating_sp_addr", "0")
        fields.setdefault("cond_node_index", "0")
        for f, e in fields.items():
            lines_init.append(f"      a->{f} = {e};")
        lines_init.append("    }")
    if hbm_slot:
        syms["rp_hbm"] = len(hbm_slot) * HBM_SLOT
    fns = sorted({n.kernel_name for n in mine})
    out = [f"// Generated by bingo_cluster_replay.py: chip 0x{chip:02x} cluster {cluster}, {len(mine)} tasks in the window.",
           "#pragma once",
           f"#define RP_N {len(mine)}u",
           f"#define RP_L1_END {end}u            // the planned L1 buffers end here; the arguments follow",
           f"#define RP_CHIP 0x{chip:02x}u",
           f"#define RP_CLUSTER {cluster}u"]
    # initialised, so in .data (loaded with the ELF) and not .bss, which the runtime zeroes a store at a time
    for s, nb in sorted(syms.items()):
        out.append(f"static uint8_t {s}[{nb}] __attribute__((aligned(64))) = {{1}};")
    # the arrays the app hashes at the end, what the cluster's tasks leave outside its L1: the part of each its
    # arguments reach
    out.append(f"#define RP_NSYMS {len(syms)}u")
    out.append("static uint8_t *const rp_sym_ptr[RP_NSYMS] = {" + ", ".join(sorted(syms)) + "};")
    def span(k):
        lo, hi = touched.get(k, (0, syms[k]))
        lo &= ~63
        return lo, min(syms[k], hi + (1 << 16)) - lo
    out.append("static const uint32_t rp_sym_lo[RP_NSYMS] = {" + ", ".join(str(span(k)[0]) for k in sorted(syms)) + "};")
    out.append("static const uint32_t rp_sym_len[RP_NSYMS] = {" + ", ".join(str(span(k)[1]) for k in sorted(syms)) + "};")
    out.append("static const char *const rp_sym_name[RP_NSYMS] = {" + ", ".join(f'"{k}"' for k in sorted(syms)) + "};")
    out.append(f"static bingo_kernel_scratchpad_t rp_sp[RP_N];")
    out.append("static void *rp_args[RP_N];")
    out.append(f"static const uint8_t rp_core[RP_N] = {{{', '.join(str(n.assigned_core_id) for n in mine)}}};")
    flat, start = [], []
    for d in deps:
        start.append(len(flat))
        flat += d
    start.append(len(flat))
    out.append(f"static const uint16_t rp_dep_start[RP_N + 1] = {{{', '.join(map(str, start))}}};")
    out.append(f"static const uint16_t rp_dep[{max(1, len(flat))}] = {{{', '.join(map(str, flat)) or '0'}}};")
    out.append("typedef uint64_t (*rp_fn_t)(void *);")
    out.append(f"static const rp_fn_t rp_fn[RP_N] = {{{', '.join('(rp_fn_t)&' + n.kernel_name for n in mine)}}};")
    out.append(f"#define RP_NGATED {n_gated}u")
    out.append(f"static const uint16_t rp_gate_anc[RP_N] = {{{', '.join(map(str, gate_anc))}}};")
    out.append(f"static const uint32_t rp_gate_dly[RP_N] = {{{', '.join(map(str, gate_dly))}}};")
    out.append("#define RP_HAS_MIN_DUR 1")
    out.append(f"static const uint32_t rp_min_dur[RP_N] = {{{', '.join(map(str, min_dur))}}};")
    # arguments some classes write as raw C (a ChainGather's hops): HeMAiA's host pointer to one of this cluster's
    # planned buffers, ptr_<buffer>, becomes its place in this TCDM
    raw = sorted({m for m in re.findall(r"\bptr_([A-Za-z0-9_]+)", "\n".join(lines_init)) if m in off})
    for b in raw:
        out.append(f"#define ptr_{b} ((uint64_t)(RP_L1 + {off[b]}u))")
    out.append("static void rp_init_args(uintptr_t RP_L1, uintptr_t RP_ARGS, uintptr_t RP_SINK) {")
    out.append("    (void)RP_SINK;")
    out.append("    uint32_t o = 0;")
    out += lines_init
    out.append("}")
    for b in raw:
        out.append(f"#undef ptr_{b}")
    with open(out_path, "w") as f:
        f.write("\n".join(out) + "\n")
    names_path = out_path.rsplit(".", 1)[0] + "_names.txt"
    with open(names_path, "w") as f:
        for i, n in enumerate(mine):
            f.write(f"{i} {ROLE[n.assigned_core_id]} {n.node_name} {n.kernel_name}\n")
    print(f"[replay] chip 0x{chip:02x} cluster {cluster}: {len(mine)} tasks, {len(flat)} dependencies, kernels "
          f"{len(fns)}; {n_gated} gated; L1 planned to {end} B; {out_path}")
    return dict(tasks=len(mine), deps=len(flat), l1_end=end, kernels=fns)
