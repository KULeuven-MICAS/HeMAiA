#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""trace_ffn.py <task dir> <workload dir>: every task's start and end, from the core traces.

THE MAPPING. Every core runs its tasks in the order of its descriptors in the workload's
offload_bingo_hw.h (the per-core order the BINGO manager demuxes; NOT the node-id order of
final_dfg.csv), and the device runtime brackets each kernel it runs with two magic NOPs,
BINGO_TRACE_MGR_RUN_KERNEL_START / _END (perf_tracing.h). So the k-th bracket in a core's trace
is the k-th normal task of that core: no task id needs to be in the trace.

    harts   0 = the host core (trace_chip_XX_hart_00000.log, disassembled), and
            1 + 4 c + core for cluster c's cores 0..3 (.dasm, raw instruction words)

Writes <task dir>/tasks.csv (name, chip, cluster, core, start_ns, end_ns) and returns the
phase summary parse_ffn.py records: each the latest end of its tasks, in ns after the start
anchor's end.
"""

import csv
import os
import re
import sys
from pathlib import Path

RUN_START, RUN_END = 0x114, 0x115                 # BINGO_TRACE_MGR_RUN_KERNEL_START / _END
NOP = lambda imm: (imm << 20) | (4 << 12) | 0x13  # xori x0, x0, imm
# descriptor lists: one comment line a descriptor (static image), or the older one store a word
DESC_IMAGE = re.compile(r"// Node ID (\d+)\s*\n\s*//\s+Type=(\d) TaskID=\d+ Assigned=(\w\w)/(\d+)/(\d+)")
DESC = re.compile(r"bingo_hw_scheduler_task_desc_list_chip_(\w\w)\[\d+ \* BINGO_TASK_DESC_WORDS"
                  r" \+ 0\] = \w+; // Node ID (\d+)\s*\n\s*// Fields: Type=(\d), TaskID=\d+"
                  r"\s*\n\s*//\s*Assigned: Chiplet=\w+, Cluster=(\d+), Core=(\d+)")
HOST_CORE = 4


def core_orders(header: Path, dfg_csv: Path):
    """{(chip, cluster, core): [task name, ...]} in the order the hardware runs them."""
    names = {r["ID"]: r["Name"] for r in csv.DictReader(dfg_csv.open())}
    out = {}
    txt = header.read_text()
    rows = [(c, n, t, cl, co) for n, t, c, cl, co in DESC_IMAGE.findall(txt)] or DESC.findall(txt)
    for chip, nid, ty, cl, core in rows:
        if ty == "0":
            out.setdefault((chip, int(cl), int(core)), []).append(names.get(nid, nid))
    return out


def data_start(f) -> int:
    """Where a file's data begins: past a hole punched into its head (trim_live.py), which would
    otherwise read as one line of NULs the size of the hole."""
    try:
        return os.lseek(f.fileno(), 0, os.SEEK_DATA)
    except OSError:                                 # empty, or all hole, or no SEEK_DATA support
        return 0


STALL_NS = 100.0   # a gap between two retired instructions longer than this is a stall


def spans_dasm(path: Path, stalls=None):
    """(start, end) of each bracketed kernel, in ns. With `stalls`, also appends one (stall ns,
    stalls) a span: the gaps longer than STALL_NS between consecutive retired instructions in it --
    the core waiting on an instruction fetch or a load, as a polling wait retires instructions."""
    s, e = f"DASM({NOP(RUN_START):08x})".encode(), f"DASM({NOP(RUN_END):08x})".encode()
    out, t0 = [], None
    prev, st_ns, st_n = 0, 0.0, 0
    with path.open("rb") as f:
        f.seek(data_start(f))
        for line in f:
            if t0 is not None and stalls is not None:
                try:
                    t = int(line.split(None, 1)[0])
                except (ValueError, IndexError):
                    t = prev
                if t - prev > STALL_NS * 1000:
                    st_ns += (t - prev) / 1000.0
                    st_n += 1
                prev = t
            if s in line:
                t0 = int(line.split()[0]) / 1000.0
                prev, st_ns, st_n = int(t0 * 1000), 0.0, 0
            elif e in line and t0 is not None:
                out.append((t0, int(line.split()[0]) / 1000.0))
                if stalls is not None:
                    stalls.append((round(st_ns, 1), st_n))
                t0 = None
    return out


def spans_log(path: Path):
    s, e = f"xori           x0, {RUN_START}", f"xori           x0, {RUN_END}"
    out, t0 = [], None
    with path.open(errors="replace") as f:
        for line in f:
            if s in line or e in line:
                ns = float(line.split()[0].rstrip("ns"))
                if s in line:
                    t0 = ns
                elif t0 is not None:
                    out.append((t0, ns))
                    t0 = None
    return out


def tasks(task_dir: Path, wl_dir: Path, ncl: int = 2):
    orders = core_orders(wl_dir / "offload_bingo_hw.h", wl_dir / "final_dfg.csv")
    logs = task_dir / "bin" / "logs"
    rows = []
    for (chip, cl, core), names in orders.items():
        if core == HOST_CORE:
            p = logs / f"trace_chip_{chip}_hart_00000.log"
            sp = spans_log(p) if p.exists() else []
        else:
            p = logs / f"trace_chip_{chip}_hart_{1 + 4 * cl + core:05d}.dasm"
            stalls = []
            sp = spans_dasm(p, stalls) if p.exists() else []
        for i, (nm, (a, b)) in enumerate(zip(names, sp)):
            st = stalls[i] if core != HOST_CORE and i < len(stalls) else ("", "")
            rows.append(dict(name=nm, chip=chip, cluster=cl, core=core, start_ns=a, end_ns=b,
                             stall_ns=st[0], stalls=st[1]))
    return rows


def phases(rows):
    """The FFN's phases, each the latest end of its tasks, in ns after the start anchor."""
    by = {r["name"]: r for r in rows}
    st = next((r for r in rows if r["name"].startswith("A0_start")), None)
    if st is None:
        return {}
    t0 = st["end_ns"]
    last = lambda pat: max((r["end_ns"] for r in rows if re.search(pat, r["name"])), default=None)
    first = lambda pat: min((r["end_ns"] for r in rows if re.search(pat, r["name"])), default=None)
    # ffn: gu_g<g>, dn_g<g>; moe: shgu_g<g> / gu_s<s>_g<g>, shdn_g<g> / dn_s<s>_g<g>, and the
    # route (the root's softmax over the folded or gathered logits; every slot chip's record);
    # fa: pass 1 (s1_<i>_g<g>), m* made on the root (mg_chain / mg_rowmax) and in every
    # cluster (m_wait / m_pull), pass 2 (pv<i>_g<g>), the normalised output (o_transpose)
    out = {"x": last(r"^x_(wait|pull)_"),
           "route": last(r"^route_k"),
           "first_chunk": first(r"(_Ld_linear_w0_|^Ld_K0_)"),
           "pass1": last(r"^s1_\d+_g"),
           "mstar": last(r"^mg_(chain|rowmax)"),
           "marr": last(r"^m_(wait|pull)_c"),
           "pass2": last(r"^pv\d+_g"),
           "out": last(r"^o_transpose"),
           "gu": last(r"^(sh)?gu_(s\d+_)?g\d+_Gemm"),
           "dn": last(r"^(sh)?dn_(s\d+_)?g\d+_Gemm"),
           "loc": last(r"^loc_add_"),
           "reduce": last(r"^y_(chain|add\d+)_"),
           "y16": last(r"^y16"),
           "end": last(r"^A\d+_end")}
    return {k: (round(v - t0, 1) if v is not None else None) for k, v in out.items()}


def main():
    task_dir, wl_dir = Path(sys.argv[1]), Path(sys.argv[2])
    rows = tasks(task_dir, wl_dir)
    with (task_dir / "tasks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "chip", "cluster", "core", "start_ns", "end_ns"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["start_ns"]))
    print(f"[trace_ffn] {len(rows)} tasks -> {task_dir / 'tasks.csv'}")
    print(phases(rows))


if __name__ == "__main__":
    main()
