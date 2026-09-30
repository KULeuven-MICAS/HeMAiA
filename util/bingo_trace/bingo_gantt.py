#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Per-task Gantt chart and bottleneck report of one BINGO offload simulation.

Reads the cluster cores' raw instruction traces (logs/trace_chip_*_hart_*.dasm) directly --
no spike-dasm / gen_trace pass -- and joins every task to its DFG node by the global task id
the core reads from CSR 0x5fe when it picks the task up:

    per hart   the BINGO manager's markers (xori x0, x0, imm; perf_tracing.h):
               GET_READY (0x110/0x111), PREP, RUN_KERNEL (0x114/0x115), WRITE_DONE, and the
               kernels' own CFG/RUN spans (0x2xx/0x3xx)
    task id    csrr rd, 0x5fe right after GET_READY_START; the value retires on the
               accelerator port (retire_acc, acc_pdata_32) on a later line
    names      final_dfg.csv (ID, Chiplet, Cluster, Core, Type, Kernel, Name)

and, optionally, the cluster iDMA backend traces (dma_trace_<hart>_<ch>.log) for bytes read
and written over time.

Outputs (in --out-dir): tasks.csv (one row per executed task), stages.csv (per stage and
cluster: first start, last end, busy time), gantt.png (a lane per cluster core, coloured by
stage), dma_bw.png and dma_bw.csv (per-cluster iDMA read / write bandwidth in bins), and
report.txt (the summary).

    python3 util/bingo_trace/bingo_gantt.py --log-dir <task>/bin/logs --dma-dir <task>/bin \\
        --dfg-csv <workload>/final_dfg.csv --out-dir <dir> [--cores-per-cluster 4]

Times are simulation nanoseconds (the dasm's first column is ps), the one axis every hart
shares; each hart's cycle counter has its own offset.
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

MARK_MASK, MARK_BITS = 0xFFFFF, 0x04013          # xori x0, x0, imm
CSR_TASK = 0x5FE
_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+\d+\s+0x([0-9a-fA-F]+)\s+DASM\(([0-9a-fA-F]+)\)")
_ACC = re.compile(r"'retire_acc': 0x1,.*?'acc_pdata_32': 0x([0-9a-fA-F]+)")

M_GET_S, M_GET_E, M_PREP_S, M_PREP_E = 0x110, 0x111, 0x112, 0x113
M_RUN_S, M_RUN_E, M_DONE_S, M_DONE_E = 0x114, 0x115, 0x116, 0x117


def parse_dasm(path):
    """Yield (task_id, spans) per task a cluster core ran; spans = {name: [t0_ns, t1_ns]}."""
    task = None
    want_acc = False
    open_marks = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _LINE.match(line)
            if not m:
                continue
            t_ns = int(m.group(1)) / 1000.0
            insn = int(m.group(4), 16)
            if want_acc and "'retire_acc': 0x1" in line:
                a = _ACC.search(line)
                if a and task is not None and task["id"] is None:
                    task["id"] = int(a.group(1), 16)
                    want_acc = False
            if (insn & MARK_MASK) == MARK_BITS:
                imm = insn >> 20
                if imm == M_GET_S:
                    if task is not None and task.get("run_e") is not None:
                        yield task
                    task = {"id": None, "get_s": t_ns, "phases": []}
                    open_marks = {}
                elif task is None:
                    continue
                elif imm == M_RUN_S:
                    task["run_s"] = t_ns
                elif imm == M_RUN_E:
                    task["run_e"] = t_ns
                elif imm == M_DONE_E:
                    task["done_e"] = t_ns
                elif imm >= 0x200 and imm < 0x400:
                    base = imm & ~1
                    if imm & 1 == 0:
                        open_marks[base] = t_ns
                    elif base in open_marks:
                        task["phases"].append((base, open_marks.pop(base), t_ns))
            elif (insn >> 20) == CSR_TASK and (insn & 0x707F) == 0x2073 and task is not None:
                want_acc = True
                a = _ACC.search(line)
                if a and task["id"] is None:
                    task["id"] = int(a.group(1), 16)
                    want_acc = False
    if task is not None and task.get("run_e") is not None:
        yield task


def load_dfg(path):
    nodes = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                nodes[int(row["ID"])] = row
            except (KeyError, ValueError):
                continue
    return nodes


def stage_of(name):
    """The pipeline stage a node belongs to: its name up to the block's own part. Block nodes
    are named <stage>_<node>_cl<c> (libs Pipeline scopes); dummy nodes keep their target."""
    n = re.sub(r"_cl\d+$", "", name or "?")
    n = re.sub(r"^dummy_(set|check)_", "", n)
    parts = n.split("_")
    # stage names are the first token plus _c<k> / _s<k> suffix tokens when present
    out = [parts[0]]
    for p in parts[1:3]:
        if re.fullmatch(r"(c|s)\d+", p):
            out.append(p)
        else:
            break
    return "_".join(out)


def parse_dma(path, bin_ns):
    """Per-bin (read bytes, write bytes) of one iDMA backend trace."""
    rd, wr = defaultdict(int), defaultdict(int)
    t_re = re.compile(r"'time': 0x([0-9a-fA-F]+)")
    # 'bus':{'axi_req_ready': r,'axi_req_strobe': s,'axi_req_valid': v,
    #        'axi_rsp_ready': r,'axi_rsp_valid': v}: rsp = read data (R), req = write data (W)
    bus_re = re.compile(r"'axi_req_ready': 0x([01]),'axi_req_strobe': 0x([0-9a-fA-F]+),"
                        r"'axi_req_valid': 0x([01]),'axi_rsp_ready': 0x([01]),"
                        r"'axi_rsp_valid': 0x([01])")
    with open(path, "r", errors="replace") as f:
        for line in f:
            tm = t_re.search(line)
            bm = bus_re.search(line) if tm else None
            if not bm:
                continue
            b = int(tm.group(1), 16) // bin_ns
            if bm.group(4) == "1" and bm.group(5) == "1":
                rd[b] += 64
            if bm.group(1) == "1" and bm.group(3) == "1":
                wr[b] += bin(int(bm.group(2), 16)).count("1")
    return rd, wr


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--dfg-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dma-dir", default=None)
    ap.add_argument("--cores-per-cluster", type=int, default=4)
    ap.add_argument("--bin-ns", type=int, default=20000)
    ap.add_argument("--chip", default="00")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    nodes = load_dfg(args.dfg_csv)
    roles = {0: "GEMM", 1: "SIMD", 2: "xDMA", 3: "iDMA"}

    rows = []
    for path in sorted(glob.glob(os.path.join(args.log_dir,
                                              f"trace_chip_{args.chip}_hart_*.dasm"))):
        hart = int(re.search(r"hart_([0-9a-fA-F]+)\.dasm$", path).group(1), 16)
        if hart == 0:
            continue
        cl, core = (hart - 1) // args.cores_per_cluster, (hart - 1) % args.cores_per_cluster
        for t in parse_dasm(path):
            nd = nodes.get(t["id"], {})
            name = nd.get("Name", f"task{t['id']}")
            gemm_run = sum(e - s for b, s, e in t["phases"] if b in (0x336, 0x332, 0x334, 0x330))
            rows.append(dict(id=t["id"], name=name, stage=stage_of(name),
                             kernel=nd.get("Kernel", "?"), cluster=cl, core=core,
                             role=roles.get(core, str(core)), get_s=t["get_s"],
                             run_s=t.get("run_s", t["get_s"]), run_e=t["run_e"],
                             done_e=t.get("done_e", t["run_e"]),
                             engine_ns=gemm_run))
    rows.sort(key=lambda r: r["run_s"])
    if not rows:
        sys.exit("no tasks found in the traces")
    with open(os.path.join(args.out_dir, "tasks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- per stage and cluster ---------------------------------------------------------
    t0 = min(r["get_s"] for r in rows)
    t1 = max(r["done_e"] for r in rows)
    stages = defaultdict(lambda: dict(start=1e30, end=0, busy=0.0, n=0))
    for r in rows:
        k = (r["stage"], r["cluster"])
        s = stages[k]
        s["start"] = min(s["start"], r["run_s"])
        s["end"] = max(s["end"], r["run_e"])
        s["busy"] += r["run_e"] - r["run_s"]
        s["n"] += 1
    order = sorted(stages.items(), key=lambda kv: kv[1]["start"])
    with open(os.path.join(args.out_dir, "stages.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "cluster", "tasks", "start_us", "end_us", "span_us", "busy_us"])
        for (st, cl), s in order:
            w.writerow([st, cl, s["n"], f"{(s['start'] - t0) / 1e3:.2f}",
                        f"{(s['end'] - t0) / 1e3:.2f}", f"{(s['end'] - s['start']) / 1e3:.2f}",
                        f"{s['busy'] / 1e3:.2f}"])

    # ---- per core utilisation ------------------------------------------------------------
    busy = defaultdict(float)
    for r in rows:
        busy[(r["cluster"], r["core"])] += r["run_e"] - r["run_s"]
    lines = [f"window {t0 / 1e3:.1f} .. {t1 / 1e3:.1f} us ({(t1 - t0) / 1e3:.1f} us), "
             f"{len(rows)} cluster tasks"]
    lines.append("core busy (kernel running) as % of the window:")
    for (cl, core), b in sorted(busy.items()):
        lines.append(f"  cluster {cl} {roles.get(core, core):5s} {100 * b / (t1 - t0):5.1f}%  "
                     f"({b / 1e3:.1f} us)")
    gemm = defaultdict(float)
    for r in rows:
        gemm[r["cluster"]] += r["engine_ns"]
    lines.append("GEMM array RUN spans (GEMV / FA) as % of the window:")
    for cl, b in sorted(gemm.items()):
        lines.append(f"  cluster {cl}: {100 * b / (t1 - t0):5.1f}%")

    # ---- DMA bandwidth ---------------------------------------------------------------------
    dma = {}
    if args.dma_dir:
        for path in sorted(glob.glob(os.path.join(args.dma_dir, "dma_trace_*_*.log"))):
            hart = int(re.search(r"dma_trace_([0-9a-fA-F]+)_", path).group(1), 16)
            cl = (hart - 1) // args.cores_per_cluster
            if os.path.getsize(path) == 0:
                continue
            dma[cl] = parse_dma(path, args.bin_ns)
        if dma:
            with open(os.path.join(args.out_dir, "dma_bw.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bin_start_us", "cluster", "read_GBps", "write_GBps"])
                for cl, (rd, wr) in sorted(dma.items()):
                    for b in sorted(set(rd) | set(wr)):
                        w.writerow([f"{b * args.bin_ns / 1e3:.1f}", cl,
                                    f"{rd[b] / args.bin_ns:.3f}", f"{wr[b] / args.bin_ns:.3f}"])
            tot_rd = sum(sum(rd.values()) for rd, _ in dma.values())
            lines.append(f"iDMA bytes read: {tot_rd / 2**20:.2f} MiB over the window "
                         f"-> {tot_rd / (t1 - t0):.3f} GB/s average")

    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
        f.write("\nstages (start/end relative to the first task, us):\n")
        for (st, cl), s in order:
            f.write(f"  {st:24s} cl{cl} {s['n']:4d} tasks  {(s['start'] - t0) / 1e3:10.1f} "
                    f"-> {(s['end'] - t0) / 1e3:10.1f}  busy {s['busy'] / 1e3:9.1f}\n")
    print("\n".join(lines))

    # ---- pictures ------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    stage_names = []
    for (st, _), _s in order:
        if st not in stage_names:
            stage_names.append(st)
    cmap = plt.get_cmap("tab20")
    colour = {st: cmap(i % 20) for i, st in enumerate(stage_names)}
    lanes = sorted({(r["cluster"], r["core"]) for r in rows})
    fig, ax = plt.subplots(figsize=(22, 0.45 * len(lanes) + 2.5))
    for i, lane in enumerate(lanes):
        for r in rows:
            if (r["cluster"], r["core"]) != lane:
                continue
            ax.broken_barh([((r["run_s"] - t0) / 1e3, max((r["run_e"] - r["run_s"]) / 1e3, 0.05))],
                           (i - 0.4, 0.8), facecolors=colour[r["stage"]], linewidth=0)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([f"cl{c} {roles.get(k, k)}" for c, k in lanes])
    ax.set_xlabel("us from the first task")
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[s]) for s in stage_names[:60]]
    ax.legend(handles, stage_names[:60], ncol=10, fontsize=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "gantt.png"), dpi=110)
    plt.close(fig)
    if dma:
        fig, ax = plt.subplots(figsize=(22, 4))
        for cl, (rd, _wr) in sorted(dma.items()):
            xs = sorted(rd)
            ax.plot([(b * args.bin_ns - t0) / 1e3 for b in xs], [rd[b] / args.bin_ns for b in xs],
                    label=f"cluster {cl} iDMA read")
        ax.set_xlabel("us from the first task")
        ax.set_ylabel("GB/s")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "dma_bw.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
