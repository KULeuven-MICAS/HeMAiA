#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""replay_pf.py <workload under workloads/ffn> --run <measured run dir> [--run-wl <its workload>]

Replays an FFN prefill task graph with the task durations measured in a finished run, to project
what a schedule change does before it is simulated. The graph and the order the hardware runs it in
are the generator's own: the workload is generated in-process (into a scratch directory), its graph
after every compiler pass (relays and dummy tasks included) gives the dependencies, and the emitted
descriptor lists give each chip's task order.

The model of the BINGO manager: each chip fetches its descriptor list in order into one waiting queue
per core (depth --wq); a full queue stops the chip's fetch. A core checks its queue head only: the head
is released when every predecessor is done (plus --msg-us for a predecessor on another chip) and at
most 8 released tasks of that core are not done yet (the checkout queue). A released task runs on its
core when the core is free, for its measured duration (start to done in the run's bingo_progress.log,
matched by task name); a dummy task takes no core time. Spin tasks whose measured time is mostly
waiting are given their shortest measured time (x_wait: the flag is set when its put is done).

Prints the window (start anchor done to the last store done), each iteration's store and X put, and,
with --compare, the measured values of the run next to them."""
import argparse
import collections
import contextlib
import csv
import io
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE.split("/target/sim/")[0]
W = f"{REPO}/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/ffn"
SN = f"{REPO}/deps/snitch_cluster"
sys.path.insert(0, f"{W}/common")
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")

CHECKOUT = 8
SPIN_MIN = ("x_wait_",)          # spin tasks replayed at their shortest measured time


def generate(wl, out):
    """Generate workload wl into out; returns the PBuild object (its dfg after every pass)."""
    import hjson
    import _bingo_paths  # noqa: F401
    from bingo_platform import load_cluster_cfg, parse_platform_cfg
    import ffn_prefill_mc as M
    args = argparse.Namespace(output_dir=out, data_h=os.path.join(out, "ffn_data.h"),
                              output_offload_file_name="offload_bingo_hw.h", cfg=f"{W}/{wl}/params.hjson",
                              hwcfg=f"{SN}/target/snitch_cluster/cfg/snax_split_cluster.hjson",
                              platformcfg=f"{REPO}/target/sw/shared/platform/generated/occamy.h",
                              dsv2_dir=f"{SN}/target/snitch_cluster/sw/apps/dsv2")
    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        S = M.PBuild(args, p, parse_platform_cfg(args.platformcfg), load_cluster_cfg(args.hwcfg))
        S.build()
        S.finish(f"replay {wl}")
    return S


def measured(run, wl_dfg):
    """{task name: (start, done)} in us, and the start anchor's done, from a run's scheduler log."""
    names = {int(r["ID"]): r["Name"] for r in csv.DictReader(open(wl_dfg))}
    log = next(p for p in (os.path.join(run, d, "bin", "bingo_progress.log") for d in os.listdir(run)
                           if d.startswith("task_")) if os.path.exists(p))
    ev = collections.defaultdict(dict)
    for ln in open(log, errors="replace"):
        if not ln.startswith("[BINGO EV]"):
            continue
        kv = dict(re.findall(r"(\w+)=(\S+)", ln))
        n = names.get(int(kv["id"]))
        if n:
            ev[n].setdefault(kv["ev"], int(kv["t"]) / 1e6)
    return {n: (e["S"], e["D"]) for n, e in ev.items() if "S" in e and "D" in e}


def durations(meas):
    dur = {n: d - s for n, (s, d) in meas.items()}
    kind = lambda n: re.sub(r"\d+", "#", n)
    lo = collections.defaultdict(lambda: 1e9)
    for n, d in dur.items():
        if n.startswith(SPIN_MIN):
            lo[kind(n)] = min(lo[kind(n)], d)
    for n in dur:
        if n.startswith(SPIN_MIN):
            dur[n] = lo[kind(n)]
    return dur


def size_of(nd):
    """The size of a task's work, from its kernel arguments: array steps for a GEMM, bytes for a DMA or a
    gather, elements or beats for the SIMD and the reshape; None if the arguments do not say."""
    a = nd.kernel_args
    if a is None:
        return None
    if all(hasattr(a, x) for x in ("M", "K", "N")):
        return a.M * a.K * a.N
    for x in ("size",):
        if isinstance(getattr(a, x, None), int):
            return getattr(a, x)
    if isinstance(getattr(a, "rows", None), int) and isinstance(getattr(a, "cols", None), int):
        return a.rows * a.cols
    if isinstance(getattr(a, "beats", None), int):
        return a.beats * (a.rows if isinstance(getattr(a, "rows", None), int) else 1)
    tb = getattr(a, "temporal_bounds_src", None)
    if tb:
        n = 1
        for b in tb:
            n *= int(b)
        return n
    return None


def kind_of(name):
    """A task's kind: its name without numbers; per-slice tasks (_j) count as per-iteration ones (_i), and a
    per-row-tile dequantise (deq_cXmY) as the whole-chunk one."""
    name = re.sub(r"_j(\d)", r"_i\1", name)
    name = re.sub(r"_i(\d+)b\d+", r"_i\1", name)          # a block of a split task (x_blocks, post_blocks)
    name = re.sub(r"^deq_c(\d+)m\d+", r"deq_c\1", name)
    return re.sub(r"\d+", "#", name)


def scaled(S_target, S_meas, dur, gemm_model=None):
    """Durations for every task of S_target: measured by name when the measured run had that task, else the
    measured median of the same kind scaled by size (array steps, bytes, elements). gemm_model (fixed us,
    ns per step for gate|up, ns per step for down) replaces every GEMM's duration."""
    ksz = collections.defaultdict(list)
    for nd in S_meas.dfg.node_list:
        n = nd.node_name
        z = size_of(nd)
        if n in dur and z:
            ksz[kind_of(n)].append(dur[n] / z)
    rate = {k: sorted(v)[len(v) // 2] for k, v in ksz.items()}
    msz = {nd.node_name: size_of(nd) for nd in S_meas.dfg.node_list}
    steps, last = {}, {}
    for nd in sorted(S_target.dfg.node_list, key=lambda t: t.node_id):
        key = (nd.assigned_chiplet_id, nd.assigned_cluster_id, nd.assigned_core_id)
        if (nd.kernel_name or "").endswith("gemm_full"):
            last[key] = size_of(nd)
        elif (nd.kernel_name or "").endswith("gemm_minimal"):
            steps[nd.node_name] = last.get(key)
    out, how = {}, collections.Counter()
    for nd in S_target.dfg.node_list:
        n = nd.node_name
        z = size_of(nd)
        if gemm_model and re.match(r"(gu|dn)_c\d+_i\d", n):
            f, rg, rd = gemm_model[:3]
            if (nd.kernel_name or "").endswith("gemm_minimal"):
                f = gemm_model[3] if len(gemm_model) > 3 else f
                z = steps.get(n) or z
            if not z:
                continue
            out[n] = f + z * (rg if n.startswith("gu") else rd) * 1e-3
            how["gemm model"] += 1
        elif n in dur and (not z or not msz.get(n) or msz[n] == z):
            out[n] = dur[n]
            how["measured"] += 1
        elif n in dur:
            out[n] = dur[n] * z / msz[n]                 # the same task, another size: scaled
            how["measured, scaled"] += 1
        elif z and kind_of(n) in rate:
            out[n] = rate[kind_of(n)] * z
            how["kind, scaled"] += 1
    return out, how


def replay(S, header, dur, wq=32, msg_us=1.0, fetch_us=0.05, dc_us=0.01, free_relays=False):
    from chiplet_mc import desc_rows
    dfg = S.dfg
    node = {str(n.node_id): n for n in dfg.node_list}
    streams = collections.defaultdict(list)                  # chip -> [(nid, type, core key)]
    free = set()                                             # --free-relays: relays outside the queues (a bound)
    for chip, nid, ty, cl, core in desc_rows(open(header).read()):
        if free_relays and node[nid].node_name.startswith("relay"):
            free.add(nid)
            continue
        streams[chip].append((nid, ty, (chip, cl, core)))
    preds = {str(n.node_id): [str(q.node_id) for q in dfg.predecessors(n)] for n in dfg.node_list}
    chip_of = {nid: f"{node[nid].assigned_chiplet_id:02x}" for nid in node}
    fetch_t, release_t, done_t, why = {}, {}, {}, {}
    queues = collections.defaultdict(collections.deque)      # core key -> fetched, not released
    core_free = collections.defaultdict(float)
    last_rel = collections.defaultdict(float)
    released = collections.defaultdict(list)                 # core key -> done times of released real tasks
    ptr = {c: 0 for c in streams}
    last_task, last_rel_task = {}, {}
    last_fetch = {c: 0.0 for c in streams}
    missing = set()
    total = sum(len(v) for v in streams.values())
    now = 0.0                                                # events are taken in time order
    while len(release_t) - len([f for f in free if f in release_t]) < total:
        for f in [f for f in free if f not in done_t and all(p in done_t for p in preds.get(f, []))]:
            ps = preds.get(f, [])
            release_t[f] = done_t[f] = max([0.0] + [done_t[p] + (msg_us if chip_of[p] != chip_of[f] else 0.0) for p in ps])
            why[f] = ("relay, free", max(ps, key=lambda p: done_t[p]) if ps else None)
        best = None
        # fetches: the next descriptor of every chip whose core queue has room
        for c, lst in streams.items():
            if ptr[c] < len(lst):
                nid, ty, key = lst[ptr[c]]
                if len(queues[key]) < wq:
                    t = max(now, last_fetch[c] + fetch_us)
                    if best is None or t < best[0]:
                        best = (t, "F", c)
        # releases: every core's queue head whose predecessors are all done
        for key, q in queues.items():
            if not q:
                continue
            nid, ty = q[0]
            ps = preds.get(nid, [])
            if any(p not in done_t for p in ps):
                continue
            cands = [(fetch_t[nid], ("fetch", None)), (last_rel[key] + dc_us, ("queue head", last_rel_task.get(key)))] + \
                    [(done_t[p] + (msg_us if chip_of[p] != chip_of[nid] else 0.0), ("dep", p)) for p in ps]
            if ty == "0" and len(released[key]) >= CHECKOUT:
                cands.append((released[key][-CHECKOUT], ("checkout full", None)))   # a core's done times only grow
            t, r = max(cands, key=lambda c: c[0])
            if t < now:
                t, r = now, ("clock", None)
            if best is None or t < best[0]:
                best = (t, "R", key, r)
        if best is None:
            raise RuntimeError(f"replay stuck with {total - len(release_t)} descriptors unreleased")
        t, what, x = best[:3]
        now = t
        if what == "F":
            nid, ty, key = streams[x][ptr[x]]
            ptr[x] += 1
            last_fetch[x] = t
            fetch_t[nid] = t
            queues[key].append((nid, ty))
        else:
            nid, ty = queues[x].popleft()
            release_t[nid] = t
            why[nid] = best[3]
            last_rel_task[x] = nid
            last_rel[x] = t
            if ty == "0":
                name = node[nid].node_name
                d = dur.get(name)
                if d is None:
                    missing.add(name)
                    d = 0.0
                start = max(t, core_free[x])
                if core_free[x] > t:
                    why[nid] = ("core busy", last_task.get(x))
                last_task[x] = nid
                done_t[nid] = start + d
                core_free[x] = done_t[nid]
                released[x].append(done_t[nid])
            else:
                done_t[nid] = t
    by_name = {node[n].node_name: (release_t[n], done_t[n]) for n in done_t}
    nm = lambda n: node[n].node_name
    def chain(end_name, steps=40):
        """The tasks that set the end task's time, walked back: (name, release, done, reason)."""
        nid = next(n for n in done_t if nm(n) == end_name)
        out = []
        while nid is not None and len(out) < steps:
            r, p = why.get(nid, ("?", None))
            out.append((nm(nid), release_t[nid], done_t[nid], r))
            nid = p
        return out
    return by_name, missing, chain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workload", help="e.g. dsv2_l0_prefill/all_sp_v20")
    ap.add_argument("--run", required=True, help="the measured run dir (runs/<tag>)")
    ap.add_argument("--run-wl", default="dsv2_l0_prefill/all", help="the workload that run used (its final_dfg.csv names the ids)")
    ap.add_argument("--wq", type=int, default=32)
    ap.add_argument("--msg-us", type=float, default=1.0)
    ap.add_argument("--compare", action="store_true", help="print the run's measured milestones too")
    ap.add_argument("--free-relays", action="store_true", help="a bound: relays outside the in-order queues")
    ap.add_argument("--scale", action="store_true", help="tasks the run did not have: their kind's measured time, scaled by size "
                    "(generates --run-wl too)")
    ap.add_argument("--gemm-model", metavar="FIXED,RATE_GU,RATE_DN[,FIXED_MIN]",
                    help="every GEMM = FIXED us + steps x RATE ns (gate|up, down); FIXED_MIN for gemm_minimal tasks")
    ap.add_argument("--phases", action="store_true", help="per-cluster phase breakdown of the busiest GEMM core")
    ap.add_argument("--why", type=int, default=0, metavar="N", help="print the last N steps of the critical path")
    ap.add_argument("--why-from", action="append", default=[], metavar="TASK",
                    help="walk the --why path back from this task instead of the last store (repeatable)")
    ap.add_argument("--dur", action="append", default=[], metavar="REGEX",
                    help="print the replayed release, start and done of the tasks whose name matches")
    a = ap.parse_args()
    meas = measured(a.run, f"{W}/{a.run_wl}/final_dfg.csv")
    dur = durations(meas)
    with tempfile.TemporaryDirectory(prefix="replay_pf_") as out:
        S = generate(a.workload, out)
        if a.scale or a.gemm_model:
            with tempfile.TemporaryDirectory(prefix="replay_pf_meas_") as out2:
                Sm = generate(a.run_wl, out2) if a.scale else S
            gm = tuple(float(x) for x in a.gemm_model.split(",")) if a.gemm_model else None
            dur, how = scaled(S, Sm, dur, gm)
            print("durations: " + ", ".join(f"{k} {v}" for k, v in how.items()))
        res, missing, chain = replay(S, os.path.join(out, "offload_bingo_hw.h"), dur, a.wq, a.msg_us,
                                     free_relays=a.free_relays)
    t0 = res["A0_start_k00_cl0"][1]
    m0 = meas["A0_start_k00_cl0"][1] if "A0_start_k00_cl0" in meas else None
    NI = S.NI
    print(f"replay of {a.workload} with the durations of {os.path.basename(a.run)} (waiting queue {a.wq}, "
          f"cross-chip message {a.msg_us} us); {len(missing)} real tasks without a measured duration (0 used)")
    if getattr(S, "single", False):              # single pass: one store per down slice
        rows = [("window", f"y_store_j{S.NDN - 1}_k00_cl0")] + \
               [(f"store slice {j}", f"y_store_j{j}_k00_cl0") for j in range(S.NDN)]
    else:
        rows = [("window", f"y_store_i{NI - 1}_k00_cl0")] + \
               [(f"store i{i}", f"y_store_i{i}_k00_cl0") for i in range(NI)] + \
               [(f"X put i{i} released", f"x_bput_i{i}_k00_cl0") for i in range(NI)]
    for label, n in rows:
        if n not in res:
            continue
        r = res[n][1] - t0 if not label.startswith("X") else res[n][0] - t0
        line = f"  {label:22s} replay {r:8.1f} us"
        if a.compare and n in meas and m0 is not None:
            mv = (meas[n][1] - m0) if not label.startswith("X") else (meas[n][0] - m0)
            line += f"   measured {mv:8.1f} us   ({100 * (r - mv) / mv:+5.1f} %)"
        print(line)
    if a.phases:
        gem = [(n, r) for n, r in res.items() if re.match(r"(gu|dn)_c\d+_i\d", n)]
        per = collections.defaultdict(list)
        for n, (rel, done) in gem:
            per[n.split("_k")[-1]].append((done - dur.get(n, 0.0) - t0, done - t0, n))
        busy = {k: sum(d - s for s, d, _ in v) for k, v in per.items()}
        kmax = max(busy, key=busy.get)
        v = sorted(per[kmax])
        gu = [x for x in v if x[2].startswith("gu")]; dn = [x for x in v if x[2].startswith("dn")]
        WIN = res[rows[0][1]][1] - t0
        idle_gu = (gu[-1][1] - gu[0][0]) - sum(d - s for s, d, _ in gu)
        idle_dn = (dn[-1][1] - dn[0][0]) - sum(d - s for s, d, _ in dn)
        print(f"busiest GEMM core {kmax}: busy {busy[kmax]:.1f} of {WIN:.1f} us ({100 * busy[kmax] / WIN:.1f} %)")
        print(f"  head (start anchor -> first gate|up) {gu[0][0]:.1f} us; gate|up {gu[0][0]:.1f} -> {gu[-1][1]:.1f} us "
              f"(busy {sum(d - s for s, d, _ in gu):.1f}, idle {idle_gu:.1f}); gap -> first down {dn[0][0] - gu[-1][1]:.1f} us; "
              f"down {dn[0][0]:.1f} -> {dn[-1][1]:.1f} (busy {sum(d - s for s, d, _ in dn):.1f}, idle {idle_dn:.1f}); tail {WIN - dn[-1][1]:.1f} us")
        big = sorted(((b[0] - a[1], a[2], b[2]) for a, b in zip(v, v[1:])), reverse=True)[:6]
        for g, a_, b_ in big:
            print(f"    idle {g:6.1f} us between {a_} and {b_}")
    for last in (a.why_from or [rows[0][1]]) if a.why else []:
        print(f"critical path into {last}, walked back (us after the start anchor; reason = what set the release or start):")
        for n, r, d, why in chain(last, a.why):
            print(f"  {n:34s} released {r - t0:8.1f}  done {d - t0:8.1f}  ({why})")
    for rx in a.dur:
        for n, (r, d) in sorted(res.items(), key=lambda x: x[1][1]):
            if re.search(rx, n):
                print(f"  {n:40s} released {r - t0:8.1f}  start {d - dur.get(n, 0.0) - t0:8.1f}  done {d - t0:8.1f}")
    if missing:
        print("  without duration: " + ", ".join(sorted(missing)[:8]) + (" ..." if len(missing) > 8 else ""))


if __name__ == "__main__":
    main()
