#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The cerf_node numbers, from one simulation run.

    report.py <run bin dir> [--map cerf_node_map.csv] [--csv out.csv]

Reads bingo_progress.log (simulation time, ps: [BINGO EV] task events, [BINGO CERF] CERF writes,
[BINGO LAUNCH] START writes) and the hosts' uart_chip_*.log (the host arm's own cycles), and
joins them through the map main_bingo.py wrote. Every interval is in ns; the host arm's own
counts are host cycles.

  cerf    write -> node   the gate's CERF write landing -> the gated node starts
          done -> node    the gate's done -> the node starts (the manager's part)
          gate -> node    the gate starts -> the node starts
  skip    write -> after  the CERF write (bit cleared) -> the skipped node's consumer starts
  cross   as cerf, the node on the far chiplet: the window rides the dep-set (cerf_carry)
  dev     as cerf, the gate on a device core
          store issue->latch: the CERF store leaving the gate's core ([CERF ISSUE]) -> it latching
  tiles   the device gate's CERF writes, and which tiles started against which it enabled
  host    build, sync and the launch call are host cycles; restart is the START write landing
          -> the relaunched node starts, and fetch the START write -> its descriptor fetched
  ways    a whole-CERF update several ways (cerf_node_kernels.h, __snax_bingo_kernel_cerf_cost), from
          0x00's host and from a device core: each try lies between two marker stores to read-only
          CERF_STATUS words. start -> latched: the first marker -> the try's last CERF write landing;
          start -> confirmed: the first marker -> the second, which follows the status wait (or fence).
          The host also prints its own cycles a way ("CNW").
"""

import argparse
import collections
import csv
import glob
import os
import re
import statistics
import sys

EV = re.compile(r"\[BINGO EV\] t=(\d+) chip=([0-9a-f]+) cl=(\d+) co=(\d+) ev=(\w) id=(\d+)")
CERF = re.compile(r"\[BINGO CERF\] t=(\d+) chip=([0-9a-f]+) mask=([0-9a-f]+) data=([0-9a-f]+)")
LAUNCH = re.compile(r"\[BINGO LAUNCH\] t=(\d+) chip=([0-9a-f]+) base=([0-9a-f]+) n=(\d+)")
ISSUE = re.compile(r"\[CERF ISSUE\] t=(\d+) chip=([0-9a-f]+) addr=[0-9a-f]+ (\S+)")
INST = re.compile(r"i_hemaia_(\d+)_(\d+)\.")
HOSTARM = re.compile(r"^CN ([0-9a-f]{2}) (\d+) (\d+)(?: (\d+) (\d+) (\d+))?\s*$")
HOSTWAY = re.compile(r"^CNW (\d+) (\d+) (\d+) (\d+)\s*$")
# The ways of a whole-CERF update, in the order the micro-benchmarks run them, a writer
WAYS = {"host": ["one store (a gate)", "SET+CLR, every word", "RMW, status wait", "RMW, fence",
                 "RMW, 32-group code"],
        "dev": ["one store (a gate)", "SET+CLR, every word", "RMW, status wait"]}
MARK_START, MARK_END = 0x88, 0x7C        # CERF_STATUS_3 and CERF_STATUS_0: the low address byte


def parse_log(path):
    ev = collections.defaultdict(lambda: collections.defaultdict(list))   # (chip, id) -> ev -> [t]
    cerf = collections.defaultdict(list)                                  # chip -> [(t, mask, data)]
    launch = collections.defaultdict(list)                                # chip -> [(t, base, n)]
    issue = collections.defaultdict(list)                                 # chip -> [t]: a CERF store leaving a core
    stores = collections.defaultdict(list)    # (chip, "host" | "dev") -> [(t, low address byte)]
    with open(path, errors="replace") as f:
        for line in f:
            m = ISSUE.match(line)
            if m:
                chip = int(m.group(2), 16)
                if chip == 0xFF:                  # a cluster's port: the chip is in the instance path
                    i = INST.search(m.group(3))
                    chip = (int(i.group(1)) << 4) | int(i.group(2))
                issue[chip].append(int(m.group(1)))
                port = "host" if "cva6" in m.group(3) else "dev"
                stores[(chip, port)].append((int(m.group(1)), int(line.split("addr=")[1].split()[0], 16) & 0xFF))
                continue
            m = EV.match(line)
            if m:
                t, chip, _, _, e, tid = m.groups()
                ev[(int(chip, 16), int(tid))][e].append(int(t))
                continue
            m = CERF.match(line)
            if m:
                cerf[int(m.group(2), 16)].append((int(m.group(1)), int(m.group(3), 16),
                                                  int(m.group(4), 16)))
                continue
            m = LAUNCH.match(line)
            if m:
                launch[int(m.group(2), 16)].append((int(m.group(1)), int(m.group(3), 16),
                                                    int(m.group(4))))
    return ev, cerf, launch, issue, stores


def parse_uart(bin_dir):
    rows = collections.defaultdict(dict)     # (chip, round) -> {key: cycles}
    ways = {}                                # way -> {key: host cycles}
    for path in glob.glob(os.path.join(bin_dir, "uart_chip_*.log")):
        with open(path, errors="replace") as f:
            for line in f:
                w = HOSTWAY.match(line.strip())
                if w:
                    ways[int(w.group(1))] = {"issue": int(w.group(2)), "total": int(w.group(3)),
                                                      "total_min": int(w.group(4))}
                    continue
                m = HOSTARM.match(line.strip())
                if not m:
                    continue
                g = m.groups()
                kv = ({"build": g[2], "sync": g[3], "launch_call": g[4], "launch_to_seen": g[5]}
                      if g[3] is not None else {"sync": g[2]})
                rows[(int(g[0], 16), int(g[1]))].update({k: int(v) for k, v in kv.items()})
    return rows, ways


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n=0"
    return (f"n={len(vals):3d}  min {min(vals):8.1f}  median {statistics.median(vals):8.1f}  "
            f"mean {statistics.mean(vals):8.1f}  max {max(vals):8.1f}")


def first(lst):
    return lst[0] if lst else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bin_dir")
    ap.add_argument("--map", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "cerf_node_map.csv"))
    ap.add_argument("--csv", default=None, help="write every per-round value here")
    a = ap.parse_args()

    ev, cerf, launch, issue, stores = parse_log(os.path.join(a.bin_dir, "bingo_progress.log"))
    host, hostways = parse_uart(a.bin_dir)
    with open(a.map) as f:
        rows = list(csv.DictReader(f))

    ns = lambda ps: None if ps is None else ps / 1000.0
    out = []                                   # (arm, round, metric, value)
    metrics = collections.defaultdict(list)

    def put(arm, r, key, val):
        out.append((arm, r, key, val))
        metrics[(arm, key)].append(val)

    def cerf_write(chip, t0, t1, group):
        """The write of `group` landing on `chip` within [t0, t1]."""
        for t, mask, data in cerf.get(chip, []):
            if t0 <= t <= t1 and (mask >> group) & 1:
                return t
        return None

    problems = []
    for row in rows:
        arm = row["arm"]
        if arm not in ("cerf", "skip", "cross", "dev"):
            continue
        r = int(row["round"])
        gchip, nchip = int(row["gate_chip"], 16), int(row["node_chip"], 16)
        g = ev[(gchip, int(row["gate"]))]
        n = ev[(nchip, int(row["node"]))]
        aft = ev[(nchip, int(row["after"]))]
        gS, gD = first(g["S"]), first(g["D"])
        if gS is None or gD is None:
            problems.append(f"{arm} round {r}: the gate never ran")
            continue
        w = cerf_write(gchip, gS, gD, int(row["group"]))
        if w is None:
            problems.append(f"{arm} round {r}: no CERF write of group {row['group']} while the gate ran")
        nS, aS = first(n["S"]), first(aft["S"])
        if arm == "skip":
            if nS is not None:
                problems.append(f"skip round {r}: the node started though its group was cleared")
            put(arm, r, "write->after", ns(aS - w) if (aS and w) else None)
            put(arm, r, "done->after", ns(aS - gD) if aS else None)
        else:
            if nS is None:
                problems.append(f"{arm} round {r}: the gated node never started")
                continue
            put(arm, r, "write->node", ns(nS - w) if w else None)
            put(arm, r, "done->node", ns(nS - gD))
            put(arm, r, "gate->node", ns(nS - gS))
        put(arm, r, "gate start->write", ns(w - gS) if w else None)
        i = max((t for t in issue.get(gchip, []) if w and t <= w), default=None)
        put(arm, r, "store issue->latch", ns(w - i) if (w and i is not None) else None)
        put(arm, r, "write->gate done", ns(gD - w) if w else None)

    # The tile arm: one device gate, many tiles.
    tiles = [row for row in rows if row["arm"].startswith("tile")]
    if tiles:
        chip = int(tiles[0]["gate_chip"], 16)
        g = ev[(chip, int(tiles[0]["gate"]))]
        gS, gD = first(g["S"]), first(g["D"])
        writes = [t for t, _, _ in cerf.get(chip, []) if gS is not None and gS <= t <= (gD or t)]
        started = {int(row["node"]) for row in tiles if ev[(chip, int(row["node"]))]["S"]}
        enabled = {int(row["node"]) for row in tiles if row["arm"] == "tile1"}
        wrong = started ^ enabled
        if wrong:
            problems.append(f"tiles: {len(started)} started, {len(enabled)} enabled, "
                            f"{len(wrong)} differ (first ids {sorted(wrong)[:6]})")
        tS = [first(ev[(chip, int(row["node"]))]["S"]) for row in tiles if row["arm"] == "tile1"]
        tS = [t for t in tS if t is not None]
        aS = first(ev[(chip, int(tiles[0]["after"]))]["S"])
        put("tiles", 0, "CERF writes", len(writes))
        if writes and tS:
            put("tiles", 0, "gate start->first write", ns(writes[0] - gS))
            put("tiles", 0, "last write->first tile", ns(min(tS) - writes[-1]))
        if gD is not None and aS is not None:
            put("tiles", 0, "gate done->after", ns(aS - gD))
        put("tiles", 0, "tiles started", len(started))
        put("tiles", 0, "tiles enabled", len(enabled))

    # The host arm: R relaunches of one node.
    hrow = next((row for row in rows if row["arm"] == "host"), None)
    if hrow:
        nchip = int(hrow["node_chip"], 16)
        n = ev[(nchip, int(hrow["node"]))]
        relaunch = [x for x in launch.get(nchip, []) if x[2] == 1]
        for r, (t, _, _) in enumerate(relaunch):
            s = next((x for x in n["S"] if x >= t), None)
            fe = next((x for x in n["F"] if x >= t), None)
            put("host", r, "restart (START -> node)", ns(s - t) if s else None)
            put("host", r, "START -> fetched", ns(fe - t) if fe else None)
        for (chip, r), kv in sorted(host.items()):
            if chip == nchip:
                for key in ("build", "launch_call", "launch_to_seen"):
                    if key in kv:
                        put("host", r, f"{key} (cc)", kv[key])
                put("host", r, "sync node chiplet (cc)", kv.get("sync"))
            put("host", r, "sync any chiplet (cc)", kv.get("sync"))
        per_round = collections.defaultdict(list)
        for (chip, r), kv in host.items():
            if "sync" in kv:
                per_round[r].append(kv["sync"])
        for r, v in sorted(per_round.items()):
            put("host", r, "sync slowest chiplet (cc)", max(v))
        if len(relaunch) != len({r for (c, r) in host if c == nchip}):
            problems.append(f"host: {len(relaunch)} one-task launches on 0x{nchip:02x} logged, "
                            f"{len({r for (c, r) in host if c == nchip})} rounds printed")

    # The whole-CERF update ways: the tries between marker stores, a writer, on chip 0x00
    for port, names in WAYS.items():
        tries, open_ = [], None
        for t, lo in stores.get((0, port), []):
            if lo == MARK_START:
                open_ = (t, [])
            elif lo == MARK_END and open_:
                tries.append((open_[0], open_[1], t))
                open_ = None
            elif open_:
                open_[1].append(t)
        if not tries:
            continue
        if len(tries) % len(names):
            problems.append(f"ways {port}: {len(tries)} tries, not a multiple of {len(names)} ways")
            continue
        reps = len(tries) // len(names)
        for i, (t0, st, t1) in enumerate(tries):
            arm = f"ways {port}: {names[i // reps]}"
            lat = [t for t, _, _ in cerf.get(0, []) if t0 <= t <= t1]
            put(arm, i % reps, "stores", len(st))
            put(arm, i % reps, "CERF writes", len(lat))
            put(arm, i % reps, "start -> latched", ns(lat[-1] - t0) if lat else None)
            put(arm, i % reps, "start -> confirmed", ns(t1 - t0))
            if port == "host" and i // reps in hostways:
                kv = hostways[i // reps]
                if i % reps == 0:
                    for key in ("issue", "total", "total_min"):
                        put(arm, 0, f"host cycles: {key}", kv[key])

    order = ["cerf", "skip", "cross", "dev", "tiles", "host"] + [f"ways {p}: {n}" for p, ns_ in WAYS.items()
                                                                for n in ns_]
    for arm in order:
        keys = [k for (a_, k) in metrics if a_ == arm]
        if not keys:
            continue
        print(f"\n{arm}")
        for k in dict.fromkeys(keys):
            print(f"  {k:28s} {stats(metrics[(arm, k)])}")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print("  " + p)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "round", "metric", "value"])
            w.writerows(out)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
