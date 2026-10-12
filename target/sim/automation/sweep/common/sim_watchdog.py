#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""sim_watchdog.py: the hang guard of one HeMAiA RTL simulation. It reports and it stops.

  sim_watchdog.py watch   --run-dir RUNS/TAG [--profile NAME] [--set key=value ...]
  sim_watchdog.py status  RUNS/TAG|TAG.events    one line: what the run is doing, STALE if the watchdog died
  sim_watchdog.py analyse --log bingo_progress.log [--probes snax_probes.log]   replay a finished log
  sim_watchdog.py params  [--profile NAME]       every parameter, its value, what it does

LIVENESS IS THE SCHEDULER. The bingo_hw_manager of every chiplet is monitored in the testbench
(soc_probes.sv, snax_bingo_progress_probe) and writes bingo_progress.log next to the simulator: every
task fetched, released, started and done, and the counters and queue heads of every chiplet every few
microseconds of simulated time. A workload moves exactly when those counters move. A core's
instruction trace is not a clock: a core parks in a blocking read whenever its next task waits for a
dependency, and a long task writes nothing.

It watches ONE simulator, found by its process name and the directory it runs in, and acts on that PID
only (SIGINT so the traces flush, SIGKILL when it does not leave). It stops it on the first of:
  HANG-SCHED  a chiplet has work outstanding and no chiplet fetched, released, started or finished a task
              for max(stall_floor_us, stall_factor x the longest task seen) of simulated time
  HANG-AXI    an AXI link has offered a valid nobody takes (a [AXI-STALL] line) AND the scheduler is quiet
              for axi_confirm_us (axi_stall=stop stops on the line alone, warn only reports it)
  WEDGE       the simulation clock and every log stood still for wedge_min of wall time
  ASSERT      an assertion failed in the transcript (sim_live.log)
  QUOTA / LOG disk headroom below min_headroom_gb; the transcript or the trace directory over its cap
  BOUND       the simulation clock passed bound_us
  NOT-ARMED   no chiplet started its task list by arm_deadline_us
  THREADS     fewer simulator threads than min_threads (a parallel simulator that did not get its cores)
  HANG-EXIT   every chiplet drained its list and the simulator is still up after post_complete_min
It writes <run dir>.events (HEARTBEAT every heartbeat_min, STOP, warnings), <tag>.hang.txt (the evidence
of a stop), and <tag>.watchdog.json (the live status, rewritten every cycle; `status` reads it).
Nothing is ever killed by a name pattern. A check that raises is reported once and the loop goes on.
"""
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.dirname(HERE)
PROFILES = os.path.join(HERE, "sim_watchdog.profiles.json")

# name, default, meaning (the type of the default is the type of the parameter)
PARAMS = [
    ("interval_s", 30, "seconds between two checks"),
    ("heartbeat_min", 10.0, "minutes between two HEARTBEAT lines in the events file"),
    ("find_timeout_s", 7200, "seconds to wait for the simulator process to appear"),
    ("sim_comm", "occamy_chip.vcs", "process name of the simulator (/proc/PID/comm)"),
    ("bound_us", 0, "stop when the simulation clock passes this many us (0 = off)"),
    ("arm_deadline_us", 0, "stop when no chiplet started its task list by this simulated time in us (0 = off)"),
    ("stall_floor_us", 300, "scheduler quiet time that is a hang, at least, in simulated us"),
    ("stall_factor", 6.0, "... and at least this many times the longest task (started to done) seen so far"),
    ("stall_cap_us", 1000, "... but never more than this (0 = no cap)"),
    ("axi_stall", "confirm", "[AXI-STALL] lines: off | warn | confirm (stop when the scheduler is also quiet) | stop"),
    ("axi_confirm_us", 100, "scheduler quiet time that confirms an AXI stall, in simulated us"),
    ("anomaly", "warn", "[BINGO ANOMALY] lines (a queue push the FIFO accounting forbids): off | warn | stop"),
    ("wedge_min", 30.0, "wall minutes with the clock and every log unchanged = a frozen simulator; must exceed "
                      "the wall time of one progress period at the slowest expected rate"),
    ("min_headroom_gb", 25, "stop below this much room under the disk quota"),
    ("quota_hard_gb", 150, "the hard quota of the account in GB"),
    ("max_log_gb", 2, "stop when the transcript sim_live.log is over this size"),
    ("max_trace_gb", 12, "stop when the trace directory is over this size (0 = off)"),
    ("min_threads", 0, "stop when the simulator has fewer threads than this after threads_grace_s (0 = off)"),
    ("threads_grace_s", 120, "seconds after the simulator appears before the thread count is judged"),
    ("min_rate_us_per_min", 0.0, "report SLOW when the simulation advances slower than this, simulated us per wall minute (0 = off)"),
    ("rate_grace_min", 20.0, "wall minutes of run before the rate is judged"),
    ("post_complete_min", 30.0, "wall minutes the simulator may stay up after every chiplet drained its list"),
    ("stop_grace_s", 60, "seconds between SIGINT and SIGKILL"),
    ("on_stop_cmd", "", "shell command run after the simulator is gone; {task_dir} {tag} {sweep} are substituted"),
    ("dfg", "", "final_dfg.csv of the workload: names the task ids in the evidence"),
]
DEFAULTS = {n: d for n, d, _ in PARAMS}


def coerce(name, text):
    d = DEFAULTS[name]
    if isinstance(d, bool):
        return str(text).lower() in ("1", "true", "yes")
    if isinstance(d, int):
        return int(float(text))
    if isinstance(d, float):
        return float(text)
    return str(text)


def load_params(profile, overrides):
    p = dict(DEFAULTS)
    if profile and profile != "default":
        with open(PROFILES) as f:
            table = json.load(f)
        if profile not in table:
            sys.exit("unknown profile %r, known: %s" % (profile, ", ".join(sorted(table))))
        for k, v in table[profile].items():
            if k.startswith("_"):
                continue
            if k not in DEFAULTS:
                sys.exit("profile %s: unknown parameter %s" % (profile, k))
            p[k] = coerce(k, v)
    for kv in overrides or []:
        k, _, v = kv.partition("=")
        if k not in DEFAULTS:
            sys.exit("unknown parameter %r (see `params`)" % k)
        p[k] = coerce(k, v)
    return p


# ---------------------------------------------------------------------------------------------------
# the scheduler log
# ---------------------------------------------------------------------------------------------------
KV = re.compile(r"(\w+)=(\S+)")
CELL = re.compile(r"(\d+)\.(\d+):w(-?\d+)@(\S+?),r(-?\d+)@(\S+?),c(-?\d+)@(\S+?),x(-?\d+)")


def parse_kv(text):
    return {k: v for k, v in KV.findall(text)}


def parse_cells(text):
    out = []
    for m in CELL.finditer(text):
        cl, co, w, wh, r, rh, c, ch, x = m.groups()
        out.append(dict(cl=int(cl), co=int(co), w=int(w), wh=wh, r=int(r), rh=rh, c=int(c), ch=ch, x=int(x)))
    return out


class Chip:
    def __init__(self, cid):
        self.id = cid
        self.armed = False
        self.complete = False
        self.ep = 0
        self.fetched = 0
        self.nlist = 0
        self.rel = 0
        self.start = 0
        self.done = 0
        self.quiet_cc = 0
        self.anom = 0
        self.t = 0
        self.cells = []
        self.stall_hint = None

    def activity(self):
        return self.fetched + self.rel + self.start + self.done


class Bingo:
    """The state of every chiplet's scheduler as the log tells it."""

    def __init__(self):
        self.chips = {}
        self.sim_ps = 0
        self.last_progress_ps = 0
        self.ever_armed = False
        self.first_arm_ps = None
        self.longest_task_ps = 0
        self.starts = {}
        self.anomalies = []
        self.stall_hints = []
        self.lines = 0

    def chip(self, cid):
        if cid not in self.chips:
            self.chips[cid] = Chip(cid)
        return self.chips[cid]

    def _progress(self, t):
        if t > self.last_progress_ps:
            self.last_progress_ps = t

    def feed(self, line):
        if not line.startswith("[BINGO "):
            return
        tag, _, rest = line[7:].partition("]")
        head, _, tail = rest.partition(" |")
        kv = parse_kv(head)
        if "t" not in kv or "chip" not in kv:
            return
        try:
            t = int(kv["t"])
        except ValueError:
            return
        self.lines += 1
        if t > self.sim_ps:
            self.sim_ps = t
        ch = self.chip(kv["chip"])
        if tag == "EV":
            self._progress(t)
            key = (ch.id, kv.get("cl"), kv.get("co"), kv.get("id"))
            if kv.get("ev") == "S":
                self.starts[key] = t
            elif kv.get("ev") == "D":
                s = self.starts.pop(key, None)
                if s is not None and kv.get("ty") != "1" and t - s > self.longest_task_ps:
                    self.longest_task_ps = t - s
        elif tag == "ARM":
            ch.armed, ch.complete = True, False
            ch.ep = int(kv.get("ep", ch.ep))
            ch.nlist = int(kv.get("n_task", 0))
            ch.fetched = ch.rel = ch.start = ch.done = 0
            self.ever_armed = True
            if self.first_arm_ps is None:
                self.first_arm_ps = t
            self._progress(t)
        elif tag in ("PROG", "CMPL", "STALL", "FINAL"):
            before = ch.activity()
            ch.t = t
            ch.ep = int(kv.get("ep", ch.ep))
            ch.armed = kv.get("armed", "0") == "1"
            f, _, n = kv.get("fetch", "0/0").partition("/")
            ch.fetched, ch.nlist = int(f), int(n or 0)
            ch.rel, ch.start, ch.done = int(kv.get("rel", 0)), int(kv.get("start", 0)), int(kv.get("done", 0))
            ch.quiet_cc = int(kv.get("quiet", 0))
            ch.anom = int(kv.get("anom", 0))
            ch.cells = parse_cells(tail)
            if ch.armed:
                self.ever_armed = True
                if self.first_arm_ps is None:
                    self.first_arm_ps = t
            if tag == "CMPL":
                ch.complete = True
                self._progress(t)
            elif tag == "STALL":
                ch.stall_hint = (t, ch.quiet_cc)
                self.stall_hints.append((ch.id, t, ch.quiet_cc))
            if ch.activity() > before:
                self._progress(t)
        elif tag == "ANOMALY":
            if len(self.anomalies) < 40:
                self.anomalies.append(line.strip())

    def armed(self):
        return [c for c in self.chips.values() if c.armed]

    def totals(self):
        cs = self.chips.values()
        return dict(fetched=sum(c.fetched for c in cs), n_task=sum(c.nlist for c in cs),
                    rel=sum(c.rel for c in cs), start=sum(c.start for c in cs), done=sum(c.done for c in cs))


class Names:
    """final_dfg.csv: task id -> (chip, cluster, name). The file is read on first use (a run's build may
    still be writing it when the watchdog starts); a name is shown only when chip and cluster agree."""

    def __init__(self, path):
        self.path = path
        self.rows = None

    def _load(self):
        self.rows = {}
        if self.path and os.path.exists(self.path):
            with open(self.path, errors="replace") as f:
                for r in csv.DictReader(f):
                    try:
                        self.rows[int(r["ID"])] = (r["Chiplet"], int(r["Cluster"]), r["Name"], r.get("Kernel", ""))
                    except (KeyError, ValueError):
                        pass

    def get(self, chip, cl, tid):
        if self.rows is None or (not self.rows and self.path):
            self._load()
        try:
            c, k, n, _ = self.rows[int(tid)]
        except (KeyError, ValueError):
            return str(tid)
        return n if (c.lower() == chip.lower() and k == int(cl)) else "%s(?)" % tid

    def is_exit(self, chip, cl, tid):
        """True/False when the DFG names the task, None when it cannot tell."""
        if self.rows is None or (not self.rows and self.path):
            self._load()
        try:
            c, k, n, kern = self.rows[int(tid)]
        except (KeyError, ValueError):
            return None
        if c.lower() != chip.lower() or k != int(cl):
            return None
        return "kernel_exit" in kern or "kernel_exit" in n


class Tail:
    """Reads the new complete lines of a growing file; survives the file being replaced."""

    def __init__(self, path, limit=16 << 20):
        self.path, self.off, self.ino, self.buf, self.limit = path, 0, None, b"", limit

    def size(self):
        try:
            return os.stat(self.path).st_size
        except OSError:
            return 0

    def read(self):
        try:
            st = os.stat(self.path)
        except OSError:
            return []
        if self.ino is not None and (st.st_ino != self.ino or st.st_size < self.off):
            self.off, self.buf = 0, b""
        self.ino = st.st_ino
        if st.st_size <= self.off:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.off)
            data = f.read(self.limit)
        self.off += len(data)
        data = self.buf + data
        parts = data.split(b"\n")
        self.buf = parts.pop()
        return [p.decode("utf-8", "replace") for p in parts]


AXI = re.compile(r"writes out (\d+), reads out (\d+) \| aw (\d)/(\d) w (\d)/(\d) b (\d)/(\d) ar (\d)/(\d) r (\d)/(\d)")


def real_axi_stall(line):
    """An [AXI-STALL] line that is a stall: a valid offered that nobody takes, or writes outstanding.
    Not one: a single read outstanding with every valid low (a host blocked on a register that answers when
    work arrives, by design), nor the unsigned counters of a few peripheral links wrapping below zero."""
    if not line.startswith("[AXI-STALL]"):
        return False
    m = AXI.search(line)
    if not m:
        return True
    w = int(m.group(1))
    valid = [int(m.group(i)) for i in (3, 5, 7, 9, 11)]
    return any(valid) or 0 < w < 100000


# ---------------------------------------------------------------------------------------------------
# the judgement from the logs alone (no process, no clock): what `analyse` replays
# ---------------------------------------------------------------------------------------------------
class Judge:
    def __init__(self, p, names=None):
        self.p = p
        self.b = Bingo()
        self.names = names or Names("")
        self.axi = []
        self.axi_seen = 0
        self.assert_line = None
        self.anomaly_seen = 0

    def feed_bingo(self, lines):
        for ln in lines:
            self.b.feed(ln)

    def feed_probes(self, lines):
        for ln in lines:
            if real_axi_stall(ln):
                self.axi_seen += 1
                if len(self.axi) < 40:
                    self.axi.append(ln.strip())

    def feed_transcript(self, lines):
        for ln in lines:
            if self.assert_line is None and (ln.startswith('Error: "') or "Too many assertion failures" in ln):
                self.assert_line = ln.strip()

    def drained(self, c):
        """A chiplet whose list is fetched and whose only open task is its exit task. The host's exit kernel
        starts once its other work is done and reports done only when the program ends, so the monitor never
        sees such a chiplet complete. With the DFG the exit task is named; without it, the open task must be
        alone on the highest core index (the host lane) with nothing waiting or ready."""
        if not c.armed or c.fetched < c.nlist:
            return False
        if any(k["w"] or k["r"] for k in c.cells):
            return False
        open_cells = [k for k in c.cells if k["c"] or k["x"]]
        if not open_cells:
            return True
        host = max([k["co"] for k in c.cells] + [4])
        for k in open_cells:
            if k["c"] > 1 or k["x"] > 1:
                return False
            ex = self.names.is_exit(c.id, k["cl"], k["ch"]) if k["ch"] != "-" else None
            if ex is False or (ex is None and k["co"] != host):
                return False
        return True

    def busy(self):
        return [c for c in self.b.chips.values() if c.armed and not self.drained(c)]

    def all_complete(self):
        return self.b.ever_armed and not self.busy()

    def threshold_us(self):
        p = self.p
        t = max(p["stall_floor_us"], p["stall_factor"] * self.b.longest_task_ps / 1e6)
        if p["stall_cap_us"] > 0:
            t = min(t, max(p["stall_cap_us"], p["stall_floor_us"]))
        return t

    def quiet_us(self):
        return max(0, self.b.sim_ps - self.b.last_progress_ps) / 1e6

    def verdict(self):
        """(kind, detail) of the first rule that fires, else None."""
        p, b = self.p, self.b
        sim_us = b.sim_ps / 1e6
        if self.assert_line:
            return "ASSERT", self.assert_line[:200]
        if p["bound_us"] and sim_us > p["bound_us"]:
            return "BOUND", "simulation clock %.1f us past the bound %d us" % (sim_us, p["bound_us"])
        if p["arm_deadline_us"] and not b.ever_armed and sim_us > p["arm_deadline_us"]:
            return "NOT-ARMED", "no chiplet started its task list by %.1f us (deadline %d us)" % (sim_us, p["arm_deadline_us"])
        if p["anomaly"] == "stop" and b.anomalies:
            return "ANOMALY", b.anomalies[0][:200]
        busy = self.busy()
        if busy:
            q, thr = self.quiet_us(), self.threshold_us()
            if self.axi and p["axi_stall"] == "stop":
                return "HANG-AXI", self.axi[0][:200]
            if self.axi and p["axi_stall"] == "confirm" and q >= p["axi_confirm_us"]:
                return "HANG-AXI", "scheduler quiet %.1f us and %s" % (q, self.axi[0][:160])
            if q >= thr:
                return "HANG-SCHED", ("no task fetched, released, started or done on any chiplet for %.1f us "
                                      "(limit %.1f us); %d chiplets have work outstanding" % (q, thr, len(busy)))
        return None

    def summary(self):
        b, t = self.b, self.b.totals()
        return ("sim %.1f us, busy %d/%d chips, fetched %d/%d, released %d, started %d, done %d, quiet %.1f us (limit %.1f)"
                % (b.sim_ps / 1e6, len(self.busy()), len(b.chips), t["fetched"], t["n_task"], t["rel"], t["start"],
                   t["done"], self.quiet_us(), self.threshold_us()))

    def evidence(self):
        b = self.b
        out = ["scheduler at the stop: " + self.summary(), ""]
        busy = sorted(self.busy(), key=lambda c: c.id)
        out.append("chiplets with work outstanding (%d):" % len(busy))
        for c in busy:
            out.append("  chip %s  fetch %d/%d rel %d start %d done %d  quiet %d cycles  last record %.1f us"
                       % (c.id, c.fetched, c.nlist, c.rel, c.start, c.done, c.quiet_cc, c.t / 1e6))
            for k in c.cells:
                def nm(h):
                    return "-" if h == "-" else self.names.get(c.id, k["cl"], h)
                out.append("      cluster %d core %d: waiting %d (head %s), ready %d (head %s), checkout %d (head %s), running %d"
                           % (k["cl"], k["co"], k["w"], nm(k["wh"]), k["r"], nm(k["rh"]), k["c"], nm(k["ch"]), k["x"]))
        if b.stall_hints:
            out.append("")
            out.append("per-chiplet quiet reports from the RTL: " + ", ".join(
                "%s@%.0fus" % (c, t / 1e6) for c, t, _ in b.stall_hints[:20]))
        if self.axi:
            out.append("")
            out.append("real [AXI-STALL] lines (%d, first %d):" % (self.axi_seen, min(12, len(self.axi))))
            for ln in self.axi[:12]:
                out.append("  " + re.sub(r"testharness\.i_dut\.|i_occamy_chip\.|i_occamy\.i_occamy_soc\.", "", ln)[:300])
        if b.anomalies:
            out.append("")
            out.append("queue anomalies:")
            out.extend("  " + a[:240] for a in b.anomalies[:10])
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------
# the process side
# ---------------------------------------------------------------------------------------------------
def my_sims(comm, cwd):
    """PIDs of this user's simulators named comm whose working directory is cwd."""
    out = []
    uid = os.getuid()
    for e in os.listdir("/proc"):
        if not e.isdigit():
            continue
        try:
            if os.stat("/proc/%s" % e).st_uid != uid:
                continue
            with open("/proc/%s/comm" % e) as f:
                if f.read().strip() != comm:
                    continue
            if os.path.realpath(os.readlink("/proc/%s/cwd" % e)) == os.path.realpath(cwd):
                out.append(int(e))
        except (OSError, ValueError):
            continue
    return out


def thread_count(pid):
    try:
        for ln in open("/proc/%d/status" % pid):
            if ln.startswith("Threads:"):
                return int(ln.split()[1])
    except (OSError, ValueError):
        pass
    return None


def quota_headroom_gb(hard_gb):
    try:
        out = subprocess.run(["quota", "-u", os.environ.get("USER", "")], capture_output=True, text=True, timeout=20).stdout
        used_kb = int(out.strip().splitlines()[-1].split()[0].rstrip("*"))
        return hard_gb - used_kb / 1048576.0
    except Exception:
        return None


def dir_gb(path):
    try:
        out = subprocess.run(["du", "-sk", path], capture_output=True, text=True, timeout=120).stdout
        return int(out.split()[0]) / 1048576.0
    except Exception:
        return 0.0


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


class Watch:
    def __init__(self, a):
        self.p = load_params(a.profile, a.set)
        self.run_dir = os.path.abspath(a.run_dir.rstrip("/"))
        self.tag = os.path.basename(self.run_dir)
        self.events = a.events or self.run_dir + ".events"
        self.hang = self.run_dir + ".hang.txt"
        self.status = self.run_dir + ".watchdog.json"
        self.judge = Judge(self.p, Names(self.p["dfg"]))
        self.errors = set()
        self.warned = set()
        self.marker = None
        self.marker_t = time.time()
        self.t0 = time.time()
        self.last_hb = time.time()
        self.last_trace_check = 0.0
        self.trace_gb = 0.0
        self.rate_t = None
        self.slow_reported = 0.0
        self.complete_at = None
        self.pid = None
        self.verdict = None

    def ev(self, msg):
        with open(self.events, "a") as f:
            f.write("%s %s WATCHDOG %s\n" % (time.strftime("%H:%M"), self.tag, msg))

    def once_error(self, where, e):
        k = "%s:%s" % (where, e)
        if k not in self.errors:
            self.errors.add(k)
            self.ev("CHECK-ERROR %s: %r (the watchdog keeps running)" % (where, e))

    def find(self):
        deadline = time.time() + self.p["find_timeout_s"]
        while time.time() < deadline:
            names = sorted(x for x in os.listdir(self.run_dir) if x.startswith("task_")) if os.path.isdir(self.run_dir) else []
            if names:
                self.task_dir = os.path.join(self.run_dir, names[0])
                self.bin = os.path.join(self.task_dir, "bin")
                pids = my_sims(self.p["sim_comm"], self.bin)
                if pids:
                    return pids[0]
            time.sleep(15)
        return None

    def stop(self, kind, detail):
        self.verdict = kind
        self.ev("STOP %s: %s" % (kind, detail))
        with open(self.hang, "a") as f:
            f.write("== %s %s: %s\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), kind, detail, self.judge.evidence()))
        pid = self.pid
        if pid and my_sims(self.p["sim_comm"], self.bin).count(pid):
            os.kill(pid, signal.SIGINT)
            end = time.time() + self.p["stop_grace_s"]
            while time.time() < end and os.path.exists("/proc/%d" % pid) and my_sims(self.p["sim_comm"], self.bin).count(pid):
                time.sleep(2)
            if my_sims(self.p["sim_comm"], self.bin).count(pid):
                self.ev("the simulator did not leave on SIGINT; SIGKILL")
                os.kill(pid, signal.SIGKILL)

    def read_logs(self):
        j = self.judge
        j.feed_bingo(self.t_bingo.read())
        j.feed_probes(self.t_probes.read())
        j.feed_transcript(self.t_live.read())

    def checks(self):
        p, j = self.p, self.judge
        # reads of the logs first: every rule below judges what they say
        self.read_logs()
        now = time.time()
        if p["axi_stall"] in ("warn", "confirm") and j.axi_seen and "axi" not in self.warned:
            self.warned.add("axi")
            self.ev("WARN real [AXI-STALL] line: %s" % j.axi[0][:200])
        if p["anomaly"] in ("warn", "stop") and j.b.anomalies and "anom" not in self.warned:
            self.warned.add("anom")
            self.ev("WARN queue anomaly: %s" % j.b.anomalies[0][:220])
        if j.b.lines == 0 and now - self.t0 > 600 and "nobingo" not in self.warned:
            self.warned.add("nobingo")
            self.ev("WARN no bingo_progress.log after 10 min: this simulator was built without the scheduler "
                    "monitor; only the wall-clock, disk, assertion and AXI checks are active")
        # the clock: the simulation time, or any log growing
        marker = (j.b.sim_ps, sum(t.size() for t in (self.t_bingo, self.t_probes, self.t_live)))
        if marker != self.marker:
            self.marker, self.marker_t = marker, now
        if now - self.marker_t > p["wedge_min"] * 60:
            return "WEDGE", "simulation clock %.1f us and every log unchanged for %s min" % (j.b.sim_ps / 1e6, p["wedge_min"])
        v = j.verdict()
        if v:
            return v
        hr = quota_headroom_gb(p["quota_hard_gb"])
        self.headroom = hr
        if hr is not None and hr < p["min_headroom_gb"]:
            return "QUOTA", "%.0f GB below the quota, limit %d GB" % (hr, p["min_headroom_gb"])
        if self.t_live.size() > p["max_log_gb"] * (1 << 30):
            return "LOG", "sim_live.log over %s GB" % p["max_log_gb"]
        if p["max_trace_gb"] and now - self.last_trace_check > 120:
            self.last_trace_check = now
            self.trace_gb = dir_gb(os.path.join(self.bin, "logs"))
        if p["max_trace_gb"] and self.trace_gb > p["max_trace_gb"]:
            return "LOG", "trace directory %.1f GB over %s GB" % (self.trace_gb, p["max_trace_gb"])
        if p["min_threads"] and now - self.t0 > p["threads_grace_s"]:
            n = thread_count(self.pid)
            if n is not None and n < p["min_threads"]:
                return "THREADS", "%d simulator threads, expected at least %d" % (n, p["min_threads"])
        if j.all_complete():
            self.complete_at = self.complete_at or now
            if now - self.complete_at > p["post_complete_min"] * 60:
                return "HANG-EXIT", "every chiplet drained its task list %s min ago and the simulator is still up" % p["post_complete_min"]
        else:
            self.complete_at = None
        return None

    def rate(self):
        j = self.judge
        sim_us = j.b.sim_ps / 1e6
        if self.rate_t is None and sim_us > 0:
            self.rate_t = (time.time(), sim_us)
        if self.rate_t is None:
            return None
        dt = (time.time() - self.rate_t[0]) / 60.0
        return (sim_us - self.rate_t[1]) / dt if dt > 1 else None

    def write_status(self, state):
        j, b = self.judge, self.judge.b
        t = b.totals()
        atomic_json(self.status, dict(
            tag=self.tag, pid=self.pid, wd_pid=os.getpid(), state=state, verdict=self.verdict, updated=time.time(),
            wall_min=round((time.time() - self.t0) / 60, 1), sim_us=round(b.sim_ps / 1e6, 1),
            rate_us_per_min=self.rate(), busy_chips=len(j.busy()), chips=len(b.chips), complete=j.all_complete(),
            fetched=t["fetched"], n_task=t["n_task"], released=t["rel"], started=t["start"], done=t["done"],
            quiet_us=round(j.quiet_us(), 1), limit_us=round(j.threshold_us(), 1),
            longest_task_us=round(b.longest_task_ps / 1e6, 1), axi_stalls=j.axi_seen, anomalies=len(b.anomalies),
            headroom_gb=getattr(self, "headroom", None), interval_s=self.p["interval_s"]))

    def other_watchdog(self):
        """The PID of another live watchdog of this run, if there is one (two would both stop the simulator)."""
        try:
            with open(self.status) as f:
                s = json.load(f)
            pid = s.get("wd_pid")
            if pid and pid != os.getpid() and s.get("state") == "running":
                with open("/proc/%d/cmdline" % pid) as c:
                    if "sim_watchdog" in c.read():
                        return pid
        except (OSError, ValueError):
            pass
        return None

    def run(self):
        p = self.p
        os.makedirs(os.path.dirname(self.events) or ".", exist_ok=True)
        other = self.other_watchdog()
        if other:
            print("a watchdog (pid %d) already watches %s" % (other, self.tag))
            return 1
        self.ev("ARMED profile params: " + " ".join("%s=%s" % (k, p[k]) for k in sorted(p) if p[k] != DEFAULTS[k]))
        self.pid = self.find()
        if not self.pid:
            self.ev("no simulator found for %s" % self.tag)
            self.write_status("no-simulator")
            return 1
        self.t0 = self.marker_t = time.time()
        self.t_bingo = Tail(os.path.join(self.bin, "bingo_progress.log"))
        self.t_probes = Tail(os.path.join(self.bin, "snax_probes.log"))
        self.t_live = Tail(os.path.join(self.bin, "sim_live.log"))
        self.ev("armed on simulator %d (%s)" % (self.pid, self.bin))
        while my_sims(p["sim_comm"], self.bin).count(self.pid):
            try:
                v = self.checks()
            except Exception as e:  # a failing check must never take the watchdog down
                self.once_error("checks", e)
                v = None
            try:
                self.write_status("running")
            except Exception as e:
                self.once_error("status", e)
            if v:
                self.stop(*v)
                break
            if time.time() - self.last_hb > p["heartbeat_min"] * 60:
                self.last_hb = time.time()
                r = self.rate()
                self.ev("HEARTBEAT %s, %s us/min%s" % (self.judge.summary(), "%.1f" % r if r else "?",
                                                       ", headroom %.0f GB" % self.headroom if getattr(self, "headroom", None) is not None else ""))
            r = self.rate()
            if (p["min_rate_us_per_min"] and r is not None and (time.time() - self.t0) > p["rate_grace_min"] * 60
                    and r < p["min_rate_us_per_min"] and time.time() - self.slow_reported > 1800):
                self.slow_reported = time.time()
                self.ev("SLOW %.2f simulated us per wall minute, expected at least %s" % (r, p["min_rate_us_per_min"]))
            time.sleep(p["interval_s"])
        try:
            self.read_logs()
        except Exception as e:
            self.once_error("final read", e)
        self.ev("simulator %d gone%s: %s" % (self.pid, " (stopped: %s)" % self.verdict if self.verdict else "", self.judge.summary()))
        if p["on_stop_cmd"]:
            cmd = p["on_stop_cmd"].format(task_dir=self.task_dir, tag=self.tag, sweep=SWEEP)
            with open(self.run_dir + ".after.log", "a") as out:
                rc = subprocess.run(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT).returncode
            self.ev("after-stop command finished rc=%d" % rc)
        self.write_status("exited")
        return 0


# ---------------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------------
def cmd_status(a):
    target = a.target
    base = target[:-len(".events")] if target.endswith(".events") else target
    if not os.path.exists(base + ".watchdog.json") and os.path.exists(os.path.join(SWEEP, a.exp, "runs", target + ".watchdog.json")):
        base = os.path.join(SWEEP, a.exp, "runs", target)
    try:
        with open(base + ".watchdog.json") as f:
            s = json.load(f)
    except (OSError, ValueError):
        print("no watchdog status for %s" % base)
        return 2
    age = time.time() - s["updated"]
    alive = s["pid"] and os.path.exists("/proc/%d" % s["pid"])
    stale = s["state"] == "running" and alive and age > 3 * max(s["interval_s"], 30) + 60
    flag = ""
    if s["state"] == "running" and not alive:
        flag = "  [watchdog says running but the simulator is gone]"
    if stale:
        flag = "  [STALE: status not updated for %d s, the watchdog is dead or stuck]" % age
    rate = s["rate_us_per_min"]
    print("%s %s: sim %.1f us (%s us/min), busy %d/%d chips, fetched %d/%d released %d started %d done %d, "
          "quiet %.1f us of %.1f, longest task %.1f us, axi stalls %d%s%s"
          % (s["tag"], s["state"], s["sim_us"], "%.1f" % rate if rate else "?", s["busy_chips"], s["chips"],
             s["fetched"], s["n_task"], s["released"], s["started"], s["done"], s["quiet_us"], s["limit_us"],
             s["longest_task_us"], s["axi_stalls"], "  STOPPED: " + s["verdict"] if s["verdict"] else "", flag))
    return 3 if stale else 0


def cmd_analyse(a):
    p = load_params(a.profile, a.set)
    j = Judge(p, Names(p["dfg"]))
    first = None
    probes = []
    if a.probes and os.path.exists(a.probes):
        with open(a.probes, errors="replace") as f:
            probes = f.readlines()
    j.feed_probes(probes)
    last_report = -1.0
    with open(a.log, errors="replace") as f:
        for ln in f:
            j.feed_bingo([ln])
            if ln.startswith("[BINGO PROG]"):
                v = j.verdict()
                if v and not first:
                    first = v + (j.b.sim_ps / 1e6,)
                    print("FIRST VERDICT at %.1f us: %s: %s" % (first[2], v[0], v[1]))
                if j.b.sim_ps / 1e6 - last_report >= a.every:
                    last_report = j.b.sim_ps / 1e6
                    print(j.summary())
    print("END: " + j.summary())
    print("complete: %s; anomalies: %d; real AXI stalls: %d; longest task %.1f us"
          % (j.all_complete(), len(j.b.anomalies), j.axi_seen, j.b.longest_task_ps / 1e6))
    if a.evidence:
        print(j.evidence())
    return 0 if not first else 1


def cmd_params(a):
    p = load_params(a.profile, a.set)
    w = max(len(n) for n, _, _ in PARAMS)
    for n, d, h in PARAMS:
        mark = "*" if p[n] != d else " "
        print("%s %-*s = %-16s %s" % (mark, w, n, p[n], h))
    print("(* = differs from the default)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--run-dir", required=True, help="the run directory (<exp>/runs/<tag>), which holds task_*/bin")
    w.add_argument("--events", help="events file (default <run dir>.events)")
    w.add_argument("--profile", default="default")
    w.add_argument("--set", action="append", default=[], metavar="key=value")
    s = sub.add_parser("status")
    s.add_argument("target", help="run dir, events file or tag")
    s.add_argument("--exp", default="ffn")
    an = sub.add_parser("analyse")
    an.add_argument("--log", required=True)
    an.add_argument("--probes")
    an.add_argument("--profile", default="default")
    an.add_argument("--set", action="append", default=[], metavar="key=value")
    an.add_argument("--every", type=float, default=100.0, help="print a summary every this many simulated us")
    an.add_argument("--evidence", action="store_true")
    pa = sub.add_parser("params")
    pa.add_argument("--profile", default="default")
    pa.add_argument("--set", action="append", default=[], metavar="key=value")
    a = ap.parse_args()
    return {"watch": lambda x: Watch(x).run(), "status": cmd_status, "analyse": cmd_analyse, "params": cmd_params}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
