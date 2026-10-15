#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""clean_stale.py [--dry-run] [--age-h H] [--repo DIR] [--under DIR ...]: delete what the simulation flows leave
behind that no flow builds from or reads back. `make clean` runs it (`make clean-stale` alone).

What goes, once idle (see below):
  pycache     __pycache__ under target/sim/automation and the workloads (any age)
  old runs    the run directories of the runner-driven flows: sweep/<group>/<exp>/task_<n>,
              .../prev_*task*, .../wave_run, ci/.../task_<n>, test/task_<n>
  summaries   the runner's simulation_summary_*.md next to them
  run bulk    in every run of the hsweep experiments (sweep/<group>/<exp>/runs/<tag>/task_*/bin): the core
              traces, after their task spans are taken into tasks.csv (sweep/llm_blocks/ffn/trim_traces.py;
              a run whose spans cannot be taken keeps its traces), and the staged program images
              and simulator copies (app_chip_*, hbm, mempool, *.daidir, occamy_chip.vcs, simv*)
  sim copies  simulators copied into an experiment directory (sweep/<group>/<exp>/simv*)
  build logs  sweep/common/builds: every build's log and events but the newest of each cfg
  wl builds   the workloads' generated build/ directories (HBM and memory-chip images; their
              Makefiles regenerate them)
What stays: tracked files, results/, figs/, cache/, weights/, and every run's UART, events, logs,
progress logs and tasks.csv.

A path counts as idle when nothing in it has changed for --age-h hours (default 24): a running
simulation writes its logs all the time, so its directory never is -- also inside the container
`make clean` runs in, where the host's processes are not visible. On the host, a directory some
process works in is skipped too. A run directory holding a file named KEEP is never touched.
--under limits every rule to paths under the directories given (one experiment's, say).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

STAGED = ("app_chip_*", "hbm", "mempool", "*.daidir", "occamy_chip.vcs", "simv*")
HOST_TRACE = "trace_chip_00_hart_00000"     # the host trace: small, holds the sim-time stamps


def newest(path):
    """The newest modification time of `path` and everything under it (symlinks not followed)."""
    try:
        t = path.lstat().st_mtime
    except OSError:
        return 0.0
    if path.is_dir() and not path.is_symlink():
        for root, dirs, files in os.walk(path):
            for n in dirs + files:
                try:
                    t = max(t, os.lstat(os.path.join(root, n)).st_mtime)
                except OSError:
                    pass
    return t


def size(path):
    if path.is_symlink() or path.is_file():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                n += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return n


def busy_dirs():
    """The working directories of the processes this side can see (empty in a container)."""
    out = set()
    for pid in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if pid.isdigit():
            try:
                out.add(os.path.realpath(f"/proc/{pid}/cwd"))
            except OSError:
                pass
    return out


class Cleaner:
    def __init__(self, repo, age_h, dry, under=()):
        self.repo, self.dry = Path(repo), dry
        self.under = [os.path.realpath(u) for u in under]
        self.auto = self.repo / "target/sim/automation"
        self.now, self.age_s = time.time(), age_h * 3600
        self.busy = busy_dirs()
        self.freed, self.count, self.notes = {}, {}, []

    def idle(self, path):
        if path.is_dir():
            real = os.path.realpath(path)
            if any(b == real or b.startswith(real + os.sep) for b in self.busy):
                return False
        return self.now - newest(path) >= self.age_s

    def inside(self, path):
        """Whether --under (if given) covers path."""
        if not self.under:
            return True
        real = os.path.realpath(path)
        return any(real == u or real.startswith(u + os.sep) for u in self.under)

    def rm(self, rule, path, nbytes=None):
        if not self.inside(path):
            return
        n = size(path) if nbytes is None else nbytes
        self.freed[rule] = self.freed.get(rule, 0) + n
        self.count[rule] = self.count.get(rule, 0) + 1
        if self.dry:
            print(f"  would remove [{rule}] {path.relative_to(self.repo)} ({n / 2**20:.1f} MiB)")
            return
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as e:
            self.notes.append(f"{path}: {e}")

    def pycache(self):
        roots = [self.auto] + list(self.repo.glob("target/sw/host/apps/*/*/workloads"))
        for r in roots:
            for p in r.rglob("__pycache__"):
                if p.is_dir():
                    self.rm("pycache", p)

    def old_runs(self):
        for pat in ("sweep/*/*/task_[0-9]*", "sweep/*/*/prev_*task*", "sweep/*/*/wave_run",
                    "ci/*/*/task_[0-9]*", "ci/*/task_[0-9]*", "test/task_[0-9]*"):
            for p in self.auto.glob(pat):
                if p.is_dir() and self.idle(p):
                    self.rm("old runs", p)
        for pat in ("sweep/*/*/simulation_summary_*.md", "ci/*/*/simulation_summary_*.md",
                    "ci/*/simulation_summary_*.md", "test/simulation_summary_*.md"):
            for p in self.auto.glob(pat):
                if self.idle(p):
                    self.rm("summaries", p)

    def run_bulk(self):
        trim = self.auto / "sweep/llm_blocks/ffn/trim_traces.py"
        for run in sorted(p for p in self.auto.glob("sweep/*/*/runs/*") if p.is_dir()):
            # trim_traces.py truncates traces itself, so a run outside --under is not even read
            if not self.inside(run) or (run / "KEEP").exists() or not self.idle(run):
                continue
            for task in sorted(run.glob("task_*")):
                b = task / "bin"
                logs = [p for p in b.glob("logs/*") if p.is_file() and not p.name.startswith(HOST_TRACE)]
                held = {p: p.stat().st_size for p in logs}
                if any(held.values()):
                    ok = (task / ".tasks_extracted").exists()
                    if not ok and not self.dry:
                        r = subprocess.run([sys.executable, str(trim), str(task), "--force"],
                                           capture_output=True, text=True)
                        ok = r.returncode == 0 and (task / ".tasks_extracted").exists()
                        if not ok:
                            self.notes.append(f"{task.relative_to(self.repo)}: task spans not taken, "
                                              f"traces kept ({(r.stderr or r.stdout).strip()[-160:]})")
                    if ok or self.dry:
                        # a dry run cannot tell yet whether the spans of an unparsed run can be taken
                        rule = "run traces" if ok else "run traces, if their spans can be taken"
                        for p in logs:
                            self.rm(rule, p, held[p])
                for pat in STAGED:
                    for p in b.glob(pat):
                        self.rm("run images", p)

    def sim_copies(self):
        for p in self.auto.glob("sweep/*/*/simv*"):
            if self.idle(p):
                self.rm("sim copies", p)

    def build_logs(self):
        d = self.auto / "sweep/common/builds"
        newest_of = {}
        for p in d.glob("*.log"):
            cfg = re.sub(r"_\d{4}_\d{4}[a-z]?$", "", p.stem)
            if cfg not in newest_of or p.stat().st_mtime > newest_of[cfg].stat().st_mtime:
                newest_of[cfg] = p
        keep = {p.stem for p in newest_of.values()}
        for p in sorted(d.glob("*.log")) + sorted(d.glob("*.events")):
            if p.stem not in keep and self.idle(p):
                self.rm("build logs", p)

    def wl_builds(self):
        for p in self.repo.glob("target/sw/host/apps/*/*/workloads/**/build"):
            if p.is_dir() and ((p / "hbm").exists() or any(p.glob("mempool*"))) and self.idle(p):
                self.rm("wl builds", p)

    def run(self):
        for step in (self.pycache, self.old_runs, self.run_bulk, self.sim_copies, self.build_logs,
                     self.wl_builds):
            step()
        verb = "would free" if self.dry else "freed"
        total = 0
        for rule, n in self.freed.items():
            total += n
            print(f"[clean_stale] {rule:11s} {self.count[rule]:5d} paths, {verb} {n / 2**30:6.2f} GiB")
        print(f"[clean_stale] {verb} {total / 2**30:.2f} GiB in all (idle >= {self.age_s / 3600:g} h)")
        for n in self.notes:
            print(f"[clean_stale] note: {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="list what would go, delete nothing")
    ap.add_argument("--age-h", type=float, default=24.0,
                    help="hours a path must have been left unchanged (default: %(default)s)")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--under", nargs="+", default=[], metavar="DIR",
                    help="only paths under these directories")
    a = ap.parse_args()
    Cleaner(a.repo, a.age_h, a.dry_run, a.under).run()


if __name__ == "__main__":
    main()
