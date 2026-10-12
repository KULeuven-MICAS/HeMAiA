#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""hsweep: the shared driver behind every sweep's one-button run.sh.

    hsweep.py build  --cfg <rtl cfg> [--tree <repo>]
    hsweep.py run    --exp <name> --tag <run> --points <points.yaml> [--tree <repo>] [--jobs N]
                     [--fgp [N]]
    hsweep.py verdict <run dir>
    hsweep.py prune   <run dir>

WHY NOT run_sweep_cli. The stock runner takes one hardware cfg per invocation, keeps one simv
per clone, records the cfg the simv was built for but never checks it (so --sw-only on a
different cfg runs mismatched hardware silently), and exits 0 when tasks fail. This keeps the
runner's build and simulation steps and adds what a sweep needs around them:

  build    RTL and bootrom for the cfg, then the simulation compile, then a stamp next to the
           simv holding the cfg's content hash and the snax commit. Builds are serialised by
           a lock on the tree, and every Chisel generation in the shared snax tree by a lock
           there too.
  run      refuses to start unless the standing simv's stamp matches the points' cfg; builds
           the apps under the tree lock (the runner clears and refills target/sim/bin, which
           two builds at once would corrupt); starts its simulations only while this
           session's running simulations plus its own stay within SIM_SLOTS (default 3).
  verdict  PASS needs the end-of-computation marker, no failed check on any UART, no live
           watchdog and no timeout; the runner's own verdict is not trusted alone.
  prune    deletes the traces once a run has been parsed.

A points file is YAML:

    cfg: target/rtl/cfg/hemaia_16chiplet_split.hjson
    fgp_threads: 8          # optional: VCS FGP worker threads a simulation (N + 1 cores)
    points:
      - {name: reduce_p16, host_app_type: offload_bingo_hw, chip_type: multi_chip,
         workload: collective/reduce, dev_app: snax-bingo-offload}
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

MAIN_TREE = Path("/users/micas/fkong/no_backup/HeMAiA")
SNAX_TREE = Path("/users/micas/fkong/no_backup/snax_cluster")
SNAX_LOCK = SNAX_TREE / ".hsweep_chisel.lock"
STAMP = "target/sim/bin/.hsweep_stamp.json"
SIM_SLOTS = int(os.environ.get("SIM_SLOTS", "3"))
SLOT_LOCK = Path(__file__).resolve().parent / ".sim_slots.lock"
# simulators of this session: their binaries sit in task dirs under these roots
MINE = ("/target/sim/automation/sweep/",
        "/085aec8d-86ae-4bc5-9d98-708e5da2403c/scratchpad/")


@contextmanager
def flocked(path: Path, shared: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    # read-write: on NFS flock is emulated with byte-range locks, and a shared (read) lock on a
    # descriptor not open for reading fails with EBADF
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _runner_module(tree: Path):
    sys.path.insert(0, str(tree / "util" / "automation_scripts"))
    import hemaia_sim_runner as hsr  # noqa: E402
    return hsr


def cfg_sha(tree: Path, cfg: str) -> str:
    return hashlib.sha256((tree / cfg).read_bytes()).hexdigest()[:16]


def snax_head(tree: Path) -> str:
    # the snax the tree builds: its bender checkout (deps/snitch_cluster), or the shared clone
    # when Bender.local overrides snitch_cluster to it
    snax = tree / "deps" / "snitch_cluster"
    snax = snax.resolve() if snax.exists() else SNAX_TREE
    r = subprocess.run(["git", "-C", str(snax), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(snax), "status", "--porcelain", "-uno"],
                           capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ("+dirty" if dirty else "")


def _make_runner(hsr, tree: Path, cfg: str, out: Path, *, build: bool, jobs: int,
                 memchip: bool, timeout_h: float, fgp_threads: int = 0, assert_maxfail: int = 1):
    return hsr.HeMAiASimRunner(
        repo_root=tree, output_dir=out, engine="vcs", with_waveform=False, cfg=cfg,
        sim_cfg="target/sim/cfg/sim_rtl.hjson", with_macro=memchip, with_d2d=memchip,
        with_pll=False, build_jobs=os.cpu_count(), max_jobs=jobs, skip_setup=True,
        skip_build=not build, skip_compile=not build, build_sw_fleet=False,
        fail_on_task_failure=False, timeout_seconds=int(timeout_h * 3600),
        fgp_threads=fgp_threads, assert_maxfail=assert_maxfail)


def cmd_build(a):
    tree = Path(a.tree)
    hsr = _runner_module(tree)
    out = tree / "target/sim/automation/sweep/common/build_out"
    r = _make_runner(hsr, tree, a.cfg, out, build=True, jobs=1, memchip=True, timeout_h=4)
    t0 = time.time()
    with flocked(tree / ".hsweep_build.lock"):
        r.effective_cfg = hsr.resolve_rtl_cfg(r.repo_root, r.cfg, with_pll=False)
        print(f"[hsweep build] {a.cfg} -> {r.effective_cfg}", flush=True)
        # RTL first. Its Chisel generation writes into the shared snax tree, and the VCS
        # compile reads what it wrote, so the snax lock is held across both: another tree's
        # build regenerating those files mid-compile would hand this one a torn copy
        with flocked(SNAX_LOCK):
            r.build_hw_sw()
            print(f"[hsweep build] rtl+bootrom done after {time.time() - t0:.0f} s", flush=True)
            r.prepare_simulation_inputs(["vcs"])
            r.compile_simulation()
        stamp = {"cfg": a.cfg, "cfg_sha": cfg_sha(tree, a.cfg), "snax": snax_head(tree),
                 "built": time.strftime("%Y-%m-%d %H:%M:%S")}
        (tree / STAMP).write_text(json.dumps(stamp, indent=2) + "\n")
    print(f"[hsweep build] DONE in {time.time() - t0:.0f} s: {stamp}", flush=True)
    print("RUN EXIT 0", flush=True)


def _my_sims() -> int:
    ps = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "args="],
                        capture_output=True, text=True).stdout
    n = 0
    for ln in ps.splitlines():
        exe = ln.split(" ", 1)[0]
        if exe.endswith("occamy_chip.vcs") and any(m in exe for m in MINE):
            n += 1
    return n


def cmd_run(a):
    tree = Path(a.tree)
    spec = yaml.safe_load(Path(a.points).read_text())
    cfg, points = spec["cfg"], spec["points"]
    stamp_p = tree / STAMP
    if not stamp_p.exists():
        sys.exit(f"[hsweep run] no simv stamp in {tree}: run `hsweep.py build --cfg {cfg}`")
    stamp = json.loads(stamp_p.read_text())
    if stamp.get("cfg_sha") != cfg_sha(tree, cfg) or stamp.get("cfg") != cfg:
        sys.exit(f"[hsweep run] the standing simv was built for {stamp.get('cfg')} "
                 f"({stamp.get('cfg_sha')}), these points need {cfg} "
                 f"({cfg_sha(tree, cfg)}): rebuild first")
    hsr = _runner_module(tree)
    out = Path(a.out) if a.out else tree / "target/sim/automation/sweep" / a.exp / "runs" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    memchip = bool(spec.get("memchip", True))
    # FGP worker threads a simulation: the --fgp flag, else the points file's fgp_threads
    fgp = a.fgp_threads if a.fgp_threads is not None else int(spec.get("fgp_threads", 0))
    r = _make_runner(hsr, tree, cfg, out, build=False, jobs=a.jobs, memchip=memchip,
                     timeout_h=float(spec.get("timeout_h", 3)), fgp_threads=fgp,
                     assert_maxfail=a.assert_maxfail)
    tasks = []
    for p in points:
        t = {"host_app_type": p["host_app_type"], "chip_type": p.get("chip_type", ""),
             "workload": p.get("workload", ""), "dev_app": p.get("dev_app", "")}
        # the runner names each task directory task_<idx>_<ci_name>
        t["ci_name"] = (hsr.make_task(**t)["ci_name"] if hasattr(hsr, "make_task")
                        else "_".join(v for v in t.values() if v))
        tasks.append(t)
    (out / "points.json").write_text(json.dumps(points, indent=2) + "\n")

    build = r.build_apps_and_stage

    def locked_build(t):
        # the tree lock: the runner clears and refills target/sim/bin. The snax lock, SHARED:
        # the apps only read the shared snax tree's generated headers and runtime, so app
        # builds in different trees may overlap; what writes there (sbt, an RTL build) takes
        # it exclusively and waits for every reader
        with flocked(tree / ".hsweep_build.lock"), flocked(SNAX_LOCK, shared=True):
            info = build(t)
            # the generator's own tables (*.csv next to the workload: phase labels, point
            # metadata) belong to THIS build: a later regeneration must not relabel this run
            apps = tree / "target/sw/host/apps"
            for (task_dir, _), p in zip(info, points):
                wl = apps / p["host_app_type"] / p.get("chip_type", "") / "workloads" / p.get("workload", "")
                snap = Path(task_dir) / "workload_tables"
                snap.mkdir(parents=True, exist_ok=True)
                for csv_f in wl.glob("*.csv"):
                    shutil.copy2(csv_f, snap / csv_f.name)
            return info
    r.build_apps_and_stage = locked_build

    sims = r.run_simulations

    def gated_sims(*args, **kw):
        # Taking a slot is atomic across runners: the check and the start happen under one
        # lock, held until this run's first simulation process exists, so two runners that
        # wait together cannot both see the same free slot.
        out = []
        with flocked(SLOT_LOCK):
            while True:
                n = _my_sims()
                if n + a.jobs <= SIM_SLOTS:
                    break
                print(f"[hsweep run] {n} of this session's sims running; waiting for {a.jobs} "
                      f"slot(s) of {SIM_SLOTS}", flush=True)
                time.sleep(30)
            print(f"[hsweep run] SIM START ({len(tasks)} task(s))", flush=True)
            th = threading.Thread(target=lambda: out.append(sims(*args, **kw)), daemon=True)
            th.start()
            t0 = time.time()
            while th.is_alive() and _my_sims() < n + 1 and time.time() - t0 < 600:
                time.sleep(2)
        th.join()
        return out[0] if out else None
    r.run_simulations = gated_sims
    r.run(tasks)
    v = verdict(out)
    print(f"[hsweep run] verdicts: {json.dumps({k: x['verdict'] for k, x in v.items()})}",
          flush=True)
    print("RUN EXIT 0", flush=True)


CHECK_RE = re.compile(r"Check \[([^\]]+)\]: (PASS|FAIL)")
SUM_RE = re.compile(r"Checks: (\d+) PASS, (\d+) FAIL")


def verdict(run: Path) -> dict:
    pts = json.loads((run / "points.json").read_text()) if (run / "points.json").exists() else []
    res = {}
    for d in sorted(run.glob("task_*")):
        idx = int(d.name.split("_")[1])
        name = pts[idx]["name"] if idx < len(pts) else d.name
        b = d / "bin"
        uart = "".join(f.read_text(errors="replace") for f in sorted(b.glob("uart_chip_*.log")))
        simlog = (b / "sim_run.log").read_text(errors="replace") if (b / "sim_run.log").exists() else ""
        probes = (b / "snax_probes.log").read_text(errors="replace") if (b / "snax_probes.log").exists() else ""
        fails = [m.group(1) for m in CHECK_RE.finditer(uart) if m.group(2) == "FAIL"]
        passes = [m.group(1) for m in CHECK_RE.finditer(uart) if m.group(2) == "PASS"]
        sp = sf = 0
        for m in SUM_RE.finditer(uart):
            sp += int(m.group(1)); sf += int(m.group(2))
        eoc = "All chips finished successfully" in simlog
        bad = "finished with errors" in simlog
        live = "[LIVE WATCHDOG]" in probes
        tmo = "TIMEOUT" in simlog
        ok = eoc and not bad and not fails and not sf and not live and not tmo
        res[name] = {"task": d.name, "verdict": "PASS" if ok else "FAIL", "eoc": eoc,
                     "checks_pass": len(passes) + sp, "checks_fail": len(fails) + sf,
                     "failed": fails[:20], "live_watchdog": live, "timeout": tmo,
                     "errors": bad}
    (run / "verdict.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def cmd_verdict(a):
    v = verdict(Path(a.run))
    for k, x in v.items():
        print(f"{x['verdict']:4s} {k}: checks {x['checks_pass']} pass / {x['checks_fail']} fail"
              f"{' LIVE-WATCHDOG' if x['live_watchdog'] else ''}{' TIMEOUT' if x['timeout'] else ''}"
              f"{'' if x['eoc'] else ' NO-EOC'}")


def cmd_prune(a):
    run = Path(a.run)
    if not (run / "verdict.json").exists():
        sys.exit("[hsweep prune] no verdict.json: run `hsweep.py verdict` (and your parser) first")
    freed = 0
    for d in run.glob("task_*/bin"):
        for p in list(d.glob("logs/*")) + list(d.glob("*.daidir")) + list(d.glob("occamy_chip.vcs")):
            if p.name.startswith("trace_chip_00_hart_00000") and p.suffix == ".log":
                continue                 # the host trace: small, holds the sim-time stamps
            try:
                freed += p.stat().st_size if p.is_file() else 0
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except OSError:
                pass
    print(f"[hsweep prune] freed ~{freed / 2**30:.1f} GiB in {run}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--cfg", required=True)
    b.add_argument("--tree", default=str(MAIN_TREE)); b.set_defaults(fn=cmd_build)
    r = sub.add_parser("run"); r.add_argument("--exp", required=True)
    r.add_argument("--tag", required=True); r.add_argument("--points", required=True)
    r.add_argument("--tree", default=str(MAIN_TREE)); r.add_argument("--out", default=None)
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--fgp", dest="fgp_threads", type=int, nargs="?", const=8, default=None,
                   metavar="N", help="VCS Fine-Grained Parallelism, N worker threads a "
                   "simulation (a bare --fgp means 8); overrides the points file's fgp_threads")
    r.add_argument("--assert-maxfail", dest="assert_maxfail", type=int, default=1, metavar="N",
                   help="stop a simulation once its SystemVerilog assertions have failed N times "
                   "in all; 0 never stops (default: %(default)s)")
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("verdict"); v.add_argument("run"); v.set_defaults(fn=cmd_verdict)
    p = sub.add_parser("prune"); p.add_argument("run"); p.set_defaults(fn=cmd_prune)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
