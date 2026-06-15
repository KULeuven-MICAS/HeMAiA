#!/usr/bin/env python3
"""
HeMAiA CI automation using Questasim
====================================

Orchestrates the full build-and-simulate flow for the HeMAiA project with
Questasim.  Some steps run on the host (Questasim, proprietary downloads)
while hardware generation runs inside a Docker/Podman container.

Flow overview:
  Step 0 - Parse task definitions from task_vsim.yaml
  Step 1 - Reset private repos and generated files
  Step 2 - Initialise private repos (host-side, needs credentials)
  Step 3 - Compile HW and SW inside the container
  Step 4 - Build application hex files and create per-task directories
  Step 5 - Prepare Questasim testharness (container) and compile (host)
  Step 6 - Run simulations concurrently
  Step 7 - Write Markdown summary
"""

from __future__ import annotations

import argparse
import atexit
import getpass
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

CI_CFG = "target/rtl/cfg/hemaia_multichip_ci.hjson"
SIM_CFG = "target/sim/cfg/sim_rtl.hjson"
SIM_TIMEOUT_SECONDS = 2 * 60 * 60  # 2 hours per simulation thread
# Keep in sync with the JOBS default in target/sim/automation/Makefile.
DEFAULT_MAX_SIM_JOBS = 16

# ---------------------------------------------------------------------------
# Process-group tracking -- guarantees no stale vsim threads survive
# ---------------------------------------------------------------------------
#
# The ``occamy_chip.vsim`` launcher spawns ``vish`` -> ``vsimk`` as descendant
# processes.  Killing only the immediate child (as ``subprocess.run`` does on
# timeout) leaves those descendants orphaned onto ``init`` where they keep
# pinning a CPU core forever.  We therefore start each simulation in its own
# session/process group and tear the *whole group* down on timeout, crash, or
# interrupt, so the script never leaves stale threads behind.

_active_pgids: Set[int] = set()
_active_pgids_lock = threading.Lock()


def _register_pgid(pgid: int) -> None:
    with _active_pgids_lock:
        _active_pgids.add(pgid)


def _unregister_pgid(pgid: int) -> None:
    with _active_pgids_lock:
        _active_pgids.discard(pgid)


def _kill_pgid(pgid: int) -> None:
    """SIGKILL an entire process group, ignoring already-dead groups."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def _terminate_all_sims() -> None:
    """Kill every still-running simulation process group.

    Registered with ``atexit`` and the SIGINT/SIGTERM handlers so a crash,
    timeout, or Ctrl-C tears down the full vsim/vish/vsimk process tree instead
    of orphaning it onto init.
    """
    with _active_pgids_lock:
        pgids = list(_active_pgids)
        _active_pgids.clear()
    for pgid in pgids:
        _kill_pgid(pgid)


def _install_cleanup_handlers() -> None:
    """Ensure simulation process groups are reaped on exit or signal."""
    atexit.register(_terminate_all_sims)

    def _handler(signum, _frame):
        _terminate_all_sims()
        # Restore default disposition and re-raise so the exit status is correct.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line options for the CI simulation flow."""
    parser = argparse.ArgumentParser(
        description="Build and run HeMAiA Questasim CI tasks.",
    )
    parser.add_argument(
        "-j", "--max-sim-jobs",
        type=int,
        default=DEFAULT_MAX_SIM_JOBS,
        help="maximum number of simulations to run in parallel "
             f"(default: {DEFAULT_MAX_SIM_JOBS})",
    )
    args = parser.parse_args()
    if args.max_sim_jobs < 1:
        parser.error("--max-sim-jobs must be >= 1")
    return args


def ensure_vsim_available() -> None:
    """Abort early if ``vsim`` is not on PATH (EDA env not sourced)."""
    if shutil.which("vsim") is None:
        print(
            "ERROR: 'vsim' not found on PATH.\n"
            "Please source the EDA setup script"
            "and re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)


def run_host_script(script_path: Path, script_args: List[str] = None) -> None:
    """Execute a shell script on the host via ``bash``."""
    if not script_path.exists():
        raise FileNotFoundError(f"Host script {script_path} does not exist")
    subprocess.run(
        ["bash", str(script_path)] + (script_args or []),
        check=True,
        cwd=script_path.parent,
    )


def run_in_container(
    repo_root: Path,
    docker_image: str,
    working_dir: Path,
    command: List[str],
) -> None:
    """Run *command* inside *docker_image*, mounting *repo_root* at the same path."""
    docker_cmd: List[str] = [
        "podman", "run", "--rm",
        "-v", f"{repo_root}:{repo_root}",
        "-w", str(working_dir),
        docker_image,
    ]
    docker_cmd.extend(command)
    subprocess.run(docker_cmd, check=True)


def _copy_path(src: Path, dst: Path) -> None:
    """Copy a file or directory from *src* to *dst*, replacing any existing target."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Minimal YAML loader (no PyYAML dependency required)
# ---------------------------------------------------------------------------

def _simple_task_yaml_loader(task_yaml: Path) -> Dict:
    """Parse the limited ``task_vsim.yaml`` format without PyYAML.

    Supports a top-level ``runs:`` key containing a sequence of dict entries,
    each with one or more ``KEY: VALUE`` attribute lines (the first key sits
    on the same line as the ``- `` dash).
    """
    def _parse_kv(text: str):
        k, v = [s.strip() for s in text.split(":", 1)]
        if (v.startswith('"') and v.endswith('"')) or (
            v.startswith("'") and v.endswith("'")
        ):
            v = v[1:-1]
        return k, v

    runs: list = []
    with task_yaml.open("r") as f:
        lines = f.readlines()

    # Advance to the 'runs:' section
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("runs:"):
            i += 1
            break
        i += 1

    current: Dict[str, str] = None
    item_indent: int = None

    while i < len(lines):
        raw = lines[i]
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        indent = len(raw) - len(stripped)

        if stripped.startswith("- "):
            if current is not None:
                runs.append(current)
                current = None
            after = stripped[2:].strip()
            if ":" not in after:
                break
            k, v = _parse_kv(after)
            current = {k: v}
            item_indent = indent + 2
        else:
            if current is None or item_indent is None:
                break
            if indent < item_indent:
                runs.append(current)
                current = None
                break
            if ":" in stripped:
                k, v = _parse_kv(stripped)
                current[k] = v
        i += 1

    if current is not None:
        runs.append(current)

    return {"runs": runs}


# ---------------------------------------------------------------------------
# Step 0 -- Parse task definitions
# ---------------------------------------------------------------------------

def _derive_ci_name(host_app_type: str, chip_type: str, workload: str, dev_app: str) -> str:
    """Derive a human-readable, unique task name from its attributes."""
    parts = [host_app_type, chip_type]
    if workload and workload != "None":
        parts.append(workload)
    if dev_app and dev_app != "None":
        parts.append(dev_app)
    return "_".join(p for p in parts if p)


def step0_parse_tasks(task_yaml: Path) -> List[Dict[str, str]]:
    """Parse CI tasks from *task_yaml*.

    Expected YAML structure::

        runs:
          - HOST_APP_TYPE: offload_bingo_sw
            CHIP_TYPE: single_chip
            WORKLOAD: gemm_tiled_1cluster
            DEV_APP: snax-bingo-offload
    """
    if yaml is not None:
        with open(task_yaml, "r") as f:
            param_data = yaml.safe_load(f)
    else:
        param_data = _simple_task_yaml_loader(task_yaml)

    tasks: List[Dict[str, str]] = []
    for attr_dict in param_data.get("runs", []):
        if "HOST_APP_TYPE" not in attr_dict:
            raise ValueError(f"Task is missing 'HOST_APP_TYPE': {attr_dict}")

        host_app_type = str(attr_dict.get("HOST_APP_TYPE", ""))
        chip_type     = str(attr_dict.get("CHIP_TYPE", ""))
        workload      = str(attr_dict.get("WORKLOAD", ""))
        dev_app       = str(attr_dict.get("DEV_APP", ""))

        tasks.append({
            "host_app_type": host_app_type,
            "chip_type":     chip_type,
            "workload":      workload,
            "dev_app":       dev_app,
            "ci_name":       _derive_ci_name(host_app_type, chip_type, workload, dev_app),
        })
    return tasks


# ---------------------------------------------------------------------------
# Step 1 -- Reset environment
# ---------------------------------------------------------------------------

def step1_clean(repo_root: Path, docker_image: str) -> None:
    """Run make clean to restore a clean state before the flow starts."""
    run_in_container(repo_root, docker_image, repo_root, ["make", "clean"])


# ---------------------------------------------------------------------------
# Step 2 -- Initialise private repos (host-side)
# ---------------------------------------------------------------------------

def step2_init_private_hemaia_repos(repo_root: Path) -> None:
    """Run ``1_git_pull_private_modules.sh`` with D2D and macro support enabled."""
    # ci is not running with the PLL, so we disable it to avoid unnecessary errors about missing PLL
    run_host_script(
        repo_root / "target/tapeout/1_git_pull_private_modules.sh",
        script_args=["--pll=0", "--d2d=1", "--macro=1"],
    )


# ---------------------------------------------------------------------------
# Step 3 -- Compile HW/SW inside the container
# ---------------------------------------------------------------------------

def step3_compile_hemaia_sw_rtl(repo_root: Path, docker_image: str) -> None:
    """Clean, then build SW, bootrom, and RTL inside the container."""
    # make sw
    run_in_container(repo_root, docker_image, repo_root, ["make", "sw", f"CFG_OVERRIDE={CI_CFG}"])
    # make bootrom
    run_in_container(repo_root, docker_image, repo_root, ["make", "bootrom", f"CFG_OVERRIDE={CI_CFG}"])
    # make rtl
    run_in_container(repo_root, docker_image, repo_root, ["make", "rtl", f"CFG_OVERRIDE={CI_CFG}"])


# ---------------------------------------------------------------------------
# Step 4 -- Build applications and prepare per-task directories
# ---------------------------------------------------------------------------

def step4_build_apps_and_prepare_tasks(
    repo_root: Path,
    docker_image: str,
    tasks: List[Dict[str, str]],
    task_base_dir: Path,
) -> List[Tuple[Path, str]]:
    """Build hex files inside the container, then copy them into per-task dirs.

    Returns a list of ``(task_dir, ci_name)`` tuples.
    """
    results: List[Tuple[Path, str]] = []
    app_subdirs = ["app_chip_0_0", "app_chip_0_1", "app_chip_1_0", "app_chip_1_1", "mempool"]

    for idx, task in enumerate(tasks):
        ci_name = task["ci_name"]

        # -- Build the application inside the container --
        make_cmd: List[str] = ["make", "apps", f"HOST_APP_TYPE={task['host_app_type']}"]
        for key, var in [("chip_type", "CHIP_TYPE"), ("workload", "WORKLOAD"), ("dev_app", "DEV_APP")]:
            val = task.get(key, "")
            if val and val != "None":
                make_cmd.append(f"{var}={val}")
        make_cmd.append(f"CFG_OVERRIDE={CI_CFG}")
        make_cmd.append("DEBUG_LEVEL=0")

        run_in_container(repo_root, docker_image, repo_root, make_cmd)

        # -- Copy generated hex directories into a unique task folder --
        task_dir = task_base_dir / f"task_{idx}"
        bin_dest = task_dir / "bin"
        bin_dest.mkdir(parents=True, exist_ok=True)

        sim_bin = repo_root / "target/sim/bin"
        for subdir in app_subdirs:
            src = sim_bin / subdir
            if src.exists():
                dst = bin_dest / subdir
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        results.append((task_dir, ci_name))
    return results


# ---------------------------------------------------------------------------
# Step 5 -- Prepare Questasim testharness and compile simulation
# ---------------------------------------------------------------------------

def step5_prepare_testharness(
    repo_root: Path,
    docker_image: str,
    tasks_info: List[Tuple[Path, str]],
) -> None:
    """Generate filelists in-container, compile on host, distribute artefacts."""
    sim_cfg_abs = str(repo_root / SIM_CFG)

    # Generate Questasim filelists inside the container
    run_in_container(
        repo_root, docker_image, repo_root,
        ["make", "hemaia_system_vsim_preparation", f"SIM_CFG={sim_cfg_abs}", "SIM_WITH_WAVEFORM=0"],
    )

    # Compile the Questasim simulation on the host
    subprocess.run(
        ["make", "hemaia_system_vsim", f"SIM_CFG={sim_cfg_abs}", "SIM_WITH_WAVEFORM=0"],
        cwd=repo_root, check=True,
    )

    # Copy simulation artefacts into each task directory
    sim_root = repo_root / "target/sim"
    artifacts = [
        sim_root / "bin" / "occamy_chip.vsim",
        sim_root / "work-vsim",
    ]
    for task_dir, _ in tasks_info:
        for src in artifacts:
            rel = src.relative_to(sim_root)
            _copy_path(src, task_dir / rel)


def _format_duration(seconds: float) -> str:
    """Format a wall-clock duration as e.g. '1h02m11s', '18m44s', or '44s'."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Step 6 -- Run simulations concurrently
# ---------------------------------------------------------------------------

def step6_run_simulations(
    tasks_info: List[Tuple[Path, str]],
    max_workers: int,
) -> Dict[str, Tuple[bool, float]]:
    """Execute simulations in parallel.

    Returns a map of ``task_dir -> (passed, wall_clock_seconds)``.
    """
    results: Dict[str, Tuple[bool, float]] = {}

    def _worker(task_dir: Path, ci_name: str):
        sim_binary = task_dir / "bin" / "occamy_chip.vsim"
        proc = None
        pgid = None
        start = time.monotonic()
        try:
            # start_new_session=True makes the child a session/group leader, so
            # its pgid equals its pid and *all* descendants (vish, vsimk) share
            # that group and can be killed as a unit -- never orphaned.
            proc = subprocess.Popen(
                [str(sim_binary)],
                cwd=task_dir / "bin",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            pgid = proc.pid
            _register_pgid(pgid)
            try:
                out_bytes, _ = proc.communicate(timeout=SIM_TIMEOUT_SECONDS)
                elapsed = time.monotonic() - start
                out = out_bytes.decode(errors="replace") if out_bytes else ""
                return str(task_dir), ci_name, proc.returncode == 0, elapsed, out
            except subprocess.TimeoutExpired:
                _kill_pgid(pgid)
                out_bytes, _ = proc.communicate()
                elapsed = time.monotonic() - start
                partial = out_bytes.decode(errors="replace") if out_bytes else ""
                msg = (
                    f"TIMEOUT: simulation exceeded {SIM_TIMEOUT_SECONDS}s "
                    f"({SIM_TIMEOUT_SECONDS // 3600}h) and was killed.\n"
                    + partial
                )
                return str(task_dir), ci_name, False, elapsed, msg
        except Exception:
            elapsed = time.monotonic() - start
            if pgid is not None:
                _kill_pgid(pgid)
            return str(task_dir), ci_name, False, elapsed, traceback.format_exc()
        finally:
            if pgid is not None:
                _unregister_pgid(pgid)

    if not tasks_info:
        return results

    worker_count = min(max_workers, len(tasks_info))
    print(
        f"Running {len(tasks_info)} simulations with up to "
        f"{worker_count} parallel job(s)"
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {}
        for td, cn in tasks_info:
            print(f"Starting simulation for {cn} ({td})")
            future_to_task[executor.submit(_worker, td, cn)] = (td, cn)

        for future in as_completed(future_to_task):
            td, cn = future_to_task[future]
            try:
                task_dir_str, ci_name, passed, elapsed, output = future.result()
                results[task_dir_str] = (passed, elapsed)
                status = "PASS" if passed else "FAIL"
                print(f"{ci_name}: {status} ({_format_duration(elapsed)})")
                if output:
                    for line in output.strip().splitlines()[-10:]:
                        print(line)
            except Exception:
                print(f"Error obtaining result for task {td}:")
                traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# Step 7 -- Write Markdown summary
# ---------------------------------------------------------------------------

def write_summary(
    summary_path: Path,
    tasks_info: List[Tuple[Path, str]],
    results: Dict[str, Tuple[bool, float]],
) -> None:
    """Write a Markdown report listing each task's pass/fail status and runtime."""
    with summary_path.open("w") as f:
        f.write("# Simulation Summary\n\n")
        f.write(f"Run by **{getpass.getuser()}** at **{datetime.now():%Y-%m-%d %H:%M:%S}**\n\n")
        for idx, (task_dir, ci_name) in enumerate(tasks_info):
            passed, elapsed = results.get(str(task_dir), (False, None))
            status = "PASS" if passed else "FAIL"
            runtime = _format_duration(elapsed) if elapsed is not None else "n/a"
            f.write(f"- **task_{idx}** ({ci_name}) \u2014 **{status}** \u2014 {runtime}\n")


# ---------------------------------------------------------------------------
# Step 8 -- Clean generated task directories
# ---------------------------------------------------------------------------

def cleanup_stale_task_dirs(task_base_dir: Path) -> None:
    """Delete any leftover ``task_*`` run directories from previous runs.

    Step 8 only removes folders created by the current run, so a crashed or
    interrupted run leaves stale ``task_*`` directories behind.  Sweep them all
    before starting a fresh flow.
    """
    removed = 0
    # Match only numbered run dirs (task_0, task_1, ...)
    for stale_dir in sorted(task_base_dir.glob("task_[0-9]*")):
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir)
            removed += 1
            print(f"Removed stale simulation folder: {stale_dir}")

    if removed:
        print(f"Cleaned up {removed} stale simulation task folder(s)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[4]          # target/sim/automation/ci -> repo root
    task_base_dir = script_path.parent
    docker_image = "ghcr.io/kuleuven-micas/hemaia:main"
    task_yaml = script_path.parent / "task_vsim.yaml"

    if not task_yaml.exists():
        raise FileNotFoundError(f"Task YAML file {task_yaml} does not exist")

    ensure_vsim_available()

    # Guarantee any simulation we launch is fully torn down on exit, timeout,
    # crash, or Ctrl-C -- no orphaned vsim/vish/vsimk threads left pinning CPUs.
    _install_cleanup_handlers()

    # Remove leftover task_* directories from any previous (crashed) run
    cleanup_stale_task_dirs(task_base_dir)

    # Step 0: Parse task definitions from YAML
    tasks = step0_parse_tasks(task_yaml)

    # Step 1: Reset environment to a clean state
    step1_clean(repo_root, docker_image)

    # Step 2: Initialise private repos on the host
    step2_init_private_hemaia_repos(repo_root)

    # Step 3: Compile HW and SW inside the container
    step3_compile_hemaia_sw_rtl(repo_root, docker_image)

    # Step 4: Build application hex files and create per-task directories
    tasks_info = step4_build_apps_and_prepare_tasks(repo_root, docker_image, tasks, task_base_dir)

    # Step 5: Prepare and compile the Questasim testharness
    step5_prepare_testharness(repo_root, docker_image, tasks_info)

    # Step 6: Run all simulations concurrently
    results = step6_run_simulations(tasks_info, args.max_sim_jobs)

    # Step 7: Write Markdown summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = script_path.parent / f"simulation_summary_{timestamp}.md"
    write_summary(summary_path, tasks_info, results)
    print(f"Simulation summary written to {summary_path}")


if __name__ == "__main__":
    main()
