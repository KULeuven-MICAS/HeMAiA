#!/usr/bin/env python3
"""
HeMAiA simulation automation -- common runner
=============================================

A single, parameterised driver shared by the HeMAiA test, CI, and sweep flows
under ``target/sim/automation``.  All of them share the same backbone:

  Step 0 - Parse / accept the task list
  Step 1 - ``make clean`` (container) then reset + init the private vendor repos
  Step 2 - Build SW / bootrom / RTL inside the build container
  Step 3 - Build per-task application hex and create ``task_<idx>`` directories
  Step 4 - Prepare the testharness (container) and compile the simulation (host)
  Step 5 - Run the simulations concurrently
  Step 6 - Write a Markdown summary

The flows differ only by parameters captured on :class:`HeMAiASimRunner`:

  * ``engine``       -- ``"vcs"`` (CI / sweep), ``"vsim"`` (interactive test), or
    ``"vlt"`` (Verilator)
  * ``with_waveform``-- ``SIM_WITH_WAVEFORM`` (0 for CI/sweep, 1 for test)
  * ``cfg`` / ``sim_cfg``           -- RTL and simulation config files
  * ``with_macro`` / ``with_d2d``   -- private vendor modules
  * ``with_pll``     -- the vendor PLL: selects the private clk/rst controller
    *and* sets ``use_vendor_pll`` in the RTL cfg (see :func:`resolve_rtl_cfg`)
  * ``spm_wide_size``-- sets ``spm_wide.length`` in the RTL cfg, sizing the
    ``WIDE_SPM`` region a simulation gets

The ``engine`` is reduced to a small lookup (:data:`ENGINES`) of the make
targets and artefacts that differ between the simulators; everything else is
engine-agnostic.  ``with_waveform`` is forwarded to the make targets, which gate
the per-engine waveform flags (``log -r /*`` for vsim, ``+vcs+fsdbon`` for vcs;
Verilator ignores it).
"""

from __future__ import annotations

import atexit
import argparse
import getpass
import os
import re
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
from typing import Dict, List, Optional, Set, Tuple

import hjson

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_DOCKER_IMAGE = "ghcr.io/kuleuven-micas/hemaia:main"
SIM_TIMEOUT_SECONDS = 4 * 60 * 60  # 4 hours per simulation thread
DEFAULT_MAX_SIM_JOBS = 16

# Per-engine differences: the make targets to run, where the compile happens, the
# produced binary, and the artefacts to stage into each task directory (paths are
# relative to target/sim).  vsim/vcs split a container "preparation" step from a
# host-side compile (the EDA tools are host-only); Verilator builds entirely in
# the container in a single target.  The vsim launcher is a wrapper script that
# references ``../work-vsim``; the vcs simv binary needs its ``.daidir`` companion
# next to it; the vlt binary is a self-contained executable.
ENGINES: Dict[str, Dict] = {
    "vsim": {
        "tool": "vsim",                  # host EDA tool required to compile
        "prep_target": "hemaia_system_vsim_preparation",
        "compile_target": "hemaia_system_vsim",
        "compile_in_container": False,
        "binary": "occamy_chip.vsim",
        "stage": ["bin/occamy_chip.vsim", "work-vsim"],
    },
    "vcs": {
        "tool": "vcs",
        "prep_target": "hemaia_system_vcs_preparation",
        "compile_target": "hemaia_system_vcs",
        "compile_in_container": False,
        "binary": "occamy_chip.vcs",
        "stage": ["bin/occamy_chip.vcs", "bin/occamy_chip.vcs.daidir"],
        # With many parallel jobs the VCS runtime license seats run out; without
        # this flag the simv that loses the race exits immediately ("Licensed
        # number of users already reached for VCS-BASE-RUNTIME") and is wrongly
        # reported as FAIL.  +vcs+lic+wait makes it queue for a seat instead.
        "run_args": ["+vcs+lic+wait"],
    },
    "vlt": {
        "tool": None,                    # Verilator builds in-container; no host tool
        "prep_target": None,             # single build target, no separate prep
        "compile_target": "hemaia_system_vlt",
        "compile_in_container": True,
        "binary": "occamy_chip.vlt",
        "stage": ["bin/occamy_chip.vlt"],
    },
}


# Derived RTL cfgs land here (gitignored, wiped by ``make clean-repo``).
GENERATED_CFG_DIR = "target/rtl/cfg/generated"


# The testharness' verdict, printed by ``target/sim/testharness/util/check_finish_*.sv``
# once every chip has written its end-of-computation marker.
SIM_OK_MARKER = "All chips finished successfully"
SIM_ERR_MARKER = "finished with errors"


def _sim_passed(returncode: int, out: str) -> bool:
    """Did the simulation actually succeed?

    The exit code cannot answer this. ``$finish(-1)`` does not set one -- in
    SystemVerilog that argument selects the diagnostic verbosity, not a status -- so a
    simulator whose testharness detected a nonzero EOC code still exits 0. A workload
    that aborts (an L1 OOM, a failed golden check) is therefore indistinguishable from a
    clean run by exit code alone.

    The testharness' own verdict is authoritative, so require it: every chip must have
    reached the success marker, and no chip may have reported an error. Absence of both
    means the run died before the EOC check (a crash, a kill, a licence failure) and is
    not a pass either.
    """
    if returncode != 0:
        return False
    if SIM_ERR_MARKER in out:
        return False
    return SIM_OK_MARKER in out


# ---------------------------------------------------------------------------
# RTL config resolution
# ---------------------------------------------------------------------------

def resolve_rtl_cfg(
    repo_root: Path,
    cfg: str,
    *,
    with_pll: bool,
    spm_wide_size: Optional[int] = None,
) -> str:
    """Return the repo-relative RTL cfg to pass as ``CFG_OVERRIDE``.

    A flow names a tapeout cfg and states the two fields a simulation may need to
    differ on; this reconciles them:

    * ``with_pll`` is the single source of truth for the vendor PLL.  It selects
      the private clk/rst controller in ``Bender.local``, and it must agree with
      ``use_vendor_pll`` in the RTL cfg: ``occamy_chip.sv.tpl`` instantiates
      ``hemaia_clk_rst_controller`` unconditionally, so a cfg claiming
      ``use_vendor_pll: true`` under ``with_pll=False`` elaborates the *public*
      module with ``USE_VENDOR_PLL(1)``.
    * ``spm_wide_size`` (bytes, ``None`` = leave the cfg alone) sets
      ``spm_wide.length``, i.e. the ``WIDE_SPM`` region in ``host.ld``.  128 kiB
      is too small for parts of the SW tree -- a 2 MiB ``.devicebin`` and the gemm
      sweep's ``D_cfg`` tables overflow it at link time.

    A cfg that already matches on every requested field is returned untouched; a
    patched copy is only written when something actually differs, so flows that
    ask for exactly what their cfg declares run it byte-for-byte.  The copy lands
    under :data:`GENERATED_CFG_DIR`, named after the fields it changed.
    """
    src = repo_root / cfg
    cfg_dict = hjson.loads(src.read_text())

    suffixes: List[str] = []
    if bool(cfg_dict.get("use_vendor_pll", False)) != with_pll:
        cfg_dict["use_vendor_pll"] = with_pll
        suffixes.append("pll" if with_pll else "nopll")
    if spm_wide_size is not None and cfg_dict["spm_wide"]["length"] != spm_wide_size:
        cfg_dict["spm_wide"]["length"] = spm_wide_size
        suffixes.append(f"spm{spm_wide_size // 1024}k")

    if not suffixes:
        return cfg

    gen_dir = repo_root / GENERATED_CFG_DIR
    gen_dir.mkdir(parents=True, exist_ok=True)
    dst = gen_dir / f"{'_'.join([src.stem, *suffixes])}.hjson"
    # The dump must be deterministic: the root Makefile only re-copies
    # CFG_OVERRIDE onto lru.hjson (and thus retriggers every cfg-dependent
    # rebuild) when the two differ byte-for-byte.
    dst.write_text(hjson.dumps(cfg_dict, indent=4) + "\n")
    print(f"Derived RTL cfg from {cfg} [{', '.join(suffixes)}]: {dst}")
    return str(dst.relative_to(repo_root))


# ---------------------------------------------------------------------------
# Process-group tracking -- guarantees no stale sim threads survive
# ---------------------------------------------------------------------------
#
# A simulation launcher spawns descendant processes (vish -> vsimk for vsim).
# Killing only the immediate child (as ``subprocess.run`` does on timeout) leaves
# those descendants orphaned onto ``init`` where they keep pinning a CPU core
# forever.  We therefore start each simulation in its own session/process group
# and tear the *whole group* down on timeout, crash, or interrupt.

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
    """Kill every still-running simulation process group."""
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
# Generic command helpers
# ---------------------------------------------------------------------------

def run_host_script(script_path: Path, script_args: Optional[List[str]] = None) -> None:
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
    """Run *command* inside *docker_image*, mounting *repo_root* at the same path.

    Prefers ``podman`` when available; otherwise falls back to ``apptainer exec``.
    """
    if shutil.which("podman") is not None:
        runner_cmd: List[str] = [
            "podman", "run", "--rm",
            "-v", f"{repo_root}:{repo_root}",
            "-w", str(working_dir),
            docker_image,
        ]
    elif shutil.which("apptainer") is not None:
        runner_cmd = [
            "apptainer", "exec",
            "--writable-tmpfs",
            "--pwd", str(working_dir),
            "-B", f"{repo_root}:{repo_root}",
            f"docker://{docker_image}",
        ]
    else:
        raise RuntimeError(
            "Neither 'podman' nor 'apptainer' is available on PATH; "
            "install one of them to run the HeMAiA build container."
        )

    runner_cmd.extend(command)
    subprocess.run(runner_cmd, check=True)


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
# Task-list parsing
# ---------------------------------------------------------------------------

def _simple_task_yaml_loader(task_yaml: Path) -> Dict:
    """Parse the limited ``task_*.yaml`` format without PyYAML.

    Supports a top-level ``runs:`` key containing a sequence of dict entries,
    each with one or more ``KEY: VALUE`` attribute lines (the first key sits on
    the same line as the ``- `` dash).
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

    current: Optional[Dict[str, str]] = None
    item_indent: Optional[int] = None

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


def _derive_ci_name(host_app_type: str, chip_type: str, workload: str, dev_app: str) -> str:
    """Derive a human-readable, unique task name from its attributes."""
    parts = [host_app_type, chip_type]
    if workload and workload != "None":
        parts.append(workload)
    if dev_app and dev_app != "None":
        parts.append(dev_app)
    return "_".join(p for p in parts if p)


def task_dir_name(idx: int, ci_name: str) -> str:
    """Run-directory name for task *idx*: ``task_<idx>_<ci_name>``.

    The numeric index stays in front and stays authoritative: it is the task's
    position in the task YAML, which is how the sweep gatherers pair a run
    directory with the workload that produced it (see :func:`find_task_dir`).
    The ``ci_name`` suffix is what makes the directory readable on disk.

    Every task directory carries the suffix -- :func:`find_task_dir` looks for
    ``task_<idx>_*`` and would not see a bare ``task_<idx>``.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ci_name).strip("_")
    if not slug:
        raise ValueError(f"task {idx}: ci_name {ci_name!r} has no usable characters")
    return f"task_{idx}_{slug}"


def find_task_dir(root, idx: int) -> Optional[Path]:
    """Locate the run directory for task *idx* under *root*, or None.

    Matches on the ``task_<idx>_`` prefix rather than reconstructing the full
    name, so a consumer only needs the index, not the task list.  The trailing
    underscore is what keeps ``task_1_*`` from also matching ``task_12_...``.
    """
    hits = sorted(Path(root).glob(f"task_{idx}_*"))
    return hits[0] if hits else None


def make_task(
    *,
    host_app_type: str,
    chip_type: str = "",
    workload: str = "",
    dev_app: str = "",
) -> Dict[str, str]:
    """Build a single task dict (used by the test drivers for a 1-element list)."""
    return {
        "host_app_type": host_app_type,
        "chip_type": chip_type,
        "workload": workload,
        "dev_app": dev_app,
        "ci_name": _derive_ci_name(host_app_type, chip_type, workload, dev_app),
    }


def resolve_task_yaml(arg) -> Path:
    """Resolve a ``--task-yaml`` value to a concrete path: an absolute path is
    returned as-is, a relative one is resolved against the current directory.

    Drivers set the argparse default to ``<script_dir>/<name>.yaml`` (an absolute
    path), so the no-flag case already points next to the script regardless of
    cwd.  Shared so every driver resolves task lists the same way.
    """
    p = Path(arg)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def parse_tasks(task_yaml: Path) -> List[Dict[str, str]]:
    """Parse simulation tasks from *task_yaml*.

    Expected YAML structure::

        runs:
          - HOST_APP_TYPE: offload_bingo_sw
            CHIP_TYPE: single_chip
            WORKLOAD: gemm_tiled_1cluster
            DEV_APP: snax-bingo-offload

    Kept as a stable module-level function: the xDMA sweep post-processing
    (``gather_xdma_luts.py``) imports it to recover the task_<idx> -> workload
    order.
    """
    if yaml is not None:
        with open(task_yaml, "r") as f:
            param_data = yaml.safe_load(f)
    else:
        param_data = _simple_task_yaml_loader(Path(task_yaml))

    tasks: List[Dict[str, str]] = []
    for attr_dict in param_data.get("runs", []):
        if "HOST_APP_TYPE" not in attr_dict:
            raise ValueError(f"Task is missing 'HOST_APP_TYPE': {attr_dict}")
        tasks.append(make_task(
            host_app_type=str(attr_dict.get("HOST_APP_TYPE", "")),
            chip_type=str(attr_dict.get("CHIP_TYPE", "")),
            workload=str(attr_dict.get("WORKLOAD", "")),
            dev_app=str(attr_dict.get("DEV_APP", "")),
        ))
    return tasks


# ---------------------------------------------------------------------------
# Shared CLI argument parsers
# ---------------------------------------------------------------------------

def parse_workload_args(
    *,
    default_host_app_type: str,
    default_chip_type: str,
    default_workload: str,
    default_dev_app: str,
    default_engine: str = "vsim",
    default_waveform: int = 1,
    description: Optional[str] = None,
) -> argparse.Namespace:
    """Parse CLI overrides for the SW workload-selection knobs (test drivers).

    Each entrypoint passes its module-level constants as the defaults, so the
    script keeps working with no arguments while a user can pick a different
    workload from the command line.  The ``--engine`` / ``--waveform`` knobs are
    also exposed so any flow can be flipped (e.g. debug a CI failure under
    vsim+waveform), defaulting to the test convention (vsim, waveform on).
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host-app-type", default=default_host_app_type,
        help="host_only | offload_legacy | offload_bingo_hw | offload_bingo_sw "
             "(default: %(default)s)")
    parser.add_argument(
        "--chip-type", default=default_chip_type,
        help="single_chip | multi_chip (default: %(default)s)")
    parser.add_argument(
        "--workload", default=default_workload,
        help="workload directory name, or None (default: %(default)s)")
    parser.add_argument(
        "--dev-app", default=default_dev_app,
        help="device app (default: %(default)s)")
    parser.add_argument(
        "--engine", choices=sorted(ENGINES), default=default_engine,
        help="simulation engine (default: %(default)s)")
    parser.add_argument(
        "--waveform", type=int, choices=(0, 1), default=default_waveform,
        help="SIM_WITH_WAVEFORM: record a waveform/log (default: %(default)s)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

class HeMAiASimRunner:
    """Drives the full build-and-simulate flow for a list of tasks."""

    def __init__(
        self,
        *,
        repo_root: Path,
        output_dir: Path,
        engine: str,
        with_waveform: bool,
        cfg: str,
        sim_cfg: str,
        with_macro: bool,
        with_d2d: bool,
        with_pll: bool,
        spm_wide_size: Optional[int] = None,
        max_jobs: int = DEFAULT_MAX_SIM_JOBS,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        in_container: bool = False,
        skip_setup: bool = False,
        skip_build: bool = False,
        skip_compile: bool = False,
        compile_jobs: Optional[int] = None,
        build_jobs: Optional[int] = None,
    ) -> None:
        if engine not in ENGINES:
            raise ValueError(f"Unknown engine {engine!r}; choose from {sorted(ENGINES)}")
        self.repo_root = repo_root.resolve()
        self.output_dir = output_dir.resolve()
        self.engine = engine
        self.spec = ENGINES[engine]
        self.waveform = "1" if with_waveform else "0"
        self.cfg = cfg              # repo-relative RTL cfg as requested
        # The cfg actually handed to make as CFG_OVERRIDE.  ``run()`` re-resolves
        # it through resolve_rtl_cfg(), which derives a patched copy when the cfg
        # disagrees with ``with_pll`` / ``spm_wide_size``.
        self.effective_cfg = cfg
        self.sim_cfg = sim_cfg      # repo-relative sim cfg
        self.with_macro = with_macro
        self.with_d2d = with_d2d
        self.with_pll = with_pll
        # spm_wide.length (bytes) to patch into the cfg; None = keep the cfg's own.
        self.spm_wide_size = spm_wide_size
        self.max_jobs = max_jobs
        self.docker_image = docker_image
        # Optional flow switches (e.g. the git CI runs inside the build container
        # but has no SSH for the private vendor repos):
        #   in_container -- run the "container" make steps directly, no podman
        #   skip_setup   -- skip reset + private-repo init
        #   skip_build   -- skip the sw/bootrom/rtl build (built elsewhere)
        #   skip_compile -- skip the sim prepare+compile, but still stage the
        #                   already-built binary into each task dir
        self.in_container = in_container
        self.skip_setup = skip_setup
        self.skip_build = skip_build
        self.skip_compile = skip_compile
        # Parallelism for the sim compile (``make -j``); None = no -j flag. The
        # Verilator build in particular is far too slow serially on a CI runner.
        self.compile_jobs = compile_jobs
        # `make -jN` for the SW build. Only `sw` is parallel-safe (and it is the slow
        # one: ~96 host apps x ~20 device apps). `rtl` runs occamygen + bender script
        # generation and `bootrom` takes seconds -- both stay serial.
        self.build_jobs = build_jobs

    # -- helpers -----------------------------------------------------------

    def _ensure_engine_available(self) -> None:
        tool = self.spec["tool"]
        if tool is None:  # e.g. Verilator: built and run without a host EDA tool
            return
        if shutil.which(tool) is None:
            print(
                f"ERROR: '{tool}' not found on PATH.\n"
                "Please source the EDA setup script and re-run this script.",
                file=sys.stderr,
            )
            sys.exit(1)

    def _container(self, command: List[str]) -> None:
        """Run a build step in the container, or directly if already inside one."""
        if self.in_container:
            subprocess.run(command, cwd=self.repo_root, check=True)
        else:
            run_in_container(self.repo_root, self.docker_image, self.repo_root, command)

    # -- Step 1: reset + init private repos --------------------------------

    def reset_and_init(self) -> None:
        """``make clean`` (container), then reset + init the private vendor repos.

        Unifies on the canonical tapeout shell scripts so a single-chip run does
        not inherit tsmc16/D2D/PLL from a previous multi-chip run.
        """
        self._container(["make", "clean"])
        tapeout = self.repo_root / "target/tapeout"
        run_host_script(tapeout / "0_reset_private_modules.sh")
        run_host_script(
            tapeout / "1_git_pull_private_modules.sh",
            script_args=[
                f"--macro={1 if self.with_macro else 0}",
                f"--d2d={1 if self.with_d2d else 0}",
                f"--pll={1 if self.with_pll else 0}",
            ],
        )

    # -- Step 2: build SW / bootrom / RTL ----------------------------------

    def build_hw_sw(self) -> None:
        """Build SW, bootrom, and RTL inside the container."""
        for target in ("sw", "bootrom", "rtl"):
            cmd = ["make"]
            if self.build_jobs and target == "sw":
                # Never inside a recipe: every recursion uses $(MAKE), so one top-level
                # -jN hands the whole tree a single jobserver.
                cmd.append(f"-j{self.build_jobs}")
            cmd += [target, f"CFG_OVERRIDE={self.effective_cfg}"]
            self._container(cmd)

    # -- Step 3: build apps and stage per-task directories -----------------

    def build_apps_and_stage(self, tasks: List[Dict[str, str]]) -> List[Tuple[Path, str]]:
        """Build hex inside the container, then copy them into per-task dirs.

        Returns a list of ``(task_dir, ci_name)`` tuples.
        """
        results: List[Tuple[Path, str]] = []
        app_subdirs = ["app_chip_0_0", "app_chip_0_1", "app_chip_1_0", "app_chip_1_1", "mempool"]
        sim_bin = self.repo_root / "target/sim/bin"

        for idx, task in enumerate(tasks):
            make_cmd: List[str] = ["make", "apps", f"HOST_APP_TYPE={task['host_app_type']}"]
            for key, var in [("chip_type", "CHIP_TYPE"), ("workload", "WORKLOAD"), ("dev_app", "DEV_APP")]:
                val = task.get(key, "")
                if val and val != "None":
                    make_cmd.append(f"{var}={val}")
            make_cmd.append(f"CFG_OVERRIDE={self.effective_cfg}")
            make_cmd.append("DEBUG_LEVEL=0")
            self._container(make_cmd)

            task_dir = self.output_dir / task_dir_name(idx, task["ci_name"])
            bin_dest = task_dir / "bin"
            bin_dest.mkdir(parents=True, exist_ok=True)
            for subdir in app_subdirs:
                _copy_path(sim_bin / subdir, bin_dest / subdir)

            results.append((task_dir, task["ci_name"]))
        return results

    # -- Step 4: prepare + compile the simulation --------------------------

    def prepare_and_compile(self, tasks_info: List[Tuple[Path, str]]) -> None:
        """Build the simulation and distribute its artefacts to each task dir.

        vsim/vcs prepare the testharness/filelists in the container, then compile
        on the host (EDA tools are host-only).  Verilator has no separate prep step
        and builds entirely in the container.
        """
        if not self.skip_compile:
            sim_cfg_abs = str(self.repo_root / self.sim_cfg)
            wave = f"SIM_WITH_WAVEFORM={self.waveform}"

            # Optional container-side preparation (testharness/filelists).
            if self.spec["prep_target"]:
                self._container(["make", self.spec["prep_target"], f"SIM_CFG={sim_cfg_abs}", wave])

            # Compile the simulation -- in the container (Verilator) or on the host.
            compile_cmd = ["make", self.spec["compile_target"], f"SIM_CFG={sim_cfg_abs}", wave]
            if self.compile_jobs:
                compile_cmd.append(f"-j{self.compile_jobs}")
            if self.spec["compile_in_container"]:
                self._container(compile_cmd)
            else:
                subprocess.run(compile_cmd, cwd=self.repo_root, check=True)

        # Insist the compile produced a binary before staging it.  ``_copy_path()``
        # silently skips a missing source, so a ``make`` that exits 0 without one (a dead
        # EDA tool, e.g. a VCS/glibc mismatch) would otherwise surface as every task dying
        # at launch with a bare FileNotFoundError -- which reads like N failing tests
        # rather than one failed build.
        sim_root = self.repo_root / "target/sim"
        binary = sim_root / "bin" / self.spec["binary"]
        if not binary.exists():
            raise RuntimeError(
                f"{self.engine} compile produced no {binary.name}: {binary} does not "
                f"exist.\nThe make target exited 0 anyway -- inspect "
                f"target/sim/work-{self.engine}/ for the real error "
                f"(e.g. {self.spec['tool'] or self.engine} failing to start)."
            )

        # Copy the engine's simulation artefacts into each task directory.
        for rel in self.spec["stage"]:
            src = sim_root / rel
            for task_dir, _ in tasks_info:
                _copy_path(src, task_dir / rel)

    # -- Step 5: run simulations concurrently ------------------------------

    def run_simulations(
        self, tasks_info: List[Tuple[Path, str]],
    ) -> Tuple[Dict[str, Tuple[bool, float]], float]:
        """Execute simulations in parallel.

        Returns ``(results, total_wall_clock_seconds)`` where *results* maps
        ``task_dir -> (passed, wall_clock_seconds)``.
        """
        results: Dict[str, Tuple[bool, float]] = {}
        binary_name = self.spec["binary"]

        def _worker(task_dir: Path, ci_name: str):
            sim_binary = task_dir / "bin" / binary_name
            proc = None
            pgid = None
            start = time.monotonic()
            try:
                # start_new_session=True makes the child a session/group leader,
                # so all descendants share that group and can be killed as a unit.
                proc = subprocess.Popen(
                    [str(sim_binary), *self.spec.get("run_args", [])],
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
                    return str(task_dir), ci_name, _sim_passed(proc.returncode, out), elapsed, out
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
            return results, 0.0

        worker_count = min(self.max_jobs, len(tasks_info))
        print(f"Running {len(tasks_info)} {self.engine} simulation(s) with up to "
              f"{worker_count} parallel job(s)")

        phase_start = time.monotonic()
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
                    # Persist the full sim stdout/stderr per task so a failure can
                    # be triaged afterwards (the console only shows the last lines).
                    if output:
                        try:
                            (Path(task_dir_str) / "bin" / "sim_run.log").write_text(output)
                        except OSError:
                            pass
                        for line in output.strip().splitlines()[-10:]:
                            print(line)
                except Exception:
                    print(f"Error obtaining result for task {td}:")
                    traceback.print_exc()

        total_elapsed = time.monotonic() - phase_start
        print(f"All {len(tasks_info)} simulation(s) finished in "
              f"{_format_duration(total_elapsed)}")
        return results, total_elapsed

    # -- Step 6: write Markdown summary ------------------------------------

    def write_summary(
        self,
        summary_path: Path,
        tasks_info: List[Tuple[Path, str]],
        results: Dict[str, Tuple[bool, float]],
        total_seconds: float,
    ) -> None:
        """Write a Markdown report listing each task's pass/fail status and runtime."""
        with summary_path.open("w") as f:
            f.write("# Simulation Summary\n\n")
            f.write(f"Run by **{getpass.getuser()}** at "
                    f"**{datetime.now():%Y-%m-%d %H:%M:%S}** "
                    f"(engine: {self.engine}, waveform: {self.waveform})\n\n")
            # The directory name already carries the ci_name, so it identifies the
            # task on its own and doubles as the path to go look at.
            for task_dir, _ci_name in tasks_info:
                passed, elapsed = results.get(str(task_dir), (False, None))
                status = "PASS" if passed else "FAIL"
                runtime = _format_duration(elapsed) if elapsed is not None else "n/a"
                f.write(f"- **{task_dir.name}** — **{status}** — {runtime}\n")

            cumulative = sum(e for _, e in results.values() if e is not None)
            f.write(
                f"\n**Total wall-clock time: {_format_duration(total_seconds)}**"
                f" (cumulative across {len(tasks_info)} task(s): "
                f"{_format_duration(cumulative)})\n"
            )

    # -- Housekeeping ------------------------------------------------------

    def cleanup_stale_task_dirs(self) -> None:
        """Delete leftover ``task_*`` run directories from previous runs."""
        removed = 0
        for stale_dir in sorted(self.output_dir.glob("task_[0-9]*")):
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)
                removed += 1
                print(f"Removed stale simulation folder: {stale_dir}")
        if removed:
            print(f"Cleaned up {removed} stale simulation task folder(s)")

    # -- Orchestrator ------------------------------------------------------

    def run(self, tasks: List[Dict[str, str]]) -> Path:
        """Run the full flow for *tasks* and return the summary path."""
        self._ensure_engine_available()
        # Guarantee any simulation we launch is torn down on exit/timeout/crash.
        _install_cleanup_handlers()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_stale_task_dirs()

        print(f"Engine: {self.engine}  waveform: {self.waveform}  cfg: {self.cfg}")
        spm = "cfg" if self.spm_wide_size is None else f"{self.spm_wide_size // 1024} kiB"
        print(f"  (macro={self.with_macro}, d2d={self.with_d2d}, pll={self.with_pll}, "
              f"spm_wide={spm})")

        if self.skip_setup:
            print("[Step 1] Skipping reset/init (skip_setup)")
        else:
            print("[Step 1] Cleaning and initialising private repos")
            self.reset_and_init()

        # Must run after Step 1: reset_and_init() calls ``make clean``, which wipes
        # the generated-cfg directory.
        self.effective_cfg = resolve_rtl_cfg(
            self.repo_root, self.cfg,
            with_pll=self.with_pll, spm_wide_size=self.spm_wide_size)

        if self.skip_build:
            print("[Step 2] Skipping SW/bootrom/RTL build (skip_build)")
        else:
            print("[Step 2] Building SW/bootrom/RTL")
            self.build_hw_sw()

        print("[Step 3] Building apps and staging per-task directories")
        tasks_info = self.build_apps_and_stage(tasks)

        if self.skip_compile:
            print("[Step 4] Staging the pre-built simulation (skip_compile)")
        else:
            print("[Step 4] Preparing and compiling the simulation")
        self.prepare_and_compile(tasks_info)

        print("[Step 5] Running simulations")
        results, total_elapsed = self.run_simulations(tasks_info)

        print("[Step 6] Writing summary")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"simulation_summary_{timestamp}.md"
        self.write_summary(summary_path, tasks_info, results, total_elapsed)
        print(f"Simulation summary written to {summary_path}")
        return summary_path
