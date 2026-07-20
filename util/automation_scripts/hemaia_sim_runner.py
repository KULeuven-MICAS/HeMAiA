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

The local netlist flow can split those steps across machines: ``prepare`` builds
all application images and produces relocatable VCS/Questa compile inputs,
whereas ``simulate`` validates that immutable hand-off before invoking the EDA
compiler and running the tasks.  The default ``all`` phase retains the original
end-to-end behaviour.

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
import hashlib
import json
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
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

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
PREPARATION_MANIFEST = "netlist_ci_preparation_manifest.json"
PREPARATION_MANIFEST_VERSION = 6
MEMPOOL_BANK_COUNT = 16
_HEX_LINE_RE = re.compile(r"(?:[0-9a-fA-F_xXzZ]+|@[0-9a-fA-F_]+)\Z")
_HDL_SOURCE_SUFFIXES = {".sv", ".v", ".vhd", ".vhdl"}
_HDL_INCLUDE_SUFFIXES = {".svh", ".vh"}


class SimulationSuiteFailed(RuntimeError):
    """Raised after writing a summary when one or more simulations failed."""


class PlatformLayout(NamedTuple):
    """Memory geometry needed to validate and stage simulation images."""

    coordinates: List[Tuple[int, int]]
    compute_bank_count: int
    compute_bank_depth: int
    mempool_bank_count: int
    mempool_bank_depth: Optional[int]

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
        # This testbench never uses $save/$restart.  Disabling that facility
        # avoids VCS re-executing every simv under ASLR before time zero.
        "run_args": ["+vcs+lic+wait", "-no_save"],
        # The matching compile-time option is enabled by target/sim/Makefile.
        # VCS requires it again at runtime to choose random 0/1 initialization.
        # Each task has a distinct bin/ working directory, so VCS writes one
        # live transcript per simulation while stdout remains captured by CI.
        "netlist_run_args": [
            "+vcs+initreg+random", "-l", "transcript", "+vcs+flush+log",
        ],
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
) -> str:
    """Return the repo-relative RTL cfg to pass as ``CFG_OVERRIDE``.

    A flow names a tapeout cfg and states the ``with_pll`` field a simulation may
    need to differ on; this reconciles it:

    * ``with_pll`` is the single source of truth for the vendor PLL.  It selects
      the private clk/rst controller in ``Bender.local``, and it must agree with
      ``use_vendor_pll`` in the RTL cfg: ``occamy_chip.sv.tpl`` instantiates
      ``hemaia_clk_rst_controller`` unconditionally, so a cfg claiming
      ``use_vendor_pll: true`` under ``with_pll=False`` elaborates the *public*
      module with ``USE_VENDOR_PLL(1)``.

    A cfg that already matches on every requested field is returned untouched; a
    patched copy is only written when something actually differs, so flows that
    ask for exactly what their cfg declares run it byte-for-byte.  The copy lands
    under :data:`GENERATED_CFG_DIR`, named after the fields it changed.
    """
    # Config derivation happens only in the build/preparation phases.  Keep the
    # dependency lazy so an EDA backend can validate and consume a prepared
    # hand-off with only the Python standard library installed.
    try:
        import hjson  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "The hjson Python package is required to derive the RTL config; "
            "run preparation in the HeMAiA build image."
        ) from error

    src = repo_root / cfg
    cfg_dict = hjson.loads(src.read_text())

    suffixes: List[str] = []
    if bool(cfg_dict.get("use_vendor_pll", False)) != with_pll:
        cfg_dict["use_vendor_pll"] = with_pll
        suffixes.append("pll" if with_pll else "nopll")

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


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        max_jobs: int = DEFAULT_MAX_SIM_JOBS,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        in_container: bool = False,
        skip_setup: bool = False,
        skip_build: bool = False,
        skip_compile: bool = False,
        compile_jobs: Optional[int] = None,
        build_jobs: Optional[int] = None,
        build_sw_fleet: bool = True,
        task_yaml: Optional[Path] = None,
        fail_on_task_failure: bool = False,
        timeout_seconds: int = SIM_TIMEOUT_SECONDS,
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
        # disagrees with ``with_pll``.
        self.effective_cfg = cfg
        self.sim_cfg = sim_cfg      # repo-relative sim cfg
        self.with_macro = with_macro
        self.with_d2d = with_d2d
        self.with_pll = with_pll
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
        # The broad ``make sw`` fleet intentionally includes applications that
        # do not target every hardware profile.  Fixed netlist CI instead builds
        # only its checked-in task list through strict ``make apps`` calls.
        self.build_sw_fleet = build_sw_fleet
        # The task source is optional for legacy/test callers.  Split CI flows
        # provide it so the hand-off manifest can detect task-list changes.
        self.task_yaml = task_yaml.resolve() if task_yaml is not None else None
        self.fail_on_task_failure = fail_on_task_failure
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")
        self.timeout_seconds = timeout_seconds

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

    def _repo_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.repo_root / candidate

    def _platform_layout(self) -> PlatformLayout:
        """Read chip coordinates and memory geometry from the effective cfg.

        ``hjson`` is part of the HeMAiA build image.  The deliberately small
        fallback parser covers the generated cfg format as well, keeping the
        backend-only phase usable with the Python standard library alone.
        """
        cfg_path = self._repo_path(self.effective_cfg)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"RTL/SW configuration does not exist: {cfg_path}")

        try:
            import hjson  # type: ignore

            with cfg_path.open("r") as source:
                data = hjson.load(source)
            multichip = data["hemaia_multichip"]
            testbench_cfg = multichip.get("testbench_cfg")
            if testbench_cfg is None and bool(multichip.get("single_chip")):
                coordinates = [(0, 0)]
            else:
                coordinates = [
                    (int(chip["coordinate"][0]), int(chip["coordinate"][1]))
                    for chip in testbench_cfg["hemaia_compute_chip"]
                ]
            compute_bank_count = int(data["spm_wide"]["banks"])
            compute_size = int(data["spm_wide"]["length"])
            mem_sizes = (
                []
                if testbench_cfg is None
                else [int(chip["mem_size"]) for chip in testbench_cfg["hemaia_mem_chip"]]
            )
        except (ImportError, KeyError, TypeError, ValueError):
            text = cfg_path.read_text()
            chip_start = text.find("hemaia_compute_chip")
            mem_start = text.find("hemaia_mem_chip", chip_start)
            if chip_start >= 0 and mem_start >= 0:
                coordinates = [
                    (int(x), int(y))
                    for x, y in re.findall(
                        r"coordinate\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]",
                        text[chip_start:mem_start],
                    )
                ]
            elif re.search(r"\bsingle_chip\s*:\s*true\b", text):
                coordinates = [(0, 0)]
            else:
                raise ValueError(
                    f"Cannot determine compute-chip layout from {cfg_path}"
                )
            spm_match = re.search(
                r"spm_wide\s*:\s*\{(?P<body>.*?)\}",
                text,
                flags=re.DOTALL,
            )
            if spm_match is None:
                raise ValueError(f"Cannot determine SPM geometry from {cfg_path}")
            spm_body = spm_match.group("body")
            bank_match = re.search(r"\bbanks\s*:\s*(\d+)", spm_body)
            size_match = re.search(r"\blength\s*:\s*(\d+)", spm_body)
            if bank_match is None or size_match is None:
                raise ValueError(f"Cannot determine SPM geometry from {cfg_path}")
            compute_bank_count = int(bank_match.group(1))
            compute_size = int(size_match.group(1))

            mem_chip_start = text.find("hemaia_mem_chip")
            mem_list_start = text.find("[", mem_chip_start)
            if mem_chip_start < 0 or mem_list_start < 0:
                raise ValueError(f"Cannot determine memory-chip layout from {cfg_path}")
            mem_chip_end = -1
            bracket_depth = 0
            for offset in range(mem_list_start, len(text)):
                if text[offset] == "[":
                    bracket_depth += 1
                elif text[offset] == "]":
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        mem_chip_end = offset + 1
                        break
            if mem_chip_end < 0:
                raise ValueError(f"Cannot determine memory-chip layout from {cfg_path}")
            mem_sizes = [
                int(size)
                for size in re.findall(
                    r"\bmem_size\s*:\s*(\d+)",
                    text[mem_chip_start:mem_chip_end],
                )
            ]

        if not coordinates:
            raise ValueError(f"No compute chips found in {cfg_path}")
        if len(set(coordinates)) != len(coordinates):
            raise ValueError(f"Duplicate compute-chip coordinates in {cfg_path}")
        if compute_bank_count < 1:
            raise ValueError(
                f"Invalid SPM bank count {compute_bank_count} in {cfg_path}"
            )
        compute_bank_bytes = 8 * compute_bank_count
        if compute_size < compute_bank_bytes or compute_size % compute_bank_bytes:
            raise ValueError(
                f"SPM size {compute_size} in {cfg_path} is not a positive multiple "
                f"of {compute_bank_count} 64-bit banks"
            )
        compute_bank_depth = compute_size // compute_bank_bytes

        mempool_bank_depth: Optional[int] = None
        if mem_sizes:
            mempool_bank_bytes = 8 * MEMPOOL_BANK_COUNT
            invalid_sizes = [
                size
                for size in mem_sizes
                if size < mempool_bank_bytes or size % mempool_bank_bytes
            ]
            if invalid_sizes:
                raise ValueError(
                    f"Memory-chip size(s) {invalid_sizes} in {cfg_path} are not positive "
                    f"multiples of {MEMPOOL_BANK_COUNT} 64-bit banks"
                )
            # The same mempool image is loaded into every memory chip.  Its safe
            # upper bound is therefore the smallest configured chip.
            mempool_bank_depth = min(mem_sizes) // mempool_bank_bytes

        return PlatformLayout(
            coordinates=coordinates,
            compute_bank_count=compute_bank_count,
            compute_bank_depth=compute_bank_depth,
            mempool_bank_count=MEMPOOL_BANK_COUNT,
            mempool_bank_depth=mempool_bank_depth,
        )

    def _clear_generated_app_images(self) -> None:
        """Remove images that ``make apps`` otherwise leaves between workloads."""
        sim_bin = self.repo_root / "target/sim/bin"
        for generated in sim_bin.glob("app_chip_*"):
            if generated.is_dir():
                shutil.rmtree(generated)
        for generated in (
            sim_bin / "mempool",
            self.repo_root / "target/sim/apps/mempool",
        ):
            if generated.is_dir():
                shutil.rmtree(generated)
        mempool_bin = self.repo_root / "target/sim/apps/mempool.bin"
        if mempool_bin.exists():
            mempool_bin.unlink()

    @staticmethod
    def _validate_hex_dir(
        directory: Path,
        bank_count: int,
        label: str,
        max_depth: Optional[int] = None,
    ) -> None:
        """Require a complete bank set with valid, in-bounds readmemh data."""
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {label} hex directory: {directory}")
        if bank_count < 1:
            raise ValueError(f"Invalid {label} bank count: {bank_count}")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"Invalid {label} bank depth: {max_depth}")

        expected = {f"bank_{idx}.hex" for idx in range(bank_count)}
        actual = {path.name for path in directory.glob("bank_*.hex") if path.is_file()}
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected {', '.join(unexpected)}")
            raise ValueError(f"Invalid {label} bank set in {directory}: {'; '.join(details)}")

        for name in sorted(expected):
            path = directory / name
            saw_data = False
            address = 0
            with path.open("r", encoding="ascii") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    if _HEX_LINE_RE.fullmatch(line) is None:
                        raise ValueError(
                            f"Invalid readmemh data in {path}:{line_number}: {line!r}"
                        )
                    if line.startswith("@"):
                        address_text = line[1:].replace("_", "")
                        try:
                            address = int(address_text, 16)
                        except ValueError as error:
                            raise ValueError(
                                f"Invalid readmemh address in {path}:{line_number}: {line!r}"
                            ) from error
                        if max_depth is not None and address >= max_depth:
                            raise ValueError(
                                f"Readmemh address 0x{address:x} in {path}:{line_number} "
                                f"exceeds {label} bank depth {max_depth}"
                            )
                        continue

                    saw_data = True
                    if max_depth is not None and address >= max_depth:
                        raise ValueError(
                            f"Readmemh data in {path}:{line_number} exceeds {label} "
                            f"bank depth {max_depth}"
                        )
                    address += 1
            if not saw_data:
                raise ValueError(f"Empty hex bank is not allowed: {path}")

    def _sim_make_args(self) -> List[str]:
        return [
            f"CFG_OVERRIDE={self.effective_cfg}",
            f"SIM_CFG={self._repo_path(self.sim_cfg)}",
            f"SIM_WITH_WAVEFORM={self.waveform}",
            f"SIM_WITH_PLL={1 if self.with_pll else 0}",
        ]

    def _sim_cfg_flag(self, name: str) -> bool:
        text = self._repo_path(self.sim_cfg).read_text()
        match = re.search(
            rf"^[ \t]*{re.escape(name)}[ \t]*[:=][ \t]*(true|false|0|1)\b",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        return match is not None and match.group(1).lower() in ("true", "1")

    def _compile_input_path(self, engine: str) -> Optional[Path]:
        if engine == "vcs":
            return self.repo_root / "target/sim/work-vcs/compile.sh"
        if engine == "vsim":
            return self.repo_root / "target/sim/work-vsim/compile.vsim.tcl"
        return None

    def _make_compile_input_relocatable(self, engine: str) -> None:
        """Make a generated compiler script relocatable and netlist-safe."""
        path = self._compile_input_path(engine)
        if path is None:
            return
        if not path.is_file():
            raise FileNotFoundError(f"{engine} preparation did not produce {path}")

        contents = path.read_text()
        if engine == "vcs":
            replacement = (
                'ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." '
                '&& pwd)"'
            )
            contents, replacements = re.subn(
                r"^ROOT=.*$", replacement, contents, count=1, flags=re.MULTILINE
            )
        else:
            replacement = (
                "set ROOT [file normalize [file join [file dirname [info script]] "
                ".. .. ..]]"
            )
            contents, replacements = re.subn(
                r"^set ROOT\s+.*$", replacement, contents, count=1, flags=re.MULTILINE
            )

        if replacements != 1:
            raise ValueError(f"Cannot locate Bender ROOT prologue in {path}")

        # The Questa Makefile appends its UART DPI source and -I flag after the
        # Bender-generated body.  SIM_MKFILE_DIR is absolute, so rewrite those
        # prefixes (and any future repo-local appended arguments) through the
        # same script-local ROOT variable as the HDL list.
        contents = contents.replace(str(self.repo_root), "$ROOT")
        if self._sim_cfg_flag("sim_with_netlist"):
            contents = self._postprocess_netlist_compile_input(engine, path, contents)
        path.write_text(contents)
        self._validate_relocatable_compile_input(engine)

    @staticmethod
    def _postprocess_netlist_compile_input(
        engine: str,
        path: Path,
        contents: str,
    ) -> str:
        """Keep the mapped chip and reject accidental compute-chip RTL.

        Selecting Bender's ``hemaia`` target solely to obtain CVA6's HeMAiA
        configuration also selects the synthesizable compute-chip hierarchy.
        A gate-level build must instead retain the narrow ``hemaia_netlist``
        source set and replace only CVA6's generic configuration package in
        the generated compiler script.
        """
        generic_config = "/core/include/config_pkg.sv"
        hemaia_config = "/core/include/config_hemaia_pkg.sv"
        generic_count = contents.count(generic_config)
        hemaia_count = contents.count(hemaia_config)
        if generic_count == 1 and hemaia_count == 0:
            contents = contents.replace(generic_config, hemaia_config, 1)
        elif generic_count != 0 or hemaia_count != 1:
            raise ValueError(
                f"Prepared {engine} netlist input has an ambiguous CVA6 config "
                f"selection in {path}: generic={generic_count}, hemaia={hemaia_count}"
            )

        required_sources = (
            "/target/rtl/bootrom/bootrom_netlist_sim/bootrom.v",
            "/target/tapeout/HeMAiAv2_tapeout/outputs/"
            "hemaia_mapped_bootrom_commented.v",
            "/hw/hemaia/hemaia_mem_system/hemaia_mem_chip.sv",
            "/target/sim/testharness/testharness.sv",
        )
        missing = [source for source in required_sources if source not in contents]
        if missing:
            raise ValueError(
                f"Prepared {engine} netlist input is missing required sources in "
                f"{path}: {missing}"
            )

        compute_rtl_sources = (
            "/target/rtl/bootrom/bootrom_chip/bootrom.sv",
            "/target/rtl/bootrom/bootrom_sim/bootrom.sv",
            "/hw/hemaia/hemaia_mem_system/hemaia_xdma.sv",
            "/hw/hemaia/hemaia_mem_system/hemaia_xdma_wrapper.sv",
            "/hw/hemaia/hemaia_mem_system/hemaia_mem_system.sv",
            "/target/rtl/src/occamy_chip.sv",
            "/target/rtl/src/hemaia.sv",
        )
        unexpected = [source for source in compute_rtl_sources if source in contents]
        if unexpected:
            raise ValueError(
                f"Prepared {engine} netlist input also selects compute-chip RTL in "
                f"{path}: {unexpected}"
            )
        return contents

    def _validate_relocatable_compile_input(self, engine: str) -> None:
        path = self._compile_input_path(engine)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing prepared {engine} compile input: {path}")
        contents = path.read_text()
        marker = "${BASH_SOURCE[0]}" if engine == "vcs" else "[info script]"
        if marker not in contents:
            raise ValueError(f"Prepared {engine} compile input is not relocatable: {path}")
        if str(self.repo_root) in contents:
            raise ValueError(f"Prepared {engine} compile input embeds {self.repo_root}: {path}")

    def _compile_dependency_state(
        self,
        engines: Iterable[str],
    ) -> Tuple[Set[Path], Set[Path]]:
        """Resolve compile inputs and include directories unavailable on this host.

        Hashing only ``compile.sh``/``compile.vsim.tcl`` is insufficient: the
        scripts name the real mapped netlist, SRAM models, D2D/PLL sources, and
        headers by path.  Step 6.1 can run without technology access, so explicit
        sources and include roots that do not exist yet are returned for deferred
        validation by step 6.2 instead of making preparation fail.
        """
        dependencies: Set[Path] = set()
        include_dirs: Set[Path] = set()
        missing_include_dirs: Set[Path] = set()
        compile_cwd = self.repo_root / "target/sim"

        def _resolve_script_path(value: str) -> Path:
            expanded = value.replace("${ROOT}", str(self.repo_root))
            expanded = expanded.replace("$ROOT", str(self.repo_root))
            candidate = Path(expanded)
            if not candidate.is_absolute():
                candidate = compile_cwd / candidate
            return candidate.resolve()

        for engine in dict.fromkeys(engines):
            script = self._compile_input_path(engine)
            if script is None:
                continue
            if not script.is_file():
                raise FileNotFoundError(f"Missing prepared {engine} compile input: {script}")
            contents = script.read_text()

            # Bender quotes every HDL source and +incdir argument in both its
            # shell and Tcl backends.  Ignore quoted compiler flags and retain
            # only paths with an HDL suffix.
            for value in re.findall(r'"([^"\n]+)"', contents):
                if value.startswith("+incdir+"):
                    include_dirs.add(_resolve_script_path(value[len("+incdir+"):]))
                    continue
                path = _resolve_script_path(value)
                if path.suffix.lower() in _HDL_SOURCE_SUFFIXES:
                    dependencies.add(path)

        # These are consumed by the simulator-link recipes rather than emitted
        # into Bender's HDL scripts, so record them explicitly as well.
        for direct_input in (
            self.repo_root / "Makefile",
            self.repo_root / "target/sim/Makefile",
            self.repo_root / "target/sim/sim.mk",
            self.repo_root / "target/sim/testharness/uartdpi/uartdpi.c",
            self.repo_root / "target/sim/testharness/uartdpi/uartdpi.h",
        ):
            dependencies.add(direct_input.resolve())

        # Included SV/V headers do not appear as compilation units.  Traverse
        # only declared include roots and only hash HDL header suffixes.
        for include_dir in include_dirs:
            if not include_dir.is_dir():
                missing_include_dirs.add(include_dir)
                continue
            for candidate in include_dir.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in _HDL_INCLUDE_SUFFIXES:
                    dependencies.add(candidate.resolve())
        return dependencies, missing_include_dirs

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
        """Build the requested shared SW fleet, bootrom, and RTL in-container."""
        targets = ("sw", "bootrom", "rtl") if self.build_sw_fleet else ("bootrom", "rtl")
        for target in targets:
            cmd = ["make", target, f"CFG_OVERRIDE={self.effective_cfg}"]
            if self.build_jobs and target == "sw":
                # `sw` parallelises through SW_JOBS: the Makefile applies -j to the sw
                # sub-make only. Set that rather than a top-level -j, which would fight
                # the sub-make's jobserver. bootrom/rtl are not -j-safe, so they stay
                # serial.
                cmd.append(f"SW_JOBS={self.build_jobs}")
            self._container(cmd)

    # -- Step 3: build apps and stage per-task directories -----------------

    def build_apps_and_stage(self, tasks: List[Dict[str, str]]) -> List[Tuple[Path, str]]:
        """Build hex inside the container, then copy them into per-task dirs.

        Returns a list of ``(task_dir, ci_name)`` tuples.
        """
        results: List[Tuple[Path, str]] = []
        sim_bin = self.repo_root / "target/sim/bin"
        layout = self._platform_layout()
        app_subdirs = [f"app_chip_{x}_{y}" for x, y in layout.coordinates]

        for idx, task in enumerate(tasks):
            # ``target/sim/apps/Makefile`` only overwrites mempool when a
            # workload produces one.  Clear all generated images first so the
            # next task cannot silently inherit the previous task's contents.
            self._clear_generated_app_images()
            make_cmd: List[str] = ["make", "apps", f"HOST_APP_TYPE={task['host_app_type']}"]
            app_arguments = [
                ("chip_type", "CHIP_TYPE"),
                ("workload", "WORKLOAD"),
                ("dev_app", "DEV_APP"),
            ]
            for key, var in app_arguments:
                val = task.get(key, "")
                # Forward the explicit ``None`` sentinel as well.  The root
                # Makefile now has nonempty workload/device defaults; omitting
                # a task's None would let those defaults leak into host-only or
                # legacy builds and compile an unrelated device application.
                if val:
                    make_cmd.append(f"{var}={val}")
            make_cmd.append(f"CFG_OVERRIDE={self.effective_cfg}")
            make_cmd.append("DEBUG_LEVEL=0")
            self._container(make_cmd)

            generated_apps = {
                path.name for path in sim_bin.glob("app_chip_*") if path.is_dir()
            }
            if generated_apps != set(app_subdirs):
                raise ValueError(
                    "Generated compute-chip images do not match the configuration: "
                    f"expected {sorted(app_subdirs)}, got {sorted(generated_apps)}"
                )
            for subdir in app_subdirs:
                self._validate_hex_dir(
                    sim_bin / subdir,
                    layout.compute_bank_count,
                    subdir,
                    layout.compute_bank_depth,
                )

            generated_mempool = sim_bin / "mempool"
            has_real_mempool = generated_mempool.is_dir()
            if has_real_mempool:
                self._validate_hex_dir(
                    generated_mempool,
                    layout.mempool_bank_count,
                    "mempool",
                    layout.mempool_bank_depth,
                )

            task_dir = self.output_dir / task_dir_name(idx, task["ci_name"])
            if task_dir.is_symlink() or task_dir.is_file():
                task_dir.unlink()
            elif task_dir.is_dir():
                shutil.rmtree(task_dir)
            bin_dest = task_dir / "bin"
            bin_dest.mkdir(parents=True, exist_ok=True)
            for subdir in app_subdirs:
                _copy_path(sim_bin / subdir, bin_dest / subdir)

            mempool_dest = bin_dest / "mempool"
            if has_real_mempool:
                _copy_path(generated_mempool, mempool_dest)
                mempool_kind = "generated"
            else:
                mempool_dest.mkdir(parents=True, exist_ok=True)
                for bank in range(layout.mempool_bank_count):
                    (mempool_dest / f"bank_{bank}.hex").write_text("0\n")
                mempool_kind = "zero_fallback"
            (mempool_dest / ".image_kind").write_text(f"{mempool_kind}\n")

            for subdir in app_subdirs:
                self._validate_hex_dir(
                    bin_dest / subdir,
                    layout.compute_bank_count,
                    subdir,
                    layout.compute_bank_depth,
                )
            self._validate_hex_dir(
                mempool_dest,
                layout.mempool_bank_count,
                "mempool",
                layout.mempool_bank_depth,
            )

            results.append((task_dir, task["ci_name"]))
        return results

    # -- Step 4: prepare + compile the simulation --------------------------

    def prepare_simulation_inputs(self, engines: Sequence[str]) -> None:
        """Generate testbench/filelist inputs without invoking an EDA compiler."""
        unique_engines = list(dict.fromkeys(engines))
        for engine in unique_engines:
            if engine not in ENGINES:
                raise ValueError(f"Unknown engine {engine!r}")

        # These Make targets do not encode every simulation-mode flag in their
        # output dependencies.  Remove only generated testbench/filelist state
        # so changing RTL -> netlist (or PLL mode) cannot reuse a stale harness.
        for generated in self._generated_simulation_files([]):
            if generated.is_file():
                generated.unlink()
        work_dirs = [
            self.repo_root / f"target/sim/work-{engine}"
            for engine in unique_engines
            if engine in ("vcs", "vsim")
        ]
        for work_dir in work_dirs:
            if work_dir.is_symlink():
                work_dir.unlink()
            elif work_dir.is_dir():
                shutil.rmtree(work_dir)
        if "vcs" in unique_engines:
            for compiled_product in (
                self.repo_root / "target/sim/AN.DB",
                self.repo_root / "target/sim/work.lib++",
                self.repo_root / "target/sim/vc_hdrs.h",
            ):
                if compiled_product.is_symlink() or compiled_product.is_file():
                    compiled_product.unlink()
                elif compiled_product.is_dir():
                    shutil.rmtree(compiled_product)
        sim_bin = self.repo_root / "target/sim/bin"
        for engine in unique_engines:
            if engine not in ("vcs", "vsim"):
                continue
            binary = sim_bin / ENGINES[engine]["binary"]
            for old_product in (
                binary,
                Path(f"{binary}.gui"),
                Path(f"{binary}.daidir"),
            ):
                if old_product.is_symlink():
                    old_product.unlink()
                elif old_product.is_dir():
                    shutil.rmtree(old_product)
                elif old_product.exists():
                    old_product.unlink()

        for engine in unique_engines:
            spec = ENGINES[engine]
            prep_target = spec["prep_target"]
            if prep_target is None:
                # Verilator has no separable preparation phase.
                continue

            print(f"Preparing relocatable {engine} simulation inputs")
            self._container(["make", prep_target, *self._sim_make_args()])
            self._make_compile_input_relocatable(engine)

    def compile_simulation(self, *, prepared_inputs: bool = False) -> None:
        """Compile only the selected engine's shared simulation model.

        ``prepared_inputs`` is used by the EDA-backend half of the split flow.
        It tells make to consume the transferred testharness and compiler script
        without consulting Bender or regenerating either from backend-local
        source state.
        """
        if self.skip_compile:
            return
        if self.spec["prep_target"]:
            self._validate_relocatable_compile_input(self.engine)

        # Do not let a binary copied from the preparation host make this target
        # look up-to-date on the EDA backend.
        sim_root = self.repo_root / "target/sim"
        binary = sim_root / "bin" / self.spec["binary"]
        if binary.is_symlink() or binary.is_file():
            binary.unlink()
        elif binary.is_dir():
            shutil.rmtree(binary)
        daidir = Path(f"{binary}.daidir")
        if daidir.is_symlink() or daidir.is_file():
            daidir.unlink()
        elif daidir.is_dir():
            shutil.rmtree(daidir)

        compile_cmd = ["make", self.spec["compile_target"], *self._sim_make_args()]
        if prepared_inputs:
            compile_cmd.append("PREPARED_SIM_INPUTS=1")
        if self.compile_jobs:
            compile_cmd.append(f"-j{self.compile_jobs}")
        if self.spec["compile_in_container"]:
            self._container(compile_cmd)
        else:
            subprocess.run(compile_cmd, cwd=self.repo_root, check=True)

        # A few EDA launch failures still let make return zero.  Surface that as
        # one compile failure instead of N opaque task-launch failures.
        if not binary.exists():
            raise RuntimeError(
                f"{self.engine} compile produced no {binary.name}: {binary} does not "
                f"exist.\nThe make target exited 0 anyway -- inspect "
                f"target/sim/work-{self.engine}/ for the real error "
                f"(e.g. {self.spec['tool'] or self.engine} failing to start)."
            )

    def stage_simulation_artifacts(self, tasks_info: List[Tuple[Path, str]]) -> None:
        """Copy the selected shared simulator artefacts to every prepared task."""
        sim_root = self.repo_root / "target/sim"
        link_large_directories = self._sim_cfg_flag("sim_with_netlist")
        for rel in self.spec["stage"]:
            src = sim_root / rel
            if not src.exists():
                raise FileNotFoundError(
                    f"Simulation compile did not produce required {self.engine} artefact: {src}"
                )
            for task_dir, _ in tasks_info:
                dst = task_dir / rel
                # Compiled libraries can be hundreds of gigabytes for a gate
                # netlist.  Keep one backend copy and use relocatable links from
                # each task rather than duplicating work-vsim/simv.daidir.
                if src.is_dir() and link_large_directories:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.is_symlink() or dst.is_file():
                        dst.unlink()
                    elif dst.is_dir():
                        shutil.rmtree(dst)
                    dst.symlink_to(
                        os.path.relpath(src, start=dst.parent),
                        target_is_directory=True,
                    )
                else:
                    _copy_path(src, dst)

    def prepare_and_compile(self, tasks_info: List[Tuple[Path, str]]) -> None:
        """Compatibility wrapper for the original all-in-one runner API."""
        if not self.skip_compile:
            self.prepare_simulation_inputs([self.engine])
            self.compile_simulation()
        self.stage_simulation_artifacts(tasks_info)

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
        run_args = list(self.spec.get("run_args", []))
        if self._sim_cfg_flag("sim_with_netlist"):
            run_args.extend(self.spec.get("netlist_run_args", []))

        def _worker(task_dir: Path, ci_name: str):
            sim_binary = task_dir / "bin" / binary_name
            # A backend rerun intentionally preserves the prepared hex images,
            # but an empty/crashed rerun must not leave the previous log looking
            # like its output.
            old_log = task_dir / "bin/sim_run.log"
            if old_log.exists():
                old_log.unlink()
            proc = None
            pgid = None
            start = time.monotonic()
            try:
                # start_new_session=True makes the child a session/group leader,
                # so all descendants share that group and can be killed as a unit.
                proc = subprocess.Popen(
                    [str(sim_binary), *run_args],
                    cwd=task_dir / "bin",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                pgid = proc.pid
                _register_pgid(pgid)
                try:
                    out_bytes, _ = proc.communicate(timeout=self.timeout_seconds)
                    elapsed = time.monotonic() - start
                    out = out_bytes.decode(errors="replace") if out_bytes else ""
                    return str(task_dir), ci_name, _sim_passed(proc.returncode, out), elapsed, out
                except subprocess.TimeoutExpired:
                    _kill_pgid(pgid)
                    out_bytes, _ = proc.communicate()
                    elapsed = time.monotonic() - start
                    partial = out_bytes.decode(errors="replace") if out_bytes else ""
                    msg = (
                        f"TIMEOUT: simulation exceeded {self.timeout_seconds}s "
                        f"({_format_duration(self.timeout_seconds)}) and was killed.\n"
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

    # -- Split-flow hand-off manifest --------------------------------------

    def _portable_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(self.repo_root))
        except ValueError:
            return str(resolved)

    def _manifest_path_to_host(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo_root / path

    def _generated_simulation_files(self, engines: Iterable[str]) -> List[Path]:
        testharness = self.repo_root / "target/sim/testharness"
        generated = [
            testharness / "testharness.sv",
            testharness / "dut.sv",
            testharness / "io_wrapper.sv",
            testharness / "util/load_binary_rtl.sv",
            testharness / "util/load_binary_mem_macro.sv",
            testharness / "util/load_binary_netlist.sv",
            testharness / "util/check_finish_rtl.sv",
            testharness / "util/check_finish_mem_macro.sv",
            testharness / "util/check_finish_netlist.sv",
        ]
        generated.extend(
            path
            for path in (self._compile_input_path(engine) for engine in engines)
            if path is not None
        )
        return generated

    def _task_hex_state(
        self,
        tasks: List[Dict[str, str]],
        tasks_info: List[Tuple[Path, str]],
    ) -> List[Dict[str, Any]]:
        if len(tasks) != len(tasks_info):
            raise ValueError(
                f"Task metadata mismatch: {len(tasks)} task(s), "
                f"{len(tasks_info)} task directories"
            )
        layout = self._platform_layout()
        app_subdirs = [f"app_chip_{x}_{y}" for x, y in layout.coordinates]
        state: List[Dict[str, Any]] = []

        for idx, (task, (task_dir, ci_name)) in enumerate(zip(tasks, tasks_info)):
            expected_dir_name = task_dir_name(idx, task["ci_name"])
            if task_dir.name != expected_dir_name:
                raise ValueError(
                    f"Expected {expected_dir_name} at manifest position {idx}, got {task_dir}"
                )
            if ci_name != task["ci_name"]:
                raise ValueError(
                    f"Task {idx} name mismatch: metadata has {ci_name!r}, "
                    f"task list has {task['ci_name']!r}"
                )
            bin_dir = task_dir / "bin"
            hex_files: Dict[str, str] = {}
            for subdir in app_subdirs:
                bank_dir = bin_dir / subdir
                self._validate_hex_dir(
                    bank_dir,
                    layout.compute_bank_count,
                    subdir,
                    layout.compute_bank_depth,
                )
                for bank in range(layout.compute_bank_count):
                    path = bank_dir / f"bank_{bank}.hex"
                    hex_files[str(path.relative_to(task_dir))] = _sha256(path)

            mempool_dir = bin_dir / "mempool"
            self._validate_hex_dir(
                mempool_dir,
                layout.mempool_bank_count,
                "mempool",
                layout.mempool_bank_depth,
            )
            for bank in range(layout.mempool_bank_count):
                path = mempool_dir / f"bank_{bank}.hex"
                hex_files[str(path.relative_to(task_dir))] = _sha256(path)

            kind_path = bin_dir / "mempool/.image_kind"
            if not kind_path.is_file():
                raise FileNotFoundError(f"Missing mempool image marker: {kind_path}")
            mempool_kind = kind_path.read_text().strip()
            if mempool_kind not in ("generated", "zero_fallback"):
                raise ValueError(f"Invalid mempool image marker in {kind_path}: {mempool_kind!r}")

            state.append({
                "index": idx,
                "ci_name": ci_name,
                "parameters": task,
                "mempool_image": mempool_kind,
                "hex_files": dict(sorted(hex_files.items())),
            })
        return state

    def _manifest_settings(self) -> Dict[str, Any]:
        """Return the complete build/profile identity for a split hand-off."""
        layout = self._platform_layout()
        return {
            "cfg": self._portable_path(self._repo_path(self.cfg)),
            "effective_cfg": self._portable_path(self._repo_path(self.effective_cfg)),
            "sim_cfg": self._portable_path(self._repo_path(self.sim_cfg)),
            "waveform": int(self.waveform),
            "with_macro": self.with_macro,
            "with_d2d": self.with_d2d,
            "with_pll": self.with_pll,
            "build_sw_fleet": self.build_sw_fleet,
            "platform": {
                "compute_coordinates": [list(coordinate) for coordinate in layout.coordinates],
                "compute_bank_count": layout.compute_bank_count,
                "compute_bank_depth": layout.compute_bank_depth,
                "mempool_bank_count": layout.mempool_bank_count,
                "mempool_bank_depth": layout.mempool_bank_depth,
            },
        }

    def write_preparation_manifest(
        self,
        tasks: List[Dict[str, str]],
        tasks_info: List[Tuple[Path, str]],
        prepared_engines: Sequence[str],
    ) -> Path:
        """Record every input required to safely resume on an EDA backend."""
        engines = list(dict.fromkeys(prepared_engines))
        for engine in engines:
            self._validate_relocatable_compile_input(engine)

        inputs: Dict[str, Dict[str, str]] = {}
        input_paths: List[Tuple[str, Path]] = [
            ("cfg", self._repo_path(self.cfg)),
            ("effective_cfg", self._repo_path(self.effective_cfg)),
            ("sim_cfg", self._repo_path(self.sim_cfg)),
        ]
        if self.task_yaml is not None:
            input_paths.append(("task_yaml", self.task_yaml))

        for name, path in input_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing preparation input {name}: {path}")
            portable_path = self._portable_path(path)
            if Path(portable_path).is_absolute():
                raise ValueError(
                    f"Split preparation input {name} must reside inside the repository: {path}"
                )
            inputs[name] = {"path": portable_path, "sha256": _sha256(path)}

        generated_files: Dict[str, str] = {}
        for path in self._generated_simulation_files(engines):
            if not path.is_file():
                raise FileNotFoundError(f"Missing generated simulation input: {path}")
            generated_files[self._portable_path(path)] = _sha256(path)

        # Record the actual files named by both prepared compiler scripts, not
        # just the scripts themselves.  Step 6.1 deliberately runs on a host
        # without the mapped netlist or technology views, so keep their paths
        # as deferred backend dependencies rather than trying to read them.
        already_recorded = {entry["path"] for entry in inputs.values()}
        compile_sources: Dict[str, str] = {}
        backend_dependencies: List[str] = []
        dependency_paths, missing_include_dirs = self._compile_dependency_state(engines)
        for path in sorted(dependency_paths):
            portable_path = self._portable_path(path)
            if portable_path in already_recorded:
                continue
            if path.is_file():
                compile_sources[portable_path] = _sha256(path)
            else:
                backend_dependencies.append(portable_path)

        manifest: Dict[str, Any] = {
            "schema_version": PREPARATION_MANIFEST_VERSION,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "settings": self._manifest_settings(),
            "prepared_engines": engines,
            "inputs": inputs,
            "generated_files": dict(sorted(generated_files.items())),
            "compile_sources": dict(sorted(compile_sources.items())),
            "backend_dependencies": sorted(backend_dependencies),
            "backend_include_dirs": sorted(
                self._portable_path(path) for path in missing_include_dirs
            ),
            "tasks": self._task_hex_state(tasks, tasks_info),
        }

        manifest_path = self.output_dir / PREPARATION_MANIFEST
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
        print(f"Preparation manifest written to {manifest_path}")
        return manifest_path

    def validate_preparation_manifest(
        self,
        tasks: List[Dict[str, str]],
    ) -> Tuple[Path, List[Tuple[Path, str]]]:
        """Validate a preparation hand-off and reconstruct existing task dirs."""
        manifest_path = self.output_dir / PREPARATION_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Preparation manifest not found: {manifest_path}; run phase 'prepare' first"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != PREPARATION_MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported preparation manifest version in {manifest_path}: "
                f"{manifest.get('schema_version')!r}"
            )

        prepared_settings = manifest.get("settings")
        if not isinstance(prepared_settings, dict):
            raise ValueError("Preparation manifest has invalid settings")

        requested_settings = {
            "cfg": self._portable_path(self._repo_path(self.cfg)),
            "sim_cfg": self._portable_path(self._repo_path(self.sim_cfg)),
            "waveform": int(self.waveform),
            "with_macro": self.with_macro,
            "with_d2d": self.with_d2d,
            "with_pll": self.with_pll,
            "build_sw_fleet": self.build_sw_fleet,
        }
        prepared_request = {
            name: prepared_settings.get(name) for name in requested_settings
        }
        if prepared_request != requested_settings:
            raise ValueError(
                "Preparation settings do not match this simulation request:\n"
                f"prepared: {prepared_request}\nrequested: {requested_settings}"
            )

        # The derived cfg may have been generated only on the preparation host.
        # Adopt its recorded repo-relative name before parsing the platform on a
        # backend (which intentionally need not have hjson installed).
        effective_cfg = prepared_settings.get("effective_cfg")
        if not isinstance(effective_cfg, str) or not effective_cfg:
            raise ValueError("Preparation manifest does not identify its effective cfg")
        if Path(effective_cfg).is_absolute():
            raise ValueError("Preparation effective cfg must be repository-relative")
        effective_cfg_path = (self.repo_root / effective_cfg).resolve()
        try:
            effective_cfg_path.relative_to(self.repo_root)
        except ValueError as error:
            raise ValueError(
                f"Preparation effective cfg escapes the repository: {effective_cfg}"
            ) from error
        self.effective_cfg = str(effective_cfg_path.relative_to(self.repo_root))
        expected_settings = self._manifest_settings()
        if prepared_settings != expected_settings:
            raise ValueError(
                "Prepared effective configuration/profile no longer matches:\n"
                f"prepared: {prepared_settings}\ncurrent: {expected_settings}"
            )

        effective_entry = manifest.get("inputs", {}).get("effective_cfg")
        if not isinstance(effective_entry, dict) or effective_entry.get("path") != effective_cfg:
            raise ValueError("Preparation manifest has inconsistent effective cfg metadata")
        if self.task_yaml is not None:
            task_yaml_entry = manifest.get("inputs", {}).get("task_yaml")
            if task_yaml_entry is None:
                raise ValueError("Preparation manifest does not identify its task YAML")
            if task_yaml_entry.get("path") != self._portable_path(self.task_yaml):
                raise ValueError(
                    "Requested task YAML differs from the preparation manifest: "
                    f"{self.task_yaml}"
                )
        if self.engine not in manifest.get("prepared_engines", []):
            raise ValueError(
                f"Engine {self.engine!r} was not prepared; available: "
                f"{manifest.get('prepared_engines', [])}"
            )

        for name, entry in manifest.get("inputs", {}).items():
            path = self._manifest_path_to_host(entry["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Prepared input {name} is missing: {path}")
            digest = _sha256(path)
            if digest != entry["sha256"]:
                raise ValueError(f"Prepared input {name} changed after preparation: {path}")

        for path_text, expected_digest in manifest.get("generated_files", {}).items():
            path = self._manifest_path_to_host(path_text)
            if not path.is_file():
                raise FileNotFoundError(f"Generated simulation input is missing: {path}")
            if _sha256(path) != expected_digest:
                raise ValueError(f"Generated simulation input changed after preparation: {path}")
        self._validate_relocatable_compile_input(self.engine)

        prepared_engines = manifest.get("prepared_engines", [])
        recorded_compile_sources = manifest.get("compile_sources")
        if not isinstance(recorded_compile_sources, dict) or not recorded_compile_sources:
            raise ValueError("Preparation manifest does not record compiler source dependencies")
        backend_dependencies = manifest.get("backend_dependencies")
        if not isinstance(backend_dependencies, list) or not all(
            isinstance(path, str) for path in backend_dependencies
        ):
            raise ValueError("Preparation manifest has invalid backend dependencies")
        backend_include_dirs = manifest.get("backend_include_dirs")
        if not isinstance(backend_include_dirs, list) or not all(
            isinstance(path, str) for path in backend_include_dirs
        ):
            raise ValueError("Preparation manifest has invalid backend include directories")

        deferred_include_roots = [
            self._manifest_path_to_host(path).resolve() for path in backend_include_dirs
        ]
        for include_dir in deferred_include_roots:
            if not include_dir.is_dir():
                raise FileNotFoundError(
                    f"Backend HDL include directory is unavailable: {include_dir}"
                )

        dependency_paths, missing_include_dirs = self._compile_dependency_state(
            prepared_engines
        )
        if missing_include_dirs:
            raise FileNotFoundError(
                "Backend HDL include directories are unavailable: "
                + ", ".join(str(path) for path in sorted(missing_include_dirs))
            )

        recorded_input_paths = {
            entry["path"] for entry in manifest.get("inputs", {}).values()
        }
        recorded_dependencies = set(recorded_compile_sources) | set(backend_dependencies)

        # Headers below an include directory which was absent during 6.1 can
        # only be discovered on the backend.  Validate that directory exists,
        # while retaining exact-set checking for every explicitly named source
        # and every include tree that was accessible during preparation.
        expected_dependencies: Set[str] = set()
        for path in dependency_paths:
            portable_path = self._portable_path(path)
            if portable_path in recorded_input_paths:
                continue
            newly_discovered_header = (
                portable_path not in recorded_dependencies
                and any(
                    path == include_root or path.is_relative_to(include_root)
                    for include_root in deferred_include_roots
                )
            )
            if not newly_discovered_header:
                expected_dependencies.add(portable_path)

        if recorded_dependencies != expected_dependencies:
            missing = sorted(expected_dependencies - recorded_dependencies)
            stale = sorted(recorded_dependencies - expected_dependencies)
            raise ValueError(
                "Prepared compiler dependency set changed after preparation: "
                f"missing manifest entries={missing}, stale manifest entries={stale}"
            )
        for path_text, expected_digest in recorded_compile_sources.items():
            path = self._manifest_path_to_host(path_text)
            if not path.is_file():
                raise FileNotFoundError(f"Prepared compiler source is missing: {path}")
            if _sha256(path) != expected_digest:
                raise ValueError(f"Prepared compiler source changed after preparation: {path}")
        for path_text in backend_dependencies:
            path = self._manifest_path_to_host(path_text)
            if not path.is_file():
                raise FileNotFoundError(f"Backend compiler source is unavailable: {path}")

        prepared_tasks = manifest.get("tasks", [])
        prepared_parameters = [entry.get("parameters") for entry in prepared_tasks]
        if prepared_parameters != tasks:
            raise ValueError("Task list changed after preparation")
        tasks_info = [
            (self.output_dir / task_dir_name(idx, task["ci_name"]), task["ci_name"])
            for idx, task in enumerate(tasks)
        ]
        current_state = self._task_hex_state(tasks, tasks_info)
        if current_state != prepared_tasks:
            raise ValueError("Staged application/mempool hex changed after preparation")

        print(f"Validated preparation manifest: {manifest_path}")
        return manifest_path, tasks_info

    def _backend_dependency_hashes(self, manifest_path: Path) -> Dict[str, str]:
        """Return hashes for explicit and include-discovered backend inputs."""
        manifest = json.loads(manifest_path.read_text())
        dependencies = manifest.get("backend_dependencies")
        include_dirs = manifest.get("backend_include_dirs")
        if not isinstance(dependencies, list) or not all(
            isinstance(path, str) for path in dependencies
        ):
            raise ValueError("Preparation manifest has invalid backend dependencies")
        if not isinstance(include_dirs, list) or not all(
            isinstance(path, str) for path in include_dirs
        ):
            raise ValueError("Preparation manifest has invalid backend include directories")

        state: Dict[str, str] = {}
        for path_text in dependencies:
            path = self._manifest_path_to_host(path_text)
            if not path.is_file():
                raise FileNotFoundError(f"Backend compiler source is unavailable: {path}")
            state[path_text] = _sha256(path)

        # An include root can be absent on the preparation machine, in which
        # case its headers cannot be listed in the manifest. Discover and hash
        # the complete header tree once it is available on the backend.
        for include_text in include_dirs:
            include_dir = self._manifest_path_to_host(include_text)
            if not include_dir.is_dir():
                raise FileNotFoundError(
                    f"Backend HDL include directory is unavailable: {include_dir}"
                )
            for path in sorted(include_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in _HDL_INCLUDE_SUFFIXES:
                    state[self._portable_path(path)] = _sha256(path)
        return state

    def snapshot_backend_dependencies(self, manifest_path: Path) -> Dict[str, str]:
        """Hash deferred backend-only compiler inputs before EDA compilation.

        These files are intentionally unavailable during the software/filelist
        preparation phase, so their digests cannot be stored in the hand-off
        manifest.  Snapshot them as soon as the backend has validated the
        manifest, then compare the same state after compilation.  In
        particular, this prevents the selected mapped netlist from being
        replaced between compile and simulation staging.
        """
        state = self._backend_dependency_hashes(manifest_path)
        print(f"Snapshotted {len(state)} backend-only compiler input(s)")
        return state

    def validate_backend_dependencies_unchanged(
        self,
        manifest_path: Path,
        expected_state: Dict[str, str],
    ) -> None:
        """Require deferred backend compiler inputs to match their snapshot."""
        current_state = self._backend_dependency_hashes(manifest_path)
        if current_state != expected_state:
            expected_paths = set(expected_state)
            current_paths = set(current_state)
            missing = sorted(expected_paths - current_paths)
            added = sorted(current_paths - expected_paths)
            changed = sorted(
                path
                for path in expected_paths & current_paths
                if expected_state[path] != current_state[path]
            )
            raise ValueError(
                "Backend compiler inputs changed during compilation: "
                f"missing={missing}, added={added}, modified={changed}"
            )
        print("Backend-only compiler inputs remained unchanged during compilation")

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

    def run(
        self,
        tasks: List[Dict[str, str]],
        *,
        phase: str = "all",
        prepare_engines: Sequence[str] = ("vcs", "vsim"),
    ) -> Path:
        """Run ``all``, preparation-only, or backend simulation-only.

        The keyword-only additions preserve every existing ``run(tasks)``
        caller.  ``prepare`` intentionally avoids the destructive repository
        reset/clean step.  It generates all software, hex, testbench, and
        compiler inputs without accessing the mapped netlist or technology
        source files; those are supplied and validated by step 6.2.
        """
        if phase not in ("all", "prepare", "simulate"):
            raise ValueError(f"Unknown phase {phase!r}; choose all, prepare, or simulate")
        if not tasks:
            raise ValueError("At least one simulation task is required")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Phase: {phase}  engine: {self.engine}  waveform: {self.waveform}  "
            f"cfg: {self.cfg}  sim_cfg: {self.sim_cfg}"
        )
        print(
            f"  (macro={self.with_macro}, d2d={self.with_d2d}, pll={self.with_pll})"
        )

        if phase == "simulate":
            manifest_path, tasks_info = self.validate_preparation_manifest(tasks)
            backend_dependency_state = self.snapshot_backend_dependencies(manifest_path)
            self._ensure_engine_available()
            _install_cleanup_handlers()
            print("[Backend 1/5] Compiling the prepared simulation model")
            self.compile_simulation(prepared_inputs=True)
            print("[Backend 2/5] Revalidating immutable preparation inputs")
            _, tasks_info = self.validate_preparation_manifest(tasks)
            self.validate_backend_dependencies_unchanged(
                manifest_path, backend_dependency_state
            )
            print("[Backend 3/5] Staging the shared simulation artefacts")
            self.stage_simulation_artifacts(tasks_info)
            print("[Backend 4/5] Running simulations")
            results, total_elapsed = self.run_simulations(tasks_info)
            print("[Backend 5/5] Writing summary")
            summary_path = self._finish_run(tasks_info, results, total_elapsed)
            return summary_path

        self.cleanup_stale_task_dirs()
        stale_manifest = self.output_dir / PREPARATION_MANIFEST
        if stale_manifest.exists():
            stale_manifest.unlink()

        if phase == "all":
            self._ensure_engine_available()
            _install_cleanup_handlers()
            if self.skip_setup:
                print("[Step 1] Skipping reset/init (skip_setup)")
            else:
                print("[Step 1] Cleaning and initialising private repos")
                self.reset_and_init()
        else:
            print("[Prepare 1/4] Preserving generated RTL and local package manifests")

        # Must run after Step 1: reset_and_init() calls ``make clean``, which wipes
        # the generated-cfg directory.
        self.effective_cfg = resolve_rtl_cfg(
            self.repo_root, self.cfg, with_pll=self.with_pll)

        if self.skip_build:
            print("[Step 2] Skipping shared SW/bootrom/RTL build (skip_build)")
            print("[Step 3] Building apps and staging per-task directories")
            tasks_info = self.build_apps_and_stage(tasks)
        elif self.build_sw_fleet:
            print("[Step 2] Building SW/bootrom/RTL")
            self.build_hw_sw()
            print("[Step 3] Building apps and staging per-task directories")
            tasks_info = self.build_apps_and_stage(tasks)
        else:
            # A clean bootrom build consumes generated platform headers.  The
            # strict selected app builds create those headers, so netlist CI
            # deliberately performs software/binary generation first.
            print("[Step 2] Building selected apps and staging task directories")
            tasks_info = self.build_apps_and_stage(tasks)
            print("[Step 3] Building bootrom/RTL for the selected hardware")
            self.build_hw_sw()

        if phase == "prepare":
            engines = list(dict.fromkeys(prepare_engines))
            if set(engines) != {"vcs", "vsim"}:
                raise ValueError(
                    "Split CI preparation must generate both VCS and Questa inputs"
                )
            print("[Prepare 4/4] Preparing VCS and Questa inputs (no EDA compile)")
            self.prepare_simulation_inputs(engines)
            return self.write_preparation_manifest(tasks, tasks_info, engines)

        if self.skip_compile:
            print("[Step 4] Staging the pre-built simulation (skip_compile)")
        else:
            print("[Step 4] Preparing and compiling the simulation")
        self.prepare_and_compile(tasks_info)

        print("[Step 5] Running simulations")
        results, total_elapsed = self.run_simulations(tasks_info)
        print("[Step 6] Writing summary")
        return self._finish_run(tasks_info, results, total_elapsed)

    def _finish_run(
        self,
        tasks_info: List[Tuple[Path, str]],
        results: Dict[str, Tuple[bool, float]],
        total_elapsed: float,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"simulation_summary_{timestamp}.md"
        self.write_summary(summary_path, tasks_info, results, total_elapsed)
        print(f"Simulation summary written to {summary_path}")

        failed = [
            ci_name
            for task_dir, ci_name in tasks_info
            if not results.get(str(task_dir), (False, 0.0))[0]
        ]
        if failed and self.fail_on_task_failure:
            raise SimulationSuiteFailed(
                f"{len(failed)} of {len(tasks_info)} simulation(s) failed; "
                f"see {summary_path}"
            )
        return summary_path
