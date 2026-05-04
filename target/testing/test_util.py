"""
Shared helpers for the HeMAiA tapeout sim drivers.

The three entrypoints in this directory (single-chiplet / multi-chiplet /
tapeout) follow the same four-step backbone:

  Step 0: reset the environment (mirrors ``../0_reset.sh``)
  Step 1: clone/refresh the private vendor repos (mirrors ``../1_init_outside_docker.sh``)
  Step 2: build SW/RTL inside the build container (mirrors ``../2_init_inside_docker.sh``)
  Step 3: compile the Questasim simulation on the host
  Step 4: run the compiled vsim binary and report pass/fail

The differences between flows are captured by a small set of parameters
(``cfg_name``, ``sim_cfg_name``, ``with_macro``, ``with_d2d``, ``with_pll``,
plus the SW workload knobs). ``run_sim_flow`` glues the steps together so
each entrypoint is a one-liner.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Optional

DEFAULT_DOCKER_IMAGE = "ghcr.io/kuleuven-micas/hemaia:main"

VENDOR_MODULES = (
    "tech_cells_tsmc16",
    "hemaia_d2d_link",
    "hemaia_clk_rst_controller",
)

# Helper that kills stuck bender/git fetch trees and clears NFS silly-rename
# files inside .bender/ so that ``make clean-bender`` can succeed.
UNSTICK_BENDER_SCRIPT = (
    Path(__file__).resolve().parents[3] / "util" / "unstick_bender.py"
)


# ---------------------------------------------------------------------------
# Generic command helpers
# ---------------------------------------------------------------------------

def ensure_vsim_available() -> None:
    """Abort early if ``vsim`` is not on PATH (EDA env not sourced)."""
    if shutil.which("vsim") is None:
        print(
            "ERROR: 'vsim' not found on PATH.\n"
            "Please source the EDA setup script and re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)


def run_host_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """Execute *cmd* on the host."""
    subprocess.run(cmd, cwd=cwd, check=True)


def run_in_container(
    repo_root: Path,
    docker_image: str,
    working_dir: Path,
    command: List[str],
) -> None:
    """Run *command* inside *docker_image*, mounting *repo_root* at the same path.

    Prefers ``podman`` when it is available on PATH. Otherwise falls back to
    ``apptainer exec`` with ``docker://<image>``."""
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


def _clone_or_pull(repo_url: str, dest: Path) -> None:
    """Clone *repo_url* into *dest*, or ``git pull`` if it already exists."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_host_cmd(["git", "clone", repo_url, str(dest)])
    else:
        run_host_cmd(["git", "pull"], cwd=dest)


def _upsert_bender_entry(text: str, key: str, replacement: str) -> str:
    """Replace the line for *key* in Bender.local with *replacement*, or append it."""
    pattern = re.compile(rf"^[ \t]*{re.escape(key)}:.*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    if text and not text.endswith("\n"):
        text += "\n"
    return text + replacement + "\n"


# ---------------------------------------------------------------------------
# Step 0 -- Reset environment (mirrors ../0_reset.sh)
# ---------------------------------------------------------------------------

def step0_reset(repo_root: Path) -> None:
    """Remove vendor modules, drop their Bender.local entries, restore the
    open-source ``tech_cells_generic`` entry, delete Bender.lock, and run
    ``make clean``."""
    for vendor in VENDOR_MODULES:
        vendor_dir = repo_root / "hw" / "hemaia" / vendor
        if vendor_dir.exists():
            shutil.rmtree(vendor_dir)

    bender_local = repo_root / "Bender.local"
    if bender_local.exists():
        new_lines: List[str] = []
        for line in bender_local.read_text().splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("hemaia_d2d_link:"):
                continue
            if stripped.startswith("hemaia_clk_rst_controller:"):
                continue
            if stripped.startswith("tech_cells_generic:"):
                indent = line[: len(line) - len(stripped)]
                new_lines.append(
                    f"{indent}tech_cells_generic: "
                    "{ git: https://github.com/KULeuven-MICAS/tech_cells_generic.git, rev    : master }\n"
                )
                continue
            new_lines.append(line)
        bender_local.write_text("".join(new_lines))

    bender_lock = repo_root / "Bender.lock"
    if bender_lock.exists():
        bender_lock.unlink()

    _make_clean_with_unstick(repo_root)


def _make_clean_with_unstick(repo_root: Path) -> None:
    """Run ``make clean``; if it fails (typically a stuck .bender/ on NFS),
    run ``util/unstick_bender.py --purge`` and retry once."""
    try:
        run_host_cmd(["make", "clean"], cwd=repo_root)
        return
    except subprocess.CalledProcessError:
        print("WARNING: 'make clean' failed; running unstick_bender --purge "
              "and retrying.", file=sys.stderr)

    if UNSTICK_BENDER_SCRIPT.exists():
        subprocess.run(
            [sys.executable, str(UNSTICK_BENDER_SCRIPT),
             "--yes", "--purge", "--repo-root", str(repo_root)],
            cwd=repo_root, check=False,
        )
    run_host_cmd(["make", "clean"], cwd=repo_root)


# ---------------------------------------------------------------------------
# Step 1 -- Init outside docker (mirrors ../1_init_outside_docker.sh)
# ---------------------------------------------------------------------------

def step1_init_outside_docker(
    repo_root: Path,
    *,
    with_macro: bool,
    with_d2d: bool,
    with_pll: bool,
) -> None:
    """Clone the private vendor repos enabled for this flow and refresh Bender.local."""
    if with_macro:
        _clone_or_pull(
            "git@github.com:IveanEx/tech_cells_tsmc16.git",
            repo_root / "hw/hemaia/tech_cells_tsmc16",
        )
    if with_d2d:
        _clone_or_pull(
            "git@github.com:IveanEx/hemaia_d2d_link.git",
            repo_root / "hw/hemaia/hemaia_d2d_link",
        )
    if with_pll:
        _clone_or_pull(
            "git@github.com:IveanEx/hemaia_clk_rst_controller.git",
            repo_root / "hw/hemaia/hemaia_clk_rst_controller",
        )

    bender_local = repo_root / "Bender.local"
    text = bender_local.read_text() if bender_local.exists() else ""
    if with_macro:
        text = _upsert_bender_entry(
            text, "tech_cells_generic",
            "  tech_cells_generic:  { path: hw/hemaia/tech_cells_tsmc16 }",
        )
    if with_d2d:
        text = _upsert_bender_entry(
            text, "hemaia_d2d_link",
            "  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }",
        )
    if with_pll:
        text = _upsert_bender_entry(
            text, "hemaia_clk_rst_controller",
            "  hemaia_clk_rst_controller:  { path: hw/hemaia/hemaia_clk_rst_controller }",
        )
    bender_local.write_text(text)


# ---------------------------------------------------------------------------
# Step 2 -- Init inside docker (mirrors ../2_init_inside_docker.sh)
# ---------------------------------------------------------------------------

def step2_init_inside_docker(
    repo_root: Path,
    docker_image: str,
    *,
    cfg_name: str,
    sim_cfg_name: str,
    host_app_type: str,
    chip_type: str,
    workload: str,
    dev_app: str,
) -> None:
    """Build apps / bootrom / RTL and prepare the testharness inside the container."""
    cfg_override = f"target/rtl/cfg/{cfg_name}"
    sim_cfg_path = repo_root / "target/sim/cfg" / sim_cfg_name

    run_in_container(
        repo_root, docker_image, repo_root,
        ["make", "apps",
         f"CFG_OVERRIDE={cfg_override}",
         f"HOST_APP_TYPE={host_app_type}",
         f"CHIP_TYPE={chip_type}",
         f"WORKLOAD={workload}",
         f"DEV_APP={dev_app}"],
    )
    run_in_container(
        repo_root, docker_image, repo_root,
        ["make", "bootrom", f"CFG_OVERRIDE={cfg_override}"],
    )
    run_in_container(
        repo_root, docker_image, repo_root,
        ["make", "rtl", f"CFG_OVERRIDE={cfg_override}"],
    )
    run_in_container(
        repo_root, docker_image, repo_root,
        ["make", "hemaia_system_vsim_preparation", f"SIM_CFG={sim_cfg_path}"],
    )


# ---------------------------------------------------------------------------
# Step 3 -- Compile the Questasim simulation on the host
# ---------------------------------------------------------------------------

def step3_compile_vsim(repo_root: Path, sim_cfg_name: str) -> None:
    """Compile the Questasim simulation on the host (Questasim is host-only)."""
    sim_cfg_abs = str(repo_root / "target/sim/cfg" / sim_cfg_name)
    subprocess.run(
        ["make", "hemaia_system_vsim", f"SIM_CFG={sim_cfg_abs}"],
        cwd=repo_root,
        check=True,
    )


# ---------------------------------------------------------------------------
# Step 4 -- Run the simulation binary
# ---------------------------------------------------------------------------

def step4_run_simulation(repo_root: Path) -> bool:
    """Run the compiled ``occamy_chip.vsim`` binary and report pass/fail."""
    sim_bin_dir = repo_root / "target/sim/bin"
    sim_binary = sim_bin_dir / "occamy_chip.vsim"
    if not sim_binary.exists():
        raise FileNotFoundError(f"Simulation binary {sim_binary} does not exist")

    print(f"Starting simulation: {sim_binary}")
    try:
        proc = subprocess.run(
            [str(sim_binary)],
            cwd=sim_bin_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = proc.stdout.decode(errors="replace") if proc.stdout else ""
        passed = proc.returncode == 0
    except Exception:
        output = traceback.format_exc()
        passed = False

    status = "PASS" if passed else "FAIL"
    print(f"Simulation: {status}")
    if output:
        for line in output.strip().splitlines()[-10:]:
            print(line)
    return passed


# ---------------------------------------------------------------------------
# Repo-root resolution and full-flow driver
# ---------------------------------------------------------------------------

def resolve_repo_root(script_path: Path) -> Path:
    """Resolve the HeMAiA repo root from a script under ``target/tapeout/testing``."""
    # testing -> tapeout -> target -> repo root
    return script_path.resolve().parents[3]


def run_sim_flow(
    *,
    cfg_name: str,
    sim_cfg_name: str,
    with_macro: bool,
    with_d2d: bool,
    with_pll: bool,
    host_app_type: str,
    chip_type: str,
    workload: str,
    dev_app: str,
    script_path: Path,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
) -> None:
    """Execute the full reset → init → build → compile → run flow."""
    repo_root = resolve_repo_root(script_path)

    ensure_vsim_available()

    print(f"[Step 0] Resetting environment under {repo_root}")
    step0_reset(repo_root)

    print("[Step 1] Initialising private repos (host-side)")
    step1_init_outside_docker(
        repo_root,
        with_macro=with_macro,
        with_d2d=with_d2d,
        with_pll=with_pll,
    )

    print("[Step 2] Compiling HW/SW and preparing testharness (container)")
    step2_init_inside_docker(
        repo_root,
        docker_image,
        cfg_name=cfg_name,
        sim_cfg_name=sim_cfg_name,
        host_app_type=host_app_type,
        chip_type=chip_type,
        workload=workload,
        dev_app=dev_app,
    )

    print("[Step 3] Compiling Questasim simulation (host)")
    step3_compile_vsim(repo_root, sim_cfg_name)

    print("[Step 4] Running simulation binary")
    step4_run_simulation(repo_root)
