#!/usr/bin/env python3
"""
HeMAiA tapeout sim
=============================
You can modify the HOST_APP_TYPE, CHIP_TYPE, WORKLOAD and DEV_APP variables below to select which application to run in the simulation. Then simply run this script to execute the full flow of preparing the environment, compiling the HW/SW, and running the Questasim simulation. 

Notice it will use the tapeout cfg with sim_rtl_pll.hjson, which has the PLL enabled.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Optional

CFG_NAME = "hemaia_tapeout.hjson"
SIM_CFG_NAME = "sim_rtl_with_pll.hjson"

# This one is for the 4chiplets
HOST_APP_TYPE = "offload_bingo_hw" # Options: host_only, offload_legacy, offload_bingo_hw, offload_bingo_sw
CHIP_TYPE = "multi_chip"
WORKLOAD = "gemm_seperate_load_and_compute_parallel_4chiplet_1cluster"
DEV_APP = "snax-bingo-offload"

# You can also run the single-chiplet app
# The single-chiplet app will run on the first chiplet, the others will be exit from the init.
# HOST_APP_TYPE = "offload_bingo_hw"
# CHIP_TYPE = "single_chip"
# WORKLOAD = "gemm_stacked_1cluster"
# DEV_APP = "snax-bingo-offload"
# ---------------------------------------------------------------------------
# Utility helpers
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
    """Run *command* inside *docker_image*, mounting *repo_root* at the same path."""
    docker_cmd: List[str] = [
        "podman", "run", "--rm",
        "-v", f"{repo_root}:{repo_root}",
        "-w", str(working_dir),
        docker_image,
    ]
    docker_cmd.extend(command)
    subprocess.run(docker_cmd, check=True)


def _clone_or_pull(repo_url: str, dest: Path) -> None:
    """Clone *repo_url* into *dest*, or ``git pull`` if it already exists."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_host_cmd(["git", "clone", repo_url, str(dest)])
    else:
        run_host_cmd(["git", "pull"], cwd=dest)


# ---------------------------------------------------------------------------
# Step 0 -- Reset environment (mirrors 0_reset.sh)
# ---------------------------------------------------------------------------

def step0_reset(repo_root: Path, script_dir: Path) -> None:
    """Remove vendor modules, Bender.lock and reset Bender.local."""
    # Remove vendor-specific modules
    for vendor in ("tech_cells_tsmc16", "hemaia_d2d_link", "hemaia_clk_rst_controller"):
        vendor_dir = repo_root / "hw" / "hemaia" / vendor
        if vendor_dir.exists():
            shutil.rmtree(vendor_dir)

    # Reset Bender.local entries
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

    # Remove the Bender.lock
    bender_lock = repo_root / "Bender.lock"
    if bender_lock.exists():
        bender_lock.unlink()
    # Make clean
    run_host_cmd(["make", "clean"], cwd=repo_root)

# ---------------------------------------------------------------------------
# Step 1 -- Init outside docker (mirrors 1_init_outside_docker.sh)
# ---------------------------------------------------------------------------

def _upsert_bender_entry(text: str, key: str, replacement: str) -> str:
    """Replace the line for *key* in Bender.local with *replacement*, or append it."""
    pattern = re.compile(rf"^[ \t]*{re.escape(key)}:.*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    if text and not text.endswith("\n"):
        text += "\n"
    return text + replacement + "\n"


def step1_init_outside_docker(repo_root: Path, script_dir: Path) -> None:
    """Clone/update the private vendor repos and refresh Bender.local entries."""
    # Private vendor modules
    _clone_or_pull(
        "git@github.com:IveanEx/tech_cells_tsmc16.git",
        repo_root / "hw/hemaia/tech_cells_tsmc16",
    )
    _clone_or_pull(
        "git@github.com:IveanEx/hemaia_d2d_link.git",
        repo_root / "hw/hemaia/hemaia_d2d_link",
    )
    _clone_or_pull(
        "git@github.com:IveanEx/hemaia_clk_rst_controller.git",
        repo_root / "hw/hemaia/hemaia_clk_rst_controller",
    )

    # Update Bender.local with the local paths
    bender_local = repo_root / "Bender.local"
    text = bender_local.read_text() if bender_local.exists() else ""
    text = _upsert_bender_entry(
        text, "hemaia_d2d_link",
        "  hemaia_d2d_link:    { path: hw/hemaia/hemaia_d2d_link }",
    )
    text = _upsert_bender_entry(
        text, "tech_cells_generic",
        "  tech_cells_generic:  { path: hw/hemaia/tech_cells_tsmc16 }",
    )
    text = _upsert_bender_entry(
        text, "hemaia_clk_rst_controller",
        "  hemaia_clk_rst_controller:  { path: hw/hemaia/hemaia_clk_rst_controller }",
    )
    bender_local.write_text(text)


# ---------------------------------------------------------------------------
# Step 2 -- Init inside docker (mirrors 2_init_inside_docker.sh)
# ---------------------------------------------------------------------------

def step2_init_inside_docker(
    repo_root: Path,
    script_dir: Path,
    docker_image: str,
) -> None:
    """Build SW/bootrom/RTL, generate the three testharness variants, run bender."""
    cfg_override = f"target/rtl/cfg/{CFG_NAME}"
    sim_cfg_dir = repo_root / "target/sim/cfg"

    # apps / bootrom / RTL
    run_in_container(
        repo_root, docker_image, repo_root,
        ["make", "apps",
         f"HOST_APP_TYPE={HOST_APP_TYPE}",
         f"CHIP_TYPE={CHIP_TYPE}",
         f"WORKLOAD={WORKLOAD}",
         f"DEV_APP={DEV_APP}"],
    )
    run_in_container(
        repo_root, docker_image, repo_root,
        ["make", "bootrom", f"CFG_OVERRIDE={cfg_override}"],
    )
    run_in_container(
        repo_root, docker_image, repo_root,
        ["make", "rtl", f"CFG_OVERRIDE={cfg_override}"],
    )

    sim_cfg_path = sim_cfg_dir / SIM_CFG_NAME
    run_in_container(
        repo_root, docker_image, repo_root,
        ["make", "hemaia_system_vsim_preparation", f"SIM_CFG={sim_cfg_path}"],
    )


# ---------------------------------------------------------------------------
# Step 3 -- Compile the Questasim simulation on the host
# ---------------------------------------------------------------------------

def step3_compile_vsim(repo_root: Path) -> None:
    """Compile the Questasim simulation on the host (Questasim is host-only)."""
    sim_cfg_abs = str(repo_root / "target/sim/cfg" / SIM_CFG_NAME)
    subprocess.run(
        ["make", "hemaia_system_vsim", f"SIM_CFG={sim_cfg_abs}"],
        cwd=repo_root, check=True,
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
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    repo_root = script_path.parents[2]   # tapeout -> target -> repo root
    docker_image = "ghcr.io/kuleuven-micas/hemaia:main"

    ensure_vsim_available()

    print(f"[Step 0] Resetting environment under {repo_root}")
    step0_reset(repo_root, script_dir)

    print("[Step 1] Initialising private repos (host-side)")
    step1_init_outside_docker(repo_root, script_dir)

    print("[Step 2] Compiling HW/SW and preparing testharnesses (container)")
    step2_init_inside_docker(repo_root, script_dir, docker_image)

    print("[Step 3] Compiling Questasim simulation (host)")
    step3_compile_vsim(repo_root)

    print("[Step 4] Running simulation binary")
    step4_run_simulation(repo_root)


if __name__ == "__main__":
    main()
