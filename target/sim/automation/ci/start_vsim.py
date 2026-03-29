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

import getpass
import multiprocessing
import os
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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

    Supports only the structure used by this project: a top-level ``runs:``
    key containing a sequence of named blocks, each with ``- KEY: VALUE``
    attribute entries.
    """
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

    # Parse each ``- name:`` block
    while i < len(lines):
        raw = lines[i]
        stripped = raw.lstrip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("- "):
            after = stripped[2:].strip()
            if after.endswith(":"):
                ci_name = after[:-1].strip()
                attrs: list = []
                i += 1
                while i < len(lines):
                    nl = lines[i]
                    if not nl.strip():
                        i += 1
                        continue
                    # Stop when indentation returns to the parent level
                    if (len(nl) - len(nl.lstrip())) <= (len(raw) - len(stripped)):
                        break
                    nstripped = nl.lstrip()
                    if nstripped.startswith("- "):
                        after_dash = nstripped[2:].strip()
                        if ":" in after_dash:
                            k, v = [s.strip() for s in after_dash.split(":", 1)]
                            if (v.startswith('"') and v.endswith('"')) or (
                                v.startswith("'") and v.endswith("'")
                            ):
                                v = v[1:-1]
                            attrs.append({k: v})
                    i += 1
                runs.append({ci_name: attrs})
                continue
        i += 1

    return {"runs": runs}


# ---------------------------------------------------------------------------
# Step 0 -- Parse task definitions
# ---------------------------------------------------------------------------

def step0_parse_tasks(task_yaml: Path) -> List[Dict[str, str]]:
    """Parse CI tasks from *task_yaml*.

    Expected YAML structure::

        runs:
          - ci_app_name:
            - HOST_APP_TYPE: offload_bingo_sw
            - CHIP_TYPE: single_chip
            - WORKLOAD: gemm_tiled
            - DEV_APP: snax-bingo-offload
    """
    if yaml is not None:
        with open(task_yaml, "r") as f:
            param_data = yaml.safe_load(f)
    else:
        param_data = _simple_task_yaml_loader(task_yaml)

    tasks: List[Dict[str, str]] = []
    for app_entry in param_data.get("runs", []):
        ci_app_name = str(list(app_entry.keys())[0])
        attributes = app_entry[ci_app_name]
        attr_dict: Dict[str, str] = {}
        for attr in attributes:
            attr_dict.update(attr)

        if "HOST_APP_TYPE" not in attr_dict:
            raise ValueError(f"Task {ci_app_name} is missing 'HOST_APP_TYPE'")

        tasks.append({
            "host_app_type": str(attr_dict.get("HOST_APP_TYPE", "")),
            "chip_type":     str(attr_dict.get("CHIP_TYPE", "")),
            "workload":      str(attr_dict.get("WORKLOAD", "")),
            "dev_app":       str(attr_dict.get("DEV_APP", "")),
            "ci_name":       ci_app_name,
        })
    return tasks


# ---------------------------------------------------------------------------
# Step 1 -- Reset environment
# ---------------------------------------------------------------------------

def step1_reset(repo_root: Path) -> None:
    """Run ``0_reset.sh`` to restore a clean state before the flow starts."""
    run_host_script(repo_root / "target/tapeout/0_reset.sh")


# ---------------------------------------------------------------------------
# Step 2 -- Initialise private repos (host-side)
# ---------------------------------------------------------------------------

def step2_init_private_hemaia_repos(repo_root: Path) -> None:
    """Run ``1_init_outside_docker.sh`` with D2D and macro support enabled."""
    run_host_script(
        repo_root / "target/tapeout/1_init_outside_docker.sh",
        script_args=["--pll=0", "--d2d=1", "--macro=1"],
    )


# ---------------------------------------------------------------------------
# Step 3 -- Compile HW/SW inside the container
# ---------------------------------------------------------------------------

def step3_compile_hemaia_sw_rtl(repo_root: Path, docker_image: str) -> None:
    """Clean, then build SW, bootrom, and RTL inside the container."""
    ci_cfg = "target/rtl/cfg/hemaia_tapeout_sim.hjson"

    # make sw
    run_in_container(repo_root, docker_image, repo_root, ["make", "sw", f"CFG_OVERRIDE={ci_cfg}"])
    # make bootrom
    run_in_container(repo_root, docker_image, repo_root, ["make", "bootrom", f"CFG_OVERRIDE={ci_cfg}"])
    # make rtl
    run_in_container(repo_root, docker_image, repo_root, ["make", "rtl", f"CFG_OVERRIDE={ci_cfg}"])


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
    sim_cfg: str = "target/rtl/cfg/sim_rtl.hjson",
) -> None:
    """Generate filelists in-container, compile on host, distribute artefacts."""
    sim_cfg_abs = str(repo_root / sim_cfg)

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


# ---------------------------------------------------------------------------
# Step 6 -- Run simulations concurrently
# ---------------------------------------------------------------------------

def step6_run_simulations(tasks_info: List[Tuple[Path, str]]) -> Dict[str, bool]:
    """Execute simulations in parallel and return a pass/fail map."""
    results: Dict[str, bool] = {}

    def _worker(task_dir: Path, ci_name: str):
        sim_binary = task_dir / "bin" / "occamy_chip.vsim"
        try:
            proc = subprocess.run(
                [str(sim_binary)],
                cwd=task_dir / "bin",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            out = proc.stdout.decode(errors="replace") if proc.stdout else ""
            return str(task_dir), ci_name, proc.returncode == 0, out
        except Exception:
            return str(task_dir), ci_name, False, traceback.format_exc()

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_to_task = {}
        for td, cn in tasks_info:
            print(f"Starting simulation for {cn} ({td})")
            future_to_task[executor.submit(_worker, td, cn)] = (td, cn)

        for future in as_completed(future_to_task):
            td, cn = future_to_task[future]
            try:
                task_dir_str, ci_name, passed, output = future.result()
                results[task_dir_str] = passed
                status = "PASS" if passed else "FAIL"
                print(f"{ci_name}: {status}")
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
    results: Dict[str, bool],
) -> None:
    """Write a Markdown report listing each task's pass/fail status."""
    with summary_path.open("w") as f:
        f.write("# Simulation Summary\n\n")
        f.write(f"Run by **{getpass.getuser()}** at **{datetime.now():%Y-%m-%d %H:%M:%S}**\n\n")
        for idx, (task_dir, ci_name) in enumerate(tasks_info):
            passed = results.get(str(task_dir), False)
            status = "PASS" if passed else "FAIL"
            f.write(f"- **task_{idx}** ({ci_name}) \u2014 **{status}**\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[4]          # target/sim/automation/ci -> repo root
    task_base_dir = script_path.parent
    docker_image = "ghcr.io/kuleuven-micas/hemaia:main"
    task_yaml = script_path.parent / "task_vsim.yaml"
    sim_cfg =  "target/rtl/cfg/sim_rtl.hjson"


    if not task_yaml.exists():
        raise FileNotFoundError(f"Task YAML file {task_yaml} does not exist")
    # Clean the repo
    run_in_container(repo_root, docker_image, repo_root, ["make", "clean"])
    
    # Step 0: Parse task definitions from YAML
    tasks = step0_parse_tasks(task_yaml)

    # Step 1: Reset environment to a clean state
    step1_reset(repo_root)

    # Step 2: Initialise private repos on the host
    step2_init_private_hemaia_repos(repo_root)

    # Step 3: Compile HW and SW inside the container
    step3_compile_hemaia_sw_rtl(repo_root, docker_image)

    # Step 4: Build application hex files and create per-task directories
    tasks_info = step4_build_apps_and_prepare_tasks(repo_root, docker_image, tasks, task_base_dir)

    # Step 5: Prepare and compile the Questasim testharness
    step5_prepare_testharness(repo_root, docker_image, tasks_info, sim_cfg=sim_cfg)

    # Step 6: Run all simulations concurrently
    results = step6_run_simulations(tasks_info)

    # Step 7: Write Markdown summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = script_path.parent / f"simulation_summary_{timestamp}.md"
    write_summary(summary_path, tasks_info, results)
    print(f"Simulation summary written to {summary_path}")


if __name__ == "__main__":
    main()
