#!/usr/bin/env python3
"""
HeMAiA CI automation using Questasim
====================================

This script orchestrates the build and simulation flow for the HeMAiA
project when using the proprietary Questasim simulator.  The previous
implementation relied on Verilator and ran entirely inside a Docker
container.  The new flow requires a mixture of host-side and container
operations because some tools (Questasim and proprietary code download
scripts) are only available on the host machine while the hardware
generation happens inside a pre-built Docker image.

The sequence performed by this script is:

1. **Initialisation outside Docker.**  The script runs
   ``target/tapeout/1_init_outside_docker.sh`` from the repository
   root.  This script is responsible for pulling proprietary code and
   must run on the host machine because it relies on credentials and
   tools not baked into the container.

2. **Initialisation inside Docker.**  A container based on
   ``ghcr.io/kuleuven-micas/snax@sha256:4ff37cad4e85d6a898cda3232ee04a1210833eb4618d1f1fd183201c03c4c57c``
   (or ``podman`` equivalent) is launched with the repository mounted at
   exactly the same path as on the host.  Inside this container the
   script runs ``target/tapeout/2_init_inside_docker.sh`` to generate
   the RTL and build infrastructure.  Keeping the mount point
   identical ensures that absolute paths embedded in the generated
   files continue to work.

3. **Application build.**  The YAML file ``task.yaml`` (located in
   the same directory as this script) lists CI tasks under the key
   ``runs``.  Each entry contains a ``host_app`` and optionally a
   ``device_app``.  For each run the script invokes ``make apps``
   inside the container to build the corresponding hex files using
   ``HOST_APP=<host_app>`` and ``DEVICE_APP=<device_app>`` when
   provided.  After each invocation the newly generated directories
   ``app_chip_0_0``, ``app_chip_0_1``, ``app_chip_1_0`` and
   ``app_chip_1_1`` from ``target/sim/bin`` are copied into a unique
   task folder ``task_<n>/bin`` under ``target/sim/automation/ci``.

4. **Simulation preparation.**  Still inside the container the
   script calls ``make hemaia_system_vsim_preparation`` from the
   repository root to build the Questasim file lists.

5. **Simulation compile.**  On the host machine the script calls
   ``make hemaia_system_vsim`` in the repository root.  This step
   compiles the Questasim simulation and produces two important
   directories: ``target/sim/bin`` (containing ``occamy_chip.sim`` and
   ancillary scripts) and ``target/sim/work-vsim`` (Questasim work
   library).  These directories are copied into each previously
   created task folder so that simulations can run independently in
   parallel.

6. **Simulation run.**  Using a thread pool, the script executes
   ``bin/occamy_chip.sim`` in each task folder concurrently.  The
   return code of each process is recorded to determine whether the
   test passed (return code 0) or failed.

7. **Summary generation.**  A Markdown file ``simulation_summary.md``
   is written in the same directory as this script.  It lists each
   task along with its host and device apps and reports whether the
   simulation succeeded.  The file can be published as an artifact for
   CI systems.

To avoid the need for an external YAML library (PyYAML) this script
includes a minimal parser for the specific ``task.yaml`` format used in
this project.  Machines without PyYAML installed can therefore run
this automation without additional dependencies.
"""

from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from typing import Dict, List, Tuple, Optional
import traceback


def run_host_script(script_path: Path) -> None:
    """Execute a shell script on the host.

    Parameters
    ----------
    script_path:
        Absolute path to the script to run.  The script must be
        executable; if not, it will be invoked with ``bash``.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Host script {script_path} does not exist")
    # Use bash explicitly to ensure scripts with non-executable flags still run
    subprocess.run(["bash", str(script_path)], check=True, cwd=script_path.parent)


def run_in_container(repo_root: Path, docker_image: str, working_dir: Path, command: List[str]) -> None:
    """Run a command inside the provided container image.

    This helper mounts the repository root into the container at the
    same absolute path and sets the working directory accordingly.  It
    executes the provided command list and waits for completion.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root on the host.
    docker_image:
        Full name (including tag or digest) of the container image to run.
    working_dir:
        Directory inside the mounted volume where the command should be
        executed.
    command:
        List of command arguments to run inside the container.
    """
    # Build the full run command.  We use '--rm' to clean up containers
    # immediately and '-v' to mount the repository at the identical
    # location.  '-w' sets the working directory within the container.
    docker_cmd: List[str] = [
        "podman",
        "run",
        "--rm",
        "-v",
        f"{repo_root}:{repo_root}",
        "-w",
        str(working_dir),
        docker_image,
    ]
    docker_cmd.extend(command)
    subprocess.run(docker_cmd, check=True)


def parse_tasks(task_yaml: Path) -> List[Dict[str, Optional[str]]]:
    """Parse the CI tasks from a YAML file without using an external library.

    The expected YAML format is a top-level ``runs`` key containing a
    list where each item is a dictionary keyed by an ``app_*`` name.
    Each dictionary value is a list of single-entry dictionaries
    specifying ``host_app`` and optionally ``device_app``.  Since this
    format is simple and regular, we parse it by examining lines rather
    than using a third-party parser.

    Parameters
    ----------
    task_yaml:
        Path to the YAML file describing the tasks.

    Returns
    -------
    List[Dict[str, Optional[str]]]
        A list of task dictionaries with keys ``host_app`` and
        ``device_app`` (``None`` if omitted or set to ``null``/``None`` in
        the YAML).
    """
    tasks: List[Dict[str, Optional[str]]] = []
    current_host: Optional[str] = None
    current_device: Optional[str] = None
    with task_yaml.open("r") as f:
        for line in f:
            stripped = line.strip()
            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                continue
            # Skip the top-level 'runs:' key
            if stripped.startswith("runs:"):
                continue
            # When a new app entry begins, flush the previous task
            if stripped.startswith("- app_"):
                if current_host is not None:
                    tasks.append(
                        {"host_app": current_host, "device_app": current_device}
                    )
                # Reset for the next entry
                current_host = None
                current_device = None
                continue
            # Parse host_app assignment
            if stripped.startswith("- host_app:"):
                parts = stripped.split(":", 1)
                current_host = parts[1].strip()
                continue
            # Parse device_app assignment
            if stripped.startswith("- device_app:"):
                parts = stripped.split(":", 1)
                value = parts[1].strip()
                # Interpret 'null', 'none' or empty value as None
                if value.lower() in ("null", "none", ""):
                    current_device = None
                else:
                    current_device = value
                continue
    # Append the last parsed task if present
    if current_host is not None:
        tasks.append({"host_app": current_host, "device_app": current_device})
    # Validate tasks
    for t in tasks:
        if t.get("host_app") is None:
            raise ValueError(
                "host_app must be specified for every task in the configuration"
            )
    return tasks


def build_apps(
    repo_root: Path,
    docker_image: str,
    tasks: List[Dict[str, Optional[str]]],
    task_base_dir: Path,
) -> List[Tuple[Path, str, Optional[str]]]:
    """Build application hex files inside the container and prepare task directories.

    For each task entry this function runs ``make apps`` inside the
    container with the appropriate variables.  After each build the
    newly generated hex directories are copied into a unique task folder
    under ``task_base_dir``.  The function returns a list of tuples
    containing the task directory, host_app and device_app for further
    processing.
    """
    results: List[Tuple[Path, str, Optional[str]]] = []
    for idx, task in enumerate(tasks):
        host_app = task["host_app"]
        device_app = task.get("device_app")
        # Build the application inside the container
        make_cmd: List[str] = ["make", "apps", f"HOST_APP={host_app}"]
        # If device_app is None or explicitly "null", provide an empty
        # assignment so the variable exists in the make invocation
        if device_app not in [None, "null", "None"]:
            make_cmd.append(f"DEVICE_APP={device_app}")
        else:
            make_cmd.append("DEVICE_APP=")
        # Execute make inside the container at the repository root
        run_in_container(repo_root, docker_image, repo_root, make_cmd)
        # Prepare task directory
        task_dir = task_base_dir / f"task_{idx}"
        bin_dest = task_dir / "bin"
        bin_dest.mkdir(parents=True, exist_ok=True)
        # Copy the generated app directories from target/sim/bin
        sim_bin = repo_root / "target/sim/bin"
        for subdir in ["app_chip_0_0", "app_chip_0_1", "app_chip_1_0", "app_chip_1_1", "mempool"]:
            src = sim_bin / subdir
            if src.exists():
                dst = bin_dest / subdir
                # Remove any existing directory to ensure fresh contents
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        results.append((task_dir, host_app, device_app))
    return results


def prepare_and_copy_sim(repo_root: Path, tasks_info: List[Tuple[Path, str, Optional[str]]]) -> None:
    """Compile the Questasim simulation and copy outputs to each task directory."""
    # After make hemaia_system_vsim_preparation inside the container, we
    # compile on the host.  Running make hemaia_system_vsim outside
    # generates target/sim/bin and target/sim/work-vsim.
    subprocess.run(["make", "hemaia_system_vsim"], cwd=repo_root, check=True)
    sim_bin_src = repo_root / "target/sim/bin"
    work_vsim_src = repo_root / "target/sim/work-vsim"
    for task_dir, _, _ in tasks_info:
        for src in [sim_bin_src, work_vsim_src]:
            if not src.exists():
                continue
            dst = task_dir / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def run_simulations(tasks_info: List[Tuple[Path, str, Optional[str]]]) -> Dict[str, bool]:
    """Run the compiled simulations in parallel and record pass/fail per task.

    Returns
    -------
    Dict[str, bool]
        Mapping of task directory string to whether the simulation
        returned exit code 0.
    """
    results: Dict[str, bool] = {}

    def worker(task_dir: Path, host_app: str, device_app: Optional[str]) -> None:
        sim_binary = task_dir / "bin" / "occamy_chip.vsim"
        # The simulation must be run from within the bin directory so that
        # relative paths (e.g. work-vsim) are resolved correctly.
        try:
            print(f"Running simulation with HOST_APP={host_app}, DEVICE_APP={device_app}")
            result = subprocess.run(
                [str(sim_binary)],
                cwd=task_dir / "bin",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            print(f"Simulation with HOST_APP={host_app}, DEVICE_APP={device_app} is finished with retval of {result.stdout.decode()}")
            passed = result.returncode == 0
        except Exception:
            passed = False
            print(f"Simulation in {task_dir} failed with exception:")
            traceback.print_exc()
        results[str(task_dir)] = passed

    # Use a ThreadPoolExecutor with as many workers as logical CPUs
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(worker, td, ha, da) for td, ha, da in tasks_info]
        # Ensure all tasks complete
        for future in futures:
            future.result()
    return results


def write_summary(
    summary_path: Path,
    tasks_info: List[Tuple[Path, str, Optional[str]]],
    results: Dict[str, bool],
) -> None:
    """Write a Markdown summary of the simulation results."""
    with summary_path.open("w") as f:
        f.write("# Simulation Summary\n\n")
        for idx, (task_dir, host_app, device_app) in enumerate(tasks_info):
            key = str(task_dir)
            passed = results.get(key, False)
            status = "PASS" if passed else "FAIL"
            f.write(
                f"- **task_{idx}** (HOST_APP={host_app}, DEVICE_APP={device_app}) \u2014 **{status}**\n"
            )


def main() -> None:
    # Determine repository root and base directories
    script_path = Path(__file__).resolve()
    # The script resides in target/sim/automation/ci; we ascend four levels to reach the repo root
    repo_root = script_path.parents[4]
    task_base_dir = script_path.parent
    # Image digest for the SNAX toolchain
    docker_image = (
        "ghcr.io/kuleuven-micas/snax@sha256:4ff37cad4e85d6a898cda3232ee04a1210833eb4618d1f1fd183201c03c4c57c"
    )
    # Paths to initialisation scripts
    init_outside = repo_root / "target/tapeout/1_init_outside_docker.sh"
    init_inside = repo_root / "target/tapeout/2_init_inside_docker.sh"
    # Run initialisation outside the container (step 1)
    run_host_script(init_outside)
    # Run initialisation inside the container (step 2)
    run_in_container(repo_root, docker_image, repo_root, ["bash", str(init_inside)])
    # Parse tasks (step 3)
    task_yaml = script_path.parent / "task.yaml"
    tasks = parse_tasks(task_yaml)
    # Build applications and prepare task directories (step 3)
    tasks_info = build_apps(repo_root, docker_image, tasks, task_base_dir)
    # Prepare Questasim filelists inside the container (step 4)
    run_in_container(
        repo_root,
        docker_image,
        repo_root,
        ["make", "hemaia_system_vsim_preparation", "SIM_WITH_WAVEFORM=0"],
    )
    # Compile Questasim simulation outside the container and copy artefacts (step 5)
    prepare_and_copy_sim(repo_root, tasks_info)
    # Run simulations concurrently (step 6)
    results = run_simulations(tasks_info)
    # Write summary (step 7)
    summary_path = script_path.parent / "simulation_summary.md"
    write_summary(summary_path, tasks_info, results)
    print(f"Simulation summary written to {summary_path}")


if __name__ == "__main__":
    main()
