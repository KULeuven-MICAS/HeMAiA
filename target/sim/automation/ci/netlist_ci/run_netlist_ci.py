#!/usr/bin/env python3
"""Prepare or run the fixed HeMAiA gate-level CI profiles.

Netlist hardware is not configurable at run time.  ``--hardware`` selects one
of the four checked-in tapeout configurations and its categorized local-CI task
list; all four configurations describe the physical 128 KiB, 16-bank SRAM.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT.parents[5]  # target/sim/automation/ci/netlist_ci -> repo root
sys.path.insert(0, str(_REPO_ROOT / "util" / "automation_scripts"))

from hemaia_sim_runner import (  # noqa: E402
    HeMAiASimRunner,
    SimulationSuiteFailed,
    parse_tasks,
)

SIM_CFG = "target/sim/cfg/sim_netlist.hjson"
MAIN_MEMORY_BYTES = 128 * 1024
MAIN_MEMORY_BANKS = 16
MAIN_MEMORY_WORDS_PER_BANK = 1024
MAIN_MEMORY_WORD_BITS = 64
DEFAULT_NETLIST_SIM_JOBS = 1


@dataclass(frozen=True)
class HardwareProfile:
    cfg: str
    task_yaml: str
    clusters: tuple[str, ...]


HARDWARE_PROFILES = {
    "1c": HardwareProfile(
        cfg="target/rtl/cfg/hemaia_tapeout.hjson",
        task_yaml="target/sim/automation/ci/local_ci/tapeout/task_local_ci.yaml",
        clusters=("snax_versacore_to_cluster",),
    ),
    "1c_simd": HardwareProfile(
        cfg="target/rtl/cfg/hemaia_tapeout_1c_simd.hjson",
        task_yaml=(
            "target/sim/automation/ci/local_ci/tapeout_1c_simd/"
            "task_local_ci.yaml"
        ),
        clusters=("snax_versacore_to_simd_cluster",),
    ),
    "2c": HardwareProfile(
        cfg="target/rtl/cfg/hemaia_tapeout_2c.hjson",
        task_yaml="target/sim/automation/ci/local_ci/tapeout_2c/task_local_ci.yaml",
        clusters=(
            "snax_versacore_to_256KB_cluster",
            "snax_versacore_to_256KB_cluster",
        ),
    ),
    "2c_simd": HardwareProfile(
        cfg="target/rtl/cfg/hemaia_tapeout_2c_simd.hjson",
        task_yaml=(
            "target/sim/automation/ci/local_ci/tapeout_2c_simd/"
            "task_local_ci.yaml"
        ),
        clusters=(
            "snax_versacore_to_256KB_simd_cluster",
            "snax_versacore_to_256KB_simd_cluster",
        ),
    ),
}


def _integer_field(body: str, field: str, cfg_path: Path) -> int:
    match = re.search(rf"\b{re.escape(field)}\s*:\s*(0x[0-9a-fA-F]+|\d+)", body)
    if match is None:
        raise ValueError(f"Cannot determine {field!r} from {cfg_path}")
    return int(match.group(1), 0)


def validate_physical_memory(cfg_path: Path) -> None:
    """Reject a profile that no longer describes the fabricated SRAM."""
    text = cfg_path.read_text()
    spm_match = re.search(r"\bspm_wide\s*:\s*\{(.*?)\}", text, re.DOTALL)
    if spm_match is None:
        raise ValueError(f"Cannot determine main-memory layout from {cfg_path}")

    size = _integer_field(spm_match.group(1), "length", cfg_path)
    banks = _integer_field(spm_match.group(1), "banks", cfg_path)
    data_width = _integer_field(text, "data_width", cfg_path)
    if (size, banks, data_width) != (
        MAIN_MEMORY_BYTES,
        MAIN_MEMORY_BANKS,
        MAIN_MEMORY_WORD_BITS,
    ):
        raise ValueError(
            f"Netlist profile {cfg_path} has unsupported main memory: "
            f"{size} bytes, {banks} banks, {data_width}-bit words; expected "
            f"{MAIN_MEMORY_BYTES} bytes, {MAIN_MEMORY_BANKS} banks, "
            f"{MAIN_MEMORY_WORD_BITS}-bit words"
        )

    words_per_bank = size * 8 // (banks * data_width)
    if words_per_bank != MAIN_MEMORY_WORDS_PER_BANK:
        raise ValueError(
            f"Netlist profile {cfg_path} has {words_per_bank} words per bank; "
            f"expected {MAIN_MEMORY_WORDS_PER_BANK}"
        )


def validate_hardware_profile(cfg_path: Path, profile: HardwareProfile) -> None:
    """Reject a cfg whose fabricated cluster identity or SRAM has drifted."""
    validate_physical_memory(cfg_path)
    text = cfg_path.read_text()
    clusters_match = re.search(r"\bclusters\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if clusters_match is None:
        raise ValueError(f"Cannot determine cluster list from {cfg_path}")
    clusters = tuple(re.findall(r'[\"\']([^\"\']+)[\"\']', clusters_match.group(1)))
    if clusters != profile.clusters:
        raise ValueError(
            f"Netlist profile {cfg_path} has unsupported clusters {clusters}; "
            f"expected {profile.clusters}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or run a fixed HeMAiA gate-level CI profile."
    )
    parser.add_argument(
        "--hardware",
        choices=tuple(HARDWARE_PROFILES),
        required=True,
        help="fabricated compute-chip profile (required)",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "prepare", "simulate"),
        default="all",
        help="end-to-end, software/filelist preparation, or backend simulation "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--engine",
        choices=("vcs", "vsim"),
        default="vcs",
        help="gate-level simulation engine (default: %(default)s)",
    )
    parser.add_argument(
        "--waveform",
        type=int,
        choices=(0, 1),
        default=0,
        help="record a simulator waveform (default: %(default)s)",
    )
    parser.add_argument(
        "-j",
        "--max-sim-jobs",
        type=int,
        default=DEFAULT_NETLIST_SIM_JOBS,
        help=f"maximum parallel simulations (default: {DEFAULT_NETLIST_SIM_JOBS})",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=24.0,
        help="timeout for each gate-level simulation (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.max_sim_jobs < 1:
        parser.error("--max-sim-jobs must be >= 1")
    if args.timeout_hours <= 0:
        parser.error("--timeout-hours must be > 0")
    return args


def main() -> None:
    args = parse_args()
    profile = HARDWARE_PROFILES[args.hardware]
    cfg_path = _REPO_ROOT / profile.cfg
    task_yaml = _REPO_ROOT / profile.task_yaml
    validate_hardware_profile(cfg_path, profile)
    if not task_yaml.is_file():
        raise FileNotFoundError(f"Netlist task profile does not exist: {task_yaml}")

    print(
        f"Netlist hardware {args.hardware}: cfg={profile.cfg}, "
        f"clusters={len(profile.clusters)}, tasks={profile.task_yaml}, "
        "main_memory=128KiB/16x1024x64"
    )
    runner = HeMAiASimRunner(
        repo_root=_REPO_ROOT,
        output_dir=_SCRIPT.parent,
        engine=args.engine,
        with_waveform=bool(args.waveform),
        cfg=profile.cfg,
        sim_cfg=SIM_CFG,
        with_macro=True,
        with_d2d=True,
        with_pll=True,
        max_jobs=args.max_sim_jobs,
        build_sw_fleet=False,
        task_yaml=task_yaml,
        fail_on_task_failure=True,
        timeout_seconds=max(1, int(args.timeout_hours * 60 * 60)),
    )
    try:
        result_path = runner.run(parse_tasks(task_yaml), phase=args.phase)
    except SimulationSuiteFailed as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"{args.phase.capitalize()} output: {result_path}")


if __name__ == "__main__":
    main()
