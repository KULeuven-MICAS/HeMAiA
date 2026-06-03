#!/usr/bin/env python3
"""Generate gemm_mem_chip_1cluster workload rows.

Each CSV row is one ``params.hjson`` test case for
``target/sw/host/apps/offload_bingo_hw/single_chip/workloads/gemm_mem_chip_1cluster``.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import hjson

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "util/sim"))
from gemm_sim_utils import (  # noqa E402
    _bytes_for_elements,
    _gemm_operand_widths,
    get_gemm_mesh_dims,
)

MEMORY_LIMIT_BYTES = int(128 * 1024 * 0.8) # 80% of 512KB to leave some room for metadata, etc.
DEFAULT_CSV = Path(__file__).with_name("testing_workload.csv")

PARAM_FIELDS = [
    "num_clusters",
    "array_shape",
    "data_type",
    "transposed_A",
    "transposed_B",
    "int4_a_enable",
    "int4_b_enable",
    "K",
    "N",
    "M",
    "addNonZeroC",
    "addZeroC",
    "accumPrevC",
    "quantization_enable",
    "int32tofp16_enable",
]

CSV_FIELDS = ["test_name", "suite", "operand_bytes"] + PARAM_FIELDS

BASE_PARAMS = {
    "num_clusters": 1,
    "array_shape": 0,
    "data_type": 0,
    "transposed_A": 0,
    "transposed_B": 0,
    "int4_a_enable": 0,
    "int4_b_enable": 0,
    "K": 32,
    "N": 3,
    "M": 2,
    "addNonZeroC": 0,
    "addZeroC": 1,
    "accumPrevC": 0,
    "quantization_enable": 0,
    "int32tofp16_enable": 0,
}


def repo_root() -> Path:
    return REPO_ROOT


def find_default_hwcfg() -> Path:
    root = repo_root()
    candidates = sorted(
        root.glob(
            ".bender/git/checkouts/snitch_cluster-*/target/snitch_cluster/cfg/"
            "snax_versacore_to_cluster.hjson"
        )
    )
    if candidates:
        return candidates[0]

    try:
        out = subprocess.check_output(
            ["bender", "path", "snitch_cluster"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        out = ""
    if out:
        return Path(out) / "target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson"

    raise FileNotFoundError(
        "Could not locate snax_versacore_to_cluster.hjson. Pass --hwcfg."
    )


def load_hwcfg(hwcfg_path: Path) -> dict:
    with hwcfg_path.open() as f:
        return hjson.loads(f.read())


def acc_cfg(hwcfg: dict) -> dict:
    return hwcfg["snax_versacore_core_template"]["snax_acc_cfg"][0]


def merged_params(params: Dict[str, int], hwcfg: dict) -> dict:
    return {**params, **hwcfg}


def operand_bytes(params: Dict[str, int], hwcfg: dict) -> int:
    merged = merged_params(params, hwcfg)
    mesh_row, tile_size, mesh_col = get_gemm_mesh_dims(merged)
    a_bits, b_bits, c_bits, d_bits = _gemm_operand_widths(merged)
    m, k, n = params["M"], params["K"], params["N"]
    return (
        _bytes_for_elements(m * k * mesh_row * tile_size, a_bits)
        + _bytes_for_elements(k * n * tile_size * mesh_col, b_bits)
        + _bytes_for_elements(m * n * mesh_row * mesh_col, c_bits)
        + _bytes_for_elements(m * n * mesh_row * mesh_col, d_bits)
    )


def valid_shape_rows(
    hwcfg: dict,
    *,
    suite: str,
    array_shapes: Sequence[int],
    overrides: Optional[Dict[str, int]] = None,
    memory_limit: int = MEMORY_LIMIT_BYTES,
) -> Iterator[dict]:
    base = dict(BASE_PARAMS)
    if overrides:
        base.update(overrides)

    data_type = base["data_type"]
    for array_shape in array_shapes:
        if array_shape >= len(
            acc_cfg(hwcfg)["snax_versacore_spatial_unrolling"][data_type]
        ):
            continue

        probe = {**base, "array_shape": array_shape, "M": 1, "K": 1, "N": 1}
        probe_merged = merged_params(probe, hwcfg)
        mesh_row, tile_size, mesh_col = (
            int(dim) for dim in get_gemm_mesh_dims(probe_merged)
        )
        a_bits, b_bits, c_bits, d_bits = (
            int(width) for width in _gemm_operand_widths(probe_merged)
        )

        a_unit = _bytes_for_elements(mesh_row * tile_size, a_bits)
        b_unit = _bytes_for_elements(tile_size * mesh_col, b_bits)
        cd_unit = _bytes_for_elements(mesh_row * mesh_col, c_bits) + _bytes_for_elements(
            mesh_row * mesh_col, d_bits
        )

        max_mn = int(max(1, memory_limit // cd_unit))
        for m in range(1, max_mn + 1):
            for n in range(1, int(max_mn // m) + 1):
                remaining = memory_limit - (m * n * cd_unit)
                if remaining < a_unit * m + b_unit * n:
                    continue
                max_k = int(remaining // (a_unit * m + b_unit * n))
                for k in range(1, max_k + 1):
                    params = {**base, "array_shape": array_shape, "M": m, "K": k, "N": n}
                    size = operand_bytes(params, hwcfg)
                    if size <= memory_limit:
                        yield {
                            "test_name": f"{suite}_as{array_shape}_m{m}_k{k}_n{n}",
                            "suite": suite,
                            "operand_bytes": size,
                            **params,
                        }


def generate_base_int8_shapes(hwcfg: dict, memory_limit: int) -> Iterable[dict]:
    return valid_shape_rows(
        hwcfg,
        suite="base_int8",
        array_shapes=(0, 1, 2),
        memory_limit=memory_limit,
    )


def iter_workloads(
    hwcfg: dict,
    *,
    memory_limit: int = MEMORY_LIMIT_BYTES,
) -> Iterator[dict]:
    seen = set()
    suites = (
        generate_base_int8_shapes,
    )
    for suite in suites:
        for row in suite(hwcfg, memory_limit):
            key = tuple(row[field] for field in PARAM_FIELDS)
            if key in seen:
                continue
            seen.add(key)
            yield row


def sample_workloads(
    hwcfg: dict,
    *,
    memory_limit: int,
    max_cases: int,
    seed: int,
) -> tuple[List[dict], int]:
    rng = random.Random(seed)
    sample: List[dict] = []
    total = 0

    for row in iter_workloads(hwcfg, memory_limit=memory_limit):
        total += 1
        if len(sample) < max_cases:
            sample.append(row)
            continue

        replacement_idx = rng.randrange(total)
        if replacement_idx < max_cases:
            sample[replacement_idx] = row

    sample.sort(key=lambda row: row["test_name"])
    return sample, total


def write_csv(rows: Iterable[dict], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--memory-limit", type=int, default=MEMORY_LIMIT_BYTES)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=128,
        help="Randomly sample at most this many rows. Zero means no cap.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hwcfg_path = find_default_hwcfg()
    hwcfg = load_hwcfg(hwcfg_path)
    if args.max_cases:
        rows, total = sample_workloads(
            hwcfg,
            memory_limit=args.memory_limit,
            max_cases=args.max_cases,
            seed=args.seed,
        )
        count = write_csv(rows, args.output)
        print(f"Randomly sampled {count} of {total} test cases to {args.output}")
    else:
        count = write_csv(
            iter_workloads(hwcfg, memory_limit=args.memory_limit),
            args.output,
        )
        print(f"Wrote {count} test cases to {args.output}")


if __name__ == "__main__":
    main()
