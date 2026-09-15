#!/usr/bin/env python3
"""Resolve and record the hardware used by a VersaCore workload sweep."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import hjson

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CFG = REPO_ROOT / "target/rtl/cfg/lru.hjson"
L1_MEMORY_FRACTION = 0.8

sys.path.insert(0, str(REPO_ROOT / "util"))
from resolve_cluster_cfg import (  # noqa: E402
    cluster_config_paths,
    find_snitch_root,
    gemm_config_path,
)


def read_config(path: Path) -> dict:
    return hjson.loads(path.read_text())


def fingerprint(config: dict) -> str:
    """Ignore formatting and comments when comparing hardware descriptions."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def default_l1_memory_limit(hwcfg: dict) -> int:
    return int(int(hwcfg["cluster"]["tcdm"]["size"]) * 1024 * L1_MEMORY_FRACTION)


@dataclass(frozen=True)
class SweepConfig:
    cfg_path: Path
    hwcfg_path: Path
    cfg: dict
    hwcfg: dict
    num_clusters: int


@contextmanager
def snapshot_cfg(config: SweepConfig):
    """Keep an active LRU config available across the launcher's initial clean."""
    directory = REPO_ROOT / "target/rtl/cfg"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=directory, prefix=".versacore-sweep-", suffix=".hjson",
        delete=False,
    ) as stream:
        path = Path(stream.name)
        stream.write(hjson.dumps(config.cfg, indent=2) + "\n")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def resolve_sweep_config(
    cfg_path: Path = DEFAULT_CFG, hwcfg_path: Path | None = None
) -> SweepConfig:
    cfg_path = cfg_path.expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"SoC config does not exist: {cfg_path}; pass --cfg.")
    snitch_root = find_snitch_root(REPO_ROOT)
    selected_hwcfg = gemm_config_path(cfg_path, snitch_root)
    cfg = read_config(cfg_path)
    selected = read_config(selected_hwcfg)
    if hwcfg_path is not None:
        hwcfg_path = hwcfg_path.expanduser().resolve()
        if fingerprint(read_config(hwcfg_path)) != fingerprint(selected):
            raise ValueError(
                f"--hwcfg {hwcfg_path} differs from {selected_hwcfg} selected by "
                f"--cfg {cfg_path}. Select a matching SoC config."
            )
    else:
        hwcfg_path = selected_hwcfg
    return SweepConfig(
        cfg_path, hwcfg_path, cfg, selected,
        len(cluster_config_paths(cfg_path, snitch_root)),
    )


def metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".meta.json")


def write_metadata(
    csv_path: Path, config: SweepConfig, l1_memory_limit: int, l3_memory_limit: int
) -> None:
    try:
        cfg_path = str(config.cfg_path.relative_to(REPO_ROOT))
    except ValueError:
        cfg_path = str(config.cfg_path)
    metadata = {
        "schema_version": 1,
        "cfg": cfg_path,
        "cfg_sha256": fingerprint(config.cfg),
        "hwcfg": config.hwcfg_path.name,
        "hwcfg_sha256": fingerprint(config.hwcfg),
        "num_clusters": config.num_clusters,
        "l1_memory_limit": l1_memory_limit,
        "l3_memory_limit": l3_memory_limit,
    }
    metadata_path(csv_path).write_text(json.dumps(metadata, indent=2) + "\n")


def config_for_csv(
    csv_path: Path, cfg_path: Path | None = None, hwcfg_path: Path | None = None
) -> SweepConfig:
    path = metadata_path(csv_path)
    if not path.is_file():
        if cfg_path is None:
            raise ValueError(
                f"{path} is missing. Regenerate the workload CSV, or pass --cfg "
                "explicitly for a CSV created before config metadata was added."
            )
        return resolve_sweep_config(cfg_path, hwcfg_path)
    metadata = json.loads(path.read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported sweep metadata version in {path}")
    if cfg_path is None:
        cfg_path = REPO_ROOT / metadata["cfg"]
    config = resolve_sweep_config(cfg_path, hwcfg_path)
    if (fingerprint(config.cfg) != metadata["cfg_sha256"] or
            fingerprint(config.hwcfg) != metadata["hwcfg_sha256"] or
            config.num_clusters != metadata["num_clusters"]):
        raise ValueError(
            f"Selected hardware differs from the config recorded in {path}. "
            "Regenerate the workload CSV for this config."
        )
    return config
