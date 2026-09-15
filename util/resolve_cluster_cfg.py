#!/usr/bin/env python3
"""Resolve cluster configurations from the same SoC CFG used by occamygen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import hjson


REPO_ROOT = Path(__file__).resolve().parents[1]


def find_snitch_root(repo_root: Path = REPO_ROOT) -> Path:
    """Use Bender's selected checkout; deps is a fallback when Bender is absent."""
    if shutil.which("bender"):
        result = subprocess.run(
            ["bender", "path", "snitch_cluster"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        )
        path = Path(result.stdout.strip())
        if not path.is_absolute():
            path = repo_root / path
    else:
        path = repo_root / "deps/snitch_cluster"
    if not path.is_dir():
        raise ValueError("Cannot resolve snitch_cluster; run Bender or pass --snitch-root")
    return path.resolve()


def cluster_config_paths(cfg_path: Path, snitch_root: Path) -> list[Path]:
    """Return cluster HJSON paths in SoC order, including repeated clusters."""
    with cfg_path.open() as stream:
        cfg = hjson.load(stream)
    names = cfg.get("clusters")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{cfg_path}: expected a nonempty 'clusters' list")
    paths = []
    for name in names:
        path = snitch_root / "target/snitch_cluster/cfg" / f"{name}.hjson"
        if not path.is_file():
            raise ValueError(f"{cfg_path}: selected cluster config does not exist: {path}")
        paths.append(path.resolve())
    return paths


def gemm_config_path(cfg_path: Path, snitch_root: Path) -> Path:
    """Resolve one GEMM config, rejecting incompatible cluster geometries."""
    paths = list(dict.fromkeys(cluster_config_paths(cfg_path, snitch_root)))
    signature = None
    for path in paths:
        with path.open() as stream:
            cfg = hjson.load(stream)
        try:
            acc = cfg["snax_versacore_core_template"]["snax_acc_cfg"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"{path}: no VersaCore GEMM accelerator configuration") from exc
        # These fields describe GEMM geometry, precision, and streamer wiring.
        # XDMA extensions can differ without changing the GEMM shape table.
        current = {key: value for key, value in acc.items()
                   if key.startswith(("snax_versacore_", "granularity_"))
                   or key == "snax_streamer_cfg"}
        if signature is not None and current != signature:
            raise ValueError(f"{cfg_path}: clusters have incompatible GEMM configurations")
        signature = current
    return paths[0]


def update_stamp(stamp: Path, paths: list[Path]) -> bool:
    """Record selection and contents; preserve mtime when neither changed."""
    records = [{"path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in dict.fromkeys(paths)]
    content = json.dumps(records, indent=2) + "\n"
    if stamp.exists() and stamp.read_text() == content:
        return False
    stamp.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=stamp.parent,
                                         prefix=stamp.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, stamp)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=REPO_ROOT / "target/rtl/cfg/lru.hjson")
    parser.add_argument("--snitch-root", type=Path)
    parser.add_argument("--format", choices=("paths", "gemm"), default="gemm")
    parser.add_argument("--stamp", type=Path,
                        help="update a Make dependency stamp only when config selection/content changes")
    parser.add_argument("--hwcfg", type=Path,
                        help="include an explicit workload GEMM_HWCFG override in the stamp")
    args = parser.parse_args()
    try:
        snitch_root = args.snitch_root or find_snitch_root()
        paths = cluster_config_paths(args.cfg, snitch_root)
        if args.stamp:
            inputs = [args.cfg, *paths]
            if args.hwcfg:
                inputs.append(args.hwcfg)
            update_stamp(args.stamp, inputs)
        elif args.format == "paths":
            print(" ".join(str(path) for path in dict.fromkeys(paths)))
        else:
            print(gemm_config_path(args.cfg, snitch_root))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Cluster configuration error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
