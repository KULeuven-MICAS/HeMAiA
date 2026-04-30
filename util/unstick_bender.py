#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Unstick the .bender/ directory after a hung bender/git fetch on NFS.

When ``bender`` shells out to ``git fetch`` and the network stalls, the
spawned ``git index-pack`` (or ``git-remote-https``) keeps a pack file
open inside ``.bender/git/db/<pkg>/objects/pack/``. If a later
``rm -rf .bender`` (e.g. from ``make clean-bender``) tries to delete
that file while the handle is still alive, NFS silly-renames it to
``.nfsXXXXXXXX`` and refuses to remove the parent directory:

    rm: cannot remove '.bender/.../pack': Directory not empty

This script:

  1. Scans ``.bender/`` for ``.nfsXXXX`` files and uses ``lsof`` to find
     the PIDs holding them open.
  2. Walks ``/proc`` for the current user's ``git`` / ``bender`` /
     ``git-remote-http(s)`` processes whose ``cwd`` is inside this
     repository -- those are the stuck fetch trees that have not yet
     been silly-renamed but will block the next clean.
  3. Sends SIGTERM, then SIGKILL to any survivors.
  4. With ``--purge``, removes ``.bender/``, ``deps/``, and
     ``Bender.lock`` so the next bender invocation starts fresh.

Examples:

    util/unstick_bender.py                  # diagnose and fix stuck handles
    util/unstick_bender.py --dry-run        # show what would happen
    util/unstick_bender.py --purge --yes    # full reset, no prompt
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Set


# Process names we are willing to signal. Anything outside this set is
# reported but never killed -- avoids accidentally hitting an editor,
# language server, or unrelated job that happens to keep a file open.
SAFE_COMMS = {
    "git",
    "git-remote-http",
    "git-remote-https",
    "bender",
}


def find_nfs_silly_files(bender_dir: Path) -> List[Path]:
    if not bender_dir.exists():
        return []
    # NFS silly-rename pattern: ".nfs" + hex chars. Stick to files only;
    # directories never get silly-renamed.
    pat = re.compile(r"^\.nfs[0-9a-fA-F]+$")
    return sorted(p for p in bender_dir.rglob(".nfs*")
                  if p.is_file() and pat.match(p.name))


def lsof_pids_for_path(path: Path) -> Set[int]:
    """Return PIDs holding ``path`` open, via ``lsof -t``."""
    try:
        out = subprocess.run(
            ["lsof", "-t", "--", str(path)],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        print("[unstick-bender] WARNING: lsof not found; "
              "cannot identify silly-rename holders.", file=sys.stderr)
        return set()
    return {int(tok) for tok in out.stdout.split() if tok.strip().isdigit()}


def proc_comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def proc_cwd(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return ""


def proc_uid(pid: int) -> int:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return -1
    m = re.search(r"^Uid:\s+(\d+)", status, re.M)
    return int(m.group(1)) if m else -1


def find_stuck_bender_tree(repo_root: Path) -> Set[int]:
    """Find current-user bender/git processes whose cwd is in repo_root."""
    repo_str = str(repo_root.resolve())
    uid = os.getuid()
    pids: Set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if proc_uid(pid) != uid:
            continue
        if proc_comm(pid) not in SAFE_COMMS:
            continue
        cwd = proc_cwd(pid)
        if cwd.startswith(repo_str):
            pids.add(pid)
    return pids


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_pids(pids: Iterable[int], *, force: bool) -> None:
    sig = signal.SIGKILL if force else signal.SIGTERM
    label = "SIGKILL" if force else "SIGTERM"
    for pid in sorted(pids):
        try:
            os.kill(pid, sig)
            print(f"  {label} -> {pid}")
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"  PERMISSION DENIED on {pid} (not your process?)",
                  file=sys.stderr)


def describe_pid(pid: int) -> str:
    return (f"pid={pid:<7}  comm={proc_comm(pid):<18} "
            f"cwd={proc_cwd(pid) or '?':<60} "
            f"cmd={proc_cmdline(pid)[:120]}")


def do_purge(repo_root: Path, *, dry_run: bool) -> None:
    targets = [
        repo_root / ".bender",
        repo_root / "deps",
        repo_root / "Bender.lock",
    ]
    for t in targets:
        if not t.exists() and not t.is_symlink():
            continue
        if dry_run:
            print(f"[unstick-bender] (dry-run) would remove {t}")
            continue
        print(f"[unstick-bender] removing {t}")
        try:
            if t.is_dir() and not t.is_symlink():
                shutil.rmtree(t)
            else:
                t.unlink()
        except OSError as e:
            print(f"[unstick-bender] failed to remove {t}: {e}",
                  file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="HeMAiA repo root (default: parent of util/).",
    )
    parser.add_argument(
        "--purge", action="store_true",
        help="After unsticking, remove .bender/, deps/, and Bender.lock.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be killed/removed without doing it.",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Do not prompt before killing processes (for scripts).",
    )
    parser.add_argument(
        "--wait", type=float, default=5.0,
        help="Seconds to wait after SIGTERM before escalating to SIGKILL "
             "(default: 5.0).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    bender_dir = (repo_root / ".bender").resolve()
    print(f"[unstick-bender] repo root : {repo_root}")
    print(f"[unstick-bender] target    : {bender_dir}")

    # 1. Silly-rename files and the PIDs holding them.
    nfs_files = find_nfs_silly_files(bender_dir)
    holder_pids: Set[int] = set()
    for f in nfs_files:
        holder_pids |= lsof_pids_for_path(f)

    # 2. Stuck bender/git tree rooted in this repo.
    tree_pids = find_stuck_bender_tree(repo_root)

    # 3. Filter to whitelist.
    candidates = holder_pids | tree_pids
    safe_pids = sorted(p for p in candidates if proc_comm(p) in SAFE_COMMS)
    skipped = sorted(p for p in candidates if proc_comm(p) not in SAFE_COMMS)

    print(f"[unstick-bender] .nfs files : {len(nfs_files)}")
    for f in nfs_files:
        try:
            size = f.stat().st_size
        except OSError:
            size = -1
        print(f"    {f}  ({size} bytes)")

    print(f"[unstick-bender] candidates : {len(safe_pids)}")
    for pid in safe_pids:
        print(f"    {describe_pid(pid)}")
    if skipped:
        print(f"[unstick-bender] skipped (not in whitelist): {skipped}")

    if not nfs_files and not safe_pids:
        print("[unstick-bender] nothing stuck.")
        if args.purge:
            do_purge(repo_root, dry_run=args.dry_run)
        return 0

    if args.dry_run:
        print("[unstick-bender] dry-run: not killing anything.")
        if args.purge:
            do_purge(repo_root, dry_run=True)
        return 0

    if safe_pids and not args.yes:
        try:
            reply = input("[unstick-bender] kill these PIDs? [y/N] ")
        except EOFError:
            reply = ""
        if reply.strip().lower() not in ("y", "yes"):
            print("[unstick-bender] aborted.")
            return 1

    if safe_pids:
        kill_pids(safe_pids, force=False)
        deadline = time.monotonic() + max(args.wait, 0.0)
        while time.monotonic() < deadline:
            if not any(alive(p) for p in safe_pids):
                break
            time.sleep(0.2)
        leftovers = [p for p in safe_pids if alive(p)]
        if leftovers:
            print(f"[unstick-bender] still alive after SIGTERM: {leftovers} "
                  f"-- escalating.")
            kill_pids(leftovers, force=True)
            time.sleep(1.0)

    # NFS may take a moment to actually unlink the silly-rename file once
    # the last handle closes. Poll briefly.
    for _ in range(10):
        leftover_nfs = find_nfs_silly_files(bender_dir)
        if not leftover_nfs:
            break
        time.sleep(0.5)
    else:
        leftover_nfs = find_nfs_silly_files(bender_dir)

    if leftover_nfs:
        print(f"[unstick-bender] {len(leftover_nfs)} .nfs file(s) still "
              f"present; NFS has not released them yet. Re-run in a few "
              f"seconds, or check ``lsof`` for new holders.")
        for f in leftover_nfs:
            print(f"    {f}")
    else:
        print("[unstick-bender] all silly-rename files released.")

    if args.purge:
        do_purge(repo_root, dry_run=False)

    print("[unstick-bender] done. You can now rerun "
          "``make clean-bender`` (or your test flow).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
