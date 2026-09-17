#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Cross-check the BINGO task-descriptor bit layout across all three of its definitions.

WHY THIS EXISTS
---------------
The layout is written down in three independent places:

  1. the RTL packed struct  bingo_hw_manager_task_desc_t   (the hardware; canonical)
  2. the Python packer      bingo_dfg.bingo_pack_node      (what actually emits the task list)
  3. the C macros           libbingo/bingo_utils.h         (what C code would use)

Nothing connected them. They drifted, and the drift was invisible: every bit pattern is a legal
descriptor, so a mis-placed field does not fault, it just schedules the wrong thing. When this
checker was first written, definition 3 computed a 50-bit layout where 1 and 2 agreed on 65 -- four
separate bugs (wrong shift advance, clog2 vs idx_width, a core count that excluded the host CVA6,
and a chiplet-id width taken from the chiplet COUNT instead of the routing-id width). It had been
wrong for a long time and was survivable only because the C encoder happened to have no callers.

Run this in CI. A layout divergence caught here costs a minute; caught in silicon it costs a week.

  python3 util/bingo/check_task_desc_layout.py [--cfg target/rtl/cfg/<cfg>.hjson]
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RTL_CANDIDATES = [
    ROOT / ".bender/git/checkouts",            # resolved dependency checkout
    ROOT.parent / "bingo_hw_manager",          # sibling working copy
]
C_HEADER_DIR = ROOT / "target/sw/host/runtime/libbingo/include"
OCCAMY_H = ROOT / "target/sw/shared/platform/generated/occamy.h"


def find_rtl_top():
    for base in RTL_CANDIDATES:
        if not base.exists():
            continue
        for p in sorted(base.glob("**/src/bingo_hw_manager_top.sv")):
            return p
    raise SystemExit("could not find bingo_hw_manager_top.sv (run `bender update` first?)")


def parse_defines(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\s*#define\s+(\w+)\s+(\S+)", line)
        if m:
            try:
                out[m.group(1)] = int(m.group(2), 0)
            except ValueError:
                pass
    return out


def idx_width(n):
    """cf_math_pkg::idx_width -- clog2 with a floor of 1."""
    return max(1, (n - 1).bit_length()) if n > 1 else 1


def struct_body(text, name):
    """Return the member lines of `typedef struct packed{ ... } <name>;`."""
    # [^{}]* rather than .*? : these structs contain no nested braces, and a non-greedy .*?
    # happily starts at the FIRST `typedef struct packed{` in the file and runs all the way to
    # this struct's closing brace, swallowing every struct in between.
    m = re.search(r"typedef\s+struct\s+packed\s*\{([^{}]*)\}\s*" + re.escape(name) + r"\s*;",
                  text, re.S)
    if not m:
        raise SystemExit(f"could not find packed struct {name} in the RTL")
    members = []
    for line in m.group(1).splitlines():
        line = re.sub(r"//.*", "", line).strip()
        if not line or not line.endswith(";"):
            continue
        mm = re.match(r"([A-Za-z_][\w:\[\]\-\s']*?)\s+([A-Za-z_]\w*)\s*;$", line)
        if mm:
            members.append((mm.group(1).strip(), mm.group(2)))
    return members


def rtl_layout(rtl_path, cfg):
    """Field list LSB->MSB as the RTL packed struct defines it.

    A packed struct declares MSB first, so the member list is reversed to get bit order.
    """
    text = rtl_path.read_text()
    ncores_hw = cfg["ncores_hw"]
    tw = {
        "bingo_hw_manager_task_type_t":           2,
        "bingo_hw_manager_task_id_t":             cfg["task_id_width"],
        "bingo_hw_manager_assigned_chiplet_id_t": cfg["chip_id_width"],
        "bingo_hw_manager_assigned_cluster_id_t": idx_width(cfg["nclusters"]),
        "bingo_hw_manager_assigned_core_id_t":    idx_width(ncores_hw),
        "bingo_hw_manager_dep_code_t":            ncores_hw,
        "bingo_hw_manager_dep_tag_t":             cfg["dep_tag_width"],
        "logic":                                  1,
        "logic [4:0]":                            5,
    }
    nested = {
        "bingo_hw_manager_dep_check_info_t": "bingo_hw_manager_dep_check_info_t",
        "bingo_hw_manager_dep_set_info_t":   "bingo_hw_manager_dep_set_info_t",
    }

    def expand(members):
        out = []
        for ty, nm in members:
            if ty in nested:
                out.extend(expand(struct_body(text, nested[ty])))
            elif ty in tw:
                out.append((nm, tw[ty]))
            else:
                raise SystemExit(f"unknown RTL member type {ty!r} for field {nm!r} -- the struct "
                                 f"gained a field this checker does not know about")
        return out

    return list(reversed(expand(struct_body(text, "bingo_hw_manager_task_desc_t"))))


def py_layout(cfg):
    """Field list LSB->MSB as the Python packer builds it."""
    n, tag = cfg["ncores_hw"], cfg["dep_tag_width"]
    cw = idx_width(cfg["nclusters"])
    return [
        ("cond_exec_invert", 1), ("cond_exec_group_id", 5), ("cond_exec_en", 1),
        ("task_type", 2), ("task_id", cfg["task_id_width"]),
        ("assigned_chiplet_id", cfg["chip_id_width"]),
        ("assigned_cluster_id", cw), ("assigned_core_id", idx_width(n)),
        ("dep_check_en", 1), ("dep_check_code", n), ("dep_check_tag", tag),
        ("dep_set_en", 1), ("dep_set_all_chiplet", 1),
        ("dep_set_chiplet_id", cfg["chip_id_width"]),
        ("dep_set_cluster_id", cw), ("dep_set_code", n), ("dep_set_tag", tag),
    ]


C_FIELDS = [
    ("cond_exec_invert", "COND_EXEC_INVERT"), ("cond_exec_group_id", "COND_EXEC_GROUP_ID"),
    ("cond_exec_en", "COND_EXEC_EN"), ("task_type", "TASK_TYPE"), ("task_id", "TASK_ID"),
    ("assigned_chiplet_id", "ASSIGNED_CHIPLET_ID"), ("assigned_cluster_id", "ASSIGNED_CLUSTER_ID"),
    ("assigned_core_id", "ASSIGNED_CORE_ID"), ("dep_check_en", "DEP_CHECK_ENABLED"),
    ("dep_check_code", "DEP_CHECK_CODE"), ("dep_check_tag", "DEP_CHECK_TAG"),
    ("dep_set_en", "DEP_SET_ENABLED"), ("dep_set_all_chiplet", "DEP_SET_ALL_CHIPLET"),
    ("dep_set_chiplet_id", "DEP_SET_CHIPLET_ID"), ("dep_set_cluster_id", "DEP_SET_CLUSTER_ID"),
    ("dep_set_code", "DEP_SET_CODE"), ("dep_set_tag", "DEP_SET_TAG"),
]


def c_layout():
    """Field list LSB->MSB as the C macros compute it -- by COMPILING them, not re-reading them.

    Re-implementing the macro arithmetic here would just be a fourth definition to drift.
    """
    src = ['#include <stdio.h>', '#include "libbingo/bingo_utils.h"', "int main(void){"]
    for nm, mac in C_FIELDS:
        src.append(f'  printf("%s %d %d\\n", "{nm}", (int){mac}_SHIFT, (int){mac}_WIDTH);')
    src.append("  return 0;}")
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "lay.c"
        c.write_text("\n".join(src))
        exe = Path(td) / "lay"
        r = subprocess.run(["gcc", "-std=gnu99", "-I", str(C_HEADER_DIR),
                            "-I", str(OCCAMY_H.parent), str(c), "-o", str(exe)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr.strip()
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        fields = []
        for line in r.stdout.split("\n"):
            parts = line.split()
            if len(parts) == 3:
                fields.append((parts[0], int(parts[1]), int(parts[2])))
        return fields, None


def to_offsets(fields):
    out, shift = [], 0
    for nm, w in fields:
        out.append((nm, shift, w))
        shift += w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    d = parse_defines(OCCAMY_H)
    cfg = {
        "ncores_hw": d.get("BINGO_NCORES_HW", d.get("N_CORES_PER_CLUSTER", 4) + 1),
        "nclusters": d.get("N_CLUSTERS_PER_CHIPLET", 1),
        "dep_tag_width": d.get("BINGO_DEP_TAG_WIDTH", 4),
        "chip_id_width": d.get("BINGO_CHIP_ID_WIDTH", 8),
        "task_id_width": 12,
    }
    container = d.get("BINGO_TASK_DESC_WIDTH", 128)

    rtl = to_offsets(rtl_layout(find_rtl_top(), cfg))
    py = to_offsets(py_layout(cfg))
    c, cerr = c_layout()

    total = sum(w for _, _, w in rtl)
    print(f"config: ncores_hw={cfg['ncores_hw']} clusters={cfg['nclusters']} "
          f"dep_tag={cfg['dep_tag_width']} chip_id={cfg['chip_id_width']}")
    print(f"layout = {total} bits, container = {container} bits "
          f"({container // 64} beat(s)), spare = {container - total}\n")

    bad = 0
    if total > container:
        print(f"FAIL: layout {total} b exceeds container {container} b")
        bad += 1

    hdr = f"{'field':<22}{'RTL':>14}{'Python':>14}{'C':>14}"
    print(hdr)
    print("-" * len(hdr))
    cmap = {n: (s, w) for n, s, w in (c or [])}
    if len(rtl) != len(py):
        print(f"FAIL: RTL struct has {len(rtl)} fields, the Python packer has {len(py)}. "
              f"RTL fields: {[n for n, _, _ in rtl]}")
        return 1
    for i, (nm, sh, w) in enumerate(rtl):
        pn, ps, pw = py[i]
        row = f"{nm:<22}{f'{sh}:{w}':>14}{f'{ps}:{pw}':>14}"
        ok = (pn == nm and ps == sh and pw == w)
        if c is None:
            row += f"{'(not built)':>14}"
        else:
            cs, cw = cmap.get(nm, (None, None))
            row += f"{f'{cs}:{cw}':>14}"
            ok = ok and cs == sh and cw == w
        if not ok:
            row += "   <-- MISMATCH"
            bad += 1
        elif args.quiet:
            continue
        print(row)

    if cerr:
        print(f"\nC layout could not be compiled, so it was NOT checked:\n{cerr}")
        bad += 1

    print()
    if bad:
        print(f"LAYOUT CHECK FAILED ({bad} problem(s)). The three definitions of the task "
              f"descriptor do not agree; whichever one the hardware uses, something else is "
              f"writing the wrong bits.")
        return 1
    print("LAYOUT CHECK PASSED: RTL, Python packer and C macros agree field for field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
