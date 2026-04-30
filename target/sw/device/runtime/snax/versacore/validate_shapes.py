#!/usr/bin/env python3
# Copyright 2025 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
# Xiaoling Yi <xiaoling.yi@kuleuven.be>
"""
Validate offload_hw_kernels/gemm_shapes.h against the active hwcfg.

Reads the hwcfg (snax_versacore_to_cluster.hjson), recomputes the expected
per-shape parameter table and shape-invariant widths, parses the hand-written
gemm_shapes.h, and diffs. Exits 0 on match, 1 on mismatch (with a clear
message pointing to the offending field).

This script does NOT generate code. The header is hand-maintained; this just
catches drift between C and the hwcfg at build time.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys

import hjson


# ----------------------------------------------------------------------
# Expected values (from hwcfg) — mirrors compute_shape_params() and
# _channel_enable_bits() in gen_bingo_gemm_kernels.py.
# ----------------------------------------------------------------------
def _channel_enable_bits(bits: int, csr_num: int) -> list[int]:
    out = [0] * csr_num
    for i in range(bits):
        idx, pos = i // 32, i % 32
        if idx < csr_num:
            out[idx] |= 1 << pos
    out.reverse()
    return [int(x) for x in out]


def expected_from_hwcfg(hwcfg: dict) -> tuple[dict, list[dict]]:
    acc = hwcfg["snax_versacore_core_template"]["snax_acc_cfg"][0]
    spatial = acc["snax_versacore_spatial_unrolling"][0]
    a_len = acc["snax_versacore_input_a_element_width"][0]
    b_len = acc["snax_versacore_input_b_element_width"][0]
    c_len = acc["snax_versacore_input_c_element_width"][0]
    d_len = acc["snax_versacore_output_d_element_width"][0]
    a_aw  = acc["snax_versacore_array_input_a_width"]
    b_aw  = acc["snax_versacore_array_input_b_width"]
    serial_cd = acc["snax_versacore_serial_c_d_width"]
    bw = 64

    globals_ = {
        "BINGO_BANK_WIDTH":       bw,
        "BINGO_A_ELEM_LEN":       a_len,
        "BINGO_B_ELEM_LEN":       b_len,
        "BINGO_C_ELEM_LEN":       c_len,
        "BINGO_D32_ELEM_LEN":     d_len,
        "BINGO_SERIAL_C_D_WIDTH": serial_cd,
        "BINGO_A_CSR_NUM":        int(math.ceil(a_aw      / bw / 32)),
        "BINGO_B_CSR_NUM":        int(math.ceil(b_aw      / bw / 32)),
        "BINGO_C_CSR_NUM":        int(math.ceil(serial_cd / bw / 32)),
        "BINGO_D32_CSR_NUM":      int(math.ceil(serial_cd / bw / 32)),
        "BINGO_NUM_ARRAY_SHAPES": len(spatial),
    }

    shapes: list[dict] = []
    for mr, ts, mc in spatial:
        a_bits = max(8, int((mr * ts * a_len / bw + 7) // 8 * 8))
        b_bits = max(8, int((mc * ts * b_len / bw + 7) // 8 * 8))
        c_bits = int((mr * mc * c_len / bw + 7) // 8 * 8)
        c_sp = (serial_cd if mr * mc * c_len >= serial_cd
                else mr * mc * c_len) / bw
        ctlstride0 = int(c_sp * (bw / 8))
        ctlbound0  = max(1, int(mr * mc * c_len / serial_cd))
        shapes.append({
            "meshRow":      int(mr),
            "tileSize":     int(ts),
            "meshCol":      int(mc),
            "Ctlbound0":    ctlbound0,
            "Ctlstride0":   ctlstride0,
            "D32tlbound0":  ctlbound0,
            "D32tlstride0": ctlstride0,
            "channel_en_A":   _channel_enable_bits(a_bits, globals_["BINGO_A_CSR_NUM"]),
            "channel_en_B":   _channel_enable_bits(b_bits, globals_["BINGO_B_CSR_NUM"]),
            "channel_en_C":   _channel_enable_bits(c_bits, globals_["BINGO_C_CSR_NUM"]),
            "channel_en_D32": _channel_enable_bits(c_bits, globals_["BINGO_D32_CSR_NUM"]),
        })
    return globals_, shapes


# ----------------------------------------------------------------------
# Actual values (from gemm_shapes.h) — regex-parsed.
# ----------------------------------------------------------------------
_DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+(BINGO_[A-Z0-9_]+)\s+(\S+)\s*$", re.MULTILINE)

_BLOCK_START_RE = re.compile(r"\[\s*(\d+)\s*\]\s*=\s*\{")

_SCALAR_FIELD_RE = re.compile(
    r"\.\s*(meshRow|tileSize|meshCol|Ctlbound0|Ctlstride0|D32tlbound0|D32tlstride0)"
    r"\s*=\s*([0-9a-fA-FxX]+)[uU]?\s*,")

_ARRAY_FIELD_RE = re.compile(
    r"\.\s*(channel_en_A|channel_en_B|channel_en_C|channel_en_D32)\s*=\s*\{([^}]*)\}")


def _parse_int(tok: str) -> int:
    """Parse a C integer literal (with optional u/U suffix, 0x or decimal)."""
    tok = tok.strip().rstrip("uU")
    return int(tok, 0)


def _find_matching_brace(src: str, open_pos: int) -> int:
    """Return the index of the `}` that matches the `{` at src[open_pos-1].
    open_pos must point one past the opening `{`."""
    depth = 1
    i = open_pos
    n = len(src)
    while i < n:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unmatched brace in shape initializer")


def parse_header(src: str) -> tuple[dict, list[dict]]:
    globals_: dict = {}
    for m in _DEFINE_RE.finditer(src):
        name, rhs = m.group(1), m.group(2).strip()
        try:
            globals_[name] = _parse_int(rhs)
        except ValueError:
            # Non-integer define (unlikely for our BINGO_ keys); skip.
            continue

    shapes: list[tuple[int, dict]] = []
    for m in _BLOCK_START_RE.finditer(src):
        idx = int(m.group(1))
        close = _find_matching_brace(src, m.end())
        body = src[m.end():close]
        s: dict = {}
        for sm in _SCALAR_FIELD_RE.finditer(body):
            s[sm.group(1)] = _parse_int(sm.group(2))
        for am in _ARRAY_FIELD_RE.finditer(body):
            elems = [_parse_int(t) for t in am.group(2).split(",") if t.strip()]
            s[am.group(1)] = elems
        shapes.append((idx, s))

    shapes.sort(key=lambda p: p[0])
    return globals_, [s for _, s in shapes]


# ----------------------------------------------------------------------
# Diff.
# ----------------------------------------------------------------------
def diff(expected_g: dict, expected_s: list[dict],
         actual_g: dict, actual_s: list[dict]) -> list[str]:
    errors: list[str] = []

    for key, exp in expected_g.items():
        act = actual_g.get(key)
        if act is None:
            errors.append(f"missing #define {key} (expected {exp})")
        elif act != exp:
            errors.append(f"{key}: expected {exp}, header has {act}")

    if len(actual_s) != len(expected_s):
        errors.append(
            f"shape count: expected {len(expected_s)} entries in "
            f"bingo_gemm_shape_params[], header has {len(actual_s)}")
        return errors

    for i, (exp, act) in enumerate(zip(expected_s, actual_s)):
        for key, exp_val in exp.items():
            act_val = act.get(key)
            if act_val is None:
                errors.append(f"shape[{i}].{key}: missing in header "
                              f"(expected {exp_val})")
            elif act_val != exp_val:
                errors.append(f"shape[{i}].{key}: expected {exp_val!r}, "
                              f"header has {act_val!r}")
    return errors


# ----------------------------------------------------------------------
# Entry point.
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True,
                    help="path to snax_versacore_to_cluster.hjson")
    ap.add_argument("--header", type=pathlib.Path, required=True,
                    help="path to offload_hw_kernels/gemm_shapes.h")
    args = ap.parse_args()

    if not args.hwcfg.is_file():
        print(f"error: hwcfg not found: {args.hwcfg}", file=sys.stderr)
        return 2
    if not args.header.is_file():
        print(f"error: header not found: {args.header}", file=sys.stderr)
        return 2

    hwcfg = hjson.loads(args.hwcfg.read_text())
    exp_g, exp_s = expected_from_hwcfg(hwcfg)
    act_g, act_s = parse_header(args.header.read_text())

    errors = diff(exp_g, exp_s, act_g, act_s)
    if errors:
        print(f"[validate_shapes] {args.header} is out of sync with {args.hwcfg}:",
              file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        print("Fix the header (see derivation comment block at the top) "
              "and rerun `make sw`.", file=sys.stderr)
        return 1

    print(f"[validate_shapes] {args.header.name} matches hwcfg "
          f"({exp_g['BINGO_NUM_ARRAY_SHAPES']} shapes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
