#!/usr/bin/env python3
# Copyright 2025 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""
Emit the HW-parameterized bingo GEMM kernels directly as C, bypassing the
previous versacore_hw_param.h macro layer.

Invoked twice by the libsnaxkernel Makefile — once per kernel layer:

  Cluster-level (bingo-sw) :
    --template offload_sw_kernels/gemm_kernels.template.c
    --out      offload_sw_kernels/gemm.h
  Core-level (bingo-hw)    :
    --template offload_hw_kernels/gemm_kernels.template.c
    --out      offload_hw_kernels/gemm.h

Inputs:
  - `--template`: committed template containing hand-written kernel bodies
    that still reference the macros (meshRow_N, channel_en_A_N_M, …) and
    dispatch via `switch (array_shape_idx)` / `if (array_shape_idx == …)`.
  - `--hwcfg` snax_versacore_to_cluster.hjson: the active cluster config.

Output:
  - `--out`: a committed-friendly emitted header with
    (1) literal integers in place of every macro and
    (2) `switch (array_shape_idx)` blocks containing only the cases present
        in the hwcfg (no padding to MAX_NUM_SHAPES) plus a `default:` arm
        returning BINGO_RET_FAIL for the core-level kernels (cluster-level
        kernels already have their own `return;` error arms in-template).

Both generated files get a `#ifndef BINGO_NUM_ARRAY_SHAPES` guard around
the shape-count `#define` so the TU that includes both headers doesn't hit
a redefinition error.

The generated files are gitignored (see libsnaxkernel/.gitignore). Do not
hand-edit them. Edit the corresponding `gemm_kernels.template.c`.
"""
from __future__ import annotations

import argparse
import datetime
import math
import pathlib
import re
import sys

import hjson


# -------------------------------------------------------------------------
# Shape-parameter computation — mirrors the old gen_versacore_hw_param.py
# so we emit the same integers the macro pipeline used to produce.
# -------------------------------------------------------------------------
def _channel_enable_bits(bits: int, csr_num: int) -> list[int]:
    """Return a list of uint32 CSR values forming a bitmask with the low
    `bits` bits set, packed most-significant-CSR first (matches
    gen_channel_enable_CSR in the legacy script)."""
    out = [0] * csr_num
    for i in range(bits):
        idx = i // 32
        pos = i % 32
        if idx < csr_num:
            out[idx] |= 1 << pos
    out.reverse()
    return [int(x) for x in out]


def compute_shape_params(hwcfg: dict) -> tuple[dict, list[dict]]:
    acc_cfg = hwcfg["snax_versacore_core_template"]["snax_acc_cfg"][0]
    spatial_unrolling = acc_cfg["snax_versacore_spatial_unrolling"][0]

    a_len = acc_cfg["snax_versacore_input_a_element_width"][0]
    b_len = acc_cfg["snax_versacore_input_b_element_width"][0]
    c_len = acc_cfg["snax_versacore_input_c_element_width"][0]
    d_len = acc_cfg["snax_versacore_output_d_element_width"][0]

    a_array_width = acc_cfg["snax_versacore_array_input_a_width"]
    b_array_width = acc_cfg["snax_versacore_array_input_b_width"]
    c_array_width = acc_cfg["snax_versacore_array_input_c_width"]
    d_array_width = acc_cfg["snax_versacore_array_output_d_width"]
    assert c_array_width == d_array_width, "C and D array width must be the same"

    serial_c_d_width = acc_cfg["snax_versacore_serial_c_d_width"]
    bank_width = 64

    globals_ = {
        "A_elem_len": a_len,
        "B_elem_len": b_len,
        "C_elem_len": c_len,
        "D32_elem_len": d_len,
        "bankWidth": bank_width,
        "versacore_serial_c_d_width": serial_c_d_width,
    }

    a_csr_num = int(math.ceil(a_array_width / bank_width / 32))
    b_csr_num = int(math.ceil(b_array_width / bank_width / 32))
    c_csr_num = int(math.ceil(serial_c_d_width / bank_width / 32))
    d_csr_num = int(math.ceil(serial_c_d_width / bank_width / 32))

    shapes: list[dict] = []
    for idx, (mesh_row, tile_size, mesh_col) in enumerate(spatial_unrolling):
        a_bits = max(8, int((mesh_row * tile_size * a_len / bank_width + 7) // 8 * 8))
        b_bits = max(8, int((mesh_col * tile_size * b_len / bank_width + 7) // 8 * 8))
        c_bits = int((mesh_row * mesh_col * c_len / bank_width + 7) // 8 * 8)

        channel_en_a = _channel_enable_bits(a_bits, a_csr_num)
        channel_en_b = _channel_enable_bits(b_bits, b_csr_num)
        channel_en_c = _channel_enable_bits(c_bits, c_csr_num)
        channel_en_c_null = _channel_enable_bits(0, c_csr_num)
        channel_en_d32 = _channel_enable_bits(c_bits, d_csr_num)

        if mesh_col * mesh_row * c_len >= serial_c_d_width:
            c_spatial_bound_0 = serial_c_d_width / bank_width
        else:
            c_spatial_bound_0 = mesh_col * mesh_row * c_len / bank_width
        ctlstride0 = int(c_spatial_bound_0 * (bank_width / 8))
        ctlbound0 = max(1, int(mesh_col * mesh_row * c_len / serial_c_d_width))
        d32tlstride0 = ctlstride0
        d32tlbound0 = ctlbound0

        shapes.append({
            "index": idx,
            "meshRow": int(mesh_row),
            "tileSize": int(tile_size),
            "meshCol": int(mesh_col),
            "channel_en_A": channel_en_a,       # len == a_csr_num
            "channel_en_B": channel_en_b,       # len == b_csr_num
            "channel_en_C": channel_en_c,       # len == c_csr_num
            "channel_en_C_null": channel_en_c_null,
            "channel_en_D32": channel_en_d32,   # len == d_csr_num
            "Ctlstride0": ctlstride0,
            "Ctlbound0": ctlbound0,
            "D32tlstride0": d32tlstride0,
            "D32tlbound0": d32tlbound0,
        })
    return globals_, shapes


# -------------------------------------------------------------------------
# Substitution table — maps every macro identifier the template references
# to its concrete integer value (or to 0 for shape slots the hwcfg doesn't
# define, so the text substitution is total; those case blocks are stripped
# by remove_unused_cases below so the 0 is never reached).
# -------------------------------------------------------------------------
MAX_TEMPLATE_SHAPES = 5  # template currently dispatches case 0..4


def build_substitutions(globals_: dict, shapes: list[dict]) -> dict[str, str]:
    subs: dict[str, str] = {k: str(v) for k, v in globals_.items()}

    by_idx = {s["index"]: s for s in shapes}

    def lookup(i: int, field: str, sub_field: int | None = None) -> str:
        if i not in by_idx:
            return "0u"
        val = by_idx[i][field]
        if sub_field is not None:
            val = val[sub_field]
        return f"{int(val):#x}u" if isinstance(val, int) and field.startswith("channel_en") \
            else str(int(val))

    for i in range(MAX_TEMPLATE_SHAPES):
        subs[f"meshRow_{i}"]  = lookup(i, "meshRow")
        subs[f"tileSize_{i}"] = lookup(i, "tileSize")
        subs[f"meshCol_{i}"]  = lookup(i, "meshCol")
        subs[f"Ctlstride0_{i}"]   = lookup(i, "Ctlstride0")
        subs[f"Ctlbound0_{i}"]    = lookup(i, "Ctlbound0")
        subs[f"D32tlstride0_{i}"] = lookup(i, "D32tlstride0")
        subs[f"D32tlbound0_{i}"]  = lookup(i, "D32tlbound0")
        # channel_en_A_{i}_0, channel_en_C_{i}_0, channel_en_C_null_{i}_0,
        # channel_en_D32_{i}_0: single-CSR families. channel_en_B_{i}_{0|1}
        # has two CSRs. We dynamically emit entries for whatever CSR count
        # the hwcfg produced so the template's `channel_en_X_{i}_{j}` refs
        # resolve to the right integers.
        if i in by_idx:
            for j, v in enumerate(by_idx[i]["channel_en_A"]):
                subs[f"channel_en_A_{i}_{j}"] = f"{int(v):#x}u"
            for j, v in enumerate(by_idx[i]["channel_en_B"]):
                subs[f"channel_en_B_{i}_{j}"] = f"{int(v):#x}u"
            for j, v in enumerate(by_idx[i]["channel_en_C"]):
                subs[f"channel_en_C_{i}_{j}"] = f"{int(v):#x}u"
            for j, v in enumerate(by_idx[i]["channel_en_C_null"]):
                subs[f"channel_en_C_null_{i}_{j}"] = f"{int(v):#x}u"
            for j, v in enumerate(by_idx[i]["channel_en_D32"]):
                subs[f"channel_en_D32_{i}_{j}"] = f"{int(v):#x}u"
        else:
            # Emit harmless 0 fallbacks; these slots sit inside unused case
            # bodies that remove_unused_cases will strip anyway.
            for fam in ("channel_en_A", "channel_en_B",
                        "channel_en_C", "channel_en_C_null", "channel_en_D32"):
                for j in range(2):  # 2 is the max CSR-count across families
                    subs[f"{fam}_{i}_{j}"] = "0u"
    return subs


# -------------------------------------------------------------------------
# Text rewriting: drop case N: bodies where N >= num_shapes from every
# `switch (array_shape_idx) { ... }` block, then substitute macros.
# -------------------------------------------------------------------------
_SWITCH_RE = re.compile(r"switch\s*\(\s*array_shape_idx\s*\)\s*\{")
_CASE_RE = re.compile(r"\bcase\s+(\d+)\s*:")
_DEFAULT_RE = re.compile(r"\bdefault\s*:")


def _find_matching(text: str, open_idx: int, open_ch: str, close_ch: str) -> int:
    depth = 1
    i = open_idx + 1
    n = len(text)
    while i < n:
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unmatched brace")


def _split_switch_labels(body: str) -> list[tuple[int, str, str]]:
    """Split a switch body into (label_kind, label_text, case_body) triples
    at top level. `label_kind` is the case number for `case N:` or None for
    `default:`. Case body text excludes the label itself."""
    # Walk the body tracking brace/paren depth. Record label positions.
    # Anything before the first label is a prologue (usually just whitespace);
    # we attach it to "pre-switch" and return separately.
    depth_brace = 0
    depth_paren = 0
    labels: list[tuple[int | None, int, int]] = []  # (kind, start, label_end)
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == "{":
            depth_brace += 1
            i += 1
            continue
        if c == "}":
            depth_brace -= 1
            i += 1
            continue
        if c == "(":
            depth_paren += 1
            i += 1
            continue
        if c == ")":
            depth_paren -= 1
            i += 1
            continue
        if depth_brace == 0 and depth_paren == 0:
            m = _CASE_RE.match(body, i)
            if m:
                labels.append((int(m.group(1)), m.start(), m.end()))
                i = m.end()
                continue
            m = _DEFAULT_RE.match(body, i)
            if m:
                labels.append((None, m.start(), m.end()))
                i = m.end()
                continue
        i += 1

    # Build segments: each label runs until the next label's start (or end).
    result: list[tuple[int | None, str, str]] = []
    for idx, (kind, start, label_end) in enumerate(labels):
        next_start = labels[idx + 1][1] if idx + 1 < len(labels) else n
        label_text = body[start:label_end]
        case_body = body[label_end:next_start]
        result.append((kind, label_text, case_body))
    return result


def remove_unused_cases(text: str, num_shapes: int) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _SWITCH_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.end()])  # include up to and incl. `{`
        brace_open = m.end() - 1
        brace_close = _find_matching(text, brace_open, "{", "}")
        body = text[brace_open + 1:brace_close]
        segments = _split_switch_labels(body)
        if not segments:
            # No labels found — leave body as-is.
            out.append(body)
        else:
            # Prologue (before first label) preserved verbatim.
            prologue = body[:segments[0][1] if False else 0]
            # Actually, recompute: segments[0][1] is label text; we need the
            # original start offset. Find it again via _CASE_RE/_DEFAULT_RE.
            # Simpler: splicing directly from labels list.
            first_label_start = None
            for km in _CASE_RE.finditer(body):
                first_label_start = km.start()
                break
            for dm in _DEFAULT_RE.finditer(body):
                if first_label_start is None or dm.start() < first_label_start:
                    first_label_start = dm.start()
                break
            if first_label_start is not None:
                out.append(body[:first_label_start])
            kept_any_case = False
            has_default = False
            for kind, label_text, case_body in segments:
                if kind is None:
                    has_default = True
                    out.append(label_text)
                    out.append(case_body)
                elif kind < num_shapes:
                    kept_any_case = True
                    out.append(label_text)
                    out.append(case_body)
                # else: skip unused case block entirely
            if not has_default:
                # Inject a default arm that fails on out-of-range shape; this
                # protects the kernel now that the template's 0..4 cases can
                # shrink to whatever the hwcfg actually exposes.
                out.append(
                    "\n    default:\n"
                    "        VERSACORE_DEBUG_PRINT("
                    "\"Error! array_shape_idx=%d invalid (only %d shapes in hwcfg)\\r\\n\", "
                    "array_shape_idx, BINGO_NUM_ARRAY_SHAPES);\n"
                    "        return BINGO_RET_FAIL;\n"
                )
            # Suppress -Wunused-variable if a switch kept no case (edge
            # case, shouldn't happen in practice).
            if not kept_any_case and not has_default:
                out.append("    (void)array_shape_idx;\n")
        out.append("}")
        i = brace_close + 1
    return "".join(out)


# -------------------------------------------------------------------------
# Identifier-level text substitution. Uses a word boundary regex so we don't
# clobber longer identifiers that happen to share a prefix (meshRow_1 vs
# meshRow_10, etc. — harmless today but cheap insurance).
# -------------------------------------------------------------------------
_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\b")


def apply_substitutions(text: str, subs: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1)
        return subs.get(name, name)
    return _IDENT.sub(repl, text)


# -------------------------------------------------------------------------
# Some C tokens collide with substitution keys (e.g. `case 0:` → after
# substitution becomes `case 0:` — fine). But a function parameter named
# `array_shape` (in set_versacore_streamer_csr) is NOT a macro and must be
# kept intact. Our subs table only contains macro names, so unrelated
# identifiers pass through unchanged.
# -------------------------------------------------------------------------


def render(template: pathlib.Path, hwcfg_path: pathlib.Path,
           hwcfg: dict) -> str:
    globals_, shapes = compute_shape_params(hwcfg)
    subs = build_substitutions(globals_, shapes)
    num_shapes = len(shapes)

    body = template.read_text()
    body = remove_unused_cases(body, num_shapes)
    body = apply_substitutions(body, subs)

    banner = (
        "// Copyright 2025 KU Leuven.\n"
        "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "//\n"
        "// AUTO-GENERATED by libsnaxkernel/gen_bingo_gemm_kernels.py.\n"
        "// DO NOT EDIT — any changes will be overwritten on the next `make sw`.\n"
        "//\n"
        f"// Source cfg : {hwcfg_path}\n"
        f"// Template   : {template}\n"
        f"// Generated  : {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')}\n"
        "//\n"
        f"// Shape count: {num_shapes}\n"
        f"// Shapes     : "
        + ", ".join(f"[{s['meshRow']},{s['tileSize']},{s['meshCol']}]" for s in shapes)
        + "\n"
        "\n"
        "#pragma once\n"
        "\n"
        "#include \"../macros.h\"\n"
        "#include <snax_versacore_lib.h>\n"
        "\n"
        "// Guarded because both the cluster-level and core-level generated\n"
        "// headers include this define — the translation unit includes both.\n"
        "#ifndef BINGO_NUM_ARRAY_SHAPES\n"
        f"#define BINGO_NUM_ARRAY_SHAPES {num_shapes}\n"
        "#endif\n"
        "\n"
    )
    return banner + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="path to emit the generated gemm.h "
                         "(offload_sw_kernels/ or offload_hw_kernels/)")
    ap.add_argument("--template", type=pathlib.Path, required=True,
                    help="path to the committed gemm_kernels.template.c "
                         "(cluster- or core-level)")
    args = ap.parse_args()

    template = args.template
    if not template.is_file():
        print(f"error: template not found: {template}", file=sys.stderr)
        return 2
    if not args.hwcfg.is_file():
        print(f"error: hwcfg not found: {args.hwcfg}", file=sys.stderr)
        return 2

    hwcfg = hjson.loads(args.hwcfg.read_text())
    out_text = render(template, args.hwcfg, hwcfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_text)
    print(f"[gen_bingo_gemm_kernels] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
