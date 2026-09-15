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
import os
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

    # The C/D port declares its 32 channels as a spatial NEST, and the streamer writes one
    # stride per declared dimension. The kernel has to size its stride array to match: a
    # short array leaves the remaining dimensions reading whatever follows it on the stack,
    # which addresses garbage rather than faulting. Check both the dimension count and the
    # innermost bound, since the ordinary GEMM layout derives sl1 = sl0 * bound0 from it.
    # snax_streamer_cfg is usually a `$ref` into a top-level template, and hjson does not
    # resolve those -- follow it by hand, accepting an inline block too.
    st = acc["snax_streamer_cfg"]
    if "$ref" in st:
        st = hwcfg[st["$ref"].lstrip("#/").split("/")[-1]]
    rw = st["data_reader_writer_params"]["spatial_bounds"][0]

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
        "BINGO_CD_SPATIAL_NUM":    len(rw),
        "BINGO_CD_SPATIAL_BOUND0": int(rw[0]),
        # Was in the header but NOT derived here, so a cfg that changed it would have
        # passed validation against a stale value.
        "BINGO_GRANULARITY_A":    int(acc.get("granularity_a", 1)),
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

# ----------------------------------------------------------------------
# Emit the header from the cfg.
# ----------------------------------------------------------------------
def emit_header(hwcfg_path: pathlib.Path, g: dict, shapes: list[dict]) -> str:
    """Render gemm_shapes.h. Every value here comes from expected_from_hwcfg().

    The prose is carried over from the hand-written header this replaces, because the
    formulas are the part a reader needs and they do not change with the cfg. The NUMBERS
    in it are interpolated, so the commentary cannot drift from the table the way
    "[4, 8], 32 channels" did when the port was narrowed to 16.
    """
    def mask_list(v):
        return "{ " + ", ".join(f"0x{x:x}u" for x in v) + " }"

    sp = ", ".join(f"[{s['meshRow']}, {s['tileSize']}, {s['meshCol']}]" for s in shapes)
    L = []
    L.append("// Copyright 2025 KU Leuven.")
    L.append("// Licensed under the Apache License, Version 2.0, see LICENSE for details.")
    L.append("// SPDX-License-Identifier: Apache-2.0")
    L.append("//")
    L.append("// GENERATED by device/runtime/snax/versacore/validate_shapes.py --emit.")
    L.append("// DO NOT EDIT: `make sw` regenerates it whenever the cluster cfg changes.")
    L.append(f"// Source cfg : {hwcfg_path}")
    L.append(f"// Array shape: (meshRow, tileSize, meshCol) = {sp}")
    L.append("//")
    L.append("// This table is a pure function of the cluster hjson. It used to be")
    L.append("// hand-written and cross-checked, which meant every cfg change failed the")
    L.append("// build and then had to be re-derived by hand; now the same derivation runs")
    L.append("// once and writes the answer. The check remains available as --check, and")
    L.append("// CI can use it to prove a committed header still matches its cfg.")
    L.append("//")
    L.append("// Derivation (bw := BINGO_BANK_WIDTH, ceil_8(x) := ((x + 7) / 8) * 8):")
    L.append("//")
    L.append("//   Ctlbound0    = max(1, meshRow*meshCol*BINGO_C_ELEM_LEN / BINGO_SERIAL_C_D_WIDTH)")
    L.append("//   Ctlstride0   = min(BINGO_SERIAL_C_D_WIDTH, meshRow*meshCol*BINGO_C_ELEM_LEN)")
    L.append("//                  / bw * (bw / 8)")
    L.append("//   D32tlbound0  = Ctlbound0 ;  D32tlstride0 = Ctlstride0")
    L.append("//")
    L.append("//   channel_en_A: bits = max(8, ceil_8(meshRow*tileSize*BINGO_A_ELEM_LEN / bw))")
    L.append("//   channel_en_B: bits = max(8, ceil_8(meshCol*tileSize*BINGO_B_ELEM_LEN / bw))")
    L.append("//   channel_en_C: bits =        ceil_8(meshRow*meshCol *BINGO_C_ELEM_LEN / bw)")
    L.append("//   channel_en_D32: same bits as channel_en_C")
    L.append("")
    L.append("#pragma once")
    L.append("")
    L.append("#include <stdint.h>")
    L.append("")
    for k in ("BINGO_BANK_WIDTH", "BINGO_A_ELEM_LEN", "BINGO_B_ELEM_LEN",
              "BINGO_C_ELEM_LEN", "BINGO_D32_ELEM_LEN", "BINGO_SERIAL_C_D_WIDTH"):
        L.append(f"#define {k:<25} {g[k]}")
    L.append("")
    L.append("// A-reader sparse-interconnect access granularity, in banks. The A reader's")
    L.append("// sparse TCDM crossbar wires read-port i only to banks of parity")
    L.append("// (i % BINGO_GRANULARITY_A), so each A-reader K-tile stride must be a multiple")
    L.append("// of this many banks -- otherwise a later K step walks port 0 onto an unroutable")
    L.append("// bank and SparseInterconnect raises \"Illegal bank access\". A DENSE TCDM")
    L.append("// (tcdm.sparse_interconnect = false) gives 1, and no stride needs rounding.")
    L.append(f"#define {'BINGO_GRANULARITY_A':<25} {g['BINGO_GRANULARITY_A']}")
    L.append("")
    L.append("// Per-stream CSR counts: how many uint32 words each channel-enable mask spans.")
    for k in ("BINGO_A_CSR_NUM", "BINGO_B_CSR_NUM", "BINGO_C_CSR_NUM", "BINGO_D32_CSR_NUM"):
        L.append(f"#define {k:<25} {g[k]}")
    L.append("")
    L.append("// SPATIAL geometry of the C/D port. The one bidirectional reader_writer declares")
    L.append("// its channels as a nest, and the streamer reads ONE STRIDE PER DIMENSION:")
    L.append("//")
    L.append("//     channel i sits at  sl0 * (i % B0)  +  sl1 * ((i / B0) % B1)")
    L.append("//")
    L.append(f"// This cfg declares {g['BINGO_CD_SPATIAL_NUM']} dimension(s) with B0 = {g['BINGO_CD_SPATIAL_BOUND0']}.")
    L.append("// Passing fewer strides than dimensions is not a smaller mistake than passing")
    L.append("// none: the streamer loop reads S_STRIDE_NUM_READER_WRITER_* entries whatever the")
    L.append("// caller provided, so a short array feeds the remaining dimensions from whatever")
    L.append("// follows it on the stack and those channels address garbage without faulting.")
    L.append(f"#define {'BINGO_CD_SPATIAL_NUM':<25} {g['BINGO_CD_SPATIAL_NUM']}")
    L.append(f"#define {'BINGO_CD_SPATIAL_BOUND0':<25} {g['BINGO_CD_SPATIAL_BOUND0']}")
    L.append("")
    L.append(f"#define {'BINGO_NUM_ARRAY_SHAPES':<25} {g['BINGO_NUM_ARRAY_SHAPES']}")
    L.append("")
    L.append("typedef struct {")
    for f in ("meshRow", "tileSize", "meshCol", "Ctlbound0", "Ctlstride0",
              "D32tlbound0", "D32tlstride0"):
        L.append(f"    uint32_t {f};")
    L.append("    uint32_t channel_en_A  [BINGO_A_CSR_NUM];")
    L.append("    uint32_t channel_en_B  [BINGO_B_CSR_NUM];")
    L.append("    uint32_t channel_en_C  [BINGO_C_CSR_NUM];")
    L.append("    uint32_t channel_en_D32[BINGO_D32_CSR_NUM];")
    L.append("} bingo_gemm_shape_params_t;")
    L.append("")
    L.append("static const bingo_gemm_shape_params_t")
    L.append("    bingo_gemm_shape_params[BINGO_NUM_ARRAY_SHAPES] = {")
    for i, sh in enumerate(shapes):
        L.append(f"    [{i}] = {{")
        for f in ("meshRow", "tileSize", "meshCol", "Ctlbound0", "Ctlstride0",
                  "D32tlbound0", "D32tlstride0"):
            L.append(f"        .{f:<12} = {sh[f]}u,")
        for f in ("channel_en_A", "channel_en_B", "channel_en_C", "channel_en_D32"):
            L.append(f"        .{f:<12} = {mask_list(sh[f])},")
        L.append("    },")
    L.append("};")
    L.append("")
    L.append("// All-zero C mask: selects \"add nothing\" without streaming a zero buffer. A")
    L.append("// disabled reader channel still pops its address and the responser substitutes a")
    L.append("// zero beat, so the array sees C = 0 with no TCDM traffic at all.")
    L.append("static const uint32_t bingo_channel_en_C_null[BINGO_C_CSR_NUM] = { 0u };")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True,
                    help="path to snax_versacore_to_cluster.hjson")
    ap.add_argument("--header", type=pathlib.Path, required=True,
                    help="path to gemm_shapes.h -- written by --emit, read by --check")
    ap.add_argument("--emit", action="store_true",
                    help="GENERATE the header from the cfg (default mode for the build)")
    ap.add_argument("--check", action="store_true",
                    help="only verify an existing header against the cfg; do not write")
    args = ap.parse_args()

    if not args.hwcfg.is_file():
        print(f"error: hwcfg not found: {args.hwcfg}", file=sys.stderr)
        return 2

    hwcfg = hjson.loads(args.hwcfg.read_text())
    exp_g, exp_s = expected_from_hwcfg(hwcfg)

    if args.emit or not args.check:
        text = emit_header(args.hwcfg, exp_g, exp_s)
        # Only rewrite when the content actually changes, so an unchanged cfg does not
        # retouch the header and force every dependent TU to recompile.
        if not args.header.is_file() or args.header.read_text() != text:
            args.header.parent.mkdir(parents=True, exist_ok=True)
            # Write via a unique temp file + atomic rename. `make sw -j` can have the
            # snax-sw-gen emitter and a device app's own rule run this concurrently, and a
            # plain write_text would let a parallel compile open a half-written header.
            tmp = args.header.with_suffix(f".h.tmp.{os.getpid()}")
            tmp.write_text(text)
            os.replace(tmp, args.header)
            print(f"[gemm_shapes] wrote {args.header.name} from {args.hwcfg.name} "
                  f"({exp_g['BINGO_NUM_ARRAY_SHAPES']} shape(s), "
                  f"serial C/D {exp_g['BINGO_SERIAL_C_D_WIDTH']} b)")
        else:
            print(f"[gemm_shapes] {args.header.name} already matches {args.hwcfg.name}")
        return 0

    if not args.header.is_file():
        print(f"error: header not found: {args.header}", file=sys.stderr)
        return 2
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
