#!/usr/bin/env python3
# Copyright 2025 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""
Structural sync check for
  target/sw/device/runtime/snax/versacore/snax_versacore_lib.h    (local)
vs
  .bender/git/checkouts/snitch_cluster-*/target/snitch_cluster/sw/snax/
      versacore-to/src/snax-versacore-to-lib.c                    (upstream)

The local header is a deliberate hand port (different types, inline bodies,
GEMMX_*→VERSACORE_* CSR family rename, extra local helpers). A textual diff
will always trip on those intentional differences, so instead we compare the
*structural* semantics that must stay in lock-step:

  1. Every function defined in upstream .c must also be defined in local .h.
     Local-only helpers (start_streamer, set_minimal_streamer_cfg, …) are
     fine — they're additions, not drift.

  2. For each common function, the ordered sequence of csrw_ss / csrr_ss
     calls must match, using only the first argument (the CSR identifier
     expression) after applying ALIAS below. The second csrw_ss arg is
     intentionally ignored: local and upstream pass deliberately different
     payloads (e.g. `A_addr` vs `delta_local_a + snrt_l1_next()`), but the
     *set of CSRs programmed and their order* must be identical.

What we do NOT check:
  - #define values for CSR addresses. Those come from the generated
    streamer_csr_addr_map.h (which the snax-sw-gen pipeline already
    regenerates per $(CFG)).
  - Non-CSR body code (e.g. the checking loops inside
    check_versacore_result_D32). The invariant the user cares about is
    which CSRs the port programs; loop bodies are free to differ.

Usage:
  check_snax_versacore_lib_sync.py <local_h> <upstream_c>

Exit 0 on match, 1 on any drift, 2 on usage/IO errors.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# GEMMX family was renamed to VERSACORE_* in the local port. Canonicalize both
# sides to the upstream spelling before comparing so the rename doesn't count
# as drift. Keep this list short and explicit — if upstream introduces a new
# rename candidate, adding it here is a deliberate decision.
ALIAS = {
    "VERSACORE_START_CSR":           "GEMMX_START",
    "VERSACORE_BUSY":                "GEMMX_BUSY",
    "VERSACORE_PERFORMANCE_COUNTER": "GEMMX_PERFORMANCE_COUNTER",
    "VERSACORE_CSR_ADDR_BASE":       "GEMMX_CSR_ADDR_BASE",
}

# Regex for a C identifier.
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*\b")


def strip_comments(src: str) -> str:
    """Strip // and /* */ comments. These files have no string literals that
    could contain comment-like sequences, but we still skip anything between
    quotes just to be safe."""
    out = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = src.find("\n", i)
            if j == -1:
                break
            i = j  # keep the newline so line numbers aren't needed here
        elif c == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            if j == -1:
                break
            i = j + 2
        elif c in ("'", '"'):
            # Skip string / char literal.
            quote = c
            j = i + 1
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if src[j] == quote:
                    j += 1
                    break
                j += 1
            out.append(src[i:j])
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _find_matching(src: str, open_idx: int, open_ch: str, close_ch: str) -> int:
    """Return index of matching close bracket, or -1."""
    depth = 1
    i = open_idx + 1
    n = len(src)
    while i < n:
        c = src[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


# Matches `<return_type> <name>(` at top level. We're lenient: any sequence of
# tokens that ends in `name(` at depth 0 qualifies as a candidate function
# header, which we then confirm by looking for a following `{` (definition) vs
# `;` (declaration).
_FUNC_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(")

# Keywords that look like a function call site but aren't. Anything here is
# never treated as a function-definition name.
_NOT_FUNC_NAMES = {
    "if", "while", "for", "switch", "return", "sizeof", "do",
    "defined",
    # Upstream/local both call these inside bodies; they're CSR helpers or
    # snrt helpers, not function definitions.
    "csrw_ss", "csrr_ss", "snrt_l1_next", "printf",
}


def extract_functions(src: str) -> dict[str, str]:
    """Return a mapping {function_name: body_without_braces}.

    A function definition is detected as:  <...> NAME ( <params> ) { <body> }
    at brace depth 0. We precompute a depths[] array for this source once so
    the outer loop is O(n) instead of quadratic, and we avoid any module-
    level cache keyed by `id(src)` — intermediate strings get GC'd and id()
    values can be recycled, which previously caused mid-call aliasing.
    """
    src = strip_comments(src)
    n = len(src)

    depths = [0] * (n + 1)
    depth = 0
    for i, c in enumerate(src):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        depths[i + 1] = depth

    funcs: dict[str, str] = {}
    for m in _FUNC_NAME_RE.finditer(src):
        name = m.group(1)
        if name in _NOT_FUNC_NAMES:
            continue
        open_paren = m.end() - 1  # position of `(`
        if depths[open_paren] != 0:
            continue
        close_paren = _find_matching(src, open_paren, "(", ")")
        if close_paren == -1:
            continue
        i = close_paren + 1
        while i < n and src[i].isspace():
            i += 1
        if i >= n or src[i] != "{":
            continue  # pure declaration or something else
        close_brace = _find_matching(src, i, "{", "}")
        if close_brace == -1:
            continue
        funcs[name] = src[i + 1:close_brace]
    return funcs


def _apply_alias(expr: str) -> str:
    """Replace ALIAS keys occurring as full identifiers in `expr`."""
    def sub(m: re.Match) -> str:
        name = m.group(0)
        return ALIAS.get(name, name)
    return _IDENT_RE.sub(sub, expr)


def _first_arg(src: str, lparen: int) -> str | None:
    """Given position of `(`, return the first argument (up to the first
    top-level `,` or `)`), stripped. Depth-tracked for nested parens."""
    depth = 1
    i = lparen + 1
    start = i
    n = len(src)
    while i < n:
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return src[start:i].strip()
        elif c == "," and depth == 1:
            return src[start:i].strip()
        i += 1
    return None


def csr_calls(body: str) -> list[str]:
    """Extract an ordered list of `csrw_ss:<expr>` / `csrr_ss:<expr>` tokens
    from a function body. The <expr> is alias-normalized and whitespace-
    collapsed so trivial formatting doesn't count as drift."""
    calls: list[str] = []
    for m in re.finditer(r"\b(csrw_ss|csrr_ss)\s*\(", body):
        fn = m.group(1)
        lparen = m.end() - 1
        arg = _first_arg(body, lparen)
        if arg is None:
            continue
        arg = _apply_alias(arg)
        # Collapse runs of whitespace so `T_BOUND_BASE_READER_0  +  i` matches
        # `T_BOUND_BASE_READER_0 + i`.
        arg = re.sub(r"\s+", " ", arg)
        calls.append(f"{fn}:{arg}")
    return calls


def compare(local_h: Path, upstream_c: Path) -> int:
    try:
        local_src = local_h.read_text()
        upstream_src = upstream_c.read_text()
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    local_fns = extract_functions(local_src)
    upstream_fns = extract_functions(upstream_src)

    problems: list[str] = []

    # 1. Every upstream fn must exist locally.
    missing = sorted(set(upstream_fns) - set(local_fns))
    if missing:
        problems.append(
            "Functions defined upstream but missing locally — port required:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    # 2. For each common function, compare the CSR call sequence.
    common = sorted(set(upstream_fns) & set(local_fns))
    for name in common:
        up_calls = csr_calls(upstream_fns[name])
        lo_calls = csr_calls(local_fns[name])
        if up_calls != lo_calls:
            problems.append(_format_call_diff(name, up_calls, lo_calls))

    if problems:
        print(
            f"error: snax_versacore_lib.h diverged from snax-versacore-to-lib.c\n"
            f"  local    : {local_h}\n"
            f"  upstream : {upstream_c}\n",
            file=sys.stderr,
        )
        for p in problems:
            print(p, file=sys.stderr)
            print("", file=sys.stderr)
        print(
            "Action: re-review the upstream source and port any relevant\n"
            "changes into the local header (respecting the GEMMX_*→VERSACORE_*\n"
            "rename). When the structural sequences match, this check passes.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[check-snax-versacore-sync] OK "
        f"({len(common)} shared functions match; "
        f"{len(set(local_fns) - set(upstream_fns))} local-only helpers ignored)"
    )
    return 0


def _format_call_diff(name: str, upstream: list[str], local: list[str]) -> str:
    """Return a human-readable side-by-side diff of two CSR-call lists."""
    lines = [f"Function `{name}` CSR access sequence drifted:"]
    width = max([len(c) for c in upstream] + [0]) + 2
    hdr = f"  {'upstream (.c)'.ljust(width)}   local (.h)"
    lines.append(hdr)
    lines.append("  " + "-" * (width + 15))
    for i in range(max(len(upstream), len(local))):
        u = upstream[i] if i < len(upstream) else "<missing>"
        l = local[i] if i < len(local) else "<missing>"
        mark = "  " if u == l else "* "
        lines.append(f"{mark}{u.ljust(width)}   {l}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("local_h", type=Path,
                    help="target/sw/device/runtime/snax/versacore/snax_versacore_lib.h")
    ap.add_argument("upstream_c", type=Path,
                    help="snax-versacore-to-lib.c in the bender-managed checkout")
    args = ap.parse_args()

    for p in (args.local_h, args.upstream_c):
        if not p.is_file():
            print(f"error: not a file: {p}", file=sys.stderr)
            return 2

    return compare(args.local_h, args.upstream_c)


if __name__ == "__main__":
    sys.exit(main())
