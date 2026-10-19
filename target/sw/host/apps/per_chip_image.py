#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Builds the per-chiplet host images of a bingo workload with params per_chip_image.

The generated offload header holds every chiplet's block behind `#if BINGO_ONLY_CHIP == <id>`, and the
generator writes the chiplet list (per_chip_ids) beside it. Every chiplet's image is that header compiled
with -DBINGO_ONLY_CHIP=<id> and linked with the app's own host.ld (generated from runtime/host.ld.tpl, the one
layout every host image uses), so its L3 holds the shared code and data and its own tables only, not every
chiplet's. The C sources are compiled once per chiplet, in
parallel (each compile optimises one chiplet's block); the assembly once.

Each image is linked twice: unpadded, which measures the region that differs between chiplets (code,
read-only data, the chiplet's own tables), then padded to the largest, so every symbol outside that region
-- the workload's L3 data, the runtime's globals, the BSS, the device binary (snitch_main) and the L3 heap
-- sits at one address in every image. The check refuses the build when one does not, or when writable
data other than a chiplet's own tables lands in the varying region. --gc-sections is off for these links, so a
symbol only some chiplets' code uses (the root's check goldens) is in every image and moves nothing.

Outputs: <out stem>_chip<id>.elf per chiplet (and .bin with --bin), and <out> itself, a copy of the first
chiplet's image, for the steps that expect one ELF (registration of the device origin, dumps, staging).
"""

import argparse
import concurrent.futures as cf
import os
import re
import shlex
import shutil
import subprocess
import sys


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode:
        sys.stdout.write(" ".join(shlex.quote(c) for c in cmd) + "\n" + p.stdout)
        raise SystemExit(f"[per-chip] failed: {cmd[0]} (exit {p.returncode})")
    return p.stdout


def symbols(nm, elf):
    """{name: (local address, type)} of the image's defined symbols, the chip id (bits 40 and up) dropped so the
    images of different chiplets compare (a name defined twice keeps the first)."""
    out = {}
    for line in run([nm, "-n", elf]).splitlines():
        f = line.split()
        if len(f) == 3:
            out.setdefault(f[2], (int(f[0], 16) & ((1 << 40) - 1), f[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", required=True, help="the per_chip_ids file the generator wrote")
    ap.add_argument("--cc", required=True)
    ap.add_argument("--cflags", default="")
    ap.add_argument("--ldflags", default="")
    ap.add_argument("--define", action="append", default=[], help="an extra -D for every compile")
    ap.add_argument("--objcopy", default=None, help="with --bin: the objcopy that writes the .bin files")
    ap.add_argument("--bin", action="store_true", help="also write <out stem>_chip<id>.bin")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", required=True)
    ap.add_argument("srcs", nargs="+")
    a = ap.parse_args()

    ids = open(a.ids).read().split()
    if not ids:
        raise SystemExit(f"[per-chip] {a.ids} lists no chiplet")
    cflags = shlex.split(a.cflags) + [f"-D{d}" for d in a.define]
    all_ld = shlex.split(a.ldflags)
    scripts = [f[2:] for f in all_ld if f.startswith("-T")]
    if len(scripts) != 1:
        raise SystemExit(f"[per-chip] --ldflags names {len(scripts)} linker scripts (-T); one, the app's host.ld")
    ld_path = scripts[0]
    ldflags = [f for f in all_ld if not f.startswith("-T") and f != "-Wl,--gc-sections"] + ["-Wl,--no-gc-sections"]
    nm = a.cc[:-len("gcc")] + "nm" if a.cc.endswith("gcc") else "nm"
    stem = a.out[:-4] if a.out.endswith(".elf") else a.out
    objdir = stem + "_chips"
    os.makedirs(objdir, exist_ok=True)
    c_srcs = [s for s in a.srcs if s.endswith(".c")]
    asm_srcs = [s for s in a.srcs if not s.endswith(".c")]
    tag = lambda c: c[2:] if c.startswith("0x") else c
    elf = lambda c, sfx="": f"{stem}_chip{tag(c)}{sfx}.elf"

    # Each image is linked at its own chiplet's global addresses (chip id << 40 above the local ones). A host runs
    # at those addresses (its PC carries its chip id), so a PC-relative address names its own memory, but an
    # address fixed at link time -- a pointer in static data (the kernel-name table), a jump table -- names chip
    # 0x00 when linked at the local addresses: with one image for every chip, chip 0x00 held the same bytes there
    # and the access only went remote; with per-chiplet images it reads another image. Linked at its own prefix,
    # every address in an image is its own chiplet's; the generated code already masks the low 40 bits before it
    # puts another chiplet's prefix on an address.
    ld_text = open(ld_path).read()
    regions = {m.group(1): (m.group(0), int(m.group(2), 0), int(m.group(3), 0))
               for m in re.finditer(r"(\w+) \(rwxa\) : ORIGIN = (\w+), LENGTH = (\w+)", ld_text)}
    if "WIDE_SPM" not in regions:
        raise SystemExit(f"[per-chip] {ld_path}: no WIDE_SPM region")

    def ld_script(c):
        path = os.path.join(objdir, f"host_chip{tag(c)}.ld")
        text = ld_text
        for region, (line, origin, length) in regions.items():
            text = text.replace(line, f"{region} (rwxa) : ORIGIN = {(int(c, 16) << 40) + origin:#x}, "
                                      f"LENGTH = {length}")
        open(path, "w").write(text)
        return path

    def obj(src, c=None):
        base = os.path.splitext(os.path.basename(src))[0]
        return os.path.join(objdir, f"{base}.o" if c is None else f"{base}_chip{tag(c)}.o")

    jobs = []
    for s in asm_srcs:
        jobs.append([a.cc] + cflags + ["-c", s, "-o", obj(s)])
    # -fno-toplevel-reorder keeps every file-scope static variable, used or not (GCC drops unused ones, so a
    # static only the root's checks touch would exist in the root's image alone and move the data after it)
    for c in ids:
        for s in c_srcs:
            jobs.append([a.cc] + cflags + ["-fno-toplevel-reorder", f"-DBINGO_ONLY_CHIP={c}", "-c", s,
                                           "-o", obj(s, c)])
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        list(ex.map(run, jobs))

    # every image asks the libraries for the same symbols -- the union of what any chiplet's code needs --
    # so every image takes the same archive members, and their data (the runtime's globals) sits at the same
    # place in every image; otherwise a member only the root's checks pull in shifts what follows it
    undef = set()
    for o in [obj(s, c) for c in ids for s in c_srcs] + [obj(s) for s in asm_srcs]:
        undef |= {ln.split()[-1] for ln in run([nm, "-u", o]).splitlines() if ln.strip()}
    want = [f"-Wl,--undefined={u}" for u in sorted(undef)]

    def link(c, out, size=None):
        objs = [obj(s, c) for s in c_srcs] + [obj(s) for s in asm_srcs]
        pad = [f"-Wl,--defsym=__bingo_var_size={size}"] if size is not None else []
        run([a.cc] + cflags + want + objs + ldflags + [f"-T{ld_script(c)}"] + pad + ["-o", out])
        return out

    # 1. unpadded: the size of every chiplet's varying region
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        list(ex.map(lambda c: link(c, elf(c, ".unpadded")), ids))
    var = {}
    for c in ids:
        s = symbols(nm, elf(c, ".unpadded"))
        var[c] = s["__bingo_var_end"][0] - s["__bingo_var_start"][0]
        os.remove(elf(c, ".unpadded"))
    size = -(-max(var.values()) // 64) * 64
    # 2. padded to the largest
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        list(ex.map(lambda c: link(c, elf(c), size), ids))

    # 3. the check: one address for everything outside the varying region
    syms = {c: symbols(nm, elf(c)) for c in ids}
    first = ids[0]
    v0 = syms[first]["__bingo_var_start"][0]
    problems = []
    for c in ids:
        vs = syms[c]["__bingo_var_start"][0]
        if vs != v0:
            problems.append(f"chip {c}: varying region at {vs:#x}, chip {first}'s at {v0:#x}")
            continue
        # a .bss symbol at the region's very start is the end of .bss (__bss_end when .bss ends where the varying
        # region begins), not inside it: it is checked like every other shared symbol
        inside = lambda adr, t="": vs <= adr < vs + size and not (adr == vs and t in "Bb")
        cdata = {n for n in syms[c] if n.startswith("__bingo_args_image_chip")}
        for n, (adr, t) in syms[c].items():
            if n == "__bingo_var_end":           # where this image's own varying part ends
                continue
            if inside(adr, t):
                if t in "DdBbGgSs" and n not in cdata:
                    problems.append(f"chip {c}: writable symbol {n} ({t}) is in the varying region")
                continue
            ref = syms[first].get(n)
            if ref is None or ref[0] != adr:
                problems.append(f"chip {c}: {n} at {adr:#x}, chip {first}: "
                                + (f"{ref[0]:#x}" if ref else "absent"))
        for n, (adr, t0) in syms[first].items():
            if not inside(adr, t0) and n not in syms[c] and n != "__bingo_var_end":
                problems.append(f"chip {c}: {n} absent (chip {first}: {adr:#x})")
    if problems:
        for p in problems[:20]:
            print(f"[per-chip] {p}")
        raise SystemExit(f"[per-chip] {len(problems)} symbol(s) not at one address in every image")
    _, l3_base, l3_len = regions["WIDE_SPM"]
    end = max(syms[c]["__l3_heap_start"][0] for c in ids) - l3_base
    print(f"[per-chip] {len(ids)} images: varying region {min(var.values()):,} .. {max(var.values()):,} B "
          f"(padded to {size:,} B at {v0:#x}); shared symbols at one address; image end "
          f"{end:,} B of the {l3_len:,} B L3 ({l3_len - end:,} B left for the L3 heap)")

    shutil.copyfile(elf(first), a.out)
    if a.bin:
        for c in ids:
            run([a.objcopy, "-O", "binary", elf(c), f"{stem}_chip{tag(c)}.bin"])


if __name__ == "__main__":
    main()
