#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
"""Write the HBM image tb_hemaia_hbm_model.sv expects into <outdir>/hbm/."""
import pathlib
import struct
import sys


def pattern(tag, words):
    return b"".join(struct.pack("<Q", (tag << 48) | i) for i in range(words))


out = pathlib.Path(sys.argv[1]) / "hbm"
out.mkdir(parents=True, exist_ok=True)
(out / "a.bin").write_bytes(pattern(0xA5A5, 8192))        # 64 KiB
(out / "b.bin").write_bytes(pattern(0xB0B0, 1024))        # 8 KiB
(out / "c.bin").write_bytes(pattern(0xC0C0, 512))         # another memchip's
(out / "d.bin").write_bytes(pattern(0xD0D0, 512))
(out / "e.bin").write_bytes(bytes(0xEE ^ k for k in range(100)))  # not a whole line
(out / "manifest.txt").write_text(
    "# offset        file     [chip=<id>]\n"
    "0x0             a.bin\n"
    "0x2_0000_0000   b.bin               # 8 GiB up\n"
    "0x1000_0000     c.bin    chip=0x21  # not ours\n"
    "0x1000_2000     d.bin    chip=0x20\n"
    "0x3000_0040     e.bin\n"
)
