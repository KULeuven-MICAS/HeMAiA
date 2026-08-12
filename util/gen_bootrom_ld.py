#!/usr/bin/env python3
# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

"""Render the boot ROM linker script from its mako template.

The boot ROM hands control to the snitches by reading the entry point out of the
SoC control scratch registers, and it does so from assembly (`la t0,
__soc_ctrl_scratch0`), so the addresses have to be linker symbols. A linker
script cannot include a C header, so those addresses used to be written out by
hand -- and silently went stale the moment the register layout moved, leaving
the boot ROM reading zero and every snitch jumping to PC 0.

Derive them instead from the generated headers the host software already uses:
`occamy_base_addr.h` for the block base and `occamy_soc_ctrl.h` for the register
offsets. Both are reggen/occamygen output, so the boot ROM and the host cannot
disagree about where the scratch registers live.
"""

import argparse
import re
import sys
from pathlib import Path

from mako.template import Template

# Scratch registers the boot ROM needs, in the order the template expects them.
#   scratch0 -> snitch_main function pointer
#   scratch1 -> comm buffer pointer
#   scratch2 -> snitch exit code
NUM_SCRATCH = 3


def parse_define(header: Path, name: str) -> int:
    """Return the integer value of `#define <name> <int>` in *header*."""
    pattern = re.compile(r"^\s*#define\s+" + re.escape(name) + r"\s+(0[xX][0-9a-fA-F]+|\d+)\s*$",
                         re.MULTILINE)
    match = pattern.search(header.read_text())
    if not match:
        raise KeyError(f"{name} not found in {header}")
    return int(match.group(1), 0)


def scratch_offset(header: Path, idx: int) -> int:
    """Offset of SCRATCH_<idx>, tolerating reggen's single-register degeneration.

    With NumScratchRegs == 1 reggen drops the index from the macro name, exactly
    as occamy_memory_map.h.tpl already has to handle.
    """
    try:
        return parse_define(header, f"OCCAMY_SOC_SCRATCH_{idx}_REG_OFFSET")
    except KeyError:
        if idx == 0:
            return parse_define(header, "OCCAMY_SOC_SCRATCH_REG_OFFSET")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tpl", required=True, type=Path,
                        help="bootrom.ld.tpl mako template")
    parser.add_argument("--headers", required=True, type=Path,
                        help="directory holding the generated platform headers")
    parser.add_argument("--out", required=True, type=Path,
                        help="linker script to write")
    args = parser.parse_args()

    base_addr_h = args.headers / "occamy_base_addr.h"
    soc_ctrl_h = args.headers / "occamy_soc_ctrl.h"
    for header in (base_addr_h, soc_ctrl_h):
        if not header.is_file():
            parser.error(f"missing generated header: {header}. Build the platform "
                         f"headers before the boot ROM.")

    mem_map_h = args.headers / "occamy_memory_map.h"
    if not mem_map_h.is_file():
        parser.error(f"missing generated header: {mem_map_h}. Build the platform "
                     f"headers before the boot ROM.")

    base = parse_define(base_addr_h, "SOC_CTRL_BASE_ADDR")
    scratch = [base + scratch_offset(soc_ctrl_h, i) for i in range(NUM_SCRATCH)]

    rendered = Template(filename=str(args.tpl)).render(
        soc_ctrl_base_addr=base,
        soc_ctrl_scratch=scratch,
        # Memory regions. Sizes are emitted as hex literals rather than the linker's
        # K/M suffixes so they reproduce the header values exactly.
        # The boot ROM only ever links into the ROM itself and the narrow SPM, so
        # the wide SPM is not passed through at all.
        bootrom_base=parse_define(base_addr_h, "BOOTROM_BASE_ADDR"),
        bootrom_size=hex(parse_define(mem_map_h, "BOOTROM_SIZE")),
        narrow_spm_base=parse_define(base_addr_h, "SPM_NARROW_BASE_ADDR"),
        narrow_spm_size=hex(parse_define(mem_map_h, "NARROW_SPM_SIZE")),
    )
    args.out.write_text(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
