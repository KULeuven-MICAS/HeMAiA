#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Copyright 2025 KU Leuven.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Yunhao Deng <yunhao.deng@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>

"""Generate the netlist-simulation boot ROM from a linked boot ROM binary.

The synthesized netlist contains the boot ROM as a mapped standard-cell block,
which simulates far too slowly to be useful. This module is dropped in as an
in-place replacement (see the `hemaia_netlist` target in Bender.yml): same
module name and same flattened port names as the mapped block, but the contents
are a plain case statement.

It used to be maintained by hand, with nothing tying it to `bootrom.bin` -- so
it silently kept the old contents whenever the boot ROM changed, and a netlist
simulation would run code that no other flow was running. Generating it from the
same binary `bootrom_sim/bootrom.sv` comes from removes that failure mode.

The port list is dictated by the synthesized netlist, not by us: the mapped
block exposes `addr_i` as individual escaped bits, and only `addr_i[9:2]` are
decoded (256 words of 4 bytes). Keep it in sync with
target/tapeout/.../hemaia_mapped_bootrom_commented.v if the ROM is ever resized.
"""

import argparse
import struct
import sys
from pathlib import Path

# The mapped block decodes addr_i[9:2]: 256 words. A boot ROM that outgrows this
# cannot be swapped in without regenerating the netlist, so overflow is an error
# rather than a silent truncation.
NUM_WORDS = 256

HEADER = """// Copyright 2022 ETH Zurich and University of Bologna.
// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Author: Yunhao Deng <yunhao.deng@kuleuven.be>
// This file is used as the in-place replacement of the bootrom in the synthesized netlist for the fast simulation.\x20

module bootrom (
    output reg  [31:0] data_o,
    input  wire        \\addr_i[12] ,
    input  wire        \\addr_i[11] ,
    input  wire        \\addr_i[10] ,
    input  wire        \\addr_i[9] ,
    input  wire        \\addr_i[8] ,
    input  wire        \\addr_i[7] ,
    input  wire        \\addr_i[6] ,
    input  wire        \\addr_i[5] ,
    input  wire        \\addr_i[4] ,
    input  wire        \\addr_i[3] ,
    input  wire        \\addr_i[2]\x20
);

    localparam AddrWidth = 32;
    localparam DataWidth = 32;

    localparam NumWords = {num_words};
    wire [7:0] word;

    assign word[7] = \\addr_i[9] ;
    assign word[6] = \\addr_i[8] ;
    assign word[5] = \\addr_i[7] ;
    assign word[4] = \\addr_i[6] ;
    assign word[3] = \\addr_i[5] ;
    assign word[2] = \\addr_i[4] ;
    assign word[1] = \\addr_i[3] ;
    assign word[0] = \\addr_i[2] ;

    always @* begin
        data_o = 32'd0;
        case (word)
"""

FOOTER = """            default: data_o = '0;
        endcase
    end

endmodule
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", required=True, type=Path,
                        help="linked boot ROM binary (bootrom_sim/bootrom.bin)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Verilog file to write")
    args = parser.parse_args()

    raw = args.bin.read_bytes()
    if len(raw) % 4:
        raw += b"\0" * (4 - len(raw) % 4)
    words = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
    if len(words) > NUM_WORDS:
        parser.error(f"{args.bin} is {len(words)} words, but the netlist ROM decodes "
                     f"only {NUM_WORDS}. Shrink the boot ROM or regenerate the netlist.")
    words += [0] * (NUM_WORDS - len(words))

    # Entry 0 carries the indentation of the `case` line it follows; the rest are
    # indented to line up under it. This mirrors util/bin2sv.py's output so the two
    # boot ROM renderings stay visually comparable.
    body = "".join(
        f"{'        ' if i == 0 else '            '}{i:03d}: data_o = 32'h{w:08x}"
        f" /* 0x{i * 4:04x} */;\n"
        for i, w in enumerate(words))

    args.out.write_text(HEADER.format(num_words=NUM_WORDS) + body + FOOTER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
