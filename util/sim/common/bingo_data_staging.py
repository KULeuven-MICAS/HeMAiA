# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Where a workload's baked inputs and goldens live, decided by the platform.

There are two places, and which one is right is a property of the CONFIG, not of the
workload:

  MEMCHIP   the arrays go into ``build/mempool.bin`` and the workload addresses them
            through the memory chiplet. This is what the tapeout configs need: their
            ``spm_wide`` is 128 KiB and already holds .data, .bss and an embedded 88 KiB
            device binary, so a sweep's arrays simply do not fit.

  HOST      the arrays are emitted into the workload's ``data.h`` as ordinary C arrays
            and addressed by SYMBOL. This is what a single-chip config needs, because it
            has no memory chiplet at all -- and it can afford to, because such a config
            gives the host 16 MiB of ``spm_wide``.

Getting this wrong does not fault. A memchip address on a config with ``N_MEM_CHIPS = 0``
is simply not mapped: the loads return whatever the fabric gives back, the kernels run
happily on it, and the checks compare one piece of garbage against another. The failure
looks like an arithmetic bug -- plausible-looking outputs against implausible goldens --
which is a long way from the actual cause.

Usage:

    st = DataStaging(platform)
    h_in     = st.put("in_0",     "uint16_t", x.view(np.uint16))
    h_golden = st.put("golden_0", "uint16_t", y.view(np.uint16))
    ...                                   # h_* are BINGO memory handles, use them directly
    st.emit(args.data_h, args.output_dir)  # writes data.h and, if needed, mempool.bin
"""

import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.append(_THIS)

from data_utils import (  # noqa: E402
    format_vector_declaration, format_vector_definition)

_LIBBINGO = os.path.normpath(os.path.join(
    _THIS, "../../../target/sw/host/runtime/libbingo/mini_compiler"))
if _LIBBINGO not in sys.path:
    sys.path.append(_LIBBINGO)

import _bingo_paths  # noqa: F401,E402  (puts mini_compiler's grouped subdirs on sys.path)
from bingo_helpers import chiplet_addr_transform_loc      # noqa: E402
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol  # noqa: E402

# Default virtual address of the memory chiplet's pool, matching the tapeout configs.
MEMPOOL_VADDR = 0x8000_0000

# numpy dtype -> the C type the emitted array is declared with. The arrays are compared
# byte for byte by the device and the host, so the declared type only has to have the
# right WIDTH; uint16_t for FP16 keeps the emitter away from float literals, which would
# round.
_CTYPE = {
    np.dtype(np.uint8): "uint8_t",
    np.dtype(np.int8): "int8_t",
    np.dtype(np.uint16): "uint16_t",
    np.dtype(np.int16): "int16_t",
    np.dtype(np.uint32): "uint32_t",
    np.dtype(np.int32): "int32_t",
}


class DataStaging:
    """Collects arrays, then emits them where the platform can actually reach them."""

    def __init__(self, platform, mempool_loc=None, mempool_vaddr=MEMPOOL_VADDR):
        self.n_mem_chips = int(platform.get("num_mem_chips", 0))
        if mempool_loc is None:
            mempool_loc = (int(platform.get("mem_chip_loc_x", 0)),
                           int(platform.get("mem_chip_loc_y", 0)))
        self.mempool_loc = mempool_loc
        self.base = chiplet_addr_transform_loc(*mempool_loc, mempool_vaddr)
        self._items = []          # (name, ctype, memchip offset or None, array)
        self._zeros = []          # (name, ctype, count) -- host path only
        self._blob = bytearray()
        self._names = set()

    def _claim(self, name):
        """Reject a repeated name here rather than in the C compiler.

        On the host path every array becomes a C symbol, so a duplicate is a redefinition
        error hundreds of lines into a generated header; on the memchip path it is worse,
        because nothing complains at all and the second array simply shadows the first in
        whatever the workload does with the handles.
        """
        if name in self._names:
            raise ValueError(
                f"staged array {name!r} twice -- every array needs its own name, since on "
                f"the host path the name IS the C symbol")
        self._names.add(name)

    @property
    def on_memchip(self):
        return self.n_mem_chips > 0

    def put(self, name, ctype, arr):
        """Record *arr* and return the BINGO handle that addresses it.

        `ctype` may be None to infer it from the array's dtype.
        """
        self._claim(name)
        arr = np.ascontiguousarray(arr)
        if ctype is None:
            ctype = _CTYPE.get(arr.dtype)
            if ctype is None:
                raise TypeError(f"no C type for dtype {arr.dtype}; pass ctype explicitly")
        if self.on_memchip:
            # 64-B align every array: the iDMA and the host loads both want aligned
            # memchip words, and an unaligned golden silently shifts the comparison.
            while len(self._blob) % 64:
                self._blob.append(0)
            off = len(self._blob)
            self._blob.extend(arr.tobytes())
            self._items.append((name, ctype, off, arr))
            return BingoMemFixedAddr(self.base + off)
        self._items.append((name, ctype, None, arr))
        return BingoMemSymbol(name)

    def put_zeros(self, name, ctype, count):
        """Record a zero-filled array and return its handle, WITHOUT spelling out zeros.

        On the host path the array is DECLARED and not defined, so it lands in .bss, which
        `initialize_bss` clears before main -- zero bytes in the image for an arbitrarily
        large buffer. Spelling out a 64 KiB zero bias as C literals would add ~400 KiB of
        source for no information. On the memchip path there is no such trick: the pool is
        a flat binary, so the zeros are real bytes there.
        """
        self._claim(name)
        width = {"uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2,
                 "uint32_t": 4, "int32_t": 4}[ctype]
        if self.on_memchip:
            while len(self._blob) % 64:
                self._blob.append(0)
            off = len(self._blob)
            self._blob.extend(bytes(count * width))
            self._items.append((name, ctype, off, np.zeros(count, dtype=f"u{width}")))
            return BingoMemFixedAddr(self.base + off)
        self._zeros.append((name, ctype, count))
        return BingoMemSymbol(name)

    def emit(self, data_h_path, output_dir):
        """Write data.h, and mempool.bin when the arrays live on the memory chiplet."""
        lines = ["#include <stdint.h>", ""]
        if self.on_memchip:
            lines += [
                "// Inputs and goldens live on the MEMORY CHIPLET (build/mempool.bin), not",
                "// here: this platform's spm_wide is too small to also hold them.",
                f"// {len(self._items)} array(s), {len(self._blob)} bytes.",
            ]
        else:
            lines += [
                "// Inputs and goldens are baked into the host image and addressed by",
                "// symbol: this platform has no memory chiplet (N_MEM_CHIPS = 0), and its",
                "// spm_wide is large enough to carry them.",
            ]
            for name, ctype, _off, arr in self._items:
                lines.append(format_vector_definition(
                    ctype, name, arr.reshape(-1).tolist(), alignment=64))
                lines.append("")
            for name, ctype, count in self._zeros:
                lines.append(format_vector_declaration(ctype, name, range(count),
                                                       alignment=64))
                lines.append("")

        if data_h_path is not None:
            with open(data_h_path, "w") as f:
                f.write("\n".join(lines) + "\n")

        if self.on_memchip:
            out_build = os.path.join(output_dir, "build")
            os.makedirs(out_build, exist_ok=True)
            with open(os.path.join(out_build, "mempool.bin"), "wb") as f:
                f.write(bytes(self._blob))
        return len(self._blob) if self.on_memchip else sum(
            a.nbytes for _n, _c, _o, a in self._items)
