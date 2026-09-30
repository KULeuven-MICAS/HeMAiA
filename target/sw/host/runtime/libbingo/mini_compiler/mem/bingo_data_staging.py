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

A third place is the memory chiplet's HBM, and unlike the two above it is the WORKLOAD's
choice, not the platform's: it holds what does not fit anywhere else -- a layer's weights --
and a block streams from it straight into L1. `put_hbm` places an array there, at a fixed
offset, and `emit` writes `build/hbm/manifest.txt` plus the image it lists, which the
testharness maps into the HBM at time zero (hw/hemaia/hemaia_mem_system/hbm/README.md).
On a platform without an HBM it refuses, for the same reason as above: the address would
simply be unmapped.

Usage:

    st = DataStaging(platform)
    h_in     = st.put("in_0",     "uint16_t", x.view(np.uint16))
    h_golden = st.put("golden_0", "uint16_t", y.view(np.uint16))
    h_w      = st.put_hbm("w_q", w.view(np.int8))      # a weight, in the HBM
    ...                                   # h_* are BINGO memory handles, use them directly
    st.emit(args.data_h, args.output_dir)  # data.h and, if needed, mempool.bin / build/hbm
"""

import os

import numpy as np

from bingo_helpers import chiplet_addr_transform_loc
from bingo_mem_handle import BingoMemFixedAddr, BingoMemSymbol

# Default virtual address of the memory chiplet's pool, matching the tapeout configs.
MEMPOOL_VADDR = 0x8000_0000
# The HBM image is one file, loaded at this offset above the HBM base; every array in it
# starts on this boundary. 4 KiB is the HBM's channel interleave, so an array's first
# burst never straddles two pseudo-channels for alignment's sake alone.
HBM_ALIGN = 4096
HBM_IMAGE = "hbm_image.bin"

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


def _c_array(ctype, name, n, alignment, values=None):
    """An integer C array of n elements: DEFINED with its values, or (values None) only
    DECLARED, so it lands in .bss."""
    head = f"{ctype} {name}[{n}] __attribute__ ((aligned ({alignment})))"
    if values is None:
        return head + ";"
    return head + " = {\n" + "".join(f"\t{v},\n" for v in values) + "};"


class DataStaging:
    """Collects arrays, then emits them where the platform can actually reach them."""

    def __init__(self, platform, mempool_loc=None, mempool_vaddr=MEMPOOL_VADDR,
                 on_host=None):
        """`on_host` overrides the platform's choice for put()/put_zeros(): True keeps the
        arrays in the host image even when a memory chiplet exists. That is only right on
        a config whose spm_wide can hold them -- the 16 MiB ones -- and it keeps every
        check a local read instead of a D2D round trip per word."""
        self.n_mem_chips = int(platform.get("num_mem_chips", 0))
        if mempool_loc is None:
            mempool_loc = (int(platform.get("mem_chip_loc_x", 0)),
                           int(platform.get("mem_chip_loc_y", 0)))
        self.mempool_loc = mempool_loc
        self.base = chiplet_addr_transform_loc(*mempool_loc, mempool_vaddr)
        self._on_host = on_host
        self._items = []          # (name, ctype, memchip offset or None, array)
        self._zeros = []          # (name, ctype, count) -- host path only
        self._blob = bytearray()
        self._names = set()
        # The HBM: chip-local base and size from the platform; 0 size = none.
        self.hbm_size = int(platform.get("hbm_size", 0)) if self.n_mem_chips else 0
        self.hbm_base = (chiplet_addr_transform_loc(*mempool_loc,
                                                    int(platform.get("hbm_base", 0)))
                         if self.hbm_size else None)
        self._hbm = bytearray()
        self._hbm_items = []      # (name, offset, nbytes)
        self._l4_items = []       # (name, offset, nbytes): put_l4, in _blob whatever on_host

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
        if self._on_host is not None:
            return not self._on_host
        return self.n_mem_chips > 0

    @property
    def has_hbm(self):
        return self.hbm_size > 0

    def put_hbm(self, name, arr):
        """Place *arr* in the memory chiplet's HBM and return its handle.

        The handle is a fixed global address that records its pool (mem_level "HBM"), so
        a port bound to it knows where it is. Arrays are packed into one image in call
        order, each on a HBM_ALIGN boundary.
        """
        if not self.has_hbm:
            raise ValueError(
                f"put_hbm({name!r}): this platform has no HBM (HBM_SIZE is 0 or there is "
                f"no memory chiplet). Only a cfg whose hemaia_mem_chip entry has an `hbm` "
                f"object has one -- hemaia_twochiplet_16MBL3_4cluster.hjson -- and an HBM "
                f"address on any other is unmapped.")
        self._claim(name)
        arr = np.ascontiguousarray(arr)
        while len(self._hbm) % HBM_ALIGN:
            self._hbm.append(0)
        off = len(self._hbm)
        self._hbm.extend(arr.tobytes())
        if len(self._hbm) > self.hbm_size:
            raise ValueError(
                f"put_hbm({name!r}): the image is {len(self._hbm):,} B, past the HBM's "
                f"{self.hbm_size:,} B.")
        self._hbm_items.append((name, off, arr.nbytes))
        return BingoMemFixedAddr(self.hbm_base + off, mem_level="HBM")

    @property
    def hbm_bytes(self):
        return len(self._hbm)

    def put_l4(self, name, arr):
        """Place *arr* in the memory chiplet's SRAM -- the "L4", build/mempool.bin, loaded
        at time zero -- whatever put() does, and return its fixed address.

        This is the steady state of a memchip-side prefetch: what a host iDMA on the
        memory chiplet would have copied HBM -> L4 ahead of use. The address is returned
        WITHOUT a level: no block has an L4 weight level, and a weight streams from any
        address the same way (an iDMA read over the D2D link).
        """
        if not self.n_mem_chips:
            raise ValueError(f"put_l4({name!r}): this platform has no memory chiplet.")
        self._claim(name)
        arr = np.ascontiguousarray(arr)
        while len(self._blob) % HBM_ALIGN:
            self._blob.append(0)
        off = len(self._blob)
        self._blob.extend(arr.tobytes())
        self._l4_items.append((name, off, arr.nbytes))
        return BingoMemFixedAddr(self.base + off)

    @property
    def l4_bytes(self):
        return sum(n for _name, _off, n in self._l4_items)

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
                vals = arr.reshape(-1).tolist()
                lines.append(_c_array(ctype, name, len(vals), 64, vals))
                lines.append("")
            for name, ctype, count in self._zeros:
                lines.append(_c_array(ctype, name, count, 64))
                lines.append("")

        if data_h_path is not None:
            with open(data_h_path, "w") as f:
                f.write("\n".join(lines) + "\n")

        if self.on_memchip or self._l4_items:
            out_build = os.path.join(output_dir, "build")
            os.makedirs(out_build, exist_ok=True)
            with open(os.path.join(out_build, "mempool.bin"), "wb") as f:
                f.write(bytes(self._blob))
        self._emit_hbm(output_dir)
        return len(self._blob) if self.on_memchip else sum(
            a.nbytes for _n, _c, _o, a in self._items)

    def _emit_hbm(self, output_dir):
        """build/hbm/manifest.txt and the one image it lists; nothing if put_hbm was never
        called. A stale manifest from an earlier build is removed either way: the
        testharness loads whatever build/hbm/ holds."""
        hbm_dir = os.path.join(output_dir, "build", "hbm")
        manifest = os.path.join(hbm_dir, "manifest.txt")
        if not self._hbm_items:
            if os.path.exists(manifest):
                os.remove(manifest)
            return
        os.makedirs(hbm_dir, exist_ok=True)
        with open(os.path.join(hbm_dir, HBM_IMAGE), "wb") as f:
            f.write(bytes(self._hbm))
        lines = ["# Generated by bingo_data_staging.py -- the workload's HBM image.",
                 "# offset (above the HBM base)  file",
                 f"0x0  {HBM_IMAGE}",
                 "#",
                 "# what the image holds (offset above the HBM base, bytes):"]
        lines += [f"#   0x{off:08x}  {n:>10}  {name}" for name, off, n in self._hbm_items]
        with open(manifest, "w") as f:
            f.write("\n".join(lines) + "\n")
