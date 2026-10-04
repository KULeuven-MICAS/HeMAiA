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

A platform may have several memory chiplets (occamy.h MEM_CHIP_ID_<k>), each holding
different data -- say, the weights of the layers its neighbours run. Every put* takes
`mem_chip=(x, y)` to pick one; without it the array goes to the HOME memory chiplet, the
first one or `mempool_loc`. `emit` writes the home chip's SRAM to build/mempool.bin (the
shared image every memory chip loads unless it has its own) and each other chip's to
build/mempool_chip_<x>_<y>.bin; with more than one memory chip, the HBM manifest tags every
image chip=<id> so it loads into that chip only.

Usage:

    st = DataStaging(platform)
    h_in     = st.put("in_0",     "uint16_t", x.view(np.uint16))
    h_golden = st.put("golden_0", "uint16_t", y.view(np.uint16))
    h_w      = st.put_hbm("w_q", w.view(np.int8))      # a weight, in the HBM
    h_w2     = st.put_hbm("w_k", w2, mem_chip=(3, 0))  # ...in another memory chip's HBM
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


def _c_array(ctype, name, n, alignment, values=None, section=None):
    """An integer C array of n elements: DEFINED with its values, or (values None) only
    DECLARED, so it lands in .bss -- or, with `section`, zeros in that (data) section."""
    sec = f", section (\"{section}\")" if section else ""
    head = f"{ctype} {name}[{n}] __attribute__ ((aligned ({alignment}){sec}))"
    if values is None:
        return head + ";"
    return head + " = {\n" + "".join(f"\t{v},\n" for v in values) + "};"


class _MemChip:
    """One memory chiplet's share of the staged data: its SRAM ("L4") image and its HBM
    image, and where both sit in the global address space."""

    def __init__(self, x, y, mempool_vaddr, hbm_base, hbm_size):
        self.loc = (x, y)
        self.chip_id = (x << 4) | y
        self.base = chiplet_addr_transform_loc(x, y, mempool_vaddr)
        self.hbm_size = int(hbm_size)
        self.hbm_base = (chiplet_addr_transform_loc(x, y, int(hbm_base))
                         if self.hbm_size else None)
        self.blob = bytearray()
        self.hbm = bytearray()
        self.hbm_items = []       # (name, offset, nbytes)
        self.l4_items = []        # (name, offset, nbytes): put_l4, in blob whatever on_host


def _platform_mem_chips(platform):
    """[(x, y, hbm_base, hbm_size)] of every memory chiplet. A platform dict from before
    `mem_chips` existed describes one, at mem_chip_loc_x/y."""
    chips = platform.get("mem_chips")
    if chips is not None:
        return [(c["id"] >> 4, c["id"] & 0xF, c.get("hbm_base", 0), c.get("hbm_size", 0))
                for c in chips]
    if not int(platform.get("num_mem_chips", 0)):
        return []
    return [(int(platform.get("mem_chip_loc_x", 0)), int(platform.get("mem_chip_loc_y", 0)),
             int(platform.get("hbm_base", 0)), int(platform.get("hbm_size", 0)))]


class DataStaging:
    """Collects arrays, then emits them where the platform can actually reach them."""

    def __init__(self, platform, mempool_loc=None, mempool_vaddr=MEMPOOL_VADDR,
                 on_host=None):
        """`on_host` overrides the platform's choice for put()/put_zeros(): True keeps the
        arrays in the host image even when a memory chiplet exists. That is only right on
        a config whose spm_wide can hold them -- the 16 MiB ones -- and it keeps every
        check a local read instead of a D2D round trip per word.

        `mempool_loc` picks the home memory chiplet (default: the first); it need not be
        one the platform lists, for a header that predates `mem_chips`."""
        chips = _platform_mem_chips(platform)
        self.n_mem_chips = int(platform.get("num_mem_chips", len(chips)))
        self._chips = {(x, y): _MemChip(x, y, mempool_vaddr, hb, hs) for x, y, hb, hs in chips}
        if mempool_loc is None:
            mempool_loc = chips[0][:2] if chips else (int(platform.get("mem_chip_loc_x", 0)),
                                                     int(platform.get("mem_chip_loc_y", 0)))
        mempool_loc = tuple(mempool_loc)
        if mempool_loc not in self._chips:
            hbm = (int(platform.get("hbm_base", 0)), int(platform.get("hbm_size", 0))
                   if self.n_mem_chips else 0)
            self._chips[mempool_loc] = _MemChip(*mempool_loc, mempool_vaddr, *hbm)
        self.mempool_loc = mempool_loc
        self._home = self._chips[mempool_loc]
        self.base = self._home.base
        self.hbm_size = self._home.hbm_size
        self.hbm_base = self._home.hbm_base
        self._on_host = on_host
        self._items = []          # (name, ctype, memchip offset or None, array) -- home chip
        self._zeros = []          # (name, ctype, count) -- host path only
        self._names = set()

    def _chip(self, mem_chip, what):
        """The _MemChip `mem_chip` names: (x, y), a chip id, or None for the home one."""
        if mem_chip is None:
            return self._home
        if isinstance(mem_chip, int):
            mem_chip = (mem_chip >> 4, mem_chip & 0xF)
        chip = self._chips.get(tuple(mem_chip))
        if chip is None:
            raise ValueError(
                f"{what}: no memory chiplet at {tuple(mem_chip)}; this platform has "
                f"{sorted(self._chips)}")
        return chip

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

    def put_hbm(self, name, arr, mem_chip=None):
        """Place *arr* in a memory chiplet's HBM (the home one unless `mem_chip`) and
        return its handle.

        The handle is a fixed global address that records its pool (mem_level "HBM"), so
        a port bound to it knows where it is. Arrays are packed into one image per chip in
        call order, each on a HBM_ALIGN boundary.
        """
        chip = self._chip(mem_chip, f"put_hbm({name!r})")
        if not self.n_mem_chips or not chip.hbm_size:
            raise ValueError(
                f"put_hbm({name!r}): this platform has no HBM on memory chiplet "
                f"{chip.loc} (HBM_SIZE is 0 or there is no memory chiplet). Only a cfg whose "
                f"hemaia_mem_chip entry has an `hbm` object has one -- "
                f"hemaia_twochiplet_16MBL3_4cluster.hjson -- and an HBM address on any other "
                f"is unmapped.")
        self._claim(name)
        arr = np.ascontiguousarray(arr)
        while len(chip.hbm) % HBM_ALIGN:
            chip.hbm.append(0)
        off = len(chip.hbm)
        chip.hbm.extend(arr.tobytes())
        if len(chip.hbm) > chip.hbm_size:
            raise ValueError(
                f"put_hbm({name!r}): the image of memory chiplet {chip.loc} is "
                f"{len(chip.hbm):,} B, past its HBM's {chip.hbm_size:,} B.")
        chip.hbm_items.append((name, off, arr.nbytes))
        return BingoMemFixedAddr(chip.hbm_base + off, mem_level="HBM")

    def fill_hbm(self, handle, data):
        """Overwrite bytes of an array put_hbm already placed, from `handle` (its handle or
        one into it) on: for contents decided after their address -- compressed weight
        chunks laid out while the graph is built, in a region reserved before it."""
        addr = int(handle.address)
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        for chip in self._chips.values():
            if chip.hbm_size and chip.hbm_base <= addr < chip.hbm_base + len(chip.hbm):
                off = addr - chip.hbm_base
                if off + data.size > len(chip.hbm):
                    raise ValueError(f"fill_hbm: {data.size} B at {addr:#x} runs past the "
                                     f"image of memory chiplet {chip.loc}")
                chip.hbm[off:off + data.size] = data.tobytes()
                return
        raise ValueError(f"fill_hbm: {addr:#x} is in no memory chiplet's HBM image")

    @property
    def hbm_bytes(self):
        return sum(len(c.hbm) for c in self._chips.values())

    def put_l4(self, name, arr, mem_chip=None):
        """Place *arr* in the memory chiplet's SRAM -- the "L4", build/mempool.bin, loaded
        at time zero -- whatever put() does, and return its fixed address.

        This is the steady state of a memchip-side prefetch: what a host iDMA on the
        memory chiplet would have copied HBM -> L4 ahead of use. The address is returned
        WITHOUT a level: no block has an L4 weight level, and a weight streams from any
        address the same way (an iDMA read over the D2D link). `mem_chip` picks the
        memory chiplet (default: the home one).
        """
        if not self.n_mem_chips:
            raise ValueError(f"put_l4({name!r}): this platform has no memory chiplet.")
        chip = self._chip(mem_chip, f"put_l4({name!r})")
        self._claim(name)
        arr = np.ascontiguousarray(arr)
        while len(chip.blob) % HBM_ALIGN:
            chip.blob.append(0)
        off = len(chip.blob)
        chip.blob.extend(arr.tobytes())
        chip.l4_items.append((name, off, arr.nbytes))
        return BingoMemFixedAddr(chip.base + off)

    @property
    def l4_bytes(self):
        return sum(n for c in self._chips.values() for _name, _off, n in c.l4_items)

    def put(self, name, ctype, arr, mem_chip=None):
        """Record *arr* and return the BINGO handle that addresses it.

        `ctype` may be None to infer it from the array's dtype. `mem_chip` puts it on that
        memory chiplet instead of the home one; the host path has no such choice.
        """
        chip = self._chip(mem_chip, f"put({name!r})")
        if mem_chip is not None and not self.on_memchip:
            raise ValueError(f"put({name!r}, mem_chip={mem_chip}): the arrays stay in the "
                             f"host image here (on_host, or no memory chiplet)")
        self._claim(name)
        arr = np.ascontiguousarray(arr)
        if ctype is None:
            ctype = _CTYPE.get(arr.dtype)
            if ctype is None:
                raise TypeError(f"no C type for dtype {arr.dtype}; pass ctype explicitly")
        if self.on_memchip:
            # 64-B align every array: the iDMA and the host loads both want aligned
            # memchip words, and an unaligned golden silently shifts the comparison.
            while len(chip.blob) % 64:
                chip.blob.append(0)
            off = len(chip.blob)
            chip.blob.extend(arr.tobytes())
            self._items.append((name, ctype, off, arr))
            return BingoMemFixedAddr(chip.base + off)
        self._items.append((name, ctype, None, arr))
        return BingoMemSymbol(name)

    def put_zeros(self, name, ctype, count, mem_chip=None, preload=False):
        """Record a zero-filled array and return its handle, WITHOUT spelling out zeros.

        On the host path the array is DECLARED and not defined, so it lands in .bss, which
        `initialize_bss` clears before main -- zero bytes in the image for an arbitrarily
        large buffer. Spelling out a 64 KiB zero bias as C literals would add ~400 KiB of
        source for no information. On the memchip path there is no such trick: the pool is
        a flat binary, so the zeros are real bytes there.

        `preload`: zeros in a .data section of their own instead, so the testbench's image
        preload clears it at time zero; in .bss the boot clears it with the system DMA, which
        for megabytes (a weight ring's slots) takes hundreds of microseconds of SIMULATED time
        before the program starts.
        """
        chip = self._chip(mem_chip, f"put_zeros({name!r})")
        if mem_chip is not None and not self.on_memchip:
            raise ValueError(f"put_zeros({name!r}, mem_chip={mem_chip}): the arrays stay "
                             f"in the host image here (on_host, or no memory chiplet)")
        self._claim(name)
        width = {"uint8_t": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2,
                 "uint32_t": 4, "int32_t": 4}[ctype]
        if self.on_memchip:
            while len(chip.blob) % 64:
                chip.blob.append(0)
            off = len(chip.blob)
            chip.blob.extend(bytes(count * width))
            self._items.append((name, ctype, off, np.zeros(count, dtype=f"u{width}")))
            return BingoMemFixedAddr(chip.base + off)
        self._zeros.append((name, ctype, count, bool(preload)))
        return BingoMemSymbol(name)

    def emit(self, data_h_path, output_dir):
        """Write data.h, and the memory chiplets' images: build/mempool.bin (the home chip)
        and build/mempool_chip_<x>_<y>.bin (each other chip given data), build/hbm."""
        home_bytes = len(self._home.blob)
        lines = ["#include <stdint.h>", ""]
        if self.on_memchip:
            lines += [
                "// Inputs and goldens live on the MEMORY CHIPLET (build/mempool.bin), not",
                "// here: this platform's spm_wide is too small to also hold them.",
                f"// {len(self._items)} array(s), "
                f"{sum(len(c.blob) for c in self._chips.values())} bytes.",
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
            for name, ctype, count, preload in self._zeros:
                lines.append(_c_array(ctype, name, count, 64,
                                      section=f".data.{name}" if preload else None))
                lines.append("")

        if data_h_path is not None:
            with open(data_h_path, "w") as f:
                f.write("\n".join(lines) + "\n")

        out_build = os.path.join(output_dir, "build")
        # A chip's own image from an earlier build would be loaded instead of the shared
        # one: remove every one this build does not write.
        if os.path.isdir(out_build):
            for f in os.listdir(out_build):
                if f.startswith("mempool_chip_") and f.endswith(".bin"):
                    os.remove(os.path.join(out_build, f))
        if self.on_memchip or self._home.l4_items:
            os.makedirs(out_build, exist_ok=True)
            with open(os.path.join(out_build, "mempool.bin"), "wb") as f:
                f.write(bytes(self._home.blob))
        for chip in self._chips.values():
            if chip is not self._home and chip.blob:
                os.makedirs(out_build, exist_ok=True)
                with open(os.path.join(out_build, "mempool_chip_%d_%d.bin" % chip.loc),
                          "wb") as f:
                    f.write(bytes(chip.blob))
        self._emit_hbm(output_dir)
        return home_bytes if self.on_memchip else sum(
            a.nbytes for _n, _c, _o, a in self._items)

    def _emit_hbm(self, output_dir):
        """build/hbm/manifest.txt and the images it lists; nothing if put_hbm was never
        called. A stale manifest from an earlier build is removed either way: the
        testharness loads whatever build/hbm/ holds. With one memory chiplet the image is
        untagged, as it always was; with more, each is tagged chip=<id>."""
        hbm_dir = os.path.join(output_dir, "build", "hbm")
        manifest = os.path.join(hbm_dir, "manifest.txt")
        chips = [c for c in self._chips.values() if c.hbm_items]
        if not chips:
            if os.path.exists(manifest):
                os.remove(manifest)
            return
        os.makedirs(hbm_dir, exist_ok=True)
        tagged = len(self._chips) > 1
        lines = ["# Generated by bingo_data_staging.py -- the workload's HBM image(s).",
                 "# offset (above the HBM base)  file  [chip=<id>: that memory chiplet only]"]
        notes = []
        for chip in chips:
            image = HBM_IMAGE if chip is self._home else "hbm_image_chip_%d_%d.bin" % chip.loc
            with open(os.path.join(hbm_dir, image), "wb") as f:
                f.write(bytes(chip.hbm))
            lines.append(f"0x0  {image}" + (f"  chip=0x{chip.chip_id:02x}" if tagged else ""))
            notes += ["#", f"# what {image} holds (offset above the HBM base, bytes):"]
            notes += [f"#   0x{off:08x}  {n:>10}  {name}" for name, off, n in chip.hbm_items]
        with open(manifest, "w") as f:
            f.write("\n".join(lines + notes) + "\n")
