#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

"""Data for xdma_remote_write_4chiplet_1cluster.

Emits two things:

  * build/mempool.bin -- [payload | poison], both `payload_bytes` long. The
    payload is what chiplet 00 sends; the poison is what chiplet 01 pre-loads
    into its receive buffer so a PASS cannot come from a stale/lucky L1.
  * a header with `xdma_rw_golden_l3[]` (== the payload) as a local L3 symbol.
    The golden lives in the ELF rather than in the MemPool chip so chiplet 01's
    HOST core checks against its OWN L3 -- no cross-chip host read in the
    checker, which would confound the very thing this workload measures.
"""

import argparse
import os
import pathlib

import hjson
import numpy as np


POISON_WORD = 0xDEADBEEF


def _payload_bytes(kwargs):
    payload_bytes = int(kwargs.get("payload_bytes", 1024))
    if payload_bytes <= 0:
        raise ValueError(f"payload_bytes must be positive, got {payload_bytes}")
    if payload_bytes % 4 != 0:
        raise ValueError(f"payload_bytes ({payload_bytes}) must be divisible by sizeof(uint32_t)")
    if payload_bytes % 64 != 0:
        raise ValueError(f"payload_bytes ({payload_bytes}) must be 64-byte aligned for xDMA")
    return payload_bytes


def emit_header_file(**kwargs):
    return "\n\n".join(["#include <stdint.h>", *emit_xdma_remote_write_data(**kwargs)]) + "\n"


def emit_xdma_remote_write_data(**kwargs):
    payload_bytes = _payload_bytes(kwargs)
    num_elements = payload_bytes // 4

    # A pattern that is wrong in every word if the transfer half-lands: the
    # index is in the low bits and a fixed tag in the high bits.
    payload = np.array(
        [0xA5A50000 + i for i in range(num_elements)], dtype=np.uint32
    )
    poison = np.full(num_elements, POISON_WORD, dtype=np.uint32)

    out_dir = kwargs.get("out_dir", "./build/")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mempool.bin"), "wb") as f:
        payload.tofile(f)
        poison.tofile(f)

    golden_words = ", ".join(f"0x{w:08x}" for w in payload)
    lines = [
        f"uint32_t xdma_rw_payload_bytes = {payload_bytes};",
        f"uint32_t xdma_rw_num_elements = {num_elements};",
        # The golden copy chiplet 01 checks against, in its own L3.
        f"uint32_t xdma_rw_golden_l3[{num_elements}] = {{{golden_words}}};",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser(description="xdma_remote_write_4chiplet_1cluster data")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument("--out_dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    merged = {**param, **hw}
    if args.out_dir is not None:
        merged["out_dir"] = str(args.out_dir)

    content = emit_header_file(**merged)
    with args.output.open("w") as f:
        f.write(content)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
