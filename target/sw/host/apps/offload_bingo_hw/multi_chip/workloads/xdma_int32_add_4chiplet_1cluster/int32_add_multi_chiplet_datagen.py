#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""Generate MemPool-backed data for xdma_int32_add_4chiplet_1cluster."""

import argparse
import os
import pathlib

import hjson
import numpy as np


def _input_bytes(kwargs):
    input_bytes = int(kwargs.get("input_bytes", kwargs.get("bytes_per_input", 1024)))
    if input_bytes <= 0:
        raise ValueError(f"input_bytes must be positive, got {input_bytes}")
    if input_bytes % 4 != 0:
        raise ValueError(f"input_bytes ({input_bytes}) must be divisible by sizeof(int32_t)")
    if input_bytes % 64 != 0:
        raise ValueError(f"input_bytes ({input_bytes}) must be 64-byte aligned for xDMA")
    return input_bytes


def emit_header_file(**kwargs):
    return "\n\n".join(["#include <stdint.h>", *emit_int32_add_data(**kwargs)]) + "\n"


def emit_int32_add_data(**kwargs):
    num_chiplets = int(kwargs.get("num_chiplets", 4))
    if num_chiplets != 4:
        raise ValueError(f"xdma_int32_add_4chiplet_1cluster expects num_chiplets=4, got {num_chiplets}")

    input_bytes = _input_bytes(kwargs)
    num_elements = input_bytes // 4

    # Four 1 KiB int32 tiles occupy 4 KiB total: values 0..1023.
    values = np.arange(num_chiplets * num_elements, dtype=np.int32)
    inputs = [
        values[i * num_elements:(i + 1) * num_elements].copy()
        for i in range(num_chiplets)
    ]

    running_sum = inputs[0].copy()
    golden_sums = []
    for i in range(1, num_chiplets):
        running_sum = (
            running_sum.astype(np.int64) + inputs[i].astype(np.int64)
        ).astype(np.int32)
        golden_sums.append(running_sum.copy())

    out_dir = kwargs.get("out_dir", "./build/")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mempool.bin"), "wb") as f:
        for array in inputs:
            array.astype(np.int32).tofile(f)
        for array in golden_sums:
            array.astype(np.int32).tofile(f)

    lines = [
        f"uint32_t int32_add_num_chiplets = {num_chiplets};",
        f"uint32_t int32_add_input_bytes = {input_bytes};",
        f"uint32_t int32_add_num_elements = {num_elements};",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser(description="xdma_int32_add_4chiplet_1cluster data")
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
