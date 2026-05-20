#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""Data generator for dma_write_from_other_chip_4chiplet_1cluster."""

import argparse
import os
import pathlib
import sys

import hjson

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/")
)
from data_utils import (  # noqa E402
    format_scalar_definition,
    format_vector_declaration,
    format_vector_definition,
)


def _byte_ramp(size):
    return [idx & 0xFF for idx in range(size)]


def emit_header_file(**kwargs):
    num_data = int(kwargs.get("num_data", 4))
    data_bytes = int(kwargs.get("data_bytes", 1024))
    if num_data != 4:
        raise ValueError(
            "dma_write_from_other_chip_4chiplet_1cluster "
            f"expects num_data=4, got {num_data}"
        )
    if data_bytes <= 0:
        raise ValueError(f"data_bytes must be positive, got {data_bytes}")

    total_bytes = num_data * data_bytes
    golden = _byte_ramp(total_bytes)

    out_dir = kwargs.get("out_dir", "./build/")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mempool.bin"), "wb") as f:
        f.write(bytes(golden))

    lines = [
        "#include <stdint.h>",
        format_scalar_definition("uint32_t", "dma_cross_num_data", num_data),
        format_scalar_definition("uint32_t", "dma_cross_data_bytes", data_bytes),
        format_scalar_definition("uint32_t", "dma_cross_total_bytes", total_bytes),
        "// Per-chip local golden image for A1..A4. The memory-chip binary uses the same bytes.",
        format_vector_definition(
            "uint8_t",
            "A_golden_l3",
            golden,
            alignment=64,
        ),
        "// One local L3 load buffer per chiplet. Same symbol address, different chiplet base.",
        format_vector_declaration(
            "uint8_t",
            "A_local_l3",
            [0] * data_bytes,
            alignment=64,
        ),
        "// Chip00 receive slots for A2, A3, A4 written by chiplets 01, 10, 11.",
        format_vector_declaration(
            "uint8_t",
            "A_recv_chip00_l3",
            [0] * ((num_data - 1) * data_bytes),
            alignment=64,
        ),
    ]
    return "\n\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generating data for DMA write-from-other-chip test")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    print(emit_header_file(**{**param, **hw}))


if __name__ == "__main__":
    main()
