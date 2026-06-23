#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>

"""Generate local L3 golden data and a matching memory-chip binary."""

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/")
)
from data_utils import (  # noqa E402
    format_scalar_definition,
    format_vector_declaration,
    format_vector_definition,
)


np.random.seed(42)


def _parse_int(value):
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def _generate_data(total_bytes):
    return np.random.randint(0, 256, total_bytes, dtype=np.uint8)


def emit_header_file(**kwargs):
    num_chunks = _parse_int(kwargs.get("num_chunks", 4))
    data_bytes = _parse_int(kwargs.get("data_bytes", kwargs.get("length_data", 1024)))
    mem_chip_local_base = _parse_int(kwargs.get("mem_chip_local_base", 0x80000000))

    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if data_bytes <= 0:
        raise ValueError(f"data_bytes must be positive, got {data_bytes}")

    total_bytes = num_chunks * data_bytes
    data = _generate_data(total_bytes)

    out_dir = pathlib.Path(kwargs.get("out_dir", os.path.dirname(__file__)))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "mempool.bin").open("wb") as f:
        f.write(data.tobytes())

    lines = [
        "#include <stdint.h>",
        format_scalar_definition(
            "uint32_t", "mem_chip_access_num_chunks", num_chunks
        ),
        format_scalar_definition(
            "uint32_t", "mem_chip_access_data_bytes", data_bytes
        ),
        format_scalar_definition(
            "uint32_t", "mem_chip_access_total_bytes", total_bytes
        ),
        format_scalar_definition(
            "uint64_t", "mem_chip_access_local_base", mem_chip_local_base
        ),
        "// Golden copy in local L3. data/mempool.bin contains the same bytes.",
        format_vector_definition(
            "uint8_t",
            "mem_chip_access_golden_l3",
            data.tolist(),
            alignment=64,
        ),
        "// Destination for the system-iDMA read from the memory chip into local L3.",
        format_vector_declaration(
            "uint8_t",
            "mem_chip_access_result_l3",
            [0] * data_bytes,
            alignment=64,
        ),
    ]
    return "\n\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate mem-chip-access data")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select parameter config file",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent,
        help="Directory where mempool.bin is written",
    )
    args = parser.parse_args()

    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    print(emit_header_file(**{**param, "out_dir": args.out_dir}))


if __name__ == "__main__":
    main()
