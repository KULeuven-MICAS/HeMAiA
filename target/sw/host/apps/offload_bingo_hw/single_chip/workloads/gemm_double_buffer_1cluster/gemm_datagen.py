#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# gemm_double_buffer stages the full A, B and golden D on the MEMCHIP
# (mempool.bin) instead of baking them into the host WIDE_SPM `.data`. With the
# 88 KiB device binary included, the baked operands left the bingo L3 heap far
# too small for this workload's 33-node double-buffered DFG (L3 malloc failed).
# A tiles are offsets into A; golden D tiles are offsets into D.

import argparse
import os
import pathlib
import sys

import hjson

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)
from gemm_sim_utils import generate_gemm_test_data  # noqa E402


def emit_header_file(**kwargs):
    return "\n\n".join(["#include <stdint.h>", *emit_matmul_data(**kwargs)])


def emit_matmul_data(**kwargs):
    gemm_data = generate_gemm_test_data(**kwargs)

    out_dir = kwargs.get("out_dir", "./build/")
    os.makedirs(out_dir, exist_ok=True)
    # Layout MUST match the base addresses in main_bingo.py:
    #   A_mp = base, B_mp = base + A_size, D_mp = base + A_size + B_size.
    with open(os.path.join(out_dir, "mempool.bin"), "wb") as f:
        gemm_data["A"].astype("int8").tofile(f)
        gemm_data["B"].astype("int8").tofile(f)
        gemm_data["D"].tofile(f)

    return []


def main():
    parser = argparse.ArgumentParser(description="Generating data for GEMM double-buffer test")
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
