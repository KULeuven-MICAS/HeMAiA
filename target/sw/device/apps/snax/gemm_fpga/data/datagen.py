#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Emit small, deterministic INT8 GEMM inputs and an independent dense golden.

Every array is part of the device image in local L3. No mempool.bin is needed.
The packed formats consumed with transpose_A/B=0 are A[m,k,row,inner],
B[n,k,col,inner] and D[m,n,row,col].
"""

import argparse
import math
from pathlib import Path

import hjson


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def spatial_strides(bounds):
    strides = []
    stride = 8
    for bound in bounds:
        strides.append(stride)
        stride *= bound
    return strides


def channel_mask(bits, width):
    words = (width + 2047) // 2048
    return [((1 << bits) - 1) >> (32 * i) & 0xFFFFFFFF
            for i in reversed(range(words))]


def make_data(sw, hw):
    m, k, n = (int(sw[key]) for key in ("M", "K", "N"))
    shape = int(sw["array_shape"])
    if min(m, k, n) <= 0:
        raise ValueError("M, K and N must be positive tile counts")
    if "snax_versacore_core_template" not in hw:
        raise ValueError("selected cluster has no VersaCore accelerator")
    acc = hw["snax_versacore_core_template"]["snax_acc_cfg"][0]
    if not 0 <= shape < len(acc["snax_versacore_spatial_unrolling"][0]):
        raise ValueError("array_shape is absent from the selected cluster")
    row, inner, col = acc["snax_versacore_spatial_unrolling"][0][shape]
    for operand, bits in (("a", 8), ("b", 8), ("c", 32)):
        if acc[f"snax_versacore_input_{operand}_element_width"][0] != bits:
            raise ValueError("gemm_fpga requires INT8 inputs and INT32 accumulation")
        if acc[f"snax_versacore_input_{operand}_data_type"][0] != "SInt":
            raise ValueError("gemm_fpga requires integer data_type 0")

    streamer = acc["snax_streamer_cfg"]
    if "$ref" in streamer:
        streamer = hw[streamer["$ref"].split("/")[-1]]
    rd = streamer["data_reader_params"]
    rw = streamer["data_reader_writer_params"]
    if rd["temporal_dim"][0] < 3 or rd["temporal_dim"][1] < 3:
        raise ValueError("A and B need at least three temporal loops")
    if any(dim < 4 for dim in rw["temporal_dim"]):
        raise ValueError("C and D need at least four temporal loops")

    # Build logical row-major matrices first; the golden does not depend on the
    # streamer's blocked indexing. Values avoid the -128 receive-buffer sentinel.
    rows, depth, cols = 2 * m * row, k * inner, n * col
    a_dense = [[(r * 7 + q * 3 + 1) % 11 - 5 for q in range(depth)]
               for r in range(rows)]
    b_dense = [[(q * 5 + c * 2 + 3) % 13 - 6 for c in range(cols)]
               for q in range(depth)]
    d_dense = [[sum(a_dense[r][q] * b_dense[q][c] for q in range(depth))
                for c in range(cols)] for r in range(rows)]
    a = [a_dense[mm * row + rr][kk * inner + qq]
         for mm in range(2 * m) for kk in range(k)
         for rr in range(row) for qq in range(inner)]
    b = [b_dense[kk * inner + qq][nn * col + cc]
         for nn in range(n) for kk in range(k)
         for cc in range(col) for qq in range(inner)]
    golden = [d_dense[mm * row + rr][nn * col + cc]
              for mm in range(2 * m) for nn in range(n)
              for rr in range(row) for cc in range(col)]

    a_tile, b_tile, d_tile = row * inner, col * inner, row * col * 4
    a_align = int(acc.get("granularity_a", 1)) * 8
    b_align = int(acc.get("granularity_b", 1)) * 8
    cd_align = int(acc.get("granularity_c_d", 1)) * 8
    # Dense tile packing requires strides permitted by the sparse interconnect.
    if a_tile % a_align or b_tile % b_align or d_tile % cd_align:
        raise ValueError("selected shape requires padded tiles; select a dense INT8 shape")
    serial = int(acc["snax_versacore_serial_c_d_width"]) // 8
    if d_tile > serial and d_tile % serial:
        raise ValueError("output tile must contain whole serializer beats")
    a_bytes, b_bytes, d_bytes = len(a) // 2, len(b), len(golden) * 2
    # Preserve the runtime's heap manager and first fragment header (currently
    # 688 bytes total). This app uses fixed buffers and never allocates from it.
    a_off = align(1024, a_align)
    b_off = align(a_off + a_bytes, b_align)
    d_off = align(b_off + b_bytes, cd_align)
    tx_off = align(d_off + d_bytes, 64)
    constants = {
        "M": m, "K": k, "N": n, "ARRAY_SHAPE": shape,
        "MESH_ROW": row, "TILE_SIZE": inner, "MESH_COL": col,
        "A_BYTES": a_bytes, "B_BYTES": b_bytes, "D_BYTES": d_bytes,
        "A_TILE_BYTES": a_tile, "B_TILE_BYTES": b_tile,
        "D_TILE_BYTES": d_tile, "CD_BEATS": max(1, d_tile // serial),
        "CD_BEAT_BYTES": min(d_tile, serial),
        "A_OFF": a_off, "B_OFF": b_off, "D_OFF": d_off,
        "TX_OFF": tx_off, "L1_BYTES": tx_off + 64,
    }
    config = {
        "a_spatial": spatial_strides(rd["spatial_bounds"][0]),
        "b_spatial": spatial_strides(rd["spatial_bounds"][1]),
        "c_spatial": spatial_strides(rw["spatial_bounds"][0]),
        "d_spatial": spatial_strides(rw["spatial_bounds"][1]),
        # The streamer gates input channels in groups of eight, matching the
        # platform's generated gemm_shapes.h channel masks.
        "a_channels": channel_mask(align(math.ceil(a_tile / 8), 8), acc["snax_versacore_array_input_a_width"]),
        "b_channels": channel_mask(align(math.ceil(b_tile / 8), 8), acc["snax_versacore_array_input_b_width"]),
        "c_channels": channel_mask(0, serial * 8),
        "d_channels": channel_mask(math.ceil(min(d_tile, serial) / 8), serial * 8),
    }
    return constants, config, a, b, golden


def emit(sw, hw):
    constants, config, a, b, golden = make_data(sw, hw)
    lines = ["// Generated by gemm_fpga/data/datagen.py. Do not edit.",
             "#pragma once", "#include <stdint.h>", ""]
    lines += [f"#define FPGA_GEMM_{key} {value}u" for key, value in constants.items()]
    lines += [""]
    for name, values in config.items():
        # Even the all-zero C mask must be in the loaded image: the custom
        # device startup does not initialize global L3 .bss.
        lines.append(f"static uint32_t {name}[] __attribute__((section(\".data.fpga_config\"))) = {{ " +
                     ", ".join(f"{value}u" for value in values) + " };")
    for ctype, name, values in (("int8_t", "a_l3", a), ("int8_t", "b_source_l3", b),
                                ("int32_t", "golden_l3", golden)):
        lines.append(f"static const {ctype} {name}[{len(values)}] __attribute__((aligned(64))) = {{")
        lines += ["    " + ", ".join(map(str, values[i:i + 16])) + ","
                  for i in range(0, len(values), 16)]
        lines += ["};", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swcfg", required=True, type=Path)
    parser.add_argument("--hwcfg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = emit(hjson.loads(args.swcfg.read_text()), hjson.loads(args.hwcfg.read_text()))
    args.output.write_text(result)


if __name__ == "__main__":
    main()
