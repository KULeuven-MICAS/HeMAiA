#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Check packed data against the accelerator's stream address walk, without RTL."""

import argparse
import copy
from pathlib import Path

import hjson

from datagen import emit, make_data


def check_stream_walk(hw, sw):
    cfg, arrays, a, b, golden = make_data(sw, hw)
    m, k, n = (cfg[key] for key in ("M", "K", "N"))
    row, inner, col = (cfg[key] for key in ("MESH_ROW", "TILE_SIZE", "MESH_COL"))
    assert cfg["A_BYTES"] == m * k * row * inner
    assert cfg["B_BYTES"] == n * k * col * inner
    assert cfg["D_BYTES"] == m * n * row * col * 4
    assert cfg["A_OFF"] + cfg["A_BYTES"] <= cfg["B_OFF"]
    assert cfg["B_OFF"] + cfg["B_BYTES"] <= cfg["D_OFF"]
    assert cfg["D_OFF"] + cfg["D_BYTES"] <= cfg["TX_OFF"]
    assert min(a) > -128 and min(b) > -128
    assert a[:cfg["A_BYTES"]] != a[cfg["A_BYTES"]:]

    # Emulate the actual C kernel's K,N,M temporal loops. Each A pass is
    # row-major [row,inner]; each B pass presents [col,inner]. This independent
    # address walk must match the generator's dense row-major reference.
    result = []
    for chip in range(2):
        for mm in range(m):
            for nn in range(n):
                block = [0] * (row * col)
                for kk in range(k):
                    a_addr = chip * cfg["A_BYTES"] + (mm * k + kk) * cfg["A_TILE_BYTES"]
                    b_addr = (nn * k + kk) * cfg["B_TILE_BYTES"]
                    for rr in range(row):
                        for cc in range(col):
                            for qq in range(inner):
                                block[rr * col + cc] += (
                                    a[a_addr + rr * inner + qq] *
                                    b[b_addr + cc * inner + qq])
                result.extend(block)
    assert result == golden, "streamed operands do not produce the dense golden"
    header = emit(sw, hw)
    assert "c_channels[] __attribute__((section(\".data.fpga_config\")))" in header
    assert "mem_chip" not in header and "mempool" not in header
    return arrays


def expect_rejection(hw, sw, message):
    try:
        make_data(sw, hw)
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"configuration was accepted; expected {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwcfg", required=True, type=Path)
    args = parser.parse_args()
    hw = hjson.loads(args.hwcfg.read_text())
    sw = hjson.loads(Path(__file__).with_name("params.hjson").read_text())
    check_stream_walk(hw, sw)
    shapes = hw["snax_versacore_core_template"]["snax_acc_cfg"][0]["snax_versacore_spatial_unrolling"][0]
    checked = 0
    for shape in range(len(shapes)):
        varied = {**sw, "M": 3, "N": 2, "K": 3, "array_shape": shape}
        try:
            check_stream_walk(hw, varied)
        except ValueError as exc:
            # Some hardware shapes need padded rather than dense operands; the
            # application deliberately rejects them instead of emitting bad data.
            assert "padded tiles" in str(exc), str(exc)
            continue
        checked += 1

    # Also exercise split-style multidimensional spatial nests and six B loops.
    # This is a synthetic CSR-layout regression, not an RTL execution claim.
    split = copy.deepcopy(hw)
    acc = split["snax_versacore_core_template"]["snax_acc_cfg"][0]
    st = acc["snax_streamer_cfg"]
    if "$ref" in st:
        st = split[st["$ref"].split("/")[-1]]
    rd, rw = st["data_reader_params"], st["data_reader_writer_params"]
    b_channels = rd["num_channel"][1]
    cd_channels = rw["num_channel"][0]
    assert b_channels % 8 == 0 and cd_channels % 4 == 0
    rd["spatial_bounds"][1] = [8, b_channels // 8]
    rd["temporal_dim"][1] = 6
    rw["spatial_bounds"] = [[4, cd_channels // 4], [4, cd_channels // 4]]
    arrays = check_stream_walk(split, sw)
    assert arrays["b_spatial"] == [8, 64]
    assert arrays["c_spatial"] == arrays["d_spatial"] == [8, 32]

    # The split cluster has exactly the three C/D loops used by run_gemm().
    rw["temporal_dim"] = [3, 3]
    check_stream_walk(split, sw)
    for dims in ([2, 3], [3, 2]):
        rw["temporal_dim"] = dims
        expect_rejection(split, sw, "C and D need at least three temporal loops")

    expect_rejection(hw, {**sw, "M": 0}, "positive tile counts")
    expect_rejection(hw, {**sw, "array_shape": len(shapes)}, "absent")
    expect_rejection({}, sw, "no VersaCore accelerator")
    alu = args.hwcfg.parent / "snax_alu_cluster.hjson"
    if alu.exists():
        expect_rejection(hjson.loads(alu.read_text()), sw, "no VersaCore accelerator")
    print(f"PASS: default + {checked} asymmetric shapes, split-style spatial/temporal layout, invalid configs")


if __name__ == "__main__":
    main()
