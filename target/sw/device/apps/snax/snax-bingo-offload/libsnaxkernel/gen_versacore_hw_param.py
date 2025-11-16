#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

# Generate versacore hardware parameter header file for snax-bingo-offload kernels
# specifically, the 

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import math
import struct
import datetime

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition, format_scalar_define_no_capital  # noqa E402

np.random.seed(42)

def gen_channel_enable_CSR(channel_en_CSR, channel_en_bits):
    for i in range(channel_en_bits):
        element_index = i // 32  # Determine which element to modify
        bit_position = i % 32  # Position within the element
        if element_index < len(channel_en_CSR):
            channel_en_CSR[element_index] |= 1 << (bit_position)

    channel_en_CSR = [int(x) for x in channel_en_CSR][::-1]
    return channel_en_CSR

# Add stdint.h header
def emit_header_file(**kwargs):
    # get current time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emit_str = f'''// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>

// Generated at {current_time}

#pragma once

// #define VERSACORE_DEBUG
#ifdef VERSACORE_DEBUG
#define VERSACORE_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define VERSACORE_DEBUG_PRINT(...)
#endif

'''
    emit_str += emit_matmul_data(**kwargs)
    return emit_str

def emit_matmul_data(**kwargs):

    # -------------------------------------------------------------
    # matmul workload settings
    # -------------------------------------------------------------
    data_str = []

    # -------------------------------------------------------------
    # -----------------------hardware parameters--------------------
    # --------------------------------------------------------------

    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]

    # generate the spatial dataflow parameters
    for i, (meshRow, tileSize, meshCol) in enumerate(snax_acc_cfg["snax_versacore_spatial_unrolling"][0]):
        data_str += [format_scalar_define_no_capital(f"meshRow_{i}", meshRow)]
        data_str += [format_scalar_define_no_capital(f"tileSize_{i}", tileSize)]
        data_str += [format_scalar_define_no_capital(f"meshCol_{i}", meshCol)]

    # only support one data type for now in the sw
    a_len = snax_acc_cfg["snax_versacore_input_a_element_width"][0]
    b_len = snax_acc_cfg["snax_versacore_input_b_element_width"][0]
    c_len = snax_acc_cfg["snax_versacore_input_c_element_width"][0]
    d_len = snax_acc_cfg["snax_versacore_output_d_element_width"][0]
    data_str += [format_scalar_define_no_capital("A_elem_len", a_len)]
    data_str += [format_scalar_define_no_capital("B_elem_len", b_len)]
    data_str += [format_scalar_define_no_capital("C_elem_len", c_len)]
    data_str += [format_scalar_define_no_capital("D32_elem_len", d_len)]

    a_array_width = snax_acc_cfg["snax_versacore_array_input_a_width"]
    b_array_width = snax_acc_cfg["snax_versacore_array_input_b_width"]
    c_array_width = snax_acc_cfg["snax_versacore_array_input_c_width"]
    d_array_width = snax_acc_cfg["snax_versacore_array_output_d_width"]

    assert c_array_width == d_array_width, "C and D array width must be the same"
    snax_versacore_serial_c_d_width = snax_acc_cfg["snax_versacore_serial_c_d_width"]
    data_str += [format_scalar_define_no_capital("versacore_serial_c_d_width", snax_versacore_serial_c_d_width)]

    bankWidth = 64
    data_str += [format_scalar_define_no_capital("bankWidth", bankWidth)]

    # -------------------------------------------------------------
    # -----------------------A streamer setting channel enable-------------------------------
    # --------------------------------------------------------------

    A_enabled_channel_CSR_num = int(math.ceil(a_array_width / bankWidth / 32))
    for i, (meshRow, tileSize, meshCol) in enumerate(snax_acc_cfg["snax_versacore_spatial_unrolling"][0]):
        channel_en_A = [0] * A_enabled_channel_CSR_num
        # related to if this is a wide channel or not
        # if wide, must be divisible by 8
        # if narrow, must be divisible by 1
        channel_en_A_bits = max(
            8, int((meshRow * tileSize * a_len / bankWidth + 7) // 8 * 8)
        )
        channel_en_A = gen_channel_enable_CSR(
            channel_en_A,
            channel_en_A_bits,
        )
        for j in range(len(channel_en_A)):
            data_str += [format_scalar_define_no_capital(f"channel_en_A_{i}_{j}", channel_en_A[j])]

    # -------------------------------------------------------------
    # -----------------------B setting-------------------------------
    # --------------------------------------------------------------

    B_enabled_channel_CSR_num = int(math.ceil(b_array_width / bankWidth / 32))
    for i, (meshRow, tileSize, meshCol) in enumerate(snax_acc_cfg["snax_versacore_spatial_unrolling"][0]):
        channel_en_B = [0] * B_enabled_channel_CSR_num
        channel_en_B_bits = max(
            8, int((meshCol * tileSize * b_len / bankWidth + 7) // 8 * 8)
        )
        channel_en_B = gen_channel_enable_CSR(
            channel_en_B,
            channel_en_B_bits,
        )
        for j in range(len(channel_en_B)):
            data_str += [format_scalar_define_no_capital(f"channel_en_B_{i}_{j}", channel_en_B[j])]

    # -----------------------------------------------------------
    # ---------------------streamer C settings---------------------
    # -----------------------------------------------------------
    # spatial settings

    for i, (meshRow, tileSize, meshCol) in enumerate(snax_acc_cfg["snax_versacore_spatial_unrolling"][0]):
        if meshCol * meshRow * c_len >= snax_versacore_serial_c_d_width:
            c_spatial_bound_0 = snax_versacore_serial_c_d_width / bankWidth
        else:
            c_spatial_bound_0 = meshCol * meshRow * c_len / bankWidth
        data_str += [format_scalar_define_no_capital(f"Ctlstride0_{i}", int(c_spatial_bound_0 * (bankWidth / 8)))]
        # temporal settings
        # serial input for C
        data_str += [
            format_scalar_define_no_capital(
                f"Ctlbound0_{i}",
                max(
                    1,
                    int(meshCol * meshRow * c_len / snax_versacore_serial_c_d_width),
                ),
            )
        ]

        C_enabled_channel_CSR_num = int(
            math.ceil(snax_versacore_serial_c_d_width / bankWidth / 32)
        )
        channel_en_C = [0] * C_enabled_channel_CSR_num

        # enable full C
        channel_en_C_bits = int((meshRow * meshCol * c_len / bankWidth + 7) // 8 * 8)
        channel_en_C = gen_channel_enable_CSR(
            channel_en_C,
            channel_en_C_bits,
        )
        for j in range(len(channel_en_C)):
            data_str += [format_scalar_define_no_capital(f"channel_en_C_{i}_{j}", channel_en_C[j])]

        channel_en_C = [0] * C_enabled_channel_CSR_num
        channel_en_C_bits = 0
        channel_en_C = gen_channel_enable_CSR(
            channel_en_C,
            channel_en_C_bits,
        )
        for j in range(len(channel_en_C)):
            data_str += [format_scalar_define_no_capital(f"channel_en_C_null_{i}_{j}", channel_en_C[j])]

    # -----------------------------------------------------------
    # streamer D settings
    # -----------------------------------------------------------
    # spatial settings

    for i, (meshRow, tileSize, meshCol) in enumerate(snax_acc_cfg["snax_versacore_spatial_unrolling"][0]):
        if meshCol * meshRow * c_len >= snax_versacore_serial_c_d_width:
            d_spatial_bound_0 = snax_versacore_serial_c_d_width / bankWidth
        else:
            d_spatial_bound_0 = meshCol * meshRow * c_len / bankWidth
        data_str += [
            format_scalar_define_no_capital(f"D32tlstride0_{i}", int(d_spatial_bound_0 * (bankWidth / 8)))
        ]

        # temporal settings
        data_str += [
            format_scalar_define_no_capital(f"D32tlbound0_{i}", max(
                1,
                int(meshCol * meshRow * c_len / snax_versacore_serial_c_d_width),
            ))
        ]

        D_enabled_channel_CSR_num = int(
            math.ceil(snax_versacore_serial_c_d_width / bankWidth / 32)
        )

        channel_en_D = [0] * D_enabled_channel_CSR_num
        channel_en_D_bits = int((meshRow * meshCol * c_len / bankWidth + 7) // 8 * 8)
        channel_en_D = gen_channel_enable_CSR(
            channel_en_D,
            channel_en_D_bits,
        )
        data_str += [
            format_scalar_define_no_capital(f"channel_en_D32_{i}_{j}", channel_en_D[j]) for j in range(len(channel_en_D))
        ]

    data_str = "\n\n".join(data_str)

    return data_str


def main():
    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generate data for kernels")
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        required=True,
        help="Select hardware config file kernel",
    )
    args = parser.parse_args()

    # Load hardware config file
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    # Emit header file
    print(emit_header_file(**hw))


if __name__ == "__main__":
    main()
