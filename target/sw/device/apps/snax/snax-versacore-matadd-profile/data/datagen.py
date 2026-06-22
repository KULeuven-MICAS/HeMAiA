#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import math

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/"))
import _usg_paths  # noqa: F401,E402  (registers util/sim/{common,gemm,xdma,ara} on sys.path)
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

# Add golden model path
from sim_golden_models import (  # noqa E402
    block_gemm_golden_model,
)  # noqa E402
from gemm_sim_utils import align_wide_addr  # noqa E402

np.random.seed(42)


# Add stdint.h header
def emit_header_file(**kwargs):
    emit_str = "#include <stdint.h>\n\n"
    emit_str += emit_matadd_data(**kwargs)
    return emit_str


def signed_int_range(bits):
    min_val = -(2 ** (bits - 1))
    max_val = 2 ** (bits - 1) - 1
    return min_val, max_val


def gen_channel_enable_CSR(channel_en_CSR, channel_en_bits):
    for i in range(channel_en_bits):
        element_index = i // 32  # Determine which element to modify
        bit_position = i % 32  # Position within the element
        if element_index < len(channel_en_CSR):
            channel_en_CSR[element_index] |= 1 << (bit_position)

    channel_en_CSR = [int(x) for x in channel_en_CSR][::-1]
    return channel_en_CSR


def emit_matadd_data(**kwargs):

    # -------------------------------------------------------------
    # matadd workload settings
    # -------------------------------------------------------------
    data_str = []

    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]
    M2 = kwargs.get("M2", M)
    N2 = kwargs.get("N2", N)
    K2 = kwargs.get("K2", 1)
    assert (
        M == 1 and N == 1 and K2 == 1
    ), "two-call accumulator reuse test expects one hardware tile (M=1, N=1, K2=1)"
    assert M2 >= 1 and N2 >= 1, "software-loop dimensions M2/N2 must be positive"

    data_str += [format_scalar_definition("uint32_t", "M", M)]
    data_str += [format_scalar_definition("uint32_t", "K", K)]
    data_str += [format_scalar_definition("uint32_t", "N", N)]
    data_str += [format_scalar_definition("uint32_t", "M2", M2)]
    data_str += [format_scalar_definition("uint32_t", "N2", N2)]
    data_str += [format_scalar_definition("uint32_t", "K2", K2)]

    array_shape = kwargs["array_shape"]
    data_str += [format_scalar_definition("uint32_t", "array_shape", array_shape)]
    data_type = kwargs["data_type"]
    data_str += [format_scalar_definition("uint32_t", "data_type", data_type)]

    # -------------------------------------------------------------
    # -----------------------hardware parameters--------------------
    # --------------------------------------------------------------

    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        0
    ]
    tileSize = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        1
    ]
    meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][
        2
    ]

    a_len = (
        snax_acc_cfg["snax_versacore_input_a_element_width"][data_type]
        if kwargs["int4_a_enable"] == 0
        else 4
    )
    b_len = (
        snax_acc_cfg["snax_versacore_input_b_element_width"][data_type]
        if kwargs["int4_b_enable"] == 0
        else 4
    )
    c_len = snax_acc_cfg["snax_versacore_input_c_element_width"][data_type]

    # if kwargs["int4_a_enable"] == 1 or kwargs["int4_b_enable"] == 1:
    #     assert (
    #         kwargs["array_shape"] == 4
    #     ), "Int4 A and B input is only supported in array shape 4"

    data_str += [
        format_scalar_definition("uint32_t", "int4_a_enable", kwargs["int4_a_enable"])
    ]
    data_str += [
        format_scalar_definition("uint32_t", "int4_b_enable", kwargs["int4_b_enable"])
    ]

    streamer_cfg = kwargs["snax_versacore_streamer_template"]
    if kwargs["stationary"] == 1:
        assert len(streamer_cfg["data_reader_params"]["fifo_depth"]) == 3

    dependency_redudancy = 5
    if kwargs["stationary"] == 1:
        # heuristic data dependency
        assert kwargs["channel_en_C"]
        assert (
            kwargs["M"] * kwargs["N"]
            >= streamer_cfg["data_reader_params"]["fifo_depth"][2]
            + dependency_redudancy
        )

    if kwargs["stationary"] == 2:
        # heuristic data dependency
        assert kwargs["channel_en_C"]
        assert (
            kwargs["M"] * kwargs["N"]
            >= streamer_cfg["data_reader_params"]["fifo_depth"][2]
            + dependency_redudancy
        )

    a_array_width = snax_acc_cfg["snax_versacore_array_input_a_width"]
    b_array_width = snax_acc_cfg["snax_versacore_array_input_b_width"]
    c_array_width = snax_acc_cfg["snax_versacore_array_input_c_width"]
    d_array_width = snax_acc_cfg["snax_versacore_array_output_d_width"]
    assert c_array_width == d_array_width, "C and D array width must be the same"
    snax_versacore_serial_c_d_width = snax_acc_cfg["snax_versacore_serial_c_d_width"]

    bankWidth = 64

    granularity_a = snax_acc_cfg.get("granularity_a", 1)
    granularity_b = snax_acc_cfg.get("granularity_b", 1)
    granularity_c_d = snax_acc_cfg.get("granularity_c_d", 1)

    data_str += [format_scalar_definition("uint32_t", "meshRow", meshRow)]
    data_str += [format_scalar_definition("uint32_t", "tileSize", tileSize)]
    data_str += [format_scalar_definition("uint32_t", "meshCol", meshCol)]

    stationary = kwargs["stationary"]
    data_str += [format_scalar_definition("uint32_t", "stationary", stationary)]
    assert (
        stationary == 0 or stationary == 1 or stationary == 2
    ), "Invalid stationary setting"
    output_stationary = 0
    weight_stationary = 1
    input_stationary = 2
    assert stationary == output_stationary, "matadd profile expects output-stationary mode"
    assert kwargs["channel_en_C"] == 1, "matadd profile needs C enabled"
    assert kwargs["quantization_enable"] == 0, "matadd profile checks int32 D directly"
    assert kwargs["int32tofp16_enable"] == 0, "matadd profile checks int32 D directly"
    assert kwargs["int4_a_enable"] == 0, "matadd profile uses int8 A"
    assert kwargs["int4_b_enable"] == 0, "matadd profile uses int8 B"
    assert (
        snax_acc_cfg["snax_versacore_input_a_data_type"][data_type] == "SInt"
    ), "matadd profile only supports signed integer A"
    assert (
        snax_acc_cfg["snax_versacore_input_b_data_type"][data_type] == "SInt"
    ), "matadd profile only supports signed integer B"
    assert (
        snax_acc_cfg["snax_versacore_input_c_data_type"][data_type] == "SInt"
    ), "matadd profile only supports signed integer C"
    assert a_len == 8, "matadd profile uses int8 A"
    assert b_len == 8, "matadd profile uses int8 B"
    assert c_len == 32, "matadd profile uses int32 C/D"
    assert (
        kwargs["transposed_B"] == 1
    ), "software-loop N2 tiling expects transposed_B=1"
    # -------------------------------------------------------------
    # -----------------------A streamer setting-------------------------------
    # --------------------------------------------------------------
    data_str += [format_scalar_definition("int32_t", "Aslstride0", bankWidth / 8)]

    Atlstride0 = a_len * tileSize * meshRow / 8
    Atlstride1 = 0
    Atlstride2 = K * a_len * tileSize * meshRow / 8
    Atlstride3 = 0
    Atlstride4 = 0
    Atlstride5 = 0

    if stationary == output_stationary:
        Atlbound0 = K
        Atlstride0 = a_len * tileSize * meshRow / 8
        Atlbound1 = N
        Atlstride1 = 0
        Atlbound2 = M
        Atlstride2 = K * a_len * tileSize * meshRow / 8
    elif stationary == weight_stationary:
        Atlbound0 = M
        Atlstride0 = K * a_len * tileSize * meshRow / 8
        Atlbound1 = N
        Atlstride1 = 0
        Atlbound2 = K
        Atlstride2 = a_len * tileSize * meshRow / 8
    elif stationary == input_stationary:
        Atlbound0 = N
        Atlstride0 = 0
        Atlbound1 = M
        Atlstride1 = K * a_len * tileSize * meshRow / 8
        Atlbound2 = K
        Atlstride2 = a_len * tileSize * meshRow / 8

    # Checker for tstrides of operand A
    assert Atlstride0 % (bankWidth / 8 * granularity_a) == 0
    assert Atlstride1 % (bankWidth / 8 * granularity_a) == 0
    assert Atlstride2 % (bankWidth / 8 * granularity_a) == 0
    assert Atlstride3 % (bankWidth / 8 * granularity_a) == 0
    assert Atlstride4 % (bankWidth / 8 * granularity_a) == 0
    assert Atlstride5 % (bankWidth / 8 * granularity_a) == 0

    data_str += [format_scalar_definition("int32_t", "Atlbound0", Atlbound0)]
    data_str += [format_scalar_definition("int32_t", "Atlstride0", Atlstride0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound1", Atlbound1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride1", Atlstride1)]
    data_str += [format_scalar_definition("int32_t", "Atlbound2", Atlbound2)]
    data_str += [format_scalar_definition("int32_t", "Atlstride2", Atlstride2)]

    data_str += [format_scalar_definition("int32_t", "Atlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride3", Atlstride3)]
    data_str += [format_scalar_definition("int32_t", "Atlbound4", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride4", Atlstride4)]
    data_str += [format_scalar_definition("int32_t", "Atlbound5", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride5", Atlstride5)]

    A_enabled_channel_CSR_num = int(math.ceil(a_array_width / bankWidth / 32))
    channel_en_A = [0] * A_enabled_channel_CSR_num
    channel_en_A_bits = max(1, int(math.ceil(meshRow * tileSize * a_len / bankWidth)))
    channel_en_A = gen_channel_enable_CSR(
        channel_en_A,
        channel_en_A_bits,
    )
    data_str += [
        "int32_t channel_en_A[] = { " + ", ".join(map(str, channel_en_A)) + " };"
    ]

    a_k_tile_data_length = int(meshRow * tileSize * a_len / 8)
    a_sw_tile_data_length = int(K * a_k_tile_data_length)
    a_data_length = int(M2 * a_sw_tile_data_length)
    data_str += [
        format_scalar_definition(
            "int32_t", "a_sw_tile_data_length", a_sw_tile_data_length
        )
    ]
    data_str += [format_scalar_definition("int32_t", "a_data_length", a_data_length)]

    # -------------------------------------------------------------
    # -----------------------B setting-------------------------------
    # --------------------------------------------------------------

    data_str += [format_scalar_definition("int32_t", "Bslstride0", bankWidth / 8)]

    b_k_tile_data_length = int(tileSize * meshCol * b_len / 8)

    if stationary == output_stationary:
        Btlbound0 = K
        Btlstride0 = N2 * b_k_tile_data_length
        Btlbound1 = N
        Btlstride1 = K * N2 * b_k_tile_data_length
        Btlbound2 = M
        Btlstride2 = 0
    elif stationary == weight_stationary:
        Btlbound0 = M
        Btlstride0 = 0
        Btlbound1 = N
        Btlstride1 = K * b_len * tileSize * meshCol / 8
        Btlbound2 = K
        Btlstride2 = b_len * tileSize * meshCol / 8
    elif stationary == input_stationary:
        Btlbound0 = N
        Btlstride0 = K * b_len * tileSize * meshCol / 8
        Btlbound1 = M
        Btlstride1 = 0
        Btlbound2 = K
        Btlstride2 = b_len * tileSize * meshCol / 8

    # Checker for tstrides of operand B
    assert Btlstride0 % (bankWidth / 8 * granularity_b) == 0
    assert Btlstride1 % (bankWidth / 8 * granularity_b) == 0
    assert Btlstride2 % (bankWidth / 8 * granularity_b) == 0

    data_str += [format_scalar_definition("int32_t", "Btlbound0", Btlbound0)]
    data_str += [format_scalar_definition("int32_t", "Btlstride0", Btlstride0)]
    data_str += [format_scalar_definition("int32_t", "Btlbound1", Btlbound1)]
    data_str += [format_scalar_definition("int32_t", "Btlstride1", Btlstride1)]
    data_str += [format_scalar_definition("int32_t", "Btlbound2", Btlbound2)]
    data_str += [format_scalar_definition("int32_t", "Btlstride2", Btlstride2)]

    B_enabled_channel_CSR_num = int(math.ceil(b_array_width / bankWidth / 32))
    channel_en_B = [0] * B_enabled_channel_CSR_num
    channel_en_B_bits = max(1, int(math.ceil(meshCol * tileSize * b_len / bankWidth)))
    channel_en_B = gen_channel_enable_CSR(
        channel_en_B,
        channel_en_B_bits,
    )
    data_str += [
        "int32_t channel_en_B[] = { " + ", ".join(map(str, channel_en_B)) + " };"
    ]

    data_str += [
        format_scalar_definition("int32_t", "b_k_tile_data_length", b_k_tile_data_length)
    ]
    b_data_length = int(K * N2 * b_k_tile_data_length)
    data_str += [format_scalar_definition("int32_t", "b_data_length", b_data_length)]

    # -----------------------------------------------------------
    # ---------------------streamer C settings---------------------
    # -----------------------------------------------------------
    # spatial settings
    data_str += [format_scalar_definition("int32_t", "Cslstride0", bankWidth / 8)]
    if meshCol * meshRow * c_len >= snax_versacore_serial_c_d_width:
        c_spatial_bound_0 = snax_versacore_serial_c_d_width / bankWidth
    else:
        c_spatial_bound_0 = meshCol * meshRow * c_len / bankWidth
    # temporal settings
    # serial input for C
    Ctlbound0 = max(
        1,
        meshCol * meshRow * c_len / snax_versacore_serial_c_d_width,
    )
    Ctlstride0 = c_spatial_bound_0 * (bankWidth / 8)

    if stationary == output_stationary:
        Ctlbound1 = N
        Ctlstride1 = c_len * meshRow * meshCol / 8
        Ctlbound2 = M
        Ctlstride2 = N * c_len * meshRow * meshCol / 8
        # C is not used in this case
        Ctlbound3 = 1
        Ctlstride3 = 0
    elif stationary == weight_stationary:
        Ctlbound1 = M
        Ctlstride1 = N * c_len * meshRow * meshCol / 8
        Ctlbound2 = N
        Ctlstride2 = c_len * meshRow * meshCol / 8
        Ctlbound3 = K
        Ctlstride3 = 0
    elif stationary == input_stationary:
        Ctlbound1 = N
        Ctlstride1 = c_len * meshRow * meshCol / 8
        Ctlbound2 = M
        Ctlstride2 = N * c_len * meshRow * meshCol / 8
        Ctlbound3 = K
        Ctlstride3 = 0

    # Checker for tstrides of operand C
    assert Ctlstride0 % (bankWidth / 8 * granularity_c_d) == 0
    assert Ctlstride1 % (bankWidth / 8 * granularity_c_d) == 0
    assert Ctlstride2 % (bankWidth / 8 * granularity_c_d) == 0
    assert Ctlstride3 % (bankWidth / 8 * granularity_c_d) == 0

    data_str += [format_scalar_definition("int32_t", "Ctlbound0", Ctlbound0)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride0", Ctlstride0)]
    data_str += [format_scalar_definition("int32_t", "Ctlbound1", Ctlbound1)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride1", Ctlstride1)]
    data_str += [format_scalar_definition("int32_t", "Ctlbound2", Ctlbound2)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride2", Ctlstride2)]
    data_str += [format_scalar_definition("int32_t", "Ctlbound3", Ctlbound3)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride3", Ctlstride3)]

    disable_C = kwargs["channel_en_C"] == 0
    enable_full_C = kwargs["channel_en_C"] == 1

    assert disable_C or enable_full_C, "Invalid C settings"

    C_enabled_channel_CSR_num = int(
        math.ceil(snax_versacore_serial_c_d_width / bankWidth / 32)
    )
    channel_en_C = [0] * C_enabled_channel_CSR_num

    if enable_full_C == 1:
        channel_en_C_bits = max(
            1,
            min(
                int(math.ceil(meshRow * meshCol * c_len / bankWidth)),
                int(math.ceil(snax_versacore_serial_c_d_width / bankWidth)),
            ),
        )
    else:
        channel_en_C_bits = 0

    channel_en_C = gen_channel_enable_CSR(
        channel_en_C,
        channel_en_C_bits,
    )
    data_str += [
        "int32_t channel_en_C[] = { " + ", ".join(map(str, channel_en_C)) + " };"
    ]
    data_str += [
        "int32_t channel_en_C_zero[] = { "
        + ", ".join(map(str, [0] * C_enabled_channel_CSR_num))
        + " };"
    ]

    c_tile_data_length = int(meshRow * meshCol * c_len / 8)
    c_data_length = int(M2 * N2 * c_tile_data_length)
    data_str += [
        format_scalar_definition("int32_t", "c_tile_data_length", c_tile_data_length)
    ]
    data_str += [format_scalar_definition("int32_t", "c_data_length", c_data_length)]

    # -----------------------------------------------------------
    # streamer D settings
    # -----------------------------------------------------------
    # spatial settings

    non_datapath_extension_d_len = c_len
    if kwargs["quantization_enable"] == 1:
        datapath_extension_d_len = 8
    elif kwargs["int32tofp16_enable"] == 1:
        datapath_extension_d_len = 16
    else:
        datapath_extension_d_len = c_len
    data_str += [format_scalar_definition("int32_t", "D32slstride0", bankWidth / 8)]
    data_str += [
        format_scalar_definition(
            "int32_t", "quantization_enable", kwargs["quantization_enable"]
        )
    ]
    if kwargs["quantization_enable"] == 1 or kwargs["int32tofp16_enable"] == 1:
        actual_d_width = snax_versacore_serial_c_d_width
    elif (
        meshCol * meshRow * non_datapath_extension_d_len
        >= snax_versacore_serial_c_d_width
    ):
        actual_d_width = snax_versacore_serial_c_d_width
    else:
        actual_d_width = meshCol * meshRow * non_datapath_extension_d_len

    d_spatial_bound_0 = actual_d_width / bankWidth

    # temporal settings
    if kwargs["quantization_enable"] == 1 or kwargs["int32tofp16_enable"] == 1:
        if (
            meshCol * meshRow * datapath_extension_d_len
            > snax_versacore_serial_c_d_width
        ):
            D32tlbound0 = (
                meshCol
                * meshRow
                * datapath_extension_d_len
                / snax_versacore_serial_c_d_width
            )
        else:
            D32tlbound0 = 1
            assert (
                kwargs["M"]
                * kwargs["N"]
                * meshRow
                * meshCol
                * datapath_extension_d_len
                % snax_versacore_serial_c_d_width
                == 0
            ), "The quantization extension cannot output correct result."
    elif (
        meshCol * meshRow * non_datapath_extension_d_len
        > snax_versacore_serial_c_d_width
    ):
        D32tlbound0 = (
            meshCol
            * meshRow
            * non_datapath_extension_d_len
            / snax_versacore_serial_c_d_width
        )
    else:
        D32tlbound0 = 1

    D32tlstride0 = d_spatial_bound_0 * (bankWidth / 8)

    if stationary == output_stationary:
        if kwargs["quantization_enable"] == 1 or kwargs["int32tofp16_enable"] == 1:
            output_matrix_per_store = (
                1
                if meshCol * meshRow * datapath_extension_d_len
                > snax_versacore_serial_c_d_width
                else snax_versacore_serial_c_d_width
                / (meshCol * meshRow * datapath_extension_d_len)
            )
            data_str += [
                format_scalar_definition(
                    "int32_t", "output_matrix_per_store", output_matrix_per_store
                )
            ]
            D32tlbound1 = N * M / output_matrix_per_store
            D32tlstride1 = (
                output_matrix_per_store
                * datapath_extension_d_len
                * meshRow
                * meshCol
                / 8
            )
            D32tlbound2 = 1
            D32tlstride2 = 0
        else:
            D32tlbound1 = N
            D32tlstride1 = non_datapath_extension_d_len * meshRow * meshCol / 8
            D32tlbound2 = M
            D32tlstride2 = N * non_datapath_extension_d_len * meshRow * meshCol / 8

        # D is not used in this case
        D32tlbound3 = 1
        D32tlstride3 = 0

    elif stationary == weight_stationary:
        assert kwargs["quantization_enable"] == 0, "invalid configuration"
        D32tlbound1 = M
        D32tlstride1 = N * non_datapath_extension_d_len * meshRow * meshCol / 8
        D32tlbound2 = N
        D32tlstride2 = non_datapath_extension_d_len * meshRow * meshCol / 8
        D32tlbound3 = K
        D32tlstride3 = 0

    elif stationary == input_stationary:
        assert kwargs["quantization_enable"] == 0, "invalid configuration"
        D32tlbound1 = N
        D32tlstride1 = non_datapath_extension_d_len * meshRow * meshCol / 8
        D32tlbound2 = M
        D32tlstride2 = N * non_datapath_extension_d_len * meshRow * meshCol / 8
        D32tlbound3 = K
        D32tlstride3 = 0

    # Checker for tstrides of operand D
    assert D32tlstride0 % (bankWidth / 8 * granularity_c_d) == 0
    assert D32tlstride1 % (bankWidth / 8 * granularity_c_d) == 0
    assert D32tlstride2 % (bankWidth / 8 * granularity_c_d) == 0
    assert D32tlstride3 % (bankWidth / 8 * granularity_c_d) == 0

    data_str += [format_scalar_definition("int32_t", "D32tlbound0", D32tlbound0)]
    data_str += [format_scalar_definition("int32_t", "D32tlstride0", D32tlstride0)]
    data_str += [format_scalar_definition("int32_t", "D32tlbound1", D32tlbound1)]
    data_str += [format_scalar_definition("int32_t", "D32tlstride1", D32tlstride1)]
    data_str += [format_scalar_definition("int32_t", "D32tlbound2", D32tlbound2)]
    data_str += [format_scalar_definition("int32_t", "D32tlstride2", D32tlstride2)]
    data_str += [format_scalar_definition("int32_t", "D32tlbound3", D32tlbound3)]
    data_str += [format_scalar_definition("int32_t", "D32tlstride3", D32tlstride3)]

    D_enabled_channel_CSR_num = int(
        math.ceil(snax_versacore_serial_c_d_width / bankWidth / 32)
    )

    channel_en_D = [0] * D_enabled_channel_CSR_num
    if kwargs["quantization_enable"] == 1 or kwargs["int32tofp16_enable"] == 1:
        channel_en_D_bits = int(math.ceil(snax_versacore_serial_c_d_width / bankWidth))
    else:
        channel_en_D_bits = min(
            int(
                math.ceil(meshRow * meshCol * non_datapath_extension_d_len / bankWidth)
            ),
            int(math.ceil(snax_versacore_serial_c_d_width / bankWidth)),
        )

    channel_en_D = gen_channel_enable_CSR(
        channel_en_D,
        channel_en_D_bits,
    )
    data_str += [
        "int32_t channel_en_D[] = { " + ", ".join(map(str, channel_en_D)) + " };"
    ]

    d_tile_data_length = int(
        meshRow
        * meshCol
        * (
            datapath_extension_d_len
            if kwargs["quantization_enable"] == 1 or kwargs["int32tofp16_enable"] == 1
            else non_datapath_extension_d_len
        )
        / 8
    )
    d_data_length = int(M2 * N2 * d_tile_data_length)
    data_str += [
        format_scalar_definition("int32_t", "d_tile_data_length", d_tile_data_length)
    ]
    data_str += [format_scalar_definition("int32_t", "d_data_length", d_data_length)]

    # -----------------------------------------------------------
    # -------------------------base address----------------------
    # -----------------------------------------------------------

    delta_local_a = 0
    delta_local_a = align_wide_addr(delta_local_a, snax_acc_cfg["granularity_a"] * 8)
    delta_local_b = a_data_length
    # the address alignment for B is 128 bytes
    # in the new sparse interconnect
    # as the data granularity now is 16*64 bits!!!
    delta_local_b = align_wide_addr(delta_local_b, snax_acc_cfg["granularity_b"] * 8)
    delta_local_c = delta_local_b + b_data_length
    # the address alignment for B is 32 bytes
    # in the new sparse interconnect
    # as the data granularity now is 4*64 bits!!!
    delta_local_c = align_wide_addr(delta_local_c, snax_acc_cfg["granularity_c_d"] * 8)

    if stationary == output_stationary:
        delta_local_d = delta_local_c
    elif stationary == weight_stationary:
        delta_local_d = delta_local_c
    elif stationary == input_stationary:
        delta_local_d = delta_local_c

    data_str += [format_scalar_definition("int32_t", "delta_local_a", delta_local_a)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b", delta_local_b)]
    data_str += [format_scalar_definition("int32_t", "delta_local_c", delta_local_c)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d", delta_local_d)]

    # -----------------------------------------------------------
    # Test Data generation
    # -----------------------------------------------------------

    # Generating random 8 integer a and b for subtraction
    # subtraction_a = np.random.randint(MIN, MAX)
    # subtraction_b = np.random.randint(MIN, MAX)
    subtraction_a = 0
    subtraction_b = 0

    # Writing the subtraction value to data.h
    data_str += [format_scalar_definition("int8_t", "subtraction_a", subtraction_a)]
    data_str += [format_scalar_definition("int8_t", "subtraction_b", subtraction_b)]

    A_MIN, A_MAX = signed_int_range(a_len)
    B_MIN, B_MAX = signed_int_range(b_len)
    C_MIN, C_MAX = -2_000_000, 2_000_000

    A = np.random.randint(A_MIN, A_MAX, size=(M2, K, meshRow, tileSize))
    data_str += [
        format_vector_definition(
            "int8_t", "A", A.reshape(-1), hex_bits=8, cast_hex=True
        )
    ]

    B = np.random.randint(B_MIN, B_MAX, size=(K, N2, tileSize, meshCol))
    data_str += [
        format_vector_definition(
            "int8_t", "B", B.reshape(-1), hex_bits=8, cast_hex=True
        )
    ]

    C = np.random.randint(C_MIN, C_MAX, size=(M2, N2, meshRow, meshCol)).reshape(-1)
    data_str += [
        format_vector_definition("int32_t", "C", C, hex_bits=32, cast_hex=True)
    ]

    A = A.reshape(-1)
    if kwargs["transposed_A"] == 1:
        A = A.reshape(M2, K, meshRow, tileSize)
        A = A.transpose(0, 1, 3, 2).reshape(-1)
    # Streamer memory is K-major; the golden model consumes B as N-major.
    B = B.transpose(1, 0, 3, 2).reshape(-1)

    data_str += [
        format_scalar_definition("int32_t", "transposed_A", kwargs["transposed_A"])
    ]
    data_str += [
        format_scalar_definition("int32_t", "transposed_B", kwargs["transposed_B"])
    ]

    zero_C = np.zeros_like(C)
    AB = block_gemm_golden_model(
        M2,
        K,
        N2,
        meshRow,
        tileSize,
        meshCol,
        A,
        B,
        subtraction_a,
        subtraction_b,
        zero_C,
    )

    D1 = block_gemm_golden_model(
        M2,
        K,
        N2,
        meshRow,
        tileSize,
        meshCol,
        A,
        B,
        subtraction_a,
        subtraction_b,
        C,
    )

    D2 = D1 + AB
    data_str += [
        format_vector_definition("int32_t", "D1", D1, hex_bits=32, cast_hex=True)
    ]
    data_str += [
        format_vector_definition("int32_t", "D2", D2, hex_bits=32, cast_hex=True)
    ]

    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_C", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D32", 0)]

    data_str += [format_scalar_definition("uint32_t", "shift_i", 0)]
    data_str += [format_scalar_definition("uint32_t", "multiplier_i", 1)]
    data_str += [format_scalar_definition("int32_t", "input_zp_i", 0)]
    data_str += [format_scalar_definition("int32_t", "output_zp_i", 0)]
    data_str += [format_scalar_definition("int32_t", "int32tofp16_enable", 0)]

    data_str = "\n\n".join(data_str)

    return data_str


def main():
    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generate data for kernels")
    parser.add_argument(
        "--swcfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        required=True,
        help="Select hardware config file kernel",
    )
    args = parser.parse_args()

    # Load param config file
    with args.swcfg.open() as f:
        param = hjson.loads(f.read())

    # Load hardware config file
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    # Merge dictionaries (hw overrides param in case of conflicts)
    merged_config = {**param, **hw}

    # Emit header file
    print(emit_header_file(**merged_config))


if __name__ == "__main__":
    main()
