#!/usr/bin/env python3

# Copyright 2024 KU Leuven.
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

# Add data utility path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

# Add golden model path
from snax_utils import (  # noqa E402
    block_gemm_golden_model,
    postprocessing_simd_golden_model,
    align_wide_addr,
)  # noqa E402

np.random.seed(42)


# Add stdint.h header
def emit_header_file(**kwargs):
    emit_str = "#include <stdint.h>\n\n"
    emit_str += emit_mha_data(**kwargs)
    return emit_str


MIN = -128
MAX = 127

bankWidth = 64
input_data_width = 8
output_data_width = 32
quantized_output_data_width = 8


def emit_mha_data(**kwargs):

    meshRow = kwargs["meshRow"]
    tileSize = kwargs["tileSize"]
    meshCol = kwargs["meshCol"]

    data_str = []

    data_str += [format_scalar_definition("int", "Batch", 1)]

    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # generate streamer config for Q=Xwq, K=XWk, V=XWv
    # --------------------------------------------------------------
    # --------------------------------------------------------------
    data_str += [format_scalar_definition("int", "M_1", kwargs["M_1"])]
    data_str += [format_scalar_definition("int", "K_1", kwargs["K_1"])]
    data_str += [format_scalar_definition("int", "N_1", kwargs["N_1"])]

    # Aslstride0 is idle
    data_str += [format_scalar_definition("int32_t", "Aslstride0_1", 1)]
    data_str += [format_scalar_definition("int32_t", "Aslstride1_1", bankWidth / 8)]
    data_str += [format_scalar_definition("int32_t", "Atlbound0_1", kwargs["K_1"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Atlstride0_1", input_data_width * tileSize * meshRow / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Atlbound1_1", kwargs["N_1"])]
    data_str += [format_scalar_definition("int32_t", "Atlstride1_1", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound2_1", kwargs["M_1"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Atlstride2_1",
            kwargs["K_1"] * input_data_width * tileSize * meshRow / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Atlbound3_1", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride3_1", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound4_1", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride4_1", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound5_1", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride5_1", 0)]

    data_str += [format_scalar_definition("int32_t", "Bslstride0_1", 1)]
    data_str += [format_scalar_definition("int32_t", "Bslstride1_1", bankWidth / 8)]
    data_str += [format_scalar_definition("int32_t", "Btlbound0_1", kwargs["K_1"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Btlstride0_1", input_data_width * tileSize * meshCol / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Btlbound1_1", kwargs["N_1"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Btlstride1_1",
            kwargs["K_1"] * input_data_width * tileSize * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Btlbound2_1", kwargs["M_1"])]
    data_str += [format_scalar_definition("int32_t", "Btlstride2_1", 0)]

    data_str += [format_scalar_definition("int32_t", "Cslstride0_1", bankWidth / 8)]
    c32_spatial_bound_0 = 8
    data_str += [
        format_scalar_definition(
            "int32_t", "Cslstride1_1", c32_spatial_bound_0 * (bankWidth / 8)
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound0_1", kwargs["N_1"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Ctlstride0_1", output_data_width * meshRow * meshCol / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound1_1", kwargs["M_1"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Ctlstride1_1",
            kwargs["N_1"] * output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound2_1", 1)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride2_1", 0)]

    data_str += [format_scalar_definition("int32_t", "D8slstride0_1", 1)]
    data_str += [format_scalar_definition("int32_t", "D8slstride1_1", bankWidth / 8)]
    data_str += [format_scalar_definition("int32_t", "D8tlbound0_1", kwargs["N_1"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D8tlstride0_1",
            quantized_output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D8tlbound1_1", kwargs["M_1"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D8tlstride1_1",
            kwargs["N_1"] * quantized_output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D8tlbound2_1", 1)]
    data_str += [format_scalar_definition("int32_t", "D8tlstride2_1", 0)]


    # --------------------------------------------------------------
    # ----------------------------------------------------S=QKt, Z=SV
    # --------------------------------------------------------------
    # --------------------------------------------------------------
    data_str += [format_scalar_definition("int", "M_2", kwargs["M_2"])]
    data_str += [format_scalar_definition("int", "K_2", kwargs["K_2"])]
    data_str += [format_scalar_definition("int", "N_2", kwargs["N_2"])]

    # Aslstride0 is idle
    data_str += [format_scalar_definition("int32_t", "Aslstride0_2", 1)]
    data_str += [format_scalar_definition("int32_t", "Aslstride1_2", bankWidth / 8)]
    data_str += [format_scalar_definition("int32_t", "Atlbound0_2", kwargs["K_2"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Atlstride0_2", input_data_width * tileSize * meshRow / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Atlbound1_2", kwargs["N_2"])]
    data_str += [format_scalar_definition("int32_t", "Atlstride1_2", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound2_2", kwargs["M_2"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Atlstride2_2",
            kwargs["K_2"] * input_data_width * tileSize * meshRow / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Atlbound3_2", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride3_2", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound4_2", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride4_2", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound5_2", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride5_2", 0)]

    data_str += [format_scalar_definition("int32_t", "Bslstride0_2", 1)]
    data_str += [format_scalar_definition("int32_t", "Bslstride1_2", bankWidth / 8)]
    data_str += [format_scalar_definition("int32_t", "Btlbound0_2", kwargs["K_2"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Btlstride0_2", input_data_width * tileSize * meshCol / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Btlbound1_2", kwargs["N_2"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Btlstride1_2",
            kwargs["K_2"] * input_data_width * tileSize * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Btlbound2_2", kwargs["M_2"])]
    data_str += [format_scalar_definition("int32_t", "Btlstride2_2", 0)]

    data_str += [format_scalar_definition("int32_t", "Cslstride0_2", bankWidth / 8)]
    c32_spatial_bound_0 = 8
    data_str += [
        format_scalar_definition(
            "int32_t", "Cslstride1_2", c32_spatial_bound_0 * (bankWidth / 8)
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound0_2", kwargs["N_2"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "Ctlstride0_2", output_data_width * meshRow * meshCol / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound1_2", kwargs["M_2"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "Ctlstride1_2",
            kwargs["N_2"] * output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "Ctlbound2_2", 1)]
    data_str += [format_scalar_definition("int32_t", "Ctlstride2_2", 0)]

    data_str += [format_scalar_definition("int32_t", "D8slstride0_2", 1)]
    data_str += [format_scalar_definition("int32_t", "D8slstride1_2", bankWidth / 8)]
    data_str += [format_scalar_definition("int32_t", "D8tlbound0_2", kwargs["N_2"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D8tlstride0_2",
            quantized_output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D8tlbound1_2", kwargs["M_2"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D8tlstride1_2",
            kwargs["N_2"] * quantized_output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D8tlbound2_2", 1)]
    data_str += [format_scalar_definition("int32_t", "D8tlstride2_2", 0)]

    data_str += [format_scalar_definition("int32_t", "delta_local_d32", 0)]


    ###################################memory allcation shedule#####################

    delta_local_a_X = 0
    # Q=XWq
    delta_local_b_Wq = (
        kwargs["K_1"] * kwargs["M_1"] * (meshRow * tileSize * input_data_width / 8)
    )
    delta_local_b_Wq = align_wide_addr(delta_local_b_Wq, 64)
    delta_local_d8_Q = delta_local_b_Wq + kwargs["K_1"] * kwargs["N_1"] * (
        meshCol * tileSize * input_data_width / 8
    )
    delta_local_d8_Q = align_wide_addr(delta_local_d8_Q, 64)
    # K=XWk
    delta_local_b_Wk = delta_local_d8_Q + kwargs["M_1"] * kwargs["N_1"] * (
        meshCol * tileSize * quantized_output_data_width / 8
    )
    delta_local_b_Wk = align_wide_addr(delta_local_b_Wk, 64)
    delta_local_d8_K = delta_local_b_Wk + kwargs["K_1"] * kwargs["N_1"] * (
        meshCol * tileSize * quantized_output_data_width / 8
    )
    delta_local_d8_K = align_wide_addr(delta_local_d8_K, 64)
    # S=QK
    # replace Wq
    delta_local_d8_S = delta_local_b_Wq
    # V=XWv
    delta_local_b_Wv = delta_local_b_Wk
    delta_local_d8_V = delta_local_d8_Q
    # Z=SV
    delta_local_d8_Z = delta_local_d8_K

    data_str += [format_scalar_definition("int32_t", "delta_local_a_X", delta_local_a_X)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b_Wq", delta_local_b_Wq)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d8_Q", delta_local_d8_Q)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b_Wk", delta_local_b_Wk)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d8_K", delta_local_d8_K)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d8_S", delta_local_d8_S)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b_Wv", delta_local_b_Wv)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d8_V", delta_local_d8_V)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d8_Z", delta_local_d8_Z)]

    # --------------------------------------------------
    # --------------------------------------------------
    # -----------other configuration gen ---------------
    # --------------------------------------------------
    # --------------------------------------------------
    # Generating random 8 integer a and b for subtraction
    subtraction_a = np.random.randint(MIN, MAX)
    subtraction_b = np.random.randint(MIN, MAX)

    # Writing the subtraction value to data.h
    data_str += [format_scalar_definition("int8_t", "subtraction_a", subtraction_a)]
    data_str += [format_scalar_definition("int8_t", "subtraction_b", subtraction_b)]

    disable_C = kwargs["broadcast_C"] == 0 and kwargs["channel_en_C"] == 0

    assert disable_C, "Invalid C settings"

    data_str += [format_scalar_definition("int32_t", "channel_en_C", 0)]

    data_str += [
        format_scalar_definition("int32_t", "broadcast_C", kwargs["broadcast_C"])
    ]

    data_str += [
        format_scalar_definition("int32_t", "transposed_A_1", kwargs["transposed_A_1"])
    ]
    data_str += [
        format_scalar_definition("int32_t", "transposed_B_1", kwargs["transposed_B_1"])
    ]
    
    data_str += [
        format_scalar_definition("int32_t", "transposed_A_2", kwargs["transposed_A_2"])
    ]
    data_str += [
        format_scalar_definition("int32_t", "transposed_B_2", kwargs["transposed_B_2"])
    ]

    # Postprocessing cfg
    bypassSIMD = kwargs["bypassSIMD"]
    data_str += [format_scalar_definition("int32_t", "bypassSIMD", bypassSIMD)]

    # Generating random constant values
    group_num = 8
    input_zp_i = np.random.randint(MIN, MAX)
    output_zp_i = np.random.randint(MIN, MAX)
    max_int_i = MAX
    min_int_i = MIN
    double_round_i = np.random.randint(0, 1)

    shift_i = np.random.randint(0, 63, size=group_num)  # values between 0-63
    multiplier_i = np.random.randint(-(2**31), 2**31 - 1, size=group_num)

    # Writing the constant values to data.h
    data_str += [
        format_scalar_definition("int8_t", "input_zp_i", input_zp_i),
        format_scalar_definition("int8_t", "output_zp_i", output_zp_i),
        format_scalar_definition("int8_t", "max_int_i", max_int_i),
        format_scalar_definition("int8_t", "min_int_i", min_int_i),
        format_scalar_definition("int8_t", "double_round_i", double_round_i),
    ]

    shared_bitpacked_shift0 = (
        (shift_i[3] << 24) | (shift_i[2] << 16) | (shift_i[1] << 8) | shift_i[0]
    )
    shared_bitpacked_shift1 = (
        (shift_i[7] << 24) | (shift_i[6] << 16) | (shift_i[5] << 8) | shift_i[4]
    )
    data_str += [
        format_scalar_definition(
            "int32_t", "shared_bitpacked_shift0", shared_bitpacked_shift0
        )
    ]
    data_str += [
        format_scalar_definition(
            "int32_t", "shared_bitpacked_shift1", shared_bitpacked_shift1
        )
    ]

    data_str += [
        format_scalar_definition("int32_t", "shared_multiplier0", multiplier_i[0])
    ]
    data_str += [
        format_scalar_definition("int32_t", "shared_multiplier1", multiplier_i[1])
    ]
    data_str += [
        format_scalar_definition("int32_t", "shared_multiplier2", multiplier_i[2])
    ]
    data_str += [
        format_scalar_definition("int32_t", "shared_multiplier3", multiplier_i[3])
    ]
    data_str += [
        format_scalar_definition("int32_t", "shared_multiplier4", multiplier_i[4])
    ]
    data_str += [
        format_scalar_definition("int32_t", "shared_multiplier5", multiplier_i[5])
    ]
    data_str += [
        format_scalar_definition("int32_t", "shared_multiplier6", multiplier_i[6])
    ]
    data_str += [
        format_scalar_definition("int32_t", "shared_multiplier7", multiplier_i[7])
    ]

    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_C", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D32", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D8", 0)]

    # D32 data streamer cfg, no need to change
    data_str += [format_scalar_definition("int32_t", "D32slstride0", bankWidth / 8)]
    d32_spatial_bound_0 = 8
    data_str += [
        format_scalar_definition(
            "int32_t", "D32slstride1", d32_spatial_bound_0 * (bankWidth / 8)
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D32tlbound0", kwargs["N_1"])]
    data_str += [
        format_scalar_definition(
            "int32_t", "D32tlstride0", output_data_width * meshRow * meshCol / 8
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D32tlbound1", kwargs["M_1"])]
    data_str += [
        format_scalar_definition(
            "int32_t",
            "D32tlstride1",
            kwargs["N_1"] * output_data_width * meshRow * meshCol / 8,
        )
    ]
    data_str += [format_scalar_definition("int32_t", "D32tlbound2", 1)]
    data_str += [format_scalar_definition("int32_t", "D32tlstride2", 0)]

    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------
    ###############test data generation and golden model###############
    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------

    X = np.random.randint(
        MIN, MAX, size=( kwargs["M_1"], kwargs["K_1"], meshRow, tileSize)
    ).reshape(-1)
    data_str += [format_vector_definition("int8_t", "X", X)]

    Wq = np.random.randint(
        MIN, MAX, size=(kwargs["N_1"], kwargs["K_1"], meshRow, tileSize)
    ).reshape(-1)
    data_str += [format_vector_definition("int8_t", "Wq", Wq)]

    Wk = np.random.randint(
        MIN, MAX, size=(kwargs["N_1"], kwargs["K_1"], meshRow, tileSize)
    ).reshape(-1)
    data_str += [format_vector_definition("int8_t", "Wk", Wk)]

    Wv = np.random.randint(
        MIN, MAX, size=(kwargs["N_1"], kwargs["K_1"], meshRow, tileSize)
    ).reshape(-1)
    data_str += [format_vector_definition("int8_t", "Wv", Wv)]

    # Q=XWq
    Q = block_gemm_golden_model(
        kwargs["M_1"],
        kwargs["K_1"],
        kwargs["N_1"],
        meshRow,
        tileSize,
        meshCol,
        X,
        Wq,
        subtraction_a,
        subtraction_b,
        np.zeros(shape=(kwargs["M_1"], kwargs["N_1"], meshRow, meshCol), dtype=np.int32).reshape(-1),
    )

    Q8 = np.zeros_like(Q, dtype=np.uint8)
    for i in range(group_num):
        Q8[i::group_num] = postprocessing_simd_golden_model(
            Q[i::group_num],
            input_zp_i,
            output_zp_i,
            shift_i[i],
            max_int_i,
            min_int_i,
            double_round_i,
            multiplier_i[i],
        )
    data_str += [format_vector_definition("int8_t", "Q8_golden", Q8)]

    # K = XWk
    K = block_gemm_golden_model(
        kwargs["M_1"],
        kwargs["K_1"],
        kwargs["N_1"],
        meshRow,
        tileSize,
        meshCol,
        X,
        Wk,
        subtraction_a,
        subtraction_b,
        np.zeros(shape=(kwargs["M_1"], kwargs["N_1"], meshRow, meshCol), dtype=np.int32).reshape(-1),
    )

    K8 = np.zeros_like(K, dtype=np.uint8)
    for i in range(group_num):
        K8[i::group_num] = postprocessing_simd_golden_model(
            K[i::group_num],
            input_zp_i,
            output_zp_i,
            shift_i[i],
            max_int_i,
            min_int_i,
            double_round_i,
            multiplier_i[i],
        )
    data_str += [format_vector_definition("int8_t", "K8_golden", K8)]

    # V = XWv
    V = block_gemm_golden_model(
        kwargs["M_1"],
        kwargs["K_1"],
        kwargs["N_1"],
        meshRow,
        tileSize,
        meshCol,
        X,
        Wv,
        subtraction_a,
        subtraction_b,
        np.zeros(shape=(kwargs["M_1"], kwargs["N_1"], meshRow, meshCol), dtype=np.int32).reshape(-1),
    )

    V8 = np.zeros_like(V, dtype=np.uint8)
    for i in range(group_num):
        V8[i::group_num] = postprocessing_simd_golden_model(
            V[i::group_num],
            input_zp_i,
            output_zp_i,
            shift_i[i],
            max_int_i,
            min_int_i,
            double_round_i,
            multiplier_i[i],
        )
    data_str += [format_vector_definition("int8_t", "V8_golden", V8)]

    # S=QKt
    K8 = K8.reshape(kwargs["M_1"], kwargs["N_1"], tileSize, meshCol)
    K8t = K8.reshape(-1)
    # K8t = K8.transpose(0, 1, 3, 2).reshape(-1)
    S = block_gemm_golden_model(
        kwargs["M_2"],
        kwargs["K_2"],
        kwargs["N_2"],
        meshRow,
        tileSize,
        meshCol,
        Q8,
        K8t,
        subtraction_a,
        subtraction_b,
        np.zeros(shape=(kwargs["M_2"], kwargs["N_2"], meshRow, meshCol), dtype=np.int32).reshape(-1),
    )

    S8 = np.zeros_like(S, dtype=np.uint8)
    for i in range(group_num):
        S8[i::group_num] = postprocessing_simd_golden_model(
            S[i::group_num],
            input_zp_i,
            output_zp_i,
            shift_i[i],
            max_int_i,
            min_int_i,
            double_round_i,
            multiplier_i[i],
        )
    data_str += [format_vector_definition("int8_t", "S8_golden", S8)]

    # Z = SV
    Z = block_gemm_golden_model(
        kwargs["M_2"],
        kwargs["K_2"],
        kwargs["N_2"],
        meshRow,
        tileSize,
        meshCol,
        S8,
        V8,
        subtraction_a,
        subtraction_b,
        np.zeros(shape=(kwargs["M_2"], kwargs["N_2"], meshRow, meshCol), dtype=np.int32).reshape(-1),
    )

    Z8 = np.zeros_like(Z, dtype=np.uint8)
    for i in range(group_num):
        Z8[i::group_num] = postprocessing_simd_golden_model(
            Z[i::group_num],
            input_zp_i,
            output_zp_i,
            shift_i[i],
            max_int_i,
            min_int_i,
            double_round_i,
            multiplier_i[i],
        )

    data_str += [format_vector_definition("int8_t", "Z8_golden", Z8)]

    data_str = "\n\n".join(data_str)

    return data_str


def main():
    # Parsing cmd args
    parser = argparse.ArgumentParser(description="Generate data for kernels")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    # Emit header file
    print(emit_header_file(**param))


if __name__ == "__main__":

    main()
