#!/usr/bin/env python3

# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import sys
import numpy as np
import os

# -----------------------
# Add data utility path
# -----------------------
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../../util/sim/")
)
from data_utils import format_scalar_definition, format_vector_definition  # noqa: E402


def conv2d_nhwc_reference(input_nhwc, weights_oickk, stride, pad):
    """Pure-numpy reference convolution (NHWC input, OICKK weights) -> NHWC output."""
    N, IH, IW, IC = input_nhwc.shape
    OC, IC_, KH, KW = weights_oickk.shape
    assert IC == IC_

    OH = (IH + 2 * pad - KH) // stride + 1
    OW = (IW + 2 * pad - KW) // stride + 1

    input_pad = np.pad(input_nhwc, ((0,0),(pad,pad),(pad,pad),(0,0)),
                       mode='constant', constant_values=0)

    output = np.zeros((N, OH, OW, OC), dtype=np.int32)
    for n in range(N):
        for oc in range(OC):
            for oh in range(OH):
                for ow in range(OW):
                    acc = np.int32(0)
                    for ic in range(IC):
                        for kh in range(KH):
                            for kw in range(KW):
                                acc += (np.int32(input_pad[n, oh*stride+kh, ow*stride+kw, ic])
                                      * np.int32(weights_oickk[oc, ic, kh, kw]))
                    output[n, oh, ow, oc] = acc
    return output


def maxpool2d_nhwc_reference(input_nhwc, pool_size, stride, pad):
    """Pure-numpy reference MaxPool (NHWC) reusing conv stride and pad."""
    N, IH, IW, C = input_nhwc.shape

    OH = (IH + 2 * pad - pool_size) // stride + 1
    OW = (IW + 2 * pad - pool_size) // stride + 1

    # Pad with INT32_MIN so out-of-bounds positions never win the max
    input_pad = np.pad(input_nhwc,
                       ((0,0),(pad,pad),(pad,pad),(0,0)),
                       mode='constant', constant_values=np.iinfo(np.int32).min)

    output = np.full((N, OH, OW, C), np.iinfo(np.int32).min, dtype=np.int32)
    for n in range(N):
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    for ph in range(pool_size):
                        for pw in range(pool_size):
                            val = input_pad[n, oh*stride+ph, ow*stride+pw, c]
                            if val > output[n, oh, ow, c]:
                                output[n, oh, ow, c] = val
    return output


def main():

    # -----------------------
    # Conv layer parameters
    # -----------------------
    BATCH     = 1
    # IN_H      = 8
    # IN_W      = 8
    # IN_C      = 64
    # OUT_C     = 64

    IN_H      = 4
    IN_W      = 4
    IN_C      = 16
    OUT_C     = 16

    KH        = 3
    KW        = 3
    STRIDE    = 1   # shared by conv and maxpool
    PAD       = 1   # shared by conv and maxpool
    POOL_SIZE = 2   # maxpool window (2x2)

    # -----------------------
    # Generate random int8 data
    # -----------------------
    np.random.seed(42)
    input_nhwc    = np.random.randint(-128, 127, size=(BATCH, IN_H, IN_W, IN_C), dtype=np.int8)
    weights_oickk = np.random.randint(-128, 127, size=(OUT_C, IN_C, KH, KW),    dtype=np.int8)

    # -----------------------
    # Compute golden outputs
    # -----------------------
    conv_out_nhwc = conv2d_nhwc_reference(input_nhwc, weights_oickk, STRIDE, PAD)
    pool_out_nhwc = maxpool2d_nhwc_reference(conv_out_nhwc, POOL_SIZE, STRIDE, PAD)

    # Flatten in the same order the C code indexes them
    conv_input   = input_nhwc.flatten().tolist()
    conv_weights = weights_oickk.flatten().tolist()
    conv_output  = conv_out_nhwc.flatten().tolist()
    pool_output  = pool_out_nhwc.flatten().tolist()

    # -----------------------
    # Format definitions
    # -----------------------
    BATCH_str     = format_scalar_definition("int32_t", "BATCH",     BATCH)
    IN_H_str      = format_scalar_definition("int32_t", "IN_H",      IN_H)
    IN_W_str      = format_scalar_definition("int32_t", "IN_W",      IN_W)
    IN_C_str      = format_scalar_definition("int32_t", "IN_C",      IN_C)
    OUT_C_str     = format_scalar_definition("int32_t", "OUT_C",     OUT_C)
    KH_str        = format_scalar_definition("int32_t", "KH",        KH)
    KW_str        = format_scalar_definition("int32_t", "KW",        KW)
    STRIDE_str    = format_scalar_definition("int32_t", "STRIDE",    STRIDE)
    PAD_str       = format_scalar_definition("int32_t", "PAD",       PAD)
    POOL_SIZE_str = format_scalar_definition("int32_t", "POOL_SIZE", POOL_SIZE)

    conv_input_str   = format_vector_definition("int8_t",  "conv_input",   conv_input)
    conv_weights_str = format_vector_definition("int8_t",  "conv_weights", conv_weights)
    conv_output_str  = format_vector_definition("int32_t", "conv_output",  conv_output)
    pool_output_str  = format_vector_definition("int32_t", "pool_output",  pool_output)

    # -----------------------
    # Assemble and print
    # -----------------------
    f_str = "\n\n".join([
        BATCH_str,
        IN_H_str,
        IN_W_str,
        IN_C_str,
        OUT_C_str,
        KH_str,
        KW_str,
        STRIDE_str,
        PAD_str,
        POOL_SIZE_str,
        conv_input_str,
        conv_weights_str,
        conv_output_str,
        pool_output_str,
    ])
    f_str += "\n"

    print(f_str)


if __name__ == "__main__":
    sys.exit(main())