# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""GEMV helpers of the multi-chiplet workloads' goldens (ffn, moe, the llama3 layers; ../llm/model_data.py holds
the data helpers):

    exact_gemv(fp, xq, Wq, k)   hwmodel.gemv bit for bit, on float64 BLAS (`fp`: ../common/fp.py)
    a_row_bytes(v)              INT8 values in the a_row layout a GEMV reads its x in
"""

import numpy as np


def exact_gemv(fp, xq, Wq, k):
    """hwmodel.gemv (../common/hwmodel.py) on float64 BLAS (numpy has no BLAS for integers): every product and partial sum of INT8 x
    INT4/INT8 over the model's depths is an integer far below 2^53, so the result is the INT32 accumulation bit for bit;
    a bound at or above that falls back to the integer product."""
    x, w = np.asarray(xq), np.asarray(Wq)
    bound = float(np.max(np.abs(x), initial=0)) * float(np.max(np.abs(w), initial=0)) * x.shape[-1]
    if bound >= 2.0 ** 52:
        acc = x.astype(np.int64) @ w.astype(np.int64)
    else:
        acc = np.rint(x.astype(np.float64) @ w.astype(np.float64)).astype(np.int64)
    return fp.d_port(acc, k), acc


def a_row_bytes(v):
    """int8 values as a_row: value c at (c // 4) * 8 + c % 4, the rest 0 (the GEMV's x)."""
    v = np.asarray(v, dtype=np.int8).reshape(-1)
    o = np.zeros(2 * v.size, dtype=np.int8)
    o.reshape(-1, 8)[:, :4] = v.reshape(-1, 4)
    return o
