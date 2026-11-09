# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""What every multi-chip block's data and golden share (in workloads/: ffn/common/ffn_data.py, fa/common/*_data.py,
moe/common/*_data.py, llama3/common/llama3_data.py, the dsv2 layers): the golden's modules, the checkpoint's BF16
tensors and the RMSNorm epsilon. The GEMV helpers are ../gemm/gemv_utils.py's.

    import_golden()             the golden's modules: fp, hwmodel, layout, pack (../common) and reference (here)
    read_bf16(path, shape)      a BF16 tensor of the sweep caches (fetch_weights.py stores them raw) as float32
    RMS_EPS                     the RMSNorm epsilon of every model here

THE GOLDEN is the device's arithmetic, bit for bit, and DeepSeek-V2-Lite's float reference:

    ../common/fp.py       the bit-level numerics every stage shares
    ../simd/simd.py       the SIMD's FP32 arithmetic: its fused multiply-add, tables and reduce order
    ../common/hwmodel.py  the cluster's stages as it computes them: every stage bit-exact
    ../common/layout.py   the array's operand layouts
    ../common/pack.py     the weights as the device reads them: INT8 or INT4 per column, B-layout
    dsv2_model.py         DeepSeek-V2-Lite's constants, read from its config.json, and its YaRN RoPE
    reference.py          DeepSeek-V2-Lite's layer in float32, following modeling_deepseek.py

They are snax_cluster's target/snitch_cluster/sw/apps/dsv2/util (snax_cluster 0f7be164), imported by their flat
names (model.py renamed dsv2_model.py: bingo_hw_manager's sim check imports a `model` package): hwmodel, pack and
reference import their siblings by name instead of relatively, and simd.py's SIMD geometry
starts on snax_split_cluster's and follows the cluster cfg a driver passes to use_hwcfg (../common/driver.py)
instead of reading snax_cluster's cfg. Importing this module puts ../common and ../simd on sys.path.
"""

import os
import sys

import numpy as np

RMS_EPS = 1e-6
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # util/sim
for _d in ("common", "simd"):
    if os.path.join(SIM, _d) not in sys.path:
        sys.path.insert(0, os.path.join(SIM, _d))


def import_golden():
    """The golden's modules: fp, hwmodel, layout, pack (../common) and reference (here). Each, and the simd and model
    hwmodel imports, must be the one in its util/sim directory, not another module of the same name."""
    import fp
    import hwmodel
    import layout
    import dsv2_model
    import pack
    import reference
    import simd
    for mod, where in ((fp, "common"), (hwmodel, "common"), (layout, "common"), (pack, "common"),
                       (simd, "simd"), (dsv2_model, "llm"), (reference, "llm")):
        got = os.path.dirname(os.path.abspath(mod.__file__))
        if got != os.path.join(SIM, where):
            raise ImportError(f"imported `{mod.__name__}` from {got}, not from {os.path.join(SIM, where)}")
    return fp, hwmodel, layout, pack, reference


def read_bf16(path, shape):
    """A raw little-endian BF16 tensor (a fetch_weights.py cache file) as float32 of `shape`."""
    u = np.fromfile(path, dtype="<u2")
    if u.size != int(np.prod(shape)):
        raise ValueError(f"{path}: {u.size} values, expected {shape}")
    return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
