#!/usr/bin/env python3
# Golden data generator for single-head attention on HeMAiA single-chip.
# Each GEMM produces an atomic (non-tiled) golden D. At runtime, the
# compiled main_bingo.py executes each GEMM as K-split tiles on multiple
# clusters + INT32 inter-cluster accumulation; the sum equals this
# atomic golden byte-for-byte.

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import struct

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import block_gemm_golden_model  # noqa E402
from layout_convert import (  # noqa E402
    row_major_to_a, row_major_to_b, row_major_to_d,
    d_to_row_major,
)

np.random.seed(42)


# ────────────────────────────────────────────────────────────────────────────
# HW-exact softmax (mirrors __host_bingo_kernel_fp32_softmax in host_kernel_lib.h)
# Uses the same Cephes polynomial exp, FMA via fp64 intermediate trick,
# chunked ordered reduction (VLMAX=4 on Ara NrLanes=2 VLEN=128), and
# multiplication by 1/sum (not direct division).
# This keeps goldens bit-identical (or within ≤1 ULP) to the HW output,
# which is required for check_type=1 (fp32 tolerance) to pass.
# ────────────────────────────────────────────────────────────────────────────

_VLMAX_F32 = 4  # Ara: VLEN=128, NrLanes=2 → VLMAX = 128/32 = 4 fp32 elems


def _f32(x):
    """Coerce to Python float that round-trips through fp32."""
    return float(np.float32(x))


def _fma(a, b, c):
    """Bit-exact FMA: a*b + c with single rounding to fp32.
    Trick: fp32 inputs fit exactly in fp64. fp32*fp32 product needs ≤48 mantissa
    bits which fits in fp64's 53-bit mantissa exactly. Final rounding to fp32
    gives the same result as true FMA."""
    return _f32(_f32(a) * _f32(b) + _f32(c))


def _bits_to_f32(bits):
    return struct.unpack('<f', struct.pack('<I', np.uint32(bits) & 0xFFFFFFFF))[0]


_EXP_HI = _f32(88.3762626647949)
_EXP_LO = _f32(-88.3762626647949)
_LOG2EF = _f32(1.44269504088896341)
_C1 = _f32(0.693359375)
_C2 = _f32(-2.12194440e-4)
_HALF = _f32(0.5)
_ONE = _f32(1.0)
_POLY = [
    _f32(1.9875691500e-4),
    _f32(1.3981999507e-3),
    _f32(8.3334519073e-3),
    _f32(4.1665795894e-2),
    _f32(1.6666665459e-1),
    _f32(5.0000001201e-1),
]


def bingo_exp_f32(x_val):
    """Mirror of HW __bingo_exp_f32 — Cephes polynomial approximation."""
    x = _f32(min(max(x_val, _EXP_LO), _EXP_HI))
    fx = _fma(x, _LOG2EF, _HALF)
    tmp = int(np.float32(fx).astype(np.int32))
    ftmp = _f32(tmp)
    if fx < ftmp:
        ftmp = _f32(ftmp - _ONE)
    tmp = int(np.float32(ftmp).astype(np.int32))

    x = _f32(x - _f32(ftmp * _C1))
    x = _f32(x - _f32(ftmp * _C2))
    z = _f32(x * x)

    y = _POLY[0]
    y = _fma(y, x, _POLY[1])
    y = _fma(y, x, _POLY[2])
    y = _fma(y, x, _POLY[3])
    y = _fma(y, x, _POLY[4])
    y = _fma(y, x, _POLY[5])
    y = _fma(y, z, _f32(x + _ONE))

    shift_bits = (tmp + 127) << 23
    shift_f = _bits_to_f32(shift_bits)
    return _f32(y * shift_f)


def bingo_softmax_row(x_row):
    """Mirror HW __host_bingo_kernel_fp32_softmax for a single row.
    Chunked ordered reduction with VLMAX=4.
    """
    row_length = len(x_row)
    # Pass 1: max
    max_val = _f32(x_row[0])
    for v in x_row[1:]:
        vv = _f32(v)
        if vv > max_val:
            max_val = vv
    # Pass 2: exp(x - max) + chunked ordered sum
    expv = np.empty(row_length, dtype=np.float32)
    for i in range(row_length):
        expv[i] = bingo_exp_f32(_f32(_f32(x_row[i]) - max_val))
    sum_val = _f32(0.0)
    for cs in range(0, row_length, _VLMAX_F32):
        ce = min(cs + _VLMAX_F32, row_length)
        rsum = _f32(0.0)
        for i in range(cs, ce):
            rsum = _f32(rsum + _f32(expv[i]))
        sum_val = _f32(sum_val + rsum)
    # Pass 3: mul by 1/sum (not direct div)
    inv_sum = _f32(_ONE / sum_val)
    out = np.empty(row_length, dtype=np.float32)
    for i in range(row_length):
        out[i] = _f32(_f32(expv[i]) * inv_sum)
    return out


def bingo_softmax_rowwise(x_2d):
    """Row-wise softmax on a 2D fp32 array, bit-exact to HW."""
    out = np.empty_like(x_2d, dtype=np.float32)
    for r in range(x_2d.shape[0]):
        out[r] = bingo_softmax_row(x_2d[r])
    return out


def quantize_symmetric_int8(tensor):
    abs_max = float(np.max(np.abs(tensor)))
    if abs_max < 1e-10:
        abs_max = 1e-10
    scale = abs_max / 127.0
    quantized = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
    return quantized, scale


def gemm_int8_roundtrip(A_fp32_log, B_fp32_log, M, K, N, meshRow, tileSize, meshCol):
    int8_A_log, scale_A = quantize_symmetric_int8(A_fp32_log)
    int8_B_log, scale_B = quantize_symmetric_int8(B_fp32_log)
    int8_A_ablk = row_major_to_a(int8_A_log, M, K, meshRow, tileSize)
    int8_B_bblk = row_major_to_b(int8_B_log, K, N, tileSize, meshCol)
    C_zero = np.zeros(M * N * meshRow * meshCol, dtype=np.int32)
    int32_D_dblk = block_gemm_golden_model(
        M, K, N, meshRow, tileSize, meshCol,
        int8_A_ablk, int8_B_bblk, 0, 0, C_zero,
    )
    # Mirror HW dequantize arithmetic: int32 -> fp32, fp32 * fp32.
    # Golden is in ROW-MAJOR (logical) layout — main_bingo.py emits an
    # xdma_d_to_row_major reshape node after each GEMM dequantize to convert
    # the HW's block-layout output to row-major before the fp32 check.
    combined_scale = np.float32(scale_A * scale_B)
    fp32_input = d_to_row_major(int32_D_dblk, M, N, meshRow, meshCol).astype(np.float32)
    fp32_D_log = (fp32_input * combined_scale).astype(np.float32)
    return {
        "int8_A_log": int8_A_log, "scale_A": scale_A,
        "int8_B_log": int8_B_log, "scale_B": scale_B,
        "int8_A_ablk": int8_A_ablk, "int8_B_bblk": int8_B_bblk,
        "int32_D_dblk": int32_D_dblk, "fp32_D_log": fp32_D_log,
        "combined_scale": float(combined_scale),
    }


def emit_scale(name, value):
    # Array form (not scalar) so BingoMemSymbol address decay works correctly.
    return f"float {name}[1] __attribute__((aligned(8))) = {{ {value:.10e}f }};"


def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_attention_data(**kwargs)
    return "\n\n".join(emit_str)


def emit_attention_data(**kwargs):
    data_str = []
    array_shape = kwargs["array_shape"]
    batch = kwargs["batch"]
    seq_len = kwargs["seq_len"]
    d_model = kwargs["d_model"]
    d_head = kwargs["d_head"]

    data_type = 0  # int8
    snax_acc_cfg = kwargs["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, tileSize, meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    assert batch == 1
    assert seq_len % meshRow == 0
    assert d_model % tileSize == 0 and d_model % meshCol == 0
    assert d_head % tileSize == 0 and d_head % meshCol == 0

    M_proj = seq_len // meshRow
    K_proj = d_model // tileSize
    N_proj = d_head // meshCol
    M_qk = seq_len // meshRow
    K_qk = d_head // tileSize
    N_qk = seq_len // meshCol
    M_av = seq_len // meshRow
    K_av = seq_len // tileSize
    N_av = d_head // meshCol
    M_o = seq_len // meshRow
    K_o = d_head // tileSize
    N_o = d_model // meshCol

    for k, v in [
        ("array_shape", array_shape), ("meshRow", meshRow),
        ("tileSize", tileSize), ("meshCol", meshCol),
        ("seq_len", seq_len), ("d_model", d_model), ("d_head", d_head),
        ("M_proj", M_proj), ("K_proj", K_proj), ("N_proj", N_proj),
        ("M_qk", M_qk), ("K_qk", K_qk), ("N_qk", N_qk),
        ("M_av", M_av), ("K_av", K_av), ("N_av", N_av),
        ("M_o", M_o), ("K_o", K_o), ("N_o", N_o),
    ]:
        data_str.append(format_scalar_definition("uint32_t", k, v))

    # FP32 inputs
    X_fp32 = np.random.uniform(-1.0, 1.0, size=(seq_len, d_model)).astype(np.float32)
    Wq_fp32 = np.random.uniform(-0.5, 0.5, size=(d_model, d_head)).astype(np.float32)
    Wk_fp32 = np.random.uniform(-0.5, 0.5, size=(d_model, d_head)).astype(np.float32)
    Wv_fp32 = np.random.uniform(-0.5, 0.5, size=(d_model, d_head)).astype(np.float32)
    Wo_fp32 = np.random.uniform(-0.5, 0.5, size=(d_head, d_model)).astype(np.float32)

    data_str += [
        format_vector_definition("float", "fp32_X", X_fp32.reshape(-1)),
        format_vector_definition("float", "fp32_Wq", Wq_fp32.reshape(-1)),
        format_vector_definition("float", "fp32_Wk", Wk_fp32.reshape(-1)),
        format_vector_definition("float", "fp32_Wv", Wv_fp32.reshape(-1)),
        format_vector_definition("float", "fp32_Wo", Wo_fp32.reshape(-1)),
    ]

    # Quantize X for quantize-kernel check
    int8_X_rm, scale_X = quantize_symmetric_int8(X_fp32)
    data_str += [
        format_vector_definition("int8_t", "golden_int8_X_rm", int8_X_rm.reshape(-1)),
        emit_scale("golden_scale_X", scale_X),
    ]

    # proj_q/k/v: X @ W* -> Q/K/V
    gq = gemm_int8_roundtrip(X_fp32, Wq_fp32, M_proj, K_proj, N_proj, meshRow, tileSize, meshCol)
    gk = gemm_int8_roundtrip(X_fp32, Wk_fp32, M_proj, K_proj, N_proj, meshRow, tileSize, meshCol)
    gv = gemm_int8_roundtrip(X_fp32, Wv_fp32, M_proj, K_proj, N_proj, meshRow, tileSize, meshCol)
    data_str += [
        format_vector_definition("int8_t", "golden_X_A_layout", gq["int8_A_ablk"]),
        format_vector_definition("int8_t", "golden_Wq_B_layout", gq["int8_B_bblk"]),
        format_vector_definition("int8_t", "golden_Wk_B_layout", gk["int8_B_bblk"]),
        format_vector_definition("int8_t", "golden_Wv_B_layout", gv["int8_B_bblk"]),
        format_vector_definition("int32_t", "golden_Q_int32", gq["int32_D_dblk"]),
        format_vector_definition("int32_t", "golden_K_int32", gk["int32_D_dblk"]),
        format_vector_definition("int32_t", "golden_V_int32", gv["int32_D_dblk"]),
        # Emit fp32 goldens under both old (Q/K/V) and gemm-name (proj_q/k/v) labels.
        # The task graph's check_result tasks reference golden_{gemm}_fp32, so those
        # must exist. Legacy Q/K/V names are kept for any external consumers.
        # Row-major (logical) layout — HW path converts block→row-major via
        # xdma_d_to_row_major after dequantize, then the check compares here.
        format_vector_definition("float", "golden_Q_fp32", gq["fp32_D_log"].reshape(-1)),
        format_vector_definition("float", "golden_K_fp32", gk["fp32_D_log"].reshape(-1)),
        format_vector_definition("float", "golden_V_fp32", gv["fp32_D_log"].reshape(-1)),
        format_vector_definition("float", "golden_proj_q_fp32", gq["fp32_D_log"].reshape(-1)),
        format_vector_definition("float", "golden_proj_k_fp32", gk["fp32_D_log"].reshape(-1)),
        format_vector_definition("float", "golden_proj_v_fp32", gv["fp32_D_log"].reshape(-1)),
        emit_scale("combined_scale_Q", gq["combined_scale"]),
        emit_scale("combined_scale_K", gk["combined_scale"]),
        emit_scale("combined_scale_V", gv["combined_scale"]),
    ]

    # qk_matmul with K^T transpose + fold 1/sqrt(d_head)
    Q_fp32 = gq["fp32_D_log"]
    K_fp32 = gk["fp32_D_log"]
    V_fp32 = gv["fp32_D_log"]
    K_T_fp32 = K_fp32.T.copy()
    gqk = gemm_int8_roundtrip(Q_fp32, K_T_fp32, M_qk, K_qk, N_qk, meshRow, tileSize, meshCol)
    data_str += [
        format_vector_definition("int8_t", "golden_Q_A_layout", gqk["int8_A_ablk"]),
        format_vector_definition("int8_t", "golden_K_T_B_layout", gqk["int8_B_bblk"]),
        format_vector_definition("int32_t", "golden_QK_int32", gqk["int32_D_dblk"]),
    ]
    inv_sqrt_d_head = np.float32(1.7677669832e-01)
    combined_scale_QK_scaled = np.float32(gqk["combined_scale"]) * inv_sqrt_d_head
    # Row-major golden — HW produces row-major via xdma_d_to_row_major after dequant.
    qk_fp32_input = d_to_row_major(gqk["int32_D_dblk"], M_qk, N_qk, meshRow, meshCol).astype(np.float32)
    QK_scaled_fp32 = (qk_fp32_input * combined_scale_QK_scaled).astype(np.float32)
    data_str += [
        format_vector_definition("float", "golden_QK_scaled_fp32", QK_scaled_fp32.reshape(-1)),
        format_vector_definition("float", "golden_qk_matmul_fp32", QK_scaled_fp32.reshape(-1)),
        emit_scale("combined_scale_QK_scaled", combined_scale_QK_scaled),
    ]

    # Softmax row-wise — HW-exact (Cephes exp polynomial + chunked vfredosum
    # with VLMAX=4 + mul-by-reciprocal). Must be HW-exact for fp32 check to pass
    # without tolerance; with tolerance mode the small ULP diffs fall within
    # the threshold.
    attn_weights_fp32 = bingo_softmax_rowwise(QK_scaled_fp32.astype(np.float32))
    data_str.append(
        format_vector_definition("float", "golden_attn_weights_fp32", attn_weights_fp32.reshape(-1))
    )

    # av_matmul
    gav = gemm_int8_roundtrip(attn_weights_fp32, V_fp32, M_av, K_av, N_av, meshRow, tileSize, meshCol)
    data_str += [
        format_vector_definition("int8_t", "golden_attn_A_layout", gav["int8_A_ablk"]),
        format_vector_definition("int8_t", "golden_V_B_layout", gav["int8_B_bblk"]),
        format_vector_definition("int32_t", "golden_attn_out_int32", gav["int32_D_dblk"]),
        format_vector_definition("float", "golden_attn_out_fp32", gav["fp32_D_log"].reshape(-1)),
        format_vector_definition("float", "golden_av_matmul_fp32", gav["fp32_D_log"].reshape(-1)),
        emit_scale("combined_scale_attn_out", gav["combined_scale"]),
    ]

    # proj_o
    attn_out_fp32 = gav["fp32_D_log"]
    go = gemm_int8_roundtrip(attn_out_fp32, Wo_fp32, M_o, K_o, N_o, meshRow, tileSize, meshCol)
    data_str += [
        format_vector_definition("int8_t", "golden_attn_out_A_layout", go["int8_A_ablk"]),
        format_vector_definition("int8_t", "golden_Wo_B_layout", go["int8_B_bblk"]),
        format_vector_definition("int32_t", "golden_Y_int32", go["int32_D_dblk"]),
        format_vector_definition("float", "golden_Y_fp32", go["fp32_D_log"].reshape(-1)),
        format_vector_definition("float", "golden_proj_o_fp32", go["fp32_D_log"].reshape(-1)),
        emit_scale("combined_scale_Y", go["combined_scale"]),
    ]

    A_proj_bytes = M_proj * K_proj * meshRow * tileSize
    B_proj_bytes = N_proj * K_proj * meshCol * tileSize
    D_proj_bytes = M_proj * N_proj * meshRow * meshCol * 4
    A_qk_bytes = M_qk * K_qk * meshRow * tileSize
    B_qk_bytes = N_qk * K_qk * meshCol * tileSize
    D_qk_bytes = M_qk * N_qk * meshRow * meshCol * 4
    A_av_bytes = M_av * K_av * meshRow * tileSize
    B_av_bytes = N_av * K_av * meshCol * tileSize
    D_av_bytes = M_av * N_av * meshRow * meshCol * 4
    A_o_bytes = M_o * K_o * meshRow * tileSize
    B_o_bytes = N_o * K_o * meshCol * tileSize
    D_o_bytes = M_o * N_o * meshRow * meshCol * 4
    for k, v in [
        ("A_proj_bytes", A_proj_bytes), ("B_proj_bytes", B_proj_bytes), ("D_proj_bytes", D_proj_bytes),
        ("A_qk_bytes", A_qk_bytes), ("B_qk_bytes", B_qk_bytes), ("D_qk_bytes", D_qk_bytes),
        ("A_av_bytes", A_av_bytes), ("B_av_bytes", B_av_bytes), ("D_av_bytes", D_av_bytes),
        ("A_o_bytes", A_o_bytes), ("B_o_bytes", B_o_bytes), ("D_o_bytes", D_o_bytes),
        ("X_fp32_bytes", seq_len * d_model * 4),
        ("X_int8_bytes", seq_len * d_model),
        ("num_softmax_rows", seq_len),
        ("softmax_row_length", seq_len),
    ]:
        data_str.append(format_scalar_definition("uint32_t", k, v))

    # ── Tile-level goldens for K-split GEMMs ────────────────────
    # Each GEMM's full A-layout and B-layout arrays are K-sliced into
    # per-tile inputs for the DFG's iDMA loads; per-cluster partial Ds
    # and inter-cluster reduction sums are computed for check_result.
    #
    # The tile plans are INLINED here (no external tile_plans.json needed).
    # Must match the specs in main_bingo.py → build_attention_specs().
    # BASELINE: all GEMMs use k_split=4 with exactly ONE K-chunk per cluster.
    # No multi-chunk-per-cluster accumulation anywhere.
    _inline_tile_plans = [
        {"name": "proj_q",    "k_split": 4, "cluster_kchunks": {i: [i] for i in range(4)}},
        {"name": "proj_k",    "k_split": 4, "cluster_kchunks": {i: [i] for i in range(4)}},
        {"name": "proj_v",    "k_split": 4, "cluster_kchunks": {i: [i] for i in range(4)}},
        {"name": "qk_matmul", "k_split": 4, "cluster_kchunks": {i: [i] for i in range(4)}},
        {"name": "av_matmul", "k_split": 4, "cluster_kchunks": {i: [i] for i in range(4)}},
        {"name": "proj_o",    "k_split": 4, "cluster_kchunks": {i: [i] for i in range(4)}},
    ]
    if True:  # scope for readability
        # Map GEMM names to local (A_ablk, B_bblk) variable refs
        _gemm_arrays = {
            "proj_q":    (gq["int8_A_ablk"], gq["int8_B_bblk"], M_proj, K_proj, N_proj),
            "proj_k":    (gk["int8_A_ablk"], gk["int8_B_bblk"], M_proj, K_proj, N_proj),
            "proj_v":    (gv["int8_A_ablk"], gv["int8_B_bblk"], M_proj, K_proj, N_proj),
            "qk_matmul": (gqk["int8_A_ablk"], gqk["int8_B_bblk"], M_qk, K_qk, N_qk),
            "av_matmul":  (gav["int8_A_ablk"], gav["int8_B_bblk"], M_av, K_av, N_av),
            "proj_o":    (go["int8_A_ablk"], go["int8_B_bblk"], M_o, K_o, N_o),
        }

        for tc in _inline_tile_plans:
            gname = tc["name"]
            if gname not in _gemm_arrays:
                continue
            k_split = tc["k_split"]
            if k_split <= 1:
                continue
            A_arr, B_arr, M_T_g, K_T_full_g, N_T_g = _gemm_arrays[gname]
            K_T_tile_g = K_T_full_g // k_split
            cluster_kchunks = {int(k): v for k, v in tc["cluster_kchunks"].items()}

            # Per-tile A/B K-slices (A-layout: [M_T, K_T, meshRow, tileSize])
            A_4d = A_arr.reshape(M_T_g, K_T_full_g, meshRow, tileSize)
            B_4d = B_arr.reshape(N_T_g, K_T_full_g, meshCol, tileSize)
            for kc in range(k_split):
                ks = kc * K_T_tile_g; ke = ks + K_T_tile_g
                data_str.append(format_vector_definition("int8_t", f"golden_{gname}_A_k{kc}", A_4d[:, ks:ke, :, :].reshape(-1).copy()))
                data_str.append(format_vector_definition("int8_t", f"golden_{gname}_B_k{kc}", B_4d[:, ks:ke, :, :].reshape(-1).copy()))

            # Per-cluster partial D
            D_size = M_T_g * N_T_g * meshRow * meshCol
            C_zero = np.zeros(D_size, dtype=np.int32)
            partial_Ds = {}
            for cid in sorted(cluster_kchunks.keys()):
                kchunks = cluster_kchunks[cid]
                accum = C_zero.copy()
                for ki, kc in enumerate(kchunks):
                    ks = kc * K_T_tile_g; ke = ks + K_T_tile_g
                    A_sl = A_4d[:, ks:ke, :, :].reshape(-1).copy()
                    B_sl = B_4d[:, ks:ke, :, :].reshape(-1).copy()
                    accum = block_gemm_golden_model(
                        M_T_g, K_T_tile_g, N_T_g, meshRow, tileSize, meshCol,
                        A_sl, B_sl, 0, 0, accum if ki > 0 else C_zero.copy(),
                    )
                partial_Ds[cid] = accum
                data_str.append(format_vector_definition("int32_t", f"golden_{gname}_D_partial_c{cid}", accum))

            # Inter-cluster reduction sums
            sorted_cids = sorted(partial_Ds.keys())
            running = partial_Ds[sorted_cids[0]].copy()
            for i, cid in enumerate(sorted_cids[1:], 1):
                running = (running.astype(np.int64) + partial_Ds[cid].astype(np.int64)).astype(np.int32)
                label = "_".join(f"c{c}" for c in sorted_cids[:i+1])
                data_str.append(format_vector_definition("int32_t", f"golden_{gname}_sum_{label}", running))

            # Per-tile byte sizes
            A_tb = M_T_g * K_T_tile_g * meshRow * tileSize
            B_tb = N_T_g * K_T_tile_g * meshCol * tileSize
            D_tb = M_T_g * N_T_g * meshRow * meshCol * 4
            data_str.append(format_scalar_definition("uint32_t", f"{gname}_A_tile_bytes", A_tb))
            data_str.append(format_scalar_definition("uint32_t", f"{gname}_B_tile_bytes", B_tb))
            data_str.append(format_scalar_definition("uint32_t", f"{gname}_D_tile_bytes", D_tb))
            data_str.append(format_scalar_definition("uint32_t", f"{gname}_k_split", k_split))

    return data_str


def main():
    parser = argparse.ArgumentParser(description="Attention golden data generator")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with open(args.cfg) as f: param = hjson.loads(f.read())
    with open(args.hwcfg) as f: hw = hjson.loads(f.read())
    merged = {**param, **hw}
    content = emit_header_file(**merged)
    with open(args.output, "w") as f: f.write(content)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
