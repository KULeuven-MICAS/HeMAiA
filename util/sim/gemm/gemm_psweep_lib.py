# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Shared builder for the per-precision GEMM cycle-sweep workloads
# (gemm_psweep_<prec>_1cluster). Lives in util/sim/ alongside gemm_sim_utils.py;
# each workload's main_bingo.py adds util/sim to sys.path and calls run("<prec>").
#
# Each workload runs ONE precision over a grid of (M, K, N, array_shape) sizes in
# a single DFG, using the clean precision-named wrapper kernels
# (offload_hw_kernels/gemm.h). Per config the DFG emits:
#     Load_A -> Load_B -> Gemm<prec> -> Check_D
# and the device GEMM kernel brackets its compute with BINGO_TRACE_GEMM_FULL_RUN_*
# markers, so target/sim/automation/sweep/gemm/gather_gemm_luts.py recovers one
# cycle count per config (paired with configs.json) — no host mcycle bracketing.
#
# Precision is split across workloads (one per precision) because it changes the
# generated data layout (int4 packing / quant int8 / fp16 output) and so cannot
# be looped inside one binary. Running the 5 workloads as a task list gives the
# parallel multi-precision sweep.
#
# Data + golden come from the validated gemm_sim_utils.generate_gemm_test_data
# (handles int4 packing, int8 requantization, int32->fp16) so this stays DRY.

import os
import sys
import json
import random
import argparse
import pathlib

import hjson

_THIS = os.path.dirname(os.path.abspath(__file__))         # .../util/sim/gemm
ROOT = os.path.normpath(os.path.join(_THIS, "../../.."))   # repo root
sys.path.append(os.path.join(ROOT, "target/sw/host/runtime/libbingo/mini_compiler"))
sys.path.append(os.path.join(ROOT, "util/sim/common"))     # data_utils
sys.path.append(_THIS)  # util/sim/gemm — gemm_sim_utils (sibling)

from bingo_dfg import BingoDFG                                       # noqa: E402
from bingo_node import BingoNode                                     # noqa: E402
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol           # noqa: E402
from bingo_platform import guard_cluster_count, parse_platform_cfg   # noqa: E402
from bingo_kernel_args import (                                      # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    HostBingoKernelCheckResultArgs,
    SnaxBingoKernelGemmI8I8I32Args,
    SnaxBingoKernelGemmI8I4I32Args,
    SnaxBingoKernelGemmI4I4I32Args,
    SnaxBingoKernelGemmI8I4F16Args,
    SnaxBingoKernelGemmI8I8F16Args,
    SnaxBingoKernelGemmI8I8I8Args,
)
from gemm_sim_utils import generate_gemm_test_data                   # noqa: E402
from data_utils import format_vector_definition                     # noqa: E402

# Size grid (M, K, N, array_shape) in mesh-tile units — shared by all precisions
# so the cycle LUTs are directly comparable. Same grid as gemm_sweep_1cluster.
CONFIGS = [
    # shape 0 (meshRow=32, tileSize=2, meshCol=32)
    (1, 1, 1, 0), (1, 2, 1, 0), (1, 4, 1, 0), (1, 8, 1, 0),
    (2, 2, 2, 0), (2, 4, 2, 0), (4, 4, 4, 0),
    # shape 1 (meshRow=1, tileSize=16, meshCol=32)
    (1, 1, 1, 1), (1, 2, 1, 1), (1, 4, 1, 1), (1, 8, 1, 1),
    (2, 2, 2, 1), (2, 4, 2, 1), (4, 4, 4, 1),
    # shape 2 (meshRow=16, tileSize=8, meshCol=16)
    (1, 1, 1, 2), (1, 2, 1, 2), (1, 4, 1, 2), (1, 8, 1, 2),
    (2, 2, 2, 2), (2, 4, 2, 2), (4, 4, 4, 2),
]

_CTYPE_BYTES = {"int8_t": 1, "uint16_t": 2, "int32_t": 4}

# BINGO_SERIAL_C_D_WIDTH from gemm_shapes.h — the VersaCore C/D serializer beat
# width (bits). The D-extension (requant->int8 / int32->fp16) output stage has two
# kernel paths (offload_hw_kernels/gemm.h): when one narrowed output tile spans
# >= one beat (meshRow*meshCol*d_bits >= 1024 — array_shape 0 and 2 here) the
# streamer splits it into whole beats and works; when one tile is SMALLER than a
# beat (the narrow meshRow=1 array_shape 1) the "pack multiple tiles into one
# beat" path stalls the accelerator (observed: shape-1 hangs even when M*N fills
# whole beats, e.g. (2,2,2,1)). So a D-extension config is only emittable when one
# output tile already fills >= one serializer beat.
_SERIAL_C_D_WIDTH = 1024


def _d_extension_emittable(spec, gd):
    """Whether the D-extension output stage can emit this config without stalling.
    int32-output precisions have no D extension -> always emittable. For the
    requant-int8 / int32->fp16 precisions, only configs whose single narrowed
    output tile fills >= one serializer beat work (the small-tile packing path
    hangs); in this sweep that excludes every array_shape-1 config."""
    flags = spec["flags"]
    if not (flags.get("quantization_enable", 0) or flags.get("int32tofp16_enable", 0)):
        return True
    d_bits = _CTYPE_BYTES[spec["d_ctype"]] * 8
    one_output_tile_bits = gd["meshRow"] * gd["meshCol"] * d_bits
    return one_output_tile_bits >= _SERIAL_C_D_WIDTH


def _plain_args(cls):
    def make(A, B, C, D, M, K, N, shape, gd):
        return cls(input_A_addr=A, input_B_addr=B, input_C_addr=C, output_D_addr=D,
                   M=M, K=K, N=N, array_shape_idx=shape)
    return make


def _quant_args(A, B, C, D, M, K, N, shape, gd):
    cfg = gd["config"]
    return SnaxBingoKernelGemmI8I8I8Args(
        input_A_addr=A, input_B_addr=B, input_C_addr=C, output_D_addr=D,
        M=M, K=K, N=N, array_shape_idx=shape,
        shift_i=cfg["shift_i"], multiplier_i=cfg["multiplier_i"],
        input_zp_i=cfg["input_zp_i"], output_zp_i=cfg["output_zp_i"])


# Precision registry: name -> kernel symbol, datagen flags, args factory, D ctype.
# Names + flags mirror the GEMM_PRECISIONS test contract and the gemm.h wrappers.
PRECISIONS = {
    "i8i8_i32": dict(
        kernel="__snax_bingo_kernel_gemm_i8i8_i32",
        flags=dict(int4_a_enable=0, int4_b_enable=0, quantization_enable=0, int32tofp16_enable=0),
        make_args=_plain_args(SnaxBingoKernelGemmI8I8I32Args), d_ctype="int32_t"),
    "i8i4_i32": dict(
        kernel="__snax_bingo_kernel_gemm_i8i4_i32",
        flags=dict(int4_a_enable=0, int4_b_enable=1, quantization_enable=0, int32tofp16_enable=0),
        make_args=_plain_args(SnaxBingoKernelGemmI8I4I32Args), d_ctype="int32_t"),
    "i4i4_i32": dict(
        kernel="__snax_bingo_kernel_gemm_i4i4_i32",
        flags=dict(int4_a_enable=1, int4_b_enable=1, quantization_enable=0, int32tofp16_enable=0),
        make_args=_plain_args(SnaxBingoKernelGemmI4I4I32Args), d_ctype="int32_t"),
    "i8i8_i8": dict(
        kernel="__snax_bingo_kernel_gemm_i8i8_i8",
        flags=dict(int4_a_enable=0, int4_b_enable=0, quantization_enable=1, int32tofp16_enable=0),
        make_args=_quant_args, d_ctype="int8_t"),
    "i8i4_f16": dict(
        kernel="__snax_bingo_kernel_gemm_i8i4_f16",
        flags=dict(int4_a_enable=0, int4_b_enable=1, quantization_enable=0, int32tofp16_enable=1),
        make_args=_plain_args(SnaxBingoKernelGemmI8I4F16Args), d_ctype="uint16_t"),
    "i8i8_f16": dict(
        kernel="__snax_bingo_kernel_gemm_i8i8_f16",
        flags=dict(int4_a_enable=0, int4_b_enable=0, quantization_enable=0, int32tofp16_enable=1),
        make_args=_plain_args(SnaxBingoKernelGemmI8I8F16Args), d_ctype="uint16_t"),
}


def _pad64(n):
    return (n + 63) & ~63


def _gen_configs(configs, hwcfg, flags):
    """One generate_gemm_test_data() per config (deterministic). Returns the list
    of gemm_data dicts (A/B/D arrays + the config incl. any quant params)."""
    random.seed(320)  # make requant params reproducible across builds
    out = []
    for (M, K, N, shape) in configs:
        kwargs = {
            **hwcfg,
            "M": M, "K": K, "N": N, "array_shape": shape, "data_type": 0,
            "transposed_A": 0, "transposed_B": 0,
            "addNonZeroC": 0, "addZeroC": 1, "accumPrevC": 0,
            **flags,
        }
        out.append(generate_gemm_test_data(**kwargs))
    return out


def _emit_header(gen, d_ctype):
    lines = ["#include <stdint.h>"]
    for i, gd in enumerate(gen):
        lines.append("")
        lines.append(f"// cfg{i}: M={gd['M']} K={gd['K']} N={gd['N']} "
                     f"mesh={gd['meshRow']}x{gd['tileSize']}x{gd['meshCol']}")
        lines.append(format_vector_definition("int8_t", f"A_cfg{i}", gd["A"]))
        lines.append(format_vector_definition("int8_t", f"B_cfg{i}", gd["B"]))
        lines.append(format_vector_definition(d_ctype, f"D_cfg{i}", gd["D"]))
    return "\n".join(lines) + "\n"


def _create_dfg(configs, gen, spec, platform):
    dbytes = _CTYPE_BYTES[spec["d_ctype"]]
    a_sz = [_pad64(len(gd["A"])) for gd in gen]   # xDMA-aligned
    b_sz = [len(gd["B"]) for gd in gen]
    d_sz = [len(gd["D"]) * dbytes for gd in gen]
    max_A, max_B, max_D = max(a_sz), max(b_sz), max(d_sz)

    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"],
        num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"],
        is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )
    gemm_core, dma_core, host_core = 0, 1, 2
    l1_A = BingoMemAlloc("l1_A", size=max_A, mem_level="L1", chip_id=0, cluster_id=0)
    l1_B = BingoMemAlloc("l1_B", size=max_B, mem_level="L1", chip_id=0, cluster_id=0)
    l1_D = BingoMemAlloc("l1_D", size=max_D, mem_level="L1", chip_id=0, cluster_id=0)

    prev_check = None
    for i, (M, K, N, shape) in enumerate(configs):
        A_l3 = BingoMemSymbol(f"A_cfg{i}")
        B_l3 = BingoMemSymbol(f"B_cfg{i}")
        D_l3 = BingoMemSymbol(f"D_cfg{i}")
        load_A = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
            node_name=f"Load_A_cfg{i}", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(A_l3, l1_A, a_sz[i]))
        load_B = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=dma_core,
            node_name=f"Load_B_cfg{i}", kernel_name="__snax_bingo_kernel_idma_1d_copy",
            kernel_args=SnaxBingoKernelIdma1dCopyArgs(B_l3, l1_B, b_sz[i]))
        gemm = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=gemm_core,
            node_name=f"Gemm_cfg{i}", kernel_name=spec["kernel"],
            kernel_args=spec["make_args"](l1_A, l1_B, 0, l1_D, M, K, N, shape, gen[i]))
        check = BingoNode(
            assigned_chiplet_id=0, assigned_cluster_id=0, assigned_core_id=host_core,
            node_name=f"Check_cfg{i}", kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                name=f"D{i}", golden_data_addr=D_l3, output_data_addr=l1_D,
                data_size=min(d_sz[i], 256)))
        for n in (load_A, load_B, gemm, check):
            dfg.bingo_add_node(n)
        dfg.bingo_add_edge(load_A, load_B)
        dfg.bingo_add_edge(load_B, gemm)
        dfg.bingo_add_edge(gemm, check)
        if prev_check is not None:
            dfg.bingo_add_edge(prev_check, load_A)
        prev_check = check
    return dfg


def _check_fits_in_l1(gen, spec, l1_size_kb):
    dbytes = _CTYPE_BYTES[spec["d_ctype"]]
    budget = l1_size_kb * 1024
    frag = lambda x: (x + 128 + 255) & ~255
    for i, gd in enumerate(gen):
        total = frag(_pad64(len(gd["A"]))) + frag(len(gd["B"])) + frag(len(gd["D"]) * dbytes)
        assert total <= budget, (
            f"cfg{i} (M={gd['M']},K={gd['K']},N={gd['N']}) needs {total} B > L1 {budget}")


def _get_args():
    p = argparse.ArgumentParser(description="per-precision GEMM cycle sweep")
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    p.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    p.add_argument("--hwcfg", type=pathlib.Path, required=True)
    p.add_argument("--platformcfg", type=pathlib.Path, required=True)
    p.add_argument("--data_h", type=pathlib.Path, default=None)
    p.add_argument("--configs_out", type=pathlib.Path, default=None)
    p.add_argument("--l1_size_kb", type=int, default=500)
    return p.parse_args()


def run(prec_name):
    """Entry point used by each gemm_psweep_<prec>_1cluster/main_bingo.py."""
    assert prec_name in PRECISIONS, f"unknown precision '{prec_name}'"
    spec = PRECISIONS[prec_name]
    args = _get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.cfg) as f:
        param = hjson.loads(f.read())
    with open(args.hwcfg) as f:
        hwcfg = hjson.loads(f.read())

    # Drop configs whose array_shape isn't exposed by the active hwcfg.
    num_shapes = len(
        hwcfg["snax_versacore_core_template"]["snax_acc_cfg"][0]
        ["snax_versacore_spatial_unrolling"][0])
    configs = [c for c in CONFIGS if c[3] < num_shapes]
    if len(configs) != len(CONFIGS):
        print(f"[gemm_psweep:{prec_name}] hwcfg exposes {num_shapes} shapes; "
              f"kept {len(configs)}/{len(CONFIGS)} configs")

    gen = _gen_configs(configs, hwcfg, spec["flags"])

    # Drop configs the D-extension output serializer can't emit (would hang the
    # accelerator). Only affects the requant-int8 / int32->fp16 precisions at the
    # shapes/sizes whose narrowed output doesn't fill whole serializer beats
    # (here: shape 1 with M*N too small). Padding M/N to fill a beat would change
    # the measured GEMM, so these points are simply absent from the cycle LUT;
    # int32-output precisions keep every config. Filter configs+gen together so
    # configs.json (used by gather_gemm_luts.py) stays aligned with the DFG.
    keep = [_d_extension_emittable(spec, gd) for gd in gen]
    if not all(keep):
        dropped = [c for c, k in zip(configs, keep) if not k]
        print(f"[gemm_psweep:{prec_name}] dropped {len(dropped)} D-extension config(s) "
              f"whose narrowed output can't fill whole {_SERIAL_C_D_WIDTH}-bit serializer "
              f"beats (would hang): {dropped}")
        configs = [c for c, k in zip(configs, keep) if k]
        gen = [gd for gd, k in zip(gen, keep) if k]

    _check_fits_in_l1(gen, spec, args.l1_size_kb)

    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(_emit_header(gen, spec["d_ctype"]))
        print(f"Written data header: {args.data_h}")
    if args.configs_out is not None:
        with open(args.configs_out, "w") as f:
            json.dump({"precision": prec_name, "configs": configs}, f, indent=2)
        print(f"Written configs list: {args.configs_out}")

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(param, platform, args.output_dir, args.output_offload_file_name):
        return
    dfg = _create_dfg(configs, gen, spec, platform)
    dfg.bingo_compile_dfg(
        f"Single-Chip GEMM precision sweep ({prec_name})",
        args.output_dir, args.output_offload_file_name,
        extra_include_header_list=["gemm_data.h"])
    print(f"Generated DFG: {prec_name} x {len(configs)} configs")
