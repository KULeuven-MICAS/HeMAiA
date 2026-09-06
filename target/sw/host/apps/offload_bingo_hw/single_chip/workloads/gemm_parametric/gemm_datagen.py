#!/usr/bin/env python3
import numpy as np
import argparse
import pathlib
import hjson
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
import _usg_paths  # noqa: F401,E402
from data_utils import format_scalar_definition, format_vector_definition
from sim_golden_models import block_gemm_golden_model

np.random.seed(320)

def emit_header_file(**cfg):
    out = ["#pragma once", "#include <stdint.h>"]
    out += emit_gemm_data(**cfg)
    return "\n\n".join(out)

def emit_gemm_data(**cfg):
    data = []
    M1, K1, N1 = cfg["M1"], cfg["K1"], cfg["N1"]

    data += [
        format_scalar_definition("uint32_t", "M1", M1),
        format_scalar_definition("uint32_t", "K1", K1),
        format_scalar_definition("uint32_t", "N1", N1),
    ]

    array_shape = cfg["array_shape"]
    data += [format_scalar_definition("uint32_t", "array_shape", array_shape)]

    acc_cfg = cfg["snax_versacore_core_template"]["snax_acc_cfg"][0]
    data_type = 0
    meshRow, tileSize, meshCol = acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape]

    data += [
        format_scalar_definition("uint32_t", "meshRow", meshRow),
        format_scalar_definition("uint32_t", "tileSize", tileSize),
        format_scalar_definition("uint32_t", "meshCol", meshCol),
        format_scalar_definition("uint32_t", "transposed_A", cfg["transposed_A"]),
        format_scalar_definition("uint32_t", "transposed_B", cfg["transposed_B"]),
        format_scalar_definition("uint32_t", "accumPrevC", cfg["accumPrevC"]),
    ]

    A = np.random.randint(-128, 127, size=(M1, K1, meshRow, tileSize), dtype=np.int8)
    B = np.random.randint(-128, 127, size=(K1, N1, tileSize, meshCol), dtype=np.int8)
    
    # Compute the real (golden) result in Python so that the check is meaningful
    C = np.zeros((M1, N1, meshRow, meshCol), dtype=np.int32)
    D = block_gemm_golden_model(
        M1, K1, N1, meshRow, tileSize, meshCol,
        A.reshape(-1), B.reshape(-1), 0, 0, C.reshape(-1)
    )

    data += [
        format_vector_definition("int8_t",  "A", A.reshape(-1)),
        format_vector_definition("int8_t",  "B", B.reshape(-1)),
        format_vector_definition("int32_t", "D", D.reshape(-1)),
    ]

    return data

def main():
    parser = argparse.ArgumentParser(description="Generating data")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with args.cfg.open() as f: param = hjson.loads(f.read())
    with args.hwcfg.open() as f: hw = hjson.loads(f.read())

    merged_config = {**param, **hw}
    print(emit_header_file(**merged_config))

if __name__ == "__main__": main()