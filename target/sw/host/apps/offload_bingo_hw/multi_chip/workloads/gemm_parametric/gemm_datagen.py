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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gemm_check_plan import (  # noqa: E402
    build_check_plan, resolve_inner_tiles, L1_BUDGET_BYTES,
)

np.random.seed(320)

def emit_header_file(**cfg):
    out = ["#pragma once", "#include <stdint.h>"]
    out += emit_gemm_data(**cfg)
    return "\n\n".join(out)


_ARRAY_CACHE = {}


def _build_arrays(**cfg):
    """Build A / B / golden D. emit_gemm_data and emit_mempool_bin must see THE
    SAME arrays, which is why they are built in one place. If the two ever
    diverged, the golden D would not match the A/B written into the mem chip,
    and the check would silently validate a wrong result."""
    M1, K1, N1 = cfg["M1"], cfg["K1"], cfg["N1"]
    array_shape = cfg["array_shape"]
    acc_cfg = cfg["snax_versacore_core_template"]["snax_acc_cfg"][0]
    meshRow, tileSize, meshCol = acc_cfg["snax_versacore_spatial_unrolling"][0][array_shape]

    # MEMOISE: emit_gemm_data and emit_mempool_bin both call this, and the
    # golden model is EXPENSIVE at large sizes (~30 s for 1024x256x128, so a
    # full minute wasted when called twice). Identical arrays are REQUIRED
    # anyway (otherwise A/B on the mem chip and the golden disagree), so the
    # cache is safe correctness-wise - it actually reinforces the intent.
    key = (M1, K1, N1, array_shape)
    if key in _ARRAY_CACHE:
        return _ARRAY_CACHE[key]

    np.random.seed(320)
    A = np.random.randint(-128, 127, size=(M1, K1, meshRow, tileSize), dtype=np.int8)
    B = np.random.randint(-128, 127, size=(K1, N1, tileSize, meshCol), dtype=np.int8)
    C = np.zeros((M1, N1, meshRow, meshCol), dtype=np.int32)
    D = block_gemm_golden_model(
        M1, K1, N1, meshRow, tileSize, meshCol,
        A.reshape(-1), B.reshape(-1), 0, 0, C.reshape(-1)
    )
    _ARRAY_CACHE[key] = (A, B, D, (meshRow, tileSize, meshCol))
    return _ARRAY_CACHE[key]


def emit_mempool_bin(out_dir, **cfg):
    """Mem-chip image: [A][B], in exactly the same order as the A_mp_base /
    B_mp_base layout in gemm_parametric.py.

    target/sim/apps/Makefile picks this file up (APP_MEM), converts it to hex
    bank files with bin2hex.py, hemaia_sim_runner splits it into banks and the
    testbench loads it into the mem chip. So THIS is how data gets onto the mem
    chip - the DFG NEVER writes to the mem chip, it only reads."""
    A, B, _D, mesh = _build_arrays(**cfg)
    meshRow, tileSize, meshCol = mesh
    M1, K1, N1 = cfg["M1"], cfg["K1"], cfg["N1"]
    grid_K = cfg.get("grid_K", 1)
    K1s = K1 // grid_K

    # A and B go to the memory chiplet partitioned into K slabs. With grid_K > 1
    # a worker reads only k in [w_k*K1s, (w_k+1)*K1s), which is strided on every
    # M row in a full-K layout; inside a slab the row width is K1s, so a tile
    # covering K is one contiguous chunk and costs one load per operand. At
    # 1024x4096x4096 on a 4x4x4 grid that is 64 nodes per worker instead of 1184.
    # Only the order of the image changes; golden D is unaffected.
    Av = A.reshape(M1, K1, meshRow, tileSize)
    # CAREFUL: B's SEMANTIC layout is (N1, K1, meshCol, tileSize) - N-MAJOR.
    # The (K1, N1, ...) shape in datagen only allocates a buffer of the right
    # size; the golden model reinterprets those same flat bytes as N-major. We
    # MUST slice against that real layout, not against the numpy shape.
    Bv = B.reshape(-1).reshape(N1, K1, meshCol, tileSize)

    os.makedirs(out_dir, exist_ok=True)

    # B is partitioned across the memory chiplets. mem_n_ranges comes from
    # mem_chip_n_ranges() in gemm_parametric.py, which is also where the DFG gets
    # its B load offsets; they must come from that one dict or workers read real
    # but wrong bytes and the check validates them. Each image holds the full A,
    # K-slab partitioned (one east-edge chiplet serves a whole column of w_m),
    # and for each slab only Bv[n_lo:n_hi]. With one memory chiplet, or
    # grid_N == 1, the range is [0, N1) and the image is byte-identical to the
    # unpartitioned one.
    mem_n_ranges = cfg.get("mem_n_ranges") or {}
    if not mem_n_ranges:
        mem_n_ranges = {None: (0, N1)}

    written = []
    for coord, (n_lo, n_hi) in sorted(mem_n_ranges.items(), key=lambda kv: (kv[0] is not None, kv[0])):
        name = "mempool.bin" if coord is None else f"mempool_{coord[0]}_{coord[1]}.bin"
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            for kk in range(grid_K):
                f.write(np.ascontiguousarray(Av[:, kk * K1s:(kk + 1) * K1s]).tobytes())
            b_bytes = 0
            for kk in range(grid_K):
                blk = np.ascontiguousarray(Bv[n_lo:n_hi, kk * K1s:(kk + 1) * K1s]).tobytes()
                f.write(blk)
                b_bytes += len(blk)
        written.append(path)
        print(f"{name} written: A={A.nbytes} B @0, "
              f"B blocks [{n_lo},{n_hi}) = {b_bytes} B @{A.nbytes} "
              f"(full B would be {B.nbytes} B)")

    # A single-mem-chip cfg keeps the historical mempool.bin name so the rest of
    # the flow (target/sim/apps/Makefile, hemaia_sim_runner) is unchanged.
    if len(written) == 1 and os.path.basename(written[0]) != "mempool.bin":
        import shutil
        legacy = os.path.join(out_dir, "mempool.bin")
        shutil.copyfile(written[0], legacy)
    return written[0]

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

    _A, _B, D, _mesh = _build_arrays(**cfg)

    # A and B are not emitted into the header: the operands are read from the
    # memory chiplet and the symbols are referenced nowhere in the DFG, they are
    # only needed here to compute golden D. Emitting them cost 24 MB per chip at
    # 2048x4096x4096. Golden D is likewise emitted only where the check nodes
    # read it, as D_check.
    grid_M = cfg.get("grid_M", 2)
    grid_N = cfg.get("grid_N", 2)
    check_bytes = cfg.get("check_bytes", 4096)
    block_bytes_D = meshRow * meshCol * 4

    # The inner tile choice comes from the SAME function the DFG uses: the
    # result buffer's (tile-packed) layout is derived from the tile shape, so
    # datagen must know tm/tn as well. If the two sides decided differently,
    # the check would compare the wrong bytes.
    grid_K = cfg.get("grid_K", 1)
    block_bytes_A = meshRow * tileSize * 1
    block_bytes_B = tileSize * meshCol * 1
    tm, _tk, tn = resolve_inner_tiles(
        cfg, M1 // grid_M, K1 // grid_K, N1 // grid_N,
        block_bytes_A, block_bytes_B, meshRow, meshCol,
        L1_BUDGET_BYTES)

    plan, total_bytes = build_check_plan(M1, N1, grid_M, grid_N, block_bytes_D,
                                         check_bytes, tm, tn)

    Dflat = D.reshape(-1)
    assert Dflat.dtype == np.int32, f"expected golden D to be int32, got {Dflat.dtype}"
    chunks = []
    expect = 0
    for w_m in range(grid_M):
        for w_n in range(grid_N):
            for (goff, _coff, nbytes, compact_off) in plan[(w_m, w_n)]:
                assert compact_off == expect, (
                    f"D_check plan inconsistent: compact_off={compact_off} expected={expect}"
                )
                assert goff % 4 == 0 and nbytes % 4 == 0, "offset/size must be divisible by 4"
                assert (goff + nbytes) <= Dflat.nbytes, (
                    f"golden run runs past the end of D: {goff}+{nbytes} > {Dflat.nbytes}"
                )
                chunks.append(Dflat[goff // 4: (goff + nbytes) // 4])
                expect += nbytes
    assert expect == total_bytes

    D_check = (np.concatenate(chunks) if chunks
               else np.zeros(0, dtype=np.int32))
    assert D_check.nbytes == total_bytes

    print(f"// D_check: {total_bytes} B ({len(chunks)} runs), "
          f"the full golden D would have been {Dflat.nbytes} B", file=sys.stderr)

    data += [format_vector_definition("int32_t", "D_check", D_check)]

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