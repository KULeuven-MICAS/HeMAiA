import os
import re
import glob
import sys
import argparse
import pathlib
import hjson

current_dir = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR = os.path.dirname(current_dir)
sys.path.append(WORKLOADS_DIR)
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)

sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gemm_datagen import emit_header_file, emit_mempool_bin
from gemm_check_plan import (
    build_check_plan, resolve_inner_tiles, _inner_tile_bytes, tile_node_count,
    L1_BUDGET_BYTES,
)

from bingo_dfg import BingoDFG
from bingo_platform import guard_cluster_count, guard_chiplet_count, parse_platform_cfg
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView, BingoMemSymbol, BingoMemFixedAddr
from bingo_helpers import chiplet_addr_transform_loc
from bingo_kernel_args import (
    SnaxBingoKernelXdmaSubmatrix2dArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelAraAddI32Args,
    SnaxBingoKernelGemmFullArgs,
)

# spm_wide is read from the system cfg; this is only the fallback. It matters:
# the tapeout cfgs have 128 KiB, not the 16 MiB of the single-chiplet ones, and
# the device binary already takes a large part of that, so anything big has to
# live on the memory chiplet.
DEFAULT_SPM_WIDE_BYTES = 16 * 1024 * 1024

# MEASURED on 2026-08-21 (this used to be an assumed "88 KiB, from a reference
# note"): the snax-bingo-offload device binary occupies exactly 82208 B in the
# .devicebin section (stat, multi_chip/gemm_parametric build).
DEVICE_BINARY_L3_BYTES = 82208       # MEASURED, deliberately NOT rounded up - see
# device_binary_l3_bytes(): the real .bin is stat'ed when available, and this is
# only the fallback. Rounding up to "stay on the safe side" DOES NOT WORK here:
# an extra 4 KiB of slack wrongly REJECTED a config that in reality links with
# 6.9 KiB to spare.
L3_MARGIN_BYTES = 16 * 1024  # margin for task-desc lists, scratchpads and other small allocs

# L3 cost of the generated host code. The DFG is unrolled statically, so the
# mini-compiler emits inline code per task into the host binary, which lives in
# WIDE_SPM. Without this term a config passes the datagen assert and then fails
# in the linker with "region `WIDE_SPM' overflowed", seven minutes into a build.
# Fitted on 16-chiplet builds: 520 tasks -> 62608 B, 2312 -> 267096 B .appl.
HOST_CODE_FIXED_BYTES = 12288      # runtime + printf + bingo host lib (task-independent)
HOST_CODE_BYTES_PER_TASK = 120     # generated code + read-only task tables, per task
# Room left for the runtime bingo_l3_alloc heap after the static image. An
# empirical bound: 168 tasks left 8024 B of heap and passed, 192 tasks left
# 4352 B and failed with "L3 malloc failed". Summing the per-chip alloc sizes
# out of the generated header would replace the guess.
HOST_L3_MARGIN_BYTES = 8 * 1024

MAIN_MEM_BASE_ADDR = 0x8000_0000

def read_system_cfg(system_cfg_path):
    """Read spm_wide and the mem-chip / compute-chip coordinates from the
    system cfg. Nothing is assumed - the tapeout and single-chiplet cfgs differ
    in ALL THREE of these values."""
    with open(system_cfg_path) as f:
        cfg = hjson.loads(f.read())
    tb = cfg.get("hemaia_multichip", {}).get("testbench_cfg", {})
    compute = [tuple(c["coordinate"]) for c in tb.get("hemaia_compute_chip", [])] or [(0, 0)]
    mem = tb.get("hemaia_mem_chip", [])
    # ALL mem chips, not just the first. A cfg may place several (the 4x4 grid
    # puts one on the east edge of each row); every one of them is a real
    # chiplet the workers can read from. mem_coord / mem_size stay as entry [0]
    # so single-mem-chip callers keep working unchanged.
    mem_chips = [{"coord": tuple(m["coordinate"]), "size": int(m["mem_size"])}
                 for m in mem]
    mem_coord = mem_chips[0]["coord"] if mem_chips else None
    mem_size = mem_chips[0]["size"] if mem_chips else 0
    return {
        "spm_wide_bytes": int(cfg.get("spm_wide", {}).get("length", DEFAULT_SPM_WIDE_BYTES)),
        "compute_coords": compute,
        "mem_chips": mem_chips,
        "mem_coord": mem_coord,
        "mem_size": mem_size,
        "clusters_per_chip": len(cfg.get("clusters", [])) or 1,
    }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--systemcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()

def build_worker_grid(grid_M, grid_K, grid_N, compute_coords, clusters_per_chip=1):
    """Place (w_m, w_k, w_n) on (chiplet, cluster):

        chiplet index = w_m * grid_N + w_n
        cluster index = w_k

    Putting w_k on the cluster axis keeps every partial of one (m,n) cell on the
    same chiplet, so the K reduction is chip-local and no summation crosses D2D.
    grid_M=4, grid_N=4, grid_K=4 is 16 chiplets of 4 clusters. A 4x1x1 grid on 4
    single-cluster chiplets falls out of the same rule.
    """
    n_chips = len(compute_coords)
    if grid_M * grid_N != n_chips:
        raise ValueError(
            f"grid_M*grid_N ({grid_M}*{grid_N}={grid_M*grid_N}) must equal the "
            f"number of chiplets ({n_chips}) - (w_m,w_n) maps to the chiplet axis "
            f"and w_k to the cluster axis. Fix either the compute chip count in "
            f"the system cfg or grid_M/grid_N."
        )
    if grid_K != clusters_per_chip:
        raise ValueError(
            f"grid_K ({grid_K}) must equal the clusters per chip "
            f"({clusters_per_chip}) - w_k maps to the cluster axis, which is what "
            f"keeps the K reduction chip-local."
        )

    workers = []
    for w_m in range(grid_M):
        for w_n in range(grid_N):
            cx, cy = compute_coords[w_m * grid_N + w_n]
            chip_id = (cx << 4) | cy
            for w_k in range(grid_K):
                workers.append({"chip": chip_id, "cluster": w_k, "loc": (cx, cy),
                                "w_m": w_m, "w_k": w_k, "w_n": w_n})
    return workers

def assign_worker_mem_chips(workers, mem_chips):
    """Give every worker the memory chiplet it reads A and B from.

    The nearest one by Manhattan distance from the worker's own compute chiplet,
    ties broken by (x, y) so the choice is deterministic. On the 4x4 grid with
    memory chiplets on the east edge at (4, 0..3) that hands every chiplet the
    one in its own row: a straight X hop, with four independent D2D links
    carrying the traffic instead of all sixteen chiplets funnelling through the
    link at (4,0). mem_chip_n_ranges then sizes each chiplet's image from this.
    """
    if not mem_chips:
        raise ValueError("assign_worker_mem_chips called with no mem chips")
    for w in workers:
        cx, cy = w["loc"]
        w["mem_loc"] = min(
            (mc["coord"] for mc in mem_chips),
            key=lambda c: (abs(c[0] - cx) + abs(c[1] - cy), c[0], c[1]),
        )
    return workers

def mem_chip_n_ranges(workers, N1_share, N1):
    """Memory chiplet coordinate -> [n_lo, n_hi) of the B blocks it must hold.

    The single source of truth for the partitioned image: the datagen writes each
    chiplet's image from this and the DFG computes its B offsets from the same
    dict. If the two disagreed, workers would read real but wrong bytes and the
    check would validate them.

    Only B is split. A worker (w_m, w_k, w_n) reads A by (w_m, w_k) and B by
    (w_n, w_k), and its chiplet sits at (x=w_m, y=w_n), so one memory chiplet
    serves every w_m for a single w_n: it needs all of A but only its own N-share
    of B. B is the big one anyway, 512 B a block against A's 16 B.

    The range is the hull of the n-ranges of the workers on that chiplet, so it
    is a superset and stays correct for any placement. With grid_N == 1, or one
    memory chiplet, it is [0, N1) and the image is byte-identical to the
    unpartitioned one.
    """
    ranges = {}
    for w in workers:
        lo = w["w_n"] * N1_share
        hi = lo + N1_share
        c = w["mem_loc"]
        if c in ranges:
            ranges[c] = (min(ranges[c][0], lo), max(ranges[c][1], hi))
        else:
            ranges[c] = (lo, hi)
    for c, (lo, hi) in ranges.items():
        assert 0 <= lo < hi <= N1, f"bad n-range {(lo, hi)} for mem chip {c} (N1={N1})"
    return ranges

def assert_mem_chip_budget(params, syscfg, mem_n_ranges):
    """Per-mem-chip capacity check, now that B is partitioned.

    Each chip holds the whole of A plus only its own hull of B, so the cost is
    A + (n_hi-n_lo)*K1*block_bytes_B -- not A + B. This is what buys the matrix
    size back after splitting one big mem chip into several smaller ones.
    """
    sizes = {mc["coord"]: mc["size"] for mc in syscfg["mem_chips"]}
    blkB = params["tileSize"] * params["meshCol"]
    worst = 0
    for coord, (lo, hi) in sorted(mem_n_ranges.items()):
        need = params["A_static_size"] + (hi - lo) * params["K1"] * blkB
        have = sizes[coord]
        worst = max(worst, need)
        assert need <= have, (
            f"mem chip {coord} needs {need} B (A {params['A_static_size']} + B "
            f"blocks [{lo},{hi}) = {(hi-lo) * params['K1'] * blkB}) but has "
            f"{have} B. Shrink the matrix, raise grid_N (a narrower B share per "
            f"chip), or give the mem chips more memory."
        )
    full = params["A_static_size"] + params["B_static_size"]
    print(f"mem chip budget: worst chip {worst} B vs {min(sizes.values())} B "
          f"available (unpartitioned would need {full} B, "
          f"{full / worst if worst else 1:.2f}x more)")

def define_workload_params(cfg_path, hwcfg_path, syscfg):
    with open(cfg_path) as f:
        param = hjson.loads(f.read())
    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    merged = {**param, **hw}

    data_type, array_shape = 0, merged["array_shape"]
    meshRow, tileSize, meshCol = merged["snax_versacore_core_template"]["snax_acc_cfg"][0][
        "snax_versacore_spatial_unrolling"][data_type][array_shape]

    grid_M, grid_K, grid_N = merged.get("grid_M", 2), merged.get("grid_K", 1), merged.get("grid_N", 2)
    M1, K1, N1 = merged["M1"], merged["K1"], merged["N1"]
    assert M1 % grid_M == 0, f"M1({M1}) must divide exactly by grid_M({grid_M})"
    assert K1 % grid_K == 0, f"K1({K1}) must divide exactly by grid_K({grid_K})"
    assert N1 % grid_N == 0, f"N1({N1}) must divide exactly by grid_N({grid_N})"
    M1_share, K1_share, N1_share = M1 // grid_M, K1 // grid_K, N1 // grid_N

    block_bytes_A = meshRow * tileSize * 1  # int8
    block_bytes_B = tileSize * meshCol * 1  # int8

    # A worker's whole (M1_share x K1_share x N1_share) share need not fit in L1:
    # it is split into inner tiles that do, along M and N first and K if needed.
    # One L1 buffer per worker is allocated once and reused across the tiles.
    inner_tile_M, inner_tile_K, inner_tile_N = resolve_inner_tiles(
        merged, M1_share, K1_share, N1_share,
        block_bytes_A, block_bytes_B, meshRow, meshCol, L1_BUDGET_BYTES)

    assert M1_share % inner_tile_M == 0, f"M1_share ({M1_share}) must be divisible by inner_tile_M ({inner_tile_M})"
    assert K1_share % inner_tile_K == 0, f"K1_share ({K1_share}) must be divisible by inner_tile_K ({inner_tile_K})"
    assert N1_share % inner_tile_N == 0, f"N1_share ({N1_share}) must be divisible by inner_tile_N ({inner_tile_N})"

    fit_bytes = _inner_tile_bytes(inner_tile_M, inner_tile_K, inner_tile_N,
                                   block_bytes_A, block_bytes_B, meshRow, meshCol)
    assert fit_bytes <= L1_BUDGET_BYTES, (
        f"inner tile (M={inner_tile_M}, K={inner_tile_K}, N={inner_tile_N}) exceeds the L1 budget "
        f"({fit_bytes} B > {L1_BUDGET_BYTES} B) - shrink inner_tile_M/K/N by hand or increase grid_M/grid_K/grid_N"
    )

    n_tile_m = M1_share // inner_tile_M
    n_tile_k = K1_share // inner_tile_K
    n_tile_n = N1_share // inner_tile_N

    D_full_size = M1 * N1 * meshRow * meshCol * 4
    A_static_size = M1 * K1 * block_bytes_A
    B_static_size = K1 * N1 * block_bytes_B

    # A and B live on the memory chiplet, not in L3; each worker pulls its own
    # share into L1 over D2D. What occupies L3 is the compiled-in golden D and
    # that chip's own D slice.
    spm_wide = syscfg["spm_wide_bytes"]
    # Per-worker D sub-block: (M1_share x N1_share) blocks, PACKED.
    # With grid_N=1 this is the same as D_full_size//grid_M; with grid_N>1 this
    # is the correct one (D_full_size//(grid_M*grid_N)).
    d_slice_size = (M1 // grid_M) * (N1 // grid_N) * meshRow * meshCol * 4

    # Check plan from the module shared with the datagen: the datagen emits only
    # the pieces this plan points at as D_check, and the check nodes read their
    # compact offsets from the same plan.
    check_bytes_param = merged.get("check_bytes", 4096)
    check_plan, check_plan_bytes = build_check_plan(
        M1, N1, grid_M, grid_N, meshRow * meshCol * 4, check_bytes_param,
        inner_tile_M, inner_tile_N)
    # Only the parts of golden D that the checks read are compiled in, so that
    # term is independent of matrix size. The one item that still grows with it
    # is the chip-local D slice, grid_K * d_slice_size; going bigger means
    # raising grid_M * grid_N, or moving D onto the memory chiplet as well.
    l3_required = (check_plan_bytes
                   + grid_K * d_slice_size
                   + DEVICE_BINARY_L3_BYTES + L3_MARGIN_BYTES)
    assert l3_required <= spm_wide, (
        f"L3 (spm_wide={spm_wide} B, read from the system cfg) is insufficient: "
        f"golden samples D_check ({check_plan_bytes} B) + per-chip D slice "
        f"(grid_K={grid_K} x {d_slice_size} B = {grid_K * d_slice_size} B) + "
        f"device binary ({DEVICE_BINARY_L3_BYTES} B) + margin ({L3_MARGIN_BYTES} B) "
        f"= {l3_required} B required.\n"
        f"  -> to shrink d_slice_size, raise grid_M*grid_N (more chips) or reduce "
        f"M1/N1. Reducing check_bytes shrinks D_check."
    )

    # ---- MEM CHIP layout ---------------------------------------------
    # Must match EXACTLY the order datagen writes into mempool.bin:
    #   [0]              A  (A_static_size bytes)
    #   [A_static_size]  B  (B_static_size bytes)
    if syscfg["mem_coord"] is None:
        raise ValueError(
            "This multi-chip workload requires a mem chip, but "
            "hemaia_multichip.testbench_cfg.hemaia_mem_chip is EMPTY in the system "
            "cfg. Use hemaia_tapeout_1c.hjson."
        )
    mem_chips = syscfg["mem_chips"]

    # ALL MEM CHIPS MUST BE THE SAME SIZE. occamygen.py sizes the SHARED
    # hemaia_mem_chip_xdma_wrapper module from mem_chip[0] and hands every
    # instance its own .WideSRAMSize, so unequal sizes build mismatched
    # hardware. occamygen rejects it too; this catches it before the build.
    mem_sizes = {mc["size"] for mc in mem_chips}
    if len(mem_sizes) > 1:
        raise ValueError(
            f"mem chips have differing mem_size ({sorted(mem_sizes)}). The "
            "generated hemaia_mem_chip_xdma_wrapper is one shared module sized "
            "from entry [0], so unequal sizes silently build wrong hardware."
        )

    # OFFSETS ARE MEM-CHIP-LOCAL. The image is replicated into every mem chip,
    # so [A][B] sits at the same local offsets in all of them and only the
    # chip id in the top address bits differs per worker -- which is why the
    # per-worker base is built at the load node from w["mem_loc"], not here.
    A_mp_local = MAIN_MEM_BASE_ADDR
    B_mp_local = A_mp_local + A_static_size
    mem_x, mem_y = syscfg["mem_coord"]
    A_mp_base = chiplet_addr_transform_loc(mem_x, mem_y, A_mp_local)
    B_mp_base = chiplet_addr_transform_loc(mem_x, mem_y, B_mp_local)

    # The real per-chip fit check needs the n-ranges, which need the worker
    # grid, so it lives in assert_mem_chip_budget() called from main(). This one
    # is only the cheap upper bound: A is replicated on every chip, so A alone
    # must fit no matter how B is partitioned.
    smallest = min(mc["size"] for mc in mem_chips)
    assert A_static_size <= smallest, (
        f"A ({A_static_size} B) is replicated onto every mem chip and does not "
        f"fit the smallest one ({smallest} B). A is NOT partitionable with mem "
        f"chips on the east edge: each serves a whole column of w_m."
    )

    params = {
        "M1": M1, "K1": K1, "N1": N1, "grid_M": grid_M, "grid_K": grid_K, "grid_N": grid_N,
        "M1_share": M1_share, "K1_share": K1_share, "N1_share": N1_share,
        "meshRow": meshRow, "tileSize": tileSize, "meshCol": meshCol,
        "arrayShapeIdx": array_shape, "transposeA": 0, "transposeB": 0,
        "inner_tile_M": inner_tile_M, "inner_tile_K": inner_tile_K, "inner_tile_N": inner_tile_N,
        "n_tile_m": n_tile_m, "n_tile_k": n_tile_k, "n_tile_n": n_tile_n,
        "D_full_size": D_full_size,
        # check_bytes: bytes verified per worker, 0 = everything.
        # __host_bingo_kernel_check_result compares byte by byte on CVA6 at
        # ~14.9 cycles/byte, so checking a 4 MiB D costs about 32 h of simulation
        # against ~31 min for the GEMM it verifies. The reference workloads use a
        # window for the same reason.
        "check_bytes": check_bytes_param,
        "check_plan": check_plan,
        "check_plan_bytes": check_plan_bytes,
        "A_static_size": A_static_size,
        "B_static_size": B_static_size,
        "d_slice_size": d_slice_size,
        "A_mp_base": A_mp_base,
        "B_mp_base": B_mp_base,
        # Mem-chip-LOCAL offsets (no chip id). The load nodes add the chip id of
        # whichever mem chip their worker was assigned; see assign_worker_mem_chips.
        "A_mp_local": A_mp_local,
        "B_mp_local": B_mp_local,
        "spm_wide_bytes": spm_wide,
    }
    return params, merged

def device_binary_l3_bytes(output_dir):
    """Real size of .devicebin; fall back to the measured constant if absent.

    The device binary is built BEFORE the generator runs, so it is normally
    available here. Stat'ing it keeps the constant current - if the device
    application grows, the budget tightens automatically.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        *([os.pardir] * 8)))
    pat = os.path.join(root, "target", "sw", "device", "apps", "snax",
                       "snax-bingo-offload", "build", "*snax-bingo-offload.bin")
    hits = glob.glob(pat)
    if not hits:
        return DEVICE_BINARY_L3_BYTES, "constant (bin not found)"
    return max(os.path.getsize(h) for h in hits), "measured"

def assert_host_l3_budget(params, output_dir, offload_file_name):
    """Check the L3 budget AFTER the DFG is compiled, with the real task count.

    The assert in define_workload_params only knows the DATA items; the amount
    of generated code is only known once the DFG has been compiled. This check
    still runs in the first seconds of a build, so it brings forward the
    "region `WIDE_SPM' overflowed" error the linker would otherwise report
    7 minutes later.
    """
    hdr = os.path.join(output_dir, offload_file_name)
    m = re.search(r"num_total_tasks\s*=\s*(\d+)", open(hdr).read())
    if m is None:
        return  # header not in the expected format - skip quietly, the linker will catch it
    tasks = int(m.group(1))

    spm_wide = params["spm_wide_bytes"]
    dev_bytes, dev_src = device_binary_l3_bytes(output_dir)
    host_code = HOST_CODE_FIXED_BYTES + HOST_CODE_BYTES_PER_TASK * tasks
    static_data = params["check_plan_bytes"]                       # D_check
    dynamic = params["grid_K"] * params["d_slice_size"]            # chip-local partial D
    total = dev_bytes + host_code + static_data + dynamic + HOST_L3_MARGIN_BYTES

    print(f"L3 budget (estimate): device binary {dev_bytes} [{dev_src}] + "
          f"generated code {host_code} ({tasks} tasks x {HOST_CODE_BYTES_PER_TASK} B) + "
          f"D_check {static_data} + D slice {dynamic} + margin {HOST_L3_MARGIN_BYTES} "
          f"= {total} / {spm_wide} B")

    assert total <= spm_wide, (
        f"L3 (spm_wide={spm_wide} B) is insufficient once the GENERATED CODE is "
        f"included: {total} B required ({total - spm_wide} B over).\n"
        f"  device binary {dev_bytes} B [{dev_src}] + generated code {host_code} B "
        f"({tasks} tasks) + D_check {static_data} B + D slice {dynamic} B "
        f"+ margin {HOST_L3_MARGIN_BYTES} B\n"
        f"  -> THE DOMINANT TERM IS USUALLY THE GENERATED CODE. Reduce the task "
        f"count: fewer tiles per worker (INCREASE inner_tile_M/K/N), or\n"
        f"     set inner_tile_K == K1_share - that emits a single Load_A + a "
        f"single Load_B per tile (the fast path instead of the strided fallback).\n"
        f"  -> on a platform with a 128 KiB spm_wide the ceiling is ~"
        f"{(spm_wide - dev_bytes - HOST_CODE_FIXED_BYTES - HOST_L3_MARGIN_BYTES) // HOST_CODE_BYTES_PER_TASK}"
        f" tasks."
    )

def create_dfg(params, platform, workers):
    dfg = BingoDFG(
        num_chiplets=platform["num_chiplets"], num_clusters_per_chiplet=platform["num_clusters_per_chiplet"],
        num_cores_per_cluster=platform["num_cores_per_cluster"], is_host_as_acc=True,
        chiplet_ids=platform["chiplet_ids"],
    )

    gemm_core, dma_core, host_core = 0, 1, 2
    grid_K = params["grid_K"]
    # For grid_K > 1 each worker needs its own partial-D buffer, or several w_k
    # workers write the same (m,n) location and silently overwrite each other.
    # Each buffer is chip-local and only as large as that worker's own slice, so
    # offsets into it are slice-relative (m_local below). The slice sits packed
    # there while the same sub-block is strided in golden D, which is why the
    # check runs per row-run.
    partial_D = {
        (w["chip"], w["w_k"]): BingoMemAlloc(
            f"computed_D_chip{w['chip']:02x}_k{w['w_k']}_l3",
            size=params["d_slice_size"], mem_level="L3", chip_id=w["chip"])
        for w in workers
    }

    block_bytes_A = params["meshRow"] * params["tileSize"] * 1
    block_bytes_B = params["tileSize"] * params["meshCol"] * 1

    print(f"inner_tile=(M={params['inner_tile_M']}, K={params['inner_tile_K']}, N={params['inner_tile_N']}), "
          f"n_tile=(m={params['n_tile_m']}, k={params['n_tile_k']}, n={params['n_tile_n']}) per worker")

    # Aligned with the worker order: each worker's last (Store) node.
    worker_last = []

    for w in workers:
        tag = f"chip{w['chip']:02x}_c{w['cluster']}"

        # ONE L1 buffer per worker for A/B/D, reused across all inner tiles
        # (see the project note: calling BingoMemAlloc inside the loop with a
        # per-tile name makes the reserved space grow multiplicatively).
        l1_A = BingoMemAlloc(f"A_l1_{tag}", size=params["inner_tile_M"] * params["inner_tile_K"] * block_bytes_A,
                              mem_level="L1", chip_id=w["chip"], cluster_id=w["cluster"])
        l1_B = BingoMemAlloc(f"B_l1_{tag}", size=params["inner_tile_K"] * params["inner_tile_N"] * block_bytes_B,
                              mem_level="L1", chip_id=w["chip"], cluster_id=w["cluster"])
        l1_D = BingoMemAlloc(f"D_l1_{tag}",
                              size=params["inner_tile_M"] * params["inner_tile_N"] * params["meshRow"] * params["meshCol"] * 4,
                              mem_level="L1", chip_id=w["chip"], cluster_id=w["cluster"])

        prev_node = None  # nothing to wait on before the worker's first load

        for n_idx in range(params["n_tile_n"]):
            n_lo = w["w_n"] * params["N1_share"] + n_idx * params["inner_tile_N"]

            for m_idx in range(params["n_tile_m"]):
                m_lo = w["w_m"] * params["M1_share"] + m_idx * params["inner_tile_M"]

                # K is innermost: for one (m_idx, n_idx) piece, successive
                # matmuls accumulate in l1_D through accumPrevC. That accumulator
                # is a single VersaCore register holding one tile, so it is only
                # correct at inner_tile_M == inner_tile_N == 1; gemm_check_plan
                # enforces it, because violating it is silently wrong.
                for k_idx in range(params["n_tile_k"]):
                    k_lo = w["w_k"] * params["K1_share"] + k_idx * params["inner_tile_K"]
                    itag = f"{tag}_m{m_idx}_n{n_idx}_k{k_idx}"

                    # iDMA. The K-slab layout of the operand image makes an
                    # inner tile one contiguous run per operand, so there is no
                    # reshape for xDMA to do here.
                    #
                    # A is blocked (M1, K1) row-major, B is (K1, N1) row-major. A
                    # sub-block spanning a full row is one contiguous run,
                    # otherwise it is one run per row.
                    #
                    # A and B are in K-slab order on the memory chiplet (see
                    # emit_mempool_bin): slab kk = k // K1_share, and inside a slab
                    # the row width is K1_share rather than K1. So when an inner
                    # tile covers K completely, both A's M rows and B's N rows are
                    # contiguous and a tile costs one load of each. k_rel is the
                    # K-block offset inside the slab.
                    K1s = params["K1_share"]
                    k_rel = k_lo - w["w_k"] * K1s
                    a_slab = w["w_k"] * params["M1"] * K1s * block_bytes_A
                    # B IS PARTITIONED ACROSS MEM CHIPS: this worker's chip holds
                    # only B blocks [n_mem_lo, n_mem_hi), so the B slab is that
                    # much narrower and n is counted from n_mem_lo, not from 0.
                    # mem_chip_n_ranges() is the single source of truth the
                    # datagen writes the image from. With one mem chip (or
                    # grid_N == 1) the range is [0, N1) and this is exactly the
                    # old formula.
                    n_mem_lo, n_mem_hi = params["mem_n_ranges"][w["mem_loc"]]
                    N1_mem = n_mem_hi - n_mem_lo
                    b_slab = w["w_k"] * N1_mem * K1s * block_bytes_B
                    a_row = K1s * block_bytes_A
                    a_tile_row = params["inner_tile_K"] * block_bytes_A
                    if params["inner_tile_K"] == K1s:
                        a_chunks = [(a_slab + m_lo * a_row,
                                     params["inner_tile_M"] * a_row, 0)]
                    else:
                        a_chunks = [(a_slab + (m_lo + r) * a_row + k_rel * block_bytes_A,
                                     a_tile_row, r * a_tile_row)
                                    for r in range(params["inner_tile_M"])]

                    # B is N-major: (N1, K1, meshCol, tileSize). The datagen
                    # allocates it as (K1, N1, tileSize, meshCol), but the golden
                    # model reinterprets the same flat bytes as b.reshape(n, k,
                    # col, size), and that is the layout that counts. The two
                    # orderings coincide at N1 = 1, so this is invisible until
                    # grid_N > 1. L1 follows the same order.
                    b_row = K1s * block_bytes_B
                    if params["inner_tile_K"] == K1s:
                        # For each n the K run is a full row inside the slab, and
                        # successive n's are adjacent too -> one single chunk.
                        b_chunks = [(b_slab + (n_lo - n_mem_lo) * b_row,
                                     params["inner_tile_N"] * params["inner_tile_K"] * block_bytes_B,
                                     0)]
                    else:
                        b_chunks = [(b_slab + (n_lo - n_mem_lo + nn) * b_row + k_rel * block_bytes_B,
                                     params["inner_tile_K"] * block_bytes_B,
                                     nn * params["inner_tile_K"] * block_bytes_B)
                                    for nn in range(params["inner_tile_N"])]

                    for ci, (src_off, nbytes, dst_off) in enumerate(a_chunks):
                        node_A = BingoNode(
                            assigned_chiplet_id=w["chip"], assigned_cluster_id=w["cluster"], assigned_core_id=dma_core,
                            node_name=f"Load_A_{itag}_p{ci}",
                            kernel_name="__snax_bingo_kernel_idma_1d_copy",
                            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                                # A comes from the MEM CHIP (mempool.bin), not from
                                # a symbol in the chip's own L3.
                                # From THIS WORKER'S mem chip (see
                                # assign_worker_mem_chips): the image is
                                # replicated, so the offset is identical on
                                # every chip and only the chip id differs.
                                src_addr=BingoMemFixedAddr(chiplet_addr_transform_loc(
                                    w["mem_loc"][0], w["mem_loc"][1],
                                    params["A_mp_local"] + src_off)),
                                dst_addr=l1_A if dst_off == 0 else BingoMemAllocView(l1_A, dst_off),
                                size=nbytes,
                            ),
                        )
                        dfg.bingo_add_node(node_A)
                        if prev_node is not None:
                            dfg.add_edge(prev_node, node_A)
                        prev_node = node_A

                    for ci, (src_off, nbytes, dst_off) in enumerate(b_chunks):
                        node_B = BingoNode(
                            assigned_chiplet_id=w["chip"], assigned_cluster_id=w["cluster"], assigned_core_id=dma_core,
                            node_name=f"Load_B_{itag}_p{ci}",
                            kernel_name="__snax_bingo_kernel_idma_1d_copy",
                            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                                src_addr=BingoMemFixedAddr(chiplet_addr_transform_loc(
                                    w["mem_loc"][0], w["mem_loc"][1],
                                    params["B_mp_local"] + src_off)),
                                dst_addr=l1_B if dst_off == 0 else BingoMemAllocView(l1_B, dst_off),
                                size=nbytes,
                            ),
                        )
                        dfg.bingo_add_node(node_B)
                        dfg.add_edge(prev_node, node_B)
                        prev_node = node_B

                    node_Gemm = BingoNode(
                        assigned_chiplet_id=w["chip"], assigned_cluster_id=w["cluster"], assigned_core_id=gemm_core,
                        node_name=f"Gemm_{itag}", kernel_name="__snax_bingo_kernel_gemm_full",
                        kernel_args=SnaxBingoKernelGemmFullArgs(
                            input_A_addr=l1_A, input_B_addr=l1_B, input_C_addr=0, output_D_addr=l1_D,
                            M=params["inner_tile_M"], K=params["inner_tile_K"], N=params["inner_tile_N"],
                            array_shape_idx=params["arrayShapeIdx"],
                            transpose_A=params["transposeA"], transpose_B=params["transposeB"],
                            accumPrevC=1 if k_idx > 0 else 0,
                        ),
                    )
                    dfg.bingo_add_node(node_Gemm)
                    dfg.add_edge(prev_node, node_Gemm)

                    prev_node = node_Gemm

                # The k loop is done: write the finished D piece to partial D.
                # One 1D copy per M-block row, not an xdma_6d descriptor -- the
                # AGU stride and bound values for a blocked-to-blocked sub-region
                # came out wrong and most elements were never written.
                block_bytes_D = params["meshRow"] * params["meshCol"] * 4
                # Tile-packed store, one node per tile. partial_D packs tiles back
                # to back rather than keeping block-row order, so all of l1_D is
                # contiguous at the destination. In row-major order a tile narrower
                # than the row was strided and cost inner_tile_M nodes, which is a
                # DFG that cannot be compiled at size. build_check_plan derives the
                # same layout, which is why it is a shared module.
                tile_bytes_D = params["inner_tile_M"] * params["inner_tile_N"] * block_bytes_D
                tile_off = (m_idx * params["n_tile_n"] + n_idx) * tile_bytes_D
                node_Store = BingoNode(
                    assigned_chiplet_id=w["chip"], assigned_cluster_id=w["cluster"], assigned_core_id=dma_core,
                    node_name=f"Store_D_{tag}_m{m_idx}_n{n_idx}",
                    kernel_name="__snax_bingo_kernel_idma_1d_copy",
                    kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                        src_addr=l1_D,
                        dst_addr=BingoMemAllocView(partial_D[(w["chip"], w["w_k"])], tile_off),
                        size=tile_bytes_D,
                    ),
                )
                dfg.bingo_add_node(node_Store)
                dfg.add_edge(prev_node, node_Store)
                prev_node = node_Store

        # Wired directly into the check node, or into the reduction add for
        # grid_K > 1.
        worker_last.append(prev_node)

    # Workers are NOT chained to each other. The mini-compiler already splits a
    # node with several local successors or predecessors into a dummy chain
    # (bingo_transform_dfg_add_dummy_{set,check}_nodes), so entry can fan out to
    # every worker root and the check can be fed by every worker's last store.
    # Chaining them by hand adds edges the CI references never exercise, and it
    # deadlocked entry's dispatch to the first worker.
    # K reduction and final check. The placement rule puts the grid_K workers of
    # one (m,n) cell on the same chiplet in different clusters, so the reduction
    # is chip-local: partial[(chip,0)] is the accumulator, k = 1..grid_K-1 are
    # added onto it in sequence, and the result is checked on that chip's host.
    block_bytes_D = params["meshRow"] * params["meshCol"] * 4

    # chip -> the workers on that chip (ordered by w_k) and their last store nodes
    by_chip = {}
    for w, last in zip(workers, worker_last):
        by_chip.setdefault(w["chip"], []).append((w, last))
    for chip in by_chip:
        by_chip[chip].sort(key=lambda t: t[0]["w_k"])

    result_ready = {}   # chip -> the node at which the result is ready
    result_buf = {}     # chip -> the result buffer
    for chip, entries in by_chip.items():
        w0, last0 = entries[0]
        acc = partial_D[(chip, 0)]
        result_buf[chip] = acc
        if grid_K == 1:
            result_ready[chip] = last0
            continue
        # Chip-local host add_i32 chain. Xiaoling's reference pattern.
        # NOTE: the add node INHERENTLY has 2 inputs (accumulator ready + the
        # k-th worker's store finished). This is the only unavoidable fan-in in
        # the project; no presence-bit collision is expected because different
        # cells set different tags. grid_K>1 has since been validated in real
        # RTL (two runs, both PASS).
        prev_ready = last0
        for k in range(1, grid_K):
            wk, lastk = entries[k]
            add_node = BingoNode(
                assigned_chiplet_id=chip, assigned_cluster_id=0, assigned_core_id=host_core,
                node_name=f"Add_chip{chip:02x}_k0_to_k{k}",
                kernel_name="__host_bingo_kernel_add_i32",
                kernel_args=HostBingoKernelAraAddI32Args(
                    input_a_addr=acc, input_b_addr=partial_D[(chip, k)], output_addr=acc,
                    num_elements=params["d_slice_size"] // 4,
                ),
            )
            dfg.bingo_add_node(add_node)
            dfg.add_edge(prev_ready, add_node)
            dfg.add_edge(lastk, add_node)
            prev_ready = add_node
        result_ready[chip] = prev_ready

    # Final check, per row-run and chip-local. A worker's D sub-block is strided
    # inside golden D but packed on the computed side, and each block row is
    # contiguous on both, so one check per row. The checks run on the owning
    # chip's own host core, chained rather than fanned out. The run list comes
    # from the shared plan in gemm_check_plan, which the datagen also uses to emit
    # D_check.
    check_plan = params["check_plan"]

    for chip, entries in by_chip.items():
        w = entries[0][0]
        wtag = f"chip{chip:02x}"
        wbuf = result_buf[chip]

        runs = check_plan[(w["w_m"], w["w_n"])]

        prev = result_ready[chip]
        for i, (goff, coff, nbytes, compact_off) in enumerate(runs):
            cn = BingoNode(
                assigned_chiplet_id=chip, assigned_cluster_id=0, assigned_core_id=host_core,
                node_name=f"Host_Check_D_{wtag}_r{i}",
                kernel_name="__host_bingo_kernel_check_result",
                kernel_args=HostBingoKernelCheckResultArgs(
                    name=f"D_{wtag}_r{i}",
                    # goff (the offset into the full golden D) is NO LONGER USED;
                    # datagen wrote that piece into D_check at compact_off.
                    golden_data_addr=BingoMemSymbol("D_check", offset=compact_off),
                    output_data_addr=(BingoMemAllocView(wbuf, coff) if coff else wbuf),
                    data_size=nbytes,
                ),
            )
            dfg.bingo_add_node(cn)
            dfg.add_edge(prev, cn)
            prev = cn

    return dfg

def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.cfg) as f:
        raw_cfg = hjson.loads(f.read())

    grid_M = raw_cfg.get("grid_M", 4)
    grid_K = raw_cfg.get("grid_K", 1)
    grid_N = raw_cfg.get("grid_N", 1)
    clusters_per_chip = raw_cfg.get("clusters_per_chip", 1)

    syscfg = read_system_cfg(args.systemcfg)
    print(f"system cfg: spm_wide={syscfg['spm_wide_bytes']} B, "
          f"compute chips={syscfg['compute_coords']}, clusters/chip="
          f"{syscfg['clusters_per_chip']}")
    print(f"{len(syscfg['mem_chips'])} mem chip(s): " + ", ".join(
        f"({mc['coord'][0]},{mc['coord'][1]})={mc['size']}B"
        for mc in syscfg["mem_chips"]) + "  [A replicated, B partitioned]")

    workers = build_worker_grid(grid_M, grid_K, grid_N, syscfg["compute_coords"],
                                 clusters_per_chip=clusters_per_chip)
    assign_worker_mem_chips(workers, syscfg["mem_chips"])
    print(f"{len(workers)} worker: " + ", ".join(
        f"chip{w['chip']:02x}/c{w['cluster']}->w_m{w['w_m']}" for w in workers))
    # One line per mem chip rather than per worker: with 64 workers the
    # per-worker list is unreadable, and what matters is the FAN-OUT -- how
    # many compute chiplets ended up on each mem chip's link.
    by_mem = {}
    for w in workers:
        by_mem.setdefault(w["mem_loc"], set()).add(w["loc"])
    print("mem-chip fan-out: " + ", ".join(
        f"({mx},{my})<-{sorted(locs)}" for (mx, my), locs in sorted(by_mem.items())))

    params, merged = define_workload_params(args.cfg, args.hwcfg, syscfg)
    merged["num_clusters"] = merged.get("num_clusters", clusters_per_chip)
    merged["num_chiplets"] = merged.get("num_chiplets", raw_cfg.get("num_chips", 1))

    # B IS PARTITIONED PER MEM CHIP. One dict, computed once, used by BOTH the
    # datagen (which writes each chip's image) and the DFG (which computes the
    # load offsets). They MUST agree -- see mem_chip_n_ranges().
    mem_n_ranges = mem_chip_n_ranges(workers, params["N1_share"], params["N1"])
    params["mem_n_ranges"] = mem_n_ranges
    merged["mem_n_ranges"] = mem_n_ranges
    assert_mem_chip_budget(params, syscfg, mem_n_ranges)
    print("mem-chip B partition: " + ", ".join(
        f"({c[0]},{c[1]})=blocks[{lo},{hi})" for c, (lo, hi) in sorted(mem_n_ranges.items())))

    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(**merged))
        # Mem-chip image. We produce the same data with the SAME call used for
        # gemm_data.h; the order matches A_mp_base/B_mp_base in params exactly.
        emit_mempool_bin(os.path.join(args.output_dir, "build"), **merged)

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_chiplet_count(merged, platform, args.output_dir, args.output_offload_file_name):
        return
    if not guard_cluster_count(merged, platform, args.output_dir, args.output_offload_file_name):
        return

    dfg = create_dfg(params, platform, workers)
    dfg.bingo_compile_dfg("GEMM_LargeScale", args.output_dir, args.output_offload_file_name,
                           extra_include_header_list=["gemm_data.h"])
    # The REAL L3 budget including generated code - fail before the linker does.
    assert_host_l3_budget(params, args.output_dir, args.output_offload_file_name)

if __name__ == "__main__":
    main()
