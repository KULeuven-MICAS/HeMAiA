import os
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
from gemm_datagen import emit_header_file

from bingo_dfg import BingoDFG
from bingo_platform import guard_cluster_count, parse_platform_cfg
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView, BingoMemSymbol
from bingo_kernel_args import (
    SnaxBingoKernelXdmaSubmatrix2dArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    HostBingoKernelCheckResultArgs,
    HostBingoKernelAraAddI32Args,
    SnaxBingoKernelGemmFullArgs,
)

# Observed L1 capacity (from the OOM error message): capacity=516864 B, with
# ~66048 B already reserved for something else at that moment. With some margin
# this is an estimate we can use safely without re-measuring the real L1
# capacity - if the real value differs (e.g. on another cluster/chip cfg),
# update it here.
L1_BUDGET_BYTES = 400 * 1024  # ASSUMPTION - verify

# Golden D is always resident in L3, and the grid_K partial-D buffers ask for
# the same size again through bingo_l3_alloc, so a single D matrix occupies L3
# twice. (grid_K+1)*D_full_size + A_static + B_static is checked against this to
# get an explicit assert instead of an opaque runtime "L3 malloc failed".
SPM_WIDE_BYTES = 16 * 1024 * 1024
L3_MARGIN_BYTES = 256 * 1024  # margin for task-desc lists, scratchpads and other small allocs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_offload_file_name", type=str, default="offload_bingo_hw.h")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("--platformcfg", type=pathlib.Path, required=True)
    parser.add_argument("--data_h", type=pathlib.Path, default=None)
    return parser.parse_args()

def build_worker_grid(grid_M, grid_K, grid_N, num_rows=1, num_cols=1, clusters_per_chip=4):
    """Place clusters_per_chip workers on a (w_m,w_k,w_n) 3D grid within one chip.
    grid_K>1 is now supported: each worker writes into its own partial-D L3
    buffer, and at the end create_dfg reduces them with a host add_i32 (see the
    reduction chain)."""
    workers = []
    worker_idx = 0
    for r in range(num_rows):
        for c in range(num_cols):
            chip_id = (r << 4) | c
            for cluster in range(clusters_per_chip):
                w_m = (worker_idx // (grid_K * grid_N)) % grid_M
                w_k = (worker_idx // grid_N) % grid_K
                w_n = worker_idx % grid_N
                workers.append({"chip": chip_id, "cluster": cluster, "w_m": w_m, "w_k": w_k, "w_n": w_n})
                worker_idx += 1
    return workers

def _largest_divisor_leq(share, limit):
    """Largest divisor of `share` that is not greater than `limit` (inclusive)."""
    limit = min(limit, share)
    if limit < 1:
        return 1
    for d in range(limit, 0, -1):
        if share % d == 0:
            return d
    return 1

def _inner_tile_bytes(tm, tk, tn, block_bytes_A, block_bytes_B, meshRow, meshCol):
    return (tm * tk * block_bytes_A) + (tk * tn * block_bytes_B) + (tm * tn * meshRow * meshCol * 4)

def _auto_inner_tiles(M1_share, K1_share, N1_share, block_bytes_A, block_bytes_B,
                       meshRow, meshCol, budget):
    """Split a worker's (M1_share x K1_share x N1_share) share into
    (inner_tile_M x inner_tile_K x inner_tile_N) pieces that fit in L1 at once.

    M is shrunk FIRST, then N (if needed), and K last (if needed) - not
    round-robin: each dimension is shrunk until exhausted (down to 1) before
    moving on. Shrinking M/N is free because it produces independent output
    regions; shrinking K instead requires successive matmuls for the same (m,n)
    piece to accumulate in L1 via accumPrevC (see create_dfg), and accumPrevC
    carries a one-tile constraint, so it only kicks in when M and N are
    exhausted and it still does not fit."""
    def fits(tm, tk, tn):
        return _inner_tile_bytes(tm, tk, tn, block_bytes_A, block_bytes_B, meshRow, meshCol) <= budget

    tm, tk, tn = M1_share, K1_share, N1_share
    while tm > 1 and not fits(tm, tk, tn):
        tm = _largest_divisor_leq(M1_share, tm - 1)
    while tn > 1 and not fits(tm, tk, tn):
        tn = _largest_divisor_leq(N1_share, tn - 1)
    while tk > 1 and not fits(tm, tk, tn):
        tk = _largest_divisor_leq(K1_share, tk - 1)
    return tm, tk, tn

def define_workload_params(cfg_path, hwcfg_path):
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
    inner_tile_M = merged.get("inner_tile_M")
    inner_tile_K = merged.get("inner_tile_K")
    inner_tile_N = merged.get("inner_tile_N")

    if inner_tile_M is None or inner_tile_K is None or inner_tile_N is None:
        auto_tm, auto_tk, auto_tn = _auto_inner_tiles(
            M1_share, K1_share, N1_share, block_bytes_A, block_bytes_B, meshRow, meshCol, L1_BUDGET_BYTES,
        )
        inner_tile_M = inner_tile_M if inner_tile_M is not None else auto_tm
        inner_tile_K = inner_tile_K if inner_tile_K is not None else auto_tk
        inner_tile_N = inner_tile_N if inner_tile_N is not None else auto_tn

    assert M1_share % inner_tile_M == 0, f"M1_share ({M1_share}) must divide exactly by inner_tile_M ({inner_tile_M})"
    assert K1_share % inner_tile_K == 0, f"K1_share ({K1_share}) must divide exactly by inner_tile_K ({inner_tile_K})"
    assert N1_share % inner_tile_N == 0, f"N1_share ({N1_share}) must divide exactly by inner_tile_N ({inner_tile_N})"

    fit_bytes = _inner_tile_bytes(inner_tile_M, inner_tile_K, inner_tile_N,
                                   block_bytes_A, block_bytes_B, meshRow, meshCol)
    assert fit_bytes <= L1_BUDGET_BYTES, (
        f"inner tile (M={inner_tile_M}, K={inner_tile_K}, N={inner_tile_N}) L1 butcesini asiyor "
        f"({fit_bytes} B > {L1_BUDGET_BYTES} B) - shrink inner_tile_M/K/N by hand or increase grid_M/grid_K/grid_N"
    )

    n_tile_m = M1_share // inner_tile_M
    n_tile_k = K1_share // inner_tile_K
    n_tile_n = N1_share // inner_tile_N

    D_full_size = M1 * N1 * meshRow * meshCol * 4
    A_static_size = M1 * K1 * block_bytes_A
    B_static_size = K1 * N1 * block_bytes_B
    l3_required = (grid_K + 1) * D_full_size + A_static_size + B_static_size + L3_MARGIN_BYTES
    assert l3_required <= SPM_WIDE_BYTES, (
        f"L3 (SPM_WIDE={SPM_WIDE_BYTES} B) is insufficient: golden D ({D_full_size} B, static) + "
        f"{grid_K} partial-D buffers ({grid_K * D_full_size} B, dynamic) + A ({A_static_size} B) + "
        f"B ({B_static_size} B) + margin ({L3_MARGIN_BYTES} B) = {l3_required} B required - "
        f"reduce M1/K1/N1 or lower grid_K"
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
        # against ~31 min for the GEMM it verifies.
        "check_bytes": merged.get("check_bytes", 4096),
    }
    return params, merged

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
    # They are accumulated onto partial_D[0] by a host add_i32 chain at the end.
    partial_D = {
        k: BingoMemAlloc(f"computed_D_partial_k{k}_l3", size=params["D_full_size"], mem_level="L3",
                          chip_id=workers[0]["chip"])
        for k in range(grid_K)
    }
    computed_D = partial_D[0]

    block_bytes_A = params["meshRow"] * params["tileSize"] * 1
    block_bytes_B = params["tileSize"] * params["meshCol"] * 1

    print(f"inner_tile=(M={params['inner_tile_M']}, K={params['inner_tile_K']}, N={params['inner_tile_N']}), "
          f"n_tile=(m={params['n_tile_m']}, k={params['n_tile_k']}, n={params['n_tile_n']}) per worker")

    group_final_stores = {k: [] for k in range(grid_K)}

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
                # correct at inner_tile_M == inner_tile_N == 1; violating it is
                # silently wrong.
                for k_idx in range(params["n_tile_k"]):
                    k_lo = w["w_k"] * params["K1_share"] + k_idx * params["inner_tile_K"]
                    itag = f"{tag}_m{m_idx}_n{n_idx}_k{k_idx}"

                    # iDMA, not xDMA: an xDMA read issued from cluster 3 commits
                    # on XDMA_COMMIT_REMOTE and then spins forever because
                    # XDMA_FINISH_REMOTE never advances. Every CI-validated
                    # multi-cluster workload uses idma_1d_copy only.
                    #
                    # A is blocked (M1, K1) row-major, B is (K1, N1) row-major. A
                    # sub-block spanning a full row is one contiguous run,
                    # otherwise it is one run per row.
                    a_full_row = params["K1"] * block_bytes_A
                    a_tile_row = params["inner_tile_K"] * block_bytes_A
                    if params["inner_tile_K"] == params["K1"]:
                        a_chunks = [(m_lo * a_full_row,
                                     params["inner_tile_M"] * a_full_row, 0)]
                    else:
                        a_chunks = [((m_lo + r) * a_full_row + k_lo * block_bytes_A,
                                     a_tile_row, r * a_tile_row)
                                    for r in range(params["inner_tile_M"])]

                    # B is N-major: (N1, K1, meshCol, tileSize), so block (n, k)
                    # is at (n * K1 + k) * block_bytes_B. The datagen allocates it
                    # as (K1, N1, tileSize, meshCol), but the golden model
                    # reinterprets the same flat bytes as b.reshape(n, k, col,
                    # size), and that is the layout that counts. The two coincide
                    # at N1 = 1, so this is invisible until grid_N > 1. L1 follows
                    # the same order.
                    if params["inner_tile_K"] == params["K1"]:
                        # For each n the K run is a full row, and successive n's
                        # are adjacent -> one single chunk.
                        # (itK == K1 => grid_K == 1 => k_lo == 0)
                        assert k_lo == 0, "k_lo must be 0 when itK==K1"
                        b_chunks = [(n_lo * params["K1"] * block_bytes_B,
                                     params["inner_tile_N"] * params["inner_tile_K"] * block_bytes_B,
                                     0)]
                    else:
                        b_chunks = [((((n_lo + nn) * params["K1"]) + k_lo) * block_bytes_B,
                                     params["inner_tile_K"] * block_bytes_B,
                                     nn * params["inner_tile_K"] * block_bytes_B)
                                    for nn in range(params["inner_tile_N"])]

                    for ci, (src_off, nbytes, dst_off) in enumerate(a_chunks):
                        node_A = BingoNode(
                            assigned_chiplet_id=w["chip"], assigned_cluster_id=w["cluster"], assigned_core_id=dma_core,
                            node_name=f"Load_A_{itag}_p{ci}",
                            kernel_name="__snax_bingo_kernel_idma_1d_copy",
                            kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                                src_addr=BingoMemSymbol("A", offset=src_off),
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
                                src_addr=BingoMemSymbol("B", offset=src_off),
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
                # came out wrong and most elements were never written. Each
                # M-block row is contiguous on both sides; only the stride between
                # rows differs.
                block_bytes_D = params["meshRow"] * params["meshCol"] * 4
                row_write_bytes = params["inner_tile_N"] * block_bytes_D
                for row in range(params["inner_tile_M"]):
                    src_row = BingoMemAllocView(l1_D, row * row_write_bytes)
                    dst_row_offset = ((m_lo + row) * params["N1"] + n_lo) * block_bytes_D
                    dst_row = BingoMemAllocView(partial_D[w["w_k"]], dst_row_offset)
                    node_Store = BingoNode(
                        assigned_chiplet_id=w["chip"], assigned_cluster_id=w["cluster"], assigned_core_id=dma_core,
                        node_name=f"Store_D_{tag}_m{m_idx}_n{n_idx}_row{row}",
                        kernel_name="__snax_bingo_kernel_idma_1d_copy",
                        kernel_args=SnaxBingoKernelIdma1dCopyArgs(
                            src_addr=src_row, dst_addr=dst_row, size=row_write_bytes,
                        ),
                    )
                    dfg.bingo_add_node(node_Store)
                    dfg.add_edge(prev_node, node_Store)
                    prev_node = node_Store

        # This worker's LAST node (Store) will be wired directly into
        # check_node (or the reduction add_node for grid_K>1) as one fan-in
        # edge - see "FIX (2026-08-19, second pass)" below.
        group_final_stores[w["w_k"]].append(prev_node)

    # Workers are NOT chained to each other. The mini-compiler already splits a
    # node with several local successors or predecessors into a dummy chain
    # (bingo_transform_dfg_add_dummy_{set,check}_nodes), so entry can fan out to
    # every worker root and the check can be fed by every worker's last store.
    # Chaining them by hand adds edges the CI references never exercise, and it
    # deadlocked entry's dispatch to the first worker.
    if grid_K > 1:
        # Host-side add_i32 reduction, the pattern from Xiaoling's reference:
        # partial_D[0] is the accumulator and partial_D[1..] are added onto it
        # in sequence. D_full_size is in bytes; add_i32 wants an element count
        # (int32 = 4 bytes).
        prev_sum = partial_D[0]
        prev_ready_nodes = group_final_stores[0]
        for k in range(1, grid_K):
            add_node = BingoNode(
                assigned_chiplet_id=workers[0]["chip"], assigned_cluster_id=0, assigned_core_id=host_core,
                node_name=f"Add_partial_k0_to_k{k}", kernel_name="__host_bingo_kernel_add_i32",
                kernel_args=HostBingoKernelAraAddI32Args(
                    input_a_addr=prev_sum, input_b_addr=partial_D[k], output_addr=prev_sum,
                    num_elements=params["D_full_size"] // 4,
                ),
            )
            dfg.bingo_add_node(add_node)
            for n in prev_ready_nodes:
                dfg.add_edge(n, add_node)
            for n in group_final_stores[k]:
                dfg.add_edge(n, add_node)
            prev_ready_nodes = [add_node]
        check_deps = prev_ready_nodes
    else:
        check_deps = group_final_stores[0]

    # One check node per worker, never a four-way fan-in. The workers' last store
    # nodes all DepSet the same tag into the host cell, and the hardware dep
    # matrix is a presence-bit scoreboard rather than a counter: four sets of one
    # tag collapse into one bit, the first consumer clears it and the rest wait
    # forever. gemm_nsplit_accumprevc_4cluster uses a per-cluster check for the
    # same reason; each worker owns a contiguous M-slice of D, so its slice can be
    # checked on its own.
    block_bytes_D = params["meshRow"] * params["meshCol"] * 4
    if grid_K > 1:
        # grid_K>1 already funnels through the single host add_i32 reduction
        # chain, so check_deps is one node -> no fan-in.
        check_node = BingoNode(
            assigned_chiplet_id=platform["chiplet_ids"][0], assigned_cluster_id=0, assigned_core_id=host_core,
            node_name="Host_Check_D", kernel_name="__host_bingo_kernel_check_result",
            kernel_args=HostBingoKernelCheckResultArgs(
                name="D_Check",
                golden_data_addr=BingoMemSymbol("D"), output_data_addr=computed_D,
                data_size=params["D_full_size"],
            ),
        )
        dfg.bingo_add_node(check_node)
        for node in check_deps:
            dfg.add_edge(node, check_node)
    elif params["grid_N"] == 1:
        slice_bytes = params["M1_share"] * params["N1"] * block_bytes_D
        # Verification WINDOWS per worker. With check_bytes=0 the whole slice is
        # checked (slow, see the note in params); otherwise one window from the
        # START and one from the END of the slice - offset bugs show up most at
        # slice boundaries, which is why it is head+tail rather than a single
        # head window.
        cb = params["check_bytes"]
        for w, last in zip(workers, group_final_stores[0]):
            base = w["w_m"] * slice_bytes
            if cb <= 0 or cb >= slice_bytes:
                windows = [("", base, slice_bytes)]
            else:
                half = max(block_bytes_D, (cb // 2 // block_bytes_D) * block_bytes_D)
                windows = [("_head", base, half),
                           ("_tail", base + slice_bytes - half, half)]
            # The windows are CHAINED on the SAME host core (no fan-out):
            # last -> check_head -> check_tail. A plain linear chain, the same
            # shape as the rest of the DFG - so it cannot hit the presence-bit
            # fan-in trap.
            prev = last
            for tag, off, nbytes in windows:
                cn = BingoNode(
                    assigned_chiplet_id=platform["chiplet_ids"][0], assigned_cluster_id=0, assigned_core_id=host_core,
                    node_name=f"Host_Check_D_c{w['cluster']}{tag}",
                    kernel_name="__host_bingo_kernel_check_result",
                    kernel_args=HostBingoKernelCheckResultArgs(
                        name=f"D_c{w['cluster']}{tag}",
                        golden_data_addr=BingoMemSymbol("D", offset=off),
                        output_data_addr=(BingoMemAllocView(computed_D, off) if off else computed_D),
                        data_size=nbytes,
                    ),
                )
                dfg.bingo_add_node(cn)
                dfg.add_edge(prev, cn)
                prev = cn
    else:
        # grid_N>1: a worker owns a STRIDED sub-block of D, so it has no single
        # contiguous slice to check, and a single shared check node would
        # reintroduce the presence-bit fan-in deadlock above. Fail loudly here
        # rather than burn an hour discovering it in RTL.
        raise NotImplementedError(
            f"grid_N={params['grid_N']} (>1) with grid_K=1 is not supported yet: each "
            "worker's D region is strided, so per-worker Check_D nodes cannot cover it "
            "with one contiguous check, and one shared Check_D would deadlock on the "
            "presence-bit dep matrix. Use grid_N=1, or "
            "emit one check per contiguous D row-run."
        )

    return dfg

def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(args.cfg) as f:
        raw_cfg = hjson.loads(f.read())

    grid_M = raw_cfg.get("grid_M", 2)
    grid_K = raw_cfg.get("grid_K", 1)
    grid_N = raw_cfg.get("grid_N", 2)
    clusters_per_chip = raw_cfg.get("clusters_per_chip", 4)

    workers = build_worker_grid(grid_M, grid_K, grid_N, num_rows=1, num_cols=1,
                                 clusters_per_chip=clusters_per_chip)

    params, merged = define_workload_params(args.cfg, args.hwcfg)
    merged["num_clusters"] = merged.get("num_clusters", clusters_per_chip)

    if args.data_h is not None:
        with open(args.data_h, "w") as f:
            f.write(emit_header_file(**merged))

    platform = parse_platform_cfg(args.platformcfg)
    if not guard_cluster_count(merged, platform, args.output_dir, args.output_offload_file_name):
        return

    dfg = create_dfg(params, platform, workers)
    dfg.bingo_compile_dfg("GEMM_LargeScale", args.output_dir, args.output_offload_file_name,
                           extra_include_header_list=["gemm_data.h"])

if __name__ == "__main__":
    main()
