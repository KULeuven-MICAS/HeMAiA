#!/usr/bin/env python3
"""SHARED LAYOUT MODULE - used by BOTH gemm_parametric.py (the DFG)
and gemm_datagen.py (the header).

Two things live here, because both sides MUST arrive at the SAME answer:

  1. inner tile selection (_auto_inner_tiles) - splits a worker's share into
     pieces that fit in L1. datagen needs to know it too, because the LAYOUT
     of the result buffer is derived from the tile shape (see below).
  2. the golden-D verification plan (build_check_plan).

WHY SHARED: datagen no longer compiles the WHOLE golden D, only the pieces the
check nodes actually read (`D_check`). Both sides MUST produce the same run
list in the SAME ORDER - otherwise the check compares the wrong bytes and
silently "validates" a wrong result. Single source of truth = this file.

OLD BEHAVIOUR (before 2026-08-20): datagen emitted the entire golden as
`int32_t D[]` and that array was compiled into EVERY chip's L3. At large matrix
sizes this alone blew the L3 budget (32 MB for 2048x4096x4096, against a 16 MiB
spm_wide). The cost is now ~= check_bytes * number of chips, i.e. INDEPENDENT
of matrix size.

=====================================================================
THE RESULT BUFFER IS NOW TILE-PACKED (2026-08-20)
=====================================================================
partial_D used to be the worker's share in (M1_share, N1_share) block-row
order. In that layout an inner tile's write was STRIDED at the destination
(tile width inner_tile_N, row width N1_share), so it needed inner_tile_M nodes
per tile. Since the DFG is unrolled STATICALLY in Python, that meant a direct
node explosion:

    1024x4096x4096, 4x4x4 grid -> 1024 store nodes per worker

The new layout packs tiles back to back:

    tile (m_idx, n_idx) offset = (m_idx * n_tile_n + n_idx) * tm * tn * blkD
    INSIDE a tile              = (tm rows, tn blocks) row-major

so the WHOLE of l1_D is written by a single contiguous 1D copy -> 1 node per
tile (instead of 1024). The golden comparison has to know this layout, which is
why build_check_plan takes tm/tn. On every earlier rung M1_share == N1_share
== 1, hence tm == tn == 1, so the layout is IDENTICAL to the old one.
"""

# L1 heap capacity is ~516864 B/cluster (measured from an allocator failure).
# Leave margin.
# SHARED: the DFG and datagen must pick the same tile, so this constant cannot
# be duplicated in two places.
L1_BUDGET_BYTES = 400 * 1024

def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]

def _inner_tile_bytes(tm, tk, tn, block_bytes_A, block_bytes_B, meshRow, meshCol):
    return (tm * tk * block_bytes_A) + (tk * tn * block_bytes_B) + (tm * tn * meshRow * meshCol * 4)

def tile_node_count(tm, tk, tn, M1_share, K1_share, N1_share, K1_full=None):
    """Number of DFG nodes one worker will generate with this tile choice.

    Because the DFG is unrolled STATICALLY in Python, this number is directly
    compile time, memory and dep-tag pressure - THIS is the thing to minimise,
    not the tile COUNT.

    Per tile:
      Load_A : one contiguous chunk if tk == K1_share, else 1 per M row
               (the mem chip stores A/B in K-slab order, so the threshold is
               the worker's K share, not the full K1 - see emit_mempool_bin)
      Load_B : same condition (B is N-major, a full K run is contiguous inside
               a slab), else tn
      Gemm   : 1
      Store_D: ALWAYS 1 thanks to the tile-packed layout (outside the k loop,
               once per (m_idx, n_idx) pair)
    """
    n_tm, n_tk, n_tn = M1_share // tm, K1_share // tk, N1_share // tn
    full_k = (tk == K1_share)
    n_a = 1 if full_k else tm
    n_b = 1 if full_k else tn
    return n_tm * n_tn * (n_tk * (n_a + n_b + 1) + 1)

def _auto_inner_tiles(M1_share, K1_share, N1_share, block_bytes_A, block_bytes_B,
                      meshRow, meshCol, budget, K1_full=None):
    """Among all (tm, tk, tn) divisor triples that fit in L1, pick the one that
    minimises the DFG NODE COUNT.

    THE OLD POLICY shrank M first (down to 1 if needed), then N, then K last.
    Its reasoning was "shrinking M/N is free, shrinking K needs accumPrevC" -
    right about K, but driving M down to 1 exploded the tile count for no
    benefit: the term that dominates a tile is the B piece
    (tk * tn * block_bytes_B, blkB=512), which shrinking M does not reduce AT ALL.

    !! accumPrevC CONSTRAINT (MEASURED IN RTL) !!
    If tk < K1_share then K is split across several tiles and the k_idx>0 calls
    accumulate in L1 with accumPrevC=1. The accumPrevC accumulator is a SINGLE
    VersaCore REGISTER = ONE TILE, so it is only correct when M=1 and N=1 per
    call (see gemm_nsplit_accumprevc_4cluster/main_bingo.py line 18; the same
    constraint is what the "assert M1==1 and N1==1" in gemm_tiled_1cluster's
    datagen enforces). With tm>1 or tn>1, accumPrevC produces SILENTLY WRONG
    RESULTS - all four chips returned a D that did not match golden.

    So tk == K1_share is no longer a tie-break PREFERENCE but a candidate
    FILTER: either K stays whole, or tm == tn == 1. The divisor counts are
    small, so the exhaustive search is cheap.
    """
    def fits(tm, tk, tn):
        return _inner_tile_bytes(tm, tk, tn, block_bytes_A, block_bytes_B,
                                 meshRow, meshCol) <= budget

    best = None
    for tk in _divisors(K1_share):
        for tm in _divisors(M1_share):
            for tn in _divisors(N1_share):
                if not fits(tm, tk, tn):
                    continue
                # accumPrevC constraint: if K is split, one tile per call is mandatory.
                if tk != K1_share and not (tm == 1 and tn == 1):
                    continue
                nodes = tile_node_count(tm, tk, tn, M1_share, K1_share, N1_share)
                tiles = (M1_share // tm) * (K1_share // tk) * (N1_share // tn)
                key = (nodes, tiles, 0 if tk == K1_share else 1,
                       -_inner_tile_bytes(tm, tk, tn, block_bytes_A,
                                          block_bytes_B, meshRow, meshCol))
                if best is None or key < best[0]:
                    best = (key, (tm, tk, tn))

    if best is None:
        # No divisor triple fits -> return the smallest one and let the
        # caller's assert stop with an explicit error message.
        return 1, 1, 1
    return best[1]

def resolve_inner_tiles(cfg, M1_share, K1_share, N1_share,
                        block_bytes_A, block_bytes_B, meshRow, meshCol, budget):
    """Honour hand-written inner_tile_* values from cfg and auto-pick the rest.

    The DFG and datagen both call this so they get the SAME answer - two
    copy-pasted decisions would be free to drift apart.
    """
    tm = cfg.get("inner_tile_M")
    tk = cfg.get("inner_tile_K")
    tn = cfg.get("inner_tile_N")
    if tm is None or tk is None or tn is None:
        a_tm, a_tk, a_tn = _auto_inner_tiles(
            M1_share, K1_share, N1_share, block_bytes_A, block_bytes_B,
            meshRow, meshCol, budget, K1_full=cfg["K1"])
        tm = a_tm if tm is None else tm
        tk = a_tk if tk is None else tk
        tn = a_tn if tn is None else tn

    # Hand-written values are subject to the same constraint (the auto search
    # already filters for it). Without this assert such a config silently
    # produces WRONG RESULTS - it does not raise an error.
    if K1_share % tk == 0 and K1_share // tk > 1:
        assert tm == 1 and tn == 1, (
            f"accumPrevC constraint violated: inner_tile_K={tk} < K1_share={K1_share}, "
            f"so K is split into {K1_share // tk} tiles and the k_idx>0 calls use "
            f"accumPrevC=1. The accumPrevC accumulator is a ONE-TILE register, so "
            f"inner_tile_M and inner_tile_N MUST BE 1 (currently M={tm}, N={tn}).\n"
            f"  -> either set inner_tile_K equal to K1_share ({K1_share}) for the fast "
            f"path (one Load_A/Load_B per tile), or set inner_tile_M=inner_tile_N=1.\n"
            f"  Measured in RTL: D mismatched golden on all 4 chips."
        )
    return tm, tk, tn

def build_check_plan(M1, N1, grid_M, grid_N, block_bytes_D, check_bytes, tm, tn):
    """Build the verification plan in canonical worker order (w_m outer, w_n
    inner - the same order as the `workers` list).

    Returns: (plan, total_bytes)
      plan[(w_m, w_n)] = [(golden_off, computed_off, nbytes, compact_off), ...]
        golden_off   : byte offset within the FULL golden D (used by datagen)
        computed_off : offset within that chip's TILE-PACKED result buffer
        compact_off  : byte offset within the emitted `D_check` array
      total_bytes : total size of D_check
    """
    M1s, N1s = M1 // grid_M, N1 // grid_N
    n_tile_n = N1s // tn
    tile_bytes = tm * tn * block_bytes_D
    run_bytes = tn * block_bytes_D
    cb = check_bytes

    plan = {}
    compact = 0
    for w_m in range(grid_M):
        for w_n in range(grid_N):
            # Smallest unit contiguous on both sides: tn blocks for a fixed
            # (m_l, n_idx). In golden these are consecutive N blocks, and in the
            # packed buffer they are consecutive too -> contiguous either way.
            runs = []
            for m_l in range(M1s):
                m_idx, r = divmod(m_l, tm)
                for n_idx in range(n_tile_n):
                    goff = (((w_m * M1s + m_l) * N1)
                            + (w_n * N1s + n_idx * tn)) * block_bytes_D
                    coff = (m_idx * n_tile_n + n_idx) * tile_bytes + r * run_bytes
                    runs.append((goff, coff, run_bytes))

            # Merge adjacent runs, but only those consecutive on BOTH sides.
            # This reproduces the old "grid_N == 1 -> a single run" behaviour
            # automatically (and gives a byte-identical run list on every
            # earlier rung, where tm == tn == 1).
            merged = []
            for (g, c, n) in runs:
                if merged and merged[-1][0] + merged[-1][2] == g and merged[-1][1] + merged[-1][2] == c:
                    pg, pc, pn = merged[-1]
                    merged[-1] = (pg, pc, pn + n)
                else:
                    merged.append((g, c, n))
            runs = merged

            # check_bytes budget: trim the NUMBER of runs (head + tail) rather
            # than cutting mid-byte - offset bugs show up most at the two ends
            # of the sub-block.
            total = sum(n for (_g, _c, n) in runs)
            if cb > 0 and total > cb:
                head, tail, acc = [], [], 0
                i, j = 0, len(runs) - 1
                while i <= j:
                    if len(head) <= len(tail):
                        nxt = runs[i]
                        if acc + nxt[2] > cb and (head or tail):
                            break
                        head.append(nxt); acc += nxt[2]; i += 1
                    else:
                        nxt = runs[j]
                        if acc + nxt[2] > cb and (head or tail):
                            break
                        tail.insert(0, nxt); acc += nxt[2]; j -= 1
                runs = head + tail if (head or tail) else runs[:1]

            out = []
            for (goff, coff, nbytes) in runs:
                out.append((goff, coff, nbytes, compact))
                compact += nbytes
            plan[(w_m, w_n)] = out

    return plan, compact
