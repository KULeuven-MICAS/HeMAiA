# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Deriving the strided xDMA transfer that performs a layout change -- and checking it.

A layout is a permutation of the same flat length, so a conversion that is wrong moves
every byte, writes every byte, and produces a scrambled tensor that no byte count catches
and that PASSES on random test data, because the golden is scrambled identically. That is
why nothing here is trusted: `convert_args` derives the strides and then walks them in
numpy against the ground-truth index maps of BOTH layouts, at graph-build time, on the real
bounds. The derivation is checked by something that does not share its arithmetic.

The generic derivation reproduces `d_to_a_args` below -- the hand-checked nest the MoE
workload validated on RTL -- byte-pair for byte-pair at every shape tried.
"""

import numpy as np

from bingo_kernel_args import SnaxBingoKernelXdma6dArgs


# THE TWO HARDWARE NUMBERS, and they are different things that are both 8.
#   _ATOM  bytes each lane moves in one beat. The innermost run must be contiguous on BOTH
#          sides and exactly this long. A shorter one falls off the hardware path onto the
#          CPU fallback -- which is why an A-layout conversion works at fp16, where the
#          tileSize run is 4*2 = 8 B, and not at int8, where it is 4 B.
#   _LANES how many of those beats move at once. That count is the transfer's `spatial`
#          dimension, so the spatial bound is always exactly this.
_ATOM = 8
_LANES = 8


# ======================================================================================
# What each layout does with an (row, col) element
# ======================================================================================
# The ground truth, as an index map. Written out rather than derived so that the stride
# derivation below has something INDEPENDENT to be checked against -- a model that shared
# its arithmetic with the thing it verifies would confirm its own mistakes.

def index_map(layout: str, rows: int, cols: int, mesh: tuple) -> np.ndarray:
    """(rows, cols) -> the element's position in `layout`'s flat buffer."""
    mr, ts, mc = mesh
    r_i = np.arange(rows)[:, None]
    c_i = np.arange(cols)[None, :]
    if layout == "row_major":
        return np.broadcast_to(r_i * cols + c_i, (rows, cols)).copy()
    if layout == "col_major":
        # THE TRANSPOSE, AS AN INDEX MAP LIKE EVERY OTHER LAYOUT. This is the whole of the
        # orientation axis: a [rows, cols] tensor whose bytes are laid out [cols, rows].
        # It used to be a boolean beside the enum, which meant byte order was described in
        # two languages -- a verified permutation for the blocking and an ad-hoc flag for
        # the orientation -- and every place they met needed hand-written reconciliation.
        # One map removes all of it.
        return np.broadcast_to(c_i * rows + r_i, (rows, cols)).copy()
    if layout == "A":
        # (m, k, r, s): row = m*meshRow + r, col = k*tileSize + s
        m, r = divmod(r_i, mr)
        k, s = divmod(c_i, ts)
        return ((m * (cols // ts) + k) * mr + r) * ts + s
    if layout == "B":
        # (n, k, c, s): col = n*meshCol + c, row = k*tileSize + s
        k, s = divmod(r_i, ts)
        n, c = divmod(c_i, mc)
        return ((n * (rows // ts) + k) * mc + c) * ts + s
    if layout == "D":
        # (m, n, r, c): row = m*meshRow + r, col = n*meshCol + c
        m, r = divmod(r_i, mr)
        n, c = divmod(c_i, mc)
        return ((m * (cols // mc) + n) * mr + r) * mc + c
    raise ValueError(
        f"no index map for layout {layout!r}. 'd32' is deliberately absent: it is the "
        f"D-port INT32 SCATTER, a different bijection from 'D', and it is not a transport "
        f"source -- see the un-permute in the layer.")


def _contig_axis(layout: str, rows: int, cols: int, mesh: tuple) -> str:
    """Which way this layout runs contiguously: along a row, or down a column.

    Two layouts that disagree here differ by a transpose, which no pair of strides can
    express -- the xDMA has dedicated transposer kernels for that, on a different path.
    """
    # The FIRST step in each direction, not every step. A blocked layout is contiguous
    # only WITHIN its block -- B's run of tileSize elements down a column restarts at each
    # block boundary -- so demanding contiguity everywhere reports "neither" for every
    # layout that is not plain row-major, and the transpose then hides behind a message
    # about precision.
    mp = index_map(layout, rows, cols, mesh)
    if cols > 1 and mp[0, 1] - mp[0, 0] == 1:
        return "row"
    if rows > 1 and mp[1, 0] - mp[0, 0] == 1:
        return "column"
    return "neither"


def _divides(layout: str, rows: int, cols: int, mesh: tuple) -> None:
    """Refuse a tensor the layout cannot tile. A partial tile is not a layout error the
    hardware reports; it writes a short buffer and leaves the tail uninitialised."""
    mr, ts, mc = mesh
    need = {"A": (mr, ts), "B": (ts, mc), "D": (mr, mc),
            "row_major": (1, 1), "col_major": (1, 1)}[layout]
    if rows % need[0] or cols % need[1]:
        raise ValueError(
            f"a {rows}x{cols} tensor does not tile layout {layout!r} on a {mesh} array: "
            f"it needs rows % {need[0]} == 0 and cols % {need[1]} == 0. A partial tile "
            f"leaves the tail of the destination unwritten, and unwritten TCDM reads X.")


# ======================================================================================
# Deriving the transfer
# ======================================================================================

def _axes(rows: int, cols: int, mesh: tuple):
    """The finest factorisation of (row, col) that EVERY block layout is affine in.

    Row splits by meshRow then tileSize, column by meshCol then tileSize. That refines all
    of A, B, D and packed at once: A blocks the column by tileSize and D by meshCol, and
    meshCol is a multiple of tileSize, so a three-way column split serves both. Bounds of
    1 collapse away afterwards.
    """
    mr, ts, mc = mesh
    out = []
    for extent, blocks in ((rows, (mr, ts)), (cols, (mc, ts))):
        big, small = blocks
        if extent % big == 0 and big % small == 0:
            out.append((extent // big, big // small, small))
        elif extent % small == 0:
            out.append((extent // small, 1, small))
        else:
            out.append((extent, 1, 1))
    (rh, rm, rl), (ch, cm, cl) = out
    return ["r_hi", "r_mid", "r_lo", "c_hi", "c_mid", "c_lo"], [rh, rm, rl, ch, cm, cl]


def _strides(layout, rows, cols, mesh, bounds):
    """Elements between consecutive positions along each axis, or None where not constant.

    MEASURED ON THE RESHAPED MAP, one axis at a time. Taking the difference on the flat
    (row, col) map instead compares pairs that straddle a block boundary, where the step
    is genuinely different -- which reads as "this axis has no constant stride" for every
    blocked layout and refuses conversions that are perfectly expressible.
    """
    mp = index_map(layout, rows, cols, mesh).reshape(*bounds)
    out = []
    for d, b in enumerate(bounds):
        if b == 1:
            out.append(0)
            continue
        lo = mp.take(indices=range(b - 1), axis=d)
        hi = mp.take(indices=range(1, b), axis=d)
        u = np.unique(hi - lo)
        out.append(int(u[0]) if u.size == 1 else None)
    return out


def convert_args(src_layout, dst_layout, rows, cols, mesh, elem_bytes, src, dst):
    """One xDMA pass that reads `src_layout` and writes `dst_layout`. Verified.

    Returns SnaxBingoKernelXdma6dArgs. Raises if the conversion is not expressible as a
    strided nest within the kernel's 1 spatial + 5 temporal dimensions, rather than
    emitting a nest that moves the right byte count to the wrong offsets.
    """
    for lay in (src_layout, dst_layout):
        _divides(lay, rows, cols, mesh)
    eb = int(elem_bytes)

    names, bounds = _axes(rows, cols, mesh)
    ss_all = _strides(src_layout, rows, cols, mesh, bounds)
    ds_all = _strides(dst_layout, rows, cols, mesh, bounds)
    axes = []
    for nm, b, ss, ds in zip(names, bounds, ss_all, ds_all):
        if b == 1:
            continue
        if ss is None or ds is None:
            raise ValueError(
                f"{src_layout}->{dst_layout}: axis {nm} has no constant stride on "
                f"{'the source' if ss is None else 'the destination'}, so the conversion "
                f"is not a strided nest. It needs a gather kernel or a host repack.")
        axes.append((nm, b, ss * eb, ds * eb))

    # THE ATOM COMES OFF FIRST. The innermost axes that run contiguously on both sides are
    # not nest dimensions at all -- they are the bytes inside one beat. Peeling them here
    # is what makes the remaining nest addressable in beats, the unit the kernel's strides
    # are in.
    #
    # WHICH axis is innermost is SEARCHED FOR, not assumed to be last. A-layout's innermost
    # index is a column (s within a tileSize run) but B-layout's is a ROW -- B is
    # (n, k, c, s) with s = row % tileSize -- so a peel that always took the last axis
    # would find no contiguous run for B and refuse every B conversion.
    run, nest = 1, list(axes)
    while run * eb < _ATOM:
        want = run * eb
        hit = next((i for i, (_, _, ss, ds) in enumerate(nest)
                    if ss == want and ds == want), None)
        if hit is None:
            break
        nm, b, ss, ds = nest.pop(hit)
        if run * b * eb > _ATOM:
            # the run is longer than one beat: keep the excess as a nest axis
            keep = (run * b * eb) // _ATOM
            nest.append((f"{nm}/beat", keep, _ATOM, _ATOM))
            run = _ATOM // eb
            break
        run *= b
    atom = run * eb
    if atom != _ATOM:
        # WHY it is short decides what the caller should do, so separate the two causes.
        # If the two layouts run contiguously along DIFFERENT axes -- one down rows, the
        # other across columns -- no strided nest can help: that is a transpose, and the
        # xDMA has separate transposer kernels for it. Otherwise the run is simply too
        # narrow at this precision.
        src_dir = _contig_axis(src_layout, rows, cols, mesh)
        dst_dir = _contig_axis(dst_layout, rows, cols, mesh)
        if src_dir != dst_dir:
            raise ValueError(
                f"{src_layout}->{dst_layout} is a TRANSPOSE, not a reshape: {src_layout} "
                f"runs contiguously along {src_dir}s and {dst_layout} along {dst_dir}s, so "
                f"no pair of strides makes a common run.\n"
                f"A TRANSPOSE IS A KERNEL, NOT A NEST, and which kernel depends on the "
                f"pair. row_major <-> col_major is the plain axis exchange a per-row "
                f"reduction wants, and comm.transfer routes it to the 8x8 block "
                f"transposer automatically -- so a conversion BETWEEN a blocked layout "
                f"and col_major is planned as two steps through row_major and needs "
                f"nothing from you. This message is the other case: a transpose between "
                f"two BLOCKED layouts.\n"
                f"Use a dedicated transposer kernel instead -- "
                f"__snax_bingo_kernel_xdma_row_major_to_b and friends, via "
                f"kernels/kernel_layout.py. They take the array shape as RUNTIME ARGS, so "
                f"one kernel covers every tiling including this one; the old per-shape "
                f"wrappers (K2N32, K16N32, K8N16) are gone.\n"
                f"What to expect on a (16, 4, 16) array: the transposer's FAST path needs "
                f"tileSize % 8 == 0 and meshCol % 8 == 0 at elem_bytes == 1, because the "
                f"unit is an 8x8 BYTE-granular block transposer. tileSize=4 misses it, so "
                f"the conversion is correct but takes the kernel's CPU loop. That is a "
                f"speed problem, not a correctness one -- and not one software can fix "
                f"here, since within a B-tile the (k, s) indices are not 8-strided and no "
                f"pair of k-tiles composes into an 8-deep block.")
        raise ValueError(
            f"{src_layout}->{dst_layout} at {eb} B/element: the innermost run common to "
            f"both layouts is {atom} B, but the xDMA moves {_ATOM} B per lane. A shorter "
            f"run falls off the hardware path onto the CPU fallback. Convert at a wider "
            f"precision -- an A-layout reshape is an fp16 operation, so quantise AFTER "
            f"the reshape rather than before it.")
    if not nest:
        raise ValueError(f"{src_layout}->{dst_layout}: nothing left to transfer after the "
                         f"{_ATOM} B atom; the two layouts are identical at this shape.")

    # Order the survivors by how they sit in the SOURCE, innermost first, so the collapse
    # below sees genuinely adjacent loops. The list order out of _axes is the (row, col)
    # factorisation, which is not memory order for every layout.
    nest.sort(key=lambda t: t[2])

    # Collapse axes that compose, innermost first: two nested loops whose outer stride is
    # exactly inner_stride * inner_bound ON BOTH SIDES are one loop of the product. This
    # is what brings the factorisation inside the kernel's 6 dimensions, and it is exact.
    merged = []
    for nm, b, ss, ds in nest:
        if merged:
            pnm, pb, pss, pds = merged[-1]
            if ss == pss * pb and ds == pds * pb:
                merged[-1] = (f"{pnm}+{nm}", pb * b, pss, pds)
                continue
        merged.append((nm, b, ss, ds))

    # THE LANE DIMENSION IS CHOSEN, NOT TAKEN. The 8 lanes need an axis whose bound is a
    # multiple of 8 and whose stride is constant on both sides -- and that is often NOT the
    # innermost one. For D->A the innermost surviving axis is the tileSize-group with bound
    # 4; the lanes go on `r` (bound meshRow), which is exactly what the hand-written nest
    # in layout.py does. Taking the innermost blindly refuses a conversion the hardware
    # performs. Temporal order is free -- a nest visits the same set whatever the order --
    # so lifting one axis out of the middle costs nothing.
    lane = next((i for i, (_, b, _, _) in enumerate(merged) if b % _LANES == 0), None)
    if lane is None:
        raise ValueError(
            f"{src_layout}->{dst_layout} at {rows}x{cols}: no axis has a bound that is a "
            f"multiple of the xDMA's {_LANES} lanes "
            f"(bounds {[b for _, b, _, _ in merged]}), so the transfer cannot fill them.")
    sp_name, sp_bound, sp_ss, sp_ds = merged[lane]
    temporal = [t for i, t in enumerate(merged) if i != lane]
    if sp_bound > _LANES:
        temporal.append((f"{sp_name}/lane", sp_bound // _LANES,
                         sp_ss * _LANES, sp_ds * _LANES))
        sp_bound = _LANES
    if len(temporal) > 5:
        raise ValueError(
            f"{src_layout}->{dst_layout} at {rows}x{cols} needs {len(temporal)} temporal "
            f"dimensions; __snax_bingo_kernel_xdma_6d has 5.")

    args = SnaxBingoKernelXdma6dArgs(
        src, dst,
        spatial_stride_src=sp_ss, spatial_stride_dst=sp_ds,
        temporal_strides_src=[t[2] for t in temporal] or [0],
        temporal_bounds_src=[t[1] for t in temporal] or [1],
        temporal_strides_dst=[t[3] for t in temporal] or [0],
        temporal_bounds_dst=[t[1] for t in temporal] or [1])
    _verify_nest(args, src_layout, dst_layout, rows, cols, mesh, eb,
                 sp_bound, [t[1] for t in temporal], atom // eb)
    return args


def _verify_nest(args, src_layout, dst_layout, rows, cols, mesh, eb, sp_bound,
                 t_bounds, atom_elems):
    """Walk the emitted nest and compare it against both index maps, element by element.

    This is the check that makes the derivation above safe to change. A nest that is wrong
    by one dimension still moves the right number of bytes -- it just moves them to the
    wrong offsets -- so a byte count cannot catch it and a simulation would only show a
    scrambled tensor several kernels downstream.
    """
    src_off, dst_off = [], []
    idx = [0] * len(t_bounds)
    total = int(np.prod(t_bounds)) if t_bounds else 1
    for _ in range(total):
        base_s = sum(i * s for i, s in zip(idx, args.temporal_strides_src[:len(t_bounds)]))
        base_d = sum(i * s for i, s in zip(idx, args.temporal_strides_dst[:len(t_bounds)]))
        for lane in range(sp_bound):
            s0 = base_s + lane * args.spatial_stride_src
            d0 = base_d + lane * args.spatial_stride_dst
            # each lane moves `atom_elems` CONTIGUOUS elements on both sides
            for e in range(atom_elems):
                src_off.append(s0 + e * eb)
                dst_off.append(d0 + e * eb)
        for d in reversed(range(len(t_bounds))):
            idx[d] += 1
            if idx[d] < t_bounds[d]:
                break
            idx[d] = 0

    n = rows * cols
    if len(src_off) != n:
        raise ValueError(
            f"{src_layout}->{dst_layout}: the nest visits {len(src_off)} elements but the "
            f"tensor has {n}. Whatever it does not write stays uninitialised, and "
            f"uninitialised TCDM reads X.")

    # The nest pairs (src byte offset -> dst byte offset). Reading the source map at each
    # source offset must give the same ELEMENT that the destination map places at the
    # paired destination offset.
    s_at = np.empty(n, dtype=np.int64)
    d_at = np.empty(n, dtype=np.int64)
    s_at[(index_map(src_layout, rows, cols, mesh)).reshape(-1)] = np.arange(n)
    d_at[(index_map(dst_layout, rows, cols, mesh)).reshape(-1)] = np.arange(n)
    so = np.asarray(src_off) // eb
    do = np.asarray(dst_off) // eb
    if so.max() >= n or do.max() >= n or so.min() < 0 or do.min() < 0:
        raise ValueError(f"{src_layout}->{dst_layout}: the nest addresses outside the "
                         f"tensor (src max {so.max()}, dst max {do.max()}, size {n}).")
    bad = int(np.count_nonzero(s_at[so] != d_at[do]))
    if bad:
        raise ValueError(
            f"{src_layout}->{dst_layout} at {rows}x{cols}: the derived nest misplaces "
            f"{bad} of {n} elements. It moves the right number of bytes to the wrong "
            f"offsets, which no byte count and no golden on random data would catch.")


# ======================================================================================
# The hand-written D -> A nest, kept
# ======================================================================================
# convert_args derives this one generically and agrees with it. It stays because it is the
# RTL-validated reference the generic path is checked against, and because MoE calls it.

def d_to_a_args(mesh, M_T, N_up, K_down, src, dst) -> SnaxBingoKernelXdma6dArgs:
    """D-layout (m, n, r, c) fp16 -> A-layout (m, k, r, s) fp16, on the xDMA AGU.

    The hidden index is n*meshCol + c on one side and k*tileSize + s on the other, so
    writing c = j*tileSize + s gives k = n*(meshCol/tileSize) + j and the innermost
    tileSize elements are contiguous on BOTH sides. That run is tileSize*2 = 8 bytes,
    which is the xDMA's own lane width -- the same 8-byte beat every xdma_layout_* kernel
    moves. In int8 it would be 4 bytes and fall off the hardware path onto the CPU
    fallback, which is why the reshape happens BEFORE the quantisation and not after.

    The 8 spatial lanes walk `r`, whose stride is constant on both sides.
    """
    mr, ts, mc = mesh
    eb = 2
    j_bound = mc // ts
    lanes = 8
    if mr % lanes:
        raise ValueError(f"meshRow={mr} is not a multiple of the xDMA's {lanes} lanes.")
    src_r, dst_r = mc * eb, ts * eb                 # one step of r
    # The M dimension is NOT optional, even though M_T == 1 hides it. D-layout is
    # (m, n, r, c) and A-layout is (m, k, r, s), so one step of m is a whole (n, r, c)
    # block on the source and a whole (k, r, s) block on the destination. Leaving it out
    # transfers only the FIRST m tile: at M_T=2 exactly half of `act_a` is never written,
    # stays uninitialised TCDM (= X), and the X reaches the host through quant -> down
    # GEMM -> push. At M_T=1 the bounds are identical either way and everything passes,
    # so the mistake is invisible until the shape grows.
    src_m = N_up * mr * mc * eb
    dst_m = K_down * mr * ts * eb

    # A layout conversion that does not move every byte leaves the rest of the destination
    # uninitialised, and uninitialised TCDM is X, which never fails a check -- it kills the
    # host instead. So count the bytes here rather than discovering it in a simulation.
    moved = lanes * (ts * eb) * (mr // lanes) * j_bound * N_up * M_T
    want = M_T * N_up * mr * mc * eb
    if moved != want:
        raise ValueError(
            f"reshape would move {moved} B but the tensor is {want} B. Every element of "
            f"the destination must be written; the shortfall stays uninitialised (X) and "
            f"reaches the host through quant -> down GEMM -> push. Check the temporal "
            f"dimension list against the layouts.")

    return SnaxBingoKernelXdma6dArgs(
        src, dst,
        spatial_stride_src=src_r, spatial_stride_dst=dst_r,
        #                r-group,           j,             n,             m
        temporal_strides_src=[lanes * src_r, ts * eb,       mr * mc * eb,  src_m],
        temporal_bounds_src=[mr // lanes,    j_bound,       N_up,      M_T],
        temporal_strides_dst=[lanes * dst_r, mr * ts * eb,  j_bound * mr * ts * eb, dst_m],
        temporal_bounds_dst=[mr // lanes,    j_bound,       N_up,      M_T])


