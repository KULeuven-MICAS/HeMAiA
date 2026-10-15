# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The MoE prefill's kernels (device: offload_hw_kernels/moe_prefill.h): the route of every
token as a table, each chip's row lists, the row gather, and the SIMD passes over D-layout row
tiles of array shape 3 (blocks of 8 rows x 16 FP16, 256 B)."""

from typing import Dict, Sequence

from bingo_mem_handle import BingoMemAlloc
from kernel_base import BingoKernelArgs

D_BLOCK = 256                 # one D-layout block of array shape 3: 8 rows x 16 FP16
ROUTE_ENTRY_BYTES = 32        # one token's entry of the route table
ROUTE_W_OFFSET = 8            # its FP16 weights, after 8 id bytes


class SnaxBingoKernelMoeTopkRowsArgs(BingoKernelArgs):
    """The top k of each token's n FP16 probabilities (tokens [t0, t1), token t's at p_addr +
    t * p_pitch), largest first, ties to the lower index: token t's 32-B entry of the route
    table at out_addr + t * 32 (k uint8 ids, 0xFF past k; then k FP16 weights at byte 8). Any
    core; every address in its cluster's L1. With ctr_addr (a word zero before the first of them
    runs), several tasks share [t0, t1), each taking `chunk` tokens at a time."""

    KERNEL_NAME = "__snax_bingo_kernel_moe_topk_rows"
    MAX_N, MAX_K = 64, 8

    def __init__(self, p_addr, p_pitch: int, n: int, k: int, t0: int, t1: int, out_addr,
                 ctr_addr=0, chunk: int = 1):
        if not 0 < n <= self.MAX_N or not 0 < k <= min(n, self.MAX_K) or not 0 <= t0 <= t1:
            raise ValueError(f"n={n} (<= {self.MAX_N}), k={k} (<= {self.MAX_K}), tokens [{t0}, {t1})")
        if p_pitch < 2 * n or p_pitch % 2:
            raise ValueError(f"p_pitch={p_pitch}: past a row of {n} FP16")
        if int(chunk) < 1:
            raise ValueError(f"chunk={chunk}: at least one token")
        self.p_addr, self.p_pitch, self.n, self.k = p_addr, int(p_pitch), int(n), int(k)
        self.t0, self.t1, self.out_addr = int(t0), int(t1), out_addr
        self.ctr_addr, self.chunk = ctr_addr, int(chunk)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_moe_topk_rows_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.p_addr, "p_addr", a, handle_name_map, split_64bit=False)
        a["p_pitch"] = str(self.p_pitch)
        a["n"] = str(self.n)
        a["k"] = str(self.k)
        a["t0"] = str(self.t0)
        a["t1"] = str(self.t1)
        self._process_addr(self.out_addr, "out_addr", a, handle_name_map, split_64bit=False)
        self._process_addr(self.ctr_addr, "ctr_addr", a, handle_name_map, split_64bit=False)
        a["chunk"] = str(self.chunk)
        return a


class SnaxBingoKernelMoeDispatchArgs(BingoKernelArgs):
    """The route table -> the row lists of a chip's expert slots: slot s (expert ids[s]) takes
    4 * caps[s] bytes from list_addr, slots back to back: caps[s] uint16 token rows, then caps[s]
    FP16 weights; rows past its tokens are pad_row with weight 0. cnts[s] is the number of
    tokens the graph expects the slot to get; another count fails the task. Any core; every
    address in its cluster's L1. With state_addr, the task does tokens [t0, t1) only: the
    ranges of one set of lists run in order on one core, the counts kept at state_addr."""

    KERNEL_NAME = "__snax_bingo_kernel_moe_dispatch"
    MAX_SLOTS = 4

    def __init__(self, table_addr, tokens: int, k: int, ids: Sequence[int], list_addr,
                 caps: Sequence[int], cnts: Sequence[int], pad_row: int, err_addr=0,
                 t0: int = 0, t1: int = None, state_addr=0):
        self.t0, self.t1, self.state_addr = int(t0), int(tokens if t1 is None else t1), state_addr
        if not 0 <= self.t0 <= self.t1 <= int(tokens):
            raise ValueError(f"tokens [{t0}, {t1}) of {tokens}")
        ids, caps, cnts = list(ids), list(caps), list(cnts)
        if not 0 < len(ids) <= self.MAX_SLOTS or len(caps) != len(ids) or len(cnts) != len(ids):
            raise ValueError(f"{len(ids)} slots ({self.MAX_SLOTS} at most), one cap and count each")
        if any(not 0 <= e < 256 for e in ids) or any(c < n for c, n in zip(caps, cnts)):
            raise ValueError(f"expert ids {ids} (bytes), capacities {caps} >= counts {cnts}")
        self.table_addr, self.tokens, self.k = table_addr, int(tokens), int(k)
        self.ids, self.caps, self.cnts = ids, caps, cnts
        self.list_addr, self.pad_row, self.err_addr = list_addr, int(pad_row), err_addr

    @staticmethod
    def list_bytes(caps: Sequence[int]) -> int:
        return 4 * sum(int(c) for c in caps)

    @staticmethod
    def rows_offset(caps: Sequence[int], s: int) -> int:
        """Bytes from list_addr to slot s's token rows."""
        return 4 * sum(int(c) for c in list(caps)[:s])

    @staticmethod
    def w_offset(caps: Sequence[int], s: int) -> int:
        """Bytes from list_addr to slot s's weights."""
        return 4 * sum(int(c) for c in list(caps)[:s]) + 2 * int(list(caps)[s])

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_moe_dispatch_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.table_addr, "table_addr", a, handle_name_map, split_64bit=False)
        a["tokens"] = str(self.tokens)
        a["k"] = str(self.k)
        a["n_slots"] = str(len(self.ids))
        a["expert_ids"] = f"0x{sum(int(e) << (8 * s) for s, e in enumerate(self.ids)):08x}u"
        self._process_addr(self.list_addr, "list_addr", a, handle_name_map, split_64bit=False)
        for s in range(self.MAX_SLOTS):
            a[f"cap[{s}]"] = str(self.caps[s] if s < len(self.caps) else 0)
            a[f"cnt[{s}]"] = str(self.cnts[s] if s < len(self.cnts) else 0)
        a["pad_row"] = str(self.pad_row)
        a["t0"] = str(self.t0)
        a["t1"] = str(self.t1)
        self._process_addr(self.state_addr, "state_addr", a, handle_name_map, split_64bit=False)
        if self.err_addr:
            self._process_addr(self.err_addr, "err_addr", a, handle_name_map, split_64bit=False)
        else:
            a["err_addr"] = "0"
        return a


class SnaxBingoKernelIdmaGatherRowsArgs(BingoKernelArgs):
    """Rows ids[i] (uint16, this cluster's L1) of the matrix at src (row r at src + r *
    row_bytes; any memory the iDMA reaches) into dst + i * dst_pitch, i < n. DM core."""

    KERNEL_NAME = "__snax_bingo_kernel_idma_gather_rows"

    def __init__(self, src_addr, row_bytes: int, ids_addr, n: int, dst_addr, dst_pitch: int):
        if row_bytes <= 0 or n <= 0 or dst_pitch < row_bytes:
            raise ValueError(f"row_bytes={row_bytes}, n={n}, dst_pitch={dst_pitch} >= a row")
        self.src_addr, self.row_bytes, self.ids_addr, self.n = src_addr, int(row_bytes), ids_addr, int(n)
        self.dst_addr, self.dst_pitch = dst_addr, int(dst_pitch)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_idma_gather_rows_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        a["row_bytes"] = str(self.row_bytes)
        self._process_addr(self.ids_addr, "ids_addr", a, handle_name_map, split_64bit=False)
        a["n"] = str(self.n)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["dst_pitch"] = str(self.dst_pitch)
        return a


class SnaxBingoKernelSimdMulTilesArgs(BingoKernelArgs):
    """x (D layout: `tiles` row tiles of nt column tiles, x_tile_stride bytes apart) times the
    one-row-tile scale tile s (nt blocks), elementwise FP16: row tile m into dst + m *
    dst_tile_stride, or with dst_pitch into a row-major matrix of that pitch. dst may be x.
    SIMD core."""

    KERNEL_NAME = "__snax_bingo_kernel_simd_mul_tiles"

    def __init__(self, x_addr, s_addr, dst_addr, tiles: int, nt: int, x_tile_stride: int = 0,
                 dst_tile_stride: int = 0, dst_pitch: int = 0):
        if tiles <= 0 or nt <= 0 or (dst_pitch and dst_pitch % 32):
            raise ValueError(f"tiles={tiles}, nt={nt}, dst_pitch={dst_pitch} (beat-aligned)")
        self.x_addr, self.s_addr, self.dst_addr = x_addr, s_addr, dst_addr
        self.tiles, self.nt = int(tiles), int(nt)
        self.x_tile_stride = int(x_tile_stride) or nt * D_BLOCK
        self.dst_tile_stride = int(dst_tile_stride) or nt * D_BLOCK
        self.dst_pitch = int(dst_pitch)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_simd_mul_tiles_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.x_addr, "x_addr", a, handle_name_map, split_64bit=False)
        self._process_addr(self.s_addr, "s_addr", a, handle_name_map, split_64bit=False)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map, split_64bit=False)
        a["tiles"] = str(self.tiles)
        a["tile_beats"] = str(8 * self.nt)
        a["x_tile_stride"] = str(self.x_tile_stride)
        a["dst_tile_stride"] = str(self.dst_tile_stride)
        a["dst_pitch"] = str(self.dst_pitch)
        return a


class SnaxBingoKernelSimdDeqSwigluArgs(BingoKernelArgs):
    """A gate|up GEMM output g ([tiles][2 nb][8][16] FP16: each row tile's nb gate blocks, then
    its nb up blocks) times its scale tile s (2 nb blocks) in place, then silu(gate) * up of
    each row tile into out + m * out_tile_stride (nb blocks). sg: a scratch as large as g. SIMD
    core."""

    KERNEL_NAME = "__snax_bingo_kernel_simd_deq_swiglu"

    def __init__(self, g_addr, s_addr, sg_addr, out_addr, tiles: int, nb: int, out_tile_stride: int):
        if tiles <= 0 or nb <= 0 or out_tile_stride < nb * D_BLOCK:
            raise ValueError(f"tiles={tiles}, nb={nb}, out_tile_stride={out_tile_stride} >= nb blocks")
        self.g_addr, self.s_addr, self.sg_addr, self.out_addr = g_addr, s_addr, sg_addr, out_addr
        self.tiles, self.nb, self.out_tile_stride = int(tiles), int(nb), int(out_tile_stride)

    @staticmethod
    def scratch_bytes(tiles: int, nb: int) -> int:
        return int(tiles) * 2 * int(nb) * D_BLOCK

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_simd_deq_swiglu_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        for nm in ("g_addr", "s_addr", "sg_addr", "out_addr"):
            self._process_addr(getattr(self, nm), nm, a, handle_name_map, split_64bit=False)
        a["tiles"] = str(self.tiles)
        a["nb"] = str(self.nb)
        a["out_tile_stride"] = str(self.out_tile_stride)
        return a


class SnaxBingoKernelSimdQuantRowsArgs(BingoKernelArgs):
    """Rows of a D-layout A16 (nb column tiles; row tiles src_tile_stride apart), each times its
    FP16 weight (w_addr: one per row; 0: 1), quantised with the static inv scale into row-major
    INT8 rows at dst + i * dst_pitch. An odd nb reads one beat past the row and writes 16 values
    past it. SIMD core."""

    KERNEL_NAME = "__snax_bingo_kernel_simd_quant_rows"

    def __init__(self, src_addr, rows: int, nb: int, src_tile_stride: int, inv_scale_f32bits: int,
                 dst_addr, dst_pitch: int, w_addr=0):
        if rows <= 0 or nb <= 0 or dst_pitch % 32 or dst_pitch < 16 * (nb + (nb & 1)):
            raise ValueError(f"rows={rows}, nb={nb}, dst_pitch={dst_pitch} (beat-aligned, >= a row)")
        self.src_addr, self.rows, self.nb, self.src_tile_stride = src_addr, int(rows), int(nb), int(src_tile_stride)
        self.inv_scale_f32bits, self.dst_addr, self.dst_pitch = int(inv_scale_f32bits), dst_addr, int(dst_pitch)
        self.w_addr = w_addr

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_simd_quant_rows_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map, split_64bit=False)
        a["rows"] = str(self.rows)
        a["nb"] = str(self.nb)
        a["src_tile_stride"] = str(self.src_tile_stride)
        if self.w_addr:
            self._process_addr(self.w_addr, "w_addr", a, handle_name_map, split_64bit=False)
        else:
            a["w_addr"] = "0"
        a["inv_scale"] = f"0x{self.inv_scale_f32bits & 0xFFFFFFFF:08x}u"
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map, split_64bit=False)
        a["dst_pitch"] = str(self.dst_pitch)
        return a


class SnaxBingoKernelSimdScatterAddRowsArgs(BingoKernelArgs):
    """y[ids[i]] += d[i], i < n, FP16 rows of nt column tiles, both D layout (row tiles
    y_tile_stride / d_tile_stride apart; default nt blocks); ids uint16 in this cluster's L1.
    SIMD core."""

    KERNEL_NAME = "__snax_bingo_kernel_simd_scatter_add_rows"

    def __init__(self, y_addr, d_addr, nt: int, ids_addr, n: int, y_tile_stride: int = 0,
                 d_tile_stride: int = 0):
        if nt <= 0 or n < 0:
            raise ValueError(f"nt={nt}, n={n}")
        self.y_addr, self.d_addr, self.nt, self.ids_addr, self.n = y_addr, d_addr, int(nt), ids_addr, int(n)
        self.y_tile_stride = int(y_tile_stride) or nt * D_BLOCK
        self.d_tile_stride = int(d_tile_stride) or nt * D_BLOCK

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_simd_scatter_add_rows_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.y_addr, "y_addr", a, handle_name_map, split_64bit=False)
        a["y_tile_stride"] = str(self.y_tile_stride)
        self._process_addr(self.d_addr, "d_addr", a, handle_name_map, split_64bit=False)
        a["d_tile_stride"] = str(self.d_tile_stride)
        a["nt"] = str(self.nt)
        self._process_addr(self.ids_addr, "ids_addr", a, handle_name_map, split_64bit=False)
        a["n"] = str(self.n)
        return a
