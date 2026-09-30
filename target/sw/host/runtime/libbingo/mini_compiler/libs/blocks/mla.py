# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Multi-head latent attention (DeepSeek-V2), one token per pass, as blocks.

Ports of the snax reference's MLA block (sw/apps/dsv2/include/snax-dsv2-mla.h), which is
bit-exact on one snax_split_cluster against the DeepSeek-V2-Lite golden
(sw/apps/dsv2/util/hwmodel.py). What happens around them -- the projections, the norms,
the per-column dequantisation -- is the ordinary Linear / row blocks; these are the pieces
that exist only in MLA:

    RopeRows      RoPE at one position over the heads' q_pe (and the k_pe), gathered from
                  the projections' outputs into the kernel's four-operand block
    CacheAppend   the token's row [c8 | kpe8] into both cache copies: the KEY copy (A
                  layout of [cap, 576], runs of 4 bytes) and the VALUE copy (A layout of
                  V^T = [512, cap], runs of 1 byte)
    QAssemble     the score matmul's query operand Q8 [32, 576]: 16 heads of [q~ | q_pe]
                  in rows 0-15, two segments with two scales, rows 16-31 zero
    MlaAttention  the online softmax over key tiles on ONE cluster, then o~ = O / l per
                  query and its transpose: [heads, 512] rows for W_UV

THE ATTENTION IS FlashAttention's, TRANSPOSED (gemm_fa.h, simd.h): S^T = K . Q^T with one
key per 64-byte beat and one query per lane, so the softmax reduces along beats, and
O^T = V^T . P^T accumulates in place with corr applied on the C read path. MLA only
changes the shapes -- d_qk = 576 ([c | k_pe]), d_v = 512 (the latent) -- and adds the last
tile's FP16 exit (GEMM_FA_D_FP16) and the c / l normalisation. The recurrence is exactly
hwmodel.mla_attention; no tolerance is involved anywhere.
"""

from dataclasses import dataclass, field
from typing import Optional

from bingo_kernel_args import (
    SnaxBingoKernelGemmFaPvArgs,
    SnaxBingoKernelGemmFaQkArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelIdma2dCopyArgs,
    SnaxBingoKernelIdmaPairwiseSwapArgs,
    SnaxBingoKernelSimdFaSoftmaxArgs,
    SnaxBingoKernelSimdMlaNormaliseArgs,
    SnaxBingoKernelSimdQuantARowsArgs,
    SnaxBingoKernelSimdRopeArgs,
    SnaxBingoKernelXdmaMemsetArgs,
    SnaxBingoKernelXdmaTranspose2dArgs,
)

from ..comm import Block, BlockResult, Ctx, DType, Layout, MemLevel, Port, PortSpec
from ..comm.ports import at_offset
from .linear import LoadStream

BEAT = 64


def _row(shape, level, cluster, doc, dtype=DType.F16, layout=Layout.ROW_MAJOR):
    return PortSpec(layout, dtype, shape, mem_level=level,
                    cluster=cluster if level == MemLevel.L1 else None, doc=doc)


# ======================================================================================
# RoPE over the heads of one token
# ======================================================================================

@dataclass(frozen=True)
class RopeRowsCfg:
    heads: int                 # q_pe rows: the heads this cluster projected
    q_head: int = 192          # a head's slice of q: [q_nope | q_pe]
    q_nope: int = 128
    rope: int = 64
    kpe: bool = False          # one more row: k_pe, at column kv_rank of ckv
    kv_cols: int = 576
    kv_rank: int = 512
    cluster: int = 0

    @property
    def rows(self) -> int:
        return self.heads + (1 if self.kpe else 0)


class RopeRows(Block):
    """RoPE at one position over `heads` q_pe rows (and the k_pe row, last).

      in   q     [1, heads * q_head] fp16, L1: the heads' q, [q_nope | q_pe] each
           kv    [1, kv_cols] fp16, L1 (with kpe): ckv, k_pe at column kv_rank
           cos   [rows, rope] fp16, L3: cos repeated per pair
           sin   [rows, rope] fp16, L3: sin with the pair's sign, [-s0, +s0, ...]
      out  y     [rows, rope] fp16, L1: the rotated rows

    The kernel reads ONE block [x | cos | xswap | sin] of equally spaced operands (simd.h
    simd_rope), so the block owns it and fills it: x by a strided gather of the q_pe
    segments (idma_2d_copy) and a copy of k_pe, the tables from L3, the pair swap on the
    iDMA. Bit-exact hwmodel.rope: RNE(RNE(x cos) + RNE(swap(x) sin)).
    """

    name = "rope_rows"

    def __init__(self, cfg: RopeRowsCfg = None, **params):
        self.cfg = cfg if cfg is not None else RopeRowsCfg(**params)
        c = self.cfg
        if c.rope % 32 or c.heads < 1:
            raise ValueError(f"RopeRows: rope={c.rope} (a multiple of 32), heads={c.heads}.")

    @property
    def inputs(self) -> dict:
        c = self.cfg
        ins = {"q": _row((1, c.heads * c.q_head), MemLevel.L1, c.cluster, "the heads' q"),
               "cos": _row((c.rows, c.rope), MemLevel.L3, None, "cos per pair, per row"),
               "sin": _row((c.rows, c.rope), MemLevel.L3, None, "signed sin, per row")}
        if c.kpe:
            ins["kv"] = _row((1, c.kv_cols), MemLevel.L1, c.cluster, "ckv")
        return ins

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": _row((c.rows, c.rope), MemLevel.L1, c.cluster, "rotated rows")}

    def simd_passes(self) -> int:
        return 1

    def idma_passes(self) -> int:
        return 4 + (1 if self.cfg.kpe else 0)

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        row_b = c.rows * c.rope * 2
        ops = g.l1(f"{self.name}_ops", 4 * row_b)
        out = g.l1(f"{self.name}_y", row_b)
        q = bound["q"]
        gq = g.node("Gather_qpe", ctx.dm, "__snax_bingo_kernel_idma_2d_copy",
                    SnaxBingoKernelIdma2dCopyArgs(at_offset(q.handle, 2 * c.q_nope), ops,
                                                  2 * c.rope, 2 * c.q_head, 2 * c.rope,
                                                  c.heads),
                    list(q.ends))
        nodes, xs = [gq], [gq]
        if c.kpe:
            kv = bound["kv"]
            gk = g.node("Copy_kpe", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(at_offset(kv.handle, 2 * c.kv_rank),
                                                      at_offset(ops, c.heads * 2 * c.rope),
                                                      2 * c.rope),
                        list(kv.ends))
            nodes.append(gk)
            xs.append(gk)
        lc = g.node("Ld_cos", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(bound["cos"].handle, at_offset(ops, row_b),
                                                  row_b))
        ls = g.node("Ld_sin", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(bound["sin"].handle,
                                                  at_offset(ops, 3 * row_b), row_b))
        sw = g.node("Swap", ctx.dm, "__snax_bingo_kernel_idma_pairwise_swap",
                    SnaxBingoKernelIdmaPairwiseSwapArgs(ops, at_offset(ops, 2 * row_b),
                                                        c.rows * c.rope, 2),
                    xs)
        rp = g.node("Rope", ctx.simd, "__snax_bingo_kernel_simd_rope",
                    SnaxBingoKernelSimdRopeArgs(ops, out, c.rope, c.rows), [sw, lc, ls])
        nodes += [lc, ls, sw, rp]
        ins = {"q": Port(self.inputs["q"], q.handle, (gq,), name="q"),
               "cos": Port(self.inputs["cos"], bound["cos"].handle, (lc,), name="cos"),
               "sin": Port(self.inputs["sin"], bound["sin"].handle, (ls,), name="sin")}
        if c.kpe:
            ins["kv"] = Port(self.inputs["kv"], bound["kv"].handle, (nodes[1],), name="kv")
        return BlockResult(outputs={"y": Port(self.outputs["y"], out, (rp,), name="y")},
                           inputs=ins, nodes=nodes, sources=[lc, ls])


# ======================================================================================
# The cache append
# ======================================================================================

@dataclass(frozen=True)
class CacheAppendCfg:
    pos: int                   # the token's row
    cap: int                   # tokens both copies hold, a multiple of 16
    kv_rank: int = 512
    rope: int = 64
    cluster: int = 0           # where c8 and kpe8 are; its DM core writes the row


class CacheAppend(Block):
    """The token's row [c8 | kpe8] into both cache copies (snax dsv2_cache_append).

      in   c8, kpe8  int8 rows in this cluster's L1
           key       A layout of [cap, kv_rank + rope], int8, L3
           val       A layout of V^T = [kv_rank, cap], int8, L3
      out  key, val  the same arrays, with the row in them

    KEY: token t is row t of 16-token blocks of 4 values: kv_rank + rope values as runs of 4
    bytes, 64 B apart, from (t / 16) * (kv_rank + rope) / 4 * 64 + (t % 16) * 4.
    VALUE: element (i, t) at (i / 16) * cap * 16 + (t / 4) * 64 + (i % 16) * 4 + t % 4, so
    the row is kv_rank / 16 runs of 16 single bytes, 4 apart.
    """

    name = "cache_append"

    def __init__(self, cfg: CacheAppendCfg = None, **params):
        self.cfg = cfg if cfg is not None else CacheAppendCfg(**params)
        c = self.cfg
        if c.cap % 16 or not 0 <= c.pos < c.cap:
            raise ValueError(f"CacheAppend: pos={c.pos} in a cache of {c.cap} (a multiple of "
                             f"16).")

    @property
    def _d(self):
        return self.cfg.kv_rank + self.cfg.rope

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"c8": _row((1, c.kv_rank), MemLevel.L1, c.cluster, "the latent", DType.I8),
                "kpe8": _row((1, c.rope), MemLevel.L1, c.cluster, "the rotated k_pe",
                             DType.I8),
                "key": _row((c.cap, self._d), MemLevel.L3, None, "key copy", DType.I8,
                            Layout.A),
                "val": _row((c.kv_rank, c.cap), MemLevel.L3, None, "value copy", DType.I8,
                            Layout.A)}

    @property
    def outputs(self) -> dict:
        ins = self.inputs
        return {"key": ins["key"], "val": ins["val"]}

    def idma_passes(self) -> int:
        return 3

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        t, d = c.pos, self._d
        k0 = (t // 16) * (d // 4) * 64 + (t % 16) * 4
        key, val = bound["key"].handle, bound["val"].handle
        after_key = list(bound["key"].ends)
        after_val = list(bound["val"].ends)
        ak = g.node("Append_key_c", ctx.dm, "__snax_bingo_kernel_idma_2d_copy",
                    SnaxBingoKernelIdma2dCopyArgs(bound["c8"].handle, at_offset(key, k0), 4, 4,
                                                  64, c.kv_rank // 4),
                    list(bound["c8"].ends) + after_key)
        ap = g.node("Append_key_pe", ctx.dm, "__snax_bingo_kernel_idma_2d_copy",
                    SnaxBingoKernelIdma2dCopyArgs(bound["kpe8"].handle,
                                                  at_offset(key, k0 + (c.kv_rank // 4) * 64),
                                                  4, 4, 64, c.rope // 4),
                    list(bound["kpe8"].ends) + after_key)
        v0 = (t // 4) * 64 + (t % 4)
        # The three copies are CHAINED, so the last one implies the others and is the one
        # end both outputs carry: a consumer then waits on one node, not three.
        av = g.node("Append_val", ctx.dm, "__snax_bingo_kernel_idma_2d_copy",
                    SnaxBingoKernelIdma2dCopyArgs(bound["c8"].handle, at_offset(val, v0), 1, 1,
                                                  4, 16, outer=c.kv_rank // 16, src_outer=16,
                                                  dst_outer=c.cap * 16),
                    [ap] + after_val)
        g.dfg.bingo_add_edge(ak, ap)
        return BlockResult(
            outputs={"key": Port(self.outputs["key"], key, (av,), name="key"),
                     "val": Port(self.outputs["val"], val, (av,), name="val")},
            inputs={"c8": Port(self.inputs["c8"], bound["c8"].handle, (ak,), name="c8"),
                    "kpe8": Port(self.inputs["kpe8"], bound["kpe8"].handle, (ap,),
                                 name="kpe8"),
                    "key": Port(self.inputs["key"], key, (ak,), name="key"),
                    "val": Port(self.inputs["val"], val, (av,), name="val")},
            nodes=[ak, ap, av])


# ======================================================================================
# The query operand
# ======================================================================================

@dataclass(frozen=True)
class QAssembleCfg:
    inv_qt_f32bits: int        # q~'s quantiser
    inv_qpe_f32bits: int       # q_pe's: s_q~ s_c = s_qpe s_kpe keeps one exact dot product
    heads: int = 16
    br: int = 32               # query lanes of the score matmul
    kv_rank: int = 512
    rope: int = 64
    cluster: int = 0
    tokens: int = 1            # 2: a second token's heads in rows 16-31 (inputs qt1, qpe1)


class QAssemble(Block):
    """Q8, the score matmul's query operand: A layout of [br, kv_rank + rope] int8.

      in   qt   [heads, kv_rank] fp16, L1: the absorbed queries q~
           qpe  [heads, rope] fp16, L1: the rotated q_pe
      out  q8   [br, kv_rank + rope] int8, A layout, L1

    Rows 0 .. heads-1 are the heads -- q~ quantised at inv_qt into the first kv_rank / 4
    blocks, q_pe at inv_qpe into the rest -- and rows heads .. br-1 are ZERO, the query
    lanes one token leaves empty (zeroed on the xDMA first). Read as the B operand of
    S^T = K . Q^T it is the same bytes (meshRow == meshCol).
    """

    name = "q_assemble"

    def __init__(self, cfg: QAssembleCfg = None, **params):
        self.cfg = cfg if cfg is not None else QAssembleCfg(**params)
        if self.cfg.heads != 16:
            raise ValueError("QAssemble: the query rows are one 16-row m-block; heads=16.")
        if self.cfg.tokens not in (1, 2) or self.cfg.tokens * self.cfg.heads > self.cfg.br:
            raise ValueError("QAssemble: one or two tokens of 16 heads in the br lanes.")

    @property
    def _d(self):
        return self.cfg.kv_rank + self.cfg.rope

    @property
    def inputs(self) -> dict:
        c = self.cfg
        ins = {"qt": _row((c.heads, c.kv_rank), MemLevel.L1, c.cluster, "q~ per head"),
               "qpe": _row((c.heads, c.rope), MemLevel.L1, c.cluster, "q_pe per head")}
        if c.tokens == 2:
            ins["qt1"] = _row((c.heads, c.kv_rank), MemLevel.L1, c.cluster, "token 1's q~")
            ins["qpe1"] = _row((c.heads, c.rope), MemLevel.L1, c.cluster, "token 1's q_pe")
        return ins

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"q8": _row((c.br, self._d), MemLevel.L1, c.cluster, "Q8", DType.I8,
                           Layout.A)}

    def simd_passes(self) -> int:
        return 2

    def xdma_passes(self) -> int:
        return 1

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        nbytes = c.br * self._d
        q8 = g.l1(f"{self.name}_q8", nbytes)
        z = g.node("ZeroQ8", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                   SnaxBingoKernelXdmaMemsetArgs(q8, nbytes,
                                                 SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO))
        qt = g.node("Quant_qt", ctx.simd, "__snax_bingo_kernel_simd_quant_a_rows",
                    SnaxBingoKernelSimdQuantARowsArgs(bound["qt"].handle, 2 * c.kv_rank, q8,
                                                      c.kv_rank // 4, c.inv_qt_f32bits),
                    [z] + list(bound["qt"].ends))
        qp = g.node("Quant_qpe", ctx.simd, "__snax_bingo_kernel_simd_quant_a_rows",
                    SnaxBingoKernelSimdQuantARowsArgs(bound["qpe"].handle, 2 * c.rope,
                                                      at_offset(q8, (c.kv_rank // 4) * BEAT),
                                                      c.rope // 4, c.inv_qpe_f32bits),
                    [qt] + list(bound["qpe"].ends))
        nodes = [z, qt, qp]
        ins = {"qt": Port(self.inputs["qt"], bound["qt"].handle, (qt,), name="qt"),
               "qpe": Port(self.inputs["qpe"], bound["qpe"].handle, (qp,), name="qpe")}
        if c.tokens == 2:
            # token 1 in the second 16-row m-block: A layout is m-block major, one block of
            # d / 4 beats per 16 rows
            m1 = (self._d // 4) * BEAT
            qt1 = g.node("Quant_qt1", ctx.simd, "__snax_bingo_kernel_simd_quant_a_rows",
                         SnaxBingoKernelSimdQuantARowsArgs(bound["qt1"].handle, 2 * c.kv_rank,
                                                           at_offset(q8, m1), c.kv_rank // 4,
                                                           c.inv_qt_f32bits),
                         [qp] + list(bound["qt1"].ends))
            qp1 = g.node("Quant_qpe1", ctx.simd, "__snax_bingo_kernel_simd_quant_a_rows",
                         SnaxBingoKernelSimdQuantARowsArgs(
                             bound["qpe1"].handle, 2 * c.rope,
                             at_offset(q8, m1 + (c.kv_rank // 4) * BEAT), c.rope // 4,
                             c.inv_qpe_f32bits),
                         [qt1] + list(bound["qpe1"].ends))
            nodes += [qt1, qp1]
            ins["qt1"] = Port(self.inputs["qt1"], bound["qt1"].handle, (qt1,), name="qt1")
            ins["qpe1"] = Port(self.inputs["qpe1"], bound["qpe1"].handle, (qp1,), name="qpe1")
        return BlockResult(
            outputs={"q8": Port(self.outputs["q8"], q8, (nodes[-1],), name="q8")},
            inputs=ins, nodes=nodes, sources=[z])


# ======================================================================================
# The attention
# ======================================================================================

@dataclass(frozen=True)
class MlaAttnCfg:
    keys: int                  # pos + 1: the cache rows this token attends to
    cap: int                   # the cache copies' capacity
    k_s: int                   # the score matmul's D-port shift (depth 576)
    k_o: int                   # the last PV's D-port shift (depth `keys`)
    a_exp: float               # a' = softmax_scale s_q~ s_c 2^k_s, an FP32 value
    a_n_f32bits: int           # 1 / c, c = 2^k_o s_c / 127
    cluster: int = 0
    bc: int = 64               # keys per tile
    br: int = 32               # query lanes
    d_qk: int = 576
    d_v: int = 512
    # The cluster's LoadStream: K and V of a tile share one slab.
    stream: Optional[LoadStream] = field(default=None, compare=False, repr=False)
    # With a second stream, K goes through `stream` and V through `vstream`, each slab
    # released by its own reader (QK, PV). One slab each (36 + 32 KiB) still hides every
    # load but tile 1's K behind the other matmul: the GEMM runs QK(j+1) before PV(j), so
    # K(j+2) loads under PV(j) and V(j+1) under QK(j+2) -- half the L1 of two K|V slabs.
    vstream: Optional[LoadStream] = field(default=None, compare=False, repr=False)
    # Scores a query lane must not see (a later token of the same pass): (key, lane0, lanes)
    # runs, each overwritten with -65504 after its tile's QK and before its softmax by an
    # iDMA copy from `mask_src` (>= 64 B of 0xFBFF, in L3), so its P is exactly 0
    # (hwmodel.mla_attention `masked`). A key row is one 64-B beat, lane l at 2 l.
    masks: tuple = ()
    mask_src: object = field(default=None, compare=False, repr=False)

    @property
    def nt(self) -> int:
        return self.keys // self.bc


class MlaAttention(Block):
    """MLA attention for one token, on ONE cluster: 16 heads against `keys` cached rows.

      in   q8    [br, d_qk] int8, A layout, L1 (QAssemble)
           key   [cap, d_qk] int8, A layout, L3 (the key copy, row appended)
           val   [d_v, cap] int8, A layout of V^T, L3 (the value copy)
      out  ot    [d_v, br] fp16, L1: o~^T = O16 (.) c / l, one latent per 64-B beat
           xt    [br, d_v] fp16, L1: its transpose, one query (head) per row

    Per tile j of bc keys (all in the cluster's LoadStream: K and V of a tile share one
    slab, loaded after whatever last read it):
        Ld K_j   bc x d_qk bytes, contiguous (the key copy's tile)
        Ld V_j   d_v / 16 runs of 16 bc bytes, 16 cap apart (the value copy's tile)
        QK_j     S^T = K . Q^T -> FP16 at k_s, into s16[j % 2] behind a latch beat
        SM_j     the online softmax (simd_fa_softmax, EXACT: P interleaved for PV, corr
                 exported for its column scaler), m and l carried in the arena
        PV_j     O^T = corr (.) O^T + V^T . P^T, INT32 in place; the last tile leaves as
                 FP16 at k_o (GEMM_FA_D_FP16)
    then c / l into O16's latch and o~ (simd_mla_normalise), and the transpose (xDMA).

    Every QK waits for q8 -- the first two by an edge, the rest through the softmax two
    tiles back -- and the tile holding the appended row waits for the cache writes: a node
    with no path from its producer may be dispatched before it.
    """

    name = "mla_attn"

    def __init__(self, cfg: MlaAttnCfg = None, **params):
        self.cfg = cfg if cfg is not None else MlaAttnCfg(**params)
        c = self.cfg
        if c.keys % c.bc or c.keys > c.cap:
            raise ValueError(
                f"MlaAttention: keys={c.keys} is not whole tiles of {c.bc} (or exceeds the "
                f"cache's {c.cap}). A partial last tile needs its padding keys' scores "
                f"masked to -65504 before the softmax, which this block does not emit yet.")
        if (c.br, c.bc % 16, c.d_qk % 4, c.d_v % 16) != (32, 0, 0, 0):
            raise ValueError("MlaAttention: br = 32 query lanes, bc a multiple of 16.")
        if c.stream is None or c.stream.cluster != c.cluster:
            raise ValueError("MlaAttention: pass this cluster's LoadStream (stream=).")
        for key, l0, nl in c.masks:
            if not (0 <= key < c.keys and 0 <= l0 and nl > 0 and l0 + nl <= c.br):
                raise ValueError(f"MlaAttention: mask {(key, l0, nl)} outside {c.keys} keys "
                                 f"x {c.br} lanes.")
        if c.masks and c.mask_src is None:
            raise ValueError("MlaAttention: masks need mask_src, 0xFBFF bytes in L3.")
        if c.vstream is not None:
            if c.vstream.cluster != c.cluster or c.vstream is c.stream:
                raise ValueError("MlaAttention: vstream is a second LoadStream of this "
                                 "cluster.")
            if c.stream.nbytes < c.bc * c.d_qk or c.vstream.nbytes < c.d_v * c.bc:
                raise ValueError(f"MlaAttention: K tiles ({c.bc * c.d_qk} B) and V tiles "
                                 f"({c.d_v * c.bc} B) do not fit slabs of "
                                 f"{c.stream.nbytes} B and {c.vstream.nbytes} B.")
        elif c.stream.nbytes < c.bc * c.d_qk + c.d_v * c.bc:
            raise ValueError(f"MlaAttention: a tile's K and V ({c.bc * (c.d_qk + c.d_v)} B) "
                             f"do not fit the stream's {c.stream.nbytes} B slab.")

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"q8": _row((c.br, c.d_qk), MemLevel.L1, c.cluster, "Q8", DType.I8, Layout.A),
                "key": _row((c.cap, c.d_qk), MemLevel.L3, None, "key copy", DType.I8,
                            Layout.A),
                "val": _row((c.d_v, c.cap), MemLevel.L3, None, "value copy", DType.I8,
                            Layout.A)}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"ot": _row((c.d_v, c.br), MemLevel.L1, c.cluster, "o~^T"),
                "xt": _row((c.br, c.d_v), MemLevel.L1, c.cluster, "o~, a head per row")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        fa = SnaxBingoKernelSimdFaSoftmaxArgs
        pv_cls = SnaxBingoKernelGemmFaPvArgs
        # The softmax arena at the LOWEST address of the attention's buffers: two of its
        # tasks reach into s16 and p8 with unsigned strides (PLACEMENT_ORDER). The runtime
        # allocator places by name, and "arena" sorts first. dhead = 1: under CORR_EXPORT
        # the arena's O region is never touched, so it need not be d_v beats.
        dhead = 1
        arena = g.l1(f"{self.name}_arena", fa.arena_bytes(c.bc, dhead))
        lay = fa.layout(c.bc, dhead)
        s16_b = BEAT + c.bc * c.br * 2
        s16 = [g.l1(f"{self.name}_s16_{i}", s16_b) for i in range(2)]
        p8_b = fa.p8_bytes(c.bc, c.br, fa.EXACT)
        p8 = [g.l1(f"{self.name}_p8_{i}", p8_b) for i in range(2)]
        oacc = g.l1(f"{self.name}_oacc", c.d_v * c.br * 4)
        o16 = g.l1(f"{self.name}_o16", BEAT + c.d_v * c.br * 2)
        # o~^T and its transpose live in the INT32 accumulator's bytes: it is dead once the
        # last PV has read it, and both come after that.
        ot16 = at_offset(oacc, c.d_v * c.br * 2)
        xt = at_offset(oacc, 0)
        corr = fa.corr_offset(c.bc, c.br)
        q8 = bound["q8"]
        q8_after = list(q8.ends)
        # Only the tiles holding rows the cache writes just produced wait for those writes
        # (the last tile, for one token). Every other tile is already in the copies and
        # loads as soon as its slab is free.
        kv_after = list(dict.fromkeys(list(bound["key"].ends) + list(bound["val"].ends)))
        fresh = {(c.keys - 1) // c.bc} if kv_after else set()
        kbytes, vbytes = c.bc * c.d_qk, c.d_v * c.bc
        nodes, qk, sm, pv, slab_of = [], {}, {}, {}, {}

        split = c.vstream is not None
        k_slab, v_slab = {}, {}          # tile -> (slab index, slab handle) when split

        def load_k(j, slab, war):
            n = g.node(f"Ld_K{j}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(
                           at_offset(bound["key"].handle, j * kbytes), slab, kbytes),
                       war + (kv_after if j in fresh else []))
            nodes.append(n)
            return n

        def load_v(j, dst, war):
            n = g.node(f"Ld_V{j}", ctx.dm, "__snax_bingo_kernel_idma_2d_copy",
                       SnaxBingoKernelIdma2dCopyArgs(
                           at_offset(bound["val"].handle, j * 16 * c.bc), dst, 16 * c.bc,
                           16 * c.cap, 16 * c.bc, c.d_v // 16),
                       war + (kv_after if j in fresh else []))
            nodes.append(n)
            return n

        def loads(j):
            si, slab, war = c.stream.take(kbytes + vbytes)
            slab_of[j] = (si, slab)
            return load_k(j, slab, war), load_v(j, at_offset(slab, kbytes), war)

        def split_k(j):
            si, slab, war = c.stream.take(kbytes)
            k_slab[j] = (si, slab)
            return load_k(j, slab, war)

        def split_v(j):
            si, slab, war = c.vstream.take(vbytes)
            v_slab[j] = (si, slab)
            return load_v(j, slab, war)

        ld = {}
        if split:
            nk, nv = len(c.stream.slabs), len(c.vstream.slabs)
            ldk = {j: split_k(j) for j in range(min(nk, c.nt))}
            ldv = {j: split_v(j) for j in range(min(nv, c.nt))}
            nbuf = 0
        else:
            nbuf = len(c.stream.slabs)
            for j in range(min(nbuf, c.nt)):
                ld[j] = loads(j)

        # tile j's K and V: where they sit and the loads that put them there
        def k_at(j):
            return k_slab[j][1] if split else slab_of[j][1]

        def v_at(j):
            return v_slab[j][1] if split else at_offset(slab_of[j][1], kbytes)

        def k_ld(j):
            return ldk[j] if split else ld[j][0]

        def v_ld(j):
            return ldv[j] if split else ld[j][1]

        def make_pv(j):
            last, first = j == c.nt - 1, j == 0
            flags = pv_cls.B_KMAJOR
            if not first:
                flags |= pv_cls.C_COLSCALE
            if last:
                flags |= pv_cls.D_FP16
            deps = [sm[j], v_ld(j)] + ([pv[j - 1]] if j else [])
            pv[j] = g.node(f"PV{j}", ctx.gemm, "__snax_bingo_kernel_gemm_fa_pv",
                           pv_cls(v_at(j),
                                  p8[j % 2], 0 if first else oacc,
                                  at_offset(o16, BEAT) if last else oacc,
                                  M=c.d_v // 16, K=c.bc // 4, N=c.br // 16,
                                  flags=flags,
                                  corr_addr=0 if first else at_offset(p8[j % 2], corr),
                                  d_shift=c.k_o if last else 0),
                           deps)
            nodes.append(pv[j])
            if not split:
                c.stream.release(slab_of[j][0], pv[j])
                return
            c.vstream.release(v_slab[j][0], pv[j])
            if j + nv < c.nt:
                ldv[j + nv] = split_v(j + nv)

        for j in range(c.nt):
            qk[j] = g.node(f"QK{j}", ctx.gemm, "__snax_bingo_kernel_gemm_fa_qk",
                           SnaxBingoKernelGemmFaQkArgs(
                               k_at(j), q8.handle, 0, at_offset(s16[j % 2], BEAT),
                               M=c.bc // 16, K=c.d_qk // 4, N=c.br // 16, d_shift=c.k_s),
                           # QK(j >= 2) waits for Softmax(j-2), which waits for QK(j-2):
                           # the first two carry the edge to q8 and the rest inherit it.
                           [k_ld(j)] + (q8_after if j < 2 else [sm[j - 2]]))
            if split:
                c.stream.release(k_slab[j][0], qk[j])
                if j + nk < c.nt:
                    ldk[j + nk] = split_k(j + nk)
            # the pass's causal masks in this tile: runs of lanes of one key row, merged
            # across consecutive rows when they cover the whole row
            mk = []
            for key, l0, nl in sorted(m for m in c.masks if j * c.bc <= m[0] < (j + 1) * c.bc):
                off, nb = BEAT + (key - j * c.bc) * BEAT + 2 * l0, 2 * nl
                if mk and mk[-1][0] + mk[-1][1] == off:
                    mk[-1][1] += nb
                else:
                    mk.append([off, nb])
            mask_nodes = []
            for i, (off, nb) in enumerate(mk):
                mask_nodes.append(g.node(
                    f"Mask{j}_{i}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(c.mask_src, at_offset(s16[j % 2], off), nb),
                    [qk[j]] + mask_nodes[-1:]))
            nodes += mask_nodes
            sm[j] = g.node(f"Softmax{j}", ctx.simd, "__snax_bingo_kernel_simd_fa_softmax",
                           fa(at_offset(s16[j % 2], BEAT), p8[j % 2], arena, c.bc, dhead,
                              tile_idx=j, seed_state=1, geom_mode=fa.GEOM_SELF,
                              score_scale=c.a_exp, p_mode=fa.EXACT, p8_pitch=0),
                           [qk[j]] + mask_nodes[-1:] + ([sm[j - 1]] if j else []) +
                           ([pv[j - 2]] if j >= 2 else []))
            nodes += [qk[j], sm[j]]
            if j >= 1:
                make_pv(j - 1)
                if not split and j - 1 + nbuf < c.nt:
                    ld[j - 1 + nbuf] = loads(j - 1 + nbuf)
        make_pv(c.nt - 1)
        last = c.nt - 1
        nrm = g.node("Normalise", ctx.simd, "__snax_bingo_kernel_simd_mla_normalise",
                     SnaxBingoKernelSimdMlaNormaliseArgs(at_offset(arena, lay["lrun"]), o16,
                                                         at_offset(o16, BEAT), ot16, c.d_v,
                                                         c.a_n_f32bits),
                     [pv[last], sm[last]])
        tr = g.node("Transpose", ctx.xdma, "__snax_bingo_kernel_xdma_transpose_2d",
                    SnaxBingoKernelXdmaTranspose2dArgs(ot16, xt, c.d_v, c.br, 2), [nrm])
        nodes += [nrm, tr]
        return BlockResult(
            outputs={"ot": Port(self.outputs["ot"], ot16, (nrm,), name="ot"),
                     "xt": Port(self.outputs["xt"], xt, (tr,), name="xt")},
            inputs={"q8": Port(self.inputs["q8"], q8.handle, (qk[0],), name="q8"),
                    "key": Port(self.inputs["key"], bound["key"].handle, (k_ld(0),),
                                name="key"),
                    "val": Port(self.inputs["val"], bound["val"].handle, (v_ld(0),),
                                name="val")},
            nodes=nodes,
            extra={"arena": arena, "layout": lay, "o16": at_offset(o16, BEAT),
                   "oacc": oacc, "qk": qk, "softmax": sm, "pv": pv})
