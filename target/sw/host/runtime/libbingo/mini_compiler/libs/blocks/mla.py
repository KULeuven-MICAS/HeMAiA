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
    SnaxBingoKernelSimdAddF16Args,
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
    fused: bool = False        # one input "qq" [heads, kv_rank + rope]: each head's q~ then q_pe


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
        if self.cfg.fused and self.cfg.tokens != 1:
            raise ValueError("QAssemble(fused): one token.")

    @property
    def _d(self):
        return self.cfg.kv_rank + self.cfg.rope

    @property
    def inputs(self) -> dict:
        c = self.cfg
        if c.fused:
            return {"qq": _row((c.heads, self._d), MemLevel.L1, c.cluster,
                               "per head: q~, then the rotated q_pe")}
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
        # fused: one [heads, kv_rank + rope] input, both halves read at its row pitch
        qt_in, qpe_in = (bound["qq"], bound["qq"]) if c.fused else (bound["qt"], bound["qpe"])
        qt_pitch, qpe_pitch = (2 * self._d,) * 2 if c.fused else (2 * c.kv_rank, 2 * c.rope)
        qpe_src = at_offset(qpe_in.handle, 2 * c.kv_rank) if c.fused else qpe_in.handle
        qt = g.node("Quant_qt", ctx.simd, "__snax_bingo_kernel_simd_quant_a_rows",
                    SnaxBingoKernelSimdQuantARowsArgs(qt_in.handle, qt_pitch, q8,
                                                      c.kv_rank // 4, c.inv_qt_f32bits),
                    [z] + list(qt_in.ends))
        qp = g.node("Quant_qpe", ctx.simd, "__snax_bingo_kernel_simd_quant_a_rows",
                    SnaxBingoKernelSimdQuantARowsArgs(qpe_src, qpe_pitch,
                                                      at_offset(q8, (c.kv_rank // 4) * BEAT),
                                                      c.rope // 4, c.inv_qpe_f32bits),
                    [qt] + list(qpe_in.ends))
        nodes = [z, qt, qp]
        if c.fused:
            ins = {"qq": Port(self.inputs["qq"], qt_in.handle, (qp,), name="qq")}
        else:
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
        # The key / value copies' readers, for the edges from their producer (the append): the
        # fresh tile's loads. Named as tile 0's, every tile waited for the cache writes (tile
        # 0 first in the chain) and the attention could not start on the old keys.
        rd = max(fresh) if fresh else 0
        return BlockResult(
            outputs={"ot": Port(self.outputs["ot"], ot16, (nrm,), name="ot"),
                     "xt": Port(self.outputs["xt"], xt, (tr,), name="xt")},
            inputs={"q8": Port(self.inputs["q8"], q8.handle, (qk[0],), name="q8"),
                    "key": Port(self.inputs["key"], bound["key"].handle, (k_ld(rd),),
                                name="key"),
                    "val": Port(self.inputs["val"], bound["val"].handle, (v_ld(rd),),
                                name="val")},
            nodes=nodes,
            extra={"arena": arena, "layout": lay, "o16": at_offset(o16, BEAT),
                   "oacc": oacc, "qk": qk, "softmax": sm, "pv": pv,
                   # the last tile's K and V loads: a stream's next loads can wait for them
                   "last_loads": [k_ld(c.nt - 1), v_ld(c.nt - 1)]})


# ---- the attention split over clusters by key (params att_split) -----------------------------
#
# Every cluster attends over its own shard of key tiles for all 16 heads. Pass 1 finds the
# shard's row maximum; the clusters exchange them and each takes the lanewise max m*; pass 2
# runs the shard's softmax SEEDED at m* (seed_state 0), so every shard quantises P on one scale
# and its PV needs no correction across clusters; the FP16 partials and row sums are then added
# (AddRow) and normalised on the attention's cluster. The golden is
# dsv2_datagen.split_attention_golden, the same recurrence with the same seed.

@dataclass(frozen=True)
class MlaShardCfg:
    tile0: int                 # the shard's first key tile
    nt: int                    # its tiles
    cap: int
    k_s: int
    k_o: int
    a_exp: float
    a_n_f32bits: int = 0
    cluster: int = 0
    bc: int = 64
    br: int = 32
    d_qk: int = 576
    d_v: int = 512
    # MlaShardScores loads the shard's V tiles too (mode "key": its own PV reads them); mode
    # "dim" loads a latent slice of V^T over every key instead (MlaDimPV)
    with_v: bool = True
    # the key tile holding the rows the cache writes just produced (global index): only its
    # loads wait for those writes. -1: every tile of the shard waits (as before)
    fresh_tile: int = -1


class MlaShardScores(Block):
    """Pass 1 of a key shard: its K and V tiles into this L1, S^T = K . Q^T per tile, and the
    online softmax with the kernel's own seed -- for its row maximum only.

      in   q8   [br, d_qk] int8 A, L1;  key, val  the cache copies in L3 (this chip's)
      out  m    one beat: the shard's row maximum per query lane (the arena's mrun)
           s16  the score tiles, one 64-B prefix beat then bc x br fp16 each (pass 2 reads
                them again)
           v    the V tiles, d_v x bc bytes each (pass 2's PV)"""

    name = "mla_shard1"

    def __init__(self, cfg: MlaShardCfg = None, **params):
        self.cfg = cfg if cfg is not None else MlaShardCfg(**params)

    @property
    def inputs(self) -> dict:
        c = self.cfg
        ins = {"q8": _row((c.br, c.d_qk), MemLevel.L1, c.cluster, "Q8", DType.I8, Layout.A),
               "key": _row((c.cap, c.d_qk), MemLevel.L3, None, "key copy", DType.I8, Layout.A)}
        if c.with_v:
            ins["val"] = _row((c.d_v, c.cap), MemLevel.L3, None, "value copy", DType.I8,
                              Layout.A)
        return ins

    @property
    def outputs(self) -> dict:
        c = self.cfg
        s16b = c.nt * (BEAT + c.bc * c.br * 2)
        outs = {"m": _row((1, c.br), MemLevel.L1, c.cluster, "row maximum"),
                "s16": _row((1, s16b // 2), MemLevel.L1, c.cluster, "score tiles")}
        if c.with_v:
            outs["v"] = _row((1, c.nt * c.d_v * c.bc), MemLevel.L1, c.cluster, "V tiles",
                             DType.I8)
        return outs

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        fa = SnaxBingoKernelSimdFaSoftmaxArgs
        kbytes, vbytes, sbytes = c.bc * c.d_qk, c.d_v * c.bc, BEAT + c.bc * c.br * 2
        arena = g.l1(f"{self.name}_arena", fa.arena_bytes(c.bc, 1))
        kt = g.l1(f"{self.name}_k", c.nt * kbytes)
        vt = g.l1(f"{self.name}_v", c.nt * vbytes) if c.with_v else None
        s16 = g.l1(f"{self.name}_s16", c.nt * sbytes)
        p8 = g.l1(f"{self.name}_p8", fa.p8_bytes(c.bc, c.br, fa.EXACT))
        key_after = list(bound["key"].ends)
        val_after = list(bound["val"].ends) if c.with_v else []
        # fresh_tile: only the tile with the appended rows waits for the cache writes (the
        # shard's other tiles are in the copies already, and pass 1 starts on them)
        waits = (lambda j: j == c.fresh_tile) if c.fresh_tile >= 0 else (lambda j: True)
        rd = max(c.fresh_tile - c.tile0, 0) if 0 <= c.fresh_tile - c.tile0 < c.nt else 0
        nodes, lk, lv, sm = [], [], [], []
        for i in range(c.nt):
            j = c.tile0 + i
            lk.append(g.node(f"Ld_K{i}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                             SnaxBingoKernelIdma1dCopyArgs(
                                 at_offset(bound["key"].handle, j * kbytes),
                                 at_offset(kt, i * kbytes), kbytes),
                             key_after if waits(j) else []))
            if c.with_v:
                lv.append(g.node(f"Ld_V{i}", ctx.dm, "__snax_bingo_kernel_idma_2d_copy",
                                 SnaxBingoKernelIdma2dCopyArgs(
                                     at_offset(bound["val"].handle, j * 16 * c.bc),
                                     at_offset(vt, i * vbytes), 16 * c.bc, 16 * c.cap,
                                     16 * c.bc, c.d_v // 16),
                                 val_after if waits(j) else []))
            qk = g.node(f"QK{i}", ctx.gemm, "__snax_bingo_kernel_gemm_fa_qk",
                        SnaxBingoKernelGemmFaQkArgs(
                            at_offset(kt, i * kbytes), bound["q8"].handle, 0,
                            at_offset(s16, i * sbytes + BEAT),
                            M=c.bc // 16, K=c.d_qk // 4, N=c.br // 16, d_shift=c.k_s),
                        [lk[i]] + list(bound["q8"].ends))
            sm.append(g.node(f"Softmax{i}", ctx.simd, "__snax_bingo_kernel_simd_fa_softmax",
                             fa(at_offset(s16, i * sbytes + BEAT), p8, arena, c.bc, 1,
                                tile_idx=i, seed_state=1, geom_mode=fa.GEOM_SELF,
                                score_scale=c.a_exp, p_mode=fa.EXACT, p8_pitch=0),
                             [qk] + sm[-1:]))
            nodes += [lk[i]] + lv[i:i + 1] + [qk, sm[i]]
            qk0 = qk if i == 0 else qk0
        lay = fa.layout(c.bc, 1)
        outs = {"m": Port(self.outputs["m"], at_offset(arena, lay["mrun"]), (sm[-1],), name="m"),
                "s16": Port(self.outputs["s16"], s16, (sm[-1],), name="s16")}
        # (the copies' readers, for the edges from their producer: the fresh tile's loads)
        ins = {"q8": Port(self.inputs["q8"], bound["q8"].handle, (qk0,), name="q8"),
               "key": Port(self.inputs["key"], bound["key"].handle, (lk[rd],), name="key")}
        if c.with_v:
            outs["v"] = Port(self.outputs["v"], vt, tuple(lv), name="v")
            ins["val"] = Port(self.inputs["val"], bound["val"].handle, (lv[rd],), name="val")
        return BlockResult(outputs=outs, inputs=ins, nodes=nodes)


@dataclass(frozen=True)
class MlaRowMaxCfg:
    rows: int                  # beats to reduce (even)
    br: int = 32
    cluster: int = 0


class MlaRowMax(Block):
    """The lanewise maximum of `rows` fp16 beats in this L1: the FA softmax kernel run over
    them as one score tile (score_scale 1, its own seed), its arena's mrun left holding
    max(rows, -65504) exactly -- the max has no rounding.

      in   x  [rows, br] fp16, L1        out  m  one beat"""

    name = "mla_rowmax"

    def __init__(self, cfg: MlaRowMaxCfg = None, **params):
        self.cfg = cfg if cfg is not None else MlaRowMaxCfg(**params)
        if self.cfg.rows % 2:
            raise ValueError("MlaRowMax: an even number of rows (the kernel packs P 2:1)")

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"x": _row((c.rows, c.br), MemLevel.L1, c.cluster, "rows")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"m": _row((1, c.br), MemLevel.L1, c.cluster, "their lanewise max")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        fa = SnaxBingoKernelSimdFaSoftmaxArgs
        arena = g.l1(f"{self.name}_arena", fa.arena_bytes(c.rows, 1))
        # the kernel writes -m into the beat just below its score tile: a prefixed copy
        tile = g.l1(f"{self.name}_tile", BEAT + c.rows * BEAT)
        p8 = g.l1(f"{self.name}_p8", fa.p8_bytes(c.rows, c.br, 0))
        cp = g.node("Copy", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                    SnaxBingoKernelIdma1dCopyArgs(bound["x"].handle, at_offset(tile, BEAT),
                                                  c.rows * BEAT), list(bound["x"].ends))
        sm = g.node("Max", ctx.simd, "__snax_bingo_kernel_simd_fa_softmax",
                    fa(at_offset(tile, BEAT), p8, arena, c.rows, 1, tile_idx=0, seed_state=1,
                       geom_mode=fa.GEOM_SELF, score_scale=1.0, p_mode=0, p8_pitch=0), [cp])
        lay = fa.layout(c.rows, 1)
        return BlockResult(
            outputs={"m": Port(self.outputs["m"], at_offset(arena, lay["mrun"]), (sm,),
                               name="m")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle, (cp,), name="x")},
            nodes=[cp, sm])


class MlaShardPV(Block):
    """Pass 2 of a key shard: the softmax seeded at the global maximum m* (its arena's mrun =
    m*, lrun = 0, seed_state 0) over pass 1's score tiles, and PV; the shard's last tile
    leaves O^T as FP16 through the D port at k_o, as MlaAttention's last tile does.

      in   m  one beat, m*;  s16, v  MlaShardScores'
      out  o  [d_v x br] fp16 (O16^T), L1;  l  one beat, the shard's row sums"""

    name = "mla_shard2"

    def __init__(self, cfg: MlaShardCfg = None, **params):
        self.cfg = cfg if cfg is not None else MlaShardCfg(**params)

    @property
    def inputs(self) -> dict:
        c = self.cfg
        s16b = c.nt * (BEAT + c.bc * c.br * 2)
        return {"m": _row((1, c.br), MemLevel.L1, c.cluster, "m*"),
                "s16": _row((1, s16b // 2), MemLevel.L1, c.cluster, "score tiles"),
                "v": _row((1, c.nt * c.d_v * c.bc), MemLevel.L1, c.cluster, "V tiles",
                          DType.I8)}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"o": _row((1, c.d_v * c.br), MemLevel.L1, c.cluster, "O16^T partial"),
                "l": _row((1, c.br), MemLevel.L1, c.cluster, "row sums")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        fa, pv_cls = SnaxBingoKernelSimdFaSoftmaxArgs, SnaxBingoKernelGemmFaPvArgs
        vbytes, sbytes = c.d_v * c.bc, BEAT + c.bc * c.br * 2
        arena = g.l1(f"{self.name}_arena", fa.arena_bytes(c.bc, 1))
        lay = fa.layout(c.bc, 1)
        p8 = [g.l1(f"{self.name}_p8_{i}", fa.p8_bytes(c.bc, c.br, fa.EXACT))
              for i in range(min(2, c.nt))]
        o16 = g.l1(f"{self.name}_o16", c.d_v * c.br * 2)
        oacc = g.l1(f"{self.name}_oacc", c.d_v * c.br * 4) if c.nt > 1 else None
        corr = fa.corr_offset(c.bc, c.br)
        seed_m = g.node("SeedM", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(bound["m"].handle,
                                                      at_offset(arena, lay["mrun"]), BEAT),
                        list(bound["m"].ends))
        seed_l = g.node("SeedL", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                        SnaxBingoKernelXdmaMemsetArgs(at_offset(arena, lay["lrun"]), BEAT,
                                                      SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO))
        nodes, sm, pv = [seed_m, seed_l], [], []
        for i in range(c.nt):
            first, last = i == 0, i == c.nt - 1
            sm.append(g.node(f"Softmax{i}", ctx.simd, "__snax_bingo_kernel_simd_fa_softmax",
                             fa(at_offset(bound["s16"].handle, i * sbytes + BEAT), p8[i % 2],
                                arena, c.bc, 1, tile_idx=i, seed_state=0,
                                geom_mode=fa.GEOM_SELF, score_scale=c.a_exp, p_mode=fa.EXACT,
                                p8_pitch=0),
                             [seed_m, seed_l] + list(bound["s16"].ends) + sm[-1:] +
                             (pv[i - 2:i - 1] if i >= 2 else [])))
            flags = pv_cls.B_KMAJOR | (0 if first else pv_cls.C_COLSCALE) | \
                (pv_cls.D_FP16 if last else 0)
            pv.append(g.node(f"PV{i}", ctx.gemm, "__snax_bingo_kernel_gemm_fa_pv",
                             pv_cls(at_offset(bound["v"].handle, i * vbytes), p8[i % 2],
                                    0 if first else oacc, o16 if last else oacc,
                                    M=c.d_v // 16, K=c.bc // 4, N=c.br // 16, flags=flags,
                                    corr_addr=0 if first else at_offset(p8[i % 2], corr),
                                    d_shift=c.k_o if last else 0),
                             [sm[i]] + list(bound["v"].ends) + pv[-1:]))
            nodes += [sm[i], pv[i]]
        return BlockResult(
            outputs={"o": Port(self.outputs["o"], o16, (pv[-1],), name="o"),
                     "l": Port(self.outputs["l"], at_offset(arena, lay["lrun"]), (sm[-1],),
                               name="l")},
            inputs={"m": Port(self.inputs["m"], bound["m"].handle, (seed_m,), name="m"),
                    "s16": Port(self.inputs["s16"], bound["s16"].handle, (sm[0],), name="s16"),
                    "v": Port(self.inputs["v"], bound["v"].handle, (pv[0],), name="v")},
            nodes=nodes)


class MlaShardOut(Block):
    """The attention's cluster's end of the split: the LAST partials added (O16 and l, FP16,
    RNE -- AddRow's SIMD pass) straight into the normalise's operand, o~ = O16 c / l, and
    its transpose -- MlaAttention's two outputs.

      in   o_acc, o_last  [d_v x br] fp16;  l_acc, l_last  one beat each
      out  ot  [d_v, br] fp16;  xt  [br, d_v] fp16, one query (head) per row"""

    name = "mla_shard_out"

    def __init__(self, cfg: MlaShardCfg = None, **params):
        self.cfg = cfg if cfg is not None else MlaShardCfg(**params)

    @property
    def inputs(self) -> dict:
        c = self.cfg
        o = _row((1, c.d_v * c.br), MemLevel.L1, c.cluster, "O16^T partial")
        l = _row((1, c.br), MemLevel.L1, c.cluster, "row sums")
        return {"o_acc": o, "o_last": o, "l_acc": l, "l_last": l}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"ot": _row((c.d_v, c.br), MemLevel.L1, c.cluster, "o~^T"),
                "xt": _row((c.br, c.d_v), MemLevel.L1, c.cluster, "o~, a head per row")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        # the normalise reads its latch beat DIRECTLY BELOW O16
        o16 = g.l1(f"{self.name}_o16", BEAT + c.d_v * c.br * 2)
        lsum = g.l1(f"{self.name}_l", BEAT)
        ot16 = g.l1(f"{self.name}_ot", c.d_v * c.br * 2)
        xt = g.l1(f"{self.name}_xt", c.br * c.d_v * 2)
        add_o = g.node("AddO", ctx.simd, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdAddF16Args(bound["o_acc"].handle,
                                                     bound["o_last"].handle,
                                                     at_offset(o16, BEAT), rows=1,
                                                     cols=c.d_v * c.br),
                       list(bound["o_acc"].ends) + list(bound["o_last"].ends))
        add_l = g.node("AddL", ctx.simd, "__snax_bingo_kernel_simd_stream_elementwise",
                       SnaxBingoKernelSimdAddF16Args(bound["l_acc"].handle,
                                                     bound["l_last"].handle, lsum, rows=1,
                                                     cols=c.br),
                       list(bound["l_acc"].ends) + list(bound["l_last"].ends) + [add_o])
        nrm = g.node("Normalise", ctx.simd, "__snax_bingo_kernel_simd_mla_normalise",
                     SnaxBingoKernelSimdMlaNormaliseArgs(lsum, o16, at_offset(o16, BEAT), ot16,
                                                         c.d_v, c.a_n_f32bits), [add_o, add_l])
        tr = g.node("Transpose", ctx.xdma, "__snax_bingo_kernel_xdma_transpose_2d",
                    SnaxBingoKernelXdmaTranspose2dArgs(ot16, xt, c.d_v, c.br, 2), [nrm])
        return BlockResult(
            outputs={"ot": Port(self.outputs["ot"], ot16, (nrm,), name="ot"),
                     "xt": Port(self.outputs["xt"], xt, (tr,), name="xt")},
            inputs={"o_acc": Port(self.inputs["o_acc"], bound["o_acc"].handle, (add_o,),
                                  name="o_acc"),
                    "o_last": Port(self.inputs["o_last"], bound["o_last"].handle, (add_o,),
                                   name="o_last"),
                    "l_acc": Port(self.inputs["l_acc"], bound["l_acc"].handle, (add_l,),
                                  name="l_acc"),
                    "l_last": Port(self.inputs["l_last"], bound["l_last"].handle, (add_l,),
                                   name="l_last")},
            nodes=[add_o, add_l, nrm, tr])


# ---- att_split_mode dim: PV split by latent, not by key --------------------------------------
#
# QK and both softmax passes stay split by key (MlaShardScores with_v=False, MlaRowMax), but
# pass 2 only writes P: every cluster gathers every shard's P (P^T's k-major blocks of one
# shard are one contiguous run, so the shards back to back in key order ARE the whole P^T)
# and computes d_v / G rows of O^T over all the keys in ONE matmul, out of a V^T slice that
# is one 2-D copy of the value cache. O never meets an FP16 add: only the row sums do, as a
# halving tree on every cluster, so each normalises its own slice; the attention's cluster
# gathers the slices and transposes. The golden is dsv2_datagen.split_attention_golden (mode
# dim).

class MlaShardP(Block):
    """Pass 2 of a key shard, P only: the softmax seeded at m* (arena mrun = m*, lrun = 0,
    seed_state 0) over pass 1's score tiles, tile i's P into a strip at i bc br bytes -- its
    trailing row-sum and corr beats land on tile i+1's P before that is written, and nothing
    reads them after their own pass.

      in   m  one beat, m*;  s16  MlaShardScores'
      out  p  [nt bc, br] int8, P^T in PV's k-major B layout;  l  one beat, the row sums"""

    name = "mla_shard2"

    def __init__(self, cfg: MlaShardCfg = None, **params):
        self.cfg = cfg if cfg is not None else MlaShardCfg(**params)

    @property
    def inputs(self) -> dict:
        c = self.cfg
        s16b = c.nt * (BEAT + c.bc * c.br * 2)
        return {"m": _row((1, c.br), MemLevel.L1, c.cluster, "m*"),
                "s16": _row((1, s16b // 2), MemLevel.L1, c.cluster, "score tiles")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"p": _row((1, c.nt * c.bc * c.br), MemLevel.L1, c.cluster, "P^T, k-major",
                          DType.I8),
                "l": _row((1, c.br), MemLevel.L1, c.cluster, "row sums")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        fa = SnaxBingoKernelSimdFaSoftmaxArgs
        sbytes, pb = BEAT + c.bc * c.br * 2, c.bc * c.br
        arena = g.l1(f"{self.name}_arena", fa.arena_bytes(c.bc, 1))
        lay = fa.layout(c.bc, 1)
        p8 = g.l1(f"{self.name}_p8", (c.nt - 1) * pb + fa.p8_bytes(c.bc, c.br, fa.EXACT))
        seed_m = g.node("SeedM", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(bound["m"].handle,
                                                      at_offset(arena, lay["mrun"]), BEAT),
                        list(bound["m"].ends))
        seed_l = g.node("SeedL", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                        SnaxBingoKernelXdmaMemsetArgs(at_offset(arena, lay["lrun"]), BEAT,
                                                      SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO))
        nodes, sm = [seed_m, seed_l], []
        for i in range(c.nt):
            sm.append(g.node(f"Softmax{i}", ctx.simd, "__snax_bingo_kernel_simd_fa_softmax",
                             fa(at_offset(bound["s16"].handle, i * sbytes + BEAT),
                                at_offset(p8, i * pb), arena, c.bc, 1, tile_idx=i,
                                seed_state=0, geom_mode=fa.GEOM_SELF, score_scale=c.a_exp,
                                p_mode=fa.EXACT, p8_pitch=0),
                             [seed_m, seed_l] + list(bound["s16"].ends) + sm[-1:]))
        nodes += sm
        return BlockResult(
            outputs={"p": Port(self.outputs["p"], p8, (sm[-1],), name="p"),
                     "l": Port(self.outputs["l"], at_offset(arena, lay["lrun"]), (sm[-1],),
                               name="l")},
            inputs={"m": Port(self.inputs["m"], bound["m"].handle, (seed_m,), name="m"),
                    "s16": Port(self.inputs["s16"], bound["s16"].handle, (sm[0],), name="s16")},
            nodes=nodes)


@dataclass(frozen=True)
class MlaDimPVCfg:
    slot: int                  # this cluster's slot: O^T rows [slot d_v/G, (slot+1) d_v/G)
    G: int                     # the slots
    keys: int
    cap: int
    k_o: int
    a_n_f32bits: int
    # keys [0, k_static) come from `val`, the rest from `valf` (the appended rows, which only
    # the attention's chip's copy has); k_static = keys reads them all from `val`
    k_static: int
    cluster: int = 0
    br: int = 32
    d_v: int = 512


class MlaDimPV(Block):
    """One latent slice of the attention: O^T[rows] = V^T[rows, :keys] . P^T over EVERY key in
    one matmul (FP16 out of the D port at k_o, as the one-cluster attention's last tile), the
    gathered row sums added as a halving tree (slot i + slot i + n/2, n = G, G/2, .. 2), and
    the slice normalised, o~ = O16 (.) c / l.

      in   p    [keys, br] int8, every shard's P back to back (P^T, k-major), L1
           l    [G, br] fp16, every shard's row sums in slot order, L1
           val  [d_v, cap] int8 A, L3: the value copy keys [0, k_static) are read from
           valf the same, for keys [k_static, keys) (only when k_static < keys)
      out  ot   [d_v / G, br] fp16: this slice of o~^T"""

    name = "mla_dimpv"

    def __init__(self, cfg: MlaDimPVCfg = None, **params):
        self.cfg = cfg if cfg is not None else MlaDimPVCfg(**params)
        c = self.cfg
        d_s = c.d_v // c.G
        if c.d_v % c.G or d_s % 16 or c.keys % 4 or c.br != 32 or c.G & (c.G - 1):
            raise ValueError(f"MlaDimPV: d_v={c.d_v} over G={c.G} (a power of two) slots of "
                             f"whole 16-row blocks, keys={c.keys} a multiple of 4, br = 32")
        if not 0 < c.k_static <= c.keys or c.k_static % 4:
            raise ValueError(f"MlaDimPV: k_static={c.k_static} in (0, {c.keys}], 4-aligned")

    @property
    def d_s(self) -> int:
        return self.cfg.d_v // self.cfg.G

    @property
    def inputs(self) -> dict:
        c = self.cfg
        v = _row((c.d_v, c.cap), MemLevel.L3, None, "value copy", DType.I8, Layout.A)
        ins = {"p": _row((1, c.keys * c.br), MemLevel.L1, c.cluster, "P^T, k-major", DType.I8),
               "l": _row((c.G, c.br), MemLevel.L1, c.cluster, "row sums"),
               "val": v}
        if c.k_static < c.keys:
            ins["valf"] = v
        return ins

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"ot": _row((self.d_s, c.br), MemLevel.L1, c.cluster, "o~^T slice")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        pv_cls = SnaxBingoKernelGemmFaPvArgs
        d_s = self.d_s
        # V^T's A layout: a 16-row block is one run of 16 cap bytes, 4 keys per 64-B beat
        row, m0 = 16 * c.cap, c.slot * d_s // 16
        vb = g.l1(f"{self.name}_v", d_s * c.keys)
        ldv = [g.node("Ld_V", ctx.dm, "__snax_bingo_kernel_idma_2d_copy",
                      SnaxBingoKernelIdma2dCopyArgs(at_offset(bound["val"].handle, m0 * row),
                                                    vb, 16 * c.k_static, row, 16 * c.keys,
                                                    d_s // 16),
                      list(bound["val"].ends))]
        if c.k_static < c.keys:
            ldv.append(g.node("Ld_Vf", ctx.dm, "__snax_bingo_kernel_idma_2d_copy",
                              SnaxBingoKernelIdma2dCopyArgs(
                                  at_offset(bound["valf"].handle, m0 * row + 16 * c.k_static),
                                  at_offset(vb, 16 * c.k_static),
                                  16 * (c.keys - c.k_static), row, 16 * c.keys, d_s // 16),
                              list(bound["valf"].ends)))
        # the normalise reads its latch beat DIRECTLY BELOW O16
        o16 = g.l1(f"{self.name}_o16", BEAT + d_s * c.br * 2)
        pv = g.node("PV", ctx.gemm, "__snax_bingo_kernel_gemm_fa_pv",
                    pv_cls(vb, bound["p"].handle, 0, at_offset(o16, BEAT),
                           M=d_s // 16, K=c.keys // 4, N=c.br // 16,
                           flags=pv_cls.B_KMAJOR | pv_cls.D_FP16, corr_addr=0, d_shift=c.k_o),
                    ldv + list(bound["p"].ends))
        adds, src, n, deps = [], bound["l"].handle, c.G, list(bound["l"].ends)
        while n > 1:
            h = n // 2
            dst = g.l1(f"{self.name}_l{h}", h * BEAT)
            adds.append(g.node(f"AddL{h}", ctx.simd,
                               "__snax_bingo_kernel_simd_stream_elementwise",
                               SnaxBingoKernelSimdAddF16Args(src, at_offset(src, h * BEAT), dst,
                                                             rows=1, cols=h * c.br), deps))
            src, n, deps = dst, h, adds[-1:]
        ot = g.l1(f"{self.name}_ot", d_s * c.br * 2)
        nrm = g.node("Normalise", ctx.simd, "__snax_bingo_kernel_simd_mla_normalise",
                     SnaxBingoKernelSimdMlaNormaliseArgs(src, o16, at_offset(o16, BEAT), ot, d_s,
                                                         c.a_n_f32bits),
                     [pv] + adds[-1:] + (list(bound["l"].ends) if not adds else []))
        ins = {"p": Port(self.inputs["p"], bound["p"].handle, (pv,), name="p"),
               "l": Port(self.inputs["l"], bound["l"].handle, ((adds or [nrm])[0],), name="l"),
               "val": Port(self.inputs["val"], bound["val"].handle, (ldv[0],), name="val")}
        if c.k_static < c.keys:
            ins["valf"] = Port(self.inputs["valf"], bound["valf"].handle, (ldv[1],),
                               name="valf")
        return BlockResult(
            outputs={"ot": Port(self.outputs["ot"], ot, (nrm,), name="ot")},
            inputs=ins, nodes=ldv + [pv] + adds + [nrm])


class MlaDimOut(Block):
    """The attention's cluster's end of att_split_mode dim: the gathered o~^T slices ARE o~^T
    (slot order is row order), and its transpose -- MlaAttention's two outputs.

      in   ot  [d_v, br] fp16, L1        out  ot  the same;  xt  [br, d_v] fp16"""

    name = "mla_dim_out"

    def __init__(self, cfg: MlaShardCfg = None, **params):
        self.cfg = cfg if cfg is not None else MlaShardCfg(**params)

    @property
    def inputs(self) -> dict:
        c = self.cfg
        return {"ot": _row((c.d_v, c.br), MemLevel.L1, c.cluster, "o~^T")}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"ot": _row((c.d_v, c.br), MemLevel.L1, c.cluster, "o~^T"),
                "xt": _row((c.br, c.d_v), MemLevel.L1, c.cluster, "o~, a head per row")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c = self.cfg
        g = ctx.at(c.cluster)
        xt = g.l1(f"{self.name}_xt", c.br * c.d_v * 2)
        tr = g.node("Transpose", ctx.xdma, "__snax_bingo_kernel_xdma_transpose_2d",
                    SnaxBingoKernelXdmaTranspose2dArgs(bound["ot"].handle, xt, c.d_v, c.br, 2),
                    list(bound["ot"].ends))
        return BlockResult(
            outputs={"ot": Port(self.outputs["ot"], bound["ot"].handle, tuple(bound["ot"].ends),
                                name="ot"),
                     "xt": Port(self.outputs["xt"], xt, (tr,), name="xt")},
            inputs={"ot": Port(self.inputs["ot"], bound["ot"].handle, (tr,), name="ot")},
            nodes=[tr])
