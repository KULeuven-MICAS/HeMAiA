# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FlashAttention as DFG fragments: per-cluster shards, and the fold that joins them.

Moved here from workloads/fa_decode_4cluster so a larger graph can call it. The code is
the tuned pipeline unchanged -- node CREATION ORDER is dispatch order on this machine and
several orderings here are load-bearing (the head builder runs inside the per-cluster pass,
QK(i+1) is emitted before PV(i)), so this is a move, not a rewrite.

CONFIGURATION IS MODULE-GLOBAL, deliberately and with a limit. `configure(param)` sets the
~14 derived shapes the builders read. They are globals because that is how the tuned code
was written and rewriting 700 lines to thread a config object would risk the schedule for
no gain today. The consequence is real and worth stating: ONE attention shape per process.
A graph needing two different shapes needs the de-globalisation first.
"""

import json
import argparse
import pathlib
import hjson
import numpy as np
from dataclasses import dataclass, fields, replace

from ..comm.paths import add_sim_paths          # noqa: E402

ROOT_DIR = add_sim_paths("common", "gemm")

from bingo_kernel_args import (                           # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelXdma1dCopyArgs,
    SnaxBingoKernelXdmaMulticastArgs,
    SnaxBingoKernelXdmaMemsetArgs,
    SnaxBingoKernelSimdFaSoftmaxArgs,
    SnaxBingoKernelGemmFaQkArgs,
    SnaxBingoKernelGemmFaPvArgs,
    SnaxBingoKernelGemmPerfReportArgs,
    HostBingoKernelIdmaArgs,
    HostBingoKernelIdmaMultiArgs,
)

from bingo_mem_handle import BingoMemAlloc                 # noqa: E402
from sim_golden_models import block_gemm_golden_model      # noqa: E402

from ..comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                    PortSpec, at_offset, transfer)
from ..verify import checks

# Core placement, from the generated map (snax_core_roles_defs.h).
# The array these kernels were written for. configure() CHECKS the cluster cfg against
# it rather than trusting it, since every buffer size and tile count derives from it.
_MESH_DEFAULT = (16, 4, 16)

# A GENUINE CONSTANT, not a parameter -- nothing chooses it. It is a bit pattern: fp16
# -inf in both halves of a 32-bit word, for seeding the running max. Every actual
# parameter lives on FaCfg, and the comparison kinds live in checks.py under names.
_NEG_INF16 = 0xFBFFFBFF

def _pull_owner(cfg, j):
    """Which cluster fetches tile j from main memory; the rest pull it from that cluster."""
    return cfg.clusters[j % cfg.ncl] if cfg.pull_rotate else cfg.clusters[0]


def _bcast_owner(cfg, j, skew=0):
    """Which cluster's xDMA issues the broadcast of KV tile j.

    With BCAST_SPREAD the skew rotates tile ownership; without it the skew still selects a
    FIXED engine, which is how K and V get one issuer each. That two-issuer split is the
    useful middle point: it halves the head serialisation without asking four clusters to
    issue at once, and it is also the bisect the adapter hang wants -- if TWO concurrent
    issuers already hang, the star multicast is irrelevant and any pair reproduces it.
    """
    return (cfg.clusters[(j + skew) % cfg.ncl] if cfg.bcast_spread
            else cfg.clusters[skew % cfg.ncl])


@dataclass(frozen=True)
class FaCfg:
    """The single source of truth for an attention instance.

    Four kinds of value, kept apart because they answer to different questions:

      HARDWARE   mesh, monoid_slots -- properties of the silicon, not choices. `mesh` is
                 checked against the cluster cfg by configure(); the default is the array
                 the kernels were written for.
      ARGUMENTS  nkv, nq, clusters, decomp, score_shift_extra -- the actual parameters.
      TUNING     pipeline depths and where operands come from. None of them changes the
                 ANSWER, and every default is the measured best on this quadrant, so a
                 caller who just wants attention never names one. validate() refuses a
                 knob the chosen decomposition does not read.
      DERIVED    everything else, below, as properties. bc/br/dhead and the second
                 matmul's shape are arithmetic over the groups above, so they cannot drift
                 from them.
    """
    m: int
    k: int
    n: int
    nkv: int
    nq: int = 1
    # WHICH CLUSTERS THE PIPELINES RUN ON. A tuple, like every other block's placement,
    # so the layer states where attention goes rather than only how wide it is.
    clusters: tuple = (0, 1, 2, 3)
    decomp: str = "headpar"
    # WHERE Q, K AND V ARE HANDED OVER. The per-tile loads are cluster iDMA and xDMA reads
    # of main memory, so L3 is where they have to be by the time those run; an operand in
    # the memory-chiplet pool is hoisted once in the prologue instead of fetched per tile
    # across the D2D link. One field for all three: they come from one staging pass.
    operand_level: MemLevel = MemLevel.L3
    score_shift_extra: int = 0

    # ---- hardware: checked, not chosen --------------------------------------------------
    mesh: tuple = (16, 4, 16)
    monoid_slots: int = 8

    # ---- internal performance tuning ----------------------------------------------------
    # NONE OF THE BELOW CHANGES THE ANSWER. They change where operands come from and how
    # deep the pipeline runs; every default is the measured best on this quadrant. They are
    # grouped by which decomposition READS them, because a knob the chosen decomp never
    # looks at is silently inert -- validate() refuses one rather than let a sweep report
    # "no effect" for a knob that was never consulted.

    # Read by BOTH decompositions -- pipeline depth, which is a property of the per-cluster
    # loop and so exists however the work was split.
    nscore: int = 2                # score buffers; 2 is the minimum that overlaps QK/PV
    nkbuf: int = 2                 # K prefetch depth
    nvbuf: int = 2                 # V prefetch depth
    v_push_tiles: tuple = ()       # tiles the host pushes instead of the cluster pulling
    v_push_pairs: bool = False     # one push per PAIR of tiles, on the system iDMA

    # Read ONLY under decomp="headpar". Under headpar every cluster wants the SAME K and V
    # -- four query heads over one sequence -- so one cluster can read memory and the rest
    # take it from that L1. Under kvsplit the shards are disjoint: there is no shared tile
    # to pull, multicast or push, and each of these has nothing to act on.
    v_pull_from_cl0: bool = True   # read once from memory, fan out over the fabric
    k_pull_from_cl0: bool = True
    pull_rotate: bool = True       # rotate which L1 serves tile j
    pull_skip_first: int = 0       # leading tiles each cluster reads itself
    bcast_k: bool = False          # multicast instead of pull; needs k_pull_from_cl0=False
    bcast_v: bool = False          # ... and bcast_v needs v_pull_from_cl0=False
    bcast_spread: bool = False
    bcast_v_skew: int = 0          # only consulted when bcast_v is on
    bcast_skip_first: bool = False
    bcast_warmup: bool = False     # MEASURED AND REFUTED: costs the issuer, saves nothing
    v_on_xdma: bool = False        # never with a broadcast; validate() refuses that pair
    # Host pushes on the system iDMA, the second wide pipe. MEASURED: real bandwidth, but
    # its dispatch sits on a chain four clusters wait on, so none of these wins.
    k_push_all: bool = False
    k_push_owner: bool = False     # only consulted while k_pull_from_cl0 is on
    v_push_owner: bool = False     # only consulted while v_pull_from_cl0 is on

    # Read ONLY under decomp="kvsplit", where all four clusters issue their first V at the
    # same instant and queue on one cold config.
    stagger_first_v: bool = False

    # instrumentation and debug -- these add nodes, so they are not free
    measure_array: bool = False
    o_check_elems: int = 512
    debug_loose_ml: bool = False
    debug_per_tile_m: bool = False
    debug_s16_compare: bool = False
    s16_cmp_buf: int = 2

    # ---- derived ---------------------------------------------------------------------
    @property
    def mesh_row(self): return self.mesh[0]
    @property
    def tile_size(self): return self.mesh[1]
    @property
    def mesh_col(self): return self.mesh[2]
    @property
    def nscore_war(self): return self.nscore
    @property
    def bc(self): return self.m * self.mesh[0]          # key columns per tile
    @property
    def br(self): return self.n * self.mesh[2]          # query rows
    @property
    def dhead(self): return self.k * self.mesh[1]       # head dimension, a model property
    # Shape 2, O^T = V^T.P^T. Bc and d are independent, so the two matmuls differ in M, K.
    @property
    def s2_m(self): return self.dhead // self.mesh[0]
    @property
    def s2_k(self): return self.bc // self.mesh[1]
    @property
    def s2_n(self): return self.n                       # N is Br/meshCol either way
    @property
    def qshift(self): return qshift(self.dhead) + self.score_shift_extra
    @property
    def ncl(self): return len(self.clusters)
    @property
    def nkv_per(self): return self.nkv // self.ncl

    # ---- the constraints the kernels impose ------------------------------------------
    def validate(self) -> "FaCfg":
        # THE ARRAY IS CHECKED FIRST, because every other rule here is DOWNSTREAM of it.
        # Br must be 32 because a query-row vector is one SIMD beat -- but "32" is
        # mesh_col * 2 / 2, so on a (32, 2, 32) array the Br rule fires first and reports
        # "Br must be 32, grow Bc instead", which is advice for a problem the caller does
        # not have. Naming the array first turns a confusing constraint into the real one.
        if tuple(self.mesh) != _MESH_DEFAULT:
            raise ValueError(
                f"this attention is written for (Mu, Ku, Nu) = {_MESH_DEFAULT} but the "
                f"cluster declares {tuple(self.mesh)}. Bc, Br and d all derive from the "
                f"array, so every descriptor and every golden would be wrong -- and "
                f"neither would fault.")
        if self.decomp not in ("headpar", "kvsplit"):
            raise ValueError(f"decomp={self.decomp!r} must be 'headpar' or 'kvsplit'")
        # THE PER-CLUSTER LISTS ARE POSITIONAL. Every buffer table, shard list and operand
        # slice in this file is built in cluster order and then read back by cluster id,
        # so a placement whose position is not its id would index one cluster's buffers
        # with another's number and compute a wrong answer nothing faults on. Lifting this
        # is an audit of every such index, not a parameter change, so it is refused here
        # rather than half-supported.
        if tuple(self.clusters) != tuple(range(self.ncl)):
            raise ValueError(
                f"clusters={tuple(self.clusters)}: this attention places its pipelines on "
                f"clusters 0..n-1 and indexes its per-cluster tables by position. Pass a "
                f"contiguous tuple starting at 0.")
        # Br IS NOT A FREE KNOB. Every per-query-row vector in the softmax arena -- the
        # running max, the running sum, every correction factor -- is ONE SIMD beat, and a
        # score row is one beat too. A beat is SIMD_WIDTH = 512 b, so it holds exactly
        # 512/16 = 32 fp16 lanes and Br must be 32. A larger Br does not fault: it writes
        # the first 32 rows scrambled and leaves the rest zero, which only a host check
        # catches. Raising it means a second temporal dimension in all eleven SIMD task
        # shapes and in the arena layout -- a kernel change, not a parameter. Bc is the
        # knob that is actually free, and L1 is what bounds it.
        if self.br != 32:
            raise ValueError(
                f"Br={self.br} (n={self.n}) but the softmax kernel packs one query-row "
                f"vector per {self.mesh[2] * 2}-byte SIMD beat, so Br must be 32. Grow Bc "
                f"(m) instead, or rework simd_fa_layout() and the SIMD task shapes.")
        # Bc IS the beat count of the score tile -- the transposed layout puts one key, all
        # Br queries, in each 64-B beat -- and the quantiser packs 2:1, so it must be even.
        if self.bc % 2:
            raise ValueError(f"Bc={self.bc} must be even: the quantiser packs two beats "
                             f"into one")
        if self.nkv % self.ncl:
            raise ValueError(
                f"nkv={self.nkv} must divide across {self.ncl} clusters: each cluster "
                f"owns a disjoint run of KV tiles and the merge assumes every shard covers "
                f"the same count.")
        if self.br % self.monoid_slots:
            raise ValueError(
                f"Br={self.br} must be a multiple of monoid_slots={self.monoid_slots}: the "
                f"monoid packs that many query rows per beat and a partial beat would fold "
                f"lanes the pack never wrote.")
        # A stream needs at least two buffers to BE a stream. The WAR edge is written as
        # "V(j) waits on PV(j - nvbuf)", so at nvbuf = 1 it reads V(j) waits on PV(j) --
        # the very task that consumes what V(j) is about to write. The builder does not
        # see a cycle; it indexes a pv[] entry that does not exist yet and dies with an
        # IndexError from deep inside the loop, which says nothing about the cause.
        for name, v in (("nkbuf", self.nkbuf), ("nvbuf", self.nvbuf),
                        ("nscore", self.nscore)):
            if v < 2:
                raise ValueError(
                    f"{name}={v}: a single buffer makes the producer wait on the consumer "
                    f"of the same buffer, which is a cycle, not a schedule. Use 2 or more.")
        # A KNOB THE CHOSEN DECOMPOSITION NEVER READS IS A LIE, NOT A DEFAULT. Each one
        # below changes the emitted graph under the decomp it is listed with and changes
        # nothing under the other. Unchecked, setting one on the wrong decomp builds, runs,
        # and gives the same cycle count as not setting it -- which reads as "the
        # optimisation does not help" rather than "it was never applied".
        _ONLY = {"headpar": ("v_pull_from_cl0", "k_pull_from_cl0", "pull_rotate",
                             "pull_skip_first", "bcast_k", "bcast_v", "bcast_spread",
                             "bcast_v_skew", "bcast_skip_first", "bcast_warmup",
                             "v_on_xdma", "k_push_all", "k_push_owner", "v_push_owner"),
                 "kvsplit": ("stagger_first_v",)}
        other = "kvsplit" if self.decomp == "headpar" else "headpar"
        dflt = {f.name: f.default for f in fields(self)}
        stuck = [k for k in _ONLY[other] if getattr(self, k) != dflt[k]]
        if stuck:
            raise ValueError(
                f"{', '.join(stuck)} {'is' if len(stuck) == 1 else 'are'} read only under "
                f"decomp={other!r}, but this cfg is decomp={self.decomp!r}. Under "
                f"{self.decomp!r} the builder never consults "
                f"{'it' if len(stuck) == 1 else 'them'}, so the setting would change "
                f"nothing and the arm would look like a measurement of a no-op.")
        if self.v_on_xdma and (self.bcast_k or self.bcast_v):
            raise ValueError(
                "v_on_xdma with a broadcast: a cluster cannot absorb an incoming remote "
                "write while its own xDMA transfers, and every receiving xDMA wedged.")
        return self

    # ---- constructors ------------------------------------------------------------------
    @classmethod
    def from_params(cls, param: dict, mesh=(16, 4, 16), **kw) -> "FaCfg":
        """From a params dict (M/K/N/NKV/ACTIVE_CLUSTERS/DECOMP)."""
        ncl = int(param.get("ACTIVE_CLUSTERS", param.get("num_clusters", 4)))
        if not 1 <= ncl <= int(param.get("num_clusters", ncl)):
            raise ValueError(f"ACTIVE_CLUSTERS={ncl} must be between 1 and num_clusters")
        return cls(m=int(param["M"]), k=int(param["K"]), n=int(param["N"]),
                   nkv=int(param["NKV"]), nq=int(param.get("NQ", 1)),
                   clusters=tuple(range(ncl)),
                   decomp=str(param.get("DECOMP", "headpar")),
                   score_shift_extra=int(param.get("SCORE_SHIFT_EXTRA", 0)),
                   mesh=tuple(mesh), **kw).validate()

    @classmethod
    def from_shape(cls, *, bc, br, dhead, mesh=(16, 4, 16), **kw) -> "FaCfg":
        """From the tile in ELEMENTS, which is how a caller thinks about a layer.

        Bc key columns, Br query rows, d the head dimension. The mesh-tile counts every
        GEMM descriptor wants are derived, so a caller never states m/k/n.
        """
        mr, ts, mc = mesh
        for nm, val, unit in (("bc", bc, mr), ("br", br, mc), ("dhead", dhead, ts)):
            if val % unit:
                raise ValueError(f"{nm}={val} does not tile the array ({unit}).")
        return cls(m=bc // mr, n=br // mc, k=dhead // ts, mesh=tuple(mesh), **kw).validate()

    @classmethod
    def from_ports(cls, *, q, k, mesh=(16, 4, 16), **kw) -> "FaCfg":
        """From the SHAPES of the bound inputs, which is the direction composition wants.

        A query port of [Br, d] and a key port of [Bc, d] plus the array give m, n and k
        directly, so the block's shape follows from what it is fed rather than being
        declared twice and hoped to agree.
        """
        br, dq = q
        bc, dk = k
        if dq != dk:
            raise ValueError(f"q and k disagree on the head dimension: {dq} vs {dk}.")
        mr, ts, mc = mesh
        for nm, val, unit in (("Br", br, mc), ("Bc", bc, mr), ("d", dq, ts)):
            if val % unit:
                raise ValueError(f"{nm}={val} does not tile the array ({unit}).")
        return cls(m=bc // mr, n=br // mc, k=dq // ts, mesh=tuple(mesh), **kw).validate()


def mesh_from_hwcfg(hwcfg_path, array_shape: int = 0) -> tuple:
    """(Mu, Ku, Nu) as the RTL was elaborated, read from the cluster hjson.

    The kernels' idea of the array and the descriptors' idea of it must be the same one.
    Trusting a literal instead is a silent wrong answer, not a fault: every buffer size and
    every tile count derives from the mesh, so a mismatch produces a graph that builds,
    runs, and computes the wrong thing.
    """
    import hjson
    with open(hwcfg_path) as f:
        hw = hjson.loads(f.read())
    unroll = (hw["snax_versacore_core_template"]["snax_acc_cfg"][0]
                ["snax_versacore_spatial_unrolling"][0][array_shape])
    return tuple(int(x) for x in unroll)


def qshift(d):
    """How far to bound the INT8 operands so a score cannot leave FP16.

    A score is a sum of d products of two shifted INT8s, so |S| <= (128>>q)^2 * d.
    Int32ToFp16 SATURATES past 65504 and exp(inf - inf) is NaN, so an overflowing row
    fails the softmax rather than degrading. Real attention divides by sqrt(d) for the
    same reason; bounding the operands is cheaper here and needs no extra pass.
    """
    for q in range(8):
        if (128 >> q) ** 2 * d <= 65504:
            return q
    raise ValueError(f"no INT8 shift keeps d={d} scores inside FP16")




def build_data(cfg, a=None, b=None, v=None, seed=42):
    """Operands plus the float model of the softmax over ONE shard's tile.

    Follows the hardware's own sequence at the precision each stage works in:

        S       exact INT32 out of the mesh, converted to FP16 on the D32 port
        m       max over KEYS, per query row
        P       exp(S16 - m), FP32 internally, stored FP16
        rowsum  summed over KEYS in the FP32 accumulator, narrowed once to FP16

    Every KV tile inside a shard is fed the SAME K, so the running maximum stops moving
    after tile 0 and the shard's final m is this tile's m; rowsum likewise.

    The operands are arguments rather than locals so build_shards() can give every cluster
    a DIFFERENT K while they share one Q and one V -- see the note there on why identical
    shards would make the cross-cluster fold untestable.
    """
    rng = np.random.RandomState(seed)
    # Block operand layouts, as block_gemm_golden_model reads them and as the streamer
    # descriptors walk them: A is [M][K][meshRow][tileSize], B is [N][K][meshCol][tileSize].
    if a is None:
        a = rng.randint(-128, 127, size=cfg.m * cfg.k * cfg.mesh_row * cfg.tile_size).astype(np.int8) >> cfg.qshift
    if b is None:
        b = rng.randint(-128, 127, size=cfg.n * cfg.k * cfg.mesh_col * cfg.tile_size).astype(np.int8) >> cfg.qshift
    # V for the second matmul: [S2_M][S2_K][meshRow][tileSize]. Its value never reaches a
    # check (O is a cycle measurement), but it must be a real operand so the matmul does
    # the work -- a zero V would still take the same cycles, but a denormal-free operand
    # keeps the array off any slow path.
    if v is None:
        v = rng.randint(-128, 127,
                        size=cfg.s2_m * cfg.s2_k * cfg.mesh_row * cfg.tile_size).astype(np.int8) >> cfg.qshift

    d32 = block_gemm_golden_model(
        cfg.m, cfg.k, cfg.n, cfg.mesh_row, cfg.tile_size, cfg.mesh_col, a, b, 0, 0,
        np.zeros(cfg.m * cfg.n * cfg.mesh_row * cfg.mesh_col, dtype=np.int64))
    # Block order [M][N][meshRow][meshCol] -> [key][query], which is how the D port lays
    # the tile down (its [4, 8] channel grouping interleaves the two N blocks inside one
    # key row) and how the SIMD core reads it back.
    s = np.asarray(d32).reshape(cfg.m, cfg.n, cfg.mesh_row, cfg.mesh_col)
    s = s.transpose(0, 2, 1, 3).reshape(cfg.bc, cfg.br)

    s16 = s.astype(np.float16)
    m = s16.max(axis=0)
    p = np.exp(s16.astype(np.float32) - m.astype(np.float32)).astype(np.float16)
    rowsum = p.astype(np.float32).sum(axis=0).astype(np.float16)

    # O, the accumulated PV output. This is the ONLY quantity that exercises the GEMM's
    # accumulation: m and rowsum are per-tile values that are IDENTICAL on every tile here,
    # so a run that drops tiles still passes both of them.
    #
    # P leaves the softmax quantised to INT8 -- exp(S-m) is in [0,1], so the baked scale is
    # 127.0 (BINGO_SIMD_I8_SCALE_UNIT) -- and PV contracts over KEYS: O^T = V^T . P^T.
    # P goes in with NO permutation, and that is the whole subtlety.
    #
    # `s` above is ALREADY the score buffer's memory order. The QK GEMM's D port writes with
    # a spatial stride of one key row (Dsl = {bw/8, key_row}), which interleaves the two N
    # blocks as it writes -- "a beat is exactly one key, all Br queries" in gemm_fa.h. So
    # memory holds [key][query], and the transpose in build_data(cfg) is how the golden model's
    # [M][N][meshRow][meshCol] output is brought INTO that order, not out of it.
    #
    # PV then reads that same flat buffer as B = [N][K][meshCol][tileSize]
    # (Btb = {K, N, M}, Bts = {b_tile, K*b_tile, 0} -- broadcast over M, b_tile = 64 B).
    # block_gemm_golden_model takes B as a flat array in exactly that convention, so the
    # quantised [key][query] array IS the operand. Permuting it first yields the right
    # VALUES in the wrong PLACES.
    p_b = np.clip(np.rint(p.astype(np.float32) * 127.0),
                  -128, 127).astype(np.int8).reshape(-1)
    o_tile = block_gemm_golden_model(
        cfg.s2_m, cfg.s2_k, cfg.s2_n, cfg.mesh_row, cfg.tile_size, cfg.mesh_col, v, p_b, 0, 0,
        np.zeros(cfg.s2_m * cfg.s2_n * cfg.mesh_row * cfg.mesh_col, dtype=np.int64))
    # Every KV tile is fed the same bytes, so m never moves and every correction factor is
    # exactly 1 -- the accumulator is NKV copies of one tile's contribution.
    o = np.asarray(o_tile, dtype=np.int64) * cfg.nkv_per

    # ---- and now put O where the D32 port ACTUALLY writes it ---------------------------
    #
    # This block is the difference between an O check that passes and one that reports
    # 3338 of 4096 elements wrong while the hardware is perfectly correct.
    #
    # The C/D spatial map is chosen so the FP16 score tile lands row-major -- one key per
    # 64 B beat, which is what the LANEWISE reduce needs. C and D share those strides and
    # the INT32 side is twice as wide, so at INT32 the SAME strides do NOT come out
    # row-major: O lands PERMUTED. That is deliberate and harmless to the arithmetic (C and
    # D permute identically, so O += P.V still accumulates against itself -- the cluster
    # cfg says exactly this), but it means a golden in canonical block order is not
    # comparable with what the host reads back.
    #
    # The map is DERIVED from the descriptors, not assumed: the array serialises a
    # meshRow x meshCol block into chunks of serial_c_d_width, each chunk into channels of
    # bankWidth, and the AGU places channel i at sl0*(i%4) + sl1*((i/4)%4). Ported from the
    # reference's data/datagen.py, which is the model that makes the cluster app's own
    # exact-equality O check pass.
    BANK_W, OUT_W, SERIAL_CD = 64, 32, 1024   # bits; snax_versacore_serial_c_d_width
    SBOUNDS = [4, 4]                          # data_reader_writer_params.spatial_bounds[0]
    slstride = [BANK_W // 8, cfg.s2_n * cfg.mesh_col * 16 // 8]
    ts0, ts1 = SERIAL_CD * cfg.s2_n // 8, cfg.mesh_col * 16 // 8

    def chan_off(ch):
        off, rem = 0, ch
        for dim, bnd in enumerate(SBOUNDS):
            off += slstride[dim] * (rem % bnd)
            rem //= bnd
        return off

    def scatter(width, blocks, ts2, elem_bytes):
        """Byte offset of every (block, n, row, col) element, in the port's order."""
        per_chunk = SERIAL_CD // width        # elements in one serialised chunk
        per_chan = BANK_W // width            # elements in one channel
        out = {}
        for mm in range(blocks):
            for nn in range(cfg.s2_n):
                for r in range(cfg.mesh_row):
                    for c in range(cfg.mesh_col):
                        e = r * cfg.mesh_col + c
                        out[(mm, nn, r, c)] = (
                            mm * ts2 + nn * ts1 + (e // per_chunk) * ts0
                            + chan_off((e % per_chunk) // per_chan)
                            + (e % per_chan) * elem_bytes)
        return out

    # SELF-TEST, and it is the load-bearing part: run the same model at FP16 and it must
    # reproduce the plain [key][query] row-major layout that `s` above already assumes. If
    # the serialisation model were wrong this assert fires instead of the golden silently
    # disagreeing with the hardware.
    for (mm, nn, r, c), byte in scatter(16, 1, cfg.s2_n * 16 * cfg.mesh_row * cfg.mesh_col // 8, 2).items():
        want = ((mm * cfg.mesh_row + r) * (cfg.s2_n * cfg.mesh_col) + nn * cfg.mesh_col + c) * 2
        assert byte == want, "D32 address model disagrees with the FP16 row-major layout"

    ts2 = cfg.s2_n * OUT_W * cfg.mesh_row * cfg.mesh_col // 8
    o_can = o.reshape(cfg.s2_m, cfg.s2_n, cfg.mesh_row, cfg.mesh_col)
    o_mem = np.zeros(o.size, dtype=np.int64)
    seen = np.zeros(o.size, dtype=bool)
    for (mm, nn, r, c), byte in scatter(OUT_W, cfg.s2_m, ts2, 4).items():
        w = byte // 4
        assert byte % 4 == 0 and not seen[w], "D32 INT32 address map is not a bijection"
        seen[w] = True
        o_mem[w] = o_can[mm, nn, r, c]
    assert seen.all(), "D32 INT32 address map does not cover the output"
    return a, b, v, m, rowsum, o_mem.astype(np.int32)


def build_shards(cfg):
    """NCL shards with DIFFERENT K, plus the global merge the fabric is supposed to compute.

    Every shard gets its own K and therefore its own (m_c, l_c). That is not incidental:
    with one shared K -- which is what the single-cluster workload uses, since there the
    recurrence just repeats one tile -- all four partials are identical, and a fold that
    silently dropped three of them, or returned the collector's own operand untouched,
    would produce exactly the right answer. Distinct shards are what make the gather's
    result evidence that the gather happened.

    Q and V are shared, as they are in the real decomposition: splitting KV leaves the
    query tile common to every cluster.

    The merge itself is the online-softmax combine the monoid junction implements:

        m* = max_c m_c              l* = sum_c exp(m_c - m*) * l_c

    computed here in FP32 on FP16 inputs, mirroring the device: the arena holds m and l in
    FP16, pack_fa_partial widens the bit pattern to FP32, and the junction folds in FP32.
    """
    rng = np.random.RandomState(1234)
    b = rng.randint(-128, 127, size=cfg.n * cfg.k * cfg.mesh_col * cfg.tile_size).astype(np.int8) >> cfg.qshift
    v = rng.randint(-128, 127,
                    size=cfg.s2_m * cfg.s2_k * cfg.mesh_row * cfg.tile_size).astype(np.int8) >> cfg.qshift

    if cfg.decomp == "headpar":
        # ONE K and ONE V for the whole group -- that IS the GQA relation, not a
        # simplification: the group's query heads share a KV head by construction. What
        # varies per cluster is Q, so every cluster still gets its own (m_c, l_c) and the
        # per-shard checks stay evidence that the shard ran. There is nothing to fold.
        #
        # The distinct-shard argument the kvsplit docstring makes does not apply here for
        # the opposite reason: no fold is being tested, so identical partials would prove
        # nothing either way. Distinct Q is what keeps the four checks independent.
        a_shared = (rng.randint(-128, 127, size=cfg.m * cfg.k * cfg.mesh_row * cfg.tile_size)
                       .astype(np.int8) >> cfg.qshift)
        shards, b_list = [], []
        for c in cfg.clusters:
            b_c = (rng.randint(-128, 127, size=cfg.n * cfg.k * cfg.mesh_col * cfg.tile_size)
                      .astype(np.int8) >> cfg.qshift)
            b_list.append(b_c)
            shards.append(build_data(cfg, a=a_shared, b=b_c, v=v))
        m_c = np.stack([np.asarray(sh[3], dtype=np.float16) for sh in shards])
        l_c = np.stack([np.asarray(sh[4], dtype=np.float16) for sh in shards])
        # a_list holds the SAME array object NCL times so stage() can see, by identity,
        # that one staged copy serves every cluster.
        # O PER CLUSTER, not shards[0]'s. Under headpar every cluster gets its own b_c
        # (its own query head), so every cluster's O is different and one golden would
        # only ever have validated cluster 0.
        return [a_shared] * cfg.ncl, b_list, v, m_c, l_c, None, [sh[5] for sh in shards]

    shards = []
    for c in cfg.clusters:
        a_c = (rng.randint(-128, 127, size=cfg.m * cfg.k * cfg.mesh_row * cfg.tile_size)
                  .astype(np.int8) >> cfg.qshift)
        shards.append(build_data(cfg, a=a_c, b=b, v=v))

    m_c = np.stack([np.asarray(sh[3], dtype=np.float16) for sh in shards])     # [NCL, BR]
    l_c = np.stack([np.asarray(sh[4], dtype=np.float16) for sh in shards])

    # The merged golden belongs to the FOLD, not to the shards, so it is built by the
    # module that folds. Its lane packing is the junction's, and nothing here should have
    # to know that geometry.
    from .gather import merged_golden
    merged = merged_golden(cfg, m_c, l_c)

    a_list = [sh[0] for sh in shards]
    # b is shared under kvsplit; the caller indexes per cluster either way.
    return a_list, [b] * cfg.ncl, v, m_c, l_c, merged, [sh[5] for sh in shards]


def stage(cfg, st, a, b, v, m, rowsum, o):
    """Hand every array to the staging helper and return the handles.

    WHERE these land is the platform's business, not the caller's: a config with a
    memory chiplet gets a mempool.bin, one without gets C arrays in the host image. See
    util/sim/common/bingo_data_staging.py -- addressing a memory chiplet that a config
    does not have reads unmapped memory rather than faulting, which surfaces as an
    arithmetic bug a long way from the cause.
    """
    # ONE ARRAY PER TENSOR, NOT ONE PER CLUSTER. The block takes q, k and v as whole
    # tensors and slices them itself, so what it needs here is a single contiguous buffer
    # per tensor -- cluster c's tile is then a byte offset into it rather than a separate
    # symbol. That is also what a real layer hands over: a projection writes one Q, not
    # four.
    #
    # An operand every cluster SHARES is staged once and stays one tile; only a genuinely
    # per-cluster one is concatenated. Sharing is decided by object identity, which
    # build_shards() already expresses by handing back the same array object N times, so
    # it is not restated here.
    def put_stacked(tag, arrays, ctype, conv):
        shared = len({id(x) for x in arrays}) == 1
        if shared:
            return st.put(tag, ctype, conv(arrays[0])), 1
        flat = np.concatenate([np.asarray(conv(a)).reshape(-1) for a in arrays])
        return st.put(tag, ctype, flat), len(arrays)

    a_h, _ = put_stacked("fa_k8", a, "int8_t", lambda x: np.asarray(x).astype(np.int8))
    b_h, _ = put_stacked("fa_q8", b, "int8_t", lambda x: np.asarray(x).astype(np.int8))

    return {
        "a": a_h,
        "b": b_h,
        "v": st.put("fa_v8", "int8_t", v.astype(np.int8)),
        # The score matmul's C is a zero BIAS and the O accumulator starts at zero. The
        # larger of the two is C, so one region serves both -- and on the host path it
        # costs no image bytes at all, because an uninitialised array lands in .bss.
        "zero": st.put_zeros("fa_zero", "int32_t", cfg.m * cfg.n * cfg.mesh_row * cfg.mesh_col),
        # Per-shard goldens, one per cluster. Keeping them is what separates "the fold is
        # wrong" from "a shard is wrong": a cluster whose recurrence quietly did nothing
        # still produces a well-formed partial, and the merged result alone cannot say
        # which of the four it came from.
        "m": [st.put(f"fa_m_golden_c{c}", "uint16_t",
                     np.asarray(m[c]).astype(np.float16).view(np.uint16))
              for c in cfg.clusters],
        "rowsum": [st.put(f"fa_rowsum_golden_c{c}", "uint16_t",
                          np.asarray(rowsum[c]).astype(np.float16).view(np.uint16))
                   for c in cfg.clusters],
        "o": [st.put(f"fa_o_golden_c{c}", "int32_t", np.asarray(o[c]).astype(np.int32))
              for c in cfg.clusters],
    }


def stage_merged(st, merged):
    """The gathered (m*, l*), in the junction's own lane order, as FP32."""
    return st.put("fa_ml_merged_golden", "float", np.asarray(merged, dtype=np.float32))

def _alloc_cluster(ctx, cfg, c):
    """Every L1 buffer one cluster's pipeline owns, and nothing else -- no nodes.

    Split out of _build_cluster because a BROADCAST load has to name its destinations in
    all four clusters, so every cluster's buffers must exist before the first load node is
    created. This moves no address: the compiler sorts handles by NAME when it lays out a
    heap (bingo_dfg._collect_memory_handles), so creation order across clusters -- and
    within one -- does not decide the layout. The ordering comment below is about which
    NAME sorts first, and that is unchanged.
    """
    g = ctx.at(c)
    # ---- L1 ---------------------------------------------------------------------------
    # Allocated FIRST, before the score buffers, and that order is load-bearing: the negate
    # task writes -m_new to rmax (here) and to the one-beat prefix of the live score buffer
    # (below) as a single two-beat shape, and a shape stride is unsigned. Arena first keeps
    # it positive.
    # ONE ARENA PER QUERY TILE. Each carries its own independent (m, l, O) recurrence,
    # which is exactly why NQ costs nothing in correctness: the query tiles never interact.
    arena = [g.l1(f"fa_arena_{q}",
                  SnaxBingoKernelSimdFaSoftmaxArgs.arena_bytes(cfg.bc, cfg.dhead))
             for q in range(cfg.nq)]

    # K and V are DOUBLE BUFFERED because they STREAM: one pair per KV tile, refilled from
    # main memory while the previous tile computes. Every tile is fed the same bytes -- the
    # golden relies on that, and m stops moving after tile 0 -- but they are RE-FETCHED per
    # tile, so the memory traffic is what a real KV stream would cost. Loading once and
    # replaying leaves the iDMA idle and makes any utilisation figure exclude memory.
    # TWO K buffers. A third would let the loads run one tile further ahead, which is the
    # right diagnosis -- every QK is gated by its own K load -- but it also raises the
    # number of simultaneously-ready DMA nodes past what the manager's ready set tolerates,
    # and the run wedges in teardown. Revisit only alongside that limit.
    k8 = [g.l1(f"fa_k8_{i}", cfg.m * cfg.k * cfg.mesh_row * cfg.tile_size) for i in range(cfg.nkbuf)]
    q8 = [g.l1(f"fa_q8_{q}", cfg.n * cfg.k * cfg.mesh_col * cfg.tile_size)      # B of the score matmul
          for q in range(cfg.nq)]
    v8 = [g.l1(f"fa_v8_{i}", cfg.s2_m * cfg.s2_k * cfg.mesh_row * cfg.tile_size) for i in range(cfg.nvbuf)]
    # The score matmul masks every C channel off (see gemm_fa.h), so nothing is ever
    # READ through this pointer -- the AGU walks addresses that are never dereferenced.
    # It exists only because the descriptor needs a base pointer. One block, not M*N.
    # fa_cz REMOVED. Every C channel of the score matmul is masked, so the AGU walks
    # addresses that are never dereferenced -- and the kernel already substitutes D_addr
    # when C_addr is 0 (the same path PV's first tile takes). A whole 1,024 B allocation
    # existed only to give that never-read pointer somewhere to point.
    cz = 0
    # TWO score buffers. A third lets QK run three deep and does remove the WAR edge it
    # targets, but it loses overall: the extra QK streams A, B and D through TCDM alongside
    # the softmax, and the softmax is the critical path. Slack bought for the GEMM is paid
    # for in SIMD bandwidth.
    # ONE BEAT OF HEADROOM in front of each score buffer. The softmax's fused exp pass reads
    # [-m_new][the tile] as a single contiguous stream. Reserving the beat here lets that
    # pass read the GEMM's own output buffer directly, instead of copying the whole tile
    # into the arena beside a -m_new slot -- a copy comparable in size to the K/V stream
    # itself. The GEMM writes its D at +64; the beat below it is the prefix.
    s16 = [g.l1(f"fa_s16_{i}", 64 + cfg.bc * cfg.br * 2) for i in range(cfg.nscore)]  # score tile, fp16
    # ONE TRAILING BEAT past the quantised P. The fused exp pass emits bc/2 INT8 beats and
    # then the tapped row sum, unnarrowed, as ONE contiguous stream -- and no shape can span
    # two allocations, so the row sum lives here rather than in the arena.
    p8 = [g.l1(f"fa_p8_{i}", cfg.bc * cfg.br + 64) for i in range(cfg.nscore)]   # quantised P + row sum
    # One arena, one Q buffer and one O accumulator PER QUERY TILE -- that is the
    # entire cost of the reuse. K, V, the score tile and P stay single sets: they are
    # exactly what we are trying not to re-read.
    oacc = [g.l1(f"fa_oacc32_{q}", cfg.br * cfg.dhead * 4) for q in range(cfg.nq)]
    # Destination of the BCAST_WARMUP multicast. Its own allocation rather than a corner of
    # fa_warm_buf: that one is the SIMD warm-up's, and a broadcast landing in it while the
    # SIMD warm-up reads it would be a data race for no reason. 64 B x 4 clusters.
    bwarm = g.l1("fa_bcast_warm", 64)
    return dict(arena=arena, k8=k8, q8=q8, v8=v8, cz=cz, s16=s16, p8=p8, oacc=oacc,
                bwarm=bwarm)


def _shard_handles(h_all, c):
    """The staged handles this cluster reads, with the per-cluster lists resolved.

    Which of these are genuinely per-cluster depends on DECOMP: kvsplit gives every cluster
    its own K, headpar its own Q. stage() already collapsed whatever is shared onto one
    handle, so indexing is uniform here either way.
    """
    h = dict(h_all)
    for key in ("a", "b", "m", "rowsum", "o"):
        # m/rowsum/o are GOLDENS and are absent when the block builds compute only, which
        # is the composed case: the operands came from a projection, so there is nothing to
        # compare against here and the next block is the consumer, not a host check.
        h[key] = h_all[key][c] if h_all.get(key) is not None else None
    return h


def _build_head(ctx, cfg, c, h_all, buf):
    """Everything a cluster does BEFORE its first KV tile: warm-ups, Q, the arena fills.

    Split from the body so that under headpar EVERY cluster's head nodes are created before
    ANY broadcast node is. That ordering is what makes it safe to spread the broadcasts
    over all four xDMAs: a core dispatches in creation order, so a broadcast placed on
    cluster c's xDMA sits BEHIND c's own fills rather than in front of them. In front, a
    later broadcast would wait on a PV that waits on a softmax that waits on the very fills
    queued behind it -- a genuine cycle, not a slowdown.
    """
    g = ctx.at(c)
    h = _shard_handles(h_all, c)
    arena, k8, q8, v8 = buf["arena"], buf["k8"], buf["q8"], buf["v8"]
    cz, s16, p8, oacc = buf["cz"], buf["s16"], buf["p8"], buf["oacc"]
    def load(tag, key, dst, nbytes, after=None):
        return g.node(f"Load_{tag}", ctx.dm, "__snax_bingo_kernel_idma_1d_copy",
                      SnaxBingoKernelIdma1dCopyArgs(h[key], dst, nbytes), after)

    def xload(tag, key, dst, nbytes, after=None):
        """Same transfer on the xDMA hart instead of the DM core's iDMA."""
        return g.node(f"Load_{tag}", ctx.xdma, "__snax_bingo_kernel_xdma_1d_copy",
                      SnaxBingoKernelXdma1dCopyArgs(h[key], dst, nbytes), after)


    # The loads are chained: one iDMA engine, and serialising them here keeps the graph's
    # ready set small rather than handing the manager four nodes that must queue anyway.
    # ORDER MATTERS, and it is not the order the buffers are declared in. There is one
    # iDMA engine so these serialise regardless; what the chain decides is WHICH of them
    # QK(0) has to wait behind. QK reads K, Q and the zero C bias. V and the O zero are
    # PV's, and PV cannot run until SM(0) is done anyway -- so they belong AFTER the
    # loads QK needs, where they stream underneath QK(0) and SM(0) instead of delaying
    # the first matmul by their own duration.
    # ---- warm every core's config path BEFORE the first byte moves ---------------------
    # A core's FIRST dispatch of a given kernel costs far more than its next, and the
    # difference is instruction-cache line refills. A refill is tens of cycles on an idle
    # fabric and well over a thousand while the iDMA is streaming a tile through the same
    # path, so what the first dispatch costs is decided by WHEN it runs, not by what it
    # does. Left to itself every core takes its first dispatch with the load chain already
    # saturating the fabric, and all three pay the contended price at once.
    #
    # These three nodes take that hit while nothing else is running. They are tiny, they
    # write only their own scratch, and the whole load chain is anchored behind them, so
    # the refills happen on an idle fabric and every real dispatch afterwards hits in the
    # icache. The cost is that the first load starts a little later; the gain is that the
    # first QK, the first softmax and the first arena fill all configure at warm speed.
    # ---- the array's own hardware counters ---------------------------------------------
    # Five words the two FA matmuls accumulate into (busy, stall_a, stall_b, stall_d,
    # dispatches), read straight out of VersaCore's read-only CSRs. This is what makes the
    # utilisation figure comparable to the snax reference's `GEMM core busy %`, which is
    # also a counter read and not a timed span -- see device_kernel_args.h.
    #
    # It is a knob because it is not free: five RO CSR reads per dispatch, and one more L1
    # allocation per cluster. Both are small against a dispatch, but the point of the
    # measurement IS the dispatch cost, so it should be possible to take the instrument out
    # and confirm the number did not move.
    perf = g.l1(f"gemm_perf_c{c}", 64) if cfg.measure_array else 0

    warm_buf = g.l1("fa_warm_buf", 1024)
    warm_a = g.l1("fa_warm_a", cfg.mesh_row * cfg.tile_size)
    warm_b = g.l1("fa_warm_b", cfg.mesh_col * cfg.tile_size)
    # Zeroed first, on the otherwise idle xDMA core. Not decoration: the warm GEMM reads
    # these, and TCDM that nothing has written reads back X -- harmless in the array's
    # datapath, but there is no reason to feed it X when the same node also warms the
    # xdma_memset path that the arena fills use.
    # ZERO THE COUNTER ACCUMULATOR BEFORE ANY MATMUL TOUCHES IT.
    #
    # The matmuls do `pf[i] += csrr(...)`, a read-modify-write. TCDM is tc_sram with
    # SimInit="none", so a word this run has not written reads back X -- and X in an
    # accumulator reaches printf, where it trips the RegWriteKnown assertion and the hart
    # dies. That is not a hang: the sim runs to its wall-clock budget with the UART frozen
    # mid-line ("[Cluster 0] GEMM-ARRAY busy="), which reads exactly like a fabric deadlock.
    # Cost is one 64-B fill on the otherwise idle xDMA core, ahead of everything.
    perf_z = ([g.node(f"PerfZero_c{c}", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                      SnaxBingoKernelXdmaMemsetArgs(
                          perf, 64, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO))]
              if cfg.measure_array else [])

    warm_z = g.node("WarmZero", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                    SnaxBingoKernelXdmaMemsetArgs(
                        warm_buf, 1024, SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO),
                    perf_z)
    warm_a_z = g.node("WarmZeroA", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                      SnaxBingoKernelXdmaMemsetArgs(
                          warm_a, cfg.mesh_row * cfg.tile_size,
                          SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), warm_z)
    # warm_b needs the same fill as warm_a: WarmGemm below takes it as BOTH its B and its C
    # operand, and TCDM that nothing has written reads back X. A node, not a knob -- feeding
    # the array X is never correct, and the pair is what the name WarmZeroAB always implied.
    warm_ab = g.node("WarmZeroB", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                     SnaxBingoKernelXdmaMemsetArgs(
                         warm_b, cfg.mesh_col * cfg.tile_size,
                         SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), warm_a_z)

    # One array block: the smallest dispatch gemm_fa_qk accepts, purely to fetch its
    # config path. C is masked off for the score matmul, so warm_b doubles as its base.
    # Only the xDMA's MEMSET path is warmed, not its 1d_copy path, even though the V
    # stream uses the latter and pays a cold config for it. A warm-up copy hangs: the
    # engine is armed but its finish counter never advances, and xdma_wait_task has no
    # bound, so the xDMA core spins forever. The size is a legal multiple of the datapath
    # width, so it is not the alignment guard in bingo_helpers -- the descriptor is being
    # rejected for some other reason. Diagnose that before adding one back.
    warm_gemm = g.node("WarmGemm", ctx.gemm, "__snax_bingo_kernel_gemm_fa_qk",
                       SnaxBingoKernelGemmFaQkArgs(warm_a, warm_b, warm_b,
                                                   warm_buf.view(64), 1, 1, 1), warm_ab)
    # Only the CONFIG paths are warmed, not the SIMD task-firing path. Warming that too --
    # a full tiny softmax here -- does remove the first tile's cold run, but it re-times the
    # pipeline so that later softmax runs land under the K/V stream instead, and their
    # contended cost is larger than the cold run it saved. Net zero, so it is not done.
    # ---- build the SIMD task geometry off the critical path ----------------------------
    # The per-tile SIMD kernel opens by writing its task-shape descriptors into the head of
    # the arena and memoises them, so only the first tile of each query tile pays. That
    # first call is expensive for two separate reasons:
    #
    # 1. THE BUILD, a few hundred scalar stores. The cost is not the stores but where they
    #    land: with the iDMA streaming a K/V tile through the same TCDM ports, each store
    #    costs several times its uncontended price, scaling with traffic in flight.
    #
    # 2. THE DISPATCH TAIL, the code after the build, whose instruction-cache lines are
    #    cold and whose refills pay the same contended price.
    #
    # A PROLOGUE node runs both before the load chain has spun up, on the real arena with
    # the real geometry, against buffers tile 0 overwrites anyway. Every real tile then
    # passes PRIMED and takes the memo path. It moves no data and queues no accelerator
    # task, so it does not disturb m, l or O.
    #
    # Making the prologue run the FULL kernel instead -- firing the tasks as well -- is a
    # regression: its own run and drain then sit on the critical path ahead of the first
    # real tile on the same in-order core, which costs more than the cold tail it saves.
    # One per query tile: the descriptors hold absolute pointers into their own arena.
    warm = [g.node(f"Geom{q}", ctx.simd, "__snax_bingo_kernel_simd_fa_softmax",
                   SnaxBingoKernelSimdFaSoftmaxArgs(
                       s16[0].view(64), p8[0], arena[q],
                       bc=cfg.bc, dhead=cfg.dhead, tile_idx=0, seed_state=0,
                       geom_mode=SnaxBingoKernelSimdFaSoftmaxArgs.GEOM_PROLOGUE))
            for q in range(cfg.nq)]

    # Q is the query tile: fixed for the whole run, loaded once.
    ld_q = [load(f"Q{q}", "b", q8[q], cfg.n * cfg.k * cfg.mesh_col * cfg.tile_size, [warm_gemm])
            for q in range(cfg.nq)]
    # No Czero load: with the C channels masked the score matmul never reads this buffer,
    # which keeps 64 KiB of zeros off the head of the chain, in front of QK(0).
    # O starts at zero: the O matmul sets take_in_new_c, so every output block starts from
    # C, and C is oacc itself. A prefix of the same zero region does it.

    # The softmax's OWN running O lives in the arena and is a different buffer from the
    # INT32 oacc32 above. Zeroing it from the SIMD core costs dhead beats -- ~8 KiB, and
    # the single largest serial bubble in the run -- so the idle DM core takes the same
    # zero region instead. The device side then only seeds mrun/lrun, which is two beats.
    _lay = SnaxBingoKernelSimdFaSoftmaxArgs.layout(cfg.bc, cfg.dhead)
    # ZERO THE O ACCUMULATOR ON THE xDMA, not the iDMA.
    #
    # This is a CONSTANT, so fetching it from main memory is wasted traffic on the one
    # engine that is on the critical head: at NQ query tiles it is NQ separate transfers
    # standing in front of K(0), which is the largest single gap in the fill. The xDMA core
    # is otherwise idle -- it owns nothing but its exit node -- so it generates the zeros in
    # place, with the reader channels disabled so no TCDM read is issued either.
    #
    # It does not wait on ld_q: generating a constant needs no operand, so this runs from
    # cycle zero and overlaps the Q load rather than queueing behind it.
    # THE WHOLE RECURRENCE STATE IS GENERATED, NOT LOADED.
    #
    # m = -inf, l = 0 and O = 0 are constants. Seeding m and l from the SIMD kernel and
    # carrying O's zeros on the iDMA both sit on the critical head: at NQ query tiles that
    # is NQ transfers in front of K(0) plus NQ scalar fills in front of the first softmax.
    # None of it needs an operand, so the idle xDMA core generates all three in place --
    # reader channels disabled, so not even a TCDM read.
    #
    # -65504 is FP16's most negative finite value, which is what the kernel used; a true
    # -inf would make the first exp(S - m) NaN rather than 0. The pattern is 32 bits
    # because no single BYTE repeats into 0xFBFF.

    ld_az = []
    for q in range(cfg.nq):
        # ANCHORED BEHIND Q, deliberately.
        #
        # These need no operand, so leaving them dependency-free looks right -- but
        # "eligible" is not "runs first". Nothing runs until the staging barrier lifts,
        # and when it does every ready task competes for the manager at once; unanchored,
        # the fills win that race and push the SIMD prologue and K(0) behind them.
        #
        # Anchoring them behind Q costs nothing real -- Q is short and on a different
        # engine -- and puts them behind the critical chain in the dispatch order rather
        # than ahead of it. The fills are cheap and carry no main-memory traffic, which is
        # the whole reason to keep them on the xDMA.
        z = g.node(f"ArenaOzero{q}", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                   SnaxBingoKernelXdmaMemsetArgs(
                       arena[q].view(_lay["oacc"]), cfg.dhead * 64,
                       SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), ld_q[q])
        # m and l are one beat each and NOT adjacent (rmax, mrun, mnew, delta, corrL,
        # lrun), so they are two fills rather than one. Chained after the O fill: they
        # share the one xDMA engine, and stating the order costs nothing at run time while
        # leaving the dep-tag allocator one chain instead of three concurrent ones.
        mfill = g.node(f"SeedM{q}", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                       SnaxBingoKernelXdmaMemsetArgs(
                           arena[q].view(_lay["mrun"]), 64, _NEG_INF16), z)
        lfill = g.node(f"SeedL{q}", ctx.xdma, "__snax_bingo_kernel_xdma_memset",
                       SnaxBingoKernelXdmaMemsetArgs(
                           arena[q].view(_lay["lrun"]), 64,
                           SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), mfill)
        ld_az.append(lfill)

    return dict(warm_gemm=warm_gemm, warm=warm, ld_q=ld_q, ld_az=ld_az,
                perf=perf, load=load, xload=xload)


def _build_cluster(ctx, cfg, c, h_all, m_all, rowsum_all, buf, bcast=None,
                   all_bufs=None, verify=True):
    """The KV-tile pipeline and this cluster's own checks, on top of _build_head.

    Byte-for-byte the tuned one-cluster graph -- same double buffering, same load-chain
    order -- with three differences and no others: it runs NKV_PER tiles instead of NKV, it
    reads this cluster's operands, and it ends at the partial rather than at a check.

    `bcast` is None under kvsplit, where this cluster loads its own K and V. Under headpar
    it is {"k": [...], "v": [...]}: the MULTICAST nodes, already created by build(), that
    fill every cluster's k8/v8 from one read. This cluster then only depends on them.
    """
    g = ctx.at(c)
    h = _shard_handles(h_all, c)
    # numpy goldens, for the check tolerances only; absent when verify is off.
    m = m_all[c] if m_all is not None else None
    rowsum = rowsum_all[c] if rowsum_all is not None else None
    arena, k8, q8, v8 = buf["arena"], buf["k8"], buf["q8"], buf["v8"]
    cz, s16, p8, oacc = buf["cz"], buf["s16"], buf["p8"], buf["oacc"]

    # HEAD FIRST, IN THIS CLUSTER'S OWN PASS -- do not hoist it into build().
    #
    # MEASURED: building every cluster's head before any body changes the GLOBAL node
    # creation order, and that order IS the task-descriptor list the BINGO manager streams.
    # It cost several points of array utilisation on a graph that was otherwise identical --
    # same per-core kernel sequences in every cell, same allocations. Dispatch is
    # order-sensitive; the list order is not cosmetic.
    head = _build_head(ctx, cfg, c, h_all, buf)
    warm_gemm, warm = head["warm_gemm"], head["warm"]
    ld_q, ld_az, perf = head["ld_q"], head["ld_az"], head["perf"]
    load, xload = head["load"], head["xload"]

    KBYTES = cfg.m * cfg.k * cfg.mesh_row * cfg.tile_size
    VBYTES = cfg.s2_m * cfg.s2_k * cfg.mesh_row * cfg.tile_size

    # One tiny multicast, on the engine that owns tile 0, behind this cluster's own head
    # chain so it runs on a quiet fabric. Tile 0's broadcast then waits on it instead of on
    # ld_q, which both orders the two and gives the real transfer a warm config path.
    bwarm_node = None
    if bcast is not None and cfg.bcast_warmup and c == _bcast_owner(cfg, 0):
        bwarm_node = g.node(
            f"BcastWarm_c{c}", ctx.xdma, "__snax_bingo_kernel_xdma_multicast",
            SnaxBingoKernelXdmaMulticastArgs(
                h_all["a"][0], [bf["bwarm"] for bf in all_bufs], 64),
            ld_q)

    # SOFTWARE-PIPELINED EMISSION: QK(i+1) is created BEFORE PV(i).
    #
    # The manager dispatches a core's tasks in CREATION ORDER, so a ready task cannot
    # overtake a blocked one -- head-of-line blocking on the GEMM's own queue. Emitted
    # naively as QK(0), PV(0), QK(1)..., PV(0) waits on SM(0) and QK(1) sits behind it even
    # though its K landed long before, and the array idles for the whole softmax.
    #
    # Interleaving gives the GEMM queue QK0, QK1, PV0, QK2, PV1, ... so while SM(i) runs the
    # array always has QK(i+1) in front of it. This is what the cluster-side kernel does by
    # construction: it has both QK(i+1) and PV(i-1) available and between them they cover
    # the softmax's latency.
    #
    # Dependencies are unchanged -- they are by reference, not by order -- and every one
    # still names a node created earlier: SM(i) reaches back to pv[i-2] (created at step
    # i-1) and PV(i-1) to sm[i-1] (created at step i-1).
    ld_k, ld_v = [], []
    qk, sm, pv = [], [], []
    # (index into qk/pv, tile) for consumers whose broadcast producer is owned by a
    # cluster that has not been built yet. Closed in build().
    pending_k, pending_v = [], []
    TOTAL = cfg.nkv_per * cfg.nq          # this cluster's shard, not the whole KV axis
    for i in range(TOTAL + 1):
        if i < TOTAL:
            j, q = divmod(i, cfg.nq)
            bcast_this_k = (bcast is not None and cfg.bcast_k and not cfg.k_pull_from_cl0
                            and not (cfg.bcast_skip_first and j == 0))
            if q == 0 and bcast is not None and cfg.k_push_all:
                # One host task delivers this tile's K into all four clusters' k8. Built on
                # cluster 0's pass only -- like the broadcast, and for the same reason: the
                # node order is load-bearing, so the producer must appear where the operand
                # is first consumed rather than hoisted into a prologue.
                if c == 0:
                    k_after = ld_q if j < cfg.nkbuf else [qk[(j - cfg.nkbuf + 1) * cfg.nq - 1]]
                    bcast["k"][j] = g.node(
                        f"PushK{j}", ctx.host, "__host_bingo_kernel_idma_multi",
                        HostBingoKernelIdmaMultiArgs(
                            [(h_all["a"][0], bf["k8"][j % cfg.nkbuf]) for bf in all_bufs],
                            KBYTES),
                        k_after, cluster=0)
                ld_k.append(bcast["k"].get(j))
            elif q == 0 and bcast is not None and cfg.k_pull_from_cl0:
                # K on the xDMA: cluster 0 reads main memory, the rest read cluster 0's L1.
                k_after = ld_q if j < cfg.nkbuf else [qk[(j - cfg.nkbuf + 1) * cfg.nq - 1]]
                own = _pull_owner(cfg, j)
                if j < cfg.pull_skip_first:
                    # Every cluster reads this tile itself. xload, not load: K lives on the
                    # xDMA in the steady state and moving it to the iDMA would put it behind V.
                    n = xload(f"K{j}", "a", k8[j % cfg.nkbuf], KBYTES, k_after)
                    if c == own:
                        bcast["ksrc"][j] = n
                elif c == own:
                    if cfg.k_push_owner:
                        n = g.node(f"PushK{j}_c{c}", ctx.host,
                                   "__host_bingo_kernel_idma",
                                   HostBingoKernelIdmaArgs(h["a"], k8[j % cfg.nkbuf], KBYTES),
                                   k_after, cluster=0)
                    else:
                        n = xload(f"K{j}", "a", k8[j % cfg.nkbuf], KBYTES, k_after)
                    bcast["ksrc"][j] = n
                else:
                    # Under rotation the owner may be a cluster built LATER, so its node can be
                    # missing here. Create the pull without that edge and record the hole;
                    # build() closes it once every pass has run.
                    src = bcast["ksrc"].get(j)
                    after = list(k_after) + ([src] if src is not None else [])
                    n = g.node(f"PullK{j}", ctx.xdma,
                               "__snax_bingo_kernel_xdma_1d_copy",
                               SnaxBingoKernelXdma1dCopyArgs(
                                   all_bufs[own]["k8"][j % cfg.nkbuf], k8[j % cfg.nkbuf], KBYTES),
                               after)
                    bcast["kpull"].append((j, n))
                    if src is None:
                        bcast.setdefault("kpend", []).append((j, n))
                ld_k.append(n)
            elif q == 0 and bcast_this_k:
                # --- headpar: K arrives by broadcast, from ONE read ----------------------
                # The GQA group shares its KV head, so this tile's K is the same bytes on
                # every cluster. Cluster 0 issues the multicast HERE, inside the tile loop
                # and with the SAME predecessors the unicast load would have had; the other
                # three clusters only take the dependency.
                #
                # THOSE PREDECESSORS ARE LOAD-BEARING. Created dependency-free -- so K0, K1,
                # V0 and V1 are all ready at once on this single xDMA -- the run WEDGES IN
                # TEARDOWN: the whole pipeline completes and the host epilogue never runs.
                # That is exactly the failure the k8 double-buffer comment above predicts
                # for "more simultaneously-ready DMA nodes than the manager's ready set
                # tolerates". Measured twice, from two different directions.
                if c == _bcast_owner(cfg, j):
                    k_after = ld_q if j < cfg.nkbuf else [qk[(j - cfg.nkbuf + 1) * cfg.nq - 1]]
                    if j == 0 and bwarm_node is not None:
                        k_after = [bwarm_node]
                    bcast["k"][j] = g.node(
                        f"BcastK{j}", ctx.xdma, "__snax_bingo_kernel_xdma_multicast",
                        SnaxBingoKernelXdmaMulticastArgs(
                            h_all["a"][0], [bf["k8"][j % cfg.nkbuf] for bf in all_bufs],
                            KBYTES),
                        k_after)
                # Under BCAST_SPREAD a cluster built LATER owns some tiles, so the node
                # can be missing here. Leave a hole and let build() close it once every
                # pass has run -- the same deferral the cross-cluster WAR edges already
                # need, and for the same reason.
                ld_k.append(bcast["k"].get(j))
            elif q == 0:
                # Tile 0 under BCAST_SKIP_FIRST lands here too, and wants exactly what this
                # branch already does: this cluster's own iDMA pulling the shared K into its
                # own k8. stage() collapsed headpar's K onto one handle, so h["a"] is that
                # shared array for every cluster -- four reads of the same bytes, in parallel.
                # --- stream this KV tile's K and V --------------------------------------
                # Into the buffer every query tile of step j-2 has finished with, so it
                # waits for the LAST of them. Before tile 2 there is nothing but Q.
                k_after = ld_q if j < cfg.nkbuf else [qk[(j - cfg.nkbuf + 1) * cfg.nq - 1]]
                ld_k.append(load(f"K{j}", "a", k8[j % cfg.nkbuf], KBYTES, k_after))
            if q == 0 and bcast is not None and cfg.bcast_v:
                if c == _bcast_owner(cfg, j, cfg.bcast_v_skew):
                    v_after = list(ld_az) if j < cfg.nvbuf \
                        else [pv[(j - cfg.nvbuf + 1) * cfg.nq - 1]]
                    bcast["v"][j] = g.node(
                        f"BcastV{j}", ctx.xdma, "__snax_bingo_kernel_xdma_multicast",
                        SnaxBingoKernelXdmaMulticastArgs(
                            h_all["v"], [bf["v8"][j % cfg.nvbuf] for bf in all_bufs], VBYTES),
                        v_after)
                ld_v.append(bcast["v"].get(j))
            elif q == 0:
                # Behind the arena fills, not just behind Q. Issuing V(0) BEFORE the
                # fills instead does clear the first softmax of V traffic -- its config and
                # run drop to their warm values -- but it pushes the fills far enough back
                # that the head grows by more than the contention cost. The fills go first.
                # V and the fills now share
                # the xDMA engine, and the fills are needed by the FIRST softmax while V
                # is not needed until the first PV. Left to the tie-break the V stream
                # gets in front of them and the first softmax waits out two whole V
                # transfers. The fills are a few tens of cycles, so stating the order
                # costs nothing.
                v_after = list(ld_az) if j < cfg.nvbuf else [pv[(j - cfg.nvbuf + 1) * cfg.nq - 1]]
                # V streams on the xDMA, K on the iDMA. The two engines then move their
                # tiles CONCURRENTLY instead of queueing on one, which takes the K/V pair
                # off the critical path whenever the array is the longer of the two.
                if cfg.v_push_pairs:
                    # PUSH, PAIRED. The system iDMA (SOC_WIDE_XBAR_IN_SYS_IDMA_MST) writes
                    # into this cluster's L1 over quadrant_wide_in while the cluster's own
                    # iDMA pulls K over quadrant_wide_out -- two disjoint 512-bit pipes.
                    # One task carries V(j) and V(j+1); the ODD tile reuses the node the
                    # even one created, so the pair costs one dispatch instead of two.
                    # Pinned to cluster 0 because every host kernel is; the DESTINATIONS are
                    # still this cluster's L1, reached by absolute address.
                    if j % 2 == 0:
                        nxt = j + 1
                        pairs = [(h["v"], v8[j % cfg.nvbuf])]
                        if nxt < cfg.nkv_per:
                            pairs.append((h["v"], v8[nxt % cfg.nvbuf]))
                        ld_v.append(g.node(f"LoadV{j}_{nxt}_c{c}", ctx.host,
                                           "__host_bingo_kernel_idma_multi",
                                           HostBingoKernelIdmaMultiArgs(pairs, VBYTES),
                                           v_after, cluster=0))
                    else:
                        # the pair's partner: same node, already issued
                        ld_v.append(ld_v[-1])
                elif j in cfg.v_push_tiles:
                    ld_v.append(g.node(f"LoadV{j}_c{c}", ctx.host,
                                       "__host_bingo_kernel_idma",
                                       HostBingoKernelIdmaArgs(h["v"], v8[j % cfg.nvbuf], VBYTES),
                                       v_after, cluster=0))
                elif bcast is not None and cfg.v_pull_from_cl0:
                    # HEADPAR, V PULLED: cluster 0 reads main memory, the rest read
                    # cluster 0's L1. Every cluster still issues exactly one transfer per
                    # tile on its own iDMA, so no engine gains work -- only the SOURCE
                    # changes, and with it three quarters of the main-memory traffic.
                    own = _pull_owner(cfg, j)
                    if j < cfg.pull_skip_first:
                        # Same as K: the leading tiles have nothing to overlap their second
                        # hop, so every cluster reads them itself. V stays on the iDMA.
                        n = load(f"V{j}", "v", v8[j % cfg.nvbuf], VBYTES, v_after)
                        if c == own:
                            bcast["vsrc"][j] = n
                    elif c == own:
                        if cfg.v_push_owner:
                            n = g.node(f"PushV{j}_c{c}", ctx.host,
                                       "__host_bingo_kernel_idma",
                                       HostBingoKernelIdmaArgs(h["v"], v8[j % cfg.nvbuf], VBYTES),
                                       v_after, cluster=0)
                        else:
                            n = load(f"V{j}", "v", v8[j % cfg.nvbuf], VBYTES, v_after)
                        bcast["vsrc"][j] = n
                    else:
                        # wait for cluster 0's copy of THIS tile as well as our own WAR edge
                        src = bcast["vsrc"].get(j)
                        after = list(v_after) + ([src] if src is not None else [])
                        n = g.node(f"PullV{j}", ctx.dm,
                                   "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(
                                       all_bufs[own]["v8"][j % cfg.nvbuf], v8[j % cfg.nvbuf], VBYTES),
                                   after)
                        bcast["vpull"].append((j, n))
                        if src is None:
                            bcast.setdefault("vpend", []).append((j, n))
                    ld_v.append(n)
                elif bcast is not None and not cfg.v_on_xdma:
                    # HEADPAR, V NOT BROADCAST: pull it on the iDMA, never the xDMA.
                    #
                    # A RECEIVING CLUSTER'S xDMA MUST BE IDLE. An incoming multicast write
                    # is handled by the destination cluster's own xDMA finish manager -- the
                    # same block whose i_read_stall_watchdog fires in the 4-issuer deadlock
                    # -- so a cluster cannot be running its own transfer and absorbing a
                    # broadcast at the same time.
                    #
                    # MEASURED. With V on the cluster xDMAs and K broadcast into them, ALL
                    # FOUR xDMA harts wedged in their wait loops (three at 1.8 GB of trace).
                    # The arm where every cluster xDMA did nothing but five tiny memsets
                    # while the broadcasts landed is the one that PASSED. Under headpar the
                    # iDMA is free anyway -- it carries only Q, not K -- so this costs
                    # nothing and keeps the receiving xDMAs clear.
                    ld_v.append(load(f"V{j}", "v", v8[j % cfg.nvbuf], VBYTES, v_after))
                else:
                    ld_v.append(xload(f"V{j}", "v", v8[j % cfg.nvbuf], VBYTES, v_after))

            # SAME-CORE EDGES ARE NOT REDUNDANT -- do not "optimise" them away.
            #
            # It is tempting to drop a dependency whose producer sits on the SAME core as
            # its consumer, on the grounds that the manager's per-core waiting queue is
            # FIFO and the core runs one task at a time, so the order is guaranteed anyway.
            # It is not: that edge is what PINS the order. The queue order is the
            # compiler's topological order, and with the edge gone the sort is free to
            # interleave the core's tasks differently, which costs far more than the dummy
            # nodes the edge creates.
            #
            # The cost they carry is real but must be paid another way: every predecessor
            # beyond the first per core becomes a dummy_check node
            # (bingo_transform_dfg_add_dummy_check_nodes), which occupies a slot in the
            # CONSUMER's waiting queue. The fix for that is stream ORDER, not edge removal
            # -- see bingo_dfg.bingo_stream_order().
            if q != 0:
                deps = [qk[i - 1]]
            elif ld_k[j] is not None:
                deps = [ld_k[j]]
            else:
                deps = []
                pending_k.append((len(qk), j))
            # UNDER headpar THIS EDGE IS NOT REDUNDANT. In the unicast case Q and K share
            # the iDMA and the load chain puts K(0) behind Q, so QK inherits the wait. The
            # broadcast moves K to the xDMA, which breaks that chain -- without this edge
            # QK(0) would be free to read a q8 the iDMA has not filled yet, and TCDM reads
            # X rather than faulting (SimInit="none"), which kills the hart silently.
            if bcast is not None and j == 0:
                deps.append(ld_q[q])
            # WAR on the score buffer: QK(i) overwrites the tile SM(i-NSCORE) read.
            if i >= cfg.nscore_war:
                deps.append(sm[i - cfg.nscore_war])
            qk.append(g.node(f"QK_{j}_{q}", ctx.gemm,
                             "__snax_bingo_kernel_gemm_fa_qk",
                             SnaxBingoKernelGemmFaQkArgs(k8[j % cfg.nkbuf], q8[q], cz,
                                                         s16[i % cfg.nscore].view(64), cfg.m, cfg.k, cfg.n,
                                                         perf_addr=perf),
                             deps))

            deps = [qk[i]]
            if j == 0:
                deps.append(ld_az[q])
                deps.append(warm[q])
            if i >= 1:
                deps.append(sm[i - 1])
            # WAR on the probability buffer: SM(i) overwrites what PV(i-NSCORE) read.
            if i >= cfg.nscore_war:
                deps.append(pv[i - cfg.nscore_war])
            sm.append(g.node(f"SM_{j}_{q}", ctx.simd,
                             "__snax_bingo_kernel_simd_fa_softmax",
                             SnaxBingoKernelSimdFaSoftmaxArgs(
                                 s16[i % cfg.nscore].view(64), p8[i % cfg.nscore], arena[q],
                                 bc=cfg.bc, dhead=cfg.dhead, tile_idx=j,
                                 seed_state=0,
                                 # THE FULL CSR PROGRAM RUNS ONCE PER CORE, NOT PER TILE.
                                 # snax_simd_program_fast writes 21 CSRs that depend only
                                 # on (bc, dhead), so every softmax after the first on this
                                 # core re-writes the same values. CSR_PRIMED says 'the
                                 # block still holds a same-geometry program, write only the
                                 # six per-task CSRs'. Only the host may assert it: this
                                 # cluster's SIMD core runs nothing but these softmaxes and
                                 # the prologue that programmed the CSRs in the first place.
                                 geom_mode=(SnaxBingoKernelSimdFaSoftmaxArgs.GEOM_PRIMED
                                            if i == 0 else
                                            SnaxBingoKernelSimdFaSoftmaxArgs.GEOM_CSR_PRIMED)),
                             deps))

        if i >= 1:
            # PV for the PREVIOUS step, emitted after this step's QK.
            p = i - 1
            pj, pq = divmod(p, cfg.nq)
            deps = [sm[p]] if p == 0 else [sm[p], pv[p - 1]]
            if pq == 0:
                if ld_v[pj] is not None:
                    deps.append(ld_v[pj])
                else:
                    pending_v.append((len(pv), pj))
            pv.append(g.node(f"PV_{pj}_{pq}", ctx.gemm,
                             "__snax_bingo_kernel_gemm_fa_pv",
                             SnaxBingoKernelGemmFaPvArgs(v8[pj % cfg.nvbuf], p8[p % cfg.nscore],
                                                         oacc[pq] if pj else 0, oacc[pq],
                                                         cfg.s2_m, cfg.s2_k, cfg.s2_n,
                                                         perf_addr=perf),
                             deps))

    # The array counters, printed once per cluster. Anchored on the last PV so it cannot
    # run until every matmul this cluster owns has retired, and placed on the GEMM core
    # because the counters are that core's accelerator CSRs -- no other core can read them.
    if cfg.measure_array:
        # One KV tile is QK (Bc x Br x d MAC) + PV (Br x d x Bc MAC) = 4.19 M MAC, which is
        # 4096 cycles at meshRow*meshCol*tileSize = 1024 MAC/cc. The reference states the
        # same figure (snax-flashattn-decode.c: "4.19 M MAC and 4096 GEMM cycles").
        ideal_cc = (2 * (cfg.bc * cfg.br * cfg.dhead)) // (cfg.mesh_row * cfg.mesh_col * cfg.tile_size) \
                   * (cfg.nkv_per * cfg.nq)
        g.node(f"ArrayPerf_c{c}", ctx.gemm, "__snax_bingo_kernel_gemm_perf_report",
               SnaxBingoKernelGemmPerfReportArgs(perf, ideal_cc), pv[-1])

    # ---- this shard's own statistics ---------------------------------------------------
    # Checked per shard, not only after the merge. A cluster whose recurrence quietly did
    # nothing still hands the gather a well-formed partial, and the merged result alone
    # cannot say which of the four it came from. Read straight out of the arena: layout()
    # mirrors the device's own simd_fa_layout(), so there is no second copy of the offsets.
    lay = SnaxBingoKernelSimdFaSoftmaxArgs.layout(cfg.bc, cfg.dhead)
    last = sm[cfg.nkv_per * cfg.nq - 1]

    # Host kernels live on cluster 0 core HOST_CORE and may not be placed anywhere else, so
    # these are pinned there explicitly even though the data they read is on cluster c.
    def host(name, kname, kargs, after):
        return g.node(name, ctx.host, kname, kargs, after, cluster=0)

    # VERIFICATION, and it is optional. A composed graph feeds attention's output to the
    # next block rather than to a host check, and then the goldens do not exist: the Q, K
    # and V it runs on came from a projection, not from a datagen. With verify=False the
    # block's only inputs are the operands, which is what makes it composable.
    st_m = ck_o = None
    if verify:
        # The tolerances are derived from the goldens themselves rather than fixed, because an
        # absolute tolerance means nothing without a magnitude: one FP16 step at a score of
        # 1000 is 1.0, and at 0.5 it is 0.0005. m is a max of converted integers and is
        # expected exact, so two steps is already slack; rowsum accumulates Bc terms in FP32
        # and narrows once, so it gets four.
        tol_m = float(2 * np.max(np.spacing(m.astype(np.float16))))
        tol_s = float(4 * np.max(np.spacing(rowsum.astype(np.float16))))
        if cfg.debug_loose_ml:
            # DEBUG ONLY. m and rowsum are the FIRST checks dispatched, and a failing host
            # kernel breaks the scheduler loop (bingo_api.c:906), so one bad m costs the other
            # eleven results. Widening these two lets the run reach the O check, which is the
            # one that says whether the COMPUTATION is wrong or only the m/rowsum readback.
            tol_m = tol_s = 1.0e4

        if cfg.debug_per_tile_m and c == 2:
            # One store+check per tile on the failing cluster only. Ordered sm[i] -> store ->
            # sm[i+1] so the read happens after THIS tile's commit and before the next tile
            # overwrites mrun. The run aborts at the first failing check, so the UART names the
            # tile directly.
            for _i, _smn in enumerate(sm):
                _l3 = BingoMemAlloc(f"dbg_m_c{c}_t{_i}", size=64, mem_level=MemLevel.L3)
                _st = host(f"DbgStoreM_c{c}_t{_i}", "__host_bingo_kernel_idma",
                           HostBingoKernelIdmaArgs(arena[_i % cfg.nq].view(lay["mrun"]), _l3, 64),
                           _smn)
                checks.check_fp16(ctx, f"DbgCheckM_c{c}_t{_i}", golden=h["m"], got=_l3,
                                  elems=cfg.br, tol=tol_m, after=_st,
                                  label=f"dbg_m_c{c}_t{_i}")
                if _i + 1 < len(sm):
                    g.dfg.bingo_add_edge(_st, sm[_i + 1])

        if cfg.debug_s16_compare and c == 2 and len(qk) > cfg.s16_cmp_buf + cfg.nscore:
            # THE WHOLE score tile, not one beat. Comparing 64 B of 32,832 B says nothing about
            # the 511 beats the rowmax also reads, and the rowmax is what feeds m.
            _n = 64 + cfg.bc * cfg.br * 2
            _a = BingoMemAlloc(f"dbg_s16_first_c{c}", size=_n, mem_level=MemLevel.L3)
            _b = BingoMemAlloc(f"dbg_s16_again_c{c}", size=_n, mem_level=MemLevel.L3)
            _i0, _i1 = cfg.s16_cmp_buf, cfg.s16_cmp_buf + cfg.nscore
            _sa = host(f"DbgS16First_c{c}", "__host_bingo_kernel_idma",
                       HostBingoKernelIdmaArgs(s16[_i0 % cfg.nscore], _a, _n), qk[_i0])
            _sb = host(f"DbgS16Again_c{c}", "__host_bingo_kernel_idma",
                       HostBingoKernelIdmaArgs(s16[_i1 % cfg.nscore], _b, _n), qk[_i1])
            # the first store must complete before QK(_i1) overwrites the buffer
            g.dfg.bingo_add_edge(_sa, qk[_i1])
            checks.check_bytes(ctx, f"DbgS16Cmp_c{c}", golden=_a, got=_b, nbytes=_n,
                               after=_sb, label=f"dbg_s16_c{c}")

        l3_m = BingoMemAlloc(f"out_fa_m_c{c}", size=64, mem_level=MemLevel.L3)
        st_m = host(f"Store_m_c{c}", "__host_bingo_kernel_idma",
                    HostBingoKernelIdmaArgs(arena[cfg.nq - 1].view(lay["mrun"]), l3_m, 64), last)
        ck_m = checks.check_fp16(ctx, f"Check_m_c{c}", golden=h["m"], got=l3_m,
                                 elems=cfg.br, tol=tol_m, after=st_m, label=f"fa_m_c{c}")

        # rsum lives in the LAST tile's p8 buffer, straight after its P beats, because that is
        # where the fused pass's contiguous output stream puts it -- not in the arena.
        l3_rs = BingoMemAlloc(f"out_fa_rowsum_c{c}", size=64, mem_level=MemLevel.L3)
        st_rs = host(f"Store_rowsum_c{c}", "__host_bingo_kernel_idma",
                     HostBingoKernelIdmaArgs(p8[(cfg.nkv_per * cfg.nq - 1) % cfg.nscore].view((cfg.bc // 2) * 64),
                                             l3_rs, 64), [last, ck_m])
        ck_rs = checks.check_fp16(ctx, f"Check_rowsum_c{c}", golden=h["rowsum"], got=l3_rs,
                                  elems=cfg.br, tol=tol_s, after=st_rs,
                                  label=f"fa_rowsum_c{c}")

        # CHECK O. Without this the suite validates m and rowsum only, and both are per-tile
        # quantities that are IDENTICAL on every tile here -- so neither covers the
        # accumulation across KV tiles, which is the entire point of FlashAttention. O is the
        # only checked value that depends on every tile having been folded in correctly.
        #
        # It is also what makes a layout experiment trustworthy. The static-L1 work found a
        # write that lands past the end of fa_v8_1, and which buffer it damages depends on what
        # the packer put next: with fa_s16_0 there, every softmax row maximum is wrong and the
        # suite says so; with fa_oacc32_0 there, nothing looked at the damage. A green run whose
        # checks cannot see the failure mode under test is not evidence, and adding this check
        # is cheaper than reasoning about which layouts happen to be observable.
        o_bytes = cfg.br * cfg.dhead * 4
        l3_o = BingoMemAlloc(f"out_fa_o_c{c}", size=o_bytes, mem_level=MemLevel.L3)
        # WAITS ON THE LAST PV, NOT ON `last`. `last` is the last SOFTMAX, and every other store
        # here reads something a softmax wrote (m lives in the arena, rowsum in p8) -- but O is
        # written by the last PV, which is a SIBLING of this node, not an ancestor: PV(TOTAL-1) is
        # emitted in the loop's +1 iteration and depends on sm[TOTAL-1] exactly as this store did.
        # So with `last` the host iDMA was free to read fa_oacc32_0 while the VersaCore was still
        # accumulating into it.
        #
        # It is a RAW hazard, so the visible failure is not a wrong number -- the run HANGS, with
        # the host parked and the UART stopping after the eighth check (all four m, all four
        # rowsum) because the first Store_o/Check_o pair is where the host iDMA first overlaps a
        # live VersaCore write to the same TCDM.
        #
        # It was latent at NSCORE=2: SM(i) takes a WAR edge on PV(i-NSCORE), so a smaller NSCORE
        # drains the PV chain further before `last` retires. At NSCORE=2 only PV(TOTAL-1) can
        # still be outstanding at that point; at NSCORE=3 three PVs can be, and the window is
        # wide enough to hit every time.
        st_o = host(f"Store_o_c{c}", "__host_bingo_kernel_idma",
                    HostBingoKernelIdmaArgs(buf["oacc"][cfg.nq - 1], l3_o, o_bytes), [pv[-1], ck_rs])
        ck_o = checks.check_int32_rel(
            ctx, f"Check_o_c{c}", golden=h["o"], got=l3_o,
            elems=min(cfg.o_check_elems, cfg.br * cfg.dhead), rtol=0.02, after=st_o,
            label=f"fa_o_c{c}")

    # The shard checks are chained rather than left as unordered peers. There is one host
    # core, so they run serially regardless; saying so costs nothing at run time and keeps
    # the per-edge dep tags affordable -- four shards of unordered store->check pairs all
    # land in the same (cluster 0, host, host) cell and each would otherwise need its own.
    return {
        # the first REAL V tile (index 0 of ld_v), so build() can serialise the cold
        # xdma_1d_copy config across clusters
        "first_v": ld_v[0] if ld_v else None,
        "last_sm": last,
        "arena": arena[cfg.nq - 1],
        "p8_last": p8[(cfg.nkv_per * cfg.nq - 1) % cfg.nscore],
        "checks": ck_o,   # the tail of this shard's store->check chain
        "first_store": st_m,  # the head of it -- build() gates this on EVERY cluster
        # The consumers of the broadcast buffers. build() reaches back through these to
        # close the cross-cluster WAR edges, which cannot be stated while this cluster is
        # being built because three quarters of the consumers do not exist yet.
        "qk": qk,
        "pv": pv,
        # The per-tile K/V FILL nodes, indexed by tile. Whatever kind of node filled the
        # slot -- an owner's load or a pull from another cluster -- ld_k[t] is what wrote
        # THIS cluster's k8[t % NKBUF]. The cross-cluster WAR closure needs exactly that,
        # and it cannot be reconstructed from bcast["ksrc"], which holds owner loads only.
        "ld_k": ld_k,
        "ld_v": ld_v,
        # The Q loads, so a caller can state a dependency on Q the same way it can on K.
        "ld_q": ld_q,
        "pending_k": pending_k,
        "pending_v": pending_v,
    }

def fa_attention(ctx, cfg, h, m_all, rowsum_all, verify=True):
    """Four cluster pipelines, split either over KV or over the GQA group's query heads.

    Under DECOMP = "headpar" (the default) the four clusters hold four query heads of one
    GQA group. They share a KV head by construction, so each KV tile is read from main
    memory ONCE and multicast into all four L1s, and each cluster's (m, l, O) is already a
    complete result -- there is nothing to fold. That is the configuration that keeps the
    quadrant compute-bound; see the DECOMP notes in the configuration section.

    Under DECOMP = "kvsplit" the rest of this docstring applies.

    The shards are independent: each is the tuned single-cluster pipeline over its own KV
    tiles, producing its own (m_c, l_c). What makes this different from four
    copies of the one-cluster run is the epilogue -- the online-softmax merge

        m* = max_c m_c        l* = sum_c exp(m_c - m*) * l_c

    is not gathered to one cluster and reduced there. It is computed BY THE FABRIC: each
    cluster packs its partial into the monoid junction's lane geometry, and one
    ChainGather walks the four of them, folding at each hop, so the collector's buffer
    receives the answer rather than the operands.
    """
    # Every cluster's L1 first, with no nodes: a broadcast names destinations in all four,
    # so the handles have to exist before the first load node does.
    bufs = [_alloc_cluster(ctx, cfg, c) for c in cfg.clusters]

    if cfg.decomp == "headpar":
        # ---- one read per tile, fanned out in the writer --------------------------------
        # ALL BROADCASTS ON ONE ENGINE (cluster 0). Spreading them over the four clusters'
        # xDMAs DEADLOCKS the fabric -- measured, and root-caused on a waveform to the
        # adapter's single from_remote context: all four clusters' receive windows open
        # inside 1.3 us and none ever closes, so every finish manager sticks in ReadBusy.
        # See docs/xdma_per_source_remote_contexts.md. BCAST_SPREAD reproduces it on
        # purpose; never enable it for a measurement.
        #
        # The nodes themselves are created inside cluster 0's own _build_cluster pass, so
        # the global node order stays what it was -- see the note there for what hoisting
        # them cost.
        # "vsrc" carries cluster 0's V load nodes so the other three can depend on them,
        # the same deferral the broadcast nodes already use.
        bcast = {"k": {}, "v": {}, "vsrc": {}, "vpull": [],
                 "ksrc": {}, "kpull": []}
        shards = [_build_cluster(ctx, cfg, c, h, m_all, rowsum_all, bufs[c], bcast, bufs,
                                 verify=verify)
                  for c in cfg.clusters]

        # ---- close the forward references ----------------------------------------------
        # Under BCAST_SPREAD cluster c consumes tiles issued by clusters built after it, so
        # those RAW edges could not be stated inline. They are ordinary edges; only their
        # statement is deferred.
        for sh in shards:
            for idx, j in sh["pending_k"]:
                ctx.dfg.bingo_add_edge(bcast["k"][j], sh["qk"][idx])
            for idx, j in sh["pending_v"]:
                ctx.dfg.bingo_add_edge(bcast["v"][j], sh["pv"][idx])

        # ---- close the cross-cluster WAR edges ------------------------------------------
        # A broadcast buffer is free only when EVERY cluster's consumer of the tile 2 (resp.
        # NVBUF) steps back has retired. Those consumers do not exist while the broadcast
        # node is being created, so the edges are added here. One producer, four consumers:
        # this is the coupling that head-parallelism buys its bandwidth with.
        # Only the BROADCAST needs this: one producer writes every cluster's k8, so it must
        # wait for every cluster's consumer. Under K_PULL each cluster fills its own buffer
        # and its own k_after already covers its own consumer -- the only extra edge needed is
        # puller -> cluster 0's refill, added below.
        # The K push has exactly the broadcast's shape -- one producer writing every
        # cluster's k8 -- so it needs exactly the broadcast's WAR edges.
        if (cfg.bcast_k or cfg.k_push_all) and not cfg.k_pull_from_cl0:
            for j in range(2, cfg.nkv_per):
                for sh in shards:
                    ctx.dfg.bingo_add_edge(sh["qk"][(j - 1) * cfg.nq - 1], bcast["k"][j])
        if cfg.bcast_v:
            for j in range(cfg.nvbuf, cfg.nkv_per):
                for sh in shards:
                    ctx.dfg.bingo_add_edge(sh["pv"][(j - cfg.nvbuf + 1) * cfg.nq - 1], bcast["v"][j])
        # A pull whose source cluster is built later than the puller could not name its
        # producer inline. These are ordinary RAW edges; only their statement is deferred.
        for key, srcmap in (("kpend", "ksrc"), ("vpend", "vsrc")):
            for j, n in bcast.get(key, []):
                ctx.dfg.bingo_add_edge(bcast[srcmap][j], n)

        # WAR ACROSS CLUSTERS. A puller of tile j reads the OWNER's buffer,
        # all_bufs[own(j)][slot j % NBUF]. That buffer is next overwritten by the owner's
        # OWN fill of the same slot, which is its ld_k/ld_v[j + NBUF] -- so the edge must
        # be stated against that node, on that cluster.
        #
        # Naming it through `ksrc` instead does not work once PULL_ROTATE is on: `ksrc[t]`
        # is the node on cluster t % NCL, and the node that really overwrites own(j)'s slot
        # is often a PULL, which lives in `kpull` and never appears in ksrc at all.
        #
        # Get it wrong and the source buffer is overwritten mid-pull: the puller gets a later
        # tile's K, its scores are wrong, and the running max m comes out far too large.
        # It is latent at NSCORE=2 and at four tiles per cluster -- the schedule has no
        # slack to open the window -- and fires every time at NSCORE=3 with sixteen.
        def _close_pull_war(pulls, key, nbuf):
            for j, n in pulls:
                own = _pull_owner(cfg, j)
                fills = shards[own].get(key) or []
                t = j + nbuf
                nxt = fills[t] if t < len(fills) else None
                if nxt is not None and nxt is not n:
                    ctx.dfg.bingo_add_edge(n, nxt)

        if cfg.k_pull_from_cl0:
            _close_pull_war(bcast["kpull"], "ld_k", cfg.nkbuf)
        if cfg.v_pull_from_cl0:
            _close_pull_war(bcast["vpull"], "ld_v", cfg.nvbuf)
        # THE HOST EPILOGUE WAITS FOR EVERY CLUSTER, NOT JUST ITS OWN.
        #
        # Each shard's stores only needed its OWN cluster's compute, so with four clusters
        # skewed -- which they are, increasingly so with more tiles -- Store_o_c2 would issue
        # a 16 KB host iDMA read out of cluster 2's L1 while clusters 0/1/3 were still
        # streaming K and V through the same fabric. That combination wedges the machine:
        # host and all sixteen snitch traces freeze together and VCS keeps burning CPU. It is
        # the RTL fragility in docs/soc_bottlenecks.md section 9, and this is the SW way
        # around it.
        #
        # It costs nothing measurable. Every utilisation figure here ends its window at the
        # LAST PV, so the host tail sits outside the measurement; all this does is stop the
        # tail from overlapping live traffic.
        last_pvs = [sh["pv"][-1] for sh in shards if sh["pv"]]
        for sh in shards:
            for lp in last_pvs:
                if lp is not sh["pv"][-1]:
                    if sh["first_store"] is not None:
                        ctx.dfg.bingo_add_edge(lp, sh["first_store"])

        # No fold: under head-parallelism each cluster's (m, l, O) is already the complete
        # answer for its own query head. The per-shard checks are the whole verification.
        return shards

    shards = [_build_cluster(ctx, cfg, c, h, m_all, rowsum_all, bufs[c], verify=verify)
              for c in cfg.clusters]
    g = ctx.at(0)

    # De-synchronise the cold xdma_1d_copy config (see STAGGER_FIRST_V). One edge per
    # consecutive pair; the transfers still overlap, only the cold CONFIGS are serialised.
    if cfg.stagger_first_v:
        prev = None
        for sh in shards:
            fv = sh.get("first_v")
            if fv is None:
                continue
            if prev is not None:
                ctx.dfg.bingo_add_edge(prev, fv)
            prev = fv

    return shards


# ======================================================================================
# The public surface
# ======================================================================================

def _check_mesh(cfg: FaCfg, hwcfg=None, mesh=None, array_shape: int = 0) -> FaCfg:
    """Refuse a cfg whose array is not the one the kernels were written for."""
    want = tuple(mesh) if mesh is not None else \
        (mesh_from_hwcfg(hwcfg, array_shape) if hwcfg is not None else cfg.mesh)
    if want != _MESH_DEFAULT or tuple(cfg.mesh) != _MESH_DEFAULT:
        raise ValueError(
            f"this attention is written for (Mu, Ku, Nu) = {_MESH_DEFAULT} but the cluster "
            f"declares {want}. Bc, Br and d all derive from the array, so every descriptor "
            f"and every golden would be wrong -- and neither would fault.")
    return cfg


def configure(param: dict, *, hwcfg=None, mesh=None, array_shape: int = 0) -> FaCfg:
    """Set the shape from a params dict and publish it. Returns the FaCfg.

    PASS hwcfg (or mesh). The mesh is not a knob -- it is what the RTL was elaborated
    with -- and every buffer size and tile count derives from it, so a wrong one is a
    graph that builds, runs and computes the wrong thing rather than a fault. The default
    is the array these kernels were written for, and it is CHECKED, not assumed.

    One shape per process; see the module docstring.
    """
    want = tuple(mesh) if mesh is not None else \
        (mesh_from_hwcfg(hwcfg, array_shape) if hwcfg is not None else _MESH_DEFAULT)
    return _check_mesh(FaCfg.from_params(param, mesh=want), mesh=want).validate()


def configure_cfg(cfg: FaCfg) -> FaCfg:
    """Validate an already-built FaCfg. Kept so both entry points read the same."""
    return cfg.validate()


def geometry(cfg) -> dict:
    """The derived shape as a plain dict, for a caller sizing its own buffers."""
    return {"M": cfg.m, "K": cfg.k, "N": cfg.n, "NKV": cfg.nkv, "NQ": cfg.nq,
            "NCL": cfg.ncl, "BC": cfg.bc, "BR": cfg.br, "DHEAD": cfg.dhead,
            "NKV_PER": cfg.nkv_per, "S2_M": cfg.s2_m, "S2_K": cfg.s2_k, "S2_N": cfg.s2_n,
            "QSHIFT": cfg.qshift, "DECOMP": cfg.decomp, "NSCORE": cfg.nscore,
            "NKBUF": cfg.nkbuf, "NVBUF": cfg.nvbuf,
            "MONOID_SLOTS": cfg.monoid_slots, "mesh": tuple(cfg.mesh)}


def fa_alloc(ctx: Ctx, cfg, clusters=None) -> list:
    """Every cluster's L1, with no nodes.

    Allocated before the first node on purpose: a broadcast names destinations in all four
    clusters, so the handles have to exist before the load node that references them does.
    """
    cls = cfg.clusters if clusters is None else clusters
    return [_alloc_cluster(ctx, cfg, c) for c in cls]


class FlashAttention(Block):
    """FlashAttention over one or more clusters, as one block.

    PORTS -- three, whatever the decomposition. The caller hands over Q, K and V and says
    in `cfg.clusters` which clusters to run on; how the operands are split across them is
    this block's business.

    WHY THIS ONE SPANS CLUSTERS when every other block runs on one. The four pipelines are
    INTERLEAVED, not merely concurrent: QK(i+1) is emitted before PV(i), the head builder
    runs inside the per-cluster pass, and the broadcast nodes are created inside cluster
    0's pass rather than in front. Node creation order is dispatch order here, so emitting
    four independent per-cluster blocks back to back would be a different schedule. What
    crosses the boundary is still explicit: the outputs below are per-cluster ports.

      in   q   queries, B-layout int8. [clusters*Br, d] under headpar (one query head per
               cluster, stacked), [Br, d] under kvsplit (one head, shared).
           k   keys,    A-layout int8. [clusters*Bc, d] under kvsplit (disjoint KV shards,
               stacked), [Bc, d] under headpar (one KV head, shared -- that IS the GQA
               relation, not a simplification).
           v   values,  A-layout int8. [Bc, d], shared either way.

      out  o_c{c}    cluster c's accumulator, INT32, in the D port's SCATTER layout
                     ("d32"), living in that cluster's L1.
           m_c{c}    cluster c's running max   -- kvsplit with >1 cluster only.
           l_c{c}    cluster c's running sum   -- likewise.

    WHY THE OUTPUT STAYS WHERE IT IS. There is no gather here. Under headpar there is
    nothing to fold: each cluster owns a different query head and its O is final. Under
    kvsplit each cluster holds a partial over its own slice of the KV axis, and O_c is
    NOT the answer -- it is one term of an online-softmax combine that has to happen
    somewhere. Saying so in the port list is the point: `o_c{c}` is explicitly per-cluster
    and explicitly in L1, so a consumer cannot mistake a partial for a result.

    THE PARTIAL IS THE TRIPLE (m, l, O), and it is three ports because that is three
    buffers: the running max lives in the softmax arena, the running sum in the last score
    buffer, the accumulator in its own region. `fa_gather` folds the first two in the
    fabric; scaling and summing the O_c is the caller's, and it needs the m* the fold
    returns.

    TODO: CAUSAL MASKING. There is none, here or anywhere else in this tree. Every query
    tile attends to every key tile, which is CORRECT for a decode step -- the new token
    legitimately sees the whole cache -- and WRONG for a real prefill, where query i must
    not see key j > i. Without a mask, prefill is right only for the last query row of the
    last tile and quietly wrong for every other, by an amount that shrinks as the sequence
    does; a toy sequence makes it look nearly right.

    Note that `nq` alone does not make this block a prefill. It sets how many query tiles
    the pipeline runs, and that is what fa_prefill_4cluster varies (NQ=2 against decode's
    NQ=1) -- the two workloads differ in nothing else. So "prefill" in this tree means
    multiple query tiles over FULL attention, and this block expresses exactly that.

    Adding the mask is a SIMD-kernel change, not a parameter: the score tile is masked
    between the QK matmul and the softmax, and at Br = Bc = 32 the mask is diagonal only
    on the tile where the query and key ranges overlap -- tiles strictly below it need no
    mask, tiles strictly above are skipped entirely, which is also where the saving is.

    THE THREE THINGS EVERY PORT CARRIES. Layout, precision and location are all in the
    signature, because none of the three faults when it is wrong -- a mismatched layout is
    a permutation that computes a scrambled answer, a mismatched precision reads two
    elements as one, and an address in a memory pool the platform does not have is simply
    unmapped. Whatever is bound is checked against the declaration, and this block's own
    build calls `comm.transfer` to close the gap where it can (hoisting from the memory
    chiplet, converting the layout in one xDMA pass) or refuses with the reason.

    The GOLDENS (m, rowsum, o) are constructor arguments, not ports: they are verification
    data and never flow between blocks.
    """
    name = "flash_attention"

    def __init__(self, cfg: FaCfg = None, *, goldens=None,
                 hwcfg=None, mesh=None, verify=None, **params):
        """Either pass a FaCfg, or the parameters directly:

            FlashAttention(bc=32, br=32, dhead=128, nkv=8, clusters=(0, 1),
                           decomp="kvsplit")

        `clusters` names where the pipelines run and `decomp` how the work is split, and
        the block generates a DIFFERENT graph for each: kvsplit gives every cluster a
        disjoint KV shard, headpar gives every cluster a query head. One cluster is legal
        in either.

        `goldens` is optional. Without it the block builds compute only (verify=False),
        which is what a composed graph wants -- its operands came from a projection rather
        than from a datagen, so there is nothing to compare against.
        """
        if isinstance(cfg, dict):
            # A params dict: M/K/N/NKV/ACTIVE_CLUSTERS/DECOMP.
            cfg = FaCfg.from_params(cfg, mesh=mesh or _MESH_DEFAULT)
        elif cfg is None:
            cfg = FaCfg.from_shape(mesh=mesh or _MESH_DEFAULT, **params)
        self.cfg = configure_cfg(cfg) if (hwcfg is None and mesh is None) \
            else configure_cfg(_check_mesh(cfg, hwcfg, mesh))
        self.goldens = goldens
        self.verify = (goldens is not None) if verify is None else verify
        if self.verify and goldens is None:
            raise ValueError("FlashAttention(verify=True) needs goldens: the check tolerances "
                             "are derived from them, not fixed.")

    # ---- which operand is per-cluster, and which is shared ------------------------------
    # One rule, stated once, so the ports, the slicing and the transfers cannot disagree.

    @property
    def _split(self) -> dict:
        """How many cluster-sized tiles each input holds. 1 means every cluster reads it."""
        ncl = self.cfg.ncl
        if self.cfg.decomp == "headpar":
            return {"q": ncl, "k": 1, "v": 1}
        return {"q": 1, "k": ncl, "v": 1}

    @property
    def inputs(self) -> dict:
        cfg, sp = self.cfg, self._split
        # `operand_level` IS WHERE THE CALLER KEEPS Q, K AND V -- main memory on one
        # platform, the memory-chiplet pool on another. It is stated rather than resolved
        # because it is a property of how the workload staged its data, which no
        # neighbouring block can tell it. Where this block's own engines read is
        # `_reads_from` below, and the gap between the two is what build() hoists.
        return {
            "q": PortSpec(Layout.B, DType.I8, (cfg.br * sp["q"], cfg.dhead),
                          mem_level=cfg.operand_level,
                          doc=("one query head per cluster, stacked"
                               if sp["q"] > 1 else "one query head, shared")),
            "k": PortSpec(Layout.A, DType.I8, (cfg.bc * sp["k"], cfg.dhead),
                          mem_level=cfg.operand_level,
                          doc=("one disjoint KV shard per cluster, stacked"
                               if sp["k"] > 1 else "one KV head, shared")),
            "v": PortSpec(Layout.A, DType.I8, (cfg.bc, cfg.dhead),
                          mem_level=cfg.operand_level,
                          doc="values, shared by every cluster"),
        }


    # The level this block's own transfers read from. The per-tile loads are cluster iDMA
    # and xDMA reads of main memory, so an operand has to have reached L3 by the time they
    # run; anything further out is hoisted once in the prologue rather than fetched per
    # tile across the D2D link.
    _reads_from = "L3"

    @property
    def needs(self) -> dict:
        """`inputs`, with the memory level this block's transfers actually require.

        Kept separate from `inputs` on purpose. `inputs` is the CONTRACT -- where the
        caller hands the operand over. This is the INTERNAL requirement, what build()
        hands to transfer.bring_in, and the difference between the two is exactly the
        hoist this block emits for itself.
        """
        return {nm: replace(spec, mem_level=self._reads_from)
                for nm, spec in self.inputs.items()}

    @property
    def outputs(self) -> dict:
        cfg = self.cfg
        outs = {f"o_c{c}": PortSpec(Layout.D32, DType.I32, (cfg.br, cfg.dhead),
                                    mem_level=MemLevel.L1, cluster=c,
                                    doc=f"cluster {c} accumulator, D-port scatter, NOT "
                                        f"un-permuted and NOT gathered")
                for c in cfg.clusters}
        if cfg.decomp == "kvsplit" and cfg.ncl > 1:
            # THE PARTIALS ARE OUTPUTS BECAUSE THE FOLD IS NOT PART OF THIS BLOCK, and
            # they are TWO ports rather than one because that is what they physically are:
            # the running max lives in the softmax arena and the running sum in the last
            # score buffer, written by different kernels into different allocations. The
            # single contiguous FP32 (m, l) the junction folds does not exist until the
            # gather's pack builds it, so one port here would name a buffer nothing wrote.
            for c in cfg.clusters:
                outs[f"m_c{c}"] = PortSpec(Layout.ROW_MAJOR, DType.F16, (1, cfg.br),
                                           mem_level=MemLevel.L1, cluster=c,
                                           doc=f"cluster {c} running max over its KV "
                                               f"shard, in the softmax arena")
                outs[f"l_c{c}"] = PortSpec(Layout.ROW_MAJOR, DType.F16, (1, cfg.br),
                                           mem_level=MemLevel.L1, cluster=c,
                                           doc=f"cluster {c} running sum over its KV "
                                               f"shard, in the last score buffer")
        return outs

    def _shard_handles(self, ctx, bound):
        """Slice each bound operand into the per-cluster handles the builders want.

        A shared operand is the SAME handle for every cluster, by identity -- which is
        what the loads already use to decide that one transfer serves all four, so
        sharing stays a property of the binding rather than a flag.
        """
        cfg, sp, eb = self.cfg, self._split, 1          # int8 operands
        out = {}
        for nm, tile_rows in (("q", cfg.br), ("k", cfg.bc), ("v", cfg.bc)):
            h = bound[nm].handle
            if sp[nm] == 1:
                out[nm] = [h] * cfg.ncl
            else:
                step = tile_rows * cfg.dhead * eb
                out[nm] = [at_offset(h, c * step) for c in cfg.clusters]
        return out

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        cfg, sp = self.cfg, self._split

        # THE SHAPE IS STATED TWICE -- once by the cfg, once by whatever got bound -- so
        # check the two agree rather than trusting either. Composition is exactly where
        # they drift, because the block upstream decides what it emits. The bound q and k
        # are the WHOLE tensors now, so divide out the split before comparing tiles.
        derived = FaCfg.from_ports(
            q=(bound["q"].shape[0] // sp["q"], bound["q"].shape[1]),
            k=(bound["k"].shape[0] // sp["k"], bound["k"].shape[1]),
            mesh=cfg.mesh, nkv=cfg.nkv, nq=cfg.nq, clusters=cfg.clusters,
            decomp=cfg.decomp, score_shift_extra=cfg.score_shift_extra)
        if (derived.m, derived.n, derived.k) != (cfg.m, cfg.n, cfg.k):
            raise ValueError(
                f"the bound ports say (Bc, Br, d) = ({derived.bc}, {derived.br}, "
                f"{derived.dhead}) but this block was configured for ({cfg.bc}, "
                f"{cfg.br}, {cfg.dhead}). Configure it from the producer's output shape, "
                f"or bind an operand that matches.")

        # CLOSE THE LAYOUT / PRECISION / LOCATION GAPS BEFORE BUILDING ANYTHING. When the
        # producer already emits what this block declared, `bring_in` returns the port
        # untouched and emits no node -- so composing costs nothing a hand-written graph
        # would not also pay. When it does not, the prologue appears here, ahead of every
        # load, rather than as a surprise in the middle of the pipeline.
        pre, bound = [], dict(bound)
        for nm, want in self.needs.items():
            port, nodes = transfer.bring_in(ctx, f"{self.name}_{nm}", bound[nm], want,
                                           mesh=cfg.mesh, elem_bytes=1)
            bound[nm], pre = port, pre + nodes

        g = self.goldens or {}
        sh_h = self._shard_handles(ctx, bound)
        h_all = {"a": sh_h["k"], "b": sh_h["q"], "v": sh_h["v"][0],
                 "zero": g.get("zero"), "m": g.get("m"),
                 "rowsum": g.get("rowsum"), "o": g.get("o")}

        before = set(ctx.dfg.nodes())
        shards = fa_attention(ctx, cfg, h_all, g.get("m_np"), g.get("rowsum_np"),
                              verify=self.verify)
        mine = [n for n in ctx.dfg.nodes() if n not in before]

        # The ENDS of an input port are the nodes that first read it. With q and k now one
        # tensor each, a per-cluster operand's ends are every cluster's first read of its
        # own slice -- the same set of nodes as before, differently grouped.
        outs, ins = {}, {}
        ld = {"q": [], "k": [], "v": []}
        want = self.outputs
        lay = SnaxBingoKernelSimdFaSoftmaxArgs.layout(cfg.bc, cfg.dhead)
        for c, shd in enumerate(shards or []):
            outs[f"o_c{c}"] = Port(want[f"o_c{c}"], shd["arena"],
                                   (shd["pv"][-1],), name=f"o_c{c}")
            if f"m_c{c}" in want:
                # Named at the OFFSETS the stores already use, so the ports and the checks
                # cannot point at different bytes.
                outs[f"m_c{c}"] = Port(want[f"m_c{c}"], shd["arena"].view(lay["mrun"]),
                                       (shd["sm"][-1],) if shd.get("sm") else
                                       (shd["pv"][-1],), name=f"m_c{c}")
                outs[f"l_c{c}"] = Port(want[f"l_c{c}"],
                                       shd["p8_last"].view((cfg.bc // 2) * 64),
                                       (shd["pv"][-1],), name=f"l_c{c}")
            for nm, key in (("q", "ld_q"), ("k", "ld_k"), ("v", "ld_v")):
                ld[nm] += [n for n in shd[key] if n is not None]
        for nm in ("q", "k", "v"):
            ins[nm] = Port(self.inputs[nm], bound[nm].handle, tuple(ld[nm]), name=nm)

        srcs = [n for n in mine if ctx.dfg.in_degree(n) == 0]
        return BlockResult(outputs=outs, inputs=ins, nodes=pre + mine, sources=srcs,
                           extra={"shards": shards, "prologue": pre})
