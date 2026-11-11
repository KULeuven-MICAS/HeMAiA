# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The driver of the whole-layer DECODE workloads (workloads/dsv2/layer<N>/decode/{half,layer}): one decoder layer of
DeepSeek-V2-Lite for ONE token at position P (the attention's params cache, default 1,024) over a P-token key cache, on
the 16 + 4 chiplet platform, every step on the device, in ONE graph built from the decode blocks:

    x -> RMSNorm (input_layernorm) -> MLA decode -> + x = h -> RMSNorm (post_attention_layernorm) -> MoE | FFN -> + h

    attention   ../../fa/common/fa_decode_mc.py (FaDecode): head parallel (chip = head), the cache and the new key's
                piece from the HBM of the root's memory chip, the token's key appended to the cache in HBM
    MoE         ../../moe/common/moe_decode_mc.py (layer 1): the router on the device, the routed experts' pieces
                streamed once the route is known, every GEMM INT4 but the router's (INT8)
    dense FFN   ../../ffn/common/ffn_decode_mc.py (layer 0): the prefill's mapping as one 16-row tile, INT4

Each block keeps its decode mapping (the prefill's block mapping cut to T = 1); the layer replaces only what the blocks
did on the host or through HBM at their edges: where x comes from, where the attention's y goes, where the second
block's X comes from, and where its output goes.

THE STEPS, in the order they run (C00 is the root chip, chips[0]; the glue's cluster is the attention's query cluster
of the root, params attn.q_cluster, 0):

    input norm   C00 cluster 0: x (one FP16 row, 2,048) from the HBM of the root's memory chip into L1; RMSNorm
                 (simd_rmsnorm, one row, row major), times the input_layernorm gain (FP16 multiply), the row repeated
                 into the 16 rows of the attention's query tile (one 2-D copy, source stride 0), reshaped to the A
                 layout (xDMA), quantised (simd_fp16_to_int8, the attention's s_x): the attention's x tile, which C00's
                 DM core multicasts from its L1 into every chip's L3 (fd_xsrc) behind the tile's flag (fd_xtflag), in
                 place of the block's multicast from HBM. The x row stays in L1 for the residual
    attention    FaDecode: its stages 1-8 unchanged; the y of each W_O block (row 0 of the reduced D-layout block, the
                 token's) is copied inside the collector's L1 into the layer's y row (FP16, column order) instead of
                 into HBM
    glue         C00 cluster 0: y dequantised (o_proj's per-column factors), plus x: h (kept in L1 to the end); RMSNorm,
                 times the post_attention_layernorm gain, quantised (the second block's s_h): X, one INT8 row
    X delivery   C00's DM core multicasts X from its L1 into every chip's landing area of the second block (md_xland
                 or fn_xland) behind its flag, in place of the block's multicast from HBM (see THE COLLECTIVES)
    MoE | FFN    the block unchanged, on that X
    final        the second block's output row plus h on the root (MoE: each root cluster its half of the columns, the
                 second half of h copied into cluster 1's L1 first; FFN: Y_raw dequantised by the block's y_dequant,
                 then plus h), written into the HBM of the root's memory chip (dd_out, FP16 [2048])
    append       the new key's piece from the owner's L3 into the cache's next slot in HBM, after the last chains

params layer.stop attn (the half layer): input norm, attention and glue; h and X written into the root memory chip's
HBM (dd_hout, dd_xout) and checked; no second block.

THE KEY-SPLIT ATTENTION (params layer.attn ks; hp, the default, is the head-parallel FaDecode above). The attention is
../../fa/common/fa_decode_ks_mc.py (FaDecodeKs): the keys split over the chips (one cached tile a chip on its cluster 0,
the token's tile on C00's cluster 1), every weight multiply a one-token GEMV, each chip's cache shard streamed through
its own ring. The steps that change:

    input norm   C00 cluster 0: x from HBM, RMSNorm, the input gain as above; then x in a_row, the GEMV's A: the FP16
                 row's 8-byte atoms (4 values) spread 16 bytes apart into an FP16 buffer zeroed before the window (one
                 xDMA pass; the gaps stay zero), quantised at the attention's s_x (simd_fp16_to_int8 keeps every value's
                 place): value c at byte (c // 4) * 8 + c % 4, the other bytes zero, 4,096 B. C00's DM core multicasts
                 it from its L1 into every chip's L3 (fk_xland) behind fk_xtflag, its own included; no 16-row tile
    attention    FaDecodeKs: its steps unchanged; the W_O columns of every cluster land in the root's L3 row (fk_yrow)
                 behind their flags, and once the root has all 32 the row is copied into the glue's y row in C00 cluster
                 0's L1 (in place of the block's write into HBM)
    glue         as above
    append       the new piece from C00 cluster 1's L1 into its slot of the root's cache shard in HBM (the block's two
                 copies: the key rows, then the value columns), after the second block's last chains
Every exchange buffer is zeroed before the window (the block's query and m* flags and its pass-2 arena are read before
they are written, as the block alone zeroes them). C00's prefetcher starts once x's multicast is done (the input norm's
HBM read and x's landing in C00's own L3 run beside no push into that L3). The collectives: x's multicast before every
x fetch (its flag), so before the block's query, m*, o~ and o8 multicasts and its chains, which the block orders among
themselves (fa_decode_ks_mc.py, THE COLLECTIVES); from X's multicast on as below. mlp_prefetch is after: the
attention's chains run after every one of its pushes has landed (its chains_after_stream) and the second block's pushes
start after them.

THE GEMV SECOND BLOCK (params moe.gemv, ffn.gu_gemv). X is the token's INT8 row in a_row (2 x 2,048 B, the layout
above): the glue spreads the gained FP16 row into the same zeroed FP16 buffer and quantises it at s_h; C00 multicasts
the 4,096 B into the block's landing (md_xland or fn_xland) behind its flag. The final residual takes the block's y row
as above: the MoE's root fold rows are y's two halves in column order, the FFN's Y16 row is on the root's cluster 0. The
block's checks are its GEMV forms (moe_decode_mc.DBuild.a8_gold: the A8 rows in a_row; ffn_decode_mc.DBuild.a8_root). A
GEMV advances the array's retired count, which a split dispatch's wait compares against: the compiler numbers every
array dispatch of a cluster, the GEMVs included (passes/bingo_dfg_schedule.py). stop attn writes X in a_row (4,096 B) into HBM with layer.x_arow.

THE COLLECTIVES. The fabric deadlocks on two multicast sources at once and on a multicast beside a ChainGather, so every
two multicasts and every multicast / chain pair are ordered by the graph, and check_collectives proves it (dependency
and flag edges): the input norm's x multicast (C00) before the root's ring head and the cache's multicast (both on the
root's host, which queues the memory chip's engine in order); the cache's before the owner's new piece (the block's own
wait on its last flag); the new piece before every chip's last key tile, so before W_O, the attention's chains, the
glue and X's multicast; X's multicast before every X wait of the second block (its flag), so before the MoE's route
multicast and every chain of the second block. X is multicast from C00's L1 (2 KiB), as the MoE block multicasts its
route table beside the memory chips' pushes, and not written into the memory chip's HBM and multicast from there: no
device write into a memory chip's HBM runs while its engines push, and every write into HBM (the output, the append, the
half layer's h and X) follows every multicast and chain. The only device read of HBM in the window is the input norm's
x, before the first multicast and before C00's own ring is pushed; the FFN's constants (its gate|up and output scale
factors, read from HBM) load before the window, as the attention's do.

THE SECOND BLOCK'S PUSHES (params layer.mlp_prefetch): after (default) every chip's second-block prefetcher starts once
X's multicast is done, so its pushes follow the attention's entirely, as the blocks ran on their own. early: each
chip's second-block prefetcher starts right behind the chip's attention prefetcher on the chip's host, so the memory
chip's engine pushes the second block's weights behind the attention's own and they stream during the attention: the
whole FFN, and the MoE's route-free chunks (the router's, the shared MLP's gate|up and down; the routed experts' chunks
still wait for the device's route). The two blocks' rings are separate L3 regions, so both fit at once. early runs the
attention's cache multicast and serial chains beside more pushes than the attention block alone. early_cache: as early,
but the chips fed by the root's memory chip (whose two links also carry the cache multicast) start their second-block
prefetchers only once the cache multicast is done, so the cache, which the attention's sweep waits for, has those links
to itself.

THE PUSH STREAMS. check_collectives also proves, for every ChainGather and every block's ring pushes on every chip,
that the chain either follows the landing of the last chunk of each of the chip's rings of that block or precedes the
chip's first prefetcher task of that block (no chain runs beside a memory-chip push), that every write into HBM follows
every ring's last chunk, and that no device task reads HBM in the window but the input norm's x. With the key-split
attention it raises otherwise; the head-parallel layer prints what it finds. params layer.chains_after_stream: the
second block's first chain also waits for the last chunk of every ring of that block (the FFN's gathers otherwise start
while its down chunks still stream; the MoE's tail chains already wait for them). Two chains that may run at once pass
the chips in the same order (opposite directions deadlock).

THE GOLDEN is the device's arithmetic end to end (hwmodel rmsnorm, fp.mul16, fp.add16, fp.quant_i8): the attention
block's data for the token at position P (its x1[P], its fold order), the glue's operators, the second block's data on
the glue's h (its static scales calibrated on that one row), the final residual. The checks, byte for byte: the
attention's (the cache's and the new piece's landing on the chip in the middle of the fold order, the appended piece in
HBM; ks: m* and o~ on the root, the partial and x's landing on the chip in the middle of the fold order, the appended
piece in HBM, and with stop attn that chip's query tile) and its y row; h and X on C00, X's landing on the chip in the
middle of the fold order; the second block's own (MoE: the route table, the root's row lists and A8 rows; FFN: the
root's A8); the output row in HBM (stop attn: h and X in HBM).

ATTENTION WEIGHTS OF LAYER 0. The attention data (fa_prefill_data.py, its 8K-token cache of hidden states x1) is layer
1's only, so the layer-0 workload runs layer 1's attention (the checkpoint's layer-1 q_proj, kv_a/kv_b, o_proj,
input_layernorm and the layer-1 cache; the same shapes and timing) in front of layer 0's dense FFN and its
post_attention_layernorm (the FFN data's).

PARAMETERS (params.hjson): the platform and the graph switches every block shares (num_chiplets, num_clusters,
static_l1, l1_capacity, l1_order, compact_id_maps, gid_lists, per_chip_image, desc_in_image, fast_host_init, d2d_ddr,
d2d_rx_yield, relay_per_gather), then `attn` (fa_decode_mc.py's), `moe` (moe_decode_mc.py's) or `ffn`
(ffn_decode_mc.py's, with y_dequant), and `layer`: attn (hp | ks: the attention block; ks takes fa_decode_ks_mc.py's
params in `attn`), mlp (moe | ffn: the second block), stop (full | attn), mlp_prefetch (after | early | early_cache; ks:
after), x_arow (stop attn: X in a_row; stop full takes the second block's form), chains_after_stream.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WL = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "multichip"))
sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "common"))
from driver import repo_root, run_driver  # noqa: E402
REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
for sub in ("ffn", "fa", "moe"):
    sys.path.insert(0, os.path.join(WL, sub, "common"))

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelIdma2dCopyArgs, SnaxBingoKernelIdmaPutFlaggedArgs,
    SnaxBingoKernelSimdAddF16Args, SnaxBingoKernelSimdFp16ToInt8Args, SnaxBingoKernelSimdMulF16Args,
    SnaxBingoKernelSimdRmsnormArgs, SnaxBingoKernelXdma6dArgs, SnaxBingoKernelXdmaMemsetArgs, simd_beats)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs.comm.nest import convert_args  # noqa: E402

import chiplet_mc as cm  # noqa: E402
import fa_decode_ks_mc as fdk  # noqa: E402
import fa_decode_mc as fdm  # noqa: E402
import ffn_data  # noqa: E402
import ffn_decode_mc as ffd  # noqa: E402
import model_data  # noqa: E402
import moe_decode_mc as mdm  # noqa: E402
import moe_prefill_data as mpd  # noqa: E402

HIDDEN = 2048
BR = 16                       # rows of the attention's query tile
TILE16 = 16 * 16 * 2          # one D-layout FP16 block of the (16, 4, 16) array
LINK_BW = 231.2e9 / 8         # bytes a second of one memory-chip link (the eight links' aggregate over eight)
ATOM, LANES = fdk.ATOM, fdk.LANES     # the xDMA's 8-byte atom (4 FP16 values) and its lanes a beat


def _f32bits(v):
    return int(np.asarray(v, dtype=np.float32).reshape(()).view(np.uint32))


def _spread_args(src, dst, n):
    """The xDMA pass that spreads an FP16 row of n values (src) into a_row's FP16 form (dst): each 8-byte atom 16 bytes
    apart, a beat of LANES atoms at a time; dst's other bytes are left as they are (zeroed before the window)."""
    b = n // 4 // LANES
    return SnaxBingoKernelXdma6dArgs(src, dst, ATOM, 2 * ATOM, [LANES * ATOM], [b], [LANES * 2 * ATOM], [b])


def _reads_hbm(t):
    """Whether task t is a copy whose source is in HBM (an address, or a handle of mem_level HBM)."""
    src = getattr(t.kernel_args, "src_addr", None)
    return isinstance(src, int) or getattr(src, "mem_level", None) == "HBM"


# ---- the blocks, over the layer's graph ---------------------------------------------------------------------------
class AttnPart(fdm.FaDecode):
    """The MLA decode in the layer: its window is the layer's, its x the input norm's multicast, its y kept in the
    collector's L1 for the glue, its append and checks made by the layer."""
    stream_name = "zs_fd_stream"     # sorts after the block's buffers, as "stream" does alone (the packer's order)

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def setup(self, extra=None):
        self.go = self.layer.open_window(extra)
        return self.go

    def x_multicast(self, go, bcast):
        return self.layer.attn_x_put

    def stage_store(self, i, red, half=None, clusters=None):
        """Row 0 of the collector's reduced block (every row is the token's), a 32-byte run a 16-column block, into the
        layer's y row in the same L1, at the block's columns."""
        R, dm, L = self.y_root(i, half), self.roles["dm"], self.layer
        nblk = (HIDDEN // 16) // (1 if half is None else 2)
        off = 0 if half is None else half * nblk * 32
        st = {}
        for c in (list(clusters) if clusters is not None else range(self.NC)):
            if (R, c) != (L.ROOT, L.GC):
                raise ValueError(f"the attention's y lands on chip {R:02x} cluster {c}; the glue runs on the root's "
                                 f"cluster {L.GC} (params attn.q_cluster, no y_roots)")
            deps = list(red[c]) if isinstance(red[c], (list, tuple)) else [red[c]]
            st[c] = self.node(R, c, dm, "g_yrow" + ("" if half is None else f"_h{half}"),
                              "__snax_bingo_kernel_idma_2d_copy",
                              SnaxBingoKernelIdma2dCopyArgs(self.dst(i, c, half), L.b["y"].view(off), 32, TILE16, 32,
                                                            nblk), deps)
            self._y_stores.append(st[c])
        return st

    def kv_append(self, deps):
        # the whole layer appends at its end, after the second block's last chains (LayerBuild.close)
        return super().kv_append(deps) if self.layer.stop == "attn" else None

    def end_window(self, after):
        self.layer.attn_ends = [t for t in after if t is not None]

    def decode_checks(self):
        """The layer makes the attention's checks once its end stamp exists (LayerBuild.attn_checks)."""


class AttnKsPart(fdk.FaDecodeKs):
    """The key-split MLA decode in the layer (params layer.attn ks): its window is the layer's, its x the input norm's
    multicast of the a_row, its y row copied from the root's L3 into the glue's L1, its append and checks made by the
    layer."""
    x_on_host = False                # x is the input norm's (LayerBuild.input_norm_ks)
    y_to_hbm = False                 # y stays on the root for the glue (stage_store)

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def setup(self, extra=None):
        self.go = self.layer.open_window(extra)
        return self.go

    def x_multicast(self, go, bcast):
        return self.layer.attn_x_put

    def stage_store(self, yw):
        """The root's y row (fk_yrow, FP16 in column order) from its L3 into the glue's y row in the root cluster 0's
        L1, after yw (the root's wait for every cluster's columns)."""
        L = self.layer
        st = self.copy(self.ROOT, L.GC, "g_yrow", self.l3("yrow"), L.b["y"], 2 * HIDDEN, [yw])
        self._y_stores.append(st)
        return st

    def kv_append(self, deps):
        # the whole layer appends at its end, after the second block's last chains (LayerBuild.close)
        return self.append_piece(deps) if self.layer.stop == "attn" else None

    def append_piece(self, deps):
        """The block's append of the new piece after deps: its key rows, then its value columns (both writes into HBM
        kept in append_writes). Returns the last."""
        last = fdk.FaDecodeKs.kv_append(self, deps)
        self.append_writes = list(self.dfg.predecessors(last)) + [last]
        return last

    def end_window(self, after):
        self.layer.attn_ends = [t for t in after if t is not None]

    def _checks(self):
        """The layer makes the attention's checks once its end stamp exists (LayerBuild.attn_checks_ks)."""


class MoePart(mdm.DBuild):
    """The MoE decode in the layer: its data the glue's h, its X the glue's multicast, its output the layer's final
    residual."""
    stream_name = "zs_md_stream"

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def _data(self, args, p, hw):
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        return mpd.MoePrefillData(cache, hw=hw, tokens=self.T, inter_e=p.get("inter_e"),
                                  inter_sh=p.get("inter_sh"), rows=self.layer.mlp_rows)

    def setup(self, extra=None):
        self.go = self.layer.mlp_window(extra)
        return self.go

    def x_multicast(self, go):
        return self.layer.mlp_put

    def _split_prefetch(self, go, after):
        """The block's two prefetcher tasks a chip; the first after X's multicast (after) or right behind the chip's
        attention prefetcher (early)."""
        L = self.layer
        if not L.early:
            return super()._split_prefetch(go, after)
        first = super()._split_prefetch({k: L.start for k in self.chips}, [])
        for k in self.chips:
            for dep in L.early_deps(k):
                self.edge(dep, first[k])
        return first

    def y_out(self, rows, last_red):
        return self.layer.final_moe(rows, last_red)

    def end_window(self, after):
        self.layer.close(after)
        self.end, self.check_copies = self.layer.end, self.layer.check_copies

    def _checks(self):
        """The block's checks but y in HBM (the layer checks its output there): the route table's entry on the root,
        the root cluster 1's row lists, the token's row of the root cluster 0's A8 of its shared piece and of its first
        routed piece (a8_gold: row 0 of the row-major A8, or with gemv the A8 row in a_row)."""
        L, R, Tg = self.layer, self.ROOT, self.TAG
        checks = [("route", BingoMemSymbol(f"{Tg}_rland", chip_id=R), self._route_bytes(), mdm.ROUTE_B)]
        lb = self._lists_bytes(R)
        checks.append(("lists", L.stash("lists_root_c1", R, 1, self.B[1]["lists"], lb.size, []), lb, lb.size))
        for pi, name in ((0, "a8sh"), (1, "a8_p1")):
            gold = self.a8_gold(R, pi)
            checks.append((name, L.stash(f"{name}_root", R, 0, self.B[0]["a8"][pi], gold.size, []), gold, gold.size))
        L.close_checks(checks)


class FfnPart(ffd.DBuild):
    """The dense FFN decode in the layer: its data the glue's h, its X the glue's multicast, its HBM constants loaded
    before the window, its output the layer's final residual."""
    TAG = "fn"                       # the attention block's is fd
    stream_name = "zs_fn_stream"

    def __init__(self, layer, args, p, plat, hw):
        self.layer = layer
        super().__init__(args, p, plat, hw, shared=layer)

    def _data(self, args, p):
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/llm_blocks/ffn/cache"))
        return ffn_data.FFNData(cache, p.get("model", "dsv2-lite-l0"), seed=int(p.get("seed", 1)),
                                wbits=self.wbits, tokens=1, inter=p.get("inter"), rows=self.layer.mlp_rows)

    def setup(self, extra=None):
        self.go = self.layer.mlp_window(extra)
        return self.go

    def x_hbm_bcasts(self, go):
        """X is the glue's multicast; every chip's prefetcher after it (after) or right behind the chip's attention
        prefetcher (early)."""
        L = self.layer
        for k in self.chips:
            for dep in (L.early_deps(k) if L.early else [L.mlp_put]):
                self.edge(dep, self.prefetch[k])
        return [L.mlp_put]

    def y_out(self, ready):
        return self.layer.final_ffn(ready)

    def end_window(self, after):
        self.layer.close(after)
        self.end, self.check_copies = self.layer.end, self.layer.check_copies

    def _checks(self):
        """The block's A8 check (a8_root: the first 2 KiB of the root's A8, the token's row 0 and the zero rows, or with
        gu_gemv the chip's A8 row in a_row); Y_raw is checked as the layer's output."""
        L, R = self.layer, self.ROOT
        buf, img, n = self.a8_root()
        L.close_checks([("a8_root", L.stash("a8_root", R, 0, buf, n, []), img, n)])


# ---- the layer ------------------------------------------------------------------------------------------------------
class LayerBuild(cm.ChipletBuild):
    """The whole decode layer (the module docstring), its blocks parts of the layer's graph. run_driver makes the
    program in four steps: __init__ (the parameters, the parts made), mem_handles() (the data, the golden and the
    memory: each part's in turn, between them the layer's golden that the next one reads, then the layer's own),
    build() (the task graph: the attention's, the glue's, the second block's) and finish() (the checks of the graph,
    then the compile into the C header and the data image)."""
    TAG = "dd"

    def __init__(self, args, p, plat, hw):
        """The parameters; no data, memory or task yet. ChipletBuild.__init__ with the switches every block shares;
        params layer (stop, attn, mlp, mlp_prefetch, chains_after_stream, x_arow); then the parts, each a block over the
        layer's graph and staging with its own params: the attention (AttnPart, or AttnKsPart with attn ks) and, stop
        full, the second block (MoePart or FfnPart). X's form follows the second block's (one INT8 row, or in a_row with
        the GEMV)."""
        common = {k: v for k, v in p.items() if k not in ("attn", "moe", "ffn", "layer")}
        lp = self.lp = dict(p.get("layer", {}))
        super().__init__(args, common, plat, hw)
        self.xch_zero = False
        self.stop = str(lp.get("stop", "full"))
        if self.stop not in ("full", "attn"):
            raise ValueError(f"params layer.stop={self.stop!r}: full or attn")
        self.attn_kind = str(lp.get("attn", "hp"))
        if self.attn_kind not in ("hp", "ks"):
            raise ValueError(f"params layer.attn={self.attn_kind!r}: hp (head parallel) or ks (key split)")
        self.mlp_kind = str(lp.get("mlp", "moe"))
        if self.mlp_kind not in ("moe", "ffn"):
            raise ValueError(f"params layer.mlp={self.mlp_kind!r}: moe or ffn")
        if self.stop == "full" and self.mlp_kind not in p:
            raise ValueError(f"params layer.mlp {self.mlp_kind}: needs the {self.mlp_kind} section")
        pf = str(lp.get("mlp_prefetch", "after"))
        if pf not in ("after", "early", "early_cache"):
            raise ValueError(f"params layer.mlp_prefetch={pf!r}: after, early or early_cache")
        if self.attn_kind == "ks" and pf != "after":
            raise ValueError(f"params layer.mlp_prefetch={pf!r} with attn ks: after (the second block's pushes start "
                             f"after the attention's chains)")
        self.early = pf in ("early", "early_cache")
        self.early_cache = pf == "early_cache"
        self.chains_after_stream = bool(lp.get("chains_after_stream", False))
        self.part_checks, self.hbm_writes, self.check_copies = [], [], []

        # ---- the parts -------------------------------------------------------------------------------------------
        Part = AttnPart if self.attn_kind == "hp" else AttnKsPart
        self.attn = Part(self, args, dict(common, **p["attn"]), plat, hw)
        self.mlp = None
        if self.stop == "full":
            pm = dict(common, **p[self.mlp_kind])
            if self.mlp_kind == "moe":
                M = self.mlp = MoePart(self, args, pm, plat, hw)
                self.mlp_gemv = M.gemv
            else:
                F = self.mlp = FfnPart(self, args, pm, plat, hw)
                if not F.y_dequant or not F.x_row:
                    raise ValueError("params ffn: y_dequant and x_row (the final residual adds Y16 to h; X is one row)")
                self.mlp_gemv = F.gu_gemv
            # X as the second block reads it: one INT8 row, or with the GEMV the row in a_row
            if "x_arow" in lp and bool(lp["x_arow"]) != self.mlp_gemv:
                raise ValueError(f"params layer.x_arow={lp['x_arow']}: the second block takes X "
                                 f"{'in a_row' if self.mlp_gemv else 'as one row'}")
            self.x_arow = self.mlp_gemv
        else:
            self.mlp_p = dict(p.get(self.mlp_kind, {}))     # the second block's params: its layer's post-attention gain
            self.mlp_gemv = False
            self.x_arow = bool(lp.get("x_arow", False))
        self.xbytes = (2 if self.x_arow else 1) * HIDDEN

    def mem_handles(self):
        """The data, the golden and the memory; the only tasks made are the blocks' prefetchers. The attention's
        (AttnPart's or AttnKsPart's mem_handles), then the golden of the input norm and of the glue on its data: x, the
        attention's x (checked against the block's), y, h, and the rows the second block's data is made on. Stop full:
        the second block's (MoePart's or FfnPart's mem_handles, on h), the output's golden and X's landing area and flag
        of the block; stop attn: the post-attention gain and s_h of the second block's layer. X, checked against the
        second block's golden X. One exchange buffer a cluster for both blocks. The layer's own data: x and the
        output (stop attn: h and X) in the HBM of the root's memory chip, the constants in L3, and C00 cluster 0's L1
        buffers of the input norm, the glue and the final step."""
        R = self.ROOT
        m = self.mem[R]
        loc = (m >> 4, m & 15)

        # ---- the attention, and the input norm's and the glue's golden -------------------------------------------
        A = self.attn
        A.mem_handles()
        self.GC = A.qc if self.attn_kind == "hp" else A.QC
        d = A.d
        fp, hm = d.fp, d.hm
        if self.GC != 0:
            raise ValueError("params attn.q_cluster: 0 (the glue and the FFN's output row share the root's cluster 0)")
        self.x16 = np.asarray(d.x1[A.P], dtype=np.float16)                       # the token's hidden state
        self.in_gain = np.asarray(d.in_norm, dtype=np.float16)
        self.oscale = np.asarray(d.out_scale(), dtype=np.float16)
        self.x8_in = fp.quant_i8(fp.mul16(hm.rmsnorm(self.x16), self.in_gain), d.s_x.inv)
        if self.attn_kind == "hp":
            tile = d.layout.to_a(np.repeat(self.x8_in[None, :], BR, axis=0), d.mesh).astype(np.int8)
            if not np.array_equal(tile.reshape(-1), np.asarray(A.xsrc).reshape(-1)):
                raise AssertionError("the input norm's x tile and the attention block's golden x differ")
            self.y_raw = np.asarray(A.y_raw(0, BR)[0], dtype=np.float16)          # the root's fold order
        else:
            if not np.array_equal(self.x8_in, np.asarray(A.gold.x8)):
                raise AssertionError("the input norm's x row and the key-split attention's golden x differ")
            self.y_raw = np.asarray(A.gold.y_raw, dtype=np.float16)              # the chain's fold order
        self.h16 = fp.add16(fp.mul16(self.y_raw, self.oscale), self.x16)
        self.mlp_rows = dict(x16=self.h16[None, :], cal=self.h16[None, :], cal_act=self.h16[None, :])

        # ---- the second block ------------------------------------------------------------------------------------
        if self.stop == "full":
            M = self.mlp
            M.mem_handles()                         # its data on h (mlp_rows)
            self.post_gain, self.s_h, xm = M.d.gain16, M.d.s_h, M.d.hq[0]
            if self.mlp_kind == "moe":
                self.out16 = fp.add16(np.asarray(M.y_row, dtype=np.float16), self.h16)
                self.mlp_land, mlp_xb = M.xland(), M.XT
            else:
                self.out16 = fp.add16(np.asarray(M.gold["y16"][0], dtype=np.float16), self.h16)
                self.mlp_land, mlp_xb = M._xland(0), M.XR
            self.mlp_flag = BingoMemSymbol(f"{M.TAG}_xtflag")
            if mlp_xb != self.xbytes:
                raise ValueError(f"the second block's X landing is {mlp_xb} B; the glue makes {self.xbytes}")
        else:
            self.post_gain = np.asarray(self._post_gain(self.mlp_p), dtype=np.float16)
            amax = float(np.max(np.abs(fp.mul16(hm.rmsnorm(self.h16), self.post_gain).astype(np.float32))))
            self.s_h = hm.QScale.for_amax(amax)
            xm = None
        self.post_gain = np.asarray(self.post_gain, dtype=np.float16)
        self.x8 = fp.quant_i8(fp.mul16(hm.rmsnorm(self.h16), self.post_gain), self.s_h.inv)
        if xm is not None and not np.array_equal(self.x8, np.asarray(xm)):
            raise AssertionError("the glue's X and the second block's golden X differ")

        # ---- one exchange buffer for both blocks ----------------------------------------------------------------
        sizes = [A.xch_bytes] + ([self.mlp.NSLC * self.mlp.SL if self.mlp_kind == "moe" else self.mlp.xch_bytes]
                                 if self.mlp else [])
        self.xch_bytes = max(sizes)
        self.xch = {(k, c): self.cx_chip(k).at(c).l1(f"AAAA_{self.TAG}_xch", self.xch_bytes)
                    for k in self.chips for c in range(self.NC)}
        A.xch = self.xch
        if self.mlp:
            self.mlp.xch = self.xch

        # ---- the layer's own data -------------------------------------------------------------------------------
        st, T = self.st, self.TAG
        self.h_xin = st.put_hbm(f"{T}_xin", self.x16.view(np.uint8).reshape(-1), mem_chip=loc)
        st.put(f"{T}_c", "uint16_t", np.concatenate([self.in_gain, self.oscale, self.post_gain]).view(np.uint16))
        if self.stop == "full":
            self.h_out = st.put_hbm(f"{T}_out", np.zeros(2 * HIDDEN, dtype=np.uint8), mem_chip=loc)
        else:
            self.h_hout = st.put_hbm(f"{T}_hout", np.zeros(2 * HIDDEN, dtype=np.uint8), mem_chip=loc)
            self.h_xout = st.put_hbm(f"{T}_xout", np.zeros(self.xbytes, dtype=np.uint8), mem_chip=loc)
        # C00 cluster 0's buffers of the input norm, the glue and the final step; named to sort after the blocks' (the
        # static packer places by name and lays them over the blocks' dead ones)
        l1 = lambda name, nb, c=self.GC: self.cx_chip(R).at(c).l1(f"zt_{T}_{name}", nb)    # noqa: E731
        scr = SnaxBingoKernelSimdRmsnormArgs(0, 0, 1, HIDDEN).scratch_bytes()
        if self.attn_kind == "hp":
            self.b = dict(c=l1("c", 3 * 2 * HIDDEN), xr=l1("xr", 2 * HIDDEN), na=l1("na", 2 * HIDDEN),
                          nscr=l1("nscr", scr), rep=l1("rep", BR * 2 * HIDDEN), repa=l1("repa", BR * 2 * HIDDEN),
                          xt=l1("xt", BR * HIDDEN), y=l1("y", 2 * HIDDEN), yd=l1("yd", 2 * HIDDEN),
                          h=l1("h", 2 * HIDDEN), ga=l1("ga", 2 * HIDDEN), gscr=l1("gscr", scr),
                          x8=l1("x8", self.xbytes))
        else:
            # ks: x in a_row (xa) in place of the 16-row tile
            self.b = dict(c=l1("c", 3 * 2 * HIDDEN), xr=l1("xr", 2 * HIDDEN), na=l1("na", 2 * HIDDEN),
                          nscr=l1("nscr", scr), xa=l1("xa", fdk.XB), y=l1("y", 2 * HIDDEN), yd=l1("yd", 2 * HIDDEN),
                          h=l1("h", 2 * HIDDEN), ga=l1("ga", 2 * HIDDEN), gscr=l1("gscr", scr),
                          x8=l1("x8", self.xbytes))
        if self.mlp_kind == "moe" and self.mlp:
            self.b["h1"] = l1("h1", HIDDEN, c=1)                  # h's second half on the root's cluster 1
        if self.attn_kind == "ks" or self.x_arow:
            # the FP16 form of an a_row (4 B a value), zeroed before the window: the spreads write only the atoms
            self.b["ar16"] = l1("ar16", 4 * HIDDEN)
        pushes = ("early, after the cache on its memory chip" if self.early_cache else "early") if self.early \
            else "after X"
        print(f"[{T}] layer: one token at position {A.P}; attention "
              f"{'key split' if self.attn_kind == 'ks' else 'head parallel'}; second block "
              f"{self.mlp_kind if self.mlp else 'none'}{' (GEMV)' if self.mlp_gemv else ''} (stop {self.stop}); X "
              f"{'in a_row' if self.x_arow else 'one row'}; its pushes {pushes}; |h| max "
              f"{float(np.max(np.abs(self.h16.astype(np.float32)))):.3g}, s_h {self.s_h.s:.4g}")

    def _post_gain(self, pm):
        """stop attn: the post_attention_layernorm gain of the second block's layer (pm: that block's params), read as
        its data reads it."""
        if self.mlp_kind == "ffn":
            cache = os.path.join(REPO, pm.get("weights", "target/sim/automation/sweep/llm_blocks/ffn/cache"))
            return ffn_data.load_weights(cache, pm.get("model", "dsv2-lite-l0"))["gain"]
        import json
        cache = os.path.join(REPO, pm.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        with open(os.path.join(cache, "manifest.json")) as f:
            m = json.load(f)["dsv2-lite-bf16"]
        name = "model.layers.1.post_attention_layernorm.weight"
        return model_data.read_bf16(os.path.join(cache, m["repo"].replace("/", "__"), name + ".bin"),
                                    m["tensors"][name]["shape"])

    def _node(self, c, role, name, kernel, kargs, deps):
        return self.node(self.ROOT, c, self.roles[role], name, kernel, kargs, list(dict.fromkeys(deps)))

    # ---- the window ------------------------------------------------------------------------------------------------
    def open_window(self, extra):
        """The constants into C00's L1 and the attention's preloads, the start stamp after them, the attention's
        prefetchers behind it, and the input norm with its multicast of x."""
        T, gc, b = self.TAG, self.GC, self.b
        cld = self._node(gc, "dm", "n_c_ld", "__snax_bingo_kernel_idma_1d_copy",
                         SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(f"{T}_c"), b["c"], 3 * 2 * HIDDEN), [])
        self.attn_pre = {k: list((extra or {}).get(k, [])) for k in self.chips}
        ends = [t for k in self.chips for t in self.attn_pre[k]] + [cld]
        if self.attn_kind == "ks":
            ends = self._sym_zero() + ends
        if "ar16" in b:
            ends.append(self._node(gc, "xdma", "n_arz", "__snax_bingo_kernel_xdma_memset",
                                   SnaxBingoKernelXdmaMemsetArgs(b["ar16"], 4 * HIDDEN,
                                                                 SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), []))
        self.start = start = self.stamp("start", ends)
        self.go = {k: start for k in self.chips}
        for k in self.chips:
            self.edge(start, self.attn.prefetch[k])
        if self.attn_kind == "ks":
            self.input_norm_ks(start)
            # the root's ring after x's multicast: no push into C00's L3 beside the input norm's HBM read and x's
            # landing there
            self.edge(self.attn_x_put, self.attn.prefetch[self.ROOT])
        else:
            self.input_norm(start)
        return dict(self.go)

    def _sym_zero(self):
        """ks: every cluster's exchange buffer zeroed on its xDMA before the window, a chain a chip (as
        ChipletBuild.setup zeroes the key-split block's alone: its flags and its pass-2 arena are read before they are
        written). Returns each chip's last."""
        xdma, out = self.roles["xdma"], []
        for k in self.chips:
            prev = []
            for c in range(self.NC):
                prev = [self.node(k, c, xdma, f"xch_zero_c{c}", "__snax_bingo_kernel_xdma_memset",
                                  SnaxBingoKernelXdmaMemsetArgs(self.xch[(k, c)], self.xch_bytes,
                                                                SnaxBingoKernelXdmaMemsetArgs.PATTERN_ZERO), prev)]
            out += prev
        return out

    def mlp_window(self, extra):
        """Where the second block starts on each chip: a probe behind the attention's last chain and the block's own setup tasks on the chip (the
        FFN's zeroed X tiles, the MoE's warm-up). The chain ends before the glue starts, so the probe adds nothing to
        X's path; it makes the order the X flag gives visible to the dependency tags and the static L1 plan (the
        attention's tasks before the block's). The block's preloads from HBM (the FFN's scale factors) go before the
        window: first on their cores, ahead of the chip's last attention preload, which the start stamp waits for."""
        go = {}
        for k in self.chips:
            local = []
            for t in (extra or {}).get(k, []):
                if _reads_hbm(t):
                    self.edge(t, self.attn_pre[k][-1] if self.attn_pre[k] else self.start)
                else:
                    local.append(t)
            go[k] = self.probe(k, "mlp_go", local + [self.attn._chains[-1]])
        return go

    def early_deps(self, k):
        """What chip k's second-block prefetcher waits for with mlp_prefetch early: the chip's attention prefetcher;
        with early_cache, on the chips the root's memory chip feeds, also the attention's cache multicast."""
        deps = [self.attn.prefetch[k]]
        if self.early_cache and self.mem[k] == self.mem[self.ROOT]:
            deps.append(self.attn.cache_mc)
        return deps

    def input_norm(self, start):
        """x from HBM, RMSNorm, the input gain, the 16 rows of the query tile, the A layout, INT8, and C00's multicast
        of the tile into every chip's L3 behind its flag (the attention's x)."""
        A, gc, b = self.attn, self.GC, self.b
        ld = self._node(gc, "dm", "n_ld", "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(self.h_xin.address, b["xr"], 2 * HIDDEN), [start])
        rn = self._node(gc, "simd", "n_rms", "__snax_bingo_kernel_simd_rmsnorm",
                        SnaxBingoKernelSimdRmsnormArgs(b["xr"], b["na"], 1, HIDDEN, scratch_addr=b["nscr"]), [ld])
        gn = self._node(gc, "simd", "n_gain", "__snax_bingo_kernel_simd_stream_elementwise",
                        SnaxBingoKernelSimdMulF16Args(b["na"], b["c"].view(0), b["na"], HIDDEN // 32, 32), [rn])
        rp = self._node(gc, "dm", "n_rep", "__snax_bingo_kernel_idma_2d_copy",
                        SnaxBingoKernelIdma2dCopyArgs(b["na"], b["rep"], 2 * HIDDEN, 0, 2 * HIDDEN, BR), [gn])
        ta = self._node(gc, "xdma", "n_toa", "__snax_bingo_kernel_xdma_6d",
                        convert_args("row_major", "A", BR, HIDDEN, cm.MESH, 2, b["rep"], b["repa"]), [rp])
        qz = self._node(gc, "simd", "n_q", "__snax_bingo_kernel_simd_fp16_to_int8",
                        SnaxBingoKernelSimdFp16ToInt8Args(b["repa"], b["xt"], beats=simd_beats(BR * HIDDEN * 2), rows=1,
                                                          inv_scale_f32bits=_f32bits(A.d.s_x.inv)), [ta])
        self.n_ld = ld
        self.attn_x_put = self._node(gc, "dm", "n_bput", "__snax_bingo_kernel_idma_put_flagged",
                                     SnaxBingoKernelIdmaPutFlaggedArgs(
                                         b["xt"], BingoMemSymbol(f"{A.TAG}_xsrc"), A.XB,
                                         BingoMemSymbol(f"{A.TAG}_xtflag"), value=1, bcast=True, also_here=True), [qz])

    def input_norm_ks(self, start):
        """ks: x from HBM, RMSNorm, the input gain, the row in a_row (its atoms spread into the zeroed FP16 ar16, then
        INT8 at the attention's s_x), and C00's multicast of the 4,096 B into every chip's L3 (fk_xland) behind
        fk_xtflag, its own included (the attention's x)."""
        A, gc, b = self.attn, self.GC, self.b
        ld = self._node(gc, "dm", "n_ld", "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(self.h_xin.address, b["xr"], 2 * HIDDEN), [start])
        rn = self._node(gc, "simd", "n_rms", "__snax_bingo_kernel_simd_rmsnorm",
                        SnaxBingoKernelSimdRmsnormArgs(b["xr"], b["na"], 1, HIDDEN, scratch_addr=b["nscr"]), [ld])
        gn = self._node(gc, "simd", "n_gain", "__snax_bingo_kernel_simd_stream_elementwise",
                        SnaxBingoKernelSimdMulF16Args(b["na"], b["c"].view(0), b["na"], HIDDEN // 32, 32), [rn])
        ar = self._node(gc, "xdma", "n_toar", "__snax_bingo_kernel_xdma_6d", _spread_args(b["na"], b["ar16"], HIDDEN),
                        [gn])
        qz = self._node(gc, "simd", "n_q", "__snax_bingo_kernel_simd_fp16_to_int8",
                        SnaxBingoKernelSimdFp16ToInt8Args(b["ar16"], b["xa"], beats=simd_beats(4 * HIDDEN), rows=1,
                                                          inv_scale_f32bits=_f32bits(A.d.s_x.inv)), [ar])
        self.n_ld = ld
        self.attn_x_put = self._node(gc, "dm", "n_bput", "__snax_bingo_kernel_idma_put_flagged",
                                     SnaxBingoKernelIdmaPutFlaggedArgs(
                                         b["xa"], A.l3("xland"), fdk.XB, A.l3("xtflag"), value=1, bcast=True,
                                         also_here=True), [qz])

    def glue(self):
        """y dequantised, plus x: h; RMSNorm, the post-attention gain, INT8 (with x_arow in a_row: the gained row spread
        into the zeroed FP16 ar16 first): X. Then X's multicast into every chip's landing area of the second block (stop
        full) or h and X into HBM (stop attn)."""
        gc, b, dm = self.GC, self.b, "dm"
        rows = list(self.attn._y_stores)
        dq = self._node(gc, "simd", "g_deq", "__snax_bingo_kernel_simd_stream_elementwise",
                        SnaxBingoKernelSimdMulF16Args(b["y"], b["c"].view(2 * HIDDEN), b["yd"], HIDDEN // 32, 32), rows)
        ad = self._node(gc, "simd", "g_add", "__snax_bingo_kernel_simd_stream_elementwise",
                        SnaxBingoKernelSimdAddF16Args(b["yd"], b["xr"], b["h"], HIDDEN // 32, 32), [dq])
        rn = self._node(gc, "simd", "g_rms", "__snax_bingo_kernel_simd_rmsnorm",
                        SnaxBingoKernelSimdRmsnormArgs(b["h"], b["ga"], 1, HIDDEN, scratch_addr=b["gscr"]), [ad])
        gn = self._node(gc, "simd", "g_gain", "__snax_bingo_kernel_simd_stream_elementwise",
                        SnaxBingoKernelSimdMulF16Args(b["ga"], b["c"].view(4 * HIDDEN), b["ga"], HIDDEN // 32, 32),
                        [rn])
        if self.x_arow:
            ar = self._node(gc, "xdma", "g_toar", "__snax_bingo_kernel_xdma_6d",
                            _spread_args(b["ga"], b["ar16"], HIDDEN), [gn])
            qz = self._node(gc, "simd", "g_q", "__snax_bingo_kernel_simd_fp16_to_int8",
                            SnaxBingoKernelSimdFp16ToInt8Args(b["ar16"], b["x8"], beats=simd_beats(4 * HIDDEN), rows=1,
                                                              inv_scale_f32bits=_f32bits(self.s_h.inv)), [ar])
        else:
            qz = self._node(gc, "simd", "g_q", "__snax_bingo_kernel_simd_fp16_to_int8",
                            SnaxBingoKernelSimdFp16ToInt8Args(b["ga"], b["x8"], beats=simd_beats(2 * HIDDEN), rows=1,
                                                              inv_scale_f32bits=_f32bits(self.s_h.inv)), [gn])
        self.g_h, self.g_x = ad, qz
        if self.stop == "full":
            self.mlp_put = self._node(gc, dm, "g_bput", "__snax_bingo_kernel_idma_put_flagged",
                                      SnaxBingoKernelIdmaPutFlaggedArgs(b["x8"], self.mlp_land, self.xbytes,
                                                                        self.mlp_flag, value=1, bcast=True,
                                                                        also_here=True), [qz])
            if self.mlp_kind == "moe":
                # h's second half into the root's cluster 1, where the final step adds it to that cluster's half of y
                self.h1_cp = self._node(gc, dm, "g_h1", "__snax_bingo_kernel_idma_1d_copy",
                                        SnaxBingoKernelIdma1dCopyArgs(b["h"].view(HIDDEN), b["h1"], HIDDEN), [ad])
            return
        wh = self._node(gc, dm, "g_hout", "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(b["h"], self.h_hout.address, 2 * HIDDEN), [ad])
        wx = self._node(gc, dm, "g_xout", "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(b["x8"], self.h_xout.address, self.xbytes), [qz, wh])
        self.hbm_writes += [wh, wx] + (list(self.attn.append_writes) if self.attn_kind == "ks" else [self.attn.append])
        self.end_window([wh, wx] + list(self.attn_ends))
        self.close_checks([])

    # ---- the final step ----------------------------------------------------------------------------------------------
    def final_moe(self, rows, last_red):
        """Each root cluster's half of the MoE's y row (rows[c]) plus that half of h, into HBM once both root clusters'
        last chains are done. Returns the writes."""
        M, b, hc = self.mlp, self.b, HIDDEN // self.NC
        outs = []
        for c in range(self.NC):
            hv, hdep = (b["h"].view(2 * hc * c), self.g_h) if c == self.GC else (b["h1"], self.h1_cp)
            ad = self._node(c, "simd", f"f_add_c{c}", "__snax_bingo_kernel_simd_stream_elementwise",
                            SnaxBingoKernelSimdAddF16Args(M.yrow[c], hv, M.yrow[c], hc // 32, 32), [rows[c], hdep])
            outs.append(self._node(c, "dm", f"f_out_c{c}", "__snax_bingo_kernel_idma_1d_copy",
                                   SnaxBingoKernelIdma1dCopyArgs(M.yrow[c], self.h_out.address + 2 * hc * c, 2 * hc),
                                   [ad] + list(last_red.values())))
        self.chains_done = list(last_red.values())
        return outs

    def final_ffn(self, ready):
        """Y16's row (the FFN's y_dequant, on the root's cluster 0, after every gather) plus h, into HBM. Returns the
        write."""
        F, b = self.mlp, self.b
        ad = self._node(0, "simd", "f_add", "__snax_bingo_kernel_simd_stream_elementwise",
                        SnaxBingoKernelSimdAddF16Args(F.yrow, b["h"], F.yrow, HIDDEN // 32, 32),
                        list(ready) + [self.g_h])
        self.chains_done = list(ready)
        return self._node(0, "dm", "f_out", "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(F.yrow, self.h_out.address, 2 * HIDDEN), [ad])

    def close(self, after):
        """The append (after the new piece's multicast and the second block's last chains: no push and no chain is in
        flight then) and the end stamp after it and the output writes."""
        A = self.attn
        if self.attn_kind == "ks":
            self.append = A.append_piece(list(self.chains_done))
            self.hbm_writes += list(after) + list(A.append_writes)
        else:
            self.append = fdm.FaDecode.kv_append(A, list(A.kv_mc) + list(self.chains_done))
            self.hbm_writes += list(after) + [self.append]
        self.end_window(list(after) + [self.append])

    # ---- checks ------------------------------------------------------------------------------------------------------
    def attn_checks(self):
        """The attention block's: the first and the last cached piece and the new piece as their multicasts landed on
        the chip in the middle of the fold order, the appended piece in HBM (its head and tail), and the token's y row
        on C00."""
        A, R, n = self.attn, self.ROOT, 2048
        T, KVP = A.TAG, fdm.KVP
        far = A.order[len(A.order) // 2]
        checks = []
        for nm, q, lo in (("c0a", 0, 0), ("cLb", A.NEW - 1, KVP - n), ("nka", A.NEW, 0), ("nvb", A.NEW, KVP - n)):
            img = np.asarray(A.d.piece(q)).view(np.uint8).reshape(-1)
            checks.append((f"kv_{nm}", BingoMemSymbol(f"{T}_kv", offset=q * KVP + lo, chip_id=far), img[lo:lo + n], n))
        new = np.asarray(A.d.piece(A.NEW)).view(np.uint8).reshape(-1)
        app = A.append if self.stop == "attn" else self.append
        for nm, lo in (("a", 0), ("b", KVP - n)):
            got = self.stash(f"app_{nm}", R, 0, A.h_cache.address + A.NEW * KVP + lo, n, [app])
            checks.append((f"app_{nm}", got, new[lo:lo + n], n))
        checks.append(("y_row", self.stash("y_row", R, self.GC, self.b["y"], 2 * HIDDEN, []),
                       self.y_raw.view(np.uint8), 2 * HIDDEN))
        return checks

    def attn_checks_ks(self):
        """The key-split attention's: m* and o~ on the root, the partial and x's landing (the a_row) on the chip in the
        middle of the fold order, the appended piece in HBM (the head and the tail of its key rows and of its V slot),
        and the token's y row on C00; stop attn also that chip's query tile (with a second block its exchange buffer's
        head holds that block's slots by the end); attn.checks stages also the root's query row of that chip's head, the
        root's o8 row and the new tile's K."""
        A, R, gold, n = self.attn, self.ROOT, self.attn.gold, 2048
        far = A.order[len(A.order) // 2]
        checks = [("mstar", self.stash("mstar", R, 0, A.msrc, fdk.BEAT, [A.m_made]),
                   np.asarray(gold.mstar, np.float16).view(np.uint8), fdk.BEAT),
                  (f"loc_k{far:02x}", self.stash(f"loc_k{far:02x}", far, 0, A.xch_view(far, 0, A.OFF_LOC), fdk.PB, []),
                   gold.part_bytes(far).view(np.uint8), fdk.PB),
                  ("xt", self.stash("xt", R, 0, A.xt, fdk.OB, []), gold.xt.view(np.uint8), fdk.OB),
                  ("x_far", A.l3("xland", chip=far), fdk.a_row(gold.x8).view(np.uint8), fdk.XB)]
        if self.stop == "attn":
            checks.insert(0, ("q8", self.stash("q8", far, 0, A.xch_view(far, 0, A.OFF_X), fdk.QB, []),
                              gold.q8_tile.view(np.uint8), fdk.QB))
        if A.checks_kind == "stages":
            h = A.head[far]
            checks += [("qrow", self.stash("qrow", R, 0, A.l3("qrows", fdk.QROW * h), fdk.QROW, []),
                        gold.qrow[h].view(np.uint8), fdk.QROW),
                       ("o8row", self.stash("o8row", R, 0, A.l3("oland"), len(A.chips) * fdk.OAR, []),
                        fdk.a_row(gold.o8_all).view(np.uint8), len(A.chips) * fdk.OAR),
                       ("ktnew", self.stash("ktnew", R, 1, A.b1["kt"], fdk.PK, []), gold.piece_k.view(np.uint8),
                        fdk.PK)]
        base = A.h_kv[R].address + A.kv_new
        app = A.append if self.stop == "attn" else self.append
        vt = gold.vtile_new.view(np.uint8)
        for nm, off, img in (("kapp_a", 0, gold.piece_k[:n]), ("kapp_b", fdk.PK - n, gold.piece_k[fdk.PK - n:]),
                             ("vapp_a", fdk.KB, vt[:n]), ("vapp_b", fdk.KB + fdk.VB - n, vt[fdk.VB - n:])):
            img = np.ascontiguousarray(img).view(np.uint8)
            checks.append((nm, self.stash(nm, R, 0, base + off, n, [app]), img, n))
        checks.append(("y_row", self.stash("y_row", R, self.GC, self.b["y"], 2 * HIDDEN, []),
                       self.y_raw.view(np.uint8), 2 * HIDDEN))
        return checks

    def close_checks(self, block_checks):
        """Every check of the layer, compared by the root's host after the end stamp: the attention's, the glue's (h, X
        and, stop full, X's landing on the chip in the middle of the fold order), the second block's and the output."""
        R, gc, b = self.ROOT, self.GC, self.b
        x8 = self.x8.astype(np.int8)
        x8 = (mdm.a_row_bytes(x8) if self.x_arow else x8).view(np.uint8)
        checks = self.attn_checks_ks() if self.attn_kind == "ks" else self.attn_checks()
        checks.append(("h", self.stash("h", R, gc, b["h"], 2 * HIDDEN, []), self.h16.view(np.uint8), 2 * HIDDEN))
        checks.append(("x8", self.stash("x8", R, gc, b["x8"], self.xbytes, []), x8, self.xbytes))
        if self.stop == "full":
            far = self.attn.order[len(self.attn.order) // 2]
            land = self.mlp_land
            checks.append(("x8_far", BingoMemSymbol(land.symbol_name, offset=land.offset, chip_id=far), x8,
                           self.xbytes))
            checks += list(block_checks)
            checks.append(("out", self.stash("out", R, gc, self.h_out.address, 2 * HIDDEN, []),
                           self.out16.view(np.uint8), 2 * HIDDEN))
        else:
            checks.append(("h_hbm", self.stash("h_hbm", R, gc, self.h_hout.address, 2 * HIDDEN, []),
                           self.h16.view(np.uint8), 2 * HIDDEN))
            checks.append(("x8_hbm", self.stash("x8_hbm", R, gc, self.h_xout.address, self.xbytes, []), x8,
                           self.xbytes))
        self.run_checks(checks)

    # ---- the layer ---------------------------------------------------------------------------------------------------
    def build(self):
        n0 = len(self.dfg.node_list)
        self.attn.build()
        self.attn_ids = (n0, len(self.dfg.node_list))
        self.glue()
        if self.mlp is None:
            return
        n1 = len(self.dfg.node_list)
        self.mlp.build()
        self.mlp_ids = (n1, len(self.dfg.node_list))
        if self.chains_after_stream:
            self.mlp_chains_after_stream()
        # the second block's tasks named as an attention task (its prefetchers, made before its build, and its GEMM
        # shape copies) carry its tag, so every task of the layer has a name of its own
        seen = {t.node_name for t in self.dfg.node_list if not self._in_mlp(t)}
        for t in self.dfg.node_list:
            if self._in_mlp(t) and t.node_name in seen:
                t.node_name = f"{self.mlp.TAG}_{t.node_name}"

    def mlp_chains_after_stream(self):
        """params layer.chains_after_stream: the second block's first chain also waits for the last chunk of every ring
        of the block (its load on the cluster that reads it), so no chain of the block runs beside one of its pushes:
        the block's chains run one at a time or after its first (the MoE's tail chains already wait for every load;
        the FFN's gathers are serial). Adds the edges not yet implied."""
        import networkx as nx
        lo, hi = self.mlp_ids
        chains = [t for t in self.dfg.node_list[lo:hi] if (t.kernel_name or "").endswith("xdma_chain_gather")]
        firsts = [t for t in chains if not any(nx.has_path(self.dfg, u, t) for u in chains if u is not t)]
        last = self._last_loads(lo, hi)
        added = 0
        for t in firsts:
            anc = nx.ancestors(self.dfg, t)
            for k in self.chips:
                for ld in last[k]:
                    if ld not in anc:
                        self.edge(ld, t)
                        added += 1
        print(f"[{self.TAG}] chains_after_stream: the second block's first chain(s) {[t.node_name for t in firsts]} "
              f"after every ring's last chunk, {added} edges added")

    def _last_loads(self, lo, hi):
        """{chip: the load of the last chunk of each of its rings among the ring loads with node ids lo .. hi - 1}: a
        ring a cluster, its chunks loaded in ring order."""
        last = {}
        for t in self.dfg.node_list[lo:hi]:
            if (t.kernel_name or "").endswith("idma_ring_load"):
                key = (t.assigned_chiplet_id, t.assigned_cluster_id)
                if key not in last or t.node_id > last[key].node_id:
                    last[key] = t
        return {k: [last[(k, c)] for c in range(self.NC) if (k, c) in last] for k in self.chips}

    def _in_mlp(self, t):
        """Whether task t is the second block's: made by its build, or its prefetcher (made with its rings)."""
        return self.mlp_ids[0] <= t.node_id < self.mlp_ids[1] or t in set(self.mlp.prefetch.values())

    # ---- checks of the graph -------------------------------------------------------------------------------------
    def check_collectives(self):
        """Every two multicasts, and every multicast and ChainGather, ordered by the graph (dependency and flag edges);
        every write into HBM after every multicast and chain; the window's one device read of HBM (the input norm's x)
        before every multicast and chain. Raises otherwise."""
        import networkx as nx
        gr = nx.DiGraph(self.dfg)
        gr.add_edges_from(self.dfg.order_edges)
        nodes = self.dfg.node_list
        mc = [t for t in nodes if (t.kernel_name or "").endswith("hbm_bcast")
              or ((t.kernel_name or "").endswith("idma_put_flagged") and getattr(t.kernel_args, "bcast", 0))]
        ch = [t for t in nodes if (t.kernel_name or "").endswith("xdma_chain_gather")]
        coll = mc + ch
        desc = {t: nx.descendants(gr, t) for t in coll}
        bad = [(a.node_name, b.node_name) for i, a in enumerate(coll) for b in coll[i + 1:]
               if (a in mc or b in mc) and b not in desc[a] and a not in desc[b]]
        if bad:
            raise ValueError(f"the layer: {len(bad)} multicast / chain pair(s) may run at once: {bad[:6]}")
        late = [(a.node_name, w.node_name) for a in coll for w in self.hbm_writes if w not in desc[a]]
        if late:
            raise ValueError(f"the layer: {len(late)} HBM write(s) not after a multicast or chain: {late[:6]}")
        early = [a.node_name for a in coll if a not in nx.descendants(gr, self.n_ld)]
        if early:
            raise ValueError(f"the layer: the input norm's HBM read is not before {early[:6]}")
        print(f"[{self.TAG}] collectives: {len(mc)} multicasts and {len(ch)} chains, each ordered with every "
              f"multicast; {len(self.hbm_writes)} HBM writes after all of them, the input norm's HBM read before")
        self.check_streams(gr, ch, desc)

    def check_streams(self, gr, ch, desc):
        """The push streams against the chains and the HBM traffic: for every chain, every block and every chip, the
        chain follows the load of the last chunk of each of the chip's rings of the block (every push of it has landed)
        or precedes each of the chip's prefetcher tasks of the block (none has started); every write into HBM follows
        every ring's last chunk; no device task reads HBM in the window but the input norm's x. Two chains that may run
        at once pass the chips in the same order. With a GEMV second block, each cluster's first GEMV of it follows
        every gemm_wait of the cluster. The key-split layer raises on any of these; the head-parallel layer raises on
        chains in opposite directions and on a GEMV before a wait, and prints the rest."""
        import networkx as nx
        A, M, nodes = self.attn, self.mlp, self.dfg.node_list
        pref = [t for t in nodes if "weight_prefetch" in (t.kernel_name or "")]
        streams = {}
        parts = [("attention", A, self.attn_ids)] + ([("second block", M, self.mlp_ids)] if M else [])
        for name, part, (lo, hi) in parts:
            own = set(part.prefetch.values())
            pre = {k: [t for t in pref if t.assigned_chiplet_id == k and (t in own or lo <= t.node_id < hi)]
                   for k in self.chips}
            # the key-split attention's one ring a chip: its last chunk is the chip's last push (FaDecodeKs.last_chunk)
            last = ({k: [A.last_chunk[k]] for k in self.chips} if part is A and self.attn_kind == "ks"
                    else self._last_loads(lo, hi))
            if any(not v for v in last.values()) or any(not v for v in pre.values()):
                raise ValueError(f"the layer: the {name}'s ring loads or prefetchers not found on every chip")
            streams[name] = (last, pre)
        anc = {t: nx.ancestors(gr, t) for t in list(ch) + list(self.hbm_writes)}
        beside = []
        for t in ch:
            for name, (last, pre) in streams.items():
                for k in self.chips:
                    if not (all(x in anc[t] for x in last[k]) or all(x in desc[t] for x in pre[k])):
                        beside.append((t.node_name, name, f"{k:02x}"))
        late = [(w.node_name, name, f"{k:02x}") for w in self.hbm_writes for name, (last, _) in streams.items()
                for k in self.chips if not all(x in anc[w] for x in last[k])]
        hops = lambda t: [h[0] for h in (getattr(t.kernel_args, "hops", None) or [])]    # noqa: E731
        opposite = [(a.node_name, b.node_name) for i, a in enumerate(ch) for b in ch[i + 1:]
                    if b not in desc[a] and a not in desc[b] and hops(a) != hops(b)]
        dev = {self.roles[r] for r in ("gemm", "simd", "xdma", "dm")}
        win = nx.descendants(gr, self.start) & nx.ancestors(gr, self.end)
        reads = [t.node_name for t in nodes if t in win and t is not self.n_ld and t.assigned_core_id in dev
                 and _reads_hbm(t)]
        strict = self.attn_kind == "ks"
        if beside and strict:
            raise ValueError(f"the layer: {len(beside)} (chain, block, chip) with the chain beside that block's pushes "
                             f"on that chip: {beside[:6]}" + ("" if self.chains_after_stream else
                                                              " (params layer.chains_after_stream)"))
        if late and strict:
            raise ValueError(f"the layer: {len(late)} HBM write(s) not after a ring's last chunk: {late[:6]}")
        if opposite:
            raise ValueError(f"the layer: chains that may run at once in opposite directions: {opposite[:6]}")
        if reads and strict:
            raise ValueError(f"the layer: device reads of HBM in the window: {reads[:6]}")
        print(f"[{self.TAG}] streams: {len(ch)} chains, "
              + ("each after the last chunk of, or before the first push of, every block's ring on every chip"
                 if not beside else f"{len(beside)} (chain, block, chip) beside a push, e.g. {beside[:3]}")
              + f"; {len(self.hbm_writes)} HBM writes "
              + ("after every ring's last chunk" if not late else f"with {len(late)} (write, block, chip) not after "
                                                                   f"its last chunk")
              + "; concurrent chains in one direction; "
              + (f"device HBM reads in the window: {reads[:4]}" if reads else "no device HBM read in the window but x"))

    def stream_report(self):
        """The bytes each D2D link carries (both directions: a link is half duplex), by phase: the attention (its rings,
        the cache's multicast, x's multicast, the new piece, the slices to the owner, its chains, the input's x from
        HBM; ks: its rings with the cache shards, x's, the query's, m*'s, o~'s and the o8 row's multicasts from C00, the
        gathers into C00, its chain(s), the input's x from HBM), the second block's route-free pushes, its routed
        pushes, and the rest (X's and the route's multicasts, the second block's chains, the output and the append). The
        stream floor of each prefetch setting, at LINK_BW a link: after, the attention's busiest link then the second
        block's; early, every byte of a link through it, and the routed pushes after the attention's."""
        A, M, R = self.attn, self.mlp, self.ROOT
        grid = set(self.chips) | set(self.mem.values())
        route, tree, KVP = fdm._route, fdm._bcast_tree, fdm.KVP
        load = {ph: {} for ph in ("attn", "free", "routed", "rest")}

        def add(ph, links, nb):
            for ln in links:
                load[ph][ln] = load[ph].get(ln, 0) + nb

        if self.attn_kind == "ks":
            for k in self.chips:
                add("attn", route(self.mem[k], k), A.ring_bytes[k])
                add("attn", route(k, R), fdk.QROW + fdk.OAR + self.NC * 4 * fdk.QC + 2 * fdk.QC * len(A.kva[k])
                    + (fdk.XFER if A.m_exchange == "gather" else 0))
            for nb in (fdk.XB, fdk.QB, fdk.XFER, fdk.BR * fdk.LAT, len(self.chips) * fdk.OAR):
                add("attn", tree(R, grid), nb)                # x, the query, m*, o~, the o8 row
            hops = A.order[1:] + [R]
            for a, b in zip(hops, hops[1:]):
                add("attn", route(a, b), fdk.PB + (fdk.XFER if A.m_exchange == "chain" else 0))
            add("attn", route(self.mem[R], R), 2 * HIDDEN)
            a_ring = sum(A.ring_bytes.values())
            what = f"attention rings {a_ring:,} B (the cache shards streamed in them)"
        else:
            for k in self.chips:
                add("attn", route(self.mem[k], k), sum(t[6] for t in A.ring_take[k]))
                add("attn", route(k, A.owner), (A.kvg[k][1] - A.kvg[k][0]) * TILE16)
            add("attn", tree(R, grid), A.XB)
            add("attn", tree(self.mem[R], grid), (A.NT - 1) * fdm.TILE_PIECES * KVP)
            add("attn", tree(A.owner, grid), KVP)
            add("attn", route(self.mem[R], R), 2 * HIDDEN)
            for h in range(2):
                hops = A.y_order(A.y_root(0, h))[1:] + [A.y_root(0, h)]
                for a, b in zip(hops, hops[1:]):
                    add("attn", route(a, b), A.YB // 2)
            a_ring = sum(sum(t[6] for t in A.ring_take[k]) for k in self.chips)
            what = f"attention rings {a_ring:,} B, cache {(A.NT - 1) * fdm.TILE_PIECES * KVP:,} B in one multicast"
        mlp_free = mlp_routed = 0
        if M is not None:
            add("rest", tree(R, grid), self.xbytes)
            for cl in range(self.N_CL):
                k = self.chip(cl)
                offs = [o[1] for o in M.chunk_off[cl]]
                nf = M.n_free[k] if self.mlp_kind == "moe" else len(offs)
                add("free", route(self.mem[k], k), sum(offs[:nf]))
                add("routed", route(self.mem[k], k), sum(offs[nf:]))
                mlp_free, mlp_routed = mlp_free + sum(offs[:nf]), mlp_routed + sum(offs[nf:])
            if self.mlp_kind == "moe":
                add("rest", tree(R, grid), mdm.ROUTE_B)
                for c in range(self.NC):
                    hops = M.chain_order(c)
                    for a, b in zip(hops, hops[1:]):
                        add("rest", route(a, b), M.NSLC * M.SL)
            else:
                others = M.order[1:]
                for c in range(self.NC):
                    hops = (others[::-1] if M.chain_alt and c == 1 else others) + [R]
                    for a, b in zip(hops, hops[1:]):
                        add("rest", route(a, b), M.NDN // self.NC * M.SLICE)
            add("rest", route(R, self.mem[R]), 2 * HIDDEN)
        if self.attn_kind == "ks":
            add("rest", route(R, self.mem[R]), fdk.PK + fdk.PVB)
        else:
            add("rest", route(A.owner, self.mem[R]), KVP)
        links = sorted(set().union(*load.values()))
        tot = {ln: sum(load[ph].get(ln, 0) for ph in load) for ln in links}
        mems = set(self.mem.values())
        memlinks = [ln for ln in links if (ln[0] in mems) != (ln[1] in mems)]
        mx = lambda ph: max(load[ph].values(), default=0)    # noqa: E731
        both = {ln: load["attn"].get(ln, 0) + load["free"].get(ln, 0) + load["routed"].get(ln, 0) for ln in links}
        f_attn, f_mlp = mx("attn") / LINK_BW, max((load["free"].get(ln, 0) + load["routed"].get(ln, 0)
                                                    for ln in links), default=0) / LINK_BW
        f_after = f_attn + f_mlp
        f_early = max(max(both.values()) / LINK_BW, f_attn + mx("routed") / LINK_BW)
        mlp_total = mlp_free + mlp_routed
        print(f"[{self.TAG}] stream: {what}; second block "
              f"{mlp_total:,} B ({mlp_free:,} route-free, {mlp_routed:,} routed); weights {a_ring + mlp_total:,} B / "
              f"231.2 GB/s = {(a_ring + mlp_total) / 231.2e9 * 1e6:.1f} us")
        print(f"[{self.TAG}] stream floor (busiest link at {LINK_BW / 1e9:.1f} GB/s): attention {f_attn * 1e6:.1f} us, "
              f"second block {f_mlp * 1e6:.1f} us; mlp_prefetch after {f_after * 1e6:.1f} us, early "
              f"{f_early * 1e6:.1f} us")
        top = sorted(links, key=lambda ln: -tot[ln])[:6]
        for ln in top:
            parts = ", ".join(f"{ph} {load[ph].get(ln, 0):,}" for ph in load if load[ph].get(ln, 0))
            print(f"[{self.TAG}] link {ln[0]:02x}-{ln[1]:02x}: {tot[ln]:,} B ({parts})")
        print(f"[{self.TAG}] memory-chip links: " + ", ".join(f"{a:02x}-{b:02x} {tot[(a, b)]:,}" for a, b in memlinks))
        print(f"[{self.TAG}] memory-chip links, attention / second block: "
              + ", ".join(f"{a:02x}-{b:02x} {load['attn'].get((a, b), 0):,} / "
                          f"{load['free'].get((a, b), 0) + load['routed'].get((a, b), 0):,}" for a, b in memlinks))
        self.floors = dict(attn=f_attn, mlp=f_mlp, after=f_after, early=f_early, total=a_ring + mlp_total)

    def l1_report(self):
        """Per cluster, an estimate of the L1 the host's init asks for: the static plan's peak (static_l1_layout.csv)
        plus every device task's args struct (64-byte aligned) plus the scheduler's lists (12 B a task), against
        l1_capacity. Prints the fullest clusters."""
        import csv
        from bingo_args_phases import _args_bytes
        peak = {}
        with open(os.path.join(self.args.output_dir, "static_l1_layout.csv")) as f:
            for r in csv.DictReader(f):
                key = (int(r["chip"], 16), int(r["cluster"]))
                peak[key] = max(peak.get(key, 0), int(r["end"]))
        args, ntask = {}, {}
        for t in self.dfg.node_list:
            if (t.kernel_name or "").startswith("__snax"):
                key = (t.assigned_chiplet_id, t.assigned_cluster_id)
                args[key] = args.get(key, 0) + _args_bytes(t)
                ntask[key] = ntask.get(key, 0) + 1
        cap = self.dfg.l1_capacity_bytes
        tot = {key: peak.get(key, 0) + args.get(key, 0) + 12 * ntask.get(key, 0) for key in set(peak) | set(args)}
        top = sorted(tot, key=lambda key: -tot[key])[:4]
        print(f"[{self.TAG}] L1 estimate (static peak + args + lists of {cap:,} B): "
              + "; ".join(f"{k:02x}/{c} {peak.get((k, c), 0):,} + {args.get((k, c), 0):,} + "
                          f"{12 * ntask.get((k, c), 0):,} = {tot[(k, c)]:,}" for k, c in top))
        if tot[top[0]] > cap:
            over = sum(v > cap for v in tot.values())
            print(f"[{self.TAG}] WARNING: L1 estimate over l1_capacity on {over} clusters (params args phases needed)")

    # ---- emission ----------------------------------------------------------------------------------------------------
    def finish(self, app_name, x_prefixes=("x_wait_", "x_pull_", "x_copy_", "kx_ld_")):
        M = self.mlp
        self.check_collectives()
        if self.mlp_kind == "ffn" and M is not None:
            bad = M.check_hazards()
            if bad:
                raise ValueError(f"FFN: {len(bad)} buffer hazards, e.g. " + "; ".join(bad[:4]))
            print(f"[{self.TAG}] FFN buffer hazards: none")
        self.stream_report()
        self.dfg.split_chip_functions = True
        super().finish(app_name, x_prefixes)
        self.l1_report()


def _app_name(S):
    which = {"moe": "MLA + MoE", "ffn": "MLA + dense FFN"}[S.mlp_kind] if S.mlp else "MLA and glue only"
    if S.attn_kind == "ks":
        which += ", the attention split by key, every weight multiply on the GEMV" if S.mlp_gemv or not S.mlp else \
            ", the attention split by key"
    return (f"DeepSeek-V2-Lite layer decode ({which}), one token at position {S.attn.P}, {len(S.chips)} chips x {S.NC} "
            f"clusters, x and the cache from HBM, {'the output' if S.mlp else 'h and X'} into HBM")


def run():
    run_driver("dsv2 whole-layer decode workload generator", LayerBuild, _app_name)
