# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the staged DeepSeek-V2-Lite workloads (workloads/dsv2/<platform>/stage*/).

A STAGE is one workload directory whose main_bingo.py holds that stage's code and nothing
else, as two functions:

    build(S)    the blocks the stage adds (and, for a pass of several tokens, the checks it
                stashes as it goes)
    checks(S)   the values it checks, stashed after every stage's compute

Each stage builds on the one before it: its `build` first calls the previous stage's
(`load_stage`), and so does its `checks`. So the stage-N workload builds stages 1..N, in
the order one monolithic builder would -- every stage's compute, then every stage's checks
-- and the graph (node order, and with it every core's in-order task stream) is the one the
measured runs used.

What the stages share lives here: the arguments and params, the golden and its staging, the
weight streams and rings, the helpers a stage builds with (`S.gemv`, `S.deq`, `S.view`,
`S.check`, ...), the checks after all compute, and the emission. `S` is a `Build`: the
state one stage hands the next.

    python3 <stage dir>/main_bingo.py --output_dir <dir> --data_h <dir>/dsv2_data.h \\
        -c <stage dir>/params.hjson --hwcfg <snax_split_cluster.hjson> \\
        --platformcfg <occamy.h> --dsv2_dir <snax_cluster sw/apps/dsv2>
"""

import argparse
import importlib.util
import os
import sys

import hjson
import numpy as np


def _repo_root():
    """The HeMAiA checkout this file is in: found by what it holds, not by counting levels,
    so a workload may sit at any depth under workloads/."""
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, "target/sw/host/runtime/libbingo/mini_compiler")):
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"{__file__} is not inside a HeMAiA checkout")
        d = parent
    return d


ROOT_DIR = _repo_root()
sys.path.append(f"{ROOT_DIR}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # dsv2_datagen

import _bingo_paths  # noqa: F401,E402  (puts mini_compiler's grouped subdirs on sys.path)
from bingo_data_staging import DataStaging                             # noqa: E402
from bingo_dfg import BingoDFG                                         # noqa: E402
from bingo_kernel_args import HostBingoKernelWeightPrefetchArgs        # noqa: E402
from bingo_mem_handle import BingoMemSymbol                            # noqa: E402
from bingo_platform import (core_roles, guard_chiplet_count,           # noqa: E402
                            guard_cluster_count, load_cluster_cfg,
                            parse_platform_cfg)
from libs import (Ctx, DType, Layout, MemLevel, Pipeline, Port,        # noqa: E402
                  PortSpec, at_offset)
from libs.blocks import (Linear, LoadStream, ScaleCols, Stash, View,    # noqa: E402
                        WeightRings)
from libs.verify import checks as vchecks                              # noqa: E402
import dsv2_datagen as dg                                              # noqa: E402

CHIP = 0x00
L1_CAPACITY = 514816           # the per-cluster L1 heap
MESH = (16, 4, 16)
GEMV_SHAPE = (1, 4, 32)
NCL, HPC = dg.NCL, dg.HPC
D, KV, KVR, QH, RP = dg.D_MODEL, dg.KV, dg.KV_RANK, dg.Q_HEAD, dg.ROPE
QC = HPC * QH                  # 768: one cluster's W_Q columns, its four heads
ATT = 1                        # the attention's cluster (one token)
SL = D // NCL                  # 512: each cluster's slice of a model-width row
ISH_C = dg.I_SH // NCL         # 704: each cluster's shared-expert columns, spread
L1, L3, HBM = MemLevel.L1, MemLevel.L3, MemLevel.HBM
F16, I8 = DType.F16, DType.I8
RM, A, B, A_ROW = Layout.ROW_MAJOR, Layout.A, Layout.B, Layout.A_ROW
STAGES = {1: "norm", 2: "W_Q", 3: "W_DKV", 4: "latent + absorb", 5: "attention",
          6: "MLA output", 7: "routing", 8: "the layer"}


def row(n, cluster, level=L1, dtype=F16, rows=1, layout=RM):
    return PortSpec(layout, dtype, (rows, n), mem_level=level,
                    cluster=cluster if level == L1 else None)


def load_stage(caller, name):
    """The stage in `name`, a sibling of the calling stage's directory: its main_bingo.py,
    loaded once (as module dsv2_<name>). A stage builds on the one before it through this."""
    mod = f"dsv2_{name}"
    if mod not in sys.modules:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(caller))), name,
                            "main_bingo.py")
        spec = importlib.util.spec_from_file_location(mod, path)
        sys.modules[mod] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[mod])
    return sys.modules[mod]


class Build:
    """What one stage hands the next: the params, the staged data, the graph under
    construction, every block a later stage reads, and the helpers the stages build with."""

    def __init__(self, args, p, plat, hw, stop):
        self.args, self.p, self.plat, self.hw = args, p, plat, hw
        STOP = self.STOP = stop
        T = self.T = int(p.get("tokens", 1))
        # params from_stage: 8 builds stage 8 alone (a pass of tokens > 1): the router, the union
        # route, the experts and the combine, from the pass's golden h and ha
        FROM = self.FROM = int(p.get("from_stage", 1))
        if FROM not in (1, 8) or (FROM == 8 and (T == 1 or STOP != 8)):
            raise ValueError(f"params from_stage={FROM}: 1, or 8 with tokens > 1 and stage 8")
        if T > 1 and STOP > 4 and (T % 2 or (int(p.get("L", 511)) + T) % 64):
            raise ValueError(f"params tokens={T}: stages 5-8 run a pass of an even number of "
                             f"tokens whose L + tokens keys are whole tiles of 64 (L=508 for 4)")
        wbits = self.wbits = int(p.get("wbits", 4))
        if wbits not in (8, 4):
            raise ValueError(f"params wbits={wbits}: 8 or 4")
        wb = self.wb
        chunk = self.chunk = int(p.get("w_chunk_bytes", 128 * 1024))
        nc = int(p.get("norm_cluster", 0))
        self.tc = [nc] if T == 1 else [t % NCL for t in range(T)]   # token t's norm cluster
        dkv = self.dkv = [int(v) for v in p.get("dkv_cols", [576, 0, 0, 0])]
        if len(dkv) != NCL or sum(dkv) != KV or any(n < 0 or n % 64 for n in dkv):
            raise ValueError(f"params dkv_cols={dkv}: {NCL} counts of 64-column runs summing "
                             f"to {KV}")
        dkv0 = self.dkv0 = [sum(dkv[:c]) for c in range(NCL)]
        # stages 4+ normalise the latent and rotate k_pe on cluster 0: with W_DKV split, each
        # token's kv16 row is gathered there first (64-column rows from every cluster)
        self.fence_kv = bool(p.get("fence_kv", True))
        self.hold = bool(p.get("hold_weights", True))

        # ---- data -------------------------------------------------------------------------
        data = self.data = dg.generate(p, args.dsv2_dir)
        PS = self.PS = data["pass_"]           # the pass's golden (tokens > 1), stages 5-8
        U = self.U = len(PS["order"]) if PS else dg.TOP_K    # routed slots: the union, or top k
        self.ATTS = [int(v) for v in p.get("att_clusters", [1, 2])]   # a 2-token group's attention
        for line in data["report"]:
            print(f"[dsv2] {line}")
        st = self.st = DataStaging(plat, on_host=True)
        if not st.has_hbm:
            raise ValueError(f"{args.platformcfg} has no HBM: build with "
                             f"hemaia_twochiplet_16MBL3_4cluster.hjson")
        l4 = self.l4 = tuple(p.get("l4_weights", ()))
        h = self.h = dg.stage(st, data, l4=l4, skip=("wdkv", "wq"))
        self.ks, inv = data["ks"], data["inv"]
        self.inv = inv
        toks = self.toks = data["toks"]
        fp = dg.import_dsv2(args.dsv2_dir)[3]
        x8 = fp.quant_i8(data["gold"]["xn"], np.uint32(inv["x"]).view(np.float32))
        if not np.array_equal(toks[0]["x8"], x8):
            raise AssertionError("token 0 is not the golden token")
        self.h_xh = st.put_hbm("dsv2_xh",
                               np.concatenate([tk["x16"] for tk in toks]).view(np.uint16))
        a_row = np.zeros((T, 2 * D), dtype=np.int8)                # value c at (c//4)*8 + c%4
        for t, tk in enumerate(toks):
            a_row[t].reshape(-1, 8)[:, :4] = tk["x8"].reshape(-1, 4)
        self.h_x8 = st.put_zeros("dsv2_x8", "int8_t", T * 2 * D)
        self.h_gx8 = st.put("dsv2_gold_x8", "int8_t", a_row.reshape(-1))
        # W_DKV and W_Q as one image, cluster by cluster, each cluster's matrices in its stream
        # order (W_DKV first): a B-layout matrix is n-major, so a run of whole columns is one
        # contiguous byte range of it
        blob = {"kv": data["blobs"]["wdkv"], "q": data["blobs"]["wq"]}
        runs = [([("kv", dkv0[c], dkv[c])] if dkv[c] else []) + [("q", QC * c, QC)]
                for c in range(NCL)]
        parts, self.w_at_off = [], {}
        for c in range(NCL):
            for nm, c0, n in runs[c]:
                self.w_at_off[(nm, c)] = sum(len(x) for x in parts)
                parts.append(blob[nm][wb(c0 * D): wb((c0 + n) * D)])
        img = np.concatenate(parts)
        self.h_img = (st.put_l4 if "wq" in l4 else st.put_hbm)("dsv2_wimg", img)
        # SHARED EXPERT SPREAD (params shared_spread): cluster c runs gate|up for intermediate
        # columns [704 c, 704 c + 704) -- its gate and up slices back to back in one image, their
        # factors likewise -- and the down projection for output columns [512 c, 512 c + 512),
        # from the all-gathered SwiGLU operand. Every column stays one whole dot product on one
        # cluster.
        spread = self.spread = bool(p.get("shared_spread", False)) and STOP >= 8
        self.h_shgu = self.h_s_shgu = None
        if spread:
            gu_b, s_gu = data["blobs"]["sh_gu"], data["factors"]["s_sh_gu"]
            img_gu = np.concatenate([np.concatenate([gu_b[wb(a * D): wb((a + ISH_C) * D)]
                                                     for a in (ISH_C * c, dg.I_SH + ISH_C * c)])
                                     for c in range(NCL)])
            self.h_shgu = (st.put_l4 if "sh_gu" in l4 else st.put_hbm)("dsv2_shgu_img", img_gu)
            self.h_s_shgu = st.put("dsv2_s_shgu_split", "uint16_t", np.concatenate(
                [np.concatenate([s_gu[a: a + ISH_C] for a in (ISH_C * c, dg.I_SH + ISH_C * c)])
                 for c in range(NCL)]).view(np.uint16))
        # RoPE tables per token and row count: token 0's are the layer's own
        rope_h = self.rope_h = {}
        for t in range(T):
            for n in (HPC, HPC + 1):
                if t == 0:
                    rope_h[("cos", n, 0)], rope_h[("sin", n, 0)] = h[f"cos{n}"], h[f"sin{n}"]
                else:
                    rope_h[("cos", n, t)] = st.put(f"dsv2_cos{n}_t{t}", "uint16_t",
                                                   np.tile(toks[t]["cos"], n).view(np.uint16))
                    rope_h[("sin", n, t)] = st.put(f"dsv2_sin{n}_t{t}", "uint16_t",
                                                   np.tile(toks[t]["sin"], n).view(np.uint16))

        dfg = self.dfg = BingoDFG(num_chiplets=1,
                                  num_clusters_per_chiplet=plat["num_clusters_per_chiplet"],
                                  num_cores_per_cluster=plat["num_cores_per_cluster"],
                                  is_host_as_acc=True, chiplet_ids=[CHIP],
                                  dep_tag_width=plat["dep_tag_width"])
        dfg.l1_capacity_bytes = int(p.get("l1_capacity", L1_CAPACITY))   # a dry run may look past it
        dfg.waiting_queue_depth = plat.get("waiting_queue_depth", 8)     # the hang check's model
        # params prune_fanout: drop the cross-core edges a core's own order implies (fewer dummy
        # tasks and dependency tags). The compiler does it by default; false opts out.
        if "prune_fanout" in p:
            dfg.prune_fanout = bool(p["prune_fanout"])
        # params compact_tables: each cluster's L1 holds only its own tasks' arg and kernel
        # pointers and an int16 id table (bingo_hw_scheduler_init_compact) instead of the three
        # SoC-wide tables -- 64.8 -> ~25 KiB a cluster at the 4-token layer's 9,091 tasks
        dfg.compact_task_tables = bool(p.get("compact_tables", False))
        ctx = self.ctx = Ctx(dfg=dfg, mesh=MESH, roles=core_roles(plat), hw=hw, chiplet=CHIP)
        self.pipe = Pipeline(ctx, verbose=True, gate_sources=True)
        # params w_buffers: L1 slabs per cluster's weight stream, one count or one per cluster --
        # a third slab lets a cluster copy a chunk ahead of the GEMV that frees the slab before
        wbuf = p.get("w_buffers", 2)
        wbuf = [int(wbuf)] * NCL if isinstance(wbuf, (int, float, str)) else [int(v) for v in wbuf]
        ls = self.ls = [LoadStream(ctx, c, nbytes=chunk, nbuf=wbuf[c]) for c in range(NCL)]
        # params factor_ring: a routed expert's dequant factors ride its cluster's weight ring,
        # pushed right behind its weights, instead of being pulled across the link (SlotLoad)
        self.fr = ls if p.get("factor_ring", False) else [None] * NCL

        # ---- the weight ring: the memory chiplet pushes every chunk into a per-cluster ring
        # in L3, scheduled by the host (weight_prefetch); the cluster's DM core copies it on --
        self.rings, self.h_rec_l3 = None, None
        wr = p.get("weight_ring")
        if wr:
            n_slots = int(wr.get("slots", 20))
            seq = np.zeros((1024, 8), dtype=np.uint64)
            seq[:, 0] = np.arange(1024, dtype=np.uint64)
            put_seq = st.put_l4 if l4 else st.put_hbm
            h_seq = put_seq("dsv2_ring_seq", seq.view(np.int8).reshape(-1))
            h_flags = st.put_zeros("dsv2_ring_flags", "uint8_t", NCL * n_slots * 64)
            h_rel = st.put_zeros("dsv2_ring_release", "uint8_t", NCL * n_slots * 64)
            self.h_rec_l3 = st.put_zeros("dsv2_rec_l3", "uint8_t", U * 128)
            rings = self.rings = WeightRings(slots=ctx.l3("wring_slots", NCL * n_slots * chunk),
                                             flags=h_flags, releases=h_rel, n_rings=NCL,
                                             n_slots=n_slots, slot_bytes=chunk,
                                             batch=int(wr.get("batch", 1)))
            for c in range(NCL):
                ls[c].rings = rings
            memchip = (int(plat["mem_chip_loc_x"]) << 4) | int(plat["mem_chip_loc_y"])
            ctx.host_node("weight_prefetch", "__host_bingo_kernel_weight_prefetch",
                          HostBingoKernelWeightPrefetchArgs(
                              BingoMemSymbol("dsv2_ring_sched"), rings.slots, h_flags, h_rel,
                              h_seq, self.h_rec_l3, U, NCL, n_slots, chunk, memchip,
                              batch=int(wr.get("batch", 1)), policy=int(wr.get("policy", 0))))
        self.stash = []                   # (name, stash stage, golden handle, nbytes)
        self.CHECKS_FROM = int(p.get("checks_from", 1))   # stash and check stages >= this only
        # what the stages leave for the checks after all compute
        self.packs, self.x8p = [], None           # stage 1: the x8 packs
        self.out, self.outs = None, []            # stage 8: the layer output
        self.h_ha = self.g_ha = None              # stage 7, a pass: ha and its golden

    # ---- helpers the stages build with ------------------------------------------------------
    def wb(self, n, bits=None):
        """Bytes of n weights."""
        return n * (self.wbits if bits is None else bits) // 8

    @staticmethod
    def staged(spec, handle):
        """An array the datagen placed; nothing produced it."""
        return Port(spec, handle, ())

    @staticmethod
    def wspec(d_in, cols, bits, level):
        lay, dt = (Layout.B_W4, DType.I4) if bits == 4 else (B, I8)
        return PortSpec(lay, dt, (d_in, cols), mem_level=level)

    def gemv(self, name, c, d_in, d_out, k, x, w=None, w_off=0, groups=1, w_slot=None,
             rec=None, bits=None, tokens=1, x_layout=A, x_level=L1):
        """A GEMV through cluster c's stream: the weight at w + w_off (a weight count,
        converted to bytes at `bits`), or the address record `rec` names for slot w_slot."""
        bits = self.wbits if bits is None else bits
        blk = Linear(tokens=tokens, d_in=d_in, d_out=d_out, mesh=MESH, cluster=c, gemv=True,
                     d_shift=k, x_level=x_level, x_layout=x_layout, w_level=HBM,
                     groups=groups, stream=self.ls[c], w_slot=w_slot, w_bits=bits,
                     rec_slots=self.U)
        if w_slot is None:
            wp = self.staged(self.wspec(d_in, groups * d_out, bits, HBM),
                             at_offset(w, self.wb(w_off, bits)))
            return self.pipe.add(blk, name, x=x, w=wp)
        return self.pipe.add(blk, name, x=x, rec=rec)

    def deq(self, name, c, n, x, s, s_off=0, rows=1):
        """y (.) s, the factors from L3 (s a handle) or from a stage in this L1."""
        if hasattr(s, "out"):
            return self.pipe.add(ScaleCols(cols=n, cluster=c, s_level=L1, rows=rows), name,
                                 x=x, s=s.out())
        return self.pipe.add(ScaleCols(cols=n, cluster=c, s_level=L3, rows=rows), name, x=x,
                             s=self.staged(row(n, None, L3), at_offset(s, s_off)))

    def view(self, name, x, src, dst, offset=0):
        return self.pipe.add(View(src=src, dst=dst, offset=offset), name, x=x)

    def check(self, name, stage, port, golden, nbytes, offset=0, st=99):
        """Stash `stage`'s output (a port spec of it) now, compare it at the end."""
        if st < self.CHECKS_FROM:
            return
        sp = stage.out(port) if port else stage.out()
        s = self.pipe.add(Stash(src=_spec_of(stage, port), nbytes=nbytes, offset=offset),
                          f"stash_{name}", x=sp)
        self.stash.append((name, s, golden, nbytes))

    @staticmethod
    def a_row_bytes(v):
        """int8 values as a_row: value c at (c // 4) * 8 + c % 4, the rest 0."""
        v = np.asarray(v, dtype=np.int8).reshape(-1)
        o = np.zeros(2 * v.size, dtype=np.int8)
        o.reshape(-1, 8)[:, :4] = v.reshape(-1, 4)
        return o

    def gput(self, name, arrs):
        """A golden of a pass (tokens > 1), staged in L3."""
        a = np.concatenate([np.asarray(x).reshape(-1) for x in arrs])
        return self.st.put(f"dsv2_goldT_{name}",
                           "uint16_t" if a.dtype == np.float16 else "int8_t",
                           a.view(np.uint16) if a.dtype == np.float16 else a.view(np.int8))

    # ---- after every stage: the checks, the graph, the emission -----------------------------
    def finish(self):
        T, STOP, FROM, h, data, PS = self.T, self.STOP, self.FROM, self.h, self.data, self.PS
        dfg, ctx, p, st, args = self.dfg, self.ctx, self.p, self.st, self.args

        def all_checks():
            """After ALL compute, one at a time, upstream first, all byte-exact."""
            g0 = ctx.at(0)
            last = {}
            for _, s, _, _ in self.stash:
                nd = s.out().port.ends[-1]
                cl = nd.assigned_cluster_id
                if cl in last:
                    dfg.bingo_add_edge(last[cl], nd)
                last[cl] = nd
            prev = list(last.values()) + [p_.out().port.ends[-1] for p_ in self.packs]
            if self.out is not None:
                prev.append(self.out.out().port.ends[-1])
            prev += [o.out().port.ends[-1] for o in self.outs if o is not self.out]
            order = [("x8", self.h_x8, self.h_gx8, 2 * D * T)] if self.CHECKS_FROM <= 1 else []
            t_last = (data["keys"] // dg.BC - 1) * dg.BC * KV
            if T > 1 and STOP >= 5 and self.CHECKS_FROM <= 5:
                tl = (PS["keys"] // dg.BC - 1) * dg.BC * KV
                order.append(("key_tile", at_offset(h["key"], tl),
                              self.gput("key_tile", [PS["key1"][tl: tl + dg.BC * KV]]),
                              dg.BC * KV))
            if T > 1 and STOP >= 7 and FROM == 1:
                order.append(("ha", self.h_ha, self.g_ha, 2 * D * T))
            for n, s, gl, nb in self.stash:
                order.append((n, s.out().port.handle, gl, nb))
                if n == "kpe8" and STOP >= 5:
                    order.append(("key_tile", at_offset(h["key"], t_last),
                                  at_offset(h["gold_key_t"], t_last), dg.BC * KV))
            for name, got, golden, nbytes in order:
                prev = [vchecks.check_bytes(g0, f"Check_dsv2_{name}", golden=golden, got=got,
                                            nbytes=nbytes, after=prev, label=f"dsv2_{name}")]
        self.pipe.raw(all_checks, "checks")
        self.pipe.run()

        if self.rings is not None:
            st.put("dsv2_ring_sched", "uint64_t", np.array(self.rings.schedule(), dtype=np.uint64))
            print(f"[dsv2] weight ring: {self.rings.chunks()} chunks pushed by the memchip "
                  f"({[len(e) for e in self.rings.entries]} per cluster), {self.rings.n_slots} "
                  f"slots of {self.rings.slot_bytes // 1024} KiB per cluster")

        if args.data_h:
            st.emit(args.data_h, args.output_dir)
            if p.get("d2d_ddr", False):
                with open(args.data_h, "a") as f:
                    f.write("\n// params d2d_ddr: every D2D link switched to DDR at boot\n"
                            "#define HEMAIA_D2D_DDR 1\n")
        extra = [os.path.basename(str(args.data_h))] if args.data_h else None
        dfg.bingo_compile_dfg(
            app_name=(f"DeepSeek-V2-Lite layer 1 to stage {STOP} ({STAGES[STOP]}), {T} token(s), "
                      f"INT{self.wbits}, {NCL} clusters"),
            output_dir=args.output_dir,
            output_file_name=args.output_offload_file_name,
            extra_include_header_list=extra,
            static_l1=bool(p.get("static_l1", True)),
            # params desc_in_l3: the task-descriptor list in L3 instead of the narrow SPM's L2
            # heap (123 KiB: ~7,800 descriptors of 16 B; a 4-token pass has 9,100)
            desc_list_in_narrow_spm=not bool(p.get("desc_in_l3", False)))
        print(f"[dsv2] stage {STOP} ({STAGES[STOP]}), {T} token(s), INT{self.wbits}; L4 "
              f"{st.l4_bytes / 2**20:.2f} MiB {list(self.l4)}; HBM {st.hbm_bytes / 2**20:.2f} MiB; "
              f"{sum(1 for _ in dfg.nodes)} nodes; D2D {'DDR' if p.get('d2d_ddr') else 'SDR'}")
        print(f"Generated: {os.path.join(args.output_dir, args.output_offload_file_name)}")


def run(stage):
    """Build stages 1..stage.STAGE, check them, and emit the workload's headers."""
    ap = argparse.ArgumentParser(description=(stage.__doc__ or "").strip().split("\n")[0])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_h", default=None)
    ap.add_argument("--output_offload_file_name", default="offload_bingo_hw.h")
    ap.add_argument("-c", "--cfg", required=True)
    ap.add_argument("--hwcfg", required=True)
    ap.add_argument("--platformcfg", required=True)
    ap.add_argument("--dsv2_dir", required=True,
                    help="snax_cluster's target/snitch_cluster/sw/apps/dsv2 (the golden)")
    args = ap.parse_args()

    with open(args.cfg) as f:
        p = dict(hjson.load(f))
    # the stage is the directory's, not a param; a params file written for the monolithic
    # builder may still name it, and then it must agree
    if int(p.get("stop", stage.STAGE)) != stage.STAGE:
        raise ValueError(f"{args.cfg}: stop={p['stop']}, but this is stage {stage.STAGE}")
    plat = parse_platform_cfg(args.platformcfg)
    if not (guard_chiplet_count(p, plat, args.output_dir, args.output_offload_file_name)
            and guard_cluster_count(p, plat, args.output_dir,
                                    args.output_offload_file_name)):
        return
    hw = load_cluster_cfg(args.hwcfg)
    shapes = [tuple(int(v) for v in s) for s in hw["snax_versacore_core_template"]
              ["snax_acc_cfg"][0]["snax_versacore_spatial_unrolling"][0]]
    if shapes[0] != MESH or GEMV_SHAPE not in shapes:
        raise ValueError(f"{args.hwcfg} declares array shapes {shapes}; this layer needs "
                         f"{MESH} and the one-token GEMV's {GEMV_SHAPE}")
    S = Build(args, p, plat, hw, stage.STAGE)
    stage.build(S)
    stage.checks(S)
    S.finish()


def dg_pass_records(PS, h):
    """The pass's records (moe_route.h, tokens > 1) as the device writes them, from the
    expert table staged in L3."""
    from libs.blocks import pass_record_bytes
    table = np.asarray(h["table_bytes"], dtype=np.int8)
    rec, order = pass_record_bytes([tk["ids"] for tk in PS["toks"]],
                                   [np.asarray(tk["w16"], dtype=np.float16).view(np.uint16)
                                    for tk in PS["toks"]], table)
    if list(order) != list(PS["order"]):
        raise AssertionError("the records' union order is not the golden's")
    return rec, order


def _spec_of(stage, port):
    """The spec of a stage's output, before it is built: its chosen block's, else the
    template's (a block with one realisation)."""
    blk = stage.template
    outs = blk.outputs
    return outs[port] if port else next(iter(outs.values()))
