# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The shared driver of the moe workloads (workloads/moe/<model>/<variant>/): DeepSeek-V2-Lite's
layer-1 MoE for one token at its real size, on the 16 + 4 chiplet platform, with a SHARDED
router (the platform, the rings, the anchors, the sync arm, the broadcast and the reduce are
ffn/common/chiplet_mc.py's; the data and the golden are moe_data.py's).

THE MAPPING. Each of the 6 routed slots runs on the 4 clusters of 2 chips, the shared expert
(2816) on the 8 clusters of 4 chips (the x = 0 column, the root among them): every cluster
computes 352 intermediate columns of ONE expert -- gate|up by columns, down by rows -- so its
down GEMV gives a partial over every output, which it dequantises (the expert's per-column
factors) and, for a routed slot, weights. Then the FFN's chip fold and reduce add the 32
partials into the root. Which expert a slot runs is decided at run time: the slot's weights,
factors and SwiGLU scale come from the chip's route RECORD (moe_route.h), whose entries name
the chip's own memory chip's copies (each chip has its own expert table), and its prefetcher
pushes them only once the record is in its L3.

THE ROUTER, SHARDED. Every chip computes the logits of its 4 experts (an INT8 GEMV over 32
columns, 28 of them zero: a SIMD beat) on cluster 0. The route needs them all:

    route fold    each chip's softmax partial over its 4 logits (the SIMD row kernel's max and
                  sum) and its 4 (logit, id) pairs go into a softmax-statistics group and a
                  TopK beat (moe_router_pack); two ChainGathers fold them across the chips on
                  the root, the TopK on cluster 0's xDMA, the statistics on cluster 1's, in
                  parallel: the 8 largest logits with their ids, and (m*, l*). The root
                  applies them to the 8 candidates (simd_softmax_row_ext) -- the ROUTE: 8
                  probabilities and ids.
    route gather  the root pulls every chip's 4 logits into one row of 64 and runs the softmax
                  over it (simd_softmax_row) -- the route: 64 probabilities.

The route is broadcast to every chip's cluster 0 (bcast_l1 or pull, the token's mechanism),
where moe_route picks the top 6 against the chip's own table; the record goes to the chip's L3
(its prefetcher) and to cluster 1.

params route: fold or gather -- the collectives' switch, with bcast and reduce.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))     # workloads/common
from driver import repo_root, run_driver  # noqa: E402
REPO = repo_root(__file__)
sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, os.path.join(HERE, "..", "..", "ffn", "common"))
sys.path.insert(0, HERE)

import _bingo_paths  # noqa: F401,E402
from bingo_kernel_args import (  # noqa: E402
    SnaxBingoKernelIdma1dCopyArgs, SnaxBingoKernelMoeRouteArgs, SnaxBingoKernelMoeRouterPackArgs,
    SnaxBingoKernelSimdSoftmaxRowArgs, SnaxBingoKernelSimdSoftmaxRowExtArgs,
    SnaxBingoKernelXdmaChainGatherArgs, XDMA_SSJ_GROUP_BYTES, xdma_softmax_stats_csrs)
from bingo_mem_handle import BingoMemSymbol  # noqa: E402
from libs import DType, Layout, Port, PortSpec, at_offset  # noqa: E402
from libs.blocks import (AddRow, Linear, ScaleCols, ScaleRowBySlot, SlotLoad,  # noqa: E402
                         SwigluARow, expert_table, record_bytes, record_spec)

import chiplet_mc as cm  # noqa: E402
import moe_data  # noqa: E402

L1, L3, HBM = cm.L1, cm.L3, cm.HBM
F16, I8 = DType.F16, DType.I8
RM, A_ROW = Layout.ROW_MAJOR, Layout.A_ROW
TOP_K, N_EXP, CAND = moe_data.TOP_K, moe_data.N_EXP, moe_data.CAND
PER_CHIP = 4                         # router columns per chip
RC = moe_data.ROUTER_COLS            # ... padded to a SIMD beat
TOPK_CSR = 8                         # TopK junction: keep 8, key by max
INTER = 352                          # every cluster's slice of its expert


class MBuild(cm.ChipletBuild):
    TAG = "moe"

    def __init__(self, args, p, plat, hw):
        super().__init__(args, p, plat, hw)
        if self.T != 1:
            raise ValueError("params tokens: the moe workloads run one token")
        self.route_impl = str(p.get("route", "fold"))
        if self.route_impl not in ("fold", "gather"):
            raise ValueError(f"params route={self.route_impl!r}: fold or gather")
        cache = os.path.join(REPO, p.get("weights", "target/sim/automation/sweep/common/weights/cache"))
        d = self.d = moe_data.MoeData(cache, args.dsv2_dir, seed=int(p.get("seed", 1)),
                                      wbits=self.wbits)
        self.D = d.H
        # ---- roles: the shared expert on the x = 0 column (the root's), the slots on pairs -
        chips = self.chips
        self.shared_chips = sorted(k for k in chips if k >> 4 == min(c >> 4 for c in chips))
        rest = [k for k in chips if k not in self.shared_chips]
        # a slot's two chips: the same column, rows 2 j' and 2 j' + 1
        rest.sort(key=lambda k: (k & 15) // 2 * 100 + (k >> 4) * 4 + (k & 15) % 2)
        self.slot_chips = [rest[2 * s: 2 * s + 2] for s in range(TOP_K)]
        if len(self.shared_chips) * self.NC * INTER != d.I_sh or \
                len(rest) != 2 * TOP_K or 2 * self.NC * INTER != d.I_e:
            raise ValueError("the mapping needs 4 shared chips and 12 slot chips of 2 clusters")
        self.role = {}                  # g -> ("sh" | slot s, its c0)
        for i, k in enumerate(self.shared_chips):
            for j, g in enumerate(self.gs(k)):
                self.role[g] = ("sh", (i * self.NC + j) * INTER)
        for s, pair in enumerate(self.slot_chips):
            for i, k in enumerate(pair):
                for j, g in enumerate(self.gs(k)):
                    self.role[g] = (s, (i * self.NC + j) * INTER)
        self.router_ids = {k: list(range(PER_CHIP * i, PER_CHIP * (i + 1)))
                           for i, k in enumerate(chips)}
        # ---- the route each arm computes; the experts staged are its slots' --------------
        ids_g, w_g, _ = d.route_gather()
        ids_f, w_f, *_ = d.route_fold(self.router_ids, self.chain_order())
        if ids_g != ids_f:
            raise ValueError(f"the two routes pick different experts ({ids_g} vs {ids_f}): a "
                             f"near tie; this token cannot compare the arms")
        self.ids, self.w16 = (ids_f, w_f) if self.route_impl == "fold" else (ids_g, w_g)
        for line in d.report(ids_g, ids_f, w_g, w_f):
            print(f"[moe] {line}")
        print(f"[moe] shared chips {[f'{k:02x}' for k in self.shared_chips]}; slots "
              f"{[[f'{k:02x}' for k in pr] for pr in self.slot_chips]}; fold order "
              f"{[f'{k:02x}' for k in self.order]}")
        self.gold = self._golden()

        self._stage()
        self.make_streams()
        self.make_rings(self.cimg, routed=self.cex, slices=self.cslices, rec_l3=self.h_rec_l3,
                        rec_slots=TOP_K)
        # ---- the symmetric buffer of every cluster ---------------------------------------
        xb = 2 * self.D
        self.OFF_X, self.OFF_LOC, self.OFF_FLAG = 0, 0, xb
        self.OFF_RFLAG = xb + 64
        self.OFF_ROWL, self.OFF_ROW = xb + 128, xb + 192     # the router row and its latch
        self.OFF_TOPK, self.OFF_STATS = xb + 256, xb + 320   # this chip's TopK beat / stats group
        self.OFF_ROUTE = self.OFF_STATS + XDMA_SSJ_GROUP_BYTES  # the route: p (128 B), ids
        self.make_sym(self.OFF_ROUTE + 256, xb)

    # ---- data -----------------------------------------------------------------------------
    def _golden(self):
        d, fp = self.d, self.d.fp
        parts = {}
        for g, (u, c0) in self.role.items():
            mp = d.sh_pack if u == "sh" else d.expert(self.ids[u])
            gr, _ = d.hm.gemv(d.hq, np.concatenate([mp["gu_q"][:, c0:c0 + INTER],
                                                    mp["gu_q"][:, mp["I"] + c0:mp["I"] + c0 + INTER]],
                                                   axis=1), d.k_x)
            g16 = fp.mul16(gr, d.gu_factors(mp, c0, INTER))
            a8 = fp.quant_i8(fp.mul16(d.hm.silu16(g16[:INTER]), g16[INTER:]), mp["s_a"].inv)
            y, _ = d.hm.gemv(a8, mp["dn_q"][c0:c0 + INTER, :], mp["k_dn"])
            y = fp.mul16(y, mp["dn_f"])
            parts[g] = y if u == "sh" else fp.mul16(y, np.float16(self.w16[u]))
        loc = {k: fp.add16(parts[self.gs(k)[0]], parts[self.gs(k)[1]]) for k in self.chips}
        fo = self.fold_order()
        y = loc[fo[0]]
        for k in fo[1:]:
            y = fp.add16(y, loc[k])
        # every expert's down GEMV uses one shift: the routed experts share their depth
        if len({d.expert(e)["k_dn"] for e in self.ids}) != 1:
            raise ValueError("the routed experts' down shifts differ")
        return dict(parts=parts, loc=loc, y=y)

    def _stage(self):
        st, d, T = self.st, self.d, self.TAG
        self.h_x8 = st.put(f"{T}_x8", "int8_t", d.x8())
        self.h_rec_l3 = st.put_zeros(f"{T}_rec_l3", "uint8_t", TOP_K * 128)
        self.wh, self.h_sgu, self.cimg = {}, {}, {}
        self.h_sdn = st.put(f"{T}_s_dn_sh", "uint16_t",
                            np.asarray(d.sh_pack["dn_f"], np.float16).view(np.uint16))
        # the router: each chip its own 32-column block and factors
        for k in self.chips:
            st.put(f"{T}_wr_k{k:02x}", "int8_t", d.router_blob(self.router_ids[k]))
            st.put(f"{T}_sr_k{k:02x}", "uint16_t",
                   d.router_factors(self.router_ids[k]).view(np.uint16))
        # the shared expert: a dense image per cluster, as the FFN's
        for g, (u, c0) in self.role.items():
            if u != "sh":
                continue
            k = self.chip(g)
            gu, dn = d.gu_blob(d.sh_pack, c0, INTER), d.dn_blob(d.sh_pack, c0, INTER)
            img = np.concatenate([gu, dn])
            loc = (self.mem[k] >> 4, self.mem[k] & 15)
            hd = st.put_hbm(f"{T}_wimg_g{g}", img, mem_chip=loc)
            self.wh[(g, "gu")], self.wh[(g, "dn")] = hd, at_offset(hd, gu.size)
            if self.crest:
                cap = img.nbytes + 64 * 1024
                self.cimg[g] = (hd.address, img.view(np.uint8).reshape(-1),
                                st.put_hbm(f"{T}_cimg_g{g}", np.zeros(cap, np.uint8), mem_chip=loc),
                                cap)
            self.h_sgu[g] = st.put(f"{T}_s_gu_g{g}", "uint16_t",
                                   np.asarray(d.gu_factors(d.sh_pack, c0, INTER),
                                              np.float16).view(np.uint16))
        # the routed slots: per slot chip, its clusters' slices of the slot's expert, back to
        # back per field, in its memory chip's HBM; its expert table names them
        self.tables, self.table_bytes, self.slot_off = {}, {}, {}
        self.cex, self.cslices = {}, {}
        for s, pair in enumerate(self.slot_chips):
            e = self.ids[s]
            mp = d.expert(e)
            for k in pair:
                gs = self.gs(k)
                loc = (self.mem[k] >> 4, self.mem[k] & 15)
                gu = [d.gu_blob(mp, self.role[g][1], INTER) for g in gs]
                gus = [np.asarray(d.gu_factors(mp, self.role[g][1], INTER), np.float16) for g in gs]
                dn = [d.dn_blob(mp, self.role[g][1], INTER) for g in gs]
                f = {"gu": np.concatenate(gu), "dn": np.concatenate(dn)}
                if self.crest:
                    hs = {fn: st.put_hbm(f"{T}_e{e}_{fn}_k{k:02x}", np.zeros(b.nbytes, np.uint8),
                                         mem_chip=loc) for fn, b in f.items()}
                    self.cex[k] = {fn: ({e: np.ascontiguousarray(b).view(np.uint8).reshape(-1)},
                                        {e: hs[fn]}) for fn, b in f.items()}
                else:
                    hs = {fn: st.put_hbm(f"{T}_e{e}_{fn}_k{k:02x}", b, mem_chip=loc)
                          for fn, b in f.items()}
                h_gus = st.put_hbm(f"{T}_e{e}_gus_k{k:02x}",
                                   np.concatenate(gus).view(np.int8), mem_chip=loc)
                h_dns = st.put_hbm(f"{T}_e{e}_dns_k{k:02x}",
                                   np.asarray(mp["dn_f"], np.float16).view(np.int8), mem_chip=loc)
                entry = (hs["gu"].address, h_gus.address, hs["dn"].address, h_dns.address,
                         mp["inv_a"])
                # every picked expert's entry is non-zero (moe_route refuses a zero one); this
                # chip streams only its slot's, so the others name the same copies
                table = expert_table(N_EXP, {x: entry for x in self.ids})
                self.tables[k] = st.put(f"{T}_table_k{k:02x}", "int8_t", table)
                self.table_bytes[k] = table
                for r, g in enumerate(gs):
                    self.slot_off[g] = {"gu": sum(b.size for b in gu[:r]),
                                        "gu_s": 2 * 2 * INTER * r, "dn": sum(b.size for b in dn[:r]),
                                        "dn_s": 0}
                self.cslices[k] = {r: {fn: (sum(b.size for b in parts[:r]),
                                            sum(b.size for b in parts[:r + 1]))
                                       for fn, parts in (("gu", gu), ("dn", dn))}
                                   for r in range(len(gs))}

    # ---- the layer ------------------------------------------------------------------------
    def build(self):
        dm, xdma, simd = self.roles["dm"], self.roles["xdma"], self.roles["simd"]
        D, R, T = self.D, self.ROOT, self.TAG
        xb = 2 * D
        go = self.setup()
        xin = self.broadcast("x", self.h_x8, xb, self.OFF_X, self.OFF_FLAG, go)
        # ---- the router: every chip's 4 logits on its cluster 0 --------------------------
        self.row_ready, self.pack = {}, {}
        for k in self.chips:
            g0 = self.gs(k)[0]
            x = Port(PortSpec(A_ROW, I8, (1, D), mem_level=L1, cluster=0),
                     self.sv(k, 0, self.OFF_X), (xin[(k, 0)],))
            lg = self.pipe.add(
                Linear(tokens=1, d_in=D, d_out=RC, mesh=cm.MESH, cluster=0, gemv=True,
                       d_shift=self.d.k_x, x_level=L1, x_layout=A_ROW, w_level=L3, w_bits=8),
                f"router_k{k:02x}", chiplet=k, x=x,
                w=Port(PortSpec(Layout.B, I8, (D, RC), mem_level=L3),
                       BingoMemSymbol(f"{T}_wr_k{k:02x}"), ()))
            lg16 = self.pipe.add(ScaleCols(cols=RC, cluster=0, s_level=L3), f"lg16_k{k:02x}",
                                 chiplet=k, x=lg.out(),
                                 s=Port(PortSpec(RM, F16, (1, RC), mem_level=L3),
                                        BingoMemSymbol(f"{T}_sr_k{k:02x}"), ()))
            self.pipe.raw(lambda k=k, lg16=lg16: self._router_tail(k, lg16), f"router_tail_k{k:02x}")
        self.pipe.raw(self._route, "route")
        self.pipe.raw(self._route_first, "route_first")
        # ---- the experts: per cluster a partial over every output ------------------------
        self.part = {}
        for g in range(self.G):
            k, c = self.chip(g), self.c(g)
            u, c0 = self.role[g]
            x = Port(PortSpec(A_ROW, I8, (1, D), mem_level=L1, cluster=c),
                     self.sv(k, c, self.OFF_X), (xin[(k, c)],))
            self.ls[g].hold_until(xin[(k, c)])
            if u == "sh":
                self.part[g] = self._shared(g, k, c, x)
            else:
                self.part[g] = self._slot(g, k, c, u, x)
        self.pipe.raw(lambda: self.reduce({g: s.out().port for g, s in self.part.items()}, xb),
                      "reduce")
        self.pipe.raw(self._finish_graph, "finish")
        self.pipe.run()

    def _shared(self, g, k, c, x):
        d, D = self.d, self.D
        mp = d.sh_pack
        gu = self.pipe.add(
            Linear(tokens=1, d_in=D, d_out=2 * INTER, mesh=cm.MESH, cluster=c, gemv=True,
                   d_shift=d.k_x, x_level=L1, x_layout=A_ROW, w_level=HBM, stream=self.ls[g],
                   w_bits=self.wbits),
            f"shgu_g{g}", chiplet=k, x=x,
            w=Port(PortSpec(Layout.B_W4, DType.I4, (D, 2 * INTER), mem_level=HBM),
                   self.wh[(g, "gu")], ()))
        g16 = self.pipe.add(ScaleCols(cols=2 * INTER, cluster=c, s_level=L3), f"shg16_g{g}",
                            chiplet=k, x=gu.out(),
                            s=Port(PortSpec(RM, F16, (1, 2 * INTER), mem_level=L3),
                                   self.h_sgu[g], ()))
        a8 = self.pipe.add(SwigluARow(inter=INTER, cluster=c, inv_scale_f32bits=mp["inv_a"],
                                      a_row=True), f"sha8_g{g}", chiplet=k, g=g16.out())
        dn = self.pipe.add(
            Linear(tokens=1, d_in=INTER, d_out=D, mesh=cm.MESH, cluster=c, gemv=True,
                   d_shift=mp["k_dn"], x_level=L1, x_layout=A_ROW, w_level=HBM,
                   stream=self.ls[g], w_bits=self.wbits),
            f"shdn_g{g}", chiplet=k, x=a8.out(),
            w=Port(PortSpec(Layout.B_W4, DType.I4, (INTER, D), mem_level=HBM),
                   self.wh[(g, "dn")], ()))
        return self.pipe.add(ScaleCols(cols=D, cluster=c, s_level=L3), f"shy_g{g}", chiplet=k,
                             x=dn.out(), s=Port(PortSpec(RM, F16, (1, D), mem_level=L3),
                                                self.h_sdn, ()))

    def _slot(self, g, k, c, s, x):
        d, D = self.d, self.D
        so = self.slot_off[g]
        rec = self.pipe.source(f"rec_g{g}", record_spec(TOP_K, c),
                               lambda g=g: self.rec_port[g])
        gu = self.pipe.add(
            Linear(tokens=1, d_in=D, d_out=2 * INTER, mesh=cm.MESH, cluster=c, gemv=True,
                   d_shift=d.k_x, x_level=L1, x_layout=A_ROW, w_level=HBM, stream=self.ls[g],
                   w_slot=(s, "gu"), w_bits=self.wbits, rec_slots=TOP_K, w_slot_off=so["gu"]),
            f"gu_s{s}_g{g}", chiplet=k, x=x, rec=rec.out())
        gus = self.pipe.add(SlotLoad(slot=s, field="gu_s", cols=2 * INTER, cluster=c,
                                     offset=so["gu_s"]), f"gus_s{s}_g{g}", chiplet=k,
                            rec=rec.out())
        g16 = self.pipe.add(ScaleCols(cols=2 * INTER, cluster=c, s_level=L1), f"g16_s{s}_g{g}",
                            chiplet=k, x=gu.out(), s=gus.out())
        a8 = self.pipe.add(SwigluARow(inter=INTER, cluster=c, slot=s, a_row=True),
                           f"a8_s{s}_g{g}", chiplet=k, g=g16.out(), rec=rec.out())
        dn = self.pipe.add(
            Linear(tokens=1, d_in=INTER, d_out=D, mesh=cm.MESH, cluster=c, gemv=True,
                   d_shift=d.expert(self.ids[s])["k_dn"], x_level=L1, x_layout=A_ROW,
                   w_level=HBM, stream=self.ls[g], w_slot=(s, "dn"), w_bits=self.wbits,
                   rec_slots=TOP_K, w_slot_off=so["dn"]),
            f"dn_s{s}_g{g}", chiplet=k, x=a8.out(), rec=rec.out())
        dns = self.pipe.add(SlotLoad(slot=s, field="dn_s", cols=D, cluster=c, offset=0),
                            f"dns_s{s}_g{g}", chiplet=k, rec=rec.out())
        y = self.pipe.add(ScaleCols(cols=D, cluster=c, s_level=L1), f"y_s{s}_g{g}", chiplet=k,
                          x=dn.out(), s=dns.out())
        return self.pipe.add(ScaleRowBySlot(cols=D, cluster=c, slot=s), f"yw_s{s}_g{g}",
                             chiplet=k, x=y.out(), rec=rec.out())

    # ---- the route ------------------------------------------------------------------------
    def _scratch(self, k, c, name, nbytes):
        return self.cx_chip(k).at(c).l1(f"{self.TAG}_{name}", nbytes)

    def _router_tail(self, k, lg16):
        """The chip's row of 4 logits in its symmetric buffer (both arms) and, route fold, its
        softmax partial, its TopK beat and its softmax-statistics group."""
        dm, simd = self.roles["dm"], self.roles["simd"]
        pt = lg16.out().port
        row = self.node(k, 0, dm, "rt_row", "__snax_bingo_kernel_moe_router_pack",
                        SnaxBingoKernelMoeRouterPackArgs(
                            0, n=PER_CHIP, logits_addr=pt.handle,
                            row_addr=self.sv(k, 0, self.OFF_ROW)), list(pt.ends))
        self.row_ready[k] = row
        if self.route_impl != "fold":
            return
        tmp = self._scratch(k, 0, "rt_tmp", 64)
        e = self._scratch(k, 0, "rt_e", 3 * 64)          # latch, e, s
        p = self._scratch(k, 0, "rt_p", 64)
        sm = self.node(k, 0, simd, "rt_softmax", "__snax_bingo_kernel_simd_softmax_row",
                       SnaxBingoKernelSimdSoftmaxRowArgs(self.sv(k, 0, self.OFF_ROW), tmp,
                                                         e.view(64), p, 1), [row])
        self.pack[k] = self.node(k, 0, dm, "rt_pack", "__snax_bingo_kernel_moe_router_pack",
                                 SnaxBingoKernelMoeRouterPackArgs(
                                     1, n=PER_CHIP, id_base=self.router_ids[k][0],
                                     logits_addr=self.sv(k, 0, self.OFF_ROW), m_addr=tmp,
                                     s_addr=e.view(128), topk_addr=self.sv(k, 0, self.OFF_TOPK),
                                     stats_addr=self.sv(k, 1, self.OFF_STATS)), [sm])

    def _route(self):
        """The route on the root, its broadcast, and every slot chip's record."""
        dm, xdma, simd = self.roles["dm"], self.roles["xdma"], self.roles["simd"]
        R, others = self.ROOT, self.order[1:]
        route = self.sv(R, 0, self.OFF_ROUTE)
        # what the root waits for: every chip's beats (fold) or row (gather) -- by edges, or by
        # one barrier every chip arrives at once its own is ready
        ready = self.pack if self.route_impl == "fold" else self.row_ready
        before = dict(ready)
        if self.sync == "barrier":
            b = self.barrier({k: [ready[k]] for k in self.chips})
            before = {k: b[R] for k in self.chips}
        if self.route_impl == "fold":
            deps = list(dict.fromkeys(before[k] for k in self.chips))
            tk_dst = self._scratch(R, 0, "rt_topk", 64)
            st_dst = self._scratch(R, 1, "rt_stats", XDMA_SSJ_GROUP_BYTES)
            tk = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, 0, others, src=self.sx(0, self.OFF_TOPK), dst=tk_dst.get_c_var_name(),
                size=64, junction=self.jct["TopK"], jct_csr0=TOPK_CSR, holds=[tk_dst])
            # one group, row 0 live; the logits are unscaled, so the key factor is 1.0
            st_csr0, st_csr1 = xdma_softmax_stats_csrs(1)
            st = SnaxBingoKernelXdmaChainGatherArgs.fold_into(
                cm.remote, R, 1, others, src=self.sx(1, self.OFF_STATS), dst=st_dst.get_c_var_name(),
                size=XDMA_SSJ_GROUP_BYTES, junction=self.jct["SoftmaxStats"],
                jct_csr0=st_csr0, jct_csr1=st_csr1, holds=[st_dst])
            g_tk = self.node(R, 0, xdma, "rt_topk_chain", "__snax_bingo_kernel_xdma_chain_gather",
                             tk, deps)
            g_st = self.node(R, 1, xdma, "rt_stats_chain", "__snax_bingo_kernel_xdma_chain_gather",
                             st, deps)
            cand = self._scratch(R, 0, "rt_cand", 2 * 64)      # latch, the candidates' row
            s_b = self._scratch(R, 0, "rt_s", 64)
            e = self._scratch(R, 0, "rt_ce", 2 * 64)           # latch, e
            un = self.node(R, 0, dm, "rt_unpack", "__snax_bingo_kernel_moe_router_pack",
                           SnaxBingoKernelMoeRouterPackArgs(
                               2, topk_addr=tk_dst, stats_addr=st_dst, row_addr=cand.view(64),
                               s_addr=s_b, ids_addr=self.sv(R, 0, self.OFF_ROUTE + 128)),
                           [g_tk, g_st])
            made = self.node(R, 0, simd, "rt_softmax_ext", "__snax_bingo_kernel_simd_softmax_row_ext",
                             SnaxBingoKernelSimdSoftmaxRowExtArgs(cand.view(64), s_b, e.view(64),
                                                                  route, 1), [un])
            n_cand, with_ids = CAND, True
        else:
            row = self._scratch(R, 0, "rt_row64", 3 * 64)     # latch, 2 beats of 64 logits
            pulls = []
            for i, k in enumerate(self.chips):
                src = self.sv(k, 0, self.OFF_ROW) if k == R else \
                    cm.remote(k, self.sx(0, self.OFF_ROW))
                dep = [before[k]]
                pulls.append(self.node(R, 0, dm, f"rt_pull{i}", "__snax_bingo_kernel_idma_1d_copy",
                                       SnaxBingoKernelIdma1dCopyArgs(
                                           src, row.view(64 + 2 * PER_CHIP * i), 2 * PER_CHIP),
                                       dep + pulls[-1:]))
            tmp = self._scratch(R, 0, "rt_tmp64", 64)
            e = self._scratch(R, 0, "rt_e64", 4 * 64)         # latch, e (2), s
            made = self.node(R, 0, simd, "rt_softmax64", "__snax_bingo_kernel_simd_softmax_row",
                             SnaxBingoKernelSimdSoftmaxRowArgs(row.view(64), tmp, e.view(64),
                                                               route, 2), [pulls[-1]])
            n_cand, with_ids = N_EXP, False
        self.route_made = made
        # ---- the route to every chip's cluster 0, then each slot chip's record ------------
        arrive = self.broadcast("route", route, 256, self.OFF_ROUTE, self.OFF_RFLAG,
                                {k: (made if k == R else ready[k]) for k in self.chips},
                                clusters=[0], src_ready=made)
        # what each shared chip's stream waits for: its last task of the route -- the
        # barrier before the route's pulls, else the one before the root's step, else its
        # router -- and on the root the route's arrival
        last = dict(self.bcast_barrier) or (b if self.sync == "barrier" else ready)
        self.route_gate = {k: (arrive[(R, 0)] if k == R else last[k]) for k in self.chips}
        self.rec_port, self.records = {}, {}
        for pair in self.slot_chips:
            for k in pair:
                g0, g1 = self.gs(k)
                rec0 = self._scratch(k, 0, "rec", TOP_K * 128)
                rt = self.node(k, 0, dm, "route", "__snax_bingo_kernel_moe_route",
                               SnaxBingoKernelMoeRouteArgs(
                                   self.sv(k, 0, self.OFF_ROUTE), n_cand, TOP_K,
                                   BingoMemSymbol(f"{self.TAG}_table_k{k:02x}"), rec0,
                                   ids_addr=(self.sv(k, 0, self.OFF_ROUTE + 128)
                                             if with_ids else 0)),
                               [arrive[(k, 0)]])
                self.records[k] = (rec0, rt)
                # the prefetcher's copy in this chip's L3, cluster 1's in its L1
                self.node(k, 0, dm, "rec_l3", "__snax_bingo_kernel_idma_1d_copy",
                          SnaxBingoKernelIdma1dCopyArgs(rec0, self.h_rec_l3, TOP_K * 128), [rt])
                rec1 = self._scratch(k, 1, "rec", TOP_K * 128)
                cp = self.node(k, 1, dm, "rec_pull", "__snax_bingo_kernel_idma_1d_copy",
                               SnaxBingoKernelIdma1dCopyArgs(rec0, rec1, TOP_K * 128), [rt])
                self.rec_port[g0] = Port(record_spec(TOP_K, 0), rec0, (rt,))
                self.rec_port[g1] = Port(record_spec(TOP_K, 1), rec1, (cp,))

    def _route_first(self):
        """The route ahead of the shared expert's loads on the DM core they share. Every DM
        core runs its tasks in order and a ring load blocks until its chunk lands, so a
        shared chip's router tasks (and on the root the route's own: the gather's pulls,
        the broadcast) sorted behind its shared-expert loads would wait for the shared
        stream, and every routed weight waits for the route. The prefetcher still pushes the
        shared chunks into L3 meanwhile; only their copies into L1 wait. Cluster 0 of each
        shared chip: after its last task of the route; the root's: after the route has
        reached it."""
        for k in self.shared_chips:
            g0 = self.gs(k)[0]
            self.ls[g0].wait_for(self.route_gate[k])

    # ---- the end --------------------------------------------------------------------------
    def _finish_graph(self):
        R, xb = self.ROOT, 2 * self.D
        self.end_window(list(self.reduce_end))
        checks = [("y", self.stash("y", R, 0, self.dst, xb, []),
                   np.asarray(self.gold["y"], np.float16), xb)]
        k = self.slot_chips[0][0]
        rec0, rt = self.records[k]
        golden = record_bytes(self.ids, np.asarray(self.w16, np.float16).view(np.uint16),
                              self.table_bytes[k])
        checks.append((f"rec_k{k:02x}", self.stash(f"rec_k{k:02x}", k, 0, rec0, TOP_K * 128, [rt]),
                       golden, TOP_K * 128))
        far = self.order[len(self.order) // 2]
        checks.append((f"loc_k{far:02x}", self.stash(f"loc_k{far:02x}", far, 0,
                                                     self.sv(far, 0, self.OFF_LOC), xb, []),
                       np.asarray(self.gold["loc"][far], np.float16), xb))
        self.run_checks(checks)


def _app_name(S):
    return (f"DeepSeek-V2-Lite layer {S.d.layer} MoE, one token, INT{S.wbits} experts, "
            f"{len(S.chips)} chips x {S.NC} clusters, crest={int(S.crest)} "
            f"route={S.route_impl} bcast={S.bcast_impl} reduce={S.reduce_impl} sync={S.sync}")


def run():
    run_driver("moe workload generator", MBuild, _app_name)
