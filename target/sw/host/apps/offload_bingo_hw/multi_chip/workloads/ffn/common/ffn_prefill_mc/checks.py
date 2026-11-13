# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The block's checks: the data checks the root's host runs, the buffer-hazard proof over the graph, and (sync barrier)
the proof that only the barriers synchronise the chips inside the window."""

import re

import networkx as nx
import numpy as np


class PChecks:
    """PBuild._checks, check_hazards and finish."""

    def _checks(self):
        """Y_raw sampled on the root's L3 (the first and the last 2 KiB of every pass), or with y_scatter the first, a
        middle and the last tile in their owners' L3; and the root's own A8 of the last pass (2 KiB)."""
        lay, R = self.lay, self.ROOT
        y = self.gold["y_raw"]
        checks = []
        n = 2048
        if self.y_scatter:
            nt = self.T // 16
            for q in (0, nt // 2, nt - 1):
                img = lay.to_d(y[q * 16:(q + 1) * 16], self.d.mesh).astype(np.float16).reshape(-1).view(np.uint8)
                for name, off in ((f"yt_q{q}_a", 0), (f"yt_q{q}_m", self.YT_B // 2 - n // 2), (f"yt_q{q}_b", self.YT_B - n)):
                    checks.append((name, self.yt_sym(q, off, chip_id=self.y_owner(q)[0]), img[off:off + n], n))
        for i in range(self.NI if not self.y_scatter else 0):
            rows = y[i * self.ROWS:(i + 1) * self.ROWS]
            img = np.concatenate([lay.to_d(rows[:, c0:c0 + w], self.d.mesh)
                                  for c0, w in self.y_slices()]).astype(np.float16).view(np.uint8)
            for name, off in ((f"y_i{i}_a", 0), (f"y_i{i}_b", self.YB - n)):
                checks.append((name, self._ysym(i * self.YB + off, chip_id=R),
                               img[off:off + n], n))
        cl = self.clusters(R)[0]
        c0, ncol = self.kcols[R]                # the chip's A8, both clusters
        i = self.NI - 1
        a8 = self.d.a8[i * self.ROWS:(i + 1) * self.ROWS, c0:c0 + ncol]
        img = lay.to_a(a8, self.d.mesh).reshape(-1).view(np.uint8)
        got = self.stash("a8_root", R, 0, self.a8[cl][i % 2], n, [])
        checks.append(("a8_root", got, img[:n], n))
        self.run_checks(checks)

    # ---- the graph's checks -----------------------------------------------------------------------
    def _sync_graph(self):
        """The graph the hazard check walks: the DFG, plus (sync barrier) what a chip barrier orders across chips: for
        each barrier round a node behind every chip's barrier task and ahead of every task that follows any of them."""
        if self.sync != "barrier":
            return self.dfg.bingo_order_view()
        H = nx.DiGraph(self.dfg.bingo_order_view())
        for n, rnd in enumerate(self.barrier_rounds):
            v = ("barrier_round", n)
            for t in rnd.values():
                H.add_edge(t, v)
                H.add_edges_from((v, s) for s in self.dfg.successors(t))
        return H

    def check_window_local(self):
        """sync barrier: no dependency edge crosses chips between the start barrier and the end stamp; only the barriers
        synchronise the chips there. Returns the offending edges."""
        pre = set().union(*(nx.ancestors(self.dfg, t) for t in self.go.values()))
        post = nx.descendants(self.dfg, self.end) | {self.end}
        return [(u, v) for u, v in self.dfg.edges()
                if u.assigned_chiplet_id != v.assigned_chiplet_id and u not in pre and u not in post and v not in post]

    def check_xch_offsets(self):
        """Every exchange buffer at offset 0 of its arena; with sync barrier also every X buffer right behind it, where
        the output slots after the exchange buffer's sit."""
        super().check_xch_offsets()
        if self.per_kernel:
            pl = self.dfg.static_l1_placement
            bad = [(cl, pl[id(self.xbuf[cl][0])][1] if id(self.xbuf[cl][0]) in pl else None) for cl in range(self.N_CL)
                   if id(self.xbuf[cl][0]) not in pl or pl[id(self.xbuf[cl][0])][1] != self.xch_bytes]
            if bad:
                raise ValueError(f"sync barrier: X buffers not at offset {self.xch_bytes}: {bad[:4]}")

    def check_hazards(self):
        """The passes reuse L1 buffers while an earlier slice's or pass's tail is still running on other cores; nothing
        orders those but the edges the generator adds. For every cluster and every pair of consecutive uses of a buffer
        slot, the last reader of the earlier use must reach the first writer of the later one in the DFG (the cores run
        in the order of their edges): a missing edge is a data race that no golden of one pass shows. Returns the
        violations."""
        byk = {}
        for t in self.dfg.node_list:
            m = re.match(r"^(.*)_k([0-9a-f]{2})_cl(\d)$", t.node_name)
            if m:
                byk[(m.group(1), int(m.group(2), 16), int(m.group(3)))] = t
        at = lambda name, cl: byk.get((name, self.chip(cl), self.local(cl)))     # noqa: E731  cluster cl's task `name`
        bad = []
        H = self._sync_graph()
        rd = at                                                      # a GEMM's last read of its operands

        def ordered(a, b, what):
            if a is None or b is None:
                bad.append(f"{what}: a task is missing")
            elif not nx.has_path(H, a, b):
                bad.append(f"{what}: {a.node_name} is not before {b.node_name}")

        # cluster c's output slot: its gather of a slice before the next down GEMM into that slot (out_slots slices
        # later, or in the next pass); the root's fold slot: the store of a slice before the next gather into it; every
        # A8 copy into a cluster before that cluster's first gather of the pass. Between passes: the A8, A16, A16a and X
        # buffers' last readers of a pass before their first writers of the next
        R, h, pb, GS = self.ROOT, self.NDN // self.NC, self.post_blocks, self.chain_group
        grp = lambda j, c: c * h + (j - c * h) // GS * GS       # noqa: E731  params chain_group: the gather of slice j
        store = lambda i, j, c: byk.get((f"y_store_j{grp(j, c)}" if i == 0 else f"y_store_i{i}j{grp(j, c)}", R, c))  # noqa: E731
        # the task after which every chip's output slot of slice j has been read: its gather, or (params reduce root)
        # the slice's store, behind every pull and add of it
        chain = (lambda i, j, c: byk.get((f"y_chain_j{grp(j, c)}" if i == 0 else f"y_chain_i{i}j{grp(j, c)}", R, c))) \
            if self.reduce_impl == "chain" else store
        push = lambda i, m, cl: at(f"a8_push_m{m}" if i == 0 else f"a8_push_i{i}m{m}", cl)    # noqa: E731

        def dn_first(i, c, jl, cl):    # the first down GEMM of a slice (a split first slice: its block 0)
            return at(f"dn_c{c * h + jl}_i{i}", cl) or at(f"dn_c{c * h}_i{i}b0", cl)
        for c in range(self.NC):
            uses = [(i, jl) for i in range(self.NI) for jl in range(h)]
            S = self.out_slots
            for n_, (i, jl) in enumerate(uses):
                j = c * h + jl
                nxt = next(((i2, jl2) for i2, jl2 in uses[n_ + 1:] if jl2 % S == jl % S), None)
                if nxt is not None:
                    for cl in range(self.N_CL):
                        if self.local(cl) == c:
                            ordered(chain(i, j, c), dn_first(nxt[0], c, nxt[1], cl),
                                    f"output slot cluster {cl} i{i}j{j}->i{nxt[0]}j{c * h + nxt[1]}")
                # the next gather into the same fold slot (another group); a root fold's slots are its own pipeline's
                nxt = next(((i2, jl2) for i2, jl2 in uses[n_ + 1:] if jl2 // GS % 2 == jl // GS % 2
                            and (i2, jl2 // GS) != (i, jl // GS)), None)
                if nxt is not None and self.reduce_impl == "chain":
                    ordered(store(i, j, c), chain(nxt[0], c * h + nxt[1], c),
                            f"root fold slot c{c} i{i}j{j}->i{nxt[0]}j{c * h + nxt[1]}")
        for i in range(self.NI):
            for cl in range(self.N_CL):      # every copy into cluster c before cluster c's first gather of the pass
                c = self.local(cl)
                first = chain(i, c * h, c)
                other = self.clusters(self.chip(cl))[1 - c]
                for m in range(self.MT):
                    ordered(push(i, m, other), first, f"A8 copy i{i}m{m} into cluster {cl} before its first gather")
                if i == 0:
                    continue
                nG = len(self.chunks[cl])
                last_dn = rd(f"dn_c{c * h + h - 1}_i{i - 1}", cl)
                ordered(last_dn, at(f"quant_i{i}b0m0", cl), f"A8 cluster {cl} i{i - 1}->i{i} (quantise)")
                ordered(last_dn, push(i, 0, other), f"A8 cluster {cl} i{i - 1}->i{i} (copy from cluster {other})")
                ordered(chain(i - 1, self.NDN - 1, self.NC - 1), push(i, 0, cl),
                        f"cluster {cl}'s first A8 copy of pass {i} after the last gather of pass {i - 1}")
                ordered(at(f"d2a_i{i - 1}b{pb - 1}", cl), at(f"swi_c0m0_i{i}", cl), f"a16d cluster {cl} i{i - 1}->i{i}")
                ordered(at(f"quant_i{i - 1}b{pb - 1}m{self.MT - 1}", cl), at(f"d2a_i{i}b0", cl),
                        f"a16a cluster {cl} i{i - 1}->i{i}")
                lg = rd(f"gu_c{nG - 1}_i{i - 1}b{pb - 1}", cl) or rd(f"gu_c{nG - 1}_i{i - 1}", cl)
                ordered(lg, at(f"x_wait_i{i}", cl), f"x cluster {cl} i{i - 1}->i{i}")
        if self.per_kernel:
            # the output slots in the X buffer: read by pass i - 1's reduce before X_i lands there, and X_i read by the
            # pass's last gate|up GEMM before a down GEMM writes there
            for i in range(self.NI):
                for cl in range(self.N_CL):
                    c = self.local(cl)
                    lg = rd(f"gu_c{len(self.chunks[cl]) - 1}_i{i}", cl)
                    for jl in range(self.xch_slots, h):
                        if i:
                            ordered(chain(i - 1, c * h + jl, c), at(f"x_wait_i{i}", cl),
                                    f"X buffer cluster {cl}: slot {jl} of pass {i - 1} reduced before X_{i} lands")
                        ordered(lg, dn_first(i, c, jl, cl),
                                f"X buffer cluster {cl}: X_{i} read before slot {jl} is written")
        if self.l3_overlay:
            # Y_raw_i over X_i's L3 buffers: the root clusters' fetches of X_i before every store of pass i
            for i in range(self.NI):
                blk = [f"i0b{b}" for b in range(self.x_blocks)] if i == 0 and self.x_blocks > 1 else [f"i{i}"]
                readers = [at(f"x_wait_{t}", cl) for cl in self.clusters(R) for t in blk]
                for c in range(self.NC):
                    for jl in range(h):
                        for r in readers:
                            ordered(r, store(i, c * h + jl, c), f"X_{i} read before the store over it, y i{i}j{c * h + jl}")
        return bad

    def finish(self, app_name):
        if self.sync == "barrier":
            if self.barrier_fence:              # the fences first, so the hazard check sees them
                self.fence_barriers()
                self.barrier_fence = False
            cross = self.check_window_local()
            if cross:
                raise ValueError(f"sync barrier: {len(cross)} dependency edges cross chips inside the window, e.g. "
                                 + "; ".join(f"{u.node_name} -> {v.node_name}" for u, v in cross[:4]))
            print(f"[{self.TAG}] sync barrier (kernel): {self.n_barrier} barriers a chip, no edge crosses "
                  f"chips inside the window")
        bad = self.check_hazards()
        if bad:
            raise ValueError(f"{len(bad)} buffer hazards, e.g. " + "; ".join(bad[:4]))
        print(f"[{self.TAG}] buffer hazards: none (A16, A16a, A8, X, output and fold slots; every cluster)")
        super().finish(app_name)
