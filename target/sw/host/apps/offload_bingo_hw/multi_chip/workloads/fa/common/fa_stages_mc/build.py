# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""FaStages' task graph: build() of the whole block (stop q8: up to every cluster's query tiles)."""

import re


class FaStagesBuild:

    def _next_id(self):
        """The node id the next task made gets."""
        return max((t.node_id for t in self.dfg.node_list), default=-1) + 1

    def relays(self):
        """Every dependency-tag relay of an xDMA task serves that one task; params relay_prefer {consumer role:
        [roles]}: dep-tag relays of that role's tasks go to those cores first (a relay joins its core's queue just
        before the first task there that its consumer reaches, so on a core busy to the end of the sweep, the SIMD
        core, it holds the chain back until the sweep ends)."""
        self.dfg.relay_one_consumer_cores = [self.roles["xdma"]]
        if self.p.get("relay_prefer"):
            self.dfg.relay_prefer_cores = {self.roles[c]: [self.roles[r] for r in rs]
                                           for c, rs in self.p["relay_prefer"].items()}

    def build(self):
        """Stages 0 to 3 (x from HBM, the keys by columns, the owners, the root's multicasts of the keys), then the
        chunks (full_chunk), then the last chunk's reduce, the window's end and its checks."""
        R = self.ROOT
        self.last_ld = {}
        # a slab is never loaded again before the reader of its last load is known
        for ls in self.ls:
            ls.strict = True
        pre = self.const_preload(self.stage1_preload())
        if self.kx_per_chunk:
            self._kx_l3_preload(pre)
        go = self.setup(extra=pre)
        self.relays()
        self.x_puts = self.stage_x_bcast(go)
        if self.pf_split:
            self._cluster_x_order()
        self.full_keys(go)
        if self.yr_rel1:
            if not self.y_roots:
                raise ValueError("params yr_rel1: needs y_roots")
            for i in range(1, self.NCH):
                for h in range(2):
                    k = self.y_root(i - 1, h)
                    self.ring_rel1 |= {(k, self.chunk_index(i, "head") + r) for r in range(self.per_chunk())}
            print(f"[{self.TAG}] yr_rel1: {sum(1 for k, n in self.ring_rel1 if k in self.y_roots)} ring chunks on the "
                  f"collectors released by cluster 1")
        res = dict(swp=[], post={}, parts={}, red={}, stores={}, xch_free={})
        for i in range(self.NCH):
            if i:
                self.chunk_start(i)
            self.full_chunk(i, go, res)
        parts_ready, stores = res["parts"], res["stores"]
        if self.stop == "q8" and not self.full_kv_all:
            # the window still ends with the keys on every chip: each one's wait for them, behind its stage 4
            for k in self.chips:
                if k != self.ROOT:
                    key = (k, 0, self.NCH - 1)
                    parts_ready[key] = list(parts_ready[key]) + [self.kv_all(k)]
        if self.stop == "full":
            self._reduce_chunk(self.NCH - 1, res["red"], stores, res["xch_free"])
        if stores:
            # every chunk's stores (the checks read every one)
            ends = [t for v in stores.values() for t in (v if isinstance(v, (list, tuple)) else [v])]
        else:
            # a stop before the reduce: one probe a chip behind that chip's last chunk, the root's last
            # (the end stamp sits behind the root's previous task on its core)
            last = {k: [t for c in range(self.NC) for t in parts_ready[(k, c, self.NCH - 1)]] for k in self.chips}
            ends = [self.probe(k, f"fin{k:02x}", last[k]) for k in self.chips if k != R]
            ends.append(self.probe(R, f"fin{R:02x}", last[R]))
        if self.keys_by_chunk and len(self.x_puts) > 2 and self._chains:
            # the rest of x is a multicast, and a reduce chain beside a multicast deadlocks the fabric: every chain
            # waits for it; the compiler proves the order
            for t in self._chains:
                self.edge(self.done_of(self.x_puts[2]), t)
        self.end_window(ends)
        if self.keys_by_chunk:
            self._keys_by_chunk_order()
            if self.kx_per_chunk:
                self._kx_per_chunk_edges()
            if self.kx_after_chains:
                self._kx_before_chains()
        swp_res = res["swp"][-1] if res["swp"] else {}
        self._checks(swp_res, res["post"])

    def full_keys(self, go):
        """Stages 1 to 3 of the whole block, the owners chained behind stage 1's waits; keys_by_chunk: stage 1's tasks
        carry their tile (_q<n>, the x waits _t<n>; the start's loads of W_KVA are chunk 0's), and the rest of x
        follows chunk 1's key multicast (x_rest_early: chunk 0's), as a dependency (the compiler proves the
        order). full_kv_all false: no chip waits for both chunks' keys at the end of stage 3; the sweep waits for its
        own tiles."""
        front = {k: [] for k in self.chips}
        k0 = self._next_id()
        self.stage_keys(go, front)
        if self.keys_by_chunk:
            P = self.C // 16
            for t in self.dfg.node_list:
                if t.node_id >= k0:
                    m = re.search(r"_[qt](\d+)", t.node_name)
                    t._key_chunk = int(m.group(1)) // P if m else 0
        self.stage_owner(go, front, end_wait=self.full_kv_all)
        if self.keys_by_chunk and len(self.x_puts) > 2 and len(self.kv_mc) > 1:
            self.edge(self.kv_mc[0 if self.x_rest_early else 1], self.first_of(self.x_puts[2]))

    def full_chunk(self, i, go, res):
        """Chunk i: per cluster its rows of x (x_rows), its queries (stage_proj) and its sweep; then W_UV (stage_post_a)
        and W_O (stage_post_b) of every cluster. Chunk i - 1's chains and stores of a root cluster are made behind its
        queries of chunk i (chains_after_sweep: behind its sweep), so on the root's in-order cores they run under chunk
        i. res: the sweeps', the post's and the parts' results, and the reduce's records (red, stores, xch_free)."""
        R = self.ROOT
        post_res, parts_ready = res["post"], res["parts"]
        red, stores, xch_free = res["red"], res["stores"], res["xch_free"]
        swp_res, sweeps = {}, {}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            chains_before = self.stop == "full" and i > 0 and k == R
            w = {c: [stores[(i - 2, c)]] if (i - 2, c) in stores else []}
            # the rows land without waiting for the previous chunk's readers of its sweep (the L1 planner keeps the two
            # apart)
            x = self.x_rows(cl, i, [go[k]])
            q8 = self.stage_proj(cl, i, x)
            if chains_before and not self.chains_after_sweep:
                self._block_chains(i - 1, [c], w, None, red, stores, xch_free)
            if self.stop == "q8":
                # every query tile's Q8 (each tile has its own ready tasks)
                parts_ready[(k, c, i)] = list(dict.fromkeys(t for q in sorted(q8) for t in q8[q][1]))
                post_res[cl] = dict(xt_rd=parts_ready[(k, c, i)])
                continue
            sw = self.sweep(cl, i, q8)
            swp_res[cl] = sw
            if chains_before and self.chains_after_sweep:
                self._block_chains(i - 1, [c], w, None, red, stores, xch_free)
            sweeps[cl] = sw
        res["swp"].append(swp_res)
        if self.stop == "q8":
            return
        # everything of chunk i before its W_O, then W_O, which waits for the chunk before's chains (the exchange buffer)
        post = {cl: self.stage_post_a(cl, i, sweeps[cl]) for cl in range(self.N_CL)}
        for cl in range(self.N_CL):
            k, c = self.chip(cl), self.local(cl)
            post_res[cl] = dict(xt_rd=post[cl][0])
            parts_ready[(k, c, i)] = self.stage_post_b(cl, i, post[cl][1], xch_free.get((k, c), []))
