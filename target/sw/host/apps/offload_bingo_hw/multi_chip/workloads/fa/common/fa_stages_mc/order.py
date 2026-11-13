# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""keys_by_chunk: the order of the multicasts, the reduce chains and the key tasks, the key tasks after the window, and
the check of the key transfers against the chains."""

import re


class FaStagesOrder:

    # the key transfers (the puts to the owners, the copies from HBM or L3)
    KX_TRANSFER = r"(kvc_put|kvx_put|Ld_wkvc_q|Ld_wkvxl_q|kvc_lld_q|kv_put_t)"

    def _order_collectives(self):
        """Every multicast and every reduce chain of the graph one after the other, in the order they were made (the
        graph's topological order nearest to it): an ordering edge from each to the next where no path orders them
        yet."""
        import networkx as nx
        coll = [c[:2] for c in self.collectives()]
        g = nx.DiGraph(self.dfg.bingo_order_view())
        pos = {t: n for n, t in enumerate(nx.lexicographical_topological_sort(g, key=lambda t: t.node_id))}
        coll.sort(key=lambda c: pos[c[0]])
        added = 0
        for (_, a), (b, _) in zip(coll, coll[1:]):
            if not nx.has_path(g, a, b):
                self.dfg.bingo_add_order_edge(a, b)
                g.add_edge(a, b)
                added += 1
        print(f"[{self.TAG}] collectives: {len(coll)} multicasts and chains in turn, {added} ordering edges added")

    def _check_kx_after_chains(self):
        """params kx_after_chains: every key transfer (the puts to the owners, the copies from HBM or L3) ordered with
        every reduce chain by the graph (dependencies and flag edges). Raises otherwise. (The multicasts and chains among
        themselves are the compiler's: passes/bingo_dfg_collectives.py.)"""
        if not self.kx_after_chains:
            return
        import networkx as nx
        g = nx.DiGraph(self.dfg)
        g.add_edges_from(self.dfg.order_edges)
        ch = [c[0] for c in self.collectives() if c[2] == "chain"]
        dch = {t: nx.descendants(g, t) for t in ch}
        kt = [t for t in self.dfg.node_list if re.match(self.KX_TRANSFER, t.node_name)]
        anc = {t: nx.ancestors(g, t) for t in ch}
        bad = [(a.node_name, b.node_name) for b in ch for a in kt if a not in dch[b] and a not in anc[b]]
        if bad:
            raise ValueError(f"params kx_after_chains: {len(bad)} key transfer / chain pair(s) may run at once: "
                             f"{bad[:6]}")
        print(f"[{self.TAG}] kx_after_chains: {len(kt)} key transfers, each ordered with every chain")

    def _kx_store_of(self, k, c, h):
        """kx_per_chunk with y_roots and yr_store_tile: cluster (k, c)'s store of chunk h - 2's y block when chip k
        collected it (chunk h's keys, made in chunk h - 1, follow it), else None."""
        if not (self.kx_per_chunk and self.y_roots and self.yr_store_tile is not None) or h < 2:
            return None
        if not hasattr(self, "_by_name"):
            self._by_name = {t.node_name: t for t in self.dfg.node_list}
        for hh in range(2):
            if self.y_root(h - 2, hh) == k:
                return self._by_name.get(f"y_store_c{c}_i{h - 2}_h{hh}_k{k:02x}_cl{c}")
        return None

    def _kx_per_chunk_edges(self):
        """params kx_per_chunk: on every cluster, the loads that start chunk h's key-tile buffers (h >= 1, inside the
        window: its first two tiles' x loads, its W_KVA, helped-pair and key-constant copies) depend on the cluster's
        last q projection start of chunk h - 1 (its x rows' last reader; the projections run in order on the GEMM core)
        and, on a collector of chunk h - 2, on its store of that chunk's y block (with yr_store_tile)."""
        PC = self.C // 16
        first = re.compile(r"^(kx_ld|kx_xld|Ld_wkvc|Ld_wkvxl|kvc_lld)_q(\d+)_k")
        by_cl = {}
        for t in self.dfg.node_list:
            by_cl.setdefault((t.assigned_chiplet_id, t.assigned_cluster_id), []).append(t)
        n = 0
        for (k, c), ts in by_cl.items():
            loads = {}
            for t in ts:
                m = first.match(t.node_name)
                if m and 1 <= int(m.group(2)) // PC < self.NCH:
                    loads.setdefault(int(m.group(2)) // PC, []).append((int(m.group(2)), t))
            for h, lst in loads.items():
                q2 = sorted({q for q, t in lst if t.node_name.startswith("kx_")})[:2]
                gq = [t for t in ts if re.match(rf"^gq\d+_i{h - 1}_k", t.node_name)]
                if not gq:
                    raise ValueError(f"params kx_per_chunk: no q projection of chunk {h - 1} on chip {k:02x} cluster {c}")
                srcs = [max(gq, key=lambda t: t.node_id)]
                st = self._kx_store_of(k, c, h)
                if st is not None:
                    # a collector of chunk h - 2: also behind its store of that chunk's y block (program order above)
                    srcs.append(st)
                if self.kx_after_chains and h >= 2:
                    # behind the last reduce chain of chunk h - 2 (the chains run one at a time unless chain_concurrent)
                    ch = [x for x in self._chains if re.match(rf"y_chain_c\d+_i{h - 2}_h\d+_", x.node_name)]
                    if not ch:
                        raise ValueError(f"params kx_after_chains: no chain of chunk {h - 2}")
                    srcs += ch if self.chain_concurrent else ch[-1:]
                for q, t in lst:
                    if q in q2 or not t.node_name.startswith("kx_"):
                        for src in srcs:
                            self.dfg.bingo_add_edge(src, t)
                            n += 1
        print(f"[{self.TAG}] kx_per_chunk: {n} key-buffer loads behind their cluster's last q projection of the chunk "
              f"before")

    def _kx_before_chains(self):
        """kx_after_chains: chunk h's key transfers (h >= 1: its HBM copies, its puts to the owners) ahead of chunk h - 1's
        reduce chains: each one an edge to every task of its chip, of that chunk, that those chains wait for (a cluster's
        last W_O GEMM of the chunk), so no key transfer of the chip runs beside a chain of the chunk before -- unless that
        task already precedes the transfer (a ring load behind the GEMM that frees its slot, which the chain's own
        release of the ring orders after the chain). With the edges of _kx_per_chunk_edges (behind chunk h - 2's chains)
        every key transfer is ordered with every chain (_check_kx_after_chains proves it)."""
        import networkx as nx
        PC = self.C // 16
        q_of = {t: q for q, t in getattr(self, "kv_put", {}).items()}
        feeds = {}                                  # (chunk, chip) -> the chains' inputs on that chip
        for ch in self._chains:
            m = re.match(r"y_chain_c\d+_i(\d+)_", ch.node_name)
            if m:
                i = int(m.group(1))
                for pr in self.dfg.predecessors(ch):
                    if f"_i{i}_" in pr.node_name:   # the chunk's own work (not a collector's store of the one before)
                        feeds.setdefault((i, pr.assigned_chiplet_id), set()).add(pr)
        bit = {pr: 1 << i for i, pr in enumerate(sorted({p for ps in feeds.values() for p in ps},
                                                         key=lambda x: x.node_id))}
        g = self.dfg.bingo_order_view()
        reach = {}                                  # task -> the bits of the chains' inputs that precede it
        for v in nx.topological_sort(g):
            r = bit.get(v, 0)
            for u in g.predecessors(v):
                r |= reach[u]
            reach[v] = r
        n = 0
        for t in list(self.dfg.node_list):
            if not re.match(self.KX_TRANSFER, t.node_name):
                continue
            m = re.search(r"_q(\d+)_", t.node_name)
            q = int(m.group(1)) if m else q_of.get(t)
            if q is None or not 1 <= q // PC < self.NCH:
                continue
            for pr in sorted(feeds.get((q // PC - 1, t.assigned_chiplet_id), ()), key=lambda x: x.node_id):
                if pr is not t and not reach[t] & bit[pr] and not self.dfg.has_edge(t, pr):
                    self.dfg.bingo_add_edge(t, pr)
                    n += 1
        print(f"[{self.TAG}] kx_after_chains: {n} edges, each key transfer ahead of the chunk before's chains")

    def _keys_by_chunk_order(self):
        """params keys_by_chunk: the key tasks of the chunks after the last one made (stage 1 by its tiles' names, stages 2
        and 3 by the runs stage_owner recorded) run after the window: each of them that no other one of them precedes
        waits for the end stamp. On the root, whose L1 is the fullest, the static L1 planner can then lay their buffers
        over the window's; everywhere, their transfers stay out of the window's last chains, which run two at a time
        and must not meet another chip's traffic on a hop."""
        runs = sorted(self._key_runs, key=lambda r: r[0])      # stable: a run's end before the next run's start
        for (a, h), (b, _) in zip(runs, runs[1:]):
            if h is None:
                continue
            for t in self.dfg.node_list:
                if a <= t.node_id < b:
                    t._key_chunk = h
        end_stamp = next(t for t in self.dfg.node_list if t.node_name.startswith("S1_end"))
        after = {t for t in self.dfg.node_list
                 if getattr(t, "_key_chunk", 0) >= self.NCH and t.node_id < end_stamp.node_id}
        preds = {t: set(self.dfg.predecessors(t)) for t in after}
        for x, y in self.dfg.order_edges:
            if y in preds:
                preds[y].add(x)
        heads = [t for t in after if not preds[t] & after]
        for t in sorted(heads, key=lambda t: t.node_id):
            self.dfg.bingo_add_edge(end_stamp, t)
        if heads:
            print(f"[{self.TAG}] keys_by_chunk: {len(after)} key tasks after the window, {len(heads)} behind the end "
                  f"stamp")
