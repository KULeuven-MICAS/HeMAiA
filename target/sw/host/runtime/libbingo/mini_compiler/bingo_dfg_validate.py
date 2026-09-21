# Fanchen Kong <fanchen.kong@kuleuven.be>

import networkx as nx

from bingo_dfg_common import _engine_of_kernel


class BingoDFGValidateMixin:
    """Passes that refuse a graph rather than let it fail on silicon.

    What these catch is otherwise invisible: a stuck dep-check does not raise, does not
    print and does not fault. The machine just stops, and it looks like a slow run. Each
    pass re-derives a property the lowering is supposed to guarantee instead of trusting
    that it did.

    Mixed into BingoDFG, so `self` is the whole DFG. These methods call each other
    and the other mixins' methods freely; nothing here is meant to stand alone.
    """

    def bingo_validate_no_hang(self, tag_width: int = None) -> dict:
        """Static check that the LOWERED graph cannot deadlock at runtime.

        This validates the OUTPUT of the lowering, re-deriving the properties
        the passes are supposed to guarantee instead of trusting that they did.
        An allocator bug that produces a hanging descriptor list is otherwise
        invisible until silicon: a stuck dep-check does not raise, does not
        time out and does not corrupt anything -- the machine simply stops.

        Must run LAST, after tags are allocated. Returns a summary dict; raises
        ValueError on the first violation with enough detail to act on.

        The four ways a lowered graph can hang:

        1. TAG MISMATCH -- a producer writes one tag and its consumer waits on
           another, so the bit the consumer wants is never set.
        2. UNSATISFIABLE CHECK -- a consumer checks a producer column that no
           reachable producer ever sets with that tag.
        3. CELL ALIASING -- two edges that can be live at the same time share
           one (cell, tag), i.e. ONE presence bit. The first consumer to check
           drains it and the second waits forever. This is the hazard per-edge
           tags exist to remove, so checking it here is the real oracle.
        4. CAPACITY -- a cell needs more distinct tags than the descriptor can
           encode.
        """
        if tag_width is None:
            tag_width = self.dep_tag_width
        max_tags = 1 << tag_width
        topo = self.bingo_stream_order()

        # Happens-before, including the same-core program order the manager's
        # in-order per-core queue enforces.
        hb = nx.DiGraph()
        hb.add_nodes_from(self.nodes())
        hb.add_edges_from(self.edges())
        by_core: dict = {}
        for nd in topo:
            by_core.setdefault((nd.assigned_chiplet_id, nd.assigned_cluster_id,
                                nd.assigned_core_id), []).append(nd)
        for seq in by_core.values():
            for i in range(len(seq) - 1):
                hb.add_edge(seq[i], seq[i + 1])
        _desc: dict = {}

        def reach(n):
            if n not in _desc:
                _desc[n] = nx.descendants(hb, n)
            return _desc[n]

        cells: dict = {}          # cell -> [(set_node, check_node), ...]
        covered: dict = {}        # (check_node, producer core) -> True
        for u, v in self.edges():
            if not (u.dep_set_enable and v.dep_check_enable):
                continue
            C, R = u.assigned_core_id, v.assigned_core_id
            if C not in (v.dep_check_list or []) or R not in (u.dep_set_list or []):
                continue
            # 1. TAG MISMATCH
            if u.dep_set_tag != v.dep_check_tag:
                raise ValueError(
                    f"hang check: '{u.node_name}' sets tag {u.dep_set_tag} but its "
                    f"consumer '{v.node_name}' waits on tag {v.dep_check_tag}. The "
                    f"consumer can never pass.")
            cells.setdefault((v.assigned_chiplet_id, v.assigned_cluster_id, R, C),
                             []).append((u, v))
            covered[(v, C)] = True

        # 2. UNSATISFIABLE CHECK
        for v in self.node_list:
            if not v.dep_check_enable:
                continue
            for c in (v.dep_check_list or []):
                if (v, c) not in covered:
                    raise ValueError(
                        f"hang check: '{v.node_name}' waits on producer core {c} "
                        f"with tag {v.dep_check_tag}, but no reachable producer "
                        f"sets that column with that tag. It can never dispatch.")

        # 3. CELL ALIASING + 4. CAPACITY
        peak = 0
        for cell, edges in cells.items():
            by_tag: dict = {}
            for (su, cv) in edges:
                by_tag.setdefault(su.dep_set_tag, []).append((su, cv))
            peak = max(peak, len(by_tag))
            if len(by_tag) > max_tags:
                raise ValueError(
                    f"hang check: cell {cell} holds {len(by_tag)} distinct tags "
                    f"but the descriptor encodes {max_tags} (tag_width={tag_width}).")
            for tag, el in by_tag.items():
                for i in range(len(el)):
                    for j in range(i + 1, len(el)):
                        (sa, ca), (sb, cb) = el[i], el[j]
                        fwd = (sb is ca) or (sb in reach(ca))
                        bwd = (sa is cb) or (sa in reach(cb))
                        if not fwd and not bwd:
                            raise ValueError(
                                f"hang check: two edges that can be live at the "
                                f"same time share tag {tag} on cell {cell}, which "
                                f"is ONE presence bit -- the first consumer to "
                                f"check drains it and the second waits forever.\n"
                                f"  edge A: {sa.node_name} -> {ca.node_name}\n"
                                f"  edge B: {sb.node_name} -> {cb.node_name}\n"
                                f"  cell = (chiplet, cluster, consumer core, "
                                f"producer core)")
        # 5. CROSS-DIE PREDICATE PATH. A conditionally-gated task on another die
        #    only learns its predicate if some task reachable from its gating
        #    node carries the CERF window to that die. The gating task's own
        #    dep_set does NOT: the dummy-set pass proxies every remote successor
        #    through a dummy on the gating task's core, and that proxy is what
        #    crosses. Without the carry bit the remote task reads a stale CERF
        #    and silently skips work the router selected.
        gating_targets = getattr(self, "_gating_to_targets", {})
        cross_die = 0
        for g, targets in gating_targets.items():
            remote = {t.assigned_chiplet_id for t in targets
                      if t.assigned_chiplet_id != g.assigned_chiplet_id}
            if not remote:
                continue
            carriers = {n.dep_set_chiplet_id for n in ([g] + list(nx.descendants(self, g)))
                        if getattr(n, "cerf_carry", False) and n.dep_set_enable}
            for chip in sorted(remote):
                if chip not in carriers:
                    raise ValueError(
                        f"hang check: gating task '{g.node_name}' gates work on "
                        f"chiplet {chip}, but nothing reachable from it carries "
                        f"the CERF window there (no node with cerf_carry and "
                        f"dep_set_chiplet_id={chip}). The remote task would read "
                        f"a stale CERF and skip silently.")
                cross_die += 1

        return {"edges": sum(len(e) for e in cells.values()),
                "cells": len(cells),
                "peak_tags_per_cell": peak,
                "tag_capacity": max_tags,
                "cross_die_gated": cross_die}

    def _validate_memory_handles(self, sorted_handles):
        """Validate memory allocation scopes before emitting C allocation calls."""
        valid_mem_levels = {"L1", "L2", "L3"}
        valid_chiplet_ids = set(self.chiplet_ids)

        for h in sorted_handles:
            if h.mem_level not in valid_mem_levels:
                raise ValueError(
                    f"Memory handle '{h.name}' uses invalid mem_level "
                    f"'{h.mem_level}'. Expected one of {sorted(valid_mem_levels)}."
                )

            if h.size <= 0:
                raise ValueError(
                    f"Memory handle '{h.name}' has invalid size {h.size}. "
                    "Allocation size must be positive."
                )

            if h.chip_id not in valid_chiplet_ids:
                raise ValueError(
                    f"Memory handle '{h.name}' targets chip_id 0x{h.chip_id:02x}, "
                    f"but this DFG only has chiplet IDs "
                    f"{[f'0x{cid:02x}' for cid in self.chiplet_ids]}."
                )

            if h.mem_level == "L1":
                if h.cluster_id < 0 or h.cluster_id >= self.num_clusters_per_chiplet:
                    raise ValueError(
                        f"Memory handle '{h.name}' targets cluster_id "
                        f"{h.cluster_id}, but chiplet 0x{h.chip_id:02x} has "
                        f"clusters 0..{self.num_clusters_per_chiplet - 1}."
                    )
            elif h.cluster_id != 0:
                raise ValueError(
                    f"Memory handle '{h.name}' is allocated in {h.mem_level}, "
                    f"but has cluster_id {h.cluster_id}. Only L1 allocations "
                    "use cluster_id; L2/L3 allocation calls use chip_id only."
                )

    def _validate_kernel_core_assignments(self, sorted_nodes):
        """Validate that kernel namespace matches the HW scheduler core target."""
        valid_chiplet_ids = set(self.chiplet_ids)
        host_core_id = self.num_cores_per_cluster - 1 if self.is_host_as_acc else None
        num_snax_cores = host_core_id if self.is_host_as_acc else self.num_cores_per_cluster
        failures = []
        # The generated engine->core map, if this build has one. core_roles() raises for a
        # cluster that lacks an engine and for a map that has not been generated yet;
        # neither is a placement error, so the per-node engine check is simply skipped.
        try:
            from bingo_platform import core_roles
            roles = core_roles()
        except Exception:
            roles = None

        def node_label(node):
            return (
                f"Node ID {node.node_id} ('{node.node_name}', "
                f"kernel={node.kernel_name})"
            )

        for node in sorted_nodes:
            if node.assigned_chiplet_id not in valid_chiplet_ids:
                failures.append(
                    f"{node_label(node)} targets chiplet 0x{node.assigned_chiplet_id:02x}, "
                    f"but valid chiplets are {[f'0x{cid:02x}' for cid in self.chiplet_ids]}."
                )

            if node.assigned_cluster_id < 0 or node.assigned_cluster_id >= self.num_clusters_per_chiplet:
                failures.append(
                    f"{node_label(node)} targets cluster {node.assigned_cluster_id}, "
                    f"but valid clusters are 0..{self.num_clusters_per_chiplet - 1}."
                )

            if node.assigned_core_id < 0 or node.assigned_core_id >= self.num_cores_per_cluster:
                failures.append(
                    f"{node_label(node)} targets core {node.assigned_core_id}, "
                    f"but valid cores are 0..{self.num_cores_per_cluster - 1}."
                )

            kernel_name = node.kernel_name or ""
            if not kernel_name:
                continue

            if kernel_name.startswith("__host"):
                if not self.is_host_as_acc:
                    failures.append(
                        f"{node_label(node)} is a host kernel, but this DFG was "
                        "created with is_host_as_acc=False."
                    )
                else:
                    if node.assigned_core_id != host_core_id or node.assigned_cluster_id != 0:
                        failures.append(
                            f"{node_label(node)} is a host kernel and must be assigned "
                            f"to chiplet-local host core cluster 0 core {host_core_id}; "
                            f"got cluster {node.assigned_cluster_id} core {node.assigned_core_id}."
                        )
            elif kernel_name.startswith("__snax"):
                if node.assigned_core_id >= num_snax_cores:
                    failures.append(
                        f"{node_label(node)} is a SNAX/device kernel and must be assigned "
                        f"to a real cluster core in 0..{num_snax_cores - 1}; "
                        f"got core {node.assigned_core_id}."
                    )
                # And on the RIGHT cluster core. Being in range is not enough: every hart
                # exposes its accelerator CSRs at the same offsets, so a node on the wrong
                # hart programs whatever is there -- or nothing -- and reports success.
                # Only enforced when the generated role map is readable; a cluster that
                # genuinely lacks an engine makes core_roles() raise, and a missing map is
                # a build-order problem, not a placement error.
                elif roles is not None:
                    engine = _engine_of_kernel(kernel_name)
                    want = roles.get(engine) if engine else None
                    if want is not None and node.assigned_core_id != want:
                        failures.append(
                            f"{node_label(node)} drives the {engine.upper()} engine, which "
                            f"the generated role map puts on core {want}, but it is "
                            f"assigned core {node.assigned_core_id}. This does not fault "
                            f"at run time -- that hart's CSR window answers at the same "
                            f"offsets -- so it would silently do nothing."
                        )
            else:
                failures.append(
                    f"{node_label(node)} has an unknown kernel namespace. Kernel names "
                    "must start with '__host' or '__snax' so the generated host/device "
                    "task mappings match the HW scheduler routing."
                )

            if node.kernel_args:
                struct_name = node.kernel_args.get_struct_name()
                if kernel_name.startswith("__host") and not struct_name.startswith("__host"):
                    failures.append(
                        f"{node_label(node)} uses host kernel namespace but argument "
                        f"struct '{struct_name}' is not a host argument struct."
                    )
                if kernel_name.startswith("__snax") and not struct_name.startswith("__snax"):
                    failures.append(
                        f"{node_label(node)} uses SNAX kernel namespace but argument "
                        f"struct '{struct_name}' is not a SNAX argument struct."
                    )

        if failures:
            raise ValueError(
                "Bingo kernel/core assignment check failed before C generation:\n"
                + "\n".join(f"- {failure}" for failure in failures)
            )

    def _validate_cerf_cross_group_edges(self):
        """Detect unconditional edges from CERF-gated nodes to nodes outside
        their CERF group.  Such edges produce bridge tasks that deadlock when
        the source's CERF group is skipped (the source never signals)."""
        node_to_group = getattr(self, '_node_to_cerf_group', {})
        if not node_to_group:
            return
        for u, v, data in self.edges(data=True):
            if data.get('cond', False):
                continue
            if u not in node_to_group:
                continue
            if getattr(v, 'node_type', 'normal') in ('entry', 'exit', 'gating'):
                continue
            # Exit/entry nodes added by the compiler have SW guard support
            if v.kernel_name and ('exit' in v.kernel_name or 'entry' in v.kernel_name):
                continue
            src_grp = node_to_group[u]
            dst_grp = node_to_group.get(v)
            if dst_grp != src_grp:
                dst_info = (f"CERF group {dst_grp}" if dst_grp is not None
                            else "not CERF-gated")
                raise ValueError(
                    f"Unconditional edge '{u.node_name}' (CERF group "
                    f"{src_grp}) -> '{v.node_name}' ({dst_info}) crosses a "
                    f"CERF boundary. When group {src_grp} is skipped, "
                    f"'{u.node_name}' will not signal completion, "
                    f"deadlocking '{v.node_name}'. To verify results of "
                    f"CERF-gated computations, use post_execute_code in "
                    f"bingo_compile_dfg() instead.")
