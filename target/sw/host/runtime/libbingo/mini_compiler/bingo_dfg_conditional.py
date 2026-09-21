# Fanchen Kong <fanchen.kong@kuleuven.be>

import networkx as nx

from bingo_kernel_args import BINGO_GATING_MODE_STATIC
from bingo_kernel_args import BINGO_GATING_MODE_THRESHOLD
from bingo_kernel_args import BINGO_GATING_MODE_TOP_K
from bingo_kernel_args import HostBingoKernelCerfGatingArgs
from bingo_mem_handle import BingoMemAlloc
from bingo_node import BingoNode


class BingoDFGConditionalMixin:
    """Conditional regions and their CERF groups.

    A gating task picks which of its guarded successors may run; the CERF group is the
    window that decision is published in, and the allocation here is what keeps two
    regions from sharing one.

    Mixed into BingoDFG, so `self` is the whole DFG. These methods call each other
    and the other mixins' methods freely; nothing here is meant to stand alone.
    """

    def _alloc_cerf_group(self, hint: str = "") -> int:
        """Allocate the next CERF group ID. Raises on overflow (>31)."""
        gid = self._next_cerf_group
        self._next_cerf_group += 1
        if gid >= 32:
            raise ValueError(
                f"CERF group overflow: need group {gid} but max is 31 "
                f"(WF4 violated). {hint}")
        return gid

    def bingo_compile_conditional_regions(self) -> dict:
        """Compile conditional edges into CERF group assignments.

        Also validates that all nodes have valid core assignments
        (catches missing ``bingo_auto_assign()`` calls).

        Scans every edge for the ``cond`` attribute set by
        ``bingo_add_edge(..., cond=True)``.  For each gating node (a node
        with at least one outgoing conditional edge):

        1. Collect the set of conditional targets.
        2. Build an undirected subgraph of *unconditional* edges among those
           targets and find connected components — targets connected by
           unconditional edges share one CERF group.
        3. Assign one CERF group per component and annotate the target nodes.
        4. Promote the gating node to ``node_type="gating"`` and record its
           ``cerf_write_groups``.

        Must be called **before** the dummy-node transforms.

        Returns:
            dict mapping each conditionally-gated BingoNode to its CERF
            group id.  Also stored in ``self._node_to_cerf_group``.
        """
        # -- Validate core assignments ----------------------------------------
        unassigned = [n for n in self.node_list if n.assigned_core_id < 0]
        if unassigned:
            names = ", ".join(n.node_name for n in unassigned[:5])
            suffix = f" (and {len(unassigned)-5} more)" if len(unassigned) > 5 else ""
            raise ValueError(
                f"{len(unassigned)} node(s) have no core assignment: "
                f"{names}{suffix}. "
                f"Call bingo_auto_assign() before compile, or provide "
                f"explicit (chiplet, cluster, core) in BingoNode()."
            )

        # -- Step 1: identify gating nodes and their conditional targets ------
        gating_to_targets: dict[BingoNode, set[BingoNode]] = {}
        for u, v, data in self.edges(data=True):
            if data.get("cond", False):
                gating_to_targets.setdefault(u, set()).add(v)

        if not gating_to_targets:
            self._node_to_cerf_group = {}
            return {}

        # -- Sanity: a CERF-gated node must not be the source of new cond edges --
        # If node X is a conditional target (will be CERF-gated) and also has
        # outgoing conditional edges, the compiler would insert a gating node on
        # X's (CERF-skippable) cluster. When that cluster is inactive the gating
        # node cannot run, deadlocking all downstream targets.
        all_cond_targets: set[BingoNode] = set()
        for targets in gating_to_targets.values():
            all_cond_targets.update(targets)
        for source_node in gating_to_targets:
            if source_node in all_cond_targets:
                parent = next(
                    (g.node_name for g, ts in gating_to_targets.items()
                     if source_node in ts), "?")
                raise ValueError(
                    f"Node '{source_node.node_name}' has outgoing conditional "
                    f"edges but is itself a conditional target (gated by "
                    f"'{parent}'). The auto-inserted gating node would be "
                    f"placed on a CERF-skippable cluster, causing a deadlock "
                    f"when that cluster is inactive. To verify results of "
                    f"CERF-gated computations, use post_execute_code in "
                    f"bingo_compile_dfg() instead.")

        # -- Auto-insert gating nodes for sources with cond_dic ------
        inserted_gating_nodes = {}  # source_node → (gating_node, cond_dic)
        for source_node in list(gating_to_targets.keys()):
            # Collect cond_dic from edges (all edges from same source share config)
            cond_dic = {}
            for _, v, data in self.out_edges(source_node, data=True):
                if data.get('cond', False) and data.get('cond_dic'):
                    cond_dic = data['cond_dic']
                    break

            # Skip if no cond_dic — use legacy promote-in-place
            if not cond_dic:
                continue

            # Create gating node on same core as source
            gating_node = BingoNode(
                source_node.assigned_chiplet_id,
                source_node.assigned_cluster_id,
                source_node.assigned_core_id,
                node_name=f"__gating_{source_node.node_name}",
            )

            # Splice: source → gating_node → [conditional targets]
            cond_succs = [v for v in self.successors(source_node)
                          if self[source_node][v].get('cond', False)]
            self.bingo_insert_node_after(source_node, gating_node,
                                         successors_to_move=cond_succs)

            inserted_gating_nodes[source_node] = (gating_node, cond_dic)
            # Transfer target ownership from source to gating_node
            gating_to_targets[gating_node] = gating_to_targets.pop(source_node)

        # -- WF1: Acyclicity (only checked when conditional edges exist) ------
        if not nx.is_directed_acyclic_graph(self):
            raise ValueError(
                "Conditional DFG is not a DAG — it contains a cycle. "
                "Well-formedness condition WF1 violated."
            )

        # -- WF2: validate single-gating-source per target --------------------
        target_to_gating: dict[BingoNode, BingoNode] = {}
        for gating_node, targets in gating_to_targets.items():
            for t in targets:
                if t in target_to_gating:
                    raise ValueError(
                        f"Node '{t.node_name}' is conditionally gated by both "
                        f"'{target_to_gating[t].node_name}' and "
                        f"'{gating_node.node_name}'.  Hardware supports only "
                        f"one CERF group per task (WF2 violated)."
                    )
                target_to_gating[t] = gating_node

        # -- WF5: gating precedence (each gating node is ancestor of targets) -
        for gating_node, targets in gating_to_targets.items():
            for t in targets:
                if not nx.has_path(self, gating_node, t):
                    raise ValueError(
                        f"Gating node '{gating_node.node_name}' is not an "
                        f"ancestor of conditional target '{t.node_name}'. "
                        f"Well-formedness condition WF5 violated."
                    )

        # -- Step 3: per gating node — connected-component grouping -----------
        #
        # CERF group reuse: if all gating nodes are totally ordered
        # (each is an ancestor of the next), their conditional targets
        # execute at different times and can safely share group IDs.
        # The clear-before-set protocol in the gating task ensures that
        # stale group values from a previous layer are overwritten.
        node_to_group: dict[BingoNode, int] = {}

        gating_ordered = [
            n for n in nx.topological_sort(self) if n in gating_to_targets
        ]
        reuse_groups = len(gating_ordered) > 1 and all(
            nx.has_path(self, gating_ordered[i], gating_ordered[i + 1])
            for i in range(len(gating_ordered) - 1)
        )
        pool_start = self._next_cerf_group

        for gating_node in gating_ordered:
            targets = gating_to_targets[gating_node]
            gating_node.node_type = "gating"

            # Reset group counter to pool start for reuse
            if reuse_groups:
                self._next_cerf_group = pool_start

            # Build undirected graph of unconditional edges among targets
            unc = nx.Graph()
            unc.add_nodes_from(targets)
            for t in targets:
                for _, v, d in self.out_edges(t, data=True):
                    if v in targets and not d.get("cond", False):
                        unc.add_edge(t, v)
                for u, _, d in self.in_edges(t, data=True):
                    if u in targets and not d.get("cond", False):
                        unc.add_edge(u, t)

            # Sort components deterministically by lowest node_id so that
            # expert_i always gets the same CERF group across reused layers.
            components = sorted(
                nx.connected_components(unc),
                key=lambda c: min(n.node_id for n in c),
            )

            # Assign CERF groups. If more components than 32, share groups
            # (multiple experts per CERF group — HW skip at group level,
            # SW guard at expert level within active groups).
            num_components = len(components)
            max_cerf = 32 - self._next_cerf_group
            if max_cerf <= 0:
                max_cerf = 32  # will overflow, _alloc_cerf_group raises

            # Assign cond_node_index to each target for SW guard
            expert_idx_counter = 0

            group_ids = []
            if num_components <= max_cerf:
                # Normal: 1 CERF group per component (no group sharing)
                for component in components:
                    hint = ("Sequential gating reuse is active — "
                            "too many experts per layer."
                            if reuse_groups else
                            "Consider reducing experts or making "
                            "gating nodes sequential for reuse.")
                    gid = self._alloc_cerf_group(hint)
                    for node in component:
                        node.cond_exec_en = True
                        node.cond_exec_group_id = gid
                        node.cond_exec_invert = False
                        node._cond_node_index = expert_idx_counter
                        node_to_group[node] = gid
                    expert_idx_counter += 1
                    group_ids.append(gid)
            else:
                # CERF group sharing: multiple components per group.
                # Distributes components round-robin across available groups.
                n_groups = min(max_cerf, 32)
                allocated_gids = [self._alloc_cerf_group(
                    f"Sharing mode: {num_components} components across {n_groups} groups"
                ) for _ in range(n_groups)]
                for comp_idx, component in enumerate(components):
                    gid = allocated_gids[comp_idx % n_groups]
                    for node in component:
                        node.cond_exec_en = True
                        node.cond_exec_group_id = gid
                        node.cond_exec_invert = False
                        node._cond_node_index = expert_idx_counter
                        node_to_group[node] = gid
                    expert_idx_counter += 1
                    if gid not in group_ids:
                        group_ids.append(gid)
                print(f"CERF group sharing: {num_components} components "
                      f"mapped to {n_groups} groups "
                      f"({num_components/n_groups:.1f} components/group)")

            gating_node.cerf_write_groups = sorted(set(
                gating_node.cerf_write_groups + group_ids
            ))
            # Set gating node reference on all guarded expert nodes (for SW guard wiring)
            for target in targets:
                target._gating_node = gating_node

        self._node_to_cerf_group = node_to_group

        # -- Auto-populate gating kernel args for inserted gating nodes ------
        # All modes use the unified __host_bingo_kernel_cerf_gating kernel.
        for source_node, (gating_node, cond_dic) in inserted_gating_nodes.items():
            cerf_controlled_mask = sum(1 << g for g in gating_node.cerf_write_groups)
            num_groups = len(gating_node.cerf_write_groups)
            mode_str = cond_dic.get('mode', 'static')

            gating_node.kernel_name = "__host_bingo_kernel_cerf_gating"

            if mode_str == 'top_k':
                # Count total conditional targets for per-expert activation array
                num_experts = len(gating_to_targets[gating_node])
                cerf_gids_alloc = BingoMemAlloc(
                    f"__cerf_gids_{source_node.node_name}",
                    num_experts, "L3",
                    chip_id=source_node.assigned_chiplet_id, cluster_id=0)
                # Per-expert activation array (uint8_t[num_experts]):
                # Gating kernel writes 1 for selected experts, 0 for others.
                # Expert kernels read their slot via SW guard.
                expert_activation_alloc = BingoMemAlloc(
                    f"__cond_act_{source_node.node_name}",
                    num_experts, "L3",
                    chip_id=source_node.assigned_chiplet_id, cluster_id=0)
                gating_node.kernel_args = HostBingoKernelCerfGatingArgs(
                    mode=BINGO_GATING_MODE_TOP_K,
                    cerf_controlled_mask=cerf_controlled_mask,
                    top_k_or_threshold=cond_dic['k'],
                    cerf_group_ids_addr=cerf_gids_alloc,
                    cond_activation_addr=expert_activation_alloc,
                )
                gating_node._pred_source_node = source_node

            elif mode_str == 'threshold':
                gating_node.kernel_args = HostBingoKernelCerfGatingArgs(
                    mode=BINGO_GATING_MODE_THRESHOLD,
                    cerf_controlled_mask=cerf_controlled_mask,
                    top_k_or_threshold=cond_dic['threshold'],
                )
                gating_node._pred_source_node = source_node

            elif mode_str == 'static':
                write_mask = cond_dic.get('write_mask', cerf_controlled_mask)
                gating_node.kernel_args = HostBingoKernelCerfGatingArgs(
                    mode=BINGO_GATING_MODE_STATIC,
                    cerf_controlled_mask=cerf_controlled_mask,
                    top_k_or_threshold=write_mask,
                )

            elif mode_str == 'custom':
                gating_node.kernel_name = cond_dic['kernel_name']
                args_cls = cond_dic.get('kernel_args_cls')
                args_kwargs = cond_dic.get('kernel_args_kwargs', {})
                if args_cls:
                    gating_node.kernel_args = args_cls(
                        cerf_controlled_mask=cerf_controlled_mask, **args_kwargs)

            else:
                raise ValueError(f"Unknown gating mode: '{mode_str}'")

            # Store expert→CERF group mapping for cerf_group_ids initialization
            self._gating_cerf_mappings[gating_node] = {
                i: gid for i, gid in enumerate(sorted(gating_node.cerf_write_groups))
            }

        return node_to_group

    def bingo_define_conditional_region(
        self,
        gating_node: BingoNode,
        guarded_nodes: list,
        group_per_node: bool = False,
        invert: bool = False,
    ) -> list[int]:
        """Define a conditional execution region controlled by a gating task.

        The gating_node is marked as type 'gating' (task_type=2 in RTL).
        When it completes on a core, the hardware writes the assigned CERF
        groups, causing guarded_nodes to either execute or be skipped.

        Args:
            gating_node:    The node whose completion activates the CERF groups.
            guarded_nodes:  Nodes whose execution depends on the CERF state.
            group_per_node: If True, each guarded node gets its own CERF group
                            (MoE: each expert independently gated).
                            If False, all guarded nodes share one CERF group
                            (early exit: entire stage gated together).
            invert:         If True, guarded nodes execute when group is INACTIVE.

        Returns:
            List of assigned CERF group IDs. Length equals len(guarded_nodes)
            when group_per_node=True, or [single_id] when False.
        """
        gating_node.node_type = "gating"

        if group_per_node:
            group_ids = []
            for node in guarded_nodes:
                gid = self._alloc_cerf_group()
                node.cond_exec_en = True
                node.cond_exec_group_id = gid
                node.cond_exec_invert = invert
                group_ids.append(gid)
        else:
            gid = self._alloc_cerf_group()
            for node in guarded_nodes:
                node.cond_exec_en = True
                node.cond_exec_group_id = gid
                node.cond_exec_invert = invert
            group_ids = [gid]

        gating_node.cerf_write_groups = sorted(set(
            gating_node.cerf_write_groups + group_ids
        ))
        return group_ids
