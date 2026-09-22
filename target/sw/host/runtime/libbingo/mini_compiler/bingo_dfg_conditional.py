# Fanchen Kong <fanchen.kong@kuleuven.be>

import networkx as nx

from bingo_kernel_args import BINGO_GATING_MODE_STATIC
from bingo_kernel_args import BINGO_GATING_MODE_THRESHOLD
from bingo_kernel_args import BINGO_GATING_MODE_TOP_K
from bingo_kernel_args import HostBingoKernelCerfGatingArgs
from bingo_mem_handle import BingoMemAlloc
from bingo_node import BingoNode


COMBINE_KINDS = ("weighted_sum", "select", "sum")


class BingoConditionalBranch:
    """One arm of a fork: the tasks that run together, or not at all.

    `group_key` is the identity two branches share when one is declared with
    `group=` the other; it becomes one CERF group at compile time. By default a
    branch is its own key, which is what gives MoE one group per expert.
    """

    def __init__(self, fork, nodes, invert: bool, group_key, index: int):
        self.fork = fork
        self.nodes = list(nodes)
        self.invert = invert
        self.group_key = group_key
        self.index = index

    def __repr__(self):
        return (f"BingoConditionalBranch(#{self.index}, {len(self.nodes)} nodes, "
                f"invert={self.invert})")


class BingoConditionalFork:
    """A declared branch point: who decides, what is guarded, where it rejoins.

    The four facts a conditional region needs are stated once each -- the gating
    node and its policy here, the guarded tasks in `branch()`, the reconvergence
    in `combine()` -- instead of being spread over every edge. That is what lets
    the compiler refuse the shapes that would otherwise produce a silently wrong
    answer, and what lets the combine be declared rather than hand-built.

    Nothing is lowered at declaration time beyond the conditional edges, so the
    graph a fork produces is the same graph the `cond_dic` edge form produces and
    both go down one path.
    """

    def __init__(self, dfg, gating_node, select: dict):
        if not isinstance(select, dict) or "mode" not in select:
            raise ValueError(
                "bingo_conditional_fork(select=...) wants the gating policy dict, "
                "e.g. {'mode': 'top_k', 'k': 2}. See docs/conditional_dfg_api.md.")
        self.dfg = dfg
        self.source = gating_node
        self.select = dict(select)
        self.branches: list[BingoConditionalBranch] = []
        self.combine_node = None
        self.combine_inputs: list = []
        self.combine_kind = None
        self.combine_weights = None
        self._selection = None          # (weights_view, activation_view), lazy

    def branch(self, nodes, invert: bool = False, group=None):
        """Guard `nodes` on this fork's decision.

        Adds the conditional edge from the gating node to each of them, which is
        the representation the rest of the compiler already understands. Tasks
        joined by ordinary edges inside `nodes` end up in one CERF group anyway;
        `group=` another branch merges two arms that are NOT connected.
        """
        if isinstance(nodes, (list, tuple, set)):
            nodes = list(nodes)
        else:
            nodes = [nodes]
        if not nodes:
            raise ValueError(
                f"fork on '{self.source.node_name}': branch() got no nodes. An "
                f"empty branch guards nothing and would allocate a CERF group "
                f"that never gates anything.")
        if group is not None and not isinstance(group, BingoConditionalBranch):
            raise ValueError(
                f"fork on '{self.source.node_name}': group= takes another branch "
                f"(the value branch() returned), not {type(group).__name__}.")

        br = BingoConditionalBranch(
            self, nodes, invert,
            group.group_key if group is not None else object(),
            len(self.branches))
        self.branches.append(br)
        for n in nodes:
            self.dfg.bingo_add_edge(self.source, n, cond=True)
        return br

    def _selection_record(self):
        """The gate's decision buffer, allocated on first use.

        Lazy because it is sized by the number of BRANCHES, which is not known
        when the fork is declared -- but it has to exist before the workload can
        write `weights=fork.weights` in the combine call, which happens before
        the compiler runs. First touch after the last branch() is therefore the
        only moment both are true, and the compiler re-checks the size against
        the final branch count in case something was declared afterwards.
        """
        if self._selection is None:
            if not self.branches:
                raise ValueError(
                    f"fork on '{self.source.node_name}': fork.weights and "
                    f"fork.activation are sized by the branch count, so they "
                    f"cannot be named before the first branch() call.")
            self._selection = self.dfg._alloc_selection_record(
                self.source, len(self.branches))
        return self._selection

    @property
    def weights(self):
        """float[branches], renormalised over the winners. 0.0f for the losers."""
        return self._selection_record()[0]

    @property
    def activation(self):
        """uint8_t[branches], 1 for a branch the gate selected."""
        return self._selection_record()[1]

    def combine(self, node, inputs, kind: str, weights=None):
        """Declare where the branches reconverge.

        A skipped task still fires its dependency -- the manager pushes it to the
        checkout queue retagged as a dummy (bingo_hw_manager_top.sv:1149-1155) --
        so this is an ordinary fan-in, and the combine runs once with whichever
        branches actually produced a value.

        The declaration is what the validator checks against: an unconditional
        edge out of a guarded task is a potentially unguarded read unless it
        lands here.
        """
        if kind not in COMBINE_KINDS:
            raise ValueError(
                f"fork on '{self.source.node_name}': combine kind '{kind}' is not "
                f"one of {COMBINE_KINDS}.")
        if kind == "weighted_sum" and weights is None:
            raise ValueError(
                f"fork on '{self.source.node_name}': combine(kind='weighted_sum') "
                f"needs weights=. Pass fork.weights for the router's own "
                f"renormalised probabilities. Without them the output is scaled "
                f"by the mass of the branches that did not run.")
        if self.combine_node is not None:
            raise ValueError(
                f"fork on '{self.source.node_name}' already has a combine "
                f"('{self.combine_node.node_name}'). A fork reconverges once.")
        self.combine_node = node
        self.combine_inputs = list(inputs)
        self.combine_kind = kind
        self.combine_weights = weights
        self.dfg._declared_combines[node] = self
        return node

    def __repr__(self):
        return (f"BingoConditionalFork('{self.source.node_name}', "
                f"{self.select.get('mode')}, {len(self.branches)} branches)")


class BingoDFGConditionalMixin:
    """Conditional regions and their CERF groups.

    A gating task picks which of its guarded successors may run; the CERF group is the
    window that decision is published in, and the allocation here is what keeps two
    regions from sharing one.

    Mixed into BingoDFG, so `self` is the whole DFG. These methods call each other
    and the other mixins' methods freely; nothing here is meant to stand alone.
    """

    def bingo_conditional_fork(self, gating_node, select: dict) -> "BingoConditionalFork":
        """Declare a branch point. See BingoConditionalFork and
        docs/conditional_dfg_api.md.

            fork = dfg.bingo_conditional_fork(router, {'mode': 'top_k', 'k': 2})
            for e in range(E):
                fork.branch(expert_lane[e])
            fork.combine(out, inputs=y, kind='weighted_sum', weights=fork.weights)
        """
        fork = BingoConditionalFork(self, gating_node, select)
        self._cond_forks.append(fork)
        return fork

    def _fork_of(self, node):
        """The fork declared on `node`, either as its source or as the gating
        node spliced in front of it. None if `node` declares no fork."""
        for f in getattr(self, "_cond_forks", []):
            if f.source is node or getattr(f, "_gating_node", None) is node:
                return f
        return None

    def _branch_of(self, node):
        """The branch that guards `node`, across every fork. None if untouched."""
        for f in getattr(self, "_cond_forks", []):
            for br in f.branches:
                if node in br.nodes:
                    return br
        return None

    def _alloc_cerf_group(self, hint: str = "") -> int:
        """Allocate the next CERF group ID. Raises on overflow (>31)."""
        gid = self._next_cerf_group
        self._next_cerf_group += 1
        if gid >= 32:
            raise ValueError(
                f"CERF group overflow: need group {gid} but max is 31 "
                f"(WF4 violated). {hint}")
        return gid

    # ------------------------------------------------------------------
    # Lowering, in the order bingo_compile_conditional_regions runs them
    # ------------------------------------------------------------------

    def _collect_conditional_declarations(self) -> dict:
        """Find every gating node and the targets it guards.

        Also validates core assignments, because an unassigned node here means
        ``bingo_auto_assign()`` was never called and everything downstream would
        compute groups for nodes that have no core to skip on.

        Returns a dict mapping each gating node to its set of conditional targets.
        Empty when the graph declares no conditional edges at all.
        """
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

        gating_to_targets: dict[BingoNode, set[BingoNode]] = {}
        for u, v, data in self.edges(data=True):
            if data.get("cond", False):
                gating_to_targets.setdefault(u, set()).add(v)

        if not gating_to_targets:
            return {}

        # A CERF-gated node must not be the source of new cond edges: the compiler
        # would insert a gating node on a CERF-skippable cluster, and when that
        # cluster is inactive the gating node cannot run, deadlocking everything
        # downstream of it.
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

        return gating_to_targets

    def _cond_dic_of(self, source_node) -> dict:
        """The gating policy that governs `source_node`'s conditional out-edges.

        A declared fork owns the policy, so it wins; otherwise all conditional
        edges out of one source share one, and the first found is the answer.
        Empty dict means the legacy promote-in-place path.

        Refuses a `cond_dic` that disagrees with a fork on the same gate -- the
        old code silently took whichever it processed first.
        """
        fork = self._fork_of(source_node)
        edge_dic = {}
        for _, _, data in self.out_edges(source_node, data=True):
            if data.get('cond', False) and data.get('cond_dic'):
                edge_dic = data['cond_dic']
                break
        if fork is not None:
            if edge_dic and edge_dic != fork.select:
                raise ValueError(
                    f"Gate '{source_node.node_name}' is declared twice and the "
                    f"two disagree: the fork says {fork.select} but an edge says "
                    f"{edge_dic}. Drop the cond_dic and keep the fork.")
            return fork.select
        return edge_dic

    def _insert_gating_node(self, source_node) -> BingoNode:
        """Splice a `__gating_<source>` node between a source and its cond targets.

        The gating node lands on the source's own core, which is the one place it is
        guaranteed not to be CERF-skippable itself.
        """
        gating_node = BingoNode(
            source_node.assigned_chiplet_id,
            source_node.assigned_cluster_id,
            source_node.assigned_core_id,
            node_name=f"__gating_{source_node.node_name}",
        )
        cond_succs = [v for v in self.successors(source_node)
                      if self[source_node][v].get('cond', False)]
        self.bingo_insert_node_after(source_node, gating_node,
                                     successors_to_move=cond_succs)
        return gating_node

    def _check_well_formed(self, gating_to_targets: dict) -> None:
        """WF1/WF2/WF5: acyclic, one gate per target, and every gate an ancestor."""
        if not nx.is_directed_acyclic_graph(self):
            raise ValueError(
                "Conditional DFG is not a DAG — it contains a cycle. "
                "Well-formedness condition WF1 violated."
            )

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

        for gating_node, targets in gating_to_targets.items():
            for t in targets:
                if not nx.has_path(self, gating_node, t):
                    raise ValueError(
                        f"Gating node '{gating_node.node_name}' is not an "
                        f"ancestor of conditional target '{t.node_name}'. "
                        f"Well-formedness condition WF5 violated."
                    )

    def _assign_cerf_groups(self, gating_to_targets: dict) -> dict:
        """One CERF group per independent component of each gate's target set.

        CERF group reuse: if all gating nodes are totally ordered (each an ancestor
        of the next), their targets execute at different times and can safely share
        group IDs. The clear-before-set protocol in the gating task overwrites the
        stale values from the previous layer.

        Returns the node -> group map, and records `_gating_to_targets` on the DFG
        so the cross-die carry check and the cycle model can see the gates.
        """
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

            if reuse_groups:
                self._next_cerf_group = pool_start

            components = self._cond_components(gating_node, targets)

            # If more components than the 32 CERF groups allow, share groups:
            # the HW skips at group level and the SW guard filters within an
            # active group.
            num_components = len(components)
            max_cerf = 32 - self._next_cerf_group
            if max_cerf <= 0:
                max_cerf = 32  # will overflow, _alloc_cerf_group raises

            if num_components <= max_cerf:
                hint = ("Sequential gating reuse is active — "
                        "too many experts per layer."
                        if reuse_groups else
                        "Consider reducing experts or making "
                        "gating nodes sequential for reuse.")
                gids = [self._alloc_cerf_group(hint) for _ in components]
            else:
                n_groups = min(max_cerf, 32)
                allocated = [self._alloc_cerf_group(
                    f"Sharing mode: {num_components} components across {n_groups} groups"
                ) for _ in range(n_groups)]
                gids = [allocated[i % n_groups] for i in range(num_components)]
                print(f"CERF group sharing: {num_components} components "
                      f"mapped to {n_groups} groups "
                      f"({num_components/n_groups:.1f} components/group)")

            group_ids = []
            for idx, (component, gid) in enumerate(zip(components, gids)):
                for node in component:
                    node.cond_exec_en = True
                    node.cond_exec_group_id = gid
                    node.cond_exec_invert = self._cond_invert_of(node)
                    # The SW guard and the router's score vector are indexed by
                    # BRANCH, not by CERF group. They coincide unless two branches
                    # were merged with group=, and then indexing by group would
                    # have every merged expert read one another's activation slot.
                    br = self._branch_of(node)
                    node._cond_node_index = br.index if br is not None else idx
                    node_to_group[node] = gid
                if gid not in group_ids:
                    group_ids.append(gid)

            gating_node.cerf_write_groups = sorted(set(
                gating_node.cerf_write_groups + group_ids
            ))
            # The SW guard needs to reach the gating node's scratchpad from the
            # guarded kernel; see bingo_dfg_emit wiring gating_sp_addr.
            for target in targets:
                target._gating_node = gating_node

        self._gating_to_targets = gating_to_targets
        self._node_to_cerf_group = node_to_group
        return node_to_group

    def _cond_components(self, gating_node, targets: set) -> list:
        """Group a gate's targets into branches, one CERF group each.

        A declared fork says which arm each target is on, including two arms
        merged with `group=` that no edge connects. Otherwise the grouping is
        derived: two targets joined by an UNCONDITIONAL edge run or skip
        together, so they are one branch.

        Sorted by lowest node_id so expert_i lands on the same group across layers
        that reuse the pool -- otherwise the mapping shifts run to run.
        """
        fork = self._fork_of(gating_node)
        if fork is not None and fork.branches:
            by_key: dict = {}
            for br in fork.branches:
                by_key.setdefault(br.group_key, set()).update(
                    n for n in br.nodes if n in targets)
            comps = [c for c in by_key.values() if c]
            stray = targets - set().union(*comps) if comps else set(targets)
            if stray:
                raise ValueError(
                    f"Gate '{gating_node.node_name}' guards "
                    f"{sorted(n.node_name for n in stray)}, which no branch of "
                    f"its fork declares. Every conditional target must be in a "
                    f"branch, or its CERF group is nobody's.")
            return sorted(comps, key=lambda c: min(n.node_id for n in c))

        unc = nx.Graph()
        unc.add_nodes_from(targets)
        for t in targets:
            for _, v, d in self.out_edges(t, data=True):
                if v in targets and not d.get("cond", False):
                    unc.add_edge(t, v)
            for u, _, d in self.in_edges(t, data=True):
                if u in targets and not d.get("cond", False):
                    unc.add_edge(u, t)
        return sorted(nx.connected_components(unc),
                      key=lambda c: min(n.node_id for n in c))

    def _cond_invert_of(self, node) -> bool:
        """Whether this target runs when its group is INACTIVE. Only a declared
        fork branch can ask for that; the edge form is always False."""
        br = self._branch_of(node)
        return bool(br.invert) if br is not None else False

    def _alloc_selection_record(self, source_node, num_branches: int):
        """The gate's decision, as ONE L3 record: float weights then uint8 flags.

        Both halves are written by the same gating kernel and read by the same
        combine, so they are views into one allocation rather than two -- a view
        emits `ptr_<base> + offset` and costs no second BingoMemAlloc.

            [0 .. 4E)          float   weight[E]      renormalised over the winners
            [4E .. 4E+E)       uint8_t activation[E]  1 = selected

        Returns (weights_view, activation_view).
        """
        flags = (num_branches + 3) & ~3         # keep the record 4-byte aligned
        rec = BingoMemAlloc(
            f"__cond_sel_{source_node.node_name}",
            4 * num_branches + flags, "L3",
            chip_id=source_node.assigned_chiplet_id, cluster_id=0)
        return rec.view(0), rec.view(4 * num_branches)

    def _build_gating_kernel_args(self, gating_node, source_node, cond_dic,
                                  num_branches: int) -> None:
        """Fill in the auto-inserted gating node's kernel and arguments.

        All modes use the unified __host_bingo_kernel_cerf_gating kernel; the mode
        field and the payload in `top_k_or_threshold` are what separate them.
        """
        cerf_controlled_mask = sum(1 << g for g in gating_node.cerf_write_groups)
        mode_str = cond_dic.get('mode', 'static')

        gating_node.kernel_name = "__host_bingo_kernel_cerf_gating"

        if mode_str == 'top_k':
            cerf_gids_alloc = BingoMemAlloc(
                f"__cerf_gids_{source_node.node_name}",
                num_branches, "L3",
                chip_id=source_node.assigned_chiplet_id, cluster_id=0)
            fork = self._fork_of(source_node)
            if fork is not None:
                weights, activation = fork._selection_record()
                if activation.offset != 4 * num_branches:
                    raise ValueError(
                        f"Fork on '{source_node.node_name}' allocated its "
                        f"selection record for {activation.offset // 4} branch(es) "
                        f"but now has {num_branches}. A branch was declared after "
                        f"fork.weights or fork.activation was first named; move "
                        f"every branch() call before the combine.")
            else:
                weights, activation = self._alloc_selection_record(
                    source_node, num_branches)
            gating_node.kernel_args = HostBingoKernelCerfGatingArgs(
                mode=BINGO_GATING_MODE_TOP_K,
                cerf_controlled_mask=cerf_controlled_mask,
                top_k_or_threshold=cond_dic['k'],
                cerf_group_ids_addr=cerf_gids_alloc,
                cond_activation_addr=activation,
                cond_weight_addr=weights,
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

        # Expert -> CERF group mapping, for cerf_group_ids initialization. The
        # gating kernel indexes this by the router's score index, so it is keyed
        # by BRANCH: with merged groups two experts legitimately map to one group,
        # and enumerate(sorted(groups)) would silently renumber them.
        fork = self._fork_of(source_node)
        if fork is not None and fork.branches:
            by_branch = {}
            for br in fork.branches:
                gid = next((n.cond_exec_group_id for n in br.nodes
                            if n.cond_exec_en), None)
                if gid is not None:
                    by_branch[br.index] = gid
            self._gating_cerf_mappings[gating_node] = by_branch
        else:
            self._gating_cerf_mappings[gating_node] = {
                i: gid for i, gid in enumerate(sorted(gating_node.cerf_write_groups))
            }

    def _lower_combines(self) -> None:
        """Wire and check every declared reconvergence.

        Runs once the graph is complete, because the checks are about paths: a
        branch that never reaches its combine is the failure this exists to name,
        and it cannot be seen while the branch is still being built.
        """
        for fork in getattr(self, "_cond_forks", []):
            node = fork.combine_node
            if node is None:
                continue
            gate = getattr(fork, "_gating_node", fork.source)

            # Rule 5 (cascade): an inner gate must be reachable from the outer
            # one, or its own decision is made against a predicate that may never
            # have been published. Acyclicity itself is WF1, checked earlier.
            for other in getattr(self, "_cond_forks", []):
                if other is fork:
                    continue
                if self._branch_of(other.source) is not None:
                    inner = getattr(other, "_gating_node", other.source)
                    outer = getattr(fork, "_gating_node", fork.source)
                    if (self._branch_of(other.source).fork is fork
                            and not nx.has_path(self, outer, inner)):
                        raise ValueError(
                            f"Cascaded gate '{other.source.node_name}' is guarded "
                            f"by the fork on '{fork.source.node_name}' but is not "
                            f"reachable from it, so it would decide against a "
                            f"stale predicate.")

            # Rule 5 (missing edges): the workload names the fan-in once, in
            # inputs=. Anything already connected is left alone.
            for producer in fork.combine_inputs:
                if producer is node:
                    continue
                if not nx.has_path(self, producer, node):
                    self.bingo_add_edge(producer, node)

            # Rule 1: every branch must actually reach the combine. A branch that
            # does not is silently dropped from the result.
            for br in fork.branches:
                if not any(n is node or nx.has_path(self, n, node)
                           for n in br.nodes):
                    raise ValueError(
                        f"Branch #{br.index} of the fork on "
                        f"'{fork.source.node_name}' "
                        f"({', '.join(n.node_name for n in br.nodes[:3])}...) "
                        f"never reaches its combine '{node.node_name}'. Its "
                        f"result would be computed and then dropped. Add it to "
                        f"inputs=, or take those tasks out of the branch.")

            # Rule 3: 'select' picks THE branch that ran, so more than one input
            # existing at run time makes the merge ambiguous. Two branches sharing
            # a group with neither inverted always run together.
            if fork.combine_kind == "select":
                by_key: dict = {}
                for br in fork.branches:
                    by_key.setdefault(br.group_key, []).append(br)
                for key, brs in by_key.items():
                    if len(brs) > 1 and not any(b.invert for b in brs):
                        raise ValueError(
                            f"Fork on '{fork.source.node_name}': branches "
                            f"{[b.index for b in brs]} share one CERF group with "
                            f"none inverted, so they all run together, but the "
                            f"combine is kind='select' and can take only one "
                            f"input. Invert one of them, or use 'sum'.")

            # Bind what the gate decided to the combine's arguments, without the
            # compiler needing to know the kernel: a kernel args class opts in by
            # exposing bind_combine().
            binder = getattr(node.kernel_args, "bind_combine", None)
            if binder is not None:
                binder(activation=fork.activation,
                       weights=(fork.combine_weights
                                if fork.combine_weights is not None
                                else fork.weights),
                       num_inputs=len(fork.branches))

    def bingo_compile_conditional_regions(self) -> dict:
        """Compile conditional edges into CERF group assignments.

        Scans every edge for the ``cond`` attribute set by
        ``bingo_add_edge(..., cond=True)``, splices in a gating node per source
        that declared a policy, allocates CERF groups per independent branch, and
        fills in the gating kernel's arguments.

        Must be called **before** the dummy-node transforms.

        Returns:
            dict mapping each conditionally-gated BingoNode to its CERF group id.
            Also stored in ``self._node_to_cerf_group``.
        """
        gating_to_targets = self._collect_conditional_declarations()
        if not gating_to_targets:
            self._node_to_cerf_group = {}
            return {}

        # Splice a gating node in front of every source that declared a policy.
        # A source without one keeps the legacy promote-in-place behaviour.
        inserted: dict = {}                      # gating_node -> (source, cond_dic)
        for source_node in list(gating_to_targets.keys()):
            cond_dic = self._cond_dic_of(source_node)
            if not cond_dic:
                continue
            gating_node = self._insert_gating_node(source_node)
            gating_to_targets[gating_node] = gating_to_targets.pop(source_node)
            inserted[gating_node] = (source_node, cond_dic)
            fork = self._fork_of(source_node)
            if fork is not None:
                fork._gating_node = gating_node

        self._check_well_formed(gating_to_targets)
        node_to_group = self._assign_cerf_groups(gating_to_targets)

        for gating_node, (source_node, cond_dic) in inserted.items():
            # The per-expert arrays are indexed by BRANCH, not by guarded task:
            # _cond_node_index is the component ordinal, and the router publishes
            # one score per branch. Sizing them by the target count instead --
            # four tasks per expert made it four times too long -- was harmless
            # only because nothing read past the end.
            targets = gating_to_targets[gating_node]
            fork = self._fork_of(source_node)
            num_branches = (len(fork.branches) if fork is not None and fork.branches
                            else 1 + max((t._cond_node_index for t in targets
                                          if t._cond_node_index is not None),
                                         default=-1))
            self._build_gating_kernel_args(
                gating_node, source_node, cond_dic, num_branches)

        self._lower_combines()
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
