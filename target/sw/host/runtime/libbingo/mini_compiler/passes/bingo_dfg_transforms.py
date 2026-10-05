# Fanchen Kong <fanchen.kong@kuleuven.be>

import networkx as nx

from bingo_node import BingoNode


class BingoDFGTransformsMixin:
    """Graph transforms and dependency assignment.

    These are the passes that turn a graph someone built by hand into one the hardware
    can run: the entry and exit nodes, the dummy set/check nodes that carry a dependency
    across a core or a die, the same-core sequencing edges, and the tag allocation that
    gives every concurrently-live edge its own presence bit.

    Mixed into BingoDFG, so `self` is the whole DFG. These methods call each other
    and the other mixins' methods freely; nothing here is meant to stand alone.
    """

    def bingo_transform_dfg_add_entry_node(self) -> None:
        """Transform the DFG to add one entry node."""
        # Find all the start nodes (nodes with no predecessors)
        # We do this BEFORE adding the entry node, otherwise the entry node (which has 0 predecessors initially)
        # will be included, leading to a self-loop when we connect entry_node -> start_nodes.
        start_nodes = [node for node in self.node_list if self.in_degree(node) == 0]

        # We will add one entry node at the beginning of the DFG
        if self.is_host_as_acc:
            entry_node = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
                assigned_core_id=self.num_cores_per_cluster -1,
                kernel_name="__host_bingo_kernel_entry"
            )
        else:
            entry_node = BingoNode(
                assigned_chiplet_id=0,
                assigned_cluster_id=0,
                assigned_core_id=0,
                kernel_name="__snax_bingo_kernel_entry"
            )
        self.bingo_add_node(entry_node)
        # Connect the entry node to all start nodes
        for start_node in start_nodes:
            self.bingo_add_edge(entry_node, start_node)

    def bingo_transform_dfg_add_exit_nodes(self) -> None:
        """Transform the DFG to add external nodes."""
        # Notice here the exit nodes are correlated with the hw architecture
        # Since we use the host core to do the simd ops,
        # We configure each cluster to have 2(gemm, dma) + 1(simd) accs 
        # But the host core only fetches the cluster0's simd ready queue
        # The other clusters' simd ready queue are not used
        # We need to add num_cluseter*(num_cores_per_cluster-1) [all the normal cores] + 1 [host core] exit nodes for each of the chiplets
        # We put those nodes at the end of user-specified nodes and put them in serials
        # first is all the normal cores and then finally the host core
        
        # We generate exit nodes for each chiplet in parallel
        print("Adding exit nodes for each chiplet...")
        for chiplet_id in self.chiplet_ids:
            # A chiplet is locally done when no successor remains on that same
            # chiplet; remote successors are completion signals for other chips.
            local_end_nodes = [
                node for node in self.node_list
                if node.assigned_chiplet_id == chiplet_id
                and not any(
                    succ.assigned_chiplet_id == chiplet_id
                    for succ in self.successors(node)
                )
            ]
            current_chiplet_exit_nodes = []
            
            # 1. Normal cores
            for cluster_id in range(self.num_clusters_per_chiplet):
                for core_id in range(self.num_cores_per_cluster):
                    # Skip the simd core for now
                    if self.is_host_as_acc and core_id == (self.num_cores_per_cluster -1):
                        continue
                    exit_node = BingoNode(
                        assigned_chiplet_id=chiplet_id,
                        assigned_cluster_id=cluster_id,
                        assigned_core_id=core_id,
                        kernel_name="__snax_bingo_kernel_exit"
                    )
                    self.bingo_add_node(exit_node)
                    current_chiplet_exit_nodes.append(exit_node)

            # 2. Host core
            exit_node_host = BingoNode(
                assigned_chiplet_id=chiplet_id,
                assigned_cluster_id=0,
                assigned_core_id=self.num_cores_per_cluster - 1,
                kernel_name="__host_bingo_kernel_exit"
            )
            self.bingo_add_node(exit_node_host)
            current_chiplet_exit_nodes.append(exit_node_host)

            # 3. Connect end nodes to the first exit node
            for end_node in local_end_nodes:
                self.bingo_add_edge(end_node, current_chiplet_exit_nodes[0])
            # 4. Chain the exit nodes within this chiplet
            for i in range(1, len(current_chiplet_exit_nodes)):
                self.bingo_add_edge(current_chiplet_exit_nodes[i-1], current_chiplet_exit_nodes[i])

    def bingo_transform_dfg_add_dummy_set_nodes(self) -> None:
        """Transform the DFG to add dummy nodes."""
        # The idea of the dummy set nodes is to solve the problem of this kind
        #            simd(Cl0)
        #           /         \
        #          |           |
        #          v           v
        #         dma(Cl0)    gemm(Cl1)
        # We need the dummy set task
        #            simd(Cl0)
        #           /         \\
        #          |           || <--  notice the double line here, it is a fake edge 
        #          |           ||      since we explicitly create the dummy task with the same type of the simd task
        #          v           vv      all we need to do is to push the dummy task after the simd task to describe this dependency
        #         dma(Cl0)    dummy dep set simd task(Cl1)
        #                      |
        #                      v
        #                    gemm(Cl1)
        for cur_node in self.node_list:
            # First find all the successors
            succs_list = [
                succ for succ in self.successors(cur_node)
            ]
            # For all the remote successors, we insert a dummy set node
            remote_succ_list = [
                succ for succ in succs_list
                if succ.assigned_chiplet_id != cur_node.assigned_chiplet_id
            ]
            local_succ_list = [
                succ for succ in succs_list
                if succ.assigned_chiplet_id == cur_node.assigned_chiplet_id
            ]
            if remote_succ_list:
                # Group remote successors by target cluster and core. A broadcast
                # dep-set has one (cluster, target_core, source_core) position
                # replicated across chiplets, so mixing clusters in one group
                # cannot be represented by a single dependency tag.
                remote_succs_by_cluster_core: dict[tuple[int, int], list[BingoNode]] = {}
                for remote_succ in remote_succ_list:
                    key = (remote_succ.assigned_cluster_id, remote_succ.assigned_core_id)
                    remote_succs_by_cluster_core.setdefault(key, []).append(remote_succ)

                for (_cluster_id, core_id), group in remote_succs_by_cluster_core.items():
                    chiplets_in_group = set(s.assigned_chiplet_id for s in group)
                    # A genuine broadcast covers ALL other chiplets AND there are at
                    # least two of them. The `num_chiplets > 2` guard is essential:
                    # at num_chiplets==2 a single point-to-point remote edge trivially
                    # "covers all (one) other chiplets" and would be mis-flagged as a
                    # broadcast -> the RTL multicasts (AW=0xFF) to EVERY chiplet,
                    # including the producer's own, leaving a stray (tagged) set with
                    # no consumer to drain it. Such a single edge must use a targeted
                    # dummy_set instead.
                    # A broadcast is a D2D multicast over the whole rectangle of chips, which
                    # every router between the compute chips must forward -- memory chips too
                    # (hemaia_d2d_link_initialize_grid programs them transit-only). Where a
                    # platform's routers do not (a fenced memory chip in C-M-C rows), the
                    # chips behind it never see it: dfg.remote_broadcast = False then sends
                    # one targeted dummy set per chip instead.
                    if (getattr(self, "remote_broadcast", True) and self.num_chiplets > 2
                            and len(chiplets_in_group) == (self.num_chiplets - 1)):
                        # Broadcast: one dummy_set blocks cur_node's core and sets the bit on all chiplets
                        print(f"Node {cur_node.node_name} is a broadcast node to set all chiplets for core {core_id}.")
                        dummy_set_node = BingoNode(
                            assigned_chiplet_id=cur_node.assigned_chiplet_id,
                            assigned_cluster_id=cur_node.assigned_cluster_id,      # must be the same type of the cur_node to block the execution
                            assigned_core_id=cur_node.assigned_core_id,            # must be the same type of the cur_node to block the execution
                            node_name=f"dummy_set_bcast_{cur_node.node_name}_co{core_id}",
                            kernel_name=None
                        )
                        dummy_set_node.node_type = "dummy"
                        dummy_set_node.dep_set_enable = True
                        dummy_set_node.dep_set_list = [group[0].assigned_core_id]
                        dummy_set_node.dep_set_cluster_id = group[0].assigned_cluster_id
                        dummy_set_node.dep_set_chiplet_id = group[0].assigned_chiplet_id # should be fine since it is a broadcast type
                        dummy_set_node.dep_check_enable = False
                        dummy_set_node.dep_check_list = []
                        dummy_set_node.remote_dep_set_all = True
                        # The gating task's remote edge is proxied through THIS dummy,
                        # so this is the message that must carry the CERF window.
                        dummy_set_node.cerf_carry = (cur_node.node_type == "gating")
                        # Add the dummy set node after cur_node for remote successors in this core group
                        self.bingo_insert_node_after(cur_node, dummy_set_node, group)
                    else:
                        # Normal case: one dummy_set per remote successor
                        for remote_succ in group:
                            print(f"Adding dummy set node for {cur_node.node_name} to remote successor {remote_succ.node_name}")
                            dummy_set_node = BingoNode(
                                assigned_chiplet_id=cur_node.assigned_chiplet_id,
                                assigned_cluster_id=cur_node.assigned_cluster_id,      # must be the same type of the cur_node to block the execution
                                assigned_core_id=cur_node.assigned_core_id,            # must be the same type of the cur_node to block the execution
                                node_name=f"dummy_set_{cur_node.node_name}_to_{remote_succ.node_name}",
                                kernel_name=None
                            )
                            dummy_set_node.node_type = "dummy"
                            dummy_set_node.dep_set_enable = True
                            dummy_set_node.dep_set_list = [remote_succ.assigned_core_id]
                            dummy_set_node.dep_set_cluster_id = remote_succ.assigned_cluster_id
                            dummy_set_node.dep_set_chiplet_id = remote_succ.assigned_chiplet_id
                            dummy_set_node.dep_check_enable = False
                            dummy_set_node.dep_check_list = []
                            dummy_set_node.remote_dep_set_all = False
                            # The gating task's remote edge is proxied through THIS dummy,
                            # so this is the message that must carry the CERF window.
                            dummy_set_node.cerf_carry = (cur_node.node_type == "gating")
                            # Add the dummy set node to the graph
                            self.bingo_insert_node_between(cur_node, remote_succ, dummy_set_node)
            if len(local_succ_list) > 1:
                # Now the local multiple successor case
                # We need local_successors-1 dummy set nodes
                print(f"Adding dummy set nodes for {cur_node.node_name} with local successors {[succ.node_name for succ in local_succ_list]}")

                # Prioritize edges where the successor node has the same assigned core as cur_node
                prioritized_indices = [i for i, succ in enumerate(local_succ_list)
                                      if succ.assigned_core_id == cur_node.assigned_core_id]
                other_indices = [i for i in range(len(local_succ_list)) if i not in prioritized_indices]
                # Combine prioritized first, then others
                ordered_indices = prioritized_indices + other_indices

                # Only need local_successors-1 dummy set nodes
                for idx in ordered_indices[:len(local_succ_list)-1]:
                    succ = local_succ_list[idx]
                    dummy_set_node = BingoNode(
                        assigned_chiplet_id=cur_node.assigned_chiplet_id,
                        assigned_cluster_id=cur_node.assigned_cluster_id,      # must be the same type of the cur_node to block the execution
                        assigned_core_id=cur_node.assigned_core_id,            # must be the same type of the cur_node to block the execution
                        node_name=f"dummy_set_{cur_node.node_name}_{idx}",
                        kernel_name=None
                    )
                    dummy_set_node.node_type = "dummy"
                    dummy_set_node.dep_set_enable = True
                    dummy_set_node.dep_set_list = [succ.assigned_core_id]
                    dummy_set_node.dep_set_cluster_id = succ.assigned_cluster_id
                    dummy_set_node.dep_set_chiplet_id = succ.assigned_chiplet_id
                    dummy_set_node.dep_check_enable = False
                    dummy_set_node.dep_check_list = []
                    dummy_set_node.remote_dep_set_all = False
                    # Add the dummy set node to the graph
                    self.bingo_insert_node_between(cur_node, succ, dummy_set_node)

    def bingo_transform_dfg_add_dummy_check_nodes(self) -> None:
        '''Transform the DFG to add dummy check nodes.

        Two cases require dummy_check insertion:

        Case 1 (same-core): A node has 2+ predecessors on the SAME core
        (different clusters). Both write to the same dep_matrix column.
        Insert dummy_checks to serialize consumption of that column.

        Case 2 (multi-core): A node has predecessors from 2+ DIFFERENT cores.
        Without dummy_checks, the node's dep_check_code would be a multi-bit
        mask (e.g., 0b110 for core 1 + core 2). This holds one column set
        while waiting for the other, creating a deadlock window when combined
        with the dep_matrix overlap detection and done queue HOL blocking.

        Solution: each dep_check (whether dummy or final normal task) must
        check exactly ONE core column. For N distinct predecessor cores,
        insert N-1 dummy_check nodes, each consuming one core's signal.
        The final normal task checks only the last remaining core.
        '''
        for cur_node in self.node_list:
            preds_list = [
                pred for pred in self.predecessors(cur_node)
            ]
            # Group predecessors by core_id
            predecessor_core_dict = {}
            for pred in preds_list:
                if pred.assigned_core_id not in predecessor_core_dict:
                    predecessor_core_dict[pred.assigned_core_id] = []
                predecessor_core_dict[pred.assigned_core_id].append(pred)

            # ---- Case 1: same-core groups with 2+ predecessors ----
            for core_id, preds in predecessor_core_dict.items():
                if len(preds) >= 2:
                    print(f"Adding dummy check nodes for {cur_node.node_name} "
                          f"with same-core predecessors {[p.node_name for p in preds]} (core {core_id})")
                    for i in range(len(preds) - 1):
                        dummy_check_node = BingoNode(
                            assigned_chiplet_id=cur_node.assigned_chiplet_id,
                            assigned_cluster_id=cur_node.assigned_cluster_id,
                            assigned_core_id=cur_node.assigned_core_id,
                            kernel_name=None
                        )
                        dummy_check_node.node_type = "dummy"
                        dummy_check_node.dep_check_enable = True
                        dummy_check_node.dep_check_list = [preds[i].assigned_core_id]
                        dummy_check_node.dep_set_enable = False
                        dummy_check_node.dep_set_list = []
                        dummy_check_node.dep_set_cluster_id = 0
                        dummy_check_node.dep_set_chiplet_id = 0
                        dummy_check_node.remote_dep_set_all = False
                        self.bingo_insert_node_between(preds[i], cur_node, dummy_check_node)

            # ---- Case 2: multi-core predecessors ----
            remaining_preds = [
                pred for pred in self.predecessors(cur_node)
                if not (pred.node_type == "dummy" and pred.dep_check_enable)
            ]
            remaining_core_ids = sorted(set(pred.assigned_core_id for pred in remaining_preds))

            if len(remaining_core_ids) >= 2 and self.enable_multi_col_check:
                # One multi-column dep_check instead of a dummy_check chain.
                pass
            elif len(remaining_core_ids) >= 2:
                cores_to_split = remaining_core_ids[:-1]
                for split_core in cores_to_split:
                    core_preds = [p for p in self.predecessors(cur_node)
                                  if p.assigned_core_id == split_core
                                  and not (p.node_type == "dummy" and p.dep_check_enable)]
                    if not core_preds:
                        continue
                    pred = core_preds[0]
                    print(f"Adding multi-core dummy check for {cur_node.node_name}: "
                          f"splitting {pred.node_name} (core {split_core})")
                    dummy_check_node = BingoNode(
                        assigned_chiplet_id=cur_node.assigned_chiplet_id,
                        assigned_cluster_id=cur_node.assigned_cluster_id,
                        assigned_core_id=cur_node.assigned_core_id,
                        kernel_name=None
                    )
                    dummy_check_node.node_type = "dummy"
                    dummy_check_node.dep_check_enable = True
                    dummy_check_node.dep_check_list = [split_core]
                    dummy_check_node.dep_set_enable = False
                    dummy_check_node.dep_set_list = []
                    dummy_check_node.dep_set_cluster_id = 0
                    dummy_check_node.dep_set_chiplet_id = 0
                    dummy_check_node.remote_dep_set_all = False
                    self.bingo_insert_node_between(pred, cur_node, dummy_check_node)

    def bingo_transform_add_core_sequencing_edges(self) -> int:
        """Add edges between consecutive tasks on the same core.

        Ensures deterministic execution order for tasks sharing a core,
        even when no explicit data dependency exists between them.
        Without these edges, the HW scheduler could dispatch same-core
        tasks in any topological order, leading to non-deterministic
        behavior and harder-to-debug timing.

        Algorithm:
          1. Topologically sort all nodes (respects existing dependencies).
          2. Group by (chiplet_id, cluster_id, core_id).
          3. Within each group, add an edge from node[i] to node[i+1]
             if no path already connects them (avoids redundant edges).

        Must be called AFTER entry/exit/conditional/dummy transforms
        (which insert infrastructure nodes on specific cores) and
        BEFORE dep info assignment.

        Returns:
            Number of sequencing edges added.
        """
        from collections import defaultdict

        # (with the ordering-only edges: a core's order must put every producer a spin waits
        # for ahead of the spinning consumer -- bingo_add_order_edge)
        topo_order = list(nx.topological_sort(self.bingo_order_view()))

        # Group nodes by their (chiplet, cluster, core) assignment
        core_groups: dict[tuple, list[BingoNode]] = defaultdict(list)
        for node in topo_order:
            key = (node.assigned_chiplet_id, node.assigned_cluster_id, node.assigned_core_id)
            core_groups[key].append(node)

        edges_added = 0
        for (chip, cl, core), nodes in core_groups.items():
            # nodes are already in topological order
            for i in range(len(nodes) - 1):
                prev_node = nodes[i]
                next_node = nodes[i + 1]
                # Skip if an edge (direct or transitive path) already exists
                if not self.has_edge(prev_node, next_node) and not nx.has_path(self, prev_node, next_node):
                    self.add_edge(prev_node, next_node)
                    edges_added += 1

        if edges_added > 0:
            print(f"Core sequencing: added {edges_added} edges across "
                  f"{len(core_groups)} core groups")
        return edges_added

    def bingo_transform_prune_redundant_fanout(self) -> int:
        """Drop the cross-core edges a core's own order already implies.

        After bingo_transform_add_core_sequencing_edges every core's tasks are one chain.
        So when a producer P feeds B1, B2, ... Bn on ANOTHER core of its chiplet, B2 .. Bn
        are reachable from B1 through that chain, and P -> B2 .. P -> Bn say nothing
        P -> B1 does not. Each of them would still cost a dummy set task on P's core and a
        dependency tag live in the (P core, B core) cell until its drain; a pass of four
        tokens fans each projection's last dequantisation out to 10-18 tasks of one DM core
        and runs that cell past the 32 tags of a 5-bit DepTagWidth. Only P -> B1, the first
        of them in the core's order, is kept.

        The mirror case too: a consumer B waiting on A1 .. An of ONE core (a gather's
        copies, one per source, all on the DM core) needs only the last of them, An; the
        core finishes its tasks in order, so An done means every Ai done. Both apply on the
        producer's own core as well, where the chain itself is the only edge kept.

        And both apply ACROSS CHIPS: a remote core's tasks are one chain too. There it is not
        only cheaper but required: a broadcast dummy set (one producer feeding the same
        (cluster, core) on every other chip) sets ONE tag bit per chip, so two consumers of
        it on one remote core would both wait on that bit and the second would never see it."""
        topo = {n: i for i, n in enumerate(nx.topological_sort(self))}
        removed = 0
        for b_ in list(self.nodes()):
            groups: dict = {}
            for a in self.predecessors(b_):
                key = (a.assigned_chiplet_id, a.assigned_cluster_id, a.assigned_core_id)
                groups.setdefault(key, []).append(a)
            for as_ in groups.values():
                if len(as_) < 2:
                    continue
                as_.sort(key=lambda n: topo[n])
                last = as_[-1]
                for a in as_[:-1]:
                    self.remove_edge(a, b_)
                    if not nx.has_path(self, a, last):      # the chain must still hold
                        self.add_edge(a, b_)
                        continue
                    removed += 1
        for p_ in list(self.nodes()):
            groups: dict = {}
            for b in self.successors(p_):
                key = (b.assigned_chiplet_id, b.assigned_cluster_id, b.assigned_core_id)
                groups.setdefault(key, []).append(b)
            for bs in groups.values():
                if len(bs) < 2:
                    continue
                bs.sort(key=lambda n: topo[n])
                first = bs[0]
                for b in bs[1:]:
                    self.remove_edge(p_, b)
                    if not nx.has_path(self, first, b):     # the chain must still hold
                        self.add_edge(p_, b)
                        continue
                    removed += 1
        if removed:
            print(f"Fan-out pruning: removed {removed} cross-core edges the core order implies")
        return removed

    def bingo_transform_prune_implied_edges(self) -> int:
        """Drop u -> v when another predecessor w of v is reachable from u.

        bingo_transform_prune_redundant_fanout drops what ONE core's chain implies. This is
        the general case: u -> ... -> w -> v already orders u before v whatever cores the
        path crosses, so the direct edge only costs a dependency tag live in (u's core,
        v's core) until v runs. Typical: a gather's collect waiting on two stashes of one
        chip that are themselves chained -- the second implies the first.

        Removing an implied edge never changes reachability, so the reachability computed
        once up front stays exact while edges go. Bitsets over a topological order: each
        node's set is its successors' sets OR-ed with their bits."""
        order = list(nx.topological_sort(self))
        idx = {n: i for i, n in enumerate(order)}
        reach = {}
        for n in reversed(order):
            r = 0
            for s in self.successors(n):
                r |= reach[s] | (1 << idx[s])
            reach[n] = r
        removed = 0
        for v in order:
            preds = list(self.predecessors(v))
            if len(preds) < 2:
                continue
            for u in preds:
                ru = reach[u]
                if any(w is not u and self.has_edge(w, v) and (ru >> idx[w]) & 1 for w in preds):
                    self.remove_edge(u, v)
                    removed += 1
        if removed:
            print(f"Implied-edge pruning: removed {removed} edges another path already orders")
        return removed

    def bingo_transform_keep_local_dep_set(self) -> int:
        """Give every task with a later task on its core a successor on its own chip.

        A task whose only successors are on other chips gets DepSet En=0, and the remote
        sets go on dummies after it. bingo_hw_manager pairs a task's checkout entry with its
        done-queue entry only on the LOCAL set path: an En=0 entry is routed by its (unused)
        dep_set_chiplet_id, which is 0, so on any chip but 0x00 it takes the chiplet path,
        leaves without waiting for the task's done and never pops it. From then on the
        core's next local set fires when its task is dispatched, not when it finishes, and
        the dummies behind it send their remote sets at once too. (A DM core's stash whose
        only consumer was a remote collect, once prune_implied had dropped its sequencing
        edge, released a quantize before the copy it read had landed: X in L1.)

        The edge added is to the next task on the same core, which runs after it anyway,
        so it orders nothing new; a graph that has no such task is left unchanged."""
        from collections import defaultdict

        core_tasks: dict = defaultdict(list)
        for n in nx.topological_sort(self):
            core_tasks[(n.assigned_chiplet_id, n.assigned_cluster_id,
                        n.assigned_core_id)].append(n)
        added = 0
        for (chip, _, _), tasks in core_tasks.items():
            for prev, nxt in zip(tasks, tasks[1:]):
                if any(s.assigned_chiplet_id == chip for s in self.successors(prev)):
                    continue
                self.add_edge(prev, nxt)
                added += 1
        if added:
            print(f"Local dep set: added {added} same-core edges for tasks with only remote "
                  f"successors")
        return added

    def bingo_transform_fit_dep_tag_budget(self, tag_width: int = None) -> int:
        """Make every dep-matrix cell fit in 2**tag_width tags, adding ordering edges where
        the graph would need more -- and relays where ordering cannot fit it.

        The allocator (bingo_transform_dfg_allocate_dep_tags) already uses the fewest tags
        a graph allows: two edges of a cell share a tag when one's check happens-before the
        other's set, so a cell needs as many tags as its largest set of edges that can all
        be live at once. That number is a property of the GRAPH, and when it is more than
        the hardware has, the graph has to change.

        ORDER first. An edge from the consumer of one live edge to the producer of another
        makes the second's set wait for the first's check, and the two can then share a tag.
        The producer may run later -- at most 2**tag_width of the cell's edges are in flight
        at once -- which is the trade a register allocator makes when it runs out of
        registers. Per over-budget cell, edges in the consumer core's order, greedily: an
        edge joins a chain whose last consumer already happens-before its producer; else
        opens a chain while there are fewer than 2**tag_width; else waits on the EARLIEST
        chain end it can without a cycle, through a new edge (order_only=True: no data flows
        on it). An added edge is itself a dependency in another cell, so this repeats until
        nothing is over budget.

        RELAYS when order cannot fit it. A collector waiting on producers on many chips
        holds as many live edges as there are producers, and the only order that lets them
        share tags makes a producer on one chip wait for a consumer on another. That edge
        lands in the producer chip's own cell, which a gather fills the same way, so the
        ordering spreads from chip to chip instead of converging, until a producer precedes
        every chain end and nothing can be ordered. When ordering alone raises, the edges it
        added are taken out and the fit runs again, keeping ordering inside one chip and
        relaying the rest (_relay_dep_edges): a relay is a no-op task on another core of the
        consumer's chip that waits for a group of the cell's producers and releases their
        first consumer -- the group's edges move to the relay core's cell, where there is
        room. A graph that fits by order alone never sees a relay.

        Runs on the real tasks, after core sequencing and fan-out pruning and before the
        dummy passes, so an edge or a relay added here is lowered like any other. A cell is
        what it will be after lowering: (consumer chiplet, consumer cluster, consumer core,
        producer core). The order here is weaker than after lowering -- a dummy check runs
        before its task, a dummy set after its producer -- so a cell that fits here fits in
        the allocator, which stays the final check.

        Returns the number of edges added. Raises when a cell cannot be fitted."""
        if tag_width is None:
            tag_width = self.dep_tag_width
        had = set(self.edges())
        try:
            return self._fit_dep_tag_budget(tag_width, relay=False)
        except ValueError as e:
            why = str(e).split("\n")[0]
            if "dep-tag budget" not in why:
                raise
            for u, v in [e_ for e_ in self.edges() if e_ not in had]:
                self.remove_edge(u, v)
            print(f"Tag budget: ordering alone cannot fit ({why[:200]}); fitting again with "
                  f"relays")
            return self._fit_dep_tag_budget(tag_width, relay=True)

    def _fit_dep_tag_budget(self, tag_width: int, relay: bool) -> int:
        """bingo_transform_fit_dep_tag_budget's fit: order only, or (relay) order inside one
        chip and relay the edges that would need order across chips."""
        budget = 1 << tag_width
        added_total, widest, relays_total, relayed_total = 0, 0, 0, 0
        import os as _os
        debug = bool(_os.environ.get("BINGO_TAG_DEBUG"))

        def cells_of_graph():
            cells: dict = {}
            for u, v in self.edges():
                key = (v.assigned_chiplet_id, v.assigned_cluster_id,
                       v.assigned_core_id, u.assigned_core_id)
                cells.setdefault(key, []).append((u, v))
            return cells

        for _round in range(16):
            cells = cells_of_graph()
            over = {k: es for k, es in cells.items() if len(es) > budget}
            if not over:
                break
            if debug:
                print(f"[tag-fit] round {_round}: {len(over)} cell(s) over {budget} edges: "
                      + ", ".join(f"{k}:{len(es)}" for k, es in sorted(over.items())[:12]))
            # Reachability as bitsets over a topological order: bit pos[b] of reach[a] is
            # set when a reaches b. NOT bingo_stream_order(): that one is cached, and a
            # cache taken before the dummy passes would hand the allocator and the emitter
            # an order without the dummies. Any topological order will do here -- each
            # core's tasks are already one chain, so every such order agrees on them.
            topo = list(nx.topological_sort(self))
            pos = {n: i for i, n in enumerate(topo)}
            reach: dict = {}
            for n in reversed(topo):
                r = 0
                for s in self.successors(n):
                    r |= reach[s] | (1 << pos[s])
                reach[n] = r

            def before(a, b):
                """a is b, or a happens-before b."""
                return a is b or bool((reach[a] >> pos[b]) & 1)

            def add_order(a, b):
                self.add_edge(a, b, order_only=True)
                gained = reach[b] | (1 << pos[b])
                for x in topo:
                    if x is a or (reach[x] >> pos[a]) & 1:
                        reach[x] |= gained

            def chain_cover(es):
                """The fewest chains (= tags) the cell needs as it stands."""
                n = len(es)
                g = nx.Graph()
                g.add_nodes_from(range(2 * n))
                for a in range(n):
                    for b in range(n):
                        if a != b and before(es[a][1], es[b][0]):
                            g.add_edge(a, n + b)
                m = nx.algorithms.bipartite.hopcroft_karp_matching(g, top_nodes=list(range(n)))
                return n - sum(1 for k in m if k < n)

            added = 0
            to_relay: dict = {}          # cell -> its edges that would need order across chips
            for key in sorted(over):
                es = sorted(over[key], key=lambda e: (pos[e[1]], pos[e[0]]))
                # A greedy cover is an upper bound; the exact one only when it is not enough.
                ends = []
                for p, c in es:
                    ok = [i for i, last in enumerate(ends) if before(last, p)]
                    if ok:
                        ends[max(ok, key=lambda i: pos[ends[i]])] = c
                    else:
                        ends.append(c)
                if len(ends) <= budget:
                    continue
                need = chain_cover(es)
                widest = max(widest, need)
                if need <= budget:
                    continue
                ends = []
                for p, c in es:
                    ok = [i for i, last in enumerate(ends) if before(last, p)]
                    if ok:
                        i = max(ok, key=lambda i: pos[ends[i]])
                    elif len(ends) < budget:
                        ends.append(c)
                        continue
                    else:
                        legal = [i for i, last in enumerate(ends) if not before(p, last)]
                        if relay and not (self.get_edge_data(p, c) or {}).get("cond"):
                            # order inside one chip only: across chips the new edge lands in
                            # another chip's cell and spreads; relay this edge instead
                            near = [i for i in legal
                                    if ends[i].assigned_chiplet_id == p.assigned_chiplet_id]
                            if not near:
                                to_relay.setdefault(key, []).append((p, c))
                                continue
                            legal = near
                        if not legal:
                            raise ValueError(
                                f"dep-tag budget: cell {key} (chiplet, cluster, consumer core, "
                                f"producer core) needs {need} tags and cannot be ordered into "
                                f"{budget}: {p.node_name} -> {c.node_name} precedes every chain "
                                f"end it would have to wait for. Widen DepTagWidth or split "
                                f"the fan-in in the workload.")
                        i = min(legal, key=lambda i: pos[ends[i]])
                        add_order(ends[i], p)
                        added += 1
                        if debug:
                            x = ends[i].assigned_chiplet_id != p.assigned_chiplet_id
                            print(f"[tag-fit]   cell {key} needs {need}: order "
                                  f"{ends[i].node_name} -> {p.node_name}"
                                  f"{' (across chips)' if x else ''}")
                    ends[i] = c
            if to_relay:
                made = self._relay_dep_edges(to_relay, budget, debug)
                relays_total += made
                relayed_total += sum(len(v) for v in to_relay.values())
            if not added and not to_relay:
                break
            added_total += added
        else:
            raise ValueError(f"dep-tag budget: cells still over {budget} tags after 16 rounds "
                             f"of added ordering edges{' and relays' if relay else ''}.")
        if added_total or relays_total:
            relayed = (f" and {relays_total} relays ({relayed_total} edges relayed)"
                       if relays_total else "")
            print(f"Tag budget: added {added_total} ordering edges{relayed} so every "
                  f"dep-matrix cell fits {budget} tags (the widest needed {widest})")
        self.tag_budget_edges_added = added_total
        self.tag_budget_relays = relays_total
        return added_total

    def _relay_dep_edges(self, to_relay: dict, budget: int, debug: bool = False) -> int:
        """Move each over-budget cell's leftover edges onto relays; returns the relays made.

        A relay is a normal task running __snax_bingo_kernel_sync_probe (no work, no
        print) on another core of the consumers' chip -- another core of their cluster, or a
        core of another cluster, never the host's, whose queue holds the weight prefetcher.
        A group of up to 2**tag_width / 2 of the cell's edges (p -> c, in the consumers' core
        order) becomes p -> relay for each, and relay -> c0, c0 the group's first consumer:
        the later ones follow c0 on their core, so each still runs after its producer. The
        group's edges now live in the relay core's cell, the least loaded one; c0 now also
        waits for the group's other producers, which is the price. A producer that c0
        already reaches starts a new group (it would close a cycle).

        The relay joins its core's chain just before the first task there that c0 reaches:
        everything behind it then waits for c0 anyway, so the relay holds no task back that
        did not already wait for its group. That keeps every core one chain, which the
        passes after this one rely on."""
        from bingo_kernel_args import SnaxBingoKernelSyncProbeArgs
        host = self.num_cores_per_cluster - 1 if self.is_host_as_acc else None
        topo = list(nx.topological_sort(self))
        pos = {n: i for i, n in enumerate(topo)}
        chain: dict = {}
        for n in topo:
            chain.setdefault((n.assigned_chiplet_id, n.assigned_cluster_id,
                              n.assigned_core_id), []).append(n)
        load: dict = {}
        for u, v in self.edges():
            k = (v.assigned_chiplet_id, v.assigned_cluster_id, v.assigned_core_id,
                 u.assigned_core_id)
            load[k] = load.get(k, 0) + 1
        group_max = max(1, budget // 2)
        made = 0
        for key in sorted(to_relay):
            chip, cl, R, C = key
            homes = [(c2, r2) for c2 in range(self.num_clusters_per_chiplet)
                     for r2 in range(self.num_cores_per_cluster)
                     if (c2, r2) != (cl, R) and r2 != host]
            if not homes:
                raise ValueError(f"dep-tag budget: cell {key} needs a relay, and its chip has "
                                 f"no other core to put one on.")
            groups, cur, first, below = [], [], None, set()
            for p, c in sorted(to_relay[key], key=lambda e: (pos[e[1]], pos[e[0]])):
                if cur and (len(cur) == group_max or p in below):
                    groups.append((first, cur))
                    cur = []
                if not cur:
                    first, below = c, nx.descendants(self, c)
                cur.append((p, c))
            if cur:
                groups.append((first, cur))
            for first, grp in groups:
                c2, r2 = min(homes, key=lambda h: (load.get((chip, h[0], h[1], C), 0), h))
                r = BingoNode(assigned_chiplet_id=chip, assigned_cluster_id=c2,
                              assigned_core_id=r2,
                              node_name=f"relay{made}_{first.node_name}",
                              kernel_name="__snax_bingo_kernel_sync_probe",
                              kernel_args=SnaxBingoKernelSyncProbeArgs())
                self.bingo_add_node(r)
                for p, c in grp:
                    data = dict(self.get_edge_data(p, c) or {})
                    self.remove_edge(p, c)
                    self.add_edge(p, r, **data)
                self.add_edge(r, first, order_only=True)
                below = nx.descendants(self, first)
                seq = chain.setdefault((chip, c2, r2), [])
                at = next((i for i, x in enumerate(seq) if x in below), len(seq))
                if at < len(seq):
                    self.add_edge(r, seq[at], order_only=True)
                if at > 0:
                    self.add_edge(seq[at - 1], r, order_only=True)
                seq.insert(at, r)
                load[(chip, c2, r2, C)] = load.get((chip, c2, r2, C), 0) + len(grp)
                made += 1
                if debug:
                    print(f"[tag-fit]   cell {key}: relay {r.node_name} on cluster {c2} core "
                          f"{r2} takes {len(grp)} edges")
        if not nx.is_directed_acyclic_graph(self):
            raise ValueError("dep-tag budget: a relay closed a cycle (a bug in "
                             "_relay_dep_edges).")
        return made

    def _split_broadcast_dep_set(self, b) -> None:
        """Lower broadcast dummy set `b` as one targeted dummy set per successor, as the
        dummy-set pass does with remote_broadcast off: each sits on b's producer's core,
        follows the producer, and sets one chip's cell. `b` becomes the first of them, so no
        node id is left unused. The producer's and the consumers' own set/check info do not
        change: the dep-info passes skip dummy sets, and every new set is on b's core."""
        (p,) = list(self.predecessors(b))
        succs = sorted(self.successors(b),
                       key=lambda s: (s.assigned_chiplet_id, s.node_id))
        for i, s in enumerate(succs):
            if i == 0:
                d = b
            else:
                d = BingoNode(assigned_chiplet_id=b.assigned_chiplet_id,
                              assigned_cluster_id=b.assigned_cluster_id,
                              assigned_core_id=b.assigned_core_id,
                              node_name=f"dummy_set_{p.node_name}_to_{s.node_name}",
                              kernel_name=None)
                d.node_type = "dummy"
                d.dep_check_enable = False
                d.dep_check_list = []
                d.cerf_carry = getattr(b, "cerf_carry", False)
                data = dict(self[b][s])
                self.bingo_add_node(d)
                self.remove_edge(b, s)
                self.add_edge(p, d)
                self.add_edge(d, s, **data)
            d.node_name = f"dummy_set_{p.node_name}_to_{s.node_name}"
            d.dep_set_enable = True
            d.dep_set_list = [s.assigned_core_id]
            d.dep_set_cluster_id = s.assigned_cluster_id
            d.dep_set_chiplet_id = s.assigned_chiplet_id
            d.remote_dep_set_all = False


    def bingo_assign_normal_node_dep_check_info(self) -> None:
        """Assign the dep check info for normal and gating nodes."""
        # Iterate over all nodes in the graph
        for cur_node in self.node_list:
            if cur_node.node_type in ("normal", "gating"):
                # Find predecessors
                # And not dummy check
                preds = [
                    pred for pred in self.predecessors(cur_node)
                    if not (pred.node_type == "dummy" and pred.dep_check_enable)
                ]
                # If there are local predecessors, assign dep_check info
                if preds:
                    cur_node.dep_check_enable = True
                    cur_node.dep_check_list = [pred.assigned_core_id for pred in preds]
                    # Sanity check if there are multiple same core_id
                    if len(cur_node.dep_check_list) != len(set(cur_node.dep_check_list)):
                        print(f"Warning: Multiple local predecessors with the same core_id for node {cur_node.node_id}. This is not expected, go back to DFG transformation stage!")
                    print(f"Assigned dep_check_info for node {cur_node.node_id}: "
                          f"dep_check_enable=True, dep_check_list={cur_node.dep_check_list}")
                else:
                    # If no local predecessors, disable dep_check
                    cur_node.dep_check_enable = False
                    cur_node.dep_check_list = []
                    print(f"No local predecessors for node {cur_node.node_id}. "
                          f"dep_check_enable=False")

    def bingo_assign_normal_node_dep_set_info(self) -> None:
        """Assign the dep set info for normal and gating nodes."""
        # Iterate over all nodes in the graph
        for cur_node in self.node_list:
           if cur_node.node_type in ("normal", "gating"):
                # Find succs
                # And not dummy set
                succs = [
                    succ for succ in self.successors(cur_node)
                    if not (succ.node_type == "dummy" and succ.dep_set_enable)
                ]
                if len(succs)>1:
                    print(f"Warning: More than one local successor for node {cur_node.node_name}. This is not expected, go back to DFG transformation stage!")
                elif len(succs)==1:
                    cur_node.dep_set_enable = True
                    cur_node.dep_set_list = [succ.assigned_core_id for succ in succs]
                    cur_node.remote_dep_set_all = False
                    cur_node.dep_set_chiplet_id = succs[0].assigned_chiplet_id
                    cur_node.dep_set_cluster_id = succs[0].assigned_cluster_id
                else:
                    cur_node.dep_set_enable = False
                    cur_node.dep_set_list = []
                    cur_node.remote_dep_set_all = False
                    cur_node.dep_set_cluster_id = 0
                    cur_node.dep_set_chiplet_id = 0

    def bingo_stream_order(self) -> list:
        """The ONE per-core order the manager will actually see, used by everything.

        The manager fetches the descriptor list in order and demuxes each entry into its
        assigned core's FIFO waiting queue, so this list IS the per-core execution order.
        Two things must agree on it:

          * the dep-tag allocator, whose min chain-cover decides which edges may SHARE a
            tag from a happens-before order that includes same-core sequencing;
          * this emitter.

        They are not independent. Emitting an order the allocator did not assume can leave
        two simultaneously-live edges holding one tag, and the run deadlocks. The converse
        holds as well: strip the tags and even the plain topological order deadlocks. The
        tags are what make a particular order safe, so the order and the tags have to be
        derived from the same sequence -- which is why this is computed once, here.

        The order is a PRIORITY topological sort: always a valid topological order, but
        among the currently-ready nodes it prefers the one anchored earliest, so a dummy
        lands next to the real task it serves. That placement matters because a dummy
        occupies a slot in the CONSUMER's waiting queue: a plain topological sort can put
        one belonging to a later task ahead of an earlier one, and the FIFO then makes the
        earlier task inherit a wait it has no dependency on.
        """
        if getattr(self, "_stream_order_cache", None) is not None:
            return self._stream_order_cache

        import heapq
        # the ordering-only edges count here too: a consumer that spins on a producer must
        # not be dispatched ahead of it (bingo_add_order_edge)
        g = self.bingo_order_view()
        topo_nodes = list(nx.topological_sort(g))
        pos = {n: i for i, n in enumerate(topo_nodes)}

        def _anchor(node):
            seen, cur, side = set(), node, 1
            while cur.node_type == "dummy" and cur.node_id not in seen:
                seen.add(cur.node_id)
                if cur.dep_check_enable:
                    nxt, side = list(self.successors(cur)), 0    # a check gates its consumer
                elif cur.dep_set_enable:
                    nxt, side = list(self.predecessors(cur)), 2  # a set follows its producer
                else:
                    break
                if not nxt:
                    break
                cur = min(nxt, key=lambda x: pos[x])
            return pos.get(cur, pos[node]), side

        def _key(node):
            if node.node_type == "dummy":
                a, side = _anchor(node)
                return (a, side, pos[node])
            return (pos[node], 1, 0)

        indeg = {n: g.in_degree(n) for n in g.nodes()}
        ready = [(_key(n), i, n) for i, n in enumerate(topo_nodes) if indeg[n] == 0]
        heapq.heapify(ready)
        seq, tie = [], len(topo_nodes)
        while ready:
            _, _, n = heapq.heappop(ready)
            seq.append(n)
            for succ in g.successors(n):
                indeg[succ] -= 1
                if indeg[succ] == 0:
                    heapq.heappush(ready, (_key(succ), tie, succ)); tie += 1
        assert len(seq) == len(topo_nodes), "priority topological sort dropped nodes"
        self._stream_order_cache = seq
        return seq

    def bingo_transform_dfg_allocate_dep_tags(self, tag_width: int = None) -> None:
        """Assign per-edge identity tags so a consumer drains only ITS
        producer's set, never a stray that happens to share the same
        dep-matrix cell.

        Must run LAST -- after the dummy-set / dummy-check passes and the
        dep-info assignment -- when every set/check operation is final. By then
        each dependency is a single DIRECT edge ``set_node -> check_node`` (the
        dummy passes split every fork/multi-producer into single-edge ops), so
        one ``dep_set_tag`` and one ``dep_check_tag`` per node suffice.

        Physical cell = ``(consumer_chiplet, consumer_cluster, R, C)`` with
        ``R = consumer core`` and ``C = producer core`` (the bare-core column, so
        cross-cluster / cross-chiplet producers fold onto the same cell -- and the
        tag is exactly what keeps them apart). Within a cell, two edges may share
        a tag iff a DFG happens-before path links one edge's CONSUMER to the
        other's PRODUCER (``has_path(consumerA, producerB)``): then B's set can
        only fire after A has dispatched and freed the tag, regardless of work
        delays. Tags are assigned by greedy coloring that exploits this reuse.

        ``tag_width`` is the fixed HW knob (``DepTagWidth``): a cell may hold at
        most ``2**tag_width`` concurrently-live edges. If coloring needs more we
        raise rather than silently reintroduce the aliasing bug -- the workload
        must reduce a cell's concurrency (co-locate / serialize the offending
        producers in placement) or ``DepTagWidth`` must be widened.
        """
        # Default to the DFG's configured DepTagWidth rather than a literal: the
        # descriptor reserves exactly that many bits per tag, so a hardcoded
        # default here is a second, silently disagreeing source of truth.
        if tag_width is None:
            tag_width = self.dep_tag_width
        max_tags = 1 << tag_width
        # MUST be the same order the emitter uses. Allocating tags against one
        # topological order while an emitter walks a DIFFERENT per-core order is
        # how two simultaneously-live edges end up sharing one tag -- the run then
        # deadlocks with no visible tag mismatch to catch it. The chain cover below
        # counts same-core sequencing as happens-before, so it is only sound for
        # THIS order. One order, derived once. See bingo_stream_order.
        topo = self.bingo_stream_order()
        pos = {n: i for i, n in enumerate(topo)}

        # 1. Collect dep-matrix set/check edges, grouped by physical cell.
        cells: dict = {}                      # (chip, cl, R, C) -> [(set_node, check_node), ...]
        pairs: list = []                      # (set_node, check_node, cell)
        for u, v in self.edges():
            if not (u.dep_set_enable and v.dep_check_enable):
                continue
            C, R = u.assigned_core_id, v.assigned_core_id
            if C not in v.dep_check_list or R not in u.dep_set_list:
                continue                      # u sets / v checks, but not THIS pair
            cell = (v.assigned_chiplet_id, v.assigned_cluster_id, R, C)
            cells.setdefault(cell, []).append((u, v))
            pairs.append((u, v, cell))

        # TAG GROUPS. A descriptor carries ONE dep_set_tag and ONE dep_check_tag,
        # and every set->check edge requires u.dep_set_tag == v.dep_check_tag. So
        # the tag is a property of the NODE, and any set/check ops linked by an
        # edge must agree: a "tag group" is a connected component of the
        # set-node <-> check-node bipartite graph. Today almost every group is a
        # single edge touching a single cell. A broadcast dep_set, or (once the
        # descriptor carries a mask) a multi-column join / multi-row fan-out, is
        # a group spanning several cells that must hold ONE tag in all of them.
        # DETERMINISM. Node keys are INTEGERS derived from node_id, and the
        # components are sorted. Keys containing a string (or a node object)
        # hash differently in every process -- Python randomises str hashing --
        # so component order, and therefore every tag, would change from one
        # compile to the next. Firmware has to be reproducible.
        bip = nx.Graph()
        skey = lambda n: 2 * n.node_id
        ckey = lambda n: 2 * n.node_id + 1
        for su, cv, _cell in pairs:
            bip.add_edge(skey(su), ckey(cv))
        gid = {}
        n_groups = 0
        for i, comp in enumerate(
                sorted((sorted(c) for c in nx.connected_components(bip)),
                       key=lambda c: c[0])):
            for key in comp:
                gid[key] = i
            n_groups = i + 1

        setters: dict = {}
        drainers: dict = {}
        cells_of: dict = {}
        edges_of: dict = {}
        for su, cv, cell in pairs:
            gi = gid[skey(su)]
            setters.setdefault(gi, set()).add(su)
            drainers.setdefault(gi, set()).add(cv)
            cells_of.setdefault(gi, set()).add(cell)
            edges_of.setdefault(gi, []).append((su, cv))

        # Same-core HOL reachability: at runtime each core dispatches its tasks in
        # topological (= push) order, so a same-core node that comes later is
        # effectively reachable from an earlier one. Add those consecutive same-core
        # edges to a SCRATCH graph used only for tag-reuse reachability (no real
        # edges, dep info unchanged). This collapses same-core (diagonal R==C) cells
        # -- which are serialized by the core queue -- to a chain, so they need few
        # tags instead of one per edge.
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

        def _descendants(n):
            if n not in _desc:
                _desc[n] = nx.descendants(hb, n)
            return _desc[n]

        def _precedes(a, b):
            """Group a may hand its tag on to b: every drain of a happens-before
            every set of b. They may meet at one node (a's consumer IS b's
            producer -- a node dispatches/drains before it completes/sets)."""
            for cv in drainers[a]:
                for su in setters[b]:
                    if cv is not su and su not in _descendants(cv):
                        return False
            return True

        # FAST PATH -- every group is one edge in one cell, which is what the
        # dummy passes guarantee today. Per cell the conflict graph is the
        # incomparability graph of a partial order, so by Dilworth the minimum
        # number of tags is the largest antichain, and a min chain cover
        # (bipartite matching) attains it EXACTLY. Keep using it: it is optimal,
        # and it is the path every existing graph takes, so nothing moves.
        single_edge = all(len(edges_of[g]) == 1 and len(cells_of[g]) == 1
                          for g in range(n_groups))
        if single_edge:
            for key, edges in cells.items():
                edges.sort(key=lambda e: (pos[e[0]], pos[e[1]]))
                n = len(edges)
                reach = [_descendants(cv) for (_su, cv) in edges]
                # Integer node labels and a sorted walk of the matching, for
                # the same reproducibility reason as above.
                B = nx.Graph()
                for a in range(n):
                    B.add_node(a); B.add_node(n + a)
                for a in range(n):
                    for b in range(n):
                        if a == b:
                            continue
                        if edges[b][0] is edges[a][1] or edges[b][0] in reach[a]:
                            B.add_edge(a, n + b)
                match = (nx.algorithms.bipartite.hopcroft_karp_matching(
                             B, top_nodes=list(range(n)))
                         if B.number_of_edges() else {})
                succ, has_pred = {}, set()
                for node in sorted(match):
                    if node < n:
                        succ[node] = match[node] - n; has_pred.add(match[node] - n)
                tag_of, n_chains = {}, 0
                for a in range(n):
                    if a in has_pred:
                        continue                       # not a chain head
                    cur = a
                    while True:
                        tag_of[cur] = n_chains
                        if cur in succ:
                            cur = succ[cur]
                        else:
                            break
                    n_chains += 1
                if n_chains > max_tags:
                    import os as _os
                    if _os.environ.get("BINGO_TAG_DEBUG"):
                        heads = [a for a in range(n) if a not in has_pred]
                        for a in heads:
                            print(f"[tag-debug] chain head {edges[a][0].node_name} -> "
                                  f"{edges[a][1].node_name}")
                    raise ValueError(
                        f"dep-tag allocation: cell {key} needs {n_chains} > {max_tags} "
                        f"concurrent tags (tag_width={tag_width}); reduce this cell's "
                        f"concurrency in placement or widen DepTagWidth.")
                for i, (su, cv) in enumerate(edges):
                    su.dep_set_tag = tag_of[i]
                    cv.dep_check_tag = tag_of[i]
            return

        # GENERAL PATH -- some group spans several edges or several cells (a
        # broadcast dep_set, or a multi-edge op). Tags are no longer independent
        # per cell: the group needs one tag free in EVERY cell it touches, so
        # this is a graph colouring. Conflict edges exist only between groups
        # that share a cell AND are incomparable, which is why two groups in
        # disjoint cells can still reuse the same tag.
        #
        # Merging edges into multi-edge ops tends to LOWER tag pressure rather
        # than raise it, because the merged edges share one tag instead of one
        # each. DSATUR is not provably optimal on this subgraph, so the capacity
        # check below stays.
        import itertools as _it
        groups_in_cell: dict = {}
        for gi, cs in cells_of.items():
            for cell in cs:
                groups_in_cell.setdefault(cell, []).append(gi)
        H = nx.Graph()
        H.add_nodes_from(range(n_groups))
        for cell in sorted(groups_in_cell):
            for a, b in _it.combinations(sorted(groups_in_cell[cell]), 2):
                if not _precedes(a, b) and not _precedes(b, a):
                    H.add_edge(a, b)
        colour = nx.coloring.greedy_color(H, strategy="DSATUR")
        n_tags = (max(colour.values()) + 1) if colour else 0
        H_cell = None
        if n_tags > max_tags:
            # PER-CELL ORDER. _precedes asks for every drain of a, in every cell, before
            # every set of b. A tag is a presence bit of ONE cell, though, so reuse only
            # has to be ordered where both groups are: in each cell they share, a's drains
            # IN THAT CELL before b's sets INTO it. A broadcast group spans every other
            # chip, and its drains there need not reach a later producer here: the
            # whole-group order makes it concurrent with everything a cell holds after
            # it, which the fit pass -- one edge per cell -- never counts. The hang check
            # tests reuse per cell, edge by edge, so this order is still sound. Tried only
            # when the whole-group colouring does not fit, so a graph that fits keeps its
            # tags.
            drains_in: dict = {}
            sets_in: dict = {}
            for su, cv, cell in pairs:
                gi = gid[skey(su)]
                drains_in.setdefault((gi, cell), set()).add(cv)
                sets_in.setdefault((gi, cell), set()).add(su)

            def _precedes_in(a, b, cell):
                for cv in drains_in[(a, cell)]:
                    for su in sets_in[(b, cell)]:
                        if cv is not su and su not in _descendants(cv):
                            return False
                return True

            H_cell = nx.Graph()
            H_cell.add_nodes_from(range(n_groups))
            for cell in sorted(groups_in_cell):
                for a, b in _it.combinations(sorted(groups_in_cell[cell]), 2):
                    if not _precedes_in(a, b, cell) and not _precedes_in(b, a, cell):
                        H_cell.add_edge(a, b)
            colour_cell = nx.coloring.greedy_color(H_cell, strategy="DSATUR")
            n_cell = (max(colour_cell.values()) + 1) if colour_cell else 0
            print(f"Tag allocation: ordering groups as a whole needs {n_tags} tags, "
                  f"ordering them per cell needs {n_cell} (of {max_tags})")
            if n_cell <= max_tags:
                H, colour, n_tags = H_cell, colour_cell, n_cell
            else:
                # SPLIT BROADCASTS. Each cell can fit on its own (the fit pass counts per
                # cell), but a broadcast dep-set is ONE node setting one tag in every other
                # chip's cell: its tag has to be free in all of them at once, so the cells
                # it sets are coloured together. Lower the broadcasts in the cells that
                # overflow as one targeted dummy set per chip -- what the dummy-set pass
                # emits with remote_broadcast off -- and allocate again. Every round leaves
                # fewer broadcasts, and a cell without one is coloured on its own.
                over = {cell for cell, gs in groups_in_cell.items()
                        if any(colour_cell[g] >= max_tags for g in gs)}
                bsets = sorted({su for cell in over for g in groups_in_cell[cell]
                                if len(cells_of[g]) > 1
                                for su in setters[g] if su.remote_dep_set_all},
                               key=lambda n: n.node_id)
                if bsets:
                    for b in bsets:
                        self._split_broadcast_dep_set(b)
                    self._stream_order_cache = None
                    self.tag_bcast_split = getattr(self, "tag_bcast_split", 0) + len(bsets)
                    print(f"Tag allocation: broadcast dep-sets tie the cells they set "
                          f"together; splitting {len(bsets)} into targeted sets and "
                          f"allocating again")
                    return self.bingo_transform_dfg_allocate_dep_tags(tag_width=tag_width)
        if n_tags > max_tags:
            import os as _os
            if _os.environ.get("BINGO_TAG_DEBUG"):
                # the cell whose groups are most CONCURRENT, not merely the most numerous:
                # its largest set of mutually unordered groups is what needs the tags
                best = (0, None, [])
                for cell, gs in groups_in_cell.items():
                    clq = max(nx.find_cliques(H.subgraph(gs)), key=len, default=[])
                    if len(clq) > best[0]:
                        best = (len(clq), cell, clq)
                print(f"[tag-debug] cell {best[1]}: {best[0]} mutually unordered groups")
                for gi in sorted(best[2]):
                    print(f"[tag-debug]   " + ", ".join(
                        f"{su.node_name}->{cv.node_name}" for su, cv in edges_of[gi][:2]))
                if H_cell is not None:
                    # ordered per cell: a clique here no colouring avoids; none here but
                    # too many colours means the groups shared across cells are the cause
                    clq, at = [], None
                    for cell, gs in groups_in_cell.items():
                        sub = H_cell.subgraph(gs).copy()
                        # only the conflicts of THIS cell: two groups may be unordered in
                        # another cell they share and ordered here
                        sub.remove_edges_from([(a, b) for a, b in list(sub.edges())
                                               if _precedes_in(a, b, cell)
                                               or _precedes_in(b, a, cell)])
                        c = max(nx.find_cliques(sub), key=len, default=[])
                        if len(c) > len(clq):
                            clq, at = c, cell
                    print(f"[tag-debug] ordered per cell, the largest clique in one cell "
                          f"has {len(clq)} groups, in cell {at}")
                    for gi in sorted(clq):
                        print(f"[tag-debug]   " + ", ".join(
                            f"{su.node_name}->{cv.node_name}"
                            for su, cv in edges_of[gi] if cv in drains_in[(gi, at)])[:300])
            busiest = max(groups_in_cell.items(),
                          key=lambda kv: len(kv[1]))
            detail = "\n".join(
                f"    group {g}: " + ", ".join(
                    f"{su.node_name}->{cv.node_name}" for su, cv in edges_of[g][:3])
                + (f" (+{len(edges_of[g]) - 3} more)" if len(edges_of[g]) > 3 else "")
                for g in sorted(busiest[1])[:8])
            raise ValueError(
                f"dep-tag allocation: needs {n_tags} > {max_tags} tags "
                f"(tag_width={tag_width}).\n"
                f"  cell = (chiplet, cluster, consumer core, producer core)\n"
                f"  busiest cell {busiest[0]} holds {len(busiest[1])} groups:\n{detail}\n"
                f"  Fix by serializing those producers against each other (an edge "
                f"between them lets two groups share a tag), or widen DepTagWidth "
                f"-- the descriptor carries TWO tags, so each step up costs 2 "
                f"bits and it must still fit task_desc_width "
                f"({self.task_desc_width} bits here). Raise "
                f"cfg s1_quadrant.dep_tag_width to widen it.")
        for su, cv, _cell in pairs:
            t = colour[gid[skey(su)]]
            su.dep_set_tag = t
            cv.dep_check_tag = t
