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
                    if self.num_chiplets > 2 and len(chiplets_in_group) == (self.num_chiplets - 1):
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

        topo_order = list(nx.topological_sort(self))

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
        topo_nodes = list(nx.topological_sort(self))
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

        indeg = {n: self.in_degree(n) for n in self.nodes()}
        ready = [(_key(n), i, n) for i, n in enumerate(topo_nodes) if indeg[n] == 0]
        heapq.heapify(ready)
        seq, tie = [], len(topo_nodes)
        while ready:
            _, _, n = heapq.heappop(ready)
            seq.append(n)
            for succ in self.successors(n):
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
        if n_tags > max_tags:
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
