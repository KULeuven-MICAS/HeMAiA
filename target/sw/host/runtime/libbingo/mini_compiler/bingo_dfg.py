# Fanchen Kong <fanchen.kong@kuleuven.be>

import os
import re
import subprocess
import sys
from collections import defaultdict

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    import site
    from importlib import reload
    reload(site)

try:
    import networkx as nx
except ImportError:
    print("networkx not found. Installing...")
    install_package("networkx")
    try:
        import networkx as nx
    except ImportError:
        # Manually add user site-packages if reload(site) didn't help
        import site
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.append(user_site)
        import networkx as nx

from bingo_utils import DiGraphWrapper
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc
from bingo_kernel_args import (
    BingoKernelArgs,
    HostBingoKernelCerfGatingArgs,
    BINGO_GATING_MODE_TOP_K,
    BINGO_GATING_MODE_THRESHOLD,
    BINGO_GATING_MODE_STATIC,
)

BINGO_HEAP_ALIGNMENT = 128
BINGO_FRAGMENT_SIZE_MIN = BINGO_HEAP_ALIGNMENT * 2
BINGO_KERNEL_SCRATCHPAD_SIZE_BYTES = 64
SPM_WIDE_ALIGNMENT = 1024

_DEFINE_RE = re.compile(r"^\s*#define\s+(\w+)\s+([0-9a-fA-FxX]+)\b")


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _align_down(value: int, alignment: int) -> int:
    return value & ~(alignment - 1)


BINGO_HEAP_INSTANCE_SIZE_PADDED = _align_up((64 + 1 + 5) * 8, BINGO_HEAP_ALIGNMENT)


def _bingo_heap_alloc_cost(request_size: int) -> int:
    return _align_up(request_size + BINGO_HEAP_ALIGNMENT, BINGO_FRAGMENT_SIZE_MIN)


def _bingo_heap_capacity(heap_size: int) -> int:
    min_size = BINGO_HEAP_INSTANCE_SIZE_PADDED + BINGO_FRAGMENT_SIZE_MIN
    return 0 if heap_size < min_size else _align_down(heap_size - BINGO_HEAP_INSTANCE_SIZE_PADDED, BINGO_FRAGMENT_SIZE_MIN)


def _parse_c_defines(path: str) -> dict:
    with open(path) as f:
        return {match.group(1): int(match.group(2), 0) for line in f if (match := _DEFINE_RE.match(line))}


def _default_heap_capacities() -> dict:
    memory_map_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../../shared/platform/generated/occamy_memory_map.h",
        )
    )
    if not os.path.exists(memory_map_path):
        return {}

    defines = _parse_c_defines(memory_map_path)
    sizes = {
        "L1": defines.get("CLUSTER_TCDM_SIZE"),
        "L2": _align_up(defines["NARROW_SPM_SIZE"] // 2, BINGO_HEAP_ALIGNMENT) if "NARROW_SPM_SIZE" in defines else None,
        "L3": _align_down(defines["WIDE_SPM_SIZE"] - SPM_WIDE_ALIGNMENT, SPM_WIDE_ALIGNMENT) if "WIDE_SPM_SIZE" in defines else None,
    }
    return {level: _bingo_heap_capacity(size) for level, size in sizes.items() if size is not None}


class BingoDFG(DiGraphWrapper[BingoNode]):
    """Data Flow Graph (DFG) for Bingo."""

    def __init__(self,
                 num_chiplets: int,
                 num_clusters_per_chiplet: int,
                 num_cores_per_cluster: int,
                 is_host_as_acc: bool,
                 chiplet_ids: list[int] = None) -> None:
        super().__init__()
        # HW architecture parameters
        self.num_chiplets = num_chiplets
        self.num_clusters_per_chiplet = num_clusters_per_chiplet
        self.num_cores_per_cluster = num_cores_per_cluster + 1 if is_host_as_acc else num_cores_per_cluster
        self.is_host_as_acc = is_host_as_acc
        assert num_chiplets == len(chiplet_ids) or chiplet_ids is None, "Length of chiplet_ids must match num_chiplets"
        self.chiplet_ids = chiplet_ids if chiplet_ids else list(range(num_chiplets))
        self.heap_capacities = _default_heap_capacities()
        # Node ID counter
        # Make sure the node id is starts from 0
        self.id = -1
        self._next_cerf_group = 0
        self._gating_cerf_mappings: dict = {}  # gating_node → {expert_idx: cerf_group_id}
    def bingo_add_node(self, node_obj: BingoNode) -> None:
        """Add a node to the DFG."""

        # Assign a unique ID to the node
        self.id += 1
        node_obj.node_id = self.id
        # Add the node to the graph and the lookup dictionaries
        self.add_node(node_obj)
    def bingo_add_edge(self, from_node_obj: BingoNode, to_node_obj: BingoNode, cond: bool = False, cond_dic: dict = None) -> None:
        """Add an edge to the DFG.

        Args:
            cond: If True, marks this as a conditional execution edge.
            cond_dic: Dict with gating policy. Implies cond=True. Keys:
                mode: 'top_k' | 'threshold' | 'static' | 'custom'
                k: int (for top_k), threshold: float (for threshold), etc.
        """
        if cond_dic is not None:
            cond = True
        self.add_edge(from_node_obj, to_node_obj, cond=cond, cond_dic=cond_dic or {})

    def bingo_insert_node_between(self, from_node_obj: BingoNode, to_node_obj: BingoNode, new_node_obj: BingoNode) -> None:
        """Insert a new node between two existing nodes in the DFG."""
        if not self.has_edge(from_node_obj, to_node_obj):
            raise ValueError(f"No edge exists between {from_node_obj.node_name} and {to_node_obj.node_name}")

        # Preserve edge attributes (e.g. cond) before removal
        edge_data = dict(self[from_node_obj][to_node_obj])

        self.bingo_add_node(new_node_obj)
        self.remove_edge(from_node_obj, to_node_obj)

        # src → new_node: unconditional (dummy nodes must always execute)
        self.add_edge(from_node_obj, new_node_obj)
        # new_node → dst: inherit original edge attributes
        self.add_edge(new_node_obj, to_node_obj, **edge_data)
    
    def bingo_insert_node_after(self, existing_node_obj: BingoNode, new_node_obj: BingoNode, successors_to_move: list[BingoNode] = None) -> None:
        """Insert a new node after an existing node in the DFG."""
        if successors_to_move is None:
            successors_to_move = list(self.successors(existing_node_obj))

        # Preserve edge attributes before removal
        succ_edge_data = {}
        for succ in successors_to_move:
            succ_edge_data[succ] = dict(self[existing_node_obj][succ])

        self.bingo_add_node(new_node_obj)

        for succ in successors_to_move:
            self.remove_edge(existing_node_obj, succ)

        # existing → new_node: unconditional
        self.add_edge(existing_node_obj, new_node_obj)

        # new_node → successors: inherit original edge attributes
        for succ in successors_to_move:
            self.add_edge(new_node_obj, succ, **succ_edge_data[succ])
    
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
            local_end_nodes = [
                node for node in self.node_list if node.assigned_chiplet_id == chiplet_id and self.out_degree(node) == 0
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
                # Group remote successors by assigned_core_id so that each core-group can be
                # handled independently: a per-core group that covers all other chiplets becomes
                # a single broadcast dummy_set; otherwise individual dummy_sets are emitted.
                remote_succs_by_core: dict[int, list[BingoNode]] = {}
                for remote_succ in remote_succ_list:
                    remote_succs_by_core.setdefault(remote_succ.assigned_core_id, []).append(remote_succ)

                for core_id, group in remote_succs_by_core.items():
                    chiplets_in_group = set(s.assigned_chiplet_id for s in group)
                    if len(chiplets_in_group) == (self.num_chiplets - 1):
                        # Broadcast: one dummy_set blocks cur_node's core and sets the bit on all chiplets
                        print(f"Node {cur_node.node_name} is a broadcast node to set all chiplets for core {core_id}.")
                        dummy_set_node = BingoNode(
                            assigned_chiplet_id=cur_node.assigned_chiplet_id,
                            assigned_cluster_id=cur_node.assigned_cluster_id,      # must be the same type of the cur_node to block the execution
                            assigned_core_id=cur_node.assigned_core_id,            # must be the same type of the cur_node to block the execution
                            kernel_name= None
                        )
                        dummy_set_node.node_type = "dummy"
                        dummy_set_node.dep_set_enable = True
                        dummy_set_node.dep_set_list = [group[0].assigned_core_id]
                        dummy_set_node.dep_set_cluster_id = group[0].assigned_cluster_id
                        dummy_set_node.dep_set_chiplet_id = group[0].assigned_chiplet_id # should be fine since it is a broadcast type
                        dummy_set_node.dep_check_enable = False
                        dummy_set_node.dep_check_list = []
                        dummy_set_node.remote_dep_set_all = True
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
                                kernel_name= None
                            )
                            dummy_set_node.node_type = "dummy"
                            dummy_set_node.dep_set_enable = True
                            dummy_set_node.dep_set_list = [remote_succ.assigned_core_id]
                            dummy_set_node.dep_set_cluster_id = remote_succ.assigned_cluster_id
                            dummy_set_node.dep_set_chiplet_id = remote_succ.assigned_chiplet_id
                            dummy_set_node.dep_check_enable = False
                            dummy_set_node.dep_check_list = []
                            dummy_set_node.remote_dep_set_all = False
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

            if len(remaining_core_ids) >= 2:
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
                    print(f"Warning: More than one local successor for node {cur_node.node_id}. This is not expected, go back to DFG transformation stage!")
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
    # ----------------------------------------------------------------
    # DARTS Tier 1: Conditional Execution Compilation
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Node Packing / Unpacking
    # ----------------------------------------------------------------
    def bingo_pack_node(self, node: BingoNode) -> int:
        """Pack the normal node into a 64-bit integer."""
        import math
        
        def get_idx_width(n):
            return math.ceil(math.log2(n)) if n > 1 else 1

        # Parameters
        chip_id_width = 8
        task_id_width = 12
        # Use the class members for dimensions
        num_clusters = self.num_clusters_per_chiplet
        num_cores = self.num_cores_per_cluster
        
        cluster_id_width = get_idx_width(num_clusters)
        core_id_width = get_idx_width(num_cores)
        
        # Initialize the packed value
        packed_val = 0
        current_shift = 0

        # 1. cond_exec_invert (1 bit) — DARTS Tier 1
        packed_val |= (int(node.cond_exec_invert) << current_shift)
        current_shift += 1

        # 2. cond_exec_group_id (5 bits) — DARTS Tier 1
        packed_val |= (node.cond_exec_group_id << current_shift)
        current_shift += 5

        # 3. cond_exec_en (1 bit) — DARTS Tier 1
        packed_val |= (int(node.cond_exec_en) << current_shift)
        current_shift += 1

        # 4. task_type (2 bits) — expanded from 1 bit
        # 00: Normal, 01: Dummy, 10: Gating
        task_type_map = {"normal": 0, "dummy": 1, "gating": 2}
        task_type_val = task_type_map.get(node.node_type, 0)
        packed_val |= (task_type_val << current_shift)
        current_shift += 2
        
        # 2. task_id (12 bits)
        packed_val |= (node.node_id << current_shift)
        current_shift += task_id_width
        
        # 3. assigned_chiplet_id (8 bits)
        packed_val |= (node.assigned_chiplet_id << current_shift)
        current_shift += chip_id_width
        
        # 4. assigned_cluster_id (cluster_id_width bits)
        packed_val |= (node.assigned_cluster_id << current_shift)
        current_shift += cluster_id_width
        
        # 5. assigned_core_id (core_id_width bits)
        packed_val |= (node.assigned_core_id << current_shift)
        current_shift += core_id_width
        
        # 6. dep_check_info
        
        # dep_check_en (1 bit)
        dep_check_en_val = 1 if node.dep_check_enable else 0
        packed_val |= (dep_check_en_val << current_shift)
        current_shift += 1
        
        # dep_check_code (num_cores bits)
        dep_check_code_val = 0
        for core_id in node.dep_check_list:
            dep_check_code_val |= (1 << core_id)
        packed_val |= (dep_check_code_val << current_shift)
        current_shift += num_cores
        
        # 7. dep_set_info
        
        # dep_set_en (1 bit)
        dep_set_en_val = 1 if node.dep_set_enable else 0
        packed_val |= (dep_set_en_val << current_shift)
        current_shift += 1
        
        # dep_set_all_chiplet (1 bit)
        dep_set_all_val = 1 if node.remote_dep_set_all else 0
        packed_val |= (dep_set_all_val << current_shift)
        current_shift += 1
        
        # dep_set_chiplet_id (8 bits)
        packed_val |= (node.dep_set_chiplet_id << current_shift)
        current_shift += chip_id_width
        
        # dep_set_cluster_id (cluster_id_width bits)
        packed_val |= (node.dep_set_cluster_id << current_shift)
        current_shift += cluster_id_width
        
        # dep_set_code (num_cores bits)
        dep_set_code_val = 0
        for core_id in node.dep_set_list:
            dep_set_code_val |= (1 << core_id)
        packed_val |= (dep_set_code_val << current_shift)
        current_shift += num_cores
        
        # Check if we exceeded 64 bits
        if current_shift > 64:
            raise ValueError(f"Packed task descriptor exceeds 64 bits: {current_shift} bits used.")
            
        return packed_val
    def bingo_unpack_node(self, packed_val: int) -> dict:
        """Unpack the 64-bit integer into node fields."""
        import math
        
        def get_idx_width(n):
            return math.ceil(math.log2(n)) if n > 1 else 1

        # Parameters
        chip_id_width = 8
        task_id_width = 12
        num_clusters = self.num_clusters_per_chiplet
        num_cores = self.num_cores_per_cluster
        
        cluster_id_width = get_idx_width(num_clusters)
        core_id_width = get_idx_width(num_cores)
        
        current_shift = 0
        fields = {}

        # 1. cond_exec_invert (1 bit) — DARTS Tier 1
        fields['cond_exec_invert'] = (packed_val >> current_shift) & 0x1
        current_shift += 1

        # 2. cond_exec_group_id (5 bits) — DARTS Tier 1
        fields['cond_exec_group_id'] = (packed_val >> current_shift) & 0x1F
        current_shift += 5

        # 3. cond_exec_en (1 bit) — DARTS Tier 1
        fields['cond_exec_en'] = (packed_val >> current_shift) & 0x1
        current_shift += 1

        # 4. task_type (2 bits) — 00=normal, 01=dummy, 10=gating
        fields['task_type'] = (packed_val >> current_shift) & 0x3
        current_shift += 2
        
        # 2. task_id (12 bits)
        fields['task_id'] = (packed_val >> current_shift) & ((1 << task_id_width) - 1)
        current_shift += task_id_width
        
        # 3. assigned_chiplet_id (8 bits)
        fields['assigned_chiplet_id'] = (packed_val >> current_shift) & ((1 << chip_id_width) - 1)
        current_shift += chip_id_width
        
        # 4. assigned_cluster_id
        fields['assigned_cluster_id'] = (packed_val >> current_shift) & ((1 << cluster_id_width) - 1)
        current_shift += cluster_id_width
        
        # 5. assigned_core_id
        fields['assigned_core_id'] = (packed_val >> current_shift) & ((1 << core_id_width) - 1)
        current_shift += core_id_width
        
        # 6. dep_check_info
        fields['dep_check_en'] = (packed_val >> current_shift) & 0x1
        current_shift += 1
        
        fields['dep_check_code'] = (packed_val >> current_shift) & ((1 << num_cores) - 1)
        current_shift += num_cores
        
        # 7. dep_set_info
        fields['dep_set_en'] = (packed_val >> current_shift) & 0x1
        current_shift += 1
        
        fields['dep_set_all'] = (packed_val >> current_shift) & 0x1
        current_shift += 1
        
        fields['dep_set_chiplet_id'] = (packed_val >> current_shift) & ((1 << chip_id_width) - 1)
        current_shift += chip_id_width
        
        fields['dep_set_cluster_id'] = (packed_val >> current_shift) & ((1 << cluster_id_width) - 1)
        current_shift += cluster_id_width
        
        fields['dep_set_code'] = (packed_val >> current_shift) & ((1 << num_cores) - 1)
        current_shift += num_cores
        
        return fields
    def bingo_visualize_dfg(self, filename: str = "dfg_visualization", figsize: tuple = (20, 16)) -> None:
        """Visualize the DFG with different shapes for task types and colors for chiplets."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            print("matplotlib not found. Installing...")
            install_package("matplotlib")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

        # Define shapes for different task types
        task_type_shapes = {
            "normal": "o",  # Circle
            "dummy_set": "s",   # Square
            "dummy_check": "v",  # Downward Triangle
        }

        # Define a color map for chiplets
        chiplet_colors = [
            "lightcoral", "lightblue", "lightgreen", "moccasin", "plum", "lightgray", "wheat", "lavender", "lightcyan", "mistyrose"
        ]

        # Custom Layout Calculation
        # X axis: Topological depth
        # Y axis: Hardware resource location (Chiplet > Cluster > Core)
        pos = {}
        try:
            generations = list(nx.topological_generations(self))
        except Exception as e:
            # Fallback for cycles (should not happen in DAG) or other errors
            print(f"Warning: Topological sort failed ({e}), treating all nodes as gen 0")
            generations = [list(self.nodes)]
        
        node_gen_map = {}
        for g_idx, gen in enumerate(generations):
            for node in gen:
                node_gen_map[node] = g_idx
        
        # Parameters for spacing (Compact)
        CORE_H = 1.0
        CLUSTER_PAD = 0.5
        CHIPLET_PAD = 1.5
        
        # To handle multiple nodes at same (gen, core), we shift them in X slightly
        # Key: (gen, chip, cluster, core) -> count
        overlap_tracker = {}

        num_clusters = self.num_clusters_per_chiplet
        num_cores = self.num_cores_per_cluster # includes host core if is_host_as_acc
        
        # Height of one cluster block
        cluster_block_h = (num_cores * CORE_H) + CLUSTER_PAD
        # Height of one chiplet block
        chiplet_block_h = (num_clusters * cluster_block_h) + CHIPLET_PAD
        
        # Mapping from chiplet_id to its index (0, 1, 2...) for compact layout
        sorted_chiplets = sorted(self.chiplet_ids)
        chiplet_idx_map = {cid: idx for idx, cid in enumerate(sorted_chiplets)}

        for node in self.nodes:
            gen = node_gen_map.get(node, 0)
            cid = node.assigned_chiplet_id
            c_idx = chiplet_idx_map.get(cid, 0) # Use the index, not the ID
            
            clid = node.assigned_cluster_id
            coreid = node.assigned_core_id
            
            # Base Y (Top is 0, moving down is negative)
            # Use c_idx for positioning instead of cid
            y = 0
            y -= c_idx * chiplet_block_h
            y -= clid * cluster_block_h
            y -= coreid * CORE_H
            
            # Check overlap
            key = (gen, cid, clid, coreid)
            if key not in overlap_tracker:
                overlap_tracker[key] = 0
            overlap_count = overlap_tracker[key]
            overlap_tracker[key] += 1
            
            # Shift X for overlaps
            # Shift by fraction of generation width
            x = gen + (overlap_count * 0.4)
            
            pos[node] = (x, y)

        # Separate nodes by task type and chiplet
        node_shapes = {shape: [] for shape in task_type_shapes.values()}
        node_colors = {}

        for node in self.nodes:
            task_type = node.node_type  # Get the task type as a string
            if task_type == "dummy":
                if node.dep_set_enable:
                    task_type = "dummy_set"
                elif node.dep_check_enable:
                    task_type = "dummy_check"
            assigned_chiplet = node.assigned_chiplet_id

            # Get the shape for the task type
            shape = task_type_shapes.get(task_type, "o")  # Default to circle if task_type is unknown
            node_shapes[shape].append(node)

            # Get the color for the chiplet
            color = chiplet_colors[assigned_chiplet % len(chiplet_colors)]
            node_colors[node] = color

        # Create figure
        fig, ax_graph = plt.subplots(figsize=figsize)
        
        # Calculate bounds
        all_x = [p[0] for p in pos.values()]
        min_x = min(all_x) if all_x else 0
        max_x = max(all_x) if all_x else 1

        # Draw Region Lines and Labels
        for cid in sorted_chiplets:
            c_idx = chiplet_idx_map[cid]
            # Start Y of this chiplet block, using c_idx
            chiplet_start_y = -(c_idx * chiplet_block_h)
            
            # Label for Chiplet (Column 1)
            # Position it roughly in the middle of the chiplet block vertically
            chiplet_mid_y = chiplet_start_y - ((num_clusters * cluster_block_h)/2)
            ax_graph.text(min_x - 2.5, chiplet_mid_y, 
                          f"Chip 0x{cid:02x}", fontsize=10, fontweight='bold', 
                          va='center', ha='center', color='black')
             
            # Draw Cluster separators and labels
            for clid in range(self.num_clusters_per_chiplet):
                cluster_start_y = chiplet_start_y - (clid * cluster_block_h)
                cluster_end_y = cluster_start_y - (num_cores * CORE_H)
                
                # Label for Cluster (Column 2)
                cluster_mid_y = cluster_start_y - ((num_cores * CORE_H)/2)
                ax_graph.text(min_x - 1.5, cluster_mid_y, 
                              f"Cluster {clid}", fontsize=8, 
                              va='center', ha='center', color='black')
                
                # Label for Cores (Column 3)
                for coreid in range(num_cores):
                    core_y = cluster_start_y - (coreid * CORE_H)
                    ax_graph.text(min_x - 0.8, core_y, 
                                  f"Core {coreid}", fontsize=6, 
                                  va='center', ha='center', color='black')
                    
                    # Draw Core Separator (Horizontal)
                    # Don't draw after the last core, as that is the Cluster separator
                    if coreid < num_cores - 1:
                        core_sep_y = core_y - (CORE_H / 2.0)
                        ax_graph.hlines(y=core_sep_y, xmin=min_x-1.0, xmax=max_x+0.5, 
                                        colors='gray', linestyles='dotted', alpha=0.8, linewidth=1.0)

                # Separator line Y (middle of padding)
                separator_y = cluster_end_y - (CLUSTER_PAD / 2)

                if clid < self.num_clusters_per_chiplet - 1:
                    # Inner cluster separator: dotted
                    ax_graph.hlines(y=separator_y, xmin=min_x-0.5, xmax=max_x+0.5, 
                                    colors='gray', linestyles='dotted', alpha=0.5)
            
            # Separator at bottom of chiplet: dashed
            chiplet_bottom_line_y = -(c_idx * chiplet_block_h) - (self.num_clusters_per_chiplet * cluster_block_h) - (CHIPLET_PAD/2.0)
            ax_graph.hlines(y=chiplet_bottom_line_y, xmin=min_x-3.0, xmax=max_x+1.0, 
                            colors='black', linestyles='dashed', alpha=0.6, linewidth=1.5)

        # Draw nodes with different shapes
        for shape, nodes in node_shapes.items():
            nx.draw_networkx_nodes(
                self, pos, nodelist=nodes,
                node_shape=shape,
                node_color=[node_colors[node] for node in nodes],
                node_size=300, # Smaller size
                ax=ax_graph
            )

        # Draw edges
        nx.draw_networkx_edges(self, pos, ax=ax_graph, alpha=0.6, arrows=True)

        # Draw labels
        labels = {}
        for node in self.nodes:
            # Simplified label: just the ID
            label_string = f"{node.node_id}"
            labels[node] = label_string
        nx.draw_networkx_labels(self, pos, labels=labels, font_size=6, ax=ax_graph)

        # Create a legend for task types
        legend_elements = [
            Line2D([0], [0], marker=shape, color="w", label=task_type, markerfacecolor="black", markersize=10)
            for task_type, shape in task_type_shapes.items()
        ]
        
        # Create a legend for chiplets
        chiplet_legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=f"Chiplet {cid:02x}", markerfacecolor=chiplet_colors[cid % len(chiplet_colors)], markersize=10)
            for cid in sorted(self.chiplet_ids)
        ]
        
        # Combine legends
        all_legends = legend_elements + chiplet_legend_elements
        
        ax_graph.legend(handles=all_legends, loc="best")

        # Save the visualization to a file
        plt.tight_layout()
        plt.savefig(f"{filename}.png")

    def bingo_export_dfg_to_csv(self, filename: str = "dfg_table") -> None:
        """Export the DFG node details to a CSV file."""
        import csv
        
        col_labels = ["ID", "Chiplet", "Cluster", "Core", "Type", "Kernel"]
        table_data = []
        sorted_nodes = sorted(self.nodes, key=lambda n: n.node_id)
        for node in sorted_nodes:
            t_type = node.node_type
            if t_type == "dummy":
                if node.dep_set_enable:
                    t_type = "dummy_set"
                elif node.dep_check_enable:
                    t_type = "dummy_check"
            
            row = [
                f"{node.node_id}",
                f"{node.assigned_chiplet_id:02x}",
                f"{node.assigned_cluster_id}",
                f"{node.assigned_core_id}",
                t_type,
                node.kernel_name
            ]
            table_data.append(row)

        with open(f"{filename}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(col_labels)
            writer.writerows(table_data)

    def bingo_emit_task_kernel_name_list(self) -> str:
        """Emit the list of task kernel names."""
        # Generate the kernel name list
        # Can be directly used in C code to get the function name from task id
        kernel_name_list = ""
        normal_node = [node for node in self.node_list if node.node_type == "normal"]
        num_normal_nodes = len(normal_node)
        # Sort the normal nodes by node id
        normal_node.sort(key=lambda x: x.node_id)
        kernel_name_list += f"char kernel_name_list[{num_normal_nodes}][64] = {{\n"
        for node in normal_node:
            kernel_name_list += f'    "{node.kernel_name}", // Node ID {node.node_id}\n'
        kernel_name_list += "};\n"
        return kernel_name_list



    def bingo_emit_task_desc_list(self, target_chiplet_id: int = None) -> str:
        """Emit the task description list in the DFG."""
        task_description_list = ""
        
        # Use topological sort, but ensure Dummy Set nodes 
        # appear immediately after their source node.
        # 1. Get topological sort
        topo_nodes = list(nx.topological_sort(self))
        all_nodes = topo_nodes
        # # 2. Apply grouping logic
        # all_nodes = []
        # visited = set()
        
        # for node in topo_nodes:
        #     if node in visited:
        #         continue
                
        #     all_nodes.append(node)
        #     visited.add(node)
            
        #     # Find successors that are dummy set nodes
        #     # These nodes must follow the current node immediately in the descriptor list
        #     successors = list(self.successors(node))
        #     dummy_set_succs = [
        #         s for s in successors 
        #         if s.node_type == "dummy" and s.dep_set_enable
        #     ]
            
        #     # Sort by ID for determinism
        #     dummy_set_succs.sort(key=lambda x: x.node_id)
            
        #     for dummy in dummy_set_succs:
        #         if dummy not in visited:
        #             all_nodes.append(dummy)
        #             visited.add(dummy)
        
        chiplets_to_process = [target_chiplet_id] if target_chiplet_id is not None else self.chiplet_ids

        for chiplet_id in chiplets_to_process:
            local_nodes = [node for node in all_nodes if node.assigned_chiplet_id == chiplet_id]
            num_local_nodes = len(local_nodes)
            
            # Emit num_tasks at the beginning
            task_description_list += f"uint32_t bingo_hw_scheduler_num_task_desc_chip_{chiplet_id:02x} = {num_local_nodes};\n"

            if num_local_nodes == 0:
                 # Even if size is 0, we allocate 1 element to avoid issues with size 0 allocation if allocator doesn't support it, or just use 0.
                 # Using 1 for safety, similar to original array [1]
                 task_description_list += f"uint64_t* bingo_hw_scheduler_task_desc_list_chip_{chiplet_id:02x} = (uint64_t*)bingo_l3_alloc(0x{chiplet_id:02x}, 1 * sizeof(uint64_t));\n"
                 task_description_list += f"bingo_hw_scheduler_task_desc_list_chip_{chiplet_id:02x}[0] = 0;\n"
            else:
                task_description_list += f"uint64_t* bingo_hw_scheduler_task_desc_list_chip_{chiplet_id:02x} = (uint64_t*)bingo_l3_alloc(0x{chiplet_id:02x}, bingo_hw_scheduler_num_task_desc_chip_{chiplet_id:02x} * sizeof(uint64_t));\n"
                for idx, node in enumerate(local_nodes):
                    packed_val = self.bingo_pack_node(node)
                    fields = self.bingo_unpack_node(packed_val)
                    
                    # Create a detailed comment
                    comment = f"// Node ID {node.node_id}\n"
                    comment += f"    // Fields: Type={fields['task_type']}, TaskID={fields['task_id']}\n"
                    comment += f"    //         Assigned: Chiplet={fields['assigned_chiplet_id']:02x}, Cluster={fields['assigned_cluster_id']}, Core={fields['assigned_core_id']}\n"
                    comment += f"    //         DepCheck: En={fields['dep_check_en']}, Code=0b{fields['dep_check_code']:0{self.num_cores_per_cluster}b}\n"
                    comment += f"    //         DepSet:   En={fields['dep_set_en']}, All={fields['dep_set_all']}, Chiplet={fields['dep_set_chiplet_id']:02x}, Cluster={fields['dep_set_cluster_id']}, Code=0b{fields['dep_set_code']:0{self.num_cores_per_cluster}b}"
                    
                    task_description_list += f"bingo_hw_scheduler_task_desc_list_chip_{chiplet_id:02x}[{idx}] = 0x{packed_val:016X}; {comment}\n"
            
        return task_description_list
    
    def bingo_emit_task_id_mapping_lists(self, target_chiplet_id: int = None) -> str:
        """Emit the mapping lists from global task id to dev/host task id."""
        all_nodes = self.node_list
        num_nodes = len(all_nodes)
        # Sort the nodes by node id
        all_nodes.sort(key=lambda x: x.node_id)
        
        mapping_str = ""
        chiplets_to_process = [target_chiplet_id] if target_chiplet_id is not None else self.chiplet_ids
        
        # 1. Emit global_task_id_to_dev_task_id for each chiplet
        # Also need to emit num_dev_tasks for each chiplet
        for chiplet_id in chiplets_to_process:
            mapping_str += f"int32_t* global_task_id_to_dev_task_id_chip_{chiplet_id:02x} = (int32_t*)bingo_l3_alloc(0x{chiplet_id:02x}, {num_nodes} * sizeof(int32_t));\n"
            dev_task_counter = 0
            
            for idx, node in enumerate(all_nodes):
                kernel_name = node.kernel_name
                # Check if the node is assigned to the current chiplet
                val = "-1"
                comment = ""
                if node.assigned_chiplet_id == chiplet_id:
                    if kernel_name and kernel_name.startswith("__snax"):
                         # It is a device task
                        val = str(dev_task_counter)
                        comment = f" -> Dev Task {dev_task_counter} ({node.node_name})"
                        dev_task_counter += 1
                    else:
                        comment = f" ({node.node_name})"
                
                mapping_str += f"global_task_id_to_dev_task_id_chip_{chiplet_id:02x}[{idx}] = {val}; // Node ID {node.node_id}{comment}\n"
            
            mapping_str += f"uint32_t num_dev_tasks_chip_{chiplet_id:02x} = {dev_task_counter};\n"
            
        # 2. Emit global_task_id_to_host_task_id
        for chiplet_id in chiplets_to_process:
            mapping_str += f"int32_t* global_task_id_to_host_task_id_chip_{chiplet_id:02x} = (int32_t*)bingo_l3_alloc(0x{chiplet_id:02x}, {num_nodes} * sizeof(int32_t));\n"
            host_task_counter = 0
            for idx, node in enumerate(all_nodes):
                kernel_name = node.kernel_name
                val = "-1"
                comment = ""
                if node.assigned_chiplet_id == chiplet_id:
                    if kernel_name and kernel_name.startswith("__host"):
                        val = str(host_task_counter)
                        comment = f" -> Host Task {host_task_counter} ({node.node_name})"
                        host_task_counter += 1
                    else:
                        comment = f" ({node.node_name})"

                mapping_str += f"global_task_id_to_host_task_id_chip_{chiplet_id:02x}[{idx}] = {val}; // Node ID {node.node_id}{comment}\n"
            
            mapping_str += f"uint32_t num_host_tasks_chip_{chiplet_id:02x} = {host_task_counter};\n"
        return mapping_str


    def _collect_memory_handles(self, sorted_nodes):
        """Collect and sort unique BingoMemAlloc from nodes."""
        unique_handles = set()
        for node in sorted_nodes:
            if node.kernel_args:
                for attr, value in node.kernel_args.__dict__.items():
                     if isinstance(value, BingoMemAlloc):
                         unique_handles.add(value)
        
        sorted_handles = sorted(list(unique_handles), key=lambda h: h.name)
        handle_name_map = {h: h.get_c_var_name() for h in sorted_handles}
        return sorted_handles, handle_name_map

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

    def _heap_location_label(self, key):
        mem_level, chip_id, cluster_id = key
        if mem_level == "L1":
            return f"L1 chip 0x{chip_id:02x} cluster {cluster_id}"
        return f"{mem_level} chip 0x{chip_id:02x}"

    def _add_heap_usage(self, usage, mem_level, chip_id, cluster_id, request_size, count=1):
        if count <= 0 or request_size <= 0:
            return
        usage[(mem_level, chip_id, cluster_id if mem_level == "L1" else 0)] += _bingo_heap_alloc_cost(request_size) * count

    def _estimate_kernel_arg_request_size(self, node, handle_name_map):
        kernel_name = node.kernel_name or ""
        if node.kernel_args:
            struct_name = node.kernel_args.get_struct_name()
            assignments = node.kernel_args.get_c_field_assignments(handle_name_map)
            return 4 * (len(assignments) + 3) if struct_name.startswith("__snax") else 8 * (len(assignments) + 1)

        if "exit" in kernel_name:
            if kernel_name.startswith("__snax"):
                return 16  # exit_code + BINGO_KERNEL_ARGS_TRAILER
            if kernel_name.startswith("__host"):
                return 16  # exit_code + scratchpad_ptr
        return 0

    def _estimate_heap_usage(self, sorted_nodes, sorted_handles, handle_name_map):
        usage = defaultdict(int)

        for handle in sorted_handles:
            self._add_heap_usage(
                usage,
                handle.mem_level,
                handle.chip_id,
                handle.cluster_id,
                handle.size,
            )

        topo_nodes = list(nx.topological_sort(self))
        num_total_nodes = len(sorted_nodes)
        for chiplet_id in self.chiplet_ids:
            local_nodes = [node for node in topo_nodes if node.assigned_chiplet_id == chiplet_id]
            dev_nodes = [
                node for node in local_nodes
                if node.kernel_name and node.kernel_name.startswith("__snax")
            ]
            host_nodes = [
                node for node in local_nodes
                if node.kernel_name and node.kernel_name.startswith("__host")
            ]

            for size in (
                max(len(local_nodes), 1) * 8,
                num_total_nodes * 4,
                num_total_nodes * 4,
                len(dev_nodes) * 4,
                len(dev_nodes) * 4,
                len(host_nodes) * 8,
                len(host_nodes) * 8,
            ):
                self._add_heap_usage(usage, "L3", chiplet_id, 0, size)

            for cluster_id in range(self.num_clusters_per_chiplet):
                for size in (len(dev_nodes) * 4, len(dev_nodes) * 4, num_total_nodes * 4):
                    self._add_heap_usage(usage, "L1", chiplet_id, cluster_id, size)

            for node in dev_nodes:
                self._add_heap_usage(usage, "L1", chiplet_id, node.assigned_cluster_id, BINGO_KERNEL_SCRATCHPAD_SIZE_BYTES)
                self._add_heap_usage(usage, "L1", chiplet_id, node.assigned_cluster_id, self._estimate_kernel_arg_request_size(node, handle_name_map))

            for node in host_nodes:
                self._add_heap_usage(usage, "L3", chiplet_id, 0, BINGO_KERNEL_SCRATCHPAD_SIZE_BYTES)
                self._add_heap_usage(usage, "L3", chiplet_id, 0, self._estimate_kernel_arg_request_size(node, handle_name_map))

        return usage

    def _check_heap_usage_fits(self, usage):
        failures = []
        for key in sorted(usage):
            capacity = self.heap_capacities.get(key[0])
            if capacity is None:
                continue
            required = usage[key]
            if required > capacity:
                failures.append(
                    f"{self._heap_location_label(key)}: required {required} B, "
                    f"capacity {capacity} B"
                )
        if failures:
            raise ValueError(
                "Bingo heap allocation check failed before C generation "
                "(sizes include allocator rounding):\n" + "\n".join(failures)
            )

    def _emit_headers(self, f, extra_include_header_list):
        """Emit C header includes."""
        f.write("// Auto-generated offload_hw_bingo.h\n")
        f.write("#pragma once\n")
        f.write('#include "libbingo/bingo_api.h"\n')
        f.write('#include "host.h"\n')
        for include in extra_include_header_list:
            f.write(f'#include "{include}"\n')
        f.write("\n")

    def _emit_debug_kernel_list(self, f):
        """Emit commented-out kernel name list for debugging."""
        f.write("// Kernel Name List\n")
        f.write("// Note: This list is currently for debugging purposes only and is not used in the runtime.\n")
        f.write("// It will be enabled in the future.\n")
        f.write("/*\n")
        f.write(self.bingo_emit_task_kernel_name_list())
        f.write("*/\n")
        f.write("\n")

    def _emit_task_desc_and_mappings(self, f, chiplet_id):
        """Emit task description and ID mapping lists."""
        f.write(f"        uint32_t num_total_tasks = {len(self.node_list)};\n")
        f.write("        // Task Description List\n")
        task_desc_str = self.bingo_emit_task_desc_list(chiplet_id)
        indented_task_desc = "\n".join(["        " + line for line in task_desc_str.splitlines()])
        f.write(f"{indented_task_desc}\n")

        f.write("        // Task ID Mapping Lists\n")
        mapping_str = self.bingo_emit_task_id_mapping_lists(chiplet_id)
        indented_mapping = "\n".join(["        " + line for line in mapping_str.splitlines()])
        f.write(f"{indented_mapping}\n")

    def _emit_memory_allocations(self, f, chiplet_id, sorted_handles, handle_name_map):
        """Emit memory allocation calls for handles on this chiplet."""
        local_handles = [h for h in sorted_handles if h.chip_id == chiplet_id]
        if local_handles:
            f.write("        // 1. Memory Allocations\n")
            for h in local_handles:
                c_var = handle_name_map[h]
                alloc_call = ""
                if h.mem_level == "L1":
                    alloc_call = f"bingo_l1_alloc(0x{h.chip_id:02x}, {h.cluster_id}, {h.size})"
                elif h.mem_level == "L2":
                        alloc_call = f"bingo_l2_alloc(0x{h.chip_id:02x}, {h.size})"
                else: # L3
                        alloc_call = f"bingo_l3_alloc(0x{h.chip_id:02x}, {h.size})"
                
                f.write(f"        uint64_t {c_var} = {alloc_call};\n")
            f.write("\n")

    def _emit_list_allocations(self, f, chiplet_id):
        """Emit allocations for device/host argument and kernel lists."""
        f.write(f"        // 2. Prepare device/host arg/kernel lists\n")
        f.write(f"        uint32_t* device_arg_list_chip_{chiplet_id:02x} = (uint32_t*)bingo_l3_alloc(0x{chiplet_id:02x}, num_dev_tasks_chip_{chiplet_id:02x} * sizeof(uint32_t));\n")
        f.write(f"        uint32_t* device_kernel_list_chip_{chiplet_id:02x} = (uint32_t*)bingo_l3_alloc(0x{chiplet_id:02x}, num_dev_tasks_chip_{chiplet_id:02x} * sizeof(uint32_t));\n")
        f.write(f"        uint64_t* host_arg_list_chip_{chiplet_id:02x} = (uint64_t*)bingo_l3_alloc(0x{chiplet_id:02x}, num_host_tasks_chip_{chiplet_id:02x} * sizeof(uint64_t));\n")
        f.write(f"        uint64_t* host_kernel_list_chip_{chiplet_id:02x} = (uint64_t*)bingo_l3_alloc(0x{chiplet_id:02x}, num_host_tasks_chip_{chiplet_id:02x} * sizeof(uint64_t));\n\n")

    def _emit_task_initialization(self, f, chiplet_id, sorted_nodes, handle_name_map):
        """Emit initialization for task arguments + per-kernel scratchpad."""
        f.write("        // 3. Task Arguments Init\n")

        local_nodes = [node for node in sorted_nodes if node.assigned_chiplet_id == chiplet_id]

        # Pass 0: Pre-allocate ALL scratchpads so gating node scratchpad C vars
        # are available when expert nodes reference them via SW guard.
        f.write("        // 3a. Pre-allocate scratchpads for all tasks\n")
        for node in local_nodes:
            kernel_name = node.kernel_name
            is_device = kernel_name and kernel_name.startswith("__snax")
            is_host = kernel_name and kernel_name.startswith("__host")
            if not (is_device or is_host):
                continue
            if is_device:
                sp_var = f"sp_dev_{node.node_id}"
                f.write(f"        bingo_kernel_scratchpad_t* {sp_var} = (bingo_kernel_scratchpad_t*)bingo_l1_alloc(0x{chiplet_id:02x}, {node.assigned_cluster_id}, BINGO_KERNEL_SCRATCHPAD_SIZE);\n")
            else:
                sp_var = f"sp_host_{node.node_id}"
                f.write(f"        bingo_kernel_scratchpad_t* {sp_var} = (bingo_kernel_scratchpad_t*)bingo_l3_alloc(0x{chiplet_id:02x}, BINGO_KERNEL_SCRATCHPAD_SIZE);\n")
            node._scratchpad_c_var = sp_var
        f.write("\n")

        # Now wire SW guard fields — gating node scratchpad C vars are all known
        for node in local_nodes:
            if node.kernel_args and node._gating_node is not None:
                gating_sp_var = node._gating_node._scratchpad_c_var
                if gating_sp_var:
                    is_dev = node.kernel_name and node.kernel_name.startswith("__snax")
                    cast = "(uint32_t)" if is_dev else "(uint64_t)"
                    node.kernel_args._gating_sp_c_expr = f"{cast}(uintptr_t){gating_sp_var}"
                if node._cond_node_index is not None:
                    node.kernel_args._cond_node_index = node._cond_node_index

        dev_task_idx = 0
        host_task_idx = 0

        for node in local_nodes:
            kernel_name = node.kernel_name
            is_device = kernel_name and kernel_name.startswith("__snax")
            is_host = kernel_name and kernel_name.startswith("__host")

            if not (is_device or is_host):
                continue

            f.write(f"        // Node ID: {node.node_id} {node.node_name} ({kernel_name})\n")

            # Scratchpad already allocated in pass 0 above
            sp_var = node._scratchpad_c_var
            sp_cast = f"(uint32_t)(uintptr_t){sp_var}" if is_device else f"(uint64_t)(uintptr_t){sp_var}"

            args_struct_type = ""
            if node.kernel_args:
                args_struct_type = node.kernel_args.get_struct_name()
                # Set scratchpad C expression so get_c_field_assignments_with_scratchpad includes it
                node.kernel_args._scratchpad_c_expr = sp_cast

            if is_device:
                args_var = f"args_dev_chip{chiplet_id:02x}_{node.node_id}"

                if node.kernel_args:
                    f.write(f"        {args_struct_type}* {args_var} = ({args_struct_type}*)bingo_l1_alloc(0x{chiplet_id:02x}, {node.assigned_cluster_id}, sizeof({args_struct_type}));\n")
                    field_assignments = node.kernel_args.get_c_field_assignments_with_scratchpad(handle_name_map)
                    for field, value in field_assignments.items():
                            f.write(f"        {args_var}->{field} = {value};\n")
                    # Wire pred_scratchpad_addr for auto-inserted gating nodes
                    if hasattr(node, '_pred_source_node') and node._pred_source_node:
                        pred_sp = node._pred_source_node._scratchpad_c_var
                        f.write(f"        {args_var}->pred_scratchpad_addr = (uint32_t)(uintptr_t){pred_sp};\n")
                    f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = (uint32_t)(uintptr_t){args_var};\n")
                else:
                    if "exit" in kernel_name:
                        f.write(f"        __snax_bingo_kernel_exit_args_t* {args_var} = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x{chiplet_id:02x}, {node.assigned_cluster_id}, sizeof(__snax_bingo_kernel_exit_args_t));\n")
                        f.write(f"        {args_var}->exit_code = 0;\n")
                        f.write(f"        {args_var}->scratchpad_ptr = {sp_cast};\n")
                        f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = (uint32_t)(uintptr_t){args_var};\n")
                    else:
                        f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = 0;\n")

                f.write(f"        device_kernel_list_chip_{chiplet_id:02x}[{dev_task_idx}] = (uint32_t)(uintptr_t)get_device_function(\"{kernel_name}\");\n")
                dev_task_idx += 1

            elif is_host:
                args_var = f"args_host_chip{chiplet_id:02x}_{node.node_id}"

                if node.kernel_args:
                    f.write(f"        {args_struct_type}* {args_var} = ({args_struct_type}*)bingo_l3_alloc(0x{chiplet_id:02x}, sizeof({args_struct_type}));\n")
                    field_assignments = node.kernel_args.get_c_field_assignments_with_scratchpad(handle_name_map)
                    for field, value in field_assignments.items():
                            f.write(f"        {args_var}->{field} = {value};\n")
                    # Wire pred_scratchpad_addr for auto-inserted gating nodes
                    if hasattr(node, '_pred_source_node') and node._pred_source_node:
                        pred_sp = node._pred_source_node._scratchpad_c_var
                        f.write(f"        {args_var}->pred_scratchpad_addr = (uint64_t)(uintptr_t){pred_sp};\n")
                    f.write(f"        host_arg_list_chip_{chiplet_id:02x}[{host_task_idx}] = (uint64_t)(uintptr_t){args_var};\n")
                else:
                    if "exit" in kernel_name:
                        f.write(f"        __host_bingo_kernel_exit_args_t* {args_var} = (__host_bingo_kernel_exit_args_t*)bingo_l3_alloc(0x{chiplet_id:02x}, sizeof(__host_bingo_kernel_exit_args_t));\n")
                        f.write(f"        {args_var}->exit_code = 0;\n")
                        f.write(f"        {args_var}->scratchpad_ptr = {sp_cast};\n")
                        f.write(f"        host_arg_list_chip_{chiplet_id:02x}[{host_task_idx}] = (uint64_t)(uintptr_t){args_var};\n")
                    else:
                        f.write(f"        host_arg_list_chip_{chiplet_id:02x}[{host_task_idx}] = 0;\n")

                f.write(f"        host_kernel_list_chip_{chiplet_id:02x}[{host_task_idx}] = (uint64_t)(uintptr_t)&{kernel_name};\n")
                host_task_idx += 1

            # Emit CERF group ID init for auto-inserted gating nodes
            if node in getattr(self, '_gating_cerf_mappings', {}):
                mapping = self._gating_cerf_mappings[node]
                if hasattr(node, 'kernel_args') and node.kernel_args and hasattr(node.kernel_args, 'cerf_group_ids_addr') and node.kernel_args.cerf_group_ids_addr is not None:
                    cerf_gids_handle = node.kernel_args.cerf_group_ids_addr
                    cerf_gids_var = handle_name_map.get(cerf_gids_handle)
                    if cerf_gids_var:
                        f.write(f"        // Auto-generated CERF group ID mapping\n")
                        f.write(f"        uint8_t* __cerf_gids_{node.node_id} = (uint8_t*)(uintptr_t){cerf_gids_var};\n")
                        for expert_idx, cerf_gid in mapping.items():
                            f.write(f"        __cerf_gids_{node.node_id}[{expert_idx}] = {cerf_gid};\n")
    
    def _emit_scheduler_launch(self, f, chiplet_id):
        """Emit the scheduler initialization and launch calls."""
        f.write("\n")
        f.write('        OFFLOAD_BINGO_HW_DEBUG_PRINT_SAFE("Chip(%x, %x): [Host] Init HW Bingo Scheduler\\r\\n",\n')
        f.write('               get_current_chip_loc_x(), get_current_chip_loc_y());\n\n')

        f.write(f"        bingo_hw_scheduler_init((uint64_t)(uintptr_t)device_arg_list_chip_{chiplet_id:02x},\n")
        f.write(f"                                (uint64_t)(uintptr_t)device_kernel_list_chip_{chiplet_id:02x},\n")
        f.write(f"                                num_dev_tasks_chip_{chiplet_id:02x},\n")
        f.write(f"                                (uint64_t)(uintptr_t)global_task_id_to_dev_task_id_chip_{chiplet_id:02x},\n")
        f.write(f"                                num_total_tasks,\n")
        f.write(f"                                (uint64_t)(uintptr_t)bingo_hw_scheduler_task_desc_list_chip_{chiplet_id:02x},\n")
        f.write(f"                                bingo_hw_scheduler_num_task_desc_chip_{chiplet_id:02x});\n\n")
        
        f.write(f"        uint32_t err = bingo_hw_scheduler(host_arg_list_chip_{chiplet_id:02x},\n")
        f.write(f"                                          host_kernel_list_chip_{chiplet_id:02x},\n")
        f.write(f"                                          global_task_id_to_host_task_id_chip_{chiplet_id:02x});\n")
        f.write(f"        if (err) return err;\n")

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

    def bingo_emit_offload_c_code(self, extra_include_header_list: list[str], output_path: str, app_name: str, post_execute_code: list[str] | None = None) -> None:
        """Emit the offload_hw_bingo.h file with kernel_execution logic."""
        
        # 1. Collect Handles
        sorted_nodes = sorted(self.node_list, key=lambda n: n.node_id)
        sorted_handles, handle_name_map = self._collect_memory_handles(sorted_nodes)
        self._validate_kernel_core_assignments(sorted_nodes)
        self._validate_memory_handles(sorted_handles)
        heap_usage = self._estimate_heap_usage(sorted_nodes, sorted_handles, handle_name_map)
        self._check_heap_usage_fits(heap_usage)

        # 2. Start emitting C code
        with open(output_path, "w") as f:
            # Step 1: Emit Headers
            self._emit_headers(f, extra_include_header_list)
            
            # Step 2: Emit Debug Kernel List
            self._emit_debug_kernel_list(f)

            # Step 3: Emit kernel_execution function structure
            f.write("int kernel_execution(){\n")
            f.write("    check_kernel_tab_ready();\n")
            f.write(f"    OFFLOAD_BINGO_HW_DEBUG_PRINT_SAFE(\"Chip(%x, %x): [Host] Preparing {app_name} Workload\\r\\n\", get_current_chip_loc_x(), get_current_chip_loc_y());\n")
            f.write("    uint32_t current_chip_id = get_current_chip_id();\n")
            

            # Step 4: Iterate over each chiplet to generate isolated blocks
            for chiplet_id in self.chiplet_ids:
                f.write(f"    if (current_chip_id == 0x{chiplet_id:02x}) {{\n")
                
                # A. Emit Task Description and Mapping Lists
                self._emit_task_desc_and_mappings(f, chiplet_id)
                
                # B. Emit Memory Allocations
                self._emit_memory_allocations(f, chiplet_id, sorted_handles, handle_name_map)
                
                # C. Emit List Allocations
                self._emit_list_allocations(f, chiplet_id)

                # D. Emit Task Initialization
                self._emit_task_initialization(f, chiplet_id, sorted_nodes, handle_name_map)

                # E. Emit Scheduler Launch
                self._emit_scheduler_launch(f, chiplet_id)

                # F. Emit Post-Execute Code (runs after scheduler completes)
                if post_execute_code:
                    f.write("\n        // Post-execution check\n")
                    for line in post_execute_code:
                        f.write(f"        {line}\n")

                f.write("    }\n")
            
            f.write("    return 0;\n")
            f.write("}\n")
    def bingo_compile_dfg(self, app_name: str, output_dir: str, output_file_name: str, extra_include_header_list: list[str] | None, post_execute_code: list[str] | None = None) -> None:
        """Compile the DFG by assigning dep info and emitting C code."""
        # 1. Transformations
        # Add Entry Node
        self.bingo_transform_dfg_add_entry_node()
        # Add Exit Nodes
        self.bingo_transform_dfg_add_exit_nodes()
        self.bingo_visualize_dfg(
            os.path.join(output_dir, "dfg_with_entry_exit_nodes")
        )
        # Compile conditional regions (CERF group assignment)
        # Must be called before dummy node transforms.
        # No-op for non-conditional DFGs (returns empty dict).
        self.bingo_compile_conditional_regions()
        self._validate_cerf_cross_group_edges()
        # Add Dummy Set/Check Nodes
        self.bingo_transform_add_core_sequencing_edges()
        self.bingo_transform_dfg_add_dummy_set_nodes()
        self.bingo_transform_dfg_add_dummy_check_nodes()
        self.bingo_visualize_dfg(
            os.path.join(output_dir, "final_dfg")
        )
        self.bingo_export_dfg_to_csv(
            os.path.join(output_dir, "final_dfg")
        )
        # Assign Dep Info
        self.bingo_assign_normal_node_dep_set_info()
        self.bingo_assign_normal_node_dep_check_info()
        
        # 2. Emit C Code
        self.bingo_emit_offload_c_code(
            app_name=app_name,
            output_path=os.path.join(output_dir, output_file_name),
            extra_include_header_list=extra_include_header_list,
            post_execute_code=post_execute_code,
        )
