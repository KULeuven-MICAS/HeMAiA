# Fanchen Kong <fanchen.kong@kuleuven.be>

import random
import os
from bingo_utils import DiGraphWrapper
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc
from bingo_kernel_args import BingoKernelArgs
import networkx as nx
MAX_NUM_CHIPLETS = 8
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
        # Node ID counter
        # Make sure the node id is starts from 0
        self.id = -1
    def bingo_add_node(self, node_obj: BingoNode) -> None:
        """Add a node to the DFG."""

        # Assign a unique ID to the node
        self.id += 1
        node_obj.node_id = self.id
        # Add the node to the graph and the lookup dictionaries
        self.add_node(node_obj)
    def bingo_add_edge(self, from_node_obj: BingoNode, to_node_obj: BingoNode) -> None:
        """Add an edge to the DFG using node objects."""
        self.add_edge(from_node_obj, to_node_obj)

    def bingo_insert_node_between(self, from_node_obj: BingoNode, to_node_obj: BingoNode, new_node_obj: BingoNode) -> None:
        """Insert a new node between two existing nodes in the DFG."""
        # Ensure the edge exists between the two nodes
        if not self.has_edge(from_node_obj, to_node_obj):
            raise ValueError(f"No edge exists between {from_node_obj.node_name} and {to_node_obj.node_name}")

        # Add the new node to the DFG
        self.bingo_add_node(new_node_obj)

        # Remove the edge between the two existing nodes
        self.remove_edge(from_node_obj, to_node_obj)

        # Add edges to connect the new node between the two existing nodes
        self.add_edge(from_node_obj, new_node_obj)
        self.add_edge(new_node_obj, to_node_obj)
        
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
                # We have a special situation that this node is a broadcast node to set all chiplets
                if len(set(remote_succ.assigned_chiplet_id for remote_succ in remote_succ_list)) == (MAX_NUM_CHIPLETS -1):
                    # All the remote successors must have the same core id
                    if len(set(remote_succ.assigned_core_id for remote_succ in remote_succ_list)) == 1:
                        print(f"Node {cur_node.node_name} is a broadcast node to set all chiplets.")
                        dummy_set_node = BingoNode(
                            assigned_chiplet_id=cur_node.assigned_chiplet_id,
                            assigned_cluster_id=cur_node.assigned_cluster_id,      # must be the same type of the cur_node to block the execution
                            assigned_core_id=cur_node.assigned_core_id,            # must be the same type of the cur_node to block the execution
                            kernel_name= None
                        )
                        dummy_set_node.node_type = "dummy"
                        dummy_set_node.dep_set_enable = True
                        dummy_set_node.dep_set_list = [remote_succ_list[0].assigned_core_id]
                        dummy_set_node.dep_set_cluster_id = remote_succ_list[0].assigned_cluster_id
                        dummy_set_node.dep_set_chiplet_id = remote_succ_list[0].assigned_chiplet_id # should be fine since it is a broadcast type
                        dummy_set_node.dep_check_enable = False
                        dummy_set_node.dep_check_list = []
                        dummy_set_node.remote_dep_set_all = True
                        # Add the dummy set node to the graph
                        for remote_succ in remote_succ_list:
                            self.bingo_insert_node_between(cur_node, remote_succ, dummy_set_node)
                else:
                    # Now the normal case
                    for remote_succ in remote_succ_list:
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
        '''Transform the DFG to add dummy check nodes.'''
        # The idea of the dummy check nodes is to solve the problem of this kind
        #         dma(Cl0)    dma(Cl1)
        #          |           |
        #           \         /
        #            v       v
        #            gemm(Cl0) <-cur node
        # that a node depends on two (more than 1) nodes with same assigned core
        for cur_node in self.node_list:
            # find all the predecessors
            preds_list = [
                pred for pred in self.predecessors(cur_node)
            ]
            # if there are more than 1 predecessor with the same assigned core id, we need to add dummy check nodes
            # find if there are more than 1 predecessor with same assigned core
            predecessor_core_dict = {}
            for pred in preds_list:
                if pred.assigned_core_id not in predecessor_core_dict:
                    predecessor_core_dict[pred.assigned_core_id] = []
                predecessor_core_dict[pred.assigned_core_id].append(pred)
            dummy_check_nodes_to_add = []
            for core_id, preds in predecessor_core_dict.items():
                if len(preds) >= 2:
                    print(f"Adding dummy check node for {cur_node.node_name} with predecessors {[pred.node_name for pred in preds]}")
                    for i in range(len(preds)-1):
                        dummy_check_node = BingoNode(
                            assigned_chiplet_id=cur_node.assigned_chiplet_id,
                            assigned_cluster_id=cur_node.assigned_cluster_id, # should be fine since it will not be executed
                            assigned_core_id=cur_node.assigned_core_id,       # should be the same type of the cur_node to block the execution
                            kernel_name= None
                        )
                        dummy_check_node.node_type = "dummy"
                        dummy_check_node.dep_check_enable = True
                        dummy_check_node.dep_check_list = [preds[i].assigned_core_id]
                        dummy_check_node.dep_set_enable = False
                        dummy_check_node.dep_set_list = []
                        dummy_check_node.dep_set_cluster_id = 0
                        dummy_check_node.dep_set_chiplet_id = 0
                        dummy_check_node.remote_dep_set_all = False
                        dummy_check_nodes_to_add.append(dummy_check_node)
                        # Add the dummy check node to the graph
                        self.bingo_insert_node_between(preds[i], cur_node, dummy_check_node)
            # Temporary solution
            # For the other predecessors, we connect them to the dummy check node
            for core_id, preds in predecessor_core_dict.items():
                if len(preds) == 1:
                    if dummy_check_nodes_to_add:
                        # Find the dummy dep check nodes with its predecessors is differnet from this predecessor
                        best_dummy_nodes = [dummy_node for dummy_node in dummy_check_nodes_to_add
                                           if dummy_node.dep_check_list[0] != preds[0].assigned_core_id]
                        if not best_dummy_nodes:
                            print("Warning: Cannot find suitable dummy check node to connect the predecessor, this is unexpected!")
                        best_dummy_node = random.choice(best_dummy_nodes)
                        self.remove_edge(preds[0], cur_node)
                        self.add_edge(preds[0], best_dummy_node)
                        best_dummy_node.dep_check_list.append(preds[0].assigned_core_id)

    def bingo_assign_normal_node_dep_check_info(self) -> None:
        """Assign the dep check info for normal nodes."""
        # Iterate over all nodes in the graph
        for cur_node in self.node_list:
            # Check if the node's task_type is "normal"
            if cur_node.node_type == "normal":
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
        """Assign the dep set info for normal nodes."""
        # Iterate over all nodes in the graph
        for cur_node in self.node_list:
           # Check if the node's task_type is "normal"
           if cur_node.node_type == "normal":
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
        
        # 1. task_type (1 bit)
        # 0: Normal, 1: Dummy
        task_type_val = 1 if node.node_type == "dummy" else 0
        packed_val |= (task_type_val << current_shift)
        current_shift += 1
        
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
        
        # 1. task_type (1 bit)
        fields['task_type'] = (packed_val >> current_shift) & 0x1
        current_shift += 1
        
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

        # Try to use a hierarchical layout (multipartite) based on topological generations
        # This gives a much clearer structure for DFGs compared to BFS or Spring
        try:
            # multiple start nodes are possible
            for layer, nodes in enumerate(nx.topological_generations(self)):
                for node in nodes:
                    self.nodes[node]["layer"] = layer
            # align='vertical' means layers are vertical strings (x=constant), flow is Left->Right
            pos = nx.multipartite_layout(self, subset_key="layer", align="vertical", scale=4)
        except Exception as e:
            print(f"Warning: Could not use multipartite_layout ({e}), falling back to spring_layout")
            # Fallback if cyclic or other error
            pos = nx.spring_layout(self, k=2, iterations=100, seed=42)

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

        # Draw nodes with different shapes
        for shape, nodes in node_shapes.items():
            nx.draw_networkx_nodes(
                self, pos, nodelist=nodes,
                node_shape=shape,
                node_color=[node_colors[node] for node in nodes],
                node_size=500,
                ax=ax_graph
            )

        # Draw edges
        nx.draw_networkx_edges(self, pos, ax=ax_graph)

        # Draw labels
        labels = {}
        for node in self.nodes:
            # Simplified label: just the ID
            label_string = f"{node.node_id}"
            labels[node] = label_string
        nx.draw_networkx_labels(self, pos, labels=labels, font_size=8, ax=ax_graph)

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
            task_description_list += f"uint32_t num_tasks_chip_{chiplet_id:02x} = {num_local_nodes};\n"

            if num_local_nodes == 0:
                 # Even if size is 0, we allocate 1 element to avoid issues with size 0 allocation if allocator doesn't support it, or just use 0.
                 # Using 1 for safety, similar to original array [1]
                 task_description_list += f"uint64_t* task_desc_list_chip_{chiplet_id:02x} = (uint64_t*)bingo_l3_alloc(0x{chiplet_id:02x}, 1 * sizeof(uint64_t));\n"
                 task_description_list += f"task_desc_list_chip_{chiplet_id:02x}[0] = 0;\n"
            else:
                task_description_list += f"uint64_t* task_desc_list_chip_{chiplet_id:02x} = (uint64_t*)bingo_l3_alloc(0x{chiplet_id:02x}, num_tasks_chip_{chiplet_id:02x} * sizeof(uint64_t));\n"
                for idx, node in enumerate(local_nodes):
                    packed_val = self.bingo_pack_node(node)
                    fields = self.bingo_unpack_node(packed_val)
                    
                    # Create a detailed comment
                    comment = f"// Node ID {node.node_id}\n"
                    comment += f"    // Fields: Type={fields['task_type']}, TaskID={fields['task_id']}\n"
                    comment += f"    //         Assigned: Chiplet={fields['assigned_chiplet_id']:02x}, Cluster={fields['assigned_cluster_id']}, Core={fields['assigned_core_id']}\n"
                    comment += f"    //         DepCheck: En={fields['dep_check_en']}, Code=0b{fields['dep_check_code']:0{self.num_cores_per_cluster}b}\n"
                    comment += f"    //         DepSet:   En={fields['dep_set_en']}, All={fields['dep_set_all']}, Chiplet={fields['dep_set_chiplet_id']:02x}, Cluster={fields['dep_set_cluster_id']}, Code=0b{fields['dep_set_code']:0{self.num_cores_per_cluster}b}"
                    
                    task_description_list += f"task_desc_list_chip_{chiplet_id:02x}[{idx}] = 0x{packed_val:016X}; {comment}\n"
            
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
        """Emit initialization for task arguments."""
        f.write("        // 3. Task Arguments Init\n")
        
        local_nodes = [node for node in sorted_nodes if node.assigned_chiplet_id == chiplet_id]
        
        dev_task_idx = 0
        host_task_idx = 0
        
        for node in local_nodes:
            kernel_name = node.kernel_name
            is_device = kernel_name and kernel_name.startswith("__snax")
            is_host = kernel_name and kernel_name.startswith("__host")
            
            if not (is_device or is_host):
                continue
            
            f.write(f"        // Node ID: {node.node_id} {node.node_name} ({kernel_name})\n")
            
            args_struct_type = ""
            if node.kernel_args:
                args_struct_type = node.kernel_args.get_struct_name()
            
            if is_device:
                args_var = f"args_dev_chip{chiplet_id:02x}_{node.node_id}"
                
                if node.kernel_args:
                    # Allocate args in the assigned cluster's L1 memory
                    f.write(f"        {args_struct_type}* {args_var} = ({args_struct_type}*)bingo_l1_alloc(0x{chiplet_id:02x}, {node.assigned_cluster_id}, sizeof({args_struct_type}));\n")
                    # Populate fields
                    field_assignments = node.kernel_args.get_c_field_assignments(handle_name_map)
                    for field, value in field_assignments.items():
                            f.write(f"        {args_var}->{field} = {value};\n")
                    f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = (uint32_t)(uintptr_t){args_var};\n")
                else:
                    # Here is leaved for the special kernels
                    # Now only the exit kernel needs this
                    if "exit" in kernel_name:
                        # Exit args allocated in assigned cluster's L1
                        f.write(f"        __snax_bingo_kernel_exit_args_t* {args_var} = (__snax_bingo_kernel_exit_args_t*)bingo_l1_alloc(0x{chiplet_id:02x}, {node.assigned_cluster_id}, sizeof(__snax_bingo_kernel_exit_args_t));\n")
                        f.write(f"        {args_var}->exit_code = 0;\n")
                        f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = (uint32_t)(uintptr_t){args_var};\n")
                    else:
                        f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = 0;\n")
                
                f.write(f"        device_kernel_list_chip_{chiplet_id:02x}[{dev_task_idx}] = (uint32_t)(uintptr_t)get_device_function(\"{kernel_name}\");\n")
                dev_task_idx += 1
                
            elif is_host:
                args_var = f"args_host_chip{chiplet_id:02x}_{node.node_id}"
                
                if node.kernel_args:
                    f.write(f"        {args_struct_type}* {args_var} = ({args_struct_type}*)bingo_l3_alloc(0x{chiplet_id:02x}, sizeof({args_struct_type}));\n")
                    field_assignments = node.kernel_args.get_c_field_assignments(handle_name_map)
                    for field, value in field_assignments.items():
                            f.write(f"        {args_var}->{field} = {value};\n")
                    f.write(f"        host_arg_list_chip_{chiplet_id:02x}[{host_task_idx}] = (uint64_t)(uintptr_t){args_var};\n")
                else:
                    if "exit" in kernel_name:
                        f.write(f"        __host_bingo_kernel_exit_args_t* {args_var} = (__host_bingo_kernel_exit_args_t*)bingo_l3_alloc(0x{chiplet_id:02x}, sizeof(__host_bingo_kernel_exit_args_t));\n")
                        f.write(f"        {args_var}->exit_code = 0;\n")
                        f.write(f"        host_arg_list_chip_{chiplet_id:02x}[{host_task_idx}] = (uint64_t)(uintptr_t){args_var};\n")
                    else:
                        f.write(f"        host_arg_list_chip_{chiplet_id:02x}[{host_task_idx}] = 0;\n")
                
                f.write(f"        host_kernel_list_chip_{chiplet_id:02x}[{host_task_idx}] = (uint64_t)(uintptr_t)&{kernel_name};\n")
                host_task_idx += 1
    
    def _emit_scheduler_launch(self, f, chiplet_id):
        """Emit the scheduler initialization and launch calls."""
        f.write("\n")
        f.write('        printf_safe("Chip(%x, %x): [Host] Init HW Bingo Scheduler\\r\\n",\n')
        f.write('               get_current_chip_loc_x(), get_current_chip_loc_y());\n\n')

        f.write(f"        bingo_hw_scheduler_init((uint64_t)(uintptr_t)device_arg_list_chip_{chiplet_id:02x},\n")
        f.write(f"                                (uint64_t)(uintptr_t)device_kernel_list_chip_{chiplet_id:02x},\n")
        f.write(f"                                num_dev_tasks_chip_{chiplet_id:02x},\n")
        f.write(f"                                (uint64_t)(uintptr_t)global_task_id_to_dev_task_id_chip_{chiplet_id:02x},\n")
        f.write(f"                                (uint64_t)(uintptr_t)task_desc_list_chip_{chiplet_id:02x},\n")
        f.write(f"                                num_tasks_chip_{chiplet_id:02x});\n\n")
        
        f.write(f"        uint32_t err = bingo_hw_scheduler(host_arg_list_chip_{chiplet_id:02x},\n")
        f.write(f"                                          host_kernel_list_chip_{chiplet_id:02x},\n")
        f.write(f"                                          global_task_id_to_host_task_id_chip_{chiplet_id:02x});\n")

    def bingo_emit_offload_c_code(self, extra_include_header_list: list[str], output_path: str) -> None:
        """Emit the offload_hw_bingo.h file with kernel_execution logic."""
        
        # 1. Collect Handles
        sorted_nodes = sorted(self.node_list, key=lambda n: n.node_id)
        sorted_handles, handle_name_map = self._collect_memory_handles(sorted_nodes)
        
        # 2. Start emitting C code
        with open(output_path, "w") as f:
            # Step 1: Emit Headers
            self._emit_headers(f, extra_include_header_list)
            
            # Step 2: Emit Debug Kernel List
            self._emit_debug_kernel_list(f)

            # Step 3: Emit kernel_execution function structure
            f.write("int kernel_execution(){\n")
            f.write("    check_kernel_tab_ready();\n")
            f.write("    uint32_t current_chip_id = get_current_chip_id();\n\n")

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

                f.write("    }\n")
            
            f.write("    return 0;\n")
            f.write("}\n")
    def bingo_compile_dfg(self, output_dir: str, output_file_name: str, extra_include_header_list: list[str]) -> None:
        """Compile the DFG by assigning dep info and emitting C code."""
        # 1. Transformations
        # Add Entry Node
        self.bingo_transform_dfg_add_entry_node()
        # Add Exit Nodes
        self.bingo_transform_dfg_add_exit_nodes()
        self.bingo_visualize_dfg(
            os.path.join(output_dir, "dfg_with_entry_exit_nodes")
        )
        # Add Dummy Set/Check Nodes
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
            output_path=os.path.join(output_dir, output_file_name),
            extra_include_header_list=extra_include_header_list,
        )