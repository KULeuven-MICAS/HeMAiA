# Fanchen Kong <fanchen.kong@kuleuven.be>

import random
from bingo_utils import DiGraphWrapper
from bingo_node import BingoNode
import networkx as nx
MAX_NUM_CHIPLETS = 8
class BingoDFG(DiGraphWrapper[BingoNode]):
    """Data Flow Graph (DFG) for Bingo."""

    def __init__(self,
                 num_chiplets: int,
                 num_clusters_per_chiplet: int,
                 num_cores_per_cluster: int,
                 is_host_as_acc: bool) -> None:
        super().__init__()
        # HW architecture parameters
        self.num_chiplets = num_chiplets
        self.num_clusters_per_chiplet = num_clusters_per_chiplet
        self.num_cores_per_cluster = num_cores_per_cluster + 1 if is_host_as_acc else num_cores_per_cluster
        self.is_host_as_acc = is_host_as_acc
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
        
        # first find the end nodes
        end_nodes = [node for node in self.node_list if self.out_degree(node) == 0]
        if len(end_nodes) == 0:
            raise ValueError("No end node found in the DFG.")
        # Now we generate the exit nodes
        exit_nodes = []
        for chiplet_id in range(self.num_chiplets):
            for cluster_id in range(self.num_clusters_per_chiplet):
                for core_id in range(self.num_cores_per_cluster):
                    # Skip the simd core for now
                    if self.is_host_as_acc and core_id == (self.num_cores_per_cluster -1):
                        continue
                    exit_node = BingoNode(
                        assigned_chiplet_id=chiplet_id,
                        assigned_cluster_id=cluster_id,
                        assigned_core_id=core_id,
                        assigned_kernel_name="__snax_bingo_kernel_exit"
                    )
                    self.bingo_add_node(exit_node)
                    exit_nodes.append(exit_node)
        # Finally add the host core exit node
        for chiplet_id in range(self.num_chiplets):
            exit_node = BingoNode(
                assigned_chiplet_id=chiplet_id,
                assigned_cluster_id=0,
                assigned_core_id=self.num_cores_per_cluster -1,
                assigned_kernel_name="__host_bingo_kernel_exit"
            )
            self.bingo_add_node(exit_node)
            exit_nodes.append(exit_node)
        # insert the first exit node after all end nodes
        first_exit_node = exit_nodes[0]
        for end_node in end_nodes:
            self.bingo_add_edge(end_node, first_exit_node)
        # insert the rest exit nodes in serials
        for i in range(1, len(exit_nodes)):
            self.bingo_add_edge(exit_nodes[i-1], exit_nodes[i])
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
                            assigned_kernel_name= None
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
                            assigned_kernel_name= None
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
            if len(local_succ_list)>1:
                # Now the local multiple successor case
                # We need local_successors-1 dummy set nodes
                print(f"Adding dummy set nodes for {cur_node.node_name} with local successors {[succ.node_name for succ in succs_list]}")
                for i in range(len(succs_list)-1):
                    dummy_set_node = BingoNode(
                        assigned_chiplet_id=cur_node.assigned_chiplet_id,
                        assigned_cluster_id=cur_node.assigned_cluster_id,      # must be the same type of the cur_node to block the execution
                        assigned_core_id=cur_node.assigned_core_id,            # must be the same type of the cur_node to block the execution
                        assigned_kernel_name= None
                    )
                    dummy_set_node.node_type = "dummy"
                    dummy_set_node.dep_set_enable = True
                    dummy_set_node.dep_set_list = [succs_list[i].assigned_core_id]
                    dummy_set_node.dep_set_cluster_id = succs_list[i].assigned_cluster_id
                    dummy_set_node.dep_set_chiplet_id = succs_list[i].assigned_chiplet_id
                    dummy_set_node.dep_check_enable = False
                    dummy_set_node.dep_check_list = []
                    dummy_set_node.remote_dep_set_all = False
                    # Add the dummy set node to the graph
                    self.bingo_insert_node_between(cur_node, succs_list[i], dummy_set_node)
                    
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
                            assigned_kernel_name= None
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
    def bingo_visualize_dfg(self, filename: str = "dfg_visualization.png", figsize: tuple = (10, 8)) -> None:
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
            "red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"
        ]

        # Select a start node for BFS layout
        start_node = next(iter(self.nodes), None)  # Get the first node in the graph
        if start_node is None:
            raise ValueError("The graph is empty. Cannot visualize an empty graph.")

        # Create a BFS layout for the graph
        pos = nx.bfs_layout(self, start_node, align="horizontal")

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

        # Set the figure size
        plt.figure(figsize=figsize)

        # Draw nodes with different shapes
        for shape, nodes in node_shapes.items():
            nx.draw_networkx_nodes(
                self, pos, nodelist=nodes,
                node_shape=shape,
                node_color=[node_colors[node] for node in nodes],
                node_size=500
            )

        # Draw edges
        nx.draw_networkx_edges(self, pos)

        # Draw labels
        labels = {}
        for node in self.nodes:
            cur_chiplet_id = node.assigned_chiplet_id
            cur_cluster_id = node.assigned_cluster_id
            cur_core_id = node.assigned_core_id
            cur_kernel_name = node.assigned_kernel_name
            cur_task_type = node.node_type
            if cur_task_type == "dummy":
                if node.dep_set_enable:
                    cur_task_type = "dummy_set"
                elif node.dep_check_enable:
                    cur_task_type = "dummy_check"
            label_string = f"Chiplet{cur_chiplet_id}_Cluster{cur_cluster_id}_Core{cur_core_id}\n"
            label_string += f"{cur_task_type}\n"
            label_string += f"{cur_kernel_name}\n"
            label_string += f"ID{node.node_id}"
            labels[node] = label_string
        nx.draw_networkx_labels(self, pos, labels=labels, font_size=8)

        # Create a legend for task types
        legend_elements = [
            Line2D([0], [0], marker=shape, color="w", label=task_type, markerfacecolor="black", markersize=10)
            for task_type, shape in task_type_shapes.items()
        ]
        plt.legend(handles=legend_elements, loc="best")

        # Save the visualization to a file
        plt.savefig(filename)
        plt.show()
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
        for node in self.node_list:
            kernel_name_list += f'    "{node.assigned_kernel_name}", // Node ID {node.node_id}\n'
        kernel_name_list += "};\n"
        return kernel_name_list



    def bingo_emit_task_desc_list(self) -> str:
        """Emit the task description list in the DFG."""
        task_description_list = ""
        all_nodes = self.node_list
        num_nodes = len(all_nodes)
        # Sort the nodes by node id
        all_nodes.sort(key=lambda x: x.node_id)
        task_description_list += f"uint32_t num_tasks = {num_nodes};\n"
        task_description_list += f"uint64_t task_desc_list[{num_nodes}] = {{\n"
        for node in all_nodes:
            packed_val = self.bingo_pack_node(node)
            fields = self.bingo_unpack_node(packed_val)
            
            # Create a detailed comment
            comment = f"// Node ID {node.node_id}\n"
            comment += f"    // Fields: Type={fields['task_type']}, TaskID={fields['task_id']}\n"
            comment += f"    //         Assigned: Chiplet={fields['assigned_chiplet_id']}, Cluster={fields['assigned_cluster_id']}, Core={fields['assigned_core_id']}\n"
            comment += f"    //         DepCheck: En={fields['dep_check_en']}, Code=0b{fields['dep_check_code']:0{self.num_cores_per_cluster}b}\n"
            comment += f"    //         DepSet:   En={fields['dep_set_en']}, All={fields['dep_set_all']}, Chiplet={fields['dep_set_chiplet_id']}, Cluster={fields['dep_set_cluster_id']}, Code=0b{fields['dep_set_code']:0{self.num_cores_per_cluster}b}"
            
            task_description_list += f"    0x{packed_val:016X}, {comment}\n"
        task_description_list += "};\n"
        return task_description_list
    
    def bingo_emit_task_id_mapping_lists(self) -> str:
        """Emit the mapping lists from global task id to dev/host task id."""
        all_nodes = self.node_list
        num_nodes = len(all_nodes)
        # Sort the nodes by node id
        all_nodes.sort(key=lambda x: x.node_id)
        
        global_to_dev = []
        global_to_host = []
        
        dev_task_counter = 0
        host_task_counter = 0
        
        for node in all_nodes:
            kernel_name = node.assigned_kernel_name
            if kernel_name and kernel_name.startswith("__snax"):
                # It is a device task
                global_to_dev.append(str(dev_task_counter))
                global_to_host.append("-1")
                dev_task_counter += 1
            elif kernel_name and kernel_name.startswith("__host"):
                # It is a host task
                global_to_dev.append("-1")
                global_to_host.append(str(host_task_counter))
                host_task_counter += 1
            else:
                # Neither (e.g. None or other)
                global_to_dev.append("-1")
                global_to_host.append("-1")
                
        mapping_str = ""
        
        # Emit global_task_id_to_dev_task_id
        mapping_str += f"int32_t global_task_id_to_dev_task_id[{num_nodes}] = {{\n"
        mapping_str += "    " + ", ".join(global_to_dev) + "\n"
        mapping_str += "};\n"
        
        # Emit global_task_id_to_host_task_id
        mapping_str += f"int32_t global_task_id_to_host_task_id[{num_nodes}] = {{\n"
        mapping_str += "    " + ", ".join(global_to_host) + "\n"
        mapping_str += "};\n"
        
        # Emit num_dev_tasks and num_host_tasks
        mapping_str += f"uint32_t num_dev_tasks = {dev_task_counter};\n"
        mapping_str += f"uint32_t num_host_tasks = {host_task_counter};\n"
        
        return mapping_str

    def bingo_emit_bingo_workload_header(self, output_path: str) -> None:
        """Emit the bingo_workload.h file."""
        with open(output_path, "w") as f:
            f.write("// Auto-generated bingo_workload.h file, edit the main_bingo.py instead\n\n")
            f.write("#pragma once\n")
            f.write("#include <stdint.h>\n\n")
            # Emit kernel name list
            kernel_name_list_str = self.bingo_emit_task_kernel_name_list()
            f.write("// Kernel Name List\n")
            f.write(kernel_name_list_str)
            f.write("\n")
            # Emit task description list
            task_desc_list_str = self.bingo_emit_task_desc_list()
            f.write("// Task Description List\n")
            f.write(task_desc_list_str)
            f.write("\n")
            # Emit task id mapping lists
            task_id_mapping_str = self.bingo_emit_task_id_mapping_lists()
            f.write("// Task ID Mapping Lists\n")
            f.write(task_id_mapping_str)
            f.write("\n")
