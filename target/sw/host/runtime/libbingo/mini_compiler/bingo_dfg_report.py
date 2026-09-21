# Fanchen Kong <fanchen.kong@kuleuven.be>

import networkx as nx

from bingo_dfg_common import install_package


class BingoDFGReportMixin:
    """Debugging output: the graph picture and the CSV dump.

    Neither is on a build path. Both are allowed to fail without taking a build with
    them, because a missing plotting library is not a reason to stop compiling.

    Mixed into BingoDFG, so `self` is the whole DFG. These methods call each other
    and the other mixins' methods freely; nothing here is meant to stand alone.
    """

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
        
        # Name is exported because the runtime trace does NOT carry it: bingo_trace.json
        # records span TYPES, not which node produced them, so attributing a span to a
        # node otherwise means guessing from durations. The task id IS observable -- it is
        # the value a core reads from CSR 0x5fe -- so ID -> Name makes that join exact.
        col_labels = ["ID", "Chiplet", "Cluster", "Core", "Type", "Kernel", "Name"]
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
                node.kernel_name,
                node.node_name
            ]
            table_data.append(row)

        with open(f"{filename}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(col_labels)
            writer.writerows(table_data)
