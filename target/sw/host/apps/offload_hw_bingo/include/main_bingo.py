# Fanchen Kong <fanchen.kong@kuleuven.be>

# This file is the main entry point for the bingo offload application
# Users will create the dfg in this file
# And then the mini-compiler will emit the bingo_workload.h file
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
ROOT_DIR = os.path.normpath(ROOT_DIR)
print("ROOT_DIR:", ROOT_DIR)
sys.path.append(f"{ROOT_DIR}/util/bingo_hw_manager/")
from bingo_dfg import BingoDFG
from bingo_node import BingoNode
# Example Usage
# The single chiplet configuration
num_chiplets = 1
num_clusters_per_chiplet = 1
num_cores_per_cluster = 2
is_host_as_acc = True
bingo_dfg = BingoDFG(num_chiplets, num_clusters_per_chiplet, num_cores_per_cluster, is_host_as_acc)
# Notice the assigned_kernel_name must match the kernel name in snax_kernel_lib.h
# Also make sure the device_kernel_args.h has the corresponding args structure defined as kernel_name_args_t
chiplet0_cluster0_core1_dummy_task_0 = BingoNode( assigned_chiplet_id=0,
                                                assigned_cluster_id=0,
                                                assigned_core_id=1,
                                                assigned_kernel_name="__snax_bingo_kernel_dummy")
chiplet0_cluster0_core0_dummy_task_1 = BingoNode( assigned_chiplet_id=0,
                                                assigned_cluster_id=0,
                                                assigned_core_id=0,
                                                assigned_kernel_name="__snax_bingo_kernel_dummy")
chiplet0_cluster0_core2_dummy_task_2 = BingoNode( assigned_chiplet_id=0,
                                                assigned_cluster_id=0,
                                                assigned_core_id=2,
                                                assigned_kernel_name="__host_bingo_kernel_dummy")
bingo_dfg.bingo_add_node(chiplet0_cluster0_core1_dummy_task_0)
bingo_dfg.bingo_add_node(chiplet0_cluster0_core0_dummy_task_1)
bingo_dfg.bingo_add_node(chiplet0_cluster0_core2_dummy_task_2)
# Define dependencies
bingo_dfg.bingo_add_edge(chiplet0_cluster0_core1_dummy_task_0, chiplet0_cluster0_core0_dummy_task_1)
bingo_dfg.bingo_add_edge(chiplet0_cluster0_core0_dummy_task_1, chiplet0_cluster0_core2_dummy_task_2)
bingo_dfg.bingo_visualize_dfg("original_dfg.png")
# Transform the DFG to add exit nodes
bingo_dfg.bingo_transform_dfg_add_exit_nodes()
bingo_dfg.bingo_visualize_dfg("dfg_with_exit_nodes.png")
# Transform the DFG to add dummy set nodes
bingo_dfg.bingo_transform_dfg_add_dummy_set_nodes()
bingo_dfg.bingo_transform_dfg_add_dummy_check_nodes()
bingo_dfg.bingo_visualize_dfg("final_dfg.png")
# Set the Dep Set and Dep Check for the normal nodes
bingo_dfg.bingo_assign_normal_node_dep_set_info()
bingo_dfg.bingo_assign_normal_node_dep_check_info()

#Emit the bingo_workload.h file
bingo_dfg.bingo_emit_bingo_workload_header("bingo_workload.h")