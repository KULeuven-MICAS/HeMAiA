# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The DFG the mini-compiler builds and then compiles into a BINGO task-descriptor list.
#
# WHAT IS HERE: the graph itself -- construction, node and edge insertion -- and
# bingo_compile_dfg, which is the running order of the passes. That order is the contract
# between them and is the reason this file is worth reading first.
#
# WHAT IS NOT: the passes themselves. Each concern is a mixin in its own module, and BingoDFG
# inherits all of them, so `self` is still the whole DFG and any pass can call any other. The
# split is for reading and reviewing, not isolation:
#
#   bingo_dfg_transforms   entry/exit and dummy nodes, core sequencing, dep tags
#   bingo_dfg_conditional  conditional regions and CERF groups
#   bingo_dfg_descriptor   the packed descriptor layout, pack and unpack
#   bingo_dfg_validate     the passes that refuse a graph that would hang
#   bingo_dfg_emit         the generated C the host compiles against
#   bingo_dfg_report       the graph picture and the CSV dump
#   bingo_dfg_staticl1     static L1 placement

import math
import os

try:
    import networkx as nx
except ImportError:
    print("networkx not found. Installing...")
    from bingo_utils import install_package
    install_package("networkx")
    import networkx as nx

from bingo_utils import DiGraphWrapper
from bingo_node import BingoNode
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView
from bingo_kernel_args import BingoKernelArgs

# Re-exported: these were defined in this module once, so they stay importable from here.
# Their homes are now where each is actually USED -- the kernel/engine map beside the
# core-role map it is checked against, the task-list word size beside the descriptor that
# is measured in those words. bingo_dfg_common.py held them only to break an import cycle
# with the mixins, and it no longer needs to: neither of those modules imports bingo_dfg.
from bingo_platform import _ENGINE_BY_KERNEL_TOKEN, _engine_of_kernel   # noqa: F401
from bingo_dfg_descriptor import (                                      # noqa: F401
    BINGO_TASK_LIST_WORD_BITS,
    BINGO_TASK_LIST_WORD_MASK,
)
from bingo_utils import install_package                                 # noqa: F401

from bingo_dfg_transforms import BingoDFGTransformsMixin
from bingo_dfg_conditional import BingoDFGConditionalMixin
from bingo_dfg_descriptor import BingoDFGDescriptorMixin
from bingo_dfg_validate import BingoDFGValidateMixin
from bingo_dfg_emit import BingoDFGEmitMixin
from bingo_dfg_report import BingoDFGReportMixin
from bingo_dfg_staticl1 import BingoDFGStaticL1Mixin


class BingoDFG(
    BingoDFGTransformsMixin,
    BingoDFGConditionalMixin,
    BingoDFGDescriptorMixin,
    BingoDFGValidateMixin,
    BingoDFGEmitMixin,
    BingoDFGReportMixin,
    BingoDFGStaticL1Mixin,
    DiGraphWrapper[BingoNode],
):
    """Data Flow Graph (DFG) for Bingo."""

    def __init__(self,
                 num_chiplets: int,
                 num_clusters_per_chiplet: int,
                 num_cores_per_cluster: int,
                 is_host_as_acc: bool,
                 chiplet_ids: list[int] = None,
                 dep_tag_width: int = None,
                 task_desc_width: int = None,
                 chip_id_width: int = None,
                 task_id_width: int = None) -> None:
        super().__init__()
        # HW architecture parameters
        self.num_chiplets = num_chiplets
        self.num_clusters_per_chiplet = num_clusters_per_chiplet
        self.num_cores_per_cluster = num_cores_per_cluster + 1 if is_host_as_acc else num_cores_per_cluster
        self.is_host_as_acc = is_host_as_acc
        assert num_chiplets == len(chiplet_ids) or chiplet_ids is None, "Length of chiplet_ids must match num_chiplets"
        self.chiplet_ids = chiplet_ids if chiplet_ids else list(range(num_chiplets))
        # Identity-aware dependencies. Must match the hw_manager's EnableTaggedDeps
        # / DepTagWidth parameters. When enabled, bingo_compile_dfg runs the spill +
        # per-edge tag-allocation passes; the tag bits are always present in the
        # packed descriptor (= 0 when disabled), matching the RTL struct layout.
        # MULTI-COLUMN CHECK: one descriptor checks every producer column of
        # a join instead of a chain of dummy_check tasks. The dep matrix already
        # AND-reduces dep_check_code at one tag and clears only on a full match,
        # so this needs no RTL change -- only a shared tag across the join's
        # producers, which the tag allocator's group path provides.
        self.enable_multi_col_check = False
        self.enable_tagged_deps = True
        # ------------------------------------------------------------------
        # Descriptor geometry: the three widths that decide where every field of the
        # packed descriptor lands and how far apart two descriptors sit in the task list.
        #
        #   dep_tag_width    cfg s1_quadrant.dep_tag_width    -> BINGO_DEP_TAG_WIDTH
        #   task_desc_width  cfg s1_quadrant.task_desc_width  -> BINGO_TASK_DESC_WIDTH
        #                    (derived per config when the cfg does not pin it)
        #   chip_id_width    cfg hemaia_multichip.chip_id_width -> BINGO_CHIP_ID_WIDTH
        #
        # None of them is a constant across the in-tree configs, and none of the ~50
        # workload generators passes them, so the DEFAULT has to come from the generated
        # occamy.h -- the header the RTL and the C runtime of this very build were made
        # from. Hardcoding any of them is silent, not loud: a task list packed 64 bits
        # wide against a 128-bit RTL descriptor strides the array by one word instead of
        # two, so every descriptor after the first is fetched from the wrong address, and
        # a wrong tag or chiplet width shifts fields inside each descriptor instead.
        # Passing a value explicitly overrides the header, for a caller that is
        # deliberately packing for a platform other than the one it parsed.
        if (dep_tag_width is None or task_desc_width is None or chip_id_width is None
                or task_id_width is None):
            from bingo_platform import platform_descriptor_geometry
            geometry, geometry_source = platform_descriptor_geometry()
        else:
            geometry, geometry_source = {}, "explicit BingoDFG arguments"
        # Where each width came from, quoted in the emitted task list and in the
        # overflow errors so a mismatch names the header instead of a bare number.
        self.platform_geometry_source = geometry_source
        self.dep_tag_width = (geometry["dep_tag_width"]
                              if dep_tag_width is None else dep_tag_width)
        self.chip_id_width = (geometry["chip_id_width"]
                              if chip_id_width is None else chip_id_width)
        # TaskIdWidth, from the same header. The id space is 2**width and bingo_add_node
        # hands out one id per node, so this caps the graph size; it is also a descriptor
        # field, so a disagreement with the RTL shifts every field above it.
        self.task_id_width = (geometry["task_id_width"]
                              if task_id_width is None else task_id_width)
        self.task_desc_width = (geometry["task_desc_width"]
                                if task_desc_width is None else task_desc_width)
        self.task_desc_words = (self.task_desc_width + BINGO_TASK_LIST_WORD_BITS - 1) \
            // BINGO_TASK_LIST_WORD_BITS
        # Node ID counter
        # Make sure the node id is starts from 0
        self.id = -1
        self._next_cerf_group = 0
        self._gating_cerf_mappings: dict = {}  # gating_node → {expert_idx: cerf_group_id}
        self._cond_forks: list = []            # declared BingoConditionalFork objects
        self._declared_combines: dict = {}     # combine node → the fork it closes
        self._gating_to_targets: dict = {}     # gating node → its conditional targets

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

    def bingo_compile_dfg(self, app_name: str, output_dir: str, output_file_name: str, extra_include_header_list: list[str] | None, post_execute_code: list[str] | None = None, static_l1=False, desc_list_in_narrow_spm: bool = True,
                         sim_check: bool = True) -> None:
        """Compile the DFG by assigning dep info and emitting C code.

        `static_l1` is False by default: buffers keep their runtime `bingo_l1_alloc` handles
        and nothing about the emitted addresses changes. Pass True to let the compiler place
        them, or a StaticL1Options for the tuning knobs. A workload opts in explicitly --
        see bingo_plan_static_l1 for why this is an argument and not an environment variable.

        `desc_list_in_narrow_spm` defaults to True: the task-descriptor list goes in the
        narrow SPM, which keeps descriptor fetches off the wide path. Pass False for the
        wide SPM. It stays a knob because the narrow SPM is small and shared with the host
        stack, so a DFG large enough to overflow it must be able to opt out.
        """
        self.desc_list_in_narrow_spm = desc_list_in_narrow_spm
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
        self._validate_cerf_core_sharing()
        # Identity-aware deps: per-edge tags are allocated LAST (after dep-info
        # assignment). The allocator's min-chain-cover reuses a tag across edges
        # that can never be live together (happens-before / same-core order), so
        # no separate concurrency-bounding pass is needed. The legacy
        # serialize_shared_counter_consumers mitigation was removed -- per-edge
        # tags supersede it. (Untagged mode has no counter-sharing mitigation.)
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
        # Identity-aware deps: allocate per-edge tags LAST, once every set/check
        # op (incl. dummies) is final. Packed into the descriptor by bingo_pack_node.
        if self.enable_tagged_deps:
            self.bingo_transform_dfg_allocate_dep_tags(tag_width=self.dep_tag_width)

        # COMPILE-TIME HANG CHECK. Validates the LOWERED graph rather than
        # trusting the passes that produced it: a stuck dep-check does not
        # raise, time out or corrupt anything at runtime -- the machine simply
        # stops -- so it has to be caught here or not at all.
        _hang = self.bingo_validate_no_hang(tag_width=self.dep_tag_width)
        print(f"Hang check passed: {_hang['edges']} dep edges over "
              f"{_hang['cells']} cells, peak {_hang['peak_tags_per_cell']} "
              f"tags/cell of {_hang['tag_capacity']} available, "
              f"{_hang['cross_die_gated']} cross-die gated target(s)")

        # CYCLE-MODEL CHECK, ON BY DEFAULT. The static check above cannot see
        # anything that depends on TIMING -- queue depths, arbiter order, a
        # cross-die predicate racing its dep-set. This pushes the real descriptor
        # list through the cycle model instead.
        #
        # It is on by default because it is cheap and the failure it catches is
        # the expensive one: measured at +0.2-0.6 s on 4-15 s workload builds
        # (~4%), against a hang that produces no error, no timeout and no
        # corruption -- the machine simply stops. It also degrades rather than
        # breaking: a missing model or a bridge problem is reported and skipped,
        # and only a genuine hang fails the build.
        #
        # Set BINGO_SIM_CHECK=0 to turn it off; the env var wins over the
        # argument either way.
        _sim_env = os.environ.get("BINGO_SIM_CHECK")
        _do_sim = sim_check if _sim_env is None else (_sim_env == "1")
        if _do_sim:
            try:
                from bingo_sim_check import simulate_for_hangs, ModelUnavailable
            except ImportError as _exc:
                print(f"Sim hang check SKIPPED: {_exc}")
            else:
                try:
                    _sim = simulate_for_hangs(self)
                except ModelUnavailable as _exc:
                    print(f"Sim hang check SKIPPED: {_exc}")
                except ValueError:
                    raise            # a real hang -- this MUST fail the build
                except Exception as _exc:
                    # Any other failure is a problem with the bridge or a model
                    # version skew. Report it; do not break firmware generation.
                    print(f"Sim hang check SKIPPED ({type(_exc).__name__}): {_exc}")
                else:
                    _sc = _sim.get('scenarios', ['all'])
                    print(f"Sim hang check passed: {_sim['descriptors']} descriptors, "
                          f"{_sim['real_tasks']} dispatching tasks, "
                          f"{_sim['dep_edges']} dependency edges verified over "
                          f"{_sim['seeds']} seeds x {len(_sc)} routing "
                          f"scenario(s) {_sc} (model: {_sim['model']})")

        # 1b. STATIC L1 ALLOCATION -- analysis and, when enabled, placement.
        #
        # Runs AFTER every transform, because the dummy set/check nodes those passes
        # insert are real nodes with real ordering, and liveness computed before them
        # would see a sparser graph than the hardware actually executes.
        self.bingo_plan_static_l1(static_l1, output_dir=output_dir)

        # 2. Emit C Code
        self.bingo_emit_offload_c_code(
            app_name=app_name,
            output_path=os.path.join(output_dir, output_file_name),
            extra_include_header_list=extra_include_header_list,
            post_execute_code=post_execute_code,
        )
