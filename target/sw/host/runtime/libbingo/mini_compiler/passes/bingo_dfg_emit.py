# Fanchen Kong <fanchen.kong@kuleuven.be>

from bingo_mem_handle import BingoMemAlloc
from bingo_mem_handle import BingoMemAllocView



# ======================================================================================
# ARGS-STRUCT COMPLETENESS
#
# Every args struct is carved out of the BINGO L1 arena, which is NEVER CLEARED. A field
# the args class forgets to assign is therefore read as whatever that TCDM word last held
# -- and depending on the field that is a wrong answer, a dead hart, or a task that stalls
# forever with no error (anything feeding an AGU stride: see src_row_stride in
# __snax_bingo_kernel_simd_stream_elementwise_args_t, which hung four workloads before
# this check existed).
#
# So the field list is checked against the C header the device compiles against, at emit
# time, where the fix is one line in the args class. The alternative is finding it as a
# silent hang in RTL.

_ARGS_TRAILER_FIELDS = frozenset(
    {"gating_sp_addr", "cond_node_index", "scratchpad_ptr", "pred_scratchpad_addr"})
_struct_fields_cache = None


def _args_struct_fields():
    """{struct_name: [field, ...]} parsed from the kernel-args C headers."""
    global _struct_fields_cache
    if _struct_fields_cache is not None:
        return _struct_fields_cache
    import os
    import re
    from _bingo_paths import repo_root
    inc = os.path.join(str(repo_root()),
                       "target/sw/host/runtime/libbingo/include/libbingo")
    out = {}
    for name in ("device_kernel_args.h", "host_kernel_args.h"):
        try:
            with open(os.path.join(inc, name)) as fh:
                src = fh.read()
        except OSError:
            continue                  # header absent: skip rather than block a build
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        src = re.sub(r"//[^\n]*", "", src)
        for body, struct in re.findall(
                r"(?:__SNAX_KERNEL_ARGS_DEFINE|__HOST_KERNEL_ARGS_DEFINE)\s*\w*\s*"
                r"\{([^{}]*)\}\s*(\w+_args_t)\s*;", src, re.S):
            out[struct] = re.findall(r"\b(?:uint|int)(?:8|16|32|64)_t\s+(\w+)\s*;", body)
    _struct_fields_cache = out
    return out


def _validate_args_structs(nodes):
    """Every node's args struct must assign every field its C struct declares."""
    problems = []
    for n in nodes:
        args = getattr(n, "kernel_args", None)
        if args is None:
            continue
        try:
            fields = set(args.get_c_field_assignments({}))
        except Exception:
            continue          # needs the real handle map; the emit path will surface it
        problems.extend(_args_complete_problems(args.get_struct_name(), fields, n))
    if problems:
        raise ValueError("Kernel args struct check failed before C generation:\n  "
                         + "\n  ".join(problems))


def _args_complete_problems(struct_name, assignments, node):
    """[] when every declared field is assigned, else one message per struct."""
    declared = _args_struct_fields().get(struct_name)
    if not declared:
        return []
    missing = [f for f in declared
               if f not in assignments and f not in _ARGS_TRAILER_FIELDS]
    if not missing:
        return []
    return [f"{struct_name} (node {getattr(node, 'node_name', '?')}): "
            f"get_c_field_assignments() does not set {', '.join(missing)}. Every field "
            f"must be assigned -- the struct lives in the BINGO L1 arena, which is never "
            f"cleared, so an unassigned field reads stale TCDM. Set it explicitly, even "
            f"when the value is 0."]


def _check_args_complete(struct_name, assignments, node):
    """Refuse an args struct with a field nobody assigned."""
    declared = _args_struct_fields().get(struct_name)
    if not declared:
        return                        # struct not found in the headers: nothing to check
    missing = [f for f in declared
               if f not in assignments and f not in _ARGS_TRAILER_FIELDS]
    if missing:
        raise ValueError(
            f"{struct_name} (node {getattr(node, 'node_name', '?')}): "
            f"get_c_field_assignments() does not set {', '.join(missing)}. Every field "
            f"must be assigned -- the struct lives in the BINGO L1 arena, which is never "
            f"cleared, so an unassigned field reads stale TCDM. Set it explicitly, even "
            f"when the value is 0.")


class BingoDFGEmitMixin:
    """Emission of the generated C header the host compiles against.

    Everything the runtime needs to start the graph: the descriptor list, the task-id
    mappings, the memory allocations and the scheduler launch.

    Mixed into BingoDFG, so `self` is the whole DFG. These methods call each other
    and the other mixins' methods freely; nothing here is meant to stand alone.
    """

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
        """Emit the task description list in the DFG.

        The list stays a uint64_t array, but a descriptor is task_desc_words of them:
        word 0 at the lower address holds the LOW bits, because the RTL fetch master
        reads beat 0 from the lower address into the low bits of the descriptor. Both
        the stride and that order are load-bearing -- get either wrong and the scheduler
        still runs, it just dispatches shuffled descriptors.
        """
        words = self.task_desc_words

        # The indices below stride by BINGO_TASK_DESC_WORDS and the constants are packed
        # to the field offsets `words` and the widths above imply. This file includes
        # libbingo/bingo_api.h -> bingo_utils.h -> occamy.h, so BOTH macros are already
        # defined by the time these lines are compiled: a #define here can never take
        # effect and a #ifndef around it can never fire. What IS worth emitting is the
        # agreement check, because a disagreement is otherwise completely silent -- the
        # C runtime walks the array with its stride, the packer wrote it with a different
        # one, and the scheduler simply fetches and dispatches garbage. That happens
        # whenever the generated header is regenerated for another CFG without rerunning
        # main_bingo.py, which is exactly the build state `make sw` leaves behind when a
        # workload's generator inputs have not changed.
        checks = [
            ("BINGO_TASK_DESC_WIDTH", self.task_desc_width),
            ("BINGO_TASK_DESC_WORDS", words),
            ("BINGO_DEP_TAG_WIDTH", self.dep_tag_width),
            ("BINGO_CHIP_ID_WIDTH", self.chip_id_width),
            ("BINGO_NCORES_HW", self.ncores_hw),
            # The one check that covers the whole field table rather than a single knob:
            # bingo_utils.h builds the layout from the same platform defines with its own
            # arithmetic, so equal totals means both derivations agree end to end.
            ("BINGO_TASK_DESC_LAYOUT_BITS", self.bingo_task_desc_bits()),
        ]
        task_description_list = (
            f"// Packed by the mini-compiler against {self.platform_geometry_source}.\n"
            f"// These must match the headers this file is compiled with; see\n"
            f"// libbingo/bingo_utils.h. Regenerate with `make sw` for the active CFG.\n"
        )
        for macro, value in checks:
            task_description_list += (
                f"BINGO_STATIC_ASSERT({macro} == {value},\n"
                f"                    \"bingo task list was packed for a different "
                f"{macro}\");\n")

        # One shared order for the tag allocator and this list -- see bingo_stream_order().
        all_nodes = self.bingo_stream_order()

        chiplets_to_process = [target_chiplet_id] if target_chiplet_id is not None else self.chiplet_ids

        for chiplet_id in chiplets_to_process:
            local_nodes = [node for node in all_nodes if node.assigned_chiplet_id == chiplet_id]
            num_local_nodes = len(local_nodes)
            list_name = f"bingo_hw_scheduler_task_desc_list_chip_{chiplet_id:02x}"
            # The narrow SPM by default: the manager fetches the list over a 64-bit AXI-Lite
            # master, so a read of the WIDE spm is upsized through two crossbars and costs
            # wide-path bandwidth out of proportion to the descriptor it wants. Capacity is
            # why it stays a knob -- see bingo_compile_dfg.
            desc_alloc = ("bingo_l2_alloc" if getattr(self, "desc_list_in_narrow_spm", True)
                          else "bingo_l3_alloc")
            count_name = f"bingo_hw_scheduler_num_task_desc_chip_{chiplet_id:02x}"

            # Emit num_tasks at the beginning. This counts DESCRIPTORS, not words -- it is
            # what the runtime programs into the scheduler's task count.
            task_description_list += f"uint32_t {count_name} = {num_local_nodes};\n"

            if num_local_nodes == 0:
                 # Even if size is 0, we allocate 1 element to avoid issues with size 0 allocation if allocator doesn't support it, or just use 0.
                 # Using 1 for safety, similar to original array [1]
                 task_description_list += (
                     f"uint64_t* {list_name} = (uint64_t*){desc_alloc}(0x{chiplet_id:02x}, "
                     f"1 * BINGO_TASK_DESC_WORDS * sizeof(uint64_t));\n")
                 for w in range(words):
                     task_description_list += f"{list_name}[0 * BINGO_TASK_DESC_WORDS + {w}] = 0x0000000000000000ULL;\n"
            else:
                task_description_list += (
                    f"uint64_t* {list_name} = (uint64_t*){desc_alloc}(0x{chiplet_id:02x}, "
                    f"{count_name} * BINGO_TASK_DESC_WORDS * sizeof(uint64_t));\n")
                for idx, node in enumerate(local_nodes):
                    packed_val = self.bingo_pack_node(node)
                    fields = self.bingo_unpack_node(packed_val)
                    desc_words = self.bingo_task_desc_words(packed_val)

                    # Create a detailed comment
                    comment = f"// Node ID {node.node_id}\n"
                    comment += f"    // Fields: Type={fields['task_type']}, TaskID={fields['task_id']}\n"
                    comment += f"    //         Assigned: Chiplet={fields['assigned_chiplet_id']:02x}, Cluster={fields['assigned_cluster_id']}, Core={fields['assigned_core_id']}\n"
                    comment += f"    //         DepCheck: En={fields['dep_check_en']}, Code=0b{fields['dep_check_code']:0{self.ncores_hw}b}\n"
                    comment += f"    //         DepSet:   En={fields['dep_set_en']}, All={fields['dep_set_all']}, Chiplet={fields['dep_set_chiplet_id']:02x}, Cluster={fields['dep_set_cluster_id']}, Code=0b{fields['dep_set_code']:0{self.ncores_hw}b}"

                    # Word 0 (the low half) carries the comment; the rest follow in
                    # ascending significance. Each word is masked to 64 bits -- formatting
                    # the whole descriptor as one %016X emitted a 32-digit constant that no
                    # C integer type can hold.
                    task_description_list += (
                        f"{list_name}[{idx} * BINGO_TASK_DESC_WORDS + 0] = "
                        f"0x{desc_words[0]:016X}ULL; {comment}\n")
                    for w in range(1, words):
                        task_description_list += (
                            f"{list_name}[{idx} * BINGO_TASK_DESC_WORDS + {w}] = "
                            f"0x{desc_words[w]:016X}ULL;\n")

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

        def collect(value):
            if isinstance(value, BingoMemAlloc):
                unique_handles.add(value)
            # A view is not its own allocation -- collect the buffer it points into, so a
            # base only ever referenced through a view is still allocated.
            elif isinstance(value, BingoMemAllocView):
                unique_handles.add(value.base)
            # A LIST of handles is still a reference to every one of them. Multi-destination
            # kernels (xdma_multicast, xdma_chain_gather) hold their operands this way, and
            # walking only scalar attributes left those buffers unallocated unless some
            # OTHER node happened to name the same handle -- which is how both kernels got
            # away with it so far. That is an accident, not an invariant: a destination no
            # local node reads would silently get no allocation at all.
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        for node in sorted_nodes:
            if node.kernel_args:
                for attr, value in node.kernel_args.__dict__.items():
                    collect(value)
        
        sorted_handles = sorted(list(unique_handles), key=lambda h: h.name)
        handle_name_map = {h: h.get_c_var_name() for h in sorted_handles}
        return sorted_handles, handle_name_map

    def _emit_headers(self, f, extra_include_header_list):
        """Emit C header includes."""
        f.write("// Auto-generated offload_hw_bingo.h\n")
        f.write("#pragma once\n")
        f.write('#include "libbingo/bingo_api.h"\n')
        f.write('#include "host.h"\n')
        # bingo_compile_dfg types this `list[str] | None`, and a workload with no data
        # header passes None -- which used to raise here instead of emitting nothing.
        for include in (extra_include_header_list or ()):
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
        """Emit memory allocation calls for handles on this chiplet.

        With static L1 allocation enabled, L1 buffers stop being individual allocations. The
        compiler has already decided every offset, so the runtime takes ONE allocation per
        cluster -- the packed arena -- and each buffer is a constant displacement into it.

        Taking the arena from bingo_l1_alloc rather than addressing TCDM directly is
        deliberate for this milestone: the heap keeps owning the base address and the
        capacity check, so nothing downstream has to learn a new address map, and a workload
        can be switched back with static_l1=False and no other change.
        """
        placement = getattr(self, "static_l1_placement", None)
        stats = getattr(self, "static_l1_stats", None)
        local_handles = [h for h in sorted_handles if h.chip_id == chiplet_id]

        if placement and stats:
            arenas = sorted(k for k in stats if k[0] == chiplet_id)
            if arenas:
                f.write("        // 1. Static L1 arenas (compiler-packed; one per cluster)\n")
                for (chip, cl) in arenas:
                    st = stats[(chip, cl)]
                    f.write(f"        uint64_t __bingo_l1_static_chip{chip:02x}_cl{cl} = "
                            f"bingo_l1_alloc(0x{chip:02x}, {cl}, {st['peak']});"
                            f"  // {st['count']} buffers packed from {st['sum']} B\n")
                f.write("\n")

        if local_handles:
            f.write("        // 1b. Memory Allocations\n")
            for h in local_handles:
                c_var = handle_name_map[h]
                if placement and h.mem_level == "L1" and id(h) in placement:
                    off = placement[id(h)][1]
                    f.write(f"        uint64_t {c_var} = "
                            f"__bingo_l1_static_chip{h.chip_id:02x}_cl{h.cluster_id} + {off};"
                            f"  // {h.size} B\n")
                    continue
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
        # ---- ARENAS for scratchpads and device args -------------------------------------
        #
        # These were one bingoHeapMalloc per node, so allocation cost scaled with graph size and
        # dominated start-up on any non-trivial DFG. None of it is needed: per-node scratchpads
        # and arg blocks live for the whole run and are never freed individually, so a free-list
        # allocator buys nothing and charges a bin search plus heap-metadata traffic -- the
        # latter crossing the fabric -- on every single one.
        #
        # Instead take ONE block per memory and slice it with a bump pointer. Sizes of the arg
        # structs are only known to the C compiler (sizeof), so the total is emitted as a
        # constant expression and the bump happens at runtime -- a couple of integer ops per
        # node instead of an allocator call.
        #
        # Placement is unchanged: device scratchpads and device args stay in their own cluster's
        # L1, host scratchpads stay in L3. Moving device scratchpads to L3 would turn every
        # device-side access into a fabric round trip.
        #
        # The named workload buffers are allocated BEFORE this point, so collapsing these
        # allocations cannot disturb the cross-chip name/offset agreement they rely on.
        sp_align = "ALIGN_UP(sizeof(bingo_kernel_scratchpad_t), 64)"
        l1_terms = {}   # cluster -> list of C size expressions
        l3_terms = []
        # With slot packing the DEVICE scratchpad term is (slots x sp_align), added below --
        # one term per cluster, not one per node. Emitting both would make the packed arena
        # LARGER than the unpacked one, which is what the first version of this did.
        sp_packed = bool(getattr(self, "static_l1_sp_slots", None))
        for node in local_nodes:
            kn = node.kernel_name
            if kn and kn.startswith("__snax"):
                if not sp_packed:
                    l1_terms.setdefault(node.assigned_cluster_id, []).append(sp_align)
            elif kn and kn.startswith("__host"):
                # host scratchpads live in L3 and are not packed
                l3_terms.append(sp_align)
        # device arg blocks share the cluster's L1 arena
        for node in local_nodes:
            kn = node.kernel_name
            if not (kn and kn.startswith("__snax")):
                continue
            if node.kernel_args:
                t = node.kernel_args.get_struct_name()
            elif "exit" in kn:
                t = "__snax_bingo_kernel_exit_args_t"
            else:
                continue
            l1_terms.setdefault(node.assigned_cluster_id, []).append(f"ALIGN_UP(sizeof({t}), 64)")

        # With slot packing the scratchpad half of the arena is (slots x sp_align) rather than
        # (nodes x sp_align). The ARG half is unchanged and must stay unchanged: arg structs
        # are all written by the host before the scheduler starts, so they are all live at once.
        sp_slots_map = getattr(self, "static_l1_sp_slots", None)
        slots_per_cl = {}
        if sp_slots_map:
            for nd, sl in sp_slots_map.items():
                if nd.assigned_chiplet_id == chiplet_id:
                    slots_per_cl.setdefault(nd.assigned_cluster_id, set()).add(sl)
            for cl, sl in slots_per_cl.items():
                l1_terms.setdefault(cl, []).append(f"{max(sl) + 1} * {sp_align}")

        f.write("        // 3a. One arena per memory for scratchpads and device args (bump-sliced)\n")
        for cl in sorted(l1_terms):
            base = f"__bingo_l1_arena_chip{chiplet_id:02x}_cl{cl}"
            f.write(f"        uint64_t {base} = bingo_l1_alloc(0x{chiplet_id:02x}, {cl},\n"
                    f"            {' + '.join(l1_terms[cl])});\n")
            # THE SLOT BLOCK SITS AT THE FRONT OF THIS ARENA, so the arg bump pointer has to
            # start ABOVE it. Leaving the bump at 0 makes args_dev_*[0] alias slot 0: both
            # regions live in this one arena and both would begin at offset 0. Nothing
            # downstream catches that -- the arena size still adds up, the build is clean, and
            # the only symptom is a task reading its arguments out of another node's
            # scratchpad at run time.
            if cl in slots_per_cl:
                f.write(f"        uint64_t {base}_off = "
                        f"{max(slots_per_cl[cl]) + 1} * {sp_align};\n")
            else:
                f.write(f"        uint64_t {base}_off = 0;\n")
        if l3_terms:
            f.write(f"        uint64_t __bingo_l3_arena_chip{chiplet_id:02x} = bingo_l3_alloc(0x{chiplet_id:02x},\n"
                    f"            {' + '.join(l3_terms)});\n")
            f.write(f"        uint64_t __bingo_l3_arena_chip{chiplet_id:02x}_off = 0;\n")
        f.write("\n")

        f.write("        // 3b. Pre-allocate scratchpads for all tasks\n")
        for node in local_nodes:
            kernel_name = node.kernel_name
            is_device = kernel_name and kernel_name.startswith("__snax")
            is_host = kernel_name and kernel_name.startswith("__host")
            if not (is_device or is_host):
                continue
            if is_device:
                sp_var = f"sp_dev_{node.node_id}"
                base = f"__bingo_l1_arena_chip{chiplet_id:02x}_cl{node.assigned_cluster_id}"
            else:
                sp_var = f"sp_host_{node.node_id}"
                base = f"__bingo_l3_arena_chip{chiplet_id:02x}"
            slot = None
            if is_device and getattr(self, "static_l1_sp_slots", None):
                slot = self.static_l1_sp_slots.get(node)
            if slot is not None:
                # Packed: a slot index the compiler proved is free while this node runs.
                f.write(f"        bingo_kernel_scratchpad_t* {sp_var} = "
                        f"(bingo_kernel_scratchpad_t*)({base} + {slot} * {sp_align});"
                        f"  // slot {slot}\n")
            else:
                f.write(f"        bingo_kernel_scratchpad_t* {sp_var} = "
                        f"(bingo_kernel_scratchpad_t*)({base} + {base}_off);\n")
                f.write(f"        {base}_off += {sp_align};\n")
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

        # Resolve each DISTINCT device kernel name exactly once.
        #
        # get_device_function() linearly scans the device symbol table, so emitting one call per
        # TASK made the host rescan the whole table once per task, while only a handful of names
        # are ever distinct -- most calls re-derive an answer already sitting in a register.
        # Hoisting costs one local per distinct name.
        dev_kernel_names = []
        for node in local_nodes:
            kn = node.kernel_name
            if kn and kn.startswith("__snax") and kn not in dev_kernel_names:
                dev_kernel_names.append(kn)
        if dev_kernel_names:
            n_fn = len(dev_kernel_names)
            f.write("        // Resolve every distinct device kernel in ONE pass over the device\n")
            f.write("        // symbol table (~110 entries). One call per name would walk it once per\n")
            f.write("        // name; get_device_functions tests each entry against all wanted names,\n")
            f.write("        // filtered by a 4-byte signature, and stops once all are found.\n")
            f.write(f"        static const char *const __bingo_fn_names_chip{chiplet_id:02x}[] = {{\n")
            for kn in dev_kernel_names:
                f.write(f"            \"{kn}\",\n")
            f.write("        };\n")
            f.write(f"        uint32_t __bingo_fn_addrs_chip{chiplet_id:02x}[{n_fn}];\n")
            f.write(f"        get_device_functions(__bingo_fn_names_chip{chiplet_id:02x}, "
                    f"__bingo_fn_addrs_chip{chiplet_id:02x}, {n_fn});\n")
            for i, kn in enumerate(dev_kernel_names):
                f.write(f"        const uint32_t __bingo_fn_{kn}_chip{chiplet_id:02x} = "
                        f"__bingo_fn_addrs_chip{chiplet_id:02x}[{i}];\n")
            f.write("\n")

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
                    _ab = f"__bingo_l1_arena_chip{chiplet_id:02x}_cl{node.assigned_cluster_id}"
                    f.write(f"        {args_struct_type}* {args_var} = ({args_struct_type}*)({_ab} + {_ab}_off);\n")
                    f.write(f"        {_ab}_off += ALIGN_UP(sizeof({args_struct_type}), 64);\n")
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
                        _ab = f"__bingo_l1_arena_chip{chiplet_id:02x}_cl{node.assigned_cluster_id}"
                        f.write(f"        __snax_bingo_kernel_exit_args_t* {args_var} = (__snax_bingo_kernel_exit_args_t*)({_ab} + {_ab}_off);\n")
                        f.write(f"        {_ab}_off += ALIGN_UP(sizeof(__snax_bingo_kernel_exit_args_t), 64);\n")
                        f.write(f"        {args_var}->exit_code = 0;\n")
                        f.write(f"        {args_var}->scratchpad_ptr = {sp_cast};\n")
                        f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = (uint32_t)(uintptr_t){args_var};\n")
                    else:
                        f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = 0;\n")

                f.write(f"        device_kernel_list_chip_{chiplet_id:02x}[{dev_task_idx}] = "
                        f"__bingo_fn_{kernel_name}_chip{chiplet_id:02x};\n")
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

    def bingo_emit_offload_c_code(self, extra_include_header_list: list[str], output_path: str, app_name: str, post_execute_code: list[str] | None = None) -> None:
        """Emit the offload_hw_bingo.h file with kernel_execution logic."""
        
        # 1. Collect Handles
        sorted_nodes = sorted(self.node_list, key=lambda n: n.node_id)
        sorted_handles, handle_name_map = self._collect_memory_handles(sorted_nodes)
        self._validate_kernel_core_assignments(sorted_nodes)
        self._validate_memory_handles(sorted_handles)
        # Before the file is opened: a check that raises mid-write leaves a truncated
        # header behind, and the next build reports a C syntax error instead of the
        # actual problem.
        _validate_args_structs(sorted_nodes)

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

        # AUDIT WHAT WAS ACTUALLY WRITTEN. The one failure this catches -- the device-argument
        # bump pointer starting inside the scratchpad slot block -- builds cleanly, passes
        # every upstream check, and corrupts arguments at run time. Checking the artefact is
        # the only place the three tenants of the arena (buffers, slots, args) are visible at
        # once. Refusing to build is deliberate: a wrong layout must never reach a simulation.
        from bingo_l1_packer import check_emitted_arena_text
        with open(output_path) as _f:
            _problems = check_emitted_arena_text(_f.read())
        if _problems:
            for _p in _problems:
                print(f"[static-l1] EMITTED-CODE ERROR: {_p}")
            raise RuntimeError("static L1: emitted arena layout is unsafe, refusing to build")
