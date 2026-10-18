# Fanchen Kong <fanchen.kong@kuleuven.be>

import re

from bingo_mem_handle import BingoMemAlloc
from bingo_mem_handle import BingoMemAllocView


# ======================================================================================
# COMPILE-TIME DATA
#
# Everything the host would otherwise write word by word before the scheduler starts -- the
# task-id tables, the descriptor list, the constant half of every device args struct -- is
# emitted as initialised static data instead. Static data is part of the program image the
# testbench loads at time zero, so it costs no simulated time; a store from the host costs a
# round trip through the SoC fabric each.

_C_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_C_INT_LITERAL = re.compile(r"\b(0[xX][0-9a-fA-F]+|\d+)(?:[uU]?[lL]{0,2}|[lL]{1,2}[uU]?)\b")
_C_TYPE_WORDS = frozenset({"uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t",
                           "int32_t", "int64_t", "uintptr_t", "intptr_t"})


def _is_c_constant(expr: str) -> bool:
    """True when `expr` is an integer constant expression: literals, casts to the fixed-width
    integer types and arithmetic, with no variable and no call -- a value a static
    initialiser can hold."""
    return all(t in _C_TYPE_WORDS for t in _C_IDENT.findall(_C_INT_LITERAL.sub("0", expr)))


# Words a constant expression over L1 addresses may use besides literals and casts: the
# address macros of bingo_api.h / bingo_alloc.h and the size arithmetic the emitter writes.
_C_STATIC_WORDS = _C_TYPE_WORDS | frozenset({
    "sizeof", "ALIGN_UP", "__builtin_offsetof", "bingo_kernel_scratchpad_t",
    "BINGO_CHIP_ADDR", "BINGO_L1_FIRST_ALLOC", "BINGO_HEAP_FRAGMENT_SIZE"})


# The high half of a program symbol's chip-global address (kernel_base._process_addr on a
# BingoMemSymbol, here or on another chip): the chip prefix alone, as every program symbol lies
# below 4 GiB of its chip's space (host.ld; the generated code checks __wide_spm_end).
_SYMBOL_ADDR_HI = (
    re.compile(r"^\(uint32_t\)\(\(chiplet_addr_transform\(\(uint64_t\)\(\(uintptr_t\)"
               r"[A-Za-z_]\w*(?: \+ \d+)?\)\)\) >> 32\)$"),
    re.compile(r"^\(uint32_t\)\(\(chiplet_addr_transform_full\((?P<chip>0x[0-9a-fA-F]+), "
               r"\(\(uint64_t\)\(\(uintptr_t\)[A-Za-z_]\w*(?: \+ \d+)?\)\) & "
               r"0x000000ffffffffffULL\)\) >> 32\)$"))


def _symbol_addr_hi(expr: str, chiplet_id: int):
    """The constant `expr` is when it is the high half of a program symbol's address, or None."""
    for rx in _SYMBOL_ADDR_HI:
        m = rx.match(expr)
        if m:
            chip = m.groupdict().get("chip") or f"0x{chiplet_id:02x}"
            return f"(uint32_t)(BINGO_CHIP_ADDR({chip}, 0) >> 32)"
    return None


def _c_static_form(expr: str, chiplet_id: int) -> str:
    """`expr` with the chip address transforms spelled as the BINGO_CHIP_ADDR macro, which a
    static initialiser can hold. chiplet_addr_transform is the transform of the chip the code
    runs on, and a chiplet's block only ever runs on that chiplet."""
    expr = expr.replace("chiplet_addr_transform_full(", "BINGO_CHIP_ADDR(")
    return expr.replace("chiplet_addr_transform(", f"BINGO_CHIP_ADDR(0x{chiplet_id:02x}, ")


def _is_c_static(expr: str, names) -> bool:
    """True when `expr` (in _c_static_form) is a constant expression: literals, casts, the
    words of _C_STATIC_WORDS and the macros in `names`."""
    return all(t in _C_STATIC_WORDS or t in names
               for t in _C_IDENT.findall(_C_INT_LITERAL.sub("0", expr)))


def _snax_symbol_sig(name: str) -> int:
    """host.h snax_symbol_sig_of_cstr: bytes [20, 28) of the name, little-endian, zero past its
    end."""
    return int.from_bytes(name.encode()[20:28].ljust(8, b"\0"), "little")


# A run of this many consecutive entries of a chiplet's g2l table is copied by one DMA from
# static data; a shorter one is written by the host entry by entry.
_G2L16_DMA_MIN_RUN = 16


def _g2l16_runs(pairs, num_nodes):
    """Split a chiplet's (global id, local id) pairs of the compact g2l table (bingo_api.h: a
    magic word, then one int16 a task) into DMA runs and single entries.

    Returns (runs, run_vals, singles): `runs` is [table index, entry count] pairs, each run
    8-byte aligned and a whole number of 8 bytes; `run_vals` is their contents back to back,
    with the shared template's values (magic word, -1, zero padding) wherever the run covers
    an entry that is not this chiplet's; `singles` the (global id, local id) pairs no run covers.
    """
    tmpl = {0: "(int16_t)(BINGO_G2L16_MAGIC & 0xFFFFu)", 1: "(int16_t)(BINGO_G2L16_MAGIC >> 16)"}
    pos = {2 + gid: loc for gid, loc in pairs}
    spans = []
    for p in sorted(pos):
        if spans and p == spans[-1][1]:
            spans[-1][1] = p + 1
        else:
            spans.append([p, p + 1])
    blocks = []
    for a, b in spans:
        if b - a < _G2L16_DMA_MIN_RUN:
            continue
        a, b = a & ~3, (b + 3) & ~3
        if blocks and a <= blocks[-1][1]:
            blocks[-1][1] = max(blocks[-1][1], b)
        else:
            blocks.append([a, b])
    runs, run_vals, covered = [], [], set()
    for a, b in blocks:
        runs += [a, b - a]
        for p in range(a, b):
            covered.add(p)
            run_vals.append(str(pos[p]) if p in pos else
                            tmpl.get(p, "-1" if p < 2 + num_nodes else "0"))
    singles = [(p - 2, v) for p, v in sorted(pos.items()) if p not in covered]
    return runs, run_vals, singles


def _c_static_list(name: str, ctype: str, values: list, per_line: int = 16) -> str:
    """`static const ctype name[n]` holding `values` in order (one zero entry when empty, as C has
    no empty arrays)."""
    vals = [str(v) for v in values] or ["0"]
    rows = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return (f"static const {ctype} {name}[{len(vals)}] __attribute__((aligned(8))) = {{\n    " +
            ",\n    ".join(rows) + "\n};\n")


def _c_static_table(name: str, ctype: str, length: int, entries: dict, fill=None,
                    fill_range=None, align: int = 8) -> str:
    """`static const ctype name[length]` as C source.

    `entries` maps an index to (value, comment). Every index of `fill_range` (lo, hi) without an
    entry holds `fill`, written as GNU range designators, so a table that is mostly one value is
    a few lines of source. Indices outside both are zero. No index is initialised twice (the
    build has -Woverride-init as an error)."""
    lines = [f"static const {ctype} {name}[{length}] __attribute__((aligned({align}))) = {{"]
    lo, hi = fill_range if fill_range is not None else (0, 0)
    cur = lo

    def gap(a, b):                       # fill [a, b)
        if fill is None or b <= a:
            return
        lines.append(f"    [{a}] = {fill}," if b - a == 1 else f"    [{a} ... {b - 1}] = {fill},")

    for idx in sorted(entries):
        if not 0 <= idx < length:
            raise ValueError(f"{name}: index {idx} outside [0, {length})")
        if lo <= idx < hi:
            gap(max(cur, lo), idx)
            cur = idx + 1
        value, comment = entries[idx]
        lines.append(f"    [{idx}] = {value}," + (f" // {comment}" if comment else ""))
    gap(max(cur, lo), hi)
    lines.append("};")
    return "\n".join(lines) + "\n"



# ======================================================================================
# ARGS-STRUCT COMPLETENESS
#
# Every args struct is carved out of the BINGO L1 arena, which is NEVER CLEARED. A field
# the args class forgets to assign is therefore read as whatever that TCDM word last held
# -- and depending on the field that is a wrong answer, a dead hart, or a task that stalls
# forever with no error (anything feeding an AGU stride, e.g. src_row_stride in
# __snax_bingo_kernel_simd_stream_elementwise_args_t).
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
                r"(?:__SNAX_KERNEL_ARGS_DEFINE|__HOST_BINGO_KERNEL_ARGS_DEFINE)\s*\w*\s*"
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
                 # An empty list still allocates one descriptor slot, since a zero-size allocation is not supported.
                 task_description_list += (
                     f"uint64_t* {list_name} = (uint64_t*){desc_alloc}(0x{chiplet_id:02x}, "
                     f"1 * BINGO_TASK_DESC_WORDS * sizeof(uint64_t));\n")
                 for w in range(words):
                     task_description_list += f"{list_name}[0 * BINGO_TASK_DESC_WORDS + {w}] = 0x0000000000000000ULL;\n"
            else:
                # The packed list is static data in the program image; ONE DMA puts it where
                # the manager fetches it from, instead of a host store per word.
                image = f"__bingo_desc_image_chip_{chiplet_id:02x}"
                task_description_list += (
                    f"static const uint64_t {image}[{num_local_nodes} * {words}] "
                    f"__attribute__((aligned(64))) = {{\n")
                for node in local_nodes:
                    packed_val = self.bingo_pack_node(node)
                    fields = self.bingo_unpack_node(packed_val)
                    desc_words = self.bingo_task_desc_words(packed_val)

                    # Word 0 at the lower address holds the low bits, the rest follow in
                    # ascending significance. Each word is masked to 64 bits so every
                    # constant fits a uint64_t.
                    task_description_list += (
                        "    " + ", ".join(f"0x{w:016X}ULL" for w in desc_words[:words]) +
                        f", // Node ID {node.node_id}\n"
                        f"    //   Type={fields['task_type']} TaskID={fields['task_id']}"
                        f" Assigned={fields['assigned_chiplet_id']:02x}/{fields['assigned_cluster_id']}"
                        f"/{fields['assigned_core_id']}"
                        f" DepCheck=En{fields['dep_check_en']}:0b{fields['dep_check_code']:0{self.ncores_hw}b}"
                        f" DepSet=En{fields['dep_set_en']},All{fields['dep_set_all']},"
                        f"{fields['dep_set_chiplet_id']:02x}/{fields['dep_set_cluster_id']}:"
                        f"0b{fields['dep_set_code']:0{self.ncores_hw}b}\n")
                task_description_list += "};\n"
                task_description_list += (
                    f"uint64_t* {list_name} = (uint64_t*){desc_alloc}(0x{chiplet_id:02x}, "
                    f"{count_name} * BINGO_TASK_DESC_WORDS * sizeof(uint64_t));\n"
                    f"sys_dma_blk_memcpy(0x{chiplet_id:02x}, (uint64_t)(uintptr_t){list_name},\n"
                    f"                   chiplet_addr_transform_full(0x{chiplet_id:02x}, "
                    f"(uint64_t)(uintptr_t){image}),\n"
                    f"                   sizeof({image}));\n")

        return task_description_list

    def _dev_task_numbering(self, chiplet_id: int):
        """The device task ids of a chiplet: node_id -> (index into the arg/kernel lists, index
        among its own cluster's device tasks), and the per-cluster bases of the lists.

        Original numbering: node-id order, every cluster's tasks interleaved (the local index is
        unused). compact_task_tables: cluster by cluster, node-id order within a cluster, so a
        cluster's tasks are one contiguous slice [bases[c], bases[c + 1]) and the runtime copies
        only that slice into its L1 (bingo_hw_scheduler_init_compact). gid_lists: within a
        cluster, core by core, each core's tasks in stream order, the order the manager
        dispatches them (bingo_hw_scheduler_init_gid_lists)."""
        dev = [n for n in sorted(self.node_list, key=lambda n: n.node_id)
               if n.assigned_chiplet_id == chiplet_id and n.kernel_name
               and n.kernel_name.startswith("__snax")]
        ncl = self.num_clusters_per_chiplet
        gid_lists = getattr(self, "gid_lists", False)
        if not getattr(self, "compact_task_tables", False):
            if gid_lists:
                raise ValueError("gid_lists needs compact_task_tables")
            return {n.node_id: (i, i) for i, n in enumerate(dev)}, None
        if gid_lists:
            pos = {n.node_id: i for i, n in enumerate(self.bingo_stream_order())}
            dev.sort(key=lambda n: (n.assigned_core_id, pos[n.node_id]))
        ids, bases, k = {}, [], 0
        for c in range(ncl):
            bases.append(k)
            mine = [n for n in dev if n.assigned_cluster_id == c]
            for j, n in enumerate(mine):
                ids[n.node_id] = (k + j, j)
            k += len(mine)
        bases.append(k)
        if len(ids) != len(dev):
            raise ValueError(f"compact_task_tables: {len(dev) - len(ids)} device tasks on a "
                             f"cluster outside 0..{ncl - 1}")
        if not gid_lists and max((len([n for n in dev if n.assigned_cluster_id == c])
                                  for c in range(ncl)), default=0) > 32767:
            raise ValueError("compact_task_tables: more than 32767 device tasks on one cluster "
                             "(the table is int16)")
        return ids, bases

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
            dev_ids, bases = self._dev_task_numbering(chiplet_id)
            if getattr(self, "gid_lists", False):
                # gid lists: one block a cluster, which the host copies into its L1 whole
                # (bingo_hw_scheduler_init_gid_lists): BINGO_GID_LIST_MAGIC, the core count K,
                # K + 1 core bases, then the global id of each of the cluster's device tasks in
                # local id order -- core by core, each core's in dispatch order -- so core r's
                # tasks are [base[r], base[r + 1]) and its next one is the entry after its
                # previous one. Each block a whole number of 8 bytes.
                ncores = self.num_cores_per_cluster
                local = sorted((dev_ids[n.node_id][0], n) for n in all_nodes if n.node_id in dev_ids)
                words = []
                for c in range(len(bases) - 1):
                    mine = [n for k, n in local if bases[c] <= k < bases[c + 1]]
                    core_base = [sum(1 for n in mine if n.assigned_core_id < r)
                                 for r in range(ncores + 1)]
                    block = (["BINGO_GID_LIST_MAGIC", ncores] + core_base +
                             [n.node_id for n in mine])
                    words += block + [0] * (len(block) % 2)
                mapping_str += _c_static_list(f"__bingo_gid_lists_chip_{chiplet_id:02x}",
                                              "uint32_t", words)
                mapping_str += (f"uint32_t num_dev_tasks_chip_{chiplet_id:02x} = {bases[-1]};\n")
                mapping_str += (f"uint32_t cluster_dev_base_chip_{chiplet_id:02x}[{len(bases)}] = "
                                f"{{{', '.join(str(b) for b in bases)}}};\n")
                continue
            if bases is not None and getattr(self, "compact_id_maps", False):
                # compact id maps: this chiplet's (global id, local id) pairs only; the dense table
                # (magic word, -1 for every other task) is built at boot from the shared template
                # (bingo_g2l16_build; with fast_host_init its long runs by DMA, bingo_g2l16_build_runs)
                tab = f"global_task_id_to_local_dev_id_chip_{chiplet_id:02x}"
                pairs = [(idx, dev_ids[node.node_id][1]) for idx, node in enumerate(all_nodes)
                         if node.node_id in dev_ids]
                idx_name = f"__bingo_g2l16_idx_chip_{chiplet_id:02x}"
                val_name = f"__bingo_g2l16_val_chip_{chiplet_id:02x}"
                if getattr(self, "fast_host_init", False):
                    runs_name = f"__bingo_g2l16_runs_chip_{chiplet_id:02x}"
                    runv_name = f"__bingo_g2l16_runv_chip_{chiplet_id:02x}"
                    runs, run_vals, singles = _g2l16_runs(pairs, num_nodes)
                    mapping_str += _c_static_list(runs_name, "uint32_t", runs)
                    mapping_str += _c_static_list(runv_name, "int16_t", run_vals)
                    mapping_str += _c_static_list(idx_name, "uint32_t", [i for i, _ in singles])
                    mapping_str += _c_static_list(val_name, "int16_t", [v for _, v in singles])
                    mapping_str += (f"int16_t* {tab} = bingo_g2l16_build_runs(__bingo_g2l16_dense, "
                                    f"__bingo_g2l16_template, num_total_tasks, {runs_name}, "
                                    f"{runv_name}, {len(runs) // 2}u, {idx_name}, {val_name}, "
                                    f"{len(singles)}u);\n")
                else:
                    mapping_str += _c_static_list(idx_name, "uint32_t", [i for i, _ in pairs])
                    mapping_str += _c_static_list(val_name, "int16_t", [v for _, v in pairs])
                    mapping_str += (f"int16_t* {tab} = bingo_g2l16_build(__bingo_g2l16_dense, "
                                    f"__bingo_g2l16_template, num_total_tasks, {idx_name}, {val_name}, "
                                    f"{len(pairs)}u);\n")
                mapping_str += (f"uint32_t num_dev_tasks_chip_{chiplet_id:02x} = {bases[-1]};\n")
                mapping_str += (f"uint32_t cluster_dev_base_chip_{chiplet_id:02x}[{len(bases)}] = "
                                f"{{{', '.join(str(b) for b in bases)}}};\n")
                continue
            if bases is not None:
                # compact: a magic word, then one int16 a task (see bingo_api.h); the host copies
                # ALIGN_UP(4 + 2 * num_nodes, 8) bytes of it into every cluster. Static data: a
                # task off this chiplet is -1, written as ranges.
                tab = f"global_task_id_to_local_dev_id_chip_{chiplet_id:02x}"
                image = f"__bingo_g2l16_image_chip_{chiplet_id:02x}"
                entries = {0: ("(int16_t)(BINGO_G2L16_MAGIC & 0xFFFFu)", "magic, low half"),
                           1: ("(int16_t)(BINGO_G2L16_MAGIC >> 16)", "magic, high half")}
                for idx, node in enumerate(all_nodes):
                    if node.node_id in dev_ids:
                        loc = dev_ids[node.node_id][1]
                        entries[2 + idx] = (str(loc), f"Node ID {node.node_id} -> cluster "
                                            f"{node.assigned_cluster_id} task {loc} ({node.node_name})")
                length = -(-(2 + num_nodes) // 4) * 4
                mapping_str += _c_static_table(image, "int16_t", length, entries, fill="-1",
                                               fill_range=(2, 2 + num_nodes))
                mapping_str += f"int16_t* {tab} = (int16_t*){image};\n"
                mapping_str += (f"uint32_t num_dev_tasks_chip_{chiplet_id:02x} = {bases[-1]};\n")
                mapping_str += (f"uint32_t cluster_dev_base_chip_{chiplet_id:02x}[{len(bases)}] = "
                                f"{{{', '.join(str(b) for b in bases)}}};\n")
                continue
            dev_task_counter = 0
            entries = {}
            for idx, node in enumerate(all_nodes):
                kernel_name = node.kernel_name
                if node.assigned_chiplet_id == chiplet_id and kernel_name and kernel_name.startswith("__snax"):
                    entries[idx] = (str(dev_task_counter), f"Node ID {node.node_id} -> Dev Task "
                                    f"{dev_task_counter} ({node.node_name})")
                    dev_task_counter += 1
            image = f"__bingo_g2d_image_chip_{chiplet_id:02x}"
            mapping_str += _c_static_table(image, "int32_t", max(num_nodes, 1), entries, fill="-1",
                                           fill_range=(0, num_nodes))
            mapping_str += (f"int32_t* global_task_id_to_dev_task_id_chip_{chiplet_id:02x} = "
                            f"(int32_t*){image};\n")
            mapping_str += f"uint32_t num_dev_tasks_chip_{chiplet_id:02x} = {dev_task_counter};\n"

        # 2. Emit global_task_id_to_host_task_id
        for chiplet_id in chiplets_to_process:
            if getattr(self, "compact_id_maps", False) or getattr(self, "gid_lists", False):
                # the chiplet's host tasks' global ids, ascending: a host task id is a position
                # in this list (bingo_hw_scheduler_sparse)
                gids = [idx for idx, node in enumerate(all_nodes)
                        if node.assigned_chiplet_id == chiplet_id and node.kernel_name
                        and node.kernel_name.startswith("__host")]
                mapping_str += _c_static_list(f"__bingo_g2h_gids_chip_{chiplet_id:02x}", "uint32_t", gids)
                mapping_str += f"uint32_t num_host_tasks_chip_{chiplet_id:02x} = {len(gids)};\n"
                continue
            host_task_counter = 0
            entries = {}
            for idx, node in enumerate(all_nodes):
                kernel_name = node.kernel_name
                if node.assigned_chiplet_id == chiplet_id and kernel_name and kernel_name.startswith("__host"):
                    entries[idx] = (str(host_task_counter), f"Node ID {node.node_id} -> Host Task "
                                    f"{host_task_counter} ({node.node_name})")
                    host_task_counter += 1
            image = f"__bingo_g2h_image_chip_{chiplet_id:02x}"
            mapping_str += _c_static_table(image, "int32_t", max(num_nodes, 1), entries, fill="-1",
                                           fill_range=(0, num_nodes))
            mapping_str += (f"int32_t* global_task_id_to_host_task_id_chip_{chiplet_id:02x} = "
                            f"(int32_t*){image};\n")
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
            # kernels (xdma_multicast, xdma_chain_gather) hold their operands this way, so
            # every handle in the list is collected and allocated, even when no other node
            # names it: a destination no local node reads still needs its allocation.
            # Sets and the values of dicts count too, as in bingo_liveness.collect_handle_users:
            # a handle the packer places must be one this emitter allocates.
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
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
        # header passes None, which emits no extra include.
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

    # ---- L1 ADDRESSES AT COMPILE TIME ------------------------------------------------------
    #
    # With the static args images every L1 address the host would patch into them -- buffers,
    # scratchpad slots, the args themselves -- is an offset from one of this chiplet's L1
    # allocations, and those land at addresses fixed by the heap layout: nothing but this code
    # allocates from a cluster's L1 heap before the scheduler starts, and the heap hands out
    # fragments front to back (bingo_alloc.h). So each allocation's address is emitted as a
    # macro (`<var>_at`), a constant expression the images are initialised with, and the
    # allocation is made with bingo_l1_alloc_at, which aborts the run if it lands anywhere else.
    # The host then writes only what depends on the program image or the device symbol table.

    def _l1_at_reset(self, chiplet_id):
        self._l1_at_on = bool(getattr(self, "static_l1_sp_slots", None)) and \
            getattr(self, "static_args_image", True) and getattr(self, "fast_host_init", False)
        self._l1_next = {}            # cluster -> C expression of the next allocation's address
        self._l1_at_names = set()     # the macros emitted for this chiplet
        self._const_handles = {}      # handle -> its address as a constant expression
        self._l1_const_by_name = None  # C variable -> constant expression (_c_static_value)

    def _emit_l1_alloc(self, f, chiplet_id, cl, var, size_expr, comment="", mac=None):
        """`uint64_t var = <L1 allocation>`; with compile-time L1 addresses also `#define` the
        macro holding where it lands. Returns the macro's name, or None."""
        if not self._l1_at_on:
            f.write(f"        uint64_t {var} = bingo_l1_alloc(0x{chiplet_id:02x}, {cl}, "
                    f"{size_expr});{comment}\n")
            return None
        mac = mac or f"{var}_at"
        at = self._l1_next.get(cl) or f"BINGO_L1_FIRST_ALLOC(0x{chiplet_id:02x}, {cl})"
        f.write(f"#define {mac} ({at})\n")
        f.write(f"        uint64_t {var} = bingo_l1_alloc_at(0x{chiplet_id:02x}, {cl}, "
                f"{size_expr}, {mac});{comment}\n")
        self._l1_next[cl] = f"{mac} + BINGO_HEAP_FRAGMENT_SIZE({size_expr})"
        self._l1_at_names.add(mac)
        return mac

    def _c_static_value(self, expr, chiplet_id, handle_name_map, local_nodes):
        """`expr` in _c_static_form with every local L1 variable -- a handle the chiplet
        allocated, a device scratchpad -- replaced by the constant expression of its address.
        A handle of another chiplet is addressed through the local variable of the same name
        (the cross-chip name/offset agreement), so a name stands for the local one."""
        names = getattr(self, "_l1_const_by_name", None)
        if names is None:
            names = {handle_name_map[h]: e for h, e in self._const_handles.items()}
            for nd in local_nodes:
                if getattr(nd, "_scratchpad_const", None):
                    names[nd._scratchpad_c_var] = nd._scratchpad_const
            self._l1_const_by_name = names
        return _C_IDENT.sub(lambda m: f"({names[m.group(0)]})" if m.group(0) in names
                            else m.group(0), _c_static_form(expr, chiplet_id))

    def _emit_memory_allocations(self, f, chiplet_id, sorted_handles, handle_name_map):
        """Emit memory allocation calls for handles on this chiplet.

        With static L1 allocation enabled, L1 buffers stop being individual allocations. The
        compiler has already decided every offset, so the runtime takes ONE allocation per
        cluster -- the packed arena -- and each buffer is a constant displacement into it.

        The arena comes from bingo_l1_alloc rather than from addressing TCDM directly, so the
        heap keeps owning the base address and the capacity check, nothing downstream has to
        learn a new address map, and static_l1=False switches a workload back with no other
        change.
        """
        placement = getattr(self, "static_l1_placement", None)
        stats = getattr(self, "static_l1_stats", None)
        local_handles = [h for h in sorted_handles if h.chip_id == chiplet_id]

        static_at = {}   # cluster -> macro of its static arena's address
        if placement and stats:
            arenas = sorted(k for k in stats if k[0] == chiplet_id)
            if arenas:
                f.write("        // 1. Static L1 arenas (compiler-packed; one per cluster)\n")
                for (chip, cl) in arenas:
                    st = stats[(chip, cl)]
                    static_at[cl] = self._emit_l1_alloc(
                        f, chip, cl, f"__bingo_l1_static_chip{chip:02x}_cl{cl}", st['peak'],
                        f"  // {st['count']} buffers packed from {st['sum']} B")
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
                    if static_at.get(h.cluster_id):
                        self._const_handles[h] = f"({static_at[h.cluster_id]} + {off})"
                    continue
                alloc_call = ""
                if h.mem_level == "L1":
                    mac = self._emit_l1_alloc(f, chiplet_id, h.cluster_id, c_var, h.size,
                                              mac=f"__bingo_l1_at_chip{chiplet_id:02x}_{c_var}")
                    if mac:
                        self._const_handles[h] = mac
                    continue
                elif h.mem_level == "L2":
                        alloc_call = f"bingo_l2_alloc(0x{h.chip_id:02x}, {h.size})"
                else: # L3
                        alloc_call = f"bingo_l3_alloc(0x{h.chip_id:02x}, {h.size})"
                
                f.write(f"        uint64_t {c_var} = {alloc_call};\n")
            f.write("\n")

    def _emit_list_allocations(self, f, chiplet_id):
        """Emit allocations for device/host argument and kernel lists."""
        f.write(f"        // 2. Prepare device/host arg/kernel lists\n")
        # With compile-time L1 addresses the device arg list is static data (3e)
        if not self._l1_at_on:
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
        # Per-node scratchpads and arg blocks live for the whole run and are never freed
        # individually, so a free-list allocator buys nothing and charges a bin search plus
        # heap-metadata traffic -- the latter crossing the fabric -- on every single one, and
        # its cost would scale with graph size.
        #
        # So each memory gets ONE block, sliced with a bump pointer. Sizes of the arg
        # structs are only known to the C compiler (sizeof), so the total is emitted as a
        # constant expression and the bump happens at runtime -- a couple of integer ops per
        # node instead of an allocator call.
        #
        # Device scratchpads and device args are in their own cluster's L1, host scratchpads
        # in L3. Moving device scratchpads to L3 would turn every device-side access into a
        # fabric round trip.
        #
        # The named workload buffers are allocated BEFORE this point, so collapsing these
        # allocations cannot disturb the cross-chip name/offset agreement they rely on.
        sp_align = "ALIGN_UP(sizeof(bingo_kernel_scratchpad_t), 64)"
        l1_terms = {}   # cluster -> list of C size expressions
        l3_terms = []
        # With slot packing the DEVICE scratchpad term is (slots x sp_align), added below --
        # one term per cluster, not one per node. Emitting both would make the packed arena
        # LARGER than the unpacked one.
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
        self._arena_off = {}   # cluster -> the arena's bump offset, as a constant expression
        for cl in sorted(l1_terms):
            base = f"__bingo_l1_arena_chip{chiplet_id:02x}_cl{cl}"
            if self._l1_at_on:
                self._emit_l1_alloc(f, chiplet_id, cl, base, " + ".join(l1_terms[cl]))
            else:
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
                self._arena_off[cl] = f"{max(slots_per_cl[cl]) + 1} * {sp_align}"
            else:
                f.write(f"        uint64_t {base}_off = 0;\n")
                self._arena_off[cl] = "0"
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
            node._scratchpad_const = None   # its address as a constant expression
            if slot is not None:
                # Packed: a slot index the compiler proved is free while this node runs.
                f.write(f"        bingo_kernel_scratchpad_t* {sp_var} = "
                        f"(bingo_kernel_scratchpad_t*)({base} + {slot} * {sp_align});"
                        f"  // slot {slot}\n")
                if is_device and self._l1_at_on:
                    node._scratchpad_const = f"({base}_at + {slot} * {sp_align})"
            else:
                f.write(f"        bingo_kernel_scratchpad_t* {sp_var} = "
                        f"(bingo_kernel_scratchpad_t*)({base} + {base}_off);\n")
                f.write(f"        {base}_off += {sp_align};\n")
                if is_device and self._l1_at_on:
                    cl = node.assigned_cluster_id
                    node._scratchpad_const = f"({base}_at + {self._arena_off[cl]})"
                    self._arena_off[cl] += f" + {sp_align}"
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
        # get_device_function() linearly scans the device symbol table, so the distinct names
        # are resolved once, in one pass, rather than once per task: only a handful of names
        # are ever distinct. Hoisting costs one local per distinct name.
        dev_kernel_names = []
        for node in local_nodes:
            kn = node.kernel_name
            if kn and kn.startswith("__snax") and kn not in dev_kernel_names:
                dev_kernel_names.append(kn)
        if dev_kernel_names:
            n_fn = len(dev_kernel_names)
            fast = getattr(self, "fast_host_init", False)
            f.write("        // Resolve every distinct device kernel in ONE pass over the device\n")
            f.write("        // symbol table (~110 entries). One call per name would walk it once per\n")
            if fast:
                f.write("        // name; get_device_functions_sig tests each entry against the wanted\n")
                f.write("        // names by their 8-byte signatures, computed here, and stops once all\n")
                f.write("        // are found.\n")
            else:
                f.write("        // name; get_device_functions tests each entry against all wanted names,\n")
                f.write("        // filtered by a 4-byte signature, and stops once all are found.\n")
            f.write(f"        static const char *const __bingo_fn_names_chip{chiplet_id:02x}[] = {{\n")
            for kn in dev_kernel_names:
                f.write(f"            \"{kn}\",\n")
            f.write("        };\n")
            f.write(f"        uint32_t __bingo_fn_addrs_chip{chiplet_id:02x}[{n_fn}];\n")
            if fast:
                f.write("        BINGO_STATIC_ASSERT(SNAX_SYM_SIG_OFF == 20u && SNAX_SYM_SIG_LEN == 8u,\n"
                        "                            \"kernel name signatures computed for another window\");\n")
                f.write(_c_static_list(f"__bingo_fn_sigs_chip{chiplet_id:02x}", "uint64_t",
                                       [f"0x{_snax_symbol_sig(kn):016x}ULL" for kn in dev_kernel_names],
                                       per_line=4))
                f.write(f"        get_device_functions_sig(__bingo_fn_names_chip{chiplet_id:02x}, "
                        f"__bingo_fn_sigs_chip{chiplet_id:02x}, "
                        f"__bingo_fn_addrs_chip{chiplet_id:02x}, {n_fn});\n")
            else:
                f.write(f"        get_device_functions(__bingo_fn_names_chip{chiplet_id:02x}, "
                        f"__bingo_fn_addrs_chip{chiplet_id:02x}, {n_fn});\n")
            for i, kn in enumerate(dev_kernel_names):
                f.write(f"        const uint32_t __bingo_fn_{kn}_chip{chiplet_id:02x} = "
                        f"__bingo_fn_addrs_chip{chiplet_id:02x}[{i}];\n")
            f.write("\n")

        dev_ids, dev_bases = self._dev_task_numbering(chiplet_id)
        host_task_idx = 0
        image_member = self._emit_device_args_images(f, chiplet_id, local_nodes, handle_name_map,
                                                     sp_packed)
        # With compile-time L1 addresses every device arg pointer is a constant: the list is
        # static data (3e) instead of one store a task
        arg_entries = {} if self._l1_at_on else None
        arg_checks = []
        # Device kernel list: the index of each task's kernel among dev_kernel_names, filled
        # from the resolved addresses by one loop (3e) instead of one statement a task
        kernel_idx = {}

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
                dev_task_idx = dev_ids[node.node_id][0]
                args_var = f"args_dev_chip{chiplet_id:02x}_{node.node_id}"

                if node.node_id in image_member:
                    # The args struct is a member of its cluster's static image: patch the
                    # fields only known at run time, point the list at where the image lands
                    img_var, img_t, member, runtime, l1_var, l1_at = image_member[node.node_id]
                    for field, value in runtime.items():
                        f.write(f"        {img_var}.{member}.{field} = {value};\n")
                    if arg_entries is not None:
                        arg_entries[dev_task_idx] = (
                            f"(uint32_t)({l1_at} + __builtin_offsetof({img_t}, {member}))",
                            f"{node.node_id} {node.node_name}")
                        arg_checks.append(
                            f"        if (__bingo_dev_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] != "
                            f"(uint32_t)({l1_var} + __builtin_offsetof({img_t}, {member}))) "
                            f"host_abort(BINGO_EXIT_LAYOUT);\n")
                    else:
                        f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = "
                                f"(uint32_t)({l1_var} + __builtin_offsetof({img_t}, {member}));\n")
                elif node.kernel_args:
                    assert arg_entries is None, f"node {node.node_id}: args outside the static image"
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
                        assert arg_entries is None, f"node {node.node_id}: args outside the static image"
                        _ab = f"__bingo_l1_arena_chip{chiplet_id:02x}_cl{node.assigned_cluster_id}"
                        f.write(f"        __snax_bingo_kernel_exit_args_t* {args_var} = (__snax_bingo_kernel_exit_args_t*)({_ab} + {_ab}_off);\n")
                        f.write(f"        {_ab}_off += ALIGN_UP(sizeof(__snax_bingo_kernel_exit_args_t), 64);\n")
                        f.write(f"        {args_var}->exit_code = 0;\n")
                        f.write(f"        {args_var}->scratchpad_ptr = {sp_cast};\n")
                        f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = (uint32_t)(uintptr_t){args_var};\n")
                    elif arg_entries is not None:
                        arg_entries[dev_task_idx] = ("0", f"{node.node_id} {node.node_name}")
                    else:
                        f.write(f"        device_arg_list_chip_{chiplet_id:02x}[{dev_task_idx}] = 0;\n")

                if getattr(self, "fast_host_init", False):
                    kernel_idx[dev_task_idx] = dev_kernel_names.index(kernel_name)
                else:
                    f.write(f"        device_kernel_list_chip_{chiplet_id:02x}[{dev_task_idx}] = "
                            f"__bingo_fn_{kernel_name}_chip{chiplet_id:02x};\n")

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

        num_dev = dev_bases[-1] if dev_bases is not None else len(kernel_idx)
        if kernel_idx or arg_entries:
            f.write("\n        // 3e. Device arg and kernel lists\n")
        if arg_entries is not None:
            f.write(_c_static_table(f"__bingo_dev_arg_list_chip_{chiplet_id:02x}", "uint32_t",
                                    max(num_dev, 1), arg_entries, align=8))
            f.write(f"        uint32_t* device_arg_list_chip_{chiplet_id:02x} = "
                    f"(uint32_t*)__bingo_dev_arg_list_chip_{chiplet_id:02x};\n")
            if arg_checks:
                f.write("#if BINGO_CHECK_STATIC_ARGS\n" + "".join(arg_checks) + "#endif\n")
        if kernel_idx:
            assert sorted(kernel_idx) == list(range(num_dev)), \
                f"chiplet {chiplet_id:#x}: device task ids are not 0..{num_dev - 1}"
            assert len(dev_kernel_names) <= 256, "kernel indices are bytes"
            f.write(_c_static_list(f"__bingo_dev_kernel_idx_chip_{chiplet_id:02x}", "uint8_t",
                                   [kernel_idx[i] for i in range(num_dev)], per_line=32))
            f.write(f"        bingo_fill_kernel_list(device_kernel_list_chip_{chiplet_id:02x}, "
                    f"__bingo_fn_addrs_chip{chiplet_id:02x}, "
                    f"__bingo_dev_kernel_idx_chip_{chiplet_id:02x}, {num_dev}u);\n")

        # Every runtime patch has landed in the images: one DMA a cluster puts them in L1
        copies = sorted({(v[0], v[1], v[4]) for v in image_member.values()})
        if copies:
            f.write("\n        // 3d. Device args images -> their cluster's L1 arena\n")
            f.write('        asm volatile("fence" ::: "memory");\n')
            for img_var, img_t, l1_var in copies:
                f.write(f"        sys_dma_blk_memcpy(0x{chiplet_id:02x}, {l1_var},\n"
                        f"                           chiplet_addr_transform_full(0x{chiplet_id:02x}, "
                        f"(uint64_t)(uintptr_t)&{img_var}),\n"
                        f"                           sizeof({img_t}));\n")

    def _emit_device_args_images(self, f, chiplet_id, local_nodes, handle_name_map, sp_packed):
        """One static image a cluster of the device args structs this chiplet's host would
        otherwise write field by field into L1. Returns {node_id: (image var, image type,
        member, {field: runtime value}, L1 address var, its compile-time address macro or
        None)}; empty when the images are off.

        Only with packed scratchpad slots: then the args are the arena's last tenant and sit
        back to back from the bump pointer, so a struct whose members are each 64-B aligned
        has exactly the bump layout (offset k = offset k-1 + ALIGN_UP(sizeof, 64)). Fields
        that are integer constants are initialised in the image -- it arrives with the
        program -- and only the rest (L1 addresses from the runtime arena, scratchpads,
        address transforms) is patched by the host, in L3, before one DMA moves the image.
        A field no args class assigns is zero here rather than stale TCDM contents.
        """
        if not (sp_packed and getattr(self, "static_args_image", True)):
            return {}
        # With compile-time L1 addresses a field is initialised in the image when it is constant
        # once every local L1 variable it names -- a buffer handle or a scratchpad -- is
        # replaced by the constant expression of its address (_c_static_value).
        at_on = self._l1_at_on
        per_cl = {}
        for node in local_nodes:
            kn = node.kernel_name
            if not (kn and kn.startswith("__snax")):
                continue
            sp_cast = f"(uint32_t)(uintptr_t){node._scratchpad_c_var}"
            if node.kernel_args:
                struct_t = node.kernel_args.get_struct_name()
                node.kernel_args._scratchpad_c_expr = sp_cast
                fields = dict(node.kernel_args.get_c_field_assignments_with_scratchpad(handle_name_map))
                pred = getattr(node, "_pred_source_node", None)
                if pred:
                    fields["pred_scratchpad_addr"] = f"(uint32_t)(uintptr_t){pred._scratchpad_c_var}"
            elif "exit" in kn:
                struct_t = "__snax_bingo_kernel_exit_args_t"
                fields = {"exit_code": "0", "scratchpad_ptr": sp_cast}
            else:
                continue
            per_cl.setdefault(node.assigned_cluster_id, []).append((node, struct_t, fields))

        out = {}
        symbol_hi = False
        checks = []   # (image, member, field, run-time value) of every field made static
        for cl in sorted(per_cl):
            cid = f"{chiplet_id:02x}"
            img_t = f"__bingo_args_image_chip{cid}_cl{cl}_t"
            img_var = f"__bingo_args_image_chip{cid}_cl{cl}"
            l1_var = f"__bingo_args_l1_chip{cid}_cl{cl}"
            base = f"__bingo_l1_arena_chip{cid}_cl{cl}"
            f.write(f"        // 3c. Device args of cluster {cl}: compile-time fields in a static image\n")
            f.write("        typedef struct {\n")
            for node, struct_t, _ in per_cl[cl]:
                f.write(f"            {struct_t} a{node.node_id} __attribute__((aligned(64)));\n")
            f.write(f"        }} {img_t};\n")
            f.write(f"        static {img_t} {img_var} __attribute__((aligned(64))) = {{\n")
            l1_at = f"{l1_var}_at" if at_on else None
            for node, struct_t, fields in per_cl[cl]:
                const, runtime = [], {}
                for k, v in fields.items():
                    sv = self._c_static_value(v, chiplet_id, handle_name_map, local_nodes) \
                        if at_on else v
                    hi = _symbol_addr_hi(v, chiplet_id) if at_on else None
                    if hi is not None:
                        const.append((k, hi))
                        symbol_hi = True
                        checks.append((img_var, f"a{node.node_id}", k, v))
                    elif _is_c_constant(v):
                        const.append((k, sv))
                    elif at_on and _is_c_static(sv, self._l1_at_names):
                        const.append((k, sv))
                        checks.append((img_var, f"a{node.node_id}", k, v))
                    else:
                        runtime[k] = v
                if const:
                    f.write(f"            .a{node.node_id} = {{" +
                            ", ".join(f".{k} = {v}" for k, v in const) + "},\n")
                out[node.node_id] = (img_var, img_t, f"a{node.node_id}", runtime, l1_var, l1_at)
            f.write("        };\n")
            f.write(f"        const uint64_t {l1_var} = {base} + {base}_off;\n")
            if at_on:
                f.write(f"#define {l1_at} ({base}_at + {self._arena_off[cl]})\n")
                f.write(f"        if ({l1_var} != {l1_at}) host_abort(BINGO_EXIT_LAYOUT);\n")
                self._l1_at_names.add(l1_at)
                self._arena_off[cl] += f" + sizeof({img_t})"
            f.write(f"        {base}_off += sizeof({img_t});\n\n")
        if symbol_hi:
            # The program runs at its chip's prefix (PC and symbol addresses carry it), so the
            # chip-local 40 bits are what must lie below 4 GiB
            f.write("        // the images hold the high halves of program symbols' addresses as the chip\n"
                    "        // prefix alone, so the program must lie below 4 GiB of its chip's space\n"
                    "        if ((((uint64_t)(uintptr_t)&__wide_spm_end - 1u) & 0xFFFFFFFFFFULL) >> 32) "
                    "host_abort(BINGO_EXIT_LAYOUT);\n\n")
        if checks:
            # Each field made static, against the value the host would have written
            f.write("#if BINGO_CHECK_STATIC_ARGS\n")
            for img_var, member, k, v in checks:
                f.write(f"        if ({img_var}.{member}.{k} != {v}) host_abort(BINGO_EXIT_LAYOUT);\n")
            f.write("#endif\n")
        return out

    def _emit_scheduler_launch(self, f, chiplet_id):
        """Emit the scheduler initialization and launch calls."""
        f.write("\n")
        f.write('        OFFLOAD_BINGO_HW_DEBUG_PRINT_SAFE("Chip(%x, %x): [Host] Init HW Bingo Scheduler\\r\\n",\n')
        f.write('               get_current_chip_loc_x(), get_current_chip_loc_y());\n\n')

        if getattr(self, "gid_lists", False):
            f.write(f"        bingo_hw_scheduler_init_gid_lists((uint64_t)(uintptr_t)device_arg_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                (uint64_t)(uintptr_t)device_kernel_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                cluster_dev_base_chip_{chiplet_id:02x},\n")
            f.write(f"                                (uint64_t)(uintptr_t)__bingo_gid_lists_chip_{chiplet_id:02x},\n")
            f.write(f"                                {self.num_cores_per_cluster}u,\n")
            f.write(f"                                (uint64_t)(uintptr_t)bingo_hw_scheduler_task_desc_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                bingo_hw_scheduler_num_task_desc_chip_{chiplet_id:02x});\n\n")
        elif getattr(self, "compact_task_tables", False):
            f.write(f"        bingo_hw_scheduler_init_compact((uint64_t)(uintptr_t)device_arg_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                (uint64_t)(uintptr_t)device_kernel_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                cluster_dev_base_chip_{chiplet_id:02x},\n")
            f.write(f"                                (uint64_t)(uintptr_t)global_task_id_to_local_dev_id_chip_{chiplet_id:02x},\n")
            f.write(f"                                num_total_tasks,\n")
            f.write(f"                                (uint64_t)(uintptr_t)bingo_hw_scheduler_task_desc_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                bingo_hw_scheduler_num_task_desc_chip_{chiplet_id:02x});\n\n")
        else:
            f.write(f"        bingo_hw_scheduler_init((uint64_t)(uintptr_t)device_arg_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                (uint64_t)(uintptr_t)device_kernel_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                num_dev_tasks_chip_{chiplet_id:02x},\n")
            f.write(f"                                (uint64_t)(uintptr_t)global_task_id_to_dev_task_id_chip_{chiplet_id:02x},\n")
            f.write(f"                                num_total_tasks,\n")
            f.write(f"                                (uint64_t)(uintptr_t)bingo_hw_scheduler_task_desc_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                bingo_hw_scheduler_num_task_desc_chip_{chiplet_id:02x});\n\n")
        
        if getattr(self, "compact_id_maps", False) or getattr(self, "gid_lists", False):
            f.write(f"        uint32_t err = bingo_hw_scheduler_sparse(host_arg_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                                 host_kernel_list_chip_{chiplet_id:02x},\n")
            f.write(f"                                                 __bingo_g2h_gids_chip_{chiplet_id:02x},\n")
            f.write(f"                                                 num_host_tasks_chip_{chiplet_id:02x});\n")
        else:
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

            # Step 2b: the compact id maps' shared template and the one dense table each
            # chiplet builds from it at boot
            if (getattr(self, "compact_id_maps", False) and getattr(self, "compact_task_tables", False)
                    and not getattr(self, "gid_lists", False)):
                n = len(self.node_list)
                length = -(-(2 + n) // 4) * 4
                f.write(_c_static_table("__bingo_g2l16_template", "int16_t", length,
                                        {0: ("(int16_t)(BINGO_G2L16_MAGIC & 0xFFFFu)", "magic, low half"),
                                         1: ("(int16_t)(BINGO_G2L16_MAGIC >> 16)", "magic, high half")},
                                        fill="-1", fill_range=(2, 2 + n), align=64))
                f.write(f"static int16_t __bingo_g2l16_dense[{length}] __attribute__((aligned(64)));\n\n")

            # Step 3: Emit kernel_execution function structure. With split_chip_functions every
            # chiplet's block is its own function and kernel_execution only dispatches: the
            # compiler then optimises one chiplet's block at a time, where one function
            # holding every block grows its optimisation passes faster than its length.
            split = bool(getattr(self, "split_chip_functions", False))
            if not split:
                f.write("int kernel_execution(){\n")
                f.write("    check_kernel_tab_ready();\n")
                f.write(f"    OFFLOAD_BINGO_HW_DEBUG_PRINT_SAFE(\"Chip(%x, %x): [Host] Preparing {app_name} Workload\\r\\n\", get_current_chip_loc_x(), get_current_chip_loc_y());\n")
                f.write("    uint32_t current_chip_id = get_current_chip_id();\n")


            # Step 4: Iterate over each chiplet to generate isolated blocks
            for chiplet_id in self.chiplet_ids:
                if split:
                    f.write(f"static int __attribute__((noinline)) "
                            f"kernel_execution_chip_{chiplet_id:02x}(void) {{\n")
                    f.write(f"    uint32_t current_chip_id = 0x{chiplet_id:02x};\n")
                    f.write("    (void)current_chip_id;\n")
                    f.write("    {\n")
                else:
                    f.write(f"    if (current_chip_id == 0x{chiplet_id:02x}) {{\n")

                self._l1_at_reset(chiplet_id)

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
                if split:
                    f.write("    return 0;\n")
                    f.write("}\n\n")

            if split:
                f.write("int kernel_execution(){\n")
                f.write("    check_kernel_tab_ready();\n")
                f.write(f"    OFFLOAD_BINGO_HW_DEBUG_PRINT_SAFE(\"Chip(%x, %x): [Host] Preparing {app_name} Workload\\r\\n\", get_current_chip_loc_x(), get_current_chip_loc_y());\n")
                f.write("    uint32_t current_chip_id = get_current_chip_id();\n")
                for chiplet_id in self.chiplet_ids:
                    f.write(f"    if (current_chip_id == 0x{chiplet_id:02x}) "
                            f"return kernel_execution_chip_{chiplet_id:02x}();\n")
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
