# Fanchen Kong <fanchen.kong@kuleuven.be>

import math

# The task list the host hands the scheduler stays a uint64_t array however wide the
# descriptor gets, so a descriptor always occupies a whole number of these words.
BINGO_TASK_LIST_WORD_BITS = 64
BINGO_TASK_LIST_WORD_MASK = (1 << BINGO_TASK_LIST_WORD_BITS) - 1
from bingo_node import BingoNode


class BingoDFGDescriptorMixin:
    """The packed task descriptor: its field layout, and packing a node into it.

    This is the one place that has to agree with the RTL struct in bingo_hw_manager_top.
    A disagreement is silent -- the manager dispatches shuffled fields and the run simply
    misbehaves -- so the layout is derived here and the widths are checked, never assumed.

    Mixed into BingoDFG, so `self` is the whole DFG. These methods call each other
    and the other mixins' methods freely; nothing here is meant to stand alone.
    """

    @staticmethod
    def _idx_width(n: int) -> int:
        """cf_math_pkg::idx_width -- 1 bit even for a one-element index space.

        clog2(1) is 0, and a zero-width field would pull every field above it down
        by one bit. The RTL indexes both cluster ids with idx_width, so SW must too.
        """
        return math.ceil(math.log2(n)) if n > 1 else 1

    @property
    def ncores_hw(self) -> int:
        """Core count the RTL descriptor is sized for, which is NOT the SNAX core count.

        occamy_quad_ctrl instantiates NUM_CORES_PER_CLUSTER = NrCoresPerCluster[0] + 1:
        the chiplet's CVA6 is wired in as one extra core so it can be a dep-set/dep-check
        participant. The core-id and both dep-code fields are therefore one bit / one
        core wider than the cluster's SNAX core count, whether or not this DFG places
        host nodes. self.num_cores_per_cluster already folds the +1 in when
        is_host_as_acc; add it back otherwise so the layout tracks the RTL either way.
        """
        return self.num_cores_per_cluster if self.is_host_as_acc \
            else self.num_cores_per_cluster + 1

    def bingo_task_desc_layout(self) -> list:
        """THE packed-descriptor field table, LSB -> MSB, mirroring the RTL struct
        bingo_hw_manager_task_desc_t (a packed struct's last-declared member is the LSB).

        Pack and unpack both walk this one list. They used to re-derive every width
        independently from the same inputs, which is exactly how the C header's copy
        rotted into a 50-bit layout against a real 65-bit one -- two derivations of the
        same table drift the moment one of them is edited.

        Returns [(name, width), ...]; the running sum of the widths is the bit offset of
        each field, so there are no shift constants to keep in sync either.
        """
        num_clusters = self.num_clusters_per_chiplet
        num_cores = self.ncores_hw
        cluster_id_width = self._idx_width(num_clusters)
        core_id_width = self._idx_width(num_cores)

        return [
            # DARTS Tier 1 conditional execution
            ("cond_exec_invert", 1),
            ("cond_exec_group_id", 5),
            ("cond_exec_en", 1),
            # 00 = normal, 01 = dummy, 10 = gating
            ("task_type", 2),
            ("task_id", self.task_id_width),
            # Routing encoding chip_id = (x << 4) | y, so 8 bits for a 4x4 array.
            ("assigned_chiplet_id", self.chip_id_width),
            ("assigned_cluster_id", cluster_id_width),
            ("assigned_core_id", core_id_width),
            # dep_check_info
            ("dep_check_en", 1),
            ("dep_check_code", num_cores),
            # Tag bits are always present to match the RTL struct; they are 0 when
            # identity-aware deps are disabled.
            ("dep_check_tag", self.dep_tag_width),
            # dep_set_info
            ("dep_set_en", 1),
            # RTL name is dep_set_all_chiplet; the short key is what callers already use.
            ("dep_set_all", 1),
            ("dep_set_chiplet_id", self.chip_id_width),
            ("dep_set_cluster_id", cluster_id_width),
            ("dep_set_code", num_cores),
            ("dep_set_tag", self.dep_tag_width),
            # Set by the compiler on whichever task SENDS the cross-die message
            # for a gating region -- see bingo_hw_manager_top.cerf_carry.
            ("cerf_carry", 1),
        ]

    def bingo_task_desc_bits(self) -> int:
        """Bits the layout actually occupies, below the zero padding to task_desc_width."""
        return sum(width for _, width in self.bingo_task_desc_layout())

    def bingo_pack_node(self, node: BingoNode) -> int:
        """Pack a node into the task descriptor, as a task_desc_width-bit integer."""
        task_type_map = {"normal": 0, "dummy": 1, "gating": 2}

        dep_check_code_val = 0
        for core_id in node.dep_check_list:
            dep_check_code_val |= (1 << core_id)
        dep_set_code_val = 0
        for core_id in node.dep_set_list:
            dep_set_code_val |= (1 << core_id)

        values = {
            "cerf_carry": int(getattr(node, "cerf_carry", False)),
            "cond_exec_invert": int(node.cond_exec_invert),
            "cond_exec_group_id": int(node.cond_exec_group_id),
            "cond_exec_en": int(node.cond_exec_en),
            "task_type": task_type_map.get(node.node_type, 0),
            "task_id": int(node.node_id),
            "assigned_chiplet_id": int(node.assigned_chiplet_id),
            "assigned_cluster_id": int(node.assigned_cluster_id),
            "assigned_core_id": int(node.assigned_core_id),
            "dep_check_en": 1 if node.dep_check_enable else 0,
            "dep_check_code": dep_check_code_val,
            "dep_check_tag": int(node.dep_check_tag or 0),
            "dep_set_en": 1 if node.dep_set_enable else 0,
            "dep_set_all": 1 if node.remote_dep_set_all else 0,
            "dep_set_chiplet_id": int(node.dep_set_chiplet_id),
            "dep_set_cluster_id": int(node.dep_set_cluster_id),
            "dep_set_code": dep_set_code_val,
            "dep_set_tag": int(node.dep_set_tag or 0),
        }

        packed_val = 0
        current_shift = 0
        for name, width in self.bingo_task_desc_layout():
            value = values[name]
            # An out-of-range value does not truncate, it ORs into the NEXT field, so a
            # single bad core id silently rewrites the dep tag above it. Refuse instead.
            if value < 0 or value >= (1 << width):
                raise ValueError(
                    f"Node {node.node_id} ({getattr(node, 'node_name', '?')}): "
                    f"{name}={value} does not fit its {width}-bit descriptor field "
                    f"(max {(1 << width) - 1}).")
            packed_val |= (value << current_shift)
            current_shift += width

        # The descriptor no longer has to fit one host AXI-Lite beat -- the RTL fetch
        # master reads task_desc_words beats and commits them as one atomic push -- but
        # it must still fit TaskDescBusWidth, because the RTL builds the same struct
        # from the same widths and ReservedBitsForTaskDesc goes negative on overflow:
        # it will not elaborate either. The occupancy is not fixed, it grows with the
        # cluster and core counts, so report the breakdown and name the knobs.
        if current_shift > self.task_desc_width:
            num_clusters = self.num_clusters_per_chiplet
            num_cores = self.ncores_hw
            cluster_id_width = self._idx_width(num_clusters)
            core_id_width = self._idx_width(num_cores)
            # Subtracted, not re-listed, so adding a field to the table cannot leave this
            # breakdown claiming numbers that no longer add up to the total.
            fixed = current_shift - (2 * cluster_id_width + core_id_width
                                     + 2 * num_cores + 2 * self.dep_tag_width)
            raise ValueError(
                f"Packed task descriptor exceeds {self.task_desc_width} bits: "
                f"{current_shift} bits used.\n"
                f"  fixed fields                 {fixed}\n"
                f"  assigned/dep_set cluster id  {2 * cluster_id_width}  "
                f"(num_clusters={num_clusters})\n"
                f"  assigned core id             {core_id_width}  (num_cores={num_cores})\n"
                f"  dep_check + dep_set code     {2 * num_cores}\n"
                f"  dep_check + dep_set tag      {2 * self.dep_tag_width}  "
                f"(dep_tag_width={self.dep_tag_width})\n"
                f"Widen s1_quadrant.task_desc_width in the RTL cfg (SW picks the new "
                f"value up from occamy.h's BINGO_TASK_DESC_WIDTH, and each extra 64 bits "
                f"costs one more fetch beat), or lower s1_quadrant.dep_tag_width (each "
                f"step down frees 2 bits, from occamy.h's BINGO_DEP_TAG_WIDTH).")

        return packed_val

    def bingo_unpack_node(self, packed_val: int) -> dict:
        """Unpack a task descriptor back into node fields, using the same one table."""
        fields = {}
        current_shift = 0
        for name, width in self.bingo_task_desc_layout():
            fields[name] = (packed_val >> current_shift) & ((1 << width) - 1)
            current_shift += width
        return fields

    def bingo_task_desc_words(self, packed_val: int) -> list:
        """Split a descriptor into task-list words, LEAST-SIGNIFICANT WORD FIRST.

        The RTL fetch master reads beat 0 from the lower address into the low bits of
        the descriptor, so ascending address == ascending significance. Emitting these
        the other way round swaps the halves of every descriptor, which does not fault:
        the scheduler just dispatches garbage.
        """
        return [(packed_val >> (i * BINGO_TASK_LIST_WORD_BITS)) & BINGO_TASK_LIST_WORD_MASK
                for i in range(self.task_desc_words)]
