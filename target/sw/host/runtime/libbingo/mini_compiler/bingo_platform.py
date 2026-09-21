import os
import re
import sys
from pathlib import Path


_DEFINE_RE = re.compile(r"^\s*#define\s+(\w+)\s+([0-9a-fA-FxX]+)\b")

# The BINGO task list is a uint64_t array however wide the descriptor gets, so a
# descriptor always occupies a whole number of these words.
TASK_LIST_WORD_BITS = 64

# Schema defaults, used when a generated header predates the define. They must match the
# RTL parameter defaults (bingo_hw_manager_top.sv) or a stale header silently produces a
# layout the hardware does not read back.
DEFAULT_DEP_TAG_WIDTH = 4
DEFAULT_TASK_DESC_WIDTH = 128
# ChipIdWidth. NOT fixed by the RTL struct, despite being 8 in every in-tree config:
# occamy_quad_ctrl.sv.tpl elaborates bingo_hw_manager_top with
# .ChipIdWidth(${chip_id_width}) from cfg key hemaia_multichip.chip_id_width, and
# occamygen exports the same number as BINGO_CHIP_ID_WIDTH. It looks constant only
# because a chiplet id is the D2D ROUTING encoding (x << 4) | y rather than a dense
# index, so a 4x4 array still spends a full byte on it. The descriptor carries TWO
# chiplet-id fields, so guessing this wrong moves every field above
# assigned_chiplet_id -- which is all of them but the first five.
DEFAULT_CHIP_ID_WIDTH = 8
# TaskIdWidth default, matching bingo_hw_manager_top's parameter default. A header from
# before s1_quadrant.task_id_width existed carries no define, and 12 is what those builds
# elaborated with.
DEFAULT_TASK_ID_WIDTH = 12


def _task_desc_words(width):
    return (int(width) + TASK_LIST_WORD_BITS - 1) // TASK_LIST_WORD_BITS


def _task_desc_geometry(defines, source):
    """(width, words) of the packed task descriptor, from one define.

    BINGO_TASK_DESC_WORDS is derived, never read as an independent number: two numbers
    for one fact is how the descriptor layout drifted between SW and RTL in the first
    place. A header that also states it is cross-checked rather than trusted.
    """
    width = defines.get("BINGO_TASK_DESC_WIDTH", DEFAULT_TASK_DESC_WIDTH)
    words = _task_desc_words(width)
    stated = defines.get("BINGO_TASK_DESC_WORDS")
    if stated is not None and stated != words:
        raise ValueError(
            f"{source} says BINGO_TASK_DESC_WIDTH={width} but "
            f"BINGO_TASK_DESC_WORDS={stated}; {width} bits is {words} "
            f"{TASK_LIST_WORD_BITS}-bit words.")
    return width, words


def _descriptor_geometry(defines, source):
    """Every task-descriptor knob a generated header carries, derived in ONE place.

    parse_platform_cfg() and the no-platform fallbacks below both come through here, so
    a caller that passes a platform dict and a caller that passes nothing cannot end up
    with two different readings of the same header.
    """
    width, words = _task_desc_geometry(defines, source)
    return {
        # DepTagWidth as the RTL was generated with (cfg s1_quadrant.dep_tag_width). The
        # packed descriptor must fit task_desc_width, and a 4-cluster config spends 2 more
        # bits on cluster ids than a 1-cluster one, so this is NOT a constant across
        # configs. A SW/RTL mismatch does not fault: the tag is not the top field, so
        # every bit above it -- both dep codes, the dep_set chiplet and cluster ids --
        # shifts, and tasks quietly dispatch to the wrong place.
        "dep_tag_width": defines.get("BINGO_DEP_TAG_WIDTH", DEFAULT_DEP_TAG_WIDTH),
        # ChipIdWidth as the RTL was generated with (cfg hemaia_multichip.chip_id_width).
        "chip_id_width": defines.get("BINGO_CHIP_ID_WIDTH", DEFAULT_CHIP_ID_WIDTH),
        # TaskIdWidth as the RTL was generated with (cfg s1_quadrant.task_id_width). The id
        # space is 2**width and the compiler assigns one id per task, so it caps the graph
        # size; it is also a descriptor field, so a mismatch shifts every field above it.
        "task_id_width": defines.get("BINGO_TASK_ID_WIDTH", DEFAULT_TASK_ID_WIDTH),
        # TaskDescBusWidth as the RTL was generated with (cfg s1_quadrant.task_desc_width,
        # otherwise derived: the smallest whole number of 64-bit words the layout fits).
        # NOT the host AXI-Lite data width: the fetch master reads task_desc_words beats
        # and commits them as one atomic push, so the descriptor may be wider than a bus
        # beat. The packer and the task-list emitter both read these two, so there is one
        # source for the layout width and for the array stride.
        "task_desc_width": width,
        "task_desc_words": words,
    }


def _parse_defines(path):
    defines = {}
    with open(path) as f:
        for line in f:
            match = _DEFINE_RE.match(line)
            if match:
                defines[match.group(1)] = int(match.group(2), 0)
    return defines


def _require_define(defines, name, path):
    if name not in defines:
        raise ValueError(f"Missing required define {name} in {path}")
    return defines[name]


# The platform the mini-compiler most recently parsed, and the single reason BingoDFG can
# track the descriptor geometry of the header THIS build was generated against without
# every workload editing its BingoDFG(...) call. A generator parses exactly one platform
# header (its --platformcfg) before it builds its graph, so "the last one parsed" is "the
# one this run is for"; when two are parsed, the last wins and an explicit
# BingoDFG(task_desc_width=...) is the way to be unambiguous.
_LAST_PARSED_PLATFORM = None


def _remember_platform(platform):
    global _LAST_PARSED_PLATFORM
    _LAST_PARSED_PLATFORM = platform
    return platform


def last_parsed_platform():
    """The platform dict from the most recent parse_platform_cfg(), or None."""
    return _LAST_PARSED_PLATFORM


def parse_platform_cfg(occamy_h_path):
    """Parse Bingo platform parameters from generated occamy.h."""
    occamy_h_path = Path(occamy_h_path)
    defines = _parse_defines(occamy_h_path)

    num_chiplets = _require_define(defines, "N_CHIPLETS", occamy_h_path)
    chiplet_ids = []
    for idx in range(num_chiplets):
        chiplet_ids.append(_require_define(defines, f"CHIPLET_ID_{idx}", occamy_h_path))

    if len(chiplet_ids) != num_chiplets:
        raise ValueError(
            f"Expected {num_chiplets} chiplet IDs in {occamy_h_path}, "
            f"got {len(chiplet_ids)}.")

    platform = {
        "num_chiplets": num_chiplets,
        "num_clusters_per_chiplet": _require_define(
            defines, "N_CLUSTERS_PER_CHIPLET", occamy_h_path),
        "num_cores_per_cluster": _require_define(
            defines, "N_CORES_PER_CLUSTER", occamy_h_path),
        "chiplet_ids": chiplet_ids,
        # Where this reading came from, so an error about a descriptor that does not fit
        # can name the header instead of leaving the reader to guess which one was used.
        "platform_header": str(occamy_h_path),
        # Whether this platform HAS a memory chiplet, and where. A workload that stages
        # its inputs and goldens there on a config with none does not fault -- the
        # addresses are simply unmapped, so the loads return junk and the checks compare
        # junk against junk. Defaulted rather than required so an older generated header
        # still parses. See util/sim/common/bingo_data_staging.py.
        "num_mem_chips": defines.get("N_MEM_CHIPS", 0),
        "mem_chip_loc_x": defines.get("MEM_CHIP_LOC_X", 0),
        "mem_chip_loc_y": defines.get("MEM_CHIP_LOC_Y", 0),
        # dep_tag_width / chip_id_width / task_desc_width / task_desc_words. Each is
        # defaulted inside _descriptor_geometry so an older generated header still parses.
        **_descriptor_geometry(defines, occamy_h_path),
    }
    return _remember_platform(platform)


# The generated platform header, resolved from THIS file's location the same way
# _DEFAULT_ROLES_HEADER below is: bingo_platform.py sits at
# target/sw/host/runtime/libbingo/mini_compiler/, so parents[4] is target/sw.
_DEFAULT_PLATFORM_HEADER = (
    Path(__file__).resolve().parents[4] / "shared" / "platform" / "generated" / "occamy.h"
)


def platform_descriptor_geometry():
    """(geometry, source) the packer should use when the caller named no widths.

    This is what makes the generated BINGO_TASK_DESC_WIDTH / BINGO_DEP_TAG_WIDTH /
    BINGO_CHIP_ID_WIDTH reach BingoDFG: none of the ~50 workload generators passes them,
    so without this they would all silently pack against a literal, and a task list packed
    64 bits wide against a 128-bit RTL descriptor reads every entry after the first from
    the wrong address. Preference order, most specific first:

      1. the platform this run already parsed (parse_platform_cfg) -- exactly the header
         the workload's --platformcfg pointed at, whatever its path;
      2. the generated header at its in-tree location, for a tool that builds a DFG
         without parsing a platform at all;
      3. the schema defaults, for a checkout where `make sw` has not generated one yet.

    `source` is a human-readable description of which of the three won, for error text.
    """
    if _LAST_PARSED_PLATFORM is not None:
        geometry = {key: _LAST_PARSED_PLATFORM[key]
                    for key in ("dep_tag_width", "chip_id_width", "task_id_width",
                                "task_desc_width", "task_desc_words")}
        return geometry, _LAST_PARSED_PLATFORM.get("platform_header", "parsed platform")
    try:
        defines = _parse_defines(_DEFAULT_PLATFORM_HEADER)
    except OSError:
        return ({"dep_tag_width": DEFAULT_DEP_TAG_WIDTH,
                 "chip_id_width": DEFAULT_CHIP_ID_WIDTH,
                 "task_id_width": DEFAULT_TASK_ID_WIDTH,
                 "task_desc_width": DEFAULT_TASK_DESC_WIDTH,
                 "task_desc_words": _task_desc_words(DEFAULT_TASK_DESC_WIDTH)},
                f"bingo_platform.py schema defaults ({_DEFAULT_PLATFORM_HEADER} is not "
                f"generated yet)")
    return (_descriptor_geometry(defines, _DEFAULT_PLATFORM_HEADER),
            str(_DEFAULT_PLATFORM_HEADER))


def default_dep_tag_width():
    """DepTagWidth for a caller with no platform dict. See platform_descriptor_geometry."""
    return platform_descriptor_geometry()[0]["dep_tag_width"]


def default_chip_id_width():
    """ChipIdWidth for a caller with no platform dict. See platform_descriptor_geometry."""
    return platform_descriptor_geometry()[0]["chip_id_width"]


def default_task_desc_width():
    """TaskDescBusWidth for a caller with no platform dict. See above."""
    return platform_descriptor_geometry()[0]["task_desc_width"]


def default_task_desc_words():
    """How many 64-bit task-list words one descriptor occupies. See above."""
    return platform_descriptor_geometry()[0]["task_desc_words"]


# The generated role map, mirrored into the device tree by `make snax-sw-gen`
# (target/sw/Makefile). Resolved from THIS file's location so every workload's
# core_roles(platform) keeps working unchanged: bingo_platform.py sits at
# target/sw/host/runtime/libbingo/mini_compiler/, so parents[4] is target/sw.
_DEFAULT_ROLES_HEADER = (
    Path(__file__).resolve().parents[4] / "device" / "runtime" / "snax" /
    "snax_core_roles_defs.h"
)


def core_roles(platform=None, roles_header=None):
    """Which cluster core carries which engine, for the platform being built.

    Everything here comes from the generated map -- SNAX_CORE_* in
    device/runtime/snax/snax_core_roles_defs.h, which snaxgen derives from the cluster
    hjson (snax_acc_cfg -> gemm, snax_simd_cfg -> simd, snax_xdma_cfg -> xdma, and the
    `xdma:` DMA-ISA boolean -> idma) and `make snax-sw-gen` mirrors into the device tree.
    Nothing is hardcoded, because a node placed on the wrong hart does not fault: it
    programs THAT hart's accelerator at the same CSR offsets and reports success.

        gemm  SNAX_CORE_GEMM
        simd  SNAX_CORE_SIMD
        xdma  SNAX_CORE_XDMA
        dm    SNAX_CORE_IDMA -- snRuntime's SNRT_CLUSTER_DM_CORE_NUM is 1, so
              snrt_is_dm_core() selects exactly this hart, and it is the only one with the
              DMA ISA (a dm* instruction anywhere else traps). Upstream rejects any cfg
              that puts the DMA ISA off the last core, so this is also n - 1.
        host  SNAX_CLUSTER_NUM_CORES exactly -- the chiplet's CVA6, one past the last SNAX
              core since those are 0 .. n-1. On snax_split_cluster (n = 4) that is core 4.
              A cluster hjson cannot describe it, so it is the one role derived rather than
              read. BingoDFG appends it as an extra accelerator when is_host_as_acc, and
              its validator demands every __host_bingo_kernel_* node sit there.

    `platform` is optional and is used ONLY to cross-check: two generators, two headers,
    one truth. Pass it where a parsed occamy.h is at hand and a stale pair becomes a loud
    failure instead of a silently wrong placement.

    A cluster that genuinely lacks an engine leaves SNAX_CORE_<ROLE> undefined (its
    SNAX_HAS_<ROLE>_CORE is 0), and that RAISES. Placing a SIMD node on hart 0 because no
    SIMD block exists is exactly the failure this map was introduced to remove.
    """
    header = Path(roles_header) if roles_header is not None else _DEFAULT_ROLES_HEADER
    if not header.exists():
        raise FileNotFoundError(
            f"generated core-role map {header} is missing. It is mirrored from the snax "
            f"checkout by `make snax-sw-gen`; run that (or `make sw`) for the active CFG "
            f"first.")
    defines = _parse_defines(header)

    n = _require_define(defines, "SNAX_CLUSTER_NUM_CORES", header)
    if n < 2:
        raise ValueError(f"a cluster needs at least two cores, {header} says {n}")

    if platform is not None:
        expected = int(platform["num_cores_per_cluster"])
        if expected != n:
            raise ValueError(
                f"{header} says SNAX_CLUSTER_NUM_CORES={n}, but occamy.h says "
                f"N_CORES_PER_CLUSTER={expected}. One of the two generated headers is "
                f"stale against the active CFG -- rerun `make snax-sw-gen`.")

    roles = {}
    for role in ("gemm", "simd", "xdma"):
        name = f"SNAX_CORE_{role.upper()}"
        if name not in defines:
            raise ValueError(
                f"{header} does not define {name}: this cluster has no {role.upper()} "
                f"engine, so a {role} node cannot be placed on it.")
        roles[role] = defines[name]

    roles["dm"] = _require_define(defines, "SNAX_CORE_IDMA", header)
    roles["host"] = n

    # Cheap cross-check of the invariant upstream enforces.
    if roles["dm"] != n - 1:
        raise ValueError(
            f"{header} puts the DMA ISA on hart {roles['dm']}, but snRuntime's DM core is "
            f"the last of {n}. snrt_is_dm_core() and the cluster disagree.")

    return roles


def guard_cluster_count(param, platform, output_dir, output_offload_file_name):
    expected = param.get("num_clusters")
    if expected is None:
        raise KeyError("params.hjson must define num_clusters")

    expected = int(expected)
    actual = int(platform["num_clusters_per_chiplet"])
    if expected == actual:
        return True

    offload_path = os.path.join(output_dir, output_offload_file_name)
    if os.path.exists(offload_path):
        os.remove(offload_path)
    print(
        f"WARNING: workload expects num_clusters={expected}, "
        f"but HW platform has N_CLUSTERS_PER_CHIPLET={actual}. "
        f"Not generating {offload_path}.",
        file=sys.stderr,
    )
    return False


def guard_chiplet_count(param, platform, output_dir, output_offload_file_name):
    expected = param.get("num_chiplets")
    if expected is None:
        raise KeyError("params.hjson must define num_chiplets")

    expected = int(expected)
    actual = int(platform["num_chiplets"])
    if expected == actual:
        return True

    offload_path = os.path.join(output_dir, output_offload_file_name)
    if os.path.exists(offload_path):
        os.remove(offload_path)
    print(
        f"WARNING: workload expects num_chiplets={expected}, "
        f"but HW platform has N_CHIPLETS={actual}. "
        f"Not generating {offload_path}.",
        file=sys.stderr,
    )
    return False
