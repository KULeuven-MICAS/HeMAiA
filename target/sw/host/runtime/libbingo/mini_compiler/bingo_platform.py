import os
import re
import sys
from pathlib import Path


_DEFINE_RE = re.compile(r"^\s*#define\s+(\w+)\s+([0-9a-fA-FxX]+)\b")


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

    return {
        "num_chiplets": num_chiplets,
        "num_clusters_per_chiplet": _require_define(
            defines, "N_CLUSTERS_PER_CHIPLET", occamy_h_path),
        "num_cores_per_cluster": _require_define(
            defines, "N_CORES_PER_CLUSTER", occamy_h_path),
        "chiplet_ids": chiplet_ids,
        # Whether this platform HAS a memory chiplet, and where. A workload that stages
        # its inputs and goldens there on a config with none does not fault -- the
        # addresses are simply unmapped, so the loads return junk and the checks compare
        # junk against junk. Defaulted rather than required so an older generated header
        # still parses. See util/sim/common/bingo_data_staging.py.
        "num_mem_chips": defines.get("N_MEM_CHIPS", 0),
        "mem_chip_loc_x": defines.get("MEM_CHIP_LOC_X", 0),
        "mem_chip_loc_y": defines.get("MEM_CHIP_LOC_Y", 0),
    }


# The generated role map, mirrored into the device tree by `make snax-sw-gen`
# (target/sw/Makefile). Resolved from THIS file's location so every workload's
# core_roles(platform) keeps working unchanged: bingo_platform.py sits at
# target/sw/host/runtime/libbingo/mini_compiler/, so parents[4] is target/sw.
_DEFAULT_ROLES_HEADER = (
    Path(__file__).resolve().parents[4] / "device" / "runtime" / "snax" /
    "snax_core_roles_defs.h"
)


def core_roles(platform, roles_header=None):
    """Which cluster core carries which engine, for the platform being built.

    A workload that hardcodes these breaks the moment the cluster changes shape, and it
    breaks QUIETLY on the device side: a kernel placed on the wrong hart programs THAT
    hart's accelerator at the same CSR offsets and reports success. So nothing here is
    assumed -- the per-engine roles are READ from the generated header and the two
    snRuntime-structural ones are derived from the core count:

        gemm  the core carrying snax_acc_cfg,  from SNAX_CORE_GEMM
        simd  the core carrying snax_simd_cfg, from SNAX_CORE_SIMD
        xdma  the core carrying snax_xdma_cfg, from SNAX_CORE_XDMA
        dm    the last SNAX core. snRuntime's SNRT_CLUSTER_DM_CORE_NUM is 1, so
              snrt_is_dm_core() selects exactly this hart, and it is the only one with the
              DMA ISA -- a dm* instruction anywhere else traps.
        host  N_CORES_PER_CLUSTER exactly -- i.e. one past the last SNAX core, since the
              SNAX cores are 0 .. n-1. On snax_split_cluster (n = 4) that is core 4.
              BingoDFG appends the chiplet-local host core as an extra accelerator when
              is_host_as_acc, and its validator demands every __host_bingo_kernel_* node
              sit there.

    `dm` and `host` stay derived rather than read: `host` because a cluster hjson cannot
    describe the chiplet's CVA6 at all, `dm` because it is snRuntime's own structural
    choice. Upstream ENFORCES the agreement -- it rejects any cfg that puts the DMA ISA
    anywhere but the last core -- so SNAX_CORE_IDMA == n - 1 is guaranteed and is
    cross-checked below rather than treated as a second source of truth.

    A cluster that genuinely lacks an engine leaves SNAX_CORE_<ROLE> undefined (its
    SNAX_HAS_<ROLE>_CORE is 0), and that RAISES here. Placing a SIMD node on hart 0
    because no SIMD block exists is exactly the silent-wrong-accelerator failure this map
    was introduced to remove.
    """
    n = int(platform["num_cores_per_cluster"])
    if n < 2:
        raise ValueError(f"a cluster needs at least two cores, got {n}")

    header = Path(roles_header) if roles_header is not None else _DEFAULT_ROLES_HEADER
    if not header.exists():
        raise FileNotFoundError(
            f"generated core-role map {header} is missing. It is mirrored from the snax "
            f"checkout by `make snax-sw-gen`; run that (or `make sw`) for the active CFG "
            f"first.")
    defines = _parse_defines(header)

    # Two generators, two headers, one truth. occamy.h's N_CORES_PER_CLUSTER and
    # snaxgen's SNAX_CLUSTER_NUM_CORES describe the same cluster; a mismatch means one
    # of them is stale against $(CFG), which is the failure mode the cfg-identity stamp
    # in target/sw/Makefile exists to prevent. Fail loudly rather than place nodes off
    # a half-updated picture.
    num_cores = _require_define(defines, "SNAX_CLUSTER_NUM_CORES", header)
    if num_cores != n:
        raise ValueError(
            f"{header} says SNAX_CLUSTER_NUM_CORES={num_cores}, but occamy.h says "
            f"N_CORES_PER_CLUSTER={n}. One of the two generated headers is stale "
            f"against the active CFG -- rerun `make snax-sw-gen`.")

    roles = {}
    for role in ("gemm", "simd", "xdma"):
        name = f"SNAX_CORE_{role.upper()}"
        if name not in defines:
            raise ValueError(
                f"{header} does not define {name}: this cluster has no {role.upper()} "
                f"engine, so a {role} node cannot be placed on it.")
        roles[role] = defines[name]

    roles["dm"] = n - 1
    roles["host"] = n

    # Cheap cross-check of the invariant upstream enforces.
    idma = defines.get("SNAX_CORE_IDMA")
    if idma is not None and idma != roles["dm"]:
        raise ValueError(
            f"{header} puts the DMA ISA on hart {idma}, but snRuntime's DM core is "
            f"hart {roles['dm']} (the last of {n}). snrt_is_dm_core() and the cluster "
            f"disagree.")

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
