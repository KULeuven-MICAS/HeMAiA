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
    }


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
