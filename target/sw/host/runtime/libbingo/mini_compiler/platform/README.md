# platform/ — the machine

What the compiler is allowed to assume about the hardware, which is: nothing it has not
read from a generated header. Nothing here is hardcoded, because a node placed on the wrong
hart does not fault — it programs that hart's accelerator at the same CSR offsets and
reports success.

| file | |
|---|---|
| `bingo_platform.py` | `core_roles()` — which cluster core carries which engine, read from `snax_core_roles_defs.h`, which snaxgen derives from the cluster hjson. `parse_platform_cfg()` for the chiplet/cluster/core counts, the memory chiplets and the task-descriptor geometry; `platform_descriptor_geometry()` for `BingoDFG`'s default widths; `load_cluster_cfg()` / `writer_junction_index()` for cfg-derived device constants. And `_engine_of_kernel()`, which maps a kernel name to the role it needs — the pair to `core_roles`, and what the validator checks a placement against. |
| `bingo_helpers.py` | Datapath rules the workloads also use: `chiplet_addr_transform_loc`, and the xDMA transfer-size alignment check. |

## The failure this directory exists to prevent

A cluster that genuinely lacks an engine leaves `SNAX_CORE_<ROLE>` undefined, and
`core_roles()` **raises**. Placing a SIMD node on hart 0 because no SIMD block exists is
exactly the failure the generated map was introduced to remove.

Read the map; do not remember it.

## Finding the generated headers

`_DEFAULT_ROLES_HEADER` and `_DEFAULT_PLATFORM_HEADER` locate their files through
`_bingo_paths.target_sw()`, which searches for a marker (`_bingo_paths.repo_root()`).
Counting `..` is right for one directory depth only, and a wrong count surfaces as a
missing generated header, from a module that never mentions paths.
