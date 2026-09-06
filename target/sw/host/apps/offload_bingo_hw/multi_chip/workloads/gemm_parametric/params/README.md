# Validated configurations for `gemm_parametric`

Each file here is one rung of the M/K/N-split ladder, and each has been run and
checked in RTL. They are the inputs the local CI suites under
`target/sim/automation/ci/local_ci/` name in their `HOST_DATA_CFG:` keys, so a
suite reproduces a known result rather than whatever happens to sit in
`../params.hjson`.

The file name is `grid<M>x<K>x<N>_<chips>chip_<clusters>c`: the worker grid
first, then the hardware it needs. `grid_M * grid_N` is the number of compute
chiplets and `grid_K` is the clusters per chiplet, so the two halves of the name
are consistent by construction. A trailing word distinguishes configurations
that share a grid and a platform but exercise a different code path.

A configuration is only valid against the RTL cfg it names: `guard_chiplet_count()`
and `guard_cluster_count()` compare `num_chips` / `clusters_per_chip` against the
platform header and silently SKIP the workload when they differ (a
`[skip] ... platform guard did not emit offload_bingo_hw.h` line in the build
log). Pairing is therefore part of the file, not a convention.

| file | RTL cfg | workers | real GEMM | what it adds | wall clock |
|---|---|---|---|---|---|
| `grid4x1x1_4chip_1c.hjson`             | `hemaia_tapeout_1c`                     | 4  | 4 x 64 x 32      | M split over chiplets                   | ~20 min |
| `grid2x1x2_4chip_1c.hjson`             | `hemaia_tapeout_1c`                     | 4  | 2 x 64 x 64      | `grid_N > 1`, strided D sub-blocks      | ~20 min |
| `grid3x1x1_3chip_1c_row.hjson`         | `hemaia_scalability_exp_3chip_1c_row`   | 3  | 3 x 64 x 32      | a chiplet at x = 2, non-2x2 topology    | 12 min  |
| `grid2x2x2_4chip_2c.hjson`             | `hemaia_tapeout_2c`                     | 8  | 2 x 32 x 64      | full M+K+N, first K reduction           | ~40 min |
| `grid2x2x2_4chip_2c_tilepack.hjson`    | `hemaia_tapeout_2c`                     | 8  | 4 x 64 x 128     | fast path + tile-packed partial D       | ~40 min |
| `grid2x2x2_4chip_2c_accumprevc.hjson`  | `hemaia_tapeout_2c`                     | 8  | 4 x 64 x 64      | K tiling through `accumPrevC`           | 38 min  |
| `grid2x4x2_4chip_4c.hjson`             | `hemaia_scalability_exp_4chip_4c`       | 16 | 64 x 2048 x 2048 | `grid_K = 4`, a 4-deep reduction chain  | 1h31m   |
| `grid4x4x4_16chip_4c.hjson`            | `hemaia_scalability_exp_16chip_4c_4mem` | 64 | 4 x 64 x 128     | 16 chiplets x 4 clusters                | 42 min  |
| `grid4x4x4_16chip_4c_thinM.hjson`      | `hemaia_scalability_exp_16chip_4c_4mem` | 64 | 4 x 8192 x 4096  | a shape that NEEDS the split image      | 7h14m   |

Real GEMM dimensions are `(M1*meshRow) x (K1*tileSize) x (N1*meshCol)`, which at
`array_shape: 1` is `(meshRow, tileSize, meshCol) = (1, 16, 32)`.

Wall-clock figures come from a loaded shared host and are worth +-60% at best.
Only `grid4x4x4_16chip_4c_thinM.hjson` is long enough to need planning, and it
is the one configuration in which the partitioned memory-chiplet image is
REQUIRED rather than merely exercised: replicated, its operands would need
32 MiB on a 16 MiB chip.
