# xdma_ops_lib — shared tooling for the per-op xDMA sweep workloads

This library (`util/sim/xdma/xdma_ops_lib.py`) + the LUT helper
(`util/sim/xdma/build_lut.py`) back the per-op `xdma_<op>_1cluster` workloads under
`target/sw/host/apps/offload_bingo_hw/single_chip/workloads/`.

`xdma_ci_ops_1cluster` runs every xDMA op in one DFG (good for a single
functional pass). These per-op dirs split that out so **one dir == one op**,
and each dir's `main_bingo.py` is also a **shape sweep** that emits per-config
cycle counts for a look-up table (LUT).

## What a per-op dir contains

| File | Purpose |
|------|---------|
| `main_bingo.py` | op name + `CONFIGS` (shape list) → calls `run_op_workload()` |
| `params.hjson`  | `num_clusters`, `array_shape` (for layout-conversion ops) |
| `Makefile`      | build rules (emits `<op>_data.h`, `offload_bingo_hw.h`, `configs.json`) |

All the real machinery — golden generators, the Load→op→Store→Check chain
builder, and the data/DFG emitters — lives once in
`util/sim/xdma/xdma_ops_lib.py` (the `REGISTRY` of op handlers). To add a
config, edit only the `CONFIGS` list. To add an op, add a handler to the
registry and create a per-op dir with the three files above.

## How the sweep + correctness work

For every entry in `CONFIGS` the DFG emits:

```
Load_input (iDMA L3→L1) → xDMA <op> → Store (L1→L3) → Host check vs numpy golden
```

Configs run serially, so the `BINGO_TRACE_XDMA_RUN_*` markers produce one
`XDMA_RUN` event **per config, in CONFIGS order**. Every config is also checked
against its golden, so the sweep doubles as the per-op functional test.

## Building / running

```bash
# build one op's workload (inside docker)
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
     WORKLOAD=xdma_transpose_1cluster DEV_APP=snax-bingo-offload

# or simulate it end-to-end
python3 target/sim/automation/test/0_start_single_chiplet_sim.py \
    --host-app-type offload_bingo_hw --chip-type single_chip \
    --workload xdma_transpose_1cluster --dev-app snax-bingo-offload
```

## Building the LUT

After a sim run:

```bash
cd target/sim && make traces        # produces bin/logs/bingo_trace.json
python3 util/sim/xdma/build_lut.py \
    --trace target/sim/bin/logs/bingo_trace.json \
    --configs <workload>/configs.json
```

`build_lut.py` zips the ordered `XDMA_RUN` `dur_cc` values with `configs.json`
(same order) and prints a `shape → cycles` table (CSV).
