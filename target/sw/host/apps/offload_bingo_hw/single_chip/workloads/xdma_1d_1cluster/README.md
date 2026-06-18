# xdma_1d — single xDMA 1D-copy test

A minimal functional workload that exercises just the xDMA 1D copy
(`__snax_bingo_kernel_xdma_1d_copy`). The DFG is a single chain:

```
Load_input (iDMA L3→L1) → XDMA_copy (xdma_1d_copy L1→L1) → Store (L1→L3) → Host check
```

The copy is the identity, so the golden equals the input; the host check
confirms the bytes round-trip unchanged. This is the simplest xDMA smoke test
(see the per-op `xdma_<op>_1cluster` dirs for shape/size sweeps with cycle LUTs).

## Files

| File | Purpose |
|------|---------|
| `main_bingo.py`       | builds the load→copy→store→check DFG |
| `xdma_1d_datagen.py`  | emits `input_data` + `golden_copy` (identity) |
| `params.hjson`        | `num_clusters`, `size` (copy bytes, multiple of 64) |
| `Makefile`            | build rules (emits `xdma_1d_data.h`, `offload_bingo_hw.h`) |

## Build / run

```bash
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
     WORKLOAD=xdma_1d DEV_APP=snax-bingo-offload

python3 target/sim/automation/test/0_start_single_chiplet_sim.py \
    --host-app-type offload_bingo_hw --chip-type single_chip \
    --workload xdma_1d --dev-app snax-bingo-offload
```

Change the copy size by editing `size` in `params.hjson`.
