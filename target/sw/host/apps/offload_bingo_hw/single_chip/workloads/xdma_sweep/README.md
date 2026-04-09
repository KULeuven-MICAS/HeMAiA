# xdma_sweep — xDMA RTL cycle-count characterization

This workload runs a **whole sweep of xDMA ops × sizes in a single DFG** and
measures per-config cycle counts via the `BINGO_TRACE_XDMA_RUN_*` markers
embedded in the device xDMA kernels — no host-side mcycle bracketing.

## Per-config DFG

For every config in `CONFIGS` (inside `main_bingo.py`) the DFG emits:

```
Load_Input → Xdma_Op
```

where `Xdma_Op` is one of `xdma_1d_copy`, `xdma_transpose_2d`, or
`xdma_submatrix_2d` with the configured dimensions.  The `XDMA_RUN` trace
event dur_cc is the cycle count we report.

## Running the sweep

From the repo root:

```bash
bash scripts/run_xdma_sweep.sh
```

That script builds the workload, runs the RTL sim **once**, calls
`make bingo-vis-traces`, and correlates XDMA_RUN events with configs
(in order) to emit `inputs/rtl_luts/xdma.csv`.

Fit the linear model + save figures with:

```bash
python scripts/characterize_xdma.py
```

## Adding new configs

Edit the `CONFIGS` list at the top of `main_bingo.py`.  Each entry is
`(op, size, rows, cols, elem_bytes)`.  For `copy_1d`, `size` is the total
bytes; for 2D ops, `rows × cols × elem_bytes` gives the total bytes.

## Files

| File | Purpose |
|------|---------|
| `main_bingo.py`           | CONFIGS list + DFG builder + configs.json dump |
| `xdma_sweep_datagen.py`   | Per-config input buffer generator              |
| `Makefile`                | Build rules                                    |
| `configs.json`            | Generated: list of configs for the shell driver |
| `xdma_sweep_data.h`       | Generated: per-config data header              |
