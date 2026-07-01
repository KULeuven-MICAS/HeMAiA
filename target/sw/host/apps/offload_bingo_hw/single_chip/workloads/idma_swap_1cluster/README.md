# idma_swap_1cluster

Per-op sweep workload for the **swap** op: an adjacent element-pair swap over a flat
buffer, `dst[i] = src[i ^ 1]` (`x[1],x[0],x[3],x[2],…`). This is the data half of RoPE's
`rotate_half` (`xswap`), produced **on-device** so in-layer `rope_q`/`rope_k` can swap a
runtime `Q`/`K` buffer (which the datagen can no longer precompute). It is implemented as
two strided **iDMA** copies (`__snax_bingo_kernel_idma_pairwise_swap`: odd→even and
even→odd slots, ~2 cyc/elem) — the same one-time swap staging used in the `snax-xdma-rope`
reference app. There is no HW xDMA swap path (the xDMA reader AGU strides forward only and
at 8-byte spatial granularity, so a 2-byte fp16 pair swap isn't expressible there).

Each `CONFIGS` entry (in `main_bingo.py`) runs `Load → iDMA swap → Store → Check`,
checking the device output byte-exact against a numpy golden — the sweep IS the per-op
functional test. It covers `num_elems ∈ {64,256,1024}` × `elem_bytes ∈ {1,2,4}`. The op
runs on the iDMA, so it emits `IDMA_RUN` trace markers (not the `XDMA_RUN` the xDMA
cycle-LUT keys on); cycle characterization is not this workload's purpose. Shared
machinery lives in `util/sim/xdma/xdma_ops_lib.py` (the `SwapOp` handler).
