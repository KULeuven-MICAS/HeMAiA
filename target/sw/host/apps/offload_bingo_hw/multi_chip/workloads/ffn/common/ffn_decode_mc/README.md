# ffn_decode_mc: the FFN decode block

The FFN block of the decode layers (`DBuild`, which `dsv2/common/dsv2_decode_mc.py` and
`llama3/common/llama3_decode_layer_mc.py` extend as their FfnPart): one LLM's dense SwiGLU FFN for ONE token, on the
prefill's mapping (`ffn_prefill_mc.PBuild`) with every weight multiply on VersaCore's one-row GEMV (the (1, 4, 32)
shape, `__snax_bingo_kernel_gemv`, 128 weights a pass), INT4 weights, tensor-parallel over every cluster of the 16 + 4
chiplet platform. The platform, the weight rings, the stamps and the setup are `chiplet_mc.py`'s; the data and the
golden are `ffn_data.py`'s (one token).

| module | what it holds |
|---|---|
| `__init__.py` | `DBuild`, the decode mixins on `ffn_prefill_mc.PBuild` |
| `params.py` | the decode's switches and its refusals of the prefill's multi-row ones (`DBuild.__init__`) |
| `memory.py` | X, Y and the weights in HBM, every L1 buffer a row |
| `rows.py` | X's fetch, the gate\|up GEMVs, the A16 row into the chip's A8 row |
| `passes.py` | the pass: the down GEMVs, the ChainGathers, Y's row, Y into HBM |
| `checks.py` | every array task a GEMV, the data checks |

## The mapping

The prefill's distribution (columns over every cluster, the down projection split by output columns, the rings, the
paired INT4 chunks, the same HBM weight images), only rows in L1. PBuild plans it as one pass of one 16-row tile (its
sizes); every buffer and task is the token's row.

```
X        the token's INT8 row in a_row (THE X DELIVERY below) in the HBM of the root's memory chip: the root's host
         queues that memory chip's push engine, which multicasts it into every chip's L3 behind one flag; every
         cluster fetches it into L1 in one flagged copy
gate|up  per weight chunk as it lands: GEMV X . [gate | up] chunk (the paired INT4 chunk, a_row A) -> the chunk's FP16
         row, its gate columns then its up columns, contiguous; dequantised (x the chunk's column factors, one FP16
         row); SwiGLU of the row's two halves into the cluster's A16 row (FP16, its columns in order)
A8       the A16 row spread into a_row's FP16 form on the xDMA (each 8-byte atom, 4 values, 16 bytes apart in a row
         zeroed before the window), quantised on the SIMD (Fp16ToInt8 keeps every value's place, so the INT8 lands in
         a_row) into the cluster's run of the chip's A8 row (a_row, both clusters' columns, 2 bytes a column), and that
         run copied into the other cluster's A8 row
down     per weight chunk (a slice of the cluster's outputs): GEMV of the chip's A8 row . down chunk -> the chip's
         partial of the slice (the token's row, FP16), into an output slot of the cluster's exchange buffer
reduce   per group of chain_group slices: one ChainGather through every chip, each hop's xDMA adding its partial in
         transit, into the root's fold slots; the gathers run one at a time, alternating between the two clusters'
         xDMAs, the first behind X's multicast (a multicast never meets a gather). chains_after_stream: the first
         gather also behind every cluster's last down GEMV, so every gather runs after the last weight push of every
         memory chip has landed (no gather beside a push; out_slots must hold every slice of a cluster)
Y        the root's clusters copy each folded slice into one L1 row on the root's cluster 0, Y_raw in column order;
         once both clusters' last copies are done (every gather has ended, every weight has landed) the root's DM core
         writes the row to the HBM of the root's memory chip, so no cross-chip write of Y meets a ChainGather or a
         weight push. Y_raw is FP16 before the final dequantise (x s_dn per output column), as the prefill leaves it: a
         layer applies s_dn where it adds the residual. y_dequant: the root's SIMD applies s_dn to the row first (one
         task), so the HBM holds Y16
```

The SIMD tasks are the 32-byte-beat stream kernels on whole beats: the dequantise (MulF16, a beat every 16 columns of
the chunk's row), SwiGLU (a beat a gate block), the quantise. The xDMA moves whole beats of 8 atoms (32 values), so the
spread reads the A16 row and writes its a_row form up to a whole beat; both buffers are padded to that and zeroed with
it, and the quantise reads only the row's own bytes. No array task here is a GEMM, so the GEMM-config reuse rewrites
nothing (`checks.py` proves every array task of the block a GEMV).

**The X delivery.** X is the token's INT8 row in a_row: value c at byte (c // 4) * 8 + c % 4, 2 x hidden bytes
(`gemv_utils.a_row_bytes`); the GEMV reads the first 4 bytes of each 8-byte word, so the other 4 are free (zero here).
It lands at `_xland(0)`, 2 x hidden bytes, behind the flag `<TAG>_xtflag`. A producer holding the FP16 row makes it with
two existing kernels: an xDMA spread of the row's 8-byte atoms 16 bytes apart into an FP16 row zeroed once
(`__snax_bingo_kernel_xdma_6d`; the gaps stay zero across uses), then `simd_fp16_to_int8` over that row (4 x hidden
bytes in, a_row out); a producer holding an INT8 A-layout tile with the token in row 0 compacts it with one xDMA copy
of the first 8-byte word of every 64-byte block. A contiguous INT8 row has no cheap conversion (4-byte runs; the xDMA
moves 8-byte atoms).

**The weights** stream as in the prefill's first pass: each cluster's gate|up and down chunks sit in its memory chip's
HBM; the memory chip pushes them into the chip's L3 ring and the DM core loads each into an L1 slab behind its landed
flag.

**The program order.** The compiler orders every device core's program (mini_compiler `passes/bingo_dfg_schedule.py`);
the edges this driver adds are every ordering the data needs.

**Timing.** The start stamp after every chip's setup (the scales loaded, the row buffers zeroed); the end stamp after
the HBM write of Y. X's multicast and the weight stream are inside the window.

**Checks**, byte for byte against the golden of the device's arithmetic (`ffn_data.py`, the reduce's fold order): the
Y_raw (y_dequant: Y16) row in the root memory chip's HBM, every column; the root's A8 row in a_row (its free bytes
zero). The golden is per row and elementwise.

## Parameters

`ffn_prefill_mc`'s (its sizes: `blocks`, `first_blocks`, `dn_cols`, `root_cols`, `dn_balance`, `chain_alt`,
`out_slots`, `chain_group`, `start_per_chip`, `gather_relay_only`, the data's), for one pass of one token (`tokens` 1).
The decode refuses what needs more than one pass, row tile or post block (`l3_overlay`, `y_scatter`, `split_last`,
`split_first_dn`, `restream`, `rows`, `post_blocks`, `x_blocks`) and the ablation arms (`sync barrier`, `reduce root`,
`x_cast unicast`). Its own:

| switch | default | what it sets |
|---|---|---|
| `y_dequant` | false | the root dequantises Y_raw's row (Y16 into HBM) |
| `chains_after_stream` | false | every gather after every cluster's last down GEMV (out_slots holds every slice) |
