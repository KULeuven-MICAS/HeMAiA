# DeepSeek-V2-Lite layer 1 on sixteen chiplets

Layer 1 (MLA attention + the 64-expert MoE; layers 1 to 26 all have this shape), every step on the device, as ONE
BINGO graph on `target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson`.

```
x1 -> RMSNorm -> MLA attention -> + x1 = h -> RMSNorm -> MoE -> + h
```

| directory | what it runs |
|---|---|
| `prefill/full` | a fresh 1,024-token prompt at the final block mappings ([`../common/dsv2_layer_prefill_mc.py`](../common/dsv2_layer_prefill_mc.py)): every token's x normed into HBM, the attention of `../../fa/dsv2_l1_prefill/t1024` (eight chunks of 128 queries), the glue, the MoE of `../../moe/dsv2_l1_prefill/full_t512` in two passes of 512 rows (`layer.moe_passes`), each with its own route, placement and weight stream, the residual in each pass's reduce, the output into HBM; run with `target/sim/automation/sweep/llm_layer/dsv2_prefill/stages.sh layer1_prefill_full` |
| `decode/layer` | the decode of one token over a 1,024-token cache ([`../common/dsv2_decode_mc.py`](../common/dsv2_decode_mc.py)): the attention split by keys (`../../fa/common/fa_decode_ks_mc/`, a cached 64-key tile a chip), every weight multiply a matrix-vector product, the MoE's weights streamed once the attention is done (`../../moe/common/moe_decode_mc/`) |

Each directory holds its `Makefile`, `main_bingo.py` and `params.hjson`; everything else is generated. `params.hjson`
holds the platform and graph switches every block shares, then `attn`, `moe` and `layer`; the driver documents every
step, the golden and the parameters.

## What the layer needs beyond its blocks

- **Args phases.** Every task's args struct sits in its cluster's L1 while its phase runs, beside the static buffer
  plan and the scheduler's lists. The driver splits each cluster's tasks into args phases
  (`passes/bingo_dfg_emit.py`): each phase's args share one L1 region and come in by a DM task when the cluster has
  finished the phase before.
- **Paged task lists** (`list_phases`). The scheduler's per-task list entries follow the args phases through two L1
  windows, so they do not grow with the whole graph either (the device runtime's
  `bingo_hw_offload_manager_paged` reads them).
- **One L3 arena** (`layer.l3_arena`). The attention's weight ring and scratch and the router's landing area share one
  region of each chip's L3 with the activations, which the full layer's host image needs to fit.
- **One exchange buffer** for both blocks' partials, one L3 region for both weight rings.
- **The attention's y stored by halves** (`y_store_halves`), so the root holds half of a chunk's y at a time.

The prefill runs are in `target/sim/automation/sweep/llm_layer/dsv2_prefill/runs/`.
