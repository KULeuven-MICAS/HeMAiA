# L2 MoE: DeepSeek-V2-Lite's layer-1 MoE on 16 compute + 4 memory chiplets, with an ablation

The prefill block runs whole with `stages.sh` (below), as the five rungs of the ablation in `workloads/moe/dsv2_l1_prefill_ablation/`.

## Question

The paper's three mechanisms, each against its conventional counterpart on the same hardware and
the same compute:
- **broadcast**: one D2D router multicast into every cluster's L1 and a landed flag, against
  every cluster pulling the root's copy with its iDMA;
- **reduce**: the xDMA ChainGather folding in transit, against the root pulling every partial and
  folding it on its SIMD;
- **sync**: the BINGO hardware scheduler resolving every cross-chip dependency per edge, against
  barriers: the vanilla HeMAiA chip barrier at every cross-chip sync point.

How much does each buy inside a real LLM layer whose weights stream from the memory chiplets,
alone and together?

## Measurement

As the FFN's (`sweep/llm_blocks/ffn/README.md`). **Phases** add the route: the last slot chip's record.

## Prefill (`stages.sh`)

DeepSeek-V2-Lite's layer-1 MoE over a 512-token prompt, the whole block at full size (`workloads/moe/common/moe_prefill_mc.py`,
through `workloads/moe/common/moe_stages_mc.py`, params `stage`), one rung of the ablation a run:
`./stages.sh abl_<rung>` (`workloads/moe/dsv2_l1_prefill_ablation/<rung>`). X and the router's weights come from the
HBM of the root's memory chip, multicast by its push engine inside the window (params `x_hbm`). `all` is the tuned
block: 64-column output slices, an expert slot's gate|up in row blocks of 32, `l1_capacity` 1,000,000, `desc_in_image`,
`args_phases`, `sh_quant_blocks` with staged mid-beat chunks.

| stage | rung | what it adds | run | result |
|---|---|---|---|---|
| `abl_base` | R0 | the tuned block with X unicast to every chiplet, the partials folded at the root, a barrier kernel after every kernel | `stg_abl_base` | PASS 7/7, 6,185.5 us, 35.0 % MAC |
| `abl_bcast` | R1 | + the X multicast | `stg_abl_bcast_1` | PASS 7/7, 5,408.5 us, 40.1 % MAC |
| `abl_bcast_reduce` | R2 | + the chain reduce | `stg_abl_bcast_reduce` | PASS 7/7, 3,436.1 us, 63.1 % MAC |
| `abl_fused` | R3 | + fused kernels: a barrier only where data crosses chiplets, one before each slice's reduce (`barrier_at` cross) | `stg_abl_fused` | PASS 7/7, 2,857.5 us, 75.8 % MAC |
| `abl_all` | R4 | per-edge synchronisation, no barrier | `stg_full_t512_3` | PASS 7/7 (y included), 2,720.7 us, 79.6 % MAC (35.50 GMAC useful; 83.8 % executed); the arrays bind it (94.0 % padded ceiling), 484.8 MB streamed for 291.4 MB of weights (gate|up chunks reloaded per row block) |

`runs/` holds these five runs; the `all` rung's run keeps its tag `stg_full_t512_3`, which the paper's data cites.

Concurrent ChainGathers run the same way round the ring: two in opposite directions deadlock on the
D2D links.

## Files

- `parse_moe.py` (with `../ffn/parse_ffn.py`): a run's phases into `results/moe.csv`.
- `stages.sh`: the prefill. A stage run's task spans: `../../common/timeline.py`; the generator alone:
  `../../common/gen_check.sh moe/<variant> <out dir> <log>`.
- **Workloads:** `target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/moe/`: `common/moe_prefill_mc.py` (the
  prefill driver, on `util/sim/multichip/chiplet_mc.py`), `common/moe_stages_mc.py` (params `stage`), `common/moe_prefill_data.py`
  (weights, route, golden), `common/moe.mk`, `dsv2_l1_prefill_ablation/<rung>/params.hjson`; `common/moe_mc.py` and
  `common/moe_data.py` serve the decode block (`common/moe_decode_mc.py`).
