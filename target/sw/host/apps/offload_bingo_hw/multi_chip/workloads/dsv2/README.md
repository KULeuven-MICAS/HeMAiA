# DeepSeek-V2-Lite on HeMAiA

Layer 1 of DeepSeek-V2-Lite (MLA attention + a 64-expert MoE), INT4 weights, checked byte-exact
against snax_cluster's golden (`target/snitch_cluster/sw/apps/dsv2`), grouped by platform:

```
dsv2/
  common/            what every dsv2 workload shares
    dsv2_datagen.py    the golden, cut to the mapping, and its staging (HBM, L3)
    dsv2_staged.py     the stage driver: params, data, weight rings, helpers, checks, emission
    dsv2.mk            the build a stage's Makefile includes
  two_chiplet/       one compute chiplet (4 clusters) + the memory chiplet with its HBM
    full_layer/        the whole layer, every stage, with the layer's tuned settings
    stage1_rmsnorm/ ... stage8_experts/
  sixteen_chiplet/   the multi-chiplet mapping -- planned, see its README
  dsv2_summary.py    one number sheet per run: layer time, phases, engines, HBM bytes, checks
```

## two_chiplet: the full layer, and one workload per stage

Platform: `hemaia_twochiplet_16MBL3_4cluster.hjson`, the only cfg whose memory chiplet has an HBM.
Weights are pushed by the memory chiplet into a ring per cluster in L3 and copied into L1 slabs by
each cluster's DM core.

`full_layer/` runs the layer: stages 1–8 with the layer's tuned settings — 1,705.8 µs for one
token, INT4, byte-exact; 5,750.9 µs for a pass of four tokens (1,437.7 µs a token, 1.19× over
one at a time). It holds no code of its own; the layer's code is the stage chain.

Each stage directory holds **that stage's code only** (`main_bingo.py`: `build(S)` and
`checks(S)`), and builds on the stage before it — the stage-N workload builds stages 1..N. The
build is the order one monolithic builder would use (every stage's compute, then every stage's
checks), so the graph, and every core's in-order task stream, is the one the measured runs used.

| Stage | Adds | Checks | 1 token, stages 1–N | 4-token pass |
|---|---|---|---|---|
| `stage1_rmsnorm` | x (HBM) → RMSNorm + quantiser → x8 in L3 | x8 | 4.0 µs | — |
| `stage2_wq` | W_Q by head on the four clusters | q16 | 155.0 µs | 177.1 µs |
| `stage3_wdkv` | W_DKV, ahead of W_Q in the streams | kv16 | 178.2 µs | 217.5 µs |
| `stage4_latent_rope_wuk` | the latent's RMSNorm, RoPE, W_UK absorbed | cn, c8, rot, kpe8, qt16 | 210.3 µs | 285.4 µs |
| `stage5_attention` | cache append, Q8, MLA attention over L+1 keys | q8, key tile, ot, xt | 274.7 µs | 374.5 µs |
| `stage6_wuv_wo` | W_UV, W_O by output columns, the residual → h | oh, attn, h | 378.1 µs | 529.8 µs |
| `stage7_router` | MoE RMSNorm, router, softmax, top 6 | hn, lg16, probs, route | 429.4 µs | 603.1 µs |
| `stage8_experts` | shared + routed experts, the combine → out (the layer) | experts, out | 1,705.8 µs | 5,750.9 µs (1,437.7 µs a token) |

Times are measured RTL runs (500 MHz chips, 3.5 GHz DDR D2D), each with the settings in its
directory. `params.hjson` is the stage's one-token run; `params_t4.hjson` its 4-token pass.

### Build and run

```bash
# the sweep runner: the full layer (task_dsv2.yaml), or every stage (task_dsv2_stages.yaml)
python3 target/sim/automation/sweep/dsv2/run_dsv2_sweep.py [-f task_dsv2_stages.yaml] [--sw-only]
grep -E "Check|PASS|FAIL" target/sim/automation/sweep/dsv2/task_*/bin/uart_chip_0_0.log

# or one stage, as the runner does it (inside the container)
make apps CFG_OVERRIDE=target/rtl/cfg/hemaia_twochiplet_16MBL3_4cluster.hjson \
     HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=multi_chip DEV_APP=snax-bingo-offload \
     WORKLOAD=dsv2/two_chiplet/full_layer
```

`WORKLOAD` is the directory's path under `workloads/`; the binary's name flattens it
(`offload_bingo_hw_multi_chip_dsv2_two_chiplet_full_layer_snax-bingo-offload`). A 4-token
pass builds with `DATA_CFG=<stage dir>/params_t4.hjson`, or from a copy of the directory whose
`params.hjson` is the 4-token one.

### Writing a stage

`build(S)` calls the previous stage's `build` first, then adds its blocks with the helpers on `S`
(`S.gemv`, `S.deq`, `S.view`, `S.pipe.add`, ...), leaving on `S` what later stages read. `checks(S)`
calls the previous stage's `checks`, then stashes its values with `S.check`. A pass of several
tokens stashes stages 5–8's checks inline in `build`, where they were measured. A stage that must
run BEFORE part of an earlier one hooks into it: stage 3 puts W_DKV ahead of stage 2's W_Q through
`S.before_wq`.

## sixteen_chiplet

Planned: see [sixteen_chiplet/README.md](sixteen_chiplet/README.md).
