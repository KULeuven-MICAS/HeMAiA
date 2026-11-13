# Llama-3-8B on sixteen chiplets: one decoder layer

One decoder layer of Meta-Llama-3-8B, every step on the device, as ONE BINGO graph on the paper's platform
`target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson`: sixteen compute chiplets (two clusters, 8 MiB of L3 each) and four
memory chiplets with their HBM. It runs a prefill over a fresh 1,024-token prompt (causal, no cache) and the decode of
one token over a 1,024-token cache. The weights are SYNTHETIC with the model's shapes (hidden 4096, 32 query and 8
key/value heads of 128, a SwiGLU FFN of 14336, INT4 per output column), so the bytes streamed and the work are the
model's: [`common/llama3_data.py`](common/llama3_data.py).

```
x -> RMSNorm -> grouped-query attention -> + x = h -> RMSNorm -> SwiGLU FFN -> + h
```

## The workloads

| directory | what it runs |
|---|---|
| `full_layer_prefill/full` | the prefill: the attention's 16 passes of 64 rows and the FFN's 16, with the input norm, the glue and the final residual pass by pass beside them, every boundary tensor through HBM, paged task lists (`list_phases`); driver [`full_layer_prefill/common/llama3_layer_mc.py`](full_layer_prefill/common/llama3_layer_mc.py); run with `target/sim/automation/sweep/llm_layer/llama3_prefill/stages.sh fl_full` |
| `decode/layer_ks` | the decode: the attention split by keys (a key tile a chip, half the query heads a cluster), every weight multiply a matrix-vector product, the FFN's weights streamed once the attention is done; driver [`common/llama3_decode_layer_mc.py`](common/llama3_decode_layer_mc.py) |

Each directory holds its `Makefile`, `main_bingo.py` and `params.hjson`; everything else is generated.

## The code

| file | what it builds |
|---|---|
| [`common/llama3_data.py`](common/llama3_data.py) | the model's configuration, the synthetic weights, and the device-exact golden of the attention (`LlamaAttnData`); registers the FFN with the ffn driver's data |
| [`common/llama3_attn_mc.py`](common/llama3_attn_mc.py) | the prefill attention block: a query head a cluster, a key/value group on a column's chip pair, passes of 64 tokens, the causal flash sweep (`libs/blocks/fa_sweep.py` with its mask), the output projection reduced in transit; also the FA block's Llama variant (`../fa/llama3_prefill`) |
| [`full_layer_prefill/common/llama3_layer_mc.py`](full_layer_prefill/common/llama3_layer_mc.py) | the whole-layer prefill: both blocks in one graph (the FFN of [`../ffn/common/ffn_prefill_mc/`](../ffn/common/ffn_prefill_mc/README.md), passes of 64 rows that stream the weights again), the steps between them, args phases, one L3 ring region |
| [`common/llama3_decode_mc.py`](common/llama3_decode_mc.py), [`common/llama3_decode_ks_mc.py`](common/llama3_decode_ks_mc.py), [`common/llama3_decode_layer_mc.py`](common/llama3_decode_layer_mc.py) | the decode attention (head-parallel and split by keys) and the whole decode layer with the FFN of [`../ffn/common/ffn_decode_mc/`](../ffn/common/ffn_decode_mc/README.md) |
| [`common/llama3.mk`](common/llama3.mk) | the build |

The attention alone is the FA block's Llama variant, `../fa/llama3_prefill` (run with
`target/sim/automation/sweep/llm_blocks/fa/stages.sh llama3`); the FFN alone is `../ffn/llama3_prefill` (run with
`target/sim/automation/sweep/llm_blocks/ffn/stages.sh llama3_<app>`).
