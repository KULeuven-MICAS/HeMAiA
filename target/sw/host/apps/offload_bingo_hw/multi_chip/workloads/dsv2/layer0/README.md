# DeepSeek-V2-Lite layer 0 on sixteen chiplets

Layer 0 is DeepSeek-V2-Lite's only dense layer: the same MLA attention as layer 1, then a dense SwiGLU FFN (hidden
2,048, 10,944 intermediate columns) instead of the MoE.

```
x1 -> RMSNorm -> MLA attention -> + x1 = h -> RMSNorm -> dense SwiGLU FFN -> + h
```

| directory | what it runs |
|---|---|
| `prefill/full` | a fresh 1,024-token prompt at the final block mappings ([`../common/dsv2_layer_prefill_mc.py`](../common/dsv2_layer_prefill_mc.py)): every token's x normed into HBM, the attention of `../../fa/dsv2_l1_prefill/t1024` (eight chunks of 128 queries), the glue, the FFN of `../../ffn/dsv2_l0_prefill/full` in eight passes of 128 rows, the final residual, the output into HBM; paged task lists (`list_phases`); run with `target/sim/automation/sweep/llm_layer/dsv2_prefill/stages.sh layer0_prefill_full` |
| `decode/layer` | the decode of one token over a 1,024-token cache ([`../common/dsv2_decode_mc.py`](../common/dsv2_decode_mc.py)): the attention split by keys (a cached 64-key tile a chip), every weight multiply a matrix-vector product, the FFN's weights streamed once the attention is done |

The attention is layer 1's: the attention's data (its checkpoint weights, input_layernorm gain and the cache of hidden
states) exists for layer 1 only, so the layer runs layer 1's MLA (the same shapes, so the same timing) in front of layer
0's FFN with layer 0's post_attention_layernorm gain (the FFN data's, model `dsv2-lite-l0`). The FFN block alone is
`../../ffn/dsv2_l0_prefill` (stage apps and `full`).

Each directory holds its `Makefile`, `main_bingo.py` and `params.hjson`; everything else is generated.
