# DeepSeek-V2-Lite on HeMAiA

DeepSeek-V2-Lite's two kinds of decoder layer, each ONE BINGO graph on the paper's platform,
`target/rtl/cfg/hemaia_16c4m_8MBL3_2cluster.hjson`: sixteen compute chiplets (two clusters, 8 MiB of L3 each) and four
memory chiplets with their HBM. Layer 0 is the model's only dense layer; layers 1 to 26 have the MoE, so layer 1
stands for them. Each layer runs a prefill over a fresh 1,024-token prompt (causal, no cache) and the decode of one
token over a 1,024-token cache, every step on the device. The blocks themselves are built and tuned on their own in
`../fa`, `../moe` and `../ffn`.

```
layer 0:  x1 -> RMSNorm -> MLA attention -> + x1 = h -> RMSNorm -> dense SwiGLU FFN -> + h
layer 1:  x1 -> RMSNorm -> MLA attention -> + x1 = h -> RMSNorm -> MoE              -> + h
```

```
dsv2/
  common/
    dsv2_layer_prefill_mc.py  the whole-layer prefill driver (layer{0,1}/prefill/full): the attention of
                         ../fa/common/fa_stages_mc.py and the MoE (layer 1) or the dense FFN (layer 0) in one graph,
                         the input RMSNorm, the glue and the residual, the blocks meeting through HBM
    dsv2_decode_mc.py    the one-token decode driver (layer{0,1}/decode/layer)
    dsv2_prefill.mk      the build
  layer0/                layer 0 (MLA + the dense FFN), see layer0/README.md
    prefill/full         the 1,024-token prefill: all eight attention chunks and the FFN in eight passes of 128 rows
    decode/layer         the one-token decode: the attention split by keys, every weight multiply a matrix-vector
                         product, the FFN's weights streamed once the attention is done
  layer1/                layer 1 (MLA + the MoE), see layer1/README.md
    prefill/full         the 1,024-token prefill: all eight attention chunks and the MoE in two passes of 512 rows
    decode/layer         the one-token decode, as layer 0's with the MoE
```

Run the prefills with `target/sim/automation/sweep/llm_layer/dsv2_prefill/stages.sh layer{0,1}_prefill_full`.
