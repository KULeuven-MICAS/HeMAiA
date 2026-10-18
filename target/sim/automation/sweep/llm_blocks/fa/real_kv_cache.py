#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""A REAL layer-1 latent cache of DeepSeek-V2-Lite over real text, for the L2 FlashAttention experiment.

The FA experiment streams the KV cache from the memory chiplets' HBM, CREST-compressed. A
compression ratio means something only on a real cache, so this computes one, in float32 from the
checkpoint's BF16 weights, following modeling_deepseek.py (the snax_cluster golden package's
util/reference.py conventions, which the device's absorbed MLA uses):

    tokens   real text (a pinned Project Gutenberg book), the checkpoint's own tokenizer, BOS first
    layer 0  embedding -> MLA attention (causal, YaRN RoPE) -> residual -> dense FFN -> residual
    layer 1  input_layernorm -> kv_a_proj_with_mqa -> per token:
               c_unit [512]  the latent after the RMS normalisation, WITHOUT the kv_a_layernorm
                             gain (the device folds the gain into W_UK and W_UV)
               kpe    [64]   k_pe rotated at its own position, interleaved pair order
             and the hidden state x1 entering layer 1, for the decode token's query

Layer 0's attention runs batched in the ABSORBED form (queries mapped into the latent space);
the last token is then re-computed by util/reference.attention -- the EXPANDED form, one token
against the cache -- and the two must agree, which checks the batched implementation.

    python3 real_kv_cache.py [--tokens 8193]

Writes cache/dsv2_l1_kv_<tokens>.npz (c_unit, kpe, x1, token ids) and records its inputs in
cache/manifest.json. Needs the `tokenizers` package.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
REPO_ID = "deepseek-ai/DeepSeek-V2-Lite"
REV = "604d5664dddd88a0433dbae533b7fe9472482de0"      # the commit sweep/common/weights and sweep/llm_blocks/ffn pin
TEXT_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"   # Pride and Prejudice
F32 = np.float32

_spec = importlib.util.spec_from_file_location(
    "weights_fetch", os.path.join(HERE, "..", "common", "weights", "fetch_weights.py"))
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)


def _dsv2_util():
    repo = next(p for p in Path(HERE).resolve().parents if (p / "Bender.yml").exists())
    sn = os.environ.get("SNAX_DSV2", str(repo / "deps/snitch_cluster/target/snitch_cluster/sw/apps/dsv2"))
    if sn not in sys.path:
        sys.path.insert(0, sn)
    from util import model, reference
    return model, reference


TENSORS = (["model.embed_tokens.weight"] +
           [f"model.layers.0.{n}" for n in (
               "input_layernorm.weight", "self_attn.q_proj.weight",
               "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight",
               "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight",
               "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
               "mlp.down_proj.weight")] +
           [f"model.layers.1.{n}" for n in (
               "input_layernorm.weight", "self_attn.kv_a_proj_with_mqa.weight")])


def fetch(name, wm):
    """One tensor at REV into cache/ (as stored), returned as float32 [out, in] / [n]."""
    shard = wm[name]
    base, meta = cf.header(REPO_ID, REV, shard)
    m = meta[name]
    lo, hi = m["data_offsets"]
    p = os.path.join(CACHE, REPO_ID.replace("/", "__"), name + ".bin")
    if not (os.path.exists(p) and os.path.getsize(p) == hi - lo):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".part", "wb") as f:
            for a in range(lo, hi, cf.PIECE):
                f.write(cf._get(cf.url(REPO_ID, REV, shard), base + a,
                                base + min(a + cf.PIECE, hi) - 1))
        os.replace(p + ".part", p)
    if m["dtype"] != "BF16":
        raise ValueError(f"{name} is {m['dtype']}")
    u = np.fromfile(p, dtype="<u2")
    return (u.astype(np.uint32) << 16).view(F32).reshape(m["shape"]), \
        dict(shape=m["shape"], shard=shard, sha256=cf.sha256(p))


class Layer:
    """The attention weights of one layer in [in, out] orientation, with util/reference's
    W interface (d, in_norm, wq, wdkv, kv_norm, w_uk(h), w_uv(h), wo)."""

    def __init__(self, d, t, i):
        g = lambda n: t[f"model.layers.{i}.{n}"]
        self.d = d
        self.in_norm = g("input_layernorm.weight")
        self.wdkv = np.ascontiguousarray(g("self_attn.kv_a_proj_with_mqa.weight").T)
        if i == 0:     # layer 1 contributes only its cache rows: its norm and W_DKV
            self.wq = np.ascontiguousarray(g("self_attn.q_proj.weight").T)
            self.kv_norm = g("self_attn.kv_a_layernorm.weight")
            self.wkvb = np.ascontiguousarray(g("self_attn.kv_b_proj.weight").T)   # [512, 4096]
            self.wo = np.ascontiguousarray(g("self_attn.o_proj.weight").T)
            self.post_norm = g("post_attention_layernorm.weight")
            self.mlp = {p: np.ascontiguousarray(g(f"mlp.{p}_proj.weight").T)
                        for p in ("gate", "up", "down")}

    def w_uk(self, h):
        b = h * (self.d.q_nope + self.d.v_head)
        return self.wkvb[:, b:b + self.d.q_nope]

    def w_uv(self, h):
        b = h * (self.d.q_nope + self.d.v_head) + self.d.q_nope
        return self.wkvb[:, b:b + self.d.v_head]


def layer0(L0, x, model, ref, chunk=256):
    """Layer 0 over every position of x [T, 2048] at once (causal), absorbed MLA."""
    d = L0.d
    T, H = x.shape[0], d.heads
    pos = np.arange(T)
    ang = pos[:, None] * model.yarn_inv_freq(d)[None, :]
    amp = model.yarn_get_mscale(d.rope_factor, d.mscale) / \
        model.yarn_get_mscale(d.rope_factor, d.mscale_all_dim)
    cos, sin = np.cos(ang) * amp, np.sin(ang) * amp                     # [T, 32]
    xn = ref.rmsnorm(x, L0.in_norm, d.rms_eps)
    q = (xn @ L0.wq).astype(F32).reshape(T, H, d.q_head)
    q_nope, q_pe = q[..., :d.q_nope], q[..., d.q_nope:]
    q_pe = model.rope_interleaved(q_pe.astype(np.float64), cos[:, None, :],
                                  sin[:, None, :]).astype(F32)
    ckv = (xn @ L0.wdkv).astype(F32)
    c_unit = ref.rmsnorm_unit(ckv[:, :d.kv_rank], d.rms_eps)
    kpe = model.rope_interleaved(ckv[:, d.kv_rank:].astype(np.float64), cos, sin).astype(F32)
    c_norm = (c_unit * L0.kv_norm).astype(F32)                          # [T, 512]
    q_lat = np.stack([q_nope[:, h] @ L0.w_uk(h).T for h in range(H)], axis=1).astype(F32)
    scale = F32(d.softmax_scale)
    o_lat = np.empty((T, H, d.kv_rank), dtype=F32)
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        s = (np.einsum("thc,kc->thk", q_lat[a:b], c_norm[:b]) +
             np.einsum("thr,kr->thk", q_pe[a:b], kpe[:b])).astype(F32) * scale
        mask = np.arange(b)[None, :] > np.arange(a, b)[:, None]           # key after query
        s[np.broadcast_to(mask[:, None, :], s.shape)] = -np.inf
        o_lat[a:b] = np.einsum("thk,kc->thc", ref.softmax(s, axis=-1), c_norm[:b])
    o = np.stack([o_lat[:, h] @ L0.w_uv(h) for h in range(H)], axis=1)   # [T, H, 128]
    attn = (o.reshape(T, -1) @ L0.wo).astype(F32)
    h_ = (x + attn).astype(F32)
    hn = ref.rmsnorm(h_, L0.post_norm, d.rms_eps)
    y, _ = ref.swiglu_mlp(hn, L0.mlp)
    return (h_ + y).astype(F32), attn, dict(c_unit=c_unit, kpe=kpe, xn=xn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tokens", type=int, default=8193,
                    help="the cached tokens plus the decode token (default 8192 + 1)")
    a = ap.parse_args()
    model, ref = _dsv2_util()
    d = model.load()
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    wm = cf.weight_map(REPO_ID, REV)
    t, info = {}, {}
    for n in TENSORS:
        t[n], info[n] = fetch(n, wm)
    print(f"[kv] weights in {time.time() - t0:.0f} s", flush=True)

    # ---- real text, the checkpoint's tokenizer ------------------------------------------
    from tokenizers import Tokenizer
    tp = os.path.join(CACHE, "tokenizer.json")
    if not os.path.exists(tp):
        with open(tp, "wb") as f:
            f.write(cf._get(cf.url(REPO_ID, REV, "tokenizer.json")))
    txp = os.path.join(CACHE, "pg1342.txt")
    if not os.path.exists(txp):
        with urllib.request.urlopen(TEXT_URL, timeout=120) as r, open(txp, "wb") as f:
            f.write(r.read())
    raw = open(txp, encoding="utf-8-sig").read()
    body = raw[raw.index("*** START OF"):]
    body = body[body.index("\n") + 1:]                               # past the licence banner
    ids = [100000] + Tokenizer.from_file(tp).encode(body, add_special_tokens=False).ids
    if len(ids) < a.tokens:
        raise ValueError(f"the text has {len(ids)} tokens, fewer than {a.tokens}")
    ids = np.array(ids[:a.tokens], dtype=np.int64)
    T = len(ids)

    # ---- layer 0, then layer 1's latent cache ---------------------------------------------
    L0, L1 = Layer(d, t, 0), Layer(d, t, 1)
    x0 = t["model.embed_tokens.weight"][ids].astype(F32)
    t1 = time.time()
    x1, attn0, st0 = layer0(L0, x0, model, ref)
    print(f"[kv] layer 0 over {T} tokens in {time.time() - t1:.0f} s", flush=True)

    # check: the last token through util/reference (expanded MLA, one token vs the cache)
    cache0 = ref.Cache(st0["c_unit"][:-1], st0["kpe"][:-1])
    chk = ref.attention(L0, st0["xn"][-1], T - 1, cache0)
    err = float(np.max(np.abs(chk["attn_out"] - attn0[-1])) / np.max(np.abs(chk["attn_out"])))
    print(f"[kv] layer-0 attention, last token, batched absorbed vs reference expanded: "
          f"max rel err {err:.2e}", flush=True)
    if err > 1e-3:
        raise AssertionError("the batched layer-0 attention disagrees with util/reference")

    xn1 = ref.rmsnorm(x1, L1.in_norm, d.rms_eps)
    ckv1 = (xn1 @ L1.wdkv).astype(F32)
    cos, sin = zip(*(model.rope_cos_sin(d, p) for p in range(T)))
    c_unit = ref.rmsnorm_unit(ckv1[:, :d.kv_rank], d.rms_eps)
    kpe = model.rope_interleaved(ckv1[:, d.kv_rank:].astype(np.float64), np.stack(cos),
                                 np.stack(sin)).astype(F32)
    out = os.path.join(CACHE, f"dsv2_l1_kv_{T}.npz")
    np.savez_compressed(out, c_unit=c_unit, kpe=kpe, x1=x1, ids=ids)
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    mp = os.path.join(CACHE, "manifest.json")
    man = json.load(open(mp)) if os.path.exists(mp) else {}
    man[f"dsv2_l1_kv_{T}"] = dict(repo=REPO_ID, revision=REV, text=TEXT_URL,
                                  text_sha256=cf.sha256(txp), tokens=T, file=os.path.basename(out),
                                  sha256=h, check_rel_err=err, tensors=info)
    with open(mp + ".part", "w") as f:
        json.dump(man, f, indent=1)
    os.replace(mp + ".part", mp)
    print(f"[kv] {out}: c_unit {c_unit.shape} |max| {np.max(np.abs(c_unit)):.3f}, "
          f"kpe {kpe.shape} |max| {np.max(np.abs(kpe)):.3f}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
