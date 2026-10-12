#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Fetch the FFN weights the L2 FFN experiment runs on into cache/ (Hugging Face range requests).

One dense FFN per model: gate_proj, up_proj, down_proj and the RMSNorm gain in front of them
(post_attention_layernorm), as stored (BF16). The HTTP helpers are sweep/common/weights/fetch_weights.py's;
each repository is pinned to a commit. A tensor lands at cache/<repo with / -> __>/<name>.bin and
cache/manifest.json records, per model, the repository, commit, layer and per tensor its dtype,
shape, bytes and SHA-256.

    python3 fetch_weights.py [--models KEY,...]

A tensor whose file already has the right size is not fetched again.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
MANIFEST = os.path.join(CACHE, "manifest.json")

_spec = importlib.util.spec_from_file_location(
    "weights_fetch", os.path.join(HERE, "..", "common", "weights", "fetch_weights.py"))
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)

# key: repo, commit, layer, the FFN's tensors (gate, up, down, the gain before them), the shapes
# gate / up [inter, hidden] and down [hidden, inter] as stored (nn.Linear: [out, in])
MODELS = {
    # DeepSeek-V2-Lite's layer 0 is its one dense layer (first_k_dense_replace = 1)
    "dsv2-lite-l0": ("deepseek-ai/DeepSeek-V2-Lite", "604d5664dddd88a0433dbae533b7fe9472482de0", 0,
                     "model.layers.{l}.mlp.{p}_proj.weight",
                     "model.layers.{l}.post_attention_layernorm.weight", 2048, 10944),
}


def tensor_names(key):
    _, _, layer, proj, gain, _, _ = MODELS[key]
    return {p: proj.format(l=layer, p=p) for p in ("gate", "up", "down")} | \
        {"gain": gain.format(l=layer)}


def path_of(repo, name):
    return os.path.join(CACHE, repo.replace("/", "__"), name + ".bin")


def fetch(key):
    repo, rev, layer, _, _, hidden, inter = MODELS[key]
    wm = cf.weight_map(repo, rev)
    out = {}
    for role, name in tensor_names(key).items():
        shard = wm[name]
        base, meta = cf.header(repo, rev, shard)
        m = meta[name]
        lo, hi = m["data_offsets"]
        p = path_of(repo, name)
        if not (os.path.exists(p) and os.path.getsize(p) == hi - lo):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p + ".part", "wb") as f:
                for a in range(lo, hi, cf.PIECE):
                    f.write(cf._get(cf.url(repo, rev, shard), base + a,
                                    base + min(a + cf.PIECE, hi) - 1))
            os.replace(p + ".part", p)
        out[role] = dict(name=name, dtype=m["dtype"], shape=m["shape"], bytes=hi - lo,
                         shard=shard, sha256=cf.sha256(p))
    want = {"gate": [inter, hidden], "up": [inter, hidden], "down": [hidden, inter],
            "gain": [hidden]}
    for role, shp in want.items():
        if out[role]["shape"] != shp:
            raise ValueError(f"{key}: {out[role]['name']} is {out[role]['shape']}, not {shp}")
    return dict(repo=repo, revision=rev, layer=layer, hidden=hidden, inter=inter, tensors=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", default="", help="comma-separated model keys (default: all)")
    a = ap.parse_args()
    keys = a.models.split(",") if a.models else list(MODELS)
    try:
        with open(MANIFEST) as f:
            man = json.load(f)
    except (OSError, ValueError):
        man = {}
    for key in keys:
        man[key] = fetch(key)
        os.makedirs(CACHE, exist_ok=True)
        with open(MANIFEST + ".part", "w") as f:
            json.dump(man, f, indent=1)
        os.replace(MANIFEST + ".part", MANIFEST)
        nb = sum(t["bytes"] for t in man[key]["tensors"].values())
        print(f"[fetch] {key}: {nb / 2**20:.1f} MiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
