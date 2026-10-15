#!/usr/bin/env python3
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Fetch the checkpoint layers the level-2 workloads read into cache/ (Hugging Face range requests).

The MoE and FA workloads read DeepSeek-V2-Lite's layer 1 from here; the FFN's own fetcher
(sweep/llm_blocks/ffn/fetch_weights.py) uses this file's HTTP helpers.

Each repository is pinned to the commit its API reports when fetched. For every model: the
safetensors index (or the single model.safetensors), each shard's header, then one ranged GET per
tensor (in 64 MiB pieces) for the tensors of the sampled layers that match the model's weight,
side-data or auxiliary regex. A tensor lands, as stored, at cache/<repo with / -> __>/<name>.bin;
cache/manifest.json records the repository, commit, layers, and per tensor its role, dtype, shape,
bytes and SHA-256.

    python3 fetch_weights.py [--models KEY,...] [--threads N]

A tensor whose file already has the right size is not fetched again.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
MANIFEST = os.path.join(CACHE, "manifest.json")
PIECE = 64 << 20

# key: repo, format, payload kind, layers, weight regex, side-data regex, auxiliary regex.
# Side data = scales and zero points of the quantised formats; auxiliary = tensors the device pack
# needs but that are not weights (the RMSNorm gains of DeepSeek-V2-Lite).
MODELS = {
    # DeepSeek-V2-Lite (27 layers; layer 0 dense, 1..26 MoE): layer 1, the MoE and the attention
    "dsv2-lite-bf16": ("deepseek-ai/DeepSeek-V2-Lite", "BF16 (MoE, MLA)", "bf16", [1],
                       r"(_proj(_with_mqa)?|mlp\.gate)\.weight$", None, r"layernorm\.weight$"),
}


def _get(url, a=None, b=None, binary=True):
    h = {"Range": f"bytes={a}-{b}"} if a is not None else {}
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300) as r:
                data = r.read()
            if a is not None and len(data) != b - a + 1:
                raise IOError(f"short read {len(data)} of {b - a + 1}")
            return data
        except Exception as e:  # transient network errors: retry with back-off
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url} {a}-{b}: {last}")


def revision(repo):
    """The commit the repository's main branch points at."""
    return json.loads(_get(f"https://huggingface.co/api/models/{repo}"))["sha"]


def url(repo, rev, f):
    return f"https://huggingface.co/{repo}/resolve/{rev}/{f}"


def weight_map(repo, rev):
    try:
        return json.loads(_get(url(repo, rev, "model.safetensors.index.json")))["weight_map"]
    except RuntimeError:
        _, meta = header(repo, rev, "model.safetensors")
        return {k: "model.safetensors" for k in meta if k != "__metadata__"}


_HDR = {}


def header(repo, rev, shard):
    if (repo, shard) not in _HDR:
        n = struct.unpack("<Q", _get(url(repo, rev, shard), 0, 7))[0]
        _HDR[(repo, shard)] = (8 + n, json.loads(_get(url(repo, rev, shard), 8, 8 + n - 1)))
    return _HDR[(repo, shard)]


def path_of(repo, name):
    return os.path.join(CACHE, repo.replace("/", "__"), name + ".bin")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(16 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fetch_tensor(repo, rev, name, shard):
    base, meta = header(repo, rev, shard)
    m = meta[name]
    lo, hi = m["data_offsets"]
    p = path_of(repo, name)
    if not (os.path.exists(p) and os.path.getsize(p) == hi - lo):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".part", "wb") as f:
            for a in range(lo, hi, PIECE):
                f.write(_get(url(repo, rev, shard), base + a, base + min(a + PIECE, hi) - 1))
        os.replace(p + ".part", p)
    return dict(dtype=m["dtype"], shape=m["shape"], bytes=hi - lo, shard=shard, sha256=sha256(p))


def select(names, layers, wre, sre, are):
    lay = re.compile(r"layers\.(\d+)\.")
    out = {}
    for k in names:
        mm = lay.search(k)
        if not mm or int(mm.group(1)) not in layers:
            continue
        if re.search(wre, k):
            out[k] = "weight"
        elif sre and re.search(sre, k):
            out[k] = "side"
        elif are and re.search(are, k):
            out[k] = "aux"
    return out


def load_manifest():
    try:
        with open(MANIFEST) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", default="", help="comma-separated model keys (default: all)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    keys = a.models.split(",") if a.models else list(MODELS)
    man = load_manifest()
    for key in keys:
        repo, label, kind, layers, wre, sre, are = MODELS[key]
        rev = revision(repo)
        wm = weight_map(repo, rev)
        sel = select(wm, layers, wre, sre, are)
        for shard in sorted({wm[k] for k in sel}):
            header(repo, rev, shard)
        t0 = time.time()
        with ThreadPoolExecutor(a.threads) as ex:
            got = dict(zip(sel, ex.map(lambda k: fetch_tensor(repo, rev, k, wm[k]), sel)))
        tensors = {k: dict(role=sel[k], **got[k]) for k in sorted(sel)}
        man[key] = dict(repo=repo, revision=rev, format=label, kind=kind, layers=layers, tensors=tensors)
        os.makedirs(CACHE, exist_ok=True)
        with open(MANIFEST + ".part", "w") as f:
            json.dump(man, f, indent=1)
        os.replace(MANIFEST + ".part", MANIFEST)
        nb = sum(t["bytes"] for t in tensors.values())
        roles = {r: sum(1 for t in tensors.values() if t["role"] == r) for r in ("weight", "side", "aux")}
        print(f"[fetch] {key} {repo}@{rev[:12]}: {roles}, {nb / 2**30:.2f} GiB ({time.time() - t0:.0f} s)",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
