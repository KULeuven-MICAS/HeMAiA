import sys, os, re, glob, json, collections
# NOTE: scratch output goes to /tmp/opencode/w (mkdir -p it); run with the shims first on PATH (tools/bin)
from pathlib import Path
import argparse
import hjson
REPO = str(next(p for p in Path(__file__).resolve().parents if (p / "Bender.yml").exists()))
W = f"{REPO}/target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/ffn"
sys.path.insert(0, f"{W}/common"); sys.path.insert(0, f"{REPO}/target/sw/host/runtime/libbingo/mini_compiler")
sys.path.insert(0, f"{REPO}/target/sim/automation/sweep/llm_blocks/ffn")
wl = sys.argv[1]; task = Path(sys.argv[2])
SN = f"{REPO}/deps/snitch_cluster"
import _bingo_paths  # noqa
from bingo_platform import load_cluster_cfg, parse_platform_cfg
import ffn_prefill_mc as M
args = argparse.Namespace(output_dir="/tmp/opencode/w/ready_gen", data_h=None, output_offload_file_name="x.h",
                          cfg=f"{W}/{wl}/params.hjson", hwcfg=f"{SN}/target/snitch_cluster/cfg/snax_split_cluster.hjson",
                          platformcfg=f"{REPO}/target/sw/shared/platform/generated/occamy.h", dsv2_dir=f"{SN}/target/snitch_cluster/sw/apps/dsv2")
os.makedirs(args.output_dir, exist_ok=True)
p = dict(hjson.load(open(args.cfg)))
S = M.PBuild(args, p, parse_platform_cfg(args.platformcfg), load_cluster_cfg(args.hwcfg))
S.build()
S.chain_program_order()
dfg = S.dfg
import trace_ffn as TF, trim_traces
rows = TF.tasks(task, trim_traces.wl_dir_of(task))
done = {r["name"] for r in rows}
inflight = {tuple(k): (s, n) for k, s, n, a, b in json.load(open("/tmp/opencode/w/inflight.json"))}
running = {n for (k, (s, n)) in inflight.items() if s == "RUNNING"}
print("completed spans:", len(done), "running:", len(running))
name = lambda t: t.node_name
nodes = {name(t): t for t in dfg.node_list}
ready = []
for t in dfg.node_list:
    n = name(t)
    if n in done or n in running: continue
    if t.assigned_core_id == 4 and "kernel_exit" in str(t.kernel_name): continue
    preds = list(dfg.predecessors(t))
    unmet = [name(q) for q in preds if name(q) not in done]
    if not unmet: ready.append(n)
print("READY (all predecessors done), not done, not running:", len(ready))
kind = lambda n: re.sub(r"\d+", "#", n)
print(collections.Counter(kind(n) for n in ready).most_common(12))
for n in sorted(ready)[:25]: print("  ", n)
json.dump({"ready": ready}, open("/tmp/opencode/w/ready.json", "w"))
# the unmet predecessors of the root cluster-0 waiters, for the record
for n in ("d2a_i2_g0_k00_cl0", "swi_c0m0_i3_g0_k00_cl0", "gu_c2_i3_g0_k00_cl0", "Ld_c4_i3_g0_k00_cl0"):
    t = nodes[n]; preds = list(dfg.predecessors(t))
    print(f"{n}: {len(preds)} predecessors; not done: {[name(q) for q in preds if name(q) not in done]}")
