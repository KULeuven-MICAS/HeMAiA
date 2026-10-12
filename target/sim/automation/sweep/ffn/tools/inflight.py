import re, os, glob, sys, collections
# NOTE: scratch output goes to /tmp/opencode/w (mkdir -p it); run with the shims first on PATH (tools/bin)
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, "/users/micas/fkong/no_backup/HeMAiA/target/sim/automation/sweep/ffn")
import trace_ffn as TF, trim_traces
task = Path(sys.argv[1]); wl = trim_traces.wl_dir_of(task)
orders = TF.core_orders(wl / "offload_bingo_hw.h", wl / "final_dfg.csv")
L = task / "bin/logs"
S, E = f"DASM({TF.NOP(TF.RUN_START):08x})".encode(), f"DASM({TF.NOP(TF.RUN_END):08x})".encode()
def one(key):
    chip, cl, core = key
    p = L / (f"trace_chip_{chip}_hart_00000.log" if core == 4 and False else f"trace_chip_{chip}_hart_{1 + 4 * cl + core:05d}.dasm")
    if core == 4 or not p.exists(): return key, None
    ns = ne = 0; ls = le = None; last = None
    with open(p, "rb") as f:
        f.seek(TF.data_start(f))
        for ln in f:
            if S in ln: ns += 1; ls = int(ln.split()[0]) / 1e6
            elif E in ln: ne += 1; le = int(ln.split()[0]) / 1e6
            last = ln
    lt = int(last.split()[0]) / 1e6 if last else None
    return key, (ns, ne, ls, le, lt)
if __name__ == "__main__":
    keys = [k for k in orders if k[2] != 4]
    with Pool(12) as pool: res = dict(pool.map(one, keys))
    # DM cores: harts 4 and 8 are core index 3 in the order file (core 3 = DM)
    kind = lambda n: re.sub(r"\d+", "#", n)
    run, wait, done = collections.Counter(), collections.Counter(), 0
    rows = []
    for k, v in res.items():
        if v is None: continue
        ns, ne, ls, le, lt = v; names = orders[k]
        if ns > ne: run[kind(names[ns - 1])] += 1; rows.append((k, "RUNNING", names[ns - 1], ls, lt))
        elif ne < len(names): wait[kind(names[ne])] += 1; rows.append((k, "waiting for", names[ne], le, lt))
        else: done += 1
    print("cores with all their tasks done:", done, "of", len(res))
    print("RUNNING (started, no end) by kind:", dict(run.most_common(12)))
    print("WAITING (next task not started) by kind:", dict(wait.most_common(12)))
    import json; json.dump([(list(k), s, n, a, b) for k, s, n, a, b in rows], open("/tmp/opencode/w/inflight.json", "w"))
