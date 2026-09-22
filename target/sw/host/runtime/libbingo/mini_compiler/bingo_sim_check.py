# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Run the CYCLE-ACCURATE model over a compiled DFG, as a compile-time hang check.
#
# WHY, ON TOP OF bingo_validate_no_hang. That validator is static: it proves
# properties of the lowered graph (tags pair up, no cell aliases, every check has
# a producer). It cannot see anything that depends on TIMING -- queue depths,
# arbiter order, the in-order descriptor stream stalling behind a full per-core
# waiting queue, or a cross-die predicate racing its dep-set. Those are exactly
# the failures that survive a static check and then hang the machine.
#
# So this runs the real model instead: push the descriptor list through the same
# queues, matrices and FSMs the RTL has, with randomised work delays over several
# seeds, and assert that every task completes and every dependency is respected.
#
# ON BY DEFAULT, BUT NEVER FATAL BY ACCIDENT. The model comes from the
# bingo_hw_manager bender dependency (Bender.yml), located with `bender path`.
# If the checkout is missing, or the pinned model is too old for this bridge,
# that is reported and the build continues -- firmware generation must not depend
# on it. Only a genuine hang fails the build. Set BINGO_SIM_CHECK=0 to skip.

import contextlib
import io
import os
import dataclasses
import random
import subprocess
import sys


def _model_roots():
    """Where to look for the cycle model, in order.

    1. BINGO_MODEL_PATH -- an explicit override, for developing against a working
       tree rather than the pinned checkout.
    2. `bender path bingo_hw_manager` -- the project's OWN dependency resolution.
       bingo_hw_manager is already a bender dependency of HeMAiA (Bender.yml), so
       this is the checkout a CI build actually has, at the pinned revision. No
       ad-hoc relative paths: those break the moment the tree is laid out
       differently, and they silently pick up whatever happens to be next door.
    """
    env = os.environ.get("BINGO_MODEL_PATH")
    if env:
        yield os.path.abspath(env), "BINGO_MODEL_PATH"
    try:
        out = subprocess.run(["bender", "path", "bingo_hw_manager"],
                             capture_output=True, text=True, timeout=30,
                             cwd=_hemaia_root())
        if out.returncode == 0 and out.stdout.strip():
            yield out.stdout.strip().splitlines()[-1].strip(), "bender"
    except (OSError, subprocess.SubprocessError):
        pass


def _hemaia_root():
    """The repo root, so `bender` runs where Bender.yml lives."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(12):
        if os.path.isfile(os.path.join(d, "Bender.yml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.getcwd()


class ModelUnavailable(RuntimeError):
    """The cycle model could not be imported. Not a compile error."""


def _load_model():
    """Import the model package, or raise ModelUnavailable saying where we looked."""
    tried = []
    for root, how in _model_roots():
        tried.append(f"{root} (via {how})")
        if not os.path.isdir(os.path.join(root, "model")):
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from model.bingo_sim import BingoSimulator, SimConfig, QueueDepths
            from model.bingo_sim_chiplet import TaskDescriptor
            return BingoSimulator, SimConfig, QueueDepths, TaskDescriptor, root
        except ImportError as exc:      # present but broken -- say which
            raise ModelUnavailable(
                f"found a model at {root} but could not import it: {exc}") from exc
    raise ModelUnavailable(
        "cycle model not found. It normally comes from the bingo_hw_manager "
        "bender dependency; run `bender checkout`, or set BINGO_MODEL_PATH. "
        f"Looked in: {', '.join(tried) or '(nothing)'}")


def _mask(cores):
    m = 0
    for c in cores or ():
        m |= 1 << c
    return m


def _geometry(dfg):
    """Model geometry, derived from the graph rather than assumed."""
    chips = sorted({n.assigned_chiplet_id for n in dfg.node_list})
    return {
        "chip_index": {c: i for i, c in enumerate(chips)},
        "n_chiplets": len(chips),
        "n_clusters": max((n.assigned_cluster_id for n in dfg.node_list), default=0) + 1,
        "n_cores": max((n.assigned_core_id for n in dfg.node_list), default=0) + 1,
    }


def _gate_k(dfg, gating_node) -> int:
    """How many branches this gate lets through, as declared. 1 when unstated."""
    src = getattr(gating_node, "_pred_source_node", None) or gating_node
    for f in getattr(dfg, "_cond_forks", []):
        if f.source is src or getattr(f, "_gating_node", None) is gating_node:
            return int(f.select.get("k", 1))
    for _, _, d in dfg.out_edges(src, data=True):
        if d.get("cond") and d.get("cond_dic"):
            return int(d["cond_dic"].get("k", 1))
    return 1


def _cerf_scenarios(dfg):
    """The routing outcomes the graph must survive, not just the static one.

    A conditional graph deadlocks for a SUBSET of predicates or for none at all,
    so activating every group -- which is all this used to do -- is the one case
    that is guaranteed not to exercise skipping. These four bracket it: nothing
    runs, everything runs, and the lowest and highest k groups run, which are the
    two orderings a real router can produce.
    """
    if not getattr(dfg, "_gating_to_targets", {}):
        return ["all"]
    return ["all", "none", "low_k", "high_k"]


def _descriptors(dfg, TaskDescriptor, geo, delay_of, scenario="all"):
    """The compiled DFG as per-chiplet model descriptor lists, in stream order.

    `scenario` picks which of each gate's CERF groups are activated; see
    _cerf_scenarios.
    """
    node_to_group = getattr(dfg, "_node_to_cerf_group", {})
    gating_targets = getattr(dfg, "_gating_to_targets", {})
    type_map = {"dummy": 1, "gating": 2}
    per = {i: [] for i in range(geo["n_chiplets"])}
    for n in dfg.bingo_stream_order():
        controlled = set(getattr(n, "cerf_write_groups", ()) or ())
        if controlled:
            groups = sorted({node_to_group[t] for t in gating_targets.get(n, set())
                             if t in node_to_group})
            k = max(0, min(_gate_k(dfg, n), len(groups)))
            if scenario == "none":
                activated = set()
            elif scenario == "low_k":
                activated = set(groups[:k])
            elif scenario == "high_k":
                activated = set(groups[len(groups) - k:])
            else:
                activated = set(groups)
            cerf_w = sum(1 << g for g in activated)
            cerf_c = sum(1 << g for g in controlled)
        else:
            cerf_w = cerf_c = 0
        t = TaskDescriptor(
            task_type=type_map.get(n.node_type, 0),
            task_id=n.node_id,
            assigned_chiplet_id=geo["chip_index"][n.assigned_chiplet_id],
            assigned_cluster_id=n.assigned_cluster_id,
            assigned_core_id=n.assigned_core_id,
            dep_check_en=bool(n.dep_check_enable),
            dep_check_code=_mask(n.dep_check_list),
            dep_set_en=bool(n.dep_set_enable),
            dep_set_all_chiplet=bool(n.remote_dep_set_all),
            dep_set_chiplet_id=geo["chip_index"].get(n.dep_set_chiplet_id, 0),
            dep_set_cluster_id=n.dep_set_cluster_id or 0,
            dep_set_code=_mask(n.dep_set_list),
            dep_check_tag=n.dep_check_tag,
            dep_set_tag=n.dep_set_tag,
            cond_exec_en=bool(n.cond_exec_en),
            cond_exec_group_id=n.cond_exec_group_id,
            cond_exec_invert=bool(n.cond_exec_invert),
            cerf_write_mask=cerf_w,
            cerf_controlled_mask=cerf_c,
        )
        t.work_delay = delay_of(n)
        per[geo["chip_index"][n.assigned_chiplet_id]].append(t)
    return per


def _real_predecessors(dfg):
    """For each dispatching task, the dispatching tasks it truly waits on.

    Dummies never reach a core, so the real relation is recovered by walking
    back THROUGH them.
    """
    import collections
    out = collections.defaultdict(set)
    real = lambda n: n.node_type != "dummy"
    for v in dfg.node_list:
        if not real(v):
            continue
        stack, seen = list(dfg.predecessors(v)), set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            if real(u):
                out[v.node_id].add(u.node_id)
            else:
                stack.extend(dfg.predecessors(u))
    return out


def simulate_for_hangs(dfg, seeds=3, work_delay_range=(20, 200),
                       h2h_latency=10, h2h_latency_jitter=40,
                       max_cycles=4_000_000, verbose=True):
    """Run the compiled DFG on the cycle model. Raises ValueError on a hang.

    D2D jitter is ON by default: a cross-die predicate that races its dep-set
    only misbehaves when the two can arrive out of order, so a zero-jitter run
    would report a graph as safe that is not.
    """
    BingoSimulator, SimConfig, QueueDepths, TaskDescriptor, root = _load_model()
    geo = _geometry(dfg)
    preds = _real_predecessors(dfg)
    order = dfg.bingo_stream_order()
    n_real = sum(1 for n in order if n.node_type != "dummy")

    scenarios = _cerf_scenarios(dfg)
    for scenario, seed in [(sc, sd) for sc in scenarios for sd in range(seeds)]:
        rng = random.Random(1000 + seed)
        fixed = {n.node_id: (rng.randint(*work_delay_range)
                             if n.node_type != "dummy" else 0) for n in order}
        # ADAPT TO THE MODEL WE ACTUALLY GOT. bender hands us a PINNED checkout,
        # which may predate fields this bridge would like to set. Passing an
        # unknown keyword would crash the firmware build over a version skew, so
        # ask the dataclass what it supports and drop the rest -- loudly, because
        # a missing field can silently weaken the check.
        want = dict(
            num_chiplets=geo["n_chiplets"],
            num_clusters_per_chiplet=geo["n_clusters"],
            num_cores_per_cluster=geo["n_cores"],
            queue_depths=QueueDepths(waiting=8, ready=8, checkout=8, done=32),
            work_delay_range=work_delay_range,
            h2h_latency=h2h_latency,
            h2h_latency_jitter=h2h_latency_jitter,
            push_interval=5,
            random_seed=seed,
            cerf_scope="carried",     # what the RTL does with the carried predicate
        )
        supported = {f.name for f in dataclasses.fields(SimConfig)}
        dropped = sorted(set(want) - supported)
        cfg = SimConfig(**{k: v for k, v in want.items() if k in supported})
        if dropped and seed == 0 and scenario == scenarios[0] and verbose:
            print(f"  NOTE: this model predates {', '.join(dropped)}; "
                  f"those aspects are NOT checked.")
            if "cerf_scope" in dropped:
                print("  NOTE: without cerf_scope the model applies CERF writes to "
                      "every chiplet instantly, which the hardware does not do -- "
                      "cross-die conditional execution is NOT validated here.")
        sim = BingoSimulator(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            sim.load_tasks(_descriptors(dfg, TaskDescriptor, geo,
                                        lambda n: fixed[n.node_id],
                                        scenario=scenario))
            res = sim.run(max_cycles=max_cycles)

        if res.deadlock_detected:
            missing = sorted(sim._all_task_ids - res.completed_task_ids)
            names = {n.node_id: n.node_name for n in order}
            raise ValueError(
                f"sim hang check: the compiled graph DEADLOCKED on the cycle "
                f"model (seed {seed}, routing scenario '{scenario}'). "
                f"{len(missing)} task(s) never completed, first few: "
                + ", ".join(f"{i}:{names.get(i, '?')}" for i in missing[:6]))
        missing = sim._all_task_ids - res.completed_task_ids
        if missing:
            names = {n.node_id: n.node_name for n in order}
            raise ValueError(
                f"sim hang check: {len(missing)} task(s) never completed "
                f"(seed {seed}, routing scenario '{scenario}'): "
                + ", ".join(f"{i}:{names.get(i, '?')}" for i in sorted(missing)[:6]))

        disp, done = {}, {}
        for e in res.trace.events:
            if e.event_type == "TASK_DISPATCHED":
                disp.setdefault(e.task_id, e.time)
            elif e.event_type == "TASK_DONE":
                done.setdefault(e.task_id, e.time)
        names = {n.node_id: n.node_name for n in order}
        for vid, pids in preds.items():
            if vid not in disp:
                continue
            for pid in pids:
                if pid in done and disp[vid] < done[pid]:
                    raise ValueError(
                        f"sim hang check: '{names.get(vid)}' dispatched at "
                        f"{disp[vid]} before its producer '{names.get(pid)}' "
                        f"finished at {done[pid]} (seed {seed}). The lowered "
                        f"graph does not enforce its own dependencies.")
        if verbose:
            print(f"  sim seed {seed}: {res.total_latency} cycles, "
                  f"{len(res.completed_task_ids)}/{n_real} tasks, deps OK")

    return {"model": root, "seeds": seeds, "descriptors": len(order),
            "real_tasks": n_real, "dep_edges": sum(len(v) for v in preds.values()),
            "scenarios": scenarios}
