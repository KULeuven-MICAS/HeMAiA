# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Static placement of L1 buffers, and the checker that makes it safe to switch on.
#
# Placement is first-fit over the interference graph from bingo_liveness: each buffer goes at
# the lowest offset where it does not overlap anything it interferes with. The ORDER buffers
# are considered in is a correctness property, not just a quality one -- see pack().
#
# The checker ships alongside because packing changes what a missing dependence edge costs.
# Without it such a bug corrupts one buffer and that buffer's checks fail. With it the packer
# concludes a buffer died early, puts another on top, and the corruption surfaces in something
# unrelated. So `verify` re-derives safety from the placement alone, asking the liveness oracle
# again rather than trusting the packer's bookkeeping: a packer bug and a checker bug would
# both have to be wrong to get through.
#
# MiniMalloc solves the same formulation exactly and consumes the same (size, interference)
# data. Worth adopting when `verify` reports a real gap between the achieved peak and the
# interference lower bound.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from bingo_liveness import build_interference, can_share
from bingo_mem_handle import BingoMemAlloc, BingoMemAllocView

# Alignment reproduces the PHASE the runtime allocator always delivered, not merely the
# alignment it promised. bingoHeapMalloc rounds each fragment to ALIGN_UP(size + 128, 256) and
# returns frag + 128: it promises 128, but every buffer it ever returned shares one offset
# mod 256, and a TCDM bank row is 256 B. Packing at 128 respects the contract and still puts
# buffers on a phase no streamer has seen.
L1_ALIGNMENT = 256


@dataclass(frozen=True)
class StaticL1Options:
    """How to place L1 buffers.

    Passed in as an argument and never read from the environment: this changes generated code,
    and the SW build runs inside a container, so an ambient switch can be set where a build is
    launched and absent where the compiler runs -- silently.

    `enable` defaults to False; every other field only means anything once it is set.
    """

    enable: bool = False

    # --- the shipping defaults -------------------------------------------------------
    share: bool = True            # let disjoint live ranges reuse each other's bytes
    pack_scratchpads: bool = True  # colour per-node scratchpads into slots
    order: str = "name"           # "name" = allocation order. See the order comment in pack().
    alignment: int = 256          # the phase bingoHeapMalloc always delivered
    guard_bytes: int = 0          # extra slack after every buffer; a diagnostic, not a fix

    # --- diagnostics: off unless a bisect wants them ---------------------------------
    drain_guard: bool = False     # bingo_liveness RULE 4
    debug: bool = False           # dump every sharing decision and who justified it
    pin_first: Tuple[str, ...] = ()   # name substrings to place at the front
    pin_last: Tuple[str, ...] = ()    # ...and at the back

    def __post_init__(self):
        if self.order not in ("name", "size", "reverse"):
            raise ValueError(f"order must be name, size or reverse; got {self.order!r}")
        if self.alignment <= 0 or self.alignment % 64:
            raise ValueError(f"alignment must be a positive multiple of 64; got {self.alignment}")


#: What a workload gets when it passes `static_l1=True`.
STATIC_L1_ON = StaticL1Options(enable=True)

# The old allocator also always left a gap: consecutive buffers were separated by the next
# fragment's 128-byte header, so a kernel writing one row past its buffer corrupted nothing
# anyone checks. StaticL1Options.guard_bytes restores that slack, to answer "does some kernel
# overrun?" in one run. A diagnostic, not a fix -- if it turns a failing arm green, the
# overrunning kernel still has to be found.


def _align_up(x: int, a: int = L1_ALIGNMENT) -> int:
    return (x + a - 1) // a * a


def _footprint(size: int, alignment: int = L1_ALIGNMENT, guard: int = 0) -> int:
    """Bytes a buffer occupies in the layout: its size, the diagnostic guard, then aligned."""
    return _align_up(size + guard, alignment)



def _order_with_constraints(preference, preds, items):
    """`preference` reordered so every declared predecessor comes first.

    Stable with respect to `preference`: among the buffers whose predecessors are all placed,
    the one the heuristic wanted first is taken first. A workload that declares no constraints
    therefore gets byte-identical output to the unconstrained packer.
    """
    want = {i: n for n, i in enumerate(preference)}
    subset = set(preference)
    remaining = {i: {p for p in preds[i] if p in subset} for i in preference}
    out, ready = [], sorted((i for i in preference if not remaining[i]), key=lambda i: want[i])
    while ready:
        i = ready.pop(0)
        out.append(i)
        newly = []
        for j, ps in remaining.items():
            if i in ps:
                ps.discard(i)
                if not ps and j not in out and j not in ready:
                    newly.append(j)
        ready = sorted(ready + newly, key=lambda k: want[k])
    if len(out) != len(preference):
        stuck = [items[i][0].name for i in preference if i not in out]
        raise ValueError(
            "static L1 allocation: the declared placement-order constraints are circular; "
            f"cannot order {stuck}")
    return out


def pack(handle_users, desc, level: str = "L1", constraints=None,
         opts: Optional["StaticL1Options"] = None):
    """Assign every buffer an offset within its own (chip, cluster) address space.

    Returns (placement, stats) where placement maps id(handle) -> (handle, offset) and stats
    maps (chip, cluster) -> {'peak', 'sum', 'lower_bound', 'count'}.

    `no_share` declares every pair of buffers on a cluster to interfere, so the layout becomes
    a plain concatenation at constant offsets. That is the bisect arm: it keeps the static
    addressing and drops the liveness argument, which is the only part of this file that can
    be wrong about the hardware rather than about arithmetic.
    """
    opts = opts or StaticL1Options()
    alignment, guard = opts.alignment, opts.guard_bytes
    items, interference = build_interference(handle_users, desc, level)
    if not opts.share:
        for i in range(len(items)):
            hi = items[i][0]
            for j in range(len(items)):
                if i != j and (items[j][0].chip_id, items[j][0].cluster_id) == (hi.chip_id, hi.cluster_id):
                    interference[i].add(j)

    # DECLARED PLACEMENT ORDER AS A PLACEMENT RULE, NOT ONLY A CHECK.
    #
    # `preds[i]` is the set of buffers that must sit at a LOWER address than buffer i. The
    # packer honours this in two places: the placement order is a topological order of this
    # relation, so a predecessor always has an address by the time its successor is placed;
    # and each buffer's search starts above its predecessors rather than at 0, because
    # first-fit would otherwise happily drop a successor into a gap underneath one.
    idx_of = {id(h): i for i, (h, _u) in enumerate(items)}
    preds: Dict[int, set] = {i: set() for i in range(len(items))}
    for a, b, _why in (constraints or []):
        ia, ib = idx_of.get(id(a)), idx_of.get(id(b))
        if ia is None or ib is None or ia == ib:
            continue
        ha, hb = items[ia][0], items[ib][0]
        if (ha.chip_id, ha.cluster_id) != (hb.chip_id, hb.cluster_id):
            continue          # different address spaces cannot be ordered against each other
        preds[ib].add(ia)

    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx, (h, _u) in enumerate(items):
        groups.setdefault((h.chip_id, h.cluster_id), []).append(idx)

    placement: Dict[int, Tuple[object, int]] = {}
    stats: Dict[Tuple[int, int], Dict[str, int]] = {}

    for key in sorted(groups):
        idxs = groups[key]
        # BINGO_L1_PIN=<substr>[,<substr>...] places the named buffers FIRST, in the order
        # given, and lets everything else follow the normal rule. It is the bisect instrument
        # for "which buffer's address is load-bearing": pin one candidate to the front of an
        # otherwise freely packed layout and see whether the workload comes back.
        pins = list(opts.pin_first)
        # ...and BINGO_L1_PIN_LAST=<substr> sends them to the END instead. Together the two
        # knobs move one buffer through an otherwise unchanged layout, which is how "this
        # buffer's neighbour is what matters" gets tested in one run rather than fifteen.
        pins_last = list(opts.pin_last)
        if pins_last:
            tail = [i for i in idxs if any(t in items[i][0].name for t in pins_last)]
            head = [i for i in idxs if i not in set(tail)]
            order = (sorted(head, key=lambda i: (-items[i][0].size, items[i][0].name))
                     + sorted(tail, key=lambda i: items[i][0].name))
        elif pins:
            def _pin_rank(i):
                nm = items[i][0].name
                for r, t in enumerate(pins):
                    if t in nm:
                        return (r, nm)
                return (len(pins), "")
            pinned = [i for i in idxs if _pin_rank(i)[0] < len(pins)]
            pinned.sort(key=_pin_rank)
            rest = [i for i in idxs if i not in set(pinned)]
            order = pinned + sorted(rest, key=lambda i: (-items[i][0].size, items[i][0].name))
        elif opts.order == "name":
            # ALLOCATION ORDER, THE DEFAULT, AND DELIBERATELY NOT THE BEST-PACKING ONE.
            #
            # The emitter allocates handles alphabetically, so this reproduces the order the
            # runtime allocator used. Largest-first packs the small-buffer tail into the gaps
            # the big ones leave and is worth a few percent, but a kernel that writes past the
            # end of one buffer, or computes an address as `other_buffer - my_buffer`, depends
            # on the RELATIVE ORDER of two allocations and nothing declares that. Allocation
            # order preserves every "X before Y" relationship by construction while still
            # reclaiming what the liveness analysis finds. order="size" for experiments.
            order = sorted(idxs, key=lambda i: items[i][0].name)
        elif opts.order == "reverse":
            # Every buffer moves. A bisect instrument: separates "any reordering breaks
            # this" from "this particular layout does".
            order = sorted(idxs, key=lambda i: items[i][0].name, reverse=True)
        else:
            # largest first: the big streams claim low offsets, the scratchpad tail fills gaps
            order = sorted(idxs, key=lambda i: (-items[i][0].size, items[i][0].name))
        # The heuristic above chose a preference order; the constraints now override it where
        # they disagree. Kahn's algorithm, taking the heuristic's order as the tie-break, so a
        # workload with no constraints gets exactly the layout it got before.
        order = _order_with_constraints(order, preds, items)

        placed: List[Tuple[int, int, int]] = []   # (idx, start, end)
        peak = 0
        for i in order:
            size = _footprint(items[i][0].size, alignment, guard)
            blockers = [(s, e) for (j, s, e) in placed if j in interference[i]]
            blockers.sort()
            # start above every declared predecessor: one alignment unit past its base is the
            # lowest address that still satisfies `offset[pred] < offset[i]`
            off = 0
            for pidx in preds[i]:
                ppl = placement.get(id(items[pidx][0]))
                if ppl is not None:
                    off = max(off, ppl[1] + alignment)
            for s, e in blockers:
                if off + size <= s:
                    break          # fits in the gap below this blocker
                off = max(off, e)
            placed.append((i, off, off + size))
            placement[id(items[i][0])] = (items[i][0], off)
            peak = max(peak, off + size)
        total = sum(_footprint(items[i][0].size, alignment, guard) for i in idxs)
        # A LOWER BOUND must be the weight of a set that is MUTUALLY interfering -- a clique.
        # Summing one buffer's neighbours is not that (they need not interfere with each
        # other) and overstates it, which had this reporting "bound == sum" on a layout the
        # packer then beat by 4.5%. Greedy max-weight clique instead: exact enough to tell
        # whether a better solver (MiniMalloc) has anything left to find.
        lb = 0
        for seed in sorted(idxs, key=lambda i: -items[i][0].size):
            clique = [seed]
            for c in sorted(idxs, key=lambda i: -items[i][0].size):
                if c != seed and all(c in interference[m] for m in clique):
                    clique.append(c)
            lb = max(lb, sum(_footprint(items[i][0].size, alignment, guard) for i in clique))
        stats[key] = {"peak": peak, "sum": total,
                      "lower_bound": min(lb, total), "count": len(idxs)}
    return placement, stats


def verify(placement, handle_users, desc, capacity: int = None, level: str = "L1",
           opts: Optional["StaticL1Options"] = None) -> List[str]:
    """Re-derive safety from the placement. Returns a list of problems; empty means sound.

    Deliberately does NOT reuse the packer's interference bookkeeping -- it re-asks the
    liveness oracle, so that a packer bug alone cannot produce a clean report.
    """
    opts = opts or StaticL1Options()
    _al, _gu = opts.alignment, opts.guard_bytes
    problems: List[str] = []
    items = [(h, u) for (h, u) in handle_users.values() if h.mem_level == level]

    # 1. every buffer placed exactly once
    for h, _u in items:
        if id(h) not in placement:
            problems.append(f"buffer {h.name} (cluster {h.cluster_id}) has no offset")

    by_cluster: Dict[Tuple[int, int], List] = {}
    for h, u in items:
        if id(h) in placement:
            by_cluster.setdefault((h.chip_id, h.cluster_id), []).append((h, u))

    for key, group in sorted(by_cluster.items()):
        for a in range(len(group)):
            ha, ua = group[a]
            oa = placement[id(ha)][1]
            ea = oa + _footprint(ha.size, _al, _gu)

            # 2. alignment and capacity
            if oa % _al:
                problems.append(f"{ha.name}: offset {oa} is not {_al}-byte aligned")
            if capacity is not None and ea > capacity:
                problems.append(
                    f"{ha.name}: ends at {ea:,} B, past the {capacity:,} B capacity "
                    f"of chip {key[0]:#04x} cluster {key[1]}")

            # 3. THE ONE THAT MATTERS: anything sharing bytes must be provably disjoint in time
            for b in range(a + 1, len(group)):
                hb, ub = group[b]
                ob = placement[id(hb)][1]
                eb = ob + _footprint(hb.size, _al, _gu)
                overlaps = oa < eb and ob < ea
                if overlaps and not can_share(ua, ub, desc):
                    problems.append(
                        f"UNSAFE OVERLAP on chip {key[0]:#04x} cluster {key[1]}: "
                        f"{ha.name} [{oa:,},{ea:,}) and {hb.name} [{ob:,},{eb:,}) share bytes "
                        f"but are not ordered in the task graph -- a missing dependence edge, "
                        f"or a liveness bug")
    return problems


def report(stats, capacity: int = None) -> str:
    lines = []
    for key in sorted(stats):
        s = stats[key]
        saved = s["sum"] - s["peak"]
        pct = (100.0 * saved / s["sum"]) if s["sum"] else 0.0
        cap = f" of {capacity:,} B" if capacity else ""
        lines.append(
            f"    chip {key[0]:#04x} cluster {key[1]}: {s['count']:3d} buffers  "
            f"sum {s['sum']:,} B -> peak {s['peak']:,} B{cap}  "
            f"(saved {saved:,} B, {pct:.1f}%; bound {s['lower_bound']:,} B)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# SCRATCHPAD SLOTS
#
# Every per-node scratchpad is the SAME size (bingo_kernel_scratchpad_t, aligned), which the
# mini-compiler only knows as a C sizeof expression. So the packer assigns integer SLOTS and
# lets the generated C multiply: `arena + slot * ALIGN_UP(sizeof(...), 64)`. Equal sizes also
# turn placement into graph colouring, which is exact for this shape rather than heuristic.
#
# WHAT A SCRATCHPAD'S USERS ARE. Its own node, plus any node that READS it. Two kernels do
# that today:
#   * BINGO_SW_GUARD_CHECK reads `gating_sp->return_value` through the gating_sp_addr the
#     compiler bakes in from node._gating_node's scratchpad var;
#   * __host_bingo_kernel_cerf_gating takes a predecessor's scratchpad as args[1].
# Both are edges in the graph already, but the LIFETIME they create is not, so they are added
# explicitly here. In FlashAttention every gating_sp_addr is 0 and there is no cerf gating, so
# every scratchpad is genuinely single-node -- but that is a property of the workload, not of
# the compiler, and a packer that assumed it would corrupt the first workload that used one.
#
# SIZE. A scratchpad is exactly 64 B (heterogeneous_runtime.h:72-84: return_value,
# num_return_values, then profiling words that are no-ops unless BINGO_SCRATCHPAD_PROFILING,
# which nothing defines). So on FlashAttention -- 202 scratchpads -- this reclaims ~13 KB, not
# the ~100 KB an earlier estimate assumed from a 512 B scratchpad. The mechanism matters far
# more on node-heavy workloads: a 92-configuration sweep spent ~185 KiB of 504 KiB here.
#
# RESIDUAL RISK, not yet defended against. There is NO FENCE between a kernel's last store and
# its done-write (bingo.h:352 is a bare `csrw 0x5ff`; the host path at bingo_api.c:726 is a
# bare writew). A slot's next occupant could in principle be clobbered by a store from the
# node that just retired. For SCRATCHPADS the exposure is small -- only the owning core writes
# them, with ordinary in-order stores to its own TCDM -- but it is not zero, and for a buffer
# whose last write is an outstanding DMA it would be real. If a packed workload ever shows
# non-deterministic corruption in an unrelated buffer, put a fence before the done-write and
# re-test before looking anywhere else.
#
# ARGUMENT STRUCTS ARE DELIBERATELY NOT PACKED. They look like the same kind of object and are
# not: the host populates every arg struct at init, BEFORE the scheduler starts, so an arg's
# live range is [program start, its node fires] rather than [its node, its node]. Every arg
# struct is therefore live simultaneously at init and none of them may share bytes. Modelling
# them as single-node -- which an earlier draft of the plan proposed -- would let the packer
# overlap a late node's arguments with a buffer that dies during start-up, and that node would
# then fire with whatever overwrote them.

def scratchpad_users(nodes):
    """node -> the set of nodes whose execution may touch that node's scratchpad."""
    users = {n: {n} for n in nodes}
    for n in nodes:
        g = getattr(n, "_gating_node", None)
        if g is not None and g in users:
            users[g].add(n)          # n reads g's scratchpad through gating_sp_addr
        p = getattr(n, "_pred_source_node", None)
        if p is not None and p in users:
            users[p].add(n)          # cerf gating reads the predecessor's scratchpad
    return users


def pack_scratchpad_slots(nodes, desc):
    """node -> slot index, DENSE WITHIN EACH (chip, cluster) ARENA.

    Two nodes share a slot only when can_share says every user of one precedes every user of
    the other -- the same oracle the buffer packer uses, so there is one definition of safety.

    Colouring runs per cluster rather than globally. A global colouring is equally SAFE but
    produces sparse indices -- cluster 0 might be handed slots {3, 7, 11} -- while each
    cluster's arena is sized for its own slot COUNT. `arena + 7 * sp_align` into a 5-slot
    arena walks off the end into whatever the allocator placed next. Numbering per cluster
    keeps index and size in the same units by construction.
    """
    users = scratchpad_users(nodes)
    by_arena = {}
    for n in nodes:
        by_arena.setdefault((n.assigned_chiplet_id, n.assigned_cluster_id), []).append(n)

    slots = {}
    for _key, group in sorted(by_arena.items()):
        order = sorted(group, key=lambda n: n.node_id)
        conflict = {n: set() for n in order}
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                if not can_share(users[a], users[b], desc):
                    conflict[a].add(b)
                    conflict[b].add(a)
        for n in sorted(order, key=lambda x: (-len(conflict[x]), x.node_id)):
            taken = {slots[m] for m in conflict[n] if m in slots}
            sl = 0
            while sl in taken:
                sl += 1
            slots[n] = sl
    return slots, users


def verify_scratchpad_slots(slots, users, desc):
    """Two nodes sharing a slot must be provably ordered. Re-asks the oracle."""
    problems = []
    by_slot = {}
    for n, s in slots.items():
        # slots are per-arena, so two nodes only collide if the arena matches too
        by_slot.setdefault((n.assigned_chiplet_id, n.assigned_cluster_id, s), []).append(n)
    for s, group in sorted(by_slot.items()):
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if not can_share(users[a], users[b], desc):
                    problems.append(
                        f"UNSAFE SCRATCHPAD SLOT {s}: nodes {a.node_name} and {b.node_name} "
                        f"share it but are not ordered in the task graph")
    return problems


# Post-emission audit of the generated C. The packer reasons about buffers and the slot
# colouring about scratchpads; neither can see the third tenant of the same arena, the device
# argument structs bump-allocated from `<arena>_off`. With slot packing on, the slot block
# sits at the front, so the arg bump must start above it -- otherwise args_dev_*[0] and slot 0
# are the same bytes, every static check still passes, and a task reads its arguments out of
# another node's scratchpad at run time. Checked on the artefact, because that is what ships.

_SP_ALIGN_RE = r"ALIGN_UP\(sizeof\(bingo_kernel_scratchpad_t\), 64\)"


def check_emitted_arena_text(text: str) -> List[str]:
    """Problems with the arena layout in an emitted offload header. Empty means safe."""
    import re

    problems: List[str] = []
    for m in re.finditer(r"uint64_t (__bingo_l[13]_arena_\w+) = bingo_l[13]_alloc\(", text):
        base = m.group(1)
        # how many slots does the arena RESERVE, and where does the arg bump start?
        size_m = re.search(re.escape(base) + r" = bingo_l[13]_alloc\((.*?)\);", text, re.S)
        off_m = re.search(r"uint64_t " + re.escape(base) + r"_off = (.*?);", text)
        if off_m is None:
            continue
        reserved = re.search(r"(\d+) \* " + _SP_ALIGN_RE, size_m.group(1)) if size_m else None
        off_slots = re.match(r"(\d+) \* " + _SP_ALIGN_RE, off_m.group(1).strip())
        off_slots = int(off_slots.group(1)) if off_slots else 0

        used = [int(s) for s in re.findall(
            re.escape(base) + r" \+ (\d+) \* " + _SP_ALIGN_RE, text)]
        if not used:
            continue                      # unpacked: scratchpads bump like everything else
        top = max(used) + 1
        if off_slots < top:
            problems.append(
                f"{base}: scratchpad slots occupy [0, {top}) x sizeof(scratchpad) but the "
                f"argument bump pointer starts at slot {off_slots} -- args would alias "
                f"slots {off_slots}..{top - 1}")
        if reserved is not None and int(reserved.group(1)) < top:
            problems.append(
                f"{base}: arena reserves {reserved.group(1)} scratchpad slots but slot "
                f"{top - 1} is used")
    return problems


# ---------------------------------------------------------------------------------------
# DECLARED PLACEMENT ORDER
#
# A kernel that computes an address in one buffer as an offset from another makes the two
# buffers' RELATIVE placement part of its ABI. Declared on the args class (PLACEMENT_ORDER),
# honoured by pack() and re-checked against the emitted layout: a constraint whose violation
# shows up as a hang rather than a failed check must be enforced by a tool, not remembered.

def collect_placement_order(nodes):
    """[(earlier_handle, later_handle, why)] declared by the kernels on these nodes."""
    out = []
    for n in nodes:
        args = getattr(n, "kernel_args", None)
        if args is None:
            continue
        for earlier, later in getattr(type(args), "PLACEMENT_ORDER", ()):
            a, b = getattr(args, earlier, None), getattr(args, later, None)
            a = a.base if isinstance(a, BingoMemAllocView) else a
            b = b.base if isinstance(b, BingoMemAllocView) else b
            if isinstance(a, BingoMemAlloc) and isinstance(b, BingoMemAlloc) and a is not b:
                out.append((a, b, f"{type(args).__name__}: {earlier} must precede {later}"))
    return out


def check_placement_order(placement, constraints):
    """Problems with a placement against the declared constraints. Empty means satisfied."""
    problems = []
    for a, b, why in constraints:
        pa, pb = placement.get(id(a)), placement.get(id(b))
        if pa is None or pb is None:
            continue
        if (a.chip_id, a.cluster_id) != (b.chip_id, b.cluster_id):
            continue
        if pa[1] >= pb[1]:
            problems.append(
                f"PLACEMENT ORDER VIOLATED on chip {a.chip_id:#04x} cluster {a.cluster_id}: "
                f"{a.name} is at +{pa[1]:,} but must precede {b.name} at +{pb[1]:,} -- {why}")
    return sorted(set(problems))


# The layout report. One row per buffer: its bytes, the nodes that touch it, the buffers it
# shares those bytes with, and the node whose first touch of the next occupant ends its life.
#
# `overwritten_by` is the column to read when data goes wrong: it names the node allowed to
# clobber this buffer, so a value that changes before that node ran is a liveness bug, and one
# that changes after it is working as designed.

def write_layout_csv(path, placement, handle_users, desc, topo_rank, opts, level="L1"):
    """One row per placed buffer. Returns the number of rows written."""
    import csv

    items = [(h, u) for (h, u) in handle_users.values()
             if h.mem_level == level and id(h) in placement]
    if not items:
        return 0

    def rank(n):
        return topo_rank.get(n, 1 << 30)

    def ordered(nodes):
        return sorted(nodes, key=lambda n: (rank(n), getattr(n, "node_name", "")))

    def nm(n):
        return getattr(n, "node_name", None) or f"node{getattr(n, 'node_id', '?')}"

    al, gu = opts.alignment, opts.guard_bytes
    rows = []
    for h, users in items:
        off = placement[id(h)][1]
        end = off + _footprint(h.size, al, gu)
        us = ordered(users)

        # who else lives in these bytes, and which of them takes over next
        shares, successor = [], None
        for h2, u2 in items:
            if h2 is h or (h2.chip_id, h2.cluster_id) != (h.chip_id, h.cluster_id):
                continue
            o2 = placement[id(h2)][1]
            if off < o2 + _footprint(h2.size, al, gu) and o2 < end:
                shares.append(h2.name)
                # the next occupant is the one whose users all come AFTER this buffer's
                if u2 and us and rank(ordered(u2)[0]) > rank(us[-1]):
                    first2 = ordered(u2)[0]
                    if successor is None or rank(first2) < rank(successor[1]):
                        successor = (h2.name, first2)

        rows.append({
            "chip": f"0x{h.chip_id:02x}",
            "cluster": h.cluster_id,
            "buffer": h.name,
            "offset": off,
            "end": end,
            "size_b": h.size,
            "footprint_b": _footprint(h.size, al, gu),
            "n_users": len(us),
            "first_use": nm(us[0]) if us else "",
            "last_use": nm(us[-1]) if us else "",
            "overwritten_by": nm(successor[1]) if successor else "",
            "reused_by_buffer": successor[0] if successor else "",
            "shares_bytes_with": " ".join(sorted(shares)),
            "users": " ".join(nm(n) for n in us),
        })

    rows.sort(key=lambda r: (r["chip"], r["cluster"], r["offset"], r["buffer"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_layout_plot(path, placement, handle_users, desc, topo_rank, opts,
                      level="L1", capacity=None):
    """Address against time: one rectangle per buffer, one panel per cluster.

    x is the buffer's live range, y the bytes it occupies. Two rectangles at the same height
    with no horizontal overlap are two buffers sharing those bytes; two overlapping in both
    axes would be the bug the checker prevents.

    Returns the path written, or None if matplotlib is unavailable -- a debugging aid must
    never be the reason a build fails.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:                                  # pragma: no cover
        print(f"[static-l1] layout plot skipped ({exc})")
        return None

    items = [(h, u) for (h, u) in handle_users.values()
             if h.mem_level == level and id(h) in placement]
    if not items:
        return None

    al, gu = opts.alignment, opts.guard_bytes
    groups = {}
    for h, u in items:
        groups.setdefault((h.chip_id, h.cluster_id), []).append((h, u))

    keys = sorted(groups)
    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 9), squeeze=False)
    span = max(topo_rank.values()) + 1 if topo_rank else 1

    for ax, key in zip(axes[0], keys):
        rows = sorted(groups[key], key=lambda hu: placement[id(hu[0])][1])
        peak = max(placement[id(h)][1] + _footprint(h.size, al, gu) for h, _ in rows)

        # buffers that actually share bytes get a hatch: that is what the plot is for
        boxes = {}
        for h, users in rows:
            o = placement[id(h)][1]
            boxes[id(h)] = (o, o + _footprint(h.size, al, gu))
        shared = set()
        for i, (h, _u) in enumerate(rows):
            for h2, _u2 in rows[i + 1:]:
                (a0, a1), (b0, b1) = boxes[id(h)], boxes[id(h2)]
                if a0 < b1 and b0 < a1:
                    shared.add(id(h)); shared.add(id(h2))

        thin_labels = []          # (y, text) for boxes too short to hold their own name
        for n, (h, users) in enumerate(rows):
            off = placement[id(h)][1]
            hgt = _footprint(h.size, al, gu)
            rk = sorted(topo_rank.get(x, 0) for x in users) or [0]
            x0, x1 = rk[0], max(rk[-1], rk[0] + span * 0.01)
            ax.add_patch(Rectangle((x0, off), x1 - x0, hgt,
                                   facecolor=f"C{n % 10}", edgecolor="black",
                                   linewidth=0.6, alpha=0.75,
                                   hatch="///" if id(h) in shared else None))
            label = h.name.replace("fa_", "")
            if hgt > peak * 0.025:
                ax.text(x0 + (x1 - x0) / 2, off + hgt / 2, label,
                        ha="center", va="center", fontsize=7)
            else:
                thin_labels.append((off + hgt / 2, x1, label))

        # Thin buffers cannot hold their names and land close together, so stack the labels
        # on a ladder to the right with a leader line back to each box.
        thin_labels.sort()
        ladder_x = span * 1.02
        step = peak * 0.035
        y_cursor = None
        for y, x_end, label in thin_labels:
            y_at = y if y_cursor is None else max(y, y_cursor + step)
            y_cursor = y_at
            ax.plot([x_end, ladder_x], [y, y_at], color="grey", linewidth=0.5, zorder=1)
            ax.text(ladder_x + span * 0.01, y_at, label, ha="left", va="center", fontsize=6)

        if capacity:
            ax.axhline(capacity, color="red", linestyle="--", linewidth=1)
            ax.text(0, capacity, f" L1 capacity {capacity:,} B", color="red",
                    fontsize=7, va="bottom")
        ax.axhline(peak, color="black", linestyle=":", linewidth=1)
        ax.text(span, peak, f"peak {peak:,} B ", color="black", fontsize=7,
                va="bottom", ha="right")
        ax.set_xlim(0, span * 1.30)
        ax.set_ylim(0, max(peak, capacity or 0) * 1.06)
        ax.set_title(f"chip {key[0]:#04x} cluster {key[1]}", fontsize=10)
        ax.set_xlabel("task graph order (topological rank)")
        ax.set_ylabel("L1 offset (bytes)")
        ax.grid(alpha=0.25, linewidth=0.5)

    mode = f"order={opts.order} align={opts.alignment}"
    if not opts.share:
        mode += " no-share"
    if opts.guard_bytes:
        mode += f" guard={opts.guard_bytes}B"
    fig.suptitle(f"Static L1 layout and liveness  [{mode}]\n"
                 f"x = live range (first to last node that touches the buffer),  "
                 f"y = the bytes it occupies.  Hatched = shares its bytes with another buffer "
                 f"(no horizontal overlap = provably safe)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
