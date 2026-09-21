# Fanchen Kong <fanchen.kong@kuleuven.be>

import networkx as nx


class BingoDFGStaticL1Mixin:
    """Static L1 placement: deciding where each buffer lives before the run starts.

    The planning lives here; the packing itself is bingo_l1_packer.

    Mixed into BingoDFG, so `self` is the whole DFG. These methods call each other
    and the other mixins' methods freely; nothing here is meant to stand alone.
    """

    def bingo_plan_static_l1(self, opts=None, output_dir: str = None) -> None:
        """Liveness + static placement for L1 buffers.

        `opts` is a StaticL1Options, or a bool:

          False (default) : report only, no address changes
          True            : place buffers statically, emit constant offsets
          StaticL1Options : the same, with the tuning and bisect knobs

        Report-only by default so the pass can run over every workload and have its numbers
        inspected before an address moves.
        """
        from bingo_liveness import (collect_handle_users, check_handle_identity,
                                    reachability, liveness_report,
                                    extend_users_for_engine_drain)
        from bingo_l1_packer import StaticL1Options
        from bingo_l1_packer import (pack, verify, report, collect_placement_order,
                                     check_placement_order, write_layout_csv,
                                     write_layout_plot)

        opts = opts if isinstance(opts, StaticL1Options) else StaticL1Options(enable=bool(opts))
        want_pack = opts.enable

        # Static allocation is three independent changes: constant addresses, scratchpad slot
        # colouring, and buffer sharing. Only sharing depends on the liveness argument, so the
        # options peel them apart and a failure lands on a layer rather than on the feature.
        no_sp_pack = not opts.pack_scratchpads

        nodes = sorted(self.node_list, key=lambda n: n.node_id)
        handle_users = collect_handle_users(nodes)
        if not handle_users:
            return

        problems = check_handle_identity(handle_users)
        if problems:
            for pb in problems:
                print(f"[static-l1] ERROR {pb}")
            raise ValueError("static L1 allocation: handle identity is not well formed")

        desc = reachability(self, nodes)

        # bingo_liveness RULE 4. Off by default; can only lengthen live ranges.
        if opts.drain_guard:
            handle_users = extend_users_for_engine_drain(handle_users, self)
            print("[static-l1] DRAIN_GUARD: same-core successors counted as users")

        print("[static-l1] " + liveness_report(handle_users, desc, "L1").lstrip())

        # An input to placement, not only a check afterwards. The check still runs: packer
        # and oracle would both have to be wrong to let a bad layout through.
        order_constraints = collect_placement_order(nodes)
        placement, stats = pack(handle_users, desc, "L1",
                                constraints=order_constraints, opts=opts)
        cap = getattr(self, "l1_capacity_bytes", None)
        print("[static-l1] placement if enabled:")
        print(report(stats, cap))

        # Declared on the args class (PLACEMENT_ORDER) so it can be checked rather than
        # remembered: violating one hangs the run instead of failing a check.
        order_problems = check_placement_order(placement, order_constraints)
        if order_constraints:
            print(f"[static-l1] {len(order_constraints)} declared placement-order "
                  f"constraint(s): {'satisfied' if not order_problems else 'VIOLATED'}")
        if order_problems:
            for pb in order_problems:
                print(f"[static-l1] {pb}")
            if want_pack:
                raise ValueError(
                    "static L1 allocation refused: the emitted layout violates a "
                    "placement-order constraint that the packer was given as an input. That "
                    "is a packer bug, not a workload one -- the constraint was honoured "
                    "during placement and must hold.")

        # The checker runs in BOTH modes. A failure in report-only mode means the graph has a
        # missing dependence edge that packing WOULD have turned into silent corruption --
        # worth knowing even while the layout is untouched.
        if opts.debug:
            # Dump every pair the packer decided may share bytes, with the users that
            # justified it. A sharing decision that looks wrong is either a liveness bug or a
            # missing edge, and both are invisible in the emitted header.
            from bingo_liveness import can_share as _cs
            from bingo_l1_packer import _align_up as _al
            byoff = {}
            for hid, (h, off) in placement.items():
                byoff.setdefault((h.chip_id, h.cluster_id), []).append((off, h, handle_users[hid][1]))
            for key in sorted(byoff):
                lst = sorted(byoff[key], key=lambda x: x[0])
                for a in range(len(lst)):
                    oa, ha, ua = lst[a]
                    ea = oa + (_al(ha.size))
                    for b in range(a+1, len(lst)):
                        ob, hb, ub = lst[b]
                        eb = ob + (_al(hb.size))
                        if oa < eb and ob < ea:
                            def _ab(x, y):
                                for m in x:
                                    for n2 in y:
                                        if m is n2 or n2 not in desc.get(m, ()):
                                            return False, (m.node_name, n2.node_name)
                                return True, None
                            f1, w1 = _ab(ua, ub)
                            f2, w2 = _ab(ub, ua)
                            print(f"[static-l1 dbg] cl{key[1]} SHARE {ha.name}[{oa},{ea}) "
                                  f"/ {hb.name}[{ob},{eb}) A->B={f1} B->A={f2} "
                                  f"blockAB={w1} blockBA={w2}")
                            print(f"[static-l1 dbg]    {ha.name} users: "
                                  + ", ".join(sorted(n.node_name for n in ua))[:220])
                            print(f"[static-l1 dbg]    {hb.name} users: "
                                  + ", ".join(sorted(n.node_name for n in ub))[:220])

        issues = verify(placement, handle_users, desc, cap, "L1")
        if issues:
            print(f"[static-l1] CHECKER FOUND {len(issues)} PROBLEM(S):")
            for i in issues[:20]:
                print(f"[static-l1]   {i}")
            if want_pack:
                raise ValueError(
                    "static L1 allocation refused: the placement is not provably safe. "
                    "Pass static_l1=False to fall back to runtime allocation.")
        else:
            print("[static-l1] checker: placement is provably safe")

        # Scratchpad slots. Same oracle, different granularity: every scratchpad is the same
        # size, so this is graph colouring and the generated C multiplies slot by sizeof.
        from bingo_l1_packer import pack_scratchpad_slots, verify_scratchpad_slots
        dev_nodes = [n for n in nodes
                     if n.kernel_name and n.kernel_name.startswith("__snax")]
        sp_slots, sp_users = ({}, {})
        if dev_nodes:
            sp_slots, sp_users = pack_scratchpad_slots(dev_nodes, desc)
            sp_issues = verify_scratchpad_slots(sp_slots, sp_users, desc)
            per_cl = {}
            for n, sl in sp_slots.items():
                per_cl.setdefault(n.assigned_cluster_id, set()).add(sl)
            # the arena is sized by max(slot)+1, so report that -- not the set size, which
            # would hide a sparse numbering bug
            print(f"[static-l1] scratchpads: {len(sp_slots)} nodes -> slots per cluster: "
                  + ", ".join(f"cl{c}={max(v)+1}" for c, v in sorted(per_cl.items())))
            if sp_issues:
                print(f"[static-l1] CHECKER FOUND {len(sp_issues)} SCRATCHPAD PROBLEM(S):")
                for i in sp_issues[:10]:
                    print(f"[static-l1]   {i}")
                if want_pack:
                    raise ValueError("static L1 allocation refused: unsafe scratchpad slots")

        self.static_l1_placement = placement if want_pack else None
        self.static_l1_stats = stats if want_pack else None
        self.static_l1_sp_slots = sp_slots if (want_pack and not no_sp_pack) else None
        # A report that disagrees with the header beside it is worse than none.
        if not want_pack and output_dir is not None:
            import os as _os_rm
            stale = _os_rm.path.join(output_dir, "static_l1_layout.csv")
            if _os_rm.path.exists(stale):
                _os_rm.remove(stale)
                print(f"[static-l1] removed stale layout report {stale} "
                      f"(this build does not place buffers)")

        if want_pack:
            # The whole mode, not just that packing happened: "packing happened" does not
            # say WHICH packing, which is how two bisect arms became silent duplicates.
            mode = [f"order={opts.order}"]
            if not opts.share:
                mode.append("no-share")
            if no_sp_pack:
                mode.append("no-scratchpad-slots")
            if opts.guard_bytes:
                mode.append(f"guard={opts.guard_bytes}B")
            if opts.alignment != 256:
                mode.append(f"align={opts.alignment}")
            if opts.drain_guard:
                mode.append("drain-guard")
            if opts.pin_first:
                mode.append(f"pin_first={','.join(opts.pin_first)}")
            if opts.pin_last:
                mode.append(f"pin_last={','.join(opts.pin_last)}")
            print("[static-l1] ENABLED: emitting constant offsets [" + " ".join(mode) + "]")

            # Only when the compiler owns placement: in report-only mode the addresses
            # would be ones nothing uses, which is worse than no file.
            if output_dir is not None:
                import os as _os_csv
                rank = {n: i for i, n in enumerate(nx.topological_sort(self))}
                csv_path = _os_csv.path.join(output_dir, "static_l1_layout.csv")
                n_rows = write_layout_csv(csv_path, placement, handle_users, desc, rank, opts)
                print(f"[static-l1] layout report: {csv_path} ({n_rows} buffers)")
                png = write_layout_plot(
                    _os_csv.path.join(output_dir, "static_l1_layout.png"),
                    placement, handle_users, desc, rank, opts, capacity=cap)
                if png:
                    print(f"[static-l1] layout plot:   {png}")
