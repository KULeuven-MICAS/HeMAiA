# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""MBuild's task helpers the phases share: a weight chunk's load and release, one GEMM, the HBM casts of X and the
router's weights, the residual add, and the args phases."""

from bingo_kernel_args import (INT4_B_LAYOUT_PARITY, HostBingoKernelHbmBcastArgs, HostBingoKernelHbmUcastArgs,
                               PhaseArgsLoadArgs, SnaxBingoKernelGemmFullArgs, SnaxBingoKernelIdma1dCopyArgs,
                               SnaxBingoKernelSimdAddF16Args)
from bingo_mem_handle import BingoMemSymbol

from .common import B_PARITY_OFF, SCALE_BUFS, SHAPE3, Scale


class Tasks:

    def args_phase(self, name):
        """An args phase boundary (params args_phases): each cluster's DM core gets the task that loads the next phase's
        args, after every task of the phase on the chip (a task may write the other cluster's L1, so a chip's clusters
        leave a phase together): after each of the phase's tasks on the chip that no other one of them follows. The
        tasks made from here on are of the next phase."""
        lo = self.phase_starts[-1] if self.phase_starts else 0
        mine = {}
        for n in self.dfg.node_list:
            if n.node_id >= lo and self._phase_task(n):
                mine.setdefault(n.assigned_chiplet_id, set()).add(n)
        ph = len(self.phase_starts) + 1
        for k in self.chips:
            ts = mine.get(k, set())
            deps = sorted((t for t in ts if not any(v in ts for v in self.dfg.successors(t))), key=lambda t: t.node_id)
            for c in range(self.NC):
                ld = self.node(k, c, self.roles["dm"], f"args_p{ph}", "__snax_bingo_kernel_idma_1d_copy",
                               PhaseArgsLoadArgs(k, c, ph), deps)
                ld.args_resident = True
                self.loaders[(ph, k, c)] = ld
        self.phase_starts.append(max(n.node_id for n in self.dfg.node_list) + 1)
        self.phase_names.append(name)

    @staticmethod
    def _phase_task(n):
        """A device task whose args sit in its phase's region (not the compiler's resident ones)."""
        return (n.kernel_name or "").startswith("__snax") and not getattr(n, "args_resident", False)

    def _assign_phases(self):
        """params args_phases: every device task its args phase; each task of a phase that no task of the phase on its
        cluster precedes waits for its cluster's load."""
        phase_of = {}
        for n in self.dfg.node_list:
            if self._phase_task(n):
                phase_of[n] = n.args_phase = sum(1 for s in self.phase_starts if n.node_id >= s)
        cl = lambda t: (t.assigned_chiplet_id, t.assigned_cluster_id)    # noqa: E731
        for n, ph in phase_of.items():
            if not ph:
                continue
            ld = self.loaders[(ph, *cl(n))]
            if any(phase_of.get(u) == ph and cl(u) == cl(n) for u in self.dfg.predecessors(n)):
                continue
            if not self.dfg.has_edge(ld, n):
                self.edge(ld, n)
        counts = {}
        for n in self.dfg.node_list:
            if self._phase_task(n) and n.assigned_chiplet_id == self.ROOT and n.assigned_cluster_id == 0:
                counts[n.args_phase] = counts.get(n.args_phase, 0) + 1
        print(f"[{self.TAG}] args phases: " + "; ".join(f"{q} {nm}" for q, nm in enumerate(self.phase_names))
              + f"; root cluster 0 tasks a phase {dict(sorted(counts.items(), key=str))}")

    def _resid_add(self, cl, jl, y, after):
        """The residual's slice jl of cluster cl's outputs added into the chip's partial y (after the shared down
        projection's dequantisation), a piece at a time through the cluster's hres buffer. Returns the last add."""
        k, c = self.chip(cl), self.local(cl)
        dm, simd = self.roles["dm"], self.roles["simd"]
        pc, hb = self.resid["piece"], self.B[cl]["hres"]
        prev = after
        for o in range(0, self.SL, pc):
            war = [self._hres_reader[cl]] if self._hres_reader.get(cl) is not None else []
            src = self.resid["src"]
            ld = self.node(k, c, dm, f"res_ld_j{jl}o{o}", "__snax_bingo_kernel_idma_1d_copy",
                           SnaxBingoKernelIdma1dCopyArgs(BingoMemSymbol(src.symbol_name,
                                                                        offset=src.offset + self.y_off(c, jl) + o,
                                                                        chip_id=getattr(src, "chip_id", None)),
                                                         hb, pc), war)
            prev = self.node(k, c, simd, f"res_add_j{jl}o{o}", "__snax_bingo_kernel_simd_stream_elementwise",
                             SnaxBingoKernelSimdAddF16Args(y.view(o), hb, y.view(o), pc // 64, 32), [ld, prev])
            self._hres_reader[cl] = prev
        return prev

    def _load(self, cl, j, deps=(), tag=""):
        """Weight chunk j of cluster cl from its ring into the next slab (tag: in the names of a chunk loaded again),
        then its scale tile copied from the slab into the next of the cluster's scale buffers (DM core): the
        dequantisation reads the copy, so the slab is free once the chunk's GEMMs are done. Returns (load, slab index,
        slab, its scale tile)."""
        k, c = self.chip(cl), self.local(cl)
        ls = self.ls[cl]
        off, nb, soff = self.chunk_off[cl][j]
        si, slab, war = ls.take(nb)
        ld, _ = ls.rings.load_chunk(
            lambda n, core, kern, args, after: self.node(k, c, core, n, kern, args, after), c,
            f"Ld_w{j}{tag}", src=self.img[cl].address + off, nbytes=nb, slab=slab, war=war,
            deps=deps, dm=self.roles["dm"])
        i = self.sc_next[cl]
        self.sc_next[cl] = (i + 1) % SCALE_BUFS
        buf, rd = self.B[cl]["sc"][i], self.sc_reader[cl][i]
        cp = self.node(k, c, self.roles["dm"], f"sc_w{j}{tag}", "__snax_bingo_kernel_idma_1d_copy",
                       SnaxBingoKernelIdma1dCopyArgs(slab.view(soff), buf, nb - soff),
                       [ld] + ([rd] if rd is not None else []))
        return ld, si, slab, Scale(buf, cp, i)

    def _scale_read(self, cl, sc, reader):
        """`reader` is the last task that reads scale tile `sc`: its buffer's next copy waits for it."""
        self.sc_reader[cl][sc.buf] = reader

    def _release(self, cl, si, gemm, deq, sc):
        """Slab si is free after the chunk's GEMM `gemm` (a list: the chunk's GEMMs of every row block) and the copy
        of its scale tile (the dequantisation `deq` reads the copy)."""
        gms = list(gemm) if isinstance(gemm, (list, tuple)) else [gemm]
        self.ls[cl].release(si, gms)
        self.ls[cl]._extra[si].append(sc.copy)

    def gemm(self, k, c, name, a, a_pitch, b, dst, M, K, N, k_shift, deps, int4=1):
        """One GEMM (gemm_full; int4: the weights INT4 in the parity layout, B_PARITY_OFF into their slab); returns the
        task its output is ready after."""
        lay = 0
        if int4:
            lay = INT4_B_LAYOUT_PARITY
            b = b.view(B_PARITY_OFF)
        kw = dict(input_A_addr=a, input_B_addr=b, input_C_addr=0, output_D_addr=dst, M=M, K=K, N=N,
                  array_shape_idx=SHAPE3, transpose_A=0, transpose_B=0, accumPrevC=0,
                  int32tofp16_enable=1, int4_b_enable=lay, d_shift=k_shift, a_pitch=a_pitch)
        return self.node(k, c, self.roles["gemm"], name, "__snax_bingo_kernel_gemm_full",
                         SnaxBingoKernelGemmFullArgs(**kw), list(deps))

    def hbm_bcasts(self, go):
        """The root's host queues its memory chip's push engine (the engine of the root's rings, whose posts queue after
        these): the router's weights behind one flag, then X a tile at a time, each tile and its flag multicast into
        every chip's L3 (ChipletBuild.hbm_cast: their posts and waits). Returns the two casts (HbmCast); X's `end` is
        the task after which its last flag has landed on the root."""
        R, T = self.ROOT, self.TAG
        m = self.mem[R]
        one = self.h_seq[m].address + 64                    # a word holding 1 in the memory chip's HBM: the flags' value
        if self.x_cast == "unicast":
            # params x_cast unicast: the same tiles and flags written into each chip's L3 in turn (self.order: the root
            # first), with a progress flag a (chip, tile) on the root; each task ends once its last progress flag has
            # landed
            rw = self.hbm_cast(R, f"rw_hbm_ucast_k{R:02x}", HostBingoKernelHbmUcastArgs(
                self.h_rwhbm.address, BingoMemSymbol(f"{T}_rwland"), BingoMemSymbol(f"{T}_rwflag"),
                BingoMemSymbol(f"{T}_rwuprog"), one, 1, self.rw_bytes, 1, m, self.engine[R], self.order), after=[go[R]])
            x = self.hbm_cast(R, f"x_hbm_ucast_k{R:02x}", HostBingoKernelHbmUcastArgs(
                self.h_xhbm.address, self.xland(), BingoMemSymbol(f"{T}_xtflag"), BingoMemSymbol(f"{T}_xuprog"), one, 1,
                self.XT, self.NXT, m, self.engine[R], self.order), after=[rw])
            return rw, x
        rw = self.hbm_cast(R, f"rw_hbm_bcast_k{R:02x}", HostBingoKernelHbmBcastArgs(
            self.h_rwhbm.address, BingoMemSymbol(f"{T}_rwland"), BingoMemSymbol(f"{T}_rwflag"), one, 1, self.rw_bytes,
            1, m, self.engine[R]), after=[go[R]])
        x = self.hbm_cast(R, f"x_hbm_bcast_k{R:02x}", HostBingoKernelHbmBcastArgs(
            self.h_xhbm.address, self.xland(), BingoMemSymbol(f"{T}_xtflag"), one, 1, self.XT, self.NXT, m,
            self.engine[R]), after=[rw])
        return rw, x
