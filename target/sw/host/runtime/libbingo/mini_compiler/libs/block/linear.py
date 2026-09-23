# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""A projection: one INT8 GEMM with its operand loads, as a block.

Every weight matrix in a transformer layer -- Wq, Wk, Wv, Wo, and each expert's up/gate/
down -- is this block with different shapes. It exists so a layer states `Linear(tokens,
d_in, d_out)` rather than restating the M/K/N tiling, the load pair and the C-address rule
four times over.

WHAT IT PRODUCES. D-layout FP16, because the narrowing rides the GEMM's own D port: an
INT32 beat carries half as many values as an FP16 one, so converting at the consumer would
double both the beats it reads and the L1 the tile occupies.

WHAT IT DOES NOT DO. It does not normalise, quantise or reshape its input. A projection
consumes A-layout int8 and that is what its port says; getting there from whatever the
previous stage emitted is the caller's composition, and the order is forced by hardware:
reshape at FP16, THEN quantise, because a row_major->A conversion needs an 8-byte run that
int8 does not have. See comm/nest.py.
"""

from dataclasses import dataclass

from bingo_kernel_args import (
    SnaxBingoKernelGemmFullArgs,
    SnaxBingoKernelIdma1dCopyArgs,
)

from ..comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel, Port,
                    PortSpec)


@dataclass(frozen=True)
class LinearCfg:
    """Shapes in ELEMENTS. The mesh-tile counts a descriptor wants are derived."""

    tokens: int
    d_in: int
    d_out: int
    mesh: tuple                       # (meshRow, tileSize, meshCol)
    cluster: int = 0
    array_shape_idx: int = 0
    transpose_a: int = 0
    transpose_b: int = 0

    def __post_init__(self):
        mr, ts, mc = self.mesh
        for nm, val, unit, why in (
                ("tokens", self.tokens, mr, "meshRow"),
                ("d_in", self.d_in, ts, "tileSize"),
                ("d_out", self.d_out, mc, "meshCol")):
            if val % unit:
                raise ValueError(
                    f"LinearCfg: {nm}={val} is not a multiple of {why}={unit}. A partial "
                    f"tile is not an error the array reports -- it computes over the tile "
                    f"it was given and the tail of the output is never written.")

    # ---- what a descriptor counts ------------------------------------------------------
    @property
    def M_T(self):
        return self.tokens // self.mesh[0]

    @property
    def K_T(self):
        return self.d_in // self.mesh[1]

    @property
    def N_T(self):
        return self.d_out // self.mesh[2]

    @property
    def sizes(self) -> dict:
        """Every buffer this block needs, in BYTES."""
        mr, ts, mc = self.mesh
        return {
            "x": self.M_T * self.K_T * mr * ts,            # int8  A-layout
            "w": self.K_T * self.N_T * ts * mc,            # int8  B-layout
            "y": self.M_T * self.N_T * mr * mc * 2,        # fp16  D-layout
        }


class Linear(Block):
    """y = x @ W, INT8 in, FP16 out.

      in   x  A-layout int8, [tokens, d_in]
           w  B-layout int8, [d_in, d_out]
      out  y  D-layout fp16, [tokens, d_out], in this block's cluster L1
    """

    name = "linear"

    def __init__(self, cfg: LinearCfg = None, **params):
        self.cfg = cfg if cfg is not None else LinearCfg(**params)

    @property
    def inputs(self) -> dict:
        c = self.cfg
        # No mem_level: this block fetches its own operands, so where the caller keeps them
        # is the caller's business. `needs` says what its loads read from.
        return {
            "x": PortSpec(Layout.A, DType.I8, (c.tokens, c.d_in), doc="activation"),
            "w": PortSpec(Layout.B, DType.I8, (c.d_in, c.d_out), doc="weight"),
        }

    @property
    def needs(self) -> dict:
        """L1 -- what the ARRAY reads, which is not the same as where the caller keeps it.

        Saying L3 here would be a statement about this block's loads, and it would refuse
        an operand that is ALREADY in L1 because a previous stage produced it there. That
        is the common case in a composed layer and it is the cheap one: there is nothing
        to move. So the requirement is where the engine reads, and build() emits a load
        only for what is further out.
        """
        from dataclasses import replace
        return {n: replace(s, mem_level=MemLevel.L1) for n, s in self.inputs.items()}

    @property
    def outputs(self) -> dict:
        c = self.cfg
        return {"y": PortSpec(Layout.D, DType.F16, (c.tokens, c.d_out),
                              mem_level=MemLevel.L1,
                              doc="projection output, D-layout fp16, in L1")}

    def build(self, ctx: Ctx, bound: dict) -> BlockResult:
        c, sz = self.cfg, self.cfg.sizes
        g = ctx.at(c.cluster)

        l1_y = g.l1(f"{self.name}_y", sz["y"])

        def operand(nm):
            """The L1 address the array will read, and the node that filled it.

            An operand ALREADY in L1 is used where it lies: copying it to a second L1
            buffer would cost a transfer, a dependency edge and a second allocation the
            static-L1 pass then has to fit, all to produce bytes that are already there.
            `ends` carries whatever produced it, so the ordering is unchanged.
            """
            port = bound[nm]
            if port.spec.mem_level == MemLevel.L1:
                return port.handle, list(port.ends), None
            buf = g.l1(f"{self.name}_{nm}", sz[nm])
            nd = g.node(f"Ld_{self.name}_{nm}", ctx.dm,
                        "__snax_bingo_kernel_idma_1d_copy",
                        SnaxBingoKernelIdma1dCopyArgs(port.handle, buf, sz[nm]))
            return buf, [nd], nd

        l1_x, x_after, ld_x = operand("x")
        l1_w, w_after, ld_w = operand("w")

        # input_C_addr=0, NOT l1_y. Every VersaCore GEMM computes D = A*B + C, and
        # accumPrevC=0 selects where C comes from: a NON-ZERO address READS C from memory
        # and adds it. Passing the output buffer as C reads it before anything wrote it --
        # uninitialised TCDM, which simulates as X -- and D = A*B + X = X. That X reaches
        # the host, where check_result loads it and the CVA6 issues on an unknown operand.
        # It never fails a check; it kills the host.
        gemm = g.node(f"Gemm_{self.name}", ctx.gemm, "__snax_bingo_kernel_gemm_full",
                      SnaxBingoKernelGemmFullArgs(
                          input_A_addr=l1_x, input_B_addr=l1_w, input_C_addr=0,
                          output_D_addr=l1_y, M=c.M_T, K=c.K_T, N=c.N_T,
                          array_shape_idx=c.array_shape_idx,
                          transpose_A=c.transpose_a, transpose_B=c.transpose_b,
                          accumPrevC=0, int32tofp16_enable=1),
                      x_after + w_after)

        nodes = [n for n in (ld_x, ld_w) if n is not None] + [gemm]
        return BlockResult(
            outputs={"y": Port(self.outputs["y"], l1_y, (gemm,), cluster=c.cluster,
                               name="y")},
            inputs={"x": Port(self.inputs["x"], bound["x"].handle,
                              (ld_x,) if ld_x else (gemm,), name="x"),
                    "w": Port(self.inputs["w"], bound["w"].handle,
                              (ld_w,) if ld_w else (gemm,), name="w")},
            nodes=nodes,
            # The weight load has no predecessor, so no data edge can order it against an
            # earlier block's buffers. Pipeline(gate_sources=True) is what lets its L1 be
            # reused; naming it here is what makes that possible.
            sources=[ld_w] if ld_w is not None else [],
            extra={"l1": {"x": l1_x, "w": l1_w, "y": l1_y}})
