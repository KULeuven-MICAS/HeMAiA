# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Host-side verification, named by the precision instead of numbered.

WHY THIS EXISTS. `__host_bingo_kernel_check_result` selects its comparison with an integer
`check_type`, and six of them are defined. A caller writing `check_type=5` has to remember
that 5 is the int32 relative compare, that its `num_elements` counts int32s rather than
bytes, and that its `tolerance` is a RATIO and not an absolute. Getting any of the three
wrong does not fault -- the wrong element size compares a prefix of the buffer and passes,
and an absolute tolerance passed where a ratio was wanted passes everything. So the number
is not a detail to be tidied away; it is three coupled decisions that must agree, and a
named wrapper is what makes them agree by construction.

WHAT LIVES WHERE. The comparison itself is a property of the KERNEL ABI, so one args
class per precision lives in libs/kernels/kernel_host.py (re-exported by bingo_kernel_args)
beside the element-size table it validates against -- HostBingoKernelCheckResultF32Args,
...I32Args and the rest. This module only builds the NODE: it places the kernel on the host
core and orders it. Keeping the two apart means a workload that does not use libs still
gets the named args, and this file has no ABI knowledge to drift.

    from libs.verify import checks
    ck = checks.check_int32_rel(ctx, "Check_o_c0", golden=h["o"], got=l3_o,
                                elems=512, rtol=0.02, after=pv[-1], label="fa_o_c0")
"""

from bingo_kernel_args import (
    HostBingoKernelCheckResultBytesArgs,
    HostBingoKernelCheckResultF16Args,
    HostBingoKernelCheckResultF16RelArgs,
    HostBingoKernelCheckResultI32Args,
)

_KERNEL = "__host_bingo_kernel_check_result"


# ======================================================================================
# One wrapper per comparison
# ======================================================================================
# Each names its tolerance argument after what the C kernel actually does with it: `tol`
# where the comparison is absolute, `rtol` where it is a ratio. That is the distinction the
# shared `tolerance=` keyword hides, and the one that silently passes when confused.

def check_bytes(ctx, name, *, golden, got, nbytes, after=(), label=None):
    """Byte-exact. The only mode with no tolerance, and the only one that counts BYTES.

    Use it for anything whose bits are supposed to be reproduced exactly -- a copy, a
    layout conversion, a re-read of a buffer that should not have changed. A tolerance mode
    would accept a conversion that dropped the low bits of every element.
    """
    return ctx.host_node(name, _KERNEL, HostBingoKernelCheckResultBytesArgs(
        golden, got, int(nbytes), name=label or name), after)


def check_fp16(ctx, name, *, golden, got, elems, tol, after=(), label=None):
    """FP16, ABSOLUTE tolerance. The elements are promoted to fp32 before comparing, so
    `tol` is an fp32 distance even though the buffer is fp16.

    Absolute is the right choice where the quantity has a known scale -- a running max, a
    row sum -- and the wrong one across a tensor spanning orders of magnitude, where the
    same `tol` is strict at the bottom and vacuous at the top. Use check_fp16_rel there.
    """
    return ctx.host_node(name, _KERNEL, HostBingoKernelCheckResultF16Args(
        golden, got, int(elems), tol=float(tol), name=label or name), after)


def check_fp16_rel(ctx, name, *, golden, got, elems, rtol, after=(), label=None):
    """FP16, RELATIVE tolerance: |out - g| <= rtol*|g| + 0.05.

    The additive 0.05 is the kernel's own floor and is not a parameter -- it exists so a
    golden of exactly zero does not demand a bit-exact zero back. That floor is why this
    mode is not a drop-in for check_fp16 on small-magnitude data: below ~0.05 it accepts
    anything.
    """
    return ctx.host_node(name, _KERNEL, HostBingoKernelCheckResultF16RelArgs(
        golden, got, int(elems), rtol=float(rtol), name=label or name), after)


def check_int32_rel(ctx, name, *, golden, got, elems, rtol, after=(), label=None):
    """Signed int32, RELATIVE: |out - g| <= rtol*|g| + 0.001*max|g|.

    This is the accumulator compare. An int32 GEMM accumulator spans several orders of
    magnitude within one tile, so an absolute tolerance chosen for the large entries is
    meaningless for the small ones and vice versa. The 0.001*max|g| term is the kernel's
    own floor for entries near zero, and is not a parameter.
    """
    return ctx.host_node(name, _KERNEL, HostBingoKernelCheckResultI32Args(
        golden, got, int(elems), rtol=float(rtol), name=label or name), after)
