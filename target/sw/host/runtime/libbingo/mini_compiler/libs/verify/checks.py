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

There is a second reason. Every libs port already declares its `dtype`. Once the check
is chosen BY that dtype, a block that changes precision cannot leave a stale checker behind
-- `check_out(ctx, ..., dtype=port.dtype)` follows the port. That is the multi-precision
case: one call site, right for i8, f16, f32 and i32.

WHAT LIVES WHERE. The comparison itself is a property of the KERNEL ABI, so one args
class per precision lives in bingo_kernel_args.py beside the element-size table it
validates against -- HostBingoKernelCheckResultF32Args, ...I32Args and the rest. This
module only builds the NODE: it places the kernel on the host core, orders it, and pairs it
with the readback that has to precede it. Keeping the two apart means a workload that does
not use libs still gets the named args, and this file has no ABI knowledge to drift.

    from libs.verify import checks
    st, ck = checks.readback_and_check(ctx, "fa_o_c0", src=oacc, golden=h["o"],
                                       dtype="i32", elems=512, rtol=0.02, after=pv[-1])
"""

from bingo_kernel_args import (
    HostBingoKernelCheckResultBytesArgs,
    HostBingoKernelCheckResultF16Args,
    HostBingoKernelCheckResultF16RelArgs,
    HostBingoKernelCheckResultF32Args,
    HostBingoKernelCheckResultI8Args,
    HostBingoKernelCheckResultI32Args,
    HostBingoKernelIdmaArgs,
)
from bingo_mem_handle import BingoMemAlloc

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


def check_int8(ctx, name, *, golden, got, elems, tol=0.0, after=(), label=None):
    """Signed int8, absolute tolerance in LSBs. For a quantised activation.

    `tol` is a count of quantisation steps, not a fraction: tol=1 accepts a value that
    rounded the other way. Pass 0 to demand exactness while still reporting the mismatch
    as an element index rather than a byte offset.
    """
    return ctx.host_node(name, _KERNEL, HostBingoKernelCheckResultI8Args(
        golden, got, int(elems), tol=float(tol), name=label or name), after)


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


def check_fp32(ctx, name, *, golden, got, elems, tol, after=(), label=None):
    """FP32, absolute tolerance. `elems` counts fp32 words, so the byte count is 4x it."""
    return ctx.host_node(name, _KERNEL, HostBingoKernelCheckResultF32Args(
        golden, got, int(elems), tol=float(tol), name=label or name), after)


def check_int32_rel(ctx, name, *, golden, got, elems, rtol, after=(), label=None):
    """Signed int32, RELATIVE: |out - g| <= rtol*|g| + 0.001*max|g|.

    This is the accumulator compare. An int32 GEMM accumulator spans several orders of
    magnitude within one tile, so an absolute tolerance chosen for the large entries is
    meaningless for the small ones and vice versa. The 0.001*max|g| term is the kernel's
    own floor for entries near zero, and is not a parameter.
    """
    return ctx.host_node(name, _KERNEL, HostBingoKernelCheckResultI32Args(
        golden, got, int(elems), rtol=float(rtol), name=label or name), after)


# ======================================================================================
# Choosing by the port's dtype
# ======================================================================================
# A block does not hardcode which of the six it wants; it asks for the one that matches the
# precision it declared. Adding a member to DType without adding it here is then a build
# error at the first check rather than a wrong comparison.

_ABSOLUTE = {"i8": check_int8, "f16": check_fp16, "f32": check_fp32}
_RELATIVE = {"f16": check_fp16_rel, "i32": check_int32_rel}


def checker_for(dtype: str, *, relative: bool = False):
    """The wrapper that compares this precision, or a refusal naming what is available."""
    table = _RELATIVE if relative else _ABSOLUTE
    fn = table.get(dtype)
    if fn is None:
        other = "absolute" if relative else "relative"
        alt = sorted(_ABSOLUTE if relative else _RELATIVE)
        kind = "relative" if relative else "absolute"
        raise ValueError(
            f"no {kind} comparison for dtype {dtype!r}: the kernel defines "
            f"{kind} modes for {sorted(table)}. {dtype!r} has an {other} mode "
            f"({alt}), or compare the bytes with check_bytes().")
    return fn


def check_out(ctx, name, *, golden, got, dtype, elems, tol=None, rtol=None,
              after=(), label=None):
    """Check a buffer against its golden, picking the comparison from `dtype`.

    `label` is what the UART prints on a mismatch and defaults to the node name. Keep them
    separate where they say different things: the node name locates the check in the graph
    ("Check_o_c2"), the label names the quantity a reader is looking for ("fa_o_c2").

    Exactly one of `tol` (absolute) or `rtol` (relative) selects the mode. Requiring the
    caller to say WHICH is deliberate: a bare `tolerance=0.02` means a hundredth of a unit
    in one mode and 2% in another, and the two differ by orders of magnitude on the same
    tensor.
    """
    if (tol is None) == (rtol is None):
        raise ValueError(
            f"check_out({name!r}) needs exactly one of tol= (absolute) or rtol= "
            f"(relative). They are not interchangeable: on an int32 accumulator the same "
            f"number means a fixed distance one way and a fraction of each element the "
            f"other.")
    fn = checker_for(dtype, relative=rtol is not None)
    kw = {"rtol": rtol} if rtol is not None else {"tol": tol}
    return fn(ctx, name, golden=golden, got=got, elems=elems, after=after, label=label, **kw)


# ======================================================================================
# Reading a device buffer back before comparing it
# ======================================================================================

def readback(ctx, name, *, src, nbytes, after=(), dst=None):
    """Copy a device buffer out to L3 so the host can see it. Returns (handle, node).

    The host cannot read L1. Every check is therefore two nodes, and separating them here
    is not ceremony: the store is what the NEXT compute must not race, so the caller needs
    the store node by itself to order against it. A check that owned its own hidden store
    could not be ordered that way -- which is exactly the RAW race that made FA's Store_o
    read the accumulator one PV too early.
    """
    h = dst if dst is not None else BingoMemAlloc(f"{ctx.prefix}out_{name}",
                                                  size=int(nbytes), mem_level="L3")
    st = ctx.host_node(f"Store_{name}", "__host_bingo_kernel_idma",
                       HostBingoKernelIdmaArgs(src, h, int(nbytes)), after)
    return h, st


def readback_and_check(ctx, name, *, src, golden, dtype, elems, tol=None, rtol=None,
                       nbytes=None, after=(), label=None):
    """readback() then check_out(), ordered. Returns (store_node, check_node).

    Both are returned because they order differently: downstream compute that overwrites
    `src` must wait on the STORE, while anything that only needs the verdict waits on the
    CHECK. Collapsing them to one node would force the stricter of the two everywhere.
    """
    if nbytes is None:
        nbytes = int(elems) * _ELEM_BYTES[dtype]
    h, st = readback(ctx, name, src=src, nbytes=nbytes, after=after)
    ck = check_out(ctx, f"Check_{name}", golden=golden, got=h, dtype=dtype, elems=elems,
                   tol=tol, rtol=rtol, after=st, label=label or name)
    return st, ck


_ELEM_BYTES = {"i8": 1, "f16": 2, "f32": 4, "i32": 4}
