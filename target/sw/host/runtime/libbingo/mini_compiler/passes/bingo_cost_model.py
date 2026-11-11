# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The compiler's estimate of how long a task takes, for ordering each core's program
(bingo_dfg_schedule.py).

DETERMINISTIC, SET FROM CLUSTER REPLAYS. Every estimate is a formula over the task's own arguments and
a few constants, the same for every graph: nothing is fitted to the graph being compiled. The
scheduler compares the orders it could choose with it -- a 20-us GEMM against a 50-ns flag wait, an
operand that lands soon against one that lands late -- and which accelerator's work to split, where
it matters which core is the busy one: a SIMD task's fixed cost against a VersaCore configure, a
dispatch against the work it hides. The constants are each kernel's fixed and per-element cost as
the SNAX cluster replays of the LLM stage apps run them (passes/bingo_cluster_replay.py), kept
round.

UNITS: cluster cycles. A task's time is what it holds its core for; an accelerator's work is apart from it
(engine_cycles): a split dispatch's runs while its core goes on, and a whole task on an accelerator core holds the
accelerator for its work too, so it waits for, and keeps from, the works of split dispatches there.

THE CONSTANTS:

  DMA_BYTES_PER_CYCLE      the cluster's wide AXI port: 512 bits a cycle
  FAR_BYTES_PER_CYCLE      a transfer that leaves the chip (D2D links, the memory chip's HBM)
  DMA_LATENCY, FAR_LATENCY a transfer's first beat: in the chip, across the package
  PUSH_BYTES_PER_CYCLE     a memory chip's push of weight chunks into one compute chip's L3
  SIMD_ELEMS_PER_CYCLE     the SIMD core's FlashAttention tile: a 32-byte beat (16 FP16 elements) every 2 cycles
  SIMD_ROW_ELEMS_PER_CYCLE a rows x cols SIMD kernel (SwiGLU, RMSNorm), which reads its rows more than once
  SIMD_OVERHEAD            a SIMD task's fixed cost: its CSR writes, the streamer's start, the drain
  SIMD_CONFIGURE           of which the configure part's (a split SIMD task's)
  GEMV_SETUP, GEMV_STEPS_PER_CYCLE  the one-token GEMV: its set-up, then its K x N steps
  XDMA_TRANSPOSE_BYTES_PER_CYCLE    the xDMA's 2D transpose
  CHAIN_CFG_BYTES_PER_CYCLE         a ChainGather's configure, which grows with the gather's bytes
  GEMM_CONFIGURE           a VersaCore dispatch's software set-up: every CSR of the streamer and
                           the array written, the shape looked up
  GEMM_TABLE_CONFIGURE     the same from the cluster's L1 copy of the shape's parameters (gemm_start's)
  GEMM_REUSE_CONFIGURE     a dispatch that rewrites only its operand addresses (gemm_minimal)
  GEMM_START, GEMM_WAIT    a queued dispatch's start write, a wait that finds the work done (any accelerator's)
  CONFIGURE_CYCLES         a configure task's CSR writes, per accelerator
  HOST_TASK                a host task's dispatch and its own work
  SMALL_TASK               a task with nothing to move or compute (a stamp, a flag poll)
  FIXED_TASKS              a kernel whose time does not follow from its arguments (a table build)
  TASK_DISPATCH            what every device task costs its core besides its own work: the descriptor's fetch, the
                           call and the return, the completion's report (so a split's extra parts cost their core)

  array cycles of a dispatch: M x K x N steps (each a block of the array shape) at one a cycle

EDGE LATENCY: the time from a task's end to its successor seeing it -- nothing on one core, a
little inside a cluster or a chip, a dependency signal's trip across the package between chips.
"""

DMA_BYTES_PER_CYCLE = 64
FAR_BYTES_PER_CYCLE = 16
DMA_LATENCY = 100
FAR_LATENCY = 300
PUSH_BYTES_PER_CYCLE = 32
PUSH_LATENCY = 400
SIMD_ELEMS_PER_CYCLE = 8
SIMD_ROW_ELEMS_PER_CYCLE = 2
SIMD_OVERHEAD = 380
SIMD_CONFIGURE = 250
GEMV_SETUP = 300
GEMV_STEPS_PER_CYCLE = 2
XDMA_OVERHEAD = 200
XDMA_TRANSPOSE_BYTES_PER_CYCLE = 24
CHAIN_CFG_BYTES_PER_CYCLE = 32
GEMM_CONFIGURE = 400
GEMM_TABLE_CONFIGURE = 150
GEMM_REUSE_CONFIGURE = 150
GEMM_START = 60
GEMM_WAIT = 30
HOST_TASK = 400
SMALL_TASK = 80
DEFAULT_TASK = 100
TASK_DISPATCH = 80
FIXED_TASKS = {"__snax_bingo_kernel_fa_build_shapes": 500, "__snax_bingo_kernel_gemm_shape_init": 300}

LAT_CLUSTER = 30
LAT_CHIP = 60
LAT_HOST = 400
LAT_REMOTE = 300
LAT_PER_HOP = 100

GEMM_FULL = "__snax_bingo_kernel_gemm_full"
GEMM_MINIMAL = "__snax_bingo_kernel_gemm_minimal"
GEMV = "__snax_bingo_kernel_gemv"
RING_LOAD = "__snax_bingo_kernel_idma_ring_load"
WEIGHT_PREFETCH = "__host_bingo_kernel_weight_prefetch"


def _is_far(addr, chip):
    """Whether an operand is off this chip: a handle placed on another chip, an address the generator computed
    (an HBM address, a remote expression), or a symbol of another chip."""
    if addr is None or addr == 0:
        return False
    if isinstance(addr, int):
        return True
    if isinstance(addr, str):
        return "chiplet_addr_transform" in addr
    base = getattr(addr, "base", addr)
    owner = getattr(base, "chip_id", None)
    return owner is not None and owner != chip


def dma_cycles(nbytes, src, dst, chip):
    """A copy of `nbytes` from `src` to `dst` issued on `chip`."""
    nbytes = max(int(nbytes or 0), 0)
    if _is_far(src, chip) or _is_far(dst, chip):
        return FAR_LATENCY + nbytes // FAR_BYTES_PER_CYCLE
    return DMA_LATENCY + nbytes // DMA_BYTES_PER_CYCLE


def array_cycles(args):
    """The array steps of a VersaCore dispatch with M, K and N, or 0."""
    m, k, n = (getattr(args, f, None) for f in ("M", "K", "N"))
    if not all(isinstance(v, int) for v in (m, k, n)):
        return 0
    return max(m, 1) * max(k, 1) * max(n, 1)


def is_blocking_gemm(node):
    """A VersaCore dispatch that configures, starts and waits in one task (gemm_full, the typed kernels, the
    FlashAttention matmuls)."""
    k = node.kernel_name or ""
    if k == GEMM_FULL:
        return True
    return (k.startswith(("__snax_bingo_kernel_gemm_i", "__snax_bingo_kernel_gemm_fa_q", "__snax_bingo_kernel_gemm_fa_p"))
            and not k.endswith("_start") and array_cycles(node.kernel_args) > 0)


def _simd_cycles(a):
    bc = getattr(a, "bc", None)
    if isinstance(bc, int) and getattr(a, "dhead", None) is not None:
        return SIMD_OVERHEAD + bc * 32 // SIMD_ELEMS_PER_CYCLE   # a FlashAttention score tile: Bc keys x 32 queries
    beats = getattr(a, "beats", None)
    if isinstance(beats, int) and beats > 0:
        return SIMD_OVERHEAD + 2 * beats * max(int(getattr(a, "rows", 1) or 1), 1)
    rows, cols = getattr(a, "rows", None), getattr(a, "cols", None)
    if isinstance(rows, int) and isinstance(cols, int):
        return SIMD_OVERHEAD + rows * cols // SIMD_ROW_ELEMS_PER_CYCLE
    return SIMD_OVERHEAD


def _xdma_cycles(a):
    m, n, eb = (getattr(a, f, None) for f in ("M", "N", "elem_bytes"))
    if all(isinstance(v, int) for v in (m, n, eb)):
        return XDMA_OVERHEAD + m * n * eb // XDMA_TRANSPOSE_BYTES_PER_CYCLE     # a 2D transpose
    size = getattr(a, "size", None)
    chain = getattr(a, "chain", None)
    if isinstance(size, int):
        hops = len(chain) if isinstance(chain, (list, tuple)) else 0
        return XDMA_OVERHEAD + size // DMA_BYTES_PER_CYCLE + hops * LAT_REMOTE
    bounds = getattr(a, "temporal_bounds_src", None)
    if isinstance(bounds, (list, tuple)) and bounds:
        steps = 1
        for b in bounds:
            steps *= max(int(b), 1)
        return XDMA_OVERHEAD + steps
    return XDMA_OVERHEAD


def task_cycles(node):
    """How long `node` holds its core (a blocking dispatch: its set-up and its array time)."""
    k, a = node.kernel_name or "", node.kernel_args
    chip = node.assigned_chiplet_id
    if k in FIXED_TASKS:
        return FIXED_TASKS[k]
    if k == GEMV:
        steps = 1
        for f in ("kt", "nb", "groups"):
            v = getattr(a, f, 1)
            steps *= max(int(v), 1) if isinstance(v, int) else 1
        return GEMV_SETUP + steps // GEMV_STEPS_PER_CYCLE
    if is_blocking_gemm(node):
        return GEMM_CONFIGURE + array_cycles(a)
    if k == GEMM_MINIMAL:
        return GEMM_REUSE_CONFIGURE
    if k.startswith("__host_bingo_kernel"):
        return HOST_TASK
    if k == RING_LOAD:
        return DMA_LATENCY + int(getattr(a, "size", 0)) // DMA_BYTES_PER_CYCLE
    if "idma" in k:
        size = getattr(a, "nbytes", None)                     # a strided copy's every run (Idma2dCopy)
        size = size if isinstance(size, int) else getattr(a, "size", None)
        if isinstance(size, int) and getattr(a, "bcast", 0):
            size *= 2                                         # a broadcast put: every cluster of the chip
        if isinstance(size, int):
            return dma_cycles(size, getattr(a, "src_addr", None), getattr(a, "dst_addr", None), chip)
        return DMA_LATENCY
    if "xdma" in k:
        return _xdma_cycles(a)
    if "simd" in k:
        return _simd_cycles(a)
    if "stamp" in k or "wait_flags" in k or "dummy" in k:
        return SMALL_TASK
    return DEFAULT_TASK


CONFIGURE_CYCLES = {"gemm": GEMM_CONFIGURE, "simd": SIMD_CONFIGURE, "xdma": XDMA_OVERHEAD}


def configure_cycles(args, accel):
    """A configure part's CSR writes: VersaCore's from the shape table where its start reads one (shape_addr); a
    ChainGather's (jct_csr0), which grows with the gather's bytes."""
    if accel == "gemm" and hasattr(args, "shape_addr"):
        return GEMM_TABLE_CONFIGURE
    size = getattr(args, "size", None)
    if accel == "xdma" and hasattr(args, "jct_csr0") and isinstance(size, int):
        return XDMA_OVERHEAD + size // CHAIN_CFG_BYTES_PER_CYCLE
    return CONFIGURE_CYCLES[accel]


def engine_cycles(args, accel):
    """What a task's work holds its accelerator for once started (its core's set-up apart): a VersaCore dispatch's
    array steps, a SIMD task's beats, an xDMA transfer's beats and hops."""
    if accel == "gemm":
        return array_cycles(args)
    if accel == "simd":
        return max(_simd_cycles(args) - SIMD_OVERHEAD, 0)
    if accel == "xdma":
        return max(_xdma_cycles(args) - XDMA_OVERHEAD, 0)
    return 0


def hops(a, b):
    return abs((a >> 4) - (b >> 4)) + abs((a & 15) - (b & 15))


def edge_latency(u_place, v_place, host_core):
    """Cycles from the end of a task at `u_place` (chip, cluster, core) to its successor at `v_place` seeing it."""
    (uk, uc, ur), (vk, vc, vr) = u_place, v_place
    if uk != vk:
        return LAT_REMOTE + LAT_PER_HOP * hops(uk, vk)
    if ur == host_core or vr == host_core:
        return 0 if (ur, uc) == (vr, vc) else LAT_HOST
    if uc != vc:
        return LAT_CHIP
    return 0 if ur == vr else LAT_CLUSTER
