# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The fused SIMD whole-operators.

Softmax, rmsnorm, silu, swiglu, rope, the quantiser, the MoE combine and FlashAttention's
softmax epilogue -- each ONE kernel rather than a chain of primitives, because the per-task
fixed cost is ~190 cycles on this machine and a decomposed chain spends more on dispatch
than on arithmetic.

The row operators come in an F16F16 and an F16I8 form; the second folds the quantisation
into the same pass, so a layer that narrows to int8 pays for one task, not two."""

from typing import Union, Dict, Optional
from bingo_mem_handle import BingoMemAlloc

from kernel_base import BingoKernelArgs


class SnaxBingoKernelSimdFp16ToInt8Args(BingoKernelArgs):
    """Fp16ToInt8: out = clamp(round(x * inv_scale), -128, 127) over `rows*beats` flat beats,
    on the HasFp16ToInt8 xDMA datapath -- the dedicated activation fp16 -> int8 GEMM-operand
    requant that replaces the host quantize_f16i8. inv_scale_f32bits = FP32 bits of 127/max|x|
    (the producer computes max|x| via MAX(x)+MAX(-x) reduces). dst_bound0 = rows*beats//2
    (int8 packs two elements per fp16 lane)."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_fp16_to_int8"

    def __init__(self, src_addr: Union[BingoMemAlloc, int], dst_addr: Union[BingoMemAlloc, int],
                 beats: int, rows: int, inv_scale_f32bits: int = 0,
                 csr_mode: int = 0, dst_bound0: Optional[int] = None,
                 inv_scale_addr: Union[BingoMemAlloc, int, None] = None):
        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.beats = beats
        self.rows = rows
        self.inv_scale_f32bits = inv_scale_f32bits
        self.csr_mode = csr_mode
        self.dst_bound0 = (rows * beats) // 2 if dst_bound0 is None else dst_bound0
        # 0 = use inv_scale_f32bits; a handle = read the runtime inv_scale (127/max|x| that
        # requant_scale wrote) from L1 at run time.
        self.inv_scale_addr = 0 if inv_scale_addr is None else inv_scale_addr

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_simd_fp16_to_int8_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.src_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.dst_addr, "dst_addr", a, handle_name_map)
        a["beats"] = str(self.beats)
        a["rows"] = str(self.rows)
        a["inv_scale_f32bits"] = str(self.inv_scale_f32bits)
        a["csr_mode"] = str(self.csr_mode)
        a["dst_bound0"] = str(self.dst_bound0)
        if isinstance(self.inv_scale_addr, int):
            a["inv_scale_addr_lo"] = str(self.inv_scale_addr & 0xFFFFFFFF)
            a["inv_scale_addr_hi"] = str((self.inv_scale_addr >> 32) & 0xFFFFFFFF)
        else:
            self._process_addr(self.inv_scale_addr, "inv_scale_addr", a, handle_name_map)
        return a


class SnaxBingoKernelSimdRopeArgs(BingoKernelArgs):
    """Fused FP16 RoPE: iDMA adjacent-pair swap of x + 3 StreamElementwise passes
    (x*cos_full, xswap*sin_signed, +) -> out. cos_full/sin_signed are precomputed
    tables; the kernel allocates xswap/tmp1/tmp2 scratch from L1. D = beats*32 fp16
    elements per row, rows independent token positions."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_rope"

    def __init__(self, x_addr: Union[BingoMemAlloc, int], cos_addr: Union[BingoMemAlloc, int],
                 sin_addr: Union[BingoMemAlloc, int], out_addr: Union[BingoMemAlloc, int],
                 cols: int, rows: int = 1):
        self.x_addr = x_addr
        self.cos_addr = cos_addr
        self.sin_addr = sin_addr
        self.out_addr = out_addr
        self.cols = cols        # per-row fp16 length D (a multiple of 32)
        self.rows = rows

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_simd_rope_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.x_addr, "x_addr", a, handle_name_map)
        self._process_addr(self.cos_addr, "cos_addr", a, handle_name_map)
        self._process_addr(self.sin_addr, "sin_addr", a, handle_name_map)
        self._process_addr(self.out_addr, "out_addr", a, handle_name_map)
        a["cols"] = str(self.cols)
        a["rows"] = str(self.rows)
        return a


class SnaxBingoKernelSimdAddF16Args(BingoKernelArgs):
    """out = a + b, elementwise over [rows, cols] fp16, on the SIMD core.

    WHY THIS AND NOT THE xDMA ADD. `__snax_bingo_kernel_xdma_elementwise_add_ab` drives the
    HasElementwiseAdd WRITER extension, which snax_split_cluster does not have. It stays
    CORRECT there -- the kernel has a CPU fallback -- but the fallback is a scalar loop
    over every int32 element on the xDMA hart, where the extension folds 16 lanes per
    512-bit beat. This drives HasStreamElementwise, a READER extension the cluster does
    have, so the residual is a vector op rather than a scalar one.

    It is a whole operator, not a primitive: the op, the operand count and the output
    precision are fixed here rather than being arguments, so a caller cannot ask for a
    combination the residual path does not mean.
    """

    KERNEL_NAME = "__snax_bingo_kernel_simd_stream_elementwise"

    _OP_ADD_FP16 = 1
    _OUT_FP16 = 0

    def __init__(self, a_addr: Union[BingoMemAlloc, int], b_addr: Union[BingoMemAlloc, int],
                 out_addr: Union[BingoMemAlloc, int], rows: int, cols: int):
        if cols % 32:
            raise ValueError(f"cols={cols} must be a multiple of 32 -- one SIMD beat is "
                             f"64 B = 32 fp16 lanes and a partial beat is not handled.")
        self.a_addr = a_addr
        self.b_addr = b_addr
        self.out_addr = out_addr
        self.rows = rows
        self.cols = cols
        self.beats = cols // 32

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_simd_stream_elementwise_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.a_addr, "src_addr", a, handle_name_map)
        self._process_addr(self.out_addr, "dst_addr", a, handle_name_map)
        # A NON-ZERO src_b makes the kernel derive operand_stride as (src_b - src_addr) at
        # run time, which is what lets the two operands be separate allocations. The
        # reader AGU strides FORWARD only, so b must sit at the HIGHER address; the
        # residual's two buffers come from one allocator in creation order, and the block
        # passes them in that order.
        self._process_addr(self.b_addr, "src_b_addr", a, handle_name_map)
        a["beats"] = str(self.beats)
        a["operand_stride"] = "0"          # derived from src_b at run time
        a["operand_count"] = "2"
        a["op"] = str(self._OP_ADD_FP16)
        a["rows"] = str(self.rows)
        a["csr_mode"] = "0"
        a["dst_bound0"] = str(self.rows * self.beats)
        a["out_dtype"] = str(self._OUT_FP16)
        a["inv_scale_f32bits"] = "0"
        return a


class _SimdRowOpArgs(BingoKernelArgs):
    """Shared base for the fused fp16 SIMD kernels that reduce ALONG a row and write one
    output tensor -- softmax, rmsnorm, silu. The user's args are HW-free: `input_addr` / `output_addr` are the
    [rows, cols] tensors (cols = per-row length D, a multiple of 32). The OUTPUT PRECISION is
    chosen by which subclass (kernel) you pick, not an arg:
      *F16F16Args -> fp16 output; output_addr is the fp16 buffer.
      *F16I8Args  -> int8 output (fused Fp16ToInt8, kernel-baked scale); output_addr is the
                     int8 buffer. No fp16 output is written (the fp16 result stays in scratch).
    Subclasses set KERNEL_NAME and STRUCT_NAME."""
    STRUCT_NAME = None  # subclass sets

    def __init__(self, input_addr: Union[BingoMemAlloc, int],
                 output_addr: Union[BingoMemAlloc, int], rows: int, cols: int):
        self.input_addr = input_addr
        self.output_addr = output_addr
        self.rows = rows
        self.cols = cols

    def get_struct_name(self) -> str:
        return self.STRUCT_NAME

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.input_addr, "input_addr", a, handle_name_map)
        self._process_addr(self.output_addr, "output_addr", a, handle_name_map)
        a["rows"] = str(self.rows)
        a["cols"] = str(self.cols)
        return a


class SnaxBingoKernelSimdSoftmaxF16F16Args(_SimdRowOpArgs):
    """Whole FP16 softmax in ONE DM-core kernel -> fp16 output. reduce-MAX, device negate,
    sub-max, merged EXP+Sexp, integer reciprocal (rv32iM divu), normalize. Host does only
    Load / Store / Check."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_softmax_f16_f16"
    STRUCT_NAME = "__snax_bingo_kernel_simd_softmax_args_t"


class SnaxBingoKernelSimdSoftmaxF16I8Args(_SimdRowOpArgs):
    """Same fused softmax pipeline -> int8 output (fused Fp16ToInt8, baked 127.0 scale since
    softmax output is in [0,1]). output_addr is the int8 [rows, cols] buffer."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_softmax_f16_i8"
    STRUCT_NAME = "__snax_bingo_kernel_simd_softmax_args_t"


class SnaxBingoKernelSimdRmsnormF16F16Args(_SimdRowOpArgs):
    """Whole FP16 rmsnorm in ONE DM-core kernel -> fp16 output. reduce-SUMSQ, integer
    1/sqrt(Sxx/N) (device sqrt + reciprocal, no FPU), normalize. cols is a power-of-two
    multiple of 32. Host does only Load / Store / Check."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_rmsnorm_f16_f16"
    STRUCT_NAME = "__snax_bingo_kernel_simd_rmsnorm_args_t"


class SnaxBingoKernelSimdRmsnormF16I8Args(_SimdRowOpArgs):
    """Same fused rmsnorm pipeline -> int8 output (fused Fp16ToInt8, baked 64.0 scale).
    output_addr is the int8 [rows, cols] buffer."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_rmsnorm_f16_i8"
    STRUCT_NAME = "__snax_bingo_kernel_simd_rmsnorm_args_t"


class SnaxBingoKernelSimdSiluF16F16Args(_SimdRowOpArgs):
    """Whole FP16 SiLU (x*sigmoid(x)) in ONE DM-core kernel -> fp16 output (one StreamMap pass).
    Host does only Load / Store / Check."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_silu_f16_f16"
    STRUCT_NAME = "__snax_bingo_kernel_simd_silu_args_t"


class SnaxBingoKernelSimdSiluF16I8Args(_SimdRowOpArgs):
    """Same fused silu -> int8 output (fused Fp16ToInt8, baked 16.0 scale). output_addr is the
    int8 [rows, cols] buffer."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_silu_f16_i8"
    STRUCT_NAME = "__snax_bingo_kernel_simd_silu_args_t"


class _SnaxBingoKernelSimdSwigluArgs(BingoKernelArgs):
    """Shared base for the fused fp16 SwiGLU kernel (out = silu(gate) * up): two input tensors
    (gate, up) + one output, all [rows, cols] (cols a multiple of 32). Output precision is chosen
    by the subclass: F16F16 -> fp16 output; F16I8 -> int8 (fused Fp16ToInt8, baked 16.0 scale).
    The kernel allocates the intermediate silu(gate) scratch itself. Subclasses set KERNEL_NAME."""
    STRUCT_NAME = "__snax_bingo_kernel_simd_swiglu_args_t"

    def __init__(self, gate_addr: Union[BingoMemAlloc, int], up_addr: Union[BingoMemAlloc, int],
                 output_addr: Union[BingoMemAlloc, int], rows: int, cols: int):
        self.gate_addr = gate_addr
        self.up_addr = up_addr
        self.output_addr = output_addr
        self.rows = rows
        self.cols = cols

    def get_struct_name(self) -> str:
        return self.STRUCT_NAME

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.gate_addr, "gate_addr", a, handle_name_map)
        self._process_addr(self.up_addr, "up_addr", a, handle_name_map)
        self._process_addr(self.output_addr, "output_addr", a, handle_name_map)
        a["rows"] = str(self.rows)
        a["cols"] = str(self.cols)
        return a


class SnaxBingoKernelSimdSwigluF16F16Args(_SnaxBingoKernelSimdSwigluArgs):
    """Whole FP16 SwiGLU in ONE DM-core kernel -> fp16 output (StreamMap SiLU + StreamElementwise
    MUL). Host does only Load / Store / Check."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_swiglu_f16_f16"


class SnaxBingoKernelSimdSwigluF16I8Args(_SnaxBingoKernelSimdSwigluArgs):
    """Same fused swiglu -> int8 output (fused Fp16ToInt8, baked 16.0 scale). output_addr is the
    int8 [rows, cols] buffer."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_swiglu_f16_i8"


class SnaxBingoKernelSimdMoeCombineF16Args(BingoKernelArgs):
    """The reconvergence of a conditional fork: out = SUM over the SELECTED experts of
    weight[e] * y[e], elementwise over [rows, cols] in fp16.

    Expert e's operand is src_base + e * src_stride, so the E landing slots are one
    allocation and each expert's push destination is a plain view into it.

    activation_addr and weight_addr are the gating node's decision. Leave them unset and
    fork.combine() fills them in through bind_combine() -- which is the point: the combine
    reads the SAME decision the hardware skipped on rather than re-deriving top-k, and
    nothing in the compiler has to know what this kernel's fields are called."""
    KERNEL_NAME = "__snax_bingo_kernel_simd_moe_combine_f16"
    STRUCT_NAME = "__snax_bingo_kernel_simd_moe_combine_args_t"

    def __init__(self, output_addr: Union[BingoMemAlloc, int],
                 src_base_addr: Union[BingoMemAlloc, int], src_stride: int,
                 rows: int, cols: int, num_inputs: int = 0,
                 activation_addr=None, weight_addr=None):
        self.output_addr = output_addr
        self.src_base_addr = src_base_addr
        self.src_stride = src_stride
        self.num_inputs = num_inputs
        self.rows = rows
        self.cols = cols
        self.activation_addr = activation_addr
        self.weight_addr = weight_addr

    def bind_combine(self, activation, weights, num_inputs):
        """Called by fork.combine() lowering. Anything set explicitly wins, so a
        workload can still point the kernel at its own arrays."""
        if self.activation_addr is None:
            self.activation_addr = activation
        if self.weight_addr is None:
            self.weight_addr = weights
        if not self.num_inputs:
            self.num_inputs = num_inputs

    def get_struct_name(self) -> str:
        return self.STRUCT_NAME

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        if self.activation_addr is None or self.weight_addr is None:
            raise ValueError(
                f"{type(self).__name__}: activation_addr/weight_addr are unset. Declare "
                f"the reconvergence with fork.combine(node, inputs, kind='weighted_sum', "
                f"weights=fork.weights) so the compiler binds them, or pass them here.")
        if not self.num_inputs:
            raise ValueError(
                f"{type(self).__name__}: num_inputs is 0, so the combine would read no "
                f"expert at all.")
        a = {}
        self._process_addr(self.output_addr, "output_addr", a, handle_name_map)
        self._process_addr(self.src_base_addr, "src_base_addr", a, handle_name_map)
        a["src_stride"] = str(self.src_stride)
        a["num_inputs"] = str(self.num_inputs)
        a["rows"] = str(self.rows)
        a["cols"] = str(self.cols)
        self._process_addr(self.activation_addr, "activation_addr", a, handle_name_map)
        self._process_addr(self.weight_addr, "weight_addr", a, handle_name_map)
        return a


class SnaxBingoKernelSimdFaSoftmaxArgs(BingoKernelArgs):
    """FlashAttention online-softmax epilogue: the whole per-tile SIMD half, one kernel.

    The producing GEMM node writes `s16_src` (its D32 output, fp16 [bc, 32]); this node
    writes `p8_dst` for the consuming GEMM. `arena` carries the persistent (m, l, O)
    recurrence plus all scratch, so the SAME arena handle must be passed for every KV
    tile of one query tile; tile_idx=0 seeds m, l and O.

    Size the arena with arena_bytes(bc, dhead) -- it must match
    SIMD_FA_ARENA_BYTES(bc, dhead) in offload_hw_kernels/simd.h, where the adjacencies
    inside it are load-bearing.

    THE ADJACENCIES ARE NOT ALL INSIDE THE ARENA. Two of this kernel's SIMD tasks pair
    operands in DIFFERENT buffers, reaching out of the arena with a stride:

        FA_SH_NEGM_OUT   writes -m_new to [rmax][negm]   negm = the one-beat prefix of s16_src
        FA_SH_LNEW_IN    reads  [lsc][rsum]              rsum = the trailing beat of p8_dst

    `snax_simd_shape_t.stride` is a uint32_t, so the arena must sit at a LOWER address than
    both s16_src and p8_dst. That held for free while addresses came from bingo_l1_alloc in
    alphabetical order, and stopped being free once a compiler took over placement. Breaking
    it does not corrupt the row maximum -- that chain is arena-internal -- it corrupts P and
    the row sum, and the run hangs before those checks report.
    """

    # (earlier_attr, later_attr): earlier must be placed at a lower address than later.
    PLACEMENT_ORDER = [("arena", "s16_src"), ("arena", "p8_dst")]

    KERNEL_NAME = "__snax_bingo_kernel_simd_fa_softmax"

    # Geometry modes; must match SIMD_FA_GEOM_* in offload_hw_kernels/simd.h.
    GEOM_SELF = 0
    GEOM_PROLOGUE = 1
    GEOM_PRIMED = 2
    GEOM_CSR_PRIMED = 3

    BEAT_BYTES = 64
    # The arena opens with a FIXED block reserved for the cached task geometries, which is
    # NOT sizeof(snax_simd_shape_t) * 12: that struct's size follows the cluster's
    # reader_agu_temporal_dimension, so deriving the reservation from it would make every
    # data offset below depend on the hjson as well. Must equal SIMD_FA_SHAPES_BYTES in
    # offload_hw_kernels/simd.h, where a _Static_assert checks the shapes still fit.
    SHAPE_BYTES = 1024

    @classmethod
    def arena_bytes(cls, bc: int, dhead: int) -> int:
        return cls.SHAPE_BYTES + (dhead + 10) * cls.BEAT_BYTES

    @classmethod
    def layout(cls, bc: int, dhead: int) -> Dict[str, int]:
        """Byte offsets of every field inside the arena.

        MIRRORS simd_fa_layout() in offload_hw_kernels/simd.h, field for field and in
        order. It exists so a workload can read the recurrence back -- `arena.view(
        layout(bc, d)["mrun"])` is the running row maximum -- without a second copy of the
        arithmetic in the workload itself.

        The order is load-bearing on the device side: adjacent pairs are how the SIMD
        tasks find their two operands, since a task reads ONE flat stream and pairs
        whatever is next to each other in it. Reordering anything here without doing the
        same in simd.h feeds a task the beat next door, which does not fault.
        """
        b = cls.BEAT_BYTES
        off = {}
        t = cls.SHAPE_BYTES
        # negmS, s16 and p16 are GONE: the fused exp pass reads the GEMM's own score
        # buffer through a one-beat prefix and emits INT8 directly, so neither the tile
        # copy nor the FP16 P is ever materialised. Mirrors simd_fa_layout() exactly.
        for name, beats in (("rmax", 1), ("mrun", 1),
                            ("mnew", 1), ("delta", 1), ("corrL", 1), ("lrun", 1),
                            ("lnew", 1), ("rsum", 1), ("lsc", 1),
                            ("corrO", 1), ("oacc", dhead)):
            off[name] = t
            t += beats * b
        assert t == cls.arena_bytes(bc, dhead), (
            f"arena layout walks to {t} but arena_bytes says "
            f"{cls.arena_bytes(bc, dhead)} -- the two have drifted apart")
        return off

    def __init__(self, s16_src: Union[BingoMemAlloc, int],
                 p8_dst: Union[BingoMemAlloc, int],
                 arena: Union[BingoMemAlloc, int],
                 bc: int, dhead: int, tile_idx: int, seed_state: int = 1,
                 geom_mode: int = 0):
        if bc % 2:
            raise ValueError(f"bc must be even (the quantiser packs 2:1), got {bc}")
        if bc <= 0 or dhead <= 0:
            raise ValueError(f"bc and dhead must be positive, got {bc}, {dhead}")
        self.s16_src = s16_src
        self.p8_dst = p8_dst
        self.arena = arena
        self.bc = bc
        self.dhead = dhead
        self.tile_idx = tile_idx
        # 1 keeps the kernel seeding m and l itself, which is what every existing workload
        # expects. 0 says the workload has already filled them -- see the xDMA memset.
        self.seed_state = int(seed_state)
        # SELF / PROLOGUE / PRIMED; see the geom_mode comment on the C struct. PRIMED is a
        # PROMISE about the GRAPH -- that a PROLOGUE node for this same arena is an ancestor
        # of this node -- and nothing on the device can check it, so it is opt-in per node.
        if geom_mode not in (self.GEOM_SELF, self.GEOM_PROLOGUE, self.GEOM_PRIMED,
                             self.GEOM_CSR_PRIMED):
            raise ValueError(f"geom_mode must be 0 (SELF), 1 (PROLOGUE), 2 (PRIMED) "
                             f"or 3 (CSR_PRIMED), "
                             f"got {geom_mode}")
        self.geom_mode = int(geom_mode)

    def get_struct_name(self) -> str:
        return "__snax_bingo_kernel_simd_fa_softmax_args_t"

    def get_c_field_assignments(self, handle_name_map: Dict[BingoMemAlloc, str]) -> Dict[str, str]:
        a = {}
        self._process_addr(self.s16_src, "s16_src_addr", a, handle_name_map)
        self._process_addr(self.p8_dst, "p8_dst_addr", a, handle_name_map)
        self._process_addr(self.arena, "arena_addr", a, handle_name_map)
        a["bc"] = str(self.bc)
        a["dhead"] = str(self.dhead)
        a["tile_idx"] = str(self.tile_idx)
        a["seed_state"] = str(self.seed_state)
        a["geom_mode"] = str(self.geom_mode)
        return a


