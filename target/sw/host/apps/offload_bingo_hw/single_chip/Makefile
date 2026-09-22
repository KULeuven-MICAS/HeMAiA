# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Workloads to build separate ELFs for
APPS = concurrent_dma_1cluster
APPS += gemm_double_buffer_1cluster
APPS += gemm_serial_dma_1cluster
APPS += gemm_parallel_dma_1cluster
APPS += gemm_mem_chip_1cluster
APPS += gemm_stacked_1cluster
APPS += gemm_sweep_1cluster
APPS += xdma_ci_ops_1cluster
APPS += xdma_1d_1cluster
# iDMA adjacent-pair swap (RoPE xswap enabler); shares the per-op sweep harness.
APPS += idma_swap_1cluster
# Per-op xDMA sweep workloads (one op per dir; functional check + cycle LUT).
# Shared machinery in util/sim/xdma/xdma_ops_lib.py. See util/sim/xdma/xdma_ops_lib.md.
APPS += xdma_6d_1cluster
APPS += xdma_transpose_1cluster
APPS += xdma_submatrix_1cluster
APPS += xdma_expand_1cluster
APPS += xdma_concat_1cluster
APPS += xdma_pad_1cluster
APPS += xdma_gather_1cluster
APPS += xdma_elementwise_add_1cluster
# Per-op SIMD sweep workloads (the fused whole-op kernels, one dir each). These run on
# the cluster's SIMD core; the sweep that collects their cost LUTs lives in
# target/sim/automation/sweep/simd.
APPS += simd_softmax_1cluster
APPS += simd_rmsnorm_1cluster
APPS += simd_silu_1cluster
APPS += simd_swiglu_1cluster
APPS += simd_rope_1cluster
# The five chain stages armed one at a time, so the cost model has a per-operator
# curve and not only whole-pipeline numbers.
APPS += simd_quant_1cluster
# FlashAttention: the two matmuls on the GEMM core, the whole online softmax as one
# kernel on the SIMD core, overlapped through the task graph.
# The snax port: one kernel, NQ the only difference. NQ=1 is decode (one query
# position per token, the whole KV cache read for it); NQ>1 is prefill, where many
# query tiles attend the same resident K/V. Intensity = NQ*Br exactly.
APPS += fa_decode_1cluster
APPS += fa_prefill_1cluster
# KV-sharded across four clusters, with the cross-cluster online-softmax merge done
# IN THE FABRIC by a ChainGather over the monoid junction -- not gathered to one
# cluster and reduced there.
APPS += fa_decode_4cluster
APPS += fa_prefill_4cluster
APPS += xdma_row_to_a_1cluster
APPS += xdma_a_to_row_1cluster
APPS += xdma_row_to_b_1cluster
APPS += xdma_b_to_row_1cluster
APPS += xdma_row_to_d_1cluster
APPS += xdma_d_to_row_1cluster
APPS += moe4_4cluster
APPS += basic_quantize_host
APPS += basic_dequantize_host
APPS += basic_int32_add_host
APPS += gemm_ksplit_4cluster
APPS += gemm_nsplit_accumprevc_4cluster
APPS += basic_softmax_host
APPS += attn_reference_4cluster
APPS += device_host_idma_check_1cluster
# Placeholders for the 2-cluster tapeout CI suites (hemaia_tapeout_2c[_simd]).
APPS += dummy_2cluster
APPS += dummy_simd_2cluster
# 2-cluster SIMD variant of the xDMA softmax sweep (hemaia_tapeout_2c_simd).
APPS += simd_softmax_2cluster
SUBDIRS = $(addprefix workloads/,$(APPS))


TARGET ?= all

.PHONY: all clean partial-build finalize-build $(SUBDIRS)

all partial-build finalize-build: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
