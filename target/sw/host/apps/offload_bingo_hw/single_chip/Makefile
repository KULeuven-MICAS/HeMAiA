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
APPS += basic_softmax_host
APPS += attn_reference_4cluster
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
