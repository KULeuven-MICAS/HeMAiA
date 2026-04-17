# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Workloads to build separate ELFs for
APPS =  gemm_double_buffer
APPS += gemm_seperate_load_and_compute_parallel
APPS += gemm_seperate_load_and_compute_serial
APPS += gemm_stacked
APPS += gemm_sweep
APPS += xdma_all_ops
APPS += xdma_layout_transform
APPS += xdma_sweep
APPS += dummy_double_buffer_ci
APPS += moe4
APPS += attn_test_single_gemm
APPS += attn_test_quantize
APPS += attn_test_dequantize
APPS += attn_test_int32_add
APPS += attn_test_ksplit_gemm
APPS += attn_test_softmax
APPS += attn_reference
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