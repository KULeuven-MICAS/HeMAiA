# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Usage of absolute paths is required to externally include this Makefile.
# LIBSNAXKERNEL_DIR is captured with := at include time; the outer
# apps/common.mk later reuses `MK_DIR`, so we must not rely on that name
# here.
LIBSNAXKERNEL_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
LIBSNAXKERNEL_CLEAN_GOAL := $(filter clean clean-%,$(MAKECMDGOALS))

# ------------------------------------------------------------------------
# HW-parameterized GEMM kernels
# ------------------------------------------------------------------------
# libsnaxkernel/gen_bingo_gemm_kernels.py is invoked twice, once per kernel
# layer, reading the layer's committed template and emitting the matching
# gemm.h with literal integers in every slot that used to come from
# versacore_hw_param.h. Outputs are gitignored; marker stamps keep
# `make sw` idempotent once the hwcfg is unchanged.
#
# GEMM_HWCFG is lazy (`=`) so it picks up $(SNITCH_ROOT) after the outer
# apps/common.mk has defined it (this Makefile is included first).
GEMM_HWCFG    = $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson
GEMM_SCRIPT  := $(LIBSNAXKERNEL_DIR)gen_bingo_gemm_kernels.py

GEMM_SW_GEN     := $(LIBSNAXKERNEL_DIR)offload_sw_kernels/gemm.h
GEMM_SW_MARKER  := $(LIBSNAXKERNEL_DIR)offload_sw_kernels/.gemm.stamp
GEMM_SW_TEMPL   := $(LIBSNAXKERNEL_DIR)offload_sw_kernels/gemm_kernels.template.c

GEMM_HW_GEN     := $(LIBSNAXKERNEL_DIR)offload_hw_kernels/gemm.h
GEMM_HW_MARKER  := $(LIBSNAXKERNEL_DIR)offload_hw_kernels/.gemm.stamp
GEMM_HW_TEMPL   := $(LIBSNAXKERNEL_DIR)offload_hw_kernels/gemm_kernels.template.c

# Use .SECONDEXPANSION so $(GEMM_HWCFG) in the prereq list is resolved in a
# later pass — the outer apps/common.mk defines $(SNITCH_ROOT) only after
# this Makefile is included, so the first prereq pass would see it empty.
.SECONDEXPANSION:

$(GEMM_SW_GEN) $(GEMM_SW_MARKER): $$(GEMM_HWCFG) $(GEMM_SCRIPT) $(GEMM_SW_TEMPL)
	python3 $(GEMM_SCRIPT) --hwcfg $(GEMM_HWCFG) --template $(GEMM_SW_TEMPL) --out $(GEMM_SW_GEN)
	@touch $(GEMM_SW_MARKER)

$(GEMM_HW_GEN) $(GEMM_HW_MARKER): $$(GEMM_HWCFG) $(GEMM_SCRIPT) $(GEMM_HW_TEMPL)
	python3 $(GEMM_SCRIPT) --hwcfg $(GEMM_HWCFG) --template $(GEMM_HW_TEMPL) --out $(GEMM_HW_GEN)
	@touch $(GEMM_HW_MARKER)

# The outer common.mk wires DATA_H as a prereq of every object file in this
# device app, so exposing both generated headers via DATA_H guarantees they
# exist before any kernel TU that includes them.
DATA_H = $(GEMM_SW_GEN) $(GEMM_HW_GEN)

.PHONY: clean-data clean
clean-data:
	rm -f $(GEMM_SW_GEN) $(GEMM_SW_MARKER) $(GEMM_HW_GEN) $(GEMM_HW_MARKER)

clean: clean-data
