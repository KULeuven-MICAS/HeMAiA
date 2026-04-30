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
# GEMM shape-table validation
# ------------------------------------------------------------------------
# offload_hw_kernels/gemm.h and offload_sw_kernels/gemm.h are hand-maintained
# plain C and both index bingo_gemm_shape_params[] from
# runtime/snax/versacore/gemm_shapes.h. validate_shapes.py (lives next to
# the header) cross-checks it against the active hwcfg and fails the build
# on any drift. No code is generated.
#
# GEMM_HWCFG is lazy (`=`) so it picks up $(SNITCH_ROOT) after the outer
# apps/common.mk has defined it (this Makefile is included first).
GEMM_HWCFG     = $(SNITCH_ROOT)/target/snitch_cluster/cfg/snax_versacore_to_cluster.hjson
VERSACORE_DIR := $(abspath $(LIBSNAXKERNEL_DIR)/../../../../runtime/snax/versacore)
VALIDATE_SCRIPT := $(VERSACORE_DIR)/validate_shapes.py
GEMM_SHAPES     := $(VERSACORE_DIR)/gemm_shapes.h
VALIDATE_STAMP  := $(LIBSNAXKERNEL_DIR).validate.stamp

# Use .SECONDEXPANSION so $(GEMM_HWCFG) in the prereq list is resolved in a
# later pass — the outer apps/common.mk defines $(SNITCH_ROOT) only after
# this Makefile is included, so the first prereq pass would see it empty.
.SECONDEXPANSION:

$(VALIDATE_STAMP): $$(GEMM_HWCFG) $(VALIDATE_SCRIPT) $(GEMM_SHAPES)
	python3 $(VALIDATE_SCRIPT) --hwcfg $(GEMM_HWCFG) --header $(GEMM_SHAPES)
	@touch $(VALIDATE_STAMP)

# The outer common.mk wires DATA_H as a prereq of every object file in this
# device app. Exposing the validation stamp ensures every build first runs
# the hwcfg vs. gemm_shapes.h cross-check before any kernel TU compiles.
DATA_H = $(VALIDATE_STAMP)

.PHONY: clean-data clean
clean-data:
	rm -f $(VALIDATE_STAMP)

clean: clean-data
