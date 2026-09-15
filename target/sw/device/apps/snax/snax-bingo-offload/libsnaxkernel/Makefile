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
# GEMM shape-table generation and validation
# ------------------------------------------------------------------------
# Both GEMM kernel families use the active table in runtime/generated. The
# outer common.mk adds that directory before the checked-in fallback.
include $(LIBSNAXKERNEL_DIR)/../../../../runtime/snax/versacore/gemm_shapes.mk

# A build-rule edit must invalidate validation even if the selected hardware
# file predates the existing stamp.
$(VALIDATE_STAMP): $(LIBSNAXKERNEL_DIR)Makefile

# The outer common.mk wires DATA_H as a prereq of every object file in this
# device app. Exposing the validation stamp ensures every build first runs
# the hwcfg vs. gemm_shapes.h cross-check before any kernel TU compiles.
DATA_H = $(GEMM_SHAPES) $(VALIDATE_STAMP)

.PHONY: clean-data clean
clean-data:
	rm -f $(GEMM_SHAPES) $(VALIDATE_STAMP) $(LIBSNAXKERNEL_DIR).validate.stamp

clean: clean-data
