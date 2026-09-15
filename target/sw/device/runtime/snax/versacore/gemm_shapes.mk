# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

ifndef HEMAIA_GEMM_SHAPES_MK
HEMAIA_GEMM_SHAPES_MK := 1

GEMM_SHAPES_MAKEFILE := $(realpath $(lastword $(MAKEFILE_LIST)))
VERSACORE_DIR := $(dir $(GEMM_SHAPES_MAKEFILE))
include $(VERSACORE_DIR)/../../../../../../util/cluster_cfg.mk

VALIDATE_SCRIPT := $(VERSACORE_DIR)validate_shapes.py
GEMM_SHAPES ?= $(abspath $(VERSACORE_DIR)/../../generated/gemm_shapes.h)
VALIDATE_STAMP ?= $(dir $(GEMM_SHAPES)).gemm_shapes.validate.stamp
_GEMM_SHAPES_DEFAULT_GOAL := $(.DEFAULT_GOAL)

# The selection stamp also changes when switching to an older configuration
# file. Keep this rule deferred: outer Makefiles can define SNITCH_ROOT later.
.SECONDEXPANSION:
$(VALIDATE_STAMP): $$(CLUSTER_CFG_STAMP) $$(CLUSTER_CFG_DEPS) $(VALIDATE_SCRIPT) $(GEMM_SHAPES_MAKEFILE)
	python3 $(VALIDATE_SCRIPT) --hwcfg $(GEMM_HWCFG) --header $(GEMM_SHAPES) --generate
	@touch $@

# Validation generates the header too. Recover independently if someone
# removes only the header, without forcing recompilation on no-op validation.
$(GEMM_SHAPES): | $(VALIDATE_STAMP)
	python3 $(VALIDATE_SCRIPT) --hwcfg $(GEMM_HWCFG) --header $@ --generate

.DEFAULT_GOAL := $(_GEMM_SHAPES_DEFAULT_GOAL)
endif
