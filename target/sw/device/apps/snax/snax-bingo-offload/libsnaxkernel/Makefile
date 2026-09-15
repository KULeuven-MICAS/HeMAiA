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
#
# Both are lazy (`=`): $(SNITCH_ROOT) is defined by the outer apps/common.mk, which
# is included AFTER this file.
GEMM_RTL_CFG   = $(abspath $(LIBSNAXKERNEL_DIR)/../../../../../../rtl/cfg/lru.hjson)
GEMM_CLUSTER   = $(shell python3 -c "import hjson,sys; print(hjson.load(open(sys.argv[1]))['clusters'][0])" $(GEMM_RTL_CFG) 2>/dev/null)
GEMM_HWCFG     = $(SNITCH_ROOT)/target/snitch_cluster/cfg/$(GEMM_CLUSTER).hjson
VERSACORE_DIR := $(abspath $(LIBSNAXKERNEL_DIR)/../../../../runtime/snax/versacore)
VALIDATE_SCRIPT := $(VERSACORE_DIR)/validate_shapes.py
GEMM_SHAPES     := $(VERSACORE_DIR)/gemm_shapes.h
VALIDATE_STAMP  := $(LIBSNAXKERNEL_DIR).validate.stamp

# Use .SECONDEXPANSION so $(GEMM_HWCFG) in the prereq list is resolved in a
# later pass — the outer apps/common.mk defines $(SNITCH_ROOT) only after
# this Makefile is included, so the first prereq pass would see it empty.
.SECONDEXPANSION:

# gemm_shapes.h is GENERATED from the ACTIVE cluster cfg, not hand-maintained. The table
# is a pure function of the hjson -- array shape, element widths, the serialised C/D width
# and the C/D port's spatial nest -- so deriving it is the only way it cannot go stale.
# It used to be written by hand and cross-checked, which turned every cfg change into a
# failed build plus a manual re-derivation; the same derivation now just writes the answer.
# --emit rewrites only when the content changes, so an unchanged cfg does not retouch the
# header and force every dependent TU to recompile.
$(GEMM_SHAPES): $$(GEMM_HWCFG) $(VALIDATE_SCRIPT)
	python3 $(VALIDATE_SCRIPT) --hwcfg $(GEMM_HWCFG) --header $(GEMM_SHAPES) --emit

$(VALIDATE_STAMP): $(GEMM_SHAPES)
	@touch $(VALIDATE_STAMP)

# The outer common.mk wires DATA_H as a prereq of every object file in this
# device app. Exposing the validation stamp ensures every build first runs
# the hwcfg vs. gemm_shapes.h cross-check before any kernel TU compiles.
DATA_H = $(VALIDATE_STAMP)

.PHONY: clean-data clean
clean-data:
	rm -f $(VALIDATE_STAMP) $(GEMM_SHAPES)

clean: clean-data
