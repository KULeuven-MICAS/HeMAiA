# Resolve software and generation inputs from the active SoC configuration.
# Variables are lazy: device data Makefiles are included before SNITCH_ROOT
# is initialized. Consumers must use second expansion for CLUSTER_CFG_DEPS.
ifndef HEMAIA_CLUSTER_CFG_INCLUDED
HEMAIA_CLUSTER_CFG_INCLUDED := 1
CLUSTER_CFG_MK := $(abspath $(lastword $(MAKEFILE_LIST)))
CLUSTER_CFG_ROOT := $(abspath $(dir $(CLUSTER_CFG_MK))/..)
CLUSTER_CFG_RESOLVER := $(CLUSTER_CFG_ROOT)/util/resolve_cluster_cfg.py
CLUSTER_CFG_CONSUMER ?= $(abspath $(firstword $(MAKEFILE_LIST)))
CLUSTER_CFG_STAMP ?= $(abspath $(dir $(firstword $(MAKEFILE_LIST)))/.cluster_cfg.stamp)
CFG ?= $(CLUSTER_CFG_ROOT)/target/rtl/cfg/lru.hjson

CLUSTER_CFG_ARGS = --cfg "$(CFG)" $(if $(strip $(SNITCH_ROOT)),--snitch-root "$(SNITCH_ROOT)")
# Cache each query after its first (late) expansion. A failed query must stop
# make rather than silently dropping hardware prerequisites.
define cluster_cfg_query
$(strip $(eval _cluster_cfg_result := $(shell python3 "$(CLUSTER_CFG_RESOLVER)" $(CLUSTER_CFG_ARGS) --format $(1) || echo CLUSTER_CFG_ERROR))$(if $(filter CLUSTER_CFG_ERROR,$(_cluster_cfg_result)),$(error Failed to resolve cluster configuration from $(CFG)),$(_cluster_cfg_result)))
endef
CLUSTER_CFG_FILES = $(eval CLUSTER_CFG_FILES := $(call cluster_cfg_query,paths))$(CLUSTER_CFG_FILES)
GEMM_HWCFG ?= $(eval GEMM_HWCFG := $(call cluster_cfg_query,gemm))$(GEMM_HWCFG)
CLUSTER_CFG_INPUTS = $(CFG) $(CLUSTER_CFG_FILES) $(CLUSTER_CFG_STAMP) $(CLUSTER_CFG_RESOLVER) $(CLUSTER_CFG_MK)
CLUSTER_CFG_DEPS = $(CLUSTER_CFG_INPUTS) $(GEMM_HWCFG)
# General SoC generation also supports accelerators other than VersaCore.
# Such consumers set this empty and depend on CLUSTER_CFG_INPUTS instead.
CLUSTER_CFG_STAMP_HWCFG = $(GEMM_HWCFG)

# Cleaning must work after lru.hjson or the Bender checkout was removed.
# Keep resolution enabled for mixed invocations such as 'clean gen-data'.
ifneq ($(filter clean clean-%,$(MAKECMDGOALS)),)
ifeq ($(filter-out clean clean-%,$(MAKECMDGOALS)),)
CLUSTER_CFG_INPUTS :=
CLUSTER_CFG_DEPS :=
endif
endif

# A content stamp catches switches to older files as well as config edits.
# Preserve the including Makefile's default target.
_CLUSTER_CFG_DEFAULT_GOAL := $(.DEFAULT_GOAL)
.PHONY: cluster-cfg-force
cluster-cfg-force:
$(CLUSTER_CFG_STAMP): cluster-cfg-force
	@python3 "$(CLUSTER_CFG_RESOLVER)" $(CLUSTER_CFG_ARGS) --stamp "$@" $(if $(strip $(CLUSTER_CFG_STAMP_HWCFG)),--hwcfg "$(CLUSTER_CFG_STAMP_HWCFG)")
.DEFAULT_GOAL := $(_CLUSTER_CFG_DEFAULT_GOAL)
endif
