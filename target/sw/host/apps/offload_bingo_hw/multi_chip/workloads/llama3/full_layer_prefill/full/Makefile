# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# One Llama-3-8B layer (synthetic weights), a fresh 1,024-token prompt, every pass; the code is
# ../common/llama3_layer_mc.py.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = llama3/full_layer_prefill/full
# each chiplet's compile of this header is large: at most this many at once
PER_CHIP_JOBS = 2
include $(MK_DIR)../../common/llama3.mk

# the layer's driver lives beside this workload, not in llama3/common: the generator reruns when it changes
$(DATA_H) $(OFFLOAD_H) $(HBM_MANIFEST): $(MK_DIR)../common/llama3_layer_mc.py
