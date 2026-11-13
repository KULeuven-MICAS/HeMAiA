# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Llama-3-8B's attention (synthetic weights), a 1,024-token prompt, all 16 passes, the whole attention; the code is
# ../../../llama3/common/llama3_attn_mc.py.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = fa/llama3_prefill_ablation/base
include $(MK_DIR)../../../llama3/common/llama3.mk
